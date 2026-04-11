"""
Retrieval layer: Chroma loading, intent routing, parallel retrieval, reranking.

Uses the native chromadb client (same as ingest.py) to guarantee embedding
function compatibility. Query embeddings are generated with the same
nomic-embed-text model via Ollama.

Pipeline per query:
  1. Router  (gpt-5.4-mini)  → which collections to search
  2. Parallel retrieval       → async queries to selected Chroma collections
  3. Deduplication            → drop exact duplicate chunks
  4. FlashrankRerank          → rerank merged pool, keep top N
"""

import asyncio
import os
import re
from pathlib import Path
from typing import Literal

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from dotenv import load_dotenv
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent

_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/embeddings")
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")

_DB_CONFIG = {
    "courses":  {"path": str(PROJECT_ROOT / "data" / "db" / "courses"),  "k": 20},
    "policies": {"path": str(PROJECT_ROOT / "data" / "db" / "policies"), "k": 15},
    "majors":   {"path": str(PROJECT_ROOT / "data" / "db" / "majors"),   "k": 25},
}

# ──────────────────────────────────────────────────────────────────────────────
# Embedder — same OllamaEmbeddingFunction used by ingest.py
# Using the identical function guarantees query vectors are in the same space
# as the stored document vectors.
# ──────────────────────────────────────────────────────────────────────────────

_embedding_fn = OllamaEmbeddingFunction(url=_OLLAMA_URL, model_name=_OLLAMA_MODEL)

# ──────────────────────────────────────────────────────────────────────────────
# Chroma collections (loaded once at import time)
# ──────────────────────────────────────────────────────────────────────────────

def _load_collection(name: str, db_path: str) -> chromadb.Collection:
    """Load an existing Chroma collection. Fails loudly if the DB doesn't exist."""
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Chroma DB not found at '{db_path}'. "
            f"Run ingest/ingest.py for the '{name}' collection first."
        )
    client = chromadb.PersistentClient(path=str(path))
    return client.get_collection(name=name)


_collections: dict[str, chromadb.Collection] = {
    name: _load_collection(name, cfg["path"])
    for name, cfg in _DB_CONFIG.items()
}

# ──────────────────────────────────────────────────────────────────────────────
# Router (gpt-5.4-mini with structured output)
# ──────────────────────────────────────────────────────────────────────────────

class _RouterDecision(BaseModel):
    collections: list[Literal["courses", "policies", "majors"]]
    major_keyword: str | None = None  # Program name for majors collection filtering


_router_llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0).with_structured_output(
    _RouterDecision
)

_ROUTER_SYSTEM = """\
You are a query router for a UCI student academic assistant.
Decide which knowledge bases to search based on the student's question.

Collections:
- courses   → specific course info, prerequisites, corequisites, units, descriptions, restrictions
- policies  → academic rules, grading, add/drop deadlines, academic integrity, probation, transfer credit
- majors    → major/minor requirements, degree plans, required courses for a specific major or minor

Rules:
- Return only the collections that are clearly relevant.
- When a question asks about required courses for a major/minor, return BOTH "majors" AND "courses" —
  the major page lists required course codes, and the courses collection has their full details.
- When a question spans multiple topics (e.g. "does course X count for my major"), return multiple.
- If genuinely unsure, return all three — over-retrieval is better than missing context.

major_keyword: If the question is about a specific major or minor, set this to the program name
exactly as it would appear in the UCI catalogue (e.g. "Computer Science", "Informatics",
"Mathematics", "Electrical Engineering"). Leave null if no specific program is mentioned.
This is used to filter the majors collection to only relevant programme pages.
"""


async def _route(question: str) -> tuple[list[str], str | None]:
    decision: _RouterDecision = await _router_llm.ainvoke(
        [
            {"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user", "content": question},
        ]
    )
    return decision.collections, decision.major_keyword


# ──────────────────────────────────────────────────────────────────────────────
# Per-collection async retrieval
# ──────────────────────────────────────────────────────────────────────────────

async def _query_collection(
    name: str,
    embedding: list[float],
    where_document: dict | None = None,
) -> list[Document]:
    """Query one Chroma collection and return LangChain Documents.

    where_document: optional Chroma document filter, e.g. {"$contains": "Computer Science"}.
    Applied only when provided — falls back to pure semantic search otherwise.
    """
    k = _DB_CONFIG[name]["k"]
    collection = _collections[name]

    # When filtering by document content the candidate pool is much smaller,
    # so request more results to capture all chunks from the matching page.
    effective_k = 60 if where_document else k
    query_kwargs: dict = dict(
        query_embeddings=[embedding],
        n_results=effective_k,
        include=["documents", "metadatas"],
    )
    if where_document:
        query_kwargs["where_document"] = where_document

    # chromadb.query is synchronous — run in a thread to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    try:
        results = await loop.run_in_executor(
            None,
            lambda: collection.query(**query_kwargs),
        )
    except Exception:
        # If the filter yields no candidates, Chroma may raise; fall back without filter
        results = await loop.run_in_executor(
            None,
            lambda: collection.query(
                query_embeddings=[embedding],
                n_results=k,
                include=["documents", "metadatas"],
            ),
        )

    docs = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


async def _retrieve_parallel(
    question: str,
    collections: list[str],
    major_keyword: str | None = None,
) -> list[Document]:
    """Embed once, then query all selected collections in parallel.

    major_keyword: if set, the majors collection is filtered to documents containing
    this string — bypasses nomic-embed-text's weak semantic matching for programme names.
    """
    # Use the same OllamaEmbeddingFunction as ingest.py — guarantees vector compatibility.
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(None, _embedding_fn, [question])
    embedding: list[float] = embeddings[0]

    tasks = []
    for name in collections:
        where_doc = {"$contains": major_keyword} if (name == "majors" and major_keyword) else None
        tasks.append(_query_collection(name, embedding, where_doc))

    batches = await asyncio.gather(*tasks)

    # Flatten and deduplicate on first 200 chars of content
    seen: set[str] = set()
    docs: list[Document] = []
    for batch in batches:
        for doc in batch:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                docs.append(doc)
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# Reranker (local cross-encoder — no API call)
# ──────────────────────────────────────────────────────────────────────────────

_reranker = FlashrankRerank(top_n=15)


def _rerank(docs: list[Document], question: str) -> list[Document]:
    if not docs:
        return []
    return _reranker.compress_documents(docs, question)


# ──────────────────────────────────────────────────────────────────────────────
# Public interface
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Course code extraction + direct lookup
# Semantic search is weak at exact course codes (e.g. "MATH 2A").
# Extract any codes mentioned in the question and fetch them by Chroma ID
# directly, then prepend to the semantic pool before reranking.
# ──────────────────────────────────────────────────────────────────────────────

# Matches: MATH 2A, CS 161, ICS 6B, CHC/LAT 179, BIO SCI D114, EECS 70LB
# Department: 2-8 letters, optional slash+letters (cross-listed), optional
#             second word (BIO SCI). Number: optional leading letter + digits + trailing letters.
_COURSE_CODE_RE = re.compile(
    r"\b([A-Z]{2,8}(?:\s[A-Z]{2,8})?(?:/[A-Z]{2,8})?\s+[A-Z]?\d+[A-Z]{0,2})\b"
)


def _extract_course_codes(text: str) -> list[str]:
    """Return deduplicated course codes found in text, normalising whitespace.

    Do NOT uppercase the input — that turns common words like 'take' and 'and'
    into valid-looking department codes (TAKE, AND). Only already-uppercase
    tokens in the original text (e.g. MATH, CS, ICS) should match.
    """
    return list(dict.fromkeys(
        re.sub(r"\s+", " ", m).strip()
        for m in _COURSE_CODE_RE.findall(text)
    ))


def _direct_course_lookup(codes: list[str]) -> list[Document]:
    """
    Fetch course documents from the courses collection by ID.
    Tries both 'CODE' and 'CODE::chunk_0' so chunked courses are found too.
    """
    if not codes or "courses" not in _collections:
        return []

    collection = _collections["courses"]
    ids_to_try = []
    for code in codes:
        ids_to_try.append(code)
        ids_to_try.append(f"{code}::chunk_0")

    result = collection.get(ids=ids_to_try, include=["documents", "metadatas"])
    docs = []
    for text, meta in zip(result["documents"], result["metadatas"]):
        if text:
            docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


async def retrieve(question: str) -> list[Document]:
    """
    Full retrieval pipeline for a standalone question:
      route → parallel retrieval → dedup → rerank
    Returns up to top_n reranked Documents with source metadata attached.
    """
    # Step 1: direct lookup for any course codes mentioned in the question
    codes = _extract_course_codes(question)
    direct_docs = _direct_course_lookup(codes)

    # Step 2: semantic retrieval (with optional major keyword filter)
    collections, major_keyword = await _route(question)
    semantic_docs = await _retrieve_parallel(question, collections, major_keyword)

    # Step 3: merge — direct docs first (guaranteed relevant), then semantic
    seen: set[str] = {d.page_content[:200] for d in direct_docs}
    for d in semantic_docs:
        key = d.page_content[:200]
        if key not in seen:
            seen.add(key)
            direct_docs.append(d)
    all_docs = direct_docs

    return _rerank(all_docs, question)
