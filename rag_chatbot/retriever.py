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
import hashlib
import os
import re
from functools import lru_cache
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
    "policies": {"path": str(PROJECT_ROOT / "data" / "db" / "policies"), "k": 20},
    "majors":   {"path": str(PROJECT_ROOT / "data" / "db" / "majors"),   "k": 25},
}

# ──────────────────────────────────────────────────────────────────────────────
# Embedder — same OllamaEmbeddingFunction used by ingest.py
# Using the identical function guarantees query vectors are in the same space
# as the stored document vectors.
# ──────────────────────────────────────────────────────────────────────────────

_embedding_fn = OllamaEmbeddingFunction(url=_OLLAMA_URL, model_name=_OLLAMA_MODEL)


@lru_cache(maxsize=256)
def _embed_cached(text: str) -> tuple[float, ...]:
    # nomic-embed-text is deterministic, so caching on the exact string is safe.
    # Cast to plain Python float — Chroma validates against `float | int`, and a
    # raw list of np.float32 scalars fails that check (though numpy arrays pass).
    return tuple(float(x) for x in _embedding_fn([text])[0])


def _dedup(docs: list) -> list:
    """Deduplicate Documents using (url, chunk_id|code, content hash).

    Falls back to a full-content SHA-1 when metadata is missing. Avoids the
    false-collision risk of prefix-based dedup on chunks that share a prefix
    like '[COMPSCI 161] ' or '{title} > {heading}: '.
    """
    seen: set = set()
    out: list = []
    for doc in docs:
        meta = doc.metadata or {}
        content_hash = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()
        url = meta.get("url")
        ident = meta.get("chunk_id") or meta.get("code")
        key = (url, ident, content_hash) if url or ident else content_hash
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out

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
    requires_full_requirements: bool = False  # True when student asks for the full course requirement list


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

requires_full_requirements: Set to true when the student is asking what courses are
required for a specific major or minor, OR when they are asking whether a specific course
is required/mandatory in a major or minor. Examples that should be true:
  "what courses do I need for the CS major?"
  "list the lower-division requirements for Computer Science"
  "what upper-division courses are required for Math?"
  "do I have to take COMPSCI 161 as a CS major?"
  "is COMPSCI 161 required for the Computer Science major?"
  "does the CS major require COMPSCI 163?"
  "is ICS 6B mandatory for the Informatics major?"
Set to false for questions about major overview, specialisations, admissions, sample
plans, career paths, or anything that is not about whether specific courses are required.
"""


async def _route(question: str) -> tuple[list[str], str | None, bool]:
    decision: _RouterDecision = await _router_llm.ainvoke(
        [
            {"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user", "content": question},
        ]
    )
    return decision.collections, decision.major_keyword, decision.requires_full_requirements


# ──────────────────────────────────────────────────────────────────────────────
# Per-collection async retrieval
# ──────────────────────────────────────────────────────────────────────────────

async def _query_collection(
    name: str,
    embedding: list[float],
    where_document: dict | None = None,
    effective_k: int | None = None,
) -> list[Document]:
    """Query one Chroma collection and return LangChain Documents.

    where_document: optional Chroma document filter. Supports both single
        {"$contains": "..."} and combined {"$and": [...]} Chroma operators.
    effective_k: override the result count. When None, defaults to 60 if a
        filter is active (smaller candidate pool) or the per-collection k otherwise.
    """
    k = _DB_CONFIG[name]["k"]
    collection = _collections[name]

    if effective_k is None:
        # When filtering, the candidate pool is smaller so request more results
        # to avoid missing relevant chunks from the target page.
        effective_k = 60 if where_document else k
    query_kwargs: dict = dict(
        query_embeddings=[embedding],
        n_results=effective_k,
        include=["documents", "metadatas"],
    )
    if where_document:
        query_kwargs["where_document"] = where_document

    # chromadb.query is synchronous — run in a thread to avoid blocking the event loop.
    # A no-match where_document returns an empty list, not an exception — so we do NOT
    # swallow errors here. Let real Chroma failures surface instead of silently polluting
    # results with unfiltered chunks.
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: collection.query(**query_kwargs),
    )

    docs = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


async def _retrieve_parallel(
    question: str,
    collections: list[str],
    major_keyword: str | None = None,
    requires_full_requirements: bool = False,
) -> list[Document]:
    """Embed once, then query all selected collections in parallel.

    major_keyword: if set, the majors collection is filtered to documents containing
        this string — bypasses nomic-embed-text's weak semantic matching for programme names.
    requires_full_requirements: when True alongside a major_keyword, applies a combined
        $and filter that scopes results to the "Major Requirements" section only, excluding
        overview and admissions chunks that would otherwise rank above the course lists.
    """
    loop = asyncio.get_event_loop()
    embedding_tuple = await loop.run_in_executor(None, _embed_cached, question)
    embedding: list[float] = list(embedding_tuple)

    # Chroma's $contains is case-sensitive. UCI catalogue pages use title case
    # consistently, so normalize the router's output to avoid silent no-match filters.
    if major_keyword:
        major_keyword = major_keyword.strip().title()

    tasks = []
    for name in collections:
        if name == "majors" and major_keyword:
            if requires_full_requirements:
                # Both strings must appear in the chunk: the programme name (in the
                # prefix on every chunk from that page) AND "Major Requirements" (only
                # present in the section-4 requirements chunks, not in the overview).
                # effective_k=100 captures all specialisation chunks for large majors.
                where_doc: dict = {
                    "$and": [
                        {"$contains": major_keyword},
                        {"$contains": "Major Requirements"},
                    ]
                }
                eff_k = 100
            else:
                where_doc = {"$contains": major_keyword}
                eff_k = 60
            tasks.append(_query_collection(name, embedding, where_doc, eff_k))
        else:
            tasks.append(_query_collection(name, embedding))

    batches = await asyncio.gather(*tasks)
    return _dedup([doc for batch in batches for doc in batch])


# ──────────────────────────────────────────────────────────────────────────────
# Reranker (local cross-encoder — no API call)
# ──────────────────────────────────────────────────────────────────────────────

# Minimum relevance score below which reranked chunks are considered noise and
# dropped before hitting the prompt. FlashrankRerank attaches a score in
# metadata["relevance_score"] after compression.
_RELEVANCE_FLOOR = 0.05
_MIN_KEEP = 3


def _rerank(docs: list[Document], question: str, top_n: int = 15) -> list[Document]:
    if not docs:
        return []
    # Construct per call — FlashrankRerank caches model weights internally, so
    # this is cheap and avoids a shared-state race when concurrent requests each
    # want a different top_n.
    reranker = FlashrankRerank(top_n=top_n)
    ranked = reranker.compress_documents(docs, question)

    # Drop low-relevance chunks, but always keep at least _MIN_KEEP so a weakly
    # matched question still gets some context rather than nothing.
    filtered = [
        d for d in ranked
        if d.metadata.get("relevance_score", 1.0) >= _RELEVANCE_FLOOR
    ]
    if len(filtered) < _MIN_KEEP:
        return ranked[:_MIN_KEEP]
    return filtered


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
    r"\b((?:[A-Z]&[A-Z]\s[A-Z]{2,8}|[A-Z]{2,8}(?:\s[A-Z]{2,8})?(?:/[A-Z]{2,8})?)\s+[A-Z]?\d+[A-Z]{0,2})\b"
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


# Department code alias table — maps student shorthand to UCI catalog codes.
# Students commonly abbreviate; the catalog uses full department codes.
# False-positive lookups for non-existent aliased codes are harmless —
# Chroma returns an empty result, not an error. Extend this table as needed.
_DEPT_ALIASES: dict[str, list[str]] = {
    "CS":   ["COMPSCI"],
    "CSCI": ["COMPSCI"],
    "ICS":  ["I&C SCI"],
    "INFX": ["IN4MATX"],
    "INFO": ["IN4MATX"],
    "PHYS": ["PHYSICS"],
    "BIO":  ["BIO SCI"],
    "STAT": ["STATS"],
}


def _expand_course_codes(codes: list[str]) -> list[str]:
    """Expand extracted codes with aliased department variants.

    e.g. ["CS 161"]  → ["CS 161",  "COMPSCI 161"]
         ["ICS 6B"]  → ["ICS 6B",  "I&C SCI 6B"]
    """
    expanded = list(codes)
    seen = set(codes)
    for code in codes:
        tokens = code.split()
        # Course number is the last token that contains a digit
        num_idx = next(
            (i for i in range(len(tokens) - 1, -1, -1) if any(c.isdigit() for c in tokens[i])),
            None,
        )
        if num_idx is None or num_idx == 0:
            continue
        dept = " ".join(tokens[:num_idx])
        number = tokens[num_idx]
        for alias in _DEPT_ALIASES.get(dept, []):
            aliased = f"{alias} {number}"
            if aliased not in seen:
                expanded.append(aliased)
                seen.add(aliased)
    return expanded


def _direct_course_lookup(codes: list[str]) -> list[Document]:
    """
    Fetch course documents from the courses collection by metadata filter.

    Expands codes through the alias table so student abbreviations (e.g. "CS 161")
    also match their full catalog equivalents (e.g. "COMPSCI 161"). Returns all
    chunks for every matched course — not just chunk_0.
    """
    if not codes or "courses" not in _collections:
        return []

    all_codes = _expand_course_codes(codes)
    collection = _collections["courses"]
    result = collection.get(
        where={"code": {"$in": all_codes}},
        include=["documents", "metadatas"],
    )
    docs = []
    for text, meta in zip(result["documents"], result["metadatas"]):
        if text:
            docs.append(Document(page_content=text, metadata=meta or {}))
    return docs


async def retrieve(question: str) -> list[Document]:
    """
    Full retrieval pipeline for a standalone question:
      route → parallel retrieval → dedup → rerank (or order-preserve for requirements)
    Returns up to top_n reranked Documents with source metadata attached.
    """
    # Step 1: direct lookup for any course codes mentioned in the question
    codes = _extract_course_codes(question)
    direct_docs = _direct_course_lookup(codes)

    # Step 2: semantic retrieval (with optional major keyword / requirements filter)
    collections, major_keyword, requires_full_requirements = await _route(question)
    semantic_docs = await _retrieve_parallel(
        question, collections, major_keyword, requires_full_requirements
    )

    # Step 3: merge — direct docs first (guaranteed relevant), then semantic.
    # _dedup preserves order, so direct-lookup results keep their priority.
    all_docs = _dedup(direct_docs + semantic_docs)

    # Step 4: rerank — or skip for requirements queries
    if requires_full_requirements:
        # Requirements chunks are already pre-filtered to the exact section via the
        # $and filter. The reranker penalises dense course-code lists in favour of
        # prose, reversing the ordering we want. Return as-is, capped to avoid
        # overwhelming the context window.
        return all_docs[:20]

    top_n = 25 if major_keyword else 20
    return _rerank(all_docs, question, top_n=top_n)
