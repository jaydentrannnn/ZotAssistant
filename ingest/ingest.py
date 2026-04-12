"""
Ingest crawled JSON data into a Chroma vector store.

Usage:
    python ingest/ingest.py --source data/raw/courses --collection courses --db-path data/db/courses
    python ingest/ingest.py --source data/raw/policies --collection policies --db-path data/db/policies
    python ingest/ingest.py --source data/raw/majors --collection majors --db-path data/db/majors
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks of at most max_chars characters.
    Splits on word boundaries so words are never cut mid-way.
    Returns a single-item list if the text already fits within max_chars.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start  = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Walk back to the nearest word boundary
        boundary = text.rfind(" ", start, end)
        if boundary <= start:
            boundary = end  # No space found — hard cut
        chunks.append(text[start:boundary])
        start = boundary - overlap
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Course handler
# ──────────────────────────────────────────────────────────────────────────────

def build_course_text(course: dict) -> str:
    """
    Build rich document text for a course so that prerequisites, corequisites,
    and restrictions are part of what gets semantically embedded — not just
    the title and description.
    """
    lines = [f"{course.get('code', '')} - {course.get('title', '')} ({course.get('units', '')} units)"]
    if course.get("description"):
        lines.append(f"Description: {course['description']}")
    if course.get("prerequisite"):
        lines.append(f"Prerequisite: {course['prerequisite']}")
    if course.get("corequisite"):
        lines.append(f"Corequisite: {course['corequisite']}")
    if course.get("restrictions"):
        lines.append(f"Restrictions: {course['restrictions']}")
    if course.get("grading_option"):
        lines.append(f"Grading Option: {course['grading_option']}")
    if course.get("repeatability"):
        lines.append(f"Repeatability: {course['repeatability']}")
    return "\n".join(lines)


def ingest_course_page(page: dict, collection, chunk_size: int, chunk_overlap: int) -> int:
    """
    Ingest all courses from a course_page JSON.
    Each course is one Chroma document keyed by course code.
    Long descriptions are chunked; each chunk gets an id suffix (::chunk_N).
    Returns the number of documents ingested.
    """
    courses = page.get("courses", [])
    if not courses:
        return 0

    ids, documents, metadatas = [], [], []
    for course in courses:
        code = course.get("code", "").strip()
        if not code:
            continue

        metadata = {
            "code":           course.get("code", ""),
            "title":          course.get("title", ""),
            "units":          course.get("units", ""),
            "prerequisite":   course.get("prerequisite", ""),
            "corequisite":    course.get("corequisite", ""),
            "restrictions":   course.get("restrictions", ""),
            "grading_option": course.get("grading_option", ""),
            "repeatability":  course.get("repeatability", ""),
            "url":            course.get("url", ""),
        }

        chunks = chunk_text(build_course_text(course), chunk_size, chunk_overlap)
        # Prepend the course code to every chunk beyond the first so that
        # semantic search can match later chunks (e.g. "Prerequisite: MATH 2B")
        # back to the correct course — mirrors what the policy handler does.
        prefix = f"[{code}] "
        for n, chunk in enumerate(chunks):
            doc_id = code if len(chunks) == 1 else f"{code}::chunk_{n}"
            ids.append(doc_id)
            documents.append(chunk if n == 0 else prefix + chunk)
            metadatas.append(metadata)

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


# ──────────────────────────────────────────────────────────────────────────────
# Policy / majors handler
# ──────────────────────────────────────────────────────────────────────────────

def _section_id(url: str, heading: str, suffix: str = "") -> str:
    """Generate a stable 16-char ID for a policy section from its URL + heading."""
    key = f"{url}::{heading}{suffix}".encode()
    return hashlib.sha1(key).hexdigest()[:16]


def ingest_policy_page(page: dict, collection, chunk_size: int, chunk_overlap: int) -> int:
    """
    Ingest each section of a policy_page as a separate Chroma document.
    Long sections are split into overlapping chunks. Each chunk retains the
    'Title > Heading' prefix so retrieval context is never lost.
    Returns the number of documents ingested.
    """
    sections = page.get("sections", [])
    if not sections:
        return 0

    url    = page.get("url", "")
    title  = page.get("title", "")
    source = url.split("/")[2] if url.startswith("http") else ""

    ids, documents, metadatas = [], [], []
    seen_ids: set[str] = set()

    for section in sections:
        heading = section.get("heading", "")
        content = section.get("content", "").strip()
        if not content:
            continue

        base_id = _section_id(url, heading)
        if base_id in seen_ids:
            base_id = _section_id(url, heading, suffix=f"::{len(seen_ids)}")
        seen_ids.add(base_id)

        # Prefix travels with every chunk so each chunk is self-contained
        prefix = f"{title} > {heading}: " if title else f"{heading}: "
        chunks = chunk_text(content, chunk_size - len(prefix), chunk_overlap)

        metadata = {
            "url":     url,
            "title":   title,
            "heading": heading,
            "level":   section.get("level", ""),
            "source":  source,
        }

        for n, chunk in enumerate(chunks):
            doc_id = base_id if len(chunks) == 1 else f"{base_id}::chunk_{n}"
            ids.append(doc_id)
            documents.append(prefix + chunk)
            metadatas.append(metadata)

    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

_HANDLERS = {
    "course_page": ingest_course_page,
    "policy_page": ingest_policy_page,
}


def ingest(source: Path, collection, chunk_size: int, chunk_overlap: int) -> None:
    """Recursively find all JSON files under source and ingest them."""
    files = sorted(source.rglob("*.json"))
    if not files:
        print(f"No JSON files found under {source}")
        return

    total = 0
    for i, filepath in enumerate(files, start=1):
        try:
            with open(filepath, encoding="utf-8") as f:
                page = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [skip] {filepath}: {e}")
            continue

        page_type = page.get("type")
        handler   = _HANDLERS.get(page_type)
        if handler is None:
            print(f"  [skip] Unknown type '{page_type}': {filepath.name}")
            continue

        count  = handler(page, collection, chunk_size, chunk_overlap)
        total += count
        print(f"[{i}/{len(files)}] {filepath.name} — {count} document(s)")

    print(f"\nDone. Ingested {total} documents into '{collection.name}'.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest crawled JSON data into a Chroma vector store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", required=True,
        help="Root directory of crawled JSON files (searched recursively)",
    )
    parser.add_argument(
        "--collection", required=True,
        help="Chroma collection name (e.g. courses, policies, majors)",
    )
    parser.add_argument(
        "--db-path", required=True,
        help="Directory to persist the Chroma database (e.g. data/db/courses)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1500,
        help="Max characters per embedded chunk (default fits within nomic-embed-text's context)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=150,
        help="Character overlap between consecutive chunks to preserve context across boundaries",
    )
    parser.add_argument(
        "--ollama-url", default=None,
        help="Ollama embeddings endpoint (falls back to OLLAMA_URL in .env, then http://localhost:11434/api/embeddings)",
    )
    parser.add_argument(
        "--ollama-model", default=None,
        help="Ollama embedding model (falls back to OLLAMA_MODEL in .env, then nomic-embed-text)",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        print(f"Error: '{source}' is not a valid directory.")
        raise SystemExit(1)

    ollama_url   = args.ollama_url   or os.getenv("OLLAMA_URL",   "http://localhost:11434/api/embeddings")
    ollama_model = args.ollama_model or os.getenv("OLLAMA_MODEL", "nomic-embed-text")

    print(f"Source       : {source.resolve()}")
    print(f"Collection   : {args.collection}")
    print(f"DB path      : {args.db_path}")
    print(f"Chunk size   : {args.chunk_size} chars  |  Overlap: {args.chunk_overlap} chars")
    print(f"Embedding    : {ollama_model} @ {ollama_url}")
    print()

    embedding_fn = OllamaEmbeddingFunction(url=ollama_url, model_name=ollama_model)
    client       = chromadb.PersistentClient(path=args.db_path)
    collection   = client.get_or_create_collection(
        name=args.collection,
        embedding_function=embedding_fn,
    )

    ingest(source, collection, args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()
