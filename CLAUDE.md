# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

A conversational RAG chatbot that helps UCI students select courses, navigate academic policies, and understand course relationships (prerequisites, corequisites, restrictions). Data comes from crawling UCI's course catalog, policy websites, and school major/minor pages.

## Running the Code

All scripts are CLI-first — no editing source files to change parameters.

**Activate the virtual environment first (Python 3.10.10):**
```bash
.venv\Scripts\activate
```

**Crawl the UCI course catalog:**
```bash
python crawler/crawler.py https://catalogue.uci.edu/allcourses --type course_catalog
# Output goes to data/raw/courses/ by default
```

**Crawl student-facing policy sites (run all of these for full policy coverage):**
```bash
# Academic regulations, graduation requirements, grading, probation
python crawler/crawler.py https://catalogue.uci.edu/informationforadmittedstudents --type policy --output data/raw/policies

# Academic integrity and student conduct
python crawler/crawler.py https://conduct.uci.edu --type policy --output data/raw/policies

# Registrar: enrollment deadlines, grading policy, final exam policy
python crawler/crawler.py https://www.reg.uci.edu/enrollment --type policy --output data/raw/policies
python crawler/crawler.py https://www.reg.uci.edu/grades --type policy --output data/raw/policies

# Do NOT crawl policies.uci.edu or ap.uci.edu — those are staff/HR administrative
# policies, not student-facing academic policies, and will pollute retrieval results.
```

**Crawl majors and minors (one command per school, all to the same output dir):**
```bash
python crawler/crawler.py https://catalogue.uci.edu/donaldbrenschoolofinformationandcomputersciences --type policy --output data/raw/majors
python crawler/crawler.py https://catalogue.uci.edu/schoolofengineering --type policy --output data/raw/majors
# Repeat for each school — verify exact URL slugs at catalogue.uci.edu before running
# Do NOT start from catalogue.uci.edu root — it would also hit /allcourses and parse them incorrectly
```

**Crawler options:** `--output`, `--delay` (default 0.5s), `--max` (default 500 pages), `--timeout` (default 10s)

**Ingest crawled data into Chroma (Ollama must be running locally):**
```bash
python ingest/ingest.py --source data/raw/courses --collection courses --db-path data/db/courses
python ingest/ingest.py --source data/raw/policies --collection policies --db-path data/db/policies
python ingest/ingest.py --source data/raw/majors --collection majors --db-path data/db/majors
```

**Ingest options:** `--chunk-size` (default 1500 chars), `--chunk-overlap` (default 150 chars), `--ollama-url`, `--ollama-model`

**Run the chatbot (requires Ollama running + `.env` with `OPENAI_API_KEY`):**
```bash
# Build the frontend first (one-time, or after any frontend change)
cd frontend && pnpm install && pnpm build && cd ..

# Start the FastAPI backend — serves both the API and the built frontend
python -m rag_chatbot.app                          # http://localhost:7860
python -m rag_chatbot.app --host 0.0.0.0 --port 7860
```

**Frontend development with hot reload (run both in parallel):**
```bash
# Terminal 1 — backend API
python -m rag_chatbot.app

# Terminal 2 — Vite dev server with proxy to backend
cd frontend && pnpm dev                            # http://localhost:5173
```
The Vite dev server proxies `/api/*` requests to `localhost:7860`, so both terminals are needed for full functionality during frontend development.

## Architecture

### Crawler (`crawler/crawler.py`)

BFS queue-based crawler scoped to the starting URL's domain + path prefix. Two crawl types controlled by `--type`:

- **`course_catalog`**: HTML pages → `parse_courses()` (targets `.courseblock` divs). PDF URLs → `parse_courses_from_pdf()` using `pdfplumber`, outputs same JSON format as HTML. Non-course HTML pages are silently skipped.
- **`policy`**: HTML pages → `extract_policy_sections()`. PDF URLs → `extract_pdf_text()` chunked by page. Used for policy sites and major/minor pages.

Output JSON schemas:
- Course page: `{type: "course_page", url, courses: [{code, title, units, description, prerequisite, corequisite, restrictions, grading_option, repeatability, extra, url, crawled_at}]}`
- Policy page: `{type: "policy_page", url, title, sections: [{heading, level, content}], crawled_at}`

**Key design decisions:**
- Course codes (e.g. `"COMPSCI 161"`) are unique identifiers and are used as Chroma document IDs. The same code appearing in multiple departments is the same cross-listed course.
- Content-type is checked before processing: `text/html` → HTML path, `application/pdf` → PDF path, anything else → skip.
- Links are only discovered from HTML pages, never from PDFs.
- `normalize_url()` strips fragments and query params. `is_allowed_url()` enforces same-domain + same-path-prefix scope — this is what scopes a majors crawl to a single school.

**`extract_policy_sections()` internals:**
Uses a recursive `_walk()` helper. Container tags (`div`, `section`, etc.) are recursed into. Heading tags (`h1`–`h6`) flush the current section and start a new one. Tables are handled atomically via `extract_table_text()` — each row becomes a pipe-delimited line (`MATH 2A | 4 Units`). This prevents `<td>` children from being double-visited.

### Ingestion (`ingest/ingest.py`)

Recursively finds all JSON files under `--source` using `rglob("*.json")`, reads the `"type"` field, and dispatches to the correct handler. Uses `upsert()` throughout so re-running is always safe.

**Course handler (`ingest_course_page`):** Builds rich document text including prerequisites, corequisites, and restrictions via `build_course_text()`. All 9 fields stored as Chroma metadata. ID = course code (e.g. `"COMPSCI 161"`). For multi-chunk courses, chunks beyond `chunk_0` are prefixed with `"[COMPSCI 161] "` so the course code travels with every chunk.

**Policy handler (`ingest_policy_page`):** Each section in the `sections` array becomes a separate Chroma document. Format: `"{page title} > {section heading}: {content}"` — prefix is repeated on every chunk. ID = SHA-1 hash of `url::heading` (stable across re-runs).

**Chunking:** Both handlers use `chunk_text(text, max_chars, overlap)` which splits at word boundaries. Default 1500 chars. Chunk IDs get a `::chunk_N` suffix; single-chunk documents keep their base ID.

**Embedding:** Ollama `nomic-embed-text` at `http://localhost:11434/api/embeddings`. Falls back to `OLLAMA_URL` / `OLLAMA_MODEL` in `.env`.

### RAG Chatbot (`rag_chatbot/`)

LangChain LCEL pipeline served via a FastAPI backend with a React frontend.

**Per-turn pipeline (`chain.py`):**
1. `_rewrite_chain` — rewrites the user message into a standalone question using conversation history. Also expands common abbreviations (e.g. `CS` → `COMPSCI`, `ICS` → `I&C SCI`). Must preserve the original question's polarity and framing exactly — never invert a question's meaning when resolving references from history.
2. `retrieve()` in `retriever.py` — routes → parallel Chroma queries → dedup → rerank (or order-preserve for requirements).
3. `_format_context` — if `file_context` is present in the input dict, prepends a `[User-Attached Document]` block before RAG sources. Then groups chunks by source URL so multi-chunk pages read as one coherent document rather than many separate `[Source N]` blocks.
4. `_answer_prompt` + main LLM — generates the cited answer.

The full chain is wrapped in `RunnableWithMessageHistory` keyed on `session_id`. Callers pass `{"input": "...", "file_context": str | None}` plus a `configurable` session ID. `file_context` flows through all LCEL steps automatically via `{**inputs}` spread — only `_format_context` consumes it.

**Answer prompt design intent (`chain.py` — `_answer_prompt`):**
The prompt is intentionally conversational — like a well-informed peer advisor, not a research analyst. Key behaviors enforced by the prompt:
- Ask a clarifying question only when the answer would be *fundamentally different* based on the response (e.g. which major for graduation requirements). Do not ask when context is sufficient.
- Never narrate what sources say or don't say ("the provided context does not state..."). Just answer naturally.
- Never self-grade or offer a "safer version" of the answer.
- Cite full URLs only when listing specific requirements or quoting exact policy language.
- If the question is out of scope (unrelated to UCI academics), briefly redirect rather than attempting to answer.

Do not revert these to clinical/analytical phrasing — prior versions caused the model to dissect its own sources instead of answering.

**Retriever (`retriever.py`) — critical design details:**

Always use `chromadb.PersistentClient` + `OllamaEmbeddingFunction` directly — never LangChain's Chroma wrapper or `langchain_ollama.OllamaEmbeddings`. These call `/api/embed` instead of `/api/embeddings` and produce incompatible query vectors.

**Router (`_RouterDecision`)** uses structured output and returns three fields:
- `collections: list[Literal["courses","policies","majors"]]` — which DBs to search
- `major_keyword: str | None` — programme name for filtering the majors collection (e.g. `"Computer Science"`)
- `requires_full_requirements: bool` — `True` when the student asks what courses are required to complete a major/minor, OR when they ask whether a specific course is required/mandatory in a major/minor. Controls which filter and reranking strategy is applied.

**Hybrid retrieval strategy — three layers working together:**

1. **Direct course lookup** (`_direct_course_lookup`): Before semantic search, course codes are extracted from the question via `_extract_course_codes()` and fetched by metadata filter (`where={"code": {"$in": all_codes}}`). This is mandatory because `nomic-embed-text` does not reliably rank specific course codes by semantic similarity.

2. **Department alias expansion** (`_DEPT_ALIASES`, `_expand_course_codes`): Students write `CS 161` but the catalog stores `COMPSCI 161`; `ICS 6B` → `I&C SCI 6B`. The alias table expands each extracted code to its catalog equivalent before lookup. Adding new aliases here is the correct fix when a department is not being found.

3. **Majors filter strategy** — branches on `requires_full_requirements`:
   - `True`: combined `$and` filter — `{"$contains": major_keyword}` AND `{"$contains": "Major Requirements"}` — scopes results exclusively to the requirements section chunks, excluding overview/admissions boilerplate and minor pages (which say "Minor Requirements"). `effective_k=100`. Reranker is **bypassed** entirely; dense course-code lists rank poorly against prose in the cross-encoder.
   - `False` with `major_keyword`: single `{"$contains": major_keyword}` filter. `effective_k=60`. Reranker runs with `top_n=25`.
   - No major: pure semantic search. Reranker runs with `top_n=20`.

**`_extract_course_codes()` must NOT uppercase the input** — `.upper()` turns common words ("take", "and") into valid-looking department codes. Only already-uppercase tokens in the original text match the regex.

**`_DB_CONFIG` in `retriever.py`** is the single place to tune per-collection retrieval depth (k values and DB paths).

**Deduplication (`_dedup`):** Keyed on `(url, chunk_id|code, sha1(content))`, not a content prefix. Prefix-based dedup was unsafe for chunks sharing a `"[COMPSCI 161] "` or `"{page title} > {heading}: "` prefix — different chunks could collide and get wrongly dropped. Used by both `_retrieve_parallel` and the final merge in `retrieve()`.

**Query embedding cache (`_embed_cached`):** `functools.lru_cache(maxsize=256)` keyed on the rewritten question string. `nomic-embed-text` is deterministic so no invalidation is needed. Values are cast to plain Python `float` — Chroma rejects `list[np.float32]` even though it accepts numpy arrays, so don't skip the cast.

**Reranker thread-safety:** `FlashrankRerank` is constructed **per call** inside `_rerank()`. Do not hoist it to module scope — mutating `top_n` on a shared instance races under concurrent FastAPI requests. Model weights are cached internally so per-call construction is cheap.

**Relevance floor:** After reranking, docs below `_RELEVANCE_FLOOR = 0.05` (`metadata["relevance_score"]`) are dropped. `_MIN_KEEP = 3` guarantees at least three docs survive so weak-but-valid queries still get context. The floor is skipped when `requires_full_requirements` is True (reranker is bypassed there anyway).

**`major_keyword` normalization:** Router output is normalized with `.strip().title()` before use in `$contains` filters. Chroma's `$contains` is case-sensitive and UCI catalogue pages use title case consistently; skipping this caused silent no-match filters.

**`_query_collection` does NOT catch exceptions.** A no-match `where_document` filter returns an empty list from Chroma, not an exception. Do not reintroduce a silent fallback — it pollutes filtered queries with unfiltered chunks and hides real Chroma bugs.

**Memory (`memory.py`):** Sliding window of 6 exchanges (12 messages). Swap `InMemoryChatMessageHistory` for Redis/SQLite here to persist across restarts without touching chain or app code.

**App (`app.py`):** FastAPI server. `POST /api/chat` accepts `multipart/form-data` with fields `message` (str), `session_id` (str), and optional `file` (UploadFile). File text is extracted by `file_parser.py` and passed as `file_context` into the chain. Streams the response as Server-Sent Events (SSE). Built React frontend is served as static files from `frontend/dist/`. CORS middleware allows the Vite dev server (`localhost:5173`) to reach the API during development. Session IDs are generated client-side — every page refresh creates a new UUID, starting a fresh LangChain session automatically.

**File parser (`file_parser.py`):** Extracts plain text from `.txt`, `.pdf`, and `.docx` uploads. PDF extraction tries pdfplumber first (text-layer PDFs), then falls back to OCR via `pypdfium2` + `pytesseract` for scanned documents. OCR requires the Tesseract binary installed at `C:\Program Files\Tesseract-OCR\tesseract.exe` (the default Windows install path — no PATH configuration needed). Output is hard-capped at 12,000 chars. All parse failures raise `ValueError` which becomes an HTTP 400.

### Frontend (`frontend/`)

React + Vite + Tailwind + shadcn/ui. Package manager is `pnpm`.

- `src/app/App.tsx` — top-level layout, all chat state, SSE fetch logic. Generates `sessionId` via `crypto.randomUUID()` on mount (resets on refresh). File uploads are sent as `multipart/form-data` via `FormData` — do **not** set `Content-Type` manually or the browser-generated multipart boundary will break. Accepted types: `.pdf`, `.docx`, `.txt` only (10 MB limit enforced client-side).
- `src/app/components/ChatMessage.tsx` — renders user and assistant bubbles. Uses `ReactMarkdown` with `remark-gfm` for markdown + autolink detection. Bare URLs output by the LLM are rendered as human-readable labels via `labelFromUrl()` (e.g. `https://catalogue.uci.edu/.../computerscience_bs` → `Catalogue · Computer Science B.S.`).
- `src/app/components/EmptyState.tsx` — shown when no messages exist; renders suggestion chips.
- Streaming: `App.tsx` reads the SSE stream chunk-by-chunk via `ReadableStream`, appending tokens to the assistant bubble in real time. Newlines are escaped as `\n` for SSE transport and unescaped on the client.

## Key Dependencies

- `requests`, `beautifulsoup4` — crawling
- `pdfplumber` — PDF text extraction (crawler + file upload text-layer path)
- `pypdfium2` — renders PDF pages to images for OCR (already in requirements; no poppler needed)
- `pytesseract` — OCR for scanned PDFs; requires Tesseract binary (`winget install UB-Mannheim.TesseractOCR` or the installer from github.com/UB-Mannheim/tesseract/wiki)
- `python-docx` — DOCX text extraction for file uploads
- `chromadb` — vector store; always use `chromadb.PersistentClient` + `OllamaEmbeddingFunction` directly
- `ollama` — local embeddings (`ollama serve` to start); embedding function is `chromadb.utils.embedding_functions.ollama_embedding_function.OllamaEmbeddingFunction`
- `langchain`, `langchain-openai`, `langchain-community` — LCEL chain, OpenAI LLMs, FlashrankRerank
- `flashrank` — local cross-encoder reranker (no API key needed)
- `fastapi`, `uvicorn` — backend API server with SSE streaming; `python-multipart` (already installed) is required for `Form`/`File` parameters
- `python-dotenv` — reads `.env` for `OPENAI_API_KEY` (copy `.env.example` to `.env`)
- Frontend: `react`, `vite`, `tailwindcss`, `react-markdown`, `remark-gfm`, `motion`

## Benchmarking (`eval/`)

Layered evaluation harnesses. Run from the project root with the venv active.

**Build auto-generated datasets** (requires `data/raw/` from a prior crawl):
```bash
python eval/build_dataset.py
# Emits eval/datasets/courses.jsonl, majors.jsonl, policies.jsonl
# eval/datasets/hard_cases.jsonl is hand-curated and already committed
```

**Run a specific harness:**
```bash
python eval/run_all.py --harness router --dataset hard --limit 20
python eval/run_all.py --harness retrieval --dataset courses --limit 50
python eval/run_all.py --harness e2e --dataset hard --limit 10 --judge
python eval/run_all.py --harness perf --limit 20
```

**Multi-turn / polarity tests via pytest:**
```bash
pytest eval/harnesses/multiturn_eval.py -v
```

**Compare two runs:**
```bash
python eval/report.py          # diffs latest vs previous run
python eval/report.py --list   # show all available runs
```

**`report.py` is not dataset-aware** — it diffs whichever two runs are most recent. Comparing a courses run vs. a majors run shows large false "regressions" because Recall@1 is inherently higher on courses (direct code lookup dominates). For before/after code changes, run the **same** dataset on both sides, or `git stash` the changes to capture a clean baseline first.

Results land in `eval/runs/<timestamp>/` (gitignored). `conftest.py` at the project root handles `sys.path` for pytest.

**Harness quick-reference:**

| Harness | Needs Chroma | Key metric |
|---|---|---|
| `router` | No | Collection F1, `requires_full_requirements` accuracy |
| `retrieval` | Yes | Recall@1/3/10, MRR, direct-lookup hit rate |
| `e2e` | Yes | Citation accuracy, field match, LLM judge faithfulness/relevance (1–5) |
| `multiturn` | Yes | Polarity preservation, pronoun resolution (pytest assertions) |
| `file` | Yes | `[User-Attached Document]` block present, answer references upload |
| `perf` | Yes | Latency p50/p95, tokens/query, cost/query |

**Critical implementation notes:**
- `router_eval.py` reproduces the router LLM inline — it does **not** import `rag_chatbot.retriever`, so it runs without Chroma DBs.
- All other harnesses import `rag_chatbot.retriever` which loads Chroma at import time. If DBs are missing they print an actionable message and exit with code 1; `multiturn_eval.py` skips via `pytestmark` instead of crashing pytest collection.
- The `_answer_prompt` in `chain.py` requires `chat_history` in its input dict. Harnesses pass `"chat_history": []` when calling `_format_context` directly (single-turn eval has no history).
- `eval/runs/` and the three auto-generated dataset files are gitignored. `hard_cases.jsonl` and `eval/fixtures/` are committed.

## Debugging

`diagnose.py` (project root) — standalone script that checks Chroma collections directly: scans ICS school pages by URL, verifies specific course IDs (COMPSCI 161, I&C SCI 6B/6D), and runs semantic queries. Run with venv active: `python diagnose.py`. Edit inline for different queries.

To add temporary debug logging, add `print()` statements to `retriever.py` (`retrieve()`) and `chain.py` (`_retrieve_step()`). Do not commit these.

## Data Notes

- `README.md` is the public-facing project overview. `CLAUDE.md` is the authoritative technical reference — do not duplicate content between them.
- Demo screenshots for the README live in `docs/screenshots/` and are committed to git (not LFS). The project logo is at `ZotAssistantLogo.png` in the repo root.
- `data/raw/` is gitignored — rebuild by re-running the crawler commands above.
- `data/db/` (Chroma vector databases) is tracked via **Git LFS** — `*.sqlite3` and `*.bin` files are stored in LFS. Run `git lfs install` before cloning if you need the databases locally. If LFS files are missing, re-run ingest to rebuild.
- ~5,920 courses across 118 departments on `catalogue.uci.edu/allcourses`.
- Scanned/image-based PDFs produce no extractable text in the crawler and are silently skipped there. The file upload path handles them via OCR fallback (`file_parser.py`).
- Chroma does not support graph-style relationships. Course-to-major relationships are retrievable through RAG because major requirement pages mention course codes directly in their ingested text.
- After any change to `ingest.py`, re-run the affected collection's ingest command — `upsert()` makes this safe and idempotent.
- The majors collection must be crawled per-school (one command per school URL). Do not crawl from the catalogue root.
- The policies collection must be built from student-facing sources only (see crawl commands above). The `policies.uci.edu` and `ap.uci.edu` domains contain staff/HR administrative content that degrades retrieval quality for student questions.
