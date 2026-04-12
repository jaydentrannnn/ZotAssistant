"""
FastAPI backend for the UCI Academic Assistant.

Usage:
    python -m rag_chatbot.app
    python -m rag_chatbot.app --port 8080
    python -m rag_chatbot.app --host 0.0.0.0 --port 7860

Serves the React frontend from frontend/dist/ and exposes POST /api/chat
as a Server-Sent Events stream. Each request carries its own session_id
(generated client-side as a UUID on page load) so every browser tab is
isolated and every refresh starts a fresh conversation.
"""

import argparse
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .chain import chain
from .file_parser import extract_text

app = FastAPI(title="UCI Academic Assistant")

# Allow the Vite dev server to reach the API during local development.
# In production the frontend is served from the same origin, so this is a no-op.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Chat endpoint — SSE streaming
# ──────────────────────────────────────────────────────────────────────────────

async def _sse_stream(message: str, session_id: str, file_context: Optional[str]):
    """Yield SSE-formatted token chunks from the LangChain chain."""
    async for chunk in chain.astream(
        {"input": message, "file_context": file_context},
        config={"configurable": {"session_id": session_id}},
    ):
        # Escape newlines so each event stays on a single line
        escaped = chunk.replace("\n", "\\n")
        yield f"data: {escaped}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def chat(
    message: str = Form(default=""),
    session_id: str = Form(...),
    file: Optional[UploadFile] = File(default=None),
):
    file_context: Optional[str] = None
    if file and file.filename:
        try:
            file_context = await extract_text(file)
        except ValueError as e:
            print(f"[file upload error] {e}")
            raise HTTPException(status_code=400, detail=str(e))

    if not message.strip() and not file_context:
        raise HTTPException(status_code=400, detail="Empty message")

    return StreamingResponse(
        _sse_stream(message, session_id, file_context),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Serve the built React frontend (SPA fallback to index.html)
# ──────────────────────────────────────────────────────────────────────────────

_DIST = Path(__file__).parent.parent / "frontend" / "dist"

# Mount /assets separately so Starlette serves hashed JS/CSS files efficiently
_ASSETS = _DIST / "assets"
if _ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=str(_ASSETS)), name="assets")


@app.get("/")
async def serve_index():
    index = _DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"error": "Frontend not built. Run `pnpm build` inside frontend/."}


@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """Serve static files by path; fall back to index.html for client-side routes."""
    candidate = _DIST / full_path
    if candidate.is_file():
        return FileResponse(candidate)
    index = _DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"error": "Frontend not built. Run `pnpm build` inside frontend/."}


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the UCI Academic Assistant web server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
