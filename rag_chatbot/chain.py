"""
LangChain LCEL chain assembly.

Full pipeline per user turn:
  1. History-aware rewrite  (gpt-5.4-mini) → standalone question
  2. Route + retrieve + rerank             → top 6 Documents
  3. Context formatting                    → cited text block
  4. Final answer             (gpt-5.4)    → streamed response

Wrapped with RunnableWithMessageHistory so session memory is handled
automatically by LangChain — callers only pass {"input": "..."} and
a configurable session_id.
"""

import os
from collections import defaultdict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .memory import get_session_history
from .retriever import retrieve

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# LLMs
# ──────────────────────────────────────────────────────────────────────────────

_mini_llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.2)
_main_llm = ChatOpenAI(model="gpt-5.4", temperature=0.5, streaming=True)

# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — History-aware query rewriting
# ──────────────────────────────────────────────────────────────────────────────

_rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a query rewriter for a UCI academic assistant. "
            "Given the conversation history and the student's latest message, "
            "rewrite the message as a fully self-contained question that includes "
            "all necessary context (course codes, major names, policy topics, etc.) "
            "so it can be answered without the conversation history. "
            "Expand common UCI course abbreviations to their official department codes "
            "(e.g., CS to COMPSCI, ICS to I&C SCI, INF to IN4MATX, SE to SWE, MAE to ENGRMAE, CEE to ENGRCEE). "
            "Ensure course codes are fully capitalized. "
            "If the message is already self-contained, return it unchanged. "
            "CRITICAL rules — you must follow these exactly:\n"
            "1. Preserve the original question's polarity and framing without exception. "
            "A question like 'do I have to take X' must remain a question about whether the student has to take X — "
            "never reframe it as 'is it true that X is not required' or any negative or inverted form.\n"
            "2. Use conversation history only to resolve ambiguous references such as pronouns or implied course names "
            "(e.g. 'it' → 'COMPSCI 161'). Never reference, repeat, challenge, or build on what a previous answer said.\n"
            "3. Output only the rewritten question — no explanation, no preamble.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

_rewrite_chain = _rewrite_prompt | _mini_llm | StrOutputParser()

# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Retrieve (async RunnableLambda wrapping the retriever pipeline)
# ──────────────────────────────────────────────────────────────────────────────

async def _retrieve_step(inputs: dict) -> dict:
    docs: list[Document] = await retrieve(inputs["standalone_question"])
    return {**inputs, "docs": docs}


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Format retrieved context with source citations
# ──────────────────────────────────────────────────────────────────────────────

def _format_context(inputs: dict) -> dict:
    docs: list[Document] = inputs.get("docs", [])
    file_context: str | None = inputs.get("file_context")

    parts: list[str] = []

    # User-attached document goes first so the LLM sees it before any RAG sources.
    if file_context:
        parts.append(f"[User-Attached Document]\n{file_context}")

    if not docs:
        if not file_context:
            parts.append("No relevant information found in the UCI academic database.")
    else:
        # Group chunks by source URL so that multi-chunk pages (e.g. the full CS
        # major requirements split across 6 chunks) read as one coherent document
        # rather than 6 separate [Source N] blocks. This prevents the LLM from
        # treating consecutive requirement chunks as unrelated sources.
        # dict preserves insertion order (Python 3.7+), so retrieval ranking is kept.
        groups: dict[str, list[Document]] = defaultdict(list)
        for doc in docs:
            url = doc.metadata.get("url", "") or "UCI Academic Resources"
            groups[url].append(doc)

        for i, (url, group_docs) in enumerate(groups.items(), 1):
            combined = "\n\n".join(d.page_content for d in group_docs)
            parts.append(f"[Source {i} — {url}]\n{combined}")

    context = "\n\n---\n\n".join(parts)
    return {**inputs, "context": context}


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Final answer prompt + main LLM
# ──────────────────────────────────────────────────────────────────────────────

_answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a friendly, knowledgeable academic assistant for UCI (University of California, Irvine) students. "
            "Answer questions about courses, academic policies, and major/minor requirements "
            "conversationally and directly, like a well-informed peer advisor.\n\n"
            "Guidelines:\n"
            "- Get straight to the answer — no preamble.\n"
            "- Use bullet points for lists of courses or requirements; prose for simple questions.\n"
            "- For multi-part questions, answer each part in order.\n"
            "- If you have partial information, give what you know, then add one brief sentence "
            "directing the student where to find the rest (e.g. 'Check your program page in the "
            "UCI Catalogue or ask your advisor.').\n"
            "- Only ask a clarifying question when the answer would be fundamentally different "
            "depending on the response (e.g. asking which major when graduation requirements vary "
            "by program). Do not ask when you already have enough context to give a useful answer.\n"
            "- Never narrate what your sources do or don't say. Never say 'the provided context "
            "does not state' or 'this is not supported by the context.' Just answer naturally.\n"
            "- Never analyze or grade your own answer.\n"
            "- When citing a source, use the full URL in parentheses, e.g. (https://catalogue.uci.edu/...). "
            "Cite only when listing specific requirements or quoting exact policy language.\n"
            "- Never repeat the question back. Never explain what you are about to do.\n"
            "- If the question is unrelated to UCI academics (courses, policies, majors, or student life), "
            "briefly say so and redirect the student to the appropriate resource.\n"
            "- Do not fabricate course names, codes, prerequisites, or policy details.",
        ),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {standalone_question}",
        ),
    ]
)

# ──────────────────────────────────────────────────────────────────────────────
# Chain assembly (LCEL)
# ──────────────────────────────────────────────────────────────────────────────

_base_chain = (
    RunnablePassthrough.assign(standalone_question=_rewrite_chain)
    | RunnableLambda(_retrieve_step)
    | RunnableLambda(_format_context)
    | _answer_prompt
    | _main_llm
    | StrOutputParser()
)

# ──────────────────────────────────────────────────────────────────────────────
# Public chain — wrap with session memory
# ──────────────────────────────────────────────────────────────────────────────

chain = RunnableWithMessageHistory(
    _base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
"""
Invoke:
    response = await chain.ainvoke(
        {"input": "What are the prereqs for CS 161?"},
        config={"configurable": {"session_id": "abc-123"}},
    )

Stream:
    async for chunk in chain.astream(
        {"input": "What are the prereqs for CS 161?"},
        config={"configurable": {"session_id": "abc-123"}},
    ):
        print(chunk, end="", flush=True)
"""
