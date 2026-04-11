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
            "If the message is already self-contained, return it unchanged. "
            "Output only the rewritten question — no explanation.",
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
    if not docs:
        context = "No relevant information found in the UCI academic database."
    else:
        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            # Courses store URL directly; policy/major sections store under "url"
            url = meta.get("url", "")
            label = url if url else "UCI Academic Resources"
            parts.append(f"[Source {i} — {label}]\n{doc.page_content}")
        context = "\n\n---\n\n".join(parts)
    return {**inputs, "context": context}


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Final answer prompt + main LLM
# ──────────────────────────────────────────────────────────────────────────────

_answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a concise academic assistant for UCI (University of California, Irvine) students. "
            "Answer questions about courses, academic policies, and major/minor requirements "
            "using only the information in the provided context. "
            "\n\n"
            "Guidelines:\n"
            "- Be direct and brief. Get to the answer immediately — no preamble.\n"
            "- Use bullet points for lists of courses or requirements; prose for simple yes/no answers.\n"
            "- Cite source URLs inline only when directly quoting or listing requirements.\n"
            "- If the context is insufficient, say so in one sentence and point to the UCI catalogue or an advisor.\n"
            "- Never repeat the question back. Never explain what you are about to do.\n"
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
