"""
Session memory management.

Each browser session gets a unique session_id (UUID generated in app.py).
LangChain's RunnableWithMessageHistory calls get_session_history() to load
and save per-session chat history. Sliding window trims old turns so the
rewrite prompt doesn't balloon with irrelevant history.
"""

from langchain_core.chat_history import InMemoryChatMessageHistory

# In-process store: session_id → message history
# For multi-process / persistent deployments swap this for a Redis or
# SQLite backend without touching chain.py or app.py.
_store: dict[str, InMemoryChatMessageHistory] = {}

# Keep the last N complete exchanges (each exchange = 1 human + 1 AI message)
_MAX_EXCHANGES = 6
_MAX_MESSAGES = _MAX_EXCHANGES * 2


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Return the message history for this session, creating it if needed."""
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    history = _store[session_id]
    # Trim to sliding window — remove oldest pairs when over the limit
    if len(history.messages) > _MAX_MESSAGES:
        history.messages = history.messages[-_MAX_MESSAGES:]
    return history
