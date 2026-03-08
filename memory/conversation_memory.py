from typing import Dict, List

from core.config import MEMORY_MAX_MESSAGES


class ConversationMemory:
    """
    Stores the ongoing conversation as an ordered list of messages.

    Each message is a dict with 'role' ('user' or 'assistant') and 'content'.
    History is capped at MEMORY_MAX_MESSAGES to keep the LLM context window
    manageable and CPU inference times predictable.

    Future extension points:
        - Persist to disk (JSON, SQLite) for cross-session memory
        - Semantic search over older messages via a vector store
        - Named memory slots: memory.remember("user_name", "Alex")
    """

    def __init__(self, max_messages: int = MEMORY_MAX_MESSAGES) -> None:
        self._history: List[Dict[str, str]] = []
        self._max = max_messages

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_user_message(self, text: str) -> None:
        """Append a user turn to the history."""
        self._history.append({"role": "user", "content": text})
        self._trim()

    def add_assistant_message(self, text: str) -> None:
        """Append a BMO turn to the history."""
        self._history.append({"role": "assistant", "content": text})
        self._trim()

    def get_history(self) -> List[Dict[str, str]]:
        """Return a copy of the full message history."""
        return list(self._history)

    def clear(self) -> None:
        """Wipe all conversation history."""
        self._history = []
        print("[MEMORY] Conversation history cleared.")

    def __len__(self) -> int:
        return len(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim(self) -> None:
        """Keep only the most recent _max messages."""
        if len(self._history) > self._max:
            removed = len(self._history) - self._max
            self._history = self._history[-self._max:]
            print(f"[MEMORY] Trimmed {removed} old message(s) to stay within limit.")
