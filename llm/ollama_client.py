import json
from typing import Dict, List

import requests


class OllamaClient:
    """
    HTTP client for the local Ollama inference server.

    Two interaction modes:

    chat(messages)   — sends a structured message list to /api/chat.
                       Used by the agent loop for all normal interactions.
                       Supports system prompts, conversation history, and
                       tool-result injection via the messages array.

    generate(prompt) — sends a raw string prompt to /api/generate.
                       Kept for backward compatibility and quick testing.
    """

    def __init__(
        self,
        model: str = "llama3.2:latest",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._generate_url = f"{self._base_url}/api/generate"
        self._chat_url = f"{self._base_url}/api/chat"

    # -----------------------------------------------------------------------
    # Primary: structured chat
    # -----------------------------------------------------------------------

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Send a structured message list to Ollama's /api/chat endpoint.

        Args:
            messages: List of dicts with 'role' ('system'|'user'|'assistant')
                      and 'content' keys.

        Returns:
            The assistant's reply as a plain string.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        response = requests.post(self._chat_url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"]

    # -----------------------------------------------------------------------
    # Legacy: raw prompt generation
    # -----------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Send a raw text prompt to /api/generate.

        Preserved for backward compatibility and standalone testing.
        For agent-driven interactions, use chat() instead.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        response = requests.post(self._generate_url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["response"]
