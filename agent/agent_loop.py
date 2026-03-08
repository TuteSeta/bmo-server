"""
BMO Agent Loop — the reasoning core.

Replaces the direct LLM call in VoicePipeline.run_once() with a
tool-aware iteration cycle:

    user input
        ↓
    build prompt  (system + history + user message)
        ↓
    call LLM
        ↓
    tool call detected?  ── yes ──►  execute tool
        │                                  ↓
        no                        inject result into messages
        ↓                                  ↓
    plain text response  ◄────── call LLM again (max 3 iterations)
        ↓
    store in memory → return to TTS

The loop is capped at AGENT_MAX_ITERATIONS to prevent runaway chains.
If the cap is hit without a plain-text answer, a safe fallback is returned.

Future extension points:
    - agent/planner.py  — break complex goals into sub-tasks
    - memory/knowledge.py — vector search over long-term facts
    - agent/intent_router.py — fast-path routing before hitting the LLM
"""

import json
import re
from typing import Dict, List, Optional

import tools.system_tools  # noqa: F401 — registers all tools on import

from core.config import AGENT_MAX_ITERATIONS
from core.logger import get_logger
from llm.ollama_client import OllamaClient
from llm.prompt_builder import build_messages
from memory.conversation_memory import ConversationMemory
from tools.registry import execute_tool

logger = get_logger(__name__)


def spanish_style(text: str) -> str:
    """Strip any 'BMO:' prefix the model may add before the response text."""
    return text.replace("BMO:", "").strip()


class AgentLoop:
    """
    Drives a single user turn through the reasoning cycle.

    Each call to run() is one complete user exchange:
        - reads history from ConversationMemory
        - may iterate multiple times if tools are invoked
        - writes the final user input and BMO's response back to memory
    """

    def __init__(self, llm: OllamaClient, memory: ConversationMemory) -> None:
        self.llm = llm
        self.memory = memory

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """
        Process one user turn and return the final text response.

        Args:
            user_input: Transcribed speech from the STT engine.

        Returns:
            Plain text response to be handed to TTS.
        """
        logger.debug("User input: %s", user_input)

        # Build the fresh message list for this turn from stored history.
        # We do NOT write to memory yet — we wait for the final answer.
        history = self.memory.get_history()
        messages: List[Dict[str, str]] = build_messages(user_input, history)

        response: Optional[str] = None

        for iteration in range(AGENT_MAX_ITERATIONS):
            raw = self.llm.chat(messages)
            logger.debug("LLM response: %s", raw)

            tool_call = self._parse_tool_call(raw)

            if tool_call is None:
                # The LLM replied in plain text — we are done.
                response = raw.strip()
                break

            # --- Tool execution branch ---
            tool_name = tool_call.get("tool", "")
            tool_args = tool_call.get("args", {})

            logger.debug("Executing tool: %s", tool_name)

            tool_result = execute_tool(tool_name, tool_args)
            logger.debug("Tool result: %s", tool_result)

            # Fold the tool exchange into the local messages list so the
            # LLM has full context for the next iteration.
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "tool",
                "content": tool_result,
            })

        if response is None:
            response = (
                "Llegué al límite de mi razonamiento. "
                "Intenta preguntarme de otra manera."
            )

        # Persist the exchange to long-term conversation memory.
        self.memory.add_user_message(user_input)
        self.memory.add_assistant_message(response)

        return spanish_style(response)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _parse_tool_call(self, text: str) -> Optional[Dict]:
        """
        Robustly extract a tool call JSON from the LLM response.

        Strategy:
        1. Try json.loads() on the full stripped response.
        2. If that fails, use regex to find the first {...} block (greedy,
           to capture nested args) and try parsing that.
        3. Only accept dicts that contain BOTH "tool" and "args" keys.
        4. Return None if no valid tool call is found.
        """
        stripped = text.strip()

        # Strategy 1: parse the whole response directly.
        try:
            data = json.loads(stripped)
            if isinstance(data, dict) and "tool" in data and "args" in data:
                return data
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: extract the outermost {...} block via greedy regex,
        # which handles args that contain nested objects.
        match = re.search(r'\{.*\}', stripped, re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and "tool" in data and "args" in data:
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

        return None
