"""
BMO Agent Loop — the reasoning core.

Pipeline:

    user input
        ↓
    build prompt  (system + history + user message)
        ↓
    call LLM
        ↓
    tool call detected?  ── yes ──►  execute tool
        │                                  ↓
        no                        summarize result (separate LLM call)
        ↓                                  ↓
    clean for speech  ◄──────────  return spoken summary
        ↓
    store in memory → return to TTS

Key behaviors adopted from be-more-agent:

    1. After a tool executes, a *separate* LLM call summarizes the result
       into one short spoken sentence — the main conversation is not polluted.
    2. Tool‑call JSON is extracted with multiple fallback strategies:
       direct parse → code‑fence strip → greedy regex → key scan.
    3. Known tool aliases are resolved so the LLM can say "google X"
       and it maps to "search_web".
    4. Sentinel results (invalid action, errors) are handled with fixed
       spoken responses — no extra LLM call needed.
    5. All output is cleaned for TTS: no markdown, no code fences,
       no special characters.
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
from tools.registry import execute_tool, list_tool_names

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Aliases — let the LLM use natural names for tools.
# If the model outputs {"tool": "hora", ...} we resolve it to "get_time".
# ---------------------------------------------------------------------------
TOOL_ALIASES: Dict[str, str] = {
    "hora":          "get_time",
    "tiempo":        "get_time",
    "time":          "get_time",
    "check_time":    "get_time",
    "sistema":       "get_system_status",
    "system":        "get_system_status",
    "status":        "get_system_status",
    "docker":        "list_docker_containers",
    "contenedores":  "list_docker_containers",
    "containers":    "list_docker_containers",
}


def _clean_for_speech(text: str) -> str:
    """
    Strip markdown artifacts and prefixes so the text is safe for TTS.

    Handles: code fences, bold/italic markers, bullet points, numbered lists,
    header markers, angle‑bracket quotes, and the 'BMO:' prefix the model
    sometimes prepends.
    """
    # Remove code fences (```…```)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Remove inline code (`…`)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Remove markdown markers: * _ # > ~
    text = re.sub(r"[*_#>~]", "", text)
    # Remove bullet points at line start
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    # Remove numbered list prefixes
    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)
    # Remove characters that don't belong in spoken output
    text = re.sub(r"[{}\[\]|\\]", "", text)
    # Strip "BMO:" prefix
    text = re.sub(r"^BMO:\s*", "", text.strip())
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


class AgentLoop:
    """
    Drives a single user turn through the reasoning cycle.

    Each call to run() is one complete user exchange:
        - reads history from ConversationMemory
        - may iterate multiple times if tools are invoked
        - writes the final user+assistant exchange back to memory
    """

    def __init__(self, llm: OllamaClient, memory: ConversationMemory) -> None:
        self.llm = llm
        self.memory = memory

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """
        Process one user turn and return a spoken response.

        Args:
            user_input: Transcribed speech from the STT engine.

        Returns:
            Plain‑text response ready for TTS.
        """
        logger.debug("User input: %s", user_input)

        history = self.memory.get_history()
        messages: List[Dict[str, str]] = build_messages(user_input, history)

        response: Optional[str] = None

        for iteration in range(AGENT_MAX_ITERATIONS):
            raw = self.llm.chat(messages)
            logger.debug("LLM response (iter %d): %s", iteration, raw)

            tool_call = self._parse_tool_call(raw)

            if tool_call is None:
                # Plain text — the LLM chose not to use a tool.
                response = _clean_for_speech(raw)
                break

            # --- Tool execution branch ---
            tool_name = self._resolve_tool_name(tool_call.get("tool", ""))
            tool_args = tool_call.get("args", {})
            # Also accept "value"/"query" keys (be-more-agent compat)
            if not tool_args:
                for alt_key in ("value", "query"):
                    if alt_key in tool_call:
                        tool_args = {"value": tool_call[alt_key]}
                        break

            logger.info("Executing tool: %s(%s)", tool_name, tool_args)

            # Check if the tool is valid before executing.
            if tool_name not in list_tool_names():
                logger.warning("Unknown tool: %s", tool_name)
                response = "No sé cómo hacer eso todavía."
                break

            tool_result = execute_tool(tool_name, tool_args)
            logger.debug("Tool result: %s", tool_result)

            # Summarize the result in a separate LLM call — this is the
            # key pattern from be-more-agent that produces clean speech.
            response = self._summarize_tool_result(
                tool_result, user_input, tool_name
            )
            break  # We have a spoken response — done.

        if response is None:
            response = (
                "Llegué al límite de mi razonamiento. "
                "Intenta preguntarme de otra manera."
            )

        # Persist the exchange to conversation memory.
        self.memory.add_user_message(user_input)
        self.memory.add_assistant_message(response)

        return response

    # -------------------------------------------------------------------
    # Tool‑call parsing
    # -------------------------------------------------------------------

    def _parse_tool_call(self, text: str) -> Optional[Dict]:
        """
        Robustly extract a tool‑call JSON from the LLM response.

        Strategies (tried in order):
            1. json.loads() on the full stripped response.
            2. Strip markdown code fences, then json.loads().
            3. Greedy regex to find the outermost {…} block.
            4. Scan for known tool‑call keys as a last resort.
        Returns None if no valid tool call is found.
        """
        stripped = text.strip()

        # Strategy 1: direct parse of the full response.
        parsed = self._try_parse_json(stripped)
        if parsed is not None:
            return parsed

        # Strategy 2: strip markdown code fences first.
        #   LLMs often wrap JSON in ```json … ``` blocks.
        defenced = re.sub(r"```(?:json)?\s*", "", stripped)
        defenced = defenced.replace("```", "").strip()
        if defenced != stripped:
            parsed = self._try_parse_json(defenced)
            if parsed is not None:
                return parsed

        # Strategy 3: extract the outermost {…} via greedy regex.
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match:
            parsed = self._try_parse_json(match.group(0))
            if parsed is not None:
                return parsed

        # Strategy 4: the LLM wrote something like:
        #   "Voy a revisar la hora. {"tool": "get_time", "args": {}}"
        # Try each {…} candidate individually (non‑greedy).
        for m in re.finditer(r"\{[^{}]*\}", stripped):
            parsed = self._try_parse_json(m.group(0))
            if parsed is not None:
                return parsed

        return None

    @staticmethod
    def _try_parse_json(text: str) -> Optional[Dict]:
        """
        Attempt to parse text as JSON and validate it looks like a tool call.

        Accepts two formats:
            BMO format:  {"tool": "name", "args": {}}
            be-more-agent format:  {"action": "name", "value": "..."}
        """
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

        if not isinstance(data, dict):
            return None

        # BMO native format
        if "tool" in data:
            if "args" not in data:
                data["args"] = {}
            return data

        # be-more-agent format — normalize to BMO format
        if "action" in data:
            return {
                "tool": data["action"],
                "args": {k: v for k, v in data.items() if k != "action"},
            }

        return None

    # -------------------------------------------------------------------
    # Tool‑name resolution
    # -------------------------------------------------------------------

    def _resolve_tool_name(self, raw_name: str) -> str:
        """
        Normalize a tool name: lowercase, strip whitespace, resolve aliases.
        """
        name = raw_name.strip().lower()
        return TOOL_ALIASES.get(name, name)

    # -------------------------------------------------------------------
    # Result summarization
    # -------------------------------------------------------------------

    def _summarize_tool_result(
        self, tool_result: str, user_question: str, tool_name: str
    ) -> str:
        """
        Ask the LLM to condense a raw tool result into one short spoken
        sentence, keeping BMO's personality.

        This is the core pattern from be-more-agent: a separate, focused
        LLM call with a minimal system prompt produces much cleaner output
        than asking the main conversation to incorporate raw data.
        """
        summary_messages = [
            {
                "role": "system",
                "content": (
                    "Eres BMO, un asistente de voz. "
                    "Resume el siguiente resultado en UNA oración corta y natural "
                    "para decir en voz alta. Responde en español. "
                    "No uses markdown, listas ni caracteres especiales."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Pregunta del usuario: {user_question}\n"
                    f"Herramienta usada: {tool_name}\n"
                    f"Resultado: {tool_result}"
                ),
            },
        ]

        try:
            summary = self.llm.chat(summary_messages)
            cleaned = _clean_for_speech(summary)
            if cleaned:
                return cleaned
        except Exception as exc:
            logger.error("Summarization failed: %s", exc)

        # Fallback: return the raw result cleaned up.
        return _clean_for_speech(tool_result)
