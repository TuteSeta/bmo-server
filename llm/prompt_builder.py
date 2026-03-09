"""
Prompt builder for the BMO agent.

Assembles the messages list for Ollama's /api/chat endpoint.

Design decisions (informed by be-more-agent):

    1. The personality prompt and tool‑use protocol are kept as SEPARATE
       paragraphs with clear visual breaks.  Mixing them into one blob
       confuses smaller models about when to use JSON vs. plain text.

    2. Concrete examples (### EJEMPLOS ###) show the LLM exactly what
       format to use.  be-more-agent's prompt has 4 examples and the
       model follows reliably — we replicate that pattern in Spanish.

    3. Tool descriptions are listed with their exact JSON invocation
       format, not just a name+description.  This removes ambiguity.
"""

from typing import Dict, List

from core.config import BMO_SYSTEM_PROMPT
from tools.registry import get_tool_descriptions, list_tool_names


def _build_tool_protocol() -> str:
    """
    Build the tool‑use instructions block.

    Only included when tools are actually registered.
    """
    tool_desc = get_tool_descriptions()
    if not tool_desc:
        return ""

    tool_names = list_tool_names()
    names_csv = ", ".join(tool_names)

    return (
        "\n\n--- HERRAMIENTAS ---\n\n"
        "Tienes acceso a las siguientes herramientas:\n\n"
        f"{tool_desc}\n\n"
        "INSTRUCCIONES PARA HERRAMIENTAS:\n"
        "- Si el usuario pide una acción concreta (hora, sistema, docker), "
        "responde SOLAMENTE con JSON.\n"
        "- Si el usuario solo quiere conversar, responde con texto normal.\n"
        "- El JSON debe tener exactamente este formato:\n"
        '{"tool": "nombre_herramienta", "args": {}}\n\n'
        f"Herramientas válidas: {names_csv}\n\n"
        "### EJEMPLOS ###\n\n"
        "Usuario: ¿Qué hora es?\n"
        'Tú: {"tool": "get_time", "args": {}}\n\n'
        "Usuario: ¿Cómo está el sistema?\n"
        'Tú: {"tool": "get_system_status", "args": {}}\n\n'
        "Usuario: Lista los contenedores docker\n"
        'Tú: {"tool": "list_docker_containers", "args": {}}\n\n'
        "Usuario: Hola BMO\n"
        "Tú: ¡Hola! ¡BMO está listo para jugar!\n\n"
        "### FIN EJEMPLOS ###\n\n"
        "IMPORTANTE:\n"
        "- Si usas una herramienta, responde SOLO con el JSON. "
        "No escribas nada más antes ni después.\n"
        "- Si no necesitas herramienta, responde normalmente en español.\n"
        "- Nunca uses markdown, listas ni bloques de código porque "
        "la respuesta se escucha en voz alta."
    )


def build_messages(
    user_input: str,
    history: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Assemble the full messages list for Ollama's /api/chat endpoint.

    Structure:
        [ system_message, ...conversation_history, user_message ]

    The system message contains:
        - BMO's personality definition  (from config.BMO_SYSTEM_PROMPT)
        - Tool‑use protocol with examples  (built here)

    Args:
        user_input: The user's current transcribed speech.
        history:    Previous turns from ConversationMemory.get_history().

    Returns:
        A list of dicts ready to pass to OllamaClient.chat().
    """
    system_content = BMO_SYSTEM_PROMPT + _build_tool_protocol()

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_content}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    return messages
