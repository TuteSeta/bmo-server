from typing import Dict, List

from core.config import BMO_SYSTEM_PROMPT
from tools.registry import get_tool_descriptions


def build_messages(
    user_input: str,
    history: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Assemble the full messages list for Ollama's /api/chat endpoint.

    Structure:
        [ system_message, ...conversation_history, user_message ]

    The system message contains:
        - BMO's personality definition
        - Descriptions of all registered tools and the JSON call format

    Args:
        user_input: The user's current transcribed speech.
        history:    Previous turns from ConversationMemory.get_history().

    Returns:
        A list of dicts ready to pass to OllamaClient.chat().
    """
    tool_desc = get_tool_descriptions()

    if tool_desc:
        system_content = (
            BMO_SYSTEM_PROMPT
            + "\n\n"
            + "Puedes usar herramientas cuando sea necesario.\n\n"
            + "Herramientas disponibles:\n\n"
            + tool_desc
            + "\n\n"
            + "Para usar una herramienta responde SOLO con JSON:\n"
            + '{"tool": "nombre_herramienta", "args": {}}\n\n'
            + "Si no necesitas herramienta, responde normalmente en español.\n\n"
            + "Ejemplos de comandos del usuario:\n"
            + '"¿Qué hora es?"\n'
            + '"¿Cómo está el sistema?"\n'
            + '"Lista los contenedores docker"\n'
            + '"Hola BMO"\n\n'
            + "Importante: No uses markdown ni listas porque la respuesta será hablada en voz alta."
        )
    else:
        system_content = BMO_SYSTEM_PROMPT

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_content}
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    return messages
