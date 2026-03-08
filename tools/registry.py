from typing import Any, Callable, Dict


# -----------------------------------------------------------------------
# Internal registry store
# -----------------------------------------------------------------------

_registry: Dict[str, Dict] = {}


# -----------------------------------------------------------------------
# Registration decorator
# -----------------------------------------------------------------------

def register_tool(name: str, description: str) -> Callable:
    """
    Decorator that registers a Python function as a BMO tool.

    Usage:
        @register_tool("get_time", "Returns the current date and time.")
        def get_time() -> str:
            ...

    The name and description are injected into the LLM system prompt so
    the model knows when and how to invoke each tool.
    """
    def decorator(func: Callable) -> Callable:
        _registry[name] = {
            "name": name,
            "description": description,
            "func": func,
        }
        return func
    return decorator


# -----------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------

def execute_tool(name: str, args: Dict[str, Any] = None) -> str:
    """
    Call the registered tool by name, passing args as keyword arguments.

    Returns the tool's string result, or an error message if the tool
    is unknown or raises an exception.
    """
    if name not in _registry:
        available = ", ".join(_registry.keys()) if _registry else "ninguna"
        return f"Herramienta desconocida '{name}'. Herramientas disponibles: {available}."
    try:
        result = _registry[name]["func"](**(args or {}))
        return str(result)
    except Exception as exc:
        return f"La herramienta '{name}' falló con el error: {exc}"


# -----------------------------------------------------------------------
# Introspection helpers (used by prompt_builder)
# -----------------------------------------------------------------------

def get_tool_descriptions() -> str:
    """
    Return a formatted string listing all registered tools.
    Injected into the LLM system prompt so the model knows what is available.

    Format:
        tool_name
        Description of the tool.

        next_tool_name
        Description of the next tool.
    """
    if not _registry:
        return ""
    blocks = [f"{t['name']}\n{t['description']}" for t in _registry.values()]
    return "\n\n".join(blocks)


def list_tool_names() -> list:
    """Return a list of all registered tool names."""
    return list(_registry.keys())
