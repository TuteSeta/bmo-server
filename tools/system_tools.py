"""
System tools for BMO.

Each function is registered as a tool via the @register_tool decorator.
Tools are auto-discovered when this module is imported — agent_loop.py
imports this module at startup to ensure all tools are in the registry
before the first LLM call.

Adding a new tool:
    1. Define a function that returns a plain string.
    2. Decorate it with @register_tool("name", "description").
    3. That's it — the agent loop and prompt builder pick it up automatically.
"""

import datetime
import subprocess

import psutil

from tools.registry import register_tool


@register_tool(
    "get_time",
    "Devuelve la fecha y hora actual del sistema. Úsala cuando el usuario pregunte qué hora o qué día es.",
)
def get_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%A, %d de %B de %Y, %H:%M")


@register_tool(
    "get_system_status",
    "Devuelve el uso de CPU y memoria RAM del servidor. Úsala cuando el usuario pregunte cómo está el sistema.",
)
def get_system_status() -> str:
    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory()
    return f"CPU: {cpu}%, RAM: {ram.percent}%"


@register_tool(
    "list_docker_containers",
    "Lista todos los contenedores Docker activos en el servidor con su estado.",
)
def list_docker_containers() -> str:
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Image}}"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "Docker no está disponible"
    if result.returncode != 0:
        return "Docker no está disponible"
    if not result.stdout.strip():
        return "No hay contenedores en ejecución actualmente."
    lines = result.stdout.strip().splitlines()
    rows = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) >= 3:
            rows.append(f"{parts[0]} ({parts[2]}): {parts[1]}")
    return "\n".join(rows)
