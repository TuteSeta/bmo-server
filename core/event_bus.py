from collections import defaultdict
from typing import Any, Callable


class EventBus:
    """
    Minimal publish/subscribe event bus.

    Modules emit named events with optional data payloads.
    Other modules subscribe callbacks to those event names.
    This keeps components decoupled — no direct imports needed between them.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event_name: str, callback: Callable) -> None:
        """Register a callback for the given event name."""
        self._subscribers[event_name].append(callback)

    def emit(self, event_name: str, data: Any = None) -> None:
        """
        Fire all callbacks registered for event_name.
        Exceptions in individual callbacks are caught and printed so that
        one bad listener never breaks the pipeline.
        """
        for callback in self._subscribers[event_name]:
            try:
                callback(data)
            except Exception as exc:
                print(f"[EventBus] Error in handler for '{event_name}': {exc}")


# Module-level singleton — import and use directly.
# Example:
#   from core.event_bus import bus
#   bus.subscribe("wake_word_detected", my_handler)
#   bus.emit("wake_word_detected", None)
bus = EventBus()
