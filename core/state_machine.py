from enum import Enum

from core.event_bus import bus
from core.logger import get_logger

logger = get_logger(__name__)


class BotState(Enum):
    """
    Discrete operational states for BMO.

    Subscribers on the event bus can react to state changes to drive
    UI animations, hardware LEDs, sound effects, or dashboard updates
    without any direct coupling to this module.
    """
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


_current_state: BotState = BotState.IDLE


def set_state(new_state: BotState) -> None:
    """
    Transition BMO to a new state.

    Updates the internal state variable and emits a 'state_changed' event
    on the global event bus so all subscribers are notified.

    Example subscribers (not yet implemented):
        ui/renderer.py    — swap face animation
        hardware/leds.py  — change LED colour
        audio/sfx.py      — play a state sound effect
    """
    global _current_state
    _current_state = new_state
    logger.debug("State → %s", new_state.value.upper())
    bus.emit("state_changed", new_state)


def get_state() -> BotState:
    """Return the current bot state."""
    return _current_state
