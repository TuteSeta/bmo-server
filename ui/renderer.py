"""
ui/renderer.py
--------------
Pygame-based BMO face renderer.

Displays the animated BMO face on an 800×480 window.  Subscribes to
``state_changed`` events from the EventBus and switches/animates the correct
face frames automatically.

The renderer runs inside a daemon background thread so the main voice pipeline
is never blocked.  If pygame cannot be imported or fails to initialize the
renderer degrades gracefully — BMO continues running headless.

Usage (called from voice/assistant.py)::

    renderer = BmoRenderer()
    renderer.start()   # non-blocking — launches background thread
"""

import threading

from core.event_bus import bus
from core.state_machine import BotState

# ── Optional pygame import ──────────────────────────────────────────────────
try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

# ── Constants ────────────────────────────────────────────────────────────────
WINDOW_WIDTH  = 800
WINDOW_HEIGHT = 480
FPS           = 5          # animation frames per second
BG_COLOR      = (0, 0, 0)  # black background

# Map BotState enum values to face directory names
_STATE_MAP: dict[BotState, str] = {
    BotState.IDLE:      "idle",
    BotState.LISTENING: "listening",
    BotState.THINKING:  "thinking",
    BotState.SPEAKING:  "speaking",
    BotState.ERROR:     "error",
}


class BmoRenderer:
    """
    Pygame window that renders the BMO animated face.

    Thread-safe: the EventBus callback updates ``_current_state`` under a lock;
    the render loop reads it under the same lock.
    """

    def __init__(self) -> None:
        self._current_state: str = "idle"
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    # ── EventBus callback ────────────────────────────────────────────────────

    def _on_state_changed(self, state: BotState) -> None:
        """Called from any thread when the bot state changes."""
        face_name = _STATE_MAP.get(state, "idle")
        with self._lock:
            self._current_state = face_name

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _load_surfaces(self) -> dict[str, list]:
        """
        Load all face PNG frames as pygame Surfaces after pygame has been
        initialized.  Falls back to an empty list for any state that cannot
        be loaded so the renderer never crashes on a missing frame.
        """
        from ui.face import load_face_paths

        faces: dict[str, list] = {}
        for state_name, paths in load_face_paths().items():
            surfaces = []
            for path in paths:
                try:
                    surfaces.append(pygame.image.load(path).convert())
                except Exception as exc:
                    print(f"[UI] Could not load {path}: {exc}")
            if surfaces:
                faces[state_name] = surfaces

        return faces

    # ── Render loop (runs in background thread) ───────────────────────────────

    def _run(self) -> None:
        try:
            pygame.init()
            screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("BMO")
            clock = pygame.time.Clock()

            faces = self._load_surfaces()
            frame_index = 0
            last_state  = None

            while self._running:
                # ── Process window events ──────────────────────────────────
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._running = False

                # ── Read current state ─────────────────────────────────────
                with self._lock:
                    state = self._current_state

                # Reset animation when state changes
                if state != last_state:
                    frame_index = 0
                    last_state  = state

                # ── Pick frame list ────────────────────────────────────────
                frames = faces.get(state) or faces.get("idle") or []

                # ── Draw ───────────────────────────────────────────────────
                screen.fill(BG_COLOR)

                if frames:
                    surface = frames[frame_index % len(frames)]
                    scaled  = pygame.transform.scale(
                        surface, (WINDOW_WIDTH, WINDOW_HEIGHT)
                    )
                    screen.blit(scaled, (0, 0))
                    frame_index = (frame_index + 1) % len(frames)

                pygame.display.flip()
                clock.tick(FPS)

        except Exception as exc:
            print(f"[UI] Renderer error — running headless: {exc}")
        finally:
            try:
                pygame.quit()
            except Exception:
                pass

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Subscribe to state events and launch the render loop in a daemon thread.
        Returns immediately.
        """
        if not _PYGAME_AVAILABLE:
            print("[UI] pygame not installed — running headless")
            return

        bus.subscribe("state_changed", self._on_state_changed)
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="BmoRenderer",
        )
        self._thread.start()
        print("[UI] BMO renderer started (800×480)")

    def stop(self) -> None:
        """Signal the render loop to exit."""
        self._running = False
