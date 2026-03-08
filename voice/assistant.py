import signal
import sys

from core.logger import get_logger
from voice.pipeline import VoicePipeline

logger = get_logger(__name__)


class Assistant:
    """
    Top-level assistant entrypoint.

    Owns the VoicePipeline and the optional pygame UI renderer.
    Handles clean shutdown on Ctrl+C.
    """

    def __init__(self) -> None:
        self.pipeline = VoicePipeline()
        self._renderer = None

    def _handle_shutdown(self, sig, frame) -> None:
        logger.info("Shutting down BMO...")
        if self._renderer is not None:
            self._renderer.stop()
        self.pipeline.stop()
        sys.exit(0)

    def start(self) -> None:
        """Start the renderer thread, then run the assistant loop."""
        signal.signal(signal.SIGINT, self._handle_shutdown)

        # Start UI renderer in a background thread before the pipeline loop.
        # Wrapped in a broad except so a missing/broken pygame never stops BMO.
        try:
            from ui.renderer import BmoRenderer
            self._renderer = BmoRenderer()
            self._renderer.start()
        except Exception as exc:
            logger.debug("UI renderer unavailable: %s", exc)

        self.pipeline.start()
