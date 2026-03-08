import numpy as np
from faster_whisper import WhisperModel

from core.config import WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL
from core.logger import get_logger

logger = get_logger(__name__)


class WhisperEngine:
    """
    Transcribes speech audio to text using faster-whisper.

    The model is loaded once on construction.  Transcription is performed
    locally — no network call required.
    """

    def __init__(self) -> None:
        self.model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Convert a numpy audio array to text.

        Args:
            audio: 1-D float32 numpy array at 16 000 Hz.

        Returns:
            Transcribed text (lowercase, stripped), or an empty string if
            nothing was detected or the result is too short to be meaningful.
        """
        segments, info = self.model.transcribe(
            audio,
            beam_size=5,
            language="es",
            task="transcribe",
        )

        segment_list = list(segments)

        # Confidence — average log-probability across segments
        if segment_list:
            avg_logprob = sum(s.avg_logprob for s in segment_list) / len(segment_list)
            confidence = max(0.0, min(1.0, (avg_logprob + 1.0)))  # rough 0–1 scale
            logger.debug("STT confidence: %.2f (avg_logprob=%.3f)", confidence, avg_logprob)
        else:
            logger.debug("STT: no segments")

        raw = " ".join(s.text.strip() for s in segment_list)

        # Clean: strip whitespace, normalise to lowercase
        result = raw.strip().lower()

        # Discard noise-triggered single-character or empty results
        if len(result) < 2:
            logger.debug("STT discarded (too short): %r", result)
            return ""

        logger.debug("STT result: %r", result)
        return result
