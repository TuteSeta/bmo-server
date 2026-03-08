import numpy as np
from openwakeword.model import Model

from core.config import WAKEWORD_MODEL, WAKEWORD_THRESHOLD
from core.logger import get_logger

logger = get_logger(__name__)


class WakeWordEngine:
    """
    Wraps OpenWakeWord to detect a configured wake word in an audio chunk.

    OpenWakeWord expects int16 audio at 16 000 Hz.  The pipeline feeds
    float32 chunks, so this class handles the conversion internally.
    """

    def __init__(
        self,
        model_name: str = WAKEWORD_MODEL,
        threshold: float = WAKEWORD_THRESHOLD,
    ) -> None:
        self.threshold = threshold
        # inference_framework="onnx" avoids a tflite dependency on most systems.
        self.model = Model(wakeword_models=[model_name], inference_framework="onnx")

    def detect(self, audio_chunk: np.ndarray) -> bool:
        """
        Return True if the wake word is detected in the given audio chunk.

        Args:
            audio_chunk: 1-D float32 numpy array at 16 000 Hz.
        """
        # Convert float32 [-1, 1] → int16 as required by OpenWakeWord.
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        prediction: dict = self.model.predict(audio_int16)
        for label, score in prediction.items():
            if score > 0.1:
                logger.debug("Wakeword score — %s: %.4f", label, score)
        return any(score >= self.threshold for score in prediction.values())
