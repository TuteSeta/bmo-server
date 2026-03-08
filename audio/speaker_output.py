import numpy as np
import sounddevice as sd

from core.config import AUDIO_SAMPLE_RATE


class SpeakerOutput:
    """
    Plays a numpy audio array through the default system speakers.

    Expects float32 samples normalised to [-1.0, 1.0].
    """

    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate

    def play(self, audio: np.ndarray, sample_rate: int = None) -> None:
        """
        Play audio and block until playback is complete.

        Args:
            audio:       1-D float32 numpy array of audio samples.
            sample_rate: Sample rate of the audio.  Defaults to the instance
                         sample_rate (AUDIO_SAMPLE_RATE).  Pass the TTS
                         engine's sample_rate here when playing synthesized
                         speech so the pitch and speed are correct.
        """
        rate = sample_rate if sample_rate is not None else self.sample_rate
        sd.play(audio, samplerate=rate)
        sd.wait()
