import io
import wave

import numpy as np
from scipy.signal import resample as scipy_resample
from piper.config import SynthesisConfig
from piper.voice import PiperVoice

from core.config import PIPER_MODEL_PATH, VOICE_SPEED, VOICE_TONE, VOICE_VARIATION

# +2 semitones pitch shift ratio: 2^(2/12) ≈ 1.1225
# Shrinking the sample count and playing at the original sample_rate raises
# both pitch and speed, producing the cartoonish BMO voice effect.
_PITCH_SEMITONES = 2
_PITCH_RATIO = 2 ** (_PITCH_SEMITONES / 12)  # ≈ 1.1225


class PiperEngine:
    """
    Text-to-speech engine backed by Piper.

    Piper synthesises audio offline using an ONNX voice model.
    speak() returns a float32 numpy array that SpeakerOutput can play directly.

    Post-processing applies a +2-semitone pitch shift via resampling to give
    BMO's characteristic higher, more cartoonish voice.

    Voice character tuning
    ----------------------
    All three parameters below are read from core/config.py so they can be
    changed in one place without touching this file.

    VOICE_SPEED  (length_scale)
        Controls speaking rate.
        Lower = faster delivery.  Higher = slower, more deliberate speech.
        Typical range: 0.5 – 2.0

    VOICE_TONE   (noise_scale)
        Controls tonal expressiveness.  Higher values make the voice sound
        more lively and varied; lower values produce a flatter, more robotic
        tone.  Typical range: 0.0 – 1.0

    VOICE_VARIATION  (noise_w)
        Controls phoneme-level pronunciation variation — how much each
        individual sound deviates from the model's "average" pronunciation.
        Higher values sound more natural and less synthetic.
        Typical range: 0.0 – 1.0
    """

    def __init__(self, model_path: str = PIPER_MODEL_PATH) -> None:
        self.voice = PiperVoice.load(model_path)
        # Expose the model's native sample rate so the speaker can play at
        # the correct rate.  Piper voices vary (e.g. 22050 Hz for medium
        # models) and are NOT the same as Whisper's 16 000 Hz input rate.
        self.sample_rate: int = self.voice.config.sample_rate

        # Voice character parameters — edit these in core/config.py.
        self.length_scale: float = VOICE_SPEED      # speaking rate (lower = faster)
        self.noise_scale: float  = VOICE_TONE        # expressiveness (higher = livelier)
        self.noise_w: float      = VOICE_VARIATION   # phoneme variation (higher = more natural)

    def _synthesize_wav(self, text: str) -> bytes:
        """Render text to WAV bytes using Piper with tuning parameters."""
        syn_config = SynthesisConfig(
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w,
        )
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file, syn_config=syn_config)
        return buf.getvalue()

    def _pitch_shift(self, audio: np.ndarray) -> np.ndarray:
        """
        Shift pitch up by _PITCH_SEMITONES via resampling.

        Resamples the audio to fewer samples (n / ratio), which when played
        at the original sample_rate sounds faster and higher-pitched — the
        classic chipmunk / cartoonish voice effect that matches BMO's character.
        """
        n_target = max(1, int(len(audio) / _PITCH_RATIO))
        shifted = scipy_resample(audio, n_target).astype(np.float32)
        return shifted

    def speak(self, text: str) -> np.ndarray:
        """
        Convert text to a playable audio array.

        Returns:
            1-D float32 numpy array normalised to [-1.0, 1.0].
            Play it at self.sample_rate for correct pitch and speed.
        """
        wav_bytes = self._synthesize_wav(text)
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wav_file:
            raw_frames = wav_file.readframes(wav_file.getnframes())

        # Piper outputs 16-bit PCM; normalise to float32 for sounddevice.
        audio = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32)
        audio /= 32768.0

        # Apply pitch shift for BMO's cartoonish voice
        audio = self._pitch_shift(audio)
        return audio
