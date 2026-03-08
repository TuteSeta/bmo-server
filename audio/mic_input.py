import json
import pathlib
import queue

import numpy as np
import sounddevice as sd

from core.config import AUDIO_CHANNELS, AUDIO_DTYPE, AUDIO_SAMPLE_RATE, MIC_DEVICE_INDEX
from core.logger import get_logger

logger = get_logger(__name__)

# config_local.json lives at the project root alongside core/, audio/, etc.
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_LOCAL_CONFIG_PATH = _PROJECT_ROOT / "config_local.json"

# RMS level below which we consider audio absent (silence ≈ 0.001).
_SILENCE_THRESHOLD = 0.005

# VAD settings
_VAD_THRESHOLD = 0.5
_VAD_CHUNK_SIZE = 512       # samples — required by Silero VAD at 16 kHz (32 ms per chunk)
_VAD_SILENCE_CHUNKS = 45    # ~1.44 s of silence (45 × 32 ms) before stopping
_VAD_MAX_SECONDS = 15.0     # hard cap on VAD recording length

# Microphone gain multiplier — amplifies quiet input before sending to Whisper
_MIC_GAIN = 5.0


def list_microphones() -> None:
    """Print all available input devices to help identify the correct device index."""
    devices = sd.query_devices()
    logger.debug("Available input devices:")
    for i, device in enumerate(devices):
        if device["max_input_channels"] > 0:
            logger.debug("  %d: %s", i, device["name"])


# ---------------------------------------------------------------------------
# Local config helpers
# ---------------------------------------------------------------------------

def _load_local_config() -> dict:
    """Load persisted settings from config_local.json. Returns {} on any error."""
    if _LOCAL_CONFIG_PATH.exists():
        try:
            with open(_LOCAL_CONFIG_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_local_config(data: dict) -> None:
    """Persist settings dict to config_local.json."""
    try:
        with open(_LOCAL_CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as exc:
        logger.warning("Could not save config: %s", exc)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

def resolve_mic_device(default: int | None = MIC_DEVICE_INDEX) -> int | None:
    """
    Determine which microphone device index to use.

    Priority:
      1. config_local.json  — persisted user selection (survives restarts)
      2. default argument   — MIC_DEVICE_INDEX from core/config.py
      3. Interactive prompt  — only when both above resolve to None

    Saves the chosen index back to config_local.json so the user is not
    asked again on the next run.
    """
    local = _load_local_config()

    # Local config wins if the key is present (even if value is None).
    if "mic_device_index" in local:
        device_index: int | None = local["mic_device_index"]
    else:
        device_index = default

    # If still undecided, ask the user interactively.
    if device_index is None:
        list_microphones()
        raw = input("[BMO] Select microphone index: ").strip()
        try:
            device_index = int(raw)
        except (ValueError, EOFError):
            logger.warning("Invalid input — using system default microphone")
            device_index = None

        # Persist so we never ask again.
        local["mic_device_index"] = device_index
        _save_local_config(local)
        logger.debug("Mic selection saved to %s", _LOCAL_CONFIG_PATH.name)

    # Resolve a human-readable name for the startup log.
    if device_index is not None:
        try:
            name = sd.query_devices(device_index)["name"]
        except Exception:
            name = str(device_index)
    else:
        try:
            name = sd.query_devices(kind="input")["name"]
        except Exception:
            name = "system default"

    logger.debug("Using mic device: %s", name)
    return device_index


def _try_load_silero_vad():
    """Load Silero VAD model. Returns None if silero-vad is not installed."""
    try:
        from silero_vad import load_silero_vad
        model = load_silero_vad()
        logger.debug("Silero VAD loaded")
        return model
    except Exception as exc:
        logger.debug("silero-vad not available — using fixed-length recording: %s", exc)
        return None


# ---------------------------------------------------------------------------
# MicInput
# ---------------------------------------------------------------------------

class MicInput:
    """
    Captures audio from the configured microphone using sounddevice.

    When Silero VAD is available, record() uses voice-activity detection:
    it waits for speech to begin, records until silence, then returns.
    Falls back to fixed-length recording if silero-vad is not installed.

    All returned audio has a gain boost applied (_MIC_GAIN × 5.0) and is
    clipped to [-1.0, 1.0] for safe Whisper input.

    Returns a flat float32 numpy array at the configured sample rate,
    which is the format expected by both OpenWakeWord and Whisper.
    """

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
        device: int | None = MIC_DEVICE_INDEX,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = resolve_mic_device(device)
        self._vad_model = _try_load_silero_vad()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def record(self, seconds: float = 3) -> np.ndarray:
        """
        Record audio.  Uses Silero VAD when available; falls back to
        fixed-length recording otherwise.

        Returns:
            1-D float32 numpy array of audio samples.
        """
        if self._vad_model is not None:
            return self._record_with_vad(max_seconds=max(seconds, _VAD_MAX_SECONDS))
        return self._record_fixed(seconds)

    # ------------------------------------------------------------------
    # VAD-based recording
    # ------------------------------------------------------------------

    def _record_with_vad(self, max_seconds: float = _VAD_MAX_SECONDS) -> np.ndarray:
        """
        Stream audio in 32 ms chunks and use Silero VAD to decide when
        speech starts and ends.

        Flow:
            audio stream
                ↓
            VAD detects speech  → start collecting
                ↓
            silence detected    → stop collecting
                ↓
            return collected audio
        """
        import torch

        audio_queue: queue.Queue = queue.Queue()

        def _callback(indata, frames, time, status):
            audio_queue.put(indata.copy().flatten())

        collected: list[np.ndarray] = []
        speech_started = False
        silence_count = 0
        max_chunks = int(max_seconds * self.sample_rate / _VAD_CHUNK_SIZE)

        logger.debug("Waiting for speech...")
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=AUDIO_DTYPE,
            blocksize=_VAD_CHUNK_SIZE,
            callback=_callback,
            device=self.device,
        ):
            for _ in range(max_chunks):
                try:
                    chunk = audio_queue.get(timeout=2.0)
                except queue.Empty:
                    break

                # Gain boost — amplify quiet microphones before VAD check
                chunk = np.clip(chunk * _MIC_GAIN, -1.0, 1.0)

                tensor = torch.tensor(chunk, dtype=torch.float32)
                with torch.no_grad():
                    speech_prob = float(self._vad_model(tensor, self.sample_rate))

                if speech_prob >= _VAD_THRESHOLD:
                    if not speech_started:
                        logger.debug("Speech detected")
                        speech_started = True
                    silence_count = 0
                    collected.append(chunk)
                elif speech_started:
                    collected.append(chunk)
                    silence_count += 1
                    if silence_count >= _VAD_SILENCE_CHUNKS:
                        logger.debug("Silence detected — stopping")
                        break

        if collected:
            audio = np.concatenate(collected)
            duration = len(audio) / self.sample_rate
            rms = np.sqrt(np.mean(audio ** 2))
            logger.debug("Captured %.2fs | RMS=%.5f", duration, rms)
            return audio

        logger.debug("No speech detected — returning empty audio")
        return np.zeros(_VAD_CHUNK_SIZE, dtype=np.float32)

    # ------------------------------------------------------------------
    # Fixed-length fallback recording
    # ------------------------------------------------------------------

    def _record_fixed(self, seconds: float) -> np.ndarray:
        """Fallback: fixed-length recording with gain boost."""
        logger.debug("Recording (fixed mode)...")
        num_samples = int(self.sample_rate * seconds)
        audio = sd.rec(
            num_samples,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=AUDIO_DTYPE,
            device=self.device,
        )
        sd.wait()
        flat = audio.flatten()

        # Gain boost
        flat = np.clip(flat * _MIC_GAIN, -1.0, 1.0)
        level = np.sqrt(np.mean(flat ** 2))
        logger.debug("MIC level: %.5f", level)

        if level < _SILENCE_THRESHOLD:
            logger.debug("No audio detected — check microphone device")

        return flat

