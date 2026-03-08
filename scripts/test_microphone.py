"""
Standalone microphone and STT test.

Records 5 seconds of audio, saves it to test_audio.wav, then transcribes
it with Whisper.  Use this to verify the microphone and STT pipeline work
independently of the wakeword engine.

Usage:
    python scripts/test_microphone.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

from audio.mic_input import list_microphones
from core.config import AUDIO_SAMPLE_RATE, MIC_DEVICE_INDEX
from stt.whisper_engine import WhisperEngine


RECORD_SECONDS = 5
OUTPUT_WAV = "test_audio.wav"


def main() -> None:
    list_microphones()

    print(f"\nRecording {RECORD_SECONDS} seconds — speak now...")
    audio = sd.rec(
        int(AUDIO_SAMPLE_RATE * RECORD_SECONDS),
        samplerate=AUDIO_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=MIC_DEVICE_INDEX,
    )
    sd.wait()
    audio_flat = audio.flatten()

    # RMS level check
    level = np.sqrt(np.mean(audio_flat ** 2))
    print(f"[MIC LEVEL] {level:.5f}")
    if level < 0.001:
        print("[WARNING] Audio level is very low — check microphone connection and permissions.")

    # Save to WAV (scipy expects int16)
    audio_int16 = (audio_flat * 32767).astype(np.int16)
    wav.write(OUTPUT_WAV, AUDIO_SAMPLE_RATE, audio_int16)
    print(f"[MIC] Saved recording to {OUTPUT_WAV}")

    # Transcribe
    print("[STT] Transcribing...")
    engine = WhisperEngine()
    text = engine.transcribe(audio_flat)
    if text:
        print(f"You said: {text}")
    else:
        print("(no speech detected)")


if __name__ == "__main__":
    main()
