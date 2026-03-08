import queue

import numpy as np
import sounddevice as sd

from agent.agent_loop import AgentLoop
from audio.mic_input import MicInput
from audio.speaker_output import SpeakerOutput
from core.config import (
    AUDIO_CHUNK_SECONDS,
    AUDIO_RECORD_SECONDS,
    AUDIO_SAMPLE_RATE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from core.event_bus import bus
from core.logger import get_logger
from core.state_machine import BotState, set_state
from llm.ollama_client import OllamaClient
from memory.conversation_memory import ConversationMemory
from stt.whisper_engine import WhisperEngine
from tts.piper_engine import PiperEngine
from wakeword.wakeword_engine import WakeWordEngine

logger = get_logger(__name__)

# Set to True to skip wakeword detection and go directly to speech recording.
# Useful for verifying STT works before debugging the wake word model.
DEBUG_ALWAYS_LISTEN = True


class VoicePipeline:
    """
    Orchestrates the full BMO voice agent pipeline:

        Microphone → Wake Word → Record → STT → AgentLoop → TTS → Speaker

    State transitions are broadcast through the EventBus at each stage,
    allowing UI renderers, hardware controllers, and sound effect players
    to react without any direct imports into this module.

    The AgentLoop replaces the former direct LLM call and handles:
        - system prompt injection
        - conversation memory
        - tool detection and execution
        - multi-iteration reasoning
    """

    def __init__(self) -> None:
        self.mic = MicInput()
        self.speaker = SpeakerOutput()
        self.wakeword = WakeWordEngine()
        self.stt = WhisperEngine()
        self.tts = PiperEngine()

        llm = OllamaClient(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        memory = ConversationMemory()
        self.agent = AgentLoop(llm=llm, memory=memory)

        self._running = False

    # ------------------------------------------------------------------
    # Wake-word detection helpers
    # ------------------------------------------------------------------

    def _listen_for_wakeword(self) -> bool:
        """
        Stream microphone audio in short chunks and return True as soon as
        the wake word is detected.  Returns False if the pipeline is stopped.
        """
        chunk_samples = int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_SECONDS)
        audio_queue: queue.Queue = queue.Queue()

        def _callback(indata: np.ndarray, frames: int, time, status) -> None:
            audio_queue.put(indata.copy().flatten())

        with sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            callback=_callback,
            device=self.mic.device,
        ):
            while self._running:
                try:
                    chunk = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if self.wakeword.detect(chunk):
                    return True

        return False

    # ------------------------------------------------------------------
    # Single agent cycle
    # ------------------------------------------------------------------

    def run_once(self) -> None:
        """Execute one full listen → reason → speak cycle."""

        # 1. Record speech
        set_state(BotState.LISTENING)
        audio = self.mic.record(seconds=AUDIO_RECORD_SECONDS)

        # 2. Speech → Text
        text = self.stt.transcribe(audio)
        if not text:
            logger.debug("No speech understood — skipping cycle")
            set_state(BotState.IDLE)
            return

        logger.debug("Transcription: %s", text)
        bus.emit("speech_transcribed", text)

        # 3. Agent reasoning loop → response
        set_state(BotState.THINKING)
        logger.info("Processing request")
        bus.emit("llm_thinking", text)
        try:
            response = self.agent.run(text)
        except Exception as exc:
            logger.error("Agent error: %s", exc)
            set_state(BotState.ERROR)
            response = "Lo siento, tuve un problema. Inténtalo de nuevo."

        bus.emit("llm_response", response)

        # 4. Response → Speech → Playback
        set_state(BotState.SPEAKING)
        logger.info("Speaking")
        bus.emit("tts_speaking", response)
        audio_out = self.tts.speak(response)
        # Piper's native sample rate differs from Whisper's 16 000 Hz input.
        self.speaker.play(audio_out, sample_rate=self.tts.sample_rate)
        set_state(BotState.IDLE)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Run the wake-word → agent cycle loop continuously."""
        self._running = True
        set_state(BotState.IDLE)

        logger.info("Voice system ready")

        while self._running:
            logger.info("Listening for wake word")
            bus.emit("listening", None)

            if DEBUG_ALWAYS_LISTEN:
                logger.debug("Wakeword skipped — DEBUG_ALWAYS_LISTEN active")
                set_state(BotState.LISTENING)
                self.run_once()
                set_state(BotState.IDLE)
            elif self._listen_for_wakeword():
                logger.info("Wake word detected")
                bus.emit("wake_word_detected", None)
                set_state(BotState.LISTENING)
                self.run_once()
                set_state(BotState.IDLE)

    def stop(self) -> None:
        """Signal the pipeline to stop after the current cycle."""
        self._running = False
