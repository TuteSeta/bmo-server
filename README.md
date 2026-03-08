# BMO — Local AI Voice Assistant

BMO is a fully offline, locally-running AI home assistant inspired by
the robot from *Adventure Time*. It runs entirely on your own hardware —
no cloud, no API keys, no data leaving your machine.

```
        ___________
       |  B  M  O  |
       |   (o_o)   |
       |___________|
```

---

## What BMO does

| Stage | Technology |
|-------|-----------|
| Wake word detection | [openWakeWord](https://github.com/dscripka/openWakeWord) |
| Speech recognition | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper base, CPU) |
| Language model | [Ollama](https://ollama.com) — llama3.2 |
| Text-to-speech | [Piper](https://github.com/rhasspy/piper) — en_US-lessac-medium |
| Audio I/O | [sounddevice](https://python-sounddevice.readthedocs.io) |

---

## Pipeline

```
Microphone
    │
    ▼
Wake Word Detection  ("Hey Jarvis")
    │
    ▼
Speech-to-Text  (Whisper)
    │
    ▼
LLM  (Ollama / llama3.2)
    │
    ▼
Text-to-Speech  (Piper)
    │
    ▼
Speaker Output
```

---

## Project structure

```
bmo-server/
│
├── core/               # Config and event bus
├── audio/              # Microphone input and speaker output
├── wakeword/           # openWakeWord engine
├── stt/                # faster-whisper engine
├── llm/                # Ollama HTTP client
├── tts/                # Piper TTS engine
├── voice/              # Pipeline orchestrator and Assistant class
├── services/           # Docker utility (optional)
│
├── scripts/
│   └── run_assistant.py   # Entry point
│
├── models/
│   └── README.md          # Instructions for downloading model files
│
├── ui/                 # (reserved — event-bus-driven UI layer)
├── hardware/           # (reserved — GPIO / display integrations)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- A microphone and speakers
- ~2 GB free disk space for models

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/bmo-server.git
cd bmo-server
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Piper voice model

The assistant requires an ONNX voice model. See `models/README.md` for full
details. Quick install:

```bash
mkdir -p models
cd models
curl -LO https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx
curl -LO https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx.json
cd ..
```

### 5. Start Ollama

```bash
# Install Ollama from https://ollama.com, then:
ollama serve          # start the server (runs on http://localhost:11434)
ollama pull llama3.2  # download the language model (~2 GB)
```

---

## Running BMO

```bash
python scripts/run_assistant.py
```

### Expected console output

```
========================================
  BMO — Local Voice Assistant
  Press Ctrl+C to stop
========================================
BMO listening...
[WAKE WORD DETECTED]
Listening for command...
[TRANSCRIPTION] what time is it
[LLM RESPONSE] I don't have access to real-time information, but...
[TTS PLAYING]
BMO listening...
```

Say **"Hey Jarvis"** to activate BMO, then speak your command.

---

## Configuration

All settings are in `core/config.py`:

```python
WAKEWORD_MODEL    = "hey_jarvis"   # built-in openWakeWord model
WAKEWORD_THRESHOLD = 0.5

WHISPER_MODEL     = "base"
WHISPER_DEVICE    = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

OLLAMA_MODEL      = "llama3.2:latest"
OLLAMA_URL        = "http://localhost:11434/api/generate"

PIPER_MODEL_PATH  = "models/en_US-lessac-medium.onnx"

AUDIO_RECORD_SECONDS = 5
```

---

## Architecture notes

- **Event bus** (`core/event_bus.py`): a lightweight pub/sub system decouples
  every pipeline stage. Events fired: `listening`, `wake_word_detected`,
  `speech_transcribed`, `llm_thinking`, `llm_response`, `tts_speaking`.
- **ui/** and **hardware/** directories are reserved for future integrations
  (terminal dashboard, GPIO, OLED display) that subscribe to bus events
  without touching the pipeline.

---

## License

MIT
