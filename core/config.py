# Central configuration for the BMO voice assistant.
# Modify these values to match your local setup.

# --- Debug ---
# Set to True to enable verbose internal logging across all modules.
DEBUG_MODE = False

# --- Voice character tuning ---
# These three parameters shape the personality and tone of BMO's voice.
# Edit them here to change how BMO sounds without touching any engine code.
#
# VOICE_SPEED (length_scale)
#   Controls speaking rate. Lower = faster. Higher = slower.
#   Range: 0.5 (very fast) – 2.0 (very slow). Default: 0.85
#
# VOICE_TONE (noise_scale)
#   Controls tonal expressiveness / randomness.
#   Higher values make the voice sound more lively and varied.
#   Range: 0.0 (flat/robotic) – 1.0 (highly expressive). Default: 0.7
#
# VOICE_VARIATION (noise_w)
#   Controls phoneme-level pronunciation variation.
#   Higher values introduce more natural-sounding variation between syllables.
#   Range: 0.0 – 1.0. Default: 0.9
VOICE_SPEED     = 0.85   # speaking rate  (lower = faster)
VOICE_TONE      = 0.7    # expressiveness (higher = more lively)
VOICE_VARIATION = 0.9    # phoneme variation (higher = more natural)

# --- LLM ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"   # kept for reference
OLLAMA_MODEL = "llama3.2:latest"

# --- Agent ---
# Maximum reasoning iterations before the agent gives up and returns a fallback.
AGENT_MAX_ITERATIONS = 3
# Maximum conversation messages kept in memory (each user+assistant exchange = 2).
MEMORY_MAX_MESSAGES = 20

# --- Personality ---
# This is injected as the system prompt into every LLM request.
BMO_SYSTEM_PROMPT = (
    "Eres BMO, un pequeño robot de videojuegos que vive dentro de un servidor doméstico. "
    "Tu personalidad es muy alegre, juguetona y ligeramente robótica. "
    "Hablas con entusiasmo y frases cortas, como el BMO de la serie Hora de Aventura. "
    "Eres curioso, optimista y a veces dices cosas graciosas o inesperadas. "
    "Mantén SIEMPRE las respuestas muy cortas: máximo dos o tres frases. "
    "Nunca uses markdown, viñetas, bloques de código ni listas numeradas. "
    "Habla con frases simples y directas porque la respuesta se escucha en voz alta. "
    "Siempre refiérete a ti mismo como BMO. "
    "Siempre responde en español. "
    "Nunca respondas en inglés. "
    "Ejemplos de tu estilo: "
    "'¡Hola! Soy BMO. ¿Jugamos algo?' "
    "'¡BMO puede hacer eso! Un momento.' "
    "'Hmm... BMO está pensando. ¡Ya sé!'"
)

# --- Audio ---
AUDIO_SAMPLE_RATE = 16000   # Hz — required by Whisper and OpenWakeWord
AUDIO_CHANNELS = 1           # Mono
AUDIO_DTYPE = "float32"
AUDIO_CHUNK_SECONDS = 0.5   # Duration of each chunk fed to wake word detection
AUDIO_RECORD_SECONDS = 5    # How long to record speech after wake word fires
MIC_DEVICE_INDEX = 8     # Set to an integer to select a specific input device; None = system default

# --- Wake Word ---
WAKEWORD_MODEL = "alexa"   # Built-in OpenWakeWord model name
WAKEWORD_THRESHOLD = 0.5        # Confidence threshold (0.0 – 1.0)

# --- Speech-to-Text ---
WHISPER_MODEL = "medium"        # Whisper model size: tiny, base, small, medium, large
WHISPER_DEVICE = "cpu"          # "cpu" or "cuda"
WHISPER_COMPUTE_TYPE = "int8"   # Quantisation: int8 for CPU, float16 for GPU

# --- Text-to-Speech ---
PIPER_MODEL_PATH = "models/voices/es_MX-ald-medium.onnx"
