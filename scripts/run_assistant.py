"""
scripts/run_assistant.py
------------------------
Launcher for the BMO voice assistant.

Run from the project root:

    python scripts/run_assistant.py

Make sure:
  1. Ollama is running:     ollama serve
  2. A Piper voice model exists at the path set in core/config.py
  3. Your microphone and speakers are connected
"""

import os
import sys
import warnings

# Suppress all Python warnings (HuggingFace, torch deprecations, etc.)
warnings.filterwarnings("ignore")

# Hide the pygame "Hello from the pygame community" startup banner.
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# Ensure the project root is on the import path regardless of where this script
# is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import get_logger
from voice.assistant import Assistant

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Booting system")
    assistant = Assistant()
    assistant.start()
