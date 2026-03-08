# UI Module — Placeholder

This directory is reserved for future user-interface components, such as:

- A terminal dashboard (Rich / Textual)
- A web dashboard (FastAPI + HTMX)
- A local GUI (Tkinter / PyQt)

The `core/event_bus` module provides the integration point:
subscribe to events like `wake_word_detected`, `speech_transcribed`,
`llm_response`, and `tts_speaking` to drive any UI layer without
modifying the voice pipeline.
