# BMO Voice Models

This directory holds the offline AI model files required by the assistant.
**Model files are not committed to git** (they are large binaries).
Download them manually and place them here before running BMO.

---

## Text-to-Speech — Piper

BMO uses [Piper](https://github.com/rhasspy/piper) for fully offline TTS.

### Required files

| File | Size |
|------|------|
| `en_US-lessac-medium.onnx` | ~63 MB |
| `en_US-lessac-medium.onnx.json` | ~3 KB |

### Download

```bash
# From the project root
mkdir -p models
cd models

curl -LO https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx
curl -LO https://github.com/rhasspy/piper/releases/download/v1.2.0/en_US-lessac-medium.onnx.json
```

Or browse all available voices at:
https://github.com/rhasspy/piper/releases

---

## Model path configuration

The model path is set in `core/config.py`:

```python
PIPER_MODEL_PATH = "models/en_US-lessac-medium.onnx"
```

Change this value if you use a different voice.

---

## Notes

- The `.onnx.json` sidecar **must** be in the same directory as the `.onnx` file and share the same base name.
- Piper voices run on CPU with no GPU required.
