"""
ui/face.py
----------
Loads face animation frame paths from public/faces/.

Each sub-directory name is a state (idle, listening, thinking, speaking, error,
capturing, warmup).  All PNG files inside are loaded and sorted alphabetically so
frames play back in the correct order.

Usage::

    from ui.face import load_face_paths

    faces = load_face_paths()
    # faces["speaking"] -> ["/abs/path/speaking 01.png", "…02.png", "…03.png"]
"""

from pathlib import Path

# Resolve path relative to this file so it always works regardless of where
# the script is launched from.
FACES_DIR = Path(__file__).parent.parent / "public" / "faces"


def load_face_paths() -> dict[str, list[str]]:
    """
    Return a dict mapping each state name to a sorted list of PNG file paths.

    Only directories that contain at least one PNG are included.  Frames within
    each state are sorted alphabetically — name them `state 01.png`,
    `state 02.png`, … to guarantee playback order.
    """
    faces: dict[str, list[str]] = {}

    if not FACES_DIR.exists():
        print(f"[UI] Faces directory not found: {FACES_DIR}")
        return faces

    for state_dir in sorted(FACES_DIR.iterdir()):
        if not state_dir.is_dir():
            continue

        frames = sorted(
            str(f) for f in state_dir.iterdir() if f.suffix.lower() == ".png"
        )

        if frames:
            faces[state_dir.name] = frames

    return faces
