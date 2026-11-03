#!/usr/bin/env python
"""Download the MediaPipe Tasks model files (only needed when mediapipe has no legacy
``mp.solutions`` API, i.e. mediapipe >= 0.10.30 / Python 3.13+).

    python scripts/download_models.py

If you are offline, download the two files manually and put them in models/mediapipe/
(or point SLT_HAND_LANDMARKER / SLT_POSE_LANDMARKER_LITE to them). URLs: see TASK_MODELS in slt/config.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slt.landmarks import ensure_task_models  # noqa: E402


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    try:
        for name, path in ensure_task_models().items():
            print(f"ok  {name}: {path}")
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
