#!/usr/bin/env python
"""Diagnose your installation: Python / package versions, MediaPipe backend, camera, extraction.

    python scripts/check_setup.py
    python scripts/check_setup.py --camera 0
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None, help="also try to grab a frame from this camera index")
    a = ap.parse_args()

    print(f"python      {sys.version.split()[0]}", "OK" if sys.version_info >= (3, 11) else "-> needs Python 3.11+")
    for mod in ("numpy", "cv2", "mediapipe", "sklearn", "joblib", "torch", "matplotlib", "tqdm", "pyttsx3"):
        try:
            m = importlib.import_module(mod)
            print(f"{mod:11s} {getattr(m, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            opt = " (optional)" if mod in ("pyttsx3", "tqdm") else ""
            print(f"{mod:11s} MISSING{opt}: {exc.__class__.__name__}")

    import numpy as np

    from slt.landmarks import LandmarkExtractor

    try:
        ext = LandmarkExtractor(video=False, use_pose=True)
    except Exception as exc:  # noqa: BLE001
        print("MediaPipe extractor FAILED:", exc)
        return
    print(f"mediapipe backend: {ext.backend}" + ("  (Tasks API: models are downloaded on first use)" if ext.backend == "tasks" else ""))
    raw = ext.process(np.zeros((480, 640, 3), np.uint8))
    print("blank-frame extraction OK, vector length", raw.shape[0])
    ext.close()

    if a.camera is not None:
        from slt.ui import open_camera

        try:
            cap = open_camera(a.camera)
            ok, frame = cap.read()
            cap.release()
            print(f"camera {a.camera}:", f"OK {frame.shape[1]}x{frame.shape[0]}" if ok else "opened but no frame")
        except Exception as exc:  # noqa: BLE001
            print(f"camera {a.camera}: FAILED - {exc}")


if __name__ == "__main__":
    main()
