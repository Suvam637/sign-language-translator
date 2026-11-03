#!/usr/bin/env python
"""Translate an image or a video file (no webcam needed).

    python scripts/predict_file.py --input photo.jpg                 # static sign in a photo (top-3)
    python scripts/predict_file.py --input clip.mp4 --output out.mp4  # video -> text, annotated video
    python scripts/predict_file.py --input clip.mp4 --show

Videos use exactly the same translator as the live script (letters + words).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from slt import config as C  # noqa: E402
from slt.inference import SignTranslator, load_predictors  # noqa: E402
from slt.landmarks import LandmarkExtractor, draw_raw, pick_hand  # noqa: E402
from slt.ui import panel, put_text, wrap_text  # noqa: E402


def predict_image(path: Path, models_dir: Path, backend: str) -> None:
    static, _ = load_predictors(models_dir, "static")
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cannot read image {path}")
    with LandmarkExtractor(video=False, use_pose=False, num_hands=1, backend=backend) as ext:
        hand = pick_hand(ext.process(img))
    if hand is None:
        print("No hand detected in the image.")
        return
    p = static.predict_proba(hand)
    for k in np.argsort(p)[::-1][:3]:
        print(f"  {static.classes[k]:>12s}  {p[k]:.3f}")


def predict_video(path: Path, models_dir: Path, mode: str, dynamic_mode: str, out_path: Path | None,
                  show: bool, backend: str) -> None:
    static, dynamic = load_predictors(models_dir, mode)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if out_path:
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    tr = SignTranslator(static, dynamic, dynamic_mode=dynamic_mode)
    ext = LandmarkExtractor(video=True, use_pose=dynamic is not None, num_hands=2 if dynamic else 1, backend=backend)
    i, t = 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            raw = ext.process(frame, t)
            out = tr.update(raw, t)
            for kind, tok in out["committed"]:
                print(f"[{t:7.2f}s] {kind:7s} -> {tok}")
            if writer or show:
                draw_raw(frame, raw)
                panel(frame, 60 + 36 * 1, bottom=True)
                put_text(frame, wrap_text(out["text"] or "...", 55)[-1], (15, h - 22), 1.0)
                if writer:
                    writer.write(frame)
                if show:
                    cv2.imshow("predict_file", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
            i += 1
        for kind, tok in tr.flush(t)["committed"]:
            print(f"[{t:7.2f}s] {kind:7s} -> {tok}")
    finally:
        ext.close()
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    print("\nTRANSLATION:", tr.sentence.text() or "(nothing recognised)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, help="write an annotated video here (mp4)")
    ap.add_argument("--mode", default="auto", choices=["auto", "static", "dynamic", "both"])
    ap.add_argument("--dynamic-mode", default="segment", choices=["segment", "sliding"])
    ap.add_argument("--models-dir", type=Path, default=C.MODELS_DIR)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--backend", default="auto", choices=["auto", "legacy", "tasks"])
    a = ap.parse_args()
    if a.input.suffix.lower() in C.IMAGE_EXTS:
        predict_image(a.input, a.models_dir, a.backend)
    else:
        predict_video(a.input, a.models_dir, a.mode, a.dynamic_mode, a.output, a.show, a.backend)


if __name__ == "__main__":
    main()
