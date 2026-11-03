#!/usr/bin/env python
"""Real-time sign language translation from the webcam.

    python scripts/predict_live.py                      # uses whichever models exist (static and/or dynamic)
    python scripts/predict_live.py --mode dynamic --dynamic-mode sliding

Keys:  C clear sentence | BACKSPACE (or B) delete last | S speak sentence | L toggle landmarks | Q / ESC quit

How to use it
    * Letters / digits: hold the hand shape still for about half a second.
    * Words: lower your hands, perform the sign, lower your hands again (segment mode),
      or keep signing continuously with --dynamic-mode sliding.
    * A pause of ~1.5 s after letters ends the spelled word.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from slt import config as C  # noqa: E402
from slt.inference import SignTranslator, load_predictors, speak_async  # noqa: E402
from slt.landmarks import LandmarkExtractor, draw_raw  # noqa: E402
from slt.ui import open_camera, panel, put_text, wrap_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="auto", choices=["auto", "static", "dynamic", "both"])
    ap.add_argument("--dynamic-mode", default="segment", choices=["segment", "sliding"])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--models-dir", type=Path, default=C.MODELS_DIR)
    ap.add_argument("--min-prob-static", type=float, default=0.85)
    ap.add_argument("--min-prob-dynamic", type=float, default=0.80)
    ap.add_argument("--hold-static", type=int, default=10, help="frames a letter must stay stable (default 10)")
    ap.add_argument("--hand-gap", type=float, default=0.4, help="seconds without hands that ends a sign (segment mode)")
    ap.add_argument("--auto-speak", action="store_true", help="speak every recognised word / phrase (needs pyttsx3)")
    ap.add_argument("--no-landmarks", action="store_true", help="do not draw the skeleton")
    ap.add_argument("--backend", default="auto", choices=["auto", "legacy", "tasks"])
    args = ap.parse_args()

    static, dynamic = load_predictors(args.models_dir, args.mode)
    tr = SignTranslator(static, dynamic, dynamic_mode=args.dynamic_mode, min_prob_static=args.min_prob_static,
                        min_prob_dynamic=args.min_prob_dynamic, hold_static=args.hold_static,
                        hand_gap_sec=args.hand_gap)
    print(f"[live] static model: {'yes' if static else 'no'} | dynamic model: {'yes' if dynamic else 'no'}")
    if static:
        print(f"[live] static classes : {static.classes}")
    if dynamic:
        print(f"[live] dynamic classes: {dynamic.classes}")

    cap = open_camera(args.camera)
    ext = LandmarkExtractor(video=True, use_pose=dynamic is not None, num_hands=2 if dynamic else 1,
                            backend=args.backend)
    draw, t0, fps, last_f = not args.no_landmarks, time.perf_counter(), 0.0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[live] camera frame not available")
                break
            t = time.perf_counter() - t0
            raw = ext.process(frame, t)
            out = tr.update(raw, t)
            for kind, tok in out["committed"]:
                print(f"[{t:7.2f}s] {kind:7s} -> {tok:15s} | {out['text']}")
                if args.auto_speak and kind == "dynamic" and tr.sentence.words:
                    speak_async(tr.sentence.words[-1])
            if draw:
                draw_raw(frame, raw)
            view = cv2.flip(frame, 1)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last_f, 1e-6)
            last_f = now

            panel(view, 70)
            s = f"{out['static'][0]} {out['static'][1]:.2f}" if out["static"] else "-"
            d = f"{out['dynamic'][0]} {out['dynamic'][1]:.2f}" if out["dynamic"] else "-"
            put_text(view, f"static: {s}    dynamic: {d}", (15, 30), 0.8, (0, 255, 255))
            put_text(view, f"{out['state']}   {fps:4.1f} fps", (15, 58), 0.6, (0, 255, 0) if out["hands"] else (200, 200, 200))
            lines = wrap_text(out["text"] or "...", 55)[-3:]
            h = view.shape[0]
            panel(view, 20 + 38 * len(lines), bottom=True)
            for i, line in enumerate(lines):
                put_text(view, line, (15, h - 20 - 38 * (len(lines) - 1 - i)), 1.0)
            cv2.imshow("Sign Language Translator", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                tr.clear()
            elif key in (8, ord("b"), ord("B")):
                tr.sentence.backspace()
            elif key in (ord("s"), ord("S")):
                speak_async(out["text"])
            elif key in (ord("l"), ord("L")):
                draw = not draw
    finally:
        ext.close()
        cap.release()
        cv2.destroyAllWindows()
    print("[live] final text:", tr.sentence.text())


if __name__ == "__main__":
    main()
