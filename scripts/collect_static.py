#!/usr/bin/env python
"""Record YOUR OWN static hand-sign samples (letters, digits, single-pose signs) from the webcam.

    python scripts/collect_static.py                       # A-Z
    python scripts/collect_static.py --labels A B C 1 2 3  # custom labels

Controls
    type a label key (e.g. A)  select that label (single-character labels)
    . / ,                      next / previous label
    SPACE                      countdown, then record a burst of samples
    ESC                        quit

Tips for a model that generalises
    * Record each label several times (several bursts), moving the hand slowly through
      different distances, heights and slight rotations during a burst.
    * Different lighting / background / clothing / people give the biggest accuracy gains.
    * Each burst is one "group" in the train/val/test split, so record >= 5 bursts per label.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from slt import config as C  # noqa: E402
from slt.data import append_static_rows, load_static, normalize_label, static_csv_path  # noqa: E402
from slt.landmarks import LandmarkExtractor, draw_raw, pick_hand  # noqa: E402
from slt.ui import open_camera, panel, put_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", default=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    ap.add_argument("--burst", type=int, default=60, help="samples recorded per SPACE press (default 60)")
    ap.add_argument("--interval", type=float, default=0.06, help="min seconds between samples (default 0.06)")
    ap.add_argument("--countdown", type=float, default=2.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    ap.add_argument("--backend", default="auto", choices=["auto", "legacy", "tasks"])
    args = ap.parse_args()

    labels = [normalize_label(l) for l in args.labels]
    csv_path = static_csv_path("own", args.data_dir)
    existing = load_static(("own",), args.data_dir)
    counts = {l: int((existing["labels"] == l).sum()) for l in labels}
    session = datetime.now().strftime("%Y%m%d%H%M%S")
    burst_id, cur = 0, 0
    state, t_state, buf = "idle", 0.0, []
    last_sample = 0.0
    key_map = {ord(l.lower()): i for i, l in enumerate(labels) if len(l) == 1}
    key_map.update({ord(l.upper()): i for i, l in enumerate(labels) if len(l) == 1})

    cap = open_camera(args.camera)
    ext = LandmarkExtractor(video=True, use_pose=False, num_hands=1, backend=args.backend)
    print(f"[collect] saving to {csv_path}\n[collect] SPACE = record a burst, ESC = quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[collect] camera frame not available")
                break
            now = time.perf_counter()
            raw = ext.process(frame, now)
            hand = pick_hand(raw)
            draw_raw(frame, raw, pose=False)
            view = cv2.flip(frame, 1)
            label = labels[cur]

            if state == "countdown" and now - t_state >= args.countdown:
                state, buf, last_sample = "recording", [], 0.0
            if state == "recording" and hand is not None and now - last_sample >= args.interval:
                buf.append(hand)
                last_sample = now
                if len(buf) >= args.burst:
                    burst_id += 1
                    group = f"{session}_b{burst_id}"
                    append_static_rows(csv_path, [(label, group, h) for h in buf])
                    counts[label] += len(buf)
                    print(f"[collect] saved {len(buf)} samples for {label} (total {counts[label]})")
                    state, buf = "idle", []

            panel(view, 100)
            put_text(view, f"Label: {label}   samples: {counts[label]}", (15, 35), 1.0, (0, 255, 255))
            if state == "idle":
                put_text(view, "SPACE: record burst | . , : change label | ESC: quit", (15, 75), 0.6)
            elif state == "countdown":
                put_text(view, f"Get ready... {max(0.0, args.countdown - (now - t_state)):.1f}s", (15, 75), 0.9, (0, 200, 255))
            else:
                msg = f"RECORDING {len(buf)}/{args.burst}" + ("" if hand is not None else "  - show your hand!")
                put_text(view, msg, (15, 75), 0.9, (0, 0, 255) if hand is None else (0, 255, 0))
            cv2.imshow("collect_static", view)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if state == "idle":
                if key == 32:
                    state, t_state = "countdown", now
                elif key == ord("."):
                    cur = (cur + 1) % len(labels)
                elif key == ord(","):
                    cur = (cur - 1) % len(labels)
                elif key in key_map:
                    cur = key_map[key]
    finally:
        ext.close()
        cap.release()
        cv2.destroyAllWindows()
    total = sum(counts.values())
    print(f"[collect] done. {total} own static samples in {csv_path}")


if __name__ == "__main__":
    main()
