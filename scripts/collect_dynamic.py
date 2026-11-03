#!/usr/bin/env python
"""Record YOUR OWN dynamic signs (words / phrases that involve movement) from the webcam.

    python scripts/collect_dynamic.py --labels HELLO THANK_YOU YES NO HELP WATER IDLE

Controls
    1-9          select label by its number
    . / ,        next / previous label
    SPACE        countdown, then record one clip (default 3 s) - perform the sign once
    U            undo (delete) the last clip saved in this session
    ESC          quit

How it works
    Each clip is trimmed to the stretch where a hand is visible and resampled to 30 frames
    (the same preprocessing is used for imported datasets and for live prediction), so you
    do not have to time the sign precisely. Keep your hands DOWN / out of view before and
    after the sign.

Recommended: 30-60 clips per word, from different distances, speeds, lighting and signers.
Also record an IDLE class (hands raised but not signing, random gestures, resting hand).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from slt import config as C  # noqa: E402
from slt.data import dynamic_dir, normalize_label, save_sequence  # noqa: E402
from slt.features import hands_mask, prepare_sequence  # noqa: E402
from slt.landmarks import LandmarkExtractor, draw_raw  # noqa: E402
from slt.ui import open_camera, panel, put_text  # noqa: E402

DEFAULT_LABELS = ["HELLO", "THANK_YOU", "YES", "NO", "HELP", "WATER", "IDLE"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    ap.add_argument("--seconds", type=float, default=3.0, help="clip length in seconds (default 3)")
    ap.add_argument("--countdown", type=float, default=2.0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=C.SEQ_LEN)
    ap.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    ap.add_argument("--backend", default="auto", choices=["auto", "legacy", "tasks"])
    args = ap.parse_args()

    labels = [normalize_label(l) for l in args.labels]
    counts = {l: len(list((dynamic_dir("own", args.data_dir) / l).glob("*.npy"))) for l in labels}
    session = datetime.now().strftime("%Y%m%d%H%M%S")
    cur, state, t_state = 0, "idle", 0.0
    clip: list[tuple[float, np.ndarray]] = []
    saved: list[Path] = []
    message, msg_until = "", 0.0

    cap = open_camera(args.camera)
    ext = LandmarkExtractor(video=True, use_pose=True, num_hands=2, backend=args.backend)
    print(f"[collect] saving to {dynamic_dir('own', args.data_dir)}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[collect] camera frame not available")
                break
            now = time.perf_counter()
            raw = ext.process(frame, now)
            draw_raw(frame, raw)
            view = cv2.flip(frame, 1)
            label = labels[cur]

            if state == "countdown" and now - t_state >= args.countdown:
                state, t_state, clip = "recording", now, []
            if state == "recording":
                clip.append((now, raw))
                if now - t_state >= args.seconds:
                    raws = np.stack([r for _, r in clip])
                    seq = prepare_sequence(raws, args.seq_len, keep_if_empty=label in C.IDLE_LABELS)
                    if seq is None:
                        message = "No hand detected - clip discarded, try again"
                    else:
                        counts[label] += 1
                        name = f"{session}_{counts[label]:04d}"
                        saved.append(save_sequence("own", label, name, seq, args.data_dir))
                        message = f"Saved clip #{counts[label]} for {label} (hands in {hands_mask(raws).mean():.0%} of frames)"
                    print("[collect]", message)
                    msg_until = now + 3.0
                    state, clip = "idle", []

            panel(view, 100)
            put_text(view, f"[{cur + 1}] {label}   clips: {counts[label]}", (15, 35), 1.0, (0, 255, 255))
            if state == "idle":
                put_text(view, message if now < msg_until else "SPACE: record | 1-9 . , : label | U: undo | ESC: quit",
                         (15, 75), 0.6, (0, 255, 0) if now < msg_until else (255, 255, 255))
            elif state == "countdown":
                put_text(view, f"Get ready... {max(0.0, args.countdown - (now - t_state)):.1f}s (hands down)", (15, 75), 0.8, (0, 200, 255))
            else:
                left = max(0.0, args.seconds - (now - t_state))
                put_text(view, f"SIGN NOW!  {left:.1f}s", (15, 75), 0.9, (0, 0, 255))
            cv2.imshow("collect_dynamic", view)

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
                elif ord("1") <= key <= ord("9") and key - ord("1") < len(labels):
                    cur = key - ord("1")
                elif key in (ord("u"), ord("U")) and saved:
                    p = saved.pop()
                    p.unlink(missing_ok=True)
                    lab = p.parent.name
                    counts[lab] = max(0, counts.get(lab, 1) - 1)
                    message, msg_until = f"Deleted {p.name}", now + 2.0
    finally:
        ext.close()
        cap.release()
        cv2.destroyAllWindows()
    print("[collect] done. clips per label:", counts)


if __name__ == "__main__":
    main()
