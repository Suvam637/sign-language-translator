#!/usr/bin/env python
"""Import an OUTSIDE video dataset of signs (words / phrases) as landmark sequences.

Expected layout (one sub-folder per word, any video names)::

    /path/to/videos/
        hello/  clip001.mp4 clip002.mp4 ...
        thank_you/ ...

or a CSV with one row per video::

    python scripts/import_external_dynamic.py --csv labels.csv --video-root /path/to/videos \
           --path-col video --label-col gloss

Each video is run through MediaPipe (pose + hands), trimmed to the stretch where a hand is
visible, resampled to 30 frames and stored as ``data/external/dynamic/<LABEL>/<name>.npy``.
The videos themselves are not copied.

Examples
    python scripts/import_external_dynamic.py --root ~/datasets/wlasl_subset --max-per-class 40
    python scripts/import_external_dynamic.py --root ~/datasets/include --label-map map.json --classes HELLO YES NO

Dataset notes (verify availability and licence yourself): large public sign video corpora such as
WLASL / MS-ASL (American Sign Language) or INCLUDE (Indian Sign Language) are commonly used. Many
require you to download the videos from their own pages. Convert them to the folder layout above
(one sub-folder per sign) and run this script.

Make sure label names match the names you use in scripts/collect_dynamic.py (use --label-map).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from slt import config as C  # noqa: E402
from slt.data import dynamic_dir, normalize_label, save_sequence  # noqa: E402
from slt.features import hands_mask, prepare_sequence  # noqa: E402
from slt.landmarks import LandmarkExtractor  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kw):
        return it


def collect_items(args) -> list[tuple[Path, str]]:
    items = []
    if args.csv:
        root = Path(args.video_root or Path(args.csv).parent)
        with open(args.csv, newline="") as fh:
            for row in csv.DictReader(fh):
                items.append((root / row[args.path_col], row[args.label_col]))
    else:
        root = Path(args.root)
        if not root.is_dir():
            raise SystemExit(f"--root {root} is not a directory")
        for cls in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(cls.rglob("*")):
                if f.suffix.lower() in C.VIDEO_EXTS:
                    items.append((f, cls.name))
    return items


def video_to_raw(path: Path, max_frames: int, max_side: int, backend: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, math.ceil(n / max_frames)) if n > 0 else 1
    ext = LandmarkExtractor(video=True, use_pose=True, num_hands=2, backend=backend)
    raws, i = [], 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                h, w = frame.shape[:2]
                if max(h, w) > max_side:
                    s = max_side / max(h, w)
                    frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                raws.append(ext.process(frame, i / fps))
            i += 1
    finally:
        ext.close()
        cap.release()
    return np.stack(raws) if raws else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="folder with one sub-folder per sign")
    ap.add_argument("--csv", help="CSV with video path and label columns (alternative to --root)")
    ap.add_argument("--video-root", help="base folder for relative paths in --csv")
    ap.add_argument("--path-col", default="path")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--classes", nargs="*", help="only import these classes (after --label-map)")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--label-map", type=Path, help="JSON file {external_name: your_name}")
    ap.add_argument("--max-per-class", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=120, help="process at most this many frames per video")
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--min-hand-ratio", type=float, default=0.3,
                    help="discard videos where a hand is visible in less than this share of frames")
    ap.add_argument("--seq-len", type=int, default=C.SEQ_LEN)
    ap.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    ap.add_argument("--backend", default="auto", choices=["auto", "legacy", "tasks"])
    args = ap.parse_args()
    if not args.root and not args.csv:
        ap.error("give --root or --csv")

    label_map = {normalize_label(k): normalize_label(v) for k, v in
                 (json.loads(args.label_map.read_text()).items() if args.label_map else [])}
    skip = {normalize_label(s) for s in args.skip}
    only = {normalize_label(s) for s in args.classes} if args.classes else None

    per_class: dict[str, list[Path]] = defaultdict(list)
    for path, label in collect_items(args):
        lab = label_map.get(normalize_label(label), normalize_label(label))
        if lab in skip or (only is not None and lab not in only):
            continue
        per_class[lab].append(path)
    if not per_class:
        raise SystemExit("No videos found.")
    for lab in per_class:
        per_class[lab] = sorted(per_class[lab])[: args.max_per_class] if args.max_per_class else sorted(per_class[lab])
    total = sum(len(v) for v in per_class.values())
    print(f"[import] {len(per_class)} classes, {total} videos")

    done = skipped = 0
    for lab, files in sorted(per_class.items()):
        ok_here = 0
        for f in tqdm(files, desc=lab, leave=False):
            out = dynamic_dir("external", args.data_dir) / lab / f"{f.stem}.npy"
            if out.exists():
                ok_here += 1
                continue
            raw = video_to_raw(f, args.max_frames, args.max_side, args.backend)
            if raw is None or hands_mask(raw).mean() < args.min_hand_ratio:
                skipped += 1
                continue
            seq = prepare_sequence(raw, args.seq_len)
            if seq is None:
                skipped += 1
                continue
            save_sequence("external", lab, f.stem, seq, args.data_dir)
            ok_here += 1
            done += 1
        print(f"[import] {lab:>16s}: {ok_here}/{len(files)} videos usable")
    print(f"[import] done: {done} new sequences, {skipped} videos skipped (unreadable / hands rarely visible)")


if __name__ == "__main__":
    main()
