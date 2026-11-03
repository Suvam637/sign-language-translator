#!/usr/bin/env python
"""Import an OUTSIDE image dataset of static signs (e.g. an ASL / ISL alphabet photo set).

Expected layout (one sub-folder per class, any image names)::

    /path/to/dataset/
        A/ img001.jpg img002.jpg ...
        B/ ...

or a CSV file with one row per image::

    python scripts/import_external_static.py --csv labels.csv --image-root /path/to/images \
           --path-col filename --label-col class

Every image is passed through MediaPipe; only the 21 hand landmarks are stored (the
images themselves are not copied), into ``data/external/static/landmarks.csv``.

Examples
    python scripts/import_external_static.py --root ~/datasets/asl_alphabet_train --max-per-class 1500 \
           --skip DEL NOTHING SPACE
    python scripts/import_external_static.py --root ~/datasets/isl --label-map my_label_map.json

Labels are upper-cased and spaces become underscores, so "thank you" -> THANK_YOU. Use
--label-map (a JSON dict such as {"del": "DELETE"}) to rename external classes so they
match the labels you used for your own data.

Check the licence / terms of use of every dataset you import.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from slt import config as C  # noqa: E402
from slt.data import append_static_rows, normalize_label, static_csv_path  # noqa: E402
from slt.landmarks import LandmarkExtractor, pick_hand  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(it, **_kw):
        return it


def collect_items(args) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    if args.csv:
        root = Path(args.image_root or Path(args.csv).parent)
        with open(args.csv, newline="") as fh:
            for row in csv.DictReader(fh):
                items.append((root / row[args.path_col], row[args.label_col]))
    else:
        root = Path(args.root)
        if not root.is_dir():
            raise SystemExit(f"--root {root} is not a directory")
        for cls in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(cls.rglob("*")):
                if f.suffix.lower() in C.IMAGE_EXTS:
                    items.append((f, cls.name))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="dataset folder with one sub-folder per class")
    ap.add_argument("--csv", help="CSV with image path and label columns (alternative to --root)")
    ap.add_argument("--image-root", help="base folder for relative paths in --csv")
    ap.add_argument("--path-col", default="path")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--max-per-class", type=int, default=None, help="limit images processed per class")
    ap.add_argument("--skip", nargs="*", default=[], help="class names to ignore (e.g. NOTHING SPACE)")
    ap.add_argument("--label-map", type=Path, help="JSON file {external_name: your_name}")
    ap.add_argument("--group-chunk", type=int, default=50,
                    help="consecutive images (sorted by name) are treated as one group when splitting, "
                         "because neighbouring frames of a dataset are usually near-duplicates (default 50)")
    ap.add_argument("--max-side", type=int, default=640, help="downscale larger images for speed")
    ap.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    ap.add_argument("--overwrite", action="store_true", help="delete the existing external static CSV first")
    ap.add_argument("--backend", default="auto", choices=["auto", "legacy", "tasks"])
    args = ap.parse_args()
    if not args.root and not args.csv:
        ap.error("give --root or --csv")

    label_map = {normalize_label(k): normalize_label(v) for k, v in
                 (json.loads(args.label_map.read_text()).items() if args.label_map else [])}
    skip = {normalize_label(s) for s in args.skip}
    out_csv = static_csv_path("external", args.data_dir)
    if args.overwrite and out_csv.exists():
        out_csv.unlink()
    elif out_csv.exists() and out_csv.stat().st_size:
        print(f"[import] NOTE: {out_csv} already exists - new rows are appended "
              "(use --overwrite to start fresh, otherwise re-importing creates duplicates)")

    per_class: dict[str, list[Path]] = defaultdict(list)
    for path, label in collect_items(args):
        lab = label_map.get(normalize_label(label), normalize_label(label))
        if lab in skip or normalize_label(label) in skip:
            continue
        per_class[lab].append(path)
    if not per_class:
        raise SystemExit("No images found.")
    for lab in per_class:
        per_class[lab] = sorted(per_class[lab])
        if args.max_per_class and len(per_class[lab]) > args.max_per_class:
            step = len(per_class[lab]) / args.max_per_class            # even subsample, keeps variety
            per_class[lab] = [per_class[lab][int(i * step)] for i in range(args.max_per_class)]
    total = sum(len(v) for v in per_class.values())
    print(f"[import] {len(per_class)} classes, {total} images to process")

    ext = LandmarkExtractor(video=False, use_pose=False, num_hands=1, backend=args.backend)
    stats = {}
    try:
        for lab, files in sorted(per_class.items()):
            rows = []
            for i, f in enumerate(tqdm(files, desc=lab, leave=False)):
                img = cv2.imread(str(f))
                if img is None:
                    continue
                h, w = img.shape[:2]
                if max(h, w) > args.max_side:
                    s = args.max_side / max(h, w)
                    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                hand = pick_hand(ext.process(img))
                if hand is not None:
                    rows.append((lab, f"{lab}_{i // args.group_chunk}", hand))
            append_static_rows(out_csv, rows)
            stats[lab] = (len(rows), len(files))
            print(f"[import] {lab:>12s}: {len(rows):5d}/{len(files):5d} images had a detectable hand")
    finally:
        ext.close()
    ok = sum(a for a, _ in stats.values())
    print(f"[import] done: {ok}/{total} samples written to {out_csv}")
    poor = [l for l, (a, b) in stats.items() if b and a / b < 0.5]
    if poor:
        print(f"[import] WARNING: < 50% detection for {poor} - check the images / mirroring / resolution")


if __name__ == "__main__":
    main()
