#!/usr/bin/env python
"""Generate SYNTHETIC landmark data (no camera needed) to test the whole pipeline.

    python scripts/make_synthetic_data.py                  # writes ./demo_data
    python scripts/train_static.py  --data-dir demo_data --models-dir demo_models
    python scripts/train_dynamic.py --data-dir demo_data --models-dir demo_models --epochs 30
    python scripts/evaluate.py      --data-dir demo_data --models-dir demo_models

The data are procedurally generated shapes, NOT real signs. Never mix them with real data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slt import config as C  # noqa: E402
from slt.synthetic import make_dynamic_dataset, make_static_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=C.ROOT / "demo_data")
    ap.add_argument("--static-labels", nargs="+", default=list("ABCDEFGH"))
    ap.add_argument("--dynamic-labels", nargs="+", default=["HELLO", "THANK_YOU", "YES", "NO", "HELP", "WATER"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make_static_dataset(a.out, a.static_labels, 60, "own", a.seed)
    make_static_dataset(a.out, a.static_labels, 150, "external", a.seed)
    make_dynamic_dataset(a.out, a.dynamic_labels, 14, "own", a.seed)
    make_dynamic_dataset(a.out, a.dynamic_labels, 40, "external", a.seed)
    print(f"synthetic data written to {a.out}")


if __name__ == "__main__":
    main()
