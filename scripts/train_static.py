#!/usr/bin/env python
"""Train the static sign model (letters / digits / single-pose signs) on own + external data.

    python scripts/train_static.py
    python scripts/train_static.py --model mlp --n-aug 6 --own-boost 5 --max-per-class 1500

Own recordings are repeated ``--own-boost`` times so they are not drowned out by a big
external dataset; ``--max-per-class`` caps how many EXTERNAL samples each class contributes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slt import config as C  # noqa: E402
from slt.training import train_static  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="auto", choices=["auto", "rf", "extra", "mlp"],
                    help="auto = train random forest and MLP, keep the one with the best validation accuracy")
    ap.add_argument("--sources", nargs="+", default=list(C.SOURCES), choices=list(C.SOURCES))
    ap.add_argument("--n-aug", type=int, default=4, help="augmented copies per training sample")
    ap.add_argument("--own-boost", type=int, default=3, help="repeat own samples this many times")
    ap.add_argument("--max-per-class", type=int, default=None, help="cap external samples per class")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    ap.add_argument("--models-dir", type=Path, default=C.MODELS_DIR)
    a = ap.parse_args()
    train_static(a.data_dir, a.models_dir, a.model, tuple(a.sources), a.n_aug, a.own_boost, a.max_per_class,
                 a.val_frac, a.test_frac, a.seed)


if __name__ == "__main__":
    main()
