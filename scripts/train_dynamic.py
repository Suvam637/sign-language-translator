#!/usr/bin/env python
"""Train the dynamic sign model (words / phrases) on own + external landmark sequences.

    python scripts/train_dynamic.py
    python scripts/train_dynamic.py --epochs 120 --hidden 192 --own-boost 5

Uses a BiGRU with attention pooling (PyTorch). Augmentation (time-warp crop, rotation, jitter,
mirroring, simulated hand drop-outs) is applied on the fly to the training split only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slt import config as C  # noqa: E402
from slt.training import train_dynamic  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=list(C.SOURCES), choices=list(C.SOURCES))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--own-boost", type=int, default=3, help="repeat own sequences this many times")
    ap.add_argument("--max-per-class", type=int, default=None, help="cap external sequences per class")
    ap.add_argument("--no-velocity", action="store_true", help="do not append frame-to-frame velocity features")
    ap.add_argument("--patience", type=int, default=15, help="early-stopping patience (epochs)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq-len", type=int, default=C.SEQ_LEN)
    ap.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    ap.add_argument("--models-dir", type=Path, default=C.MODELS_DIR)
    a = ap.parse_args()
    train_dynamic(a.data_dir, a.models_dir, tuple(a.sources), a.epochs, a.batch_size, a.lr, a.hidden, a.layers,
                  a.dropout, a.own_boost, a.max_per_class, not a.no_velocity, a.patience, a.val_frac,
                  a.test_frac, a.seed, a.device, a.seq_len)


if __name__ == "__main__":
    main()
