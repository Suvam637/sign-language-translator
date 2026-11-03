#!/usr/bin/env python
"""Evaluate trained models: classification report, per-source accuracy, confusion matrix PNG.

    python scripts/evaluate.py                       # held-out test split of both models
    python scripts/evaluate.py --task static --eval-on external

--eval-on test      the group-separated test split saved at training time (unbiased estimate)
--eval-on own/external   ALL samples of that source. If the model was trained on that source the
                         result is optimistic; it is a real cross-domain test only when the
                         model was trained WITHOUT it (train with  --sources own  then evaluate
                         on external, and vice versa) - a good check of generalisation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slt import config as C  # noqa: E402
from slt.evaluate import evaluate_dynamic, evaluate_static  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="both", choices=["static", "dynamic", "both"])
    ap.add_argument("--eval-on", default="test", choices=["test", "own", "external"])
    ap.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    ap.add_argument("--models-dir", type=Path, default=C.MODELS_DIR)
    a = ap.parse_args()
    ran = False
    for task, fn, fname in (("static", evaluate_static, "static_model.joblib"),
                            ("dynamic", evaluate_dynamic, "dynamic_model.pt")):
        if a.task in (task, "both"):
            if not (a.models_dir / fname).exists():
                print(f"[eval] skipping {task}: {a.models_dir / fname} not found")
                continue
            print(f"\n=== {task} model ({a.eval_on}) ===")
            fn(a.models_dir, a.data_dir, a.eval_on)
            ran = True
    if not ran:
        raise SystemExit("Nothing evaluated - train a model first.")


if __name__ == "__main__":
    main()
