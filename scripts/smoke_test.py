#!/usr/bin/env python
"""One-command self-test on synthetic data (no camera, ~1-2 minutes): verifies that
data -> train -> evaluate -> real-time translation all work on your machine.

    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slt.evaluate import evaluate_dynamic, evaluate_static  # noqa: E402
from slt.inference import DynamicPredictor, SignTranslator, StaticPredictor  # noqa: E402
from slt.synthetic import (make_dynamic_dataset, make_static_dataset, stream_letter,  # noqa: E402
                           stream_word)
from slt.training import train_dynamic, train_static  # noqa: E402


def main() -> None:
    letters = list("ABCDEF")
    words = ["HELLO", "THANK_YOU", "YES", "NO", "HELP"]
    with tempfile.TemporaryDirectory() as tmp:
        d, m = Path(tmp) / "data", Path(tmp) / "models"
        make_static_dataset(d, letters, 50, "own")
        make_static_dataset(d, letters, 120, "external")
        make_dynamic_dataset(d, words, 12, "own")
        make_dynamic_dataset(d, words, 30, "external")
        s = train_static(d, m, model="rf", n_aug=2, verbose=False)
        dy = train_dynamic(d, m, epochs=25, patience=10, verbose=False)
        print(f"static  test accuracy: {s['test_accuracy']:.3f}")
        print(f"dynamic test accuracy: {dy['test_accuracy']:.3f}")
        evaluate_static(m, d, verbose=False)
        evaluate_dynamic(m, d, verbose=False)

        tr = SignTranslator(StaticPredictor(m), DynamicPredictor(m))
        t_end = 0.0
        for idx in (1, 3):                                   # letters B, D
            for t, raw in stream_letter(idx, len(letters), seed=idx):
                tr.update(raw, t_end + t)
            t_end += 3.0
        for idx in (0, 2):                                   # words HELLO, YES
            for t, raw in stream_word(idx, len(words), seed=idx):
                tr.update(raw, t_end + t)
            t_end += 4.0
        tr.flush(t_end)
        print("simulated stream translated to:", tr.sentence.text())
        print("expected                      : B d Hello Yes   (the two letters are >1.5 s apart, so they are separate words)")
    print("SMOKE TEST FINISHED")


if __name__ == "__main__":
    main()
