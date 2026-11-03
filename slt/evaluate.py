"""Evaluation: accuracy, per-class report, per-source accuracy and confusion matrix image."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from . import config as C
from .data import load_dynamic, load_static
from .features import hand_features
from .inference import DynamicPredictor, StaticPredictor


def _save_confusion(cm: np.ndarray, classes: list[str], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(classes)
    size = max(5, min(0.45 * n + 2, 22))
    fig, ax = plt.subplots(figsize=(size, size))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), classes, rotation=90, fontsize=8)
    ax.set_yticks(range(n), classes, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    if n <= 30:
        for i in range(n):
            for j in range(n):
                if cm[i, j]:
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                            color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _report(y_true, y_pred, classes, source, models_dir: Path, tag: str, verbose: bool) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    labels = sorted(set(classes) | set(y_true) | set(y_pred))
    rep = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    png = models_dir / f"{tag}_confusion.png"
    _save_confusion(cm, labels, png, f"{tag}  accuracy={acc:.3f}")
    per_source = {}
    if source is not None:
        for s in np.unique(source):
            m = np.asarray(source) == s
            per_source[str(s)] = {"n": int(m.sum()), "accuracy": float(accuracy_score(np.asarray(y_true)[m], np.asarray(y_pred)[m]))}
    if verbose:
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
        print(f"overall accuracy: {acc:.4f}  ({len(y_true)} samples)")
        for s, v in per_source.items():
            print(f"  source={s:9s} n={v['n']:5d} accuracy={v['accuracy']:.4f}")
        print(f"confusion matrix saved to {png}")
    out = {"accuracy": acc, "n": int(len(y_true)), "per_source": per_source,
           "macro_f1": float(rep["macro avg"]["f1-score"])}
    (models_dir / f"{tag}_eval.json").write_text(json.dumps(out, indent=2))
    return out


def evaluate_static(models_dir=None, data_dir=None, eval_on: str = "test", verbose: bool = True) -> dict:
    """eval_on: 'test' (held-out split saved at train time) | 'own' | 'external' (all data of a source)."""
    models_dir = Path(models_dir or C.MODELS_DIR)
    pred = StaticPredictor(models_dir)
    if eval_on == "test":
        t = np.load(models_dir / "static_test.npz", allow_pickle=False)
        raw, labels, source = t["raw"], t["labels"], t["source"]
    else:
        d = load_static((eval_on,), data_dir)
        raw, labels, source = d["raw"], d["labels"], d["source"]
    if len(raw) == 0:
        raise SystemExit(f"No samples to evaluate for eval_on='{eval_on}'.")
    y_pred = pred.model.predict(hand_features(raw))
    return _report(labels, y_pred, pred.classes, source, models_dir, f"static_{eval_on}", verbose)


def evaluate_dynamic(models_dir=None, data_dir=None, eval_on: str = "test", verbose: bool = True) -> dict:
    models_dir = Path(models_dir or C.MODELS_DIR)
    pred = DynamicPredictor(models_dir)
    if eval_on == "test":
        t = np.load(models_dir / "dynamic_test.npz", allow_pickle=False)
        raw, labels, source = t["raw"], t["labels"], t["source"]
    else:
        d = load_dynamic((eval_on,), data_dir, pred.seq_len)
        raw, labels, source = d["raw"], d["labels"], d["source"]
    if len(raw) == 0:
        raise SystemExit(f"No samples to evaluate for eval_on='{eval_on}'.")
    y_pred = np.array([pred.classes[int(np.argmax(pred.predict_proba(r)))] for r in raw])
    return _report(labels, y_pred, pred.classes, source, models_dir, f"dynamic_{eval_on}", verbose)
