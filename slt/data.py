"""Dataset layout, loading, saving and leak-free splitting.

Layout (everything is plain CSV / NPY so it is easy to inspect and version)::

    data/
      own/                            <- recorded by you with the collect_* scripts
        static/landmarks.csv          label, group, x0,y0,z0 ... x20,y20,z20
        dynamic/<LABEL>/<id>.npy      (SEQ_LEN, 258) raw sequences
      external/                       <- imported from third-party datasets
        static/landmarks.csv
        dynamic/<LABEL>/<id>.npy
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from . import config as C

_STATIC_HEADER = ["label", "group"] + [f"{a}{i}" for i in range(C.N_HAND) for a in "xyz"]


def normalize_label(label: str) -> str:
    """'  thank you ' -> 'THANK_YOU'.  Keeps own and external labels consistent."""
    label = re.sub(r"\s+", "_", str(label).strip())
    return label.upper()


# ------------------------------------------------------------------- static
def static_csv_path(source: str, data_dir: Path | None = None) -> Path:
    return Path(data_dir or C.DATA_DIR) / source / "static" / "landmarks.csv"


def append_static_rows(path: Path, rows: list[tuple[str, str, np.ndarray]]) -> None:
    """rows: (label, group, hand(21,3)). Creates the file with a header when needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(_STATIC_HEADER)
        for label, group, hand in rows:
            w.writerow([normalize_label(label), group] + [f"{v:.6f}" for v in np.asarray(hand, np.float32).ravel()])


def load_static(sources=C.SOURCES, data_dir: Path | None = None):
    """Return dict(raw (N,21,3), labels (N,), source (N,), group (N,)); empty arrays if no data."""
    raws, labels, srcs, groups = [], [], [], []
    for src in sources:
        p = static_csv_path(src, data_dir)
        if not p.exists():
            continue
        with open(p, newline="") as fh:
            rd = csv.reader(fh)
            next(rd, None)
            for row in rd:
                if len(row) != 2 + C.HAND_DIM:
                    continue
                labels.append(row[0])
                groups.append(f"{src}:{row[1]}")
                srcs.append(src)
                raws.append(np.asarray(row[2:], dtype=np.float32).reshape(C.N_HAND, 3))
    raw = np.stack(raws) if raws else np.zeros((0, C.N_HAND, 3), np.float32)
    return dict(raw=raw, labels=np.array(labels, dtype=str), source=np.array(srcs, dtype=str),
                group=np.array(groups, dtype=str))


# ------------------------------------------------------------------ dynamic
def dynamic_dir(source: str, data_dir: Path | None = None) -> Path:
    return Path(data_dir or C.DATA_DIR) / source / "dynamic"


def save_sequence(source: str, label: str, name: str, seq: np.ndarray, data_dir: Path | None = None) -> Path:
    d = dynamic_dir(source, data_dir) / normalize_label(label)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.npy"
    np.save(path, np.asarray(seq, dtype=np.float32))
    return path


def load_dynamic(sources=C.SOURCES, data_dir: Path | None = None, seq_len: int = C.SEQ_LEN):
    """Return dict(raw (N,T,258), labels, source, group)."""
    seqs, labels, srcs, groups = [], [], [], []
    for src in sources:
        root = dynamic_dir(src, data_dir)
        if not root.exists():
            continue
        for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for f in sorted(cls_dir.glob("*.npy")):
                arr = np.load(f)
                if arr.ndim != 2 or arr.shape[1] != C.RAW_FRAME_DIM:
                    continue
                if len(arr) != seq_len:
                    idx = np.linspace(0, len(arr) - 1, seq_len).round().astype(int)
                    arr = arr[idx]
                seqs.append(arr.astype(np.float32))
                labels.append(cls_dir.name)
                srcs.append(src)
                groups.append(f"{src}:{f.stem}")
    raw = np.stack(seqs) if seqs else np.zeros((0, seq_len, C.RAW_FRAME_DIM), np.float32)
    return dict(raw=raw, labels=np.array(labels, dtype=str), source=np.array(srcs, dtype=str),
                group=np.array(groups, dtype=str))


def class_counts(labels, source=None) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    labels = np.asarray(labels)
    for lab in np.unique(labels):
        m = labels == lab
        entry = {"total": int(m.sum())}
        if source is not None:
            for s in np.unique(source):
                entry[str(s)] = int((m & (np.asarray(source) == s)).sum())
        out[str(lab)] = entry
    return out


# ---------------------------------------------------------------- splitting
def grouped_split(labels, groups, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
    """Stratified train/val/test split that keeps whole *groups* together.

    Consecutive webcam frames (or frames of one video) are near duplicates; putting one
    in train and the next in test inflates accuracy. Samples that share a ``group`` id
    therefore always land in the same split. Classes with fewer than 5 groups fall back to
    a plain random split of their samples.
    """
    rng = np.random.default_rng(seed)
    labels, groups = np.asarray(labels), np.asarray(groups)
    tr, va, te = [], [], []
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        gs = np.unique(groups[idx])
        if len(gs) >= 5:
            gs = rng.permutation(gs)
            n_te = max(1, int(round(len(gs) * test_frac)))
            n_va = max(1, int(round(len(gs) * val_frac)))
            te_g, va_g = set(gs[:n_te]), set(gs[n_te:n_te + n_va])
            for i in idx:
                (te if groups[i] in te_g else va if groups[i] in va_g else tr).append(i)
        else:
            idx = rng.permutation(idx)
            n = len(idx)
            if n < 3:
                tr.extend(idx)
                continue
            n_te = max(1, int(round(n * test_frac)))
            n_va = max(1, int(round(n * val_frac)))
            n_te = min(n_te, n - 1)
            n_va = max(0, min(n_va, n - 1 - n_te))      # always keep >= 1 training sample
            te.extend(idx[:n_te])
            va.extend(idx[n_te:n_te + n_va])
            tr.extend(idx[n_te + n_va:])
    return np.array(sorted(tr), dtype=int), np.array(sorted(va), dtype=int), np.array(sorted(te), dtype=int)


def cap_per_class(idx, labels, source, max_per_class: int | None, cap_source: str = "external", seed: int = 0):
    """Limit how many *cap_source* samples each class contributes (keeps own data untouched)."""
    if not max_per_class:
        return np.asarray(idx)
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx)
    keep = []
    for lab in np.unique(np.asarray(labels)[idx]):
        sel = idx[np.asarray(labels)[idx] == lab]
        capped = sel[np.asarray(source)[sel] == cap_source]
        other = sel[np.asarray(source)[sel] != cap_source]
        if len(capped) > max_per_class:
            capped = rng.choice(capped, max_per_class, replace=False)
        keep.extend(other.tolist())
        keep.extend(np.asarray(capped).tolist())
    return np.array(sorted(keep), dtype=int)


def boost_source(idx, source, boost: int = 1, boost_source_name: str = "own"):
    """Repeat indices of *boost_source_name* ``boost`` times (own data usually is far smaller)."""
    idx = np.asarray(idx)
    if boost <= 1:
        return idx
    mine = idx[np.asarray(source)[idx] == boost_source_name]
    return np.concatenate([idx] + [mine] * (boost - 1))
