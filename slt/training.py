"""Training pipelines.

* ``train_static``  - fingerspelling / hand-shape classifier (scikit-learn).
* ``train_dynamic`` - word / phrase classifier on landmark sequences (PyTorch BiGRU).

Both merge ``data/own`` and ``data/external``, split by *group* (no leakage), augment only
the training split, and save the held-out test split so ``evaluate`` is reproducible.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .augment import augment_hand, augment_sequence
from .data import (boost_source, cap_per_class, class_counts, grouped_split, load_dynamic,
                   load_static)
from .features import STATIC_FEATURE_DIM, hand_features, sequence_features


def _log(verbose: bool, *a):
    if verbose:
        print("[slt]", *a, flush=True)


# =================================================================== static
def build_static_model(name: str, seed: int = 42):
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, max_depth=24, n_jobs=-1, random_state=seed)
    if name == "extra":
        return ExtraTreesClassifier(n_estimators=400, max_depth=28, n_jobs=-1, random_state=seed)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(256, 128), alpha=1e-3, early_stopping=True, n_iter_no_change=15,
            max_iter=400, random_state=seed))
    raise ValueError(f"unknown static model '{name}' (use rf, extra, mlp or auto)")


def _augmented_static_features(raw, labels, idx, n_aug, rng):
    xs, ys = [hand_features(raw[idx])], [labels[idx]]
    for _ in range(n_aug):
        aug = np.stack([augment_hand(raw[i], rng) for i in idx])
        xs.append(hand_features(aug))
        ys.append(labels[idx])
    return np.concatenate(xs), np.concatenate(ys)


def train_static(data_dir=None, models_dir=None, model: str = "auto", sources=C.SOURCES, n_aug: int = 4,
                 own_boost: int = 3, max_per_class: int | None = None, val_frac: float = 0.15,
                 test_frac: float = 0.15, seed: int = 42, verbose: bool = True) -> dict:
    models_dir = Path(models_dir or C.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    d = load_static(sources, data_dir)
    raw, labels, source, group = d["raw"], d["labels"], d["source"], d["group"]
    if len(raw) == 0:
        raise SystemExit("No static data found. Record some with scripts/collect_static.py or import a dataset first.")
    if len(np.unique(labels)) < 2:
        raise SystemExit("Need at least two classes to train.")
    _log(verbose, f"loaded {len(raw)} samples, {len(np.unique(labels))} classes "
                  f"(own={int((source == 'own').sum())}, external={int((source == 'external').sum())})")

    tr, va, te = grouped_split(labels, group, val_frac, test_frac, seed)
    tr = cap_per_class(tr, labels, source, max_per_class, seed=seed)
    tr_b = boost_source(tr, source, own_boost)
    _log(verbose, f"split: train={len(tr)} (x{own_boost} own boost -> {len(tr_b)}), val={len(va)}, test={len(te)}")

    t0 = time.time()
    Xtr, ytr = _augmented_static_features(raw, labels, tr_b, n_aug, rng)
    Xva = hand_features(raw[va]) if len(va) else None
    yva = labels[va] if len(va) else None
    _log(verbose, f"feature matrix {Xtr.shape} built in {time.time() - t0:.1f}s")

    names = ["mlp", "rf"] if model == "auto" else [model]       # fastest first: wins ties (live inference speed)
    if model == "auto" and Xva is None:
        _log(verbose, "no validation split available -> using the MLP")
        names = ["mlp"]
    best, best_name, best_acc, scores = None, None, -1.0, {}
    for name in names:
        clf = build_static_model(name, seed)
        t1 = time.time()
        clf.fit(Xtr, ytr)
        acc = accuracy_score(yva, clf.predict(Xva)) if Xva is not None else float("nan")
        scores[name] = None if np.isnan(acc) else round(float(acc), 4)
        _log(verbose, f"{name:6s} val acc = {acc:.4f}  ({time.time() - t1:.1f}s)")
        if best is None or (not np.isnan(acc) and acc > best_acc + 0.005):   # RF must be clearly better
            best, best_name, best_acc = clf, name, (acc if not np.isnan(acc) else best_acc)

    test_acc = None
    if len(te):
        test_acc = float(accuracy_score(labels[te], best.predict(hand_features(raw[te]))))
        _log(verbose, f"TEST accuracy ({best_name}) = {test_acc:.4f} on {len(te)} held-out samples")

    joblib.dump({"model": best, "name": best_name, "classes": list(best.classes_),
                 "feature_dim": STATIC_FEATURE_DIM, "feature_version": C.STATIC_FEATURE_VERSION},
                models_dir / "static_model.joblib")
    np.savez_compressed(models_dir / "static_test.npz", raw=raw[te], labels=labels[te], source=source[te])
    meta = {"model": best_name, "val_scores": scores, "test_accuracy": test_acc, "n_train": int(len(tr)),
            "n_val": int(len(va)), "n_test": int(len(te)), "classes": class_counts(labels, source)}
    (models_dir / "static_meta.json").write_text(json.dumps(meta, indent=2))
    _log(verbose, f"saved {models_dir / 'static_model.joblib'}")
    return meta


# ================================================================== dynamic
class _SeqDataset:
    """Torch-style dataset; augmentation happens on the fly on raw sequences."""

    def __init__(self, raw, y, idx, mean, std, use_velocity, augment, seed):
        self.raw, self.y, self.idx = raw, y, np.asarray(idx)
        self.mean, self.std, self.vel, self.aug = mean, std, use_velocity, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        seq = augment_sequence(self.raw[i], self.rng) if self.aug else self.raw[i]
        x = (sequence_features(seq, self.vel) - self.mean) / self.std
        return x.astype(np.float32), int(self.y[i])


def train_dynamic(data_dir=None, models_dir=None, sources=C.SOURCES, epochs: int = 80, batch_size: int = 32,
                  lr: float = 2e-3, hidden: int = 128, layers: int = 2, dropout: float = 0.3,
                  own_boost: int = 3, max_per_class: int | None = None, use_velocity: bool = True,
                  patience: int = 15, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42,
                  device: str = "auto", seq_len: int = C.SEQ_LEN, verbose: bool = True) -> dict:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from .seq_model import SignGRU, pick_device

    models_dir = Path(models_dir or C.MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = load_dynamic(sources, data_dir, seq_len)
    raw, labels, source, group = d["raw"], d["labels"], d["source"], d["group"]
    if len(raw) == 0:
        raise SystemExit("No dynamic data found. Record with scripts/collect_dynamic.py or import videos first.")
    classes = sorted(np.unique(labels).tolist())
    if len(classes) < 2:
        raise SystemExit("Need at least two classes to train.")
    y = np.array([classes.index(l) for l in labels], dtype=np.int64)
    _log(verbose, f"loaded {len(raw)} sequences, {len(classes)} classes "
                  f"(own={int((source == 'own').sum())}, external={int((source == 'external').sum())})")

    tr, va, te = grouped_split(labels, group, val_frac, test_frac, seed)
    tr = cap_per_class(tr, labels, source, max_per_class, seed=seed)
    tr_b = boost_source(tr, source, own_boost)
    if len(va) == 0:
        _log(verbose, "WARNING: no validation split (too little data) - using the training set for model selection")
        va = tr
    _log(verbose, f"split: train={len(tr)} (-> {len(tr_b)} with own boost), val={len(va)}, test={len(te)}")

    feats_tr = np.concatenate([sequence_features(raw[i], use_velocity) for i in tr])
    mean = feats_tr.mean(0).astype(np.float32)
    std = np.maximum(feats_tr.std(0), 1e-3).astype(np.float32)
    input_dim = feats_tr.shape[1]

    dev = pick_device(device)
    net = SignGRU(input_dim, len(classes), hidden, layers, dropout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    loader = DataLoader(_SeqDataset(raw, y, tr_b, mean, std, use_velocity, True, seed),
                        batch_size=batch_size, shuffle=True, drop_last=len(tr_b) > batch_size, num_workers=0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader))
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    def batch_eval(idx):
        ds = _SeqDataset(raw, y, idx, mean, std, use_velocity, False, seed)
        xs = torch.from_numpy(np.stack([ds[k][0] for k in range(len(ds))])).to(dev)
        ys = torch.from_numpy(y[idx]).long().to(dev)
        net.eval()
        with torch.no_grad():
            logits = net(xs)
            return float((logits.argmax(1) == ys).float().mean()), float(crit(logits, ys))

    best = {"acc": -1.0, "loss": 1e9, "state": None, "epoch": 0}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train()
        run = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(loss.detach()) * len(xb)
        v_acc, v_loss = batch_eval(va)
        if v_acc > best["acc"] or (v_acc == best["acc"] and v_loss < best["loss"]):
            best = {"acc": v_acc, "loss": v_loss, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}}
        if verbose and (ep == 1 or ep % 5 == 0):
            _log(verbose, f"epoch {ep:3d}  train_loss={run / len(loader.dataset):.4f}  val_acc={v_acc:.4f}  val_loss={v_loss:.4f}")
        if ep - best["epoch"] >= patience:
            _log(verbose, f"early stop at epoch {ep} (best epoch {best['epoch']})")
            break
    _log(verbose, f"training done in {time.time() - t0:.0f}s, best val acc {best['acc']:.4f} @ epoch {best['epoch']}")
    net.load_state_dict(best["state"])

    test_acc = None
    if len(te):
        test_acc, _ = batch_eval(te)
        _log(verbose, f"TEST accuracy = {test_acc:.4f} on {len(te)} held-out sequences")

    torch.save({"state_dict": best["state"], "labels": classes, "mean": torch.from_numpy(mean),
                "std": torch.from_numpy(std), "seq_len": seq_len, "use_velocity": use_velocity,
                "input_dim": input_dim, "hidden": hidden, "layers": layers, "dropout": dropout,
                "feature_version": C.DYNAMIC_FEATURE_VERSION}, models_dir / "dynamic_model.pt")
    np.savez_compressed(models_dir / "dynamic_test.npz", raw=raw[te], labels=labels[te], source=source[te])
    meta = {"best_val_acc": best["acc"], "best_epoch": best["epoch"], "test_accuracy": test_acc,
            "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
            "input_dim": int(input_dim), "classes": class_counts(labels, source)}
    (models_dir / "dynamic_meta.json").write_text(json.dumps(meta, indent=2))
    _log(verbose, f"saved {models_dir / 'dynamic_model.pt'}")
    return meta
