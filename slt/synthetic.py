"""Synthetic landmark data for smoke-testing the pipeline WITHOUT a camera.

The hands/sequences are procedurally generated and obviously NOT real sign language;
they only exist so you can verify that install -> train -> evaluate -> predict works.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as C
from .data import append_static_rows, save_sequence

_BASES = np.array([[-0.045, -0.035], [-0.03, -0.09], [0.0, -0.10], [0.03, -0.09], [0.055, -0.07]], np.float32)
_LENS = np.array([[0.035, 0.03, 0.025], [0.04, 0.03, 0.025], [0.045, 0.032, 0.026],
                  [0.04, 0.03, 0.024], [0.032, 0.024, 0.02]], np.float32)
_SPREAD = np.deg2rad([-35, -8, 0, 8, 18])


def make_hand(curl: np.ndarray, center=(0.5, 0.7), rng: np.random.Generator | None = None,
              noise: float = 0.002) -> np.ndarray:
    """Procedural 21-point hand. ``curl`` (5,) in [0, 1] is how much each finger is bent."""
    pts = np.zeros((C.N_HAND, 3), np.float32)
    pts[0, :2] = center
    for f in range(5):
        p = np.array(center, np.float32) + _BASES[f]
        ang = _SPREAD[f]
        pts[1 + 4 * f, :2] = p
        for k in range(3):
            ang = ang + curl[f] * 0.55
            p = p + _LENS[f, k] * np.array([np.sin(ang), -np.cos(ang)], np.float32)
            pts[2 + 4 * f + k, :2] = p
            pts[2 + 4 * f + k, 2] = -0.01 * curl[f] * (k + 1)
    if rng is not None:
        pts += rng.normal(0, noise, pts.shape).astype(np.float32)
    return pts


def _class_curls(n_classes: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 1, (n_classes, 5)).astype(np.float32)


def make_static_dataset(data_dir: Path, labels: list[str], per_class: int = 120, source: str = "own",
                        seed: int = 0, n_groups: int = 8) -> None:
    rng = np.random.default_rng(seed + (0 if source == "own" else 1000))
    curls = _class_curls(len(labels), 123)          # same class shapes for both sources
    rows = []
    for ci, lab in enumerate(labels):
        for i in range(per_class):
            c = np.clip(curls[ci] + rng.normal(0, 0.05, 5), 0, 1)
            centre = (rng.uniform(0.35, 0.65), rng.uniform(0.55, 0.8))
            rows.append((lab, f"s{i % n_groups}", make_hand(c, centre, rng)))
    append_static_rows(Path(data_dir) / source / "static" / "landmarks.csv", rows)


def _make_pose(rng: np.random.Generator, shoulder_w: float = 0.35, aspect: float = 1.78) -> np.ndarray:
    pose = np.zeros((C.N_POSE, 4), np.float32)
    cx, cy = 0.5 * aspect, 0.45
    pose[:, 3] = 1.0
    pose[0, :2] = (cx, cy - 0.2)
    pose[11, :2] = (cx + shoulder_w / 2, cy)
    pose[12, :2] = (cx - shoulder_w / 2, cy)
    pose[13, :2] = pose[11, :2] + (0.03, 0.2)
    pose[14, :2] = pose[12, :2] + (-0.03, 0.2)
    pose[15, :2] = pose[13, :2] + (0.0, 0.2)
    pose[16, :2] = pose[14, :2] + (0.0, 0.2)
    pose[23, :2] = (cx + 0.1, cy + 0.6)
    pose[24, :2] = (cx - 0.1, cy + 0.6)
    pose[:, :2] += rng.normal(0, 0.002, (C.N_POSE, 2))
    return pose


def make_sequence(class_idx: int, n_classes: int, rng: np.random.Generator, seq_len: int = C.SEQ_LEN) -> np.ndarray:
    """A class-specific hand-shape + wrist-trajectory sequence in raw (T, 258) format."""
    curls = _class_curls(n_classes, 321)[class_idx]
    freq = 1.0 + class_idx % 4
    phase = rng.uniform(0, 0.4)
    amp = 0.08 + 0.03 * (class_idx % 3)
    out = np.zeros((seq_len, C.RAW_FRAME_DIM), np.float32)
    for t in range(seq_len):
        s = t / (seq_len - 1)
        pose = _make_pose(rng)
        wx = pose[16, 0] + amp * np.sin(2 * np.pi * (freq * s + phase))
        wy = pose[16, 1] - 0.15 + amp * np.cos(2 * np.pi * (freq * s + phase)) * (1 if class_idx % 2 == 0 else 0.3)
        hand = make_hand(np.clip(curls + rng.normal(0, 0.04, 5), 0, 1), (wx, wy), rng)
        out[t, C.POSE_SLICE] = pose.ravel()
        out[t, C.RH_SLICE] = hand.ravel()
    return out


def make_dynamic_dataset(data_dir: Path, labels: list[str], per_class: int = 30, source: str = "own",
                         seed: int = 0) -> None:
    rng = np.random.default_rng(seed + (0 if source == "own" else 2000))
    for ci, lab in enumerate(labels):
        for i in range(per_class):
            save_sequence(source, lab, f"syn_{seed}_{i:04d}", make_sequence(ci, len(labels), rng), data_dir)


# ------------------------------------------------------------- live streams
def stream_letter(label_idx: int, n_labels: int, seconds: float = 1.2, fps: int = 30, lead: float = 0.4,
                  trail: float = 0.8, seed: int = 0) -> list[tuple[float, np.ndarray]]:
    """Simulated camera stream: hand absent, then a still hand shape (a 'letter'), then absent."""
    rng = np.random.default_rng(seed)
    curls = _class_curls(n_labels, 123)[label_idx]
    frames = []
    for i in range(int(lead * fps) + int(seconds * fps) + int(trail * fps)):
        raw = np.zeros(C.RAW_FRAME_DIM, np.float32)
        if int(lead * fps) <= i < int(lead * fps) + int(seconds * fps):
            hand = make_hand(np.clip(curls + rng.normal(0, 0.02, 5), 0, 1), (0.5, 0.7), rng, noise=0.0008)
            raw[C.RH_SLICE] = hand.ravel()
        frames.append((i / fps, raw))
    return frames


def stream_word(class_idx: int, n_classes: int, seconds: float = 1.5, fps: int = 30, lead: float = 0.5,
                trail: float = 0.8, seed: int = 0) -> list[tuple[float, np.ndarray]]:
    """Simulated camera stream: hands down, one moving sign, hands down."""
    rng = np.random.default_rng(seed)
    seq = make_sequence(class_idx, n_classes, rng, seq_len=int(seconds * fps))
    n_lead, n_trail = int(lead * fps), int(trail * fps)
    full = np.concatenate([np.zeros((n_lead, C.RAW_FRAME_DIM), np.float32), seq,
                           np.zeros((n_trail, C.RAW_FRAME_DIM), np.float32)])
    return [(i / fps, full[i]) for i in range(len(full))]
