"""Augmentation on RAW landmarks (before feature extraction).

All functions treat all-zero landmark groups as "not detected" and leave them zero.
"""
from __future__ import annotations

import numpy as np

from . import config as C

# MediaPipe pose landmark index swap when the image is mirrored (left <-> right).
_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
          (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
POSE_MIRROR_PERM = np.arange(C.N_POSE)
for _a, _b in _PAIRS:
    POSE_MIRROR_PERM[_a], POSE_MIRROR_PERM[_b] = _b, _a

_PIVOT = 1.0  # any vertical axis works: all features are relative to body/hand anchors


def _rot3(rng: np.random.Generator, yaw: float, tilt: float) -> np.ndarray:
    """Rotation about z by +-yaw degrees and about x / y by +-tilt degrees."""
    a, b, c = np.deg2rad([rng.uniform(-yaw, yaw), rng.uniform(-tilt, tilt), rng.uniform(-tilt, tilt)])
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    ry = np.array([[np.cos(c), 0, np.sin(c)], [0, 1, 0], [-np.sin(c), 0, np.cos(c)]])
    return (rz @ rx @ ry).astype(np.float32)


# ------------------------------------------------------------- static hands
def mirror_hand(hand: np.ndarray) -> np.ndarray:
    out = hand.copy()
    out[:, 0] = _PIVOT - out[:, 0]
    return out


def augment_hand(hand: np.ndarray, rng: np.random.Generator, yaw: float = 15.0, tilt: float = 12.0,
                 scale: float = 0.12, noise: float = 0.004, p_mirror: float = 0.5) -> np.ndarray:
    """Random 3-D rotation, scaling, jitter and (optionally) mirroring of one (21, 3) hand."""
    pts = hand.astype(np.float32).copy()
    c = pts.mean(0, keepdims=True)
    pts = (pts - c) @ _rot3(rng, yaw, tilt).T
    pts *= 1.0 + rng.uniform(-scale, scale)
    pts += rng.normal(0, noise, pts.shape).astype(np.float32)
    pts += c
    if rng.random() < p_mirror:
        pts = mirror_hand(pts)
    return pts


# --------------------------------------------------------------- sequences
def mirror_sequence(raw: np.ndarray) -> np.ndarray:
    """Mirror a (T, 258) raw sequence: flip x, swap left/right hand slots and pose pairs."""
    raw = np.asarray(raw, dtype=np.float32)
    T = len(raw)
    pose = raw[:, C.POSE_SLICE].reshape(T, C.N_POSE, 4).copy()
    lh = raw[:, C.LH_SLICE].reshape(T, C.N_HAND, 3).copy()
    rh = raw[:, C.RH_SLICE].reshape(T, C.N_HAND, 3).copy()
    pose_ok = np.abs(pose[:, :, :3]).sum(2) > 0
    pose[:, :, 0] = np.where(pose_ok, _PIVOT - pose[:, :, 0], 0)
    pose = pose[:, POSE_MIRROR_PERM]
    for h in (lh, rh):
        ok = np.abs(h).sum(2) > 0
        h[:, :, 0] = np.where(ok, _PIVOT - h[:, :, 0], 0)
    out = np.concatenate([pose.reshape(T, -1), rh.reshape(T, -1), lh.reshape(T, -1)], axis=1)
    return out.astype(np.float32)


def augment_sequence(raw: np.ndarray, rng: np.random.Generator, p_mirror: float = 0.3,
                     min_crop: float = 0.75, rot_deg: float = 8.0, noise: float = 0.003,
                     p_drop: float = 0.05) -> np.ndarray:
    """Time crop + re-sample (speed change), small 2-D rotation, jitter, mirroring, hand drop-outs."""
    raw = np.asarray(raw, dtype=np.float32)
    T = len(raw)
    # temporal crop then resample back to T frames (simulates timing differences)
    frac = rng.uniform(min_crop, 1.0)
    n = max(2, int(round(T * frac)))
    start = rng.integers(0, T - n + 1)
    idx = np.linspace(start, start + n - 1, T).round().astype(int)
    seq = raw[idx].copy()

    pose = seq[:, C.POSE_SLICE].reshape(T, C.N_POSE, 4)
    lh = seq[:, C.LH_SLICE].reshape(T, C.N_HAND, 3)
    rh = seq[:, C.RH_SLICE].reshape(T, C.N_HAND, 3)
    groups = [(pose, 2), (lh, 2), (rh, 2)]  # views into seq, xy are columns 0:2

    # one global 2-D rotation about the body centre
    theta = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    cs, sn = np.cos(theta), np.sin(theta)
    anchor = pose[:, [11, 12], :2].mean(1).mean(0) if np.any(pose[:, 11:13, :3]) else np.array([0.5, 0.5], np.float32)
    for g, _ in groups:
        ok = np.abs(g[:, :, :3]).sum(2) > 0
        xy = g[:, :, :2] - anchor
        rot = np.stack([cs * xy[..., 0] - sn * xy[..., 1], sn * xy[..., 0] + cs * xy[..., 1]], axis=-1) + anchor
        rot += rng.normal(0, noise, rot.shape)
        g[:, :, :2] = np.where(ok[..., None], rot, 0)

    # simulate frames where a hand was not detected
    for g in (lh, rh):
        drop = rng.random(T) < p_drop
        g[drop] = 0

    out = seq
    if rng.random() < p_mirror:
        out = mirror_sequence(out)
    return out.astype(np.float32)
