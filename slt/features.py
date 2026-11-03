"""Turn raw landmarks into translation/scale-invariant model features."""
from __future__ import annotations

import numpy as np

from . import config as C

EPS = 1e-6

# --------------------------------------------------------------- static hand
_TIPS = [4, 8, 12, 16, 20]
_TIP_PAIRS = [(i, j) for a, i in enumerate(_TIPS) for j in _TIPS[a + 1:]]          # 10 pairs
_FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
_ANGLE_TRIPLES = []                                                                  # 15 joint angles
for _f in _FINGERS:
    chain = [0] + _f
    for k in range(1, 4):
        _ANGLE_TRIPLES.append((chain[k - 1], chain[k], chain[k + 1]))

STATIC_FEATURE_DIM = C.HAND_DIM + len(_TIP_PAIRS) + len(_ANGLE_TRIPLES)             # 63 + 10 + 15 = 88


def hand_features(hands: np.ndarray) -> np.ndarray:
    """Features for static hand shapes.

    hands: (N, 21, 3) or (21, 3) raw landmarks -> (N, 88) float32:
      * 63 coordinates relative to the wrist, scaled by the wrist->middle-MCP length
      * 10 fingertip-to-fingertip distances (scaled)
      * 15 finger joint angles (radians)
    """
    single = hands.ndim == 2
    h = np.asarray(hands, dtype=np.float32)
    if single:
        h = h[None]
    local = h - h[:, :1, :]
    scale = np.linalg.norm(local[:, 9, :], axis=1).clip(min=1e-4)
    local = local / scale[:, None, None]
    dists = np.stack([np.linalg.norm(local[:, i] - local[:, j], axis=1) for i, j in _TIP_PAIRS], axis=1)
    angles = []
    for a, b, c in _ANGLE_TRIPLES:
        v1, v2 = local[:, a] - local[:, b], local[:, c] - local[:, b]
        cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + EPS)
        angles.append(np.arccos(np.clip(cos, -1.0, 1.0)))
    out = np.concatenate([local.reshape(len(h), -1), dists, np.stack(angles, axis=1)], axis=1).astype(np.float32)
    return out[0] if single else out


# --------------------------------------------------------- dynamic sequences
FRAME_FEATURE_DIM = len(C.POSE_IDX) * 2 + 2 * (C.HAND_DIM + 2) + 2                  # 14 + 130 + 2 = 146
SEQ_FEATURE_DIM_VEL = FRAME_FEATURE_DIM * 2


def frame_features(raw: np.ndarray) -> np.ndarray:
    """(T, 258) raw frames -> (T, 146) features.

    Per frame: upper-body pose (nose, shoulders, elbows, wrists) relative to the shoulder
    centre / shoulder width; for each hand the wrist-relative, size-normalised shape
    (63) plus the wrist position relative to the shoulders (2); two presence flags.
    """
    raw = np.asarray(raw, dtype=np.float32).reshape(-1, C.RAW_FRAME_DIM)
    T = len(raw)
    pose = raw[:, C.POSE_SLICE].reshape(T, C.N_POSE, 4)
    lh = raw[:, C.LH_SLICE].reshape(T, C.N_HAND, 3)
    rh = raw[:, C.RH_SLICE].reshape(T, C.N_HAND, 3)
    pose_ok = np.abs(pose[:, :, :3]).sum((1, 2)) > 0
    lh_ok = np.abs(lh).sum((1, 2)) > 0
    rh_ok = np.abs(rh).sum((1, 2)) > 0

    center = np.tile(np.array([0.5, 0.5], dtype=np.float32), (T, 1))
    scale = np.full(T, 0.3, dtype=np.float32)
    sl, sr = pose[:, 11, :2], pose[:, 12, :2]
    sw = np.linalg.norm(sl - sr, axis=1)
    ok = pose_ok & (sw > 1e-3)
    center[ok] = (sl[ok] + sr[ok]) / 2
    scale[ok] = sw[ok]

    pose_f = (pose[:, C.POSE_IDX, :2] - center[:, None, :]) / scale[:, None, None]
    pose_f[~pose_ok] = 0
    pose_f = pose_f.reshape(T, -1)

    def hand_block(h, present):
        wrist = h[:, 0, :]
        local = h - wrist[:, None, :]
        hs = np.linalg.norm(local[:, 9, :], axis=1).clip(min=1e-4)
        local = (local / hs[:, None, None]).reshape(T, -1)
        pos = (wrist[:, :2] - center) / scale[:, None]
        out = np.concatenate([local, pos], axis=1)
        out[~present] = 0
        return out

    feats = np.concatenate([pose_f, hand_block(lh, lh_ok), hand_block(rh, rh_ok),
                            lh_ok[:, None].astype(np.float32), rh_ok[:, None].astype(np.float32)], axis=1)
    return feats.astype(np.float32)


def sequence_features(raw: np.ndarray, use_velocity: bool = True) -> np.ndarray:
    """(T, 258) raw -> (T, 146) or (T, 292) with frame-to-frame velocity appended."""
    f = frame_features(raw)
    if not use_velocity:
        return f
    vel = np.diff(f, axis=0, prepend=f[:1])
    return np.concatenate([f, vel], axis=1).astype(np.float32)


def resample_sequence(seq: np.ndarray, length: int = C.SEQ_LEN) -> np.ndarray:
    """Nearest-neighbour resample of (T, D) to (length, D) (keeps 'missing' frames exactly zero)."""
    seq = np.asarray(seq)
    if len(seq) == 0:
        raise ValueError("cannot resample an empty sequence")
    idx = np.linspace(0, len(seq) - 1, length).round().astype(int)
    return seq[idx]


def hands_mask(raw_seq: np.ndarray) -> np.ndarray:
    """(T, 258) -> (T,) bool: at least one hand detected in that frame."""
    raw_seq = np.asarray(raw_seq)
    return (np.abs(raw_seq[:, C.LH_SLICE]).sum(1) + np.abs(raw_seq[:, C.RH_SLICE]).sum(1)) > 0


def active_span(raw_seq: np.ndarray, margin: int = 2) -> np.ndarray | None:
    """Trim a recording to the stretch between the first and last frame that shows a hand.

    Returns None when no hand is visible at all.
    """
    present = hands_mask(raw_seq)
    if not present.any():
        return None
    idx = np.flatnonzero(present)
    return raw_seq[max(0, idx[0] - margin): min(len(raw_seq), idx[-1] + margin + 1)]


def prepare_sequence(raw_seq: np.ndarray, length: int = C.SEQ_LEN, margin: int = 2,
                     keep_if_empty: bool = False) -> np.ndarray | None:
    """THE shared preprocessing for dynamic signs (recording, dataset import and live inference):
    trim to the active span, then resample to a fixed number of frames. This makes the model
    independent of how long the person waited before / after signing and of signing speed."""
    span = active_span(raw_seq, margin)
    if span is None:
        if not keep_if_empty:
            return None
        span = np.asarray(raw_seq)
    return resample_sequence(span, length)
