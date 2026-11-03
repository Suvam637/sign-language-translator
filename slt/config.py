"""Central configuration: paths, landmark layout and shared constants."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SLT_DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.environ.get("SLT_MODELS_DIR", ROOT / "models"))

# Two data sources are kept in separate folders so that you can always tell
# your own recordings apart from third-party datasets.
SOURCES = ("own", "external")

# ---------------------------------------------------------------- landmarks
N_POSE = 33
N_HAND = 21
POSE_DIM = N_POSE * 4          # x, y, z, visibility
HAND_DIM = N_HAND * 3          # x, y, z

# Layout of one raw frame vector (258 floats):
#   [0:132]    pose  (33 x [x, y, z, visibility])
#   [132:195]  hand slot "L" (MediaPipe label "Left")
#   [195:258]  hand slot "R" (MediaPipe label "Right")
# Missing parts are filled with zeros.  x and z are multiplied by the image
# aspect ratio (w/h) so the coordinate system is isotropic (one unit in x and
# one unit in y are the same physical distance).
RAW_FRAME_DIM = POSE_DIM + 2 * HAND_DIM
POSE_SLICE = slice(0, POSE_DIM)
LH_SLICE = slice(POSE_DIM, POSE_DIM + HAND_DIM)
RH_SLICE = slice(POSE_DIM + HAND_DIM, RAW_FRAME_DIM)

# Upper-body pose points used as features: nose, shoulders, elbows, wrists.
POSE_IDX = (0, 11, 12, 13, 14, 15, 16)

# ----------------------------------------------------------------- sequences
SEQ_LEN = 30                   # frames per dynamic-sign sample (after resampling)
WINDOW_SEC = 2.0               # seconds of video covered by one dynamic sample

# Labels that mean "nothing is being signed"; never added to the sentence.
IDLE_LABELS = ("IDLE", "NONE", "NOTHING", "NEUTRAL", "BACKGROUND")

# Feature versions (bump when feature engineering changes; stored in models)
STATIC_FEATURE_VERSION = 1
DYNAMIC_FEATURE_VERSION = 1

# MediaPipe Tasks model files. Only needed for the Tasks backend, i.e. when the
# legacy ``mp.solutions`` API is unavailable (mediapipe >= 0.10.30).
TASK_MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
