"""MediaPipe landmark extraction (pose + two hands) with two interchangeable backends.

* ``legacy``  - ``mp.solutions`` API (mediapipe <= 0.10.21, Python 3.9-3.12).
                Models are bundled in the wheel, nothing to download.
* ``tasks``   - ``mediapipe.tasks`` API (all versions; the only option for
                mediapipe >= 0.10.30 / Python 3.13+). The two ``.task`` model
                files are downloaded once into ``models/mediapipe/``.

Both backends return the same *raw frame vector* (see ``config.RAW_FRAME_DIM``)
so everything downstream is backend independent.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np

from . import config as C

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
    (18, 19), (19, 20), (0, 17),
]
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]


# ------------------------------------------------------------------ helpers
def split_raw_frame(raw: np.ndarray):
    """Return (pose[33,4], left_hand[21,3], right_hand[21,3]) views of a raw frame."""
    raw = np.asarray(raw)
    return (
        raw[C.POSE_SLICE].reshape(C.N_POSE, 4),
        raw[C.LH_SLICE].reshape(C.N_HAND, 3),
        raw[C.RH_SLICE].reshape(C.N_HAND, 3),
    )


def hands_present(raw: np.ndarray) -> bool:
    return bool(np.any(raw[C.LH_SLICE]) or np.any(raw[C.RH_SLICE]))


def pick_hand(raw: np.ndarray) -> np.ndarray | None:
    """Return the (21, 3) landmarks of the most prominent hand in a raw frame, or None."""
    _, lh, rh = split_raw_frame(raw)
    cands = [h for h in (lh, rh) if np.any(h)]
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0].copy()
    spread = [float(np.ptp(h[:, 0]) * np.ptp(h[:, 1])) for h in cands]
    return cands[int(np.argmax(spread))].copy()


def assemble_raw(pose, hands, aspect: float) -> np.ndarray:
    """Build the 258-d raw vector.

    pose  : (33, 4) normalised [x, y, z, visibility] or None
    hands : list of (landmarks (21, 3) normalised, label "Left"/"Right")
    aspect: image width / height
    """
    raw = np.zeros(C.RAW_FRAME_DIM, dtype=np.float32)
    if pose is not None:
        p = np.asarray(pose, dtype=np.float32).copy()
        p[:, 0] *= aspect
        p[:, 2] *= aspect
        raw[C.POSE_SLICE] = p.ravel()
    hs = []
    for lm, label in hands[:2]:
        h = np.asarray(lm, dtype=np.float32).copy()
        h[:, 0] *= aspect
        h[:, 2] *= aspect
        hs.append((h, label))
    if len(hs) == 1:
        slot = hs[0][1]
        (raw[C.LH_SLICE] if slot == "Left" else raw[C.RH_SLICE])[:] = hs[0][0].ravel()
    elif len(hs) == 2:
        if hs[0][1] == hs[1][1]:  # same label twice -> assign by image position
            hs.sort(key=lambda t: t[0][0, 0])
            raw[C.LH_SLICE] = hs[0][0].ravel()
            raw[C.RH_SLICE] = hs[1][0].ravel()
        else:
            for h, label in hs:
                (raw[C.LH_SLICE] if label == "Left" else raw[C.RH_SLICE])[:] = h.ravel()
    return raw


def draw_raw(frame, raw: np.ndarray, hands: bool = True, pose: bool = True) -> None:
    """Draw the landmarks of a raw frame on a BGR image (in place, backend independent)."""
    import cv2

    h, w = frame.shape[:2]
    pz, lh, rh = split_raw_frame(raw)

    def px(pt):  # raw x is (x_norm * w/h) -> x_norm * w = x_raw * h
        return int(pt[0] * h), int(pt[1] * h)

    if pose and np.any(pz[:, :3]):
        for a, b in POSE_CONNECTIONS:
            if pz[a, 3] > 0.3 and pz[b, 3] > 0.3:
                cv2.line(frame, px(pz[a]), px(pz[b]), (200, 200, 60), 2, cv2.LINE_AA)
        for i in C.POSE_IDX:
            if pz[i, 3] > 0.3:
                cv2.circle(frame, px(pz[i]), 4, (0, 220, 255), -1, cv2.LINE_AA)
    if hands:
        for hand, col in ((lh, (80, 255, 80)), (rh, (255, 120, 60))):
            if not np.any(hand):
                continue
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, px(hand[a]), px(hand[b]), col, 2, cv2.LINE_AA)
            for i in range(C.N_HAND):
                cv2.circle(frame, px(hand[i]), 3, (255, 255, 255), -1, cv2.LINE_AA)


def ensure_task_models(models_dir: Path | None = None, need_pose: bool = True) -> dict[str, str]:
    """Download the MediaPipe Tasks model files once and return their local paths."""
    models_dir = Path(models_dir or C.MODELS_DIR / "mediapipe")
    models_dir.mkdir(parents=True, exist_ok=True)
    names = ["hand_landmarker.task"] + (["pose_landmarker_lite.task"] if need_pose else [])
    out = {}
    for name in names:
        env = os.environ.get("SLT_" + name.split(".")[0].upper())
        path = Path(env) if env else models_dir / name
        if not path.exists():
            print(f"[slt] downloading {name} ...")
            try:
                urllib.request.urlretrieve(C.TASK_MODELS[name], path)
            except Exception as exc:  # pragma: no cover - network dependent
                if path.exists():
                    path.unlink()
                raise RuntimeError(
                    f"Could not download {name} from {C.TASK_MODELS[name]} ({exc}).\n"
                    f"Download it manually and place it in {models_dir}"
                ) from exc
        out[name] = str(path)
    return out


# ---------------------------------------------------------------- extractor
class LandmarkExtractor:
    """Detect pose + hands in BGR frames and return raw frame vectors.

    Parameters
    ----------
    video      : True for webcam / video (tracking mode), False for independent still images.
    use_pose   : disable to speed up pure hand-sign (fingerspelling) use cases.
    num_hands  : 1 or 2.
    backend    : "auto" | "legacy" | "tasks".
    """

    def __init__(self, video: bool = True, use_pose: bool = True, num_hands: int = 2,
                 min_detection_conf: float = 0.5, min_tracking_conf: float = 0.5,
                 model_complexity: int = 1, backend: str = "auto"):
        import mediapipe as mp

        self.mp = mp
        self.video = video
        self.use_pose = use_pose
        self.num_hands = num_hands
        if backend == "auto":
            backend = "legacy" if hasattr(mp, "solutions") else "tasks"
        if backend == "legacy" and not hasattr(mp, "solutions"):
            raise RuntimeError("This mediapipe version has no legacy `solutions` API; use backend='tasks'.")
        self.backend = backend
        self._last_ts_ms = -1
        if backend == "legacy":
            sol = mp.solutions
            self._hands = sol.hands.Hands(
                static_image_mode=not video, max_num_hands=num_hands, model_complexity=min(model_complexity, 1),
                min_detection_confidence=min_detection_conf, min_tracking_confidence=min_tracking_conf)
            self._pose = sol.pose.Pose(
                static_image_mode=not video, model_complexity=model_complexity, smooth_landmarks=video,
                min_detection_confidence=min_detection_conf, min_tracking_confidence=min_tracking_conf) if use_pose else None
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            paths = ensure_task_models(need_pose=use_pose)
            mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
            self._hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=paths["hand_landmarker.task"]),
                running_mode=mode, num_hands=num_hands,
                min_hand_detection_confidence=min_detection_conf,
                min_hand_presence_confidence=min_detection_conf,
                min_tracking_confidence=min_tracking_conf))
            self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=paths["pose_landmarker_lite.task"]),
                running_mode=mode, num_poses=1,
                min_pose_detection_confidence=min_detection_conf,
                min_pose_presence_confidence=min_detection_conf,
                min_tracking_confidence=min_tracking_conf)) if use_pose else None

    # ------------------------------------------------------------------ API
    def process(self, bgr: np.ndarray, timestamp_s: float | None = None) -> np.ndarray:
        """Return the raw 258-d vector for one BGR frame (zeros where nothing was found)."""
        import cv2

        h, w = bgr.shape[:2]
        aspect = w / float(h)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.backend == "legacy":
            rgb.flags.writeable = False
            hres = self._hands.process(rgb)
            pose = None
            if self._pose is not None:
                pres = self._pose.process(rgb)
                if pres.pose_landmarks:
                    pose = np.array([[l.x, l.y, l.z, l.visibility] for l in pres.pose_landmarks.landmark], dtype=np.float32)
            hands = []
            if hres.multi_hand_landmarks:
                for lm, hd in zip(hres.multi_hand_landmarks, hres.multi_handedness):
                    arr = np.array([[l.x, l.y, l.z] for l in lm.landmark], dtype=np.float32)
                    hands.append((arr, hd.classification[0].label))
            return assemble_raw(pose, hands, aspect)

        mp = self.mp
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self.video:
            ts = int((timestamp_s if timestamp_s is not None else self._last_ts_ms / 1000.0 + 0.033) * 1000)
            ts = max(ts, self._last_ts_ms + 1)  # timestamps must strictly increase
            self._last_ts_ms = ts
            hres = self._hands.detect_for_video(image, ts)
            pres = self._pose.detect_for_video(image, ts) if self._pose is not None else None
        else:
            hres = self._hands.detect(image)
            pres = self._pose.detect(image) if self._pose is not None else None
        pose = None
        if pres is not None and pres.pose_landmarks:
            pose = np.array([[l.x, l.y, l.z, l.visibility if l.visibility is not None else 1.0]
                             for l in pres.pose_landmarks[0]], dtype=np.float32)
        hands = []
        for lms, handed in zip(hres.hand_landmarks, hres.handedness):
            arr = np.array([[l.x, l.y, l.z] for l in lms], dtype=np.float32)
            hands.append((arr, handed[0].category_name))
        return assemble_raw(pose, hands, aspect)

    def close(self) -> None:
        for obj in (self._hands, self._pose):
            if obj is not None:
                try:
                    obj.close()
                except Exception:  # pragma: no cover
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def extract_hand_from_image(bgr: np.ndarray, extractor: LandmarkExtractor) -> np.ndarray | None:
    """Convenience for still images: (21, 3) aspect-corrected landmarks of the main hand, or None."""
    return pick_hand(extractor.process(bgr))
