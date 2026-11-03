import numpy as np
import pytest

from slt import config as C
from slt.landmarks import assemble_raw, draw_raw, hands_present, pick_hand, split_raw_frame
from slt.synthetic import make_hand, make_sequence


def _hand(offset=0.0):
    h = make_hand(np.full(5, 0.2, np.float32), (0.5 + offset, 0.7))
    return h


def test_assemble_one_hand_goes_to_matching_slot():
    raw = assemble_raw(None, [(_hand(), "Left")], aspect=2.0)
    _, lh, rh = split_raw_frame(raw)
    assert np.any(lh) and not np.any(rh)
    assert np.isclose(lh[0, 0], 0.5 * 2.0)                  # x is multiplied by the aspect ratio
    raw = assemble_raw(None, [(_hand(), "Right")], aspect=1.0)
    assert not np.any(split_raw_frame(raw)[1]) and np.any(split_raw_frame(raw)[2])


def test_assemble_two_hands_same_label_uses_image_position():
    raw = assemble_raw(None, [(_hand(0.2), "Right"), (_hand(-0.2), "Right")], aspect=1.0)
    _, lh, rh = split_raw_frame(raw)
    assert lh[0, 0] < rh[0, 0]


def test_pick_hand_and_presence():
    assert not hands_present(np.zeros(C.RAW_FRAME_DIM)) and pick_hand(np.zeros(C.RAW_FRAME_DIM)) is None
    raw = assemble_raw(None, [(_hand(), "Left")], 1.0)
    assert hands_present(raw) and pick_hand(raw).shape == (21, 3)


def test_draw_raw_does_not_crash():
    cv2 = pytest.importorskip("cv2")
    frame = np.zeros((360, 640, 3), np.uint8)
    draw_raw(frame, make_sequence(0, 2, np.random.default_rng(0))[10])
    assert frame.any()


@pytest.mark.parametrize("video", [False, True])
def test_mediapipe_extractor_on_blank_frame(video):
    pytest.importorskip("mediapipe")
    from slt.landmarks import LandmarkExtractor

    try:
        ext = LandmarkExtractor(video=video, use_pose=True)
    except RuntimeError as exc:                              # e.g. Tasks backend without internet access
        pytest.skip(str(exc))
    with ext:
        raw = ext.process(np.zeros((240, 320, 3), np.uint8), 0.0)
    assert raw.shape == (C.RAW_FRAME_DIM,) and not raw.any()


def _fake_tasks_extractor(video):
    """A LandmarkExtractor whose MediaPipe Tasks detectors are replaced by stubs."""
    from types import SimpleNamespace as NS

    mp = pytest.importorskip("mediapipe")
    from slt.landmarks import LandmarkExtractor

    lm = lambda x, y, z, v=None: NS(x=x, y=y, z=z, visibility=v)  # noqa: E731
    hand = [lm(0.5 + 0.01 * i, 0.6 - 0.01 * i, 0.0) for i in range(21)]
    pose = [lm(0.5, 0.2 + 0.02 * i, 0.0, 0.9) for i in range(33)]

    class Hands:
        def _res(self):
            return NS(hand_landmarks=[hand], handedness=[[NS(category_name="Right")]])
        detect = lambda self, img: self._res()                    # noqa: E731
        detect_for_video = lambda self, img, ts: self._res()      # noqa: E731

    class Pose:
        def _res(self):
            return NS(pose_landmarks=[pose])
        detect = lambda self, img: self._res()                    # noqa: E731
        detect_for_video = lambda self, img, ts: self._res()      # noqa: E731

    ext = object.__new__(LandmarkExtractor)
    ext.mp, ext.video, ext.use_pose, ext.backend, ext._last_ts_ms = mp, video, True, "tasks", -1
    ext._hands, ext._pose = Hands(), Pose()
    return ext


@pytest.mark.parametrize("video", [False, True])
def test_tasks_backend_result_conversion(video):
    ext = _fake_tasks_extractor(video)
    frame = np.zeros((240, 320, 3), np.uint8)
    raw = ext.process(frame, 0.10)
    raw2 = ext.process(frame, 0.10)                      # same timestamp must not crash (made strictly increasing)
    pose, lh, rh = split_raw_frame(raw)
    assert not np.any(lh) and np.any(rh) and np.isclose(rh[0, 0], 0.5 * 320 / 240)   # aspect-corrected x
    assert np.isclose(pose[5, 3], 0.9) and raw2.shape == raw.shape


def test_tasks_option_names_match_installed_mediapipe():
    pytest.importorskip("mediapipe")
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    base = mp_python.BaseOptions(model_asset_path="dummy.task")
    vision.HandLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.VIDEO, num_hands=2,
                                 min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
                                 min_tracking_confidence=0.5)
    vision.PoseLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.IMAGE, num_poses=1,
                                 min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
                                 min_tracking_confidence=0.5)
