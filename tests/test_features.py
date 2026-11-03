import numpy as np

from slt import config as C
from slt.features import (FRAME_FEATURE_DIM, STATIC_FEATURE_DIM, active_span, frame_features, hand_features,
                          hands_mask, prepare_sequence, resample_sequence, sequence_features)
from slt.synthetic import make_hand, make_sequence


def test_hand_feature_shape_and_batch():
    h = make_hand(np.array([0.1, 0.5, 0.2, 0.9, 0.3], np.float32))
    assert hand_features(h).shape == (STATIC_FEATURE_DIM,) == (88,)
    assert hand_features(np.stack([h, h])).shape == (2, 88)


def test_hand_features_are_translation_and_scale_invariant():
    h = make_hand(np.array([0.1, 0.5, 0.2, 0.9, 0.3], np.float32))
    moved = h * 2.5 + np.array([0.3, -0.2, 0.0], np.float32)
    np.testing.assert_allclose(hand_features(h), hand_features(moved), atol=1e-4)


def test_hand_features_differ_between_shapes():
    a = hand_features(make_hand(np.zeros(5, np.float32)))
    b = hand_features(make_hand(np.ones(5, np.float32)))
    assert np.abs(a - b).sum() > 1.0


def test_frame_features_shape_and_missing_hands_are_zero():
    rng = np.random.default_rng(0)
    seq = make_sequence(0, 3, rng)
    f = frame_features(seq)
    assert f.shape == (C.SEQ_LEN, FRAME_FEATURE_DIM) == (C.SEQ_LEN, 146)
    assert np.all(f[:, -2] == 0) and np.all(f[:, -1] == 1)          # only the 'R' slot is present
    assert np.isfinite(f).all()
    assert np.allclose(frame_features(np.zeros((4, C.RAW_FRAME_DIM))), 0)


def test_sequence_features_velocity():
    seq = make_sequence(1, 3, np.random.default_rng(1))
    assert sequence_features(seq, True).shape == (C.SEQ_LEN, 292)
    assert sequence_features(seq, False).shape == (C.SEQ_LEN, 146)
    assert np.allclose(sequence_features(seq, True)[0, 146:], 0)


def test_resample_and_active_span():
    seq = np.zeros((20, C.RAW_FRAME_DIM), np.float32)
    seq[5:12, C.RH_SLICE] = 1.0
    assert resample_sequence(seq, 30).shape == (30, C.RAW_FRAME_DIM)
    span = active_span(seq, margin=1)
    assert len(span) == 9 and hands_mask(span)[1:-1].all()
    out = prepare_sequence(seq, 30)
    assert out.shape == (30, C.RAW_FRAME_DIM)
    assert prepare_sequence(np.zeros((10, C.RAW_FRAME_DIM)), 30) is None
    assert prepare_sequence(np.zeros((10, C.RAW_FRAME_DIM)), 30, keep_if_empty=True).shape[0] == 30
