import numpy as np
import pytest

from slt import config as C
from slt.augment import augment_hand, augment_sequence, mirror_hand, mirror_sequence
from slt.data import (append_static_rows, boost_source, cap_per_class, grouped_split, load_static,
                      normalize_label, save_sequence, load_dynamic)
from slt.synthetic import make_hand, make_sequence


def test_mirror_sequence_is_an_involution_and_swaps_hands():
    seq = make_sequence(2, 4, np.random.default_rng(0))
    m = mirror_sequence(seq)
    assert np.any(m[:, C.LH_SLICE]) and not np.any(m[:, C.RH_SLICE])    # right hand moved to the left slot
    np.testing.assert_allclose(mirror_sequence(m), seq, atol=1e-5)


def test_mirror_hand_involution():
    h = make_hand(np.full(5, 0.3, np.float32))
    np.testing.assert_allclose(mirror_hand(mirror_hand(h)), h, atol=1e-6)


def test_augment_hand_changes_values_keeps_shape():
    rng = np.random.default_rng(0)
    h = make_hand(np.full(5, 0.3, np.float32))
    a = augment_hand(h, rng)
    assert a.shape == h.shape and not np.allclose(a, h)


def test_augment_sequence_keeps_shape_and_missing_slots():
    rng = np.random.default_rng(0)
    seq = make_sequence(1, 3, rng)
    for _ in range(10):
        a = augment_sequence(seq, rng, p_mirror=0.0)
        assert a.shape == seq.shape and np.isfinite(a).all()
        assert not np.any(a[:, C.LH_SLICE])                              # a hand that was never there stays absent
    assert not np.allclose(augment_sequence(seq, rng, p_mirror=0.0), seq)


def test_normalize_label():
    assert normalize_label("  thank you ") == "THANK_YOU"
    assert normalize_label("a") == "A"


def test_static_csv_roundtrip(tmp_path):
    h = make_hand(np.full(5, 0.5, np.float32))
    p = tmp_path / "own" / "static" / "landmarks.csv"
    append_static_rows(p, [("a", "g1", h), ("b", "g2", h)])
    append_static_rows(p, [("a", "g3", h)])
    d = load_static(("own",), tmp_path)
    assert d["raw"].shape == (3, 21, 3) and list(d["labels"]) == ["A", "B", "A"]
    np.testing.assert_allclose(d["raw"][0], h, atol=1e-5)
    assert load_static(("external",), tmp_path)["raw"].shape == (0, 21, 3)


def test_dynamic_roundtrip(tmp_path):
    seq = make_sequence(0, 2, np.random.default_rng(0))
    save_sequence("own", "hello world", "a", seq, tmp_path)
    d = load_dynamic(("own",), tmp_path)
    assert d["raw"].shape == (1, C.SEQ_LEN, C.RAW_FRAME_DIM) and d["labels"][0] == "HELLO_WORLD"


def test_grouped_split_has_no_group_leakage():
    labels = np.repeat(["A", "B", "C"], 100)
    groups = np.array([f"{l}{i // 10}" for l in "ABC" for i in range(100)])
    tr, va, te = grouped_split(labels, groups, 0.2, 0.2, seed=1)
    assert len(set(tr) | set(va) | set(te)) == 300 and not (set(tr) & set(te)) and not (set(tr) & set(va))
    g = lambda ix: set(groups[ix])  # noqa: E731
    assert not (g(tr) & g(va)) and not (g(tr) & g(te)) and not (g(va) & g(te))
    for lab in "ABC":
        assert (labels[tr] == lab).any() and (labels[te] == lab).any() and (labels[va] == lab).any()


def test_grouped_split_small_classes_fall_back_to_random():
    labels = np.array(["A"] * 4 + ["B"] * 2)
    groups = np.array(["g"] * 6)
    tr, va, te = grouped_split(labels, groups)
    assert len(tr) >= 1 and len(set(tr) | set(va) | set(te)) == 6


def test_cap_and_boost():
    labels = np.array(["A"] * 10 + ["A"] * 3)
    source = np.array(["external"] * 10 + ["own"] * 3)
    idx = np.arange(13)
    capped = cap_per_class(idx, labels, source, 4)
    assert (source[capped] == "external").sum() == 4 and (source[capped] == "own").sum() == 3
    boosted = boost_source(idx, source, 3)
    assert len(boosted) == 13 + 2 * 3
