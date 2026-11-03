"""End-to-end tests on synthetic data: training, evaluation and real-time translation."""
import numpy as np

from slt import config as C
from slt.evaluate import evaluate_dynamic, evaluate_static
from slt.inference import DynamicPredictor, SignTranslator, StaticPredictor
from slt.synthetic import stream_letter, stream_word
from tests.conftest import LETTERS, WORDS


def test_training_reaches_high_accuracy(trained):
    _, models, s, d = trained
    assert s["test_accuracy"] > 0.9 and d["test_accuracy"] > 0.9
    for f in ("static_model.joblib", "static_test.npz", "dynamic_model.pt", "dynamic_test.npz"):
        assert (models / f).exists()


def test_evaluation_reports(trained):
    data, models, _, _ = trained
    assert evaluate_static(models, data, "test", verbose=False)["accuracy"] > 0.9
    assert evaluate_dynamic(models, data, "test", verbose=False)["accuracy"] > 0.9
    r = evaluate_static(models, data, "external", verbose=False)           # cross-source style evaluation runs too
    assert r["n"] == 120 * len(LETTERS)
    assert (models / "static_test_confusion.png").exists()


def test_predictors_return_valid_labels(trained):
    data, models, _, _ = trained
    sp, dp = StaticPredictor(models), DynamicPredictor(models)
    assert set(sp.classes) == set(LETTERS) and set(dp.classes) == set(WORDS)
    t, raw = stream_letter(2, len(LETTERS))[20]
    label, prob = sp.predict(raw[C.RH_SLICE].reshape(21, 3))
    assert label == "C" and prob > 0.5
    seq = np.stack([r for _, r in stream_word(1, len(WORDS))][20:65])
    label, prob = dp.predict(seq)
    assert label == "THANK_YOU"


def _run(tr, frames, t0, gap=1.0):
    for t, raw in frames:
        tr.update(raw, t0 + t)
    return t0 + frames[-1][0] + gap


def test_translator_letters_and_words_segment_mode(trained):
    _, models, _, _ = trained
    tr = SignTranslator(StaticPredictor(models), DynamicPredictor(models))
    quick = dict(seconds=0.6, lead=0.05, trail=0.05)                        # letters spelled one after another
    t = _run(tr, stream_letter(0, len(LETTERS), seed=1, **quick), 0.0, gap=0.0)    # A
    t = _run(tr, stream_letter(4, len(LETTERS), seed=2, **quick), t, gap=0.0)      # E -> same word "ae"
    t = _run(tr, stream_word(0, len(WORDS), seed=3), t + 3.0)               # HELLO (3 s later: new word)
    t = _run(tr, stream_word(4, len(WORDS), seed=4), t)                     # HELP
    tr.flush(t)
    assert tr.sentence.text() == "Ae Hello I need help"


def test_translator_sliding_mode(trained):
    _, models, _, _ = trained
    tr = SignTranslator(None, DynamicPredictor(models), dynamic_mode="sliding")
    t = _run(tr, stream_word(2, len(WORDS), seed=5), 0.0)                   # YES
    tr.flush(t)
    assert tr.sentence.text() == "Yes"


def test_translator_stays_silent_without_hands(trained):
    _, models, _, _ = trained
    tr = SignTranslator(StaticPredictor(models), DynamicPredictor(models))
    for i in range(300):
        out = tr.update(np.zeros(C.RAW_FRAME_DIM, np.float32), i / 30)
        assert out["committed"] == []
    assert tr.sentence.text() == ""


def test_static_only_and_dynamic_only_translators(trained):
    _, models, _, _ = trained
    only_s = SignTranslator(StaticPredictor(models), None)
    _run(only_s, stream_letter(3, len(LETTERS), seed=7), 0.0)
    assert only_s.sentence.text() == "D"
    only_d = SignTranslator(None, DynamicPredictor(models))
    t = _run(only_d, stream_word(3, len(WORDS), seed=8), 0.0)
    only_d.flush(t)
    assert only_d.sentence.text() == "No"
