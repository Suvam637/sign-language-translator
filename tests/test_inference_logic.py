from slt.inference import SentenceBuilder, StableCommitter


def test_committer_requires_hold_and_blocks_repeats():
    c = StableCommitter(hold=3, min_prob=0.8)
    out = [c.update("A", 0.9) for _ in range(6)]
    assert out == [None, None, "A", None, None, None]
    assert [c.update("B", 0.95) for _ in range(3)] == [None, None, "B"]
    c.release()
    assert [c.update("B", 0.95) for _ in range(3)] == [None, None, "B"]      # allowed again after release


def test_committer_ignores_low_probability_and_idle():
    c = StableCommitter(hold=2, min_prob=0.8)
    assert all(c.update("A", 0.5) is None for _ in range(5))
    assert all(c.update("IDLE", 0.99) is None for _ in range(5))
    assert [c.update("A", 0.9) for _ in range(2)] == [None, "A"]
    for _ in range(2):                                                   # idle in between lets the same sign repeat
        c.update("IDLE", 0.99)
    assert [c.update("A", 0.9) for _ in range(2)] == [None, "A"]


def test_sentence_builder_letters_words_and_phrases():
    s = SentenceBuilder(pause_sec=1.0)
    for i, ch in enumerate("HI"):
        s.add(ch, 0.1 * i)
    assert s.text() == "Hi"
    s.tick(0.5)
    assert s.letters == ["H", "I"]                                       # no pause yet
    s.tick(2.0)
    assert s.letters == [] and s.words == ["hi"]
    s.add("THANK_YOU", 3.0)
    s.add("WATER", 3.5)
    assert s.text() == "Hi Thank you I need water"
    s.backspace()
    assert s.text() == "Hi Thank you"
    s.clear()
    assert s.text() == ""
