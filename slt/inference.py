"""Real-time inference: model wrappers, temporal smoothing, sentence building, translator."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from . import config as C
from .features import hand_features, prepare_sequence, resample_sequence, sequence_features
from .landmarks import hands_present, pick_hand

PHRASES_PATH = Path(__file__).with_name("phrases.json")


# ---------------------------------------------------------------- predictors
class StaticPredictor:
    """Single hand shape -> letter / digit / short sign."""

    def __init__(self, models_dir: Path | str | None = None):
        import joblib

        path = Path(models_dir or C.MODELS_DIR) / "static_model.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run scripts/train_static.py first")
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.classes = list(bundle["classes"])
        try:                                   # single-sample predictions are faster without a thread pool
            self.model.set_params(n_jobs=1)
        except (ValueError, TypeError):
            pass

    def predict_proba(self, hand: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(hand_features(hand)[None])[0]

    def predict(self, hand: np.ndarray) -> tuple[str, float]:
        p = self.predict_proba(hand)
        k = int(np.argmax(p))
        return self.classes[k], float(p[k])


class DynamicPredictor:
    """Landmark sequence -> word / phrase."""

    def __init__(self, models_dir: Path | str | None = None, device: str = "cpu"):
        import torch

        from .seq_model import SignGRU

        path = Path(models_dir or C.MODELS_DIR) / "dynamic_model.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run scripts/train_dynamic.py first")
        ck = torch.load(path, map_location=device, weights_only=True)
        self.torch = torch
        self.device = torch.device(device)
        self.classes = list(ck["labels"])
        self.seq_len = int(ck["seq_len"])
        self.use_velocity = bool(ck["use_velocity"])
        self.mean = ck["mean"].numpy()
        self.std = ck["std"].numpy()
        self.net = SignGRU(ck["input_dim"], len(self.classes), ck["hidden"], ck["layers"], ck["dropout"])
        self.net.load_state_dict(ck["state_dict"])
        self.net.to(self.device).eval()

    def predict_proba(self, raw_seq: np.ndarray) -> np.ndarray:
        if len(raw_seq) != self.seq_len:
            raw_seq = resample_sequence(raw_seq, self.seq_len)
        x = (sequence_features(raw_seq, self.use_velocity) - self.mean) / self.std
        with self.torch.no_grad():
            logits = self.net(self.torch.from_numpy(x.astype(np.float32))[None].to(self.device))
            return self.torch.softmax(logits, dim=1)[0].cpu().numpy()

    def predict(self, raw_seq: np.ndarray) -> tuple[str, float]:
        p = self.predict_proba(raw_seq)
        k = int(np.argmax(p))
        return self.classes[k], float(p[k])


# ---------------------------------------------------------------- smoothing
class StableCommitter:
    """Turns noisy per-frame predictions into discrete, de-duplicated tokens.

    A label is committed after it was predicted ``hold`` times in a row with probability
    >= ``min_prob``. After a commit the same label is blocked until the prediction
    changes (or ``release()`` is called, e.g. when the hand leaves the picture).
    """

    def __init__(self, hold: int = 8, min_prob: float = 0.8, ignore=C.IDLE_LABELS):
        self.hold, self.min_prob = hold, min_prob
        self.ignore = {s.upper() for s in ignore}
        self.reset()

    def reset(self):
        self.cand, self.count, self.locked = None, 0, None

    def release(self):
        self.cand, self.count, self.locked = None, 0, None

    def update(self, label: str, prob: float) -> str | None:
        if label.upper() in self.ignore or prob < self.min_prob:
            self.cand, self.count = None, 0
            if label.upper() in self.ignore:
                self.locked = None
            return None
        if label != self.cand:
            self.cand, self.count = label, 1
            if label != self.locked:
                self.locked = None
        else:
            self.count += 1
        if self.count >= self.hold and self.locked != label:
            self.locked = label
            return label
        return None


class SentenceBuilder:
    """Collects committed tokens into text. Letters are spelled into words."""

    def __init__(self, pause_sec: float = 1.5, phrases: dict | None = None):
        if phrases is None:
            phrases = json.loads(PHRASES_PATH.read_text(encoding="utf-8")) if PHRASES_PATH.exists() else {}
        self.phrases = {k.upper(): v for k, v in phrases.items()}
        self.pause_sec = pause_sec
        self.clear()

    def clear(self):
        self.words: list[str] = []
        self.letters: list[str] = []
        self._last_t = 0.0

    def _flush_letters(self):
        if self.letters:
            self.words.append("".join(self.letters).lower())
            self.letters = []

    def add(self, token: str, t: float = 0.0):
        self._last_t = t
        if len(token) == 1:                       # fingerspelled letter / digit
            self.letters.append(token)
            return
        self._flush_letters()
        self.words.append(self.phrases.get(token.upper(), token.replace("_", " ").lower()))

    def tick(self, t: float):
        """Call regularly; a pause after letters ends the spelled word."""
        if self.letters and t - self._last_t > self.pause_sec:
            self._flush_letters()

    def backspace(self):
        if self.letters:
            self.letters.pop()
        elif self.words:
            self.words.pop()

    def text(self) -> str:
        parts = self.words + (["".join(self.letters).lower()] if self.letters else [])
        s = " ".join(parts).strip()
        return (s[:1].upper() + s[1:]) if s else ""


# ---------------------------------------------------------------- buffer
class TemporalBuffer:
    """Last ``window_sec`` seconds of raw frames (used by the 'sliding' dynamic mode)."""

    def __init__(self, window_sec: float = C.WINDOW_SEC, seq_len: int = C.SEQ_LEN):
        self.window_sec, self.seq_len = window_sec, seq_len
        self.items: deque[tuple[float, np.ndarray]] = deque()

    def add(self, t: float, raw: np.ndarray):
        self.items.append((t, raw))
        while self.items and t - self.items[0][0] > self.window_sec:
            self.items.popleft()

    def clear(self):
        self.items.clear()

    def __len__(self):
        return len(self.items)

    def duration(self) -> float:
        return self.items[-1][0] - self.items[0][0] if len(self.items) > 1 else 0.0

    def hand_ratio(self) -> float:
        return float(np.mean([hands_present(r) for _, r in self.items])) if self.items else 0.0

    def last_hand_time(self) -> float | None:
        for t, r in reversed(self.items):
            if hands_present(r):
                return t
        return None

    def sequence(self) -> np.ndarray | None:
        return prepare_sequence(np.stack([r for _, r in self.items]), self.seq_len)


# --------------------------------------------------------------- translator
class SignTranslator:
    """Frame-driven translator that fuses the static (letters) and dynamic (words) models.

    Call ``update(raw_frame, t)`` for every video frame (``t`` in seconds). It returns a dict
    with the latest predictions, newly committed tokens and the running sentence.

    dynamic_mode
      * ``"segment"`` (default) - a sign starts when a hand appears and ends when the hands have
        been out of view for ``hand_gap_sec``. The active stretch is classified once. Most
        reliable for isolated words: lower your hands between signs.
      * ``"sliding"`` - classify the last ``window_sec`` seconds every ``predict_every`` seconds.
        Works with hands that stay in view but is more prone to early / partial predictions.

    Static letters are only accepted while the hand is (nearly) still, so the movement of a
    word is not mistaken for a sequence of letters.
    """

    def __init__(self, static: StaticPredictor | None = None, dynamic: DynamicPredictor | None = None,
                 dynamic_mode: str = "segment", window_sec: float = C.WINDOW_SEC, predict_every: float = 0.2,
                 min_prob_static: float = 0.85, min_prob_dynamic: float = 0.80, hold_static: int = 10,
                 hold_dynamic: int = 2, hand_gap_sec: float = 0.4, min_sign_sec: float = 0.4,
                 max_segment_sec: float = 4.0, max_static_speed: float = 1.5, min_hand_ratio: float = 0.25,
                 pause_sec: float = 1.5):
        if static is None and dynamic is None:
            raise ValueError("Provide at least one of static / dynamic predictors")
        if dynamic_mode not in ("segment", "sliding"):
            raise ValueError("dynamic_mode must be 'segment' or 'sliding'")
        self.static, self.dynamic, self.dynamic_mode = static, dynamic, dynamic_mode
        self.seq_len = dynamic.seq_len if dynamic else C.SEQ_LEN
        self.static_commit = StableCommitter(hold_static, min_prob_static)
        self.dynamic_commit = StableCommitter(hold_dynamic, min_prob_dynamic)
        self.sentence = SentenceBuilder(pause_sec)
        self.buffer = TemporalBuffer(window_sec, self.seq_len)
        self.predict_every, self.min_hand_ratio = predict_every, min_hand_ratio
        self.hand_gap_sec, self.min_sign_sec = hand_gap_sec, min_sign_sec
        self.max_segment_sec, self.max_static_speed = max_segment_sec, max_static_speed
        self.min_prob_dynamic = min_prob_dynamic
        self.clear()

    # ------------------------------------------------------------------ state
    def clear(self):
        self.sentence.clear()
        self.buffer.clear()
        self.static_commit.reset()
        self.dynamic_commit.reset()
        self.last_static = self.last_dynamic = None
        self.state = "idle"
        self._seg: list[tuple[float, np.ndarray]] = []
        self._seg_static_hits = 0
        self._last_hand_t = -1e9
        self._next_pred_t = 0.0
        self._suppress_static_until = -1.0
        self._prev_wrist = None
        self._prev_t = None

    def _hand_speed(self, hand: np.ndarray, t: float) -> float:
        """Wrist speed in hand-sizes per second (scale invariant)."""
        size = max(float(np.linalg.norm(hand[9, :2] - hand[0, :2])), 1e-4)
        speed = 0.0
        if self._prev_wrist is not None and self._prev_t is not None and t > self._prev_t:
            speed = float(np.linalg.norm(hand[0, :2] - self._prev_wrist)) / size / (t - self._prev_t)
        self._prev_wrist, self._prev_t = hand[0, :2].copy(), t
        return speed

    # ---------------------------------------------------------------- update
    def update(self, raw: np.ndarray, t: float) -> dict:
        committed: list[tuple[str, str]] = []
        present = hands_present(raw)

        # -------- static letters / digits
        if not present:
            self.static_commit.release()
            self.last_static = None
            self._prev_wrist = None
        elif self.static is not None and t >= self._suppress_static_until:
            hand = pick_hand(raw)
            if hand is not None:
                if self._hand_speed(hand, t) > self.max_static_speed:
                    self.static_commit.update("__moving__", 0.0)
                    self.last_static = None
                else:
                    label, prob = self.static.predict(hand)
                    self.last_static = (label, prob)
                    tok = self.static_commit.update(label, prob)
                    if tok:
                        committed.append(("static", tok))
                        self._seg_static_hits += 1

        # -------- dynamic words
        if self.dynamic is not None:
            if self.dynamic_mode == "segment":
                self._update_segment(raw, t, present, committed)
            else:
                self._update_sliding(raw, t, committed)

        for _, tok in committed:
            self.sentence.add(tok, t)
        self.sentence.tick(t)
        return {"static": self.last_static, "dynamic": self.last_dynamic, "committed": committed,
                "hands": present, "state": self.state, "text": self.sentence.text()}

    def flush(self, t: float) -> dict:
        """Finish a sign that is still in progress (call at the end of a video)."""
        committed: list[tuple[str, str]] = []
        if self.dynamic is not None and self.dynamic_mode == "segment" and self._seg:
            self._finish_segment(t, committed)
        for _, tok in committed:
            self.sentence.add(tok, t)
        return {"committed": committed, "text": self.sentence.text()}

    # -------------------------------------------------------- segment mode
    def _update_segment(self, raw, t, present, committed):
        if present:
            if not self._seg:
                self._seg_static_hits = 0
            self._seg.append((t, raw))
            self._last_hand_t = t
            while self._seg and t - self._seg[0][0] > self.max_segment_sec:
                self._seg.pop(0)
            self.state = "recording sign"
        elif self._seg:
            self._seg.append((t, raw))                       # keep short drop-outs (timing stays right)
            if t - self._last_hand_t >= self.hand_gap_sec:
                self._finish_segment(t, committed)
        else:
            self.state = "idle"

    def _finish_segment(self, t, committed):
        seg, self._seg = self._seg, []
        self.state = "idle"
        duration = self._last_hand_t - seg[0][0]
        fingerspelling = self.static is not None and self._seg_static_hits > 0
        if duration < self.min_sign_sec or fingerspelling:
            return
        seq = prepare_sequence(np.stack([r for _, r in seg]), self.seq_len)
        if seq is None:
            return
        label, prob = self.dynamic.predict(seq)
        self.last_dynamic = (label, prob)
        if label.upper() not in C.IDLE_LABELS and prob >= self.min_prob_dynamic:
            committed.append(("dynamic", label))
            self._suppress_static_until = t + 0.8

    # -------------------------------------------------------- sliding mode
    def _update_sliding(self, raw, t, committed):
        self.buffer.add(t, raw)
        if t < self._next_pred_t:
            return
        self._next_pred_t = t + self.predict_every
        lt = self.buffer.last_hand_time()
        enough = len(self.buffer) >= 8 and self.buffer.duration() >= 0.5 * self.buffer.window_sec
        recent = lt is not None and (t - lt) <= self.hand_gap_sec
        seq = self.buffer.sequence() if (enough and recent and self.buffer.hand_ratio() >= self.min_hand_ratio) else None
        if seq is None:
            self.last_dynamic = None
            self.dynamic_commit.release()
            return
        label, prob = self.dynamic.predict(seq)
        self.last_dynamic = (label, prob)
        tok = self.dynamic_commit.update(label, prob)
        if tok:
            committed.append(("dynamic", tok))
            self.buffer.clear()                              # avoid re-triggering on the same sign
            self.dynamic_commit.release()
            self._suppress_static_until = t + 1.2


# ----------------------------------------------------------------- loading
def load_predictors(models_dir: Path | str | None = None, mode: str = "auto"):
    """Load the requested models. mode: auto | static | dynamic | both. Returns (static, dynamic)."""
    models_dir = Path(models_dir or C.MODELS_DIR)
    has_s = (models_dir / "static_model.joblib").exists()
    has_d = (models_dir / "dynamic_model.pt").exists()
    if mode == "auto":
        want_s, want_d = has_s, has_d
    else:
        want_s, want_d = mode in ("static", "both"), mode in ("dynamic", "both")
    if (want_s and not has_s) or (want_d and not has_d) or not (want_s or want_d):
        missing = []
        if want_s and not has_s:
            missing.append("static_model.joblib (run scripts/train_static.py)")
        if want_d and not has_d:
            missing.append("dynamic_model.pt (run scripts/train_dynamic.py)")
        raise SystemExit("Missing trained model(s) in " + str(models_dir) + ": " +
                         (", ".join(missing) if missing else "none found - train a model first"))
    return (StaticPredictor(models_dir) if want_s else None,
            DynamicPredictor(models_dir) if want_d else None)


def speak_async(text: str) -> None:
    """Speak ``text`` with the optional pyttsx3 package (non-blocking)."""
    import threading

    try:
        import pyttsx3
    except ImportError:
        print("[slt] text-to-speech needs:  pip install pyttsx3")
        return

    def _run():
        try:
            eng = pyttsx3.init()
            eng.say(text)
            eng.runAndWait()
        except Exception as exc:  # pragma: no cover - audio backend specific
            print(f"[slt] speech failed: {exc}")

    threading.Thread(target=_run, daemon=True).start()
