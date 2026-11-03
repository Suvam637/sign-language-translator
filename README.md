# Sign Language Translator (MediaPipe + OpenCV + Python)

Real-time sign recognition and translation to text (and optional speech) from a webcam or a video file.
It handles two kinds of signs:

* **Static signs** - hand shapes such as fingerspelled letters and digits -> scikit-learn classifier.
* **Dynamic signs** - words / phrases that involve movement -> PyTorch BiGRU with attention on landmark sequences.

You can train on **your own recordings**, on **outside datasets**, or on both together.

## What problem does it solve, and how?

**Problem.** Millions of Deaf and hard-of-hearing people use a sign language, but most hearing people do not,
and human interpreters are scarce and expensive - so everyday situations (clinic, bank counter, shop, classroom,
a government office) often have no communication bridge at all.

**How this project attacks it**

1. **Landmarks instead of pixels.** MediaPipe turns every frame into pose + 2 x 21 hand landmarks. The models never
   see faces, skin tone or backgrounds, so they are light (real time on a laptop CPU), more robust to lighting
   and clothing, and privacy-friendlier (no video needs to be stored - only coordinates).
2. **Invariant features.** Hand shapes are expressed relative to the wrist and hand size; body-relative
   positions are expressed relative to the shoulders. Distance from the camera and position in the frame
   do not matter.
3. **Two specialised models** (letters vs. moving words) fused by a translator that smooths predictions,
   avoids duplicates, ignores letters while the hand is moving, spells letters into words and maps
   word-signs to natural phrases (`slt/phrases.json`, e.g. `WATER` -> "I need water").
4. **Own + outside data done safely:** separate folders, one shared preprocessing path, own samples up-weighted,
   mirror / rotation / time-warp augmentation, **group-aware splits** (no near-duplicate frames in train *and*
   test) and per-source evaluation so you can measure how well a model generalises to unseen signers.

**What it is *not*.** This is isolated-sign recognition + fingerspelling. Real sign languages have their own
grammar, facial expressions and non-manual markers, so the output is a gloss-to-text approximation, not fluent
sentence translation. Treat it as an assistive prototype and involve Deaf users and interpreters when you
evaluate or deploy it.

## Pipeline

```
 webcam / video
      |
      v
 MediaPipe  ->  raw frame vector (258 floats: pose 33x4 + left hand 21x3 + right hand 21x3)
      |
      +--> static branch : features (88) --> MLP / RandomForest --> letter + confidence
      |
      +--> dynamic branch: trim to active span -> resample to 30 frames -> features (2 x 146)
                           --> BiGRU + attention --> word + confidence
      |
      v
 SignTranslator: stability filter (hold N frames) - motion gate - de-duplication - sentence builder - optional TTS
```

## Install (Python 3.11 or newer)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_setup.py --camera 0     # versions, MediaPipe backend, camera test
python scripts/smoke_test.py                 # ~1-2 min, no camera needed: trains + translates synthetic data
```

**MediaPipe versions (important).** MediaPipe dropped its classic `mp.solutions` API in 0.10.30 and version
0.10.21 (the last one that has it, with bundled models) only has wheels up to Python 3.12.
`requirements.txt` therefore installs 0.10.21 on Python 3.11 / 3.12 and a newer MediaPipe on 3.13+, and the code
picks the matching backend automatically (`--backend legacy|tasks` to force one).
On the *Tasks* backend the two model files are downloaded once into `models/mediapipe/`
(`python scripts/download_models.py`).

> Tested here: Python 3.12 + MediaPipe 0.10.21 (full test-suite, real hand photos, real video decoding).
> The *Tasks* backend was only tested with stand-in detectors (result conversion and option names), because
> the sandbox could not download its model files - run `check_setup.py` and a short live test on your machine
> if you use Python 3.13+.

## Workflow

### 1. Collect your own data

```bash
# static signs (letters / digits): SPACE records a burst of 60 samples, type a letter to pick it
python scripts/collect_static.py                       # A-Z
python scripts/collect_static.py --labels 0 1 2 3 4 5 6 7 8 9

# dynamic signs: SPACE = countdown + 3 s clip; keep hands DOWN before and after the sign
python scripts/collect_dynamic.py --labels HELLO THANK_YOU YES NO HELP WATER IDLE
```

Stored in `data/own/static/landmarks.csv` and `data/own/dynamic/<LABEL>/*.npy` (landmarks only, no images).

Guidelines: >= 5 bursts per letter in different positions; 30-60 clips per word; different lighting, distances,
backgrounds and - best of all - several different people. Record an **IDLE** class (hands raised but not signing,
random gestures) so the model learns to say "nothing".

### 2. Add outside data (optional but recommended)

```bash
python scripts/import_external_static.py  --root ~/datasets/asl_alphabet_train --max-per-class 1500 --skip DEL NOTHING SPACE
python scripts/import_external_dynamic.py --root ~/datasets/wlasl_subset --max-per-class 40 --label-map label_map.example.json
```

Details, dataset suggestions and licence reminders: [`docs/DATASETS.md`](docs/DATASETS.md).

### 3. Train

```bash
python scripts/train_static.py      # auto: trains MLP + random forest, keeps the best on validation data
python scripts/train_dynamic.py     # BiGRU; early stopping; add --device cuda if you have a GPU
```

Useful options: `--own-boost 5` (weight your own recordings more), `--max-per-class 1500` (cap outside
samples per class so a huge dataset does not drown yours), `--sources own` (ignore outside data).
Models and the held-out **test split** are written to `models/`.

### 4. Evaluate

```bash
python scripts/evaluate.py                                # held-out, group-separated test split
python scripts/train_static.py --sources own              # cross-domain check:
python scripts/evaluate.py --task static --eval-on external
```

Prints precision / recall / F1 per class, accuracy per source (own vs external) and saves confusion matrices
(`models/*_confusion.png`).

### 5. Predict

```bash
python scripts/predict_live.py                       # webcam, uses whichever models exist
python scripts/predict_live.py --mode dynamic --dynamic-mode sliding
python scripts/predict_file.py --input photo.jpg     # one image -> top-3 letters
python scripts/predict_file.py --input clip.mp4 --output annotated.mp4
```

Live keys: `C` clear, `Backspace`/`B` delete last, `S` speak (needs `pip install pyttsx3`), `L` toggle landmarks,
`Q`/`Esc` quit. Letters: hold the shape still for about half a second. Words: lower your hands, sign, lower your
hands (`segment` mode) - or sign continuously with `--dynamic-mode sliding`.

## Tests

```bash
pip install pytest
python -m pytest -q          # 35 tests, ~30-60 s (unit tests + synthetic end-to-end + real-time stream tests)
```

## Project layout

```
slt/                    library
  landmarks.py          MediaPipe backends (legacy + Tasks) -> raw frame vector, drawing
  features.py           static & sequence features, trimming / resampling
  augment.py            rotation, scale, mirror, time-warp, hand drop-out
  data.py               own/external layout, CSV/NPY IO, group-aware split, capping, boosting
  training.py           train_static (sklearn) / train_dynamic (PyTorch)
  seq_model.py          BiGRU + attention
  inference.py          predictors, StableCommitter, SentenceBuilder, SignTranslator
  evaluate.py           reports + confusion matrices
  synthetic.py          camera-free synthetic data & streams (for tests only)
  phrases.json          gloss -> natural phrase mapping (edit me)
scripts/                command-line tools (collect_*, import_*, train_*, evaluate, predict_*, smoke_test, check_setup)
tests/                  pytest suite
data/{own,external}/    your landmark data (git-ignored)
models/                 trained models (git-ignored)
```

## Measured on a 1-vCPU cloud sandbox (indicative only; a normal laptop is faster)

| Step | Time |
|---|---|
| MediaPipe hands only | ~14 ms / frame |
| MediaPipe hands + pose | ~43 ms / frame |
| Static classifier (MLP) | ~0.7 ms |
| Dynamic BiGRU | ~4 ms per sign |

Use `--mode static` (no pose model) for the fastest fingerspelling-only setup.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot open camera` | try `--camera 1`; close other apps using the webcam; allow camera access for your terminal / IDE (macOS, Windows) |
| nothing is recognised | lower `--min-prob-static/--min-prob-dynamic` (defaults 0.85/0.80), check the class list printed at start, make sure the hand is fully in frame and well lit |
| wrong letters while moving a hand | letters are only accepted when the hand is nearly still; increase `--hold-static` |
| a word fires several times / never ends | in `segment` mode the hands must leave the picture for `--hand-gap` seconds (default 0.4) |
| `mp.solutions` missing | you have mediapipe >= 0.10.30: the Tasks backend is used automatically; run `scripts/download_models.py` |
| two `cv2` packages / import errors | `pip uninstall opencv-python opencv-python-headless`, keep `opencv-contrib-python` |
| left-handed signers | mirroring is part of the augmentation; for best results record some left-handed samples too |

## Ideas for next steps

Export the static model to ONNX / TensorFlow.js for in-browser use, add a real-time web UI (FastAPI + WebSocket),
add facial landmarks for non-manual markers, move from isolated signs to continuous recognition (CTC or
transformer sequence models) and from glosses to sentences (a gloss-to-text language model).
