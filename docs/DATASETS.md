# Using outside datasets

The project keeps **your recordings** (`data/own/`) and **third-party data** (`data/external/`) in
separate folders, merges them for training, and reports accuracy per source.

> The dataset names below come from memory (no internet search was available when this project was
> written). Check that each one still exists, what it contains, and **its licence / terms of use**
> before downloading or redistributing it.

## Static signs (hand shapes, fingerspelling, digits) -> `import_external_static.py`

Need: images in one folder per class.

| Dataset (check yourself) | Notes |
|---|---|
| ASL Alphabet image set (Kaggle, ~87k photos, 29 classes) | has extra classes `del`, `nothing`, `space` -> use `--skip DEL NOTHING SPACE` |
| Sign Language Digits Dataset (GitHub: ardamavi/Sign-Language-Digits-Dataset) | digits 0-9, small 100x100 images; a few of these were used to test the importer |
| Indian Sign Language alphabet / digit image sets (Kaggle, several uploads) | quality varies - check detection rate printed by the importer |

```bash
python scripts/import_external_static.py --root ~/datasets/asl_alphabet_train \
       --max-per-class 1500 --skip DEL NOTHING SPACE
```

The importer prints how many images per class contained a detectable hand. Below ~50 % usually means tiny
images, heavy blur, or gloved / cropped hands.

## Dynamic signs (words / phrases) -> `import_external_dynamic.py`

Need: short videos, one folder per sign (or a CSV with `path,label`).

| Dataset (check yourself) | Notes |
|---|---|
| WLASL (word-level ASL, ~2000 glosses) | official release is a list of video URLs, many dead; processed re-uploads exist. Select a subset (20-50 words) |
| MS-ASL | similar situation, video links |
| INCLUDE (AI4Bharat, Indian Sign Language words) | relevant if you target ISL |
| How2Sign | continuous ASL; useful later for sentence-level work, not for this isolated-sign classifier |

```bash
python scripts/import_external_dynamic.py --root ~/datasets/wlasl_subset --max-per-class 40 \
       --label-map label_map.example.json
```

## Make labels match

`"Thank you"`, `"thank_you"` and `"THANK-YOU"` are different classes unless you map them. Labels are
upper-cased and spaces become `_`; anything else (hyphens...) must be renamed with `--label-map`
(see `label_map.example.json`) so own and external data of the same sign end up in the same class.

## Sign languages are not interchangeable

ASL, ISL, BSL ... are different languages. A model trained on ASL data will not recognise ISL signs.
Do not mix classes from different languages unless you give them different labels, and prefer data signed
by **many different people** and recorded in **different conditions** - that matters more than raw count.

## Check generalisation honestly

```bash
# train on your own recordings only, test on the external data (and the other way round)
python scripts/train_static.py --sources own
python scripts/evaluate.py --task static --eval-on external
```
A big gap between the held-out test accuracy and this cross-source accuracy means the model has
learned the quirks of one signer / camera / background rather than the sign.
