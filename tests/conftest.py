"""Shared fixtures: a small synthetic dataset and models trained on it (built once per session)."""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

LETTERS = list("ABCDEF")
WORDS = ["HELLO", "THANK_YOU", "YES", "NO", "HELP"]


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    from slt.synthetic import make_dynamic_dataset, make_static_dataset

    root = tmp_path_factory.mktemp("synth")
    data, models = root / "data", root / "models"
    make_static_dataset(data, LETTERS, 50, "own")
    make_static_dataset(data, LETTERS, 120, "external")
    make_dynamic_dataset(data, WORDS, 12, "own")
    make_dynamic_dataset(data, WORDS, 30, "external")
    return data, models


@pytest.fixture(scope="session")
def trained(synth):
    from slt.training import train_dynamic, train_static

    data, models = synth
    s = train_static(data, models, model="rf", n_aug=2, verbose=False)
    d = train_dynamic(data, models, epochs=25, patience=10, verbose=False)
    return data, models, s, d
