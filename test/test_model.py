from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_PREP_DIR = REPO_ROOT / "model" / "prep"
MODEL_OUT_DIR = REPO_ROOT / "model_out"

def test_model_weights_exist():
    weights_path = MODEL_OUT_DIR / "skill_classifier.pt"
    assert weights_path.exists(), f"Missing model weights, run pipeline.py: {weights_path}"


def test_prepared_data_exist():
    prepared_data_path = MODEL_PREP_DIR / "prepared_data.npz"
    assert prepared_data_path.exists(), f"Missing prepared data, run pipeline.py: {prepared_data_path}"


def test_label_vocab_exist():
    vocab_path = MODEL_PREP_DIR / "label_vocab.json"
    assert vocab_path.exists(), f"Missing label vocab, run pipeline.py: {vocab_path}"


def test_vectorizer_exist():
    vectorizer_path = MODEL_PREP_DIR / "vectorizer.pkl"
    assert vectorizer_path.exists(), f"Missing vectorizer, run pipeline.py: {vectorizer_path}"


