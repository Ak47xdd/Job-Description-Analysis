import gc
import json
import pickle
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="InconsistentVersionWarning: Trying to unpickle estimator*")
warnings.filterwarnings("ignore", message="Trying to unpickle estimator*")

ROOT = Path(__file__).resolve().parents[3]

_label_vocab = None
_vectorizer = None
_classifier_weights = None


def _load_model():
    global _label_vocab, _vectorizer, _classifier_weights

    if _label_vocab is not None and _vectorizer is not None and _classifier_weights is not None:
        return _label_vocab, _vectorizer, _classifier_weights

    prep_dir = ROOT / "model" / "prep"
    if not prep_dir.exists():
        alt = ROOT / "prep"
        prep_dir = alt if alt.exists() else prep_dir

    label_path = prep_dir / "v1" / "label_vocab.json"
    vector_path = prep_dir / "v1" / "vectorizer.pkl"
    weights_path = ROOT / "model_out" / "v1" / "skill_classifier.npz"

    if not label_path.exists():
        raise FileNotFoundError(f"Missing {label_path}.")
    if not vector_path.exists():
        raise FileNotFoundError(f"Missing {vector_path}.")
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing lightweight v1 classifier artifact: {weights_path}. "
            "Run python model/export_v1_classifier_numpy.py and commit the generated .npz."
        )

    with label_path.open(encoding="utf-8") as handle:
        label_vocab = json.load(handle)
    with vector_path.open("rb") as handle:
        vectorizer = pickle.load(handle)

    with np.load(weights_path, allow_pickle=False) as data:
        w1 = np.asarray(data["w1"], dtype=np.float32)
        b1 = np.asarray(data["b1"], dtype=np.float32)
        w2 = np.asarray(data["w2"], dtype=np.float32)
        b2 = np.asarray(data["b2"], dtype=np.float32)

    input_dim = len(getattr(vectorizer, "vocabulary_", {})) or vectorizer.transform([""]).shape[1]
    expected = [
        (w1, (32, input_dim)),
        (b1, (32,)),
        (w2, (len(label_vocab), 32)),
        (b2, (len(label_vocab),)),
    ]
    for value, shape in expected:
        if value.shape != shape:
            raise ValueError(f"Invalid v1 NumPy classifier shape {value.shape}; expected {shape}.")

    _label_vocab, _vectorizer, _classifier_weights = label_vocab, vectorizer, (w1, b1, w2, b2)
    return _label_vocab, _vectorizer, _classifier_weights


def JobAnalyze_6k(job_desc="", role="", job_type="", top_k=50):
    label_vocab, vectorizer, weights = _load_model()
    w1, b1, w2, b2 = weights

    features = vectorizer.transform([f"{job_desc} {role} {job_type}"]).toarray().astype(np.float32, copy=False)
    hidden = np.maximum(0.0, features @ w1.T + b1)
    logits = np.clip(hidden @ w2.T + b2, -60.0, 60.0)
    probabilities = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32, copy=False)[0]

    ranked = sorted(zip(label_vocab, probabilities), key=lambda item: -float(item[1]))
    del features, hidden, logits, probabilities
    gc.collect()
    return ranked[:top_k]
