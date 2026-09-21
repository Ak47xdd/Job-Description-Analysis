"""
data_prep.py

Build the shared v2 labels/splits used by both TF-IDF and SBERT pipelines.
Run this before model/model.py or model/model_sbert.py when the cleaned v2
job-description dataset changes.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

try:
    from .sym_map import SYNONYM_MAP
except ImportError:
    from sym_map import SYNONYM_MAP


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "clean" / "v2" / "cleaned_job_descriptions_v2.csv"
OUT_DIR = Path(__file__).resolve().parent
V2_DIR = OUT_DIR / "v2"

SKILLS_FIX = {
    "tesnorflow/pytorch": "tensorflow/pytorch",
    "numpyhugging face": "numpy",
    "sytem design": "system design",
    "python. ml": "ml",
}


def normalizer(skills) -> list[str]:
    if pd.isna(skills):
        return []
    skill = [s.strip().lower() for s in str(skills).split(",") if s.strip()]
    fixed = [SKILLS_FIX.get(s, s) for s in skill]
    return list(dict.fromkeys(fixed))


def apply_synonyms(text: str) -> str:
    text = text.lower()
    for phrase, canonical in sorted(SYNONYM_MAP.items(), key=lambda x: -len(x[0])):
        text = text.replace(phrase, canonical)
    return text


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing cleaned v2 dataset: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    required = {"tech_skills", "job_desc", "role", "type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["skill_list"] = df["tech_skills"].apply(normalizer)

    freq = Counter(s for lst in df["skill_list"] for s in lst)
    vocab = sorted([s for s, count in freq.items() if count >= 3])
    if not vocab:
        raise ValueError("No v2 labels survived the minimum frequency filter.")

    vocab_index = {label: i for i, label in enumerate(vocab)}
    y = np.zeros((len(df), len(vocab)), dtype=np.float32)
    for row_idx, skill_list in enumerate(df["skill_list"]):
        for skill in skill_list:
            label_idx = vocab_index.get(skill)
            if label_idx is not None:
                y[row_idx, label_idx] = 1.0

    if y.shape[1] != len(vocab):
        raise RuntimeError(
            f"Internal v2 label construction error: y has {y.shape[1]} columns "
            f"but vocab has {len(vocab)} labels."
        )

    jd_input = (
        df["job_desc"].fillna("").apply(apply_synonyms)
        + " "
        + df["role"].fillna("").astype(str).str.lower()
        + " "
        + df["type"].fillna("").astype(str).str.lower()
    )

    vectorizer = TfidfVectorizer(
        max_features=150,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
    )
    X = vectorizer.fit_transform(jd_input).toarray().astype(np.float32)

    indices = np.arange(len(df), dtype=np.int64)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        indices,
        test_size=0.2,
        random_state=42,
    )

    if y_train.shape[1] != len(vocab) or y_test.shape[1] != len(vocab):
        raise RuntimeError(
            "Generated v2 split labels do not match label_vocab_v2.json dimensions. "
            f"train={y_train.shape[1]}, test={y_test.shape[1]}, vocab={len(vocab)}."
        )

    V2_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        V2_DIR / "prepared_data_v2.npz",
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        idx_train=idx_train,
        idx_test=idx_test,
    )

    with (V2_DIR / "label_vocab_v2.json").open("w", encoding="utf-8") as file:
        json.dump(vocab, file, indent=2)

    with (V2_DIR / "vectorizer_v2.pkl").open("wb") as file:
        pickle.dump(vectorizer, file)

    dataset_sha256 = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()
    manifest = {
        "source_dataset": str(DATA_FILE.relative_to(ROOT)),
        "source_dataset_sha256": dataset_sha256,
        "num_rows": int(len(df)),
        "num_labels": int(len(vocab)),
        "label_vocab_sha256": hashlib.sha256(
            json.dumps(vocab, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "random_state": 42,
        "test_size": 0.2,
    }
    (V2_DIR / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"Dataset: {len(df)} rows | Vocab: {len(vocab)} labels | "
        f"TF-IDF features: {X.shape[1]}"
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Saved shared v2 labels/splits to {V2_DIR}")


if __name__ == "__main__":
    main()
