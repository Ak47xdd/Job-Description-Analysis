"""
JobAnalyze v2 Sentence-Transformer Feature Pipeline

This module is an experimental replacement for the TF-IDF feature pipeline.
It converts job descriptions into dense contextual embeddings using
sentence-transformers and saves them for a downstream PyTorch MLP.

Install:
    pip install sentence-transformers

Run from the repository root:
    python model/sentence_transformer_v2.py

The default model is all-MiniLM-L6-v2 (384-dimensional embeddings).
The role and job type are included in the text so the model can use the
same contextual input fields as the original TF-IDF pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "clean" / "v2" / "cleaned_job_descriptions_v2.csv"
DEFAULT_OUTPUT = ROOT / "model" / "prep" / "v2_sentence_transformer"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _find_column(columns: Iterable[str], candidates: Sequence[str]) -> str:
    normalized = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    raise KeyError(f"Could not find any of {list(candidates)} in columns: {list(columns)}")


def build_input_texts(df: pd.DataFrame) -> list[str]:
    """Build contextual text from description, role, and type columns."""
    description_col = _find_column(
        df.columns, ["job description", "job_desc", "description", "Job_Desc"]
    )
    role_col = _find_column(df.columns, ["role", "Role"])
    type_col = _find_column(df.columns, ["type", "job type", "job_type", "Type"])

    descriptions = df[description_col].fillna("").astype(str)
    roles = df[role_col].fillna("").astype(str)
    job_types = df[type_col].fillna("").astype(str)

    return [
        f"Role: {role}. Job type: {job_type}. Job description: {description}"
        for description, role, job_type in zip(descriptions, roles, job_types)
    ]


def encode_dataset(
    csv_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
) -> dict:
    """Encode all rows and save embeddings plus metadata."""
    df = pd.read_csv(csv_path)
    texts = build_input_texts(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_model = SentenceTransformer(model_name)

    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    ).astype(np.float32)

    # Keep the original row order so labels and embeddings remain aligned.
    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "indexes.npy", np.arange(len(df), dtype=np.int64))
    df.to_csv(output_dir / "source_rows.csv", index=False)

    metadata = {
        "model_name": model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "num_samples": int(embeddings.shape[0]),
        "normalize_embeddings": normalize_embeddings,
        "batch_size": batch_size,
        "source_csv": str(csv_path),
        "source_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "input_format": "Role + Job type + Job description",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def load_embeddings(output_dir: Path = DEFAULT_OUTPUT) -> tuple[np.ndarray, dict]:
    """Load saved embeddings and metadata for model training."""
    embeddings = np.load(output_dir / "embeddings.npy")
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    return embeddings, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Sentence-Transformer embeddings for JobAnalyze v2")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    metadata = encode_dataset(
        csv_path=args.data,
        output_dir=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        normalize_embeddings=not args.no_normalize,
    )
    print(json.dumps(metadata, indent=2))
    print(f"Saved embeddings to: {args.output / 'embeddings.npy'}")


if __name__ == "__main__":
    main()
