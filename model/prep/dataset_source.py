"""Keep the model's cleaned v2 input synchronized with the latest raw dataset.

The data-engineering notebook historically performed this copy manually. Training
scripts now call this helper so a newly expanded raw CSV cannot be ignored when
the cleaned CSV is stale.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# dataset_source.py lives at <repo>/model/prep; parents[2] is the repository root.
ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_FILE = ROOT / "data" / "raw" / "Job_descriptions_v2.csv"
CLEAN_DATA_FILE = ROOT / "data" / "clean" / "v2" / "cleaned_job_descriptions_v2.csv"
REQUIRED_COLUMNS = {"tech_skills", "job_desc", "role", "type"}


def _read_row_count(path: Path) -> int:
    # Count parsed records rather than physical lines because job descriptions
    # may contain quoted multiline text.
    return len(pd.read_csv(path))


def sync_latest_clean_dataset() -> Path:
    """Refresh the cleaned training CSV when the raw source is newer/larger."""
    if not RAW_DATA_FILE.exists():
        if not CLEAN_DATA_FILE.exists():
            raise FileNotFoundError(
                f"Neither raw nor cleaned v2 dataset exists: {RAW_DATA_FILE} / {CLEAN_DATA_FILE}"
            )
        print(f"Using cleaned v2 dataset (raw source not present): {CLEAN_DATA_FILE}")
        return CLEAN_DATA_FILE

    raw_df = pd.read_csv(RAW_DATA_FILE)
    missing = REQUIRED_COLUMNS - set(raw_df.columns)
    if missing:
        raise ValueError(
            f"Raw v2 dataset is missing required columns: {sorted(missing)}. "
            f"File: {RAW_DATA_FILE}"
        )

    raw_rows = len(raw_df)
    clean_rows = _read_row_count(CLEAN_DATA_FILE) if CLEAN_DATA_FILE.exists() else -1
    raw_is_newer = (
        CLEAN_DATA_FILE.exists()
        and RAW_DATA_FILE.stat().st_mtime_ns > CLEAN_DATA_FILE.stat().st_mtime_ns
    )

    if not CLEAN_DATA_FILE.exists() or raw_rows > clean_rows or (
        raw_rows == clean_rows and raw_is_newer
    ):
        CLEAN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Match the previous notebook's cleaning behavior without dropping rows.
        raw_df = raw_df.fillna("Not Specified")
        raw_df.to_csv(CLEAN_DATA_FILE, index=False)
        print(
            f"Refreshed cleaned training dataset from raw source: "
            f"{clean_rows:,} -> {raw_rows:,} rows"
        )
    else:
        print(
            f"Cleaned training dataset is current: {clean_rows:,} rows "
            f"(raw source: {raw_rows:,} rows)"
        )

    return CLEAN_DATA_FILE
