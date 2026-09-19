"""Prepare a Render-friendly INT8 ONNX artifact for JobAnalyze SBERT.

Run once during model preparation, not inside the Render web process:

    python model/export_sbert_onnx.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer, export_dynamic_quantized_onnx_model

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUTPUT = "model_out/sbert_onnx/all-MiniLM-L6-v2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} with the ONNX backend...")
    model = SentenceTransformer(args.model, backend="onnx", device="cpu")

    print("Exporting dynamic INT8 ONNX with the AVX2 quantization profile...")
    export_dynamic_quantized_onnx_model(
        model=model,
        quantization_config="avx2",
        model_name_or_path=str(output),
        file_suffix="qint8_avx2",
    )

    expected = output / "onnx" / "model_qint8_avx2.onnx"
    if not expected.exists():
        raise RuntimeError(f"Quantized artifact was not created at {expected}")

    print(f"Ready: {expected}")
    print("Configure Render with SBERT_ONNX_FILE=onnx/model_qint8_avx2.onnx")


if __name__ == "__main__":
    main()
