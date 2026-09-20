"""Export the trained v1 PyTorch classifier to an exact FP32 NumPy artifact.

This script is intentionally a build-time utility. Production inference uses
NumPy only, so PyTorch does not need to be imported by the Render API process.
"""

from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "model_out" / "v1" / "skill_classifier.pt"
OUTPUT = ROOT / "model_out" / "v1" / "skill_classifier.npz"


def main() -> None:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Missing trained v1 weights: {WEIGHTS}")

    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    required = ("net.0.weight", "net.0.bias", "net.3.weight", "net.3.bias")
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(f"Missing classifier tensors: {missing}")

    arrays = {
        "w1": state["net.0.weight"].detach().cpu().numpy().astype(np.float32),
        "b1": state["net.0.bias"].detach().cpu().numpy().astype(np.float32),
        "w2": state["net.3.weight"].detach().cpu().numpy().astype(np.float32),
        "b2": state["net.3.bias"].detach().cpu().numpy().astype(np.float32),
    }

    if arrays["w1"].ndim != 2 or arrays["w2"].ndim != 2:
        raise ValueError("v1 classifier weights must both be rank-2 matrices.")
    if arrays["b1"].shape != (arrays["w1"].shape[0],):
        raise ValueError("v1 hidden bias shape does not match the first layer.")
    if arrays["w2"].shape[1] != arrays["w1"].shape[0]:
        raise ValueError("v1 classifier layer dimensions do not match.")
    if arrays["b2"].shape != (arrays["w2"].shape[0],):
        raise ValueError("v1 output bias shape does not match the second layer.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT, **arrays)
    print(f"Wrote {OUTPUT}")
    print("Shapes:", {key: value.shape for key, value in arrays.items()})


if __name__ == "__main__":
    main()
