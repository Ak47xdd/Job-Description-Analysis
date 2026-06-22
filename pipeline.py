from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parent


@dataclass
class StepResult:
    name: str
    path: Path
    ok: bool
    returncode: int
    elapsed_s: float
    error: Optional[str] = None


def _now_s() -> float:
    return time.time()


def _log(msg: str) -> None:
    print(msg, flush=True)


def run_command(cmd: List[str], *, cwd: Path, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def execute_notebook(notebook_path: Path) -> subprocess.CompletedProcess:
    out_dir = REPO_ROOT / ".pipeline_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    executed_out = out_dir / f"{notebook_path.stem}__executed.ipynb"

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--output",
        str(executed_out),
        str(notebook_path),
    ]

    return run_command(cmd, cwd=REPO_ROOT)


def execute_python(script_path: Path) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(script_path)]
    return run_command(cmd, cwd=REPO_ROOT)


def run_step(name: str, path_str: str) -> StepResult:
    path = (REPO_ROOT / path_str).resolve()
    start = _now_s()

    if not path.exists():
        return StepResult(
            name=name,
            path=path,
            ok=False,
            returncode=1,
            elapsed_s=_now_s() - start,
            error=f"File not found: {path}",
        )

    try:
        _log(f"\n[PIPELINE] Starting step: {name} -> {path.relative_to(REPO_ROOT)}")
        if path.suffix == ".ipynb":
            proc = execute_notebook(path)
        else:
            proc = execute_python(path)

        elapsed = _now_s() - start
        ok = proc.returncode == 0

        if not ok:
            # Keep stderr; stdout can be large.
            _log(f"[PIPELINE] Step FAILED: {name} (rc={proc.returncode}) in {elapsed:.1f}s")
            err_snippet = proc.stderr.strip()[-4000:]
            _log(f"[PIPELINE] stderr (last 4000 chars):\n{err_snippet}")
        else:
            _log(f"[PIPELINE] Step OK: {name} in {elapsed:.1f}s")

        return StepResult(
            name=name,
            path=path,
            ok=ok,
            returncode=proc.returncode,
            elapsed_s=elapsed,
            error=(proc.stderr.strip() if not ok else None),
        )

    except Exception as e:  # noqa: BLE001
        elapsed = _now_s() - start
        _log(f"[PIPELINE] Step EXCEPTION: {name} -> {e}")
        return StepResult(
            name=name,
            path=path,
            ok=False,
            returncode=1,
            elapsed_s=elapsed,
            error=str(e),
        )


def main() -> int:
    steps = [
        ("01_EDA", "notebooks/01_EDA.ipynb"),
        ("02_Data_Engineering", "notebooks/02_Data_Engineering.ipynb"),
        ("data_prep", "model/prep/data_prep.py"),
        ("train_model", "model/model.py"),
        ("evaluate", "model/eval.py"),
    ]

    results: List[StepResult] = []

    # Ensure notebooks execute from repo root.
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    for name, rel_path in steps:
        # Run step, continue on failure.
        step_res = run_step(name, rel_path)
        results.append(step_res)

        # Optional: if a step fails, continue anyway (as requested).
        # If you want to stop on first failure, change this behavior.

    # Summarize
    ok_steps = [r for r in results if r.ok]
    failed_steps = [r for r in results if not r.ok]

    summary = {
        "ok": len(ok_steps),
        "failed": len(failed_steps),
        "results": [
            {
                "name": r.name,
                "path": str(r.path.relative_to(REPO_ROOT)) if r.path.exists() else str(r.path),
                "ok": r.ok,
                "returncode": r.returncode,
                "elapsed_s": round(r.elapsed_s, 3),
                "error": (r.error[:2000] if r.error else None),
            }
            for r in results
        ],
    }

    out_file = REPO_ROOT / ".pipeline_runs" / f"pipeline_summary_{int(time.time())}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _log("\n[PIPELINE] DONE")
    _log(f"[PIPELINE] OK steps: {len(ok_steps)}/{len(results)}")
    if failed_steps:
        _log(f"[PIPELINE] Failed steps: {[r.name for r in failed_steps]}")
        _log(f"[PIPELINE] Summary written to: {out_file}")
        # Return non-zero because something failed.
        return 1

    _log(f"[PIPELINE] Summary written to: {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

