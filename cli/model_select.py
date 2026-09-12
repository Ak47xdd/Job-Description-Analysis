"""
model_select.py — inference routing for JobAnalyze model selection.
"""
from pathlib import Path
from dotenv import load_dotenv
from rich import print
import os
import requests
import sys

MODEL_6K = "JobAnalyze 6k"
MODEL_SBERT = "JobAnalyze v2 SBERT"


def _load_env() -> None:
    candidates = [
        Path.cwd() / ".env",
        Path.home() / ".jobselect" / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for path in candidates:
        if path.exists():
            load_dotenv(dotenv_path=path, override=False)
            return


def _resolve_config() -> tuple[str, str]:
    _load_env()
    return (
        os.getenv("JOBSELECT_API_URL", "").strip().rstrip("/"),
        os.getenv("JOBSELECT_API_KEY", "").strip(),
    )


def _resolve_key(env_key: str) -> str:
    try:
        from . import api_val
        if api_val.key:
            return api_val.key
    except ImportError:
        try:
            import api_val
            if api_val.key:
                return api_val.key
        except ImportError:
            pass
    return env_key


def _local_predict(jd: str, role: str, job_type: str, model: str):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if model == MODEL_SBERT:
        from api.JobAnalyze.v2.pred_v2 import JobAnalyze_v2_SBERT
        return JobAnalyze_v2_SBERT(jd, role=role, job_type=job_type)

    from api.JobAnalyze.v1.pred_v1 import JobAnalyze_6k
    return JobAnalyze_6k(jd, role=role, job_type=job_type)


def _call_api(jd: str, role: str, job_type: str, api_key: str, api_url: str, model: str):
    if model == MODEL_SBERT:
        endpoint = f"{api_url}/JobAnalyze_SBERT"
    else:
        endpoint = f"{api_url}/JobAnalyze_6k"

    headers = {"JobAnalyze_6k_Key": api_key}
    payload = {"Job_Desc": jd, "Role": role, "Type": job_type}

    response = requests.post(endpoint, json=payload, headers=headers, timeout=(10, 120))
    response.raise_for_status()
    data = response.json()
    return [(skill, float(score)) for skill, score in data["answer"]]


def predict(
    jd: str,
    role: str,
    job_type: str,
    model: str = MODEL_6K,
    force_local: bool = False,
) -> tuple[list[tuple[str, float]], str]:
    """Run the selected model through the API, falling back to the same local model."""
    if model not in (MODEL_6K, MODEL_SBERT):
        raise ValueError(f"Unsupported model: {model}")

    if force_local:
        return _local_predict(jd, role, job_type, model), "LOCAL"

    api_url, env_key = _resolve_config()
    api_key = _resolve_key(env_key)

    if api_url and api_key:
        try:
            return _call_api(jd, role, job_type, api_key, api_url, model), "API"
        except requests.exceptions.ConnectionError as e:
            print(f"[red][JobSelect] Connection error: {e} — falling back to LOCAL")
        except requests.exceptions.Timeout:
            print("[red][JobSelect] Request timed out/Under Maintenance — falling back to LOCAL")
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            print(f"[red][JobSelect] API error {code} — falling back to LOCAL")
        except Exception as e:
            print(f"[red][JobSelect] Unexpected error: {e} — falling back to LOCAL")

    return _local_predict(jd, role, job_type, model), "LOCAL"
