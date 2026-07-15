"""
model_select.py — inference routing (API first, local fallback).
Never prompts the user. Key is resolved lazily at predict() call time.
"""

from pathlib import Path
from dotenv import load_dotenv
from rich import print
import os
import requests
import sys
 
 
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
    url = os.getenv("JOBSELECT_API_URL", "").strip().rstrip("/")
    key = os.getenv("JOBSELECT_API_KEY", "").strip()
    return url, key
 
 
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
 
 
def _local_predict(jd: str, role: str, job_type: str):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from model.pred import JobAnalyze_6k as _job_analyze
    return _job_analyze(jd, role=role, job_type=job_type)
 
 
def _call_api(
    jd: str, role: str, job_type: str, api_key: str, api_url: str
) -> list[tuple[str, float]]:
    endpoint = f"{api_url}/JobAnalyze_6k"
    headers  = {"JobAnalyze_6k_Key": api_key}
    payload  = {"Job_Desc": jd, "Role": role, "Type": job_type}
 
    response = requests.post(endpoint, json=payload, headers=headers, timeout=(10, 120))
    response.raise_for_status()
 
    data = response.json()
    return [(skill, float(score)) for skill, score in data["answer"]]
 
 
def predict(jd: str, role: str, job_type: str, force_local: bool = False) -> tuple[list[tuple[str, float]], str]:
    if force_local:
        results = _local_predict(jd, role=role, job_type=job_type)
        return results, "LOCAL"

    api_url, env_key = _resolve_config()
    api_key = _resolve_key(env_key)

    if api_url and api_key:
        try:
            results = _call_api(jd, role, job_type, api_key, api_url)
            return results, "API"
        except requests.exceptions.ConnectionError as e:
            print(f"[red][JobSelect] Connection error: {e} — falling back to LOCAL")
        except requests.exceptions.Timeout:
            print("[red][JobSelect] Request timed out/Under Maintainance — falling back to LOCAL")
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            print(f"[red][JobSelect] API error {code} — falling back to LOCAL")
        except Exception as e:
            print(f"[red][JobSelect] Unexpected error: {e} — falling back to LOCAL")
 
    results = _local_predict(jd, role=role, job_type=job_type)
    return results, "LOCAL"