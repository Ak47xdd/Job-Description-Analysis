from fastapi import Depends, HTTPException
from starlette.requests import Request
from fastapi import APIRouter
import traceback

from JobAnalyze.v1.pred_v1 import JobAnalyze_6k
from rate_limit import limiter
from helpers import _build_analysis, _SKILL_TO_CAT, SKILL_CATEGORIES
from schemas import ModelRequest
from auth import verify

router = APIRouter(tags=["items"])

MAX_JD_LENGTH = 30000

# Public response policy. A skill is "detected" when the model probability
# reaches this minimum. Required-vs-preferred is NOT determined by this
# threshold; explicit JD sections are authoritative in helpers.py.
DETECTION_MIN_SCORE = 0.15
REQUIRED_DEFINITION = "skills matched within the JD's Required/Qualifications section"


def _detected_categories(predicted: list[tuple[str, float]]) -> list[dict]:
    """Build categories using the same documented detection threshold."""
    buckets: dict[str, list[dict]] = {cat: [] for cat in SKILL_CATEGORIES}
    for skill, probability in predicted:
        if probability < DETECTION_MIN_SCORE:
            continue
        category = _SKILL_TO_CAT.get(skill)
        if not category:
            continue
        buckets[category].append({
            "name": skill.title(),
            "status": "found",
            "importance": int(probability * 100),
        })
    return [
        {"name": category, "skills": sorted(skills, key=lambda item: -item["importance"])}
        for category, skills in buckets.items()
        if skills
    ]


def _finalize_analysis(analysis: dict, predicted: list[tuple[str, float]]) -> dict:
    """Apply the public threshold/count contract to a built analysis."""
    detected = [(skill, probability) for skill, probability in predicted if probability >= DETECTION_MIN_SCORE]
    summary = analysis.get("summary") or {}
    required = summary.get("required") or []

    # requiredTechCount is deliberately derived from summary.required so the
    # API exposes one source of truth instead of maintaining two calculations.
    analysis["technicalSkillCount"] = len(detected)
    analysis["requiredTechCount"] = len(required)
    analysis["categories"] = _detected_categories(predicted)
    analysis["thresholds"] = {
        "detectionMinScore": DETECTION_MIN_SCORE,
        "requiredDefinition": REQUIRED_DEFINITION,
    }
    return analysis


@router.post(
    "/web_analyze",
    operation_id="web_analyze"
)
@limiter.limit("5/minute")
async def web_analyze(request: Request, data: ModelRequest) -> dict:
    if len(data.Job_Desc) > MAX_JD_LENGTH:
        raise HTTPException(status_code=413, detail="Job description is too large.")
    try:
        predicted = [
            (skill, float(score))
            for skill, score in JobAnalyze_6k(
                job_desc=data.Job_Desc,
                role=data.Role,
                job_type=data.Type,
            )
        ]
        analysis = _build_analysis(
            predicted=predicted,
            role=data.Role,
            job_type=data.Type,
            jd_text=data.Job_Desc,
        )

        # Keep the public /web_analyze response compatible with the
        # authenticated /JobAnalyze_6k contract. The frontend expects the
        # raw model predictions under `answer`; the richer analysis remains
        # available under `analysis`.
        return {
            "answer": predicted,
            "analysis": _finalize_analysis(analysis, predicted),
        }
    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Web Analyzer failed while building the analysis response.")


@router.post(
    "/JobAnalyze_6k",
    operation_id="analyze_job_description"
)
@limiter.limit("10/minute")
async def JobAnalyze_Pred(request: Request, data: ModelRequest, api_client: dict = Depends(verify)) -> dict:
    if len(data.Job_Desc) > MAX_JD_LENGTH:
        raise HTTPException(status_code=413, detail="Job description is too large.")
    predicted = [(skill, float(score)) for skill, score in JobAnalyze_6k(job_desc=data.Job_Desc, role=data.Role, job_type=data.Type)]
    analysis = _build_analysis(predicted=predicted, role=data.Role, job_type=data.Type, jd_text=data.Job_Desc)
    return {"answer": predicted, "analysis": _finalize_analysis(analysis, predicted)}
