from fastapi import Depends, HTTPException
from starlette.requests import Request
from fastapi import APIRouter
import traceback

from JobAnalyze.v1.pred_v1 import JobAnalyze_6k
from rate_limit import limiter
from helpers import _build_analysis, _SKILL_TO_CAT, SKILL_CATEGORIES, _get_compatibility
from section_skills import classify_required_preferred
from schemas import ModelRequest
from auth import verify

router = APIRouter(tags=["items"])

MAX_JD_LENGTH = 30000

# Public response policy. A skill is "detected" when the model probability
# reaches this minimum. Required-vs-preferred is determined by explicit JD
# sections by section_skills.py.
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


def _finalize_analysis(analysis: dict, predicted: list[tuple[str, float]], jd_text: str) -> dict:
    """Apply the public threshold/count contract and section-aware split."""
    detected = [(skill, probability) for skill, probability in predicted if probability >= DETECTION_MIN_SCORE]
    required_pairs, preferred_pairs = classify_required_preferred(predicted, jd_text)
    summary = analysis.get("summary") or {}

    # Replace the heuristic helper split with the authoritative section-aware
    # classification. A skill mentioned in both sections belongs to required.
    summary["required"] = [skill.title() for skill, _ in required_pairs[:5]]
    summary["preferred"] = [skill.title() for skill, _ in preferred_pairs[:5]]
    analysis["summary"] = summary

    compatibility, compatibility_label = _get_compatibility(required_pairs)
    analysis["compatibility"] = compatibility
    analysis["compatibilityLabel"] = compatibility_label
    analysis["technicalSkillCount"] = len(detected)
    analysis["requiredTechCount"] = len(required_pairs)
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

        return {
            "answer": predicted,
            "analysis": _finalize_analysis(analysis, predicted, data.Job_Desc),
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
    return {"answer": predicted, "analysis": _finalize_analysis(analysis, predicted, data.Job_Desc)}
