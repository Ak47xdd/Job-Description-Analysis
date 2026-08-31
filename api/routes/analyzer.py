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

DETECTION_MIN_SCORE = 0.15
REQUIRED_DEFINITION = "skills matched within the JD's Required/Qualifications section"
REQUIRED_SELECTION_METHOD = (
    "explicit_jd_section; if no Required section exists, confidence>=0.60 fallback; "
    "summary.required is display-limited to the top 5 required skills by model confidence"
)
PREFERRED_SELECTION_METHOD = (
    "explicit_jd_section; summary.preferred is display-limited to the top 5 preferred skills by model confidence"
)


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


def _score_breakdown(
    detected: list[tuple[str, float]],
    required: list[tuple[str, float]],
    preferred: list[tuple[str, float]],
    compatibility: int,
) -> dict:
    """Expose the inputs to compatibility scoring without pretending it is a hiring probability."""
    desired = required + preferred
    desired_count = len(desired)
    present_count = len(detected)
    required_present = len(required)
    preferred_present = len(preferred)
    weighted_average = (
        sum(score for _, score in desired) / desired_count if desired_count else 0.0
    )
    return {
        "compatibilityScore": compatibility,
        "detectedSkillCount": present_count,
        "desiredSkillCount": desired_count,
        "skillsPresentInDesiredSet": min(present_count, desired_count),
        "requiredSkillCount": required_present,
        "preferredSkillCount": preferred_present,
        "weightedAverageImportance": round(weighted_average, 4),
        "weightedAverageImportancePercent": round(weighted_average * 100, 1),
        "requiredCoveragePercent": round(required_present / len(required) * 100, 1) if required else 0.0,
        "preferredCoveragePercent": round(preferred_present / len(preferred) * 100, 1) if preferred else 0.0,
        "selectionMethod": "section-aware_required_preferred; compatibility uses required skills returned by the section classifier",
    }


def _finalize_analysis(analysis: dict, predicted: list[tuple[str, float]], jd_text: str) -> dict:
    """Apply the public threshold/count and documented selection contracts."""
    detected = [(skill, probability) for skill, probability in predicted if probability >= DETECTION_MIN_SCORE]
    required_pairs, preferred_pairs = classify_required_preferred(predicted, jd_text)
    summary = analysis.get("summary") or {}

    # Section classification determines membership. The UI summary is capped
    # at five entries, selected by model confidence within each section.
    summary["required"] = [skill.title() for skill, _ in required_pairs[:5]]
    summary["preferred"] = [skill.title() for skill, _ in preferred_pairs[:5]]
    analysis["summary"] = summary

    compatibility, compatibility_label = _get_compatibility(required_pairs)
    analysis["compatibility"] = compatibility
    analysis["compatibilityLabel"] = compatibility_label
    analysis["technicalSkillCount"] = len(detected)
    analysis["requiredTechCount"] = len(required_pairs)
    analysis["preferredTechCount"] = len(preferred_pairs)
    analysis["categories"] = _detected_categories(predicted)
    analysis["scoreBreakdown"] = _score_breakdown(
        detected, required_pairs, preferred_pairs, compatibility
    )
    analysis["thresholds"] = {
        "detectionMinScore": DETECTION_MIN_SCORE,
        "requiredDefinition": REQUIRED_DEFINITION,
        "requiredSelectionMethod": REQUIRED_SELECTION_METHOD,
        "preferredSelectionMethod": PREFERRED_SELECTION_METHOD,
        "requiredTotalCount": len(required_pairs),
        "preferredTotalCount": len(preferred_pairs),
        "summaryDisplayLimit": 5,
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
