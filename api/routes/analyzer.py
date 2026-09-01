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
REQUIRED_SELECTION_METHOD = "explicit_jd_section; if no Required section exists, confidence>=0.60 fallback; summary.required is display-limited to the top 5 required skills by model confidence"
PREFERRED_SELECTION_METHOD = "explicit_jd_section; summary.preferred is display-limited to the top 5 preferred skills by model confidence"


def _canonical_skill_name(skill: str) -> str:
    return skill.strip().lower()


def _detected_categories(predicted: list[tuple[str, float]]) -> list[dict]:
    buckets: dict[str, list[dict]] = {cat: [] for cat in SKILL_CATEGORIES}
    for skill, probability in predicted:
        if probability < DETECTION_MIN_SCORE:
            continue
        canonical = _canonical_skill_name(skill)
        category = _SKILL_TO_CAT.get(canonical)
        if not category:
            continue
        buckets[category].append({"name": canonical, "displayName": skill.strip(), "status": "found", "importance": int(probability * 100)})
    return [{"name": category, "skills": sorted(skills, key=lambda item: -item["importance"])} for category, skills in buckets.items() if skills]


def _score_breakdown(detected, required, preferred, compatibility) -> dict:
    desired = required + preferred
    desired_count = len(desired)
    weighted_average = sum(score for _, score in desired) / desired_count if desired_count else 0.0
    return {
        "compatibilityScore": compatibility,
        "detectedSkillCount": len(detected),
        "desiredSkillCount": desired_count,
        "skillsPresentInDesiredSet": min(len(detected), desired_count),
        "requiredSkillCount": len(required),
        "preferredSkillCount": len(preferred),
        "weightedAverageImportance": round(weighted_average, 4),
        "weightedAverageImportancePercent": round(weighted_average * 100, 1),
        "requiredCoveragePercent": round(len(required) / len(required) * 100, 1) if required else 0.0,
        "preferredCoveragePercent": round(len(preferred) / len(preferred) * 100, 1) if preferred else 0.0,
        "selectionMethod": "section-aware_required_preferred; compatibility uses required skills returned by the section classifier",
    }


def _finalize_analysis(analysis, predicted, jd_text):
    detected = [(skill, probability) for skill, probability in predicted if probability >= DETECTION_MIN_SCORE]
    required_pairs, preferred_pairs = classify_required_preferred(predicted, jd_text)
    summary = analysis.get("summary") or {}
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
    analysis["scoreBreakdown"] = _score_breakdown(detected, required_pairs, preferred_pairs, compatibility)
    # `importance` used to duplicate the same skill/confidence data already
    # represented by `categories[].skills[].importance`. Keep the canonical
    # category representation and omit the redundant top-level array.
    analysis.pop("importance", None)
    analysis["thresholds"] = {"detectionMinScore": DETECTION_MIN_SCORE, "requiredDefinition": REQUIRED_DEFINITION, "requiredSelectionMethod": REQUIRED_SELECTION_METHOD, "preferredSelectionMethod": PREFERRED_SELECTION_METHOD, "requiredTotalCount": len(required_pairs), "preferredTotalCount": len(preferred_pairs), "summaryDisplayLimit": 5, "skillNameKey": "lowercase canonical skill identifier"}
    return analysis


@router.post(
    "/web_analyze",
    operation_id="web_analyze",
    summary="Analyze a raw job description and extract skills, requirements, compatibility, and required/preferred sections.",
    description=("Analyze raw job-description text with the JobAnalyze 6k skill classifier. "
                 "Job_Desc must contain the raw JD text, not a summary or pre-extracted skill list. "
                 "Role is the target job role and accepts the six preset roles or a custom role. "
                 "Type is the caller-supplied seniority context and must be Internship, Junior, or Senior. "
                 "Type is contextual input and is not silently inferred or corrected from the JD. "
                 "If Type conflicts with explicit seniority language in the JD, the supplied Type remains the classifier context; the raw JD remains the source for extracted requirements."))
@limiter.limit("5/minute")
async def web_analyze(request: Request, data: ModelRequest) -> dict:
    if len(data.Job_Desc) > MAX_JD_LENGTH:
        raise HTTPException(status_code=413, detail="Job description is too large.")
    try:
        predicted = [(_canonical_skill_name(skill), float(score)) for skill, score in JobAnalyze_6k(job_desc=data.Job_Desc, role=data.Role, job_type=data.Type)]
        analysis = _build_analysis(predicted=predicted, role=data.Role, job_type=data.Type, jd_text=data.Job_Desc)
        return {"answer": predicted, "analysis": _finalize_analysis(analysis, predicted, data.Job_Desc)}
    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Web Analyzer failed while building the analysis response.")


@router.post(
    "/JobAnalyze_6k",
    operation_id="analyze_job_description",
    summary="Analyze a raw job description with the JobAnalyze 6k skill classifier.",
    description=("Analyze raw job-description text. Job_Desc should be the complete raw JD text. "
                 "Role identifies the target role and accepts the six preset roles or a custom role. "
                 "Type is the caller-supplied seniority context: Internship, Junior, or Senior. "
                 "Type is not automatically inferred or corrected from the JD; if it conflicts with the JD's stated seniority, the supplied Type remains the classifier context."))
@limiter.limit("10/minute")
async def JobAnalyze_Pred(request: Request, data: ModelRequest, api_client: dict = Depends(verify)) -> dict:
    if len(data.Job_Desc) > MAX_JD_LENGTH:
        raise HTTPException(status_code=413, detail="Job description is too large.")
    predicted = [(_canonical_skill_name(skill), float(score)) for skill, score in JobAnalyze_6k(job_desc=data.Job_Desc, role=data.Role, job_type=data.Type)]
    analysis = _build_analysis(predicted=predicted, role=data.Role, job_type=data.Type, jd_text=data.Job_Desc)
    return {"answer": predicted, "analysis": _finalize_analysis(analysis, predicted, data.Job_Desc)}