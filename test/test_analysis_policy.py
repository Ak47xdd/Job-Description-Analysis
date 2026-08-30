from api.routes.analyzer import DETECTION_MIN_SCORE, _finalize_analysis


def test_required_count_matches_summary_required():
    predicted = [("python", 0.95), ("ml", 0.80), ("sql", 0.20), ("docker", 0.05)]
    analysis = {
        "summary": {"required": ["Python", "Ml"]},
        "technicalSkillCount": 999,
        "requiredTechCount": 999,
        "categories": [],
    }

    result = _finalize_analysis(analysis, predicted)

    assert result["technicalSkillCount"] == 3
    assert result["requiredTechCount"] == len(result["summary"]["required"])
    assert result["thresholds"]["detectionMinScore"] == DETECTION_MIN_SCORE
    assert result["thresholds"]["requiredDefinition"] == "skills matched within the JD's Required/Qualifications section"


def test_detection_threshold_is_inclusive():
    predicted = [("python", DETECTION_MIN_SCORE), ("sql", DETECTION_MIN_SCORE - 0.001)]
    analysis = {"summary": {"required": []}}

    result = _finalize_analysis(analysis, predicted)

    assert result["technicalSkillCount"] == 1
    assert result["categories"][0]["skills"][0]["name"] == "Python"
