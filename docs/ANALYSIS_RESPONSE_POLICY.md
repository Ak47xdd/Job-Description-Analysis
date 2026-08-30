# JobAnalyze Analysis Response Policy

## Skill counts

`technicalSkillCount` is the number of model-predicted skills whose probability is at least `DETECTION_MIN_SCORE` (`0.15`). The same threshold is used to populate the response `categories[].skills` entries.

`requiredTechCount` is **not** a second probability threshold. It is exactly `len(summary.required)`.

## Required vs preferred

Required/preferred classification is section-aware and is implemented in `api/helpers.py`:

- Skills explicitly matched in a Required/Qualifications section are `required`.
- Skills explicitly matched in a Preferred/Bonus/Nice-to-have section are `preferred`.
- If a skill is mentioned in both, Required wins and the response never duplicates it.
- If there is no Preferred section, `preferred` is empty.
- For JDs without explicit required/preferred sections, the existing conservative confidence fallback is used.

## Response metadata

Analysis responses expose:

```json
"thresholds": {
  "detectionMinScore": 0.15,
  "requiredDefinition": "skills matched within the JD's Required/Qualifications section"
}
```

This makes the count semantics reproducible for API and MCP callers.

## Consistency contract

The following relationship must always hold:

```text
requiredTechCount == len(summary.required)
```

The API route enforces this relationship as a final response-layer safety check.
