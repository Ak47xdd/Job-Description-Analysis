# Role and Seniority Support

## Supported roles

The API accepts these role values:

- AI Engineer
- AI Developer
- Data Scientist
- ML Engineer
- MLOps Engineer
- Data Analyst

The underlying JobAnalyze 6k model is a general technical-skill classifier whose current implementation accepts the role as additional text context. These added role values remove an API-validation ceiling; they do **not** claim that the model was separately trained or validated for each role. Results for newly accepted roles should therefore be treated as supported inference rather than role-specific benchmarked performance.

## Supported seniority types

The API intentionally keeps the existing seniority contract:

- Internship
- Junior
- Senior

## Validation behavior

`Role` and `Type` are validated by `ModelRequest` before either analyzer endpoint runs. Invalid values return a Pydantic validation error listing the supported values.

The role/type values are context inputs, not proof that the submitted JD belongs to that role or seniority. The JD text remains the source for extracted title, responsibilities, required/preferred skills, and other content-grounded fields.
