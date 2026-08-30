from enum import Enum

from pydantic import BaseModel, field_validator, EmailStr, Field


class SignUpRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class SignInRequest(BaseModel):
    email: EmailStr
    password: str


class JobOpeningCreate(BaseModel):
    title: str
    department: str
    type: str
    location: str = "Remote"
    description: str
    requirements: list[str] = []
    tags: list[str] = []

    @field_validator("department")
    def dept_valid(cls, v):
        allowed = ["Engineering", "Research", "Design", "Operations"]
        if v not in allowed:
            raise ValueError(f"department must be one of {allowed}")
        return v

    @field_validator("type")
    def type_valid(cls, v):
        allowed = ["Full-time", "Part-time", "Contract", "Internship"]
        if v not in allowed:
            raise ValueError(f"type must be one of {allowed}")
        return v


class NewsItemCreate(BaseModel):
    title: str
    summary: str
    category: str = "Update"
    url: str | None = None
    body: str | None = None
    is_published: bool = True

    @field_validator("title")
    def title_length(cls, v):
        if len(v) > 120:
            raise ValueError("title must be 120 characters or fewer")
        return v.strip()

    @field_validator("summary")
    def summary_length(cls, v):
        if len(v) > 280:
            raise ValueError("summary must be 280 characters or fewer")
        return v.strip()

    @field_validator("category")
    def category_valid(cls, v):
        allowed = ["Release", "Update", "Research", "Community"]
        if v not in allowed:
            raise ValueError(f"category must be one of {allowed}")
        return v


class RoleEnum(str, Enum):
    AI_ENGINEER = "AI Engineer"
    AI_DEVELOPER = "AI Developer"
    DATA_SCIENTIST = "Data Scientist"
    ML_ENGINEER = "ML Engineer"
    MLOPS_ENGINEER = "MLOps Engineer"
    DATA_ANALYST = "Data Analyst"


class TypeEnum(str, Enum):
    INTERNSHIP = "Internship"
    JUNIOR = "Junior"
    SENIOR = "Senior"


class ModelRequest(BaseModel):
    Job_Desc: str
    # Preset roles are documented here, but the analyzer also accepts a user-defined role.
    # Keeping this as str allows the TUI's editable role field to work with custom roles.
    Role: str = Field(
        description="Job role. Presets: AI Engineer, AI Developer, Data Scientist, ML Engineer, MLOps Engineer, Data Analyst. Custom roles are also accepted."
    )
    Type: TypeEnum

    @field_validator("Job_Desc")
    def jd_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Job_Desc cannot be empty")
        return v.strip()

    @field_validator("Role")
    def role_not_empty(cls, v):
        value = v.strip()
        if not value:
            raise ValueError("Role cannot be empty")
        if len(value) > 100:
            raise ValueError("Role must be 100 characters or fewer")
        return value
