from pydantic import BaseModel, field_validator, EmailStr

class SignUpRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

class SignInRequest(BaseModel):
    email: EmailStr
    password: str
    
class JobOpeningCreate(BaseModel):
    title:        str
    department:   str
    type:         str
    location:     str  = "Remote"
    description:  str
    requirements: list[str] = []
    tags:         list[str] = []

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
    title:        str
    summary:      str
    category:     str  = "Update"
    url:          str | None = None
    body:         str | None = None
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

class ModelRequest(BaseModel):
    Job_Desc: str
    Role:     str
    Type:     str

    @field_validator("Job_Desc")
    def jd_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Job_Desc cannot be empty")
        return v.strip()

    @field_validator("Role")
    def role_valid(cls, v):
        allowed = [
            "AI Engineer",
            "AI Developer",
            "Data Scientist",
            "ML Engineer",
            "MLOps Engineer",
            "Data Analyst",
        ]
        if v not in allowed:
            raise ValueError(f"Role must be one of {allowed}")
        return v

    @field_validator("Type")
    def type_valid(cls, v):
        allowed = ["Internship", "Junior", "Mid", "Senior"]
        if v not in allowed:
            raise ValueError(f"Type must be one of {allowed}")
        return v
