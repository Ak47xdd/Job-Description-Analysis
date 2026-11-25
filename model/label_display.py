"""Human-friendly display formatting for canonical skill labels.

Model keys remain lowercase and stable; this module only formats labels shown
to users. Never use display labels for vocabulary lookup or artifact hashing.
"""
from __future__ import annotations

import re

_WORD_OVERRIDES = {
    "ai": "AI",
    "ml": "ML",
    "nlp": "NLP",
    "llm": "LLM",
    "llms": "LLMs",
    "api": "API",
    "apis": "APIs",
    "rest": "REST",
    "sql": "SQL",
    "nosql": "NoSQL",
    "aws": "AWS",
    "gcp": "GCP",
    "mcp": "MCP",
    "ci": "CI",
    "cd": "CD",
    "etl": "ETL",
    "elt": "ELT",
    "seo": "SEO",
    "crm": "CRM",
    "iam": "IAM",
    "oauth": "OAuth",
    "ui": "UI",
    "ux": "UX",
    "db": "DB",
    "ide": "IDE",
    "sdk": "SDK",
    "html": "HTML",
    "css": "CSS",
    "json": "JSON",
    "xml": "XML",
    "aws/azure": "AWS/Azure",
    "ai/ml": "AI/ML",
    "ci/cd": "CI/CD",
    "ui/ux": "UI/UX",
    "genai": "GenAI",
    "vectordb": "Vector DB",
    "c#": "C#",
    "c++": "C++",
    ".net": ".NET",
    "node.js": "Node.js",
    "next.js": "Next.js",
    "power bi": "Power BI",
    "github": "GitHub",
    "git": "Git",
    "github actions": "GitHub Actions",
    "google cloud": "Google Cloud",
    "google analytics": "Google Analytics",
    "microsoft office": "Microsoft Office",
    "microsoft 365": "Microsoft 365",
    "postgresql": "PostgreSQL",
    "mongodb": "MongoDB",
    "mysql": "MySQL",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "python": "Python",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "fastapi": "FastAPI",
    "opencv": "OpenCV",
    "powerpoint": "PowerPoint",
    "excel": "Excel",
    "wordpress": "WordPress",
    "kubernetes": "Kubernetes",
    "docker": "Docker",
    "claude code": "Claude Code",
    "claude": "Claude",
    "langchain": "LangChain",
    "langgraph": "LangGraph",
    "llamaindex": "LlamaIndex",
    "openai": "OpenAI",
    "hugging face": "Hugging Face",
}


def display_label(label: str) -> str:
    """Return a readable title-cased label while preserving common acronyms."""
    value = re.sub(r"\\s+", " ", str(label or "").strip())
    if not value:
        return value
    exact = _WORD_OVERRIDES.get(value.casefold())
    if exact:
        return exact

    def format_token(token: str) -> str:
        return _WORD_OVERRIDES.get(token.casefold(), token[:1].upper() + token[1:].lower())

    # Keep separators while formatting each human-readable word.
    return re.sub(r"[^\\s/+-]+", lambda match: format_token(match.group(0)), value)
