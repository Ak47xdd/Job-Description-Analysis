from model.prep.sym_map import CANONICAL_LABEL_MAP, TAXONOMY_VERSION, canonicalize_skill_text


def test_taxonomy_version_is_explicit():
    assert TAXONOMY_VERSION == "v4-parent-buckets"


def test_aliases_share_canonical_buckets():
    assert CANONICAL_LABEL_MAP["azure"] == "aws/azure"
    assert CANONICAL_LABEL_MAP["postgresql"] == "sql"
    assert CANONICAL_LABEL_MAP["react.js"] == "react"
    assert CANONICAL_LABEL_MAP["react native"] == "react"
    assert CANONICAL_LABEL_MAP["github actions"] == "ci/cd"
    assert CANONICAL_LABEL_MAP["gitlab ci"] == "ci/cd"
    assert CANONICAL_LABEL_MAP["helm"] == "kubernetes"
    assert CANONICAL_LABEL_MAP["eks"] == "kubernetes"
    assert CANONICAL_LABEL_MAP["gemini"] == "genai"
    assert CANONICAL_LABEL_MAP["autogen"] == "agents"


def test_text_canonicalization_prevents_duplicate_skill_terms():
    text = canonicalize_skill_text(
        "Built REST APIs with React.js, PostgreSQL, Azure, Helm and GitHub Actions."
    )
    assert "apis" in text
    assert "react.js" not in text
    assert "postgresql" not in text
    assert "azure" not in text
    assert "helm" not in text
    assert "github actions" not in text
    assert "aws/azure" in text
    assert "kubernetes" in text
    assert "ci/cd" in text


def test_nlp_full_phrase_canonicalizes_to_first_class_label():
    text = canonicalize_skill_text(
        "Required: Natural Language Processing (NLP) and Generative AI."
    )
    assert "nlp" in text
    assert "natural language processing" not in text



def test_dotnet_framework_variants_collapse_to_parent_label():
    from model.prep.data_prep import normalizer

    labels = normalizer(".NET, .NET Core, ASP.NET Core, ASP.NET, .NET Framework")
    assert labels == [".net"]


def test_ai_ml_variants_collapse_to_parent_label():
    from model.prep.data_prep import normalizer

    labels = normalizer("AI, AI/ML, machine learning, ML, artificial intelligence")
    assert labels == ["ai/ml"]


def test_ai_ml_text_canonicalization_uses_parent_bucket():
    text = canonicalize_skill_text(
        "Experience in AI, machine learning, ASP.NET Core and .NET Core."
    )
    assert "ai/ml" in text
    assert "machine learning" not in text
    assert "asp.net core" not in text
    assert ".net core" not in text
    assert ".net" in text


def test_agent_frameworks_remain_distinct_labels():
    # These are different tools, not synonyms for the generic agent category.
    assert "langchain" not in CANONICAL_LABEL_MAP
    assert "langgraph" not in CANONICAL_LABEL_MAP
