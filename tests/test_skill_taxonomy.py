from model.prep.sym_map import CANONICAL_LABEL_MAP, TAXONOMY_VERSION, canonicalize_skill_text


def test_taxonomy_version_is_explicit():
    assert TAXONOMY_VERSION == "v3-canonical-buckets"


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
