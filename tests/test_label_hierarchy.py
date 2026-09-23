from model.prep.data_prep import normalizer
from model.prep.sym_map import SYNONYM_MAP


def test_label_normalizer_consolidates_aliases():
    labels = normalizer(
        "AI tools, Generative AI, Large Language Models, LLM, Full Stack, "
        "Full-Stack, Backend Services, Backend Engineering, Python"
    )

    assert "ai" in labels
    assert "genai" in labels
    assert "llms" in labels
    assert "full-stack" in labels
    assert "backend" in labels
    assert "python" in labels

    assert "ai tools" not in labels
    assert "generative ai" not in labels
    assert "large language models" not in labels
    assert "llm" not in labels
    assert "full stack" not in labels
    assert "backend services" not in labels
    assert "backend engineering" not in labels


def test_input_synonyms_match_canonical_labels():
    assert SYNONYM_MAP["generative ai"] == "genai"
    assert SYNONYM_MAP["large language models"] == "llms"
    assert SYNONYM_MAP["llm"] == "llms"
    assert SYNONYM_MAP["full stack"] == "full-stack"
    assert SYNONYM_MAP["backend services"] == "backend"
    assert SYNONYM_MAP["ai tools"] == "ai"


def test_cloud_provider_aliases_share_one_canonical_label():
    labels = normalizer(
        "AWS, Azure, Microsoft Azure, Amazon Web Services, GCP, "
        "Google Cloud, AWS/Azure"
    )

    assert labels == ["aws/azure"]


def test_cloud_provider_text_synonyms_use_one_canonical_bucket():
    from model.prep.data_prep import apply_synonyms

    text = apply_synonyms(
        "AWS, Azure, Microsoft Azure, Amazon Web Services, GCP, Google Cloud"
    )

    assert text.count("aws/azure") == 6
    assert "azure" not in text.replace("aws/azure", "")
    assert "aws" not in text.replace("aws/azure", "")
