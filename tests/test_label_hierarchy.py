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
