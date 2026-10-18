import unittest

from model.prep.embedding_text import (
    MAX_SEQ_LENGTH,
    PREPROCESSING_VERSION,
    prepare_job_description,
)


class TestEmbeddingTextPreprocessing(unittest.TestCase):
    def test_removes_benefits_section_and_keeps_technical_sections(self):
        jd = """Junior AI Engineer
Responsibilities:
Build Python and NLP applications.
Requirements:
Experience with PyTorch and SQL.
Benefits:
Competitive salary and health insurance.
Free snacks and unlimited vacation.
Equal Opportunity Employer"""
        cleaned = prepare_job_description(jd).lower()
        self.assertIn("python", cleaned)
        self.assertIn("nlp", cleaned)
        self.assertIn("pytorch", cleaned)
        self.assertIn("sql", cleaned)
        self.assertNotIn("unlimited vacation", cleaned)
        self.assertNotIn("competitive salary", cleaned)

    def test_retains_technical_line_that_mentions_benefits(self):
        jd = (
            "Develop production APIs with Python, FastAPI, PostgreSQL, "
            "and machine learning systems. We offer competitive compensation."
        )
        cleaned = prepare_job_description(jd)
        self.assertIn("FastAPI", cleaned)
        self.assertIn("PostgreSQL", cleaned)
        self.assertIn("machine learning", cleaned)

    def test_normalizes_html_and_whitespace(self):
        self.assertEqual(
            prepare_job_description("<p>Python</p>\r\n\r\n  SQL  "),
            "Python\n\nSQL",
        )

    def test_preprocessing_contract_is_versioned_and_bounded(self):
        self.assertEqual(MAX_SEQ_LENGTH, 128)
        self.assertTrue(PREPROCESSING_VERSION)


if __name__ == "__main__":
    unittest.main()
