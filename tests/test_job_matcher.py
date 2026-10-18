import contextlib
import importlib
import io
import json
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


# Stub the setup module before importing the matcher so tests never rebuild the index.
setup_stub = types.ModuleType("rag_setup_assignment")
setup_stub.collection_filtered = None
setup_stub.embedding_model = None
sys.modules["rag_setup_assignment"] = setup_stub
job_matcher = importlib.import_module("job_matcher")


class FakeEmbeddingModel:
    """Provide deterministic vectors without loading the transformer model."""

    def encode(self, text):
        """Return a fixed embedding for any test query."""
        return np.array([0.1, 0.2, 0.3])


class FakeCollection:
    """Expose the collection methods needed by the matcher using test records."""

    def __init__(self, records, semantic_ids=None):
        """Store test chunks and optionally define their semantic ranking."""
        self.records = records
        self.semantic_ids = semantic_ids or [record["id"] for record in records]

    def get(self, include):
        """Return test chunk ids, documents, and metadata in Chroma's shape."""
        return {
            "ids": [record["id"] for record in self.records],
            "documents": [record["document"] for record in self.records],
            "metadatas": [record["metadata"] for record in self.records],
        }

    def query(self, query_embeddings, n_results):
        """Return the configured semantic order for a vector query."""
        return {"ids": [self.semantic_ids[:n_results]]}


def make_record(resume_id, source_file, name, years, skills, section, document):
    """Build one resume-chunk record with the metadata expected by the matcher."""
    return {
        "id": resume_id,
        "document": document,
        "metadata": {
            "source_file": source_file,
            "name": name,
            "experience_years": years,
            "skills": skills,
            "section": section,
        },
    }


class JobMatcherTests(unittest.TestCase):
    """Test matcher helpers, candidate filtering, output validation, and metrics."""

    def setUp(self):
        """Replace the embedding dependency with its deterministic test double."""
        self.original_collection = job_matcher.collection_filtered
        self.original_embedding_model = job_matcher.embedding_model
        job_matcher.embedding_model = FakeEmbeddingModel()

    def tearDown(self):
        """Restore the matcher globals after each test case."""
        job_matcher.collection_filtered = self.original_collection
        job_matcher.embedding_model = self.original_embedding_model

    def run_matcher(self, answers, records):
        """Run the interactive matcher with canned answers and capture its JSON."""
        job_matcher.collection_filtered = FakeCollection(records)
        output = io.StringIO()
        with patch("builtins.input", side_effect=answers), contextlib.redirect_stdout(output):
            job_matcher.main()
        return json.loads(output.getvalue())

    def test_tokenize_and_exact_match_boundaries(self):
        """Check technology tokenization and prevent substring skill matches."""
        self.assertIn("c++", job_matcher.tokenize("C++ developer"))
        self.assertTrue(job_matcher.exact_match("Java", "Java backend engineer"))
        self.assertFalse(job_matcher.exact_match("Java", "JavaScript engineer"))

    def test_parse_must_haves(self):
        """Check experience thresholds, skill extraction, and empty requirements."""
        self.assertEqual(
            job_matcher.parse_must_haves("5+ years Python; AWS; 3 years Java"),
            (5, ["Python", "AWS", "Java"]),
        )
        self.assertEqual(job_matcher.parse_must_haves(""), (0, []))

    def test_candidate_match_score_is_constrained(self):
        """Ensure candidate scores accept the inclusive range from 0 through 100."""
        candidate = job_matcher.CandidateMatch(
            candidate_name="Jane Doe",
            resume_path="data/resumes/jane_doe.txt",
            match_score=100,
            matched_skills=["Python"],
            relevant_sections=["Experience"],
            relevant_excerpts=["Experience: Python"],
            reasoning="Relevant Python experience.",
        )
        self.assertEqual(candidate.match_score, 100)
        with self.assertRaises(ValueError):
            job_matcher.CandidateMatch(
                candidate_name="Invalid",
                resume_path="invalid.txt",
                match_score=101,
                matched_skills=[],
                relevant_sections=[],
                relevant_excerpts=[],
                reasoning="",
            )

    def test_hybrid_match_filters_must_haves_and_reports_metrics(self):
        """Check hybrid candidate output, must-have exclusions, evidence, and metrics."""
        records = [
            make_record("jane-exp", "jane_doe.txt", "Jane Doe", 6, "Python, AWS", "Experience", "Python backend API development"),
            make_record("jane-skills", "jane_doe.txt", "Jane Doe", 6, "Python, AWS", "Skills", "Python, AWS, Docker"),
            make_record("sam-exp", "sam_lee.txt", "Sam Lee", 2, "Python, AWS", "Experience", "Python backend development"),
            make_record("pat-exp", "pat_roy.txt", "Pat Roy", 7, "Python", "Experience", "Python backend API development"),
        ]
        response = self.run_matcher(
            ["Python backend engineer with AWS", "Python, AWS", "5+ years Python; AWS"],
            records,
        )

        self.assertEqual(response["job_description"], "Python backend engineer with AWS")
        self.assertEqual([candidate["candidate_name"] for candidate in response["top_matches"]], ["Jane Doe"])
        candidate = response["top_matches"][0]
        self.assertGreaterEqual(candidate["match_score"], 0)
        self.assertLessEqual(candidate["match_score"], 100)
        self.assertEqual(set(candidate["matched_skills"]), {"Python", "AWS"})
        self.assertIn("Experience", candidate["relevant_sections"])
        self.assertIn("Skills", candidate["relevant_sections"])
        self.assertTrue(candidate["relevant_excerpts"])

        metrics = response["performance_metrics"]
        self.assertGreaterEqual(metrics["retrieval_latency_ms"], 0)
        self.assertEqual(metrics["retrieval_accuracy_at_k"], 1.0)
        self.assertEqual(metrics["relevant_candidates_at_k"], 1)
        self.assertEqual(metrics["evaluated_candidates_at_k"], 1)
        self.assertEqual(metrics["total_chunks"], 4)
        self.assertEqual(metrics["total_candidates"], 3)

    def test_accuracy_is_null_without_critical_skills(self):
        """Mark skill-hit relevance unavailable when no critical skills are given."""
        records = [
            make_record("jane-exp", "jane_doe.txt", "Jane Doe", 6, "Python", "Experience", "Python backend engineer"),
        ]
        response = self.run_matcher(["Python backend engineer", "", ""], records)
        self.assertIsNone(response["performance_metrics"]["retrieval_accuracy_at_k"])
        self.assertIsNone(response["performance_metrics"]["relevant_candidates_at_k"])
        self.assertIn("Unavailable", response["performance_metrics"]["accuracy_basis"])

    def test_empty_collection_raises_clear_error(self):
        """Raise a useful error when no resume chunks are available for retrieval."""
        job_matcher.collection_filtered = FakeCollection([])
        with patch("builtins.input", side_effect=["Python engineer", "", ""]):
            with self.assertRaisesRegex(RuntimeError, "collection is empty"):
                with contextlib.redirect_stderr(io.StringIO()):
                    job_matcher.main()


if __name__ == "__main__":
    unittest.main()
