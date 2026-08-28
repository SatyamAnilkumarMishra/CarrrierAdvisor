import json
import unittest
from types import SimpleNamespace

from career_tools import (
    analyze_resume,
    analyze_skill_gap,
    generate_roadmap,
    search_jobs,
)
from errors import JobSearchError, LLMError
from llm_providers import LLMProvider


def _settings(**overrides):
    defaults = dict(
        has_job_search_api=False,
        job_search_default_limit=10,
        job_search_country="us",
        adzuna_app_id=None,
        adzuna_app_key=None,
        llm_request_timeout_seconds=30,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class FakeLLMProvider(LLMProvider):
    """Returns a canned JSON string regardless of input."""

    def __init__(self, response: str):
        self.response = response
        self.last_prompt = None

    def generate(self, messages, *, system_instruction=""):
        self.last_prompt = messages[-1].content
        return self.response

    def decide_needs_retrieval(self, query):
        return False


class TestSearchJobsBundled(unittest.TestCase):
    def test_role_match_returns_results(self):
        settings = _settings()
        results = search_jobs("Data Analyst", settings)
        self.assertTrue(any("Data Analyst" in job.title for job in results))
        self.assertTrue(all(job.source == "bundled" for job in results))

    def test_empty_role_raises(self):
        with self.assertRaises(JobSearchError):
            search_jobs("   ", _settings())

    def test_location_filter(self):
        settings = _settings()
        results = search_jobs("Engineer", settings, location="Remote")
        self.assertTrue(all(job.location.lower() == "remote" for job in results))

    def test_experience_level_filter(self):
        settings = _settings()
        results = search_jobs("Analyst", settings, experience_level="entry")
        self.assertTrue(all(job.experience_level == "entry" for job in results))

    def test_no_match_returns_empty_list(self):
        settings = _settings()
        results = search_jobs("Underwater Basket Weaver", settings, location="Nowhere")
        self.assertEqual(results, [])

    def test_limit_is_respected(self):
        settings = _settings()
        results = search_jobs("Engineer", settings, limit=1)
        self.assertLessEqual(len(results), 1)


class TestSkillGapAnalyzer(unittest.TestCase):
    def test_parses_well_formed_json(self):
        payload = {
            "matched_skills": ["Python"],
            "missing_skills": ["SQL"],
            "partially_met_skills": ["Excel"],
            "overall_readiness": "MEDIUM",
            "summary": "You're partway there.",
        }
        llm = FakeLLMProvider(json.dumps(payload))
        result = analyze_skill_gap(["Python"], "Data Analyst", llm)
        self.assertEqual(result.matched_skills, ["Python"])
        self.assertEqual(result.missing_skills, ["SQL"])
        self.assertEqual(result.overall_readiness, "medium")
        self.assertIn("target role", "target role: Data Analyst".lower())  # sanity

    def test_handles_markdown_fenced_json(self):
        payload = {
            "matched_skills": [],
            "missing_skills": [],
            "partially_met_skills": [],
            "overall_readiness": "low",
            "summary": "Just starting out.",
        }
        llm = FakeLLMProvider(f"```json\n{json.dumps(payload)}\n```")
        result = analyze_skill_gap([], "Data Scientist", llm)
        self.assertEqual(result.overall_readiness, "low")

    def test_empty_target_role_raises(self):
        llm = FakeLLMProvider("{}")
        with self.assertRaises(LLMError):
            analyze_skill_gap(["Python"], "  ", llm)

    def test_invalid_json_raises_llm_error(self):
        llm = FakeLLMProvider("not json at all")
        with self.assertRaises(LLMError):
            analyze_skill_gap(["Python"], "Data Analyst", llm)


class TestResumeAnalyzer(unittest.TestCase):
    def test_parses_well_formed_json(self):
        payload = {
            "extracted_skills": ["Excel", "SQL"],
            "experience_summary": "2 years in marketing analytics.",
            "strengths": ["Data-driven"],
            "gaps_or_improvements": ["No Python experience"],
            "suggested_target_roles": ["Marketing Analyst"],
        }
        llm = FakeLLMProvider(json.dumps(payload))
        result = analyze_resume("Some resume text describing experience.", llm)
        self.assertEqual(result.extracted_skills, ["Excel", "SQL"])
        self.assertEqual(result.suggested_target_roles, ["Marketing Analyst"])

    def test_empty_resume_text_raises(self):
        llm = FakeLLMProvider("{}")
        with self.assertRaises(LLMError):
            analyze_resume("   ", llm)

    def test_target_role_is_included_in_prompt(self):
        payload = {
            "extracted_skills": [],
            "experience_summary": "",
            "strengths": [],
            "gaps_or_improvements": [],
            "suggested_target_roles": [],
        }
        llm = FakeLLMProvider(json.dumps(payload))
        analyze_resume("Some resume text.", llm, target_role="Backend Engineer")
        self.assertIn("Backend Engineer", llm.last_prompt)


class TestRoadmapGenerator(unittest.TestCase):
    def test_parses_milestones(self):
        payload = {
            "milestones": [
                {
                    "title": "Foundations",
                    "duration": "Weeks 1-4",
                    "focus_skills": ["Python"],
                    "actions": ["Complete a Python course"],
                },
                {
                    "title": "Applied projects",
                    "duration": "Weeks 5-12",
                    "focus_skills": ["SQL"],
                    "actions": ["Build a portfolio project"],
                },
            ],
            "summary": "A steady path into data analytics.",
        }
        llm = FakeLLMProvider(json.dumps(payload))
        result = generate_roadmap(["Excel"], "Data Analyst", llm, timeframe_months=3)
        self.assertEqual(len(result.milestones), 2)
        self.assertEqual(result.milestones[0].title, "Foundations")
        self.assertEqual(result.target_role, "Data Analyst")

    def test_empty_target_role_raises(self):
        llm = FakeLLMProvider("{}")
        with self.assertRaises(LLMError):
            generate_roadmap(["Python"], "", llm)

    def test_timeframe_is_clamped(self):
        payload = {"milestones": [], "summary": "ok"}
        llm = FakeLLMProvider(json.dumps(payload))
        generate_roadmap(["Python"], "Data Analyst", llm, timeframe_months=999)
        self.assertIn("36 months", llm.last_prompt)


if __name__ == "__main__":
    unittest.main()
