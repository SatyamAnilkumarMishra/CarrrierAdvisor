"""Career tools: job search, skill-gap analysis, resume analysis, roadmap generation.

These are the four capabilities exposed both to end users (via `app.py`'s
"Tools" tabs) and to external MCP clients (via `mcp_server.py`). Keeping the
logic here — rather than inline in either caller — mirrors the reasoning
behind `rag_service.py`: one implementation, multiple front doors, no risk of
behavior drifting between the Streamlit UI, the MCP server, and the LangSmith
evaluation harness in `evaluation.py`.

Three of the four tools (skill-gap analysis, resume analysis, roadmap
generation) ask the LLM for a JSON object matching a specific shape and parse
it with `_generate_json()`. The job-search tool is not an LLM call: it either
queries the Adzuna API (if `ADZUNA_APP_ID`/`ADZUNA_APP_KEY` are configured)
or filters the small bundled dataset in `jobs_data.py`.

Every public function is wrapped with `@traceable` (see `tracing.py`) so runs
show up in LangSmith when tracing is enabled, and are no-ops otherwise.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from backend.config import Settings
from backend.errors import JobSearchError, LLMError
from backend.jobs_data import SAMPLE_JOBS
from backend.llm_providers import ChatMessage, LLMProvider
from backend.tracing import traceable

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JobListing:
    title: str
    company: str
    location: str
    url: str
    description: str = ""
    skills: list[str] = field(default_factory=list)
    experience_level: str | None = None
    source: str = "bundled"


@dataclass(frozen=True)
class SkillGapResult:
    matched_skills: list[str]
    missing_skills: list[str]
    partially_met_skills: list[str]
    overall_readiness: str  # e.g. "low" | "medium" | "high"
    summary: str


@dataclass(frozen=True)
class ResumeAnalysis:
    extracted_skills: list[str]
    experience_summary: str
    strengths: list[str]
    gaps_or_improvements: list[str]
    suggested_target_roles: list[str]


@dataclass(frozen=True)
class RoadmapMilestone:
    title: str
    duration: str
    focus_skills: list[str]
    actions: list[str]


@dataclass(frozen=True)
class RoadmapResult:
    target_role: str
    milestones: list[RoadmapMilestone]
    summary: str


# --------------------------------------------------------------------------
# Shared LLM-JSON helper
# --------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of `text`, tolerating markdown code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = _JSON_BLOCK_RE.search(cleaned)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise LLMError("The model returned a response that wasn't valid JSON. Please try again.")


def _generate_json(llm: LLMProvider, prompt: str, *, system_instruction: str = "") -> dict:
    text = llm.generate(
        [ChatMessage(role="user", content=prompt)], system_instruction=system_instruction
    )
    return _extract_json(text)


def _as_str_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v).strip() for v in value if str(v).strip()]


# --------------------------------------------------------------------------
# 1. Job Search Tool
# --------------------------------------------------------------------------


@traceable(name="search_jobs", run_type="tool")
def search_jobs(
    role: str,
    settings: Settings,
    *,
    skills: Sequence[str] | None = None,
    location: str | None = None,
    experience_level: str | None = None,
    limit: int | None = None,
) -> list[JobListing]:
    """Search jobs by role, skills, location, and experience level.

    Uses the Adzuna API when credentials are configured (`Settings.has_job_search_api`),
    otherwise filters the small bundled sample dataset — so this tool always
    returns something usable, even with zero external configuration.
    """
    role = (role or "").strip()
    if not role:
        raise JobSearchError("Please provide a job role or title to search for.")

    limit = limit or settings.job_search_default_limit

    if settings.has_job_search_api:
        try:
            return _search_jobs_adzuna(
                role, settings, skills=skills, location=location, limit=limit
            )
        except JobSearchError:
            raise
        except Exception as exc:
            logger.warning("Adzuna job search failed, falling back to bundled dataset: %s", exc)

    return _search_jobs_bundled(
        role, skills=skills, location=location, experience_level=experience_level, limit=limit
    )


def _search_jobs_adzuna(
    role: str,
    settings: Settings,
    *,
    skills: Sequence[str] | None,
    location: str | None,
    limit: int,
) -> list[JobListing]:
    import requests

    query_terms = " ".join([role, *(skills or [])]).strip()
    country = (settings.job_search_country or "us").lower()
    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
    params = {
        "app_id": settings.adzuna_app_id,
        "app_key": settings.adzuna_app_key,
        "what": query_terms,
        "results_per_page": min(limit, 50),
    }
    if location:
        params["where"] = location

    try:
        response = requests.get(url, params=params, timeout=settings.llm_request_timeout_seconds)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise JobSearchError(
            "We couldn't reach the job search service. Please try again shortly.", cause=exc
        ) from exc

    listings: list[JobListing] = []
    for item in payload.get("results", [])[:limit]:
        listings.append(
            JobListing(
                title=item.get("title", "Untitled role"),
                company=(item.get("company") or {}).get("display_name", "Unknown company"),
                location=(item.get("location") or {}).get("display_name", "Unspecified"),
                url=item.get("redirect_url", ""),
                description=(item.get("description") or "")[:500],
                skills=list(skills or []),
                experience_level=None,
                source="adzuna",
            )
        )
    return listings


def _search_jobs_bundled(
    role: str,
    *,
    skills: Sequence[str] | None,
    location: str | None,
    experience_level: str | None,
    limit: int,
) -> list[JobListing]:
    role_terms = {t for t in re.split(r"\s+", role.lower()) if t}
    skill_terms = {s.lower().strip() for s in (skills or []) if s.strip()}
    location_term = (location or "").lower().strip()
    experience_term = (experience_level or "").lower().strip()

    scored: list[tuple[float, dict]] = []
    for job in SAMPLE_JOBS:
        title_terms = set(re.split(r"\s+", job["title"].lower()))
        job_skills = {s.lower() for s in job["skills"]}

        role_score = len(role_terms & title_terms) / max(len(role_terms), 1)
        skill_score = (
            len(skill_terms & job_skills) / max(len(skill_terms), 1) if skill_terms else 0.0
        )

        if (
            location_term
            and location_term not in job["location"].lower()
            and location_term != "remote"
        ):
            continue
        if location_term == "remote" and job["location"].lower() != "remote":
            continue
        if experience_term and experience_term != job["experience_level"]:
            continue

        score = role_score + skill_score
        if role_score == 0 and skill_score == 0:
            continue
        scored.append((score, job))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    results = [job for _, job in scored[:limit]]

    return [
        JobListing(
            title=job["title"],
            company=job["company"],
            location=job["location"],
            url=job["url"],
            description=job["description"],
            skills=job["skills"],
            experience_level=job["experience_level"],
            source="bundled",
        )
        for job in results
    ]


# --------------------------------------------------------------------------
# 2. Skill Gap Analyzer
# --------------------------------------------------------------------------


@traceable(name="analyze_skill_gap", run_type="chain")
def analyze_skill_gap(
    user_skills: Sequence[str], target_role: str, llm: LLMProvider
) -> SkillGapResult:
    """Compare a user's current skills against a target role using the LLM."""
    target_role = (target_role or "").strip()
    if not target_role:
        raise LLMError("Please provide a target role to compare skills against.")
    skills_text = ", ".join(s.strip() for s in user_skills if s.strip()) or "(none provided)"

    prompt = (
        "You are a career-skills analyst. Compare the candidate's current "
        f"skills against what is typically required for the target role.\n\n"
        f"Target role: {target_role}\n"
        f"Candidate's current skills: {skills_text}\n\n"
        "Respond with ONLY a JSON object (no markdown, no commentary) with exactly "
        "these keys:\n"
        '  "matched_skills": array of the candidate\'s skills that are relevant to the role,\n'
        '  "missing_skills": array of important skills for the role the candidate does not have,\n'
        '  "partially_met_skills": array of skills the candidate has some but not full proficiency in,\n'
        '  "overall_readiness": one of "low", "medium", "high",\n'
        '  "summary": a 2-3 sentence plain-language summary.'
    )

    data = _generate_json(llm, prompt)
    return SkillGapResult(
        matched_skills=_as_str_list(data.get("matched_skills")),
        missing_skills=_as_str_list(data.get("missing_skills")),
        partially_met_skills=_as_str_list(data.get("partially_met_skills")),
        overall_readiness=str(data.get("overall_readiness", "medium")).lower(),
        summary=str(data.get("summary", "")).strip(),
    )


# --------------------------------------------------------------------------
# 3. Resume Analyzer
# --------------------------------------------------------------------------


@traceable(name="analyze_resume", run_type="chain")
def analyze_resume(
    resume_text: str, llm: LLMProvider, *, target_role: str | None = None
) -> ResumeAnalysis:
    """Extract skills from resume text and identify missing/improvable areas.

    If `target_role` is given, gaps and suggested roles are framed relative
    to it; otherwise the model infers plausible target roles itself.
    """
    resume_text = (resume_text or "").strip()
    if not resume_text:
        raise LLMError("The resume appears to be empty — no text could be analyzed.")

    role_instruction = (
        f'Evaluate the resume specifically against this target role: "{target_role.strip()}".'
        if target_role and target_role.strip()
        else "Suggest 2-4 target roles the candidate is well-suited for based on the resume."
    )

    prompt = (
        "You are a resume reviewer for a career-advising assistant. Read the resume "
        f"text below and extract structured information. {role_instruction}\n\n"
        f'Resume text:\n"""\n{resume_text[:12000]}\n"""\n\n'
        "Respond with ONLY a JSON object (no markdown, no commentary) with exactly "
        "these keys:\n"
        '  "extracted_skills": array of skills found or reasonably implied in the resume,\n'
        '  "experience_summary": a 2-3 sentence summary of the candidate\'s experience level and background,\n'
        '  "strengths": array of notable strengths,\n'
        '  "gaps_or_improvements": array of missing skills, weak areas, or resume-writing improvements,\n'
        '  "suggested_target_roles": array of role titles this candidate is or could become well-suited for.'
    )

    data = _generate_json(llm, prompt)
    return ResumeAnalysis(
        extracted_skills=_as_str_list(data.get("extracted_skills")),
        experience_summary=str(data.get("experience_summary", "")).strip(),
        strengths=_as_str_list(data.get("strengths")),
        gaps_or_improvements=_as_str_list(data.get("gaps_or_improvements")),
        suggested_target_roles=_as_str_list(data.get("suggested_target_roles")),
    )


# --------------------------------------------------------------------------
# 4. Career Roadmap Generator
# --------------------------------------------------------------------------


@traceable(name="generate_roadmap", run_type="chain")
def generate_roadmap(
    current_skills: Sequence[str],
    target_role: str,
    llm: LLMProvider,
    *,
    timeframe_months: int = 6,
) -> RoadmapResult:
    """Generate a structured, milestone-based learning roadmap toward `target_role`."""
    target_role = (target_role or "").strip()
    if not target_role:
        raise LLMError("Please provide a target role to build a roadmap toward.")
    skills_text = ", ".join(s.strip() for s in current_skills if s.strip()) or "(none provided)"
    timeframe_months = max(1, min(int(timeframe_months or 6), 36))

    prompt = (
        "You are a career coach building a structured learning roadmap.\n\n"
        f"Target role: {target_role}\n"
        f"Candidate's current skills: {skills_text}\n"
        f"Desired timeframe: {timeframe_months} months\n\n"
        "Break the timeframe into 3-5 sequential milestones. Respond with ONLY a "
        "JSON object (no markdown, no commentary) with exactly these keys:\n"
        '  "milestones": array of objects, each with "title" (string), "duration" '
        '(e.g. "Weeks 1-4"), "focus_skills" (array of strings), and "actions" '
        "(array of 2-4 concrete action strings),\n"
        '  "summary": a 2-3 sentence overview of the roadmap.'
    )

    data = _generate_json(llm, prompt)
    milestones = [
        RoadmapMilestone(
            title=str(m.get("title", "")).strip(),
            duration=str(m.get("duration", "")).strip(),
            focus_skills=_as_str_list(m.get("focus_skills")),
            actions=_as_str_list(m.get("actions")),
        )
        for m in (data.get("milestones") or [])
        if isinstance(m, dict)
    ]
    return RoadmapResult(
        target_role=target_role,
        milestones=milestones,
        summary=str(data.get("summary", "")).strip(),
    )
