"""LangSmith evaluation harness.

Run with: `python evaluation.py` (requires LANGSMITH_API_KEY + LANGSMITH_TRACING=true
in .env — see .env.example). Uploads/reuses a small dataset in your LangSmith
project and scores the current model/prompts against it, so a prompt or model
change can be checked for regressions instead of eyeballed.

Two things are evaluated, since the app now does more than one kind of
generation:

- `evaluate_chat()` — the RAG chat flow (`RagService.answer`), scored on
  whether the assistant actually used the uploaded document when it should
  have, and whether an LLM judge rates the answer as relevant/on-topic.
- `evaluate_career_tools()` — the four MCP tools (skill-gap analysis, resume
  analysis, roadmap generation; job search is excluded since it's not an LLM
  call), scored on whether the tool's structured JSON output contains the
  fields a downstream client depends on.

This is intentionally a small, hand-curated dataset — enough to catch obvious
regressions (empty output, malformed JSON, dropped retrieval) rather than a
comprehensive benchmark.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from backend.config import ConfigError, get_settings
from backend.tracing import configure_langsmith

logger = logging.getLogger(__name__)

# --- Chat evaluation dataset -------------------------------------------------

CHAT_EXAMPLES: list[dict[str, Any]] = [
    {
        "inputs": {"question": "What skills matter most for an entry-level data analyst role?"},
        "outputs": {"should_use_document": False},
    },
    {
        "inputs": {"question": "What should I wear to a job interview?"},
        "outputs": {"should_use_document": False},
    },
    {
        "inputs": {
            "question": "According to the guide, what are the recommended steps for a career transition?"
        },
        "outputs": {"should_use_document": True},
    },
    {
        "inputs": {"question": "How do I write a strong resume summary?"},
        "outputs": {"should_use_document": False},
    },
]

# --- Career-tools evaluation dataset ----------------------------------------

TOOL_EXAMPLES: list[dict[str, Any]] = [
    {
        "tool": "skill_gap",
        "inputs": {
            "user_skills": ["Python", "Excel", "SQL basics"],
            "target_role": "Data Analyst",
        },
    },
    {
        "tool": "roadmap",
        "inputs": {
            "current_skills": ["HTML", "CSS", "basic JavaScript"],
            "target_role": "Frontend Engineer",
            "timeframe_months": 6,
        },
    },
    {
        "tool": "resume",
        "inputs": {
            "resume_text": (
                "Jane Doe. Experience: 2 years as a marketing coordinator using "
                "Google Analytics, Excel, and email campaign tools. Bachelor's "
                "in Communications."
            ),
            "target_role": "Data Analyst",
        },
    },
]


def _build_provider():
    from backend.llm_providers import GeminiProvider

    settings = get_settings()
    return GeminiProvider(
        api_key=settings.google_api_key,
        model_name=settings.gemini_model,
        timeout_seconds=settings.llm_request_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def _chat_target(inputs: dict) -> dict:
    from backend.rag_pipeline import load_existing_vector_store
    from backend.rag_service import RagService

    settings = get_settings()
    provider = _build_provider()
    service = RagService(settings, provider)
    vectorstore = load_existing_vector_store(settings)
    if vectorstore is not None:
        service.set_vectorstore(vectorstore)

    result = service.answer(inputs["question"])
    return {"answer": result.text, "used_retrieval": result.used_retrieval}


def _retrieval_matches_expectation(run, example) -> dict:
    expected = example.outputs.get("should_use_document")
    actual = run.outputs.get("used_retrieval")
    score = 1.0 if expected is None or expected == actual else 0.0
    return {"key": "retrieval_matches_expectation", "score": score}


def _non_empty_answer(run, example) -> dict:
    answer = (run.outputs or {}).get("answer", "")
    score = 1.0 if answer and len(answer.strip()) > 0 else 0.0
    return {"key": "non_empty_answer", "score": score}


def evaluate_chat(dataset_name: str = "career-advisor-chat") -> Any:
    """Score the RAG chat flow against `CHAT_EXAMPLES` using LangSmith."""
    from langsmith import Client

    client = Client()
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(
            inputs=[e["inputs"] for e in CHAT_EXAMPLES],
            outputs=[e["outputs"] for e in CHAT_EXAMPLES],
            dataset_id=dataset.id,
        )

    return client.evaluate(
        _chat_target,
        data=dataset_name,
        evaluators=[_retrieval_matches_expectation, _non_empty_answer],
        experiment_prefix="career-advisor-chat",
    )


_TOOL_TARGETS: dict[str, Callable[[dict], dict]] = {}


def _tool_target(inputs: dict) -> dict:
    from backend.career_tools import analyze_resume, analyze_skill_gap, generate_roadmap

    tool = inputs["tool"]
    payload = inputs["inputs"]
    provider = _build_provider()

    if tool == "skill_gap":
        result = analyze_skill_gap(payload["user_skills"], payload["target_role"], provider)
        return {"result": result.__dict__}
    if tool == "roadmap":
        result = generate_roadmap(
            payload["current_skills"],
            payload["target_role"],
            provider,
            timeframe_months=payload.get("timeframe_months", 6),
        )
        return {"result": result.__dict__}
    if tool == "resume":
        result = analyze_resume(
            payload["resume_text"], provider, target_role=payload.get("target_role")
        )
        return {"result": result.__dict__}
    raise ValueError(f"Unknown tool in eval dataset: {tool!r}")


def _tool_output_well_formed(run, example) -> dict:
    result = (run.outputs or {}).get("result") or {}
    score = 1.0 if isinstance(result, dict) and len(result) > 0 else 0.0
    return {"key": "well_formed_output", "score": score}


def evaluate_career_tools(dataset_name: str = "career-advisor-tools") -> Any:
    """Score the MCP career tools against `TOOL_EXAMPLES` using LangSmith."""
    from langsmith import Client

    client = Client()
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(
            inputs=[{"tool": e["tool"], "inputs": e["inputs"]} for e in TOOL_EXAMPLES],
            dataset_id=dataset.id,
        )

    return client.evaluate(
        _tool_target,
        data=dataset_name,
        evaluators=[_tool_output_well_formed],
        experiment_prefix="career-advisor-tools",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    try:
        settings = get_settings()
    except ConfigError as exc:
        print(f"Configuration problem: {exc}")
        return

    if not settings.langsmith_tracing_enabled:
        print(
            "LANGSMITH_TRACING is not enabled. Set LANGSMITH_TRACING=true and "
            "LANGSMITH_API_KEY in .env before running evaluations."
        )
        return

    configure_langsmith(settings)

    print("Running chat evaluation…")
    evaluate_chat()
    print("Running career-tools evaluation…")
    evaluate_career_tools()
    print("Done — see the results in your LangSmith project dashboard.")


if __name__ == "__main__":
    main()
