"""LLM provider abstraction.

Why this exists: the original codebase (`crew_config.py`) hard-wired Gemini
directly into the response-generation code path, and named the class/module
after CrewAI despite never using it — there was no real agentic behavior,
just a single-pass prompt-and-response call.

This module fixes both problems:

1. `LLMProvider` is a small abstract interface. Swapping Gemini for another
   model later means writing one new class here, not touching the RAG
   pipeline, the Streamlit UI, or the CLI. Only Gemini is implemented for now
   — no other providers were requested — but the seam is clean.

2. `decide_needs_retrieval()` is a real, minimal agentic step: before running
   a similarity search, the model is asked a small, separate yes/no question
   — "does answering this need the uploaded document, or is it general
   career guidance?" — and retrieval only runs if the answer is yes. This
   avoids polluting general questions ("what should I wear to an interview?")
   with irrelevant document chunks just because a PDF happens to be loaded.

   This project remains a single-provider RAG system with one lightweight
   decision step, not a multi-agent CrewAI pipeline — the `crewai` dependency
   has been removed accordingly (see requirements.txt and README).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from backend.errors import LLMError, retry_with_backoff
from backend.tracing import traceable

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


class LLMProvider(ABC):
    """Abstract interface every LLM backend must implement."""

    @abstractmethod
    def generate(self, messages: Sequence[ChatMessage], *, system_instruction: str = "") -> str:
        """Generate a response given a bounded conversation history.

        `messages` is ordered oldest-first and should already be trimmed to a
        reasonable window by the caller (see `rag_service.py`).
        """
        raise NotImplementedError

    @abstractmethod
    def decide_needs_retrieval(self, query: str) -> bool:
        """Return True if answering `query` would benefit from document retrieval."""
        raise NotImplementedError

    def is_career_related(self, query: str) -> bool:
        """Return True if `query` is in scope for a career-advice assistant.

        Concrete rather than abstract on purpose: every existing provider (and
        every test double) keeps working without implementing it. Subclasses
        able to answer a short classification prompt more cheaply than a full
        `generate()` call should override it — `GeminiProvider` does.

        Fails open. A garbled verdict, a timeout, or a provider outage returns
        True, because wrongly refusing a real career question is a worse
        failure than occasionally letting an off-topic one through.
        """
        try:
            verdict = self.generate([ChatMessage(role="user", content=career_topic_prompt(query))])
        except Exception as exc:
            logger.warning("Topic-guard step failed, allowing the query through: %s", exc)
            return True
        return not is_off_topic_verdict(verdict)


# --- Career-scope topic guard -------------------------------------------------

CAREER_VERDICT = "CAREER"
OFF_TOPIC_VERDICT = "OFF_TOPIC"


def career_topic_prompt(query: str) -> str:
    """Build the classification prompt used by the career-scope topic guard."""
    return (
        "You are a strict topic filter for a career-advice assistant. That "
        "assistant answers ONLY questions about careers and professional "
        "development.\n\n"
        "IN SCOPE: resumes and CVs, cover letters, job search and "
        "applications, interviews, salary and negotiation, promotions, career "
        "changes, choosing a career path, skills and skill gaps, "
        "certifications, courses studied for a career, internships, workplace "
        "and professional-growth questions, networking, portfolios, and the "
        "industries or roles the user is considering working in. Greetings and "
        "questions about what the assistant itself can do are also in scope.\n\n"
        "OUT OF SCOPE: general knowledge, current affairs, politics and heads "
        "of state, celebrities, entertainment, sport, history, science trivia, "
        "maths problems, recipes, travel, shopping, medical or legal advice, "
        "coding help unrelated to the user's career, and requests to write, "
        "translate or summarise general content.\n\n"
        "A question stays out of scope even when it mentions a career word in "
        "passing: 'list Indian actors and their salaries' is about "
        "celebrities, not about the user's career.\n\n"
        "Examples:\n"
        'Q: "who is the prime minister of India" -> OFF_TOPIC\n'
        'Q: "list indian actors" -> OFF_TOPIC\n'
        'Q: "what is the capital of France" -> OFF_TOPIC\n'
        'Q: "write me a poem about the sea" -> OFF_TOPIC\n'
        'Q: "how do I become a machine learning engineer" -> CAREER\n'
        'Q: "review the skills section of my resume" -> CAREER\n'
        'Q: "is a PMP certification worth it" -> CAREER\n'
        'Q: "hi, what can you help me with?" -> CAREER\n\n'
        f'Question: "{query}"\n\n'
        f"Reply with exactly one word: {CAREER_VERDICT} or {OFF_TOPIC_VERDICT}."
    )


def is_off_topic_verdict(raw: str) -> bool:
    """Interpret a topic-guard reply.

    Only an explicit refusal counts. An empty reply, a hedge, or a model that
    ignored the output format allows the query through, which is what keeps
    the guard fail-open.
    """
    normalized = (raw or "").strip().upper()
    if not normalized:
        return False
    return OFF_TOPIC_VERDICT in normalized or normalized.startswith("OFF")


# Models tried, in order, when the configured model is unavailable (retired,
# not enabled for the key, or rate-limited).
#
# The "-latest" aliases come first deliberately: they are moving pointers to
# whatever Google currently serves, so they keep resolving as individual
# versions are retired and work with any key that has Gemini API access. The
# pinned IDs after them are a backstop in case an alias is ever unavailable.
_FALLBACK_MODELS = (
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
)

# Substrings marking an error as worth trying a *different model* for, rather
# than retrying the same one.
_TRY_NEXT_MODEL_MARKERS = ("not found", "404", "429", "quota", "rate limit", "unsupported")

# Substrings marking a transient failure worth retrying the *same* model.
_TRANSIENT_MARKERS = ("503", "500", "502", "504", "unavailable", "timeout", "timed out", "deadline")

# Substrings marking a credentials problem — never worth retrying.
_AUTH_MARKERS = ("api key", "api_key", "401", "403", "permission denied", "unauthenticated", "invalid argument: key")


class _TransientLLMFailure(Exception):
    """Internal marker for failures worth retrying against the same model."""


class GeminiProvider(LLMProvider):
    """Google Gemini implementation of LLMProvider.

    Uses the `google-genai` SDK. The older `google-generativeai` package this
    previously imported is deprecated upstream ("all support has ended") and
    emits a FutureWarning on import, so it is no longer a safe base for a
    project that needs to keep answering requests.
    """

    def __init__(
        self, api_key: str, model_name: str, *, timeout_seconds: int = 30, max_retries: int = 3
    ):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "The google-genai package is not installed. Run: pip install -r requirements.txt"
            ) from exc

        if not api_key or not api_key.strip() or api_key.strip().startswith("your_"):
            raise LLMError(
                "No usable Google API key found. Copy .env.example to .env and set "
                "GOOGLE_API_KEY to a real key from https://aistudio.google.com/apikey"
            )

        self._types = types
        # `timeout` on HttpOptions is in milliseconds.
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_seconds * 1000),
        )
        self._model_name = model_name
        self._max_retries = max_retries
        logger.info("GeminiProvider initialized (model=%s, sdk=google-genai)", model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def _candidate_models(self) -> list[str]:
        """Configured model first, then current stable fallbacks, de-duplicated."""
        return list(dict.fromkeys([self._model_name, *_FALLBACK_MODELS]))

    def _extract_text(self, response) -> str:
        """Pull text out of a response, or explain precisely why there is none.

        An empty response is not the same failure as an unreachable API, and
        conflating them (as the previous version did) sent users off to check an
        API key that was working fine.
        """
        text = getattr(response, "text", None)
        if text and text.strip():
            return text

        candidates = getattr(response, "candidates", None) or []
        if candidates:
            reason = getattr(candidates[0], "finish_reason", None)
            reason_name = getattr(reason, "name", str(reason)) if reason else None
            if reason_name in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "IMAGE_SAFETY"}:
                raise LLMError(
                    "The model declined to answer that request because it was flagged by "
                    "Gemini's safety filters. Try rephrasing your question."
                )
            if reason_name == "MAX_TOKENS":
                raise LLMError(
                    "The response was cut off before any text was produced. Try a shorter "
                    "question, or raise the model's output limit."
                )
            if reason_name == "RECITATION":
                raise LLMError(
                    "The model stopped because its answer too closely reproduced training "
                    "data. Try rephrasing your question."
                )

        feedback = getattr(response, "prompt_feedback", None)
        blocked = getattr(feedback, "block_reason", None) if feedback else None
        if blocked:
            raise LLMError(
                f"Gemini blocked this prompt ({getattr(blocked, 'name', blocked)}). "
                "Try rephrasing your question."
            )

        raise LLMError("The AI model returned an empty response. Please try again.")

    def _generate_once(self, model_name: str, prompt: str) -> str:
        """One attempt against one model, retrying only genuinely transient failures.

        Retrying a rejected API key or a retired model just delays the error the
        user needs to see, so only 5xx/timeout-class failures are retried; every
        other error is raised immediately for `_call_model` to classify.
        """

        @retry_with_backoff(
            max_retries=self._max_retries,
            base_delay_seconds=1.0,
            retry_on=(_TransientLLMFailure,),
        )
        def _attempt() -> str:
            try:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=self._request_config(),
                )
            except Exception as exc:
                err_str = str(exc).lower()
                is_auth = any(m in err_str for m in _AUTH_MARKERS)
                if not is_auth and any(m in err_str for m in _TRANSIENT_MARKERS):
                    raise _TransientLLMFailure(str(exc)) from exc
                raise
            return self._extract_text(response)

        try:
            return _attempt()
        except _TransientLLMFailure as exc:
            # Surface the original error so the caller can classify it normally.
            raise (exc.__cause__ or exc) from exc

    def _request_config(self):
        """Per-request config.

        Automatic function calling is explicitly disabled: this provider never
        passes tools, and leaving AFC on makes the SDK log a warning on every
        single call.
        """
        return self._types.GenerateContentConfig(
            automatic_function_calling=self._types.AutomaticFunctionCallingConfig(disable=True),
        )

    def _call_model(self, prompt: str) -> str:
        last_error: Exception | None = None

        for model_name in self._candidate_models():
            try:
                return self._generate_once(model_name, prompt)
            except LLMError:
                # Already a precise, user-facing diagnosis (safety block, empty
                # response, blocked prompt) — surface it rather than masking it
                # by silently trying another model.
                raise
            except Exception as exc:
                last_error = exc
                err_str = str(exc).lower()

                if any(m in err_str for m in _AUTH_MARKERS):
                    raise LLMError(
                        "Your Google API key was rejected. Check that GOOGLE_API_KEY in .env is "
                        "a valid key from https://aistudio.google.com/apikey and that the "
                        "Gemini API is enabled for it.",
                        cause=exc,
                    ) from exc

                if any(m in err_str for m in _TRY_NEXT_MODEL_MARKERS):
                    logger.warning(
                        "Model %s unavailable (%s) — trying next candidate model.", model_name, exc
                    )
                    continue

                logger.error("Model %s failed with a non-recoverable error: %s", model_name, exc)
                break

        raise LLMError(
            "I couldn't get a response from Gemini. Every candidate model failed — check your "
            "API key, your quota at https://aistudio.google.com, and your network connection.",
            cause=last_error,
        ) from last_error

    @traceable(name="GeminiProvider.generate", run_type="llm")
    def generate(self, messages: Sequence[ChatMessage], *, system_instruction: str = "") -> str:
        prompt_parts: list[str] = []
        if system_instruction:
            prompt_parts.append(system_instruction.strip())
        for msg in messages:
            speaker = "User" if msg.role == "user" else "Assistant"
            prompt_parts.append(f"{speaker}: {msg.content}")
        prompt_parts.append("Assistant:")
        prompt = "\n\n".join(prompt_parts)
        return self._call_model(prompt)

    def decide_needs_retrieval(self, query: str) -> bool:
        """Ask the model a small, separate yes/no question before retrieving.

        Falls back to `True` (retrieve) on any failure — retrieving unnecessary
        context is a much cheaper failure mode than silently skipping relevant
        context, and the relevance threshold in the retrieval step filters out
        low-quality matches anyway.
        """
        decision_prompt = (
            "You are a routing step in a career-advice assistant that has an "
            "uploaded reference document available. Decide whether answering "
            "the user's question below would meaningfully benefit from "
            "searching that document, versus being answerable with general "
            "career-advice knowledge alone.\n\n"
            f'Question: "{query}"\n\n'
            "Reply with exactly one word: YES or NO."
        )
        try:
            answer = self._call_model(decision_prompt).strip().upper()
            return answer.startswith("Y")
        except Exception as exc:
            logger.warning("Retrieval-decision step failed, defaulting to retrieve=True: %s", exc)
            return True

    def is_career_related(self, query: str) -> bool:
        """Classify the query's topic with a single short prompt.

        Overrides the base implementation to use `_call_model` directly, which
        skips the conversation-shaped request `generate()` builds — this is a
        one-word classification, not a chat turn.
        """
        try:
            verdict = self._call_model(career_topic_prompt(query))
        except Exception as exc:
            logger.warning("Topic-guard step failed, allowing the query through: %s", exc)
            return True

        off_topic = is_off_topic_verdict(verdict)
        if off_topic:
            logger.info("Topic guard refused an out-of-scope query.")
        return not off_topic
