"""Shared error types and resilience helpers.

Centralizing these means every module raises/handles errors the same way:
specific exception types, safe user-facing messages, and full detail only in
server-side logs — never leaked to the UI or CLI output.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class CareerAdvisorError(Exception):
    """Base class for all application-raised errors.

    `user_message` is safe to show directly in the UI/CLI. The original
    exception (if any) should be logged separately with full detail.
    """

    def __init__(self, user_message: str, *, cause: Exception | None = None):
        super().__init__(user_message)
        self.user_message = user_message
        self.cause = cause


class DocumentValidationError(CareerAdvisorError):
    """Raised when an uploaded document fails validation (type, size, content)."""


class QueryValidationError(CareerAdvisorError):
    """Raised when a user query fails validation (empty, too long, etc.)."""


class OffTopicQueryError(CareerAdvisorError):
    """Raised when a query falls outside the assistant's career-advice scope.

    This is a deliberate refusal, not a failure: the assistant is scoped to
    career guidance, so general-knowledge questions ("who is the PM of India",
    "list Indian actors") are turned away instead of answered.
    """


class VectorStoreError(CareerAdvisorError):
    """Raised when building or querying the vector store fails."""


class LLMError(CareerAdvisorError):
    """Raised when the underlying LLM provider fails after retries."""


class ResumeParsingError(CareerAdvisorError):
    """Raised when an uploaded resume can't be validated or its text extracted."""


class JobSearchError(CareerAdvisorError):
    """Raised when the job-search tool (live API or bundled fallback) fails."""


def safe_error_message(
    exc: Exception, *, fallback: str = "Something went wrong while processing your request."
) -> str:
    """Return a message safe to show to end users, logging the real exception server-side."""
    logger.error("Unhandled error: %s", exc, exc_info=True)
    if isinstance(exc, CareerAdvisorError):
        return exc.user_message
    return fallback


def retry_with_backoff(
    *,
    max_retries: int = 3,
    base_delay_seconds: float = 1.0,
    max_delay_seconds: float = 20.0,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Retry a function with exponential backoff + jitter on transient failures.

    A small, dependency-free stand-in for `tenacity` so we don't need to add a
    new third-party dependency just for retry logic.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except retry_on as exc:  # type: ignore[misc]
                    attempt += 1
                    if attempt > max_retries:
                        logger.error(
                            "%s failed after %d attempt(s): %s", func.__name__, attempt, exc
                        )
                        raise
                    delay = min(max_delay_seconds, base_delay_seconds * (2 ** (attempt - 1)))
                    delay += random.uniform(0, delay * 0.1)  # jitter
                    logger.warning(
                        "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                        func.__name__,
                        attempt,
                        max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)

        return wrapper

    return decorator
