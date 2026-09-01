"""Centralized application configuration.

All environment-derived settings live here. Every other module should import
`get_settings()` instead of calling `os.getenv()` directly, so there is a single
source of truth for configuration and a single place validation happens.

Settings are validated eagerly at load time: missing or malformed required
values raise a `ConfigError` with an actionable message, rather than letting a
half-configured app fail confusingly mid-conversation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

try:
    # Loads variables from a local .env file into os.environ, if present.
    # Safe no-op in environments (e.g. CI, containers) that inject env vars directly.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - python-dotenv is a required dependency,
    # but we don't want a missing import to be the failure mode for config loading.
    pass


class ConfigError(ValueError):
    """Raised when required configuration is missing or invalid."""


def _get_str(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise ConfigError(
            f"Required environment variable '{name}' is not set. "
            f"Copy .env.example to .env and provide a value."
        )
    return value


def _get_int(
    name: str, default: int, *, min_value: int | None = None, max_value: int | None = None
) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ConfigError(
                f"Environment variable '{name}' must be an integer, got: {raw!r}"
            ) from exc

    if min_value is not None and value < min_value:
        raise ConfigError(f"Environment variable '{name}' must be >= {min_value}, got: {value}")
    if max_value is not None and value > max_value:
        raise ConfigError(f"Environment variable '{name}' must be <= {max_value}, got: {value}")
    return value


def _get_float(
    name: str, default: float, *, min_value: float | None = None, max_value: float | None = None
) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ConfigError(
                f"Environment variable '{name}' must be a number, got: {raw!r}"
            ) from exc

    if min_value is not None and value < min_value:
        raise ConfigError(f"Environment variable '{name}' must be >= {min_value}, got: {value}")
    if max_value is not None and value > max_value:
        raise ConfigError(f"Environment variable '{name}' must be <= {max_value}, got: {value}")
    return value


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_VALID_ENVIRONMENTS = {"development", "production", "test"}
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@dataclass(frozen=True)
class Settings:
    """Immutable, validated application settings."""

    # --- Secrets ---
    google_api_key: str

    # --- Environment / logging ---
    app_env: str = "development"
    log_level: str = "INFO"

    # --- LLM ---
    gemini_model: str = "gemini-flash-latest"
    llm_request_timeout_seconds: int = 30
    llm_max_retries: int = 3

    # --- RAG pipeline ---
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_context_docs: int = 3
    relevance_score_threshold: float = 0.35
    chroma_persist_dir: str = "chroma_db"
    default_document_path: str = "Career_Advisor_Guide_2025.pdf"
    # Build the vector store from `default_document_path` on startup when no
    # persisted store exists yet, so retrieval works on a fresh checkout
    # instead of silently falling back to un-grounded answers.
    auto_index_default_document: bool = True

    # --- Scope ---
    # Refuse questions outside career advice (general knowledge, politics,
    # entertainment) instead of answering them.
    strict_career_only: bool = True

    # --- Limits ---
    max_upload_size_mb: int = 15
    max_query_length: int = 2000
    max_history_turns: int = 6

    # --- LangSmith (tracing / evaluation) ---
    langsmith_tracing_enabled: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "career-advisor"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # --- MCP server ---
    mcp_server_name: str = "career-advisor"
    mcp_transport: str = "stdio"

    # --- Job search ---
    adzuna_app_id: str | None = None
    adzuna_app_key: str | None = None
    job_search_country: str = "us"
    job_search_default_limit: int = 10

    # --- Resume analysis ---
    max_resume_size_mb: int = 10

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def max_resume_size_bytes(self) -> int:
        return self.max_resume_size_mb * 1024 * 1024

    @property
    def has_job_search_api(self) -> bool:
        return bool(self.adzuna_app_id and self.adzuna_app_key)


def load_settings() -> Settings:
    """Read and validate settings from the environment. Raises ConfigError on failure."""

    app_env = _get_str("APP_ENV", "development") or "development"
    if app_env not in _VALID_ENVIRONMENTS:
        raise ConfigError(f"APP_ENV must be one of {sorted(_VALID_ENVIRONMENTS)}, got: {app_env!r}")

    log_level = (_get_str("LOG_LEVEL", "INFO") or "INFO").upper()
    if log_level not in _VALID_LOG_LEVELS:
        raise ConfigError(
            f"LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}, got: {log_level!r}"
        )

    settings = Settings(
        google_api_key=_get_str("GOOGLE_API_KEY", required=True),
        app_env=app_env,
        log_level=log_level,
        gemini_model=_get_str("GEMINI_MODEL", "gemini-flash-latest"),
        llm_request_timeout_seconds=_get_int(
            "LLM_REQUEST_TIMEOUT_SECONDS", 30, min_value=1, max_value=300
        ),
        llm_max_retries=_get_int("LLM_MAX_RETRIES", 3, min_value=0, max_value=10),
        chunk_size=_get_int("CHUNK_SIZE", 500, min_value=50, max_value=10_000),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 50, min_value=0, max_value=5_000),
        max_context_docs=_get_int("MAX_CONTEXT_DOCS", 3, min_value=1, max_value=20),
        relevance_score_threshold=_get_float(
            "RELEVANCE_SCORE_THRESHOLD", 0.35, min_value=0.0, max_value=1.0
        ),
        chroma_persist_dir=_get_str("CHROMA_PERSIST_DIR", "chroma_db"),
        default_document_path=_get_str("DEFAULT_DOCUMENT_PATH", "Career_Advisor_Guide_2025.pdf"),
        auto_index_default_document=_get_bool("AUTO_INDEX_DEFAULT_DOCUMENT", True),
        strict_career_only=_get_bool("STRICT_CAREER_ONLY", True),
        max_upload_size_mb=_get_int("MAX_UPLOAD_SIZE_MB", 15, min_value=1, max_value=200),
        max_query_length=_get_int("MAX_QUERY_LENGTH", 2000, min_value=10, max_value=20_000),
        max_history_turns=_get_int("MAX_HISTORY_TURNS", 6, min_value=0, max_value=50),
        langsmith_tracing_enabled=_get_bool("LANGSMITH_TRACING", False),
        langsmith_api_key=_get_str("LANGSMITH_API_KEY"),
        langsmith_project=_get_str("LANGSMITH_PROJECT", "career-advisor") or "career-advisor",
        langsmith_endpoint=_get_str("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
        or "https://api.smith.langchain.com",
        mcp_server_name=_get_str("MCP_SERVER_NAME", "career-advisor") or "career-advisor",
        mcp_transport=_get_str("MCP_TRANSPORT", "stdio") or "stdio",
        adzuna_app_id=_get_str("ADZUNA_APP_ID"),
        adzuna_app_key=_get_str("ADZUNA_APP_KEY"),
        job_search_country=_get_str("JOB_SEARCH_COUNTRY", "us") or "us",
        job_search_default_limit=_get_int(
            "JOB_SEARCH_DEFAULT_LIMIT", 10, min_value=1, max_value=50
        ),
        max_resume_size_mb=_get_int("MAX_RESUME_SIZE_MB", 10, min_value=1, max_value=200),
    )

    if settings.langsmith_tracing_enabled and not settings.langsmith_api_key:
        raise ConfigError(
            "LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is not set. "
            "Provide a LangSmith API key or set LANGSMITH_TRACING=false."
        )

    if settings.chunk_overlap >= settings.chunk_size:
        raise ConfigError(
            f"CHUNK_OVERLAP ({settings.chunk_overlap}) must be smaller than "
            f"CHUNK_SIZE ({settings.chunk_size})."
        )

    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide cached settings. Call `get_settings.cache_clear()` in tests
    that need to reload configuration under different environment variables."""
    return load_settings()


def configure_logging(settings: Settings | None = None) -> None:
    """Configure root logging once, consistently, for whichever entry point is running."""
    settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
