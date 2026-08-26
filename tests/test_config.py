import os
import unittest

from config import ConfigError, load_settings


class TestLoadSettings(unittest.TestCase):
    def setUp(self):
        self._original_environ = dict(os.environ)
        # Start from a clean slate for every test so leftover host env vars
        # (e.g. a real GOOGLE_API_KEY on the dev machine) can't mask failures.
        for key in list(os.environ):
            if key.startswith(
                (
                    "GOOGLE_API_KEY",
                    "APP_ENV",
                    "LOG_LEVEL",
                    "GEMINI_MODEL",
                    "CHUNK_SIZE",
                    "CHUNK_OVERLAP",
                    "MAX_CONTEXT_DOCS",
                    "RELEVANCE_SCORE_THRESHOLD",
                    "CHROMA_PERSIST_DIR",
                    "DEFAULT_DOCUMENT_PATH",
                    "MAX_UPLOAD_SIZE_MB",
                    "MAX_QUERY_LENGTH",
                    "MAX_HISTORY_TURNS",
                    "LLM_REQUEST_TIMEOUT_SECONDS",
                    "LLM_MAX_RETRIES",
                    "LANGSMITH_",
                    "MCP_",
                    "ADZUNA_",
                    "JOB_SEARCH_",
                    "MAX_RESUME_SIZE_MB",
                )
            ):
                os.environ.pop(key, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._original_environ)

    def test_missing_api_key_raises(self):
        with self.assertRaises(ConfigError):
            load_settings()

    def test_valid_minimal_config_uses_defaults(self):
        os.environ["GOOGLE_API_KEY"] = "test-key"
        settings = load_settings()
        self.assertEqual(settings.google_api_key, "test-key")
        self.assertEqual(settings.app_env, "development")
        self.assertEqual(settings.gemini_model, "gemini-flash-latest")
        self.assertEqual(settings.chunk_size, 500)

    def test_invalid_app_env_raises(self):
        os.environ["GOOGLE_API_KEY"] = "test-key"
        os.environ["APP_ENV"] = "not-a-real-env"
        with self.assertRaises(ConfigError):
            load_settings()

    def test_invalid_log_level_raises(self):
        os.environ["GOOGLE_API_KEY"] = "test-key"
        os.environ["LOG_LEVEL"] = "SUPER_VERBOSE"
        with self.assertRaises(ConfigError):
            load_settings()

    def test_non_integer_chunk_size_raises(self):
        os.environ["GOOGLE_API_KEY"] = "test-key"
        os.environ["CHUNK_SIZE"] = "not-a-number"
        with self.assertRaises(ConfigError):
            load_settings()

    def test_chunk_overlap_must_be_smaller_than_chunk_size(self):
        os.environ["GOOGLE_API_KEY"] = "test-key"
        os.environ["CHUNK_SIZE"] = "100"
        os.environ["CHUNK_OVERLAP"] = "100"
        with self.assertRaises(ConfigError):
            load_settings()

    def test_relevance_threshold_out_of_range_raises(self):
        os.environ["GOOGLE_API_KEY"] = "test-key"
        os.environ["RELEVANCE_SCORE_THRESHOLD"] = "1.5"
        with self.assertRaises(ConfigError):
            load_settings()

    def test_max_upload_size_bytes_conversion(self):
        os.environ["GOOGLE_API_KEY"] = "test-key"
        os.environ["MAX_UPLOAD_SIZE_MB"] = "10"
        settings = load_settings()
        self.assertEqual(settings.max_upload_size_bytes, 10 * 1024 * 1024)

    def test_langsmith_disabled_by_default(self):
        os.environ["GOOGLE_API_KEY"] = "test-key"
        settings = load_settings()
        self.assertFalse(settings.langsmith_tracing_enabled)
        self.assertEqual(settings.langsmith_project, "career-advisor")

    def test_langsmith_tracing_without_api_key_raises(self):
        os.environ["GOOGLE_API_KEY"] = "test-key"
        os.environ["LANGSMITH_TRACING"] = "true"
        with self.assertRaises(ConfigError):
            load_settings()

    def test_langsmith_tracing_with_api_key_succeeds(self):
        os.environ["GOOGLE_API_KEY"] = "test-key"
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = "ls-test-key"
        settings = load_settings()
        self.assertTrue(settings.langsmith_tracing_enabled)
        self.assertEqual(settings.langsmith_api_key, "ls-test-key")

    def test_has_job_search_api_requires_both_adzuna_vars(self):
        os.environ["GOOGLE_API_KEY"] = "test-key"
        settings = load_settings()
        self.assertFalse(settings.has_job_search_api)

        os.environ["ADZUNA_APP_ID"] = "id"
        settings = load_settings()
        self.assertFalse(settings.has_job_search_api)

        os.environ["ADZUNA_APP_KEY"] = "key"
        settings = load_settings()
        self.assertTrue(settings.has_job_search_api)

    def test_max_resume_size_bytes_conversion(self):
        os.environ["GOOGLE_API_KEY"] = "test-key"
        os.environ["MAX_RESUME_SIZE_MB"] = "5"
        settings = load_settings()
        self.assertEqual(settings.max_resume_size_bytes, 5 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()

