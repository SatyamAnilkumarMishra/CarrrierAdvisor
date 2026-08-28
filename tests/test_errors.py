import unittest
from unittest.mock import patch

from errors import CareerAdvisorError, retry_with_backoff, safe_error_message


class TestSafeErrorMessage(unittest.TestCase):
    def test_known_error_returns_user_message(self):
        exc = CareerAdvisorError("Friendly message")
        self.assertEqual(safe_error_message(exc), "Friendly message")

    def test_unknown_error_returns_fallback_not_internals(self):
        exc = ValueError("some internal stack trace detail nobody should see")
        message = safe_error_message(exc)
        self.assertNotIn("stack trace", message)
        self.assertEqual(message, "Something went wrong while processing your request.")

    def test_unknown_error_custom_fallback(self):
        exc = RuntimeError("boom")
        message = safe_error_message(exc, fallback="Custom fallback")
        self.assertEqual(message, "Custom fallback")


class TestRetryWithBackoff(unittest.TestCase):
    def test_succeeds_without_retry(self):
        calls = {"count": 0}

        @retry_with_backoff(max_retries=3, base_delay_seconds=0)
        def always_succeeds():
            calls["count"] += 1
            return "ok"

        self.assertEqual(always_succeeds(), "ok")
        self.assertEqual(calls["count"], 1)

    @patch("errors.time.sleep", return_value=None)
    def test_retries_then_succeeds(self, _mock_sleep):
        calls = {"count": 0}

        @retry_with_backoff(max_retries=3, base_delay_seconds=0)
        def fails_twice_then_succeeds():
            calls["count"] += 1
            if calls["count"] < 3:
                raise ConnectionError("transient")
            return "ok"

        self.assertEqual(fails_twice_then_succeeds(), "ok")
        self.assertEqual(calls["count"], 3)
