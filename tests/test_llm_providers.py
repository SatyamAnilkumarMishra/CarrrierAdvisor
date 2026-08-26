import sys
import types
import unittest
from unittest.mock import MagicMock

from llm_providers import ChatMessage, GeminiProvider


class _FakeResponse:
    def __init__(self, text):
        self.text = text


def _install_fake_genai(generate_content_impl):
    """Inject a fake `google.generativeai` module into sys.modules so
    GeminiProvider can be instantiated and exercised without the real
    package (or a network call) being available."""
    fake_model = MagicMock()
    fake_model.generate_content.side_effect = generate_content_impl

    fake_genai = types.ModuleType("google.generativeai")
    fake_genai.configure = MagicMock()
    fake_genai.GenerativeModel = MagicMock(return_value=fake_model)

    fake_google = types.ModuleType("google")
    fake_google.generativeai = fake_genai

    sys.modules["google"] = fake_google
    sys.modules["google.generativeai"] = fake_genai
    return fake_model


class TestGeminiProvider(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("google.generativeai", None)
        sys.modules.pop("google", None)

    def test_generate_builds_prompt_and_returns_text(self):
        _install_fake_genai(lambda *a, **k: _FakeResponse("Here's some career advice."))
        provider = GeminiProvider(api_key="fake-key", model_name="gemini-flash-latest", max_retries=0)

        result = provider.generate(
            [ChatMessage(role="user", content="What skills do I need for data science?")],
            system_instruction="Be concise.",
        )
        self.assertEqual(result, "Here's some career advice.")

    def test_decide_needs_retrieval_true_on_yes(self):
        _install_fake_genai(lambda *a, **k: _FakeResponse("YES"))
        provider = GeminiProvider(api_key="fake-key", model_name="gemini-flash-latest", max_retries=0)
        self.assertTrue(
            provider.decide_needs_retrieval("What does the document say about resumes?")
        )

    def test_decide_needs_retrieval_false_on_no(self):
        _install_fake_genai(lambda *a, **k: _FakeResponse("NO"))
        provider = GeminiProvider(api_key="fake-key", model_name="gemini-flash-latest", max_retries=0)
        self.assertFalse(provider.decide_needs_retrieval("What should I wear to an interview?"))

    def test_decide_needs_retrieval_defaults_true_on_failure(self):
        def _boom(*a, **k):
            raise RuntimeError("API is down")

        _install_fake_genai(_boom)
        provider = GeminiProvider(api_key="fake-key", model_name="gemini-flash-latest", max_retries=0)
        self.assertTrue(provider.decide_needs_retrieval("anything"))

    def test_empty_response_raises_llm_error(self):
        from errors import LLMError

        _install_fake_genai(lambda *a, **k: _FakeResponse(""))
        provider = GeminiProvider(api_key="fake-key", model_name="gemini-flash-latest", max_retries=0)
        with self.assertRaises(LLMError):
            provider.generate([ChatMessage(role="user", content="hello")])


if __name__ == "__main__":
    unittest.main()
