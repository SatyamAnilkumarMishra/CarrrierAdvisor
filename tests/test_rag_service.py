import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from errors import QueryValidationError
from llm_providers import ChatMessage, LLMProvider
from rag_service import RagService, trim_history, validate_query


def _settings(**overrides):
    defaults = dict(
        max_query_length=2000,
        max_history_turns=2,
        max_context_docs=3,
        relevance_score_threshold=0.35,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class FakeLLMProvider(LLMProvider):
    """Minimal in-memory stand-in for LLMProvider, used across tests instead
    of hitting a real model or even the fake-genai plumbing."""

    def __init__(self, needs_retrieval=True, response="Here is your answer."):
        self.needs_retrieval = needs_retrieval
        self.response = response
        self.last_messages = None
        self.last_system_instruction = None

    def generate(self, messages, *, system_instruction=""):
        self.last_messages = list(messages)
        self.last_system_instruction = system_instruction
        return self.response

    def decide_needs_retrieval(self, query):
        return self.needs_retrieval


class TestValidateQuery(unittest.TestCase):
    def test_empty_query_raises(self):
        with self.assertRaises(QueryValidationError):
            validate_query("   ", _settings())

    def test_too_long_query_raises(self):
        with self.assertRaises(QueryValidationError):
            validate_query("x" * 50, _settings(max_query_length=10))

    def test_valid_query_is_stripped(self):
        self.assertEqual(validate_query("  hello  ", _settings()), "hello")


class TestTrimHistory(unittest.TestCase):
    def test_keeps_only_last_n_turns(self):
        history = [
            ChatMessage(role="user" if i % 2 == 0 else "assistant", content=str(i))
            for i in range(10)
        ]
        trimmed = trim_history(history, _settings(max_history_turns=2))
        self.assertEqual(len(trimmed), 4)
        self.assertEqual([m.content for m in trimmed], ["6", "7", "8", "9"])

    def test_zero_turns_returns_empty(self):
        history = [ChatMessage(role="user", content="hi")]
        self.assertEqual(trim_history(history, _settings(max_history_turns=0)), [])


class TestRagServiceAnswer(unittest.TestCase):
    def test_answers_without_document(self):
        provider = FakeLLMProvider()
        service = RagService(_settings(), provider, vectorstore=None)

        result = service.answer("What skills matter for data science?")

        self.assertFalse(result.used_retrieval)
        self.assertEqual(result.sources, [])
        self.assertEqual(result.text, "Here is your answer.")

    def test_retrieves_when_document_present_and_needed(self):
        provider = FakeLLMProvider(needs_retrieval=True)
        vectorstore = MagicMock()
        vectorstore.similarity_search_with_relevance_scores.return_value = [
            (SimpleNamespace(page_content="relevant excerpt", metadata={"page": 3}), 0.8),
        ]
        service = RagService(_settings(), provider, vectorstore=vectorstore)

        result = service.answer("What does the guide say about resumes?")

        self.assertTrue(result.used_retrieval)
        self.assertEqual(len(result.sources), 1)
        self.assertIn("relevant excerpt", provider.last_system_instruction)

    def test_skips_retrieval_when_llm_decides_not_needed(self):
        provider = FakeLLMProvider(needs_retrieval=False)
        vectorstore = MagicMock()
        service = RagService(_settings(), provider, vectorstore=vectorstore)

        result = service.answer("What should I wear to an interview?")

        vectorstore.similarity_search_with_relevance_scores.assert_not_called()
        self.assertFalse(result.used_retrieval)
        self.assertEqual(result.sources, [])

    def test_history_is_passed_through_to_provider(self):
        provider = FakeLLMProvider()
        service = RagService(_settings(), provider, vectorstore=None)
        history = [ChatMessage(role="user", content="first question")]

        service.answer("follow-up question", history=history)

        contents = [m.content for m in provider.last_messages]
        self.assertIn("first question", contents)
        self.assertIn("follow-up question", contents)

    def test_invalid_query_raises_before_calling_provider(self):
        provider = FakeLLMProvider()
        service = RagService(_settings(), provider, vectorstore=None)

        with self.assertRaises(QueryValidationError):
            service.answer("   ")
        self.assertIsNone(provider.last_messages)


if __name__ == "__main__":
    unittest.main()
