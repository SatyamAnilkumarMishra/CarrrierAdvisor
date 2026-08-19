import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from errors import DocumentValidationError, VectorStoreError
from rag_pipeline import retrieve_relevant_chunks, validate_pdf


def _settings(**overrides):
    defaults = dict(
        max_upload_size_bytes=5 * 1024 * 1024, max_context_docs=3, relevance_score_threshold=0.35
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestValidatePdf(unittest.TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(DocumentValidationError):
            validate_pdf("/tmp/does-not-exist-12345.pdf", max_size_bytes=1_000_000)

    def test_empty_file_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            path = f.name
        try:
            with self.assertRaises(DocumentValidationError):
                validate_pdf(path, max_size_bytes=1_000_000)
        finally:
            os.remove(path)

    def test_oversized_file_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\n" + b"0" * 2000)
            path = f.name
        try:
            with self.assertRaises(DocumentValidationError):
                validate_pdf(path, max_size_bytes=100)
        finally:
            os.remove(path)

    def test_wrong_header_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"not a real pdf file at all")
            path = f.name
        try:
            with self.assertRaises(DocumentValidationError):
                validate_pdf(path, max_size_bytes=1_000_000)
        finally:
            os.remove(path)

    def test_valid_pdf_header_passes(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\n%%EOF")
            path = f.name
        try:
            validate_pdf(path, max_size_bytes=1_000_000)  # should not raise
        finally:
            os.remove(path)


class TestRetrieveRelevantChunks(unittest.TestCase):
    def _fake_doc(self, content, page, source="doc.pdf"):
        return SimpleNamespace(page_content=content, metadata={"page": page, "source": source})

    def test_filters_below_threshold(self):
        vectorstore = MagicMock()
        vectorstore.similarity_search_with_relevance_scores.return_value = [
            (self._fake_doc("relevant chunk", 1), 0.9),
            (self._fake_doc("borderline chunk", 2), 0.2),
        ]
        chunks = retrieve_relevant_chunks(
            vectorstore, "some query", _settings(relevance_score_threshold=0.5)
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "relevant chunk")
        self.assertEqual(chunks[0].page, 1)

    def test_returns_empty_when_nothing_relevant(self):
        vectorstore = MagicMock()
        vectorstore.similarity_search_with_relevance_scores.return_value = [
            (self._fake_doc("off topic", 1), 0.05),
        ]
        chunks = retrieve_relevant_chunks(
            vectorstore, "some query", _settings(relevance_score_threshold=0.5)
        )
        self.assertEqual(chunks, [])

    def test_wraps_backend_errors(self):
        vectorstore = MagicMock()
        vectorstore.similarity_search_with_relevance_scores.side_effect = RuntimeError(
            "chroma exploded"
        )
        with self.assertRaises(VectorStoreError):
            retrieve_relevant_chunks(vectorstore, "query", _settings())


if __name__ == "__main__":
    unittest.main()
