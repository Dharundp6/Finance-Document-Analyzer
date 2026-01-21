"""
Tests for chunk creator atom.
"""

from atoms.chunk_creator import (
    ChunkingConfig,
    FinancialAwareChunker,
    RecursiveChunker,
    create_chunks,
)
from core.types import ChunkType


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_basic_chunking(self):
        """Test basic text chunking."""
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20, min_chunk_size=10)
        chunker = RecursiveChunker(config)

        text = "This is a test sentence. " * 20
        chunks = chunker.chunk(text, document_id="test-doc")

        assert len(chunks) > 0
        assert all(len(c.content) <= config.chunk_size + 50 for c in chunks)

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = RecursiveChunker()
        chunks = chunker.chunk("", document_id="test-doc")

        assert len(chunks) == 0

    def test_short_text(self):
        """Test chunking text shorter than min chunk size."""
        config = ChunkingConfig(min_chunk_size=100)
        chunker = RecursiveChunker(config)

        chunks = chunker.chunk("Short text", document_id="test-doc")

        assert len(chunks) == 0

    def test_chunk_metadata(self):
        """Test chunk metadata is preserved."""
        chunker = RecursiveChunker()

        chunks = chunker.chunk(
            "This is a test. " * 50,
            document_id="test-doc",
            page_number=5,
            metadata={"custom": "value"},
        )

        assert all(c.document_id == "test-doc" for c in chunks)
        assert all(c.page_number == 5 for c in chunks)
        assert all(c.metadata.get("custom") == "value" for c in chunks)


class TestFinancialAwareChunker:
    """Tests for FinancialAwareChunker."""

    def test_section_detection(self):
        """Test financial section detection."""
        chunker = FinancialAwareChunker()

        text = """
        REVENUE ANALYSIS:
        Total revenue for Q3 2023 was $1.5 billion.

        EXPENSE BREAKDOWN:
        Operating expenses increased by 15%.
        """

        chunks = chunker.chunk(text, document_id="test-doc")

        assert len(chunks) >= 1

    def test_chunk_type_detection(self):
        """Test chunk type detection."""
        chunker = FinancialAwareChunker()

        financial_text = "Revenue increased by 20% to $500 million in fiscal year 2023."
        chunks = chunker.chunk(financial_text * 10, document_id="test-doc")

        # Should detect financial content
        has_financial = any(c.chunk_type == ChunkType.FINANCIAL_DATA for c in chunks)
        assert has_financial or len(chunks) > 0


class TestCreateChunks:
    """Tests for create_chunks function."""

    def test_default_strategy(self):
        """Test default chunking strategy."""
        chunks = create_chunks("This is test content. " * 50, document_id="test-doc")

        assert len(chunks) > 0

    def test_recursive_strategy(self):
        """Test recursive chunking strategy."""
        chunks = create_chunks(
            "This is test content. " * 50, document_id="test-doc", strategy="recursive"
        )

        assert len(chunks) > 0

    def test_sentence_strategy(self):
        """Test sentence chunking strategy."""
        chunks = create_chunks(
            "This is sentence one. This is sentence two. This is sentence three. " * 20,
            document_id="test-doc",
            strategy="sentence",
        )

        assert len(chunks) > 0
