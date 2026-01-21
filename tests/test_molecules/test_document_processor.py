"""
Tests for document processor molecule.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from molecules.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Tests for DocumentProcessor."""

    @pytest.fixture
    def processor(self):
        """Create a document processor instance."""
        return DocumentProcessor()

    @pytest.fixture
    def sample_csv(self):
        """Create a sample CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df = pd.DataFrame(
                {
                    "Company": ["ACME Corp", "Tech Inc", "Finance Ltd"],
                    "Revenue": [1000000, 2500000, 1800000],
                    "Profit": [100000, 350000, 220000],
                }
            )
            df.to_csv(f.name, index=False)
            yield Path(f.name)

    @pytest.fixture
    def sample_txt(self):
        """Create a sample text file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("""
            Financial Report Q3 2023

            Revenue: $1.5 million
            Expenses: $1.2 million
            Net Profit: $300,000

            Year-over-year growth: 15%
            """)
            yield Path(f.name)

    def test_process_csv(self, processor, sample_csv):
        """Test processing CSV file."""
        result = processor.process(sample_csv)

        assert result.success
        assert len(result.chunks) > 0
        assert len(result.tables) > 0

    def test_process_txt(self, processor, sample_txt):
        """Test processing text file."""
        result = processor.process(sample_txt)

        assert result.success
        assert len(result.chunks) > 0

    def test_financial_entity_extraction(self, processor, sample_txt):
        """Test financial entity extraction."""
        result = processor.process(sample_txt, extract_financial_entities=True)

        assert result.success
        assert len(result.financial_entities) > 0

        # Check for currency entities
        currency_entities = [e for e in result.financial_entities if "currency" in e.entity_type]
        assert len(currency_entities) > 0

    def test_invalid_file(self, processor):
        """Test processing non-existent file."""
        result = processor.process(Path("/nonexistent/file.pdf"))

        assert not result.success
        assert len(result.errors) > 0
