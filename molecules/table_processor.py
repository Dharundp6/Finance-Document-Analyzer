"""
Molecule component for processing tables from financial documents.
Handles table extraction, normalization, and conversion to queryable formats.
"""

import re
import uuid
from dataclasses import dataclass
from typing import Any

import pandas as pd

from atoms.financial_extractor import FinancialExtractor
from atoms.text_cleaner import TextCleaner
from config.logging_config import logger
from core.types import ChunkType, TableData, TextChunk


@dataclass
class TableProcessingConfig:
    """Configuration for table processing."""

    min_rows: int = 2
    min_columns: int = 2
    max_rows_per_chunk: int = 50
    include_summary: bool = True
    detect_header: bool = True
    normalize_numbers: bool = True
    extract_financial_entities: bool = True


class TableNormalizer:
    """Normalizes table data for consistent processing."""

    # Common financial header patterns
    HEADER_PATTERNS = [
        r"(?i)^(date|period|year|quarter|month)$",
        r"(?i)^(revenue|sales|income|profit|loss)$",
        r"(?i)^(expense|cost|expenditure)$",
        r"(?i)^(asset|liability|equity)$",
        r"(?i)^(total|subtotal|net|gross)$",
        r"(?i)^(q[1-4]|fy\s*\d{2,4})$",
        r"(?i)^(actual|budget|forecast|variance)$",
        r"(?i)^(ytd|mtd|qtd)$",
        r"(?i)^(amount|value|balance)$",
        r"(?i)^(%|percent|ratio|margin)$",
    ]

    # Number patterns
    NUMBER_PATTERN = re.compile(r"^[\$€£¥₹]?\s*[\(\-]?\s*[\d,]+\.?\d*\s*[\)]?\s*[%MBKTmk]?$")
    CURRENCY_PATTERN = re.compile(r"[\$€£¥₹]")
    PERCENTAGE_PATTERN = re.compile(r"%\s*$")

    def __init__(self):
        self.text_cleaner = TextCleaner()

    def normalize_table(
        self, headers: list[str], rows: list[list[Any]], detect_header: bool = True
    ) -> tuple[list[str], list[list[Any]]]:
        """
        Normalize table headers and data.

        Args:
            headers: Table headers
            rows: Table rows
            detect_header: Whether to detect if first row is header

        Returns:
            Tuple of (normalized_headers, normalized_rows)
        """
        # Clean headers
        normalized_headers = [self._clean_header(h) for h in headers]

        # Check if headers look like data
        if detect_header and self._headers_look_like_data(normalized_headers) and rows:
            # First row might be actual header
            potential_headers = [self._clean_header(str(cell)) for cell in rows[0]]
            if self._is_likely_header(potential_headers):
                normalized_headers = potential_headers
                rows = rows[1:]

        # Generate headers if empty
        if not normalized_headers or all(not h for h in normalized_headers):
            normalized_headers = [f"Column_{i + 1}" for i in range(len(rows[0]) if rows else 0)]

        # Normalize rows
        normalized_rows = []
        for row in rows:
            normalized_row = [self._normalize_cell(cell) for cell in row]

            # Pad or trim row to match header count
            if len(normalized_row) < len(normalized_headers):
                normalized_row.extend([""] * (len(normalized_headers) - len(normalized_row)))
            elif len(normalized_row) > len(normalized_headers):
                normalized_row = normalized_row[: len(normalized_headers)]

            normalized_rows.append(normalized_row)

        return normalized_headers, normalized_rows

    def _clean_header(self, header: Any) -> str:
        """Clean and normalize a header."""
        if header is None:
            return ""

        header_str = str(header).strip()

        # Remove excessive whitespace
        header_str = " ".join(header_str.split())

        # Remove common prefixes/suffixes
        header_str = re.sub(r"^[\*\#\-\•]\s*", "", header_str)
        header_str = re.sub(r"\s*[\*\#]$", "", header_str)

        return header_str

    def _normalize_cell(self, cell: Any) -> str:
        """Normalize a cell value."""
        if cell is None:
            return ""

        if isinstance(cell, (int, float)):
            if pd.isna(cell):
                return ""
            # Format numbers consistently
            if isinstance(cell, float) and cell == int(cell):
                return str(int(cell))
            return str(cell)

        cell_str = str(cell).strip()

        # Handle common null representations
        if cell_str.lower() in ["nan", "none", "null", "n/a", "na", "-", "--", "—"]:
            return ""

        return cell_str

    def _headers_look_like_data(self, headers: list[str]) -> bool:
        """Check if headers look like data values."""
        if not headers:
            return False

        # If most headers are numbers, they're probably data
        numeric_count = sum(1 for h in headers if self.NUMBER_PATTERN.match(h))
        return numeric_count > len(headers) * 0.5

    def _is_likely_header(self, row: list[str]) -> bool:
        """Check if a row is likely to be a header."""
        if not row:
            return False

        # Check against header patterns
        pattern_matches = 0
        for cell in row:
            for pattern in self.HEADER_PATTERNS:
                if re.match(pattern, cell):
                    pattern_matches += 1
                    break

        # If more than half match header patterns
        if pattern_matches > len(row) * 0.3:
            return True

        # If row contains mostly non-numeric values
        non_numeric = sum(1 for cell in row if not self.NUMBER_PATTERN.match(cell))
        return non_numeric > len(row) * 0.7

    def detect_column_types(self, headers: list[str], rows: list[list[str]]) -> dict[str, str]:
        """
        Detect the type of each column.

        Args:
            headers: Column headers
            rows: Data rows

        Returns:
            Dictionary mapping header to type
        """
        column_types = {}

        for i, header in enumerate(headers):
            values = [row[i] for row in rows if i < len(row) and row[i]]

            if not values:
                column_types[header] = "empty"
                continue

            # Check for currency
            currency_count = sum(1 for v in values if self.CURRENCY_PATTERN.search(v))
            if currency_count > len(values) * 0.5:
                column_types[header] = "currency"
                continue

            # Check for percentage
            percent_count = sum(1 for v in values if self.PERCENTAGE_PATTERN.search(v))
            if percent_count > len(values) * 0.5:
                column_types[header] = "percentage"
                continue

            # Check for numeric
            numeric_count = sum(1 for v in values if self.NUMBER_PATTERN.match(v))
            if numeric_count > len(values) * 0.7:
                column_types[header] = "numeric"
                continue

            # Check for date
            date_patterns = [r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}", r"Q[1-4]\s*\d{4}", r"FY\s*\d{4}"]
            date_count = sum(
                1 for v in values if any(re.search(p, v, re.IGNORECASE) for p in date_patterns)
            )
            if date_count > len(values) * 0.5:
                column_types[header] = "date"
                continue

            column_types[header] = "text"

        return column_types


class TableAnalyzer:
    """Analyzes tables to extract insights and summaries."""

    def __init__(self):
        self.financial_extractor = FinancialExtractor()

    def analyze_table(
        self, table: TableData, config: TableProcessingConfig = None
    ) -> dict[str, Any]:
        """
        Analyze a table and extract insights.

        Args:
            table: Table data to analyze
            config: Processing configuration

        Returns:
            Dictionary with analysis results
        """
        config = config or TableProcessingConfig()

        # Convert to DataFrame for easier analysis
        df = self._table_to_dataframe(table)

        if df.empty:
            return {"error": "Empty table"}

        analysis = {
            "table_id": table.table_id,
            "dimensions": {"rows": len(df), "columns": len(df.columns)},
            "columns": list(df.columns),
            "column_types": {},
            "numeric_summary": {},
            "financial_entities": [],
            "key_metrics": [],
        }

        # Analyze each column
        for col in df.columns:
            col_data = df[col].dropna()

            if col_data.empty:
                analysis["column_types"][col] = "empty"
                continue

            # Try to convert to numeric
            numeric_data = pd.to_numeric(col_data, errors="coerce")
            non_null_numeric = numeric_data.dropna()

            if len(non_null_numeric) > len(col_data) * 0.5:
                analysis["column_types"][col] = "numeric"
                analysis["numeric_summary"][col] = {
                    "min": float(non_null_numeric.min()),
                    "max": float(non_null_numeric.max()),
                    "mean": float(non_null_numeric.mean()),
                    "sum": float(non_null_numeric.sum()),
                    "count": int(len(non_null_numeric)),
                }
            else:
                analysis["column_types"][col] = "text"

        # Extract financial entities
        if config.extract_financial_entities:
            table_text = table.to_markdown()
            analysis["financial_entities"] = self.financial_extractor.extract_all(table_text)

        # Identify key metrics
        analysis["key_metrics"] = self._identify_key_metrics(df, analysis["numeric_summary"])

        return analysis

    def _table_to_dataframe(self, table: TableData) -> pd.DataFrame:
        """Convert TableData to pandas DataFrame."""
        if not table.headers or not table.rows:
            return pd.DataFrame()

        try:
            df = pd.DataFrame(table.rows, columns=table.headers)
            return df
        except Exception as e:
            logger.error(f"Error converting table to DataFrame: {e}")
            return pd.DataFrame()

    def _identify_key_metrics(
        self, df: pd.DataFrame, numeric_summary: dict[str, dict]
    ) -> list[dict[str, Any]]:
        """Identify key financial metrics in the table."""
        key_metrics = []

        # Keywords that indicate important metrics
        important_keywords = [
            "total",
            "net",
            "gross",
            "revenue",
            "income",
            "profit",
            "loss",
            "ebitda",
            "ebit",
            "margin",
            "growth",
            "ratio",
        ]

        for col in df.columns:
            col_lower = col.lower()

            # Check if column name contains important keywords
            for keyword in important_keywords:
                if keyword in col_lower:
                    if col in numeric_summary:
                        key_metrics.append(
                            {"metric": col, "keyword": keyword, "summary": numeric_summary[col]}
                        )
                    break

        # Also check row labels if first column looks like labels
        if len(df.columns) > 1:
            first_col = df.columns[0]
            if df[first_col].dtype == object:  # Text column
                for idx, value in df[first_col].items():
                    if isinstance(value, str):
                        value_lower = value.lower()
                        for keyword in important_keywords:
                            if keyword in value_lower:
                                # Get values from other columns in this row
                                row_values = {}
                                for col in df.columns[1:]:
                                    cell_value = df.loc[idx, col]
                                    if pd.notna(cell_value):
                                        row_values[col] = cell_value

                                if row_values:
                                    key_metrics.append(
                                        {"metric": value, "keyword": keyword, "values": row_values}
                                    )
                                break

        return key_metrics


class TableProcessor:
    """
    Main processor for tables from financial documents.
    Combines normalization, analysis, and chunk generation.
    """

    def __init__(self, config: TableProcessingConfig | None = None):
        """
        Initialize the table processor.

        Args:
            config: Processing configuration
        """
        self.config = config or TableProcessingConfig()
        self.normalizer = TableNormalizer()
        self.analyzer = TableAnalyzer()

        logger.info("TableProcessor initialized")

    def process_table(
        self,
        headers: list[str],
        rows: list[list[Any]],
        document_id: str,
        page_number: int | None = None,
        table_index: int = 0,
        caption: str | None = None,
    ) -> tuple[TableData, list[TextChunk], dict[str, Any]]:
        """
        Process a table and generate chunks.

        Args:
            headers: Table headers
            rows: Table rows
            document_id: Source document ID
            page_number: Page number if applicable
            table_index: Index of table in document
            caption: Table caption/title

        Returns:
            Tuple of (TableData, chunks, analysis)
        """
        logger.debug(f"Processing table {table_index} from document {document_id}")

        # Validate table
        if not self._validate_table(headers, rows):
            logger.warning(f"Table {table_index} failed validation, skipping")
            return None, [], {}

        # Normalize table
        normalized_headers, normalized_rows = self.normalizer.normalize_table(
            headers, rows, detect_header=self.config.detect_header
        )

        # Create TableData object
        table_id = str(uuid.uuid4())
        table_data = TableData(
            table_id=table_id,
            headers=normalized_headers,
            rows=normalized_rows,
            document_id=document_id,
            page_number=page_number,
            table_index=table_index,
            caption=caption,
        )

        # Analyze table
        analysis = self.analyzer.analyze_table(table_data, self.config)

        # Generate chunks
        chunks = self._generate_chunks(table_data, analysis)

        logger.debug(
            f"Processed table: {len(normalized_rows)} rows, "
            f"{len(normalized_headers)} columns, {len(chunks)} chunks"
        )

        return table_data, chunks, analysis

    def process_dataframe(
        self,
        df: pd.DataFrame,
        document_id: str,
        sheet_name: str | None = None,
        table_index: int = 0,
    ) -> tuple[TableData, list[TextChunk], dict[str, Any]]:
        """
        Process a pandas DataFrame.

        Args:
            df: DataFrame to process
            document_id: Source document ID
            sheet_name: Sheet name for Excel files
            table_index: Index of table

        Returns:
            Tuple of (TableData, chunks, analysis)
        """
        headers = df.columns.tolist()
        rows = df.values.tolist()

        return self.process_table(
            headers=headers,
            rows=rows,
            document_id=document_id,
            caption=sheet_name,
            table_index=table_index,
        )

    def _validate_table(self, headers: list[str], rows: list[list[Any]]) -> bool:
        """Validate if table meets minimum requirements."""
        # Check minimum dimensions
        if len(rows) < self.config.min_rows:
            return False

        # Check columns (from headers or first row)
        num_cols = len(headers) if headers else (len(rows[0]) if rows else 0)
        if num_cols < self.config.min_columns:
            return False

        # Check if table has any actual content
        total_cells = sum(len(row) for row in rows)
        non_empty_cells = sum(
            1 for row in rows for cell in row if cell is not None and str(cell).strip()
        )

        # Less than 10% filled is not worth processing
        return non_empty_cells >= total_cells * 0.1

    def _generate_chunks(self, table: TableData, analysis: dict[str, Any]) -> list[TextChunk]:
        """Generate text chunks from a table."""
        chunks = []
        base_metadata = {
            "table_id": table.table_id,
            "page_number": table.page_number,
            "table_index": table.table_index,
            "caption": table.caption,
        }

        # 1. Schema chunk
        schema_chunk = self._create_schema_chunk(table, analysis, base_metadata)
        chunks.append(schema_chunk)

        # 2. Summary chunk (if enabled)
        if self.config.include_summary and analysis.get("numeric_summary"):
            summary_chunk = self._create_summary_chunk(table, analysis, base_metadata)
            chunks.append(summary_chunk)

        # 3. Key metrics chunk
        if analysis.get("key_metrics"):
            metrics_chunk = self._create_metrics_chunk(table, analysis, base_metadata)
            chunks.append(metrics_chunk)

        # 4. Data chunks (split large tables)
        data_chunks = self._create_data_chunks(table, base_metadata)
        chunks.extend(data_chunks)

        return chunks

    def _create_schema_chunk(
        self, table: TableData, analysis: dict[str, Any], base_metadata: dict
    ) -> TextChunk:
        """Create a chunk describing the table schema."""
        content_parts = []

        # Title/caption
        if table.caption:
            content_parts.append(f"Table: {table.caption}")
        else:
            content_parts.append(f"Table {table.table_index + 1}")

        # Dimensions
        dims = analysis.get("dimensions", {})
        content_parts.append(
            f"Dimensions: {dims.get('rows', 0)} rows × {dims.get('columns', 0)} columns"
        )

        # Column information
        content_parts.append("\nColumns:")
        column_types = analysis.get("column_types", {})
        for header in table.headers:
            col_type = column_types.get(header, "unknown")
            content_parts.append(f"  - {header} ({col_type})")

        return TextChunk(
            chunk_id=str(uuid.uuid4()),
            content="\n".join(content_parts),
            chunk_type=ChunkType.TABLE,
            document_id=table.document_id,
            page_number=table.page_number,
            metadata={**base_metadata, "chunk_subtype": "schema"},
        )

    def _create_summary_chunk(
        self, table: TableData, analysis: dict[str, Any], base_metadata: dict
    ) -> TextChunk:
        """Create a chunk with numeric summaries."""
        content_parts = []

        if table.caption:
            content_parts.append(f"Summary Statistics for {table.caption}:")
        else:
            content_parts.append("Table Summary Statistics:")

        numeric_summary = analysis.get("numeric_summary", {})

        for col, stats in numeric_summary.items():
            content_parts.append(f"\n{col}:")
            content_parts.append(f"  - Sum: {stats.get('sum', 'N/A'):,.2f}")
            content_parts.append(f"  - Average: {stats.get('mean', 'N/A'):,.2f}")
            content_parts.append(f"  - Min: {stats.get('min', 'N/A'):,.2f}")
            content_parts.append(f"  - Max: {stats.get('max', 'N/A'):,.2f}")

        return TextChunk(
            chunk_id=str(uuid.uuid4()),
            content="\n".join(content_parts),
            chunk_type=ChunkType.TABLE,
            document_id=table.document_id,
            page_number=table.page_number,
            metadata={**base_metadata, "chunk_subtype": "summary"},
        )

    def _create_metrics_chunk(
        self, table: TableData, analysis: dict[str, Any], base_metadata: dict
    ) -> TextChunk:
        """Create a chunk with key financial metrics."""
        content_parts = []

        content_parts.append("Key Financial Metrics:")

        for metric in analysis.get("key_metrics", []):
            metric_name = metric.get("metric", "Unknown")

            if "summary" in metric:
                stats = metric["summary"]
                content_parts.append(f"\n{metric_name}:")
                content_parts.append(f"  Total: {stats.get('sum', 'N/A'):,.2f}")
                content_parts.append(f"  Average: {stats.get('mean', 'N/A'):,.2f}")

            if "values" in metric:
                content_parts.append(f"\n{metric_name}:")
                for col, value in metric["values"].items():
                    content_parts.append(f"  {col}: {value}")

        return TextChunk(
            chunk_id=str(uuid.uuid4()),
            content="\n".join(content_parts),
            chunk_type=ChunkType.FINANCIAL_DATA,
            document_id=table.document_id,
            page_number=table.page_number,
            metadata={**base_metadata, "chunk_subtype": "metrics"},
        )

    def _create_data_chunks(self, table: TableData, base_metadata: dict) -> list[TextChunk]:
        """Create chunks from table data rows."""
        chunks = []

        # If table is small enough, create one chunk with markdown
        if len(table.rows) <= self.config.max_rows_per_chunk:
            markdown = table.to_markdown()
            if markdown:
                chunks.append(
                    TextChunk(
                        chunk_id=str(uuid.uuid4()),
                        content=markdown,
                        chunk_type=ChunkType.TABLE,
                        document_id=table.document_id,
                        page_number=table.page_number,
                        metadata={**base_metadata, "chunk_subtype": "full_data"},
                    )
                )
        else:
            # Split into multiple chunks
            for i in range(0, len(table.rows), self.config.max_rows_per_chunk):
                batch_rows = table.rows[i : i + self.config.max_rows_per_chunk]

                # Create mini table
                mini_table = TableData(
                    table_id=table.table_id,
                    headers=table.headers,
                    rows=batch_rows,
                    document_id=table.document_id,
                    page_number=table.page_number,
                    table_index=table.table_index,
                )

                markdown = mini_table.to_markdown()
                if markdown:
                    chunks.append(
                        TextChunk(
                            chunk_id=str(uuid.uuid4()),
                            content=markdown,
                            chunk_type=ChunkType.TABLE,
                            document_id=table.document_id,
                            page_number=table.page_number,
                            metadata={
                                **base_metadata,
                                "chunk_subtype": "partial_data",
                                "row_start": i,
                                "row_end": i + len(batch_rows),
                            },
                        )
                    )

        # Also create row-by-row representation for detailed queries
        for row_idx, row in enumerate(table.rows):
            row_content = self._format_row(table.headers, row, row_idx)

            chunks.append(
                TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    content=row_content,
                    chunk_type=ChunkType.TABLE,
                    document_id=table.document_id,
                    page_number=table.page_number,
                    metadata={**base_metadata, "chunk_subtype": "row", "row_index": row_idx},
                )
            )

        return chunks

    def _format_row(self, headers: list[str], row: list[Any], row_idx: int) -> str:
        """Format a single row as readable text."""
        parts = [f"Row {row_idx + 1}:"]

        for header, value in zip(headers, row, strict=False):
            if value and str(value).strip():
                parts.append(f"  {header}: {value}")

        return "\n".join(parts)

    def compare_tables(self, table1: TableData, table2: TableData) -> dict[str, Any]:
        """
        Compare two tables (e.g., for year-over-year analysis).

        Args:
            table1: First table
            table2: Second table

        Returns:
            Comparison results
        """
        comparison = {
            "table1_id": table1.table_id,
            "table2_id": table2.table_id,
            "matching_columns": [],
            "differences": [],
        }

        # Find matching columns
        common_headers = set(table1.headers) & set(table2.headers)
        comparison["matching_columns"] = list(common_headers)

        # Convert to DataFrames for easier comparison
        df1 = pd.DataFrame(table1.rows, columns=table1.headers)
        df2 = pd.DataFrame(table2.rows, columns=table2.headers)

        # Compare numeric columns
        for col in common_headers:
            if col in df1.columns and col in df2.columns:
                try:
                    vals1 = pd.to_numeric(df1[col], errors="coerce").dropna()
                    vals2 = pd.to_numeric(df2[col], errors="coerce").dropna()

                    if len(vals1) > 0 and len(vals2) > 0:
                        sum1 = vals1.sum()
                        sum2 = vals2.sum()

                        if sum1 != 0:
                            change_pct = ((sum2 - sum1) / abs(sum1)) * 100
                        else:
                            change_pct = 0 if sum2 == 0 else float("inf")

                        comparison["differences"].append(
                            {
                                "column": col,
                                "table1_sum": float(sum1),
                                "table2_sum": float(sum2),
                                "change": float(sum2 - sum1),
                                "change_percent": float(change_pct),
                            }
                        )
                except Exception as e:
                    logger.debug(f"Could not compare column {col}: {e}")

        return comparison


def process_table(
    headers: list[str], rows: list[list[Any]], document_id: str, **kwargs
) -> tuple[TableData, list[TextChunk], dict[str, Any]]:
    """
    Convenience function to process a table.

    Args:
        headers: Table headers
        rows: Table rows
        document_id: Source document ID
        **kwargs: Additional arguments

    Returns:
        Tuple of (TableData, chunks, analysis)
    """
    processor = TableProcessor()
    return processor.process_table(headers, rows, document_id, **kwargs)
