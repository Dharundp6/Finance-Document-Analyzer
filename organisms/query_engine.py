"""
Organism component for query processing.
Handles query understanding and context building.
"""

import re
import uuid
from typing import Any

from atoms.financial_extractor import FinancialExtractor
from config.logging_config import logger
from config.settings import settings
from core.types import FinancialEntity, QueryContext, RetrievalResult
from molecules.retriever import HybridRetriever
from molecules.vector_store import VectorStore


class QueryEngine:
    """
    Query processing engine.
    Handles query analysis, retrieval, and context building.
    """

    # Query type patterns
    QUERY_PATTERNS = {
        "comparison": [
            r"\bcompare\b",
            r"\bversus\b",
            r"\bvs\.?\b",
            r"\bdifference\b",
            r"\bhigher\b",
            r"\blower\b",
        ],
        "trend": [
            r"\btrend\b",
            r"\bover time\b",
            r"\bgrowth\b",
            r"\bincrease\b",
            r"\bdecrease\b",
            r"\bchange\b",
        ],
        "aggregation": [
            r"\btotal\b",
            r"\bsum\b",
            r"\baverage\b",
            r"\bcount\b",
            r"\bhow many\b",
            r"\bhow much\b",
        ],
        "lookup": [
            r"\bwhat is\b",
            r"\bwhat was\b",
            r"\bwhat are\b",
            r"\bshow me\b",
            r"\bfind\b",
            r"\blist\b",
        ],
        "analytical": [
            r"\bwhy\b",
            r"\bexplain\b",
            r"\banalyze\b",
            r"\bimpact\b",
            r"\bcause\b",
            r"\breason\b",
        ],
    }

    def __init__(
        self, vector_store: VectorStore | None = None, retriever: HybridRetriever | None = None
    ):
        """
        Initialize the query engine.

        Args:
            vector_store: Vector store instance
            retriever: Hybrid retriever instance
        """
        self.vector_store = vector_store or VectorStore()
        self.retriever = retriever or HybridRetriever(self.vector_store)
        self.financial_extractor = FinancialExtractor()

        logger.info("QueryEngine initialized")

    def process_query(
        self, query: str, document_ids: list[str] | None = None, top_k: int = None
    ) -> tuple[QueryContext, list[RetrievalResult]]:
        """
        Process a query and retrieve relevant context.

        Args:
            query: User query
            document_ids: Limit search to specific documents
            top_k: Number of results to retrieve

        Returns:
            Tuple of (query context, retrieval results)
        """
        top_k = top_k or settings.top_k

        logger.info(f"Processing query: {query[:100]}...")

        # Analyze query
        query_context = self._analyze_query(query)

        # Build filters
        filters = {}
        if document_ids:
            filters["document_id"] = {"$in": document_ids}

        # Retrieve relevant chunks
        results = self.retriever.retrieve(
            query=query_context.processed_query or query,
            top_k=top_k,
            filter_dict=filters if filters else None,
        )

        logger.info(f"Retrieved {len(results)} results for query")

        return query_context, results

    def get_context_for_generation(
        self, query: str, document_ids: list[str] | None = None, max_context_length: int = 8000
    ) -> str:
        """
        Get formatted context string for response generation.

        Args:
            query: User query
            document_ids: Limit to specific documents
            max_context_length: Maximum context length

        Returns:
            Formatted context string
        """
        query_context, results = self.process_query(query, document_ids)

        # Build context string
        context_parts = []
        current_length = 0

        for result in results:
            chunk_text = self._format_chunk_for_context(result)
            chunk_length = len(chunk_text)

            if current_length + chunk_length > max_context_length:
                break

            context_parts.append(chunk_text)
            current_length += chunk_length

        context = "\n\n---\n\n".join(context_parts)

        return context

    def _analyze_query(self, query: str) -> QueryContext:
        """
        Analyze query to understand intent and extract entities.

        Args:
            query: User query

        Returns:
            Query context with analysis
        """
        query_id = str(uuid.uuid4())

        # Detect query type
        query_type = self._detect_query_type(query)

        # Extract financial entities
        entities = self.financial_extractor.extract_all(query)

        # Process query (expand/refine)
        processed_query = self._process_query_text(query, query_type)

        # Extract potential filters
        filters = self._extract_filters(query, entities)

        return QueryContext(
            query_id=query_id,
            original_query=query,
            processed_query=processed_query,
            query_type=query_type,
            entities=entities,
            filters=filters,
        )

    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query."""
        query_lower = query.lower()

        for query_type, patterns in self.QUERY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type

        return "general"

    def _process_query_text(self, query: str, query_type: str) -> str:
        """Process and potentially expand the query."""
        processed = query

        # Add context based on query type
        if query_type == "comparison":
            # Ensure comparison terms are captured
            processed = f"{query} comparison analysis"
        elif query_type == "trend":
            processed = f"{query} over time historical"
        elif query_type == "aggregation":
            processed = f"{query} total summary"

        return processed

    def _extract_filters(self, query: str, entities: list[FinancialEntity]) -> dict[str, Any]:
        """Extract potential filters from query and entities."""
        filters = {}

        # Extract date filters
        date_entities = [e for e in entities if e.entity_type == "date"]
        if date_entities:
            filters["date_mentioned"] = True

        # Extract document type preferences
        if "balance sheet" in query.lower():
            filters["preferred_type"] = "balance_sheet"
        elif "income statement" in query.lower():
            filters["preferred_type"] = "income_statement"
        elif "cash flow" in query.lower():
            filters["preferred_type"] = "cash_flow"

        return filters

    def _format_chunk_for_context(self, result: RetrievalResult) -> str:
        """Format a retrieval result for context."""
        chunk = result.chunk

        # Build header with source info
        header_parts = []

        if chunk.metadata.get("document_id"):
            header_parts.append(f"Document: {chunk.document_id[:8]}...")

        if chunk.page_number:
            header_parts.append(f"Page: {chunk.page_number}")

        if chunk.metadata.get("sheet_name"):
            header_parts.append(f"Sheet: {chunk.metadata['sheet_name']}")

        header = " | ".join(header_parts) if header_parts else "Source"

        # Format content
        formatted = f"[{header}]\n{chunk.content}"

        return formatted
