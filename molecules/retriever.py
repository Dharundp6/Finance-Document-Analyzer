"""
Hybrid retriever combining FAISS vector search with BM25.
"""

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from config.logging_config import logger
from config.settings import settings
from core.types import RetrievalResult, TextChunk
from molecules.vector_store import FAISSVectorStore  # ← Changed import


@dataclass
class RetrieverConfig:
    """Configuration for the retriever."""

    top_k: int = 10
    rerank_top_k: int = 5
    similarity_threshold: float = 0.3
    use_hybrid: bool = True
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    use_reranking: bool = True


class BM25Calculator:
    """BM25 scoring for keyword matching."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.doc_freqs = defaultdict(int)
        self.total_docs = 0
        self.index = defaultdict(list)

    def fit(self, documents: list[tuple[str, str]]) -> None:
        """Build BM25 index from documents."""
        self.total_docs = len(documents)
        total_length = 0

        for doc_id, text in documents:
            tokens = self._tokenize(text)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            term_freqs = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1

            for token, freq in term_freqs.items():
                self.index[token].append((doc_id, freq))
                self.doc_freqs[token] += 1

        self.avg_doc_length = total_length / max(self.total_docs, 1)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search using BM25."""
        query_tokens = self._tokenize(query)
        scores = defaultdict(float)

        for token in query_tokens:
            if token not in self.index:
                continue

            df = self.doc_freqs[token]
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf in self.index[token]:
                doc_len = self.doc_lengths[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                scores[doc_id] += idf * numerator / denominator

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens


class HybridRetriever:
    """
    Hybrid retriever combining FAISS and BM25.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,  # ← Changed type
        config: RetrieverConfig | None = None,
    ):
        """Initialize the hybrid retriever."""
        self.vector_store = vector_store
        self.config = config or RetrieverConfig(
            top_k=settings.top_k,
            rerank_top_k=settings.rerank_top_k,
            similarity_threshold=settings.similarity_threshold,
        )

        self.bm25 = BM25Calculator()
        self._bm25_indexed = False

        logger.info("HybridRetriever initialized with FAISS backend")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_dict: dict[str, Any] | None = None,
        use_hybrid: bool = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant chunks for a query."""
        top_k = top_k or self.config.top_k
        use_hybrid = use_hybrid if use_hybrid is not None else self.config.use_hybrid

        logger.debug(f"Retrieving for: {query[:100]}...")

        # Vector search using FAISS
        vector_results = self.vector_store.search(
            query=query, top_k=top_k * 2 if use_hybrid else top_k, filter_dict=filter_dict
        )

        if use_hybrid and self._bm25_indexed:
            bm25_results = self.bm25.search(query, top_k=top_k * 2)
            combined = self._combine_results(vector_results, bm25_results, top_k)
        else:
            combined = vector_results[:top_k]

        if self.config.use_reranking:
            combined = self._rerank(query, combined)[: self.config.rerank_top_k]

        # Convert to RetrievalResult objects
        results = []
        for rank, result in enumerate(combined):
            if result["score"] >= self.config.similarity_threshold:
                chunk = TextChunk(
                    chunk_id=result["chunk_id"],
                    content=result["content"],
                    chunk_type=result["metadata"].get("chunk_type", "text"),
                    document_id=result["metadata"].get(
                        "document_id", result.get("document_id", "")
                    ),
                    page_number=result.get("page_number"),
                    metadata=result["metadata"],
                )

                results.append(
                    RetrievalResult(
                        chunk=chunk,
                        score=result["score"],
                        rank=rank + 1,
                        retrieval_method="hybrid" if use_hybrid else "vector",
                    )
                )

        logger.debug(f"Retrieved {len(results)} results")
        return results

    def build_bm25_index(self) -> None:
        """Build BM25 index from all documents in FAISS store."""
        logger.info("Building BM25 index...")

        try:
            # Get all chunks from FAISS store
            documents = []
            for chunk in self.vector_store.chunks.values():
                documents.append((chunk.chunk_id, chunk.content))

            if documents:
                self.bm25.fit(documents)
                self._bm25_indexed = True
                logger.info(f"BM25 index built with {len(documents)} documents")
            else:
                logger.info("No documents to index for BM25")

        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")

    def _combine_results(
        self, vector_results: list[dict], bm25_results: list[tuple[str, float]], top_k: int
    ) -> list[dict]:
        """Combine vector and BM25 results."""
        combined_scores = {}
        result_data = {}

        # Add vector scores
        max_vector_score = max((r["score"] for r in vector_results), default=1) or 1
        for result in vector_results:
            chunk_id = result["chunk_id"]
            normalized_score = result["score"] / max_vector_score
            combined_scores[chunk_id] = self.config.vector_weight * normalized_score
            result_data[chunk_id] = result

        # Add BM25 scores
        max_bm25_score = max((score for _, score in bm25_results), default=1) or 1
        for chunk_id, score in bm25_results:
            normalized_score = score / max_bm25_score

            if chunk_id in combined_scores:
                combined_scores[chunk_id] += self.config.keyword_weight * normalized_score
            else:
                # Try to get chunk data from vector store
                chunk_data = self.vector_store.get_chunk(chunk_id)
                if chunk_data:
                    combined_scores[chunk_id] = self.config.keyword_weight * normalized_score
                    result_data[chunk_id] = chunk_data

        # Sort and return top results
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        results = []
        for chunk_id in sorted_ids[:top_k]:
            if chunk_id in result_data:
                result = result_data[chunk_id].copy()
                result["score"] = combined_scores[chunk_id]
                results.append(result)

        return results

    def _rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Rerank results based on query relevance."""
        query_lower = query.lower()
        query_terms = set(re.findall(r"\b\w+\b", query_lower))

        for result in results:
            content_lower = result["content"].lower()
            content_terms = set(re.findall(r"\b\w+\b", content_lower))

            overlap = len(query_terms & content_terms)
            coverage = overlap / max(len(query_terms), 1)

            phrase_bonus = 0.2 if query_lower in content_lower else 0

            result["score"] = result["score"] * (1 + coverage * 0.3 + phrase_bonus)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results
