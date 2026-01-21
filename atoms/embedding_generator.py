"""
Local embedding generator using Sentence Transformers.
No API calls, no rate limits, fast and free!
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Any

from sentence_transformers import SentenceTransformer

from config.logging_config import logger
from config.settings import settings
from core.exceptions import EmbeddingError
from core.types import EmbeddingResult

# Available models with their dimensions
AVAILABLE_MODELS = {
    "all-MiniLM-L6-v2": 384,  # Fast, good quality
    "all-mpnet-base-v2": 768,  # Better quality, slower
    "multi-qa-mpnet-base-dot-v1": 768,  # Optimized for Q&A
    "all-MiniLM-L12-v2": 384,  # Balanced
    "paraphrase-MiniLM-L6-v2": 384,  # Good for similarity
}


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    cache_enabled: bool = True
    normalize_embeddings: bool = True
    show_progress: bool = True
    device: str = "cpu"  # "cpu" or "cuda"


class EmbeddingCache:
    """In-memory cache for embeddings."""

    def __init__(self, max_size: int = 50000):
        self.cache: dict[str, list[float]] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _get_key(self, text: str, model: str) -> str:
        content = f"{model}:{text[:500]}"  # Limit key size
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model: str) -> list[float] | None:
        key = self._get_key(text, model)
        result = self.cache.get(key)
        if result:
            self.hits += 1
        else:
            self.misses += 1
        return result

    def set(self, text: str, model: str, embedding: list[float]) -> None:
        if len(self.cache) >= self.max_size:
            # Remove oldest 10%
            keys_to_remove = list(self.cache.keys())[: int(self.max_size * 0.1)]
            for key in keys_to_remove:
                del self.cache[key]

        key = self._get_key(text, model)
        self.cache[key] = embedding

    def clear(self) -> None:
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def size(self) -> int:
        return len(self.cache)

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size": self.size(),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2%}",
        }


class EmbeddingGenerator:
    """
    Local embedding generator using Sentence Transformers.

    Benefits:
    - No API calls
    - No rate limits
    - Fast processing
    - Free to use
    - Works offline
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        """
        Initialize the embedding generator.

        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig(
            model_name=settings.embedding_model, batch_size=settings.embedding_batch_size
        )

        # Initialize cache
        self.cache = EmbeddingCache() if self.config.cache_enabled else None

        # Load model
        logger.info(f"Loading embedding model: {self.config.model_name}...")
        start_time = time.time()

        try:
            self.model = SentenceTransformer(self.config.model_name, device=self.config.device)

            # Get embedding dimension
            self.dimension = self.model.get_sentence_embedding_dimension()

            load_time = time.time() - start_time
            logger.info(
                f"Model loaded in {load_time:.2f}s | "
                f"Dimension: {self.dimension} | "
                f"Device: {self.config.device}"
            )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise EmbeddingError(f"Failed to load embedding model: {e}") from e

    def generate(self, text: str, use_cache: bool = True) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            use_cache: Whether to use caching

        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            logger.warning("Empty text provided, returning zero vector")
            return [0.0] * self.dimension

        # Check cache
        if use_cache and self.cache:
            cached = self.cache.get(text, self.config.model_name)
            if cached:
                return cached

        try:
            # Generate embedding
            embedding = self.model.encode(
                text, normalize_embeddings=self.config.normalize_embeddings, show_progress_bar=False
            )

            # Convert to list
            embedding_list = embedding.tolist()

            # Cache result
            if use_cache and self.cache:
                self.cache.set(text, self.config.model_name, embedding_list)

            return embedding_list

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

    def generate_batch(self, texts: list[str], show_progress: bool = True) -> list[list[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} texts...")
        start_time = time.time()

        # Separate cached and uncached texts
        embeddings = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        if self.cache:
            for i, text in enumerate(texts):
                if text and text.strip():
                    cached = self.cache.get(text, self.config.model_name)
                    if cached:
                        embeddings[i] = cached
                    else:
                        uncached_indices.append(i)
                        uncached_texts.append(text)
                else:
                    embeddings[i] = [0.0] * self.dimension
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = [t if t and t.strip() else "" for t in texts]

        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                batch_embeddings = self.model.encode(
                    uncached_texts,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=show_progress and len(uncached_texts) > 10,
                )

                # Store results
                for i, idx in enumerate(uncached_indices):
                    embedding_list = batch_embeddings[i].tolist()
                    embeddings[idx] = embedding_list

                    # Cache the embedding
                    if self.cache and uncached_texts[i]:
                        self.cache.set(uncached_texts[i], self.config.model_name, embedding_list)

            except Exception as e:
                logger.error(f"Error in batch embedding: {e}")
                # Fill with zero vectors on error
                for idx in uncached_indices:
                    if embeddings[idx] is None:
                        embeddings[idx] = [0.0] * self.dimension

        # Fill any remaining None values
        for i in range(len(embeddings)):
            if embeddings[i] is None:
                embeddings[i] = [0.0] * self.dimension

        elapsed = time.time() - start_time
        logger.info(
            f"Generated {len(texts)} embeddings in {elapsed:.2f}s "
            f"({len(texts) / elapsed:.1f} texts/sec)"
        )

        return embeddings

    def generate_query_embedding(self, query: str) -> list[float]:
        """
        Generate embedding for a search query.

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        return self.generate(query, use_cache=False)

    def generate_with_metadata(self, chunk_id: str, text: str) -> EmbeddingResult:
        """
        Generate embedding with full metadata.

        Args:
            chunk_id: ID of the chunk
            text: Text to embed

        Returns:
            EmbeddingResult with metadata
        """
        embedding = self.generate(text)

        return EmbeddingResult(
            chunk_id=chunk_id,
            embedding=embedding,
            model=self.config.model_name,
            dimension=len(embedding),
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.config.model_name,
            "dimension": self.dimension,
            "device": self.config.device,
            "batch_size": self.config.batch_size,
            "cache_stats": self.cache.stats() if self.cache else None,
        }

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension


# Global instance
_embedding_generator: EmbeddingGenerator | None = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create the global embedding generator."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator


def reset_embedding_generator() -> None:
    """Reset the global embedding generator."""
    global _embedding_generator
    if _embedding_generator and _embedding_generator.cache:
        _embedding_generator.cache.clear()
    _embedding_generator = None


def generate_embedding(text: str, **kwargs) -> list[float]:
    """Convenience function to generate a single embedding."""
    return get_embedding_generator().generate(text, **kwargs)


def generate_embeddings(texts: list[str], **kwargs) -> list[list[float]]:
    """Convenience function to generate multiple embeddings."""
    return get_embedding_generator().generate_batch(texts, **kwargs)


def get_embedding_dimension() -> int:
    """Get the embedding dimension."""
    return get_embedding_generator().get_dimension()
