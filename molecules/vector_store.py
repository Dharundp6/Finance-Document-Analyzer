"""
FAISS-based vector store for document embeddings.
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import numpy as np

import faiss

from config.settings import settings
from config.logging_config import logger
from core.types import TextChunk, ChunkType
from core.exceptions import VectorStoreError
from atoms.embedding_generator import EmbeddingGenerator, get_embedding_generator


@dataclass
class StoredChunk:
    """Chunk data stored alongside FAISS index."""
    chunk_id: str
    content: str
    document_id: str
    chunk_type: str
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class VectorStoreConfig:
    """Configuration for FAISS vector store."""
    persist_directory: str = "./data/vector_db"
    index_file: str = "faiss_index.bin"
    metadata_file: str = "metadata.pkl"
    embedding_dimension: int = 384  # MiniLM default
    use_gpu: bool = False
    index_type: str = "flat"


class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search.
    """
    
    def __init__(
        self,
        config: Optional[VectorStoreConfig] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        """Initialize the FAISS vector store."""
        
        # Get embedding generator first to know the dimension
        self.embedding_generator = embedding_generator or get_embedding_generator()
        
        # Get actual dimension from the model
        actual_dimension = self.embedding_generator.get_dimension()
        
        self.config = config or VectorStoreConfig(
            persist_directory=settings.vector_db_path,
            index_file=settings.faiss_index_file,
            metadata_file=settings.metadata_file,
            embedding_dimension=actual_dimension  # Use actual dimension
        )
        
        # Override with actual dimension
        self.config.embedding_dimension = actual_dimension
        
        # Storage paths
        self.persist_dir = Path(self.config.persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.persist_dir / self.config.index_file
        self.metadata_path = self.persist_dir / self.config.metadata_file
        
        # Initialize storage
        self.index: Optional[faiss.Index] = None
        self.chunks: Dict[int, StoredChunk] = {}
        self.chunk_id_to_faiss_id: Dict[str, int] = {}
        self.next_id: int = 0
        
        self._load_or_create_index()
        
        logger.info(
            f"FAISSVectorStore initialized: {self.count()} vectors, "
            f"dimension={self.config.embedding_dimension}"
        )
    
    def _load_or_create_index(self) -> None:
        """Load existing index or create new one."""
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                self._load_index()
                
                # Check if dimension matches
                if self.index.d != self.config.embedding_dimension:
                    logger.warning(
                        f"Index dimension ({self.index.d}) doesn't match "
                        f"model dimension ({self.config.embedding_dimension}). "
                        "Creating new index."
                    )
                    self._create_index()
                else:
                    logger.info(f"Loaded existing FAISS index with {self.count()} vectors")
                return
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Creating new index.")
        
        self._create_index()
        logger.info("Created new FAISS index")
    
    def _create_index(self) -> None:
        """Create a new FAISS index."""
        dimension = self.config.embedding_dimension
        
        if self.config.index_type == "flat":
            self.index = faiss.IndexFlatIP(dimension)
        elif self.config.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        else:
            self.index = faiss.IndexFlatIP(dimension)
        
        self.index = faiss.IndexIDMap(self.index)
        
        self.chunks = {}
        self.chunk_id_to_faiss_id = {}
        self.next_id = 0
        
        self._save_index()
    
    def _load_index(self) -> None:
        """Load index and metadata from disk."""
        self.index = faiss.read_index(str(self.index_path))
        
        with open(self.metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data.get('chunks', {})
            self.chunk_id_to_faiss_id = data.get('chunk_id_to_faiss_id', {})
            self.next_id = data.get('next_id', 0)
    
    def _save_index(self) -> None:
        """Save index and metadata to disk."""
        try:
            faiss.write_index(self.index, str(self.index_path))
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks,
                    'chunk_id_to_faiss_id': self.chunk_id_to_faiss_id,
                    'next_id': self.next_id
                }, f)
            
            logger.debug("Index saved to disk")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise VectorStoreError(f"Failed to save index: {e}")
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def add_chunks(
        self,
        chunks: List[TextChunk],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> int:
        """Add chunks to the vector store."""
        if not chunks:
            return 0
        
        logger.info(f"Adding {len(chunks)} chunks to FAISS index...")
        
        # Generate all embeddings at once (fast with local model!)
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_generator.generate_batch(texts, show_progress=show_progress)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        for i in range(len(embeddings_array)):
            embeddings_array[i] = self._normalize_vector(embeddings_array[i])
        
        # Assign IDs and store metadata
        ids = []
        for i, chunk in enumerate(chunks):
            faiss_id = self.next_id
            self.next_id += 1
            
            ids.append(faiss_id)
            self.chunk_id_to_faiss_id[chunk.chunk_id] = faiss_id
            
            self.chunks[faiss_id] = StoredChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                document_id=chunk.document_id,
                chunk_type=chunk.chunk_type.value if isinstance(chunk.chunk_type, ChunkType) else str(chunk.chunk_type),
                page_number=chunk.page_number,
                metadata=chunk.metadata
            )
        
        # Add to FAISS index
        ids_array = np.array(ids, dtype=np.int64)
        self.index.add_with_ids(embeddings_array, ids_array)
        
        # Save to disk
        self._save_index()
        
        logger.info(f"Successfully added {len(chunks)} chunks to index")
        return len(chunks)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        if self.count() == 0:
            logger.warning("Index is empty")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_query_embedding(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            query_vector = self._normalize_vector(query_vector.flatten()).reshape(1, -1)
            
            # Search
            search_k = min(top_k * 3, self.count())
            distances, indices = self.index.search(query_vector, search_k)
            
            # Process results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:
                    continue
                
                if idx not in self.chunks:
                    continue
                
                chunk = self.chunks[idx]
                
                # Apply filters
                if filter_dict and not self._matches_filter(chunk, filter_dict):
                    continue
                
                score = float(distance)
                
                results.append({
                    'chunk_id': chunk.chunk_id,
                    'content': chunk.content,
                    'document_id': chunk.document_id,
                    'score': score,
                    'page_number': chunk.page_number,
                    'metadata': {
                        'chunk_type': chunk.chunk_type,
                        'document_id': chunk.document_id,
                        **chunk.metadata
                    }
                })
                
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise VectorStoreError(f"Search failed: {e}")
    
    def _matches_filter(self, chunk: StoredChunk, filter_dict: Dict[str, Any]) -> bool:
        """Check if chunk matches filter criteria."""
        for key, value in filter_dict.items():
            if key == 'document_id':
                if chunk.document_id != value:
                    return False
            elif key == 'chunk_type':
                if chunk.chunk_type != value:
                    return False
            elif key in chunk.metadata:
                if chunk.metadata[key] != value:
                    return False
        return True
    
    def search_by_document(
        self,
        query: str,
        document_id: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search within a specific document."""
        return self.search(query, top_k, filter_dict={'document_id': document_id})
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a chunk by ID."""
        faiss_id = self.chunk_id_to_faiss_id.get(chunk_id)
        if faiss_id is None:
            return None
        
        chunk = self.chunks.get(faiss_id)
        if chunk is None:
            return None
        
        return {
            'chunk_id': chunk.chunk_id,
            'content': chunk.content,
            'document_id': chunk.document_id,
            'page_number': chunk.page_number,
            'metadata': {
                'chunk_type': chunk.chunk_type,
                **chunk.metadata
            }
        }
    
    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        chunks_to_delete = [
            faiss_id for faiss_id, chunk in self.chunks.items()
            if chunk.document_id == document_id
        ]
        
        if not chunks_to_delete:
            return 0
        
        deleted_count = len(chunks_to_delete)
        
        for faiss_id in chunks_to_delete:
            chunk = self.chunks.pop(faiss_id, None)
            if chunk:
                self.chunk_id_to_faiss_id.pop(chunk.chunk_id, None)
        
        self._rebuild_index()
        
        logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
        return deleted_count
    
    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from stored chunks."""
        if not self.chunks:
            self._create_index()
            return
        
        old_chunks = dict(self.chunks)
        self._create_index()
        
        if not old_chunks:
            return
        
        # Re-embed all chunks
        texts = [chunk.content for chunk in old_chunks.values()]
        embeddings = self.embedding_generator.generate_batch(texts, show_progress=False)
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        for i in range(len(embeddings_array)):
            embeddings_array[i] = self._normalize_vector(embeddings_array[i])
        
        ids = []
        for chunk in old_chunks.values():
            new_faiss_id = self.next_id
            self.next_id += 1
            
            ids.append(new_faiss_id)
            self.chunks[new_faiss_id] = chunk
            self.chunk_id_to_faiss_id[chunk.chunk_id] = new_faiss_id
        
        ids_array = np.array(ids, dtype=np.int64)
        self.index.add_with_ids(embeddings_array, ids_array)
        
        self._save_index()
    
    def count(self) -> int:
        """Get the number of vectors in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        document_ids = set()
        chunk_types = {}
        
        for chunk in self.chunks.values():
            document_ids.add(chunk.document_id)
            chunk_type = chunk.chunk_type
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        return {
            'total_chunks': self.count(),
            'total_documents': len(document_ids),
            'chunk_types': chunk_types,
            'index_type': self.config.index_type,
            'embedding_dimension': self.config.embedding_dimension,
            'embedding_model': self.embedding_generator.config.model_name
        }
    
    def clear(self) -> None:
        """Clear all data from the index."""
        self._create_index()
        logger.info("Vector store cleared")
    
    def get_all_document_ids(self) -> List[str]:
        """Get all unique document IDs."""
        return list(set(chunk.document_id for chunk in self.chunks.values()))


# Alias for backward compatibility
VectorStore = FAISSVectorStore