"""
Core type definitions for the Financial RAG system.
Defines all data models and type hints used across the application.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime
import uuid


class DocumentType(Enum):
    """Supported document types."""
    PDF = "pdf"
    WORD = "docx"
    EXCEL = "xlsx"
    CSV = "csv"
    TEXT = "txt"


class ChunkType(Enum):
    """Types of content chunks."""
    TEXT = "text"
    TABLE = "table"
    HEADER = "header"
    FINANCIAL_DATA = "financial_data"
    METADATA = "metadata"


@dataclass
class DocumentMetadata:
    """Metadata associated with a document."""
    filename: str
    file_type: DocumentType
    file_size: int
    upload_date: datetime = field(default_factory=datetime.utcnow)
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    page_count: Optional[int] = None
    author: Optional[str] = None
    title: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    chunk_id: str
    content: str
    chunk_type: ChunkType
    document_id: str
    page_number: Optional[int] = None
    start_index: int = 0
    end_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = str(uuid.uuid4())
        # Handle string chunk_type
        if isinstance(self.chunk_type, str):
            try:
                self.chunk_type = ChunkType(self.chunk_type)
            except ValueError:
                self.chunk_type = ChunkType.TEXT


@dataclass
class TableData:
    """Represents extracted table data."""
    table_id: str
    headers: List[str]
    rows: List[List[Any]]
    document_id: str
    page_number: Optional[int] = None
    table_index: int = 0
    caption: Optional[str] = None
    
    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        if not self.headers or not self.rows:
            return ""
        
        # Clean headers
        clean_headers = [str(h) if h else "" for h in self.headers]
        
        # Header row
        md = "| " + " | ".join(clean_headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(clean_headers)) + " |\n"
        
        # Data rows
        for row in self.rows:
            clean_row = [str(cell) if cell is not None else "" for cell in row]
            # Pad row if needed
            while len(clean_row) < len(clean_headers):
                clean_row.append("")
            md += "| " + " | ".join(clean_row[:len(clean_headers)]) + " |\n"
        
        return md
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert table to dictionary format."""
        return {
            "table_id": self.table_id,
            "headers": self.headers,
            "rows": self.rows,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "caption": self.caption
        }


@dataclass
class FinancialEntity:
    """Represents an extracted financial entity."""
    entity_type: str  # e.g., "currency", "percentage", "date", "metric"
    value: str
    normalized_value: Optional[Union[float, datetime]] = None
    context: Optional[str] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            "entity_type": self.entity_type,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "context": self.context,
            "confidence": self.confidence
        }


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    chunk_id: str
    embedding: List[float]
    model: str
    dimension: int
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    chunk: TextChunk
    score: float
    rank: int
    retrieval_method: str = "vector"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "content": self.chunk.content,
            "score": self.score,
            "rank": self.rank,
            "retrieval_method": self.retrieval_method,
            "document_id": self.chunk.document_id,
            "page_number": self.chunk.page_number,
            "metadata": self.chunk.metadata
        }


@dataclass
class QueryContext:
    """Context for a user query."""
    query_id: str
    original_query: str
    processed_query: Optional[str] = None
    query_type: Optional[str] = None  # e.g., "factual", "analytical", "comparison"
    entities: List[FinancialEntity] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.query_id:
            self.query_id = str(uuid.uuid4())


@dataclass
class GeneratedResponse:
    """Generated response from the RAG system."""
    response_id: str
    query_id: str
    answer: str
    sources: List[RetrievalResult]
    confidence: float
    processing_time: float
    token_usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.response_id:
            self.response_id = str(uuid.uuid4())


@dataclass
class ProcessingResult:
    """Result of document processing."""
    document_id: str
    success: bool
    chunks: List[TextChunk] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    financial_entities: List[FinancialEntity] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.document_id:
            self.document_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "success": self.success,
            "chunk_count": len(self.chunks),
            "table_count": len(self.tables),
            "entity_count": len(self.financial_entities),
            "errors": self.errors,
            "processing_time": self.processing_time,
            "metadata": self.metadata
        }


@dataclass 
class DocumentInfo:
    """Information about an ingested document."""
    document_id: str
    filename: str
    file_type: DocumentType
    file_size: int
    upload_date: datetime
    chunk_count: int = 0
    table_count: int = 0
    status: str = "processed"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type.value if isinstance(self.file_type, DocumentType) else self.file_type,
            "file_size": self.file_size,
            "upload_date": self.upload_date.isoformat() if isinstance(self.upload_date, datetime) else self.upload_date,
            "chunk_count": self.chunk_count,
            "table_count": self.table_count,
            "status": self.status,
            "metadata": self.metadata
        }


@dataclass
class SearchQuery:
    """Represents a search query with filters."""
    query_text: str
    document_ids: Optional[List[str]] = None
    top_k: int = 10
    similarity_threshold: float = 0.5
    filters: Dict[str, Any] = field(default_factory=dict)
    include_metadata: bool = True


@dataclass
class SearchResult:
    """Result of a search operation."""
    query: str
    results: List[RetrievalResult]
    total_results: int
    search_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "search_time": self.search_time
        }