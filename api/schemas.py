"""
API request and response schemas.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    PDF = "pdf"
    WORD = "docx"
    EXCEL = "xlsx"
    CSV = "csv"
    TEXT = "txt"


# Request Schemas
class QueryRequest(BaseModel):
    """Request schema for querying documents."""

    query: str = Field(..., min_length=1, max_length=1000)
    document_ids: list[str] | None = None
    top_k: int | None = Field(default=10, ge=1, le=50)
    include_sources: bool = True


class IngestRequest(BaseModel):
    """Request schema for document ingestion."""

    metadata: dict[str, Any] | None = None


# Response Schemas
class SourceInfo(BaseModel):
    """Source information for citations."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    page_number: int | None = None
    metadata: dict[str, Any] = {}


class QueryResponse(BaseModel):
    """Response schema for queries."""

    response_id: str
    query: str
    answer: str
    sources: list[SourceInfo]
    confidence: float
    processing_time: float
    token_usage: dict[str, int]


class DocumentInfo(BaseModel):
    """Document information."""

    document_id: str
    filename: str
    file_type: str
    file_size: int
    upload_date: str
    chunk_count: int | None = None
    metadata: dict[str, Any] = {}


class IngestResponse(BaseModel):
    """Response schema for ingestion."""

    success: bool
    document_id: str
    filename: str
    chunk_count: int
    table_count: int
    processing_time: float
    errors: list[str] = []


class StatsResponse(BaseModel):
    """Response schema for system statistics."""

    total_documents: int
    total_chunks: int
    chunk_types: dict[str, int]
    collection_name: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response schema."""

    error_code: str
    message: str
    details: dict[str, Any] | None = None
