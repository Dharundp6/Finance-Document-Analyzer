"""
API routes for the Financial RAG system.
"""

import json
import os
from datetime import datetime

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from api.schemas import (
    DocumentInfo,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceInfo,
    StatsResponse,
)
from atoms.embedding_generator import get_embedding_generator
from config.logging_config import logger
from config.settings import settings
from molecules.retriever import HybridRetriever
from molecules.vector_store import FAISSVectorStore
from organisms.ingestion_pipeline import IngestionPipeline
from organisms.query_engine import QueryEngine
from organisms.response_generator import ResponseGenerator

# Create router FIRST
router = APIRouter()


# Initialize components
logger.info("Initializing RAG components...")

vector_store = FAISSVectorStore()
retriever = HybridRetriever(vector_store)
ingestion_pipeline = IngestionPipeline(vector_store=vector_store)
query_engine = QueryEngine(vector_store=vector_store, retriever=retriever)
response_generator = ResponseGenerator()

# Build BM25 index on startup
retriever.build_bm25_index()

logger.info("RAG components initialized successfully")


# ============================================
# Health & Info Endpoints
# ============================================


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy", version=settings.app_version, timestamp=datetime.utcnow().isoformat()
    )


@router.get("/model-info")
async def get_model_info():
    """Get information about the models being used."""
    emb_gen = get_embedding_generator()

    return {
        "llm_model": settings.llm_model,
        "embedding_model": emb_gen.config.model_name,
        "embedding_dimension": emb_gen.get_dimension(),
        "embedding_type": "local (sentence-transformers)",
        "vector_store": "FAISS",
        "total_chunks": vector_store.count(),
        "cache_stats": emb_gen.cache.stats() if emb_gen.cache else None,
    }


# ============================================
# Query Endpoints
# ============================================


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the financial documents.

    Args:
        request: Query request with question and optional filters

    Returns:
        Generated answer with sources
    """
    try:
        logger.info(f"Query received: {request.query[:100]}...")

        # Process query and retrieve context
        query_context, retrieval_results = query_engine.process_query(
            query=request.query, document_ids=request.document_ids, top_k=request.top_k
        )

        # Get formatted context
        context = query_engine.get_context_for_generation(
            query=request.query, document_ids=request.document_ids
        )

        # Generate response
        response = response_generator.generate(
            query=request.query,
            context=context,
            query_context=query_context,
            retrieval_results=retrieval_results,
        )

        # Format sources
        sources = []
        if request.include_sources:
            for result in retrieval_results[:5]:
                sources.append(
                    SourceInfo(
                        chunk_id=result.chunk.chunk_id,
                        document_id=result.chunk.document_id,
                        content=result.chunk.content[:500],
                        score=result.score,
                        page_number=result.chunk.page_number,
                        metadata=result.chunk.metadata,
                    )
                )

        return QueryResponse(
            response_id=response.response_id,
            query=request.query,
            answer=response.answer,
            sources=sources,
            confidence=response.confidence,
            processing_time=response.processing_time,
            token_usage=response.token_usage,
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")

        # Check for rate limit
        error_str = str(e)
        if "ResourceExhausted" in error_str or "429" in error_str:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="API rate limit reached. Please wait and try again.",
            ) from None

        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/query/stream")
async def query_documents_stream(request: QueryRequest):
    """Query with streaming response."""
    try:
        context = query_engine.get_context_for_generation(
            query=request.query, document_ids=request.document_ids
        )

        def generate():
            yield from response_generator.generate_with_streaming(
                query=request.query, context=context
            )

        return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        logger.error(f"Error in streaming query: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================
# Document Endpoints
# ============================================


@router.post("/documents/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...), metadata: str | None = None):
    """
    Upload and ingest a document.

    Args:
        file: Document file to upload
        metadata: Optional JSON metadata string

    Returns:
        Ingestion result
    """
    try:
        logger.info(f"Uploading document: {file.filename}")

        # Validate file type
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in settings.supported_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Supported: {settings.supported_extensions}",
            )

        # Read file content
        content = await file.read()

        # Parse metadata if provided
        meta_dict = None
        if metadata:
            try:
                meta_dict = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON, ignoring")

        # Ingest document
        result = ingestion_pipeline.ingest_bytes(
            file_bytes=content, filename=file.filename, metadata=meta_dict
        )

        # Rebuild BM25 index
        retriever.build_bm25_index()

        return IngestResponse(
            success=result.success,
            document_id=result.document_id,
            filename=file.filename,
            chunk_count=len(result.chunks),
            table_count=len(result.tables),
            processing_time=result.processing_time,
            errors=result.errors,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    """List all ingested documents."""
    try:
        documents = ingestion_pipeline.list_documents()

        return [
            DocumentInfo(
                document_id=doc["document_id"],
                filename=doc["filename"],
                file_type=doc["file_type"],
                file_size=doc["file_size"],
                upload_date=doc["upload_date"],
                metadata=doc.get("custom_metadata", {}),
            )
            for doc in documents
        ]

    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/documents/{document_id}", response_model=DocumentInfo)
async def get_document(document_id: str):
    """Get information about a specific document."""
    try:
        doc = ingestion_pipeline.get_document_info(document_id)

        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        return DocumentInfo(
            document_id=doc["document_id"],
            filename=doc["filename"],
            file_type=doc["file_type"],
            file_size=doc["file_size"],
            upload_date=doc["upload_date"],
            metadata=doc.get("custom_metadata", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its chunks."""
    try:
        success = ingestion_pipeline.delete_document(document_id)

        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        # Rebuild BM25 index
        retriever.build_bm25_index()

        return {"message": "Document deleted successfully", "document_id": document_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================
# Stats Endpoint
# ============================================


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    try:
        stats = ingestion_pipeline.get_stats()
        vs_stats = vector_store.get_stats()

        return StatsResponse(
            total_documents=stats.get("registered_documents", 0),
            total_chunks=vs_stats.get("total_chunks", 0),
            chunk_types=vs_stats.get("chunk_types", {}),
            collection_name=f"FAISS ({vs_stats.get('embedding_model', 'unknown')})",
        )

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ============================================
# Search Endpoint (Direct vector search)
# ============================================


@router.post("/search")
async def search_documents(query: str, top_k: int = 10, document_id: str | None = None):
    """
    Direct vector similarity search without LLM generation.
    Useful for testing and debugging.
    """
    try:
        filter_dict = {"document_id": document_id} if document_id else None

        results = vector_store.search(query=query, top_k=top_k, filter_dict=filter_dict)

        return {
            "query": query,
            "total_results": len(results),
            "results": [
                {
                    "chunk_id": r["chunk_id"],
                    "score": r["score"],
                    "content": r["content"][:300] + "..."
                    if len(r["content"]) > 300
                    else r["content"],
                    "document_id": r["document_id"],
                    "page_number": r.get("page_number"),
                }
                for r in results
            ],
        }

    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
