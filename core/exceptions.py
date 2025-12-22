"""
Custom exceptions for the Financial RAG system.
Provides granular error handling across the application.
"""

from typing import Optional, Dict, Any


class FinancialRAGException(Exception):
    """Base exception for Financial RAG system."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details
        }


class DocumentProcessingError(FinancialRAGException):
    """Error during document processing."""
    
    def __init__(self, message: str, document_id: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DOCUMENT_PROCESSING_ERROR", **kwargs)
        self.document_id = document_id


class UnsupportedFileTypeError(DocumentProcessingError):
    """Error when file type is not supported."""
    
    def __init__(self, file_type: str):
        super().__init__(
            message=f"Unsupported file type: {file_type}",
            error_code="UNSUPPORTED_FILE_TYPE",
            details={"file_type": file_type}
        )


class FileSizeExceededError(DocumentProcessingError):
    """Error when file size exceeds limit."""
    
    def __init__(self, file_size: int, max_size: int):
        super().__init__(
            message=f"File size {file_size} bytes exceeds maximum {max_size} bytes",
            error_code="FILE_SIZE_EXCEEDED",
            details={"file_size": file_size, "max_size": max_size}
        )


class EmbeddingError(FinancialRAGException):
    """Error during embedding generation."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="EMBEDDING_ERROR", **kwargs)


class VectorStoreError(FinancialRAGException):
    """Error with vector store operations."""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="VECTOR_STORE_ERROR", **kwargs)
        self.operation = operation


class RetrievalError(FinancialRAGException):
    """Error during retrieval."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="RETRIEVAL_ERROR", **kwargs)


class GenerationError(FinancialRAGException):
    """Error during response generation."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="GENERATION_ERROR", **kwargs)


class APIError(FinancialRAGException):
    """Error with external API calls."""
    
    def __init__(self, message: str, api_name: str, status_code: Optional[int] = None, **kwargs):
        super().__init__(message, error_code="API_ERROR", **kwargs)
        self.api_name = api_name
        self.status_code = status_code


class ValidationError(FinancialRAGException):
    """Error during validation."""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field = field