"""
Validation utilities for the Financial RAG system.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import magic  # python-magic

from config.settings import settings
from core.exceptions import (
    UnsupportedFileTypeError, 
    FileSizeExceededError,
    ValidationError
)


class FileValidator:
    """Validates uploaded files."""
    
    MIME_TYPE_MAP = {
        'application/pdf': '.pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
        'application/vnd.ms-excel': '.xls',
        'text/csv': '.csv',
        'text/plain': '.txt'
    }
    
    @classmethod
    def validate_file(
        cls,
        file_path: Path,
        allowed_extensions: Optional[List[str]] = None,
        max_size_mb: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Validate a file.
        
        Args:
            file_path: Path to the file
            allowed_extensions: List of allowed extensions
            max_size_mb: Maximum file size in MB
            
        Returns:
            Tuple of (is_valid, message)
        """
        allowed_extensions = allowed_extensions or settings.supported_extensions
        max_size_mb = max_size_mb or settings.max_file_size_mb
        
        # Check existence
        if not file_path.exists():
            return False, f"File not found: {file_path}"
        
        # Check extension
        ext = file_path.suffix.lower()
        if ext not in allowed_extensions:
            return False, f"Unsupported file type: {ext}"
        
        # Check size
        file_size = file_path.stat().st_size
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            return False, f"File size ({file_size} bytes) exceeds limit ({max_size_bytes} bytes)"
        
        # Verify MIME type matches extension
        if cls._check_mime_type(file_path, ext):
            return True, "Valid"
        else:
            return False, "File type does not match extension"
    
    @classmethod
    def _check_mime_type(cls, file_path: Path, expected_ext: str) -> bool:
        """Check if MIME type matches expected extension."""
        try:
            mime = magic.Magic(mime=True)
            detected_mime = mime.from_file(str(file_path))
            
            expected_mime_ext = cls.MIME_TYPE_MAP.get(detected_mime)
            
            # Allow text files to be flexible
            if detected_mime.startswith('text/'):
                return expected_ext in ['.txt', '.csv']
            
            return expected_mime_ext == expected_ext
            
        except Exception:
            # If magic fails, allow the file
            return True


class QueryValidator:
    """Validates user queries."""
    
    MIN_QUERY_LENGTH = 3
    MAX_QUERY_LENGTH = 2000
    
    # Potentially harmful patterns
    INJECTION_PATTERNS = [
        r'<script',
        r'javascript:',
        r'on\w+\s*=',
    ]
    
    @classmethod
    def validate_query(cls, query: str) -> Tuple[bool, str]:
        """
        Validate a user query.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check length
        if len(query) < cls.MIN_QUERY_LENGTH:
            return False, f"Query too short (minimum {cls.MIN_QUERY_LENGTH} characters)"
        
        if len(query) > cls.MAX_QUERY_LENGTH:
            return False, f"Query too long (maximum {cls.MAX_QUERY_LENGTH} characters)"
        
        # Check for injection patterns
        import re
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return False, "Query contains invalid characters"
        
        return True, "Valid"
    
    @classmethod
    def sanitize_query(cls, query: str) -> str:
        """
        Sanitize a query by removing potentially harmful content.
        
        Args:
            query: User query
            
        Returns:
            Sanitized query
        """
        import re
        import html
        
        # HTML escape
        sanitized = html.escape(query)
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized.strip()