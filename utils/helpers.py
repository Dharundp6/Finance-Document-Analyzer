"""
Helper utilities for the Financial RAG system.
"""

import os
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import re


def generate_document_id(filename: str, content_hash: Optional[str] = None) -> str:
    """
    Generate a unique document ID.
    
    Args:
        filename: Name of the file
        content_hash: Optional hash of file content
        
    Returns:
        Unique document ID
    """
    import uuid
    
    if content_hash:
        # Use content hash for deduplication
        return hashlib.md5(f"{filename}:{content_hash}".encode()).hexdigest()
    
    return str(uuid.uuid4())


def compute_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Compute hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm (md5, sha256)
        
    Returns:
        File hash
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def format_currency(
    value: float,
    currency: str = 'USD',
    locale: str = 'en_US'
) -> str:
    """
    Format a value as currency.
    
    Args:
        value: Numeric value
        currency: Currency code
        locale: Locale for formatting
        
    Returns:
        Formatted currency string
    """
    symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'INR': '₹',
        'JPY': '¥'
    }
    
    symbol = symbols.get(currency, currency)
    
    if abs(value) >= 1e12:
        return f"{symbol}{value/1e12:.2f}T"
    elif abs(value) >= 1e9:
        return f"{symbol}{value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{symbol}{value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"{symbol}{value/1e3:.2f}K"
    else:
        return f"{symbol}{value:,.2f}"


def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format a value as percentage.
    
    Args:
        value: Numeric value (0.15 = 15%)
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimal_places}f}%"


def parse_date_string(date_str: str) -> Optional[datetime]:
    """
    Parse various date string formats.
    
    Args:
        date_str: Date string
        
    Returns:
        Parsed datetime or None
    """
    formats = [
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%B %d, %Y',
        '%b %d, %Y',
        '%Y',
        'Q%q %Y',  # Fiscal quarter
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Try to extract year at least
    year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
    if year_match:
        return datetime(int(year_match.group()), 1, 1)
    
    return None


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks.
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Override dictionary
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_numbers(text: str) -> List[float]:
    """
    Extract all numbers from text.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted numbers
    """
    pattern = r'-?[\d,]+\.?\d*'
    matches = re.findall(pattern, text)
    
    numbers = []
    for match in matches:
        try:
            num = float(match.replace(',', ''))
            numbers.append(num)
        except ValueError:
            continue
    
    return numbers


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        import time
        self.end_time = time.time()
    
    @property
    def elapsed(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def __str__(self):
        return f"{self.name}: {self.elapsed:.2f}s"