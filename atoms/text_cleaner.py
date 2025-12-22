"""
Atomic component for text cleaning and normalization.
Handles preprocessing of text for financial documents.
"""

import re
import unicodedata
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from config.logging_config import logger


@dataclass
class CleaningConfig:
    """Configuration for text cleaning."""
    remove_extra_whitespace: bool = True
    normalize_unicode: bool = True
    remove_control_chars: bool = True
    preserve_numbers: bool = True
    preserve_currency: bool = True
    preserve_percentages: bool = True
    lowercase: bool = False
    remove_urls: bool = True
    remove_email: bool = False
    min_line_length: int = 3


class TextCleaner:
    """
    Cleans and normalizes text from financial documents.
    Preserves important financial information like numbers, currencies, and percentages.
    """
    
    # Regex patterns
    URL_PATTERN = re.compile(
        r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    )
    EMAIL_PATTERN = re.compile(
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    )
    MULTIPLE_SPACES = re.compile(r'\s+')
    MULTIPLE_NEWLINES = re.compile(r'\n{3,}')
    CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
    
    # Financial patterns to preserve
    CURRENCY_PATTERN = re.compile(
        r'[$€£¥₹][\d,]+(?:\.\d{1,2})?|[\d,]+(?:\.\d{1,2})?\s*(?:USD|EUR|GBP|INR|JPY)'
    )
    PERCENTAGE_PATTERN = re.compile(r'\d+(?:\.\d+)?%')
    NUMBER_PATTERN = re.compile(r'[\d,]+(?:\.\d+)?')
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        """
        Initialize the text cleaner.
        
        Args:
            config: Cleaning configuration
        """
        self.config = config or CleaningConfig()
    
    def clean(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        cleaned = text
        
        # Normalize unicode characters
        if self.config.normalize_unicode:
            cleaned = self._normalize_unicode(cleaned)
        
        # Remove control characters
        if self.config.remove_control_chars:
            cleaned = self._remove_control_chars(cleaned)
        
        # Remove URLs
        if self.config.remove_urls:
            cleaned = self._remove_urls(cleaned)
        
        # Remove emails
        if self.config.remove_email:
            cleaned = self._remove_emails(cleaned)
        
        # Clean whitespace
        if self.config.remove_extra_whitespace:
            cleaned = self._normalize_whitespace(cleaned)
        
        # Lowercase if configured
        if self.config.lowercase:
            cleaned = cleaned.lower()
        
        return cleaned.strip()
    
    def clean_for_embedding(self, text: str) -> str:
        """
        Clean text specifically for embedding generation.
        More aggressive cleaning while preserving semantic meaning.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text optimized for embedding
        """
        cleaned = self.clean(text)
        
        # Remove redundant punctuation
        cleaned = re.sub(r'[.]{2,}', '.', cleaned)
        cleaned = re.sub(r'[-]{2,}', '-', cleaned)
        
        # Normalize quotes
        cleaned = re.sub(r'["""]', '"', cleaned)
        cleaned = re.sub(r"[''']", "'", cleaned)
        
        return cleaned
    
    def clean_table_cell(self, cell: Any) -> str:
        """
        Clean a table cell value.
        
        Args:
            cell: Cell value (can be any type)
            
        Returns:
            Cleaned string value
        """
        if cell is None:
            return ""
        
        cell_str = str(cell).strip()
        
        # Handle common null representations
        if cell_str.lower() in ['nan', 'none', 'null', '-', 'n/a', 'na']:
            return ""
        
        return self.clean(cell_str)
    
    def extract_financial_values(self, text: str) -> Dict[str, List[str]]:
        """
        Extract financial values from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted values by type
        """
        return {
            "currencies": self.CURRENCY_PATTERN.findall(text),
            "percentages": self.PERCENTAGE_PATTERN.findall(text),
            "numbers": self.NUMBER_PATTERN.findall(text)
        }
    
    def remove_headers_footers(
        self, 
        pages: List[str], 
        threshold: float = 0.7
    ) -> List[str]:
        """
        Remove repeated headers and footers from pages.
        
        Args:
            pages: List of page texts
            threshold: Similarity threshold for detection
            
        Returns:
            Pages with headers/footers removed
        """
        if len(pages) < 3:
            return pages
        
        # Detect common lines at start/end of pages
        first_lines = [p.split('\n')[0] if p else '' for p in pages]
        last_lines = [p.split('\n')[-1] if p else '' for p in pages]
        
        header_to_remove = self._find_common_line(first_lines, threshold)
        footer_to_remove = self._find_common_line(last_lines, threshold)
        
        cleaned_pages = []
        for page in pages:
            lines = page.split('\n')
            
            if header_to_remove and lines and lines[0] == header_to_remove:
                lines = lines[1:]
            if footer_to_remove and lines and lines[-1] == footer_to_remove:
                lines = lines[:-1]
            
            cleaned_pages.append('\n'.join(lines))
        
        return cleaned_pages
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        return unicodedata.normalize('NFKC', text)
    
    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters."""
        return self.CONTROL_CHARS.sub('', text)
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self.URL_PATTERN.sub('', text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        return self.EMAIL_PATTERN.sub('', text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        # Replace multiple spaces with single space
        text = self.MULTIPLE_SPACES.sub(' ', text)
        # Replace multiple newlines with double newline
        text = self.MULTIPLE_NEWLINES.sub('\n\n', text)
        return text
    
    def _find_common_line(self, lines: List[str], threshold: float) -> Optional[str]:
        """Find a line that appears in most pages."""
        if not lines:
            return None
        
        line_counts = {}
        for line in lines:
            if line.strip():
                line_counts[line] = line_counts.get(line, 0) + 1
        
        if not line_counts:
            return None
        
        most_common = max(line_counts, key=line_counts.get)
        if line_counts[most_common] / len(lines) >= threshold:
            return most_common
        
        return None


# Convenience function
def clean_text(text: str, **kwargs) -> str:
    """
    Convenience function to clean text.
    
    Args:
        text: Input text
        **kwargs: Configuration options
        
    Returns:
        Cleaned text
    """
    config = CleaningConfig(**kwargs)
    cleaner = TextCleaner(config)
    return cleaner.clean(text)