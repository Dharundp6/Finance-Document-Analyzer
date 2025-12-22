"""
Atomic component for creating text chunks.
Implements various chunking strategies optimized for financial documents.
"""

import re
from typing import List, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import uuid

from config.settings import settings
from config.logging_config import logger
from core.types import TextChunk, ChunkType


@dataclass
class ChunkingConfig:
    """Configuration for chunking."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    separators: List[str] = None
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ". ", " ", ""]


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    
    @abstractmethod
    def chunk(self, text: str, document_id: str, **kwargs) -> List[TextChunk]:
        """Split text into chunks."""
        pass


class RecursiveChunker(BaseChunker):
    """
    Recursive text chunker that splits on multiple separators.
    Optimized for maintaining context in financial documents.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize the chunker.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_size=settings.min_chunk_size
        )
    
    def chunk(
        self, 
        text: str, 
        document_id: str,
        chunk_type: ChunkType = ChunkType.TEXT,
        page_number: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> List[TextChunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            document_id: ID of source document
            chunk_type: Type of chunk
            page_number: Page number if applicable
            metadata: Additional metadata
            
        Returns:
            List of TextChunk objects
        """
        if not text or len(text.strip()) < self.config.min_chunk_size:
            return []
        
        chunks = self._split_text(text)
        
        result = []
        current_index = 0
        
        for i, chunk_text in enumerate(chunks):
            end_index = current_index + len(chunk_text)
            
            chunk = TextChunk(
                chunk_id=str(uuid.uuid4()),
                content=chunk_text.strip(),
                chunk_type=chunk_type,
                document_id=document_id,
                page_number=page_number,
                start_index=current_index,
                end_index=end_index,
                metadata={
                    **(metadata or {}),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )
            result.append(chunk)
            
            # Update index accounting for overlap
            current_index = end_index - self.config.chunk_overlap
        
        logger.debug(f"Created {len(result)} chunks from text")
        return result
    
    def _split_text(self, text: str) -> List[str]:
        """Recursively split text using separators."""
        return self._recursive_split(text, self.config.separators)
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using a list of separators.
        
        Args:
            text: Text to split
            separators: List of separators to try
            
        Returns:
            List of text chunks
        """
        if not separators:
            # No more separators, just split by character count
            return self._split_by_chars(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split) + len(separator)
            
            if current_length + split_length > self.config.chunk_size and current_chunk:
                # Current chunk is full, save it
                chunk_text = separator.join(current_chunk)
                
                if len(chunk_text) > self.config.chunk_size:
                    # Chunk too large, recursively split
                    chunks.extend(self._recursive_split(chunk_text, remaining_separators))
                else:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, separator)
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text) if overlap_text else 0
            
            current_chunk.append(split)
            current_length += split_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if len(chunk_text) > self.config.chunk_size:
                chunks.extend(self._recursive_split(chunk_text, remaining_separators))
            elif len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def _split_by_chars(self, text: str) -> List[str]:
        """Split text by character count with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config.chunk_size
            chunk = text[start:end]
            
            if len(chunk) >= self.config.min_chunk_size:
                chunks.append(chunk)
            
            start = end - self.config.chunk_overlap
        
        return chunks
    
    def _get_overlap_text(self, parts: List[str], separator: str) -> str:
        """Get text for overlap from previous chunk."""
        if not parts:
            return ""
        
        overlap_chars = 0
        overlap_parts = []
        
        for part in reversed(parts):
            part_len = len(part) + len(separator)
            if overlap_chars + part_len > self.config.chunk_overlap:
                break
            overlap_parts.insert(0, part)
            overlap_chars += part_len
        
        return separator.join(overlap_parts)


class SentenceChunker(BaseChunker):
    """
    Chunker that splits on sentence boundaries.
    Better for maintaining semantic coherence.
    """
    
    SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
    
    def chunk(
        self,
        text: str,
        document_id: str,
        chunk_type: ChunkType = ChunkType.TEXT,
        page_number: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> List[TextChunk]:
        """Split text into chunks at sentence boundaries."""
        sentences = self.SENTENCE_PATTERN.split(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.config.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.config.min_chunk_size:
                    chunks.append(chunk_text)
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1
        
        # Last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(chunk_text)
        
        # Convert to TextChunk objects
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk = TextChunk(
                chunk_id=str(uuid.uuid4()),
                content=chunk_text,
                chunk_type=chunk_type,
                document_id=document_id,
                page_number=page_number,
                metadata={
                    **(metadata or {}),
                    "chunk_index": i,
                    "chunking_method": "sentence"
                }
            )
            result.append(chunk)
        
        return result


class FinancialAwareChunker(BaseChunker):
    """
    Chunker optimized for financial documents.
    Preserves financial sections, tables, and key data points.
    """
    
    # Patterns for financial sections
    SECTION_PATTERNS = [
        r'(?:^|\n)(#{1,3}\s+.+)',  # Markdown headers
        r'(?:^|\n)([A-Z][A-Z\s]+:)',  # ALL CAPS headers
        r'(?:^|\n)((?:Item|Section|Part)\s+\d+[.:]\s*.+)',  # Numbered sections
        r'(?:^|\n)((?:Revenue|Income|Expense|Asset|Liability|Equity).*:)',  # Financial headers
    ]
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.recursive_chunker = RecursiveChunker(config)
    
    def chunk(
        self,
        text: str,
        document_id: str,
        chunk_type: ChunkType = ChunkType.TEXT,
        page_number: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> List[TextChunk]:
        """
        Split text into chunks while preserving financial sections.
        """
        # First, try to split by financial sections
        sections = self._split_by_sections(text)
        
        all_chunks = []
        
        for section_idx, section in enumerate(sections):
            section_metadata = {
                **(metadata or {}),
                "section_index": section_idx
            }
            
            if len(section) <= self.config.chunk_size:
                # Section fits in one chunk
                if len(section) >= self.config.min_chunk_size:
                    chunk = TextChunk(
                        chunk_id=str(uuid.uuid4()),
                        content=section.strip(),
                        chunk_type=self._detect_chunk_type(section),
                        document_id=document_id,
                        page_number=page_number,
                        metadata=section_metadata
                    )
                    all_chunks.append(chunk)
            else:
                # Section too large, use recursive chunking
                section_chunks = self.recursive_chunker.chunk(
                    section,
                    document_id=document_id,
                    chunk_type=chunk_type,
                    page_number=page_number,
                    metadata=section_metadata
                )
                all_chunks.extend(section_chunks)
        
        return all_chunks
    
    def _split_by_sections(self, text: str) -> List[str]:
        """Split text by financial section headers."""
        # Combine all section patterns
        combined_pattern = '|'.join(self.SECTION_PATTERNS)
        
        # Find all section headers
        splits = re.split(combined_pattern, text)
        
        # Filter empty splits and rejoin headers with content
        sections = []
        current_section = []
        
        for split in splits:
            if split and split.strip():
                # Check if this is a header
                is_header = any(
                    re.match(pattern, split.strip()) 
                    for pattern in self.SECTION_PATTERNS
                )
                
                if is_header and current_section:
                    sections.append('\n'.join(current_section))
                    current_section = [split.strip()]
                else:
                    current_section.append(split.strip())
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return sections if sections else [text]
    
    def _detect_chunk_type(self, text: str) -> ChunkType:
        """Detect the type of content in the chunk."""
        text_lower = text.lower()
        
        # Check for table indicators
        if '|' in text and text.count('|') > 5:
            return ChunkType.TABLE
        
        # Check for financial data indicators
        financial_keywords = [
            'revenue', 'income', 'expense', 'profit', 'loss',
            'asset', 'liability', 'equity', 'cash flow',
            'balance sheet', 'income statement'
        ]
        if any(kw in text_lower for kw in financial_keywords):
            return ChunkType.FINANCIAL_DATA
        
        # Check for headers
        if len(text) < 200 and text.strip().isupper():
            return ChunkType.HEADER
        
        return ChunkType.TEXT


def create_chunks(
    text: str,
    document_id: str,
    strategy: str = "financial",
    **kwargs
) -> List[TextChunk]:
    """
    Convenience function to create chunks.
    
    Args:
        text: Text to chunk
        document_id: Source document ID
        strategy: Chunking strategy ('recursive', 'sentence', 'financial')
        **kwargs: Additional arguments passed to chunker
        
    Returns:
        List of TextChunk objects
    """
    chunkers = {
        "recursive": RecursiveChunker,
        "sentence": SentenceChunker,
        "financial": FinancialAwareChunker
    }
    
    chunker_class = chunkers.get(strategy, FinancialAwareChunker)
    chunker = chunker_class()
    
    return chunker.chunk(text, document_id, **kwargs)