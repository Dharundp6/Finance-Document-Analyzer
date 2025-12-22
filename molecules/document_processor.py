"""
Molecule component for processing documents.
Combines file reading, cleaning, chunking, and extraction.
"""

import os
from pathlib import Path
from typing import List, Optional, Union, BinaryIO
from datetime import datetime
import uuid
import time

from config.settings import settings
from config.logging_config import logger
from core.types import (
    DocumentMetadata, TextChunk, TableData, 
    FinancialEntity, ProcessingResult, DocumentType, ChunkType
)
from core.exceptions import (
    DocumentProcessingError, FileSizeExceededError, 
    UnsupportedFileTypeError
)
from atoms.file_reader import FileReaderFactory
from atoms.text_cleaner import TextCleaner, CleaningConfig
from atoms.chunk_creator import FinancialAwareChunker, ChunkingConfig
from atoms.financial_extractor import FinancialExtractor


class DocumentProcessor:
    """
    Processes documents through the full pipeline:
    reading -> cleaning -> chunking -> extraction.
    """
    
    EXTENSION_TO_TYPE = {
        '.pdf': DocumentType.PDF,
        '.docx': DocumentType.WORD,
        '.doc': DocumentType.WORD,
        '.xlsx': DocumentType.EXCEL,
        '.xls': DocumentType.EXCEL,
        '.csv': DocumentType.CSV,
        '.txt': DocumentType.TEXT
    }
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Initialize components
        self.text_cleaner = TextCleaner(CleaningConfig())
        self.chunker = FinancialAwareChunker(ChunkingConfig(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        ))
        self.financial_extractor = FinancialExtractor()
        
        logger.info("DocumentProcessor initialized")
    
    def process(
        self,
        file_path: Union[str, Path],
        extract_tables: bool = True,
        extract_financial_entities: bool = True
    ) -> ProcessingResult:
        """
        Process a document through the full pipeline.
        
        Args:
            file_path: Path to the document
            extract_tables: Whether to extract tables
            extract_financial_entities: Whether to extract financial entities
            
        Returns:
            ProcessingResult with chunks, tables, and entities
        """
        start_time = time.time()
        file_path = Path(file_path)
        document_id = str(uuid.uuid4())
        
        logger.info(f"Processing document: {file_path.name} (ID: {document_id})")
        
        # Validate file
        self._validate_file(file_path)
        
        # Get file type
        file_type = self._get_file_type(file_path)
        
        # Create metadata
        metadata = self._create_metadata(file_path, file_type, document_id)
        
        try:
            # Read document
            reader = FileReaderFactory.get_reader(file_type)
            raw_content = reader.read(str(file_path))
            
            # Process based on document type
            chunks = []
            tables = []
            financial_entities = []
            
            if file_type in [DocumentType.CSV, DocumentType.EXCEL]:
                # Handle tabular documents
                chunks, tables = self._process_tabular(
                    raw_content, document_id, file_type
                )
            else:
                # Handle text-based documents
                chunks = self._process_text_document(
                    raw_content, document_id, file_type
                )
                
                if extract_tables and 'tables' in raw_content:
                    tables = self._process_tables(
                        raw_content['tables'], document_id
                    )
            
            # Extract financial entities from all chunks
            if extract_financial_entities:
                all_text = ' '.join([c.content for c in chunks])
                financial_entities = self.financial_extractor.extract_all(all_text)
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"Document processed: {len(chunks)} chunks, "
                f"{len(tables)} tables, {len(financial_entities)} entities "
                f"in {processing_time:.2f}s"
            )
            
            return ProcessingResult(
                document_id=document_id,
                success=True,
                chunks=chunks,
                tables=tables,
                financial_entities=financial_entities,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return ProcessingResult(
                document_id=document_id,
                success=False,
                errors=[str(e)],
                processing_time=time.time() - start_time
            )
    
    def _validate_file(self, file_path: Path) -> None:
        """Validate file exists, size, and type."""
        if not file_path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")
        
        # Check file size
        file_size = file_path.stat().st_size
        max_size = settings.max_file_size_mb * 1024 * 1024
        
        if file_size > max_size:
            raise FileSizeExceededError(file_size, max_size)
        
        # Check extension
        ext = file_path.suffix.lower()
        if ext not in self.EXTENSION_TO_TYPE:
            raise UnsupportedFileTypeError(ext)
    
    def _get_file_type(self, file_path: Path) -> DocumentType:
        """Get document type from file extension."""
        ext = file_path.suffix.lower()
        return self.EXTENSION_TO_TYPE.get(ext)
    
    def _create_metadata(
        self,
        file_path: Path,
        file_type: DocumentType,
        document_id: str
    ) -> DocumentMetadata:
        """Create document metadata."""
        stat = file_path.stat()
        
        return DocumentMetadata(
            filename=file_path.name,
            file_type=file_type,
            file_size=stat.st_size,
            document_id=document_id,
            modified_date=datetime.fromtimestamp(stat.st_mtime)
        )
    
    def _process_text_document(
        self,
        raw_content: dict,
        document_id: str,
        file_type: DocumentType
    ) -> List[TextChunk]:
        """Process text-based documents (PDF, Word, Text)."""
        chunks = []
        
        if file_type == DocumentType.PDF:
            # Process page by page
            for page_data in raw_content.get('pages', []):
                page_text = page_data.get('text', '')
                page_num = page_data.get('page_number')
                
                if page_text.strip():
                    cleaned_text = self.text_cleaner.clean(page_text)
                    page_chunks = self.chunker.chunk(
                        cleaned_text,
                        document_id=document_id,
                        page_number=page_num,
                        metadata={'source_type': 'pdf'}
                    )
                    chunks.extend(page_chunks)
        
        elif file_type == DocumentType.WORD:
            # Combine paragraphs
            paragraphs = raw_content.get('paragraphs', [])
            full_text = '\n\n'.join([p['text'] for p in paragraphs])
            
            if full_text.strip():
                cleaned_text = self.text_cleaner.clean(full_text)
                chunks = self.chunker.chunk(
                    cleaned_text,
                    document_id=document_id,
                    metadata={'source_type': 'word'}
                )
        
        elif file_type == DocumentType.TEXT:
            text = raw_content.get('text', '')
            if text.strip():
                cleaned_text = self.text_cleaner.clean(text)
                chunks = self.chunker.chunk(
                    cleaned_text,
                    document_id=document_id,
                    metadata={'source_type': 'text'}
                )
        
        return chunks
    
    def _process_tabular(
        self,
        raw_content: dict,
        document_id: str,
        file_type: DocumentType
    ) -> tuple:
        """Process tabular documents (CSV, Excel)."""
        chunks = []
        tables = []
        
        if file_type == DocumentType.CSV:
            # Single table
            table = TableData(
                table_id=str(uuid.uuid4()),
                headers=raw_content.get('headers', []),
                rows=raw_content.get('rows', []),
                document_id=document_id
            )
            tables.append(table)
            
            # Create chunks from table
            chunks.extend(self._table_to_chunks(table, document_id))
            
            # Add summary chunk
            if 'summary' in raw_content:
                summary_chunk = TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    content=f"Table Summary:\n{self._format_summary(raw_content['summary'])}",
                    chunk_type=ChunkType.METADATA,
                    document_id=document_id,
                    metadata={'type': 'summary'}
                )
                chunks.append(summary_chunk)
        
        elif file_type == DocumentType.EXCEL:
            # Multiple sheets
            for sheet in raw_content.get('sheets', []):
                table = TableData(
                    table_id=str(uuid.uuid4()),
                    headers=sheet.get('headers', []),
                    rows=sheet.get('rows', []),
                    document_id=document_id,
                    caption=sheet.get('sheet_name')
                )
                tables.append(table)
                
                # Create chunks from each sheet
                sheet_chunks = self._table_to_chunks(
                    table, document_id,
                    metadata={'sheet_name': sheet.get('sheet_name')}
                )
                chunks.extend(sheet_chunks)
        
        return chunks, tables
    
    def _process_tables(
        self,
        tables_data: List[dict],
        document_id: str
    ) -> List[TableData]:
        """Process extracted tables from PDF/Word."""
        tables = []
        
        for idx, table_data in enumerate(tables_data):
            if not table_data.get('headers') and not table_data.get('rows'):
                continue
            
            table = TableData(
                table_id=str(uuid.uuid4()),
                headers=table_data.get('headers', []),
                rows=table_data.get('rows', []),
                document_id=document_id,
                page_number=table_data.get('page_number'),
                table_index=idx
            )
            tables.append(table)
        
        return tables
    
    def _table_to_chunks(
        self,
        table: TableData,
        document_id: str,
        metadata: dict = None
    ) -> List[TextChunk]:
        """Convert table to searchable chunks."""
        chunks = []
        
        # Schema chunk (column information)
        schema_content = f"Table Columns: {', '.join(table.headers)}"
        if table.caption:
            schema_content = f"Sheet/Table: {table.caption}\n{schema_content}"
        
        schema_chunk = TextChunk(
            chunk_id=str(uuid.uuid4()),
            content=schema_content,
            chunk_type=ChunkType.TABLE,
            document_id=document_id,
            metadata={
                **(metadata or {}),
                'table_id': table.table_id,
                'type': 'schema'
            }
        )
        chunks.append(schema_chunk)
        
        # Markdown representation of table
        markdown_table = table.to_markdown()
        if markdown_table:
            # Split large tables into chunks
            if len(markdown_table) > self.chunk_size:
                # Create row-based chunks
                for i, row in enumerate(table.rows):
                    row_content = ' | '.join(
                        f"{h}: {v}" for h, v in zip(table.headers, row)
                    )
                    row_chunk = TextChunk(
                        chunk_id=str(uuid.uuid4()),
                        content=row_content,
                        chunk_type=ChunkType.TABLE,
                        document_id=document_id,
                        metadata={
                            **(metadata or {}),
                            'table_id': table.table_id,
                            'row_index': i
                        }
                    )
                    chunks.append(row_chunk)
            else:
                table_chunk = TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    content=markdown_table,
                    chunk_type=ChunkType.TABLE,
                    document_id=document_id,
                    metadata={
                        **(metadata or {}),
                        'table_id': table.table_id,
                        'type': 'full_table'
                    }
                )
                chunks.append(table_chunk)
        
        return chunks
    
    def _format_summary(self, summary: dict) -> str:
        """Format summary statistics."""
        lines = []
        for col, stats in summary.items():
            stat_str = ', '.join(
                f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in stats.items() if v is not None
            )
            lines.append(f"{col}: {stat_str}")
        return '\n'.join(lines)