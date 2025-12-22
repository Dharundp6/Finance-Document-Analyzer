"""
Organism component for document ingestion pipeline.
Orchestrates the full ingestion process.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import uuid
import json

from config.settings import settings
from config.logging_config import logger
from core.types import ProcessingResult, DocumentMetadata
from core.exceptions import DocumentProcessingError
from molecules.document_processor import DocumentProcessor
from molecules.vector_store import VectorStore
from molecules.retriever import HybridRetriever


class IngestionPipeline:
    """
    Complete document ingestion pipeline.
    Handles upload, processing, and indexing of documents.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        document_processor: Optional[DocumentProcessor] = None
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            vector_store: Vector store instance
            document_processor: Document processor instance
        """
        self.vector_store = vector_store or VectorStore()
        self.document_processor = document_processor or DocumentProcessor()
        
        # Document registry
        self.document_registry: Dict[str, DocumentMetadata] = {}
        self._load_registry()
        
        # Ensure directories exist
        Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(settings.processed_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("IngestionPipeline initialized")
    
    def ingest_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        move_file: bool = False
    ) -> ProcessingResult:
        """
        Ingest a single file.
        
        Args:
            file_path: Path to the file
            metadata: Additional metadata
            move_file: Whether to move file to upload directory
            
        Returns:
            Processing result
        """
        file_path = Path(file_path)
        
        logger.info(f"Ingesting file: {file_path.name}")
        
        # Move/copy file if needed
        if move_file:
            dest_path = Path(settings.upload_dir) / file_path.name
            shutil.move(str(file_path), str(dest_path))
            file_path = dest_path
        
        # Process document
        result = self.document_processor.process(
            file_path,
            extract_tables=True,
            extract_financial_entities=True
        )
        
        if result.success:
            # Add chunks to vector store
            if result.chunks:
                self.vector_store.add_chunks(result.chunks)
            
            # Register document
            doc_metadata = DocumentMetadata(
                filename=file_path.name,
                file_type=self.document_processor._get_file_type(file_path),
                file_size=file_path.stat().st_size,
                document_id=result.document_id,
                custom_metadata=metadata or {}
            )
            
            self.document_registry[result.document_id] = doc_metadata
            self._save_registry()
            
            logger.info(f"Successfully ingested: {file_path.name}")
        else:
            logger.error(f"Failed to ingest: {file_path.name}")
        
        return result
    
    def ingest_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None
    ) -> Dict[str, ProcessingResult]:
        """
        Ingest all files in a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to process subdirectories
            file_extensions: List of extensions to process
            
        Returns:
            Dictionary of filename -> processing result
        """
        directory = Path(directory)
        extensions = file_extensions or settings.supported_extensions
        
        logger.info(f"Ingesting directory: {directory}")
        
        results = {}
        
        # Find all matching files
        pattern = "**/*" if recursive else "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    result = self.ingest_file(file_path)
                    results[file_path.name] = result
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
                    results[file_path.name] = ProcessingResult(
                        document_id="",
                        success=False,
                        errors=[str(e)]
                    )
        
        # Rebuild BM25 index
        if results:
            self._rebuild_bm25_index()
        
        successful = sum(1 for r in results.values() if r.success)
        logger.info(f"Ingested {successful}/{len(results)} files")
        
        return results
    
    def ingest_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Ingest file from bytes (for API uploads).
        
        Args:
            file_bytes: File content as bytes
            filename: Original filename
            metadata: Additional metadata
            
        Returns:
            Processing result
        """
        # Save to upload directory
        file_path = Path(settings.upload_dir) / filename
        
        with open(file_path, 'wb') as f:
            f.write(file_bytes)
        
        return self.ingest_file(file_path, metadata=metadata)
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and its chunks.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if successful
        """
        logger.info(f"Deleting document: {document_id}")
        
        try:
            # Delete from vector store
            self.vector_store.delete_document(document_id)
            
            # Remove from registry
            if document_id in self.document_registry:
                del self.document_registry[document_id]
                self._save_registry()
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a document."""
        metadata = self.document_registry.get(document_id)
        
        if not metadata:
            return None
        
        return {
            'document_id': document_id,
            'filename': metadata.filename,
            'file_type': metadata.file_type.value,
            'file_size': metadata.file_size,
            'upload_date': metadata.upload_date.isoformat(),
            'custom_metadata': metadata.custom_metadata
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all ingested documents."""
        return [
            self.get_document_info(doc_id)
            for doc_id in self.document_registry.keys()
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        vector_stats = self.vector_store.get_stats()
        
        return {
            **vector_stats,
            'registered_documents': len(self.document_registry)
        }
    
    def _rebuild_bm25_index(self) -> None:
        """Rebuild BM25 index after ingestion."""
        # This would be called by the retriever
        pass
    
    def _load_registry(self) -> None:
        """Load document registry from disk."""
        registry_path = Path(settings.processed_dir) / "document_registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    data = json.load(f)
                
                for doc_id, doc_data in data.items():
                    from core.types import DocumentType
                    self.document_registry[doc_id] = DocumentMetadata(
                        filename=doc_data['filename'],
                        file_type=DocumentType(doc_data['file_type']),
                        file_size=doc_data['file_size'],
                        document_id=doc_id,
                        upload_date=datetime.fromisoformat(doc_data['upload_date']),
                        custom_metadata=doc_data.get('custom_metadata', {})
                    )
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
    
    def _save_registry(self) -> None:
        """Save document registry to disk."""
        registry_path = Path(settings.processed_dir) / "document_registry.json"
        
        data = {}
        for doc_id, metadata in self.document_registry.items():
            data[doc_id] = {
                'filename': metadata.filename,
                'file_type': metadata.file_type.value,
                'file_size': metadata.file_size,
                'upload_date': metadata.upload_date.isoformat(),
                'custom_metadata': metadata.custom_metadata
            }
        
        with open(registry_path, 'w') as f:
            json.dump(data, f, indent=2)