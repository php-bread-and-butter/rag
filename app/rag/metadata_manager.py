"""
Metadata Manager Module

Provides robust metadata management for tracking file references and sources
throughout the RAG pipeline. Ensures complete traceability from ingestion to query.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import hashlib


class MetadataManager:
    """
    Manages metadata enrichment for documents and chunks.
    Ensures robust file reference tracking throughout the RAG pipeline.
    """
    
    @staticmethod
    def create_file_reference(
        filename: str,
        storage_info: Dict[str, Any],
        input_type: str,
        source_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive file reference metadata.
        
        Args:
            filename: Original filename
            storage_info: Storage information from FileStorageManager
            input_type: Input type ("files", "upload", "s3", "sql", "text")
            source_path: Original source path (for files/S3)
            collection_name: Collection name
            additional_metadata: Any additional metadata to include
            
        Returns:
            Dictionary with file reference metadata
        """
        file_ref = {
            # File identification
            "file_id": storage_info.get("file_id"),
            "original_filename": filename,
            "file_type": MetadataManager._detect_file_type(filename),
            
            # Storage information
            "storage_type": storage_info.get("storage_type", "unknown"),
            "storage_path": storage_info.get("file_path"),
            "file_size_bytes": storage_info.get("size_bytes", 0),
            
            # Source tracking
            "input_type": input_type,
            "source_path": source_path or filename,
            "collection_name": collection_name,
            
            # Timestamps
            "ingested_at": datetime.now().isoformat(),
            
            # Additional metadata
            **(additional_metadata or {})
        }
        
        # Add S3-specific info if applicable
        if storage_info.get("storage_type") == "s3":
            file_ref["s3_bucket"] = storage_info.get("s3_bucket")
            file_ref["s3_key"] = storage_info.get("s3_key")
            file_ref["s3_url"] = storage_info.get("file_path")
        
        return file_ref
    
    @staticmethod
    def enrich_document_metadata(
        document: Any,  # langchain Document
        file_reference: Dict[str, Any],
        document_index: int = 0,
        total_documents: int = 1,
        page_number: Optional[int] = None,
        section_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Enrich document metadata with file reference and document-level info.
        
        Args:
            document: LangChain Document object
            file_reference: File reference metadata
            document_index: Index of this document within the file
            total_documents: Total number of documents from this file
            page_number: Page number (for PDFs)
            section_info: Section/table info (for structured documents)
        """
        # Merge file reference
        document.metadata.update(file_reference)
        
        # Add document-level metadata
        document.metadata.update({
            "document_index": document_index,
            "total_documents_in_file": total_documents,
            "is_last_document": (document_index == total_documents - 1),
        })
        
        # Add page number if provided
        if page_number is not None:
            document.metadata["page_number"] = page_number
            document.metadata["page_index"] = page_number - 1  # 0-based index
        
        # Add section info if provided
        if section_info:
            document.metadata.update(section_info)
    
    @staticmethod
    def enrich_chunk_metadata(
        chunk: Any,  # langchain Document
        source_document: Any,  # langchain Document
        chunk_index: int,
        total_chunks_in_document: int,
        chunk_size: int,
        splitter_type: str,
        embedding_model_name: Optional[str] = None,
        embedding_provider: Optional[str] = None
    ) -> None:
        """
        Enrich chunk metadata with source tracking and processing info.
        
        Args:
            chunk: Chunk Document object
            source_document: Source document this chunk came from
            chunk_index: Index of this chunk within the document
            total_chunks_in_document: Total chunks in the source document
            chunk_size: Chunk size used
            splitter_type: Splitter type used
            embedding_model_name: Embedding model name
            embedding_provider: Embedding provider
        """
        # Preserve all source document metadata
        chunk.metadata.update(source_document.metadata)
        
        # Add chunk-specific metadata
        chunk.metadata.update({
            # Chunk identification
            "chunk_index": chunk_index,
            "total_chunks_in_document": total_chunks_in_document,
            "is_first_chunk": (chunk_index == 0),
            "is_last_chunk": (chunk_index == total_chunks_in_document - 1),
            
            # Processing information
            "processed_at": datetime.now().isoformat(),
            "chunk_size": chunk_size,
            "splitter_type": splitter_type,
            "chunk_length": len(chunk.page_content),
            
            # Embedding information
            "embedding_model": embedding_model_name,
            "embedding_provider": embedding_provider,
        })
        
        # Create chunk ID for unique identification
        chunk.metadata["chunk_id"] = MetadataManager._generate_chunk_id(
            file_id=chunk.metadata.get("file_id"),
            document_index=chunk.metadata.get("document_index", 0),
            chunk_index=chunk_index
        )
    
    @staticmethod
    def create_source_summary(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a human-readable source summary from metadata.
        Useful for displaying in query responses.
        
        Args:
            metadata: Document/chunk metadata
            
        Returns:
            Dictionary with source summary
        """
        summary = {
            "file_name": metadata.get("original_filename", "Unknown"),
            "file_type": metadata.get("file_type", "unknown"),
            "source_path": metadata.get("source_path", "Unknown"),
        }
        
        # Add storage info
        if metadata.get("storage_type") == "s3":
            summary["storage"] = {
                "type": "S3",
                "bucket": metadata.get("s3_bucket"),
                "key": metadata.get("s3_key"),
                "url": metadata.get("s3_url")
            }
        elif metadata.get("storage_type") == "local":
            summary["storage"] = {
                "type": "Local",
                "path": metadata.get("storage_path")
            }
        
        # Add document/chunk info
        if "page_number" in metadata:
            summary["page"] = metadata["page_number"]
        
        if "chunk_index" in metadata:
            summary["chunk"] = {
                "index": metadata["chunk_index"],
                "total": metadata.get("total_chunks_in_document", 1)
            }
        
        # Add timestamps
        if "ingested_at" in metadata:
            summary["ingested_at"] = metadata["ingested_at"]
        
        return summary
    
    @staticmethod
    def _detect_file_type(filename: str) -> str:
        """Detect file type from filename extension"""
        ext = Path(filename).suffix.lower()
        type_mapping = {
            ".txt": "text",
            ".text": "text",
            ".pdf": "pdf",
            ".docx": "word",
            ".doc": "word",
            ".csv": "csv",
            ".xlsx": "excel",
            ".xls": "excel",
            ".json": "json",
            ".jsonl": "jsonl",
            ".db": "sql",
            ".sqlite": "sql",
            ".sqlite3": "sql",
        }
        return type_mapping.get(ext, "unknown")
    
    @staticmethod
    def _generate_chunk_id(
        file_id: Optional[str],
        document_index: int,
        chunk_index: int
    ) -> str:
        """Generate unique chunk ID"""
        if file_id:
            base = f"{file_id}_{document_index}_{chunk_index}"
        else:
            base = f"chunk_{document_index}_{chunk_index}_{datetime.now().timestamp()}"
        
        # Create short hash for uniqueness
        hash_obj = hashlib.md5(base.encode())
        short_hash = hash_obj.hexdigest()[:8]
        return f"{base}_{short_hash}"


def enrich_metadata_for_training(
    documents: List[Any],
    storage_info: Dict[str, Any],
    input_type: str,
    filename: str,
    collection_name: str,
    source_path: Optional[str] = None,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Convenience function to enrich metadata for all documents from a file.
    
    Args:
        documents: List of Document objects
        storage_info: Storage information from FileStorageManager
        input_type: Input type
        filename: Original filename
        collection_name: Collection name
        source_path: Original source path
        additional_metadata: Additional metadata
    """
    # Create file reference
    file_ref = MetadataManager.create_file_reference(
        filename=filename,
        storage_info=storage_info,
        input_type=input_type,
        source_path=source_path,
        collection_name=collection_name,
        additional_metadata=additional_metadata
    )
    
    # Enrich each document
    total_docs = len(documents)
    for idx, doc in enumerate(documents):
        # Extract page number if available
        page_num = doc.metadata.get("page", doc.metadata.get("page_number"))
        if page_num is not None:
            try:
                page_num = int(page_num) + 1 if isinstance(page_num, (int, str)) and str(page_num).isdigit() else None
            except (ValueError, TypeError):
                page_num = None
        
        MetadataManager.enrich_document_metadata(
            document=doc,
            file_reference=file_ref,
            document_index=idx,
            total_documents=total_docs,
            page_number=page_num
        )
