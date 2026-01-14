"""
Unified Training Processor

Handles the complete training pipeline:
1. Route to appropriate data source processor
2. Process and return chunks
3. Store raw files (S3/local)
4. Store vectors (ChromaDB)
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from langchain_core.documents import Document

from app.rag.unified_loader import UnifiedDocumentLoader
from app.rag.text_splitters import TextSplitterManager
from app.rag.storage import FileStorageManager
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class TrainingProcessor:
    """
    Unified processor for RAG training.
    Routes to appropriate processors and handles the complete pipeline.
    """
    
    def __init__(self):
        self.storage_manager = FileStorageManager()
        logger.info("TrainingProcessor initialized")
    
    async def process_training_request(
        self,
        input_type: str,  # "files", "upload", "s3", "sql", "text"
        input_data: Dict[str, Any],
        collection_name: str,
        processing_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main entry point for training requests.
        
        Args:
            input_type: Type of input ("files", "upload", "s3", "sql", "text")
            input_data: Input-specific data
            collection_name: ChromaDB collection name
            processing_config: Processing parameters (splitting, embedding, etc.)
        
        Returns:
            dict with chunks and storage info:
            {
                "documents": List[Document],
                "chunks": List[Document],
                "stored_files": List[dict]
            }
        """
        logger.info(
            f"Processing training request | "
            f"Input type: {input_type} | "
            f"Collection: {collection_name}"
        )
        
        # Route to appropriate processor
        if input_type == "files":
            return await self._process_files(input_data, collection_name, processing_config)
        elif input_type == "upload":
            return await self._process_upload(input_data, collection_name, processing_config)
        elif input_type == "s3":
            return await self._process_s3(input_data, collection_name, processing_config)
        elif input_type == "sql":
            return await self._process_sql(input_data, collection_name, processing_config)
        elif input_type == "text":
            return await self._process_text(input_data, collection_name, processing_config)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
    
    async def _process_files(
        self,
        input_data: Dict[str, Any],
        collection_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process files from server paths"""
        file_paths = input_data["file_paths"]
        
        logger.info(f"Processing {len(file_paths)} file(s) from server paths")
        
        # Step 1: Store raw files (copy to storage)
        stored_files = []
        for file_path in file_paths:
            try:
                content = Path(file_path).read_bytes()
                filename = Path(file_path).name
                storage_info = await self.storage_manager.save_file(
                    content=content,
                    filename=filename,
                    collection_name=collection_name,
                    metadata={"source_path": file_path}
                )
                stored_files.append(storage_info)
            except Exception as e:
                logger.error(f"Failed to store file {file_path}: {str(e)}", exc_info=True)
                raise
        
        # Step 2: Ingest documents
        loader_config = config.get("loader_config", {})
        loader = UnifiedDocumentLoader(**loader_config)
        documents = loader.load_files(
            file_paths,
            config.get("metadata")
        )
        logger.info(f"Ingested {len(documents)} document(s)")
        
        # Step 3: Process chunks
        chunks = self._process_chunks(documents, config)
        
        return {
            "documents": documents,
            "chunks": chunks,
            "stored_files": stored_files
        }
    
    async def _process_upload(
        self,
        input_data: Dict[str, Any],
        collection_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process uploaded file"""
        file_content = input_data["content"]
        filename = input_data["filename"]
        
        logger.info(f"Processing uploaded file | Filename: {filename} | Size: {len(file_content)} bytes")
        
        # Step 1: Store raw file first
        storage_info = await self.storage_manager.save_file(
            content=file_content,
            filename=filename,
            collection_name=collection_name,
            metadata={"uploaded": "true"}
        )
        
        # Step 2: Ingest document
        loader_config = config.get("loader_config", {})
        loader = UnifiedDocumentLoader(**loader_config)
        
        # Add storage path to metadata
        metadata = config.get("metadata", {})
        metadata["storage_path"] = storage_info["file_path"]
        
        documents = loader.load_bytes(
            filename=filename,
            content=file_content,
            metadata=metadata,
            use_pymupdf=config.get("use_pymupdf", True),
            password=config.get("password"),
            max_pages=config.get("max_pages")
        )
        logger.info(f"Ingested {len(documents)} document(s) from uploaded file")
        
        # Step 3: Process chunks
        chunks = self._process_chunks(documents, config)
        
        return {
            "documents": documents,
            "chunks": chunks,
            "stored_files": [storage_info]
        }
    
    async def _process_s3(
        self,
        input_data: Dict[str, Any],
        collection_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process files from S3"""
        s3_paths = input_data["s3_paths"]  # List of S3 paths
        
        logger.info(f"Processing {len(s3_paths)} file(s) from S3")
        
        # Step 1: Download and store files
        stored_files = []
        documents = []
        
        loader_config = config.get("loader_config", {})
        loader = UnifiedDocumentLoader(**loader_config)
        
        for s3_path in s3_paths:
            try:
                # Download from S3
                content = await self.storage_manager.get_file(s3_path)
                filename = s3_path.split("/")[-1]
                
                # Store in our storage (for consistency and backup)
                storage_info = await self.storage_manager.save_file(
                    content=content,
                    filename=filename,
                    collection_name=collection_name,
                    metadata={"source_s3_path": s3_path}
                )
                stored_files.append(storage_info)
                
                # Ingest document
                metadata = config.get("metadata", {})
                metadata["s3_path"] = s3_path
                metadata["storage_path"] = storage_info["file_path"]
                
                file_docs = loader.load_bytes(
                    filename=filename,
                    content=content,
                    metadata=metadata,
                    use_pymupdf=config.get("use_pymupdf", True),
                    password=config.get("password"),
                    max_pages=config.get("max_pages")
                )
                documents.extend(file_docs)
            except Exception as e:
                logger.error(f"Failed to process S3 file {s3_path}: {str(e)}", exc_info=True)
                raise
        
        logger.info(f"Ingested {len(documents)} document(s) from S3")
        
        # Step 2: Process chunks
        chunks = self._process_chunks(documents, config)
        
        return {
            "documents": documents,
            "chunks": chunks,
            "stored_files": stored_files
        }
    
    async def _process_sql(
        self,
        input_data: Dict[str, Any],
        collection_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process SQL database"""
        from app.rag.sql_loaders import SQLDocumentLoader
        
        db_path = input_data["db_path"]
        
        logger.info(f"Processing SQL database | Path: {db_path}")
        
        # Step 1: Load SQL data
        sql_loader_config = config.get("sql_loader_config", {})
        sql_loader = SQLDocumentLoader(**sql_loader_config)
        
        documents = sql_loader.load_database(
            db_path,
            input_data.get("query"),
            config.get("metadata")
        )
        logger.info(f"Ingested {len(documents)} document(s) from SQL database")
        
        # Step 2: Store database reference (not the actual DB file, just metadata)
        db_reference = f"SQL Database: {db_path}\nCollection: {collection_name}\nProcessed: {datetime.now().isoformat()}"
        storage_info = await self.storage_manager.save_file(
            content=db_reference.encode('utf-8'),
            filename=f"{Path(db_path).name}.reference",
            collection_name=collection_name,
            metadata={
                "type": "sql_reference",
                "db_path": db_path,
                "document_count": str(len(documents))
            }
        )
        
        # Step 3: Process chunks
        chunks = self._process_chunks(documents, config)
        
        return {
            "documents": documents,
            "chunks": chunks,
            "stored_files": [storage_info]
        }
    
    async def _process_text(
        self,
        input_data: Dict[str, Any],
        collection_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process raw text"""
        text = input_data["text"]
        
        logger.info(f"Processing raw text | Length: {len(text)} characters")
        
        # Step 1: Create document
        document = Document(
            page_content=text,
            metadata=config.get("metadata", {})
        )
        
        # Step 2: Store text as file
        storage_info = await self.storage_manager.save_file(
            content=text.encode('utf-8'),
            filename="text_input.txt",
            collection_name=collection_name,
            metadata={"type": "text_input", "char_count": str(len(text))}
        )
        
        # Add storage path to document metadata
        document.metadata["storage_path"] = storage_info["file_path"]
        
        # Step 3: Process chunks
        chunks = self._process_chunks([document], config)
        
        return {
            "documents": [document],
            "chunks": chunks,
            "stored_files": [storage_info]
        }
    
    def _process_chunks(
        self,
        documents: List[Document],
        config: Dict[str, Any]
    ) -> List[Document]:
        """Process documents into chunks"""
        splitter_manager = TextSplitterManager(
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200),
            splitter_type=config.get("splitter_type", "recursive")
        )
        
        chunks = splitter_manager.split_documents(documents)
        
        # Add processing metadata to chunks
        for chunk in chunks:
            chunk.metadata["processed_at"] = datetime.now().isoformat()
            chunk.metadata["chunk_size"] = config.get("chunk_size")
            chunk.metadata["splitter_type"] = config.get("splitter_type")
        
        logger.info(f"Created {len(chunks)} chunk(s) from {len(documents)} document(s)")
        return chunks
    
    async def store_vectors(
        self,
        chunks: List[Document],
        collection_name: str,
        embedding_config: Dict[str, Any]
    ) -> List[str]:
        """
        Store chunks in vector store
        
        Args:
            chunks: List of Document chunks to store
            collection_name: ChromaDB collection name
            embedding_config: Embedding configuration
        
        Returns:
            List of chunk IDs stored in vector store
        """
        from app.rag.vector_store import get_vector_store_manager
        
        vector_store = get_vector_store_manager(
            collection_name=collection_name,
            embedding_model_name=embedding_config.get("model_name", "all-MiniLM-L6-v2"),
            embedding_provider=embedding_config.get("provider")
        )
        
        chunk_ids = vector_store.add_documents(chunks)
        logger.info(f"Stored {len(chunk_ids)} chunk(s) in vector store | Collection: {collection_name}")
        return chunk_ids
