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
from app.rag.metadata_manager import MetadataManager, enrich_metadata_for_training
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
        
        # Step 2: Ingest documents and enrich metadata
        loader_config = config.get("loader_config", {})
        loader = UnifiedDocumentLoader(**loader_config)
        
        all_documents = []
        
        # Process each file and enrich metadata
        for idx, file_path in enumerate(file_paths):
            file_docs = loader.load_files([file_path], config.get("metadata"))
            
            # Find corresponding storage info (match by index or filename)
            file_storage_info = stored_files[idx] if idx < len(stored_files) else stored_files[0] if stored_files else {}
            
            # Enrich metadata for documents from this file
            filename = Path(file_path).name
            enrich_metadata_for_training(
                documents=file_docs,
                storage_info=file_storage_info,
                input_type="files",
                filename=filename,
                collection_name=collection_name,
                source_path=file_path,
                additional_metadata=config.get("metadata")
            )
            
            all_documents.extend(file_docs)
        
        logger.info(f"Ingested {len(all_documents)} document(s)")
        
        # Step 3: Process chunks (embedding config will be added during vector storage)
        chunks = self._process_chunks(all_documents, config)
        
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
        
        # Load documents with base metadata
        documents = loader.load_bytes(
            filename=filename,
            content=file_content,
            metadata=config.get("metadata", {}),
            use_pymupdf=config.get("use_pymupdf", True),
            password=config.get("password"),
            max_pages=config.get("max_pages")
        )
        
        # Enrich metadata with file reference
        enrich_metadata_for_training(
            documents=documents,
            storage_info=storage_info,
            input_type="upload",
            filename=filename,
            collection_name=collection_name,
            source_path=filename,
            additional_metadata=config.get("metadata")
        )
        
        logger.info(f"Ingested {len(documents)} document(s) from uploaded file")
        
        # Step 3: Process chunks (embedding config will be added during vector storage)
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
                file_docs = loader.load_bytes(
                    filename=filename,
                    content=content,
                    metadata=config.get("metadata", {}),
                    use_pymupdf=config.get("use_pymupdf", True),
                    password=config.get("password"),
                    max_pages=config.get("max_pages")
                )
                
                # Enrich metadata with file reference
                enrich_metadata_for_training(
                    documents=file_docs,
                    storage_info=storage_info,
                    input_type="s3",
                    filename=filename,
                    collection_name=collection_name,
                    source_path=s3_path,
                    additional_metadata={**(config.get("metadata", {})), "s3_path": s3_path}
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
        
        # Enrich metadata with file reference
        enrich_metadata_for_training(
            documents=documents,
            storage_info=storage_info,
            input_type="sql",
            filename=f"{Path(db_path).name}",
            collection_name=collection_name,
            source_path=db_path,
            additional_metadata={**(config.get("metadata", {})), "db_path": db_path, "sql_query": input_data.get("query")}
        )
        
        # Step 3: Process chunks (embedding config will be added during vector storage)
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
        
        # Step 1: Store text as file
        storage_info = await self.storage_manager.save_file(
            content=text.encode('utf-8'),
            filename="text_input.txt",
            collection_name=collection_name,
            metadata={"type": "text_input", "char_count": str(len(text))}
        )
        
        # Step 2: Create document
        document = Document(
            page_content=text,
            metadata=config.get("metadata", {})
        )
        
        # Enrich metadata with file reference
        enrich_metadata_for_training(
            documents=[document],
            storage_info=storage_info,
            input_type="text",
            filename="text_input.txt",
            collection_name=collection_name,
            source_path="text_input",
            additional_metadata={**(config.get("metadata", {})), "char_count": len(text)}
        )
        
        # Step 3: Process chunks
        chunks = self._process_chunks([document], config, embedding_config={
            "model_name": config.get("embedding_model_name", "all-MiniLM-L6-v2"),
            "provider": config.get("embedding_provider")
        })
        
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
        """
        Process documents into chunks with enriched metadata.
        
        Args:
            documents: List of Document objects
            config: Processing configuration
            embedding_config: Embedding configuration (optional)
        """
        splitter_manager = TextSplitterManager(
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200),
            splitter_type=config.get("splitter_type", "recursive")
        )
        
        all_chunks = []
        
        # Process each document separately to track chunk indices per document
        for doc_idx, document in enumerate(documents):
            doc_chunks = splitter_manager.split_documents([document])
            
            # Enrich each chunk with metadata
            total_chunks_in_doc = len(doc_chunks)
            for chunk_idx, chunk in enumerate(doc_chunks):
                MetadataManager.enrich_chunk_metadata(
                    chunk=chunk,
                    source_document=document,
                    chunk_index=chunk_idx,
                    total_chunks_in_document=total_chunks_in_doc,
                    chunk_size=config.get("chunk_size", 1000),
                    splitter_type=config.get("splitter_type", "recursive"),
                    embedding_model_name=None,  # Will be set during vector storage
                    embedding_provider=None  # Will be set during vector storage
                )
            
            all_chunks.extend(doc_chunks)
        
        logger.info(f"Created {len(all_chunks)} chunk(s) from {len(documents)} document(s)")
        return all_chunks
    
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
        
        # Update embedding info in chunk metadata before storing
        embedding_model_name = embedding_config.get("model_name", "all-MiniLM-L6-v2")
        embedding_provider = embedding_config.get("provider")
        
        for chunk in chunks:
            chunk.metadata["embedding_model"] = embedding_model_name
            chunk.metadata["embedding_provider"] = embedding_provider
        
        chunk_ids = vector_store.add_documents(chunks)
        logger.info(f"Stored {len(chunk_ids)} chunk(s) in vector store | Collection: {collection_name}")
        return chunk_ids
