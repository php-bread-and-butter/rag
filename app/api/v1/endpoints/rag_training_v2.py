"""
RAG Training API Endpoints V2

Unified endpoint for training RAG system with multiple input types.
Stores raw files (S3/local) and vectors (ChromaDB).
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, status, Form
from pydantic import BaseModel, Field, model_validator

from app.rag.training_processor import TrainingProcessor
from app.rag.vector_store import get_vector_store_manager
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


class UnifiedTrainRequest(BaseModel):
    """Unified training request accepting multiple input types"""
    
    # Input type and data
    input_type: str = Field(
        ...,
        description="Input type: 'files' (server paths), 's3' (S3 paths), 'sql' (SQLite DB), 'text' (raw text)",
        example="files"
    )
    
    # Input-specific data (only one should be provided based on input_type)
    file_paths: Optional[List[str]] = Field(
        None,
        description="For 'files': List of server file paths",
        example=["/data/document.pdf", "/data/report.txt"]
    )
    s3_paths: Optional[List[str]] = Field(
        None,
        description="For 's3': List of S3 paths (s3://bucket/key or bucket/key)",
        example=["s3://my-bucket/documents/report.pdf"]
    )
    db_path: Optional[str] = Field(
        None,
        description="For 'sql': SQLite database path",
        example="/data/company.db"
    )
    text: Optional[str] = Field(
        None,
        description="For 'text': Raw text content",
        example="This is sample text content to train on..."
    )
    sql_query: Optional[str] = Field(
        None,
        description="For 'sql': Optional SQL query to execute"
    )
    
    # Collection configuration
    collection_name: str = Field(
        "default",
        description="ChromaDB collection name",
        example="knowledge_base"
    )
    
    # Processing configuration
    chunk_size: int = Field(1000, ge=1, le=10000, description="Chunk size for splitting")
    chunk_overlap: int = Field(200, ge=0, le=1000, description="Chunk overlap")
    splitter_type: str = Field(
        "recursive",
        description="Text splitter type: 'recursive', 'character', 'token', 'markdown', 'python'"
    )
    
    # Embedding configuration
    embedding_model_name: str = Field(
        "all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    embedding_provider: Optional[str] = Field(
        None,
        description="Embedding provider: 'huggingface' or 'openai'"
    )
    
    # File-specific options
    use_pymupdf: bool = Field(True, description="Use PyMuPDF for PDF processing")
    password: Optional[str] = Field(None, description="Password for encrypted PDFs")
    max_pages: Optional[int] = Field(None, description="Maximum pages to process for large PDFs")
    word_loader_type: str = Field("python-docx", description="Word loader type")
    csv_row_based: bool = Field(False, description="For CSV files, create one document per row")
    csv_intelligent_formatting: bool = Field(True, description="Use intelligent formatting for CSV")
    json_loader_type: str = Field("intelligent", description="JSON loader type")
    
    # SQL-specific options
    sql_loader_type: str = Field("intelligent", description="SQL loader type")
    include_sample_rows: int = Field(5, description="Number of sample rows per table")
    include_relationships: bool = Field(True, description="Include relationship documents")
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata to add to all documents")
    
    @model_validator(mode='after')
    def validate_input_data(self):
        """Validate that appropriate input data is provided based on input_type"""
        if self.input_type == "files" and not self.file_paths:
            raise ValueError("file_paths is required when input_type is 'files'")
        elif self.input_type == "s3" and not self.s3_paths:
            raise ValueError("s3_paths is required when input_type is 's3'")
        elif self.input_type == "sql" and not self.db_path:
            raise ValueError("db_path is required when input_type is 'sql'")
        elif self.input_type == "text" and not self.text:
            raise ValueError("text is required when input_type is 'text'")
        return self


class UnifiedTrainResponse(BaseModel):
    """Unified training response"""
    message: str = Field(..., example="Successfully trained RAG system")
    collection_name: str = Field(..., example="knowledge_base")
    input_type: str = Field(..., example="files")
    documents_ingested: int = Field(..., example=10)
    chunks_created: int = Field(..., example=45)
    chunks_stored: int = Field(..., example=45)
    stored_files: List[Dict[str, Any]] = Field(..., description="File storage information")
    collection_info: Dict[str, Any] = Field(..., description="Collection information after training")


@router.post(
    "/train",
    response_model=UnifiedTrainResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Unified RAG training endpoint",
    description="""
    Unified endpoint for training RAG system with multiple input types.
    
    **Supported input types:**
    - `files`: Files from server filesystem
    - `s3`: Files from S3 bucket
    - `sql`: SQLite database
    - `text`: Raw text content
    
    **Complete Process:**
    1. Routes to appropriate processor based on input_type
    2. Ingests and processes data
    3. Stores raw files (S3 if configured, else local filesystem)
    4. Splits into chunks
    5. Generates embeddings
    6. Stores vectors in ChromaDB
    
    **File Storage:**
    - If AWS S3 is configured (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET), files are stored in S3
    - Otherwise, files are stored locally in ./storage/{collection_name}/
    
    **Returns:**
    - Processing statistics
    - File storage information (S3 paths or local paths)
    - Collection information
    """,
    response_description="Training result with statistics and storage information"
)
async def unified_train(request: UnifiedTrainRequest):
    """
    Unified training endpoint that handles multiple input types.
    """
    logger.info(
        f"Unified training request | "
        f"Input type: {request.input_type} | "
        f"Collection: {request.collection_name}"
    )
    
    try:
        processor = TrainingProcessor()
        
        # Prepare input data based on input_type
        input_data = _prepare_input_data(request)
        
        # Prepare processing config
        processing_config = {
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
            "splitter_type": request.splitter_type,
            "use_pymupdf": request.use_pymupdf,
            "password": request.password,
            "max_pages": request.max_pages,
            "loader_config": {
                "word_loader_type": request.word_loader_type,
                "csv_row_based": request.csv_row_based,
                "csv_intelligent_formatting": request.csv_intelligent_formatting,
                "json_loader_type": request.json_loader_type
            },
            "sql_loader_config": {
                "loader_type": request.sql_loader_type,
                "include_sample_rows": request.include_sample_rows,
                "include_relationships": request.include_relationships
            },
            "metadata": request.metadata
        }
        
        # Process training request (ingest, split, store files)
        result = await processor.process_training_request(
            input_type=request.input_type,
            input_data=input_data,
            collection_name=request.collection_name,
            processing_config=processing_config
        )
        
        # Store vectors in ChromaDB
        embedding_config = {
            "model_name": request.embedding_model_name,
            "provider": request.embedding_provider
        }
        
        chunk_ids = await processor.store_vectors(
            chunks=result["chunks"],
            collection_name=request.collection_name,
            embedding_config=embedding_config
        )
        
        # Get collection info
        vector_store = get_vector_store_manager(
            collection_name=request.collection_name,
            embedding_model_name=request.embedding_model_name,
            embedding_provider=request.embedding_provider
        )
        collection_info = vector_store.get_collection_info()
        
        logger.info(
            f"Training completed successfully | "
            f"Input type: {request.input_type} | "
            f"Documents: {len(result['documents'])} | "
            f"Chunks: {len(result['chunks'])} | "
            f"Stored files: {len(result['stored_files'])}"
        )
        
        return UnifiedTrainResponse(
            message=f"Successfully trained RAG system with {request.input_type} input",
            collection_name=request.collection_name,
            input_type=request.input_type,
            documents_ingested=len(result["documents"]),
            chunks_created=len(result["chunks"]),
            chunks_stored=len(chunk_ids),
            stored_files=result["stored_files"],
            collection_info=collection_info
        )
    except ValueError as e:
        logger.warning(f"Invalid training request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except FileNotFoundError as e:
        logger.warning(f"File not found during training: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {str(e)}"
        )
    except Exception as e:
        logger.error(
            f"Error during unified training | "
            f"Input type: {request.input_type} | "
            f"Error: {type(e).__name__} | "
            f"Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during RAG training: {str(e)}"
        )


@router.post(
    "/train/upload",
    response_model=UnifiedTrainResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Train RAG system with uploaded file",
    description="""
    Upload a file and train the RAG system.
    This is a convenience endpoint that uses the unified training system.
    
    **Process:**
    1. Upload file
    2. Store raw file (S3/local)
    3. Ingest and process
    4. Split into chunks
    5. Generate embeddings
    6. Store vectors in ChromaDB
    """,
    response_description="Training result with statistics"
)
async def train_upload(
    file: UploadFile = File(..., description="File to upload and train on"),
    collection_name: str = Form("default"),
    splitter_type: str = Form("recursive"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    embedding_model_name: str = Form("all-MiniLM-L6-v2"),
    embedding_provider: Optional[str] = Form(None),
    use_pymupdf: bool = Form(True),
    password: Optional[str] = Form(None),
    max_pages: Optional[int] = Form(None),
    word_loader_type: str = Form("python-docx"),
    csv_row_based: bool = Form(False),
    csv_intelligent_formatting: bool = Form(True),
    json_loader_type: str = Form("intelligent"),
    metadata: Optional[str] = Form(None, description="JSON string for metadata")
):
    """
    Upload endpoint - converts to unified format and processes
    """
    logger.info(
        f"Upload training request | "
        f"File: {file.filename} | "
        f"Collection: {collection_name}"
    )
    
    try:
        # Read file content
        content = await file.read()
        
        # Parse metadata if provided
        import json
        parsed_metadata = json.loads(metadata) if metadata else None
        
        # Create unified request
        request = UnifiedTrainRequest(
            input_type="upload",
            file_paths=None,
            s3_paths=None,
            db_path=None,
            text=None,
            collection_name=collection_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            splitter_type=splitter_type,
            embedding_model_name=embedding_model_name,
            embedding_provider=embedding_provider,
            use_pymupdf=use_pymupdf,
            password=password,
            max_pages=max_pages,
            word_loader_type=word_loader_type,
            csv_row_based=csv_row_based,
            csv_intelligent_formatting=csv_intelligent_formatting,
            json_loader_type=json_loader_type,
            metadata=parsed_metadata
        )
        
        # Create input data with file content
        input_data = {
            "content": content,
            "filename": file.filename or "unknown"
        }
        
        # Process using unified processor
        processor = TrainingProcessor()
        
        processing_config = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "splitter_type": splitter_type,
            "use_pymupdf": use_pymupdf,
            "password": password,
            "max_pages": max_pages,
            "loader_config": {
                "word_loader_type": word_loader_type,
                "csv_row_based": csv_row_based,
                "csv_intelligent_formatting": csv_intelligent_formatting,
                "json_loader_type": json_loader_type
            },
            "metadata": parsed_metadata
        }
        
        # Process training request
        result = await processor.process_training_request(
            input_type="upload",
            input_data=input_data,
            collection_name=collection_name,
            processing_config=processing_config
        )
        
        # Store vectors
        embedding_config = {
            "model_name": embedding_model_name,
            "provider": embedding_provider
        }
        
        chunk_ids = await processor.store_vectors(
            chunks=result["chunks"],
            collection_name=collection_name,
            embedding_config=embedding_config
        )
        
        # Get collection info
        vector_store = get_vector_store_manager(
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
            embedding_provider=embedding_provider
        )
        collection_info = vector_store.get_collection_info()
        
        return UnifiedTrainResponse(
            message=f"Successfully trained RAG system with uploaded file: {file.filename}",
            collection_name=collection_name,
            input_type="upload",
            documents_ingested=len(result["documents"]),
            chunks_created=len(result["chunks"]),
            chunks_stored=len(chunk_ids),
            stored_files=result["stored_files"],
            collection_info=collection_info
        )
    except Exception as e:
        logger.error(
            f"Error during upload training | Error: {type(e).__name__} | Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during RAG training: {str(e)}"
        )


class QueryRequest(BaseModel):
    """Request model for querying the vector store"""
    query: str = Field(..., description="Search query", example="What is Langchain?")
    collection_name: str = Field("default", description="Name of the ChromaDB collection")
    k: int = Field(4, ge=1, le=100, description="Number of results to return")
    embedding_model_name: str = Field("all-MiniLM-L6-v2", description="Embedding model name (must match collection)")
    embedding_provider: Optional[str] = Field(None, description="Embedding provider")


class QueryResponse(BaseModel):
    """Response model for query operations"""
    message: str = Field(..., example="Query completed successfully")
    query: str = Field(..., example="What is Langchain?")
    results: List[Dict[str, Any]] = Field(..., description="Search results with scores")
    collection_name: str = Field(..., example="default")


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the trained RAG system",
    description="""
    Query the vector store to retrieve similar documents.
    
    **Use cases:**
    - Semantic search
    - Document retrieval
    - Finding relevant context for RAG
    
    **Returns:**
    - List of similar documents with similarity scores
    - Ranked by relevance
    """,
    response_description="Query results with similarity scores"
)
async def query_vector_store(request: QueryRequest):
    logger.info(
        f"Vector store query | "
        f"Query: {request.query[:50]}... | "
        f"Collection: {request.collection_name} | "
        f"K: {request.k}"
    )
    
    try:
        # Get vector store
        vector_store = get_vector_store_manager(
            collection_name=request.collection_name,
            embedding_model_name=request.embedding_model_name,
            embedding_provider=request.embedding_provider
        )
        
        # Perform similarity search with scores
        results = vector_store.similarity_search_with_score(
            query=request.query,
            k=request.k
        )
        
        # Format results
        formatted_results = [
            {
                "rank": idx + 1,
                "score": float(score),
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata
            }
            for idx, (doc, score) in enumerate(results)
        ]
        
        logger.info(f"Query completed | Results: {len(formatted_results)}")
        
        return QueryResponse(
            message=f"Query completed successfully. Found {len(formatted_results)} result(s)",
            query=request.query,
            results=formatted_results,
            collection_name=request.collection_name
        )
    except Exception as e:
        logger.error(
            f"Error querying vector store | Error: {type(e).__name__} | Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error querying vector store: {str(e)}"
        )


@router.get(
    "/collections/{collection_name}",
    status_code=status.HTTP_200_OK,
    summary="Get collection information",
    description="""
    Get information about a specific collection.
    
    **Returns:**
    - Collection name
    - Document count
    - Embedding model information
    - Persist directory
    """,
    response_description="Collection information"
)
async def get_collection_info(
    collection_name: str,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    embedding_provider: Optional[str] = None
):
    try:
        vector_store = get_vector_store_manager(
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
            embedding_provider=embedding_provider
        )
        
        collection_info = vector_store.get_collection_info()
        
        return {
            "message": f"Collection information retrieved successfully",
            "collection_info": collection_info
        }
    except Exception as e:
        logger.error(f"Error getting collection info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting collection info: {str(e)}"
        )


@router.delete(
    "/collections/{collection_name}",
    status_code=status.HTTP_200_OK,
    summary="Delete a collection",
    description="""
    Delete a collection from the vector store.
    
    **Warning:** This action cannot be undone!
    """,
    response_description="Deletion confirmation"
)
async def delete_collection(
    collection_name: str,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    embedding_provider: Optional[str] = None
):
    try:
        vector_store = get_vector_store_manager(
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
            embedding_provider=embedding_provider
        )
        
        vector_store.delete_collection()
        
        # Remove from cache
        from app.rag.vector_store import _vector_store_managers
        cache_key = f"{collection_name}_{vector_store.persist_directory}_{embedding_model_name}_{embedding_provider}"
        if cache_key in _vector_store_managers:
            del _vector_store_managers[cache_key]
        
        return {
            "message": f"Collection '{collection_name}' deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting collection: {str(e)}"
        )


def _prepare_input_data(request: UnifiedTrainRequest) -> Dict[str, Any]:
    """Prepare input data based on input type"""
    if request.input_type == "files":
        return {"file_paths": request.file_paths}
    elif request.input_type == "s3":
        return {"s3_paths": request.s3_paths}
    elif request.input_type == "sql":
        return {"db_path": request.db_path, "query": request.sql_query}
    elif request.input_type == "text":
        return {"text": request.text}
    elif request.input_type == "upload":
        # This is handled separately in train_upload endpoint
        raise ValueError("Upload input type should use /train/upload endpoint")
    else:
        raise ValueError(f"Invalid input_type: {request.input_type}")
