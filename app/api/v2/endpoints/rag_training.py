"""
RAG Training API Endpoints V2

Unified endpoint for training RAG system with multiple input types.
Stores raw files (S3/local) and vectors (ChromaDB).

This is the V2 API - use /api/v2/rag/* endpoints.
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


class RAGQueryRequest(BaseModel):
    """Request model for RAG query with LLM answer generation"""
    query: str = Field(..., description="Question to ask", example="What is machine learning?")
    collection_name: str = Field("default", description="Name of the ChromaDB collection")
    k: int = Field(3, ge=1, le=100, description="Number of documents to retrieve")
    embedding_model_name: str = Field("all-MiniLM-L6-v2", description="Embedding model name (must match collection)")
    embedding_provider: Optional[str] = Field(None, description="Embedding provider")
    llm_provider: str = Field("openai", description="LLM provider: 'openai' or 'groq'")
    llm_model_name: Optional[str] = Field(None, description="LLM model name (defaults to provider's default)")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="LLM temperature (0.0 for deterministic)")
    chain_type: str = Field("retrieval", description="Chain type: 'retrieval', 'lcel', or 'conversational'")
    chat_history: Optional[List[Dict[str, str]]] = Field(None, description="Chat history for conversational chains [{'role': 'user'/'assistant', 'content': '...'}]")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key (uses env var if not provided)")
    groq_api_key: Optional[str] = Field(None, description="GROQ API key (uses env var if not provided)")


class RAGQueryResponse(BaseModel):
    """Response model for RAG query with LLM answer"""
    message: str = Field(..., example="RAG query completed successfully")
    query: str = Field(..., example="What is machine learning?")
    answer: str = Field(..., description="LLM-generated answer")
    context: List[Dict[str, Any]] = Field(
        ..., 
        description="Retrieved context documents with source information. Each item includes: rank, content, metadata (full), and source (summary with file reference)"
    )
    collection_name: str = Field(..., example="default")
    llm_info: Dict[str, Any] = Field(..., description="LLM model information")
    sources: List[Dict[str, Any]] = Field(
        ..., 
        description="Unique source files referenced in the answer. Includes file references, storage paths, and metadata"
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query the trained RAG system (similarity search only)",
    description="""
    Query the vector store to retrieve similar documents (without LLM answer generation).
    
    **Use cases:**
    - Semantic search
    - Document retrieval
    - Finding relevant context for RAG
    
    **Returns:**
    - List of similar documents with similarity scores
    - Ranked by relevance
    
    **Note:** For LLM-generated answers, use `/query/rag` endpoint instead.
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
        
        # Format results with source summaries
        from app.rag.metadata_manager import MetadataManager
        
        formatted_results = []
        for idx, (doc, score) in enumerate(results):
            source_summary = MetadataManager.create_source_summary(doc.metadata)
            formatted_results.append({
                "rank": idx + 1,
                "score": float(score),
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata,
                "source": source_summary
            })
        
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


@router.post(
    "/query/rag",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query RAG system with LLM answer generation",
    description="""
    Query the RAG system and get an LLM-generated answer based on retrieved context.
    
    **Complete RAG Pipeline:**
    1. Retrieves relevant documents from vector store
    2. Uses LLM to generate answer based on retrieved context
    3. Returns answer with source documents
    
    **Supported Chain Types:**
    - `retrieval`: Standard retrieval chain (create_retrieval_chain)
    - `lcel`: LangChain Expression Language chain (more flexible)
    - `conversational`: Conversational chain with history-aware retriever
    
    **LLM Providers:**
    - `openai`: OpenAI models (gpt-3.5-turbo, gpt-4, etc.)
    - `groq`: GROQ models (gemma2-9b-it, llama-3.1-8b-instant, etc.)
    
    **Conversational Memory:**
    - For `conversational` chain type, provide `chat_history` to maintain context
    - Format: `[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`
    
    **Example:**
    ```json
    {
        "query": "What are the types of machine learning?",
        "collection_name": "knowledge_base",
        "llm_provider": "openai",
        "llm_model_name": "gpt-3.5-turbo",
        "chain_type": "retrieval",
        "k": 3
    }
    ```
    
    **Returns:**
    - LLM-generated answer
    - Retrieved context documents
    - LLM model information
    """,
    response_description="RAG query result with LLM answer and context"
)
async def query_rag(request: RAGQueryRequest):
    """
    Query RAG system with LLM answer generation.
    """
    logger.info(
        f"RAG query request | "
        f"Query: {request.query[:50]}... | "
        f"Collection: {request.collection_name} | "
        f"LLM Provider: {request.llm_provider} | "
        f"Chain Type: {request.chain_type}"
    )
    
    try:
        from app.rag.rag_chain import create_rag_chain
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Get vector store
        vector_store = get_vector_store_manager(
            collection_name=request.collection_name,
            embedding_model_name=request.embedding_model_name,
            embedding_provider=request.embedding_provider
        )
        
        # Convert chat history if provided
        chat_history = None
        if request.chat_history:
            chat_history = []
            for msg in request.chat_history:
                role = msg.get("role", "").lower()
                content = msg.get("content", "")
                if role == "user":
                    chat_history.append(HumanMessage(content=content))
                elif role == "assistant":
                    chat_history.append(AIMessage(content=content))
        
        # Create RAG chain
        rag_chain_manager = create_rag_chain(
            vector_store=vector_store,
            llm_provider=request.llm_provider,
            llm_model_name=request.llm_model_name,
            temperature=request.temperature,
            k=request.k,
            chain_type=request.chain_type,
            openai_api_key=request.openai_api_key,
            groq_api_key=request.groq_api_key
        )
        
        # Invoke RAG chain
        result = rag_chain_manager.invoke(
            query=request.query,
            chat_history=chat_history
        )
        
        # Extract answer and context
        answer = result.get("answer", "")
        context_docs = result.get("context", [])
        
        # Format context documents with source summaries
        from app.rag.metadata_manager import MetadataManager
        
        formatted_context = []
        unique_sources = {}  # Track unique source files
        
        for idx, doc in enumerate(context_docs):
            source_summary = MetadataManager.create_source_summary(doc.metadata)
            formatted_context.append({
                "rank": idx + 1,
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata,
                "source": source_summary
            })
            
            # Track unique sources by file_id or storage_path
            file_id = doc.metadata.get("file_id")
            storage_path = doc.metadata.get("storage_path")
            source_key = file_id or storage_path or doc.metadata.get("original_filename", "unknown")
            
            if source_key not in unique_sources:
                unique_sources[source_key] = {
                    "file_id": file_id,
                    "file_name": doc.metadata.get("original_filename", "Unknown"),
                    "file_type": doc.metadata.get("file_type", "unknown"),
                    "storage_path": storage_path,
                    "storage_type": doc.metadata.get("storage_type"),
                    "source_path": doc.metadata.get("source_path"),
                    "ingested_at": doc.metadata.get("ingested_at"),
                    "collection_name": doc.metadata.get("collection_name"),
                    "input_type": doc.metadata.get("input_type")
                }
        
        # Convert unique sources to list
        sources_list = list(unique_sources.values())
        
        # Get LLM info
        from app.rag.llm_manager import LLMManager
        llm_manager = LLMManager(
            provider=request.llm_provider,
            model_name=request.llm_model_name,
            temperature=request.temperature,
            openai_api_key=request.openai_api_key,
            groq_api_key=request.groq_api_key
        )
        llm_info = llm_manager.get_model_info()
        
        logger.info(
            f"RAG query completed successfully | "
            f"Answer length: {len(answer)} | "
            f"Context docs: {len(formatted_context)} | "
            f"Unique sources: {len(sources_list)}"
        )
        
        return RAGQueryResponse(
            message="RAG query completed successfully",
            query=request.query,
            answer=answer,
            context=formatted_context,
            collection_name=request.collection_name,
            llm_info=llm_info,
            sources=sources_list
        )
    except ValueError as e:
        logger.warning(f"Invalid RAG query request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Error during RAG query | Error: {type(e).__name__} | Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during RAG query: {str(e)}"
        )


@router.get(
    "/llm/models",
    status_code=status.HTTP_200_OK,
    summary="Get available LLM models",
    description="""
    Get a list of all available LLM models by provider.
    
    **Returns:**
    - Available OpenAI models
    - Available GROQ models
    - Default models for each provider
    """,
    response_description="Available LLM models"
)
async def get_available_llm_models():
    """Get available LLM models"""
    try:
        from app.rag.llm_manager import get_available_models, DEFAULT_MODELS
        
        models = get_available_models()
        
        return {
            "message": "Available LLM models retrieved successfully",
            "providers": list(models.keys()),
            "default_models": DEFAULT_MODELS,
            "models": models
        }
    except Exception as e:
        logger.error(f"Error getting available LLM models: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting available LLM models: {str(e)}"
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
