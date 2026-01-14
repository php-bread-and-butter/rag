"""
RAG Training API Endpoints

Endpoints for training the RAG system by ingesting, splitting, embedding, and storing documents.
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, status, Form
from pydantic import BaseModel, Field

from app.rag.unified_loader import UnifiedDocumentLoader
from app.rag.text_splitters import TextSplitterManager
from app.rag.vector_store import VectorStoreManager, get_vector_store_manager
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


class TrainDocumentsRequest(BaseModel):
    """Request model for training with file paths"""
    file_paths: List[str] = Field(..., description="List of file paths to train on", example=["/path/to/file1.pdf", "/path/to/file2.txt"])
    collection_name: str = Field("default", description="Name of the ChromaDB collection")
    # Document ingestion parameters
    use_pymupdf: bool = Field(True, description="Use PyMuPDF for PDF processing")
    password: Optional[str] = Field(None, description="Password for encrypted PDFs")
    max_pages: Optional[int] = Field(None, description="Maximum pages to process for large PDFs")
    word_loader_type: str = Field("python-docx", description="Word loader type")
    csv_row_based: bool = Field(False, description="For CSV files, create one document per row")
    csv_intelligent_formatting: bool = Field(True, description="Use intelligent formatting for CSV")
    json_loader_type: str = Field("intelligent", description="JSON loader type")
    # Text splitting parameters
    splitter_type: str = Field("recursive", description="Text splitter type")
    chunk_size: int = Field(1000, ge=1, le=10000, description="Chunk size for splitting")
    chunk_overlap: int = Field(200, ge=0, le=1000, description="Chunk overlap")
    # Embedding parameters
    embedding_model_name: str = Field("all-MiniLM-L6-v2", description="Embedding model name")
    embedding_provider: Optional[str] = Field(None, description="Embedding provider: 'huggingface' or 'openai'")
    # Metadata
    metadata: Optional[dict] = Field(None, description="Optional metadata to add to all documents")


class TrainTextRequest(BaseModel):
    """Request model for training with raw text"""
    text: str = Field(..., description="Raw text content to train on")
    collection_name: str = Field("default", description="Name of the ChromaDB collection")
    # Text splitting parameters
    splitter_type: str = Field("recursive", description="Text splitter type")
    chunk_size: int = Field(1000, ge=1, le=10000, description="Chunk size for splitting")
    chunk_overlap: int = Field(200, ge=0, le=1000, description="Chunk overlap")
    # Embedding parameters
    embedding_model_name: str = Field("all-MiniLM-L6-v2", description="Embedding model name")
    embedding_provider: Optional[str] = Field(None, description="Embedding provider")
    # Metadata
    metadata: Optional[dict] = Field(None, description="Optional metadata to add to all documents")


class TrainSQLRequest(BaseModel):
    """Request model for training with SQL database"""
    db_path: str = Field(..., description="Path to SQLite database file")
    collection_name: str = Field("default", description="Name of the ChromaDB collection")
    sql_loader_type: str = Field("intelligent", description="SQL loader type")
    sql_query: Optional[str] = Field(None, description="Optional SQL query")
    include_sample_rows: int = Field(5, description="Number of sample rows per table")
    include_relationships: bool = Field(True, description="Include relationship documents")
    # Text splitting parameters
    splitter_type: str = Field("recursive", description="Text splitter type")
    chunk_size: int = Field(1000, ge=1, le=10000, description="Chunk size for splitting")
    chunk_overlap: int = Field(200, ge=0, le=1000, description="Chunk overlap")
    # Embedding parameters
    embedding_model_name: str = Field("all-MiniLM-L6-v2", description="Embedding model name")
    embedding_provider: Optional[str] = Field(None, description="Embedding provider")
    # Metadata
    metadata: Optional[dict] = Field(None, description="Optional metadata to add to all documents")


class TrainResponse(BaseModel):
    """Response model for training operations"""
    message: str = Field(..., example="Training completed successfully")
    collection_name: str = Field(..., example="default")
    documents_ingested: int = Field(..., example=10)
    chunks_created: int = Field(..., example=45)
    chunks_stored: int = Field(..., example=45)
    collection_info: Dict[str, Any] = Field(..., description="Collection information after training")


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
    results: List[dict] = Field(..., description="Search results with scores")
    collection_name: str = Field(..., example="default")


@router.post(
    "/train/files",
    response_model=TrainResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Train RAG system with files",
    description="""
    Train the RAG system by ingesting, splitting, embedding, and storing documents from file paths.
    
    **Process:**
    1. Ingest documents from file paths (supports TXT, PDF, Word, CSV, Excel, JSON)
    2. Split documents into chunks
    3. Generate embeddings for chunks
    4. Store in ChromaDB vector store
    
    **Supported file types:**
    - `.txt`, `.text` - Plain text files
    - `.pdf` - PDF documents
    - `.docx` - Microsoft Word documents
    - `.csv` - CSV files
    - `.xlsx`, `.xls` - Excel files
    - `.json`, `.jsonl` - JSON files
    
    **Returns:**
    - Number of documents ingested
    - Number of chunks created
    - Number of chunks stored
    - Collection information
    """,
    response_description="Training result with statistics"
)
async def train_with_files(request: TrainDocumentsRequest):
    logger.info(
        f"RAG training request (files) | "
        f"Files: {len(request.file_paths)} | "
        f"Collection: {request.collection_name} | "
        f"Splitter: {request.splitter_type} | "
        f"Embedding: {request.embedding_model_name}"
    )
    
    try:
        # Step 1: Ingest documents
        unified_loader = UnifiedDocumentLoader(
            word_loader_type=request.word_loader_type,
            csv_row_based=request.csv_row_based,
            csv_intelligent_formatting=request.csv_intelligent_formatting,
            json_loader_type=request.json_loader_type
        )
        
        documents = unified_loader.load_files(
            request.file_paths,
            request.metadata,
            request.use_pymupdf,
            request.password,
            request.max_pages
        )
        
        logger.info(f"Ingested {len(documents)} documents")
        
        # Step 2: Split documents
        splitter_manager = TextSplitterManager(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            splitter_type=request.splitter_type
        )
        
        chunks = splitter_manager.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Step 3: Store in vector store
        vector_store = get_vector_store_manager(
            collection_name=request.collection_name,
            embedding_model_name=request.embedding_model_name,
            embedding_provider=request.embedding_provider
        )
        
        chunk_ids = vector_store.add_documents(chunks)
        logger.info(f"Stored {len(chunk_ids)} chunks in vector store")
        
        # Get collection info
        collection_info = vector_store.get_collection_info()
        
        return TrainResponse(
            message=f"Successfully trained RAG system with {len(request.file_paths)} file(s)",
            collection_name=request.collection_name,
            documents_ingested=len(documents),
            chunks_created=len(chunks),
            chunks_stored=len(chunk_ids),
            collection_info=collection_info
        )
    except FileNotFoundError as e:
        logger.warning(f"File not found during training: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found: {str(e)}"
        )
    except Exception as e:
        logger.error(
            f"Error during RAG training | Error: {type(e).__name__} | Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during RAG training: {str(e)}"
        )


@router.post(
    "/train/upload",
    response_model=TrainResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Train RAG system with uploaded file",
    description="""
    Train the RAG system by uploading a file, then ingesting, splitting, embedding, and storing it.
    
    **Process:**
    1. Upload and ingest document
    2. Split into chunks
    3. Generate embeddings
    4. Store in ChromaDB vector store
    
    **Supported file types:** Same as `/train/files`
    """,
    response_description="Training result with statistics"
)
async def train_with_upload(
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
    word_loader_type: str = Form("python-docx")
):
    logger.info(
        f"RAG training request (upload) | "
        f"File: {file.filename} | "
        f"Collection: {collection_name}"
    )
    
    try:
        # Step 1: Read and ingest file
        content = await file.read()
        unified_loader = UnifiedDocumentLoader()
        
        documents = unified_loader.load_bytes(
            filename=file.filename or "unknown",
            content=content,
            metadata={"source": file.filename, "uploaded": True},
            use_pymupdf=use_pymupdf,
            password=password,
            max_pages=max_pages
        )
        
        logger.info(f"Ingested {len(documents)} documents from uploaded file")
        
        # Step 2: Split documents
        splitter_manager = TextSplitterManager(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            splitter_type=splitter_type
        )
        
        chunks = splitter_manager.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 3: Store in vector store
        vector_store = get_vector_store_manager(
            collection_name=collection_name,
            embedding_model_name=embedding_model_name,
            embedding_provider=embedding_provider
        )
        
        chunk_ids = vector_store.add_documents(chunks)
        logger.info(f"Stored {len(chunk_ids)} chunks in vector store")
        
        # Get collection info
        collection_info = vector_store.get_collection_info()
        
        return TrainResponse(
            message=f"Successfully trained RAG system with uploaded file: {file.filename}",
            collection_name=collection_name,
            documents_ingested=len(documents),
            chunks_created=len(chunks),
            chunks_stored=len(chunk_ids),
            collection_info=collection_info
        )
    except Exception as e:
        logger.error(
            f"Error during RAG training (upload) | Error: {type(e).__name__} | Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during RAG training: {str(e)}"
        )


@router.post(
    "/train/text",
    response_model=TrainResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Train RAG system with raw text",
    description="""
    Train the RAG system with raw text content.
    
    **Process:**
    1. Split text into chunks
    2. Generate embeddings
    3. Store in ChromaDB vector store
    """,
    response_description="Training result with statistics"
)
async def train_with_text(request: TrainTextRequest):
    logger.info(
        f"RAG training request (text) | "
        f"Text length: {len(request.text)} | "
        f"Collection: {request.collection_name}"
    )
    
    try:
        from langchain_core.documents import Document
        
        # Step 1: Create document from text
        document = Document(
            page_content=request.text,
            metadata=request.metadata or {}
        )
        
        # Step 2: Split document
        splitter_manager = TextSplitterManager(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            splitter_type=request.splitter_type
        )
        
        chunks = splitter_manager.split_documents([document])
        logger.info(f"Created {len(chunks)} chunks from text")
        
        # Step 3: Store in vector store
        vector_store = get_vector_store_manager(
            collection_name=request.collection_name,
            embedding_model_name=request.embedding_model_name,
            embedding_provider=request.embedding_provider
        )
        
        chunk_ids = vector_store.add_documents(chunks)
        logger.info(f"Stored {len(chunk_ids)} chunks in vector store")
        
        # Get collection info
        collection_info = vector_store.get_collection_info()
        
        return TrainResponse(
            message="Successfully trained RAG system with raw text",
            collection_name=request.collection_name,
            documents_ingested=1,
            chunks_created=len(chunks),
            chunks_stored=len(chunk_ids),
            collection_info=collection_info
        )
    except Exception as e:
        logger.error(
            f"Error during RAG training (text) | Error: {type(e).__name__} | Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during RAG training: {str(e)}"
        )


@router.post(
    "/train/sql",
    response_model=TrainResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Train RAG system with SQL database",
    description="""
    Train the RAG system with data from a SQLite database.
    
    **Process:**
    1. Extract data from SQL database
    2. Split into chunks
    3. Generate embeddings
    4. Store in ChromaDB vector store
    """,
    response_description="Training result with statistics"
)
async def train_with_sql(request: TrainSQLRequest):
    logger.info(
        f"RAG training request (SQL) | "
        f"Database: {request.db_path} | "
        f"Collection: {request.collection_name}"
    )
    
    try:
        from app.rag.sql_loaders import SQLDocumentLoader
        
        # Step 1: Load SQL database
        sql_loader = SQLDocumentLoader(
            loader_type=request.sql_loader_type,
            include_sample_rows=request.include_sample_rows,
            include_relationships=request.include_relationships
        )
        
        documents = sql_loader.load_database(
            request.db_path,
            request.sql_query,
            request.metadata
        )
        
        logger.info(f"Ingested {len(documents)} documents from SQL database")
        
        # Step 2: Split documents
        splitter_manager = TextSplitterManager(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            splitter_type=request.splitter_type
        )
        
        chunks = splitter_manager.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 3: Store in vector store
        vector_store = get_vector_store_manager(
            collection_name=request.collection_name,
            embedding_model_name=request.embedding_model_name,
            embedding_provider=request.embedding_provider
        )
        
        chunk_ids = vector_store.add_documents(chunks)
        logger.info(f"Stored {len(chunk_ids)} chunks in vector store")
        
        # Get collection info
        collection_info = vector_store.get_collection_info()
        
        return TrainResponse(
            message=f"Successfully trained RAG system with SQL database: {request.db_path}",
            collection_name=request.collection_name,
            documents_ingested=len(documents),
            chunks_created=len(chunks),
            chunks_stored=len(chunk_ids),
            collection_info=collection_info
        )
    except FileNotFoundError as e:
        logger.warning(f"SQL database not found: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"SQL database not found: {str(e)}"
        )
    except Exception as e:
        logger.error(
            f"Error during RAG training (SQL) | Error: {type(e).__name__} | Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during RAG training: {str(e)}"
        )


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
