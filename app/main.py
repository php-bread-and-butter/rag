"""
Main FastAPI application entry point
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import traceback

from app.api.v1.api import api_router
from app.core.config import settings
from app.core.logging_config import setup_logging, get_logger

# Setup logging
setup_logging(log_level=settings.LOG_LEVEL, log_file=settings.LOG_FILE)
logger = get_logger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
    ## FastAPI RAG Tutorial Application
    
    A step-by-step tutorial project for building a RAG (Retrieval-Augmented Generation) system.
    
    ### Features
    
    * **Document Ingestion** - Upload and process various file types (TXT, PDF)
    * **Text Splitting** - Split documents into chunks using multiple strategies
    * **Automatic File Type Detection** - System automatically detects and processes file types
    
    ### API Endpoints
    
    * `/api/v1/documents/*` - Document ingestion endpoints
    * `/api/v1/splitting/*` - Text splitting endpoints
    
    ### Interactive Documentation
    
    * **Swagger UI**: Available at `/docs` (this page)
    * **ReDoc**: Available at `/redoc`
    """,
    openapi_url="/api/v1/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "FastAPI RAG Tutorial",
    },
    license_info={
        "name": "MIT",
    },
    tags_metadata=[
        {
            "name": "Document Ingestion",
            "description": "Upload and ingest documents. Supports automatic file type detection for TXT, PDF, Word, CSV, Excel, JSON, and SQL files.",
        },
        {
            "name": "Text Splitting",
            "description": "Split text and documents into chunks using various splitting techniques.",
        },
        {
            "name": "Embeddings",
            "description": "Generate text embeddings using HuggingFace and OpenAI models. Supports single queries, batch documents, and similarity calculations.",
        },
        {
            "name": "RAG Training (Legacy)",
            "description": "Legacy RAG training endpoints. Use RAG Training V2 for new implementations.",
        },
        {
            "name": "RAG Training V2",
            "description": "Unified RAG training system with S3/local file storage and ChromaDB vector storage. Supports multiple input types: files, S3, SQL, text. Stores raw files separately from vectors.",
        },
    ],
)

# Include API router
app.include_router(api_router, prefix="/api/v1")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all requests"""
    import time
    
    start_time = time.time()
    
    # Log request
    logger.info(f"→ {request.method} {request.url.path} | Client: {request.client.host if request.client else 'unknown'}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"← {request.method} {request.url.path} | "
            f"Status: {response.status_code} | "
            f"Time: {process_time:.3f}s"
        )
        
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"✗ {request.method} {request.url.path} | "
            f"Error: {str(e)} | "
            f"Time: {process_time:.3f}s"
        )
        raise


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with detailed logging"""
    logger.error(
        f"Unhandled exception: {type(exc).__name__}",
        exc_info=True,
        extra={
            "path": request.url.path,
            "method": request.method,
            "client": request.client.host if request.client else "unknown"
        }
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "type": type(exc).__name__,
            "message": str(exc),
            "path": request.url.path
        }
    )


@app.get(
    "/",
    summary="Root endpoint",
    description="Welcome endpoint with API information and links to documentation",
    tags=["General"]
)
async def hello_world():
    """Hello World endpoint"""
    return {
        "message": "Welcome to FastAPI RAG Tutorial",
        "version": settings.VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "api_base": "/api/v1",
        "endpoints": {
            "documents": "/api/v1/documents",
            "splitting": "/api/v1/splitting"
        }
    }


@app.get(
    "/health",
    summary="Health check",
    description="Check if the API service is running and healthy",
    tags=["General"]
)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "FastAPI Tutorial",
        "version": settings.VERSION
    }
