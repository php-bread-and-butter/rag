"""
Unified Document Ingestion API Endpoints

Single endpoint that handles all file types automatically.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, status, Form
from pydantic import BaseModel, Field

from app.rag.unified_loader import UnifiedDocumentLoader
from app.rag.sql_loaders import SQLDocumentLoader
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Initialize unified loader (default configuration)
# Can be customized per request for Word loader type
unified_loader = UnifiedDocumentLoader()


class DocumentIngestResponse(BaseModel):
    """Response model for document ingestion"""
    message: str
    file_name: str
    file_type: str
    documents_count: int
    total_chars: int
    pages: Optional[int] = None
    documents: List[dict]


class FilesIngestRequest(BaseModel):
    """Request model for ingesting files from paths"""
    file_paths: List[str] = Field(..., description="List of file paths on the server", example=["/path/to/file1.pdf", "/path/to/file2.txt"])
    use_pymupdf: bool = Field(True, description="Use PyMuPDF for PDF processing (faster, better)")
    password: Optional[str] = Field(None, description="Password for encrypted PDFs (if needed)")
    max_pages: Optional[int] = Field(None, description="Maximum pages to process for large PDFs (None for all)")
    word_loader_type: str = Field("python-docx", description="Word loader type: 'python-docx', 'docx2txt', or 'unstructured'")
    csv_row_based: bool = Field(False, description="For CSV files, create one document per row (instead of one document for entire file)")
    csv_intelligent_formatting: bool = Field(True, description="Use intelligent structured content format for CSV rows")
    json_loader_type: str = Field("intelligent", description="JSON loader type: 'intelligent' (default), 'jsonloader', or 'jsonloader_jq'")
    json_jq_schema: Optional[str] = Field(None, description="jq query schema for JSONLoader (e.g., '.employees[]' to extract each employee)")
    json_text_content: bool = Field(False, description="For JSONLoader, whether to extract text content (False for full JSON objects)")
    metadata: Optional[dict] = Field(None, description="Optional metadata to attach to all documents", example={"source": "documents", "category": "tutorial"})


class SQLIngestRequest(BaseModel):
    """Request model for ingesting SQL databases"""
    db_path: str = Field(..., description="Path to the SQLite database file", example="/path/to/company.db")
    loader_type: str = Field("intelligent", description="SQL loader type: 'intelligent' (default), 'sqldatabase', or 'sqldatabaseloader'")
    query: Optional[str] = Field(None, description="Optional SQL query to execute (for SQLDatabaseLoader)")
    include_sample_rows: int = Field(5, description="Number of sample rows to include per table (for intelligent loader)")
    include_relationships: bool = Field(True, description="Whether to create relationship documents from JOINs (for intelligent loader)")
    metadata: Optional[dict] = Field(None, description="Optional metadata to attach to all documents", example={"source": "database", "category": "company_data"})


class TextIngestRequest(BaseModel):
    """Request model for raw text ingestion"""
    text: str = Field(..., description="Raw text content to ingest", example="This is sample text to ingest into the system.")
    metadata: Optional[dict] = Field(None, description="Optional metadata for the text", example={"source": "api", "type": "user_input"})


@router.post(
    "/upload",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
    description="""
    Upload and ingest a document with automatic file type detection and issue handling.
    
    **Supported file types:**
    - `.txt`, `.text` - Plain text files
    - `.pdf` - PDF documents (with automatic issue handling)
    - `.docx` - Microsoft Word documents
    - `.csv` - CSV (Comma-Separated Values) files
    - `.xlsx`, `.xls` - Microsoft Excel files
    - `.json`, `.jsonl` - JSON and JSON Lines files
    
    **PDF Issue Handling (automatic):**
    - Text cleaning (removes excessive whitespace, fixes ligatures)
    - Empty page filtering (skips pages with <50 characters)
    - Encryption detection and handling (if password provided)
    - Scanned PDF detection (flagged in metadata)
    - Large file handling (with page limits)
    - Enhanced metadata (page numbers, char counts, etc.)
    
    **Word Document Loaders:**
    - `python-docx` (default): Direct parsing with python-docx, supports tables and metadata
    - `docx2txt`: Simple text extraction using Docx2txtLoader
    - `unstructured`: Structured element extraction using UnstructuredWordDocumentLoader (elements mode)
    
    **How it works:**
    1. Upload a file (any supported type)
    2. System automatically detects the file type
    3. Uses appropriate processor with issue handling
    4. Returns processed documents with enhanced metadata
    
    **Example:**
    - Upload a PDF: Returns one Document per page (empty pages skipped)
    - Upload a Word doc: Returns Document(s) based on loader type
    - Upload a TXT: Returns a single Document
    - Upload Excel: Returns one Document per sheet
    """,
    response_description="Document ingestion result with page count and content preview",
    responses={
        201: {
            "description": "Document ingested successfully",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Document example.pdf ingested successfully",
                        "file_name": "example.pdf",
                        "file_type": "pdf",
                        "documents_count": 5,
                        "total_chars": 12345,
                        "pages": 5,
                        "documents": [
                            {
                                "page_number": 1,
                                "content": "First 500 characters of page content...",
                                "length": 1234,
                                "metadata": {"source": "example.pdf", "page": 0}
                            }
                        ]
                    }
                }
            }
        },
        400: {"description": "Invalid file type, encoding error, or encrypted PDF without password"},
        500: {"description": "Internal server error"}
    }
)
async def upload_document(
    file: UploadFile = File(..., description="File to upload (.txt, .text, .pdf, .docx, .csv, .xlsx, .xls, .json, .jsonl)"),
    use_pymupdf: bool = Form(True, description="Use PyMuPDF for PDF processing (faster, better)"),
    password: Optional[str] = Form(None, description="Password for encrypted PDFs (if needed)"),
    max_pages: Optional[int] = Form(None, description="Maximum pages to process for large PDFs (None for all)"),
    word_loader_type: str = Form("python-docx", description="Word loader type: 'python-docx' (default), 'docx2txt', or 'unstructured'"),
    json_loader_type: str = Form("intelligent", description="JSON loader type: 'intelligent' (default), 'jsonloader', or 'jsonloader_jq'"),
    json_jq_schema: Optional[str] = Form(None, description="jq query schema for JSONLoader (e.g., '.employees[]')"),
    json_text_content: bool = Form(False, description="For JSONLoader, whether to extract text content")
):
    logger.info(
        f"Document upload request | "
        f"Filename: {file.filename} | "
        f"Content-Type: {file.content_type} | "
        f"Use PyMuPDF: {use_pymupdf}"
    )
    
    try:
        # Read file content
        content = await file.read()
        logger.debug(f"File read | Size: {len(content)} bytes")
        
        # Detect file type and load
        file_type = unified_loader.detect_file_type_from_bytes(file.filename or "unknown", content)
        
        # Create metadata
        metadata = {
            "source": file.filename,
            "file_type": file_type,
            "uploaded": True
        }
        
        # Create loader instance with custom settings if needed
        loader = unified_loader
        needs_custom_loader = (
            (file_type == 'word' and word_loader_type != "python-docx") or
            (file_type == 'json' and (json_loader_type != "intelligent" or json_jq_schema is not None))
        )
        
        if needs_custom_loader:
            from app.rag.unified_loader import UnifiedDocumentLoader
            loader = UnifiedDocumentLoader(
                word_loader_type=word_loader_type,
                json_loader_type=json_loader_type,
                json_jq_schema=json_jq_schema,
                json_text_content=json_text_content
            )
        
        # Load document using unified loader (with PDF issue handling)
        documents = loader.load_bytes(
            filename=file.filename or "unknown",
            content=content,
            metadata=metadata,
            use_pymupdf=use_pymupdf,
            password=password,
            max_pages=max_pages
        )
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        
        # Determine if PDF (has pages) or text (single document)
        is_pdf = file_type == 'pdf'
        pages_count = len(documents) if is_pdf else None
        
        logger.info(
            f"Document ingested successfully | "
            f"Filename: {file.filename} | "
            f"Type: {file_type} | "
            f"Documents: {len(documents)} | "
            f"Total chars: {total_chars}"
        )
        
        return DocumentIngestResponse(
            message=f"Document {file.filename} ingested successfully",
            file_name=file.filename or "unknown",
            file_type=file_type,
            documents_count=len(documents),
            total_chars=total_chars,
            pages=pages_count,
            documents=[
                {
                    "page_number": idx + 1 if is_pdf else None,
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "length": len(doc.page_content),
                    "metadata": doc.metadata
                }
                for idx, doc in enumerate(documents)
            ]
        )
    except ValueError as e:
        logger.warning(f"Unsupported file type | Filename: {file.filename} | Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except UnicodeDecodeError as e:
        logger.error(f"File encoding error | Filename: {file.filename} | Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File encoding error. Please ensure the file is UTF-8 encoded or a valid PDF."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error processing document | "
            f"Filename: {file.filename} | "
            f"Error: {type(e).__name__} | "
            f"Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )


@router.post(
    "/files",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest multiple files from server paths",
    description="""
    Ingest multiple files from file paths on the server.
    
    **Note:** Files must be accessible from the server's filesystem.
    
    **Supported file types:**
    - `.txt`, `.text` - Plain text files
    - `.pdf` - PDF documents
    - `.docx` - Microsoft Word documents
    - `.csv` - CSV (Comma-Separated Values) files
    - `.xlsx`, `.xls` - Microsoft Excel files
    - `.json`, `.jsonl` - JSON and JSON Lines files
    
    The system automatically detects each file's type and processes accordingly.
    
    **Note:** Excel files with multiple sheets will create one Document per sheet.
    """,
    response_description="Combined result from all ingested files"
)
async def ingest_files(request: FilesIngestRequest):
    logger.info(
        f"Files ingestion request | "
        f"Count: {len(request.file_paths)} | "
        f"Use PyMuPDF: {request.use_pymupdf}"
    )
    
    try:
        # Create loader instance with custom settings if needed
        loader = unified_loader
        needs_custom_loader = (
            request.csv_row_based or 
            not request.csv_intelligent_formatting or
            request.word_loader_type != "python-docx" or
            request.json_loader_type != "intelligent" or
            request.json_jq_schema is not None
        )
        
        if needs_custom_loader:
            from app.rag.unified_loader import UnifiedDocumentLoader
            loader = UnifiedDocumentLoader(
                word_loader_type=request.word_loader_type,
                csv_row_based=request.csv_row_based,
                csv_intelligent_formatting=request.csv_intelligent_formatting,
                json_loader_type=request.json_loader_type,
                json_jq_schema=request.json_jq_schema,
                json_text_content=request.json_text_content
            )
        
        documents = loader.load_files(
            request.file_paths,
            request.metadata,
            request.use_pymupdf,
            request.password,
            request.max_pages,
            request.csv_row_based,
            request.csv_intelligent_formatting,
            request.json_loader_type,
            request.json_jq_schema,
            request.json_text_content
        )
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        
        logger.info(
            f"Files ingested successfully | "
            f"Files: {len(request.file_paths)} | "
            f"Total documents: {len(documents)} | "
            f"Total chars: {total_chars}"
        )
        
        return DocumentIngestResponse(
            message=f"Successfully ingested {len(request.file_paths)} file(s) with {len(documents)} document(s)",
            file_name="multiple_files",
            file_type="mixed",
            documents_count=len(documents),
            total_chars=total_chars,
            documents=[
                {
                    "page_number": idx + 1,
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "length": len(doc.page_content),
                    "metadata": doc.metadata
                }
                for idx, doc in enumerate(documents)
            ]
        )
    except FileNotFoundError as e:
        logger.warning(f"File not found | Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValueError as e:
        logger.warning(f"Unsupported file type | Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Error ingesting files | "
            f"Error: {type(e).__name__} | "
            f"Message: {str(e)} | "
            f"Paths: {request.file_paths}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting files: {str(e)}"
        )


@router.post(
    "/text",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest raw text",
    description="""
    Ingest raw text data directly without a file.
    
    **Use case:** When you have text content in memory and want to process it.
    
    **Example:**
    ```json
    {
        "text": "Your text content here...",
        "metadata": {"source": "api", "type": "user_input"}
    }
    ```
    """,
    response_description="Text converted to Document object"
)
async def ingest_text(request: TextIngestRequest):
    logger.info(f"Text ingestion request | Text length: {len(request.text)} chars")
    
    try:
        from app.rag.document_loaders import load_text
        
        document = load_text(request.text, request.metadata)
        
        logger.info(
            f"Text ingested successfully | "
            f"Chars: {len(document.page_content)}"
        )
        
        return DocumentIngestResponse(
            message="Text ingested successfully",
            file_name="raw_text",
            file_type="text",
            documents_count=1,
            total_chars=len(document.page_content),
            documents=[{
                "page_number": None,
                "content": document.page_content[:500] + "..." if len(document.page_content) > 500 else document.page_content,
                "length": len(document.page_content),
                "metadata": document.metadata
            }]
        )
    except Exception as e:
        logger.error(
            f"Error ingesting text | Error: {type(e).__name__} | Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting text: {str(e)}"
        )


@router.post(
    "/sql",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest SQL database",
    description="""
    Ingest data from a SQLite database file.
    
    **Supported loader types:**
    - **intelligent** (default): Creates documents for each table with schema, sample rows, and relationships
    - **sqldatabase**: Uses SQLDatabase utility to extract table schema information
    - **sqldatabaseloader**: Uses SQLDatabaseLoader to execute queries and extract data
    
    **Intelligent Loader Features:**
    - Table schema extraction (columns, types, constraints)
    - Sample row extraction (configurable number of rows)
    - Relationship detection (JOINs between tables)
    - Enhanced metadata (table names, record counts, etc.)
    
    **Example:**
    ```json
    {
        "db_path": "/path/to/company.db",
        "loader_type": "intelligent",
        "include_sample_rows": 5,
        "include_relationships": true
    }
    ```
    """,
    response_description="SQL database converted to Document objects"
)
async def ingest_sql_database(request: SQLIngestRequest):
    logger.info(
        f"SQL database ingestion request | "
        f"Database: {request.db_path} | "
        f"Loader: {request.loader_type}"
    )
    
    try:
        sql_loader = SQLDocumentLoader(
            loader_type=request.loader_type,
            include_sample_rows=request.include_sample_rows,
            include_relationships=request.include_relationships
        )
        
        documents = sql_loader.load_database(
            request.db_path,
            request.query,
            request.metadata
        )
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        
        logger.info(
            f"SQL database ingested successfully | "
            f"Database: {request.db_path} | "
            f"Documents: {len(documents)} | "
            f"Total chars: {total_chars}"
        )
        
        return DocumentIngestResponse(
            message=f"SQL database {request.db_path} ingested successfully",
            file_name=request.db_path,
            file_type="sql",
            documents_count=len(documents),
            total_chars=total_chars,
            documents=[
                {
                    "page_number": None,
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "length": len(doc.page_content),
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
        )
    except FileNotFoundError as e:
        logger.warning(f"SQL database not found | Database: {request.db_path} | Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"SQL database not found: {str(e)}"
        )
    except Exception as e:
        logger.error(
            f"Error ingesting SQL database | "
            f"Database: {request.db_path} | "
            f"Error: {type(e).__name__} | "
            f"Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting SQL database: {str(e)}"
        )


@router.get(
    "/supported-types",
    status_code=status.HTTP_200_OK,
    summary="Get supported file types",
    description="""
    Get a list of all supported file types and their descriptions.
    
    Returns information about which file extensions are supported
    and what each type is used for.
    """,
    response_description="List of supported file types and descriptions"
)
async def get_supported_types():
    supported_types = unified_loader.get_supported_types()
    
    type_descriptions = unified_loader.get_supported_types_with_descriptions()
    
    return {
        "supported_extensions": supported_types,
        "type_descriptions": {
            ext: type_descriptions.get(ext, "Unknown type")
            for ext in supported_types
        },
        "total_types": len(supported_types)
    }
