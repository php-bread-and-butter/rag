"""
Unified Document Ingestion API Endpoints

Single endpoint that handles all file types automatically.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from pydantic import BaseModel

from app.rag.unified_loader import UnifiedDocumentLoader
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Initialize unified loader
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
    file_paths: List[str]
    use_pymupdf: bool = True
    metadata: Optional[dict] = None


class TextIngestRequest(BaseModel):
    """Request model for raw text ingestion"""
    text: str
    metadata: Optional[dict] = None


@router.post("/upload", response_model=DocumentIngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    use_pymupdf: bool = True
):
    """
    Upload and ingest a document (automatically detects file type)
    
    Supported file types:
    - .txt, .text - Text files
    - .pdf - PDF documents
    
    The system automatically detects the file type and uses the appropriate processor.
    """
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
        
        # Load document using unified loader
        documents = unified_loader.load_bytes(
            filename=file.filename or "unknown",
            content=content,
            metadata=metadata,
            use_pymupdf=use_pymupdf
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


@router.post("/files", response_model=DocumentIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_files(request: FilesIngestRequest):
    """
    Ingest multiple files from file paths (automatically detects file types)
    
    Supported file types:
    - .txt, .text - Text files
    - .pdf - PDF documents
    
    The system automatically detects each file's type and uses the appropriate processor.
    """
    logger.info(
        f"Files ingestion request | "
        f"Count: {len(request.file_paths)} | "
        f"Use PyMuPDF: {request.use_pymupdf}"
    )
    
    try:
        documents = unified_loader.load_files(
            request.file_paths,
            request.metadata,
            request.use_pymupdf
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


@router.post("/text", response_model=DocumentIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_text(request: TextIngestRequest):
    """
    Ingest raw text data
    
    This endpoint accepts raw text and converts it into a Document object.
    """
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


@router.get("/supported-types", status_code=status.HTTP_200_OK)
async def get_supported_types():
    """
    Get list of supported file types
    """
    supported_types = unified_loader.get_supported_types()
    
    type_descriptions = {
        '.txt': 'Plain text files',
        '.text': 'Plain text files',
        '.pdf': 'PDF documents',
    }
    
    return {
        "supported_extensions": supported_types,
        "type_descriptions": {
            ext: type_descriptions.get(ext, "Unknown type")
            for ext in supported_types
        },
        "total_types": len(supported_types)
    }
