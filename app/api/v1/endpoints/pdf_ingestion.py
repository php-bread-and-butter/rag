"""
Step 3: PDF Document Ingestion API Endpoints

Endpoints for ingesting and parsing PDF documents.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, status
from pydantic import BaseModel

from app.rag.pdf_loaders import load_pdf_file, load_pdf_bytes, PDFDocumentLoader
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


class PDFIngestResponse(BaseModel):
    """Response model for PDF ingestion"""
    message: str
    pages_count: int
    total_chars: int
    pages: List[dict]
    file_name: Optional[str] = None


class PDFFilesRequest(BaseModel):
    """Request model for ingesting PDFs from file paths"""
    file_paths: List[str]
    use_pymupdf: bool = True
    metadata: Optional[dict] = None


@router.post("/pdf/upload", response_model=PDFIngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf_file(
    file: UploadFile = File(...),
    use_pymupdf: bool = True
):
    """
    Upload and ingest a PDF file
    
    This endpoint accepts a PDF file upload and processes it.
    Supports both PyPDF and PyMuPDF libraries.
    
    Args:
        file: PDF file to upload
        use_pymupdf: Use PyMuPDF (faster, better) if True, PyPDF if False
    """
    logger.info(
        f"PDF upload request | "
        f"Filename: {file.filename} | "
        f"Content-Type: {file.content_type} | "
        f"Use PyMuPDF: {use_pymupdf}"
    )
    
    try:
        # Check file extension
        if not file.filename.lower().endswith('.pdf'):
            logger.warning(f"Invalid file type | Filename: {file.filename} | Expected: .pdf")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only .pdf files are supported"
            )
        
        # Read file content
        content = await file.read()
        logger.debug(f"PDF file read | Size: {len(content)} bytes")
        
        # Create metadata
        metadata = {
            "source": file.filename,
            "file_type": "application/pdf",
            "uploaded": True
        }
        
        # Load PDF from bytes
        documents = load_pdf_bytes(content, use_pymupdf=use_pymupdf, metadata=metadata)
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        
        logger.info(
            f"PDF ingested successfully | "
            f"Filename: {file.filename} | "
            f"Pages: {len(documents)} | "
            f"Total chars: {total_chars}"
        )
        
        return PDFIngestResponse(
            message=f"PDF file {file.filename} ingested successfully",
            pages_count=len(documents),
            total_chars=total_chars,
            file_name=file.filename,
            pages=[
                {
                    "page_number": idx + 1,
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "length": len(doc.page_content),
                    "metadata": doc.metadata
                }
                for idx, doc in enumerate(documents)
            ]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error processing PDF file | "
            f"Filename: {file.filename} | "
            f"Error: {type(e).__name__} | "
            f"Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing PDF file: {str(e)}"
        )


@router.post("/pdf/files", response_model=PDFIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_pdf_files(request: PDFFilesRequest):
    """
    Ingest multiple PDF files from file paths
    
    This endpoint accepts a list of PDF file paths and loads them.
    Note: Files must be accessible from the server.
    
    Args:
        request: Request containing file paths and options
    """
    logger.info(
        f"PDF files ingestion request | "
        f"Count: {len(request.file_paths)} | "
        f"Use PyMuPDF: {request.use_pymupdf}"
    )
    
    try:
        loader = PDFDocumentLoader(use_pymupdf=request.use_pymupdf)
        documents = loader.load_files(request.file_paths, request.metadata)
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        
        logger.info(
            f"PDF files ingested successfully | "
            f"Files: {len(request.file_paths)} | "
            f"Total pages: {len(documents)} | "
            f"Total chars: {total_chars}"
        )
        
        return PDFIngestResponse(
            message=f"Successfully ingested {len(request.file_paths)} PDF file(s) with {len(documents)} page(s)",
            pages_count=len(documents),
            total_chars=total_chars,
            pages=[
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
        logger.warning(f"PDF file not found | Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Error ingesting PDF files | "
            f"Error: {type(e).__name__} | "
            f"Message: {str(e)} | "
            f"Paths: {request.file_paths}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting PDF files: {str(e)}"
        )


@router.get("/pdf/info/{file_path:path}", status_code=status.HTTP_200_OK)
async def get_pdf_info(file_path: str):
    """
    Get information about a PDF file (page count, etc.)
    
    Args:
        file_path: Path to the PDF file
    """
    logger.info(f"PDF info request | File: {file_path}")
    
    try:
        loader = PDFDocumentLoader()
        page_count = loader.get_page_count(file_path)
        
        return {
            "file_path": file_path,
            "page_count": page_count,
            "status": "success"
        }
    except FileNotFoundError as e:
        logger.warning(f"PDF file not found | File: {file_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Error getting PDF info | File: {file_path} | Error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading PDF file: {str(e)}"
        )
