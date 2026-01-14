"""
Step 1: Document Ingestion API Endpoints

Endpoints for ingesting and parsing text data using document loaders.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, status, Request
from pydantic import BaseModel

from app.rag.document_loaders import load_text_file, load_text, TextDocumentLoader
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


class TextIngestRequest(BaseModel):
    """Request model for text ingestion"""
    text: str
    metadata: Optional[dict] = None


class TextIngestResponse(BaseModel):
    """Response model for text ingestion"""
    message: str
    documents_count: int
    total_chars: int
    documents: List[dict]


@router.post("/text/ingest", response_model=TextIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_text(request: TextIngestRequest):
    """
    Ingest raw text data
    
    This endpoint accepts raw text and converts it into a Document object.
    """
    logger.info(f"Text ingestion request | Text length: {len(request.text)} chars")
    
    try:
        document = load_text(request.text, request.metadata)
        
        logger.info(
            f"Text ingested successfully | "
            f"Chars: {len(document.page_content)} | "
            f"Metadata: {document.metadata}"
        )
        
        return TextIngestResponse(
            message="Text ingested successfully",
            documents_count=1,
            total_chars=len(document.page_content),
            documents=[{
                "content": document.page_content[:200] + "..." if len(document.page_content) > 200 else document.page_content,
                "metadata": document.metadata,
                "length": len(document.page_content)
            }]
        )
    except Exception as e:
        logger.error(
            f"Error ingesting text | Error: {type(e).__name__} | Message: {str(e)}",
            exc_info=True,
            extra={"text_length": len(request.text) if request.text else 0}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting text: {str(e)}"
        )


@router.post("/text/upload", response_model=TextIngestResponse, status_code=status.HTTP_201_CREATED)
async def upload_text_file(file: UploadFile = File(...)):
    """
    Upload and ingest a text file
    
    This endpoint accepts a .txt file upload and processes it.
    """
    logger.info(f"File upload request | Filename: {file.filename} | Content-Type: {file.content_type}")
    
    try:
        # Check file extension
        if not file.filename.endswith('.txt'):
            logger.warning(f"Invalid file type | Filename: {file.filename} | Expected: .txt")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only .txt files are supported"
            )
        
        # Read file content
        content = await file.read()
        logger.debug(f"File read | Size: {len(content)} bytes")
        
        text_content = content.decode('utf-8')
        
        # Create metadata
        metadata = {
            "source": file.filename,
            "file_type": "text/plain"
        }
        
        # Load as document
        document = load_text(text_content, metadata)
        
        logger.info(
            f"File ingested successfully | "
            f"Filename: {file.filename} | "
            f"Chars: {len(document.page_content)}"
        )
        
        return TextIngestResponse(
            message=f"File {file.filename} ingested successfully",
            documents_count=1,
            total_chars=len(document.page_content),
            documents=[{
                "content": document.page_content[:200] + "..." if len(document.page_content) > 200 else document.page_content,
                "metadata": document.metadata,
                "length": len(document.page_content)
            }]
        )
    except UnicodeDecodeError as e:
        logger.error(
            f"File encoding error | Filename: {file.filename} | Error: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File encoding error. Please ensure the file is UTF-8 encoded."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error processing file | Filename: {file.filename} | Error: {type(e).__name__} | Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )


@router.post("/text/files", response_model=TextIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_text_files(file_paths: List[str]):
    """
    Ingest multiple text files from file paths
    
    This endpoint accepts a list of file paths and loads them.
    Note: Files must be accessible from the server.
    """
    logger.info(f"File paths ingestion request | Count: {len(file_paths)} | Paths: {file_paths}")
    
    try:
        loader = TextDocumentLoader()
        documents = loader.load_files(file_paths)
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        
        logger.info(
            f"Files ingested successfully | "
            f"Count: {len(documents)} | "
            f"Total chars: {total_chars}"
        )
        
        return TextIngestResponse(
            message=f"Successfully ingested {len(documents)} file(s)",
            documents_count=len(documents),
            total_chars=total_chars,
            documents=[{
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata,
                "length": len(doc.page_content)
            } for doc in documents]
        )
    except FileNotFoundError as e:
        logger.warning(f"File not found | Error: {str(e)} | Paths: {file_paths}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Error ingesting files | Error: {type(e).__name__} | Message: {str(e)} | Paths: {file_paths}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting files: {str(e)}"
        )
