"""
Step 2: Text Splitting API Endpoints

Endpoints for splitting text and documents using various techniques.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.rag.text_splitters import TextSplitterManager
from app.rag.document_loaders import load_text
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


class TextSplitRequest(BaseModel):
    """Request model for text splitting"""
    text: str
    chunk_size: int = Field(default=1000, ge=1, le=10000, description="Size of each chunk")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Overlap between chunks")
    splitter_type: str = Field(
        default="recursive",
        description="Type of splitter: recursive, character, token, markdown, python"
    )


class DocumentSplitRequest(BaseModel):
    """Request model for document splitting"""
    documents: List[dict] = Field(description="List of documents with 'page_content' and optional 'metadata'")
    chunk_size: int = Field(default=1000, ge=1, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    splitter_type: str = Field(default="recursive")


class SplitResponse(BaseModel):
    """Response model for text splitting"""
    message: str
    original_length: int
    chunks_count: int
    chunks: List[dict]
    splitter_type: str
    chunk_size: int
    chunk_overlap: int


@router.post("/text/split", response_model=SplitResponse, status_code=status.HTTP_200_OK)
async def split_text(request: TextSplitRequest):
    """
    Split text into chunks using various splitting techniques
    
    Supported splitter types:
    - recursive: RecursiveCharacterTextSplitter (recommended, handles most text well)
    - character: CharacterTextSplitter (simple character-based splitting)
    - token: TokenTextSplitter (token-based splitting using tiktoken)
    - markdown: MarkdownTextSplitter (preserves markdown structure)
    - python: PythonCodeTextSplitter (for Python code)
    """
    logger.info(
        f"Text split request | "
        f"Text length: {len(request.text)} | "
        f"Chunk size: {request.chunk_size} | "
        f"Overlap: {request.chunk_overlap} | "
        f"Splitter: {request.splitter_type}"
    )
    
    try:
        splitter = TextSplitterManager(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            splitter_type=request.splitter_type
        )
        
        chunks = splitter.split_text(request.text)
        
        logger.info(
            f"Text split successfully | "
            f"Original length: {len(request.text)} | "
            f"Chunks: {len(chunks)}"
        )
        
        return SplitResponse(
            message=f"Text split into {len(chunks)} chunks",
            original_length=len(request.text),
            chunks_count=len(chunks),
            chunks=[
                {
                    "content": chunk,
                    "length": len(chunk),
                    "index": idx
                }
                for idx, chunk in enumerate(chunks)
            ],
            splitter_type=request.splitter_type,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
    except ValueError as e:
        logger.warning(f"Invalid splitter type | Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Error splitting text | Error: {type(e).__name__} | Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error splitting text: {str(e)}"
        )


@router.post("/documents/split", response_model=SplitResponse, status_code=status.HTTP_200_OK)
async def split_documents(request: DocumentSplitRequest):
    """
    Split multiple documents into chunks
    
    Documents should have 'page_content' and optional 'metadata' fields.
    """
    logger.info(
        f"Document split request | "
        f"Documents: {len(request.documents)} | "
        f"Chunk size: {request.chunk_size} | "
        f"Splitter: {request.splitter_type}"
    )
    
    try:
        # Convert dict documents to Document objects
        from langchain_core.documents import Document
        
        documents = [
            Document(
                page_content=doc.get("page_content", ""),
                metadata=doc.get("metadata", {})
            )
            for doc in request.documents
        ]
        
        splitter = TextSplitterManager(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            splitter_type=request.splitter_type
        )
        
        chunks = splitter.split_documents(documents)
        
        total_original_length = sum(len(doc.page_content) for doc in documents)
        
        logger.info(
            f"Documents split successfully | "
            f"Original docs: {len(documents)} | "
            f"Chunks: {len(chunks)}"
        )
        
        return SplitResponse(
            message=f"Split {len(documents)} document(s) into {len(chunks)} chunks",
            original_length=total_original_length,
            chunks_count=len(chunks),
            chunks=[
                {
                    "content": chunk.page_content,
                    "length": len(chunk.page_content),
                    "metadata": chunk.metadata,
                    "index": idx
                }
                for idx, chunk in enumerate(chunks)
            ],
            splitter_type=request.splitter_type,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
    except ValueError as e:
        logger.warning(f"Invalid request | Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Error splitting documents | Error: {type(e).__name__} | Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error splitting documents: {str(e)}"
        )


@router.get("/splitters", status_code=status.HTTP_200_OK)
async def list_splitters():
    """
    List available text splitter types and their descriptions
    """
    splitters = {
        "recursive": {
            "name": "RecursiveCharacterTextSplitter",
            "description": "Recommended for most text. Recursively splits on multiple separators.",
            "best_for": "General text, articles, documents"
        },
        "character": {
            "name": "CharacterTextSplitter",
            "description": "Simple character-based splitting with a single separator.",
            "best_for": "Simple text, plain documents"
        },
        "token": {
            "name": "TokenTextSplitter",
            "description": "Token-based splitting using tiktoken encoding.",
            "best_for": "When you need exact token counts (e.g., for LLM context windows)"
        },
        "markdown": {
            "name": "MarkdownTextSplitter",
            "description": "Preserves markdown structure while splitting.",
            "best_for": "Markdown files, documentation"
        },
        "python": {
            "name": "PythonCodeTextSplitter",
            "description": "Splits Python code while preserving code structure.",
            "best_for": "Python source code files"
        }
    }
    
    return {
        "available_splitters": splitters,
        "default": "recursive"
    }
