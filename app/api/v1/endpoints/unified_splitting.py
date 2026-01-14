"""
Unified Text Splitting API Endpoints

Single endpoint that handles both text and document splitting.
"""
from typing import List, Optional, Union
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.rag.text_splitters import TextSplitterManager
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


class UnifiedSplitRequest(BaseModel):
    """Unified request model for splitting text or documents"""
    # Either provide text OR documents, not both
    text: Optional[str] = Field(None, description="Raw text to split (use this OR documents, not both)", example="This is a long text that needs to be split into smaller chunks.")
    documents: Optional[List[dict]] = Field(None, description="List of documents to split. Each document should have 'page_content' and optional 'metadata' (use this OR text, not both)", example=[{"page_content": "Document content here...", "metadata": {"source": "doc1"}}])
    
    # Splitting parameters
    chunk_size: int = Field(default=1000, ge=1, le=10000, description="Size of each chunk (characters or tokens depending on splitter)")
    chunk_overlap: int = Field(default=200, ge=0, le=1000, description="Overlap between chunks to maintain context")
    splitter_type: str = Field(
        default="recursive",
        description="Type of splitter: recursive (recommended), character, token, markdown, python",
        example="recursive"
    )


class SplitResponse(BaseModel):
    """Response model for text/document splitting"""
    message: str
    input_type: str  # "text" or "documents"
    original_length: int
    chunks_count: int
    chunks: List[dict]
    splitter_type: str
    chunk_size: int
    chunk_overlap: int


@router.post(
    "/split",
    response_model=SplitResponse,
    status_code=status.HTTP_200_OK,
    summary="Split text or documents into chunks",
    description="""
    Unified endpoint for splitting text or documents into chunks.
    
    **You can provide either:**
    - `text`: Raw text string to split
    - `documents`: List of document objects with 'page_content' and optional 'metadata'
    
    **Supported splitter types:**
    - **recursive** (recommended): RecursiveCharacterTextSplitter - Handles most text well, splits on multiple separators
    - **character**: CharacterTextSplitter - Simple character-based splitting
    - **token**: TokenTextSplitter - Token-based splitting using tiktoken (useful for LLM context windows)
    - **markdown**: MarkdownTextSplitter - Preserves markdown structure
    - **python**: PythonCodeTextSplitter - For Python source code
    
    **Parameters:**
    - `chunk_size`: Maximum size of each chunk (characters or tokens)
    - `chunk_overlap`: Overlap between chunks to maintain context
    """,
    response_description="Text or documents split into chunks with metadata",
    responses={
        200: {
            "description": "Successfully split into chunks",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Text split into 5 chunks",
                        "input_type": "text",
                        "original_length": 5000,
                        "chunks_count": 5,
                        "chunks": [
                            {
                                "content": "First chunk content...",
                                "length": 1000,
                                "index": 0
                            }
                        ],
                        "splitter_type": "recursive",
                        "chunk_size": 1000,
                        "chunk_overlap": 200
                    }
                }
            }
        },
        400: {"description": "Invalid request (missing text/documents or invalid parameters)"},
        500: {"description": "Internal server error"}
    }
)
async def split_unified(request: UnifiedSplitRequest):
    """
    Split text or documents into chunks using various splitting techniques
    
    This unified endpoint accepts either raw text or a list of documents
    and splits them using the specified splitter type.
    """
    # Validate that exactly one input type is provided
    if request.text and request.documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either 'text' OR 'documents', not both"
        )
    
    if not request.text and not request.documents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either 'text' or 'documents'"
        )
    
    logger.info(
        f"Unified split request | "
        f"Input type: {'text' if request.text else 'documents'} | "
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
        
        if request.text:
            # Split text
            chunks = splitter.split_text(request.text)
            original_length = len(request.text)
            input_type = "text"
            
            logger.info(
                f"Text split successfully | "
                f"Original length: {original_length} | "
                f"Chunks: {len(chunks)}"
            )
            
            return SplitResponse(
                message=f"Text split into {len(chunks)} chunks",
                input_type=input_type,
                original_length=original_length,
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
        
        else:
            # Split documents
            from langchain_core.documents import Document
            
            # Convert dict documents to Document objects
            documents = [
                Document(
                    page_content=doc.get("page_content", ""),
                    metadata=doc.get("metadata", {})
                )
                for doc in request.documents
            ]
            
            chunks = splitter.split_documents(documents)
            total_original_length = sum(len(doc.page_content) for doc in documents)
            input_type = "documents"
            
            logger.info(
                f"Documents split successfully | "
                f"Original docs: {len(documents)} | "
                f"Chunks: {len(chunks)}"
            )
            
            return SplitResponse(
                message=f"Split {len(documents)} document(s) into {len(chunks)} chunks",
                input_type=input_type,
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
        logger.warning(f"Invalid splitter type | Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Error splitting | Input type: {'text' if request.text else 'documents'} | Error: {type(e).__name__} | Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error splitting: {str(e)}"
        )


@router.get(
    "/splitters",
    status_code=status.HTTP_200_OK,
    summary="List available text splitters",
    description="""
    Get information about all available text splitter types.
    
    Returns detailed descriptions of each splitter type, including:
    - Name and description
    - Best use cases
    - When to use each splitter
    """,
    response_description="List of available splitters with descriptions"
)
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
