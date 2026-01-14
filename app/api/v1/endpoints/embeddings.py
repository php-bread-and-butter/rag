"""
Embeddings API Endpoints

Provides endpoints for text embedding and similarity calculations.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.rag.embeddings import EmbeddingManager, get_available_models, get_default_embedding_manager
from app.core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


class EmbedQueryRequest(BaseModel):
    """Request model for embedding a single query"""
    text: str = Field(..., description="Text to embed", example="Hello, I am learning about embeddings!")
    model_name: Optional[str] = Field("all-MiniLM-L6-v2", description="Name of the embedding model to use")


class EmbedDocumentsRequest(BaseModel):
    """Request model for embedding multiple documents"""
    texts: List[str] = Field(..., description="List of texts to embed", example=[
        "The cat sat on the mat",
        "The dog played in the yard",
        "I love programming in Python"
    ])
    model_name: Optional[str] = Field("all-MiniLM-L6-v2", description="Name of the embedding model to use")


class SimilarityRequest(BaseModel):
    """Request model for calculating similarity between two texts"""
    text1: str = Field(..., description="First text", example="The cat sat on the mat")
    text2: str = Field(..., description="Second text", example="The kitten sat on the mat")
    model_name: Optional[str] = Field("all-MiniLM-L6-v2", description="Name of the embedding model to use")


class EmbeddingResponse(BaseModel):
    """Response model for embedding operations"""
    message: str = Field(..., example="Text embedded successfully")
    model_name: str = Field(..., example="sentence-transformers/all-MiniLM-L6-v2")
    embedding_size: int = Field(..., example=384)
    embedding: Optional[List[float]] = Field(None, description="Single embedding vector (for query)")
    embeddings: Optional[List[List[float]]] = Field(None, description="Multiple embedding vectors (for documents)")


class SimilarityResponse(BaseModel):
    """Response model for similarity calculations"""
    message: str = Field(..., example="Similarity calculated successfully")
    text1: str = Field(..., example="The cat sat on the mat")
    text2: str = Field(..., example="The kitten sat on the mat")
    similarity: float = Field(..., example=0.95, description="Cosine similarity score between -1 and 1")
    model_name: str = Field(..., example="sentence-transformers/all-MiniLM-L6-v2")
    interpretation: str = Field(..., example="Very similar", description="Human-readable interpretation of similarity")


def _get_interpretation(similarity: float) -> str:
    """Get human-readable interpretation of similarity score"""
    if similarity >= 0.9:
        return "Very similar"
    elif similarity >= 0.7:
        return "Similar"
    elif similarity >= 0.5:
        return "Somewhat similar"
    elif similarity >= 0.3:
        return "Not very similar"
    elif similarity >= 0.0:
        return "Not similar"
    else:
        return "Opposite meanings"


@router.post(
    "/query",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Embed a single query/text",
    description="""
    Generate an embedding vector for a single piece of text.
    
    **Use cases:**
    - Embedding user queries for semantic search
    - Converting text to vector representation
    - Single text embedding for comparison
    
    **Example:**
    ```json
    {
        "text": "Hello, I am learning about embeddings!",
        "model_name": "all-MiniLM-L6-v2"
    }
    ```
    
    **Returns:**
    - Embedding vector (list of floats)
    - Model information
    - Embedding size
    """,
    response_description="Embedding vector for the input text"
)
async def embed_query(request: EmbedQueryRequest):
    logger.info(f"Query embedding request | Text length: {len(request.text)} | Model: {request.model_name}")
    
    try:
        # Get or create embedding manager for the specified model
        if request.model_name == "all-MiniLM-L6-v2":
            # Use default manager for default model
            embedding_manager = get_default_embedding_manager()
        else:
            # Create new manager for custom model
            embedding_manager = EmbeddingManager(model_name=request.model_name)
        
        # Embed the text
        embedding = embedding_manager.embed_query(request.text)
        model_info = embedding_manager.get_model_info()
        
        logger.info(
            f"Query embedded successfully | "
            f"Text length: {len(request.text)} | "
            f"Embedding size: {len(embedding)} | "
            f"Model: {model_info['model_name']}"
        )
        
        return EmbeddingResponse(
            message="Text embedded successfully",
            model_name=model_info["model_name"],
            embedding_size=len(embedding),
            embedding=embedding,
            embeddings=None
        )
    except ValueError as e:
        logger.warning(f"Invalid input for embedding | Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Error embedding query | Model: {request.model_name} | Error: {type(e).__name__} | Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error embedding query: {str(e)}"
        )


@router.post(
    "/documents",
    response_model=EmbeddingResponse,
    status_code=status.HTTP_200_OK,
    summary="Embed multiple documents/texts",
    description="""
    Generate embedding vectors for multiple texts at once.
    
    **Use cases:**
    - Batch embedding of documents
    - Embedding document chunks for vector storage
    - Processing multiple texts efficiently
    
    **Example:**
    ```json
    {
        "texts": [
            "The cat sat on the mat",
            "The dog played in the yard",
            "I love programming in Python"
        ],
        "model_name": "all-MiniLM-L6-v2"
    }
    ```
    
    **Returns:**
    - List of embedding vectors (one per input text)
    - Model information
    - Embedding size
    """,
    response_description="List of embedding vectors for the input texts"
)
async def embed_documents(request: EmbedDocumentsRequest):
    logger.info(
        f"Documents embedding request | "
        f"Texts count: {len(request.texts)} | "
        f"Model: {request.model_name}"
    )
    
    try:
        if not request.texts:
            raise ValueError("Texts list cannot be empty")
        
        # Get or create embedding manager for the specified model
        if request.model_name == "all-MiniLM-L6-v2":
            # Use default manager for default model
            embedding_manager = get_default_embedding_manager()
        else:
            # Create new manager for custom model
            embedding_manager = EmbeddingManager(model_name=request.model_name)
        
        # Embed the documents
        embeddings = embedding_manager.embed_documents(request.texts)
        model_info = embedding_manager.get_model_info()
        
        if embeddings:
            embedding_size = len(embeddings[0])
        else:
            embedding_size = model_info.get("embedding_size", 0)
        
        logger.info(
            f"Documents embedded successfully | "
            f"Texts count: {len(request.texts)} | "
            f"Embeddings count: {len(embeddings)} | "
            f"Embedding size: {embedding_size} | "
            f"Model: {model_info['model_name']}"
        )
        
        return EmbeddingResponse(
            message=f"Successfully embedded {len(embeddings)} document(s)",
            model_name=model_info["model_name"],
            embedding_size=embedding_size,
            embedding=None,
            embeddings=embeddings
        )
    except ValueError as e:
        logger.warning(f"Invalid input for embedding | Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Error embedding documents | Model: {request.model_name} | Error: {type(e).__name__} | Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error embedding documents: {str(e)}"
        )


@router.post(
    "/similarity",
    response_model=SimilarityResponse,
    status_code=status.HTTP_200_OK,
    summary="Calculate similarity between two texts",
    description="""
    Calculate cosine similarity between two texts using embeddings.
    
    **Cosine Similarity:**
    - Result close to 1: Very similar
    - Result close to 0: Not related
    - Result close to -1: Opposite meanings
    
    **Use cases:**
    - Finding similar documents
    - Semantic search
    - Text matching and deduplication
    
    **Example:**
    ```json
    {
        "text1": "The cat sat on the mat",
        "text2": "The kitten sat on the mat",
        "model_name": "all-MiniLM-L6-v2"
    }
    ```
    
    **Returns:**
    - Similarity score (float between -1 and 1)
    - Human-readable interpretation
    - Model information
    """,
    response_description="Similarity score and interpretation"
)
async def calculate_similarity(request: SimilarityRequest):
    logger.info(
        f"Similarity calculation request | "
        f"Text1 length: {len(request.text1)} | "
        f"Text2 length: {len(request.text2)} | "
        f"Model: {request.model_name}"
    )
    
    try:
        # Get or create embedding manager for the specified model
        if request.model_name == "all-MiniLM-L6-v2":
            # Use default manager for default model
            embedding_manager = get_default_embedding_manager()
        else:
            # Create new manager for custom model
            embedding_manager = EmbeddingManager(model_name=request.model_name)
        
        # Calculate similarity
        similarity_score = embedding_manager.similarity(request.text1, request.text2)
        interpretation = _get_interpretation(similarity_score)
        model_info = embedding_manager.get_model_info()
        
        logger.info(
            f"Similarity calculated successfully | "
            f"Similarity: {similarity_score:.4f} | "
            f"Interpretation: {interpretation} | "
            f"Model: {model_info['model_name']}"
        )
        
        return SimilarityResponse(
            message="Similarity calculated successfully",
            text1=request.text1,
            text2=request.text2,
            similarity=similarity_score,
            model_name=model_info["model_name"],
            interpretation=interpretation
        )
    except ValueError as e:
        logger.warning(f"Invalid input for similarity calculation | Error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            f"Error calculating similarity | Model: {request.model_name} | Error: {type(e).__name__} | Message: {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating similarity: {str(e)}"
        )


@router.get(
    "/models",
    status_code=status.HTTP_200_OK,
    summary="List available embedding models",
    description="""
    Get information about all available embedding models.
    
    **Returns:**
    - Model names and keys
    - Embedding dimensions
    - Descriptions and use cases
    - Recommended applications
    """,
    response_description="Dictionary of available models with their specifications"
)
async def list_models():
    """List all available embedding models"""
    models = get_available_models()
    
    logger.info(f"Retrieved {len(models)} available embedding models")
    
    return {
        "message": "Available embedding models retrieved successfully",
        "models": models,
        "default": "all-MiniLM-L6-v2",
        "count": len(models)
    }
