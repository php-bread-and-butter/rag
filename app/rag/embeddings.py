"""
Embeddings Module

Provides text embedding functionality using HuggingFace models.
Supports multiple embedding models and similarity calculations.
"""
import numpy as np
from typing import List, Optional, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from app.core.logging_config import get_logger

logger = get_logger(__name__)


# Popular embedding models with their specifications
AVAILABLE_MODELS = {
    "all-MiniLM-L6-v2": {
        "size": 384,
        "description": "Fast and efficient, good quality",
        "use_case": "General purpose, real-time applications",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "all-mpnet-base-v2": {
        "size": 768,
        "description": "Best quality, slower than MiniLM",
        "use_case": "When quality matters more than speed",
        "model_name": "sentence-transformers/all-mpnet-base-v2"
    },
    "all-MiniLM-L12-v2": {
        "size": 384,
        "description": "Slightly better than L6, bit slower",
        "use_case": "Good balance of speed and quality",
        "model_name": "sentence-transformers/all-MiniLM-L12-v2"
    },
    "multi-qa-MiniLM-L6-cos-v1": {
        "size": 384,
        "description": "Optimized for question-answering",
        "use_case": "Q&A systems, semantic search",
        "model_name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "size": 384,
        "description": "Supports 50+ languages",
        "use_case": "Multilingual applications",
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    }
}


class EmbeddingManager:
    """
    Manages text embeddings using HuggingFace models.
    
    Provides functionality to:
    - Embed single queries
    - Embed multiple documents
    - Calculate cosine similarity between embeddings
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the embedding manager with a HuggingFace model.

        Args:
            model_name: Name of the model (key from AVAILABLE_MODELS or full model path)
            model_kwargs: Additional arguments for the model
            encode_kwargs: Additional arguments for encoding
        """
        # Get model name from available models or use as-is
        if model_name in AVAILABLE_MODELS:
            full_model_name = AVAILABLE_MODELS[model_name]["model_name"]
            self.model_key = model_name
        else:
            full_model_name = model_name
            self.model_key = model_name

        logger.info(f"Initializing embedding model | Model: {full_model_name}")

        # Default model kwargs
        if model_kwargs is None:
            model_kwargs = {"device": "cpu"}

        # Default encode kwargs
        if encode_kwargs is None:
            encode_kwargs = {"normalize_embeddings": False}

        try:
            self.embeddings: Embeddings = HuggingFaceEmbeddings(
                model_name=full_model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            logger.info(f"✓ Embedding model loaded successfully | Model: {full_model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model | Model: {full_model_name} | Error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize embedding model '{full_model_name}': {str(e)}")

        self.model_name = full_model_name
        self.embedding_size = AVAILABLE_MODELS.get(self.model_key, {}).get("size", None)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query/text.

        Args:
            text: Text to embed

        Returns:
            List of float values representing the embedding vector
        """
        if not text or not text.strip():
            logger.warning("Attempted to embed empty text")
            raise ValueError("Text cannot be empty")

        try:
            embedding = self.embeddings.embed_query(text)
            logger.debug(f"Embedded query | Text length: {len(text)} | Embedding size: {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Error embedding query | Error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to embed query: {str(e)}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents/texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            logger.warning("Attempted to embed empty list of texts")
            return []

        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            logger.warning("No valid texts to embed after filtering")
            return []

        try:
            embeddings = self.embeddings.embed_documents(valid_texts)
            logger.info(f"Embedded {len(embeddings)} documents | Model: {self.model_name}")
            return embeddings
        except Exception as e:
            logger.error(f"Error embedding documents | Error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to embed documents: {str(e)}")

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two embedding vectors.

        Cosine similarity measures the angle between two vectors:
        - Result close to 1: Very similar
        - Result close to 0: Not related
        - Result close to -1: Opposite meanings

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        if len(vec1) != len(vec2):
            raise ValueError(f"Vectors must have the same length. Got {len(vec1)} and {len(vec2)}")

        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)

        dot_product = np.dot(vec1_array, vec2_array)
        norm_a = np.linalg.norm(vec1_array)
        norm_b = np.linalg.norm(vec2_array)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)
        return float(similarity)

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score between -1 and 1
        """
        embedding1 = self.embed_query(text1)
        embedding2 = self.embed_query(text2)
        return self.cosine_similarity(embedding1, embedding2)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current embedding model.

        Returns:
            Dictionary with model information
        """
        info = {
            "model_name": self.model_name,
            "model_key": self.model_key,
            "embedding_size": self.embedding_size
        }

        if self.model_key in AVAILABLE_MODELS:
            info.update({
                "description": AVAILABLE_MODELS[self.model_key]["description"],
                "use_case": AVAILABLE_MODELS[self.model_key]["use_case"]
            })

        return info


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available embedding models.

    Returns:
        Dictionary mapping model keys to their information
    """
    return AVAILABLE_MODELS.copy()


# Global default embedding manager instance
_default_embedding_manager: Optional[EmbeddingManager] = None


def get_default_embedding_manager(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingManager:
    """
    Get or create the default embedding manager instance.

    Args:
        model_name: Model name to use if creating a new instance

    Returns:
        EmbeddingManager instance
    """
    global _default_embedding_manager
    if _default_embedding_manager is None:
        _default_embedding_manager = EmbeddingManager(model_name=model_name)
    return _default_embedding_manager


def reset_default_embedding_manager():
    """Reset the default embedding manager (useful for testing or model switching)."""
    global _default_embedding_manager
    _default_embedding_manager = None
