"""
Embeddings Module

Provides text embedding functionality using HuggingFace and OpenAI models.
Supports multiple embedding models and similarity calculations.
"""
import os
import numpy as np
from typing import List, Optional, Dict, Any, Literal
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

from app.core.logging_config import get_logger
from app.core.config import settings

logger = get_logger(__name__)


# HuggingFace embedding models with their specifications
HUGGINGFACE_MODELS = {
    "all-MiniLM-L6-v2": {
        "provider": "huggingface",
        "size": 384,
        "description": "Fast and efficient, good quality",
        "use_case": "General purpose, real-time applications",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2"
    },
    "all-mpnet-base-v2": {
        "provider": "huggingface",
        "size": 768,
        "description": "Best quality, slower than MiniLM",
        "use_case": "When quality matters more than speed",
        "model_name": "sentence-transformers/all-mpnet-base-v2"
    },
    "all-MiniLM-L12-v2": {
        "provider": "huggingface",
        "size": 384,
        "description": "Slightly better than L6, bit slower",
        "use_case": "Good balance of speed and quality",
        "model_name": "sentence-transformers/all-MiniLM-L12-v2"
    },
    "multi-qa-MiniLM-L6-cos-v1": {
        "provider": "huggingface",
        "size": 384,
        "description": "Optimized for question-answering",
        "use_case": "Q&A systems, semantic search",
        "model_name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "provider": "huggingface",
        "size": 384,
        "description": "Supports 50+ languages",
        "use_case": "Multilingual applications",
        "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    }
}

# OpenAI embedding models with their specifications
OPENAI_MODELS = {
    "text-embedding-3-small": {
        "provider": "openai",
        "size": 1536,
        "description": "Good balance of performance and cost",
        "cost_per_1m_tokens": 0.02,
        "use_case": "General purpose, cost-effective",
        "model_name": "text-embedding-3-small"
    },
    "text-embedding-3-large": {
        "provider": "openai",
        "size": 3072,
        "description": "Highest quality embeddings",
        "cost_per_1m_tokens": 0.13,
        "use_case": "When accuracy is critical",
        "model_name": "text-embedding-3-large"
    },
    "text-embedding-ada-002": {
        "provider": "openai",
        "size": 1536,
        "description": "Previous generation model",
        "cost_per_1m_tokens": 0.10,
        "use_case": "Legacy applications",
        "model_name": "text-embedding-ada-002"
    }
}

# Combined available models
AVAILABLE_MODELS = {**HUGGINGFACE_MODELS, **OPENAI_MODELS}


class EmbeddingManager:
    """
    Manages text embeddings using HuggingFace or OpenAI models.
    
    Provides functionality to:
    - Embed single queries
    - Embed multiple documents
    - Calculate cosine similarity between embeddings
    - Semantic search
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        provider: Optional[Literal["huggingface", "openai"]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the embedding manager with a HuggingFace or OpenAI model.

        Args:
            model_name: Name of the model (key from AVAILABLE_MODELS or full model path)
            provider: Provider type ('huggingface' or 'openai'). Auto-detected if model_name is in AVAILABLE_MODELS
            model_kwargs: Additional arguments for the model
            encode_kwargs: Additional arguments for encoding
            openai_api_key: OpenAI API key (required for OpenAI models). Falls back to OPENAI_API_KEY env var
        """
        # Determine provider and model info
        if model_name in AVAILABLE_MODELS:
            model_info = AVAILABLE_MODELS[model_name]
            self.provider = provider or model_info.get("provider", "huggingface")
            full_model_name = model_info["model_name"]
            self.model_key = model_name
            self.embedding_size = model_info.get("size")
        else:
            # Custom model name - try to detect provider
            if provider:
                self.provider = provider
            elif model_name.startswith("text-embedding") or model_name.startswith("gpt"):
                self.provider = "openai"
            else:
                self.provider = provider or "huggingface"
            full_model_name = model_name
            self.model_key = model_name
            self.embedding_size = None

        logger.info(f"Initializing embedding model | Provider: {self.provider} | Model: {full_model_name}")

        try:
            if self.provider == "openai":
                # Initialize OpenAI embeddings
                api_key = openai_api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                        "or pass openai_api_key parameter."
                    )

                # Set environment variable for langchain
                os.environ["OPENAI_API_KEY"] = api_key

                self.embeddings: Embeddings = OpenAIEmbeddings(model=full_model_name)
                logger.info(f"✓ OpenAI embedding model loaded successfully | Model: {full_model_name}")
                
                # Get embedding size by embedding a test string
                try:
                    test_embedding = self.embeddings.embed_query("test")
                    self.embedding_size = len(test_embedding)
                except Exception:
                    pass  # Will be None if we can't determine

            else:
                # Initialize HuggingFace embeddings
                if model_kwargs is None:
                    model_kwargs = {"device": "cpu"}

                if encode_kwargs is None:
                    encode_kwargs = {"normalize_embeddings": False}

                self.embeddings: Embeddings = HuggingFaceEmbeddings(
                    model_name=full_model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
                logger.info(f"✓ HuggingFace embedding model loaded successfully | Model: {full_model_name}")

        except Exception as e:
            logger.error(f"Failed to load embedding model | Provider: {self.provider} | Model: {full_model_name} | Error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize embedding model '{full_model_name}': {str(e)}")

        self.model_name = full_model_name

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

    def semantic_search(
        self,
        query: str,
        documents: List[str],
        top_k: int = 3
    ) -> List[tuple]:
        """
        Perform semantic search to find most similar documents to a query.

        Args:
            query: Search query text
            documents: List of documents to search through
            top_k: Number of top results to return

        Returns:
            List of tuples (similarity_score, document_text) sorted by similarity (descending)
        """
        if not documents:
            logger.warning("No documents provided for semantic search")
            return []

        # Embed query and documents
        query_embedding = self.embed_query(query)
        doc_embeddings = self.embed_documents(documents)

        if not doc_embeddings:
            logger.warning("No valid embeddings generated for documents")
            return []

        # Calculate similarities
        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_emb)
            similarities.append((similarity, documents[i]))

        # Sort by similarity (descending)
        similarities.sort(reverse=True, key=lambda x: x[0])

        logger.info(f"Semantic search completed | Query: {query[:50]}... | Results: {len(similarities[:top_k])}")
        return similarities[:top_k]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current embedding model.

        Returns:
            Dictionary with model information
        """
        info = {
            "model_name": self.model_name,
            "model_key": self.model_key,
            "provider": self.provider,
            "embedding_size": self.embedding_size
        }

        if self.model_key in AVAILABLE_MODELS:
            model_info = AVAILABLE_MODELS[self.model_key]
            info.update({
                "description": model_info.get("description"),
                "use_case": model_info.get("use_case")
            })
            if self.provider == "openai" and "cost_per_1m_tokens" in model_info:
                info["cost_per_1m_tokens"] = model_info["cost_per_1m_tokens"]

        return info


def get_available_models(provider: Optional[Literal["huggingface", "openai"]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get information about all available embedding models.

    Args:
        provider: Optional filter by provider ('huggingface' or 'openai')

    Returns:
        Dictionary mapping model keys to their information
    """
    if provider:
        if provider == "huggingface":
            return HUGGINGFACE_MODELS.copy()
        elif provider == "openai":
            return OPENAI_MODELS.copy()
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
