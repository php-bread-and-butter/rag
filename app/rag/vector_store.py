"""
Vector Store Module using ChromaDB

Manages vector storage and retrieval for RAG applications.
"""
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.rag.embeddings import EmbeddingManager
from app.core.logging_config import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class VectorStoreManager:
    """
    Manages ChromaDB vector store for RAG applications.
    
    Provides functionality to:
    - Create and manage collections
    - Store documents with embeddings
    - Search similar documents
    - Manage vector store collections
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: Optional[str] = None,
        embedding_model: Optional[EmbeddingManager] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_provider: Optional[str] = None
    ):
        """
        Initialize the vector store manager.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data (None for in-memory)
            embedding_model: Pre-initialized EmbeddingManager (optional)
            embedding_model_name: Name of embedding model to use
            embedding_provider: Provider type ('huggingface' or 'openai')
        """
        self.collection_name = collection_name
        
        # Set persist directory
        if persist_directory is None:
            # Use config setting or default to ./chroma_db directory
            persist_directory = settings.CHROMA_DB_PATH or os.path.join(os.getcwd(), "chroma_db")
        
        self.persist_directory = persist_directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        if embedding_model is None:
            self.embedding_manager = EmbeddingManager(
                model_name=embedding_model_name,
                provider=embedding_provider
            )
        else:
            self.embedding_manager = embedding_model
        
        # Get the embeddings instance
        self.embeddings: Embeddings = self.embedding_manager.embeddings
        
        # Initialize ChromaDB
        self._vector_store: Optional[Chroma] = None
        self._load_or_create_collection()
        
        logger.info(
            f"VectorStoreManager initialized | "
            f"Collection: {collection_name} | "
            f"Persist directory: {persist_directory} | "
            f"Embedding model: {embedding_model_name}"
        )

    def _load_or_create_collection(self):
        """Load existing collection or create a new one"""
        try:
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Could not load collection, creating new one: {str(e)}")
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add
            ids: Optional list of IDs for documents (auto-generated if None)

        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents provided to add")
            return []

        try:
            # Add documents to ChromaDB
            document_ids = self._vector_store.add_documents(
                documents=documents,
                ids=ids
            )
            
            logger.info(
                f"Added {len(document_ids)} documents to collection '{self.collection_name}'"
            )
            
            return document_ids
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to add documents to vector store: {str(e)}")

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents using semantic search.

        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of similar Document objects
        """
        try:
            results = self._vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            
            logger.info(
                f"Similarity search completed | "
                f"Query: {query[:50]}... | "
                f"Results: {len(results)}"
            )
            
            return results
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to perform similarity search: {str(e)}")

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        Search for similar documents with similarity scores.

        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of tuples (Document, score)
        """
        try:
            results = self._vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            logger.info(
                f"Similarity search with scores completed | "
                f"Query: {query[:50]}... | "
                f"Results: {len(results)}"
            )
            
            return results
        except Exception as e:
            logger.error(f"Error performing similarity search with scores: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to perform similarity search with scores: {str(e)}")

    def delete_collection(self):
        """Delete the current collection"""
        try:
            # ChromaDB doesn't have a direct delete method, but we can delete the directory
            collection_path = os.path.join(self.persist_directory, self.collection_name)
            if os.path.exists(collection_path):
                import shutil
                shutil.rmtree(collection_path)
                logger.info(f"Deleted collection: {self.collection_name}")
            
            # Recreate empty collection
            self._load_or_create_collection()
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to delete collection: {str(e)}")

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.

        Returns:
            Dictionary with collection information
        """
        try:
            # Get collection count
            collection = self._vector_store._collection
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "document_count": count,
                "embedding_model": self.embedding_manager.model_name,
                "embedding_provider": self.embedding_manager.provider
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}", exc_info=True)
            return {
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "document_count": 0,
                "error": str(e)
            }

    def get_vector_store(self) -> Chroma:
        """Get the underlying ChromaDB vector store"""
        return self._vector_store


# Global vector store managers cache
_vector_store_managers: Dict[str, VectorStoreManager] = {}


def get_vector_store_manager(
    collection_name: str = "default",
    persist_directory: Optional[str] = None,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    embedding_provider: Optional[str] = None
) -> VectorStoreManager:
    """
    Get or create a vector store manager instance.

    Args:
        collection_name: Name of the collection
        persist_directory: Directory to persist data
        embedding_model_name: Name of embedding model
        embedding_provider: Provider type

    Returns:
        VectorStoreManager instance
    """
    cache_key = f"{collection_name}_{persist_directory}_{embedding_model_name}_{embedding_provider}"
    
    if cache_key not in _vector_store_managers:
        _vector_store_managers[cache_key] = VectorStoreManager(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model_name=embedding_model_name,
            embedding_provider=embedding_provider
        )
    
    return _vector_store_managers[cache_key]
