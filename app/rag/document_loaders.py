"""
Step 1: Document Loaders for Text Data Ingestion and Parsing

This module provides document loaders for ingesting text data from various sources.
"""
from typing import List, Optional
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


class TextDocumentLoader:
    """
    Loader for plain text files
    
    Supports:
    - .txt files
    - Raw text strings
    """
    
    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize the text document loader
        
        Args:
            encoding: File encoding (default: utf-8)
        """
        self.encoding = encoding
    
    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a text file and return Document objects
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of Document objects
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        loader = TextLoader(file_path, encoding=self.encoding)
        documents = loader.load()
        
        return documents
    
    def load_text(self, text: str, metadata: Optional[dict] = None) -> Document:
        """
        Load raw text string as a Document
        
        Args:
            text: Raw text content
            metadata: Optional metadata dictionary
            
        Returns:
            Document object
        """
        return Document(
            page_content=text,
            metadata=metadata or {}
        )
    
    def load_files(self, file_paths: List[str]) -> List[Document]:
        """
        Load multiple text files
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of Document objects from all files
        """
        all_documents = []
        for file_path in file_paths:
            documents = self.load_file(file_path)
            all_documents.extend(documents)
        return all_documents


# Convenience function
def load_text_file(file_path: str, encoding: str = "utf-8") -> List[Document]:
    """
    Convenience function to load a text file
    
    Args:
        file_path: Path to the text file
        encoding: File encoding
        
    Returns:
        List of Document objects
    """
    loader = TextDocumentLoader(encoding=encoding)
    return loader.load_file(file_path)


def load_text(text: str, metadata: Optional[dict] = None) -> Document:
    """
    Convenience function to load raw text
    
    Args:
        text: Raw text content
        metadata: Optional metadata dictionary
        
    Returns:
        Document object
    """
    loader = TextDocumentLoader()
    return loader.load_text(text, metadata)
