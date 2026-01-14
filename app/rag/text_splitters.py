"""
Step 2: Text Splitting Techniques

This module provides various text splitting strategies for chunking documents.
"""
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    HTMLHeaderTextSplitter,
    HTMLSectionSplitter,
)
import tiktoken


class TextSplitterManager:
    """
    Manager for different text splitting strategies
    
    Supports:
    - RecursiveCharacterTextSplitter (recommended)
    - CharacterTextSplitter
    - TokenTextSplitter
    - MarkdownTextSplitter
    - PythonCodeTextSplitter
    - HTMLHeaderTextSplitter
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        splitter_type: str = "recursive"
    ):
        """
        Initialize text splitter
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            splitter_type: Type of splitter to use
                - "recursive" (default): RecursiveCharacterTextSplitter
                - "character": CharacterTextSplitter
                - "token": TokenTextSplitter
                - "markdown": MarkdownTextSplitter
                - "python": PythonCodeTextSplitter
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter_type = splitter_type.lower()
        self._splitter = self._create_splitter()
    
    def _create_splitter(self):
        """Create the appropriate text splitter"""
        if self.splitter_type == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
        elif self.splitter_type == "character":
            return CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n\n"
            )
        elif self.splitter_type == "token":
            encoding = tiktoken.get_encoding("cl100k_base")
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                encoding_name="cl100k_base"
            )
        elif self.splitter_type == "markdown":
            return MarkdownTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        elif self.splitter_type == "python":
            return PythonCodeTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            raise ValueError(
                f"Unknown splitter type: {self.splitter_type}. "
                "Supported types: recursive, character, token, markdown, python"
            )
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        return self._splitter.split_text(text)
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of Document chunks
        """
        return self._splitter.split_documents(documents)
    
    def split_document(self, document: Document) -> List[Document]:
        """
        Split a single document into chunks
        
        Args:
            document: Document object
            
        Returns:
            List of Document chunks
        """
        return self._splitter.split_documents([document])


class RecursiveSplitter(TextSplitterManager):
    """Convenience class for RecursiveCharacterTextSplitter"""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap, "recursive")


class CharacterSplitter(TextSplitterManager):
    """Convenience class for CharacterTextSplitter"""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap, "character")


class TokenSplitter(TextSplitterManager):
    """Convenience class for TokenTextSplitter"""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap, "token")


class MarkdownSplitter(TextSplitterManager):
    """Convenience class for MarkdownTextSplitter"""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap, "markdown")


class PythonCodeSplitter(TextSplitterManager):
    """Convenience class for PythonCodeTextSplitter"""
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap, "python")
