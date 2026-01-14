"""
Step 3: PDF Document Loaders for Ingestion and Parsing

This module provides PDF document loaders with support for various PDF formats.
"""
import os
import tempfile
from typing import List, Optional
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_core.documents import Document

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class PDFDocumentLoader:
    """
    Loader for PDF files
    
    Supports multiple PDF parsing libraries:
    - PyPDFLoader: Basic PDF parsing
    - PyMuPDFLoader: Advanced PDF parsing with better text extraction
    """
    
    def __init__(self, use_pymupdf: bool = True):
        """
        Initialize the PDF document loader
        
        Args:
            use_pymupdf: If True, use PyMuPDF (faster, better). If False, use PyPDF.
        """
        self.use_pymupdf = use_pymupdf
    
    def load_file(self, file_path: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Load a PDF file and return Document objects
        
        Args:
            file_path: Path to the PDF file
            metadata: Optional metadata to add to documents
            
        Returns:
            List of Document objects (one per page)
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        logger.debug(f"Loading PDF file | Path: {file_path} | Library: {'PyMuPDF' if self.use_pymupdf else 'PyPDF'}")
        
        try:
            if self.use_pymupdf:
                loader = PyMuPDFLoader(file_path)
            else:
                loader = PyPDFLoader(file_path)
            
            documents = loader.load()
            
            # Add custom metadata if provided
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)
                    doc.metadata["source_file"] = file_path
                    doc.metadata["file_type"] = "application/pdf"
            
            logger.info(f"PDF loaded successfully | File: {file_path} | Pages: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF | File: {file_path} | Error: {str(e)}", exc_info=True)
            raise
    
    def load_bytes(self, pdf_bytes: bytes, metadata: Optional[dict] = None) -> List[Document]:
        """
        Load PDF from bytes
        
        Args:
            pdf_bytes: PDF file content as bytes
            metadata: Optional metadata to add to documents
            
        Returns:
            List of Document objects (one per page)
        """
        import tempfile
        
        logger.debug(f"Loading PDF from bytes | Size: {len(pdf_bytes)} bytes")
        
        # Create a temporary file to save PDF bytes
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name
        
        try:
            # Load from temporary file
            documents = self.load_file(temp_path, metadata)
            logger.info(f"PDF loaded from bytes | Pages: {len(documents)}")
            return documents
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    
    def load_files(self, file_paths: List[str], metadata: Optional[dict] = None) -> List[Document]:
        """
        Load multiple PDF files
        
        Args:
            file_paths: List of PDF file paths
            metadata: Optional metadata to add to all documents
            
        Returns:
            List of Document objects from all files
        """
        all_documents = []
        for file_path in file_paths:
            documents = self.load_file(file_path, metadata)
            all_documents.extend(documents)
        return all_documents
    
    def get_page_count(self, file_path: str) -> int:
        """
        Get the number of pages in a PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Number of pages
        """
        documents = self.load_file(file_path)
        return len(documents)


# Convenience functions
def load_pdf_file(file_path: str, use_pymupdf: bool = True, metadata: Optional[dict] = None) -> List[Document]:
    """
    Convenience function to load a PDF file
    
    Args:
        file_path: Path to the PDF file
        use_pymupdf: Use PyMuPDF if True, PyPDF if False
        metadata: Optional metadata dictionary
        
    Returns:
        List of Document objects
    """
    loader = PDFDocumentLoader(use_pymupdf=use_pymupdf)
    return loader.load_file(file_path, metadata)


def load_pdf_bytes(pdf_bytes: bytes, use_pymupdf: bool = True, metadata: Optional[dict] = None) -> List[Document]:
    """
    Convenience function to load PDF from bytes
    
    Args:
        pdf_bytes: PDF file content as bytes
        use_pymupdf: Use PyMuPDF if True, PyPDF if False
        metadata: Optional metadata dictionary
        
    Returns:
        List of Document objects
    """
    loader = PDFDocumentLoader(use_pymupdf=use_pymupdf)
    return loader.load_bytes(pdf_bytes, metadata)
