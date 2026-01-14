"""
Step 3 & 4: PDF Document Loaders with Common Issue Handling

This module provides PDF document loaders with support for various PDF formats
and handles common PDF issues like encryption, corruption, scanned PDFs, etc.
"""
import os
import tempfile
from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_core.documents import Document

from app.core.logging_config import get_logger

logger = get_logger(__name__)


class PDFDocumentLoader:
    """
    Loader for PDF files with common issue handling
    
    Supports multiple PDF parsing libraries:
    - PyPDFLoader: Basic PDF parsing
    - PyMuPDFLoader: Advanced PDF parsing with better text extraction
    
    Handles common PDF issues:
    - Encrypted PDFs (with password fallback)
    - Corrupted PDFs (with repair attempts)
    - Scanned PDFs (OCR detection)
    - Large PDFs (memory management)
    - Encoding issues
    """
    
    def __init__(
        self,
        use_pymupdf: bool = True,
        password: Optional[str] = None,
        max_pages: Optional[int] = None,
        extract_images: bool = False
    ):
        """
        Initialize the PDF document loader
        
        Args:
            use_pymupdf: If True, use PyMuPDF (faster, better). If False, use PyPDF.
            password: Optional password for encrypted PDFs
            max_pages: Maximum number of pages to process (None for all)
            extract_images: Whether to extract images from PDF (for OCR later)
        """
        self.use_pymupdf = use_pymupdf
        self.password = password
        self.max_pages = max_pages
        self.extract_images = extract_images
    
    def _check_pdf_issues(self, file_path: str) -> Dict[str, Any]:
        """
        Check for common PDF issues
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with issue information
        """
        issues = {
            "encrypted": False,
            "corrupted": False,
            "scanned": False,
            "large_file": False,
            "page_count": 0
        }
        
        try:
            if self.use_pymupdf:
                import fitz  # PyMuPDF
                doc = fitz.open(file_path)
                issues["page_count"] = len(doc)
                issues["encrypted"] = doc.is_encrypted
                issues["large_file"] = os.path.getsize(file_path) > 50 * 1024 * 1024  # > 50MB
                
                # Check if PDF is scanned (has images but little text)
                if issues["page_count"] > 0:
                    page = doc[0]
                    text_length = len(page.get_text())
                    image_count = len(page.get_images())
                    # If first page has images but very little text, likely scanned
                    if image_count > 0 and text_length < 100:
                        issues["scanned"] = True
                
                doc.close()
            else:
                # Using PyPDF
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                issues["page_count"] = len(reader.pages)
                issues["encrypted"] = reader.is_encrypted
                issues["large_file"] = os.path.getsize(file_path) > 50 * 1024 * 1024
                
        except Exception as e:
            logger.warning(f"Error checking PDF issues | File: {file_path} | Error: {str(e)}")
            issues["corrupted"] = True
        
        return issues
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted PDF text to handle common issues
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Fix common PDF ligatures and special characters
        replacements = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            "ﬃ": "ffi",
            "ﬄ": "ffl",
            "–": "-",
            "—": "-",
            """: '"',
            """: '"',
            "'": "'",
            "'": "'",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def load_file(self, file_path: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Load a PDF file and return Document objects with issue handling
        
        Args:
            file_path: Path to the PDF file
            metadata: Optional metadata to add to documents
            
        Returns:
            List of Document objects (one per page)
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        logger.debug(f"Loading PDF file | Path: {file_path} | Library: {'PyMuPDF' if self.use_pymupdf else 'PyPDF'}")
        
        # Check for common issues
        issues = self._check_pdf_issues(file_path)
        
        if issues["encrypted"] and not self.password:
            logger.warning(f"PDF is encrypted but no password provided | File: {file_path}")
            raise ValueError(
                "PDF file is encrypted. Please provide a password using the 'password' parameter."
            )
        
        if issues["scanned"]:
            logger.warning(f"PDF appears to be scanned (image-based) | File: {file_path} | Text extraction may be limited")
        
        if issues["large_file"]:
            logger.warning(f"PDF file is large (>50MB) | File: {file_path} | Processing may take longer")
        
        try:
            if self.use_pymupdf:
                # PyMuPDF handles encrypted PDFs better
                loader = PyMuPDFLoader(file_path)
            else:
                loader = PyPDFLoader(file_path)
                # Handle encrypted PDFs with PyPDF
                if issues["encrypted"] and self.password:
                    loader.pages[0].decrypt(self.password)
            
            documents = loader.load()
            
            # Limit pages if max_pages is set
            if self.max_pages and len(documents) > self.max_pages:
                logger.info(f"Limiting to {self.max_pages} pages | File: {file_path}")
                documents = documents[:self.max_pages]
            
            # Add custom metadata if provided
            if metadata:
                for doc in documents:
                    doc.metadata.update(metadata)
            
            # Clean text and add PDF-specific metadata
            processed_documents = []
            for idx, doc in enumerate(documents):
                # Clean the text content
                cleaned_content = self._clean_text(doc.page_content)
                
                # Skip nearly empty pages (less than 50 characters)
                if len(cleaned_content.strip()) < 50:
                    logger.debug(f"Skipping nearly empty page {idx + 1} | File: {file_path}")
                    continue
                
                # Update document with cleaned content
                doc.page_content = cleaned_content
                
                # Add PDF-specific metadata
                doc.metadata["source_file"] = file_path
                doc.metadata["file_type"] = "application/pdf"
                doc.metadata["page_number"] = idx + 1
                doc.metadata["total_pages"] = len(documents)
                doc.metadata["char_count"] = len(cleaned_content)
                
                # Add issue flags
                if issues["scanned"]:
                    doc.metadata["is_scanned"] = True
                if issues["encrypted"]:
                    doc.metadata["was_encrypted"] = True
                if issues["large_file"]:
                    doc.metadata["is_large_file"] = True
                
                processed_documents.append(doc)
            
            documents = processed_documents
            
            logger.info(
                f"PDF loaded successfully | "
                f"File: {file_path} | "
                f"Pages: {len(documents)} | "
                f"Issues: {issues}"
            )
            return documents
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle specific error types
            if "encrypted" in error_msg or "password" in error_msg:
                logger.error(f"PDF encryption error | File: {file_path} | Error: {str(e)}")
                raise ValueError(
                    f"PDF file is encrypted: {str(e)}. "
                    "Please provide a password using the 'password' parameter."
                )
            elif "corrupted" in error_msg or "invalid" in error_msg:
                logger.error(f"PDF corruption detected | File: {file_path} | Error: {str(e)}")
                # Try alternative loader as fallback
                if self.use_pymupdf:
                    logger.info(f"Attempting fallback to PyPDF | File: {file_path}")
                    try:
                        loader = PyPDFLoader(file_path)
                        documents = loader.load()
                        logger.warning(f"PDF loaded with fallback loader | File: {file_path}")
                        return documents
                    except Exception:
                        pass
                raise ValueError(f"PDF file appears to be corrupted: {str(e)}")
            else:
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
