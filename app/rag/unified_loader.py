"""
Unified Document Loader

Automatically detects file type and uses the appropriate loader.
"""
from typing import List, Optional
from pathlib import Path
import mimetypes
from langchain_core.documents import Document

from app.rag.document_loaders import TextDocumentLoader
from app.rag.pdf_loaders import PDFDocumentLoader
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class UnifiedDocumentLoader:
    """
    Unified loader that automatically detects file type and uses appropriate loader
    """
    
    SUPPORTED_EXTENSIONS = {
        '.txt': 'text',
        '.text': 'text',
        '.pdf': 'pdf',
        # Future: .docx, .csv, .xlsx, .json, etc.
    }
    
    def __init__(self):
        self.text_loader = TextDocumentLoader()
        self.pdf_loader = PDFDocumentLoader()
    
    def detect_file_type(self, file_path: str) -> str:
        """
        Detect file type from extension
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type string (text, pdf, etc.)
        """
        extension = Path(file_path).suffix.lower()
        file_type = self.SUPPORTED_EXTENSIONS.get(extension)
        
        if not file_type:
            # Try MIME type detection
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type:
                if mime_type.startswith('text/'):
                    return 'text'
                elif mime_type == 'application/pdf':
                    return 'pdf'
        
        if not file_type:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        return file_type
    
    def detect_file_type_from_bytes(self, filename: str, content: bytes) -> str:
        """
        Detect file type from filename and content
        
        Args:
            filename: Name of the file
            content: File content as bytes
            
        Returns:
            File type string
        """
        extension = Path(filename).suffix.lower()
        file_type = self.SUPPORTED_EXTENSIONS.get(extension)
        
        if not file_type:
            # Try MIME type detection
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type:
                if mime_type.startswith('text/'):
                    return 'text'
                elif mime_type == 'application/pdf':
                    return 'pdf'
        
        if not file_type:
            # Try to detect from content (check PDF magic bytes)
            if content.startswith(b'%PDF'):
                return 'pdf'
            # Try to decode as text
            try:
                content.decode('utf-8')
                return 'text'
            except UnicodeDecodeError:
                pass
        
        if not file_type:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        return file_type
    
    def load_file(
        self,
        file_path: str,
        metadata: Optional[dict] = None,
        use_pymupdf: bool = True
    ) -> List[Document]:
        """
        Load a file using the appropriate loader based on file type
        
        Args:
            file_path: Path to the file
            metadata: Optional metadata to add
            use_pymupdf: For PDFs, use PyMuPDF if True
            
        Returns:
            List of Document objects
        """
        file_type = self.detect_file_type(file_path)
        logger.info(f"Loading file | Path: {file_path} | Type: {file_type}")
        
        if file_type == 'text':
            return self.text_loader.load_file(file_path, metadata)
        elif file_type == 'pdf':
            self.pdf_loader.use_pymupdf = use_pymupdf
            return self.pdf_loader.load_file(file_path, metadata)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def load_bytes(
        self,
        filename: str,
        content: bytes,
        metadata: Optional[dict] = None,
        use_pymupdf: bool = True
    ) -> List[Document]:
        """
        Load a file from bytes using the appropriate loader
        
        Args:
            filename: Name of the file (for type detection)
            content: File content as bytes
            metadata: Optional metadata to add
            use_pymupdf: For PDFs, use PyMuPDF if True
            
        Returns:
            List of Document objects
        """
        file_type = self.detect_file_type_from_bytes(filename, content)
        logger.info(f"Loading file from bytes | Filename: {filename} | Type: {file_type} | Size: {len(content)} bytes")
        
        if file_type == 'text':
            text_content = content.decode('utf-8')
            doc_metadata = metadata or {}
            doc_metadata["source"] = filename
            return [self.text_loader.load_text(text_content, doc_metadata)]
        elif file_type == 'pdf':
            self.pdf_loader.use_pymupdf = use_pymupdf
            return self.pdf_loader.load_bytes(content, metadata)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def load_files(
        self,
        file_paths: List[str],
        metadata: Optional[dict] = None,
        use_pymupdf: bool = True
    ) -> List[Document]:
        """
        Load multiple files using appropriate loaders
        
        Args:
            file_paths: List of file paths
            metadata: Optional metadata to add
            use_pymupdf: For PDFs, use PyMuPDF if True
            
        Returns:
            List of Document objects from all files
        """
        all_documents = []
        for file_path in file_paths:
            documents = self.load_file(file_path, metadata, use_pymupdf)
            all_documents.extend(documents)
        return all_documents
    
    def get_supported_types(self) -> List[str]:
        """
        Get list of supported file types
        
        Returns:
            List of supported file extensions
        """
        return list(self.SUPPORTED_EXTENSIONS.keys())
