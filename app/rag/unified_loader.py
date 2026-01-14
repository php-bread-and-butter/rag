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
from app.rag.word_loaders import WordDocumentLoader
from app.rag.csv_excel_loaders import CSVExcelLoader
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
        '.docx': 'word',
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.xls': 'excel',
        # Future: .json, etc.
    }
    
    def __init__(self):
        self.text_loader = TextDocumentLoader()
        self.pdf_loader = PDFDocumentLoader()
        self.word_loader = WordDocumentLoader()
        self.csv_excel_loader = CSVExcelLoader()
    
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
                elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    return 'word'
                elif mime_type == 'text/csv' or mime_type == 'application/csv':
                    return 'csv'
                elif mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                   'application/vnd.ms-excel']:
                    return 'excel'
        
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
            # Try to detect from content (check magic bytes)
            if content.startswith(b'%PDF'):
                return 'pdf'
            elif content.startswith(b'PK'):  # ZIP-based formats (.docx, .xlsx, etc.)
                # Check ZIP structure to determine file type
                try:
                    import zipfile
                    import io
                    zip_file = zipfile.ZipFile(io.BytesIO(content))
                    file_list = zip_file.namelist()
                    
                    if 'word/document.xml' in file_list:
                        return 'word'
                    elif 'xl/workbook.xml' in file_list or 'xl/sharedStrings.xml' in file_list:
                        return 'excel'
                except Exception:
                    pass
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
        use_pymupdf: bool = True,
        password: Optional[str] = None,
        max_pages: Optional[int] = None
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
            self.pdf_loader.password = password
            self.pdf_loader.max_pages = max_pages
            return self.pdf_loader.load_file(file_path, metadata)
        elif file_type == 'word':
            return self.word_loader.load_file(file_path, metadata)
        elif file_type in ['csv', 'excel']:
            return self.csv_excel_loader.load_file(file_path, metadata)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def load_bytes(
        self,
        filename: str,
        content: bytes,
        metadata: Optional[dict] = None,
        use_pymupdf: bool = True,
        password: Optional[str] = None,
        max_pages: Optional[int] = None
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
            self.pdf_loader.password = password
            self.pdf_loader.max_pages = max_pages
            return self.pdf_loader.load_bytes(content, metadata)
        elif file_type == 'word':
            return self.word_loader.load_bytes(content, metadata)
        elif file_type in ['csv', 'excel']:
            return self.csv_excel_loader.load_bytes(filename, content, metadata)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def load_files(
        self,
        file_paths: List[str],
        metadata: Optional[dict] = None,
        use_pymupdf: bool = True,
        password: Optional[str] = None,
        max_pages: Optional[int] = None
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
            documents = self.load_file(
                file_path,
                metadata,
                use_pymupdf,
                password,
                max_pages
            )
            all_documents.extend(documents)
        return all_documents
    
    def get_supported_types(self) -> List[str]:
        """
        Get list of supported file types
        
        Returns:
            List of supported file extensions
        """
        return list(self.SUPPORTED_EXTENSIONS.keys())
    
    def get_supported_types_with_descriptions(self) -> dict:
        """
        Get supported file types with descriptions
        
        Returns:
            Dictionary mapping extensions to descriptions
        """
        return {
            '.txt': 'Plain text files',
            '.text': 'Plain text files',
            '.pdf': 'PDF documents',
            '.docx': 'Microsoft Word documents',
            '.csv': 'CSV (Comma-Separated Values) files',
            '.xlsx': 'Microsoft Excel files (Excel 2007+)',
            '.xls': 'Microsoft Excel files (Excel 97-2003)',
        }
