"""
Smart PDF Processor - Combines PDF loading, cleaning, and chunking

This module provides a SmartPDFProcessor class that combines PDF loading,
text cleaning, and chunking in a single step, similar to the notebook example.
"""
from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.pdf_loaders import PDFDocumentLoader
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class SmartPDFProcessor:
    """
    Advanced PDF processing with error handling and smart chunking
    
    Combines PDF loading, text cleaning, and chunking in one step.
    Similar to the SmartPDFProcessor pattern from the notebook.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        use_pymupdf: bool = True,
        password: Optional[str] = None,
        max_pages: Optional[int] = None,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the Smart PDF Processor
        
        Args:
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            use_pymupdf: Use PyMuPDF if True, PyPDF if False
            password: Optional password for encrypted PDFs
            max_pages: Maximum pages to process (None for all)
            separators: Custom separators for text splitting (default: [" "])
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_pymupdf = use_pymupdf
        self.password = password
        self.max_pages = max_pages
        
        # Initialize text splitter with custom separators
        separators = separators or [" "]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
        
        # Initialize PDF loader
        self.pdf_loader = PDFDocumentLoader(
            use_pymupdf=use_pymupdf,
            password=password,
            max_pages=max_pages
        )
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted PDF text
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Fix common PDF ligatures
        replacements = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            "ﬃ": "ffi",
            "ﬄ": "ffl",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def process_pdf(self, pdf_path: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Process PDF with smart chunking and metadata enhancement
        
        This method combines PDF loading, text cleaning, and chunking in one step.
        
        Args:
            pdf_path: Path to the PDF file
            metadata: Optional metadata to add to all chunks
            
        Returns:
            List of Document objects (chunks) with enhanced metadata
        """
        logger.info(f"Processing PDF with SmartPDFProcessor | File: {pdf_path}")
        
        # Load PDF pages
        pages = self.pdf_loader.load_file(pdf_path, metadata)
        
        # Process each page
        processed_chunks = []
        
        for page_num, page in enumerate(pages):
            # Clean text (PDF loader already does this, but we ensure it's done)
            cleaned_text = self._clean_text(page.page_content)
            
            # Skip nearly empty pages
            if len(cleaned_text.strip()) < 50:
                logger.debug(f"Skipping nearly empty page {page_num + 1} | File: {pdf_path}")
                continue
            
            # Create chunks with enhanced metadata
            chunk_metadata = {
                **page.metadata,
                "page": page_num + 1,
                "total_pages": len(pages),
                "chunk_method": "smart_pdf_processor",
                "char_count": len(cleaned_text)
            }
            
            # Split page into chunks
            chunks = self.text_splitter.create_documents(
                texts=[cleaned_text],
                metadatas=[chunk_metadata]
            )
            
            processed_chunks.extend(chunks)
        
        logger.info(
            f"PDF processed successfully | "
            f"File: {pdf_path} | "
            f"Pages: {len(pages)} | "
            f"Chunks: {len(processed_chunks)}"
        )
        
        return processed_chunks
