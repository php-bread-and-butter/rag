"""
Tests for Step 1: Document Loaders
"""
import pytest
from pathlib import Path
import tempfile
import os

from app.rag.document_loaders import TextDocumentLoader, load_text_file, load_text


def test_load_text():
    """Test loading raw text"""
    text = "This is a test document."
    doc = load_text(text, {"source": "test"})
    
    assert doc.page_content == text
    assert doc.metadata["source"] == "test"


def test_load_text_file():
    """Test loading a text file"""
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is test content.\nLine 2.")
        temp_path = f.name
    
    try:
        docs = load_text_file(temp_path)
        assert len(docs) == 1
        assert "test content" in docs[0].page_content
    finally:
        os.unlink(temp_path)


def test_load_text_file_not_found():
    """Test loading non-existent file"""
    loader = TextDocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_file("/nonexistent/file.txt")


def test_load_multiple_files():
    """Test loading multiple files"""
    temp_files = []
    try:
        # Create multiple temp files
        for i in range(3):
            f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            f.write(f"Content {i}")
            temp_files.append(f.name)
            f.close()
        
        loader = TextDocumentLoader()
        docs = loader.load_files(temp_files)
        
        assert len(docs) == 3
    finally:
        for f in temp_files:
            os.unlink(f)
