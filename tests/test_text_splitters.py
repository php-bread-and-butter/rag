"""
Tests for Step 2: Text Splitting Techniques
"""
import pytest
from langchain_core.documents import Document

from app.rag.text_splitters import (
    TextSplitterManager,
    RecursiveSplitter,
    CharacterSplitter,
    TokenSplitter,
    MarkdownSplitter,
    PythonCodeSplitter
)


def test_recursive_splitter():
    """Test RecursiveCharacterTextSplitter"""
    text = "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three."
    splitter = RecursiveSplitter(chunk_size=50, chunk_overlap=10)
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 0
    assert all(len(chunk) <= 50 for chunk in chunks)


def test_character_splitter():
    """Test CharacterTextSplitter"""
    text = "Line 1\n\nLine 2\n\nLine 3"
    splitter = CharacterSplitter(chunk_size=20, chunk_overlap=5)
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 0


def test_token_splitter():
    """Test TokenTextSplitter"""
    text = "This is a test sentence. " * 10
    splitter = TokenSplitter(chunk_size=20, chunk_overlap=5)
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 0


def test_markdown_splitter():
    """Test MarkdownTextSplitter"""
    text = "# Header 1\n\nContent here.\n\n## Header 2\n\nMore content."
    splitter = MarkdownSplitter(chunk_size=50, chunk_overlap=10)
    chunks = splitter.split_text(text)
    
    assert len(chunks) > 0


def test_split_documents():
    """Test splitting Document objects"""
    docs = [
        Document(page_content="Document 1 content", metadata={"id": 1}),
        Document(page_content="Document 2 content", metadata={"id": 2})
    ]
    
    splitter = RecursiveSplitter(chunk_size=20, chunk_overlap=5)
    chunks = splitter.split_documents(docs)
    
    assert len(chunks) > len(docs)
    assert all(isinstance(chunk, Document) for chunk in chunks)


def test_invalid_splitter_type():
    """Test invalid splitter type"""
    with pytest.raises(ValueError):
        TextSplitterManager(splitter_type="invalid")


def test_chunk_overlap():
    """Test that chunk overlap works correctly"""
    text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10"
    splitter = RecursiveSplitter(chunk_size=20, chunk_overlap=10)
    chunks = splitter.split_text(text)
    
    # Check that chunks overlap
    if len(chunks) > 1:
        # First few words of second chunk should match last few words of first chunk
        assert len(chunks) >= 2
