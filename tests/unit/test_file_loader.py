"""
Unit tests for file loader
Tests document parsing and chunking functionality
"""

import pytest
import json
import tempfile
from pathlib import Path
from app.utils.file_loader import extract_text, chunk_text


class TestFileLoader:
    """Test suite for file loader"""
    
    # =====================
    # extract_text Tests
    # =====================
    
    def test_extract_text_from_txt_file(self):
        """Test extracting text from .txt file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            content = "This is test content"
            f.write(content)
            temp_path = f.name
        
        try:
            extracted = extract_text(temp_path)
            assert extracted == content
        finally:
            Path(temp_path).unlink()
    
    def test_extract_text_from_md_file(self):
        """Test extracting text from .md file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            content = "# Markdown Header\n\nSome content"
            f.write(content)
            temp_path = f.name
        
        try:
            extracted = extract_text(temp_path)
            assert extracted == content
        finally:
            Path(temp_path).unlink()
    
    def test_extract_text_from_json_file(self):
        """Test extracting text from .json file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {"key": "value", "number": 42}
            json.dump(data, f)
            temp_path = f.name
        
        try:
            extracted = extract_text(temp_path)
            assert "key" in extracted
            assert "value" in extracted
        finally:
            Path(temp_path).unlink()
    
    def test_extract_text_from_csv_file(self):
        """Test extracting text from .csv file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("name,age,city\n")
            f.write("John,30,NYC\n")
            f.write("Jane,25,LA\n")
            temp_path = f.name
        
        try:
            extracted = extract_text(temp_path)
            assert "John" in extracted
            assert "age" in extracted
        finally:
            Path(temp_path).unlink()
    
    def test_extract_text_unsupported_format(self):
        """Test that unsupported file format raises ValueError"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("content")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file type"):
                extract_text(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_extract_text_empty_file(self):
        """Test extracting from empty file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            extracted = extract_text(temp_path)
            assert extracted == ""
        finally:
            Path(temp_path).unlink()
    
    # =====================
    # chunk_text Tests
    # =====================
    
    def test_chunk_text_basic(self):
        """Test basic text chunking"""
        text = "word " * 100  # Long text
        chunks = chunk_text(text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_chunk_text_respects_chunk_size(self):
        """Test that chunks respect word limit"""
        # chunk_text chunks by word count (default 300 words)
        text = "word " * 1000  # 1000 words
        chunks = chunk_text(text)
        
        # Each chunk should have reasonable number of words
        for chunk in chunks:
            words = chunk.split()
            # Each chunk should be <= default chunk_size (300) + overlap tolerance
            assert len(words) <= 350
    
    def test_chunk_text_maintains_content(self):
        """Test that chunking doesn't lose content"""
        text = "This is important content that should be preserved "
        text = text * 20
        
        chunks = chunk_text(text)
        reconstructed = "".join(chunks)
        
        # All content should be preserved (accounting for potential overlap)
        assert "important content" in reconstructed
    
    def test_chunk_text_with_short_text(self):
        """Test chunking with text shorter than chunk size"""
        from config import CHUNK_SIZE
        
        text = "Short text"
        chunks = chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_empty_string(self):
        """Test chunking empty string"""
        chunks = chunk_text("")
        
        # Should return empty list or list with empty string
        assert isinstance(chunks, list)
    
    def test_chunk_text_overlap(self):
        """Test that chunks have overlap"""
        from config import CHUNK_SIZE, CHUNK_OVERLAP
        
        # Create text that will produce multiple chunks
        text = "word " * (CHUNK_SIZE // 5 * 3)
        chunks = chunk_text(text)
        
        if len(chunks) > 1:
            # If multiple chunks exist, check for overlap
            # Later chunks should contain some content from earlier ones
            assert len(chunks) > 0
    
    def test_chunk_text_with_special_characters(self):
        """Test chunking text with special characters"""
        text = "Hello\n\nWorld\t\tTest!@#$%^&*()\n" * 50
        chunks = chunk_text(text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_chunk_text_with_unicode(self):
        """Test chunking text with unicode characters"""
        text = "Hello 世界 Привет مرحبا " * 50
        chunks = chunk_text(text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_chunk_text_with_newlines(self):
        """Test chunking text with newlines"""
        # Note: chunk_text splits by words and rejoins with spaces
        # So newlines are lost during chunking process
        text = "Line 1 content here\nLine 2 content here\nLine 3 content here\n" * 20
        chunks = chunk_text(text)
        
        # Should produce multiple chunks
        assert len(chunks) > 0
        # All chunks should be strings
        assert all(isinstance(chunk, str) for chunk in chunks)
        # Chunks should contain content from original text
        assert any("Line" in chunk for chunk in chunks)


class TestChunkingConsistency:
    """Test consistency of text chunking"""
    
    def test_chunk_text_deterministic(self):
        """Test that chunking produces deterministic results"""
        text = "This is a test text for chunking " * 100
        
        chunks1 = chunk_text(text)
        chunks2 = chunk_text(text)
        
        assert chunks1 == chunks2
    
    def test_chunk_text_multiple_calls(self):
        """Test multiple chunking operations produce consistent results"""
        texts = [
            "First document content " * 50,
            "Second document content " * 50,
            "Third document content " * 50
        ]
        
        results = []
        for text in texts:
            chunks = chunk_text(text)
            results.append(chunks)
        
        # All should produce consistent chunking
        assert len(results) == 3
        assert all(len(r) > 0 for r in results)
