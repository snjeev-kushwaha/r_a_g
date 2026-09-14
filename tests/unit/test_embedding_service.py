"""
Unit tests for embedding service
Tests the embedding generation functionality
"""

import pytest
import numpy as np
from app.services.embedding_service import get_embedding, get_embeddings_batch


class TestEmbeddingService:
    """Test suite for embedding service"""
    
    def test_get_embedding_returns_vector(self):
        """Test that get_embedding returns a valid vector"""
        text = "This is a test sentence for embedding."
        embedding = get_embedding(text)
        
        assert embedding is not None
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)  # nomic-embed-text uses 768 dimensions
    
    def test_get_embedding_with_empty_string(self):
        """Test that get_embedding returns None for empty string"""
        embedding = get_embedding("")
        assert embedding is None
    
    def test_get_embedding_with_whitespace_only(self):
        """Test that get_embedding returns None for whitespace-only string"""
        embedding = get_embedding("   \n\t  ")
        assert embedding is None
    
    def test_get_embedding_with_long_text(self):
        """Test that get_embedding handles text truncation"""
        long_text = "word " * 2000  # Create text longer than MAX_CHARS
        embedding = get_embedding(long_text)
        
        assert embedding is not None
        assert embedding.shape == (768,)
    
    def test_get_embedding_consistency(self):
        """Test that same text produces similar embeddings"""
        text = "Machine Learning is a subset of AI"
        embedding1 = get_embedding(text)
        embedding2 = get_embedding(text)
        
        # Check that embeddings are identical (deterministic)
        assert np.allclose(embedding1, embedding2, rtol=1e-5)
    
    def test_get_embeddings_batch_returns_list(self):
        """Test that get_embeddings_batch returns a list of vectors"""
        texts = [
            "First text about machine learning",
            "Second text about deep learning",
            "Third text about neural networks"
        ]
        embeddings = get_embeddings_batch(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert all(isinstance(e, np.ndarray) for e in embeddings)
        assert all(e.shape == (768,) for e in embeddings)
    
    def test_get_embeddings_batch_with_empty_list(self):
        """Test get_embeddings_batch with empty list"""
        embeddings = get_embeddings_batch([])
        assert embeddings == []
    
    def test_get_embeddings_batch_filters_empty_strings(self):
        """Test that get_embeddings_batch filters out empty strings"""
        texts = [
            "Valid text",
            "",
            "   ",
            "Another valid text"
        ]
        embeddings = get_embeddings_batch(texts)
        
        # Should only have 2 embeddings (empty strings filtered)
        assert len(embeddings) == 2
    
    def test_embedding_dimensions(self):
        """Test that embedding has correct dimensions"""
        text = "Test embedding dimensions"
        embedding = get_embedding(text)
        
        assert embedding is not None
        assert len(embedding) == 768
        assert embedding.dtype == np.float32
    
    def test_get_embeddings_batch_performance(self):
        """Test that batch processing works with multiple texts"""
        texts = [f"Text number {i} about topic {i}" for i in range(10)]
        embeddings = get_embeddings_batch(texts)
        
        assert len(embeddings) == 10
        assert all(isinstance(e, np.ndarray) for e in embeddings)
