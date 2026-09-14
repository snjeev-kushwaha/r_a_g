"""
Unit tests for vector database
Tests the FAISS vector store functionality
"""

import pytest
import numpy as np
from app.db.vector_db import VectorDB


class TestVectorDB:
    """Test suite for vector database"""
    
    @pytest.fixture
    def vector_db(self):
        """Create a fresh VectorDB instance for each test"""
        return VectorDB(dim=768)
    
    def test_initialization(self, vector_db):
        """Test VectorDB initialization"""
        assert vector_db.dim == 768
        assert vector_db.id_counter == 0
        assert vector_db.doc_index_map == {}
        assert vector_db.text_store == {}
    
    def test_add_single_document(self, vector_db):
        """Test adding a single document"""
        vectors = np.random.rand(5, 768).astype("float32")
        texts = [f"Text chunk {i}" for i in range(5)]
        
        vector_db.add_document("doc1", vectors, texts)
        
        assert "doc1" in vector_db.doc_index_map
        assert len(vector_db.doc_index_map["doc1"]) == 5
        assert vector_db.id_counter == 5
    
    def test_add_multiple_documents(self, vector_db):
        """Test adding multiple documents"""
        for doc_id in range(3):
            vectors = np.random.rand(3, 768).astype("float32")
            texts = [f"Doc{doc_id} Text {i}" for i in range(3)]
            vector_db.add_document(f"doc{doc_id}", vectors, texts)
        
        assert len(vector_db.doc_index_map) == 3
        assert vector_db.id_counter == 9
    
    def test_delete_document(self, vector_db):
        """Test deleting a document"""
        vectors = np.random.rand(5, 768).astype("float32")
        texts = [f"Text {i}" for i in range(5)]
        
        vector_db.add_document("doc1", vectors, texts)
        assert "doc1" in vector_db.doc_index_map
        
        vector_db.delete_document("doc1")
        assert "doc1" not in vector_db.doc_index_map
    
    def test_delete_nonexistent_document(self, vector_db):
        """Test deleting a non-existent document doesn't raise error"""
        # Should not raise any error
        vector_db.delete_document("nonexistent")
    
    def test_search_returns_results(self, vector_db):
        """Test that search returns correct results"""
        # Add test documents
        vectors = np.array([
            [1.0, 0.0, 0.0] + [0.0] * 765,
            [0.0, 1.0, 0.0] + [0.0] * 765,
            [0.0, 0.0, 1.0] + [0.0] * 765
        ], dtype="float32")
        
        texts = ["Vector A", "Vector B", "Vector C"]
        
        # Create proper 768-dim vectors
        vectors = np.random.rand(3, 768).astype("float32")
        
        vector_db.add_document("doc1", vectors, texts)
        
        # Query with a similar vector
        query_vector = vectors[0:1]
        results = vector_db.search(query_vector, top_k=2)
        
        assert len(results) <= 2
        assert results[0] is not None
    
    def test_search_with_empty_db(self, vector_db):
        """Test search on empty database"""
        query_vector = np.random.rand(1, 768).astype("float32")
        results = vector_db.search(query_vector, top_k=3)
        
        assert results == []
    
    def test_search_top_k_parameter(self, vector_db):
        """Test that top_k parameter works correctly"""
        vectors = np.random.rand(10, 768).astype("float32")
        texts = [f"Text {i}" for i in range(10)]
        
        vector_db.add_document("doc1", vectors, texts)
        
        query_vector = vectors[0:1]
        results = vector_db.search(query_vector, top_k=3)
        
        assert len(results) <= 3
    
    def test_document_replacement(self, vector_db):
        """Test that adding document with same ID replaces it"""
        # Add first version
        vectors1 = np.random.rand(2, 768).astype("float32")
        texts1 = ["Old text 1", "Old text 2"]
        vector_db.add_document("doc1", vectors1, texts1)
        
        first_count = vector_db.id_counter
        
        # Add second version with same doc_id
        vectors2 = np.random.rand(3, 768).astype("float32")
        texts2 = ["New text 1", "New text 2", "New text 3"]
        vector_db.add_document("doc1", vectors2, texts2)
        
        # Should have 5 vectors total (2 from first + 3 from second)
        assert vector_db.id_counter == 5
    
    def test_vector_dimension_validation(self, vector_db):
        """Test that vector dimensions are validated"""
        # Wrong dimension vectors should raise assertion
        wrong_vectors = np.random.rand(5, 512).astype("float32")  # Wrong dim
        texts = [f"Text {i}" for i in range(5)]
        
        with pytest.raises(AssertionError):
            vector_db.add_document("doc1", wrong_vectors, texts)
    
    def test_1d_vector_reshaping(self, vector_db):
        """Test that 1D vectors are properly reshaped"""
        # Create a 1D vector
        vector = np.random.rand(768).astype("float32")
        texts = ["Single text"]
        
        vector_db.add_document("doc1", vector, texts)
        
        assert len(vector_db.doc_index_map["doc1"]) == 1

    def test_persistence(self, tmp_path):
        """Test saving and loading vector DB from disk"""
        storage_dir = str(tmp_path / "faiss_index")
        db1 = VectorDB(dim=768, storage_dir=storage_dir)

        vectors = np.random.rand(3, 768).astype("float32")
        texts = ["Chunk 1", "Chunk 2", "Chunk 3"]
        db1.add_document("doc_persist", vectors, texts)

        # Create new instance pointing to same storage
        db2 = VectorDB(dim=768, storage_dir=storage_dir)
        assert db2.index.ntotal == 3
        assert "doc_persist" in db2.doc_index_map
        assert len(db2.text_store) == 3

        # Search should work in reloaded instance
        results = db2.search(vectors[0:1], top_k=1)
        assert len(results) == 1
        assert results[0] == "Chunk 1"
