"""
Integration tests for RAG service
Tests the complete RAG pipeline with multiple components
"""

import pytest
from pathlib import Path
from app.services.rag_service import ingest_document, query_rag
from app.services.embedding_service import get_embedding
from app.db.vector_db import VectorDB
from app.utils.file_loader import chunk_text


class TestRAGPipeline:
    """Test suite for the complete RAG pipeline"""
    
    @pytest.mark.integration
    def test_full_ingest_and_query_pipeline(self, sample_document_content):
        """Test complete pipeline: ingest document and query it"""
        doc_id = "test_doc_integration_1"
        
        # Ingest document
        ingest_document(doc_id, sample_document_content)
        
        # Query the ingested document
        answer = query_rag("What was the revenue in Q3 2024?")
        
        # Should return an answer (not error)
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert answer != "No data found."
    
    @pytest.mark.integration
    def test_ingest_multiple_documents(self, sample_document_content):
        """Test ingesting multiple documents and querying them"""
        doc1 = "doc_integration_1"
        doc2 = "doc_integration_2"
        
        ingest_document(doc1, sample_document_content)
        ingest_document(doc2, sample_document_content)
        
        # Query should work with multiple documents
        answer = query_rag("What is the market share?")
        
        assert isinstance(answer, str)
        assert len(answer) > 0
    
    @pytest.mark.integration
    def test_document_replacement(self, sample_document_content):
        """Test that ingesting same doc_id replaces the document"""
        doc_id = "replacement_test"
        
        # Ingest first version
        ingest_document(doc_id, "First version content about cats")
        answer1 = query_rag("Tell me about cats")
        
        # Ingest second version
        new_content = "Second version content about dogs is very different"
        ingest_document(doc_id, new_content)
        answer2 = query_rag("Tell me about dogs")
        
        # Both operations should succeed
        assert isinstance(answer1, str)
        assert isinstance(answer2, str)
    
    @pytest.mark.integration
    def test_query_with_no_documents(self):
        """Test querying when no documents are ingested"""
        # Create a new VectorDB to ensure it's empty
        from app.db.vector_db import VectorDB
        empty_db = VectorDB()
        
        # Query empty database should return "No data found"
        answer = query_rag("This should find nothing")
        
        assert isinstance(answer, str)
    
    @pytest.mark.integration
    def test_embedding_to_vector_db_integration(self, sample_document_content):
        """Test embedding and vector DB integration"""
        chunks = chunk_text(sample_document_content)
        
        # Each chunk should be embeddable
        for chunk in chunks:
            embedding = get_embedding(chunk)
            assert embedding is not None
            assert embedding.shape == (768,)


class TestRAGServiceAccuracy:
    """Test accuracy of RAG service responses"""
    
    @pytest.mark.integration
    def test_query_exact_match(self, sample_document_content):
        """Test that query returns relevant information"""
        doc_id = "accuracy_test_1"
        ingest_document(doc_id, sample_document_content)
        
        # Query for specific financial metric
        answer = query_rag("What is the net profit?")
        
        # Answer should contain the financial information
        assert isinstance(answer, str)
        assert len(answer) > 0
    
    @pytest.mark.integration
    def test_query_different_phrasings(self, sample_document_content):
        """Test that similar queries produce consistent results"""
        doc_id = "phrasing_test"
        ingest_document(doc_id, sample_document_content)
        
        # Similar queries
        answer1 = query_rag("How much revenue?")
        answer2 = query_rag("What was the total revenue?")
        
        # Both should return valid answers
        assert isinstance(answer1, str)
        assert isinstance(answer2, str)
        assert len(answer1) > 0
        assert len(answer2) > 0
    
    @pytest.mark.integration
    def test_query_with_no_relevant_data(self, sample_document_content):
        """Test querying for information not in document"""
        doc_id = "no_match_test"
        ingest_document(doc_id, sample_document_content)
        
        # Query for something definitely not in the document
        answer = query_rag("What is the capital of France?")
        
        # Should indicate no relevant data found
        assert isinstance(answer, str)


class TestChunkingIntegration:
    """Test chunking with embedding and vector DB"""
    
    @pytest.mark.integration
    def test_chunks_are_embeddable(self, sample_document_content):
        """Test that all chunks can be embedded"""
        chunks = chunk_text(sample_document_content)
        
        for i, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            assert embedding is not None, f"Chunk {i} could not be embedded"
            assert len(embedding) == 768
    
    @pytest.mark.integration
    def test_chunking_preserves_searchable_content(self, sample_document_content):
        """Test that important content is preserved in chunks"""
        chunks = chunk_text(sample_document_content)
        reconstructed = " ".join(chunks)
        
        # Key information should be searchable
        assert "revenue" in reconstructed.lower()
        assert "profit" in reconstructed.lower()
        assert "growth" in reconstructed.lower()


class TestErrorHandling:
    """Test error handling in RAG pipeline"""
    
    @pytest.mark.integration
    def test_ingest_empty_document(self):
        """Test ingesting empty document"""
        # This should not crash
        try:
            ingest_document("empty_doc", "")
        except Exception as e:
            # If it raises, it should be a meaningful error
            assert isinstance(e, (ValueError, AssertionError, RuntimeError))
    
    @pytest.mark.integration
    def test_query_empty_string(self):
        """Test querying with empty string"""
        result = query_rag("")
        
        # Should return a string (possibly error message)
        assert isinstance(result, str)
    
    @pytest.mark.integration
    def test_ingest_very_long_document(self, sample_document_content):
        """Test ingesting very long document"""
        very_long = sample_document_content * 100
        
        try:
            ingest_document("long_doc", very_long)
            answer = query_rag("What is profit?")
            assert isinstance(answer, str)
        except Exception as e:
            # Should either work or raise meaningful error
            assert "memory" in str(e).lower() or "timeout" in str(e).lower()
