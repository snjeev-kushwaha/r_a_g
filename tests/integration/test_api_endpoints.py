"""
Integration tests for FastAPI endpoints
Tests the REST API functionality
"""

import pytest
import requests
from pathlib import Path


class TestAPIEndpoints:
    """Test suite for API endpoints"""
    
    @pytest.mark.integration
    def test_health_check_endpoint(self, api_url, check_api_running):
        """Test GET / health check endpoint"""
        response = requests.get(f"{api_url}/")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "RAG API is running"
    
    @pytest.mark.integration
    def test_upload_text_file(self, api_url, temp_text_file, check_api_running):
        """Test POST /upload with text file"""
        with open(temp_text_file, 'rb') as f:
            files = {'file': (temp_text_file.name, f)}
            response = requests.post(f"{api_url}/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "doc_id" in data
        assert "message" in data
        assert "indexed successfully" in data["message"].lower()
    
    @pytest.mark.integration
    def test_upload_json_file(self, api_url, temp_json_file, check_api_running):
        """Test POST /upload with JSON file"""
        with open(temp_json_file, 'rb') as f:
            files = {'file': (temp_json_file.name, f)}
            response = requests.post(f"{api_url}/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == temp_json_file.name
    
    @pytest.mark.integration
    def test_ask_question_endpoint(self, api_url, check_api_running):
        """Test POST /ask endpoint"""
        payload = {"message": "What is Machine Learning?"}
        response = requests.post(f"{api_url}/ask", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "question" in data
        assert "answer" in data
        assert data["question"] == payload["message"]
        assert isinstance(data["answer"], str)
    
    @pytest.mark.integration
    def test_ask_with_empty_message(self, api_url, check_api_running):
        """Test POST /ask with empty message"""
        payload = {"message": ""}
        response = requests.post(f"{api_url}/ask", json=payload)
        
        # Should handle gracefully (200 or 400)
        assert response.status_code in [200, 400, 422]
    
    @pytest.mark.integration
    def test_upload_without_file(self, api_url, check_api_running):
        """Test POST /upload without file should fail"""
        response = requests.post(f"{api_url}/upload")
        
        # Should return error
        assert response.status_code in [400, 422]


class TestAPIWorkflow:
    """Test end-to-end API workflow"""
    
    @pytest.mark.integration
    def test_upload_and_query_workflow(self, api_url, temp_text_file, check_api_running):
        """Test complete workflow: upload document and query it"""
        # Upload document
        with open(temp_text_file, 'rb') as f:
            files = {'file': (temp_text_file.name, f)}
            upload_response = requests.post(f"{api_url}/upload", files=files)
        
        assert upload_response.status_code == 200
        
        # Query the uploaded document
        query_payload = {"message": "What is Deep Learning?"}
        query_response = requests.post(f"{api_url}/ask", json=query_payload)
        
        assert query_response.status_code == 200
        data = query_response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_multiple_questions_after_upload(self, api_url, temp_text_file, check_api_running):
        """Test asking multiple questions after uploading a document"""
        # Upload document
        with open(temp_text_file, 'rb') as f:
            files = {'file': (temp_text_file.name, f)}
            requests.post(f"{api_url}/upload", files=files)
        
        # Ask multiple questions
        questions = [
            "What is Machine Learning?",
            "Tell me about Deep Learning",
            "What is NLP?",
            "Explain Computer Vision"
        ]
        
        responses = []
        for question in questions:
            payload = {"message": question}
            response = requests.post(f"{api_url}/ask", json=payload)
            assert response.status_code == 200
            responses.append(response.json())
        
        # All responses should be valid
        assert len(responses) == len(questions)
        assert all("answer" in r for r in responses)


class TestAPIErrorHandling:
    """Test API error handling"""
    
    @pytest.mark.integration
    def test_invalid_json_payload(self, api_url, check_api_running):
        """Test POST /ask with invalid JSON"""
        response = requests.post(
            f"{api_url}/ask",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        # Should return error
        assert response.status_code in [400, 422]
    
    @pytest.mark.integration
    def test_missing_required_field(self, api_url, check_api_running):
        """Test POST /ask without required message field"""
        payload = {"wrong_field": "value"}
        response = requests.post(f"{api_url}/ask", json=payload)
        
        # Should return validation error
        assert response.status_code in [400, 422]
    
    @pytest.mark.integration
    def test_request_timeout_handling(self, api_url, check_api_running):
        """Test that API handles timeouts gracefully"""
        # Very long message might cause timeout
        long_message = "test " * 1000
        payload = {"message": long_message}
        
        try:
            response = requests.post(f"{api_url}/ask", json=payload, timeout=120)
            # Should complete within timeout
            assert response.status_code in [200, 400, 422, 500]
        except requests.exceptions.Timeout:
            # Timeout is also acceptable for very long processing
            pass


class TestAPIResponseFormat:
    """Test API response format and structure"""
    
    @pytest.mark.integration
    def test_upload_response_format(self, api_url, temp_text_file, check_api_running):
        """Test that upload response has correct format"""
        with open(temp_text_file, 'rb') as f:
            files = {'file': (temp_text_file.name, f)}
            response = requests.post(f"{api_url}/upload", files=files)
        
        data = response.json()
        assert isinstance(data, dict)
        assert all(key in data for key in ["message", "doc_id"])
        assert isinstance(data["message"], str)
        assert isinstance(data["doc_id"], str)
    
    @pytest.mark.integration
    def test_ask_response_format(self, api_url, check_api_running):
        """Test that ask response has correct format"""
        payload = {"message": "Test question"}
        response = requests.post(f"{api_url}/ask", json=payload)
        
        data = response.json()
        assert isinstance(data, dict)
        assert all(key in data for key in ["question", "answer"])
        assert isinstance(data["question"], str)
        assert isinstance(data["answer"], str)
    
    @pytest.mark.integration
    def test_response_headers(self, api_url, check_api_running):
        """Test that API returns correct content type"""
        response = requests.get(f"{api_url}/")
        
        assert "content-type" in response.headers
        assert "application/json" in response.headers["content-type"]

    @pytest.mark.integration
    def test_bulk_upload_endpoint(self, api_url, temp_text_file, temp_json_file, check_api_running):
        """Test POST /upload/bulk with multiple files"""
        with open(temp_text_file, 'rb') as f1, open(temp_json_file, 'rb') as f2:
            files = [
                ('files', (temp_text_file.name, f1, 'text/plain')),
                ('files', (temp_json_file.name, f2, 'application/json')),
            ]
            response = requests.post(f"{api_url}/upload/bulk", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 2
        assert data["successful_uploads"] == 2
        assert data["failed_uploads"] == 0
        assert len(data["files"]) == 2
        assert all(f["status"] == "indexed" for f in data["files"])
