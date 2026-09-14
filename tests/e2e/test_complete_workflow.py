"""
End-to-End tests for complete RAG system
Tests real-world scenarios and workflows
"""

import pytest
import tempfile
from pathlib import Path


class TestE2ECompleteWorkflow:
    """Test complete end-to-end workflows"""
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_complete_rag_workflow_with_business_document(self, api_url, check_api_running, check_ollama_running):
        """Test complete workflow with business document"""
        import requests
        
        # Create a business document
        business_doc = """
        ANNUAL REPORT 2024
        
        Company Performance Overview
        
        Revenue Breakdown:
        - Product Sales: $8.5 Million (65%)
        - Service Revenue: $4.2 Million (32%)
        - Other: $0.3 Million (3%)
        Total Revenue: $13 Million
        
        Expenses:
        - Staff: $5.2 Million
        - Infrastructure: $2.1 Million
        - Marketing: $1.5 Million
        - Operations: $1.2 Million
        Total Expenses: $10 Million
        
        Net Profit: $3 Million
        
        Key Metrics:
        - Customer Retention: 94%
        - Market Growth: 35% YoY
        - Employee Count: 250
        - Office Locations: 5
        """
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(business_doc)
            temp_file = Path(f.name)
        
        try:
            # Step 1: Upload document
            print("\n[E2E Test] Step 1: Uploading business document...")
            with open(temp_file, 'rb') as f:
                files = {'file': (temp_file.name, f)}
                upload_response = requests.post(f"{api_url}/upload", files=files, timeout=30)
            
            assert upload_response.status_code == 200, f"Upload failed: {upload_response.text}"
            upload_data = upload_response.json()
            print(f"✓ Document uploaded: {upload_data['doc_id']}")
            
            # Step 2: Ask multiple business questions
            print("[E2E Test] Step 2: Asking business questions...")
            questions = [
                "What is the total revenue?",
                "How much were the staff expenses?",
                "What is the customer retention rate?",
                "Which revenue stream is the largest?"
            ]
            
            for question in questions:
                payload = {"message": question}
                query_response = requests.post(f"{api_url}/ask", json=payload, timeout=60)
                
                assert query_response.status_code == 200, f"Query failed: {query_response.text}"
                answer_data = query_response.json()
                
                print(f"Q: {question}")
                print(f"A: {answer_data['answer'][:100]}...")
                
                assert "answer" in answer_data
                assert len(answer_data["answer"]) > 0
        
        finally:
            temp_file.unlink()
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_complete_rag_workflow_with_multiple_documents(self, api_url, check_api_running, check_ollama_running):
        """Test workflow with multiple documents"""
        import requests
        
        # Document 1: Technology
        tech_doc = """
        TECHNOLOGY TRENDS 2024
        
        Artificial Intelligence: The future of technology
        - Machine Learning adoption increased by 45%
        - Large Language Models revolutionizing NLP
        - Deep Learning frameworks becoming mainstream
        
        Cloud Computing:
        - AWS leads with 32% market share
        - Multi-cloud strategies becoming common
        - Edge computing gaining traction
        
        Cybersecurity:
        - Zero-trust architecture widely adopted
        - Quantum computing threats emerging
        - Encryption standards evolving
        """
        
        # Document 2: Finance
        finance_doc = """
        FINANCIAL MARKETS REPORT
        
        Stock Market Performance:
        - Tech Index up 28%
        - Financial Sector up 12%
        - Healthcare Sector up 18%
        
        Interest Rates:
        - Federal Reserve rate: 5.25-5.50%
        - Expected cuts in Q2 2024
        - Impact on mortgage market
        
        Investment Trends:
        - ESG investing growing rapidly
        - AI-focused funds seeing inflows
        - Real estate sector recovering
        """
        
        temp_files = []
        try:
            # Create and upload both documents
            for i, (doc_content, suffix) in enumerate([
                (tech_doc, "_tech"),
                (finance_doc, "_finance")
            ]):
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(doc_content)
                    temp_file = Path(f.name)
                    temp_files.append(temp_file)
                
                # Upload
                with open(temp_file, 'rb') as f:
                    files = {'file': (temp_file.name, f)}
                    response = requests.post(f"{api_url}/upload", files=files, timeout=30)
                
                assert response.status_code == 200
                print(f"✓ Uploaded document {i+1}")
            
            # Ask questions that test both documents
            questions = [
                "What is the impact of AI on technology?",
                "What are the current interest rates?",
                "Which sector has the best performance?",
                "What is happening in cloud computing?"
            ]
            
            for question in questions:
                payload = {"message": question}
                response = requests.post(f"{api_url}/ask", json=payload, timeout=60)
                
                assert response.status_code == 200
                print(f"✓ Query answered: {question}")
        
        finally:
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_document_update_workflow(self, api_url, check_api_running, check_ollama_running):
        """Test updating documents (re-ingesting with same ID)"""
        import requests
        
        doc_id_name = "updatable_doc.txt"
        
        # Version 1: About Cats
        version1 = """
        CATS
        
        Cats are domesticated animals known for their independence.
        They have excellent night vision and are natural hunters.
        Cats communicate through meows, purrs, and body language.
        Popular cat breeds include Persian, Siamese, and Maine Coon.
        """
        
        # Version 2: About Dogs
        version2 = """
        DOGS
        
        Dogs are loyal companions known for their pack mentality.
        They have excellent sense of smell used for tracking.
        Dogs communicate through barks, whines, and tail wagging.
        Popular dog breeds include Labrador, German Shepherd, and Golden Retriever.
        """
        
        try:
            # Upload Version 1
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(version1)
                temp_file = Path(f.name)
            
            with open(temp_file, 'rb') as f:
                files = {'file': (doc_id_name, f)}
                response = requests.post(f"{api_url}/upload", files=files)
            assert response.status_code == 200
            temp_file.unlink()
            
            print("✓ Uploaded version 1 (Cats)")
            
            # Query Version 1
            payload = {"message": "Tell me about animals"}
            response = requests.post(f"{api_url}/ask", json=payload, timeout=60)
            assert response.status_code == 200
            print("✓ Queried version 1")
            
            # Upload Version 2 (replaces Version 1)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(version2)
                temp_file = Path(f.name)
            
            with open(temp_file, 'rb') as f:
                files = {'file': (doc_id_name, f)}
                response = requests.post(f"{api_url}/upload", files=files)
            assert response.status_code == 200
            temp_file.unlink()
            
            print("✓ Uploaded version 2 (Dogs)")
            
            # Query Version 2
            payload = {"message": "Tell me about animals again"}
            response = requests.post(f"{api_url}/ask", json=payload, timeout=60)
            assert response.status_code == 200
            print("✓ Queried version 2")
        
        except Exception as e:
            pytest.fail(f"Document update workflow failed: {str(e)}")


class TestE2ERobustness:
    """Test system robustness in E2E scenarios"""
    
    @pytest.mark.e2e
    def test_concurrent_queries_after_upload(self, api_url, check_api_running, check_ollama_running):
        """Test system can handle queries after upload"""
        import requests
        import threading
        
        doc_content = "Python is a popular programming language. Java is used for enterprise applications."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(doc_content)
            temp_file = Path(f.name)
        
        try:
            # Upload
            with open(temp_file, 'rb') as f:
                files = {'file': (temp_file.name, f)}
                requests.post(f"{api_url}/upload", files=files)
            
            # Query in sequence (not truly concurrent to avoid rate limiting)
            results = []
            
            def run_query(question):
                response = requests.post(
                    f"{api_url}/ask",
                    json={"message": question},
                    timeout=60
                )
                results.append(response.status_code)
            
            questions = [
                "What is Python?",
                "Tell me about Java"
            ]
            
            for question in questions:
                run_query(question)
            
            # All queries should succeed
            assert all(status == 200 for status in results)
            print("✓ All queries processed successfully")
        
        finally:
            temp_file.unlink()
    
    @pytest.mark.e2e
    def test_large_document_handling(self, api_url, check_api_running, check_ollama_running):
        """Test system can handle large documents"""
        import requests
        
        # Create a large document
        large_doc = "This is sample content. " * 500  # ~12KB
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(large_doc)
            temp_file = Path(f.name)
        
        try:
            # Upload large document
            with open(temp_file, 'rb') as f:
                files = {'file': (temp_file.name, f)}
                response = requests.post(f"{api_url}/upload", files=files, timeout=60)
            
            assert response.status_code == 200
            print("✓ Large document uploaded successfully")
            
            # Query the large document
            response = requests.post(
                f"{api_url}/ask",
                json={"message": "What is the content about?"},
                timeout=120
            )
            
            assert response.status_code == 200
            print("✓ Query on large document successful")
        
        finally:
            temp_file.unlink()


class TestE2EErrorRecovery:
    """Test error handling and recovery in E2E scenarios"""
    
    @pytest.mark.e2e
    def test_recovery_after_failed_query(self, api_url, check_api_running, check_ollama_running):
        """Test system recovers from failed queries"""
        import requests
        
        doc = "Test document content for recovery testing"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(doc)
            temp_file = Path(f.name)
        
        try:
            # Upload
            with open(temp_file, 'rb') as f:
                files = {'file': (temp_file.name, f)}
                requests.post(f"{api_url}/upload", files=files)
            
            # Attempt query with invalid data
            response1 = requests.post(f"{api_url}/ask", json={"invalid": "data"})
            # This should fail with validation error
            
            # System should recover and next query should work
            response2 = requests.post(
                f"{api_url}/ask",
                json={"message": "Valid query"},
                timeout=60
            )
            
            assert response2.status_code == 200
            print("✓ System recovered from error")
        
        finally:
            temp_file.unlink()
