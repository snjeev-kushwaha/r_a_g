"""
Tests configuration and shared fixtures for pytest
"""

import pytest
import tempfile
from pathlib import Path
from typing import Generator
import requests
import time


# ============================================
# API Configuration (Dynamically resolved from config / .env)
# ============================================

from config import API_HOST, API_PORT, OLLAMA_BASE_URL

_client_host = "localhost" if API_HOST in ("0.0.0.0", "") else API_HOST
API_URL = f"http://{_client_host}:{API_PORT}"
OLLAMA_URL = OLLAMA_BASE_URL


# ============================================
# Fixtures
# ============================================

@pytest.fixture(scope="session")
def api_url():
    """Return API base URL"""
    return API_URL


@pytest.fixture(scope="session")
def ollama_url():
    """Return Ollama base URL"""
    return OLLAMA_URL


@pytest.fixture(scope="session")
def check_api_running():
    """Check if API is running before tests"""
    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.get(f"{API_URL}/", timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            if i < max_retries - 1:
                time.sleep(2)
    
    pytest.skip("API server is not running. Start with: uvicorn main:app --reload")


@pytest.fixture(scope="session")
def check_ollama_running():
    """Check if Ollama is running before tests"""
    max_retries = 5
    for i in range(max_retries):
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            if i < max_retries - 1:
                time.sleep(2)
    
    pytest.skip("Ollama is not running. Start with: ollama serve")


@pytest.fixture
def temp_text_file() -> Generator[Path, None, None]:
    """Create a temporary text file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("""
        Machine Learning is a subset of Artificial Intelligence.
        It focuses on learning from data and making predictions.
        Deep Learning uses neural networks to process data.
        Natural Language Processing enables machines to understand language.
        Computer Vision is used for image recognition and analysis.
        """)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_json_file() -> Generator[Path, None, None]:
    """Create a temporary JSON file for testing"""
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_data = {
            "company": "TechCorp",
            "employees": 150,
            "founded": 2015,
            "locations": ["USA", "Europe", "Asia"],
            "description": "A leading AI and machine learning company specializing in RAG systems."
        }
        json.dump(test_data, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def sample_document_content() -> str:
    """Sample document content for testing"""
    return """
    QUARTERLY BUSINESS REPORT Q3 2024
    
    Executive Summary
    Our company achieved significant growth in Q3 2024 with revenue reaching $5.2 Million,
    representing a 25% increase from Q2. This growth was driven by strong product sales
    and expanding service offerings.
    
    Financial Overview
    Revenue: $5.2 Million
    - Product Sales: $3.1 Million (60%)
    - Service Revenue: $2.1 Million (40%)
    
    Expenses: $3.8 Million
    - Staff Salaries: $2.2 Million
    - Infrastructure: $1.0 Million
    - Marketing: $0.6 Million
    
    Net Profit: $1.4 Million (27% margin)
    
    Key Performance Indicators
    - Customer Growth: 25% increase YoY
    - Customer Retention Rate: 92%
    - Market Share: 5.2% (up from 4.1% in Q2)
    - Employee Satisfaction: 4.3/5.0
    
    Product Performance
    CloudSync Pro continues to be our flagship product with 12,000+ active users.
    New features launched: Real-time Collaboration, Advanced Analytics, Mobile App.
    
    Market Outlook
    The AI and ML market is expected to grow 45% annually.
    We are well-positioned to capture significant market share in 2024-2025.
    
    Recommendations
    1. Expand European operations
    2. Invest in R&D for next-generation products
    3. Hire 50 additional technical staff
    4. Launch marketing campaign in APAC region
    """


@pytest.fixture
def api_client():
    """Create a simple API client for testing"""
    class APIClient:
        def __init__(self, base_url: str = API_URL):
            self.base_url = base_url
        
        def health_check(self):
            """Check API health"""
            response = requests.get(f"{self.base_url}/", timeout=10)
            return response.status_code, response.json()
        
        def upload_document(self, file_path: Path):
            """Upload a document"""
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f)}
                response = requests.post(f"{self.base_url}/upload", files=files, timeout=30)
            return response.status_code, response.json()
        
        def ask_question(self, message: str):
            """Ask a question"""
            payload = {"message": message}
            response = requests.post(f"{self.base_url}/ask", json=payload, timeout=60)
            return response.status_code, response.json()
    
    return APIClient()


# ============================================
# Pytest Hooks
# ============================================

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Add markers based on file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
