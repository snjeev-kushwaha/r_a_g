# RAG System with Ollama

A production-ready **Retrieval-Augmented Generation (RAG)** system that combines document retrieval, semantic search, and LLM-powered answer generation using Ollama.

## 🎯 Features

- **📄 Multi-Format Document Support**: PDF, DOCX, CSV, Excel, JSON, TXT, PPTX
- **🔍 Semantic Search**: FAISS-based vector similarity search
- **🤖 LLM Integration**: Ollama for embeddings and answer generation
- **⚡ Batch Processing**: Efficient parallel embedding generation
- **🚀 FastAPI REST API**: Easy-to-use HTTP endpoints
- **📊 Comprehensive Testing**: Unit, integration, and E2E tests
- **🐳 Docker Ready**: Containerized Ollama backend

## 📋 Quick Start

### Prerequisites

- **Docker** with Ollama running (for LLM and embeddings)
- **Python 3.9+** with pip
- **2GB+ RAM** (recommended: 4GB)

### Installation

```bash
# Navigate to the project
cd d:\RAG\r_a_g

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import fastapi, ollama, faiss; print('✓ All dependencies installed')"
```

### Setup Ollama Docker Container

```bash
# Ensure Ollama is running
docker ps | grep ollama

# If not running, start it
docker run -d -p 11434:11434 --name ollama ollama/ollama:latest

# Pull required models (if not already pulled)
docker exec ollama ollama pull nomic-embed-text
docker exec ollama ollama pull llama3.2:3b

# Verify models
docker exec ollama ollama list
```

### Start the System

**Terminal 1 - FastAPI Server**:
```bash
cd d:\RAG\r_a_g
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Access the API:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/

## 📁 Project Structure

```
d:\RAG\r_a_g\
├── main.py                      # FastAPI application entry point
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Pytest configuration
│
├── api/                         # API layer
│   ├── routes.py               # Endpoint definitions
│   └── schemas.py              # Pydantic models
│
├── services/                   # Business logic
│   ├── embedding_service.py    # Ollama embedding client
│   ├── llm_service.py          # Ollama LLM client
│   └── rag_service.py          # RAG orchestration
│
├── db/                         # Data layer
│   └── vector_db.py            # FAISS vector store
│
├── utils/                      # Utilities
│   └── file_loader.py          # Document parsing & chunking
│
├── tests/                      # Test suite (ORGANIZED)
│   ├── conftest.py            # Pytest fixtures & configuration
│   ├── unit/                  # Unit tests
│   │   ├── test_embedding_service.py
│   │   ├── test_vector_db.py
│   │   └── test_file_loader.py
│   ├── integration/           # Integration tests
│   │   ├── test_rag_pipeline.py
│   │   └── test_api_endpoints.py
│   └── e2e/                   # End-to-end tests
│       └── test_complete_workflow.py
│
├── data/                       # Data storage
│   └── uploads/               # Uploaded files
│
└── docs/                       # Documentation
    ├── SETUP_GUIDE.md
    ├── DOCKER_SETUP_SUMMARY.md
    └── COMMANDS.sh
```

## 🧪 Testing

### Test Structure

The test suite is organized into three levels:

| Level | Purpose | Location | Run Time |
|-------|---------|----------|----------|
| **Unit Tests** | Test individual components | `tests/unit/` | < 1s per test |
| **Integration Tests** | Test component interactions | `tests/integration/` | 1-10s per test |
| **E2E Tests** | Test complete workflows | `tests/e2e/` | 10-60s per test |

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-timeout

# Run all tests
pytest

# Run only unit tests (fastest)
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run E2E tests (slowest, requires API running)
pytest tests/e2e/ -m e2e

# Run specific test file
pytest tests/unit/test_embedding_service.py

# Run specific test class
pytest tests/unit/test_embedding_service.py::TestEmbeddingService

# Run tests excluding slow tests
pytest -m "not slow"

# Run with coverage
pytest --cov=. --cov-report=html

# Run with verbose output
pytest -v

# Run tests in parallel
pip install pytest-xdist
pytest -n auto
```

### Test Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only E2E tests
pytest -m e2e

# Skip slow tests
pytest -m "not slow"
```

### Before Running Integration/E2E Tests

Ensure these are running:

```bash
# Check Ollama
docker exec ollama ollama ps

# Check API
curl http://localhost:8000/

# Start API if not running
uvicorn main:app --reload
```

## 🔌 API Endpoints

### 1. Health Check - Basic
```bash
GET /
```

Response: `{"status": "RAG API is running"}`

### 2. Health Check - Comprehensive
```bash
GET /health
```

Returns detailed status of all components:
```json
{
  "status": "healthy",
  "timestamp": "2024-09-14T10:30:45.123456",
  "components": {
    "api": "running",
    "ollama": "running",
    "vector_db": "running"
  },
  "details": {
    "vector_db_vectors": 150,
    "embedding_model": "working"
  }
}
```

**Status Values**: 
- `healthy` - All components working
- `degraded` - Some components have issues
- Component states: `running`, `unavailable`, `error`

### 3. Upload Document
```bash
POST /upload
Content-Type: multipart/form-data

file: <binary file>
```

Supported formats: PDF, DOCX, CSV, Excel, JSON, TXT, PPTX

### 4. Query Documents
```bash
POST /ask
Content-Type: application/json

{"message": "Your question here"}
```

## 💻 Usage Examples

### Python
```python
import requests

BASE_URL = "http://localhost:8000"

# Check health (comprehensive)
health = requests.get(f"{BASE_URL}/health").json()
print(f"API Status: {health['status']}")

# Upload document
with open("document.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post(f"{BASE_URL}/upload", files=files)
    print(response.json())

# Query documents
response = requests.post(
    f"{BASE_URL}/ask",
    json={"message": "What is the main topic?"}
)
print(response.json()["answer"])
```

### cURL
```bash
# Check health
curl -X GET "http://localhost:8000/health" | jq .

# Upload document
curl -X POST "http://localhost:8000/upload" -F "file=@document.pdf"

# Query documents
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the main topic?"}'
```

## ⚙️ Configuration

Edit `config.py`:

```python
VECTOR_DB_PATH = "./db/faiss_index"    # Database location
CHUNK_SIZE = 500                        # Text chunk size
CHUNK_OVERLAP = 50                      # Chunk overlap
TOP_K = 3                               # Retrieved chunks
OLLAMA_MODEL = "llama3.2:3b"           # LLM model
```

## 🐛 Troubleshooting

**Ollama not connecting?**
```bash
docker ps | grep ollama
docker exec ollama ollama list
```

**Model not found?**
```bash
docker exec ollama ollama pull nomic-embed-text
docker exec ollama ollama pull llama3.2:3b
```

**Slow responses?**
- Reduce CHUNK_SIZE in config.py
- Reduce TOP_K value
- Use lighter model (e.g., phi3)
- Check system resources

**Low quality answers?**
- Check source document quality
- Adjust CHUNK_SIZE
- Increase TOP_K
- Try different model

## 📚 Documentation

- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Detailed setup instructions
- [DOCKER_SETUP_SUMMARY.md](DOCKER_SETUP_SUMMARY.md) - Docker configuration
- [COMMANDS.sh](COMMANDS.sh) - Quick reference commands
- [API Documentation](http://localhost:8000/docs) - Interactive Swagger UI

## 📊 Architecture

```
FastAPI ↔ File Loader ↔ Embedding Service ↔ Vector DB (FAISS) ↔ Ollama
```

## 🚀 Performance

Typical latencies on standard hardware:

| Operation | Time |
|-----------|------|
| Embedding per chunk | 50-100ms |
| Vector search | <10ms |
| LLM response | 2-5s |
| Complete Q&A | 5-15s |

## 🔐 Security

- Input validation via Pydantic
- File type validation
- Local processing (no external APIs)
- Optional rate limiting with Redis
- Optional authentication

## 🤝 Contributing

1. Add tests in appropriate directory
2. Run `pytest` to verify
3. Ensure coverage is maintained

## 📄 License

Educational and commercial use permitted.

## ✅ Status

- **Testing**: Unit, Integration, E2E ✅
- **Documentation**: Complete ✅
- **Production Ready**: Yes ✅

---

**Last Updated**: 2024-09-14
