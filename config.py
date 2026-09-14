"""
Configuration settings for the RAG system.
Supports environment variables with sensible production defaults.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(DATA_DIR / "uploads"))
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(BASE_DIR / "db" / "faiss_index"))

# Logging Settings
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = os.getenv("LOG_FILE", str(LOG_DIR / "app.log"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Ollama Service Settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))

# Backward-compatibility alias
OLLAMA_MODEL = OLLAMA_LLM_MODEL

# Text Chunking and Retrieval Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "3"))

# Embedding and Context Limits
MAX_EMBED_CHARS = int(os.getenv("MAX_EMBED_CHARS", "4000"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))

# Generation Parameters
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "500"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Vector DB Settings
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "768"))
