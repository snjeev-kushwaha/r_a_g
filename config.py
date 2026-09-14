"""
Configuration settings for the RAG system.
Strictly loaded from .env and environment variables.
No hardcoded fallback defaults for system configuration to prevent silent misconfigurations.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory of the Project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
ENV_FILE = BASE_DIR / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=True)
else:
    # If .env does not exist, system environment variables (Docker/K8s/CI) will be checked
    load_dotenv(override=True)


def _get_required_env(key: str) -> str:
    """
    Retrieve a required environment variable from the environment (.env).
    Raises ValueError if the variable is missing or empty.
    """
    val = os.getenv(key)
    if val is None or not val.strip():
        raise ValueError(
            f"Configuration Error: Missing required environment variable '{key}'. "
            f"Ensure it is defined in your .env file."
        )
    return val.strip()


def _get_required_int(key: str) -> int:
    """Retrieve an integer environment variable, raising ValueError if missing or invalid."""
    val = _get_required_env(key)
    try:
        return int(val)
    except ValueError:
        raise ValueError(
            f"Configuration Error: Environment variable '{key}' must be an integer, got '{val}'."
        )


def _get_required_float(key: str) -> float:
    """Retrieve a float environment variable, raising ValueError if missing or invalid."""
    val = _get_required_env(key)
    try:
        return float(val)
    except ValueError:
        raise ValueError(
            f"Configuration Error: Environment variable '{key}' must be a float, got '{val}'."
        )


def _resolve_env_path(key: str) -> str:
    """Resolve a required path relative to BASE_DIR if not already absolute."""
    val = _get_required_env(key)
    p = Path(val)
    if not p.is_absolute():
        return str((BASE_DIR / p).resolve())
    return str(p.resolve())


# Server Settings
API_HOST = _get_required_env("API_HOST")
API_PORT = _get_required_int("API_PORT")

# Base Storage Paths
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = _resolve_env_path("UPLOAD_DIR")
VECTOR_DB_PATH = _resolve_env_path("VECTOR_DB_PATH")

# Logging Settings
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = _resolve_env_path("LOG_FILE")
LOG_LEVEL = _get_required_env("LOG_LEVEL")

# Ollama Service Settings (Strictly loaded from .env)
OLLAMA_BASE_URL = _get_required_env("OLLAMA_BASE_URL")
OLLAMA_LLM_MODEL = _get_required_env("OLLAMA_LLM_MODEL")
OLLAMA_EMBED_MODEL = _get_required_env("OLLAMA_EMBED_MODEL")
OLLAMA_TIMEOUT = _get_required_float("OLLAMA_TIMEOUT")

# Backward-compatibility alias
OLLAMA_MODEL = OLLAMA_LLM_MODEL

# Text Chunking and Retrieval Settings
CHUNK_SIZE = _get_required_int("CHUNK_SIZE")
CHUNK_OVERLAP = _get_required_int("CHUNK_OVERLAP")
TOP_K = _get_required_int("TOP_K")

# Embedding and Context Limits
MAX_EMBED_CHARS = _get_required_int("MAX_EMBED_CHARS")
MAX_CONTEXT_CHARS = _get_required_int("MAX_CONTEXT_CHARS")

# Generation Parameters
LLM_NUM_PREDICT = _get_required_int("LLM_NUM_PREDICT")
LLM_TEMPERATURE = _get_required_float("LLM_TEMPERATURE")

# Vector DB Settings
VECTOR_DIM = _get_required_int("VECTOR_DIM")
