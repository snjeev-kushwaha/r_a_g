"""
Ollama embedding service.
Handles generating vector embeddings for text chunks and queries.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import numpy as np
import ollama

from app.core.logging_config import get_logger
from config import (
    MAX_EMBED_CHARS,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
)

logger = get_logger(__name__)

# Initialize client with configured host
_client = ollama.Client(host=OLLAMA_BASE_URL)


def get_embedding(text: str) -> Optional[np.ndarray]:
    """
    Generate vector embedding for a single text string using Ollama.

    Args:
        text: Input text string to embed.

    Returns:
        1D numpy array with float32 dtype (e.g. 768-dim), or None if input is empty or embedding fails.
    """
    if not text or not text.strip():
        return None

    truncated_text = text[:MAX_EMBED_CHARS]

    try:
        response = _client.embeddings(
            model=OLLAMA_EMBED_MODEL,
            prompt=truncated_text,
        )
        embedding = response.get("embedding")
        if not embedding:
            logger.warning("Ollama embeddings response contained no 'embedding' field.")
            return None

        return np.array(embedding, dtype="float32")

    except Exception as e:
        logger.error(f"Error generating embedding with model '{OLLAMA_EMBED_MODEL}': {e}", exc_info=True)
        return None


def get_embeddings_batch(texts: List[str], max_workers: int = 4) -> List[np.ndarray]:
    """
    Generate embeddings for multiple texts concurrently.

    Args:
        texts: List of text strings. Empty or whitespace-only strings are skipped.
        max_workers: Number of concurrent worker threads.

    Returns:
        List of valid 1D numpy vector arrays.
    """
    valid_texts = [t for t in texts if t and t.strip()]
    if not valid_texts:
        return []

    logger.info(f"Generating embeddings batch for {len(valid_texts)} chunks using {OLLAMA_EMBED_MODEL}...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        vectors = list(executor.map(get_embedding, valid_texts))

    valid_vectors = [v for v in vectors if v is not None and isinstance(v, np.ndarray)]
    logger.info(f"Successfully generated {len(valid_vectors)} / {len(valid_texts)} embeddings.")

    return valid_vectors