import ollama
import numpy as np

def get_embedding(text: str) -> np.ndarray:
    if not text or not text.strip():
        raise ValueError("Empty text passed to embedding")

    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )

    embedding = response["embedding"]

    if not embedding:
        raise ValueError("Empty embedding returned from Ollama")

    return np.array(embedding, dtype="float32").reshape(1, -1)
