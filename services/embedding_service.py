import ollama
import numpy as np
from concurrent.futures import ThreadPoolExecutor

MAX_CHARS = 4000


def get_embedding(text: str):
    if not text.strip():
        return None

    text = text[:MAX_CHARS]

    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )

    return np.array(response["embedding"], dtype="float32")


def get_embeddings_batch(texts: list[str]):
    texts = [t for t in texts if t.strip()]

    with ThreadPoolExecutor(max_workers=4) as executor:
        vectors = list(executor.map(get_embedding, texts))

    return [v for v in vectors if v is not None]