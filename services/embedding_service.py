import ollama
import numpy as np

MAX_CHARS = 4000

def get_embedding(text: str):
    if not text.strip():
        return None

    text = text[:MAX_CHARS]

    try:
        response = ollama.embeddings(
            model="nomic-embed-text",
            prompt=text
        )

        embedding = np.array(response["embedding"], dtype="float32")
        return embedding

    except Exception as e:
        print("Embedding error:", e)
        return None