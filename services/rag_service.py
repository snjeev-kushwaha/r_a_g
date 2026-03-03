import numpy as np
from db.vector_db import VectorDB
from services.embedding_service import get_embedding
from utils.file_loader import chunk_text
from config import TOP_K
from services.llm_service import generate_answer

vector_db = VectorDB()

def ingest_text(text: str):
    chunks = chunk_text(text)

    if not chunks:
        return
    
    vectors = []
    valid_chunks = []
    
    for chunk in chunks:
        embedding = get_embedding(chunk)
        if embedding is not None:
            vectors.append(embedding)
            valid_chunks.append(chunk)
    
    if not vectors:
        raise ValueError("No valid embeddings generated.")
    
    vectors = np.vstack(vectors)
    vector_db.add_vectors(vectors, valid_chunks)


def query_rag(question: str) -> str:
    q_vec = get_embedding(question).reshape(1, -1)
    chunks = vector_db.search(q_vec, top_k=TOP_K)

    if not chunks:
        return "No data found."

    context = "\n\n".join(chunks)
    return generate_answer(question, context)