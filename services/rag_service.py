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

    vectors = np.vstack([get_embedding(chunk) for chunk in chunks])

    vector_db.add_vectors(vectors, chunks)


def query_rag(question: str) -> str:
    q_vec = get_embedding(question).reshape(1, -1)
    chunks = vector_db.search(q_vec, top_k=TOP_K)

    if not chunks:
        return "No data found."

    context = "\n\n".join(chunks)
    return generate_answer(question, context)