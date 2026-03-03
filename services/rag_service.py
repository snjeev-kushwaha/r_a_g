from db.vector_db import VectorDB
# from services.embedding_service import get_embedding
from services.embedding_service import get_embedding, get_embeddings_batch
from utils.file_loader import chunk_text
from config import TOP_K
from services.llm_service import generate_answer

vector_db = VectorDB()

def ingest_document(doc_id: str, text: str):
    # Remove old vectors if document already exists
    vector_db.delete_document(doc_id)

    chunks = chunk_text(text)
    # vectors = [get_embedding(chunk) for chunk in chunks]
    vectors = get_embeddings_batch(chunks)

    vector_db.add_document(doc_id, vectors, chunks)

def query_rag(question: str) -> str:
    q_vec = get_embedding(question)

    if q_vec is None:
        return "Embedding failed."

    q_vec = q_vec.reshape(1, -1)

    chunks = vector_db.search(q_vec, top_k=TOP_K)
    chunks = filter_chunks(question, chunks)

    if not chunks:
        return "No data found."

    context = "\n\n".join(chunks)
    return generate_answer(question, context)

def filter_chunks(question: str, chunks: list[str]):
    keywords = question.lower().split()
    
    filtered = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        if any(word in chunk_lower for word in keywords):
            filtered.append(chunk)

    return filtered if filtered else chunks

# import numpy as np
# from db.vector_db import VectorDB
# from services.embedding_service import get_embedding
# from utils.file_loader import chunk_text
# from config import TOP_K
# from services.llm_service import generate_answer

# vector_db = VectorDB()

def ingest_text(text: str):
    pass
#     chunks = chunk_text(text)

#     if not chunks:
#         return
    
#     vectors = []
#     valid_chunks = []
    
#     # for bulk upload
#     # vectors = [get_embedding(chunk) for chunk in chunks]

#     for chunk in chunks:
#         embedding = get_embedding(chunk)
#         if embedding is not None:
#             vectors.append(embedding)
#             valid_chunks.append(chunk)
    
#     if not vectors:
#         raise ValueError("No valid embeddings generated.")
    
#     vectors = np.vstack(vectors)
#     vector_db.add_vectors(vectors, valid_chunks)


# def query_rag(question: str) -> str:
#     q_vec = get_embedding(question).reshape(1, -1)
#     chunks = vector_db.search(q_vec, top_k=TOP_K)

#     if not chunks:
#         return "No data found."

#     context = "\n\n".join(chunks)
#     return generate_answer(question, context)