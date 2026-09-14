"""
RAG orchestration service.
Coordinates document ingestion, chunking, embedding, vector retrieval, and LLM answer generation.
"""

import re
import time
from typing import Any, Dict, List

from app.core.exceptions import EmbeddingError
from app.core.logging_config import get_logger
from app.db.vector_db import VectorDB
from app.services.embedding_service import get_embedding, get_embeddings_batch
from app.services.llm_service import generate_answer
from app.utils.file_loader import chunk_text
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    TOP_K,
    VECTOR_DB_PATH,
    VECTOR_DIM,
)

logger = get_logger(__name__)

# Singleton vector DB instance with persistence
vector_db = VectorDB(dim=VECTOR_DIM, storage_dir=VECTOR_DB_PATH)


def ingest_document(doc_id: str, text: str) -> int:
    """
    Ingest a document into the RAG system:
    Chunks the text, computes embeddings, and stores them in VectorDB.

    Args:
        doc_id: Unique identifier for the document (e.g., filename).
        text: Full raw text of the document.

    Returns:
        Number of successfully indexed chunks.

    Raises:
        ValueError: If doc_id or text is empty.
        EmbeddingError: If no embeddings could be generated.
    """
    if not doc_id or not doc_id.strip():
        raise ValueError("Document ID cannot be empty.")
    if not text or not text.strip():
        raise ValueError("Document text cannot be empty.")

    logger.info(f"Ingesting document '{doc_id}' (text length: {len(text)} chars)...")
    start_time = time.time()

    # Clean up existing document if already present
    vector_db.delete_document(doc_id)

    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    if not chunks:
        logger.warning(f"No chunks produced for document '{doc_id}'.")
        return 0

    logger.debug(f"Document '{doc_id}' split into {len(chunks)} chunks. Generating embeddings...")

    # Pair chunks with embeddings safely
    valid_chunks: List[str] = []
    valid_vectors = []

    # Batch embedding
    raw_vectors = get_embeddings_batch(chunks)
    if len(raw_vectors) == len(chunks):
        valid_chunks = chunks
        valid_vectors = raw_vectors
    else:
        # Fallback alignment if any chunk failed embedding
        logger.warning(
            f"Embedding count mismatch for '{doc_id}' ({len(raw_vectors)} vs {len(chunks)}). "
            "Re-aligning individual embeddings..."
        )
        for chunk in chunks:
            vec = get_embedding(chunk)
            if vec is not None:
                valid_chunks.append(chunk)
                valid_vectors.append(vec)

    if not valid_vectors:
        logger.error(f"Failed to generate any embeddings for document '{doc_id}'.")
        raise EmbeddingError(f"Could not generate embeddings for document '{doc_id}'.")

    vector_db.add_document(doc_id, valid_vectors, valid_chunks)

    elapsed = time.time() - start_time
    logger.info(
        f"Successfully ingested '{doc_id}' with {len(valid_chunks)} chunks in {elapsed:.2f}s."
    )
    return len(valid_chunks)


class RAGResult(str):
    """
    RAG query result that behaves as both a string (for backward compatibility)
    and a dictionary (with answer, sources_found, and is_relevant).
    """

    def __new__(cls, answer: str, sources_found: int = 0, is_relevant: bool = False):
        instance = super().__new__(cls, answer)
        instance.answer = answer
        instance.sources_found = sources_found
        instance.is_relevant = is_relevant
        return instance

    def __getitem__(self, key):
        if key == "answer":
            return self.answer
        if key == "sources_found":
            return self.sources_found
        if key == "is_relevant":
            return self.is_relevant
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "answer":
            return self.answer
        if key == "sources_found":
            return self.sources_found
        if key == "is_relevant":
            return self.is_relevant
        return default


def query_rag(question: str) -> RAGResult:
    """
    Query the RAG system and return a structured answer with metadata.

    Args:
        question: User query string.

    Returns:
        RAGResult containing:
            - answer (str): LLM-generated answer or fallback message.
            - sources_found (int): Number of relevant document chunks used.
            - is_relevant (bool): Whether relevant context was found in the indexed documents.
    """
    if not question or not question.strip():
        return RAGResult(
            answer="Please provide a valid question.",
            sources_found=0,
            is_relevant=False,
        )

    logger.info(f"Processing RAG query: '{question[:80]}'...")
    start_time = time.time()

    q_vec = get_embedding(question)
    if q_vec is None:
        logger.error("Failed to generate embedding for the question.")
        return RAGResult(
            answer="Error: Could not process the question due to embedding failure. Please try again.",
            sources_found=0,
            is_relevant=False,
        )

    q_vec = q_vec.reshape(1, -1)
    retrieved_chunks = vector_db.search(q_vec, top_k=TOP_K)
    logger.debug(f"Retrieved {len(retrieved_chunks)} raw chunks from vector search.")

    filtered_chunks = filter_chunks(question, retrieved_chunks)

    if not filtered_chunks:
        logger.info(f"No relevant document chunks found for question: '{question[:80]}'")
        return RAGResult(
            answer="No relevant information found in the document for your question.",
            sources_found=0,
            is_relevant=False,
        )

    context = "\n\n".join(filtered_chunks)
    answer = generate_answer(question, context)

    elapsed = time.time() - start_time
    logger.info(f"RAG query completed in {elapsed:.2f}s with {len(filtered_chunks)} sources.")

    return RAGResult(
        answer=answer,
        sources_found=len(filtered_chunks),
        is_relevant=True,
    )


def filter_chunks(question: str, chunks: List[str]) -> List[str]:
    """
    Filter retrieved chunks by checking for question keyword relevance.
    More lenient: returns original chunks if keyword filtering would remove everything.

    Args:
        question: User query string.
        chunks: List of retrieved text chunks.

    Returns:
        Filtered list of relevant text chunks.
    """
    if not chunks:
        return []

    stop_words = {
        "what", "are", "the", "is", "a", "an", "and", "or", "how", "where",
        "when", "why", "which", "for", "with", "about", "can", "you", "tell",
        "me", "this", "that", "from", "in", "on", "at", "to", "by", "of"
    }

    words = re.findall(r"\b\w+\b", question.lower())
    keywords = [w for w in words if len(w) > 2 and w not in stop_words]

    if not keywords:
        return chunks

    filtered = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        if any(kw in chunk_lower for kw in keywords):
            filtered.append(chunk)

    return filtered if filtered else chunks