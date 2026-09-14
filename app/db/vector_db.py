"""
FAISS-based vector database with persistence, concurrency safety, and metadata mapping.
"""

import os
import pickle
import threading
from typing import Dict, List, Optional
import faiss
import numpy as np

from app.core.exceptions import VectorDBError
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class VectorDB:
    """
    Thread-safe vector database using FAISS IndexIDMap with disk persistence.
    """

    def __init__(self, dim: int = 768, storage_dir: Optional[str] = None):
        self.dim = dim
        self.storage_dir = storage_dir
        self._lock = threading.Lock()

        self.id_counter: int = 0
        self.doc_index_map: Dict[str, List[int]] = {}  # doc_id -> list of vector IDs
        self.text_store: Dict[int, str] = {}           # vector_id -> text chunk

        if self.storage_dir and self._persisted_data_exists():
            self.load()
        else:
            self._init_empty_index()

    def _init_empty_index(self) -> None:
        """Initialize a fresh, empty FAISS IndexIDMap."""
        base_index = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIDMap(base_index)
        logger.debug(f"Initialized fresh empty FAISS index (dim: {self.dim})")

    def _persisted_data_exists(self) -> bool:
        """Check if both index and metadata files exist in storage directory."""
        if not self.storage_dir:
            return False
        index_path = os.path.join(self.storage_dir, "index.faiss")
        meta_path = os.path.join(self.storage_dir, "metadata.pkl")
        return os.path.isfile(index_path) and os.path.isfile(meta_path)

    def save(self) -> None:
        """
        Persist FAISS index and metadata to disk thread-safely.
        """
        if not self.storage_dir:
            return

        with self._lock:
            try:
                os.makedirs(self.storage_dir, exist_ok=True)
                index_path = os.path.join(self.storage_dir, "index.faiss")
                meta_path = os.path.join(self.storage_dir, "metadata.pkl")

                faiss.write_index(self.index, index_path)

                metadata = {
                    "id_counter": self.id_counter,
                    "doc_index_map": self.doc_index_map,
                    "text_store": self.text_store,
                    "dim": self.dim,
                }
                with open(meta_path, "wb") as f:
                    pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

                logger.info(
                    f"Saved vector database to {self.storage_dir} "
                    f"({self.index.ntotal} vectors, {len(self.doc_index_map)} docs)"
                )
            except Exception as e:
                logger.error(f"Failed to persist vector database: {e}", exc_info=True)
                raise VectorDBError(f"Failed to persist vector database: {str(e)}") from e

    def load(self) -> None:
        """
        Load FAISS index and metadata from disk thread-safely.
        """
        if not self.storage_dir:
            return

        with self._lock:
            index_path = os.path.join(self.storage_dir, "index.faiss")
            meta_path = os.path.join(self.storage_dir, "metadata.pkl")

            try:
                self.index = faiss.read_index(index_path)
                with open(meta_path, "rb") as f:
                    metadata = pickle.load(f)

                self.id_counter = metadata.get("id_counter", 0)
                self.doc_index_map = metadata.get("doc_index_map", {})
                self.text_store = metadata.get("text_store", {})
                self.dim = metadata.get("dim", self.dim)

                logger.info(
                    f"Successfully loaded vector database from {self.storage_dir} "
                    f"({self.index.ntotal} vectors, {len(self.doc_index_map)} docs)"
                )
            except Exception as e:
                logger.error(
                    f"Failed to load persisted vector database from {self.storage_dir}: {e}. "
                    "Falling back to empty index.",
                    exc_info=True,
                )
                self._init_empty_index()
                self.id_counter = 0
                self.doc_index_map = {}
                self.text_store = {}

    def add_document(self, doc_id: str, vectors: np.ndarray, texts: List[str]) -> None:
        """
        Add a document's embeddings and text chunks to the vector database.

        Args:
            doc_id: Unique document identifier.
            vectors: 2D numpy array of embeddings (shape: [N, dim]).
            texts: List of text chunk strings corresponding to the embeddings.
        """
        vectors_np = np.array(vectors, dtype="float32")

        if len(vectors_np.shape) == 1:
            vectors_np = vectors_np.reshape(1, -1)

        if vectors_np.shape[1] != self.dim:
            raise VectorDBError(
                f"Vector dimension mismatch: expected {self.dim}, got {vectors_np.shape[1]}"
            )

        if len(vectors_np) != len(texts):
            raise VectorDBError(
                f"Count mismatch: got {len(vectors_np)} vectors but {len(texts)} texts"
            )

        with self._lock:
            ids = []
            for text in texts:
                vid = self.id_counter
                self.id_counter += 1
                ids.append(vid)
                self.text_store[vid] = text

            ids_np = np.array(ids, dtype=np.int64)
            self.index.add_with_ids(vectors_np, ids_np)
            self.doc_index_map[doc_id] = ids

            logger.info(
                f"Added document '{doc_id}' with {len(ids)} chunks. "
                f"Total vectors in DB: {self.index.ntotal}"
            )

        if self.storage_dir:
            self.save()

    def delete_document(self, doc_id: str) -> None:
        """
        Remove a document and all its associated vectors from the database.

        Args:
            doc_id: Unique document identifier to delete.
        """
        with self._lock:
            if doc_id not in self.doc_index_map:
                logger.debug(f"Document '{doc_id}' not present in vector DB; skipping deletion.")
                return

            ids = np.array(self.doc_index_map[doc_id], dtype=np.int64)
            self.index.remove_ids(ids)

            for vid in ids:
                self.text_store.pop(int(vid), None)

            del self.doc_index_map[doc_id]
            logger.info(
                f"Deleted document '{doc_id}' ({len(ids)} vectors removed). "
                f"Total vectors remaining: {self.index.ntotal}"
            )

        if self.storage_dir:
            self.save()

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[str]:
        """
        Find the most similar text chunks for a given query vector.

        Args:
            query_vector: 1D or 2D query embedding vector.
            top_k: Maximum number of closest chunks to return.

        Returns:
            List of matching text strings.
        """
        if not hasattr(self, "index") or self.index is None or self.index.ntotal == 0:
            logger.debug("Vector search skipped: index is empty.")
            return []

        q_vec = np.array(query_vector, dtype="float32").reshape(1, -1)
        if q_vec.shape[1] != self.dim:
            logger.warning(
                f"Query vector dimension mismatch: expected {self.dim}, got {q_vec.shape[1]}"
            )
            return []

        with self._lock:
            _, indices = self.index.search(q_vec, min(top_k, self.index.ntotal))

        results: List[str] = []
        for idx in indices[0]:
            if idx != -1:
                val = self.text_store.get(int(idx))
                if val:
                    results.append(val)

        logger.debug(f"Vector search returned {len(results)} chunks (requested top_k: {top_k})")
        return results