# import faiss
# import numpy as np

# class VectorDB:
#     def __init__(self, dim=768):
#         self.index = faiss.IndexFlatL2(dim)
#         self.texts = []

#     def add_vectors(self, vectors, texts):
#         if not isinstance(vectors, np.ndarray):
#             raise ValueError("Vectors must be numpy array")

#         if vectors.ndim != 2:
#             raise ValueError(f"FAISS expects 2D array, got {vectors.shape}")

#         self.index.add(vectors)
#         self.texts.extend(texts)
    
#     def search(self, query_vector, top_k=3):
#         if self.index.ntotal == 0:
#             return []
    
#         query_vector = np.array(query_vector, dtype="float32").reshape(1, -1)
#         _, indices = self.index.search(query_vector, top_k)
    
#         return [
#             self.texts[i]
#             for i in indices[0]
#             if i < len(self.texts)
#         ]

import faiss
import numpy as np

class VectorDB:
    def __init__(self, dim=768):
        self.dim = dim
        base_index = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIDMap(base_index)

        self.id_counter = 0
        self.doc_index_map = {}  # doc_id -> list of vector IDs
        self.text_store = {}     # vector_id -> text

    def add_document(self, doc_id, vectors, texts):
        vectors = np.array(vectors, dtype="float32")

        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)

        assert vectors.shape[1] == self.dim

        ids = []
        for text in texts:
            vector_id = self.id_counter
            self.id_counter += 1
            ids.append(vector_id)
            self.text_store[vector_id] = text

        ids_np = np.array(ids)
        self.index.add_with_ids(vectors, ids_np)

        self.doc_index_map[doc_id] = ids

    def delete_document(self, doc_id):
        if doc_id not in self.doc_index_map:
            return

        ids = np.array(self.doc_index_map[doc_id])
        self.index.remove_ids(ids)

        for vid in ids:
            self.text_store.pop(vid, None)

        del self.doc_index_map[doc_id]

    def search(self, query_vector, top_k=3):
        query_vector = np.array(query_vector, dtype="float32").reshape(1, -1)
        _, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.text_store.get(idx))

        return results