import faiss
import numpy as np

class VectorDB:
    def __init__(self, dim=768):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add_vectors(self, vectors, texts):
        if not isinstance(vectors, np.ndarray):
            raise ValueError("Vectors must be numpy array")

        if vectors.ndim != 2:
            raise ValueError(f"FAISS expects 2D array, got {vectors.shape}")

        self.index.add(vectors)
        self.texts.extend(texts)
    
    def search(self, query_vector, top_k=3):
        if self.index.ntotal == 0:
            return []
    
        query_vector = np.array(query_vector, dtype="float32").reshape(1, -1)
        _, indices = self.index.search(query_vector, top_k)
    
        return [
            self.texts[i]
            for i in indices[0]
            if i < len(self.texts)
        ]