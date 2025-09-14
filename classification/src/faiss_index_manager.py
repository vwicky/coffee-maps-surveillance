import faiss
import os
from util_classes import SexEnum, ClassificationResult, ClassificationMetadata


class FaissIndexManager:
    _default_save_path = "../faiss_index/faiss_faces.index"
    
    def __init__(self, dim: int, threshold: float, faiss_index_path: str = None):
        self.faiss_index_path = faiss_index_path if faiss_index_path else self._default_save_path
        self.threshold = threshold
        
        if faiss_index_path and os.path.exists(faiss_index_path):
            self.load()
        else:
            self.index = faiss.IndexFlatIP(dim)
        self.idx_map = []  # maps FAISS index → metadata face_id or MongoDB embedding_id

    def add_embedding(self, embedding, metadata_idx) -> int:
        """Add embedding and return its FAISS index"""
        self.index.add(embedding)
        self.idx_map.append(metadata_idx)
        return self.index.ntotal - 1
        
    def check_if_present(self, embedding) -> int | None:
        if self.index.ntotal == 0:
            return None
        D, I = self.index.search(embedding, k=1)
        best_score = D[0][0]
        best_idx = I[0][0]
        if best_score > self.threshold:
            return self.idx_map[best_idx]
        return None
        
    def save(self):
        faiss.write_index(self.index, self.faiss_index_path)
        print(f"Index saved to {self.faiss_index_path}")
        
    def load(self):
        # idx_map must be persisted separately (e.g., in MongoDB)
        self.index = faiss.read_index(self.faiss_index_path)
        return self.index

