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

    def add_embedding(self, embedding):
        self.index.add(embedding)
        
    def check_if_present(self, embedding) -> int | None:
        if self.index.ntotal > 0:  # check existing embeddings
            D, I = self.index.search(embedding, k=1)
            best_score = D[0][0]
            best_idx = I[0][0]

            if best_score > self.threshold:
                print(f"Embedding already seen: {best_idx} (score={best_score:.3f})")
                return best_idx
            else:
                print("New embedding detected, returning None")
                return None
        else:
            print("Index empty, returning None")
            return None
        
    def save(self):
        faiss.write_index(self.index, self.faiss_index_path)
        print(f"Index saved to {self.faiss_index_path}")
        
    def load(self):
        self.index = faiss.read_index(self.faiss_index_path)
        return self.index
