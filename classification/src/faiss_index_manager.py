import faiss

from util_classes import SexEnum, ClassificationResult, ClassificationMetadata


class FaissIndexManager:
    def __init__(self, dim: int, threshold: float, faiss_index_path: str = "../faiss_index/faiss_faces.index"):
        self.index = faiss.IndexFlatIP(dim)
        self.threshold = threshold
        self.faiss_index_path = faiss_index_path
        
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
        return faiss.write_index(self.index, self.faiss_index_path)
    def load(self):
        return faiss.read_index(self.faiss_index_path)
