import json
from dataclasses import asdict
from typing import Union
from util_classes import ClassificationMetadata, ClassificationResult

class MetadataManager:
    _default_save_path = "../db/metadata.json"
    
    def __init__(self, metadata: Union[str, list[ClassificationMetadata], None] = None):
        """
        metadata can be:
          - str : path to JSON file
          - list : list of ClassificationMetadata
          - None : start with empty list
        """
        if isinstance(metadata, str):
            self.metadata: list[ClassificationMetadata] = []
            self.load(metadata)
        elif isinstance(metadata, list):
            self.metadata = metadata
        else:
            self.metadata = []

    # ------------------- Persistence -------------------
    def load(self, path: str = None):
        """Load metadata from a JSON file into ClassificationMetadata objects."""
        
        path = path if path else self._default_save_path
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:  # empty file
            self.metadata = []
            return

        raw = json.loads(content)

        self.metadata = []
        for face_id, data in raw.items():
            data = ClassificationResult(**data)
            self.metadata.append(ClassificationMetadata(idx=face_id, classification_result=data))

    def save(self, path: str = None):
        """Save metadata to a JSON file (face_id → attributes)."""
        path = path if path else self._default_save_path
        
        export = {m.idx: asdict(m.classification_result) for m in self.metadata}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)

    # ------------------- Access -------------------
    def last_index(self) -> int:
        return max(0, len(self.metadata) - 1)

    def get_length(self) -> int:
        return len(self.metadata)

    def get_by_id(self, idx: int) -> ClassificationMetadata:
        try:
            return self.metadata[idx]
        except IndexError:
            print(f"> Index {idx} was out of range, metadata has {self.get_length()} elements")
            return None

    def get_by_face_id(self, face_id: str) -> ClassificationMetadata | None:
        for m in self.metadata:
            if m.face_id == face_id:
                return m
        return None

    # ------------------- Modify -------------------
    def add(self, classification_metadata: ClassificationMetadata) -> int:
        """Append new metadata and return its index."""
        self.metadata.append(classification_metadata)
        return self.last_index()

    def remove_by_face_id(self, face_id: str) -> bool:
        """Remove metadata by face_id, return True if removed."""
        for i, m in enumerate(self.metadata):
            if m.face_id == face_id:
                del self.metadata[i]
                return True
        return False

    # ------------------- Pythonic -------------------
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int) -> ClassificationMetadata:
        return self.metadata[idx]

    def __iter__(self):
        return iter(self.metadata)
