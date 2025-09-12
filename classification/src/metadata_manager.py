
from util_classes import SexEnum, ClassificationResult, ClassificationMetadata

class MetadataManager:
  def __init__(self, metadata: list[ClassificationMetadata]):
    self.metadata = metadata if metadata else []
    
  def last_index(self) -> int:
    return len(self.metadata) - 1
    
  def get_by_id(self, id: int) -> ClassificationMetadata:
    return self.metadata[id]
  
  def add(self, classification_metadata: ClassificationMetadata) -> int:
    """
    Args:
        classification_metadata (ClassificationMetadata): what we add
    Returns:
        int: new element's id
    """
    self.metadata.append(classification_metadata)
    return self.last_index()
     