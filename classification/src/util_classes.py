from dataclasses import dataclass

class SexEnum:
    MALE = "M"
    FEMALE = "F"
    
@dataclass
class ClassificationResult:
    age: int = 0
    sex: str = ""
    race: str = ""
    
    famous: str = ""
    
@dataclass
class AdditionalMetadata:
    frame_number: int
    bbox: list
    
@dataclass
class ClassificationMetadata:
    session_id: str
    face_id: str
    embedding_id: str
    
    classification_result: ClassificationResult
    additional_metadata: AdditionalMetadata
    # old field
    idx: int = 0