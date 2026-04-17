from dataclasses import dataclass
from typing import Optional

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
    frame_number: Optional[int]
    bbox: list
    duration_sec: float = 0.0
    enter_time: str = ""
    exit_time: str = ""
    is_staff: bool = False
    
@dataclass
class ClassificationMetadata:
    session_id: str
    face_id: str
    embedding_id: str
    
    classification_result: ClassificationResult
    additional_metadata: AdditionalMetadata
    # old field
    idx: int = 0