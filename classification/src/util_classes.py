from dataclasses import dataclass

class SexEnum:
    MALE = "M"
    FEMALE = "F"
    
@dataclass
class ClassificationResult:
    age: int
    sex: str
    race: str
    
@dataclass
class ClassificationMetadata:
    classification_result: ClassificationResult
    idx: int