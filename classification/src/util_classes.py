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
class ClassificationMetadata:
    classification_result: ClassificationResult
    idx: int