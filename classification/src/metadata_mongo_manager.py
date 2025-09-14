from pymongo import MongoClient
from .util_classes import ClassificationResult, ClassificationMetadata, AdditionalMetadata
from dataclasses import asdict

class MongoMetadataManager:
    def __init__(self, mongo_uri="mongodb://localhost:27017", db_name="face_db"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.sessions = self.db["sessions"]

    def create_session(self, session_id: str, video_path: str, notes: str = ""):
        """Create a new session if it doesn't exist."""
        if self.sessions.find_one({"session_id": session_id}):
            return
        doc = {
            "session_id": session_id,
            "video_path": video_path,
            "faces": [],
            "notes": notes
        }
        self.sessions.insert_one(doc)

    def add_face(self, metadata: ClassificationMetadata):
        """Add a face to a session"""
        doc = {
            "face_id": metadata.face_id,
            "embedding_id": metadata.embedding_id,
            "classification_result": asdict(metadata.classification_result),
            "additional_metadata": asdict(metadata.additional_metadata) or {}
        }
        self.sessions.update_one(
            {"session_id": metadata.session_id},
            {"$push": {"faces": doc}}
        )

    def get_session(self, session_id: str):
        return self.sessions.find_one({"session_id": session_id})

    def get_faces(self, session_id: str):
        session = self.get_session(session_id)
        return session["faces"] if session else []

    def find_face(self, session_id: str, face_id: str):
        faces = self.get_faces(session_id)
        for face in faces:
            if face["face_id"] == face_id:
                return face
        return None
