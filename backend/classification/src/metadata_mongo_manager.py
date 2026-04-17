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
            "person_key": str(metadata.additional_metadata.frame_number) if metadata.additional_metadata.frame_number is not None else str(metadata.embedding_id),
            "classification_result": asdict(metadata.classification_result),
            "additional_metadata": asdict(metadata.additional_metadata) or {},
            "total_duration_sec": float(getattr(metadata.additional_metadata, "duration_sec", 0.0) or 0.0),
            "track_ids": [metadata.additional_metadata.frame_number] if metadata.additional_metadata.frame_number is not None else [],
        }
        self.sessions.update_one(
            {"session_id": metadata.session_id},
            {"$push": {"faces": doc}}
        )

    def merge_duplicate_time(
        self,
        session_id: str,
        canonical_person_key: str,
        duplicate_track_id: int | None,
        duration_sec: float,
    ):
        """Merge duplicate track time into canonical person entry."""
        canonical_person_key = str(canonical_person_key)
        duration_sec = float(duration_sec or 0.0)

        session = self.get_session(session_id)
        if not session:
            return

        faces = session.get("faces", [])
        target_idx = None
        for i, face in enumerate(faces):
            if str(face.get("person_key")) == canonical_person_key:
                target_idx = i
                break

        if target_idx is None:
            # If canonical person is not yet present in Mongo for this session, skip silently.
            return

        target = faces[target_idx]
        old_total = float(target.get("total_duration_sec", 0.0) or 0.0)
        new_total = old_total + duration_sec
        track_ids = target.get("track_ids") or []
        if duplicate_track_id is not None and duplicate_track_id not in track_ids:
            track_ids.append(duplicate_track_id)

        self.sessions.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    f"faces.{target_idx}.total_duration_sec": new_total,
                    f"faces.{target_idx}.track_ids": track_ids,
                }
            },
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

    def find_face_by_person_key(self, session_id: str, person_key: str):
        faces = self.get_faces(session_id)
        key = str(person_key)
        for face in faces:
            if str(face.get("person_key")) == key:
                return face
        return None
