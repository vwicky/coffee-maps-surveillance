import os
import cv2
import insightface
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import faiss
import numpy as np
from dotenv import load_dotenv
import os

# our modules
from util_classes import SexEnum, ClassificationResult, ClassificationMetadata, AdditionalMetadata
from faiss_index_manager import FaissIndexManager
from metadata_manager import MetadataManager

# subclassifier models
from sub_classifiers.race_classifier import RaceClassifier
from sub_classifiers.famous_classifier import FamousClassifier

from metadata_mongo_manager import MongoMetadataManager

# Load environment variables from .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "general_db")  # fallback

class FaceClassifier:
    def __init__(self, 
                 ctx_id=-1, #cpu
                 det_size=(640, 640), 
                 race_model_path = None, 
                 famous_model_path = None,
                 load_previous_faces: bool = False # path to faiss index file
                 ):
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
                
        # loading previous faces
        if load_previous_faces:
            self.face_fid = FaissIndexManager(
                dim=512,
                threshold=0.5,
                faiss_index_path="../faiss_index/faiss_faces.index"
            )
            self.metadata_manager = MetadataManager(metadata="../db/metadata.json")
        else:
            self.face_fid = FaissIndexManager(
                dim=512,
                threshold=0.5,
                faiss_index_path=None
            )
            self.metadata_manager = MetadataManager(metadata=None)
            
        # adding mongodb
        self.mongo_metadata_manager = MongoMetadataManager(mongo_uri=MONGO_URI, db_name=MONGO_DB)
            
        # loading Race model
        self.race_model_path = race_model_path
        if self.race_model_path:
            self.race_model = RaceClassifier(model_path=self.race_model_path)
            
        # loading Famous People model
        self.famous_model_path = famous_model_path
        if self.famous_model_path:
            self.famous_model = FamousClassifier(model_path=self.famous_model_path)
            
    def analyze_face_on_frame(self, img, session_id, frame_number=None):
        """ analyses frame and saves info in a mongodb """
        results = []
        
        faces = self.app.get(img)
        for face in faces:
            embedding = face.normed_embedding.astype("float32").reshape(1, -1)
            
            # check FAISS
            existing_id = self.face_fid.check_if_present(embedding)
            if existing_id is not None:
                continue  # already seen

            # classify new face
            result = self.detect_(face, img)

            # add to FAISS and Mongo
            embedding_id = self.face_fid.add_embedding(embedding, metadata_idx=frame_number)
            face_id = f"{session_id}_frame{frame_number}_face{embedding_id}"
            
            additional_metadata = AdditionalMetadata(
                frame_number=frame_number, bbox=face.bbox.tolist()
            )
            metadata = ClassificationMetadata(
                session_id=session_id,
                face_id=face_id,
                embedding_id=embedding_id,
                classification_result=result,
                additional_metadata=additional_metadata
            )
            
            self.mongo_metadata_manager.add_face(metadata)
            results.append(metadata)
        return results
            
    def person_with_id(self, id: int) -> ClassificationResult:
        data = self.metadata_manager.get_by_id(id)
        if data is not None:
            return data.classification_result
        return ClassificationResult()

    def analyze_image(self, img):
        faces = self.app.get(img)
        return faces

    def predict_race(self, face, img, ensemble=True, threshold=0.5):
        """
        Predict race for a detected face using landmarks-aware cropping and optional ensemble.
            Landmark-aware crop → reduces background noise.
            Ensemble → averages multiple slightly shifted crops to reduce random misclassifications.
            Resize + normalization → matches transformer expectations.
            Confidence threshold → ignore low-confidence predictions.   
        Returns: (race_label, confidence)
        """
        if self.race_model_path is None:
            return "unknown", 0.0

        return self.race_model.predict(face, img, ensemble, threshold)
    
    def detect_famous(self, img) -> str:
        if self.famous_model_path is None:
            return "unknown", 0.0
        return self.famous_model.predict(img)

    def detect_gender(self, face) -> str:
        return SexEnum.MALE if face.gender == 1 else SexEnum.FEMALE
    
    def detect_age(self, face) -> int:
        return int(face.age)
    
    def detect_race(self, face, img) -> str:
        race, percentage = self.predict_race(face, img)
        return race, percentage
    
    def detect_(self, face, img) -> ClassificationResult:
        gender = self.detect_gender(face)
        age = self.detect_age(face)
        race = self.detect_race(face, img)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
        img_pil = Image.fromarray(img)             # now PIL Image
        famous = self.detect_famous(img_pil)
        
        return ClassificationResult(age, gender, race, famous)

    def draw_face(self, face, img, cl_result):
        # draw rctangle
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # add label
        label = f"{cl_result.sex}, {cl_result.age}y, {cl_result.race}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return img

    def show_images_grid(self, imgs, titles=None, rows=1, cols=3):
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        axes = axes.flatten()
        for i in range(rows * cols):
            if i < len(imgs):
                img_rgb = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
                axes[i].imshow(img_rgb)
                axes[i].set_title(titles[i] if titles else "")
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

    def check_if_face_present(self, embedding) -> int | None:
        """
            Searches for that face embedding in FAISS, if similarity > threshold -> return index in metadata, else -> None, indicating absence
        """
        return self.face_fid.check_if_present(embedding)

    def analyze_folder(self, folder_path) -> dict[str, ClassificationMetadata]:
        image_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        results = {} # img_path -> idx in metadata
        for file_name in image_files:
            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Cannot read {file_name}")
                continue
            print(f"> amalysing {file_name}")
            faces = self.analyze_image(img)
            
            faces_idx = []
            for face in faces:
                embedding = face.normed_embedding.astype("float32").reshape(1, -1)
                
                search_face = self.check_if_face_present(embedding)
                if search_face is not None:
                    # this face was already detected
                    classification_report = self.metadata_manager.get_by_id(search_face).classification_result
                    faces_idx.append(search_face)
                else:
                    # new face - need to classify it
                    classification_report = self.detect_(face, img)
                    
                    # Add new person
                    self.face_fid.add_embedding(embedding)
                    
                    new_idx = self.metadata_manager.last_index() + 1
                    self.metadata_manager.add(ClassificationMetadata(
                        idx=new_idx,
                        classification_result=classification_report
                    ))
                    faces_idx.append(new_idx)
            results[file_name] = faces_idx
            
        return results

    def save_faiss_index(self):
        self.face_fid.save()
    
    def save_metadata(self):
        self.metadata_manager.save()
    
