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

# our modules
from util_classes import SexEnum, ClassificationResult, ClassificationMetadata
from faiss_index_manager import FaissIndexManager
from metadata_manager import MetadataManager

class FaceClassifier:
    def __init__(self, 
                 ctx_id=-1, #cpu
                 det_size=(640, 640), 
                 race_model_path= None, 
                 load_previous_faces: bool = False # path to faiss index file
                 ):
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        
        self.face_fid = FaissIndexManager(
            dim=512,
            threshold=0.5,
            faiss_index_path="faiss_index/faiss_faces.index"
        )
        
        if load_previous_faces:
            metadata = self.face_fid.load()
            self.metadata_manager = MetadataManager(metadata=metadata)
        else:
            self.metadata_manager = MetadataManager(metadata=None)

        # Load race model if provided
        self.race_model = None
        self.race_extractor = None
        
        # Define once in __init__
        self.race_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # or model expected size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        if race_model_path:
            self.race_model = AutoModelForImageClassification.from_pretrained(
                race_model_path, trust_remote_code=True
            )
            self.race_extractor = AutoFeatureExtractor.from_pretrained(
                race_model_path, trust_remote_code=True
            )
            self.race_model.eval()
            # Map numeric labels to names
            self.race_labels = self.race_model.config.id2label

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
        if self.race_model is None:
            return "unknown", 0.0

        # Get bounding box and landmarks
        x1, y1, x2, y2 = map(int, face.bbox)
        h, w, _ = img.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        # Use landmarks for tighter crop if available
        if hasattr(face, 'kps') and face.kps is not None:
            # landmarks: 5 or 68 points (x, y)
            kps = np.array(face.kps)
            x_min = max(0, int(kps[:,0].min()))
            y_min = max(0, int(kps[:,1].min()))
            x_max = min(w, int(kps[:,0].max()))
            y_max = min(h, int(kps[:,1].max()))
            x1, y1, x2, y2 = x_min, y_min, x_max, y_max

        # Crop the face
        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            return "unknown", 0.0

        # Prepare crops for ensemble
        crops = [face_crop]
        if ensemble:
            # Slight shifts
            dh, dw = int(0.05*(y2-y1)), int(0.05*(x2-x1))
            crops.append(img[max(0,y1-dh):min(h,y2+dh), max(0,x1-dw):min(w,x2+dw)])
            crops.append(img[max(0,y1+dh):min(h,y2-dh), max(0,x1+dw):min(w,x2-dw)])

        probs_list = []
        for crop in crops:
            # Convert BGR -> RGB PIL
            img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            # Resize to model expected size
            img_pil = img_pil.resize((224,224))
            # Transform
            img_tensor = self.race_transform(img_pil).unsqueeze(0)

            with torch.no_grad():
                outputs = self.race_model(img_tensor)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probs_list.append(probs)

        # Average ensemble probabilities
        probs_avg = torch.mean(torch.stack(probs_list), dim=0)
        max_prob, race_idx = torch.max(probs_avg, dim=-1)

        if max_prob.item() < threshold:
            return "unknown", 0.0

        return self.race_labels[race_idx.item()], max_prob.item()

    def detect_gender(self, face) -> str:
        return SexEnum.MALE if face.gender == 1 else SexEnum.FEMALE
    def detect_age(self, face) -> int:
        return int(face.age)
    def detect_race(self, face, img) -> str:
        race, percentage = self.predict_race(face, img)
        return f"{race}, {percentage}%"
    def detect_(self, face, img) -> ClassificationResult:
        gender = self.detect_gender(face)
        age = self.detect_age(face)
        race = self.detect_race(face, img)
        
        return ClassificationResult(age, gender, race)

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

    def analyze_folder(self, folder_path):
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
                    img_labeled = self.draw_face(face, img, classification_report)
                    faces_idx.append(search_face)
                else:
                    # new face - need to classify it
                    classification_report = self.detect_(face, img)
                    img_labeled = self.draw_face(face, img, classification_report)
                    
                    # Add new person
                    self.face_fid.add_embedding(embedding)
                    self.metadata_manager.add(ClassificationMetadata(
                        classification_report, 
                        self.metadata_manager.last_index() + 1  # new ID
                    ))
                    faces_idx.append(self.metadata_manager.last_index() + 1)
            results[file_name] = faces_idx
            
        return results

    def analyze_folder_grid(self, folder_path, rows=1, cols=3):
        """ Use in .ipynb files to display nicely """
        image_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        labeled_images = []
        titles = []
        for file_name in image_files:
            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Cannot read {file_name}")
                continue
            faces = self.analyze_image(img)
            
            for face in faces:
                embedding = face.normed_embedding.astype("float32").reshape(1, -1)
                
                search_face = self.check_if_face_present(embedding)
                if search_face is not None:
                    # this face was already detected
                    classification_report = self.metadata_manager.get_by_id(search_face).classification_result
                    img_labeled = self.draw_face(face, img, classification_report)
                else:
                    # new face - need to classify it
                    classification_report = self.detect_(face, img)
                    img_labeled = self.draw_face(face, img, classification_report)
                    
                    # Add new person
                    self.face_fid.add_embedding(embedding)
                    self.metadata_manager.add(ClassificationMetadata(
                        classification_report, 
                        len(metadata) - 1  # new ID
                    ))

                labeled_images.append(img_labeled)
                titles.append(file_name)

            if len(labeled_images) == rows * cols:
                self.show_images_grid(labeled_images, titles, rows, cols)
                labeled_images, titles = [], []

        if labeled_images:
            self.show_images_grid(labeled_images, titles, rows, cols)
    def save_faiss_index(self):
        self.face_fid.save()
    
