
from PIL import Image
import torch
import cv2
import numpy as np
from torchvision import transforms
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

from .i_classifier import Classifier

class RaceClassifier(Classifier):
    def __init__(self, model_path: str):
        self.model_path = model_path

        self.race_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # or model expected size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        self.race_model = AutoModelForImageClassification.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.race_extractor = AutoFeatureExtractor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.race_model.eval()
        # Map numeric labels to names
        self.race_labels = self.race_model.config.id2label
        
    def predict(self, face, img, ensemble=True, threshold=0.5):
        """
        Predict race for a detected face using landmarks-aware cropping and optional ensemble.
            Landmark-aware crop → reduces background noise.
            Ensemble → averages multiple slightly shifted crops to reduce random misclassifications.
            Resize + normalization → matches transformer expectations.
            Confidence threshold → ignore low-confidence predictions.   
        Returns: (race_label, confidence)
        """
        
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
  