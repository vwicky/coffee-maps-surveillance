import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import torch.nn.functional as F
import numpy as np
import cv2

from .i_classifier import Classifier

class FamousClassifier(Classifier):
    
    _default_model_path = "classification/models/resnet_famous_faces.pth"
    _default_classes_path = "classification/data/Dataset.csv"

    def __init__(self, model_path: str = None, device: str = None, threshold: float = 0.975, top_k: int = 3):
        """
        model_path: path to trained model weights
        device: "cuda" or "cpu"
        threshold: minimum probability to consider prediction confident
        top_k: number of top predictions to check for 'not famous'
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or self._default_model_path
        self.threshold = threshold
        self.top_k = top_k

        # load classes
        df = pd.read_csv(self._default_classes_path)
        self.all_labels = sorted(df['label'].unique())
        self.num_classes = len(self.all_labels)

        self.label2idx = {label: idx for idx, label in enumerate(self.all_labels)}
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}

        # load trained model
        self.model = models.resnet50()
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # proper preprocessing for ResNet
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, img):
        # convert NumPy/OpenCV BGR -> PIL RGB
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        x = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)

        top_prob, top_idx = probs.max(dim=1)
        top_prob = top_prob.item()
        top_idx = top_idx.item()

        if top_prob < self.threshold:
            return "not famous", top_prob
        else:
            return self.idx2label[top_idx], top_prob

