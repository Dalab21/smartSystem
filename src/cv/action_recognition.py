import torch
import numpy as np
from torchvision.models.video import r3d_18  # Modèle 3D CNN de PyTorch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

class ActionRecognizer:
    def __init__(self):
        self.model = r3d_18(pretrained=True)
        self.model.eval()
        self.labels = ["Marcher", "Courir", "Sauter", "Pousser", "Tomber"]  # Exemple
        self.transform = Compose([
            Resize((112, 112)),
            CenterCrop((112, 112)),
            ToTensor()
        ])

    def predict(self, video_frames):
        """Input: Liste de frames (np.array)"""
        # Préprocessing
        frames = [self.transform(Image.fromarray(frame)) for frame in video_frames]
        clip = torch.stack(frames).unsqueeze(0)  # Shape: [1, T, C, H, W]
        
        # Inference
        with torch.no_grad():
            outputs = self.model(clip)
        
        # Post-processing
        pred = torch.argmax(outputs).item()
        return self.labels[pred]