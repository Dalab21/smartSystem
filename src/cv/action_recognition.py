import torch
import numpy as np
# model 3D CNN (PyTorch)
from torchvision.models.video import r3d_18  
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor

class ActionRecognizer:
    def __init__(self):
        self.model = r3d_18(pretrained=True)
        self.model.eval()
        # exples de mouvements
        self.labels = ["Marcher", "Courir", "Sauter", "Pousser", "Tomber"]  
        self.transform = Compose([
            Resize((112, 112)),
            CenterCrop((112, 112)),
            ToTensor()
        ])

    def predict(self, video_frames):
        """Input: Liste de frames (np.array)"""
        # Preprocessing
        frames = [self.transform(Image.fromarray(frame)) for frame in video_frames]
        clip = torch.stack(frames).unsqueeze(0)  # shape: [1, T, C, H, W]
        
        # inference
        with torch.no_grad():
            outputs = self.model(clip)
        
        # post-processing
        pred = torch.argmax(outputs).item()
        return self.labels[pred]