import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torchvision.models as models

def load_fire_model(model_path='models/flame_detector.pth'):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 2)  # Fire vs Non-Fire
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

class FireClassifier:
    def __init__(self, model_path='models/flame_detector.pth'):
        self.model = load_fire_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)
            fire_confidence = prob[1].item()
        return fire_confidence
