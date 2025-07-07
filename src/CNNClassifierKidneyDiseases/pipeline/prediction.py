import numpy as np
import torch
import os

from torchvision import transforms
from PIL import Image

class PredictionPipeline:

    def __init__(self,filename):
        self.filename = filename
        self.model = torch.load(os.path.join("artifacts", "training", "model.pth") ,weights_only=False)
        self.model.eval()  # Set model to evaluation mode

    def predict(self):
        imagename =self.filename
        test_image = Image.open(imagename).convert("RGB")
        transform = transforms.Compose([transforms.Resize((224,224)) , transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.1921, 0.1921, 0.1921] , std=[0.2601, 0.2601, 0.2601])])
        
        img_tensor = transform(test_image)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        # Model inference
        with torch.no_grad():
            output = self.model(img_tensor)
            result = torch.argmax(output, dim=1).item()

        # Interpretation
        prediction = 'Tumor' if result == 1 else 'Normal'
        return [{"image": prediction}]