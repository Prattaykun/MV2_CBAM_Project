import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from .model import MobileNetV2_CBAM

class FirePredictor:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = MobileNetV2_CBAM(num_classes=2)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model path {model_path} not found. Using random weights.")
            
        self.model.to(self.device).eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Hooks for Grad-CAM
        self.gradients = None
        self.activations = None
        
        # Hook into the last feature layer of MobileNetV2
        # model.features is a Sequential
        target_layer = self.model.features[-1]
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        class_names = ['Fire', 'No_Fire'] 
        predicted_class = class_names[predicted.item()]
        
        return predicted_class, confidence.item()

    def generate_cam(self, image_path, target_class_idx=None):
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Zero grads
        self.model.zero_grad()
        
        # Forward pass needs gradients for CAM
        output = self.model(img_tensor)
        
        if target_class_idx is None:
            target_class_idx = output.argmax(dim=1).item()
            
        # Backward pass for the target class
        output[:, target_class_idx].backward()
        
        # Global Average Pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the channels by gradients
        activations = self.activations.detach().clone()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # ReLU on top of the heatmap
        heatmap = F.relu(heatmap)
        
        # Normalize the heatmap
        heatmap = heatmap.cpu().numpy()
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        
        # Resize heatmap to original image size
        # Read image to get size
        orig_image = cv2.imread(image_path)
        if orig_image is None:
             # Fallback if cv2 fails to read (e.g. specialized formats)
             orig_image = np.array(image.convert('RGB')) 
             orig_image = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)

        heatmap = cv2.resize(heatmap, (orig_image.shape[1], orig_image.shape[0]))
        
        # Convert to RGB heatmap
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Superimpose
        superimposed_img = heatmap_colored * 0.4 + orig_image
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        return superimposed_img, heatmap_uint8

if __name__ == "__main__":
    predictor = FirePredictor('ml_core/models/mv2_cbam_best.pth')
