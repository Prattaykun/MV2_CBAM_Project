
import sys
import os

# Add parent dir to path so we can import ml_core.model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import glob
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from ml_core.model import MobileNetV2_CBAM

class KaggleFlatDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Support common image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, ext)))
            
        print(f"Found {len(self.image_paths)} images in {root_dir}")
        if len(self.image_paths) == 0:
            print(f"WARNING: No images found in {root_dir}. Check path!")
            
        for path in self.image_paths:
            filename = os.path.basename(path).lower()
            if filename.startswith('fire'):
                self.labels.append(0) # 0 for Fire
            elif filename.startswith('nofire'):
                self.labels.append(1) # 1 for No_Fire
            else:
                self.labels.append(-1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = torch.zeros(3, 224, 224) 
            
        return image, label, img_path

def evaluate_on_kaggle(model_path, data_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = KaggleFlatDataset(data_dir, transform=transform)
    if len(dataset) == 0:
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = MobileNetV2_CBAM(num_classes=2)
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return
    else:
        print(f"Error: Model not found at {model_path}")
        return
        
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Starting evaluation...")
    with torch.no_grad():
        for i, (inputs, labels, paths) in enumerate(dataloader):
            if i % 5 == 0:
                print(f"Processing batch {i}...")
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Filter out invalid labels (-1)
    # Also valid labels are 0 and 1
    valid_mask = [l != -1 for l in all_labels]
    
    if not any(valid_mask):
        print("No valid labels found (fire_*.jpg or nofire_*.jpg)")
        return

    final_preds = [p for p, m in zip(all_preds, valid_mask) if m]
    final_labels = [l for l, m in zip(all_labels, valid_mask) if m]
    
    accuracy = accuracy_score(final_labels, final_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(final_labels, final_preds, average='binary', pos_label=0) # Fire is 0
    
    print("-" * 30)
    print("Evaluation Results on Kaggle Dataset")
    print("-" * 30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision (Fire): {precision:.4f}")
    print(f"Recall (Fire):    {recall:.4f}")
    print(f"F1 Score (Fire):  {f1:.4f}")
    print("-" * 30)
    
    report_content = f"""# Generalization Report - Kaggle Dataset

## Dataset Info
- **Source**: `{data_dir}`
- **Total Images Processed**: {len(final_labels)}
- **Classes**: Fire vs No_Fire

## Metrics
| Metric | Value |
| :--- | :--- |
| **Accuracy** | **{accuracy:.4f}** |
| **Precision (Fire)** | {precision:.4f} |
| **Recall (Fire)** | {recall:.4f} |
| **F1 Score** | {f1:.4f} |

## Interpretation
- **Accuracy**: Overall correctness on the new domain.
- **Recall**: Proportion of actual fires correctly detected (Critical for safety).
- **Precision**: Proportion of predicted fires that were actually fires (False Alarm rate).
"""
    
    with open("generalization_report.md", "w") as f:
        f.write(report_content)
    print("Report saved to generalization_report.md")

if __name__ == "__main__":
    # Adjust paths relative to where script is run
    MODEL_PATH = 'ml_core/models/mv2_cbam_best.pth' 
    DATA_DIR = 'dataset/Forest Fire Dataset kaggle/Testing' # Use absolute path or relative to project root
    
    evaluate_on_kaggle(MODEL_PATH, DATA_DIR)
