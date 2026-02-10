
import os
import torch
import pandas as pd
from ml_core.predict import FirePredictor


def verify_misclassifications():
    # Setup
    model_path = 'ml_core/models/mv2_cbam_best.pth'
    test_dir = 'dataset/Forest Fire Dataset kaggle/Testing'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    predictor = FirePredictor(model_path=model_path, device='cpu') # Use CPU for local test
    
    # Get all 'nofire' images (these are the ground truth negatives)
    # We want to see which ones the model thinks are 'Fire' (False Positives)
    nofire_images = [f for f in os.listdir(test_dir) if f.lower().startswith('nofire') and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(nofire_images)} 'No_Fire' images to test.")
    
    false_positives = []
    
    print("Running inference...")
    for i, img_name in enumerate(nofire_images):
        if i % 10 == 0:
            print(f"Processing {i}/{len(nofire_images)}...", end='\r')
        img_path = os.path.join(test_dir, img_name)
        try:
            prediction, confidence = predictor.predict(img_path)
            
            # Check for False Positive (Ground Truth: No_Fire, Predicted: Fire)
            if prediction == 'Fire':
                false_positives.append({
                    'image': img_name,
                    'confidence': confidence
                })
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    # Report
    print("\n" + "="*50)
    print(f"ANALYSIS RESULT: Sunset/Red Bias Check")
    print("="*50)
    print(f"Total 'No_Fire' Images Tested: {len(nofire_images)}")
    print(f"False Positives (Misclassified as Fire): {len(false_positives)}")
    print(f"False Positive Rate: {len(false_positives)/len(nofire_images)*100:.2f}%")
    print("-" * 50)
    
    if false_positives:
        print("Top 10 Misclassified Images (Highest Confidence):")
        # Sort by confidence descending
        false_positives.sort(key=lambda x: x['confidence'], reverse=True)
        for fp in false_positives[:10]:
            print(f"- {fp['image']}: {fp['confidence']:.4f}")
            
        # Save list for potential hard mining
        df = pd.DataFrame(false_positives)
        df.to_csv('false_positives.csv', index=False)
        print(f"\nFull list saved to 'false_positives.csv'")
    else:
        print("Great news! No false positives detected on this set.")

if __name__ == "__main__":
    verify_misclassifications()
