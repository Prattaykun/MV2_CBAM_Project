
import argparse
import os
from ml_core.train import train_model

def main():
    parser = argparse.ArgumentParser(description='Train MobileNetV2-CBAM Fire Detection Model')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Dataset directory '{args.data_dir}' not found.")
        return
        
    print(f"Starting training with:\n Data: {args.data_dir}\n Epochs: {args.epochs}\n Batch: {args.batch_size}\n LR: {args.lr}")
    
    train_model(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

if __name__ == "__main__":
    main()
