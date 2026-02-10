
import os
import shutil
import glob

# Source Kaggle Training Data
KAGGLE_TRAIN_DIR = r"d:\projects\MV2_CBAM_Project\dataset\Forest Fire Dataset kaggle\Training"

# Destination Project Training Data
PROJECT_TRAIN_DIR = r"d:\projects\MV2_CBAM_Project\dataset\Training"

# Mappings (Source -> Destination)
MAPPING = {
    "fire": "Fire",
    "nofire": "No_Fire"
}

def merge_data():
    print("Starting Dataset Merge...")
    
    total_copied = 0
    
    for src_sub, dest_sub in MAPPING.items():
        src_path = os.path.join(KAGGLE_TRAIN_DIR, src_sub)
        dest_path = os.path.join(PROJECT_TRAIN_DIR, dest_sub)
        
        if not os.path.exists(src_path):
            print(f"Warning: Source directory {src_path} does not exist.")
            continue
            
        # Get all images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
        images = []
        for ext in extensions:
            images.extend(glob.glob(os.path.join(src_path, ext)))
            
        print(f"Merging {len(images)} images from '{src_sub}' into '{dest_sub}'...")
        
        for img in images:
            # Create a unique filename to avoid overwrites (prefix with kaggle_)
            basename = os.path.basename(img)
            new_name = f"kaggle_{basename}"
            dest_file = os.path.join(dest_path, new_name)
            
            try:
                shutil.copy2(img, dest_file)
                total_copied += 1
            except Exception as e:
                print(f"Error copying {img}: {e}")
                
    print(f"Merge Complete! Total images added: {total_copied}")

if __name__ == "__main__":
    merge_data()
