
import os
import shutil
import os
import shutil
import glob

def merge_additional_data():
    # Source: The new dataset path
    source_root = 'dataset/FOREST_FIRE_DATASET'
    
    # Destination: Our main training directory
    dest_root = 'dataset/Training'
    
    # Mapping: Source Class Name -> Destination Class Name
    class_mapping = {
        'fire': 'Fire',
        'non fire': 'No_Fire'
    }
    
    # Subdirectories to merge (train and test)
    subdirs = ['train', 'test']
    
    print(f"Merging data from {source_root} into {dest_root}...")
    
    total_moved = 0
    
    for subdir in subdirs:
        current_source = os.path.join(source_root, subdir)
        if not os.path.exists(current_source):
            print(f"Skipping {current_source} (not found)")
            continue
            
        # Iterate over classes in the source subdirectory
        for src_class, dest_class in class_mapping.items():
            src_class_path = os.path.join(current_source, src_class)
            dest_class_path = os.path.join(dest_root, dest_class)
            
            # Get all images
            src_class_path = os.path.join(current_source, src_class)
            if not os.path.exists(src_class_path):
                 print(f"Skipping {src_class} in {subdir} (not found)")
                 continue

            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                images.extend(glob.glob(os.path.join(src_class_path, ext)))
            
            print(f"Found {len(images)} images for {src_class} in {subdir}")
            
            for img_path in images:
                filename = os.path.basename(img_path)
                # Create a unique name to avoid conflicts
                # prefix with dataset name and partition
                new_filename = f"aug_forest_{subdir}_{filename}"
                dest_path = os.path.join(dest_class_path, new_filename)
                
                try:
                    shutil.copy2(img_path, dest_path)
                    total_moved += 1
                except Exception as e:
                    print(f"Error copying {filename}: {e}")
                    
    print(f"\nSuccessfully merged {total_moved} new images into Training set.")

if __name__ == "__main__":
    merge_additional_data()
