
import os
import zipfile

def create_zip(output_filename):
    print(f"Creating submission zip: {output_filename}")
    
    # Files/Folders to include
    include_paths = [
        'backend',
        'frontend',
        'ml_core',
        'docs',
        'README.md',
        'requirements.txt',
        'run_project.ps1',
        'train_runner.py',
        'merge_dataset.py'
    ]
    
    # Patterns/Files to exclude within those folders
    exclude_patterns = [
        '__pycache__',
        'node_modules',
        '.next',
        '.env',
        'venv',
        '.git',
        '.vscode',
        'mv2_cbam_base.pth', # Exclude backup model
        # 'mv2_cbam_best.pth' # Keep best model if small enough, but usually better to exclude for email submission
    ]
    
    # Create the zip
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in include_paths:
            if os.path.isfile(item):
                zipf.write(item, arcname=item)
                print(f"Added file: {item}")
            elif os.path.isdir(item):
                for root, dirs, files in os.walk(item):
                    # Filter out excluded directories
                    dirs[:] = [d for d in dirs if d not in exclude_patterns]
                    
                    for file in files:
                        if file in exclude_patterns or file.endswith('.pyc'):
                            continue
                            
                        # Double check logical exclusion
                        if 'node_modules' in root or '__pycache__' in root:
                            continue
                            
                        file_path = os.path.join(root, file)
                        # Relative path for the zip
                        arcname = os.path.relpath(file_path, os.getcwd())
                        zipf.write(file_path, arcname=arcname)
                        # print(f"Added: {arcname}") # Verbose
        
    print("Zip creation complete!")

if __name__ == "__main__":
    create_zip("Submission_Prattay_Roy_Chowdhury.zip")
