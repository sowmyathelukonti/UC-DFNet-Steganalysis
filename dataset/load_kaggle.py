import os
import zipfile
import shutil
import cv2
import numpy as np

def setup_kaggle_credentials():
    """
    Checks if Kaggle credentials are configured.
    If not, guides the user on how to set it up.
    """
    home = os.path.expanduser("~")
    kaggle_dir = os.path.join(home, ".kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    
    if not os.path.exists(kaggle_json):
        print("\n" + "="*70)
        print("⚠️  KAGGLE API CREDENTIALS NOT FOUND!")
        print("="*70)
        print(f"To download this dataset automatically, please:")
        print(f"1. Log in to your Kaggle account (https://www.kaggle.com).")
        print(f"2. Go to your Account settings (https://www.kaggle.com/settings).")
        print(f"3. Scroll to the API section and click 'Create New Token'.")
        print(f"   This downloads a file named 'kaggle.json'.")
        print(f"4. Copy 'kaggle.json' to the folder: {kaggle_dir}")
        print("="*70 + "\n")
        return False
    return True

def download_and_integrate_kaggle(limit_per_class=500, target_dir="data"):
    """
    Downloads the 'marcozuppelli/stegoimagesdataset' from Kaggle or extracts a local ZIP file,
    unzips, resizes, and copies images into our training pipeline.
    """
    temp_download_dir = "data_kaggle_temp"
    os.makedirs(temp_download_dir, exist_ok=True)
    
    # 1. Check if the dataset zip is already present locally (manual browser download fallback)
    local_zip = "stegoimagesdataset.zip"
    if os.path.exists(local_zip):
        print(f"Detected local dataset ZIP '{local_zip}'. Extracting locally...")
        try:
            with zipfile.ZipFile(local_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_download_dir)
            print("Local extraction complete!")
        except Exception as e:
            print(f"Error unzipping local file: {e}")
            return False
    else:
        # Fall back to automated Kaggle API download
        if not setup_kaggle_credentials():
            print("Kaggle credentials not found and no local 'stegoimagesdataset.zip' detected.")
            print("Please place the downloaded ZIP file in this directory or set up your Kaggle API key.")
            return False
            
        import kaggle
        dataset_name = "marcozuppelli/stegoimagesdataset"
        print(f"Downloading dataset '{dataset_name}' from Kaggle via API...")
        try:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(dataset_name, path=temp_download_dir, unzip=True)
            print("API Download and extraction complete!")
        except Exception as e:
            print(f"Error downloading from Kaggle API: {e}")
            return False
        
    # Setup target directories
    cover_target = os.path.join(target_dir, "cover")
    stego_target = os.path.join(target_dir, "stego")
    os.makedirs(cover_target, exist_ok=True)
    os.makedirs(stego_target, exist_ok=True)
    
    print("\nScanning extracted folders for cover and stego files...")
    
    # We recursively search for images
    cover_files = []
    stego_files = []
    
    for root, dirs, files in os.walk(temp_download_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                # Determine class based on folder names in path
                path_lower = full_path.lower()
                if "cover" in path_lower:
                    cover_files.append(full_path)
                elif "stego" in path_lower:
                    stego_files.append(full_path)
                    
    print(f"Found {len(cover_files)} Cover images and {len(stego_files)} Stego images in raw dataset.")
    
    # Select subset to integrate
    selected_covers = cover_files[:limit_per_class]
    selected_stegos = stego_files[:limit_per_class]
    
    # Copy & Resize on-the-fly to 256x256 (matches model requirements)
    print(f"Integrating and resizing first {len(selected_covers)} cover images...")
    for idx, path in enumerate(selected_covers):
        try:
            img = cv2.imread(path)
            if img is not None:
                resized = cv2.resize(img, (256, 256))
                # Save as cover_k_0000.png
                target_path = os.path.join(cover_target, f"cover_k_{idx:05d}.png")
                cv2.imwrite(target_path, resized)
        except Exception as e:
            print(f"Error copying {path}: {e}")
            
    print(f"Integrating and resizing first {len(selected_stegos)} stego images...")
    for idx, path in enumerate(selected_stegos):
        try:
            img = cv2.imread(path)
            if img is not None:
                resized = cv2.resize(img, (256, 256))
                # Save as stego_k_0000.png
                target_path = os.path.join(stego_target, f"stego_k_{idx:05d}.png")
                cv2.imwrite(target_path, resized)
        except Exception as e:
            print(f"Error copying {path}: {e}")
            
    print("\nClean-up: Removing raw downloaded Kaggle files to save disk space...")
    shutil.rmtree(temp_download_dir, ignore_errors=True)
    
    print(f"Success! Integrated {len(selected_covers)} cover and {len(selected_stegos)} stego images into '{target_dir}' directory.")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kaggle Stego Dataset Downloader & Integrator")
    parser.add_argument("--limit", type=int, default=500, help="Number of images per class to load (to prevent massive storage usage)")
    parser.add_argument("--target", type=str, default="data", help="Target training data folder")
    
    args = parser.parse_args()
    
    # Ensure kaggle package is installed
    try:
        import kaggle
    except ImportError:
        print("Kaggle python package not found. Installing via pip...")
        import subprocess
        subprocess.check_call([os.sys.executable, "-m", "pip", "install", "kaggle"])
        
    download_and_integrate_kaggle(limit_per_class=args.limit, target_dir=args.target)
