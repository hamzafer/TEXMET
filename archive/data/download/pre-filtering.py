import os
import cv2
import json
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import imagehash

def automatic_prefilter():
    """Remove obvious bad images automatically"""
    
    # Load your dataset
    with open("../data/FINAL_CORRECTED_MET_TEXTILES_DATASET/objects_with_images_only/ALL_TEXTILES_AND_TAPESTRIES_WITH_IMAGES_20250705_230315.json", "r") as f:
        data = json.load(f)
    
    images_dir = "MET_TEXTILES_BULLETPROOF_DATASET/images"
    bad_ids = set()
    stats = {
        "corrupt": 0,
        "missing": 0, 
        "weird_aspect": 0,
        "too_small": 0,
        "blurry": 0,
        "total_checked": 0
    }
    
    print("🔍 Starting automatic pre-filtering...")
    
    for obj in tqdm(data, desc="Checking images"):
        obj_id = str(obj['objectID'])
        stats["total_checked"] += 1
        
        # Find image file
        img_files = [f for f in os.listdir(images_dir) if f.startswith(obj_id)]
        
        if not img_files:
            bad_ids.add(obj_id)
            stats["missing"] += 1
            continue
            
        img_path = os.path.join(images_dir, img_files[0])
        
        try:
            # 1. Check if file opens
            img = cv2.imread(img_path)
            if img is None:
                bad_ids.add(obj_id)
                stats["corrupt"] += 1
                continue
                
            h, w = img.shape[:2]
            
            # 2. Check size
            if min(h, w) < 200:
                bad_ids.add(obj_id)
                stats["too_small"] += 1
                continue
                
            # 3. Check aspect ratio (reject extreme panoramas)
            if max(h, w) / min(h, w) > 5:
                bad_ids.add(obj_id)
                stats["weird_aspect"] += 1
                continue
                
            # 4. Check blur (Laplacian variance)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < 50:
                bad_ids.add(obj_id)
                stats["blurry"] += 1
                continue
                
        except Exception as e:
            bad_ids.add(obj_id)
            stats["corrupt"] += 1
    
    # Create clean dataset
    clean_data = [obj for obj in data if str(obj['objectID']) not in bad_ids]
    try:
        print(f"\n📊 PRE-FILTERING RESULTS:")
        print(f"Original dataset: {len(data):,} objects")
        print(f"Removed images: {len(bad_ids):,}")
        print(f"  - Missing files: {stats['missing']:,}")
        print(f"  - Corrupt files: {stats['corrupt']:,}")
        print(f"  - Too small: {stats['too_small']:,}")
        print(f"  - Weird aspect: {stats['weird_aspect']:,}")
        print(f"  - Blurry: {stats['blurry']:,}")
        print(f"Clean dataset: {len(clean_data):,} objects")

        # Save clean dataset
        with open("clean_textiles_dataset.json", "w") as f:
            json.dump(clean_data, f, indent=2)
    except Exception as e:
        print(f"An error occurred during result reporting or saving: {e}")
    return clean_data, bad_ids

if __name__ == "__main__":
    clean_data, bad_ids = automatic_prefilter()