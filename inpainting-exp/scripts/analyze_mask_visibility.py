#!/home/user1/miniconda3/envs/diffassemble/bin/python
import os
import numpy as np
from PIL import Image

def analyze_mask_visibility(mask_dir, min_visibility=0.60, max_visibility=0.70):
    """Analyzes mask images for visibility (percentage of non-zero pixels)."""
    if not os.path.exists(mask_dir):
        print(f"Error: Directory not found at '{mask_dir}'")
        return

    mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not mask_files:
        print("No mask images found to analyze.")
        return

    print(f"Analyzing {len(mask_files)} mask images for visibility between {min_visibility*100:.0f}% and {max_visibility*100:.0f}%...")
    
    matching_masks = []

    for file_name in mask_files:
        mask_path = os.path.join(mask_dir, file_name)
        try:
            with Image.open(mask_path) as img:
                # Convert to grayscale to ensure single channel for pixel counting
                # Assuming masks are binary (0 or 255) or grayscale
                gray_img = img.convert('L')
                mask_array = np.array(gray_img)
                
                # Count non-zero pixels (assuming mask is 0 for background, >0 for foreground)
                visible_pixels = np.count_nonzero(mask_array)
                total_pixels = mask_array.size
                
                if total_pixels == 0:
                    visibility_percentage = 0.0
                else:
                    visibility_percentage = visible_pixels / total_pixels
                
                if min_visibility <= visibility_percentage <= max_visibility:
                    matching_masks.append((file_name, visibility_percentage))

        except Exception as e:
            print(f"Could not process {file_name}: {e}")

    if matching_masks:
        print("\nMasks with visibility between 60% and 70%:")
        for mask_name, visibility in matching_masks:
            print(f" - {mask_name}: {visibility*100:.2f}%")
    else:
        print("\nNo masks found with visibility between 60% and 70%.")

if __name__ == "__main__":
    mask_directory = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/oseberg/masks_only"
    analyze_mask_visibility(mask_directory)