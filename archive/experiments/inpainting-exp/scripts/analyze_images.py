import os
import numpy as np
from PIL import Image
import pandas as pd

# Path to the directory containing the selected images
image_dir = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/selected_images_v4"

def analyze_images(directory):
    """Analyzes images in a directory for pixel stats and dimensions."""
    if not os.path.exists(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("No images found to analyze.")
        return

    # Lists to store stats for each image
    stats = []

    print(f"Analyzing {len(image_files)} images...")
    for i, file_name in enumerate(image_files):
        img_path = os.path.join(directory, file_name)
        try:
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    continue
                
                img_array = np.array(img, dtype=np.float32) / 255.0 # Normalize to [0, 1]
                
                # Per-channel stats
                mean_r, mean_g, mean_b = np.mean(img_array, axis=(0, 1))
                std_r, std_g, std_b = np.std(img_array, axis=(0, 1))
                
                # Dimensions
                width, height = img.size
                
                stats.append({
                    'file': file_name,
                    'width': width,
                    'height': height,
                    'mean_r': mean_r,
                    'mean_g': mean_g,
                    'mean_b': mean_b,
                    'std_r': std_r,
                    'std_g': std_g,
                    'std_b': std_b
                })
        except Exception as e:
            print(f"Could not process {file_name}: {e}")

    # Create a pandas DataFrame for easy analysis
    df = pd.DataFrame(stats)

    print("\n--- Image Statistics Summary ---")
    print(f"Total images analyzed: {len(df)}")

    # Dimension statistics
    print("\n1. Image Dimensions (in pixels):")
    print(f"   - Average Size: {int(df['width'].mean())} x {int(df['height'].mean())}")
    print(f"   - Smallest Image: {df['width'].min()} x {df['height'].min()}")
    print(f"   - Largest Image:  {df['width'].max()} x {df['height'].max()}")

    # Pixel value statistics (normalized to [0, 1])
    print("\n2. Pixel Color Distribution (Normalized to [0, 1]):")
    print("   - Average Mean (Brightness):")
    print(f"     - Red:   {df['mean_r'].mean():.4f}")
    print(f"     - Green: {df['mean_g'].mean():.4f}")
    print(f"     - Blue:  {df['mean_b'].mean():.4f}")
    print("   - Average Std Dev (Contrast):")
    print(f"     - Red:   {df['std_r'].mean():.4f}")
    print(f"     - Green: {df['std_g'].mean():.4f}")
    print(f"     - Blue:  {df['std_b'].mean():.4f}")

if __name__ == "__main__":
    analyze_images(image_dir)
