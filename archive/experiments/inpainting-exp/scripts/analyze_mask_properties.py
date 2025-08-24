#!/home/user1/miniconda3/envs/diffassemble/bin/python
import matplotlib
matplotlib.use('Agg') # Use the Agg backend for non-interactive plotting

import os
import numpy as np
from PIL import Image
import glob
from scipy.ndimage import binary_erosion
import matplotlib.pyplot as plt

def calculate_perimeter_optimized(mask_array):
    """Calculates the perimeter of a binary mask using optimized erosion method."""
    eroded_mask = binary_erosion(mask_array, structure=np.ones((3,3))).astype(mask_array.dtype)
    perimeter_mask = mask_array - eroded_mask
    return np.count_nonzero(perimeter_mask)

def analyze_and_plot_mask_properties(mask_dir, output_plot_dir):
    """Analyzes mask images for masked area and irregularity (compactness) and generates aggregate plots."""
    if not os.path.exists(mask_dir):
        print(f"Error: Mask directory not found at '{mask_dir}'")
        return
    os.makedirs(output_plot_dir, exist_ok=True)

    mask_files_paths = glob.glob(os.path.join(mask_dir, '*.png')) 
    if not mask_files_paths:
        print("No mask images found to analyze.")
        return

    print(f"Analyzing {len(mask_files_paths)} mask images for masked area and irregularity...")
    
    all_masked_areas = [] # Changed variable name
    all_compactness_values = []

    for i, mask_path in enumerate(mask_files_paths):
        if (i + 1) % 10 == 0: # Log every 10 files
            print(f"  Processing mask {i + 1}/{len(mask_files_paths)}: {os.path.basename(mask_path)}")

        try:
            with Image.open(mask_path) as img:
                gray_img = img.convert('L')
                mask_array = np.array(gray_img)
                mask_array = (mask_array > 0).astype(np.uint8)

                total_pixels = mask_array.size
                visible_pixels = np.count_nonzero(mask_array)
                
                if total_pixels == 0:
                    masked_area_percentage = 0.0 # Changed variable name
                    compactness = 0.0
                else:
                    masked_area_percentage = (total_pixels - visible_pixels) / total_pixels * 100 # Changed variable name
                    area = visible_pixels
                    perimeter = calculate_perimeter_optimized(mask_array)

                    if perimeter == 0:
                        compactness = 0.0
                    else:
                        compactness = (4 * np.pi * area) / (perimeter**2)

                all_masked_areas.append(masked_area_percentage) # Changed variable name
                all_compactness_values.append(compactness)

        except Exception as e:
            print(f"Could not process {os.path.basename(mask_path)}: {e}")

    print("\nMask analysis complete.")

    if all_masked_areas:
        avg_masked_area = np.mean(all_masked_areas) # Changed variable name
        avg_compactness = np.mean(all_compactness_values)
        
        print("\n--- Summary Statistics (Oseberg Masks) ---")
        print(f"Average Masked Area: {avg_masked_area:.2f}%") # Changed output text
        print(f"Average Compactness (Irregularity): {avg_compactness:.4f}")
        print("  (Note: Higher compactness means more circular/less irregular. Max 1.0 for perfect circle)")

        # Generate and save aggregate plots
        plt.figure(figsize=(12, 5))

        # Histogram for Masked Area
        plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
        plt.hist(all_masked_areas, bins=10, edgecolor='black') # Changed variable name
        plt.title('Distribution of Masked Area') # Changed title
        plt.xlabel('Masked Area (%)') # Changed label
        plt.ylabel('Number of Masks')
        plt.grid(axis='y', alpha=0.75)

        # Histogram for Compactness
        plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
        plt.hist(all_compactness_values, bins=10, edgecolor='black')
        plt.title('Distribution of Compactness')
        plt.xlabel('Compactness Value')
        plt.ylabel('Number of Masks')
        plt.grid(axis='y', alpha=0.75)

        plt.tight_layout()
        plt.savefig(os.path.join(output_plot_dir, 'oseberg_mask_distributions.png'))
        plt.close()
        print(f"Aggregate plots saved to {os.path.join(output_plot_dir, 'oseberg_mask_distributions.png')}")

    else:
        print("No masks were successfully processed to calculate statistics.")

if __name__ == "__main__":
    mask_directory = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/original_datasets/oseberg/masks"
    output_plot_directory = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/visuals/oseberg_mask_plots"
    analyze_and_plot_mask_properties(mask_directory, output_plot_directory)
