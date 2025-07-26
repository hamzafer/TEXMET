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

def plot_mask_properties(mask_dir, output_plot_dir):
    """Analyzes and plots individual mask images for covered area and irregularity."""
    if not os.path.exists(mask_dir):
        print(f"Error: Mask directory not found at '{mask_dir}'")
        return
    os.makedirs(output_plot_dir, exist_ok=True)

    mask_files_paths = glob.glob(os.path.join(mask_dir, '*.png')) 
    if not mask_files_paths:
        print("No mask images found to analyze.")
        return

    print(f"Generating plots for {len(mask_files_paths)} mask images...")
    
    for i, mask_path in enumerate(mask_files_paths):
        file_name = os.path.basename(mask_path)
        print(f"  Processing {i + 1}/{len(mask_files_paths)}: {file_name}")

        try:
            with Image.open(mask_path) as img:
                gray_img = img.convert('L')
                mask_array = np.array(gray_img)
                mask_array = (mask_array > 0).astype(np.uint8)

                total_pixels = mask_array.size
                visible_pixels = np.count_nonzero(mask_array)
                
                if total_pixels == 0:
                    covered_area_percentage = 0.0
                    compactness = 0.0
                else:
                    covered_area_percentage = (total_pixels - visible_pixels) / total_pixels * 100
                    area = visible_pixels
                    perimeter = calculate_perimeter_optimized(mask_array)

                    if perimeter == 0:
                        compactness = 0.0
                    else:
                        compactness = (4 * np.pi * area) / (perimeter**2)

                # Plotting
                plt.figure(figsize=(6, 6))
                plt.imshow(mask_array, cmap='gray')
                plt.title(f"Mask: {file_name}\nCovered Area: {covered_area_percentage:.2f}%\nCompactness: {compactness:.4f}")
                plt.axis('off')
                plt.tight_layout()
                
                plot_file_name = os.path.splitext(file_name)[0] + '_properties.png'
                plt.savefig(os.path.join(output_plot_dir, plot_file_name))
                plt.close() # Close the figure to free memory

        except Exception as e:
            print(f"Could not process or plot {file_name}: {e}")

    print("\nPlot generation complete.")

if __name__ == "__main__":
    mask_directory = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/original_datasets/oseberg/masks"
    output_plot_directory = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/visuals/oseberg_mask_plots"
    plot_mask_properties(mask_directory, output_plot_directory)