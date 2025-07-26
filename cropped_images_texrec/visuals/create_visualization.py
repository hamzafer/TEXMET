import matplotlib.pyplot as plt
import os
import re
import matplotlib.image as mpimg

# Define the directory and file paths
image_dir = '/home/user1/Desktop/HAMZA/THESIS/TEXMET/cropped_images_texrec'
success_original = os.path.join(image_dir, 'img_002_(j)_img_002_original.png')
success_random = os.path.join(image_dir, 'img_002_(j)_img_002_random.png')
success_reconstructed_path = os.path.join(image_dir, 'img_002_(j)_img_002_reconstructed_pAcc=1_patchAcc=1.00.png')
failure_original = os.path.join(image_dir, 'img_002_(d)_img_002_original.png')
failure_random = os.path.join(image_dir, 'img_002_(d)_img_002_random.png')
failure_reconstructed_path = os.path.join(image_dir, 'img_002_(d)_img_002_reconstructed_pAcc=0_patchAcc=0.00.png')

# Function to extract accuracy from filename
def get_accuracy_from_filename(filename):
    match = re.search(r'pAcc=([\d\.]+)_patchAcc=([\d\.]+)', filename)
    if match:
        return f"pAcc={match.group(1)}, patchAcc={match.group(2)}"
    return ""

# Get accuracies
success_accuracy = get_accuracy_from_filename(os.path.basename(success_reconstructed_path))
failure_accuracy = get_accuracy_from_filename(os.path.basename(failure_reconstructed_path))


# Create a figure with 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
title_fontsize = 22

# --- Row 1: Success Case ---
# Load and display the images for the success case
axes[0, 0].imshow(mpimg.imread(success_original))
axes[0, 0].set_title('Original Image (Success)', fontsize=title_fontsize)
axes[0, 0].axis('off')

axes[0, 1].imshow(mpimg.imread(success_random))
axes[0, 1].set_title('Shuffled Image (Success)', fontsize=title_fontsize)
axes[0, 1].axis('off')

axes[0, 2].imshow(mpimg.imread(success_reconstructed_path))
axes[0, 2].set_title(f'Reconstructed (Success)\n{success_accuracy}', fontsize=title_fontsize)
axes[0, 2].axis('off')

# --- Row 2: Failure Case ---
# Load and display the images for the failure case
axes[1, 0].imshow(mpimg.imread(failure_original))
axes[1, 0].set_title('Original Image (Failure)', fontsize=title_fontsize)
axes[1, 0].axis('off')

axes[1, 1].imshow(mpimg.imread(failure_random))
axes[1, 1].set_title('Shuffled Image (Failure)', fontsize=title_fontsize)
axes[1, 1].axis('off')

axes[1, 2].imshow(mpimg.imread(failure_reconstructed_path))
axes[1, 2].set_title(f'Reconstructed (Failure)\n{failure_accuracy}', fontsize=title_fontsize)
axes[1, 2].axis('off')

# Adjust layout and save the figure
plt.tight_layout(pad=3.0)
plt.savefig('/home/user1/Desktop/HAMZA/THESIS/TEXMET/reconstruction_visualization.png')
plt.close()

print("Visualization saved as reconstruction_visualization.png")