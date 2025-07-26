

import os
import shutil

def segregate_images_and_masks(source_dir, images_dir, masks_dir):
    """Segregates image and mask files into separate directories based on simple naming convention."""
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found at '{source_dir}'")
        return

    # Create destination directories if they don't exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    files = os.listdir(source_dir)
    moved_images = 0
    moved_masks = 0

    for f in files:
        src_path = os.path.join(source_dir, f)
        if os.path.isfile(src_path):
            # Simple check: if '_mask' is in the filename, it's a mask
            if '_mask' in f.lower():
                shutil.move(src_path, os.path.join(masks_dir, f))
                moved_masks += 1
            else:
                shutil.move(src_path, os.path.join(images_dir, f))
                moved_images += 1
    
    print(f"Segregation complete.")
    print(f"Moved {moved_images} images to '{images_dir}'.")
    print(f"Moved {moved_masks} masks to '{masks_dir}'.")

    # Verify counts
    print("\n--- Verification ---")
    print(f"Count in images_only: {len(os.listdir(images_dir))}")
    print(f"Count in masks_only: {len(os.listdir(masks_dir))}")

if __name__ == "__main__":
    source_directory = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/all_unique_pngs_with_masks"
    images_directory = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/images_only"
    masks_directory = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/masks_only"
    segregate_images_and_masks(source_directory, images_directory, masks_directory)

