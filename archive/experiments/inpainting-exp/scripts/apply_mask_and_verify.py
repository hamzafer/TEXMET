#!/home/user1/miniconda3/envs/diffassemble/bin/python
import os
import numpy as np
from PIL import Image

def apply_mask_and_verify(image_dir, mask_path, output_dir, min_visibility=0.60, max_visibility=0.70):
    """Applies a given mask to images, saves them, and verifies visible area."""
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found at '{image_dir}'")
        return
    if not os.path.exists(mask_path):
        print(f"Error: Mask file not found at '{mask_path}'")
        return

    os.makedirs(output_dir, exist_ok=True)

    try:
        base_mask = Image.open(mask_path).convert('L') # Load mask as grayscale
    except Exception as e:
        print(f"Error loading mask {mask_path}: {e}")
        return

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found in the source directory to mask.")
        return

    print(f"Processing {len(image_files)} images...")
    
    all_visible_areas = []

    for file_name in image_files:
        img_path = os.path.join(image_dir, file_name)
        output_path = os.path.join(output_dir, file_name)

        try:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)

            # Resize mask to image dimensions
            resized_mask = base_mask.resize(img.size, Image.LANCZOS)
            mask_array = np.array(resized_mask) / 255.0 # Normalize mask to 0-1

            # Apply mask: set masked-out areas to black
            # Assuming mask is 0 for masked-out, 1 for visible
            # If mask is inverted (0 visible, 1 masked), use (1 - mask_array)
            masked_img_array = img_array * np.expand_dims(mask_array, axis=2)
            masked_img = Image.fromarray(masked_img_array.astype(np.uint8))
            masked_img.save(output_path)

            # Verify visible area of the *masked* image
            # Count non-black pixels (where at least one channel is > 0)
            visible_pixels_count = np.sum(np.any(masked_img_array > 0, axis=2))
            total_pixels = masked_img_array.shape[0] * masked_img_array.shape[1]
            
            if total_pixels == 0:
                current_visibility = 0.0
            else:
                current_visibility = visible_pixels_count / total_pixels

            all_visible_areas.append(current_visibility)

            if min_visibility <= current_visibility <= max_visibility:
                print(f" - {file_name}: Masked and saved. Visible area: {current_visibility*100:.2f}% (within range)")
            else:
                print(f" - {file_name}: Masked and saved. Visible area: {current_visibility*100:.2f}% (OUTSIDE range)")

        except Exception as e:
            print(f"Could not process {file_name}: {e}")

    print("\nMasking and verification complete.")

    if all_visible_areas:
        average_visible_area = np.mean(all_visible_areas) * 100
        average_masked_area = 100 - average_visible_area
        print("\n--- Summary Statistics ---")
        print(f"Average Visible Area: {average_visible_area:.2f}%")
        print(f"Average Masked Area: {average_masked_area:.2f}%")
    else:
        print("No images were successfully processed to calculate statistics.")

if __name__ == "__main__":
    image_source_directory = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/imagenet1k"
    mask_file_path = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/oseberg/masks_only/img_123_mask.png"
    output_masked_images_directory = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/masked_imagenet1k"
    
    apply_mask_and_verify(image_source_directory, mask_file_path, output_masked_images_directory)
