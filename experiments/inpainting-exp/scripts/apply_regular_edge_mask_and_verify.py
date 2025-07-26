#!/home/user1/miniconda3/envs/diffassemble/bin/python
import os
import numpy as np
from PIL import Image, ImageDraw

def apply_regular_edge_mask_and_verify(
    image_dir,
    output_dir,
    masks_output_dir, # New parameter for saving generated masks
    target_visible_area_percentage=63.48, 
    min_visibility_range=0.60, 
    max_visibility_range=0.70  
):
    """Applies a central visible square mask (masking edges) to images, saves them, and verifies visible area."""
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found at '{image_dir}'")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(masks_output_dir, exist_ok=True) # Create directory for masks

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found in the source directory to mask.")
        return

    print(f"Processing {len(image_files)} images with a central visible square mask (masking edges)...")
    
    all_visible_areas = []

    for file_name in image_files:
        img_path = os.path.join(image_dir, file_name)
        output_img_path = os.path.join(output_dir, file_name)
        # Construct mask output path (e.g., image_name_mask.png)
        mask_file_name = os.path.splitext(file_name)[0] + "_mask.png"
        output_mask_path = os.path.join(masks_output_dir, mask_file_name)

        try:
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            width, height = img.size

            # Calculate side length of the central VISIBLE square
            total_image_area = width * height
            visible_area_pixels = (target_visible_area_percentage / 100.0) * total_image_area
            
            visible_side = int(np.sqrt(visible_area_pixels))

            visible_side = min(visible_side, width, height)

            # Create a black mask (masked area) and draw a white rectangle (visible area) in the center
            mask = Image.new('L', (width, height), 0) # Black background (masked)
            draw = ImageDraw.Draw(mask)

            # Calculate coordinates for the central white square
            left = (width - visible_side) // 2
            top = (height - visible_side) // 2
            right = left + visible_side
            bottom = top + visible_side

            draw.rectangle([left, top, right, bottom], fill=255) # Draw white square (visible)

            # Save the generated mask
            mask.save(output_mask_path)

            mask_array = np.array(mask) / 255.0 # Normalize mask to 0-1

            # Apply mask: set masked-out areas to black
            masked_img_array = img_array * np.expand_dims(mask_array, axis=2)
            masked_img = Image.fromarray(masked_img_array.astype(np.uint8))
            masked_img.save(output_img_path)

            # Verify visible area of the *masked* image
            visible_pixels_count = np.sum(np.any(masked_img_array > 0, axis=2))
            total_pixels = masked_img_array.shape[0] * masked_img_array.shape[1]
            
            if total_pixels == 0:
                current_visibility = 0.0
            else:
                current_visibility = visible_pixels_count / total_pixels

            all_visible_areas.append(current_visibility)

            if min_visibility_range <= current_visibility <= max_visibility_range:
                print(f" - {file_name}: Masked and saved. Visible area: {current_visibility*100:.2f}% (within range)")
            else:
                print(f" - {file_name}: Masked and saved. Visible area: {current_visibility*100:.2f}% (OUTSIDE range)")

        except Exception as e:
            print(f"Could not process {file_name}: {e}")

    print("\nMasking and verification complete.")

    if all_visible_areas:
        average_visible_area = np.mean(all_visible_areas) * 100
        average_masked_area = 100 - average_visible_area
        print("\n--- Summary Statistics (Regular Edge Masking) ---")
        print(f"Average Visible Area: {average_visible_area:.2f}%")
        print(f"Average Masked Area: {average_masked_area:.2f}%")
    else:
        print("No images were successfully processed to calculate statistics.")

if __name__ == "__main__":
    image_source_directory = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/imagenet1k"
    output_regularly_masked_images_directory = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/regularly_masked_edges_imagenet1k"
    output_generated_masks_directory = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/regularly_generated_masks"
    
    apply_regular_edge_mask_and_verify(image_source_directory, output_regularly_masked_images_directory, output_generated_masks_directory)