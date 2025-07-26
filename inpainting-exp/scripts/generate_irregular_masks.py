#!/home/user1/miniconda3/envs/diffassemble/bin/python
import os
import shutil
from PIL import Image

def generate_resized_masks(
    image_dir,
    base_mask_path,
    output_masks_dir
):
    """Generates and saves resized versions of a base mask for each image in a directory."""
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found at '{image_dir}'")
        return
    if not os.path.exists(base_mask_path):
        print(f"Error: Base mask file not found at '{base_mask_path}'")
        return

    os.makedirs(output_masks_dir, exist_ok=True)

    try:
        base_mask = Image.open(base_mask_path).convert('L') # Load base mask as grayscale
    except Exception as e:
        print(f"Error loading base mask {base_mask_path}: {e}")
        return

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found in the source directory to generate masks for.")
        return

    print(f"Generating masks for {len(image_files)} images...")
    
    for file_name in image_files:
        img_path = os.path.join(image_dir, file_name)
        
        # Construct mask output path (e.g., image_name_mask.png)
        mask_file_name = os.path.splitext(file_name)[0] + "_mask.png"
        output_mask_path = os.path.join(output_masks_dir, mask_file_name)

        try:
            img = Image.open(img_path) # Just need dimensions, no need to convert mode
            width, height = img.size

            # Resize base mask to image dimensions
            resized_mask = base_mask.resize((width, height), Image.LANCZOS)
            resized_mask.save(output_mask_path)

        except Exception as e:
            print(f"Could not generate mask for {file_name}: {e}")

    print("\nMask generation complete.")
    print(f"Generated masks saved to '{output_masks_dir}'.")

if __name__ == "__main__":
    image_source_directory = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/imagenet1k"
    base_mask_file_path = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/oseberg/masks_only/img_123_mask.png"
    output_generated_masks_directory = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/irregularly_generated_masks"
    
    generate_resized_masks(image_source_directory, base_mask_file_path, output_generated_masks_directory)
