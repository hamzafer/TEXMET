
import os
import sys
from PIL import Image

def create_comparison_image(original_path, inpainted_path, output_path):
    try:
        original_img = Image.open(original_path).resize((512, 512))
        inpainted_img = Image.open(inpainted_path).resize((512, 512))

        # Create a new image with double the width to place images side by side
        comparison_img = Image.new('RGB', (1024, 512))
        comparison_img.paste(original_img, (0, 0))
        comparison_img.paste(inpainted_img, (512, 0))

        # Add a red border to the inpainted image to signify failure
        from PIL import ImageDraw
        draw = ImageDraw.Draw(comparison_img)
        draw.rectangle([512, 0, 1023, 511], outline="red", width=5)

        comparison_img.save(output_path)
        print(f"Saved comparison image to {output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}. One of the image paths is incorrect.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python visualize.py <original_image_path> <inpainted_image_path> <output_image_path>")
        sys.exit(1)

    create_comparison_image(sys.argv[1], sys.argv[2], sys.argv[3])
