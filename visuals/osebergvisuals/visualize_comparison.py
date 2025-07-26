
import sys
from PIL import Image

def create_comparison(image1_path, image2_path, output_path):
    try:
        img1_orig = Image.open(image1_path)
        img2 = Image.open(image2_path)

        # Calculate the new width of the first image to maintain aspect ratio based on the second image's height
        img1_height = img2.height
        img1_width = int(img1_orig.width * (img1_height / img1_orig.height))

        # Resize the first image
        img1 = img1_orig.resize((img1_width, img1_height))

        # Create a new image with the combined width and the same height
        dst = Image.new('RGB', (img1.width + img2.width, img1.height))

        # Paste images side-by-side
        dst.paste(img1, (0, 0))
        dst.paste(img2, (img1.width, 0))

        dst.save(output_path)
        print(f"Saved comparison image to {output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}. One of the image paths is incorrect.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python visualize_comparison.py <image1_path> <image2_path> <output_path>")
        sys.exit(1)

    create_comparison(sys.argv[1], sys.argv[2], sys.argv[3])
