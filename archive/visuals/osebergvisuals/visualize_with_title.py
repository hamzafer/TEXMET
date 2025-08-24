
import sys
from PIL import Image, ImageDraw, ImageFont

def create_comparison_with_title(image1_path, image2_path, output_path, title):
    try:
        img2 = Image.open(image2_path)
        img1 = Image.open(image1_path).resize(img2.size)

        # Create a new image with combined width and max height
        dst = Image.new('RGB', (img1.width + img2.width, img1.height + 50)) # +50 for title

        # Add title
        draw = ImageDraw.Draw(dst)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        draw.text((10, 10), title, font=font, fill=(255, 255, 255))

        # Paste images
        dst.paste(img1, (0, 50))
        dst.paste(img2, (img1.width, 50))

        dst.save(output_path)
        print(f"Saved comparison image to {output_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}. One of the image paths is incorrect.")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python visualize_with_title.py <image1_path> <image2_path> <output_path> <title>")
        sys.exit(1)

    create_comparison_with_title(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
