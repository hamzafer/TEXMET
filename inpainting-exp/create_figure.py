import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def get_font(size=20):
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size)
    except IOError:
        font = ImageFont.load_default()
    return font

def resize_img(img, size=(512, 512)):
    return img.resize(size)

def create_regular_masking_figure(original_image_path, mask_path, masked_image_path, inpainted_image_path, output_path):
    try:
        original_img = resize_img(Image.open(original_image_path))
        mask_img = resize_img(Image.open(mask_path))
        masked_img = resize_img(Image.open(masked_image_path))
        inpainted_img = resize_img(Image.open(inpainted_image_path))
    except FileNotFoundError as e:
        print(f"Error: {e}. One of the image files was not found.")
        return

    mask_array = np.array(mask_img.convert('L'))
    masked_percentage = 100 - (np.sum(mask_array > 0) / mask_array.size * 100)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Regular Masking Process', fontsize=24)

    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=16)
    axes[0].axis('off')

    axes[1].imshow(mask_img, cmap='gray')
    axes[1].set_title(f'Regular Mask ({masked_percentage:.2f}% masked)', fontsize=16)
    axes[1].axis('off')

    axes[2].imshow(masked_img)
    axes[2].set_title('Masked Image', fontsize=16)
    axes[2].axis('off')

    axes[3].imshow(inpainted_img)
    axes[3].set_title('Inpainted Result', fontsize=16)
    axes[3].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    original_image = '/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/original_datasets/imagenet1k/ILSVRC2012_val_00000020.JPEG'
    mask_image = '/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/regular_masking_results/generated_masks/regularly_generated_masks/ILSVRC2012_val_00000020_mask.png'
    masked_image = '/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/regular_masking_results/masked_images/regularly_masked_edges_imagenet1k/ILSVRC2012_val_00000020.JPEG'
    inpainted_image = '/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/regular_masking_results/results/images/ILSVRC2012_val_00000020.png'
    output_figure_path = '/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/regular_masking_process.png'
    
    create_regular_masking_figure(original_image, mask_image, masked_image, inpainted_image, output_figure_path)