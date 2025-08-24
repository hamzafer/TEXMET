
import os
import random
import shutil
import numpy as np
from PIL import Image
from scipy.ndimage import sobel

# 1. Set paths and seed
root_imagenet = "/media/user1/DataStorage/hamza/thesis/val"
dest = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/selected_images_v4"
random.seed(42)

# Create destination directory if it doesn't exist
if os.path.exists(dest):
    shutil.rmtree(dest)
os.makedirs(dest)

# 2. Select more classes
NUM_CLASSES_TO_SEARCH = 500
try:
    all_classes = [d for d in os.listdir(root_imagenet) if os.path.isdir(os.path.join(root_imagenet, d))]
    if len(all_classes) < NUM_CLASSES_TO_SEARCH:
        print(f"Warning: Found only {len(all_classes)} classes, but {NUM_CLASSES_TO_SEARCH} were requested. Using all available.")
        NUM_CLASSES_TO_SEARCH = len(all_classes)
    selected_classes = random.sample(all_classes, NUM_CLASSES_TO_SEARCH)
except FileNotFoundError:
    print(f"Error: The source directory '{root_imagenet}' was not found.")
    exit()

# 3. Quality filter function (with relaxed resolution)
def ok(img_path):
    """Quality filter with relaxed resolution."""
    try:
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                return False

            w, h = img.size
            # Criterion 1 (Relaxed): short side >= 256 px
            if min(w, h) < 256:
                return False
            # Criterion 2: aspect ratio <= 2:1
            if max(w, h) / min(w, h) > 2.0:
                return False

            # Criterion 3: not "too flat"
            gray_img = img.convert('L')
            img_array = np.array(gray_img, dtype=np.float32)
            grad_x = sobel(img_array, axis=0)
            grad_y = sobel(img_array, axis=1)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            if np.mean(grad_magnitude) <= 15:
                return False
        return True
    except Exception:
        return False

# 4. Loop through classes and collect images
total_collected = 0
print(f"Starting image selection with relaxed filters (256px, {NUM_CLASSES_TO_SEARCH} classes)...")
for class_name in selected_classes:
    if total_collected >= 1000:
        break

    class_path = os.path.join(root_imagenet, class_name)
    try:
        images_in_class = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        random.shuffle(images_in_class)
    except FileNotFoundError:
        continue

    collected_from_class = 0
    for img_name in images_in_class:
        # Remove the 10-per-class limit to get to 1000 faster
        # if collected_from_class >= 10:
        #     break
        
        img_path = os.path.join(class_path, img_name)
        if ok(img_path):
            shutil.copy(img_path, dest)
            collected_from_class += 1
            total_collected += 1
            if total_collected >= 1000:
                break

print(f"Finished: Collected {total_collected} images in '{dest}'.")
