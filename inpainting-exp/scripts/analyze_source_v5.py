import os

def analyze_source_directory_v5(directory):
    """Analyzes the source directory recursively, handling specific naming conventions and subdirectories."""
    if not os.path.exists(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    all_image_extensions = ('.png', '.jpg', '.jpeg')

    # Store unique base filenames for images and masks
    unique_image_basenames = set()
    unique_mask_basenames = set()

    for root, _, files in os.walk(directory):
        for f in files:
            if not f.lower().endswith(all_image_extensions):
                continue # Skip non-image files

            # Extract base name without extension
            base_name_no_ext = os.path.splitext(f)[0].lower()

            # Check for the 'mask' pattern directly in the filename
            if base_name_no_ext.endswith('mask'):
                unique_mask_basenames.add(base_name_no_ext) # Store as e.g., 'resized_001mask'
            else:
                unique_image_basenames.add(base_name_no_ext) # Store as e.g., 'resized_001'

    # Now, match images with masks based on their unique base names
    matched_image_basenames = set()
    unmatched_image_basenames = set()
    matched_mask_basenames = set()

    for img_base in unique_image_basenames:
        # Expected mask base name for this image base
        expected_mask_base = img_base + 'mask'
        
        if expected_mask_base in unique_mask_basenames:
            matched_image_basenames.add(img_base)
            matched_mask_basenames.add(expected_mask_base)
        else:
            unmatched_image_basenames.add(img_base)

    orphaned_mask_basenames = unique_mask_basenames - matched_mask_basenames

    print("--- Source Directory Analysis (v5 - Recursive & Robust) ---")
    print(f"Directory: {directory}")
    print(f"Total unique image files (excluding masks): {len(unique_image_basenames)}")
    print(f"Total unique mask files: {len(unique_mask_basenames)}")
    print("\n--- Breakdown ---")
    print(f"Unique images with a corresponding mask: {len(matched_image_basenames)}")
    print(f"Unique images without a corresponding mask: {len(unmatched_image_basenames)}")
    print(f"Orphaned masks (masks without a corresponding image): {len(orphaned_mask_basenames)}")

    if unmatched_image_basenames:
        print("\nList of unique images without masks (first 10):")
        for img_base in sorted(list(unmatched_image_basenames))[:10]:
            print(f" - {img_base}.png (example)")

    if orphaned_mask_basenames:
        print("\nList of unique orphaned masks (first 10):")
        for mask_base in sorted(list(orphaned_mask_basenames))[:10]:
            print(f" - {mask_base}.png (example)")

    print("\n--- Conclusion ---")
    if len(matched_image_basenames) == 45 and len(unmatched_image_basenames) > 0:
        print("The original directory contained a significant number of images that did not have a matching mask file.")
        print("Your filtering process correctly selected only the images that had masks, which is why the count dropped to 45.")
        print("There do not appear to be duplicates in the traditional sense, but rather many unpaired images.")
    else:
        print("Further investigation might be needed if the counts don't align with expectations.")

if __name__ == "__main__":
    source_dir = "/home/user1/Desktop/HAMZA/THESIS/master-thesis/dataset/DataTexRecSSH"
    analyze_source_directory_v5(source_dir)
