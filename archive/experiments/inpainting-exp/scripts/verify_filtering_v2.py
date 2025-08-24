import os

def verify_pairs_v2(directory):
    """Verifies that every image file has a corresponding mask file, handling complex naming."""
    if not os.path.exists(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    files = os.listdir(directory)
    images = {f for f in files if f.lower().endswith('.png') and '_mask' not in f}
    masks = {f for f in files if f.lower().endswith('.png') and '_mask' in f}

    images_without_masks = set()
    matched_masks = set()

    for img in images:
        # Standard case: img_001.png -> img_001_mask.png
        expected_mask_1 = img.replace('.png', '_mask.png')
        # Numbered case: img_004_1.png -> img_004_mask_1.png
        parts = img.rsplit('_', 1)
        expected_mask_2 = None
        if len(parts) == 2 and parts[1].replace('.png', '').isdigit():
            expected_mask_2 = f"{parts[0]}_mask_{parts[1]}"

        if expected_mask_1 in masks:
            matched_masks.add(expected_mask_1)
        elif expected_mask_2 and expected_mask_2 in masks:
            matched_masks.add(expected_mask_2)
        else:
            images_without_masks.add(img)

    orphaned_masks = masks - matched_masks

    # --- Report Results ---
    print("--- Verification Report (v2) ---")
    print(f"Directory: {directory}")
    print(f"Found {len(images)} images and {len(masks)} masks.")

    if not images_without_masks and not orphaned_masks:
        print("\n[SUCCESS] The filtering is correct!")
        print("Every image has a corresponding mask, and there are no orphaned files.")
    else:
        print("\n[FAILURE] Found some inconsistencies:")
        if images_without_masks:
            print("\nImages missing a mask:")
            for img in sorted(list(images_without_masks)):
                print(f" - {img}")
        if orphaned_masks:
            print("\nMasks missing an image (orphaned masks):")
            for mask in sorted(list(orphaned_masks)):
                print(f" - {mask}")

if __name__ == "__main__":
    target_dir = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/inpainting-exp/all_unique_pngs_with_masks"
    verify_pairs_v2(target_dir)
