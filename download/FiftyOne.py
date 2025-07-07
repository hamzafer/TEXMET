#!/usr/bin/env python
"""
Met Textiles × FiftyOne – end-to-end pipeline
"""

import os, json, fiftyone as fo, fiftyone.brain as fob
import glob
import fiftyone as fo

print("🧹 Cleaning up existing dataset...")
# fo.delete_dataset("met_textiles_27k")      # wipes the empty set

# ---------------------------------------------------------------------- #
# Paths – adjust if you've moved things
# ---------------------------------------------------------------------- #
JSON_PATH  = (
    "../FINAL_CORRECTED_MET_TEXTILES_DATASET/objects_with_images_only/"
    "ALL_TEXTILES_AND_TAPESTRIES_WITH_IMAGES_20250705_230315.json"
)
IMAGES_DIR = "MET_TEXTILES_BULLETPROOF_DATASET/images"
DATASET_NAME = "met_textiles_27k"

print(f"📂 JSON Path: {JSON_PATH}")
print(f"🖼️  Images Directory: {IMAGES_DIR}")
print(f"🏷️  Dataset Name: {DATASET_NAME}")
print()

# ---------------------------------------------------------------------- #
# Helper – map a JSON object → local image path
# ---------------------------------------------------------------------- #
def resolve_image_path(obj):
    """
    Return a matching local image filepath for this Met object, or None.

    Look for:
      1.  <IMAGES_DIR>/<objectID>.<jpg|png>
      2.  <IMAGES_DIR>/<objectID>_*_primary.<jpg|png>
      3.  A file whose basename matches the remote primary image URL
      4.  The web-large fallback
    """
    oid = obj["objectID"]

    # 1. Straight ID
    basics = [
        os.path.join(IMAGES_DIR, f"{oid}.jpg"),
        os.path.join(IMAGES_DIR, f"{oid}.png"),
    ]

    # 2. Your "*_primary" naming pattern
    primary_glob = glob.glob(os.path.join(IMAGES_DIR, f"{oid}_*_primary.*"))
    basics.extend(primary_glob)

    # 3 & 4. Basenames from remote URLs (handles any residual file names)
    url_names = [
        os.path.basename(obj.get("primaryImage", "")),
        os.path.basename(obj.get("primaryImageSmall", "")),
    ]
    basics.extend(os.path.join(IMAGES_DIR, n) for n in url_names if n)

    # return the first path that exists
    for path in basics:
        if path and os.path.exists(path):
            return path

    return None

# ---------------------------------------------------------------------- #
# (Re)create the dataset
# ---------------------------------------------------------------------- #
print("🔍 Checking if dataset exists...")
if fo.dataset_exists(DATASET_NAME):
    print(f"✅ Loading existing dataset: {DATASET_NAME}")
    ds = fo.load_dataset(DATASET_NAME)
else:
    print(f"🚀 Creating new dataset: {DATASET_NAME}")
    print("📖 Loading JSON metadata...")
    with open(JSON_PATH, encoding="utf-8") as f:
        objects = json.load(f)
    
    print(f"📊 Found {len(objects)} objects in JSON")
    print("🔗 Resolving image paths...")
    
    samples = []
    missing_images = 0
    for i, obj in enumerate(objects):
        if i % 1000 == 0 and i > 0:
            print(f"   Processed {i}/{len(objects)} objects...")
        
        fp = resolve_image_path(obj)
        if fp is None:
            missing_images += 1
            continue  # skip if image missing

        samples.append(
            fo.Sample(
                filepath=fp,
                object_id=obj["objectID"],
                department=obj.get("department", ""),
                classification=obj.get("classification", ""),
                title=obj.get("title", ""),
                met_raw=obj,  # keep full metadata for convenience
            )
        )

    print(f"✅ Created {len(samples)} samples")
    print(f"⚠️  Skipped {missing_images} objects with missing images")
    print("💾 Adding samples to dataset...")
    
    ds = fo.Dataset(DATASET_NAME, overwrite=True)
    ds.add_samples(samples)
    ds.persistent = True  # dataset survives Python restarts
    print("✅ Dataset created and saved!")

print(f"\n📈 Dataset Overview:")
print(ds)  # sanity-check (should show 27 k samples)
print()

# ---------------------------------------------------------------------- #
# Compute CLIP embeddings once (GPU-accelerated)
# ---------------------------------------------------------------------- #
EMB_FIELD = "clip_emb"
print(f"🧠 Checking for CLIP embeddings in field '{EMB_FIELD}'...")
if EMB_FIELD not in ds.get_field_schema():
    print("⚡ Computing CLIP embeddings – grab a coffee ☕")
    print("   Using model: clip-vit-base32-torch")
    print("   Batch size: 64, Workers: 8")
    ds.compute_embeddings(                      
        model="clip-vit-base32-torch",          # choose another CLIP variant if desired
        embeddings_field=EMB_FIELD,
        batch_size=64,                          # tune for your GPU
        num_workers=8,
    )
    ds.save()
    print("✅ CLIP embeddings computed and saved!")
else:
    print("✅ CLIP embeddings already exist!")
print()

# ---------------------------------------------------------------------- #
# FiftyOne Brain goodies
# ---------------------------------------------------------------------- #
print("🧠 Computing FiftyOne Brain features...")
print("   🎯 Computing uniqueness...")
fob.compute_uniqueness(ds, embeddings=EMB_FIELD)                 
print("   📊 Computing representativeness...")
fob.compute_representativeness(ds, embeddings=EMB_FIELD)         

print("   🗺️  Computing 3D UMAP visualization...")
fob.compute_visualization(                                       # 3-D UMAP for App
    ds,
    embeddings=EMB_FIELD,
    brain_key="clip_umap",
    method="umap",
    dim=3,
)                                                               

print("   🔍 Setting up similarity search (LanceDB)...")
fob.compute_similarity(                                          # fast NN search
    ds,
    embeddings=EMB_FIELD,
    brain_key="clip_sim",
    backend="lancedb",
)                                                               

print("   🔍 Finding near-duplicates...")
dup_results = fob.compute_near_duplicates(
    ds, embeddings=EMB_FIELD, threshold=0.18
)
print(f"   📋 Near-duplicate sets found: {len(dup_results.duplicate_ids)}")
print("✅ All Brain features computed!")
print()

# ---------------------------------------------------------------------- #
# Launch interactive UI
# ---------------------------------------------------------------------- #
print("🚀 Launching FiftyOne App...")
print("   Port: 5151")
print("   Opening browser tab...")
session = fo.launch_app(ds, port=5151)
session.open_tab()
session.wait()
print("🎉 FiftyOne App is ready! Enjoy exploring!")
