#!/usr/bin/env python
"""
Met Textiles × FiftyOne – end-to-end pipeline
"""

import os, json, glob, argparse
import fiftyone as fo, fiftyone.brain as fob
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from PIL import Image                # NEW

# --------------------------- CLI flag -------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--rebuild", action="store_true",
                    help="Delete existing dataset before running")
args = parser.parse_args()

Image.MAX_IMAGE_PIXELS = None        # disable Pillow size limit

JSON_PATH  = (
    "../data/FINAL_CORRECTED_MET_TEXTILES_DATASET/objects_with_images_only/"
    "ALL_TEXTILES_AND_TAPESTRIES_WITH_IMAGES_20250705_230315.json"
)
IMAGES_DIR   = "MET_TEXTILES_BULLETPROOF_DATASET/images"
DATASET_NAME = "met_textiles_27k"
EMB_FIELD    = "clip_emb"

# --------------------------- helper ---------------------------------- #
def resolve_image_path(obj):
    oid = obj["objectID"]
    cand = [
        os.path.join(IMAGES_DIR, f"{oid}.jpg"),
        os.path.join(IMAGES_DIR, f"{oid}.png"),
        *glob.glob(os.path.join(IMAGES_DIR, f"{oid}_*_primary.*")),
        *(os.path.join(IMAGES_DIR, os.path.basename(u))
          for u in (obj.get("primaryImage"), obj.get("primaryImageSmall")) if u),
    ]
    return next((p for p in cand if os.path.exists(p)), None)

# --------------------------- dataset --------------------------------- #
if args.rebuild and fo.dataset_exists(DATASET_NAME):
    print("🗑️  Deleting existing dataset …")
    fo.delete_dataset(DATASET_NAME)

if fo.dataset_exists(DATASET_NAME):
    print("✅ Loading existing dataset")
    ds = fo.load_dataset(DATASET_NAME)
else:
    print("🚀 Creating dataset from JSON")
    print(f"📂 JSON Path: {JSON_PATH}")
    print(f"🖼️  Images Directory: {IMAGES_DIR}")
    with open(JSON_PATH, encoding="utf-8") as f:
        objects = json.load(f)
    
    print(f"📊 Found {len(objects)} objects in JSON")
    print("🔗 Resolving image paths...")

    samples, miss = [], 0
    for i, obj in enumerate(objects):
        if i % 5000 == 0 and i > 0:
            print(f"   Processed {i}/{len(objects)} objects...")
        fp = resolve_image_path(obj)
        if not fp:
            miss += 1
            continue
        samples.append(
            fo.Sample(
                filepath=fp,
                object_id=obj["objectID"],
                department=obj.get("department", ""),
                classification=obj.get("classification", ""),
                title=obj.get("title", ""),
                tags=[obj.get("department", ""), obj.get("classification", "")],  # NEW
                met_raw=obj,
            )
        )
    print(f"   → {len(samples)} samples | {miss} missing images")
    print("💾 Adding samples to dataset...")

    ds = fo.Dataset(DATASET_NAME, overwrite=True)
    ds.add_samples(samples)
    ds.persistent = True
    print("✅ Dataset created and saved!")

print(f"\n📈 Dataset Overview:")
print(ds, "\n")

# --------------------------- embeddings ------------------------------ #
print(f"🧠 Loading CLIP model...")
model = foz.load_zoo_model("clip-vit-base32-torch")
print(f"🔍 Checking for embeddings in field '{EMB_FIELD}'...")

if EMB_FIELD not in ds.get_field_schema():
    print("⚡ Computing CLIP embeddings (full set)…")
    print("   Batch size: 64, Workers: 8")
    ds.compute_embeddings(
        model,
        embeddings_field=EMB_FIELD,
        batch_size=64,
        num_workers=8,
        skip_failures=True,          # NEW
    )
    ds.save()
    print("✅ CLIP embeddings computed and saved!")

# second pass for any that still failed
missing = ds.match(F(EMB_FIELD) == None)
if len(missing):
    print(f"🔄 Filling {len(missing)} missing embeddings …")
    missing.compute_embeddings(
        model,
        embeddings_field=EMB_FIELD,
        batch_size=64,
        num_workers=8,
        skip_failures=True,
    )
    ds.save()
    print("✅ Missing embeddings filled!")

# --------------------------- Brain goodies --------------------------- #
print("🧠 Computing FiftyOne Brain features...")
print("   🎯 Computing uniqueness...")
fob.compute_uniqueness(ds, embeddings=EMB_FIELD)
print("   📊 Computing representativeness...")

fob.compute_representativeness(ds, embeddings=EMB_FIELD)
print("   🗺️  Computing 3D UMAP visualization...")

fob.compute_visualization(
    ds, embeddings=EMB_FIELD, brain_key="clip_umap", method="umap", dim=3
)

# single similarity run
if "clip_sim" in ds.list_brain_runs():
    print("   🧹 Cleaning up existing similarity index...")
    ds.delete_brain_run("clip_sim")

print("   🔍 Setting up similarity search...")
try:
    fob.compute_similarity(
        ds, embeddings=EMB_FIELD, brain_key="clip_sim", backend="lancedb"
    )
    print("   ✅ LanceDB similarity backend ready!")
except Exception as e:
    print(f"   ⚠️  LanceDB missing → falling back to sklearn: {e}")
    fob.compute_similarity(
        ds, embeddings=EMB_FIELD, brain_key="clip_sim", backend="sklearn"
    )
    print("   ✅ Sklearn similarity backend ready!")

print("   🔍 Finding near-duplicates...")
dup = fob.compute_near_duplicates(ds, embeddings=EMB_FIELD, threshold=0.18)
print(f"   📋 Found {len(dup.duplicate_ids)} near-duplicate sets")
print("✅ All Brain features computed!")
print()

# 
# ---------------------------------------------------------------------- #
# Launch interactive UI
# ---------------------------------------------------------------------- #
print("🚀 Launching FiftyOne App...")
print("   📍 Port: 5151")
print("   🌐 Opening browser tab...")
session = fo.launch_app(ds, port=5151)
session.open_tab()
session.wait()
print("🎉 FiftyOne App is ready! Enjoy exploring!")
