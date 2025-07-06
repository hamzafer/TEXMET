import fiftyone as fo
import json
import os

def load_or_create_dataset():
    """Load existing dataset or create new one if needed"""
    
    dataset_name = "MET_Textiles_Persistent"
    
    # Check if dataset already exists
    existing_datasets = fo.list_datasets()
    if dataset_name in existing_datasets:
        print(f"📂 Loading existing dataset '{dataset_name}'...")
        try:
            dataset = fo.load_dataset(dataset_name)
            print(f"✅ Loaded dataset with {len(dataset)} samples")
            return dataset
        except Exception as e:
            print(f"⚠️ Error loading existing dataset: {e}")
            print("🔄 Will create new dataset...")
    
    # Create new dataset if none exists
    return create_new_dataset(dataset_name)

def create_new_dataset(dataset_name):
    """Create a new FiftyOne dataset"""
    
    print(f"📝 Creating new dataset '{dataset_name}'...")
    
    # Load your data
    with open("../FINAL_CORRECTED_MET_TEXTILES_DATASET/objects_with_images_only/ALL_TEXTILES_AND_TAPESTRIES_WITH_IMAGES_20250705_230315.json", "r") as f:
        data = json.load(f)
    
    images_dir = "MET_TEXTILES_BULLETPROOF_DATASET/images"
    
    samples = []
    print("🔄 Processing samples...")
    
    for i, obj in enumerate(data):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(data)} samples")
            
        obj_id = str(obj['objectID'])
        
        # Find image file
        img_files = [f for f in os.listdir(images_dir) if f.startswith(obj_id)]
        if not img_files:
            continue
            
        img_path = os.path.join(images_dir, img_files[0])
        
        # Check if file actually exists
        if not os.path.exists(img_path):
            continue
        
        # Safe handling of tags
        tags_data = obj.get('tags', [])
        if tags_data is None:
            tags_data = []
        tags = [tag.get('term', '') for tag in tags_data if tag is not None]
        
        try:
            sample = fo.Sample(
                filepath=img_path,
                object_id=obj['objectID'],
                title=obj.get('title', ''),
                classification=obj.get('classification', ''),
                department=obj.get('department', ''),
                culture=obj.get('culture', ''),
                period=obj.get('period', ''),
                date=obj.get('objectDate', ''),
                medium=obj.get('medium', ''),
                tags=tags,
            )
            samples.append(sample)
        except Exception as e:
            print(f"Error creating sample for {obj_id}: {e}")
            continue
    
    print(f"✅ Created {len(samples)} samples")
    
    if not samples:
        print("❌ No valid samples found!")
        return None
    
    try:
        # Create dataset
        dataset = fo.Dataset(dataset_name)
        dataset.persistent = True  # Make it persistent
        
        print("📦 Adding samples to dataset...")
        # Add samples in batches
        batch_size = 500
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            dataset.add_samples(batch)
            print(f"Added batch {i//batch_size + 1}/{(len(samples)-1)//batch_size + 1}")
        
        # Save dataset
        print("💾 Saving dataset...")
        dataset.save()
        
        print(f"📊 Dataset '{dataset_name}' created with {len(dataset)} samples")
        return dataset
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        raise

def launch_app_only():
    """Just launch the app with existing dataset"""
    
    dataset_name = "MET_Textiles_Persistent"
    
    try:
        dataset = fo.load_dataset(dataset_name)
        print(f"📂 Loaded dataset '{dataset_name}' with {len(dataset)} samples")
        
        print("🚀 Launching FiftyOne App...")
        session = fo.launch_app(dataset, port=5151)
        
        print("🎯 FiftyOne launched at http://localhost:5151")
        print("📋 Use the interface to:")
        print("  - Browse by department/classification")
        print("  - Tag keepers vs rejects")
        print("  - Filter by date, culture, etc.")
        print("  - Create custom views")
        print()
        print("💡 Server is running. Press Ctrl+C to stop...")
        
        session.wait()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔄 Dataset not found. Run with --create to create new dataset.")

def cleanup_fiftyone():
    """Clean up all FiftyOne datasets"""
    print("🧹 Cleaning up existing datasets...")
    datasets = fo.list_datasets()
    for name in datasets:
        try:
            dataset = fo.load_dataset(name)
            dataset.delete()
            print(f"Deleted dataset: {name}")
        except Exception as e:
            print(f"Error deleting {name}: {e}")

if __name__ == "__main__":
    import sys
    
    if "--cleanup" in sys.argv:
        cleanup_fiftyone()
        
    elif "--create" in sys.argv:
        # Force create new dataset
        cleanup_fiftyone()
        dataset = create_new_dataset("MET_Textiles_Persistent")
        if dataset:
            print("🚀 Launching app...")
            session = fo.launch_app(dataset, port=5151)
            session.wait()
            
    elif "--launch" in sys.argv:
        # Just launch app with existing dataset
        launch_app_only()
        
    else:
        # Default: load existing or create new
        dataset = load_or_create_dataset()
        if dataset:
            print("🚀 Launching app...")
            session = fo.launch_app(dataset, port=5151)
            print("🎯 FiftyOne launched at http://localhost:5151")
            print("💡 Server is running. Press Ctrl+C to stop...")
            print(f"session: {session}")
            session.wait()
