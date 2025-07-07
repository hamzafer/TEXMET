import fiftyone as fo
import fiftyone.brain as fob  # For embeddings and clustering
import json
import os
import sys

def extract_year(date_str):
    """Extract year from date string"""
    import re
    if not date_str:
        return None
    
    # Look for 4-digit year
    match = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', str(date_str))
    if match:
        return int(match.group(1))
    return None

def add_embeddings_and_clustering(dataset):
    """Add embeddings and clustering to the dataset"""
    print("🧠 Computing embeddings...")
    
    try:
        # Compute image embeddings using a pre-trained model
        fob.compute_embeddings(
            dataset,
            model="clip-vit-base32-torch",  # Fast and good quality
            embeddings_field="embeddings",
            batch_size=16  # Smaller batch size to avoid memory issues
        )
        
        print("🎯 Performing UMAP visualization...")
        
        # Perform UMAP visualization
        fob.compute_visualization(
            dataset,
            embeddings="embeddings",
            method="umap",  # Better than t-SNE for large datasets
            brain_key="umap_viz"
        )
        
        print("📊 Computing uniqueness scores...")
        
        # Compute uniqueness to find duplicates/similar images
        fob.compute_uniqueness(
            dataset,
            embeddings="embeddings"
        )
        
        print("✅ Embeddings and clustering complete!")
        
    except Exception as e:
        print(f"⚠️ Brain features failed: {e}")
        print("📝 Dataset created without brain features")
    
    return dataset

def setup_similarity_search(dataset):
    """Setup similarity search capabilities"""
    print("🔍 Setting up similarity search...")
    
    try:
        # Build similarity index
        fob.compute_similarity(
            dataset,
            embeddings="embeddings",
            brain_key="similarity_index"
        )
        
        print("✅ Similarity search ready!")
        
    except Exception as e:
        print(f"⚠️ Similarity search setup failed: {e}")
    
    return dataset

def create_custom_views(dataset):
    """Create useful custom views for textile analysis"""
    
    print("📋 Creating custom views...")
    
    views = {}
    
    try:
        # Group by classification
        views["by_classification"] = dataset.group_by("classification")
        
        # Items with rich metadata
        views["rich_metadata"] = dataset.match(
            (fo.ViewField("culture") != "") & 
            (fo.ViewField("period") != "") & 
            (fo.ViewField("tags").length() > 2)
        )
        
        # Recent items (if year data available)
        if "year" in dataset.get_field_schema():
            views["modern_items"] = dataset.match(fo.ViewField("year") > 1800)
        
        # Tapestries vs Textiles
        views["tapestries"] = dataset.match(fo.ViewField("classification").contains_str("Tapestry"))
        views["textiles"] = dataset.match(fo.ViewField("classification").contains_str("Textile"))
        
        print("✅ Custom views created!")
        
    except Exception as e:
        print(f"⚠️ Error creating views: {e}")
    
    return views

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
    """Create a new FiftyOne dataset with advanced features"""
    
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
                # Add metadata for better analysis
                has_primary_image=bool(obj.get('primaryImage', '')),
                year=extract_year(obj.get('objectDate', '')),
                country=obj.get('country', ''),
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
        
        # Add brain features
        print("\n🧠 Adding AI-powered features...")
        dataset = add_embeddings_and_clustering(dataset)
        dataset = setup_similarity_search(dataset)
        
        # Create custom views
        views = create_custom_views(dataset)
        
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
        print("  - Use similarity search (click on an image)")
        print("  - Explore UMAP clustering visualization")
        print("  - Find duplicate/similar images")
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

def analyze_dataset(dataset):
    """Analyze the dataset and show brain feature results"""
    
    print("\n📊 DATASET ANALYSIS")
    print("=" * 50)
    
    # Basic stats
    print(f"Total samples: {len(dataset)}")
    
    # Check for brain features
    brain_info = dataset.get_brain_info()
    
    if brain_info:
        print(f"Brain features available: {list(brain_info.keys())}")
        
        # Show embedding info
        if "embeddings" in dataset.get_field_schema():
            print("✅ Embeddings computed")
        
        # Show visualization info
        if "umap_viz" in brain_info:
            print("✅ UMAP visualization available")
        
        # Show similarity index
        if "similarity_index" in brain_info:
            print("✅ Similarity search ready")
    else:
        print("⚠️ No brain features found")
    
    # Department breakdown
    dept_counts = dataset.count_values("department")
    print(f"\nTop departments:")
    for dept, count in sorted(dept_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {dept}: {count}")
    
    # Classification breakdown
    class_counts = dataset.count_values("classification")
    print(f"\nTop classifications:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {cls}: {count}")

if __name__ == "__main__":
    
    if "--cleanup" in sys.argv:
        cleanup_fiftyone()
        
    elif "--create" in sys.argv:
        # Force create new dataset with brain features
        cleanup_fiftyone()
        dataset = create_new_dataset("MET_Textiles_Persistent")
        if dataset:
            print("\n🚀 Launching app with brain features...")
            session = fo.launch_app(dataset, port=5151)
            session.wait()
            
    elif "--launch" in sys.argv:
        # Just launch app with existing dataset
        launch_app_only()
        
    elif "--analyze" in sys.argv:
        # Analyze existing dataset
        try:
            dataset = fo.load_dataset("MET_Textiles_Persistent")
            analyze_dataset(dataset)
            
            print("\n🚀 Launching app for analysis...")
            session = fo.launch_app(dataset, port=5151)
            session.wait()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🔄 Dataset not found. Run with --create first.")
        
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
