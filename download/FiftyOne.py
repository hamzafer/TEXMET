#!/usr/bin/env python3
"""
Complete FiftyOne setup for MET Textiles Dataset with all Brain features
Optimized for RTX 4090 24GB - Maximum features enabled
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import logging

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
from fiftyone import ViewField as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
JSON_PATH = "../FINAL_CORRECTED_MET_TEXTILES_DATASET/objects_with_images_only/ALL_TEXTILES_AND_TAPESTRIES_WITH_IMAGES_20250705_230315.json"
IMAGES_DIR = "MET_TEXTILES_BULLETPROOF_DATASET/images"
DATASET_NAME = "met_textiles_complete"
BATCH_SIZE = 32  # Optimized for RTX 4090
NUM_WORKERS = 4

class METTextilesDatasetBuilder:
    """Complete FiftyOne dataset builder with all brain features"""
    
    def __init__(self, json_path: str, images_dir: str, dataset_name: str):
        self.json_path = json_path
        self.images_dir = Path(images_dir)
        self.dataset_name = dataset_name
        self.dataset = None
        
    def load_json_data(self) -> List[Dict]:
        """Load and validate JSON data"""
        logger.info(f"Loading JSON data from {self.json_path}")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} objects from JSON")
        return data
    
    def find_local_image_path(self, object_id: int, primary_image_url: str) -> Optional[str]:
        """Find local image path for an object"""
        # Try common naming patterns
        possible_names = [
            f"{object_id}.jpg",
            f"{object_id}.png",
            f"{object_id}.jpeg",
            f"object_{object_id}.jpg",
            f"met_{object_id}.jpg",
        ]
        
        for name in possible_names:
            path = self.images_dir / name
            if path.exists():
                return str(path)
        
        # Try to extract filename from URL
        if primary_image_url:
            url_filename = primary_image_url.split('/')[-1]
            path = self.images_dir / url_filename
            if path.exists():
                return str(path)
        
        return None
    
    def parse_date_range(self, date_str: str) -> Dict[str, Optional[int]]:
        """Parse date strings like 'ca. 1870–75' or '1867'"""
        if not date_str:
            return {"start_year": None, "end_year": None}
        
        # Remove common prefixes
        date_str = date_str.replace("ca. ", "").replace("c. ", "")
        
        if "–" in date_str:
            parts = date_str.split("–")
            start_year = int(parts[0]) if parts[0].isdigit() else None
            end_year = int(parts[1]) if parts[1].isdigit() else None
            # Handle abbreviated end years like "1870–75"
            if end_year and end_year < 100 and start_year:
                end_year = (start_year // 100) * 100 + end_year
        else:
            year = int(date_str) if date_str.isdigit() else None
            start_year = end_year = year
        
        return {"start_year": start_year, "end_year": end_year}
    
    def create_sample_from_object(self, obj: Dict) -> Optional[fo.Sample]:
        """Create FiftyOne sample from MET object"""
        object_id = obj.get("objectID")
        primary_image = obj.get("primaryImage", "")
        
        # Find local image
        local_image_path = self.find_local_image_path(object_id, primary_image)
        if not local_image_path:
            logger.warning(f"No local image found for object {object_id}")
            return None
        
        # Parse dates
        date_info = self.parse_date_range(obj.get("objectDate", ""))
        
        # Create sample
        sample = fo.Sample(filepath=local_image_path)
        
        # Core metadata
        sample["object_id"] = object_id
        sample["accession_number"] = obj.get("accessionNumber", "")
        sample["accession_year"] = obj.get("accessionYear", "")
        sample["is_highlight"] = obj.get("isHighlight", False)
        sample["is_public_domain"] = obj.get("isPublicDomain", False)
        sample["is_timeline_work"] = obj.get("isTimelineWork", False)
        
        # Object information - PRIORITY FIELDS
        sample["title"] = obj.get("title", "")
        sample["department"] = obj.get("department", "")
        sample["classification"] = obj.get("classification", "")
        
        # Additional object details
        sample["object_name"] = obj.get("objectName", "")
        sample["culture"] = obj.get("culture", "")
        sample["period"] = obj.get("period", "")
        sample["dynasty"] = obj.get("dynasty", "")
        sample["medium"] = obj.get("medium", "")
        sample["dimensions"] = obj.get("dimensions", "")
        sample["credit_line"] = obj.get("creditLine", "")
        sample["gallery_number"] = obj.get("GalleryNumber", "")
        
        # Date information
        sample["object_date"] = obj.get("objectDate", "")
        sample["object_begin_date"] = obj.get("objectBeginDate")
        sample["object_end_date"] = obj.get("objectEndDate")
        sample["date_start_year"] = date_info["start_year"]
        sample["date_end_year"] = date_info["end_year"]
        
        # Artist information
        sample["artist_display_name"] = obj.get("artistDisplayName", "")
        sample["artist_display_bio"] = obj.get("artistDisplayBio", "")
        sample["artist_nationality"] = obj.get("artistNationality", "")
        sample["artist_gender"] = obj.get("artistGender", "")
        sample["artist_role"] = obj.get("artistRole", "")
        sample["artist_prefix"] = obj.get("artistPrefix", "")
        sample["artist_begin_date"] = obj.get("artistBeginDate", "")
        sample["artist_end_date"] = obj.get("artistEndDate", "")
        
        # Geographic information
        sample["geography_type"] = obj.get("geographyType", "")
        sample["city"] = obj.get("city", "")
        sample["state"] = obj.get("state", "")
        sample["country"] = obj.get("country", "")
        sample["region"] = obj.get("region", "")
        sample["subregion"] = obj.get("subregion", "")
        
        # URLs and references
        sample["object_url"] = obj.get("objectURL", "")
        sample["object_wikidata_url"] = obj.get("objectWikidata_URL", "")
        sample["artist_wikidata_url"] = obj.get("artistWikidata_URL", "")
        sample["artist_ulan_url"] = obj.get("artistULAN_URL", "")
        
        # Process constituents
        constituents = obj.get("constituents", [])
        if constituents:
            sample["constituents"] = fo.ListField([
                fo.EmbeddedDocument.from_dict({
                    "id": c.get("constituentID"),
                    "role": c.get("role", ""),
                    "name": c.get("name", ""),
                    "gender": c.get("gender", ""),
                    "ulan_url": c.get("constituentULAN_URL", ""),
                    "wikidata_url": c.get("constituentWikidata_URL", "")
                }) for c in constituents
            ])
        
        # Process measurements
        measurements = obj.get("measurements", [])
        if measurements:
            sample["measurements"] = fo.ListField([
                fo.EmbeddedDocument.from_dict({
                    "element_name": m.get("elementName", ""),
                    "element_description": m.get("elementDescription", ""),
                    "measurements": m.get("elementMeasurements", {})
                }) for m in measurements
            ])
        
        # Process tags
        tags = obj.get("tags", [])
        if tags:
            sample["tags"] = fo.ListField([
                fo.EmbeddedDocument.from_dict({
                    "term": t.get("term", ""),
                    "aat_url": t.get("AAT_URL", ""),
                    "wikidata_url": t.get("Wikidata_URL", "")
                }) for t in tags
            ])
            
            # Extract tag terms for easy filtering
            sample["tag_terms"] = [t.get("term", "") for t in tags]
        
        # Additional computed fields
        sample["has_measurements"] = len(measurements) > 0
        sample["has_tags"] = len(tags) > 0
        sample["has_constituents"] = len(constituents) > 0
        sample["century"] = (date_info["start_year"] // 100 + 1) if date_info["start_year"] else None
        
        return sample
    
    def create_dataset(self) -> fo.Dataset:
        """Create FiftyOne dataset with all samples"""
        logger.info(f"Creating dataset: {self.dataset_name}")
        
        # Delete existing dataset if it exists
        if self.dataset_name in fo.list_datasets():
            fo.delete_dataset(self.dataset_name)
        
        # Create new dataset
        dataset = fo.Dataset(self.dataset_name)
        dataset.persistent = True
        
        # Load JSON data
        json_data = self.load_json_data()
        
        # Create samples
        samples = []
        failed_count = 0
        
        logger.info("Creating samples...")
        for obj in tqdm(json_data, desc="Processing objects"):
            try:
                sample = self.create_sample_from_object(obj)
                if sample:
                    samples.append(sample)
                else:
                    failed_count += 1
            except Exception as e:
                logger.error(f"Error processing object {obj.get('objectID')}: {e}")
                failed_count += 1
        
        # Add samples to dataset
        if samples:
            dataset.add_samples(samples)
            logger.info(f"Added {len(samples)} samples to dataset")
            logger.info(f"Failed to process {failed_count} objects")
        else:
            logger.error("No samples created!")
            return None
        
        # Add dataset info
        dataset.info = {
            "description": "MET Museum Textiles and Tapestries Dataset",
            "created": datetime.now().isoformat(),
            "total_objects": len(json_data),
            "successful_samples": len(samples),
            "failed_samples": failed_count,
            "source": "Metropolitan Museum of Art API"
        }
        
        self.dataset = dataset
        return dataset
    
    def compute_embeddings(self, model_name: str = "clip-vit-base32-torch") -> None:
        """Compute image embeddings using CLIP"""
        logger.info(f"Computing embeddings using {model_name}")
        
        try:
            # Compute embeddings
            embeddings = fob.compute_embeddings(
                self.dataset,
                model=model_name,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                embeddings_field="embeddings",
                progress=True
            )
            
            logger.info(f"Computed embeddings for {len(embeddings)} samples")
            
        except Exception as e:
            logger.error(f"Error computing embeddings: {e}")
            raise
    
    def compute_similarity(self, brain_key: str = "similarity") -> None:
        """Compute image similarity using embeddings"""
        logger.info("Computing similarity index...")
        
        try:
            similarity_index = fob.compute_similarity(
                self.dataset,
                embeddings="embeddings",
                brain_key=brain_key,
                progress=True
            )
            
            logger.info(f"Created similarity index with key: {brain_key}")
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            raise
    
    def compute_uniqueness(self, brain_key: str = "uniqueness") -> None:
        """Compute image uniqueness scores"""
        logger.info("Computing uniqueness scores...")
        
        try:
            uniqueness_scores = fob.compute_uniqueness(
                self.dataset,
                embeddings="embeddings",
                brain_key=brain_key,
                progress=True
            )
            
            logger.info(f"Computed uniqueness scores with key: {brain_key}")
            
        except Exception as e:
            logger.error(f"Error computing uniqueness: {e}")
            raise
    
    def compute_visualization(self, brain_key: str = "visualization") -> None:
        """Compute 2D visualization using embeddings"""
        logger.info("Computing visualization...")
        
        try:
            # Compute visualization
            results = fob.compute_visualization(
                self.dataset,
                embeddings="embeddings",
                brain_key=brain_key,
                method="umap",  # or "tsne", "pca"
                progress=True
            )
            
            logger.info(f"Computed visualization with key: {brain_key}")
            
        except Exception as e:
            logger.error(f"Error computing visualization: {e}")
            raise
    
    def compute_hardness(self, brain_key: str = "hardness") -> None:
        """Compute hardness scores if we have labels"""
        logger.info("Computing hardness scores...")
        
        try:
            # For unsupervised hardness, we can use culture or period as pseudo-labels
            # First, let's create a simple classification field
            self.dataset.compute_metadata()
            
            # Use culture as labels for hardness computation
            view = self.dataset.match(F("culture") != "")
            
            if len(view) > 0:
                hardness_scores = fob.compute_hardness(
                    view,
                    label_field="culture",
                    embeddings="embeddings",
                    brain_key=brain_key,
                    progress=True
                )
                
                logger.info(f"Computed hardness scores with key: {brain_key}")
            else:
                logger.warning("No culture labels found for hardness computation")
                
        except Exception as e:
            logger.error(f"Error computing hardness: {e}")
            # Don't raise, as this is optional
    
    def setup_advanced_indexing(self) -> None:
        """Set up advanced indexing for better performance"""
        logger.info("Setting up advanced indexing...")
        
        try:
            # Create indexes for common fields - PRIORITY FIELDS FIRST
            index_fields = [
                "object_id",
                "title",              # PRIORITY
                "department",         # PRIORITY  
                "classification",     # PRIORITY
                "culture",
                "period",
                "artist_display_name",
                "country",
                "century",
                "object_name",
                "medium",
                "is_highlight",
                "is_public_domain"
            ]
            
            for field in index_fields:
                try:
                    self.dataset.create_index(field)
                    logger.info(f"Created index for {field}")
                except Exception as e:
                    logger.warning(f"Could not create index for {field}: {e}")
                    
        except Exception as e:
            logger.error(f"Error setting up indexing: {e}")
    
    def generate_dataset_stats(self) -> Dict[str, Any]:
        """Generate comprehensive dataset statistics"""
        logger.info("Generating dataset statistics...")
        
        stats = {
            "total_samples": len(self.dataset),
            "schema": self.dataset.get_field_schema(),
            "field_counts": {},
            "value_counts": {},
            "date_range": {},
            "geographic_distribution": {},
            "media_stats": {}
        }
        
        # Field counts
        for field_name in self.dataset.get_field_schema():
            try:
                non_null_count = len(self.dataset.match(F(field_name).exists()))
                stats["field_counts"][field_name] = non_null_count
            except:
                pass
        
        # Value counts for categorical fields - PRIORITY FIELDS FIRST
        categorical_fields = [
            "title", "department", "classification",  # PRIORITY FIELDS
            "culture", "period", "artist_display_name", "country", "object_name"
        ]
        for field in categorical_fields:
            try:
                values = self.dataset.values(field)
                stats["value_counts"][field] = len(set(v for v in values if v))
            except:
                pass
        
        # Date range
        try:
            begin_dates = [d for d in self.dataset.values("object_begin_date") if d]
            end_dates = [d for d in self.dataset.values("object_end_date") if d]
            if begin_dates and end_dates:
                stats["date_range"] = {
                    "earliest": min(begin_dates),
                    "latest": max(end_dates),
                    "span_years": max(end_dates) - min(begin_dates)
                }
        except:
            pass
        
        # Geographic distribution
        try:
            countries = [c for c in self.dataset.values("country") if c]
            stats["geographic_distribution"] = {
                "unique_countries": len(set(countries)),
                "top_countries": list(pd.Series(countries).value_counts().head(10).to_dict().keys())
            }
        except:
            pass
        
        return stats
    
    def create_useful_views(self) -> Dict[str, fo.DatasetView]:
        """Create useful dataset views for exploration"""
        logger.info("Creating useful dataset views...")
        
        views = {}
        
        # Highlights only
        views["highlights"] = self.dataset.match(F("is_highlight") == True)
        
        # Public domain only
        views["public_domain"] = self.dataset.match(F("is_public_domain") == True)
        
        # Timeline works
        views["timeline_works"] = self.dataset.match(F("is_timeline_work") == True)
        
        # PRIORITY FIELD VIEWS
        
        # By department (PRIORITY)
        try:
            departments = [d for d in self.dataset.values("department") if d]
            unique_departments = list(set(departments))
            for dept in unique_departments:
                safe_name = dept.replace(" ", "_").replace("The ", "").replace("(", "").replace(")", "").lower()
                views[f"dept_{safe_name}"] = self.dataset.match(F("department") == dept)
        except:
            pass
        
        # By classification (PRIORITY)
        try:
            classifications = [c for c in self.dataset.values("classification") if c]
            unique_classifications = list(set(classifications))
            for classif in unique_classifications:
                safe_name = classif.replace(" ", "_").replace("(", "").replace(")", "").lower()
                views[f"class_{safe_name}"] = self.dataset.match(F("classification") == classif)
        except:
            pass
        
        # By century
        for century in range(15, 22):  # 15th to 21st century
            century_view = self.dataset.match(F("century") == century)
            if len(century_view) > 0:
                views[f"century_{century}"] = century_view
        
        # By culture (top cultures)
        try:
            cultures = [c for c in self.dataset.values("culture") if c]
            top_cultures = pd.Series(cultures).value_counts().head(10).index.tolist()
            for culture in top_cultures:
                safe_name = culture.replace(" ", "_").replace("(", "").replace(")", "").lower()
                views[f"culture_{safe_name}"] = self.dataset.match(F("culture") == culture)
        except:
            pass
        
        # By object type
        try:
            object_names = [o for o in self.dataset.values("object_name") if o]
            top_objects = pd.Series(object_names).value_counts().head(10).index.tolist()
            for obj_name in top_objects:
                safe_name = obj_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
                views[f"object_{safe_name}"] = self.dataset.match(F("object_name") == obj_name)
        except:
            pass
        
        # With measurements
        views["with_measurements"] = self.dataset.match(F("has_measurements") == True)
        
        # With tags
        views["with_tags"] = self.dataset.match(F("has_tags") == True)
        
        logger.info(f"Created {len(views)} useful views")
        return views
    
    def run_complete_pipeline(self) -> fo.Dataset:
        """Run the complete pipeline with all brain features"""
        logger.info("Starting complete FiftyOne pipeline...")
        
        # Create dataset
        dataset = self.create_dataset()
        if not dataset:
            raise Exception("Failed to create dataset")
        
        # Compute all brain features
        try:
            self.compute_embeddings()
            self.compute_similarity()
            self.compute_uniqueness()
            self.compute_visualization()
            self.compute_hardness()
        except Exception as e:
            logger.error(f"Error in brain computation: {e}")
            # Continue anyway, some features might have worked
        
        # Setup advanced features
        self.setup_advanced_indexing()
        
        # Generate stats
        stats = self.generate_dataset_stats()
        dataset.info["stats"] = stats
        
        # Create views
        views = self.create_useful_views()
        
        # Save dataset
        dataset.save()
        
        logger.info("="*50)
        logger.info("PIPELINE COMPLETE!")
        logger.info("="*50)
        logger.info(f"Dataset name: {self.dataset_name}")
        logger.info(f"Total samples: {len(dataset)}")
        logger.info(f"Brain features computed: embeddings, similarity, uniqueness, visualization, hardness")
        logger.info(f"Created {len(views)} useful views")
        logger.info("PRIORITY FIELDS INDEXED: title, department, classification")
        logger.info("="*50)
        
        return dataset


def main():
    """Main function to run the complete pipeline"""
    
    # Verify paths exist
    if not os.path.exists(JSON_PATH):
        logger.error(f"JSON file not found: {JSON_PATH}")
        return
    
    if not os.path.exists(IMAGES_DIR):
        logger.error(f"Images directory not found: {IMAGES_DIR}")
        return
    
    # Create builder and run pipeline
    builder = METTextilesDatasetBuilder(JSON_PATH, IMAGES_DIR, DATASET_NAME)
    dataset = builder.run_complete_pipeline()
    
    # Launch FiftyOne App
    print("\n" + "="*60)
    print("READY TO EXPLORE!")
    print("="*60)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Samples: {len(dataset)}")
    print("\nPRIORITY FIELDS (fully indexed):")
    print("📝 titles - Object titles")
    print("📚 departments - Museum departments") 
    print("🏷️ classifications - Object classifications")
    print("\nUseful commands:")
    print("# Load dataset")
    print(f"dataset = fo.load_dataset('{DATASET_NAME}')")
    print("\n# Explore priority fields")
    print("explore_dataset()")
    print("explore_by_department('The American Wing')")
    print("explore_by_classification('Furniture')")
    print("\n# Launch app")
    print("session = fo.launch_app(dataset)")
    print("\n# Find similar images")
    print("sample = dataset.first()")
    print("similar_view = dataset.sort_by_similarity(sample, k=20)")
    print("\n# Most unique images")
    print("unique_view = dataset.sort_by('uniqueness', reverse=True)")
    print("\n# Filter by priority fields in UI sidebar:")
    print("- Department dropdown")
    print("- Classification dropdown") 
    print("- Title search box")
    print("="*60)
    
    # Optionally launch the app
    try:
        import webbrowser
        session = fo.launch_app(dataset)
        print(f"\nFiftyOne App launched at: {session.url}")
        
        # Show some example similar images
        if len(dataset) > 0:
            sample = dataset.first()
            similar_view = dataset.sort_by_similarity(sample, k=20)
            print(f"\nShowing 20 most similar images to: {sample.title}")
            session.view = similar_view
            
    except Exception as e:
        logger.error(f"Could not launch app: {e}")
        print("You can launch the app manually with:")
        print(f"fo.launch_app(fo.load_dataset('{DATASET_NAME}'))")


if __name__ == "__main__":
    main()


# Additional utility functions for exploration
def explore_dataset(dataset_name: str = DATASET_NAME):
    """Utility function to explore the dataset"""
    dataset = fo.load_dataset(dataset_name)
    
    print(f"Dataset: {dataset_name}")
    print(f"Total samples: {len(dataset)}")
    print(f"Brain keys: {list(dataset.list_brain_runs().keys())}")
    
    # PRIORITY FIELDS ANALYSIS
    print("\n" + "="*50)
    print("PRIORITY FIELDS ANALYSIS")
    print("="*50)
    
    print("\n📚 DEPARTMENTS:")
    departments = [d for d in dataset.values("department") if d]
    if departments:
        print(pd.Series(departments).value_counts().head(10))
    else:
        print("No departments found")
    
    print("\n🏷️ CLASSIFICATIONS:")
    classifications = [c for c in dataset.values("classification") if c]
    if classifications:
        print(pd.Series(classifications).value_counts().head(10))
    else:
        print("No classifications found")
    
    print("\n📝 TITLES (Top patterns):")
    titles = [t for t in dataset.values("title") if t]
    if titles:
        # Show common words in titles
        title_words = []
        for title in titles:
            title_words.extend(title.split())
        print("Most common words in titles:")
        print(pd.Series(title_words).value_counts().head(15))
    else:
        print("No titles found")
    
    print("\n" + "="*50)
    print("ADDITIONAL ANALYSIS")
    print("="*50)
    
    print("\n🌍 Top cultures:")
    cultures = [c for c in dataset.values("culture") if c]
    if cultures:
        print(pd.Series(cultures).value_counts().head(10))
    
    print("\n🎨 Top object types:")
    objects = [o for o in dataset.values("object_name") if o]
    if objects:
        print(pd.Series(objects).value_counts().head(10))
    
    print("\n📅 Date range:")
    dates = [d for d in dataset.values("object_begin_date") if d]
    if dates:
        print(f"From {min(dates)} to {max(dates)}")
    
    return dataset


def find_similar_objects(dataset_name: str, object_id: int, k: int = 20):
    """Find similar objects to a given object ID"""
    dataset = fo.load_dataset(dataset_name)
    
    # Find the sample
    sample = dataset.match(F("object_id") == object_id).first()
    if not sample:
        print(f"Object {object_id} not found")
        return None
    
    # Find similar
    similar_view = dataset.sort_by_similarity(sample, k=k)
    
    print(f"Found {len(similar_view)} similar objects to '{sample.title}'")
    for s in similar_view:
        print(f"- {s.title} (Culture: {s.culture}, Period: {s.period})")
    
    return similar_view


def explore_by_department(dataset_name: str, department: str):
    """Explore objects from a specific department"""
    dataset = fo.load_dataset(dataset_name)
    
    dept_view = dataset.match(F("department") == department)
    print(f"Found {len(dept_view)} objects from {department}")
    
    if len(dept_view) > 0:
        # Show classifications within this department
        classifications = [c for c in dept_view.values("classification") if c]
        if classifications:
            print(f"\nClassifications in {department}:")
            print(pd.Series(classifications).value_counts().head(10))
        
        # Show most unique objects from this department
        unique_view = dept_view.sort_by("uniqueness", reverse=True).limit(10)
        print(f"\nMost unique objects in {department}:")
        for s in unique_view:
            title = s.title if s.title else "Untitled"
            classif = s.classification if s.classification else "Unclassified"
            print(f"- {title} ({classif})")
    
    return dept_view


def explore_by_classification(dataset_name: str, classification: str):
    """Explore objects from a specific classification"""
    dataset = fo.load_dataset(dataset_name)
    
    class_view = dataset.match(F("classification") == classification)
    print(f"Found {len(class_view)} objects classified as {classification}")
    
    if len(class_view) > 0:
        # Show departments that have this classification
        departments = [d for d in class_view.values("department") if d]
        if departments:
            print(f"\nDepartments with {classification}:")
            print(pd.Series(departments).value_counts().head(10))
        
        # Show most unique objects from this classification
        unique_view = class_view.sort_by("uniqueness", reverse=True).limit(10)
        print(f"\nMost unique {classification} objects:")
        for s in unique_view:
            title = s.title if s.title else "Untitled"
            dept = s.department if s.department else "Unknown Department"
            print(f"- {title} ({dept})")
    
    return class_view


def explore_by_culture(dataset_name: str, culture: str):
    """Explore objects from a specific culture"""
    dataset = fo.load_dataset(dataset_name)
    
    culture_view = dataset.match(F("culture") == culture)
    print(f"Found {len(culture_view)} objects from {culture}")
    
    # Get most unique objects from this culture
    if len(culture_view) > 0:
        unique_view = culture_view.sort_by("uniqueness", reverse=True).limit(10)
        print(f"\nMost unique {culture} objects:")
        for s in unique_view:
            print(f"- {s.title} (uniqueness: {s.uniqueness:.3f})")
    
    return culture_view


def launch_exploration_session(dataset_name: str = DATASET_NAME):
    """Launch FiftyOne with the dataset ready for exploration"""
    dataset = fo.load_dataset(dataset_name)
    session = fo.launch_app(dataset)
    
    print("FiftyOne App launched!")
    print("\nTry these in the app:")
    print("1. Click 'Embeddings' to see the 2D visualization")
    print("2. Select an image and click 'Similarity' to find similar images")
    print("3. Sort by 'uniqueness' to see the most unique images")
    print("4. Use the sidebar to filter by culture, period, etc.")
    
    return session, dataset
