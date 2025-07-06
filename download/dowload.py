#!/usr/bin/env python3
"""
MET Textiles Dataset Image Downloader
=====================================
Downloads all images from the MET textiles and tapestries JSON dataset.
Includes progress tracking, comprehensive logging, and CSV reporting.

Author: Hamza
Date: July 2025
"""

import json
import requests
import os
import pandas as pd
import time
from datetime import datetime
import logging
from urllib.parse import urlparse
from pathlib import Path
import hashlib
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

class METImageDownloader:
    def __init__(self, json_file_path, output_dir="MET_TEXTILES_IMAGES", max_workers=10):
        self.json_file_path = json_file_path
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        self.additional_images_dir = self.output_dir / "additional_images"
        self.additional_images_dir.mkdir(exist_ok=True)
        
        # Initialize logging
        self.setup_logging()
        
        # Statistics
        self.stats = {
            'total_objects': 0,
            'objects_with_images': 0,
            'primary_images_downloaded': 0,
            'additional_images_downloaded': 0,
            'failed_downloads': 0,
            'skipped_existing': 0,
            'total_download_size': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Download tracking
        self.download_results = []
        self.failed_downloads = []
        
        # Thread lock for stats
        self.stats_lock = threading.Lock()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f"met_download_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("=== MET TEXTILES IMAGE DOWNLOADER STARTED ===")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Log file: {log_file}")
        
    def load_json_data(self):
        """Load and validate JSON data"""
        self.logger.info(f"Loading JSON data from: {self.json_file_path}")
        
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"Successfully loaded {len(data)} objects from JSON")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load JSON data: {e}")
            raise
            
    def get_safe_filename(self, object_id, title, image_type="primary"):
        """Generate safe filename for image"""
        # Clean title for filename
        safe_title = "".join(c for c in str(title)[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = safe_title.replace(' ', '_')
        
        if not safe_title:
            safe_title = "untitled"
            
        return f"{object_id}_{safe_title}_{image_type}.jpg"
        
    def download_image(self, url, filepath, object_id, image_type="primary"):
        """Download a single image with error handling"""
        try:
            # Check if file already exists
            if filepath.exists():
                self.logger.debug(f"Image already exists, skipping: {filepath.name}")
                with self.stats_lock:
                    self.stats['skipped_existing'] += 1
                return True, "already_exists", 0
                
            # Download image
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Get content length
            content_length = int(response.headers.get('content-length', 0))
            
            # Save image
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Verify file was created and has content
            if filepath.exists() and filepath.stat().st_size > 0:
                file_size = filepath.stat().st_size
                self.logger.debug(f"Successfully downloaded: {filepath.name} ({file_size} bytes)")
                
                with self.stats_lock:
                    self.stats['total_download_size'] += file_size
                    if image_type == "primary":
                        self.stats['primary_images_downloaded'] += 1
                    else:
                        self.stats['additional_images_downloaded'] += 1
                        
                return True, "success", file_size
            else:
                self.logger.error(f"Downloaded file is empty or missing: {filepath}")
                return False, "empty_file", 0
                
        except requests.RequestException as e:
            self.logger.error(f"Download failed for {object_id} ({image_type}): {e}")
            return False, f"request_error: {e}", 0
        except Exception as e:
            self.logger.error(f"Unexpected error downloading {object_id} ({image_type}): {e}")
            return False, f"unexpected_error: {e}", 0
            
    def process_object(self, obj):
        """Process a single object and download its images"""
        object_id = obj.get('objectID')
        title = obj.get('title', 'Untitled')
        classification = obj.get('classification', 'Unknown')
        
        result = {
            'object_id': object_id,
            'title': title,
            'classification': classification,
            'primary_image_url': '',
            'primary_image_downloaded': False,
            'primary_image_filename': '',
            'primary_image_size': 0,
            'additional_images_count': 0,
            'additional_images_downloaded': 0,
            'additional_images_filenames': [],
            'total_size': 0,
            'status': 'no_images',
            'error_message': ''
        }
        
        try:
            # Download primary image
            primary_image_url = obj.get('primaryImage', '')
            if primary_image_url:
                result['primary_image_url'] = primary_image_url
                
                filename = self.get_safe_filename(object_id, title, "primary")
                filepath = self.images_dir / filename
                
                success, status, size = self.download_image(primary_image_url, filepath, object_id, "primary")
                
                if success:
                    result['primary_image_downloaded'] = True
                    result['primary_image_filename'] = filename
                    result['primary_image_size'] = size
                    result['total_size'] += size
                    result['status'] = 'success'
                else:
                    result['error_message'] = status
                    result['status'] = 'failed'
                    with self.stats_lock:
                        self.stats['failed_downloads'] += 1
            
            # Download additional images
            additional_images = obj.get('additionalImages', [])
            if additional_images:
                result['additional_images_count'] = len(additional_images)
                additional_filenames = []
                
                for i, img_url in enumerate(additional_images):
                    filename = self.get_safe_filename(object_id, title, f"additional_{i+1}")
                    filepath = self.additional_images_dir / filename
                    
                    success, status, size = self.download_image(img_url, filepath, object_id, f"additional_{i+1}")
                    
                    if success:
                        result['additional_images_downloaded'] += 1
                        additional_filenames.append(filename)
                        result['total_size'] += size
                    
                result['additional_images_filenames'] = '; '.join(additional_filenames)
            
            # Update overall status
            if result['primary_image_downloaded'] or result['additional_images_downloaded'] > 0:
                if result['status'] != 'failed':
                    result['status'] = 'success'
                    
        except Exception as e:
            self.logger.error(f"Error processing object {object_id}: {e}")
            result['status'] = 'error'
            result['error_message'] = str(e)
            
        return result
        
    def download_all_images(self):
        """Main method to download all images"""
        self.logger.info("=== STARTING IMAGE DOWNLOAD PROCESS ===")
        self.stats['start_time'] = datetime.now()
        
        # Load JSON data
        data = self.load_json_data()
        self.stats['total_objects'] = len(data)
        
        # Count objects with images
        objects_with_images = [obj for obj in data if obj.get('primaryImage') or obj.get('additionalImages')]
        self.stats['objects_with_images'] = len(objects_with_images)
        
        self.logger.info(f"Found {len(objects_with_images)} objects with images out of {len(data)} total objects")
        
        # Create progress bar
        progress_bar = tqdm(
            total=len(objects_with_images),
            desc="Downloading images",
            unit="objects",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        # Download images using thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_object = {
                executor.submit(self.process_object, obj): obj 
                for obj in objects_with_images
            }
            
            # Process completed tasks
            for future in as_completed(future_to_object):
                result = future.result()
                self.download_results.append(result)
                
                # Update progress
                progress_bar.update(1)
                
                # Update progress bar description with current stats
                progress_bar.set_postfix({
                    'Downloaded': self.stats['primary_images_downloaded'],
                    'Failed': self.stats['failed_downloads'],
                    'Size': f"{self.stats['total_download_size'] / (1024*1024):.1f}MB"
                })
                
        progress_bar.close()
        
        # Final statistics
        self.stats['end_time'] = datetime.now()
        duration = self.stats['end_time'] - self.stats['start_time']
        
        self.logger.info("=== DOWNLOAD COMPLETE ===")
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Total objects processed: {len(self.download_results)}")
        self.logger.info(f"Primary images downloaded: {self.stats['primary_images_downloaded']}")
        self.logger.info(f"Additional images downloaded: {self.stats['additional_images_downloaded']}")
        self.logger.info(f"Failed downloads: {self.stats['failed_downloads']}")
        self.logger.info(f"Skipped existing: {self.stats['skipped_existing']}")
        self.logger.info(f"Total download size: {self.stats['total_download_size'] / (1024*1024):.2f} MB")
        
    def save_results_csv(self):
        """Save download results to CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = self.output_dir / f"download_results_{timestamp}.csv"
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.download_results)
        
        # Add summary statistics
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        self.logger.info(f"Results saved to CSV: {csv_file}")
        
        # Save summary statistics
        stats_file = self.output_dir / f"download_statistics_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            # Convert datetime objects to strings for JSON serialization
            stats_for_json = self.stats.copy()
            stats_for_json['start_time'] = self.stats['start_time'].isoformat()
            stats_for_json['end_time'] = self.stats['end_time'].isoformat()
            stats_for_json['duration_seconds'] = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
            json.dump(stats_for_json, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Statistics saved to: {stats_file}")
        
        return csv_file, stats_file
        
    def create_summary_report(self):
        """Create a summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"download_summary_{timestamp}.txt"
        
        duration = self.stats['end_time'] - self.stats['start_time']
        success_rate = (self.stats['primary_images_downloaded'] / self.stats['objects_with_images'] * 100) if self.stats['objects_with_images'] > 0 else 0
        
        report = f"""
MET TEXTILES DATASET IMAGE DOWNLOAD SUMMARY
==========================================

Download completed: {self.stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}
Total duration: {duration}

STATISTICS:
-----------
Total objects in dataset: {self.stats['total_objects']:,}
Objects with images: {self.stats['objects_with_images']:,}
Primary images downloaded: {self.stats['primary_images_downloaded']:,}
Additional images downloaded: {self.stats['additional_images_downloaded']:,}
Total images downloaded: {self.stats['primary_images_downloaded'] + self.stats['additional_images_downloaded']:,}
Failed downloads: {self.stats['failed_downloads']:,}
Skipped existing files: {self.stats['skipped_existing']:,}

SUCCESS RATE: {success_rate:.2f}%

DOWNLOAD SIZE:
--------------
Total downloaded: {self.stats['total_download_size'] / (1024*1024):.2f} MB
Average per image: {(self.stats['total_download_size'] / max(1, self.stats['primary_images_downloaded'] + self.stats['additional_images_downloaded'])) / 1024:.1f} KB

DIRECTORY STRUCTURE:
-------------------
Primary images: {self.images_dir}
Additional images: {self.additional_images_dir}
Logs: {self.logs_dir}

FILES CREATED:
--------------
Images directory: {self.stats['primary_images_downloaded']:,} primary images
Additional images directory: {self.stats['additional_images_downloaded']:,} additional images
Total files: {self.stats['primary_images_downloaded'] + self.stats['additional_images_downloaded']:,} images
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(report)
        self.logger.info(f"Summary report saved to: {report_file}")
        
        return report_file

def main():
    """Main function"""
    print("🎨 MET TEXTILES DATASET IMAGE DOWNLOADER 🎨")
    print("=" * 50)
    
    # Configuration
    JSON_FILE = "/home/user1/Desktop/HAMZA/THESIS/MET/FINAL_CORRECTED_MET_TEXTILES_DATASET/objects_with_images_only/ALL_TEXTILES_AND_TAPESTRIES_WITH_IMAGES_20250706_204750.json"
    OUTPUT_DIR = "MET_TEXTILES_COMPLETE_DATASET"
    MAX_WORKERS = 15  # Adjust based on your internet connection
    
    print(f"📁 JSON File: {JSON_FILE}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    print(f"🔧 Max Workers: {MAX_WORKERS}")
    print()
    
    # Verify JSON file exists
    if not os.path.exists(JSON_FILE):
        print(f"❌ ERROR: JSON file not found: {JSON_FILE}")
        print("Please update the JSON_FILE path in the script.")
        return
        
    try:
        # Initialize downloader
        downloader = METImageDownloader(
            json_file_path=JSON_FILE,
            output_dir=OUTPUT_DIR,
            max_workers=MAX_WORKERS
        )
        
        # Start download process
        downloader.download_all_images()
        
        # Save results
        csv_file, stats_file = downloader.save_results_csv()
        report_file = downloader.create_summary_report()
        
        print("\n🎉 DOWNLOAD COMPLETE! 🎉")
        print(f"📊 Results CSV: {csv_file}")
        print(f"📈 Statistics: {stats_file}")
        print(f"📋 Summary Report: {report_file}")
        print(f"🖼️  Images saved in: {OUTPUT_DIR}")
        
    except KeyboardInterrupt:
        print("\n❌ Download interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
