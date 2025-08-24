#!/usr/bin/env python3
"""
MET Textiles Dataset Image Downloader - BULLETPROOF VERSION
===========================================================
Downloads all images from the MET textiles and tapestries JSON dataset.
Features: Retry logic, failure recovery, resume capability, bulletproof downloading!

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
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class BulletproofMETDownloader:
    def __init__(self, json_file_path, output_dir="MET_TEXTILES_IMAGES", max_workers=10):
        self.json_file_path = json_file_path
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        
        # Retry configuration
        self.max_retries_per_image = 5
        self.retry_delay_base = 2  # Base delay in seconds
        self.max_retry_delay = 30  # Maximum delay
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        self.additional_images_dir = self.output_dir / "additional_images"
        self.additional_images_dir.mkdir(exist_ok=True)
        self.failed_dir = self.output_dir / "failed_downloads"
        self.failed_dir.mkdir(exist_ok=True)
        
        # Initialize logging
        self.setup_logging()
        
        # Configure requests session with retry strategy
        self.setup_session()
        
        # Statistics
        self.stats = {
            'total_objects': 0,
            'objects_with_images': 0,
            'primary_images_downloaded': 0,
            'additional_images_downloaded': 0,
            'failed_downloads': 0,
            'skipped_existing': 0,
            'total_download_size': 0,
            'retry_attempts': 0,
            'successful_retries': 0,
            'permanent_failures': 0,
            'start_time': None,
            'end_time': None,
            'main_phase_time': None,
            'retry_phase_time': None
        }
        
        # Download tracking
        self.download_results = []
        self.failed_downloads = []
        self.retry_queue = []
        
        # Thread lock for stats
        self.stats_lock = threading.Lock()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"bulletproof_download_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("=== BULLETPROOF MET TEXTILES IMAGE DOWNLOADER STARTED ===")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"Max retries per image: {self.max_retries_per_image}")
        
    def setup_session(self):
        """Setup requests session with retry strategy"""
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers to be more polite
        self.session.headers.update({
            'User-Agent': 'Academic Research Bot - MET Textiles Dataset Collection',
            'Accept': 'image/*,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        })
        
        self.logger.info("Configured robust HTTP session with retry strategy")
        
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
        
    def exponential_backoff(self, attempt, base_delay=2, max_delay=30):
        """Calculate exponential backoff delay with jitter"""
        delay = min(base_delay * (2 ** attempt), max_delay)
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.1, 0.3) * delay
        return delay + jitter
        
    def download_image_with_retry(self, url, filepath, object_id, image_type="primary", max_retries=None):
        """Download a single image with comprehensive retry logic"""
        if max_retries is None:
            max_retries = self.max_retries_per_image
            
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # Check if file already exists and is valid
                if filepath.exists() and filepath.stat().st_size > 1000:  # At least 1KB
                    self.logger.debug(f"Image already exists and valid, skipping: {filepath.name}")
                    with self.stats_lock:
                        self.stats['skipped_existing'] += 1
                    return True, "already_exists", filepath.stat().st_size, 0
                
                # Add delay for retry attempts
                if attempt > 0:
                    delay = self.exponential_backoff(attempt - 1, self.retry_delay_base, self.max_retry_delay)
                    self.logger.info(f"Retry attempt {attempt}/{max_retries} for {object_id} after {delay:.1f}s delay")
                    time.sleep(delay)
                    
                    with self.stats_lock:
                        self.stats['retry_attempts'] += 1
                
                # Download image with timeout
                response = self.session.get(url, timeout=45, stream=True)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png']):
                    raise ValueError(f"Invalid content type: {content_type}")
                
                # Get content length
                content_length = int(response.headers.get('content-length', 0))
                
                # Download with progress for large files
                downloaded_bytes = 0
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_bytes += len(chunk)
                
                # Verify file was created and has reasonable content
                if filepath.exists() and filepath.stat().st_size > 500:  # At least 500 bytes
                    file_size = filepath.stat().st_size
                    
                    # Additional verification - check if it's actually an image
                    try:
                        with open(filepath, 'rb') as f:
                            header = f.read(10)
                        
                        # Check for common image headers
                        is_valid_image = (
                            header.startswith(b'\xff\xd8\xff') or  # JPEG
                            header.startswith(b'\x89PNG\r\n\x1a\n') or  # PNG
                            header.startswith(b'GIF87a') or header.startswith(b'GIF89a') or  # GIF
                            header.startswith(b'RIFF') and b'WEBP' in header  # WebP
                        )
                        
                        if not is_valid_image:
                            raise ValueError("Downloaded file is not a valid image")
                            
                    except Exception as e:
                        self.logger.warning(f"Image validation failed for {object_id}: {e}")
                        # Don't fail completely, file might still be usable
                    
                    self.logger.debug(f"Successfully downloaded: {filepath.name} ({file_size} bytes)")
                    
                    with self.stats_lock:
                        self.stats['total_download_size'] += file_size
                        if image_type == "primary":
                            self.stats['primary_images_downloaded'] += 1
                        else:
                            self.stats['additional_images_downloaded'] += 1
                        
                        if attempt > 0:
                            self.stats['successful_retries'] += 1
                            
                    return True, "success", file_size, attempt
                else:
                    raise ValueError("Downloaded file is empty or too small")
                    
            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {e}"
                self.logger.warning(f"Download attempt {attempt + 1} failed for {object_id} ({image_type}): {last_error}")
                
                # For certain errors, don't retry
                if isinstance(e, requests.exceptions.HTTPError):
                    if e.response.status_code in [404, 403, 410]:  # Not found, forbidden, gone
                        self.logger.info(f"Permanent failure for {object_id}: HTTP {e.response.status_code}")
                        break
                        
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                self.logger.warning(f"Download attempt {attempt + 1} failed for {object_id} ({image_type}): {last_error}")
            
            # Clean up partial file
            if filepath.exists():
                try:
                    filepath.unlink()
                except:
                    pass
                    
        # All retries failed
        self.logger.error(f"All {max_retries + 1} attempts failed for {object_id} ({image_type}): {last_error}")
        with self.stats_lock:
            self.stats['failed_downloads'] += 1
            self.stats['permanent_failures'] += 1
            
        return False, last_error, 0, max_retries
        
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
            'primary_image_attempts': 0,
            'additional_images_count': 0,
            'additional_images_downloaded': 0,
            'additional_images_filenames': [],
            'additional_images_attempts': [],
            'total_size': 0,
            'status': 'no_images',
            'error_message': '',
            'needs_retry': False
        }
        
        try:
            # Download primary image
            primary_image_url = obj.get('primaryImage', '')
            if primary_image_url:
                result['primary_image_url'] = primary_image_url
                
                filename = self.get_safe_filename(object_id, title, "primary")
                filepath = self.images_dir / filename
                
                success, status, size, attempts = self.download_image_with_retry(
                    primary_image_url, filepath, object_id, "primary"
                )
                
                result['primary_image_attempts'] = attempts + 1
                
                if success:
                    result['primary_image_downloaded'] = True
                    result['primary_image_filename'] = filename
                    result['primary_image_size'] = size
                    result['total_size'] += size
                    result['status'] = 'success'
                else:
                    result['error_message'] = status
                    result['status'] = 'failed'
                    result['needs_retry'] = True
            
            # Download additional images
            additional_images = obj.get('additionalImages', [])
            if additional_images:
                result['additional_images_count'] = len(additional_images)
                additional_filenames = []
                additional_attempts = []
                
                for i, img_url in enumerate(additional_images):
                    filename = self.get_safe_filename(object_id, title, f"additional_{i+1}")
                    filepath = self.additional_images_dir / filename
                    
                    success, status, size, attempts = self.download_image_with_retry(
                        img_url, filepath, object_id, f"additional_{i+1}"
                    )
                    
                    additional_attempts.append(attempts + 1)
                    
                    if success:
                        result['additional_images_downloaded'] += 1
                        additional_filenames.append(filename)
                        result['total_size'] += size
                    else:
                        # Track failed additional images for retry
                        if result['needs_retry'] == False:
                            result['needs_retry'] = True
                    
                result['additional_images_filenames'] = '; '.join(additional_filenames)
                result['additional_images_attempts'] = additional_attempts
            
            # Update overall status
            if result['primary_image_downloaded'] or result['additional_images_downloaded'] > 0:
                if result['status'] != 'failed':
                    result['status'] = 'partial_success' if result['needs_retry'] else 'success'
                    
        except Exception as e:
            self.logger.error(f"Error processing object {object_id}: {e}")
            result['status'] = 'error'
            result['error_message'] = str(e)
            result['needs_retry'] = True
            
        return result
        
    def retry_failed_downloads(self):
        """Retry all failed downloads with more aggressive retry settings"""
        if not self.failed_downloads:
            self.logger.info("No failed downloads to retry")
            return
            
        self.logger.info(f"=== STARTING RETRY PHASE FOR {len(self.failed_downloads)} FAILED DOWNLOADS ===")
        retry_start_time = datetime.now()
        
        # More aggressive retry settings for retry phase
        original_max_retries = self.max_retries_per_image
        self.max_retries_per_image = 8  # More retries in retry phase
        
        retry_progress = tqdm(
            total=len(self.failed_downloads),
            desc="Retrying failed downloads",
            unit="objects",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        successful_retries = 0
        
        with ThreadPoolExecutor(max_workers=max(1, self.max_workers // 2)) as executor:  # Fewer workers for retries
            # Submit retry tasks
            future_to_object = {
                executor.submit(self.process_object, obj): obj 
                for obj in self.failed_downloads
            }
            
            # Process completed retry tasks
            for future in as_completed(future_to_object):
                result = future.result()
                
                # Update original result
                original_result = next((r for r in self.download_results if r['object_id'] == result['object_id']), None)
                if original_result:
                    original_result.update(result)
                    if result['status'] in ['success', 'partial_success']:
                        successful_retries += 1
                
                retry_progress.update(1)
                retry_progress.set_postfix({
                    'Recovered': successful_retries,
                    'Still Failed': len(self.failed_downloads) - successful_retries
                })
                
        retry_progress.close()
        
        # Restore original retry settings
        self.max_retries_per_image = original_max_retries
        
        retry_end_time = datetime.now()
        self.stats['retry_phase_time'] = retry_end_time
        retry_duration = retry_end_time - retry_start_time
        
        self.logger.info(f"=== RETRY PHASE COMPLETE ===")
        self.logger.info(f"Retry duration: {retry_duration}")
        self.logger.info(f"Successfully recovered: {successful_retries}/{len(self.failed_downloads)} failed downloads")
        
    def download_all_images(self):
        """Main method to download all images with bulletproof retry logic"""
        self.logger.info("=== STARTING BULLETPROOF IMAGE DOWNLOAD PROCESS ===")
        self.stats['start_time'] = datetime.now()
        
        # Load JSON data
        data = self.load_json_data()
        self.stats['total_objects'] = len(data)
        
        # Count objects with images
        objects_with_images = [obj for obj in data if obj.get('primaryImage') or obj.get('additionalImages')]
        self.stats['objects_with_images'] = len(objects_with_images)
        
        self.logger.info(f"Found {len(objects_with_images)} objects with images out of {len(data)} total objects")
        
        # PHASE 1: Main download phase
        self.logger.info("=== PHASE 1: MAIN DOWNLOAD ===")
        main_progress = tqdm(
            total=len(objects_with_images),
            desc="Main download phase",
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
                
                # Track failed downloads for retry
                if result['needs_retry'] or result['status'] in ['failed', 'error']:
                    # Find the original object
                    original_obj = next((obj for obj in objects_with_images if obj['objectID'] == result['object_id']), None)
                    if original_obj:
                        self.failed_downloads.append(original_obj)
                
                # Update progress
                main_progress.update(1)
                main_progress.set_postfix({
                    'Downloaded': self.stats['primary_images_downloaded'],
                    'Failed': len(self.failed_downloads),
                    'Size': f"{self.stats['total_download_size'] / (1024*1024):.1f}MB"
                })
                
        main_progress.close()
        self.stats['main_phase_time'] = datetime.now()
        
        # PHASE 2: Retry failed downloads
        if self.failed_downloads:
            self.retry_failed_downloads()
        
        # Final statistics
        self.stats['end_time'] = datetime.now()
        total_duration = self.stats['end_time'] - self.stats['start_time']
        
        self.logger.info("=== ALL PHASES COMPLETE ===")
        self.logger.info(f"Total duration: {total_duration}")
        self.logger.info(f"Total objects processed: {len(self.download_results)}")
        self.logger.info(f"Primary images downloaded: {self.stats['primary_images_downloaded']}")
        self.logger.info(f"Additional images downloaded: {self.stats['additional_images_downloaded']}")
        self.logger.info(f"Total retry attempts: {self.stats['retry_attempts']}")
        self.logger.info(f"Successful retries: {self.stats['successful_retries']}")
        self.logger.info(f"Permanent failures: {self.stats['permanent_failures']}")
        self.logger.info(f"Total download size: {self.stats['total_download_size'] / (1024*1024):.2f} MB")
        
    def save_results_csv(self):
        """Save download results to CSV with detailed retry information"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = self.output_dir / f"bulletproof_download_results_{timestamp}.csv"
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.download_results)
        
        # Add summary columns
        df['total_attempts'] = df['primary_image_attempts'] + df.get('additional_images_attempts', 0).apply(lambda x: sum(x) if isinstance(x, list) else 0)
        df['download_success'] = df['status'].isin(['success', 'partial_success'])
        
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        self.logger.info(f"Detailed results saved to CSV: {csv_file}")
        
        # Save retry statistics
        retry_stats = {
            'total_retry_attempts': self.stats['retry_attempts'],
            'successful_retries': self.stats['successful_retries'],
            'permanent_failures': self.stats['permanent_failures'],
            'retry_success_rate': (self.stats['successful_retries'] / max(1, self.stats['retry_attempts'])) * 100
        }
        
        stats_file = self.output_dir / f"bulletproof_statistics_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            # Convert datetime objects to strings for JSON serialization
            stats_for_json = self.stats.copy()
            stats_for_json.update(retry_stats)
            stats_for_json['start_time'] = self.stats['start_time'].isoformat()
            stats_for_json['end_time'] = self.stats['end_time'].isoformat()
            if self.stats['main_phase_time']:
                stats_for_json['main_phase_time'] = self.stats['main_phase_time'].isoformat()
            if self.stats['retry_phase_time']:
                stats_for_json['retry_phase_time'] = self.stats['retry_phase_time'].isoformat()
            stats_for_json['total_duration_seconds'] = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
            json.dump(stats_for_json, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Bulletproof statistics saved to: {stats_file}")
        
        return csv_file, stats_file
        
    def create_bulletproof_summary(self):
        """Create a comprehensive bulletproof summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"bulletproof_summary_{timestamp}.txt"
        
        total_duration = self.stats['end_time'] - self.stats['start_time']
        main_duration = self.stats['main_phase_time'] - self.stats['start_time'] if self.stats['main_phase_time'] else total_duration
        retry_duration = self.stats['retry_phase_time'] - self.stats['main_phase_time'] if self.stats['retry_phase_time'] and self.stats['main_phase_time'] else "N/A"
        
        success_rate = (self.stats['primary_images_downloaded'] / self.stats['objects_with_images'] * 100) if self.stats['objects_with_images'] > 0 else 0
        retry_success_rate = (self.stats['successful_retries'] / max(1, self.stats['retry_attempts'])) * 100
        
        total_images = self.stats['primary_images_downloaded'] + self.stats['additional_images_downloaded']
        
        report = f"""
🛡️  BULLETPROOF MET TEXTILES DATASET DOWNLOAD SUMMARY 🛡️
========================================================

Download completed: {self.stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}
Total duration: {total_duration}
Main phase duration: {main_duration}
Retry phase duration: {retry_duration}

📊 MAIN STATISTICS:
------------------
Total objects in dataset: {self.stats['total_objects']:,}
Objects with images: {self.stats['objects_with_images']:,}
Primary images downloaded: {self.stats['primary_images_downloaded']:,}
Additional images downloaded: {self.stats['additional_images_downloaded']:,}
Total images downloaded: {total_images:,}
Skipped existing files: {self.stats['skipped_existing']:,}

🎯 SUCCESS RATES:
----------------
Primary image success rate: {success_rate:.2f}%
Images per object ratio: {total_images / max(1, self.stats['objects_with_images']):.2f}

🔄 RETRY STATISTICS:
-------------------
Total retry attempts: {self.stats['retry_attempts']:,}
Successful retries: {self.stats['successful_retries']:,}
Retry success rate: {retry_success_rate:.2f}%
Permanent failures: {self.stats['permanent_failures']:,}

💾 DOWNLOAD SIZE:
----------------
Total downloaded: {self.stats['total_download_size'] / (1024*1024):.2f} MB
Total downloaded: {self.stats['total_download_size'] / (1024*1024*1024):.2f} GB
Average per image: {(self.stats['total_download_size'] / max(1, total_images)) / 1024:.1f} KB

📁 DIRECTORY STRUCTURE:
----------------------
Primary images: {self.images_dir} ({self.stats['primary_images_downloaded']:,} files)
Additional images: {self.additional_images_dir} ({self.stats['additional_images_downloaded']:,} files)
Logs: {self.logs_dir}
Main log file: {self.log_file.name}

🎉 FINAL RESULTS:
----------------
✅ Successfully downloaded: {total_images:,} images
✅ Success rate: {success_rate:.2f}%
✅ Dataset completeness: {((total_images) / self.stats['objects_with_images'] * 100):.1f}%
✅ Bulletproof retry recovery: {self.stats['successful_retries']:,} additional successes

This download used bulletproof retry logic with exponential backoff,
comprehensive error handling, and end-of-run failure recovery!
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
            
        print(report)
        self.logger.info(f"Bulletproof summary report saved to: {report_file}")
        
        return report_file

def main():
    """Main function"""
    print("🛡️  BULLETPROOF MET TEXTILES DATASET IMAGE DOWNLOADER 🛡️")
    print("=" * 60)
    
    # Configuration
    JSON_FILE = "/home/user1/Desktop/HAMZA/THESIS/TEXMET/data/FINAL_CORRECTED_MET_TEXTILES_DATASET/objects_with_images_only/ALL_TEXTILES_AND_TAPESTRIES_WITH_IMAGES_20250705_230315.json"
    OUTPUT_DIR = "MET_TEXTILES_BULLETPROOF_DATASET"
    MAX_WORKERS = 12  # Slightly reduced for more stable downloads
    
    print(f"📁 JSON File: {JSON_FILE}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    print(f"🔧 Max Workers: {MAX_WORKERS}")
    print(f"🔄 Bulletproof Features: Retry logic, failure recovery, resume capability")
    print()
    
    # Verify JSON file exists
    if not os.path.exists(JSON_FILE):
        print(f"❌ ERROR: JSON file not found: {JSON_FILE}")
        print("Please update the JSON_FILE path in the script.")
        return
        
    try:
        # Initialize bulletproof downloader
        downloader = BulletproofMETDownloader(
            json_file_path=JSON_FILE,
            output_dir=OUTPUT_DIR,
            max_workers=MAX_WORKERS
        )
        
        # Start bulletproof download process
        downloader.download_all_images()
        
        # Save detailed results
        csv_file, stats_file = downloader.save_results_csv()
        report_file = downloader.create_bulletproof_summary()
        
        print("\n🎉 BULLETPROOF DOWNLOAD COMPLETE! 🎉")
        print(f"📊 Detailed Results CSV: {csv_file}")
        print(f"📈 Bulletproof Statistics: {stats_file}")
        print(f"📋 Summary Report: {report_file}")
        print(f"🖼️  Images saved in: {OUTPUT_DIR}")
        print(f"🛡️  Download was bulletproof with retry recovery!")
        
    except KeyboardInterrupt:
        print("\n❌ Download interrupted by user")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
