#!/usr/bin/env python3
"""
Download OPSSAT-AD dataset from Zenodo.
ESA OPS-SAT Anomaly Detection Dataset
"""

import os
import sys
import zipfile
import requests
from tqdm import tqdm

# Zenodo record for OPSSAT-AD
ZENODO_RECORD_ID = "12588359"
ZENODO_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"

def get_download_urls():
    """Get file download URLs from Zenodo API."""
    print(f"Fetching file information from Zenodo record {ZENODO_RECORD_ID}...")
    
    response = requests.get(ZENODO_URL)
    if response.status_code != 200:
        print(f"Error: Could not fetch Zenodo record (status {response.status_code})")
        return None
    
    data = response.json()
    files = data.get('files', [])
    
    if not files:
        print("Error: No files found in Zenodo record")
        return None
    
    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  - {f['key']} ({f['size'] / 1024 / 1024:.1f} MB)")
    
    return files


def download_file(url, destination, desc="Downloading"):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def main():
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data', 'raw')
    
    os.makedirs(data_dir, exist_ok=True)
    
    print("=" * 60)
    print("OPSSAT-AD Dataset Downloader")
    print("=" * 60)
    print()
    print("Dataset: OPS-SAT Anomaly Detection Dataset")
    print("Source: ESA OPS-SAT CubeSat Mission")
    print(f"Zenodo DOI: 10.5281/zenodo.{ZENODO_RECORD_ID}")
    print()
    
    # Get download URLs
    files = get_download_urls()
    
    if files is None:
        print("\nFalling back to direct download URL...")
        # Direct download URL (may need to be updated)
        direct_url = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/opssat_anomaly_dataset.zip"
        zip_path = os.path.join(data_dir, 'opssat_anomaly_dataset.zip')
        
        print(f"\nDownloading from: {direct_url}")
        try:
            download_file(direct_url, zip_path, desc="OPSSAT Dataset")
        except Exception as e:
            print(f"Error downloading: {e}")
            print("\nPlease download manually from:")
            print(f"  https://zenodo.org/records/{ZENODO_RECORD_ID}")
            return False
    else:
        # Download all files
        print("\nDownloading files...")
        for file_info in files:
            filename = file_info['key']
            download_url = file_info['links']['self']
            destination = os.path.join(data_dir, filename)
            
            if os.path.exists(destination):
                print(f"  Skipping {filename} (already exists)")
                continue
            
            print(f"\nDownloading: {filename}")
            download_file(download_url, destination, desc=filename)
    
    # Extract if zip file
    zip_files = [f for f in os.listdir(data_dir) if f.endswith('.zip')]
    for zip_file in zip_files:
        zip_path = os.path.join(data_dir, zip_file)
        print(f"\nExtracting {zip_file}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_dir)
        
        print(f"Extracted to: {data_dir}")
    
    # List extracted files
    print("\nDataset files:")
    for root, dirs, files in os.walk(data_dir):
        level = root.replace(data_dir, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = '  ' * (level + 1)
        for file in files[:10]:  # Show first 10 files
            print(f"{sub_indent}{file}")
        if len(files) > 10:
            print(f"{sub_indent}... and {len(files) - 10} more files")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print(f"\nData saved to: {data_dir}")
    print("\nNext steps:")
    print("  1. Run: python scripts/train.py")
    print("  2. Or run: python scripts/demo.py (uses synthetic data)")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
