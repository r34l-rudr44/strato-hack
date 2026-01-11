#!/usr/bin/env python3
"""
Download OPSSAT-AD dataset from Zenodo.
ESA OPS-SAT Anomaly Detection Dataset

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --force   # Re-download even if exists
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader


def main():
    parser = argparse.ArgumentParser(description='Download OPSSAT-AD dataset from Zenodo')
    parser.add_argument('--force', action='store_true', help='Force re-download even if files exist')
    parser.add_argument('--data-dir', type=str, default='data', help='Base data directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("OPSSAT-AD Dataset Downloader")
    print("=" * 60)
    print()
    print("Dataset: OPS-SAT Anomaly Detection Dataset")
    print("Source:  ESA OPS-SAT CubeSat Mission")
    print("Zenodo:  https://zenodo.org/records/12588359")
    print()
    
    # Initialize loader with specified data directory
    loader = DataLoader(data_dir=args.data_dir)
    
    try:
        # Download dataset
        opssat_dir = loader.download_opssat(force=args.force)
        
        print()
        print("=" * 60)
        print("Download complete!")
        print("=" * 60)
        print()
        print(f"Data saved to: {opssat_dir}")
        print()
        
        # List files
        csv_files = list(opssat_dir.glob("*.csv"))
        if csv_files:
            print("Downloaded files:")
            for f in csv_files:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name} ({size_mb:.1f} MB)")
        
        print()
        print("Next steps:")
        print("  1. Train models:    python scripts/train.py")
        print("  2. Evaluate:        python scripts/evaluate.py")
        print()
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        print()
        print("Manual download instructions:")
        print("  1. Visit: https://zenodo.org/records/12588359")
        print("  2. Download dataset.csv and segments.csv")
        print(f"  3. Place files in: {loader.raw_dir / 'opssat'}")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
