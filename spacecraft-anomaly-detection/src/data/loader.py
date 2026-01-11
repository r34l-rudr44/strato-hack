"""
Data Loader Module
==================

Handles loading data from various sources:
- OPSSAT-AD dataset from Zenodo
- Synthetic data for demos
- Custom CSV/Parquet files
"""

import os
import zipfile
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Union
from tqdm import tqdm
import yaml


class DataLoader:
    """
    Unified data loader for spacecraft telemetry data.
    """
    
    # OPSSAT-AD Zenodo links
    OPSSAT_ZENODO_ID = "12588359"
    OPSSAT_ZENODO_URL = f"https://zenodo.org/records/{OPSSAT_ZENODO_ID}"
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Base directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.synthetic_dir = self.data_dir / "synthetic"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.synthetic_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def download_opssat(self, force: bool = False) -> Path:
        """
        Download the OPSSAT-AD dataset from Zenodo.
        
        Args:
            force: Force re-download even if exists
            
        Returns:
            Path to downloaded data directory
        """
        opssat_dir = self.raw_dir / "opssat"
        
        if opssat_dir.exists() and not force:
            print(f"OPSSAT data already exists at {opssat_dir}")
            return opssat_dir
            
        print(f"Downloading OPSSAT-AD dataset from Zenodo...")
        print(f"Note: This requires ~50MB download")
        
        # Zenodo API to get download links
        api_url = f"https://zenodo.org/api/records/{self.OPSSAT_ZENODO_ID}"
        
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            record = response.json()
            
            # Find the main data file
            files = record.get("files", [])
            
            opssat_dir.mkdir(parents=True, exist_ok=True)
            
            for file_info in files:
                filename = file_info["key"]
                download_url = file_info["links"]["self"]
                file_path = opssat_dir / filename
                
                print(f"Downloading {filename}...")
                
                # Download with progress bar
                response = requests.get(download_url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                with open(file_path, 'wb') as f:
                    with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                            
                # Extract if zip
                if filename.endswith('.zip'):
                    print(f"Extracting {filename}...")
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(opssat_dir)
                        
            print(f"OPSSAT-AD dataset downloaded to {opssat_dir}")
            return opssat_dir
            
        except Exception as e:
            print(f"Error downloading OPSSAT data: {e}")
            print("Please download manually from: https://zenodo.org/records/12588359")
            raise
            
    def load_opssat(
        self,
        subset: str = "all"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the OPSSAT-AD dataset.
        
        Args:
            subset: Which subset to load ("train", "test", "all")
            
        Returns:
            data: Telemetry data DataFrame
            labels: Labels DataFrame
        """
        opssat_dir = self.raw_dir / "opssat"
        
        if not opssat_dir.exists():
            raise FileNotFoundError(
                f"OPSSAT data not found at {opssat_dir}. "
                "Run download_opssat() first."
            )
            
        # Find data files
        data_files = list(opssat_dir.glob("*.csv"))
        
        if not data_files:
            # Check subdirectories
            data_files = list(opssat_dir.glob("**/*.csv"))
            
        if not data_files:
            raise FileNotFoundError(f"No CSV files found in {opssat_dir}")
            
        print(f"Found {len(data_files)} data files")
        
        # Load and concatenate
        dfs = []
        for file in data_files:
            df = pd.read_csv(file)
            dfs.append(df)
            
        combined = pd.concat(dfs, ignore_index=True)
        
        # Separate features and labels
        label_columns = ['label', 'is_anomaly', 'anomaly', 'Label', 'Anomaly']
        label_col = None
        
        for col in label_columns:
            if col in combined.columns:
                label_col = col
                break
                
        if label_col:
            labels = combined[[label_col]].copy()
            labels.columns = ['is_anomaly']
            data = combined.drop(columns=[label_col])
        else:
            # No labels found - assume last column or create dummy
            print("Warning: No label column found. Using dummy labels.")
            data = combined
            labels = pd.DataFrame({'is_anomaly': np.zeros(len(combined))})
            
        return data, labels
        
    def load_synthetic(
        self,
        n_timesteps: int = 5000,
        anomaly_ratio: float = 0.05,
        regenerate: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load or generate synthetic data.
        
        Args:
            n_timesteps: Number of time steps
            anomaly_ratio: Fraction of anomalies
            regenerate: Force regeneration
            
        Returns:
            data: Telemetry data DataFrame
            labels: Labels DataFrame
        """
        cache_file = self.synthetic_dir / f"synthetic_{n_timesteps}_{int(anomaly_ratio*100)}.pkl"
        
        if cache_file.exists() and not regenerate:
            print(f"Loading cached synthetic data from {cache_file}")
            import pickle
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
                
        # Generate new data
        from .synthetic_generator import SpacecraftTelemetryGenerator
        
        print(f"Generating synthetic data: {n_timesteps} timesteps, {anomaly_ratio:.1%} anomalies")
        
        generator = SpacecraftTelemetryGenerator(n_timesteps=n_timesteps)
        data, labels = generator.generate(anomaly_ratio=anomaly_ratio)
        
        # Cache
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump((data, labels), f)
            
        return data, labels
        
    def load_csv(
        self,
        data_path: str,
        labels_path: Optional[str] = None,
        timestamp_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data from CSV files.
        
        Args:
            data_path: Path to telemetry data CSV
            labels_path: Optional path to labels CSV
            timestamp_col: Optional timestamp column name
            
        Returns:
            data: Telemetry data DataFrame
            labels: Labels DataFrame
        """
        data = pd.read_csv(data_path)
        
        # Handle timestamp
        if timestamp_col and timestamp_col in data.columns:
            data[timestamp_col] = pd.to_datetime(data[timestamp_col])
            data = data.set_index(timestamp_col)
            
        # Load or create labels
        if labels_path:
            labels = pd.read_csv(labels_path)
        else:
            # Try to find label column in data
            label_columns = ['label', 'is_anomaly', 'anomaly', 'Label', 'Anomaly']
            label_col = None
            
            for col in label_columns:
                if col in data.columns:
                    label_col = col
                    break
                    
            if label_col:
                labels = data[[label_col]].copy()
                labels.columns = ['is_anomaly']
                data = data.drop(columns=[label_col])
            else:
                labels = pd.DataFrame({'is_anomaly': np.zeros(len(data))})
                
        return data, labels


def load_data(
    source: str = "synthetic",
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load data from various sources.
    
    Args:
        source: Data source ("synthetic", "opssat", or path to CSV)
        **kwargs: Additional arguments passed to loader
        
    Returns:
        data: Telemetry data DataFrame
        labels: Labels DataFrame
    """
    loader = DataLoader()
    
    if source == "synthetic":
        return loader.load_synthetic(**kwargs)
    elif source == "opssat":
        return loader.load_opssat(**kwargs)
    elif os.path.exists(source):
        return loader.load_csv(source, **kwargs)
    else:
        raise ValueError(f"Unknown data source: {source}")


# Init file
__all__ = ['DataLoader', 'load_data']
