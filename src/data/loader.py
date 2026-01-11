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
from typing import Tuple, Optional, List
from tqdm import tqdm


class DataLoader:
    """
    Unified data loader for spacecraft telemetry data.
    """
    
    # OPSSAT-AD Zenodo links
    OPSSAT_ZENODO_ID = "12588359"
    OPSSAT_ZENODO_URL = f"https://zenodo.org/records/{OPSSAT_ZENODO_ID}"
    
    # Known label column names (case-insensitive matching applied later)
    LABEL_COLUMN_NAMES = ['label', 'is_anomaly', 'anomaly', 'class', 'target', 'y']
    
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
        
        # Check if dataset.csv already exists
        dataset_csv = opssat_dir / "dataset.csv"
        if dataset_csv.exists() and not force:
            print(f"OPSSAT data already exists at {opssat_dir}")
            return opssat_dir
            
        print("Downloading OPSSAT-AD dataset from Zenodo...")
        print("Note: This requires ~20MB download")
        
        # Zenodo API to get download links
        api_url = f"https://zenodo.org/api/records/{self.OPSSAT_ZENODO_ID}"
        
        try:
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            record = response.json()
            
            # Find the main data file
            files = record.get("files", [])
            
            opssat_dir.mkdir(parents=True, exist_ok=True)
            
            for file_info in files:
                filename = file_info["key"]
                download_url = file_info["links"]["self"]
                file_path = opssat_dir / filename
                
                # Skip if file already exists and not forcing
                if file_path.exists() and not force:
                    print(f"  Skipping {filename} (already exists)")
                    continue
                
                print(f"Downloading {filename}...")
                
                # Download with progress bar
                dl_response = requests.get(download_url, stream=True, timeout=60)
                total_size = int(dl_response.headers.get('content-length', 0))
                
                with open(file_path, 'wb') as f:
                    with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                        for chunk in dl_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                            
                # Extract if zip
                if filename.endswith('.zip'):
                    print(f"Extracting {filename}...")
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(opssat_dir)
                        
            print(f"OPSSAT-AD dataset downloaded to {opssat_dir}")
            
            # List downloaded files
            csv_files = list(opssat_dir.glob("*.csv"))
            print(f"Available CSV files: {[f.name for f in csv_files]}")
            
            return opssat_dir
            
        except Exception as e:
            print(f"Error downloading OPSSAT data: {e}")
            print("Please download manually from: https://zenodo.org/records/12588359")
            raise
    
    def _find_label_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find label column using case-insensitive matching.
        
        Args:
            df: DataFrame to search
            
        Returns:
            Column name if found, None otherwise
        """
        df_cols_lower = {col.lower(): col for col in df.columns}
        
        for candidate in self.LABEL_COLUMN_NAMES:
            if candidate.lower() in df_cols_lower:
                return df_cols_lower[candidate.lower()]
        
        return None
    
    def _select_numeric_features(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        Select only numeric columns and drop ID/text columns.
        
        Args:
            df: Input DataFrame
            exclude_cols: Additional columns to exclude
            
        Returns:
            DataFrame with only numeric feature columns
        """
        exclude_cols = exclude_cols or []
        
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Drop common ID/index columns
        id_patterns = ['id', 'index', 'idx', 'row', 'unnamed']
        cols_to_drop = []
        
        for col in numeric_df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in id_patterns):
                cols_to_drop.append(col)
            elif col in exclude_cols:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            numeric_df = numeric_df.drop(columns=cols_to_drop, errors='ignore')
        
        return numeric_df
    
    def _coerce_labels(self, labels: pd.Series) -> np.ndarray:
        """
        Coerce label values to binary 0/1.
        
        Args:
            labels: Raw label series
            
        Returns:
            Binary numpy array (0 = normal, 1 = anomaly)
        """
        labels = labels.copy()
        
        # Handle string labels
        if labels.dtype == object:
            labels_lower = labels.str.lower().str.strip()
            # Map common anomaly indicators to 1
            anomaly_indicators = ['anomaly', 'anomalous', 'abnormal', 'fault', 'failure', 'true', 'yes', '1']
            labels = labels_lower.apply(lambda x: 1 if x in anomaly_indicators else 0)
        else:
            # Numeric: ensure 0/1
            labels = labels.fillna(0).astype(int)
            # If values are not 0/1, treat non-zero as anomaly
            labels = (labels != 0).astype(int)
        
        return labels.values
            
    def load_opssat(
        self,
        prefer_dataset_csv: bool = True,
        subset: str = "all"
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load the OPSSAT-AD dataset.
        
        Args:
            prefer_dataset_csv: If True, load only dataset.csv (recommended).
                               If False, concatenate all CSV files.
            subset: Which subset to load ("train", "test", "all") - currently "all" only
            
        Returns:
            data: Telemetry features DataFrame (numeric only)
            labels: Binary labels array (0 = normal, 1 = anomaly)
        """
        opssat_dir = self.raw_dir / "opssat"
        
        # Fallback paths to search
        search_paths = [
            opssat_dir,
            self.raw_dir,
            self.data_dir / "opssat",
            self.data_dir,
        ]
        
        dataset_csv = None
        for search_dir in search_paths:
            if not search_dir.exists():
                continue
            # Look for dataset.csv specifically
            candidate = search_dir / "dataset.csv"
            if candidate.exists():
                dataset_csv = candidate
                break
            # Also check subdirectories
            candidates = list(search_dir.glob("**/dataset.csv"))
            if candidates:
                dataset_csv = candidates[0]
                break
        
        if dataset_csv is None:
            # No dataset.csv found, try any CSV
            for search_dir in search_paths:
                if not search_dir.exists():
                    continue
                csv_files = list(search_dir.glob("**/*.csv"))
                if csv_files:
                    # Prefer dataset.csv, then any other
                    for f in csv_files:
                        if 'dataset' in f.name.lower():
                            dataset_csv = f
                            break
                    if dataset_csv is None:
                        # Skip segments.csv if possible
                        non_segment = [f for f in csv_files if 'segment' not in f.name.lower()]
                        dataset_csv = non_segment[0] if non_segment else csv_files[0]
                    break
        
        if dataset_csv is None:
            raise FileNotFoundError(
                f"OPSSAT data not found. Searched in: {[str(p) for p in search_paths]}. "
                "Run download_opssat() first or place dataset.csv in data/raw/opssat/."
            )
        
        print(f"Loading OPSSAT data from: {dataset_csv}")
        
        # Load the CSV
        df = pd.read_csv(dataset_csv)
        print(f"  Loaded {len(df)} samples with {len(df.columns)} columns")
        
        # Find and extract label column
        label_col = self._find_label_column(df)
        
        if label_col:
            print(f"  Found label column: '{label_col}'")
            labels = self._coerce_labels(df[label_col])
            df = df.drop(columns=[label_col])
        else:
            print("  Warning: No label column found. Using zeros as labels.")
            labels = np.zeros(len(df), dtype=int)
        
        # Select numeric features only
        data = self._select_numeric_features(df)
        print(f"  Selected {len(data.columns)} numeric feature columns")
        
        # Handle missing values
        n_missing = data.isna().sum().sum()
        if n_missing > 0:
            print(f"  Filling {n_missing} missing values with column medians")
            data = data.fillna(data.median())
        
        # Final check
        if len(data.columns) == 0:
            raise ValueError("No numeric feature columns found in dataset")
        
        print(f"  Anomaly distribution: {labels.sum()} anomalies ({100*labels.mean():.1f}%)")
        
        return data, labels
        
    def load_synthetic(
        self,
        n_timesteps: int = 5000,
        anomaly_ratio: float = 0.05,
        regenerate: bool = False
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load or generate synthetic data.
        
        Args:
            n_timesteps: Number of time steps
            anomaly_ratio: Fraction of anomalies
            regenerate: Force regeneration
            
        Returns:
            data: Telemetry data DataFrame
            labels: Labels array
        """
        cache_file = self.synthetic_dir / f"synthetic_{n_timesteps}_{int(anomaly_ratio*100)}.pkl"
        
        if cache_file.exists() and not regenerate:
            print(f"Loading cached synthetic data from {cache_file}")
            import pickle
            with open(cache_file, 'rb') as f:
                data, labels_df = pickle.load(f)
                # Convert labels DataFrame to array
                if isinstance(labels_df, pd.DataFrame):
                    if 'is_anomaly' in labels_df.columns:
                        labels = labels_df['is_anomaly'].values
                    else:
                        labels = labels_df.values.flatten()
                else:
                    labels = np.asarray(labels_df)
                return data, labels
                
        # Generate new data
        from .synthetic_generator import SpacecraftTelemetryGenerator
        
        print(f"Generating synthetic data: {n_timesteps} timesteps, {anomaly_ratio:.1%} anomalies")
        
        generator = SpacecraftTelemetryGenerator(n_timesteps=n_timesteps)
        data, labels_df = generator.generate(anomaly_ratio=anomaly_ratio)
        
        # Cache
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump((data, labels_df), f)
        
        # Convert labels
        if isinstance(labels_df, pd.DataFrame):
            if 'is_anomaly' in labels_df.columns:
                labels = labels_df['is_anomaly'].values
            else:
                labels = labels_df.values.flatten()
        else:
            labels = np.asarray(labels_df)
            
        return data, labels
        
    def load_csv(
        self,
        data_path: str,
        labels_path: Optional[str] = None,
        timestamp_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load data from CSV files.
        
        Args:
            data_path: Path to telemetry data CSV
            labels_path: Optional path to labels CSV
            timestamp_col: Optional timestamp column name
            
        Returns:
            data: Telemetry data DataFrame
            labels: Labels array
        """
        df = pd.read_csv(data_path)
        
        # Handle timestamp
        if timestamp_col and timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.set_index(timestamp_col)
            
        # Load or create labels
        if labels_path:
            labels_df = pd.read_csv(labels_path)
            label_col = self._find_label_column(labels_df)
            if label_col:
                labels = self._coerce_labels(labels_df[label_col])
            else:
                labels = labels_df.values.flatten().astype(int)
        else:
            # Try to find label column in data
            label_col = self._find_label_column(df)
            
            if label_col:
                labels = self._coerce_labels(df[label_col])
                df = df.drop(columns=[label_col])
            else:
                labels = np.zeros(len(df), dtype=int)
        
        # Select numeric features
        data = self._select_numeric_features(df)
        
        # Fill missing
        data = data.fillna(data.median())
                
        return data, labels


def load_data(
    source: str = "synthetic",
    **kwargs
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Convenience function to load data from various sources.
    
    Args:
        source: Data source ("synthetic", "opssat", or path to CSV)
        **kwargs: Additional arguments passed to loader
        
    Returns:
        data: Telemetry data DataFrame
        labels: Labels array
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


__all__ = ['DataLoader', 'load_data']
