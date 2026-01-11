"""
Data Preprocessor Module
========================

Preprocessing and feature engineering for spacecraft telemetry data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats
from scipy.fft import fft


class DataPreprocessor:
    """
    Preprocessor for spacecraft telemetry data with feature engineering.
    """
    
    def __init__(
        self,
        normalization: str = "standard",
        window_size: int = 50,
        stride: int = 10,
        feature_config: Optional[Dict] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            normalization: Normalization method ("standard", "minmax", "robust")
            window_size: Window size for rolling features
            stride: Stride for sliding window
            feature_config: Feature engineering configuration
        """
        self.normalization = normalization
        self.window_size = window_size
        self.stride = stride
        self.feature_config = feature_config or {
            "rolling_mean": True,
            "rolling_std": True,
            "rolling_min": True,
            "rolling_max": True,
            "rate_of_change": True,
            "fft_features": True,
        }
        
        # Initialize scalers
        self.scalers: Dict[str, object] = {}
        self.fitted = False
        
    def _get_scaler(self):
        """Get the appropriate scaler based on normalization method."""
        if self.normalization == "standard":
            return StandardScaler()
        elif self.normalization == "minmax":
            return MinMaxScaler()
        elif self.normalization == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")
            
    def _add_rolling_features(
        self,
        df: pd.DataFrame,
        column: str
    ) -> pd.DataFrame:
        """Add rolling window features for a column."""
        features = {}
        
        if self.feature_config.get("rolling_mean", True):
            features[f"{column}_rolling_mean"] = df[column].rolling(
                window=self.window_size, min_periods=1
            ).mean()
            
        if self.feature_config.get("rolling_std", True):
            features[f"{column}_rolling_std"] = df[column].rolling(
                window=self.window_size, min_periods=1
            ).std().fillna(0)
            
        if self.feature_config.get("rolling_min", True):
            features[f"{column}_rolling_min"] = df[column].rolling(
                window=self.window_size, min_periods=1
            ).min()
            
        if self.feature_config.get("rolling_max", True):
            features[f"{column}_rolling_max"] = df[column].rolling(
                window=self.window_size, min_periods=1
            ).max()
            
        return pd.DataFrame(features)
        
    def _add_rate_of_change(
        self,
        df: pd.DataFrame,
        column: str
    ) -> pd.DataFrame:
        """Add rate of change (derivative) features."""
        features = {}
        
        # First derivative
        features[f"{column}_diff1"] = df[column].diff().fillna(0)
        
        # Second derivative
        features[f"{column}_diff2"] = df[column].diff().diff().fillna(0)
        
        # Percent change
        features[f"{column}_pct_change"] = df[column].pct_change().fillna(0).replace(
            [np.inf, -np.inf], 0
        )
        
        return pd.DataFrame(features)
        
    def _add_fft_features(
        self,
        df: pd.DataFrame,
        column: str,
        n_components: int = 5
    ) -> pd.DataFrame:
        """Add FFT-based frequency domain features."""
        features = {}
        
        # Compute FFT magnitude for rolling windows
        values = df[column].values
        fft_mags = []
        
        for i in range(len(values)):
            start = max(0, i - self.window_size + 1)
            window = values[start:i+1]
            
            if len(window) >= 4:  # Need minimum samples for FFT
                fft_vals = np.abs(fft(window))[:len(window)//2]
                
                # Get top frequency components
                if len(fft_vals) >= n_components:
                    top_indices = np.argsort(fft_vals)[-n_components:]
                    mag = np.mean(fft_vals[top_indices])
                else:
                    mag = np.mean(fft_vals) if len(fft_vals) > 0 else 0
            else:
                mag = 0
                
            fft_mags.append(mag)
            
        features[f"{column}_fft_mag"] = fft_mags
        
        return pd.DataFrame(features)
        
    def _add_statistical_features(
        self,
        df: pd.DataFrame,
        column: str
    ) -> pd.DataFrame:
        """Add statistical features."""
        features = {}
        
        # Z-score
        mean = df[column].mean()
        std = df[column].std()
        if std > 0:
            features[f"{column}_zscore"] = (df[column] - mean) / std
        else:
            features[f"{column}_zscore"] = 0
            
        # Distance from rolling mean
        rolling_mean = df[column].rolling(window=self.window_size, min_periods=1).mean()
        features[f"{column}_dist_from_mean"] = df[column] - rolling_mean
        
        return pd.DataFrame(features)
        
    def fit(self, data: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            data: Training data DataFrame
            
        Returns:
            self
        """
        # Fit scalers for each column
        for column in data.columns:
            scaler = self._get_scaler()
            scaler.fit(data[[column]])
            self.scalers[column] = scaler
            
        self.fitted = True
        self.feature_columns = list(data.columns)
        
        return self
        
    def transform(
        self,
        data: pd.DataFrame,
        add_features: bool = True
    ) -> pd.DataFrame:
        """
        Transform data with normalization and feature engineering.
        
        Args:
            data: Input data DataFrame
            add_features: Whether to add engineered features
            
        Returns:
            Transformed DataFrame
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
            
        result = data.copy()
        
        # Normalize
        for column in data.columns:
            if column in self.scalers:
                result[column] = self.scalers[column].transform(data[[column]])
                
        # Add engineered features
        if add_features:
            for column in data.columns:
                # Rolling features
                if self.feature_config.get("rolling_mean", True) or \
                   self.feature_config.get("rolling_std", True):
                    rolling_feats = self._add_rolling_features(result, column)
                    result = pd.concat([result, rolling_feats], axis=1)
                    
                # Rate of change
                if self.feature_config.get("rate_of_change", True):
                    roc_feats = self._add_rate_of_change(result, column)
                    result = pd.concat([result, roc_feats], axis=1)
                    
                # Statistical features
                stat_feats = self._add_statistical_features(result, column)
                result = pd.concat([result, stat_feats], axis=1)
                
            # FFT features (computed separately due to cost)
            if self.feature_config.get("fft_features", False):
                print("Computing FFT features (this may take a moment)...")
                for column in data.columns:
                    fft_feats = self._add_fft_features(result, column)
                    result = pd.concat([result, fft_feats], axis=1)
                    
        return result
        
    def fit_transform(
        self,
        data: pd.DataFrame,
        add_features: bool = True
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            data: Input data DataFrame
            add_features: Whether to add engineered features
            
        Returns:
            Transformed DataFrame
        """
        self.fit(data)
        return self.transform(data, add_features)
        
    def create_sequences(
        self,
        data: pd.DataFrame,
        labels: pd.DataFrame,
        sequence_length: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/RNN models.
        
        Args:
            data: Feature DataFrame
            labels: Labels DataFrame
            sequence_length: Length of each sequence
            
        Returns:
            X: Sequences array (n_samples, sequence_length, n_features)
            y: Labels array (n_samples,)
        """
        X, y = [], []
        
        data_values = data.values
        label_values = labels['is_anomaly'].values if 'is_anomaly' in labels.columns \
                       else labels.values.flatten()
        
        for i in range(len(data_values) - sequence_length + 1):
            X.append(data_values[i:i + sequence_length])
            # Use label of last timestep in sequence
            y.append(label_values[i + sequence_length - 1])
            
        return np.array(X), np.array(y)


def preprocess_data(
    data: pd.DataFrame,
    labels: pd.DataFrame,
    normalization: str = "standard",
    add_features: bool = True,
    create_sequences: bool = False,
    sequence_length: int = 50,
    test_split: float = 0.2,
    random_seed: int = 42
) -> Dict:
    """
    Convenience function for complete preprocessing pipeline.
    
    Args:
        data: Raw telemetry data
        labels: Labels DataFrame
        normalization: Normalization method
        add_features: Whether to engineer features
        create_sequences: Whether to create sequences for LSTM
        sequence_length: Sequence length if creating sequences
        test_split: Test set fraction
        random_seed: Random seed
        
    Returns:
        Dictionary with preprocessed data splits
    """
    from sklearn.model_selection import train_test_split
    
    # Train/test split
    n = len(data)
    split_idx = int(n * (1 - test_split))
    
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    train_labels = labels.iloc[:split_idx]
    test_labels = labels.iloc[split_idx:]
    
    # Initialize and fit preprocessor
    preprocessor = DataPreprocessor(normalization=normalization)
    
    # Fit on training data
    preprocessor.fit(train_data)
    
    # Transform
    X_train = preprocessor.transform(train_data, add_features=add_features)
    X_test = preprocessor.transform(test_data, add_features=add_features)
    
    # Get labels
    y_train = train_labels['is_anomaly'].values if 'is_anomaly' in train_labels.columns \
              else train_labels.values.flatten()
    y_test = test_labels['is_anomaly'].values if 'is_anomaly' in test_labels.columns \
             else test_labels.values.flatten()
    
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'feature_names': list(X_train.columns),
    }
    
    # Create sequences if needed
    if create_sequences:
        X_train_seq, y_train_seq = preprocessor.create_sequences(
            X_train, train_labels, sequence_length
        )
        X_test_seq, y_test_seq = preprocessor.create_sequences(
            X_test, test_labels, sequence_length
        )
        
        result.update({
            'X_train_seq': X_train_seq,
            'X_test_seq': X_test_seq,
            'y_train_seq': y_train_seq,
            'y_test_seq': y_test_seq,
        })
        
    return result


if __name__ == "__main__":
    # Demo
    from synthetic_generator import generate_demo_dataset
    
    print("Generating demo data...")
    data, labels = generate_demo_dataset(n_timesteps=1000)
    
    print("\nPreprocessing...")
    result = preprocess_data(
        data, labels,
        add_features=True,
        create_sequences=True
    )
    
    print(f"\nOriginal features: {len(data.columns)}")
    print(f"Engineered features: {len(result['feature_names'])}")
    print(f"Training samples: {len(result['X_train'])}")
    print(f"Test samples: {len(result['X_test'])}")
    
    if 'X_train_seq' in result:
        print(f"\nSequence shape: {result['X_train_seq'].shape}")
