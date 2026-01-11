"""
Statistical Anomaly Detectors
=============================

Simple statistical methods for anomaly detection:
- Z-Score based detection
- IQR (Interquartile Range) based detection
- Modified Z-Score (MAD-based)
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
from scipy import stats

from .base import BaseAnomalyDetector


class ZScoreDetector(BaseAnomalyDetector):
    """
    Z-Score based anomaly detector.
    
    Flags points that deviate more than `threshold` standard deviations
    from the mean as anomalies.
    """
    
    def __init__(self, threshold: float = 3.0, name: str = "ZScore"):
        """
        Initialize Z-Score detector.
        
        Args:
            threshold: Number of standard deviations for anomaly threshold
            name: Detector name
        """
        super().__init__(name=name)
        self.z_threshold = threshold
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None
    ) -> 'ZScoreDetector':
        """Fit by computing mean and std of training data."""
        X = np.array(X)
        
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        # Avoid division by zero
        self.std = np.where(self.std == 0, 1e-10, self.std)
        
        self.fitted = True
        self.threshold = self.z_threshold
        
        return self
        
    def score_samples(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Compute anomaly scores as maximum absolute Z-score across features.
        """
        if not self.fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
            
        X = np.array(X)
        
        # Compute Z-scores for each feature
        z_scores = np.abs((X - self.mean) / self.std)
        
        # Return maximum Z-score across features
        return np.max(z_scores, axis=1)
        
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """Predict using Z-score threshold."""
        scores = self.score_samples(X)
        thresh = threshold if threshold is not None else self.z_threshold
        return (scores > thresh).astype(int)


class IQRDetector(BaseAnomalyDetector):
    """
    Interquartile Range (IQR) based anomaly detector.
    
    Flags points outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR] as anomalies.
    """
    
    def __init__(self, multiplier: float = 1.5, name: str = "IQR"):
        """
        Initialize IQR detector.
        
        Args:
            multiplier: IQR multiplier for bounds
            name: Detector name
        """
        super().__init__(name=name)
        self.multiplier = multiplier
        self.q1: Optional[np.ndarray] = None
        self.q3: Optional[np.ndarray] = None
        self.iqr: Optional[np.ndarray] = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None
    ) -> 'IQRDetector':
        """Fit by computing quartiles of training data."""
        X = np.array(X)
        
        self.q1 = np.percentile(X, 25, axis=0)
        self.q3 = np.percentile(X, 75, axis=0)
        self.iqr = self.q3 - self.q1
        
        # Avoid division by zero
        self.iqr = np.where(self.iqr == 0, 1e-10, self.iqr)
        
        self.lower_bound = self.q1 - self.multiplier * self.iqr
        self.upper_bound = self.q3 + self.multiplier * self.iqr
        
        self.fitted = True
        
        return self
        
    def score_samples(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Compute anomaly scores based on distance from IQR bounds.
        """
        if not self.fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
            
        X = np.array(X)
        
        # Compute distance from bounds (normalized by IQR)
        below = np.maximum(0, self.lower_bound - X) / self.iqr
        above = np.maximum(0, X - self.upper_bound) / self.iqr
        
        # Combined score as max deviation across features
        scores = np.max(below + above, axis=1)
        
        return scores
        
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """Predict using IQR bounds."""
        X = np.array(X)
        
        # Point is anomaly if outside bounds for any feature
        below = np.any(X < self.lower_bound, axis=1)
        above = np.any(X > self.upper_bound, axis=1)
        
        return (below | above).astype(int)


class MADDetector(BaseAnomalyDetector):
    """
    Median Absolute Deviation (MAD) based anomaly detector.
    
    More robust to outliers than standard Z-score.
    Uses median and MAD instead of mean and std.
    """
    
    def __init__(self, threshold: float = 3.5, name: str = "MAD"):
        """
        Initialize MAD detector.
        
        Args:
            threshold: Modified Z-score threshold
            name: Detector name
        """
        super().__init__(name=name)
        self.mad_threshold = threshold
        self.median: Optional[np.ndarray] = None
        self.mad: Optional[np.ndarray] = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None
    ) -> 'MADDetector':
        """Fit by computing median and MAD of training data."""
        X = np.array(X)
        
        self.median = np.median(X, axis=0)
        self.mad = stats.median_abs_deviation(X, axis=0)
        
        # Avoid division by zero
        self.mad = np.where(self.mad == 0, 1e-10, self.mad)
        
        self.fitted = True
        self.threshold = self.mad_threshold
        
        return self
        
    def score_samples(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Compute modified Z-scores using MAD.
        """
        if not self.fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
            
        X = np.array(X)
        
        # Modified Z-score: 0.6745 is scaling factor for normal distribution
        modified_z = 0.6745 * np.abs(X - self.median) / self.mad
        
        # Return maximum across features
        return np.max(modified_z, axis=1)
        
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """Predict using modified Z-score threshold."""
        scores = self.score_samples(X)
        thresh = threshold if threshold is not None else self.mad_threshold
        return (scores > thresh).astype(int)


class RollingZScoreDetector(BaseAnomalyDetector):
    """
    Rolling Z-Score detector for time series data.
    
    Computes Z-scores relative to a rolling window of recent history.
    """
    
    def __init__(
        self,
        window_size: int = 50,
        threshold: float = 3.0,
        name: str = "RollingZScore"
    ):
        """
        Initialize Rolling Z-Score detector.
        
        Args:
            window_size: Size of rolling window
            threshold: Z-score threshold
            name: Detector name
        """
        super().__init__(name=name)
        self.window_size = window_size
        self.z_threshold = threshold
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None
    ) -> 'RollingZScoreDetector':
        """Fit (mostly for interface consistency)."""
        self.fitted = True
        self.threshold = self.z_threshold
        return self
        
    def score_samples(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Compute rolling Z-scores.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
            
        scores = np.zeros(len(X))
        
        for col in X.columns:
            # Rolling statistics
            rolling_mean = X[col].rolling(
                window=self.window_size, 
                min_periods=1
            ).mean()
            rolling_std = X[col].rolling(
                window=self.window_size, 
                min_periods=1
            ).std().fillna(1e-10)
            
            # Z-score relative to rolling window
            col_z = np.abs((X[col] - rolling_mean) / rolling_std.replace(0, 1e-10))
            scores = np.maximum(scores, col_z.values)
            
        return scores


if __name__ == "__main__":
    # Demo
    import sys
    sys.path.append('..')
    from data.synthetic_generator import generate_demo_dataset
    
    print("Generating demo data...")
    data, labels = generate_demo_dataset(n_timesteps=1000, anomaly_ratio=0.05)
    
    y_true = labels['is_anomaly'].values
    
    print("\nTesting statistical detectors:")
    
    detectors = [
        ZScoreDetector(threshold=3.0),
        IQRDetector(multiplier=1.5),
        MADDetector(threshold=3.5),
        RollingZScoreDetector(window_size=50, threshold=3.0),
    ]
    
    for detector in detectors:
        detector.fit(data)
        predictions = detector.predict(data)
        
        accuracy = np.mean(predictions == y_true)
        print(f"{detector.name}: Accuracy = {accuracy:.3f}")
