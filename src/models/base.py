"""
Base Anomaly Detector
=====================

Abstract base class for all anomaly detectors.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
import pickle
from pathlib import Path


@dataclass
class DetectionResult:
    """Container for anomaly detection results."""
    predictions: np.ndarray  # Binary predictions (0 = normal, 1 = anomaly)
    scores: np.ndarray       # Continuous anomaly scores
    threshold: float         # Threshold used for classification
    metadata: Dict[str, Any] # Additional info


class BaseAnomalyDetector(ABC):
    """
    Abstract base class for anomaly detectors.
    
    All detectors should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, name: str = "BaseDetector"):
        """
        Initialize the detector.
        
        Args:
            name: Name of the detector
        """
        self.name = name
        self.fitted = False
        self.threshold: Optional[float] = None
        self.training_scores: Optional[np.ndarray] = None
        
    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None
    ) -> 'BaseAnomalyDetector':
        """
        Fit the detector on training data.
        
        Args:
            X: Training features
            y: Optional labels (for semi-supervised methods)
            
        Returns:
            self
        """
        pass
        
    @abstractmethod
    def score_samples(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Compute anomaly scores for samples.
        
        Higher scores indicate more anomalous samples.
        
        Args:
            X: Input features
            
        Returns:
            Array of anomaly scores
        """
        pass
        
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Input features
            threshold: Custom threshold (uses fitted threshold if None)
            
        Returns:
            Binary predictions (0 = normal, 1 = anomaly)
        """
        scores = self.score_samples(X)
        
        thresh = threshold if threshold is not None else self.threshold
        if thresh is None:
            # Use default percentile-based threshold
            thresh = np.percentile(scores, 95)
            
        return (scores > thresh).astype(int)
        
    def fit_predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Fit and predict in one step.
        
        Args:
            X: Input features
            y: Optional labels
            threshold: Optional threshold
            
        Returns:
            Binary predictions
        """
        self.fit(X, y)
        return self.predict(X, threshold)
        
    def detect(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> DetectionResult:
        """
        Full detection with scores and predictions.
        
        Args:
            X: Input features
            threshold: Optional threshold
            
        Returns:
            DetectionResult with predictions, scores, and metadata
        """
        scores = self.score_samples(X)
        
        thresh = threshold if threshold is not None else self.threshold
        if thresh is None:
            thresh = np.percentile(scores, 95)
            
        predictions = (scores > thresh).astype(int)
        
        return DetectionResult(
            predictions=predictions,
            scores=scores,
            threshold=thresh,
            metadata={
                "detector": self.name,
                "n_samples": len(X),
                "n_anomalies": int(predictions.sum()),
                "anomaly_ratio": float(predictions.mean()),
                "score_min": float(scores.min()),
                "score_max": float(scores.max()),
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std()),
            }
        )
        
    def set_threshold(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        contamination: float = 0.05,
        method: str = "percentile"
    ) -> float:
        """
        Set detection threshold based on training data.
        
        Args:
            X: Training data
            contamination: Expected fraction of anomalies
            method: Method for threshold ("percentile", "std")
            
        Returns:
            Computed threshold
        """
        scores = self.score_samples(X)
        
        if method == "percentile":
            self.threshold = np.percentile(scores, 100 * (1 - contamination))
        elif method == "std":
            mean = np.mean(scores)
            std = np.std(scores)
            self.threshold = mean + 2 * std
        else:
            raise ValueError(f"Unknown threshold method: {method}")
            
        return self.threshold
        
    def save(self, path: str) -> None:
        """
        Save the detector to disk.
        
        Args:
            path: Save path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path: str) -> 'BaseAnomalyDetector':
        """
        Load a detector from disk.
        
        Args:
            path: Path to saved detector
            
        Returns:
            Loaded detector
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
            
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.fitted})"
