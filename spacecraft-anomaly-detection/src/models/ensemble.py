"""
Ensemble Anomaly Detector
=========================

Combines multiple anomaly detectors for more robust detection.
Supports various voting strategies and learned weights.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any, List
from sklearn.preprocessing import MinMaxScaler

from .base import BaseAnomalyDetector, DetectionResult


class EnsembleDetector(BaseAnomalyDetector):
    """
    Ensemble anomaly detector combining multiple base detectors.
    
    Strategies:
    - soft: Average normalized anomaly scores
    - hard: Majority voting on predictions
    - weighted: Weighted average based on validation performance
    """
    
    def __init__(
        self,
        detectors: Optional[List[BaseAnomalyDetector]] = None,
        voting: str = "soft",
        weights: Optional[List[float]] = None,
        threshold_percentile: float = 95,
        name: str = "Ensemble"
    ):
        """
        Initialize ensemble detector.
        
        Args:
            detectors: List of base detectors
            voting: Voting strategy ("soft", "hard", "weighted")
            weights: Optional weights for each detector
            threshold_percentile: Percentile for threshold (soft voting)
            name: Detector name
        """
        super().__init__(name=name)
        
        self.detectors = detectors or []
        self.voting = voting
        self.weights = weights
        self.threshold_percentile = threshold_percentile
        
        # Score normalizers
        self.scalers: Dict[str, MinMaxScaler] = {}
        
    def add_detector(self, detector: BaseAnomalyDetector) -> 'EnsembleDetector':
        """Add a detector to the ensemble."""
        self.detectors.append(detector)
        return self
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None
    ) -> 'EnsembleDetector':
        """
        Fit all detectors in the ensemble.
        
        Args:
            X: Training features
            y: Optional labels (for weight learning)
        """
        X_array = np.array(X)
        
        print(f"Fitting ensemble with {len(self.detectors)} detectors...")
        
        for detector in self.detectors:
            print(f"  Training {detector.name}...")
            detector.fit(X, y)
            
            # Fit normalizer for this detector's scores
            scores = detector.score_samples(X)
            self.scalers[detector.name] = MinMaxScaler()
            self.scalers[detector.name].fit(scores.reshape(-1, 1))
            
        # Learn weights if y is provided and weights not specified
        if y is not None and self.weights is None and self.voting == "weighted":
            self._learn_weights(X, y)
        elif self.weights is None:
            # Equal weights
            self.weights = [1.0 / len(self.detectors)] * len(self.detectors)
            
        # Set ensemble threshold
        if self.voting == "soft" or self.voting == "weighted":
            combined_scores = self._combine_scores(X)
            self.threshold = np.percentile(combined_scores, self.threshold_percentile)
            
        self.fitted = True
        print("Ensemble training complete.")
        
        return self
        
    def _learn_weights(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray
    ) -> None:
        """
        Learn optimal weights based on individual detector performance.
        """
        from sklearn.metrics import f1_score
        
        performances = []
        
        for detector in self.detectors:
            predictions = detector.predict(X)
            f1 = f1_score(y, predictions, zero_division=0)
            performances.append(f1)
            
        # Normalize to sum to 1
        total = sum(performances) + 1e-10
        self.weights = [p / total for p in performances]
        
        print("Learned weights:")
        for detector, weight in zip(self.detectors, self.weights):
            print(f"  {detector.name}: {weight:.3f}")
            
    def _normalize_scores(
        self,
        scores: np.ndarray,
        detector_name: str
    ) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        if detector_name in self.scalers:
            return self.scalers[detector_name].transform(
                scores.reshape(-1, 1)
            ).flatten()
        else:
            # Fallback to simple min-max
            scores_min = scores.min()
            scores_max = scores.max()
            if scores_max - scores_min > 0:
                return (scores - scores_min) / (scores_max - scores_min)
            else:
                return np.zeros_like(scores)
                
    def _combine_scores(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Combine scores from all detectors.
        """
        all_scores = []
        
        for detector in self.detectors:
            scores = detector.score_samples(X)
            normalized = self._normalize_scores(scores, detector.name)
            all_scores.append(normalized)
            
        all_scores = np.array(all_scores)  # (n_detectors, n_samples)
        
        if self.voting == "weighted":
            weights = np.array(self.weights).reshape(-1, 1)
            combined = np.sum(all_scores * weights, axis=0)
        else:  # soft voting - simple average
            combined = np.mean(all_scores, axis=0)
            
        return combined
        
    def score_samples(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Compute combined anomaly scores.
        """
        if not self.fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
            
        return self._combine_scores(X)
        
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict anomaly labels.
        """
        if self.voting == "hard":
            # Majority voting
            all_predictions = []
            for detector in self.detectors:
                predictions = detector.predict(X)
                all_predictions.append(predictions)
                
            all_predictions = np.array(all_predictions)
            
            # Weighted voting if weights are provided
            if self.weights is not None:
                weights = np.array(self.weights).reshape(-1, 1)
                weighted_votes = np.sum(all_predictions * weights, axis=0)
                return (weighted_votes > 0.5).astype(int)
            else:
                # Simple majority
                return (np.mean(all_predictions, axis=0) > 0.5).astype(int)
        else:
            # Threshold-based on combined scores
            scores = self.score_samples(X)
            thresh = threshold if threshold is not None else self.threshold
            return (scores > thresh).astype(int)
            
    def detect(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> DetectionResult:
        """
        Full detection with individual detector results.
        """
        scores = self.score_samples(X)
        predictions = self.predict(X, threshold)
        
        # Get individual detector results
        individual_results = {}
        for detector in self.detectors:
            individual_results[detector.name] = {
                'scores': detector.score_samples(X),
                'predictions': detector.predict(X)
            }
            
        return DetectionResult(
            predictions=predictions,
            scores=scores,
            threshold=threshold if threshold is not None else self.threshold,
            metadata={
                "detector": self.name,
                "n_samples": len(X),
                "n_anomalies": int(predictions.sum()),
                "anomaly_ratio": float(predictions.mean()),
                "voting": self.voting,
                "weights": self.weights,
                "individual_results": individual_results,
            }
        )
        
    def get_detector_agreement(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Get agreement level among detectors for each sample.
        
        Returns fraction of detectors that agree on anomaly status.
        """
        all_predictions = []
        for detector in self.detectors:
            predictions = detector.predict(X)
            all_predictions.append(predictions)
            
        all_predictions = np.array(all_predictions)
        
        # Agreement = max of (fraction predicting 0, fraction predicting 1)
        anomaly_votes = np.mean(all_predictions, axis=0)
        agreement = np.maximum(anomaly_votes, 1 - anomaly_votes)
        
        return agreement
        
    def get_detector_contributions(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> pd.DataFrame:
        """
        Get contribution of each detector to the final score.
        """
        results = {'sample_idx': range(len(X))}
        
        for i, detector in enumerate(self.detectors):
            scores = detector.score_samples(X)
            normalized = self._normalize_scores(scores, detector.name)
            
            if self.weights is not None:
                contribution = normalized * self.weights[i]
            else:
                contribution = normalized / len(self.detectors)
                
            results[detector.name] = contribution
            
        return pd.DataFrame(results)


def create_default_ensemble(
    contamination: float = 0.05,
    include_autoencoder: bool = True
) -> EnsembleDetector:
    """
    Create an ensemble with default detectors.
    
    Args:
        contamination: Expected anomaly fraction
        include_autoencoder: Whether to include LSTM autoencoder
        
    Returns:
        Configured ensemble detector
    """
    from .statistical import ZScoreDetector, MADDetector
    from .isolation_forest import IsolationForestDetector
    from .one_class_svm import OneClassSVMDetector
    from .autoencoder import LSTMAutoencoderDetector
    
    detectors = [
        ZScoreDetector(threshold=3.0),
        MADDetector(threshold=3.5),
        IsolationForestDetector(
            n_estimators=100,
            contamination=contamination
        ),
        OneClassSVMDetector(
            nu=contamination,
            kernel="rbf"
        ),
    ]
    
    if include_autoencoder:
        detectors.append(
            LSTMAutoencoderDetector(
                hidden_dim=64,
                latent_dim=16,
                seq_len=50,
                epochs=30
            )
        )
        
    ensemble = EnsembleDetector(
        detectors=detectors,
        voting="weighted",
        name="DefaultEnsemble"
    )
    
    return ensemble


if __name__ == "__main__":
    # Demo
    import sys
    sys.path.append('..')
    from data.synthetic_generator import generate_demo_dataset
    from sklearn.metrics import classification_report
    
    print("Generating demo data...")
    data, labels = generate_demo_dataset(n_timesteps=2000, anomaly_ratio=0.05)
    
    y_true = labels['is_anomaly'].values
    
    # Split data
    split = int(len(data) * 0.8)
    X_train, X_test = data.iloc[:split], data.iloc[split:]
    y_train, y_test = y_true[:split], y_true[split:]
    
    print("\nCreating ensemble detector...")
    ensemble = create_default_ensemble(
        contamination=0.05,
        include_autoencoder=False  # Skip for faster demo
    )
    
    print("\nTraining ensemble...")
    ensemble.fit(X_train, y_train)
    
    print("\nEvaluating...")
    predictions = ensemble.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Normal', 'Anomaly']))
    
    print("\nDetector Agreement Stats:")
    agreement = ensemble.get_detector_agreement(X_test)
    print(f"Mean agreement: {agreement.mean():.3f}")
    print(f"Min agreement: {agreement.min():.3f}")
