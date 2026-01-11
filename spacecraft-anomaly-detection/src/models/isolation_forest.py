"""
Isolation Forest Anomaly Detector
=================================

Isolation Forest is an unsupervised anomaly detection algorithm that
isolates observations by randomly selecting features and split values.

Key insight: Anomalies are easier to isolate, requiring fewer splits.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any
from sklearn.ensemble import IsolationForest as SKLearnIF

from .base import BaseAnomalyDetector


class IsolationForestDetector(BaseAnomalyDetector):
    """
    Isolation Forest based anomaly detector.
    
    Excellent for detecting anomalies in high-dimensional data.
    Works well with multivariate spacecraft telemetry.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[str, int] = "auto",
        contamination: float = 0.05,
        max_features: float = 1.0,
        bootstrap: bool = False,
        random_state: int = 42,
        n_jobs: int = -1,
        name: str = "IsolationForest"
    ):
        """
        Initialize Isolation Forest detector.
        
        Args:
            n_estimators: Number of base estimators (trees)
            max_samples: Number of samples per tree ("auto" = min(256, n_samples))
            contamination: Expected proportion of anomalies
            max_features: Number of features per tree
            bootstrap: Whether to use bootstrap sampling
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 = all cores)
            name: Detector name
        """
        super().__init__(name=name)
        
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model: Optional[SKLearnIF] = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None
    ) -> 'IsolationForestDetector':
        """
        Fit the Isolation Forest on training data.
        
        Note: Isolation Forest is unsupervised, y is ignored.
        """
        X = np.array(X)
        
        self.model = SKLearnIF(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            warm_start=False
        )
        
        self.model.fit(X)
        
        # Set threshold based on decision function
        # decision_function returns: negative = anomaly, positive = normal
        scores = -self.model.decision_function(X)
        self.threshold = np.percentile(scores, 100 * (1 - self.contamination))
        
        self.fitted = True
        return self
        
    def score_samples(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """
        Compute anomaly scores.
        
        Higher scores indicate more anomalous samples.
        """
        if not self.fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
            
        X = np.array(X)
        
        # Negate decision function so higher = more anomalous
        scores = -self.model.decision_function(X)
        
        return scores
        
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Can use sklearn's built-in predict or threshold-based.
        """
        if threshold is None and self.threshold is None:
            # Use sklearn's built-in prediction
            X = np.array(X)
            predictions = self.model.predict(X)
            # sklearn uses: 1 = normal, -1 = anomaly
            return (predictions == -1).astype(int)
        else:
            # Use threshold-based prediction
            scores = self.score_samples(X)
            thresh = threshold if threshold is not None else self.threshold
            return (scores > thresh).astype(int)
            
    def get_feature_importance(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        feature_names: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Estimate feature importance based on average path lengths.
        
        Note: This is an approximation. For more accurate importance,
        consider using permutation importance.
        """
        if not self.fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
            
        X = np.array(X)
        n_features = X.shape[1]
        
        # Simple approximation based on feature variance contribution
        # More sophisticated methods exist but this is fast
        feature_scores = np.zeros(n_features)
        
        # Get base anomaly scores
        base_scores = self.score_samples(X)
        
        # Measure impact of each feature
        for i in range(n_features):
            # Create copy with shuffled feature
            X_shuffled = X.copy()
            np.random.shuffle(X_shuffled[:, i])
            shuffled_scores = self.score_samples(X_shuffled)
            
            # Importance = change in scores when feature is shuffled
            feature_scores[i] = np.mean(np.abs(base_scores - shuffled_scores))
            
        # Normalize
        feature_scores = feature_scores / (feature_scores.sum() + 1e-10)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
            
        return pd.DataFrame({
            'feature': feature_names,
            'importance': feature_scores
        }).sort_values('importance', ascending=False)
        
    def get_params(self) -> Dict[str, Any]:
        """Get detector parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'contamination': self.contamination,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'random_state': self.random_state,
        }


if __name__ == "__main__":
    # Demo
    import sys
    sys.path.append('..')
    from data.synthetic_generator import generate_demo_dataset
    from sklearn.metrics import classification_report
    
    print("Generating demo data...")
    data, labels = generate_demo_dataset(n_timesteps=2000, anomaly_ratio=0.05)
    
    y_true = labels['is_anomaly'].values
    
    print("\nTraining Isolation Forest...")
    detector = IsolationForestDetector(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    
    # Split data
    split = int(len(data) * 0.8)
    X_train, X_test = data.iloc[:split], data.iloc[split:]
    y_train, y_test = y_true[:split], y_true[split:]
    
    detector.fit(X_train)
    predictions = detector.predict(X_test)
    scores = detector.score_samples(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Normal', 'Anomaly']))
    
    print("\nFeature Importance:")
    importance = detector.get_feature_importance(X_test, list(data.columns))
    print(importance.head(10))
