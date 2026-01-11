"""
One-Class SVM Anomaly Detector
==============================

One-Class SVM learns a decision boundary that encompasses "normal" data.
Points outside this boundary are classified as anomalies.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from .base import BaseAnomalyDetector


class OneClassSVMDetector(BaseAnomalyDetector):
    """
    One-Class SVM based anomaly detector.
    
    Best for medium-sized datasets where a clear boundary
    separates normal from anomalous behavior.
    """
    
    def __init__(
        self,
        kernel: str = "rbf",
        gamma: Union[str, float] = "scale",
        nu: float = 0.05,
        degree: int = 3,
        coef0: float = 0.0,
        shrinking: bool = True,
        cache_size: int = 200,
        max_iter: int = -1,
        name: str = "OneClassSVM"
    ):
        """
        Initialize One-Class SVM detector.
        
        Args:
            kernel: Kernel type ("rbf", "linear", "poly", "sigmoid")
            gamma: Kernel coefficient ("scale", "auto", or float)
            nu: Upper bound on fraction of training errors and lower bound
                on fraction of support vectors. Should be in (0, 1].
                Corresponds to expected contamination rate.
            degree: Degree for polynomial kernel
            coef0: Independent term in kernel function
            shrinking: Whether to use shrinking heuristic
            cache_size: Kernel cache size in MB
            max_iter: Hard limit on iterations (-1 = no limit)
            name: Detector name
        """
        super().__init__(name=name)
        
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.degree = degree
        self.coef0 = coef0
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.max_iter = max_iter
        
        self.model: Optional[OneClassSVM] = None
        self.scaler: Optional[StandardScaler] = None
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None
    ) -> 'OneClassSVMDetector':
        """
        Fit the One-Class SVM on training data.
        
        Note: One-Class SVM is unsupervised, y is ignored.
        Data is internally scaled for better performance.
        """
        X = np.array(X)
        
        # Scale data for SVM (important for RBF kernel)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = OneClassSVM(
            kernel=self.kernel,
            gamma=self.gamma,
            nu=self.nu,
            degree=self.degree,
            coef0=self.coef0,
            shrinking=self.shrinking,
            cache_size=self.cache_size,
            max_iter=self.max_iter
        )
        
        self.model.fit(X_scaled)
        
        # Set threshold based on decision function
        scores = -self.model.decision_function(X_scaled)
        self.threshold = np.percentile(scores, 100 * (1 - self.nu))
        
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
        X_scaled = self.scaler.transform(X)
        
        # Negate decision function so higher = more anomalous
        scores = -self.model.decision_function(X_scaled)
        
        return scores
        
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict anomaly labels.
        """
        if threshold is None and self.threshold is None:
            # Use sklearn's built-in prediction
            X = np.array(X)
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            # sklearn uses: 1 = normal, -1 = anomaly
            return (predictions == -1).astype(int)
        else:
            # Use threshold-based prediction
            scores = self.score_samples(X)
            thresh = threshold if threshold is not None else self.threshold
            return (scores > thresh).astype(int)
            
    def get_support_vectors(self) -> np.ndarray:
        """
        Get support vectors (in original scale).
        
        Support vectors define the decision boundary.
        """
        if not self.fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
            
        # Inverse transform to original scale
        return self.scaler.inverse_transform(self.model.support_vectors_)
        
    def get_params(self) -> Dict[str, Any]:
        """Get detector parameters."""
        return {
            'kernel': self.kernel,
            'gamma': self.gamma,
            'nu': self.nu,
            'degree': self.degree,
            'coef0': self.coef0,
            'shrinking': self.shrinking,
        }


class SVDDAnomalyDetector(BaseAnomalyDetector):
    """
    Support Vector Data Description (SVDD) inspired detector.
    
    Similar to One-Class SVM but explicitly focuses on finding
    the minimum enclosing hypersphere around normal data.
    
    This is a simplified implementation using sklearn's OneClassSVM
    with tuned parameters for SVDD-like behavior.
    """
    
    def __init__(
        self,
        nu: float = 0.05,
        gamma: Union[str, float] = "scale",
        name: str = "SVDD"
    ):
        """
        Initialize SVDD detector.
        
        Args:
            nu: Expected anomaly fraction
            gamma: RBF kernel width
            name: Detector name
        """
        super().__init__(name=name)
        self.nu = nu
        self.gamma = gamma
        
        # Use One-Class SVM with RBF kernel as SVDD approximation
        self._detector = OneClassSVMDetector(
            kernel="rbf",
            gamma=gamma,
            nu=nu,
            name=name
        )
        
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None
    ) -> 'SVDDAnomalyDetector':
        """Fit the SVDD detector."""
        self._detector.fit(X, y)
        self.fitted = True
        self.threshold = self._detector.threshold
        return self
        
    def score_samples(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """Compute anomaly scores."""
        return self._detector.score_samples(X)
        
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """Predict anomaly labels."""
        return self._detector.predict(X, threshold)


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
    
    print("\nTraining One-Class SVM...")
    detector = OneClassSVMDetector(
        kernel="rbf",
        gamma="scale",
        nu=0.05
    )
    
    detector.fit(X_train)
    predictions = detector.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=['Normal', 'Anomaly']))
    
    print(f"\nNumber of support vectors: {len(detector.get_support_vectors())}")
