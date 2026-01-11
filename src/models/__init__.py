"""
Anomaly Detection Models
========================

Collection of anomaly detectors for spacecraft telemetry:

- Statistical Methods: Z-Score, IQR, MAD
- Machine Learning: Isolation Forest, One-Class SVM
- Deep Learning: LSTM Autoencoder
- Ensemble: Combined detector with voting
"""

from .base import BaseAnomalyDetector, DetectionResult
from .statistical import (
    ZScoreDetector,
    IQRDetector,
    MADDetector,
    RollingZScoreDetector
)
from .isolation_forest import IsolationForestDetector
from .one_class_svm import OneClassSVMDetector, SVDDAnomalyDetector
from .ensemble import EnsembleDetector, create_default_ensemble

# Optional: Deep Learning models (require torch)
try:
    from .autoencoder import LSTMAutoencoderDetector
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    LSTMAutoencoderDetector = None

__all__ = [
    # Base
    'BaseAnomalyDetector',
    'DetectionResult',
    
    # Statistical
    'ZScoreDetector',
    'IQRDetector',
    'MADDetector',
    'RollingZScoreDetector',
    
    # ML
    'IsolationForestDetector',
    'OneClassSVMDetector',
    'SVDDAnomalyDetector',
    
    # Ensemble
    'EnsembleDetector',
    'create_default_ensemble',
]

# Add deep learning models if available
if _HAS_TORCH:
    __all__.append('LSTMAutoencoderDetector')
