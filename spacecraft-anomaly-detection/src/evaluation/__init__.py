"""
Evaluation Module
=================

Metrics and evaluation utilities for anomaly detection.
"""

from .metrics import (
    compute_all_metrics,
    compute_confusion_matrix,
    compute_roc_data,
    compute_pr_data,
    find_optimal_threshold,
    evaluate_detector,
    compare_detectors,
    generate_report
)

__all__ = [
    'compute_all_metrics',
    'compute_confusion_matrix',
    'compute_roc_data',
    'compute_pr_data',
    'find_optimal_threshold',
    'evaluate_detector',
    'compare_detectors',
    'generate_report',
]
