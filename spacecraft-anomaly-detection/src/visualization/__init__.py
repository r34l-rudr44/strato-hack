"""
Visualization Module
====================

Comprehensive visualizations for spacecraft anomaly detection.
"""

from .plots import (
    plot_telemetry_with_anomalies,
    plot_anomaly_scores,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_score_distribution,
    plot_model_comparison,
    plot_feature_importance,
    create_evaluation_dashboard
)

__all__ = [
    'plot_telemetry_with_anomalies',
    'plot_anomaly_scores',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_confusion_matrix',
    'plot_score_distribution',
    'plot_model_comparison',
    'plot_feature_importance',
    'create_evaluation_dashboard'
]
