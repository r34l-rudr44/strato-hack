"""
Evaluation Metrics Module
=========================

Comprehensive metrics for evaluating anomaly detection performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: Ground truth labels (0 = normal, 1 = anomaly)
        y_pred: Predicted labels
        y_scores: Optional anomaly scores for AUC metrics
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Specificity (true negative rate)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
    else:
        metrics['specificity'] = 0
        
    # AUC metrics if scores provided
    if y_scores is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics['roc_auc'] = 0.5
            
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_scores)
        except ValueError:
            metrics['pr_auc'] = 0.0
            
    # Additional metrics
    metrics['n_samples'] = len(y_true)
    metrics['n_anomalies_true'] = int(y_true.sum())
    metrics['n_anomalies_pred'] = int(y_pred.sum())
    metrics['anomaly_ratio_true'] = float(y_true.mean())
    metrics['anomaly_ratio_pred'] = float(y_pred.mean())
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Compute confusion matrix as DataFrame.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix DataFrame
    """
    cm = confusion_matrix(y_true, y_pred)
    
    return pd.DataFrame(
        cm,
        index=['Actual Normal', 'Actual Anomaly'],
        columns=['Predicted Normal', 'Predicted Anomaly']
    )


def compute_roc_data(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve data.
    
    Args:
        y_true: Ground truth labels
        y_scores: Anomaly scores
        
    Returns:
        fpr: False positive rates
        tpr: True positive rates
        thresholds: Threshold values
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    return fpr, tpr, thresholds


def compute_pr_data(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Precision-Recall curve data.
    
    Args:
        y_true: Ground truth labels
        y_scores: Anomaly scores
        
    Returns:
        precision: Precision values
        recall: Recall values
        thresholds: Threshold values
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    return precision, recall, thresholds


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = 'f1_score'
) -> Tuple[float, float]:
    """
    Find optimal threshold based on specified metric.
    
    Args:
        y_true: Ground truth labels
        y_scores: Anomaly scores
        metric: Metric to optimize ('f1_score', 'precision', 'recall', 'youden')
        
    Returns:
        optimal_threshold: Best threshold value
        best_metric_value: Metric value at optimal threshold
    """
    thresholds = np.percentile(y_scores, np.arange(0, 100, 1))
    best_threshold = 0
    best_value = 0
    
    for thresh in thresholds:
        y_pred = (y_scores > thresh).astype(int)
        
        if metric == 'f1_score':
            value = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            value = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            value = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'youden':
            # Youden's J statistic: sensitivity + specificity - 1
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                value = sensitivity + specificity - 1
            else:
                value = 0
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        if value > best_value:
            best_value = value
            best_threshold = thresh
            
    return best_threshold, best_value


def evaluate_detector(
    detector,
    X_test: np.ndarray,
    y_test: np.ndarray,
    optimize_threshold: bool = False,
    threshold_metric: str = 'f1_score'
) -> Dict[str, any]:
    """
    Comprehensive evaluation of an anomaly detector.
    
    Args:
        detector: Fitted anomaly detector
        X_test: Test features
        y_test: Test labels
        optimize_threshold: Whether to optimize threshold
        threshold_metric: Metric for threshold optimization
        
    Returns:
        Dictionary with metrics, curves, and predictions
    """
    # Get scores
    scores = detector.score_samples(X_test)
    
    # Optimize threshold if requested
    if optimize_threshold:
        threshold, _ = find_optimal_threshold(y_test, scores, threshold_metric)
        predictions = (scores > threshold).astype(int)
    else:
        predictions = detector.predict(X_test)
        threshold = detector.threshold
        
    # Compute metrics
    metrics = compute_all_metrics(y_test, predictions, scores)
    metrics['threshold'] = threshold
    
    # Compute curves
    fpr, tpr, roc_thresholds = compute_roc_data(y_test, scores)
    precision, recall, pr_thresholds = compute_pr_data(y_test, scores)
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'scores': scores,
        'confusion_matrix': compute_confusion_matrix(y_test, predictions),
        'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds},
        'pr_curve': {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds},
    }


def compare_detectors(
    detectors: List,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Compare multiple detectors on the same test set.
    
    Args:
        detectors: List of fitted detectors
        X_test: Test features
        y_test: Test labels
        
    Returns:
        DataFrame comparing detector performance
    """
    results = []
    
    for detector in detectors:
        eval_result = evaluate_detector(detector, X_test, y_test)
        metrics = eval_result['metrics']
        metrics['detector'] = detector.name
        results.append(metrics)
        
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['detector', 'accuracy', 'precision', 'recall', 'f1_score', 
            'specificity', 'roc_auc', 'pr_auc']
    cols = [c for c in cols if c in df.columns]
    
    return df[cols].sort_values('f1_score', ascending=False)


def generate_report(
    results: Dict,
    detector_name: str = "Detector"
) -> str:
    """
    Generate a text report of evaluation results.
    
    Args:
        results: Results from evaluate_detector
        detector_name: Name of the detector
        
    Returns:
        Formatted report string
    """
    metrics = results['metrics']
    cm = results['confusion_matrix']
    
    report = f"""
{'='*60}
ANOMALY DETECTION EVALUATION REPORT
{detector_name}
{'='*60}

SUMMARY METRICS
---------------
Accuracy:    {metrics['accuracy']:.4f}
Precision:   {metrics['precision']:.4f}
Recall:      {metrics['recall']:.4f}
F1 Score:    {metrics['f1_score']:.4f}
Specificity: {metrics['specificity']:.4f}
"""

    if 'roc_auc' in metrics:
        report += f"ROC AUC:     {metrics['roc_auc']:.4f}\n"
    if 'pr_auc' in metrics:
        report += f"PR AUC:      {metrics['pr_auc']:.4f}\n"
        
    report += f"""
CONFUSION MATRIX
----------------
{cm.to_string()}

DATA STATISTICS
---------------
Total Samples:       {metrics['n_samples']}
True Anomalies:      {metrics['n_anomalies_true']} ({metrics['anomaly_ratio_true']:.2%})
Predicted Anomalies: {metrics['n_anomalies_pred']} ({metrics['anomaly_ratio_pred']:.2%})

DETAILED COUNTS
---------------
True Positives:  {metrics.get('true_positives', 'N/A')}
False Positives: {metrics.get('false_positives', 'N/A')}
True Negatives:  {metrics.get('true_negatives', 'N/A')}
False Negatives: {metrics.get('false_negatives', 'N/A')}

{'='*60}
"""
    
    return report


if __name__ == "__main__":
    # Demo
    np.random.seed(42)
    
    # Generate dummy data
    n = 1000
    y_true = np.random.binomial(1, 0.05, n)
    y_scores = np.random.normal(0, 1, n)
    y_scores[y_true == 1] += 2  # Anomalies have higher scores
    y_pred = (y_scores > 1).astype(int)
    
    print("Computing metrics...")
    metrics = compute_all_metrics(y_true, y_pred, y_scores)
    
    print("\nMetrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
            
    print("\nConfusion Matrix:")
    print(compute_confusion_matrix(y_true, y_pred))
    
    print("\nFinding optimal threshold...")
    opt_thresh, opt_f1 = find_optimal_threshold(y_true, y_scores, 'f1_score')
    print(f"Optimal threshold: {opt_thresh:.4f}")
    print(f"Optimal F1 score: {opt_f1:.4f}")
