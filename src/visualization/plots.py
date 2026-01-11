"""
Visualization Module
====================

Comprehensive visualizations for spacecraft anomaly detection:
- Time series with anomaly highlighting
- Model performance curves (ROC, PR)
- Confusion matrices
- Feature importance
- Score distributions
- Model comparisons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path


# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def plot_telemetry_with_anomalies(
    data: pd.DataFrame,
    labels: pd.DataFrame,
    predictions: Optional[np.ndarray] = None,
    channels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    title: str = "Spacecraft Telemetry with Detected Anomalies",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot telemetry time series with anomalies highlighted.
    
    Args:
        data: Telemetry DataFrame with timestamp index
        labels: DataFrame with 'is_anomaly' column
        predictions: Optional model predictions
        channels: Specific channels to plot (default: first 4)
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    if channels is None:
        channels = list(data.columns)[:4]
        
    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    
    if n_channels == 1:
        axes = [axes]
        
    # Get anomaly indices
    y_true = labels['is_anomaly'].values if 'is_anomaly' in labels.columns \
             else labels.values.flatten()
    true_anomalies = np.where(y_true == 1)[0]
    
    if predictions is not None:
        pred_anomalies = np.where(predictions == 1)[0]
        
    for i, (ax, channel) in enumerate(zip(axes, channels)):
        # Plot signal
        ax.plot(data.index, data[channel], 'b-', linewidth=0.5, label='Normal', alpha=0.7)
        
        # Highlight true anomalies
        if len(true_anomalies) > 0:
            ax.scatter(
                data.index[true_anomalies],
                data[channel].iloc[true_anomalies],
                c='red', s=20, label='True Anomaly', zorder=5, alpha=0.7
            )
            
        # Highlight predictions
        if predictions is not None and len(pred_anomalies) > 0:
            ax.scatter(
                data.index[pred_anomalies],
                data[channel].iloc[pred_anomalies],
                c='orange', s=10, marker='x', label='Predicted', zorder=4
            )
            
        ax.set_ylabel(channel, fontsize=10)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
            
    axes[-1].set_xlabel('Time', fontsize=10)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_anomaly_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (12, 5),
    title: str = "Anomaly Scores Over Time",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot anomaly scores with threshold line.
    
    Args:
        scores: Anomaly scores array
        labels: True labels
        threshold: Detection threshold
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot scores
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    
    ax.plot(normal_idx, scores[normal_idx], 'b.', markersize=2, 
            label='Normal', alpha=0.5)
    ax.plot(anomaly_idx, scores[anomaly_idx], 'r.', markersize=5, 
            label='Anomaly', alpha=0.8)
    
    if threshold is not None:
        ax.axhline(y=threshold, color='g', linestyle='--', 
                   label=f'Threshold ({threshold:.3f})', linewidth=2)
        
    ax.set_xlabel('Sample Index', fontsize=10)
    ax.set_ylabel('Anomaly Score', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: Optional[float] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "ROC Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: Optional AUC score to display
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    label = 'ROC Curve'
    if auc_score is not None:
        label = f'ROC Curve (AUC = {auc_score:.3f})'
        
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    ap_score: Optional[float] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Precision-Recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        ap_score: Optional average precision score
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    label = 'PR Curve'
    if ap_score is not None:
        label = f'PR Curve (AP = {ap_score:.3f})'
        
    ax.plot(recall, precision, 'b-', linewidth=2, label=label)
    
    ax.set_xlabel('Recall', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_confusion_matrix(
    cm: Union[np.ndarray, pd.DataFrame],
    labels: List[str] = ['Normal', 'Anomaly'],
    figsize: Tuple[int, int] = (8, 6),
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix
        labels: Class labels
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(cm, pd.DataFrame):
        cm = cm.values
        
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar=True
    )
    
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Anomaly Score Distribution",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of anomaly scores for normal vs anomaly samples.
    
    Args:
        scores: Anomaly scores
        labels: True labels
        threshold: Optional threshold line
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red', density=True)
    
    if threshold is not None:
        ax.axvline(x=threshold, color='green', linestyle='--', 
                   label=f'Threshold ({threshold:.3f})', linewidth=2)
        
    ax.set_xlabel('Anomaly Score', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'f1_score',
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Model Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of multiple models.
    
    Args:
        comparison_df: DataFrame from compare_detectors
        metric: Metric to compare
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by metric
    df_sorted = comparison_df.sort_values(metric, ascending=True)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_sorted)))
    
    bars = ax.barh(df_sorted['detector'], df_sorted[metric], color=colors)
    
    # Add value labels
    for bar, val in zip(bars, df_sorted[metric]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
        
    ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=10)
    ax.set_ylabel('Detector', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim([0, 1.1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_multi_metric_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Multi-Metric Model Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple metrics for all models.
    
    Args:
        comparison_df: DataFrame from compare_detectors
        metrics: Metrics to plot
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(comparison_df))
    width = 0.8 / len(metrics)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
    
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, comparison_df[metric], width, 
                      label=metric.replace('_', ' ').title(),
                      color=colors[i], alpha=0.8)
        
    ax.set_xlabel('Detector', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['detector'], rotation=45, ha='right')
    ax.legend(loc='upper right', ncol=2)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 10,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Feature Importance",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance bar chart.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    df_top = importance_df.head(top_n).sort_values('importance', ascending=True)
    
    colors = plt.cm.coolwarm(np.linspace(0.3, 0.9, len(df_top)))
    
    ax.barh(df_top['feature'], df_top['importance'], color=colors)
    
    ax.set_xlabel('Importance', fontsize=10)
    ax.set_ylabel('Feature', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def create_evaluation_dashboard(
    results: Dict,
    data: pd.DataFrame,
    labels: pd.DataFrame,
    detector_name: str = "Detector",
    save_dir: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    Create complete evaluation dashboard with all visualizations.
    
    Args:
        results: Results from evaluate_detector
        data: Original telemetry data
        labels: True labels
        detector_name: Name of detector
        save_dir: Optional directory to save figures
        
    Returns:
        Dictionary of figure names to figures
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
    figures = {}
    y_true = labels['is_anomaly'].values if 'is_anomaly' in labels.columns \
             else labels.values.flatten()
    
    # 1. Telemetry with anomalies
    figures['telemetry'] = plot_telemetry_with_anomalies(
        data, labels, results['predictions'],
        title=f"Telemetry with {detector_name} Detections",
        save_path=str(save_dir / 'telemetry.png') if save_dir else None
    )
    
    # 2. Anomaly scores
    figures['scores'] = plot_anomaly_scores(
        results['scores'], y_true, results['metrics'].get('threshold'),
        title=f"{detector_name} Anomaly Scores",
        save_path=str(save_dir / 'scores.png') if save_dir else None
    )
    
    # 3. ROC curve
    figures['roc'] = plot_roc_curve(
        results['roc_curve']['fpr'],
        results['roc_curve']['tpr'],
        results['metrics'].get('roc_auc'),
        title=f"{detector_name} ROC Curve",
        save_path=str(save_dir / 'roc.png') if save_dir else None
    )
    
    # 4. PR curve
    figures['pr'] = plot_precision_recall_curve(
        results['pr_curve']['precision'],
        results['pr_curve']['recall'],
        results['metrics'].get('pr_auc'),
        title=f"{detector_name} Precision-Recall Curve",
        save_path=str(save_dir / 'pr.png') if save_dir else None
    )
    
    # 5. Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, results['predictions'])
    figures['confusion'] = plot_confusion_matrix(
        cm,
        title=f"{detector_name} Confusion Matrix",
        save_path=str(save_dir / 'confusion.png') if save_dir else None
    )
    
    # 6. Score distribution
    figures['distribution'] = plot_score_distribution(
        results['scores'], y_true, results['metrics'].get('threshold'),
        title=f"{detector_name} Score Distribution",
        save_path=str(save_dir / 'distribution.png') if save_dir else None
    )
    
    return figures


# ==============================================================================
# Convenience wrapper functions for easier script usage
# ==============================================================================

def plot_telemetry_with_anomalies(
    data: pd.DataFrame,
    true_labels: Union[np.ndarray, pd.Series, pd.DataFrame],
    pred_labels: Optional[np.ndarray] = None,
    channels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    title: str = "Spacecraft Telemetry with Detected Anomalies",
    save_path: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot telemetry time series with anomalies highlighted.
    Flexible wrapper that handles various input formats.
    """
    if channels is None:
        channels = list(data.columns)[:4]
        
    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    
    if n_channels == 1:
        axes = [axes]
    
    # Convert labels to numpy array
    if isinstance(true_labels, pd.DataFrame):
        y_true = true_labels.values.flatten()
    elif isinstance(true_labels, pd.Series):
        y_true = true_labels.values
    else:
        y_true = np.asarray(true_labels)
    
    # Limit to data length
    y_true = y_true[:len(data)]
    true_anomalies = np.where(y_true == 1)[0]
    
    if pred_labels is not None:
        pred_labels = np.asarray(pred_labels)[:len(data)]
        pred_anomalies = np.where(pred_labels == 1)[0]
        
    for i, (ax, channel) in enumerate(zip(axes, channels)):
        values = data[channel].values
        ax.plot(range(len(values)), values, 'b-', linewidth=0.5, label='Normal', alpha=0.7)
        
        if len(true_anomalies) > 0:
            valid_idx = true_anomalies[true_anomalies < len(values)]
            ax.scatter(valid_idx, values[valid_idx], c='red', s=20, 
                      label='True Anomaly', zorder=5, alpha=0.7)
            
        if pred_labels is not None and len(pred_anomalies) > 0:
            valid_idx = pred_anomalies[pred_anomalies < len(values)]
            ax.scatter(valid_idx, values[valid_idx], c='orange', s=10, 
                      marker='x', label='Predicted', zorder=4)
            
        ax.set_ylabel(channel, fontsize=10)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
            
    axes[-1].set_xlabel('Time', fontsize=10)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "ROC Curve",
    save_path: Optional[str] = None
) -> Union[plt.Figure, plt.Axes]:
    """
    Plot ROC curve from y_true and scores.
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.figure
        standalone = False
    
    lbl = label if label else 'ROC Curve'
    lbl = f'{lbl} (AUC={roc_auc:.3f})'
    
    kwargs = {'label': lbl, 'linewidth': 2}
    if color is not None:
        kwargs['color'] = color
        
    ax.plot(fpr, tpr, **kwargs)
    
    if standalone:
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
    return fig if standalone else ax


def plot_precision_recall_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    color: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "Precision-Recall Curve",
    save_path: Optional[str] = None
) -> Union[plt.Figure, plt.Axes]:
    """
    Plot Precision-Recall curve from y_true and scores.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        standalone = True
    else:
        fig = ax.figure
        standalone = False
    
    lbl = label if label else 'PR Curve'
    lbl = f'{lbl} (AP={ap:.3f})'
    
    kwargs = {'label': lbl, 'linewidth': 2}
    if color is not None:
        kwargs['color'] = color
        
    ax.plot(recall, precision, **kwargs)
    
    if standalone:
        ax.set_xlabel('Recall', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='lower left')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
    return fig if standalone else ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = ['Normal', 'Anomaly'],
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix from y_true and y_pred.
    """
    from sklearn.metrics import confusion_matrix as sk_cm
    
    cm = sk_cm(y_true, y_pred)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax, cbar=True)
    
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_score_distribution(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Anomaly Score Distribution",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of anomaly scores for normal vs anomaly samples.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
        
    labels = np.asarray(labels)
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='green', density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red', density=True)
    
    if threshold is not None:
        ax.axvline(x=threshold, color='black', linestyle='--', 
                   label=f'Threshold ({threshold:.3f})', linewidth=2)
    else:
        # Use 95th percentile as default threshold
        thresh = np.percentile(scores, 95)
        ax.axvline(x=thresh, color='black', linestyle='--', 
                   label=f'Threshold (95%)', linewidth=2)
        
    ax.set_xlabel('Anomaly Score', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_model_comparison(
    metrics_dict: Dict[str, Dict],
    metrics_to_plot: List[str] = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Model Performance Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of multiple models from metrics dictionary.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_dict).T
    
    # Filter to available metrics
    available_metrics = [m for m in metrics_to_plot if m in df.columns]
    df = df[available_metrics]
    
    x = np.arange(len(available_metrics))
    width = 0.8 / len(df)
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    
    for i, (model, row) in enumerate(df.iterrows()):
        offset = (i - len(df)/2 + 0.5) * width
        ax.bar(x + offset, row.values, width, label=model, color=colors[i])
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    
    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def plot_feature_importance(
    importance: Union[np.ndarray, Dict, pd.DataFrame],
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Feature Importance",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance bar chart.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to DataFrame
    if isinstance(importance, dict):
        df = pd.DataFrame(list(importance.items()), columns=['feature', 'importance'])
    elif isinstance(importance, np.ndarray):
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importance))]
        df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    else:
        df = importance
    
    # Sort and get top N
    df = df.sort_values('importance', ascending=True).tail(top_n)
    
    colors = plt.cm.coolwarm(np.linspace(0.3, 0.9, len(df)))
    ax.barh(df['feature'], df['importance'], color=colors)
    
    ax.set_xlabel('Importance', fontsize=10)
    ax.set_ylabel('Feature', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


if __name__ == "__main__":
    # Demo with synthetic data
    import sys
    sys.path.append('..')
    
    print("Generating demo visualizations...")
    
    # Create dummy data
    np.random.seed(42)
    n = 500
    
    # Simulate time series data
    t = np.linspace(0, 10, n)
    data = pd.DataFrame({
        'temp': 25 + 5 * np.sin(t) + np.random.normal(0, 0.5, n),
        'voltage': 28 + np.random.normal(0, 0.3, n),
        'current': 2 + 0.5 * np.sin(2*t) + np.random.normal(0, 0.1, n),
    }, index=pd.date_range('2024-01-01', periods=n, freq='h'))
    
    # Inject some anomalies
    anomaly_idx = [50, 51, 52, 200, 201, 350, 351, 352, 353]
    data.iloc[anomaly_idx, 0] += 10  # Temperature spike
    
    labels = np.zeros(n, dtype=int)
    labels[anomaly_idx] = 1
    
    # Simulate predictions
    predictions = np.zeros(n, dtype=int)
    predictions[[50, 51, 200, 350, 351, 352, 400]] = 1
    
    scores = np.random.normal(0, 1, n)
    scores[anomaly_idx] += 3
    
    print("Plotting telemetry with anomalies...")
    fig1 = plot_telemetry_with_anomalies(data, labels, predictions)
    
    print("Plotting score distribution...")
    fig2 = plot_score_distribution(scores, labels, threshold=2.0)
    
    plt.show()
    print("Done!")
