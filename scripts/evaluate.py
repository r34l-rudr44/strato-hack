#!/usr/bin/env python3
"""
StratoHack 2.0 - Model Evaluation and Visualization
Generate comprehensive evaluation results and visualizations.

Usage:
    python scripts/evaluate.py                    # Evaluate trained models
    python scripts/evaluate.py --synthetic        # Use synthetic test data
"""

import sys
import os
import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader
from src.evaluation.metrics import compute_all_metrics
from src.visualization.plots import (
    plot_telemetry_with_anomalies,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_score_distribution,
    plot_model_comparison,
)


def load_models(models_dir: Path) -> dict:
    """Load trained models from disk."""
    models = {}
    
    for filepath in models_dir.glob("*_detector.pkl"):
        # Extract model name from filename
        name = filepath.stem.replace('_detector', '')
        # Normalize name
        name_map = {
            'zscore': 'ZScore',
            'mad': 'MAD',
            'isolationforest': 'IsolationForest',
            'oneclasssvm': 'OneClassSVM',
            'ensemble': 'Ensemble',
        }
        display_name = name_map.get(name.lower(), name.title())
        
        with open(filepath, 'rb') as f:
            models[display_name] = pickle.load(f)
    
    return models


def load_preprocessor(models_dir: Path):
    """Load preprocessor from disk."""
    preprocessor_path = models_dir / 'preprocessor.pkl'
    if not preprocessor_path.exists():
        return None
    with open(preprocessor_path, 'rb') as f:
        return pickle.load(f)


def load_feature_names(models_dir: Path) -> list:
    """Load feature names from disk."""
    path = models_dir / 'feature_names.json'
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


def prepare_test_data(args, preprocessor, project_dir: Path):
    """Load and prepare test data."""
    print("\n[1/3] Loading test data...")
    
    loader = DataLoader(data_dir=str(project_dir / "data"))
    
    if args.synthetic:
        print("      Using synthetic data")
        data, labels = loader.load_synthetic(n_timesteps=5000, anomaly_ratio=0.05)
    else:
        print("      Loading OPSSAT data")
        data, labels = loader.load_opssat()
    
    # Preprocess
    if preprocessor is not None:
        X_proc = preprocessor.transform(data, add_features=False)
    else:
        # Fallback: just use data as-is (numeric columns)
        X_proc = data.select_dtypes(include=[np.number])
    
    print(f"      Loaded {len(X_proc)} samples with {X_proc.shape[1]} features")
    print(f"      Anomaly ratio: {labels.mean():.2%}")
    
    return X_proc, labels, data


def evaluate_models(models: dict, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate all models and compute metrics."""
    print("\n[2/3] Evaluating models...")
    
    results = {}
    all_metrics = {}
    
    for name, model in models.items():
        pred = model.predict(X_test)
        scores = model.score_samples(X_test)
        metrics = compute_all_metrics(y_test, pred, scores)
        
        results[name] = {
            'predictions': pred,
            'scores': scores,
            'metrics': metrics,
        }
        all_metrics[name] = metrics
        
        print(f"      {name}: F1={metrics['f1_score']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}, Acc={metrics['accuracy']:.3f}")
    
    return results, all_metrics


def generate_visualizations(results: dict, all_metrics: dict, y_test: np.ndarray, 
                           X_raw: pd.DataFrame, figures_dir: Path, feature_names: list = None):
    """Generate all evaluation visualizations."""
    print("\n[3/3] Generating visualizations...")
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Find best model
    best_name = max(all_metrics.items(), key=lambda x: x[1]['f1_score'])[0]
    best_result = results[best_name]
    
    # Color palette
    n_models = len(results)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_models, 3)))
    
    # 1. ROC curves for all models
    print("      Generating ROC curves...")
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    for i, (name, result) in enumerate(results.items()):
        plot_roc_curve(y_test, result['scores'], label=name, ax=ax_roc, color=colors[i % len(colors)])
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax_roc.set_title('ROC Curves - All Detectors', fontsize=14)
    ax_roc.legend(loc='lower right')
    fig_roc.savefig(figures_dir / 'roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig_roc)
    
    # 2. Precision-Recall curves
    print("      Generating PR curves...")
    fig_pr, ax_pr = plt.subplots(figsize=(10, 8))
    for i, (name, result) in enumerate(results.items()):
        plot_precision_recall_curve(y_test, result['scores'], label=name, ax=ax_pr, color=colors[i % len(colors)])
    ax_pr.set_title('Precision-Recall Curves', fontsize=14)
    ax_pr.legend(loc='lower left')
    fig_pr.savefig(figures_dir / 'pr_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig_pr)
    
    # 3. Confusion matrix for best model
    print("      Generating confusion matrix...")
    fig_cm = plot_confusion_matrix(y_test, best_result['predictions'], title=f'{best_name} Confusion Matrix')
    fig_cm.savefig(figures_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close(fig_cm)
    
    # 4. Score distribution for best model
    print("      Generating score distribution...")
    fig_dist = plot_score_distribution(best_result['scores'], y_test, title=f'{best_name} Score Distribution')
    fig_dist.savefig(figures_dir / 'score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close(fig_dist)
    
    # 5. Model comparison bar chart
    print("      Generating model comparison...")
    fig_comp = plot_model_comparison(all_metrics)
    fig_comp.savefig(figures_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig_comp)
    
    # 6. Anomaly timeline (sample-based visualization)
    print("      Generating anomaly timeline...")
    n_display = min(500, len(y_test))
    
    fig_timeline, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Top: anomaly scores
    ax = axes[0]
    ax.plot(range(n_display), best_result['scores'][:n_display], 'b-', alpha=0.7, linewidth=0.8, label='Anomaly Score')
    anomaly_idx = np.where(y_test[:n_display] == 1)[0]
    if len(anomaly_idx) > 0:
        ax.scatter(anomaly_idx, best_result['scores'][:n_display][anomaly_idx], 
                  c='red', s=30, zorder=5, label='True Anomaly')
    ax.axhline(y=np.percentile(best_result['scores'], 95), color='green', 
              linestyle='--', alpha=0.7, label='95th percentile')
    ax.set_ylabel('Anomaly Score')
    ax.set_title(f'{best_name} Anomaly Detection Results (first {n_display} samples)')
    ax.legend(loc='upper right')
    
    # Bottom: predictions vs true
    ax = axes[1]
    ax.fill_between(range(n_display), y_test[:n_display], alpha=0.3, color='red', label='True Anomaly')
    ax.fill_between(range(n_display), best_result['predictions'][:n_display], alpha=0.3, color='blue', label='Predicted')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Label')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Normal', 'Anomaly'])
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    fig_timeline.savefig(figures_dir / 'anomaly_timeline.png', dpi=150, bbox_inches='tight')
    plt.close(fig_timeline)
    
    # 7. Create comprehensive dashboard
    print("      Generating dashboard...")
    create_dashboard(results, all_metrics, y_test, figures_dir)
    
    print(f"      Saved all visualizations to: {figures_dir}")


def create_dashboard(results: dict, all_metrics: dict, y_test: np.ndarray, figures_dir: Path):
    """Create a comprehensive results dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('StratoHack 2.0 - Spacecraft Anomaly Detection Results', fontsize=16, fontweight='bold')
    
    best_name = max(all_metrics.items(), key=lambda x: x[1]['f1_score'])[0]
    best_result = results[best_name]
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    # 1. Anomaly scores timeline
    ax = axes[0, 0]
    n_display = min(300, len(y_test))
    ax.plot(range(n_display), best_result['scores'][:n_display], 'b-', alpha=0.7, linewidth=0.8)
    anomaly_idx = np.where(y_test[:n_display] == 1)[0]
    if len(anomaly_idx) > 0:
        ax.scatter(anomaly_idx, best_result['scores'][:n_display][anomaly_idx], c='red', s=20, zorder=5)
    ax.set_title('Anomaly Scores (Sample)')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Score')
    
    # 2. ROC curves
    ax = axes[0, 1]
    from sklearn.metrics import roc_curve, auc
    for i, (name, result) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, result['scores'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], label=f'{name} ({roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(fontsize=7, loc='lower right')
    
    # 3. Performance metrics comparison
    ax = axes[0, 2]
    metrics_df = pd.DataFrame({name: {
        'Accuracy': m['accuracy'],
        'Precision': m['precision'],
        'Recall': m['recall'],
        'F1': m['f1_score']
    } for name, m in all_metrics.items()}).T
    x = np.arange(len(metrics_df.columns))
    width = 0.12
    for i, (model, row) in enumerate(metrics_df.iterrows()):
        ax.bar(x + i*width, row.values, width, label=model, color=colors[i])
    ax.set_xticks(x + width * len(metrics_df) / 2)
    ax.set_xticklabels(metrics_df.columns)
    ax.set_ylim(0, 1.1)
    ax.set_title('Performance Metrics')
    ax.legend(fontsize=6, loc='lower right')
    
    # 4. Confusion matrix
    ax = axes[1, 0]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, best_result['predictions'])
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Anomaly'])
    ax.set_yticklabels(['Normal', 'Anomaly'])
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14, color=color)
    ax.set_title(f'{best_name} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    # 5. Score distribution
    ax = axes[1, 1]
    normal_scores = best_result['scores'][y_test == 0]
    anomaly_scores = best_result['scores'][y_test == 1]
    ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='green', density=True)
    if len(anomaly_scores) > 0:
        ax.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red', density=True)
    threshold = np.percentile(best_result['scores'], 95)
    ax.axvline(threshold, color='black', linestyle='--', label='95th pctl')
    ax.set_title(f'{best_name} Score Distribution')
    ax.set_xlabel('Anomaly Score')
    ax.legend()
    
    # 6. Summary table
    ax = axes[1, 2]
    ax.axis('off')
    best_metrics = all_metrics[best_name]
    summary_data = [
        ['Test Samples', f'{len(y_test):,}'],
        ['True Anomalies', f'{int(y_test.sum()):,}'],
        ['Anomaly Rate', f'{100*y_test.mean():.1f}%'],
        ['Best Model', best_name],
        ['Accuracy', f'{best_metrics["accuracy"]:.3f}'],
        ['Precision', f'{best_metrics["precision"]:.3f}'],
        ['Recall', f'{best_metrics["recall"]:.3f}'],
        ['F1-Score', f'{best_metrics["f1_score"]:.3f}'],
        ['ROC-AUC', f'{best_metrics["roc_auc"]:.3f}'],
    ]
    table = ax.table(cellText=summary_data, colLabels=['Metric', 'Value'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax.set_title('Summary', pad=20)
    
    plt.tight_layout()
    fig.savefig(figures_dir / 'dashboard.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_report(all_metrics: dict, reports_dir: Path):
    """Generate text report and CSV summary."""
    print("\n      Generating reports...")
    
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics CSV
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(reports_dir / 'metrics_summary.csv')
    
    # Generate text report
    best_name = max(all_metrics.items(), key=lambda x: x[1]['f1_score'])[0]
    best_m = all_metrics[best_name]
    
    report_lines = [
        "=" * 60,
        "SPACECRAFT ANOMALY DETECTION - EVALUATION REPORT",
        "StratoHack 2.0 - Problem Statement 2",
        "=" * 60,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "BEST MODEL",
        "-" * 40,
        f"Model:     {best_name}",
        f"Accuracy:  {best_m['accuracy']:.4f}",
        f"Precision: {best_m['precision']:.4f}",
        f"Recall:    {best_m['recall']:.4f}",
        f"F1-Score:  {best_m['f1_score']:.4f}",
        f"ROC-AUC:   {best_m['roc_auc']:.4f}",
        "",
        "ALL MODELS COMPARISON",
        "-" * 40,
    ]
    
    # Sort by F1
    for name, m in sorted(all_metrics.items(), key=lambda x: -x[1]['f1_score']):
        report_lines.append(
            f"{name:15s}: Acc={m['accuracy']:.3f}, Prec={m['precision']:.3f}, "
            f"Rec={m['recall']:.3f}, F1={m['f1_score']:.3f}, AUC={m['roc_auc']:.3f}"
        )
    
    report_lines.extend([
        "",
        "=" * 60,
        "OUTPUT FILES",
        "-" * 40,
        "Figures:  outputs/figures/",
        "  - roc_curves.png",
        "  - pr_curves.png",
        "  - confusion_matrix.png",
        "  - score_distribution.png",
        "  - model_comparison.png",
        "  - anomaly_timeline.png",
        "  - dashboard.png",
        "",
        "Reports:  outputs/reports/",
        "  - metrics_summary.csv",
        "  - evaluation_report.txt",
        "",
        "=" * 60,
    ])
    
    report_text = "\n".join(report_lines)
    
    with open(reports_dir / 'evaluation_report.txt', 'w') as f:
        f.write(report_text)
    
    print(f"      Report saved to: {reports_dir / 'evaluation_report.txt'}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained anomaly detection models')
    parser.add_argument('--models-dir', type=str, default='outputs/models', help='Models directory')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic test data')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.absolute()
    project_dir = script_dir.parent
    
    models_dir = project_dir / args.models_dir
    output_dir = project_dir / args.output_dir
    figures_dir = output_dir / 'figures'
    reports_dir = output_dir / 'reports'
    
    print("=" * 60)
    print("StratoHack 2.0 - Model Evaluation")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check for trained models
    if not models_dir.exists() or not any(models_dir.glob("*.pkl")):
        print(f"\nError: No trained models found in {models_dir}")
        print("Please run training first:")
        print("  python scripts/train.py")
        return 1
    
    try:
        # Load models
        print(f"\n      Loading models from: {models_dir}")
        models = load_models(models_dir)
        print(f"      Loaded {len(models)} models: {list(models.keys())}")
        
        # Load preprocessor
        preprocessor = load_preprocessor(models_dir)
        feature_names = load_feature_names(models_dir)
        
        # Load test data
        X_proc, y_test, X_raw = prepare_test_data(args, preprocessor, project_dir)
        
        # Evaluate models
        results, all_metrics = evaluate_models(models, X_proc.values, y_test)
        
        # Generate visualizations
        generate_visualizations(results, all_metrics, y_test, X_raw, figures_dir, feature_names)
        
        # Generate report
        generate_report(all_metrics, reports_dir)
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE!")
        print("=" * 60)
        
        print("\nModel Performance Summary:")
        print("-" * 50)
        metrics_df = pd.DataFrame({name: {
            'Acc': m['accuracy'],
            'Prec': m['precision'],
            'Rec': m['recall'],
            'F1': m['f1_score'],
            'AUC': m['roc_auc']
        } for name, m in all_metrics.items()}).T
        print(metrics_df.round(3).to_string())
        
        best_name = max(all_metrics.items(), key=lambda x: x[1]['f1_score'])[0]
        print(f"\nBest Model: {best_name} (F1={all_metrics[best_name]['f1_score']:.3f})")
        
        print(f"\nOutput files:")
        print(f"  Figures: {figures_dir}/")
        print(f"  Reports: {reports_dir}/")
        
        return 0
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
