#!/usr/bin/env python3
"""
StratoHack 2.0 - Model Evaluation and Visualization
Generate comprehensive evaluation results and visualizations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from src.data.synthetic_generator import SpacecraftTelemetryGenerator
from src.evaluation.metrics import compute_all_metrics, generate_report
from src.visualization.plots import (
    plot_telemetry_with_anomalies,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_score_distribution,
    plot_model_comparison,
    plot_feature_importance
)


def load_models(models_dir):
    """Load trained models from disk."""
    models = {}
    
    for filename in os.listdir(models_dir):
        if filename.endswith('_detector.pkl'):
            model_name = filename.replace('_detector.pkl', '').title()
            if model_name == 'Oneclass':
                model_name = 'OneClassSVM'
            elif model_name == 'Isolationforest':
                model_name = 'IsolationForest'
            elif model_name == 'Rollingzscore':
                model_name = 'RollingZScore'
                
            with open(os.path.join(models_dir, filename), 'rb') as f:
                models[model_name] = pickle.load(f)
    
    return models


def generate_all_visualizations(models, X_test, y_test, X_raw, output_dir):
    """Generate all evaluation visualizations."""
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    print("\n[2/3] Generating visualizations...")
    
    # Get predictions from all models
    results = {}
    all_metrics = {}
    
    for name, model in models.items():
        result = model.predict(X_test)
        results[name] = result
        all_metrics[name] = compute_all_metrics(y_test, result.predictions, result.scores)
    
    # 1. Telemetry with anomalies (using Ensemble or best model)
    best_model_name = max(all_metrics.keys(), key=lambda k: all_metrics[k]['f1_score'])
    best_result = results[best_model_name]
    
    print(f"      Generating telemetry plot...")
    fig1 = plot_telemetry_with_anomalies(
        data=X_raw.iloc[:500],
        true_labels=y_test[:500] if len(y_test) >= 500 else y_test,
        pred_labels=best_result.predictions[:500] if len(best_result.predictions) >= 500 else best_result.predictions,
        channels=list(X_raw.columns[:3]),
        title=f'Telemetry with Anomalies ({best_model_name})'
    )
    fig1.savefig(os.path.join(figures_dir, 'telemetry_anomalies.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. ROC curves for all models
    print(f"      Generating ROC curves...")
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    for (name, result), color in zip(results.items(), colors):
        plot_roc_curve(y_test, result.scores, label=name, ax=ax2, color=color)
    ax2.set_title('ROC Curves - All Detectors', fontsize=14)
    ax2.legend(loc='lower right')
    fig2.savefig(os.path.join(figures_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Precision-Recall curves
    print(f"      Generating PR curves...")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    for (name, result), color in zip(results.items(), colors):
        plot_precision_recall_curve(y_test, result.scores, label=name, ax=ax3, color=color)
    ax3.set_title('Precision-Recall Curves', fontsize=14)
    ax3.legend(loc='lower left')
    fig3.savefig(os.path.join(figures_dir, 'pr_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Confusion matrices for top models
    print(f"      Generating confusion matrices...")
    top_models = sorted(all_metrics.items(), key=lambda x: x[1]['f1_score'], reverse=True)[:4]
    fig4, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (name, _) in zip(axes.flat, top_models):
        result = results[name]
        plot_confusion_matrix(y_test, result.predictions, title=f'{name}', ax=ax)
    plt.tight_layout()
    fig4.savefig(os.path.join(figures_dir, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    # 5. Score distributions
    print(f"      Generating score distributions...")
    fig5, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (name, result) in zip(axes.flat, list(results.items())[:6]):
        plot_score_distribution(result.scores, y_test, title=f'{name}', ax=ax)
    plt.tight_layout()
    fig5.savefig(os.path.join(figures_dir, 'score_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close(fig5)
    
    # 6. Model comparison
    print(f"      Generating model comparison...")
    fig6 = plot_model_comparison(all_metrics)
    fig6.savefig(os.path.join(figures_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig6)
    
    # 7. Feature importance (for Isolation Forest)
    if 'IsolationForest' in models and hasattr(models['IsolationForest'], 'get_feature_importance'):
        print(f"      Generating feature importance...")
        importance = models['IsolationForest'].get_feature_importance()
        if importance is not None:
            fig7 = plot_feature_importance(importance, top_n=20)
            fig7.savefig(os.path.join(figures_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
            plt.close(fig7)
    
    # 8. Complete dashboard
    print(f"      Generating dashboard...")
    create_dashboard(results, all_metrics, y_test, X_raw, figures_dir)
    
    print(f"      Saved all visualizations to: {figures_dir}")
    
    return all_metrics, results


def create_dashboard(results, all_metrics, y_test, X_raw, figures_dir):
    """Create a comprehensive results dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('StratoHack 2.0 - Spacecraft Anomaly Detection Results', fontsize=16, fontweight='bold')
    
    # Get best model
    best_name = max(all_metrics.keys(), key=lambda k: all_metrics[k]['f1_score'])
    best_result = results[best_name]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    # 1. Telemetry sample
    ax = axes[0, 0]
    sample = X_raw.iloc[:200]
    cols = list(sample.columns[:2])
    for col in cols:
        ax.plot(sample[col].values, label=col, alpha=0.8)
    anomaly_idx = np.where(y_test[:200] == 1)[0]
    if len(anomaly_idx) > 0:
        ax.scatter(anomaly_idx, sample[cols[0]].values[anomaly_idx], c='red', s=50, zorder=5, label='Anomaly')
    ax.set_title('Telemetry Sample')
    ax.set_xlabel('Time')
    ax.legend(fontsize=8)
    
    # 2. ROC curves
    ax = axes[0, 1]
    for (name, result), color in zip(results.items(), colors):
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, result.scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, label=f'{name} ({roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(fontsize=7)
    
    # 3. Performance comparison
    ax = axes[0, 2]
    metrics_df = pd.DataFrame(all_metrics).T[['accuracy', 'precision', 'recall', 'f1_score']]
    x = np.arange(len(metrics_df.columns))
    width = 0.12
    for i, (model, row) in enumerate(metrics_df.iterrows()):
        ax.bar(x + i*width, row.values, width, label=model, color=colors[i])
    ax.set_xticks(x + width * len(metrics_df) / 2)
    ax.set_xticklabels(['Acc', 'Prec', 'Recall', 'F1'])
    ax.set_ylim(0, 1)
    ax.set_title('Performance Metrics')
    ax.legend(fontsize=6, loc='lower right')
    
    # 4. Best model confusion matrix
    ax = axes[1, 0]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, best_result.predictions)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Anomaly'])
    ax.set_yticklabels(['Normal', 'Anomaly'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14,
                   color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set_title(f'{best_name} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    # 5. Score distribution
    ax = axes[1, 1]
    normal_scores = best_result.scores[y_test == 0]
    anomaly_scores = best_result.scores[y_test == 1]
    ax.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='green', density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
    threshold = np.percentile(best_result.scores, 95)
    ax.axvline(threshold, color='black', linestyle='--', label=f'Threshold')
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
        ['Best Model', best_name],
        ['Accuracy', f'{best_metrics["accuracy"]:.3f}'],
        ['Precision', f'{best_metrics["precision"]:.3f}'],
        ['Recall', f'{best_metrics["recall"]:.3f}'],
        ['F1-Score', f'{best_metrics["f1_score"]:.3f}'],
        ['ROC-AUC', f'{best_metrics["roc_auc"]:.3f}']
    ]
    table = ax.table(cellText=summary_data, colLabels=['Metric', 'Value'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax.set_title('Summary', pad=20)
    
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_report_file(all_metrics, output_dir):
    """Generate evaluation report."""
    print("\n[3/3] Generating report...")
    
    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    report_path = os.path.join(reports_dir, 'evaluation_report.txt')
    generate_report(all_metrics, output_path=report_path)
    
    # Also save as CSV
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(os.path.join(reports_dir, 'metrics_summary.csv'))
    
    print(f"      Report saved to: {report_path}")
    
    return metrics_df


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained anomaly detection models')
    parser.add_argument('--models-dir', type=str, default='outputs/models', help='Models directory')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--n-samples', type=int, default=5000, help='Number of test samples (synthetic)')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    models_dir = os.path.join(project_dir, args.models_dir)
    output_dir = os.path.join(project_dir, args.output_dir)
    
    print("=" * 60)
    print("StratoHack 2.0 - Model Evaluation")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load models
    print("\n[1/3] Loading models...")
    
    if os.path.exists(models_dir) and any(f.endswith('.pkl') for f in os.listdir(models_dir)):
        models = load_models(models_dir)
        print(f"      Loaded {len(models)} models: {list(models.keys())}")
        
        # Load preprocessor and test data
        with open(os.path.join(models_dir, 'preprocessor.pkl'), 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Generate test data
        generator = SpacecraftTelemetryGenerator(n_timesteps=args.n_samples, anomaly_ratio=0.05)
        data, labels, _ = generator.generate()
        
        X_proc = preprocessor.transform(data)
        y_aligned = labels[preprocessor.window_size-1:preprocessor.window_size-1+len(X_proc)]
        
        X_test = X_proc.values
        y_test = y_aligned.values
        X_raw = data.iloc[preprocessor.window_size-1:preprocessor.window_size-1+len(X_proc)]
        
    else:
        print("      No trained models found. Running quick evaluation with new models...")
        
        # Generate data
        generator = SpacecraftTelemetryGenerator(n_timesteps=args.n_samples, anomaly_ratio=0.05)
        data, labels, _ = generator.generate()
        
        from src.data.preprocessor import DataPreprocessor
        from src.models.statistical import ZScoreDetector, MADDetector
        from src.models.isolation_forest import IsolationForestDetector
        from src.models.one_class_svm import OneClassSVMDetector
        from src.models.ensemble import EnsembleDetector
        
        preprocessor = DataPreprocessor(window_size=50)
        
        train_size = int(0.7 * len(data))
        X_train = data.iloc[:train_size]
        y_train = labels[:train_size]
        X_test_raw = data.iloc[train_size:]
        y_test_raw = labels[train_size:]
        
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test_raw)
        
        y_train_aligned = y_train[preprocessor.window_size-1:preprocessor.window_size-1+len(X_train_proc)]
        y_test = y_test_raw[preprocessor.window_size-1:preprocessor.window_size-1+len(X_test_proc)].values
        
        X_train = X_train_proc.values
        X_test = X_test_proc.values
        X_raw = X_test_raw.iloc[preprocessor.window_size-1:preprocessor.window_size-1+len(X_test_proc)]
        
        # Train quick models
        models = {
            'ZScore': ZScoreDetector(threshold=3.0),
            'MAD': MADDetector(threshold=3.5),
            'IsolationForest': IsolationForestDetector(contamination=0.05),
            'OneClassSVM': OneClassSVMDetector(nu=0.05)
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train_aligned.values)
        
        # Create ensemble
        ensemble = EnsembleDetector(
            detectors=list(models.values()),
            detector_names=list(models.keys())
        )
        ensemble.fit(X_train, y_train_aligned.values)
        models['Ensemble'] = ensemble
        
        print(f"      Trained {len(models)} models for evaluation")
    
    # Generate visualizations
    all_metrics, results = generate_all_visualizations(models, X_test, y_test, X_raw, output_dir)
    
    # Generate report
    metrics_df = generate_report_file(all_metrics, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    
    print("\nModel Performance Summary:")
    print("-" * 50)
    print(metrics_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].round(3).to_string())
    
    best_model = max(all_metrics.items(), key=lambda x: x[1]['f1_score'])
    print(f"\n🏆 Best Model: {best_model[0]} (F1={best_model[1]['f1_score']:.3f})")
    
    print(f"\nOutput files:")
    print(f"  📊 Figures: {os.path.join(output_dir, 'figures')}/")
    print(f"  📝 Reports: {os.path.join(output_dir, 'reports')}/")
    
    return all_metrics


if __name__ == '__main__':
    main()
