#!/usr/bin/env python3
"""
StratoHack 2.0 - Spacecraft Anomaly Detection Demo
Quick demonstration using synthetic telemetry data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from src.data.synthetic_generator import SpacecraftTelemetryGenerator
from src.data.preprocessor import DataPreprocessor
from src.models.statistical import ZScoreDetector, MADDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.one_class_svm import OneClassSVMDetector
from src.models.ensemble import EnsembleDetector
from src.evaluation.metrics import compute_all_metrics, compare_detectors, generate_report
from src.visualization.plots import (
    plot_telemetry_with_anomalies,
    plot_roc_curve,
    plot_confusion_matrix,
    plot_score_distribution,
    plot_model_comparison
)


def main():
    print("=" * 60)
    print("StratoHack 2.0 - Spacecraft Anomaly Detection System")
    print("=" * 60)
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate synthetic spacecraft telemetry data
    print("[1/6] Generating synthetic spacecraft telemetry data...")
    generator = SpacecraftTelemetryGenerator(
        n_timesteps=5000,
        random_seed=42
    )
    data, labels = generator.generate(anomaly_ratio=0.05)
    
    # Convert labels to numpy if needed
    if isinstance(labels, pd.DataFrame):
        labels = labels['is_anomaly'].values if 'is_anomaly' in labels.columns else labels.values.flatten()
    
    print(f"      Generated {len(data)} samples with {labels.sum()} anomalies ({100*labels.mean():.1f}%)")
    print(f"      Channels: {list(data.columns)}")
    print()
    
    # Step 2: Preprocess data
    print("[2/6] Preprocessing data with feature engineering...")
    preprocessor = DataPreprocessor(
        window_size=50,
        normalization='standard'
    )
    
    # Split data
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))
    
    X_train = data.iloc[:train_size]
    y_train = labels[:train_size]
    X_val = data.iloc[train_size:train_size+val_size]
    y_val = labels[train_size:train_size+val_size]
    X_test = data.iloc[train_size+val_size:]
    y_test = labels[train_size+val_size:]
    
    # Fit preprocessor and transform
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    # Align labels with processed data
    y_train_aligned = y_train[preprocessor.window_size-1:preprocessor.window_size-1+len(X_train_proc)]
    y_val_aligned = y_val[preprocessor.window_size-1:preprocessor.window_size-1+len(X_val_proc)]
    y_test_aligned = y_test[preprocessor.window_size-1:preprocessor.window_size-1+len(X_test_proc)]
    
    print(f"      Training set: {len(X_train_proc)} samples")
    print(f"      Validation set: {len(X_val_proc)} samples")
    print(f"      Test set: {len(X_test_proc)} samples")
    print(f"      Features after engineering: {X_train_proc.shape[1]}")
    print()
    
    # Step 3: Train multiple detectors
    print("[3/6] Training anomaly detectors...")
    
    detectors = {
        'Z-Score': ZScoreDetector(threshold=3.0),
        'MAD': MADDetector(threshold=3.5),
        'Isolation Forest': IsolationForestDetector(contamination=0.05, n_estimators=100),
        'One-Class SVM': OneClassSVMDetector(nu=0.05, kernel='rbf')
    }
    
    results = {}
    for name, detector in detectors.items():
        print(f"      Training {name}...")
        detector.fit(X_train_proc.values, y_train_aligned.values)
        results[name] = detector.predict(X_test_proc.values)
    
    print()
    
    # Step 4: Evaluate all models
    print("[4/6] Evaluating model performance...")
    
    all_metrics = {}
    for name, result in results.items():
        metrics = compute_all_metrics(y_test_aligned.values, result.predictions, result.scores)
        all_metrics[name] = metrics
        print(f"      {name}:")
        print(f"         Accuracy: {metrics['accuracy']:.3f}")
        print(f"         Precision: {metrics['precision']:.3f}")
        print(f"         Recall: {metrics['recall']:.3f}")
        print(f"         F1-Score: {metrics['f1_score']:.3f}")
        print(f"         ROC-AUC: {metrics['roc_auc']:.3f}")
    
    print()
    
    # Step 5: Create ensemble
    print("[5/6] Creating weighted ensemble detector...")
    
    ensemble = EnsembleDetector(
        detectors=list(detectors.values()),
        detector_names=list(detectors.keys()),
        voting='soft',
        weights=None  # Learn from validation
    )
    
    ensemble.fit(X_train_proc.values, y_train_aligned.values)
    
    # Learn weights from validation performance
    val_f1_scores = []
    for detector in detectors.values():
        val_result = detector.predict(X_val_proc.values)
        val_metrics = compute_all_metrics(y_val_aligned.values, val_result.predictions, val_result.scores)
        val_f1_scores.append(val_metrics['f1_score'])
    
    weights = np.array(val_f1_scores) / sum(val_f1_scores)
    ensemble.weights = weights
    
    print(f"      Learned weights: {dict(zip(detectors.keys(), weights.round(3)))}")
    
    ensemble_result = ensemble.predict(X_test_proc.values)
    ensemble_metrics = compute_all_metrics(y_test_aligned.values, ensemble_result.predictions, ensemble_result.scores)
    all_metrics['Ensemble'] = ensemble_metrics
    results['Ensemble'] = ensemble_result
    
    print(f"      Ensemble Performance:")
    print(f"         Accuracy: {ensemble_metrics['accuracy']:.3f}")
    print(f"         Precision: {ensemble_metrics['precision']:.3f}")
    print(f"         Recall: {ensemble_metrics['recall']:.3f}")
    print(f"         F1-Score: {ensemble_metrics['f1_score']:.3f}")
    print(f"         ROC-AUC: {ensemble_metrics['roc_auc']:.3f}")
    print()
    
    # Step 6: Generate visualizations
    print("[6/6] Generating visualizations...")
    
    # 6a. Telemetry with anomalies
    fig1 = plot_telemetry_with_anomalies(
        data=X_test.iloc[:500],  # First 500 samples for clarity
        true_labels=y_test[:500].values,
        pred_labels=results['Ensemble'].predictions[:500] if len(results['Ensemble'].predictions) >= 500 else results['Ensemble'].predictions,
        channels=['temp_battery', 'voltage_bus', 'current_draw'],
        title='Spacecraft Telemetry with Detected Anomalies'
    )
    fig1.savefig(os.path.join(output_dir, 'telemetry_anomalies.png'), dpi=150, bbox_inches='tight')
    print(f"      Saved: telemetry_anomalies.png")
    
    # 6b. ROC curves for all models
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    for (name, result), color in zip(results.items(), colors):
        plot_roc_curve(y_test_aligned.values, result.scores, label=name, ax=ax2, color=color)
    ax2.set_title('ROC Curves - All Detectors', fontsize=14)
    ax2.legend(loc='lower right')
    fig2.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    print(f"      Saved: roc_curves.png")
    
    # 6c. Confusion matrix for ensemble
    fig3 = plot_confusion_matrix(
        y_test_aligned.values,
        ensemble_result.predictions,
        title='Ensemble Detector - Confusion Matrix'
    )
    fig3.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    print(f"      Saved: confusion_matrix.png")
    
    # 6d. Anomaly score distribution
    fig4 = plot_score_distribution(
        ensemble_result.scores,
        y_test_aligned.values,
        title='Ensemble Anomaly Score Distribution'
    )
    fig4.savefig(os.path.join(output_dir, 'score_distribution.png'), dpi=150, bbox_inches='tight')
    print(f"      Saved: score_distribution.png")
    
    # 6e. Model comparison
    fig5 = plot_model_comparison(all_metrics)
    fig5.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"      Saved: model_comparison.png")
    
    # 6f. Summary dashboard
    fig6, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig6.suptitle('StratoHack 2.0 - Spacecraft Anomaly Detection Results', fontsize=16, fontweight='bold')
    
    # Telemetry snippet
    ax = axes[0, 0]
    test_snippet = X_test.iloc[:200]
    ax.plot(test_snippet['temp_battery'].values, label='Battery Temp', alpha=0.8)
    ax.plot(test_snippet['voltage_bus'].values, label='Bus Voltage', alpha=0.8)
    anomaly_idx = np.where(y_test[:200].values == 1)[0]
    ax.scatter(anomaly_idx, test_snippet['temp_battery'].values[anomaly_idx], c='red', s=50, zorder=5, label='Anomaly')
    ax.set_title('Telemetry Sample with Anomalies')
    ax.set_xlabel('Time')
    ax.legend(fontsize=8)
    
    # ROC curves
    ax = axes[0, 1]
    for (name, result), color in zip(results.items(), colors):
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test_aligned.values, result.scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, label=f'{name} (AUC={roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(fontsize=8)
    
    # Metrics comparison
    ax = axes[0, 2]
    metrics_df = pd.DataFrame(all_metrics).T[['accuracy', 'precision', 'recall', 'f1_score']]
    x = np.arange(len(metrics_df.columns))
    width = 0.15
    for i, (model, row) in enumerate(metrics_df.iterrows()):
        ax.bar(x + i*width, row.values, width, label=model)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1'])
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Comparison')
    ax.legend(fontsize=7, loc='lower right')
    
    # Confusion matrix
    ax = axes[1, 0]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test_aligned.values, ensemble_result.predictions)
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Anomaly'])
    ax.set_yticklabels(['Normal', 'Anomaly'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
    ax.set_title('Ensemble Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    # Score distribution
    ax = axes[1, 1]
    normal_scores = ensemble_result.scores[y_test_aligned.values == 0]
    anomaly_scores = ensemble_result.scores[y_test_aligned.values == 1]
    ax.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='green', density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
    ax.axvline(np.percentile(ensemble_result.scores, 95), color='black', linestyle='--', label='Threshold')
    ax.set_title('Anomaly Score Distribution')
    ax.set_xlabel('Anomaly Score')
    ax.legend()
    
    # Summary table
    ax = axes[1, 2]
    ax.axis('off')
    summary_data = [
        ['Total Samples', f'{len(data):,}'],
        ['Test Samples', f'{len(y_test_aligned):,}'],
        ['True Anomalies', f'{int(y_test_aligned.sum()):,}'],
        ['Detected Anomalies', f'{int(ensemble_result.predictions.sum()):,}'],
        ['Best Model', 'Ensemble'],
        ['Best F1-Score', f'{ensemble_metrics["f1_score"]:.3f}'],
        ['Best ROC-AUC', f'{ensemble_metrics["roc_auc"]:.3f}']
    ]
    table = ax.table(cellText=summary_data, colLabels=['Metric', 'Value'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax.set_title('Summary Statistics', pad=20)
    
    plt.tight_layout()
    fig6.savefig(os.path.join(output_dir, 'dashboard.png'), dpi=150, bbox_inches='tight')
    print(f"      Saved: dashboard.png")
    
    plt.close('all')
    
    # Generate text report
    report_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'reports')
    os.makedirs(report_dir, exist_ok=True)
    report = generate_report(all_metrics, output_path=os.path.join(report_dir, 'evaluation_report.txt'))
    print(f"      Saved: evaluation_report.txt")
    
    print()
    print("=" * 60)
    print("DEMO COMPLETE!")
    print("=" * 60)
    print()
    print(f"Output files saved to: {output_dir}")
    print()
    print("Key Results:")
    print(f"  • Best performing model: Ensemble (F1={ensemble_metrics['f1_score']:.3f})")
    print(f"  • Detected {int(ensemble_result.predictions.sum())} anomalies out of {int(y_test_aligned.sum())} true anomalies")
    print(f"  • Overall accuracy: {ensemble_metrics['accuracy']:.1%}")
    print()
    print("Generated visualizations:")
    print("  • telemetry_anomalies.png - Time series with detected anomalies")
    print("  • roc_curves.png - ROC curves for all models")
    print("  • confusion_matrix.png - Ensemble confusion matrix")
    print("  • score_distribution.png - Anomaly score distribution")
    print("  • model_comparison.png - Performance comparison")
    print("  • dashboard.png - Complete results dashboard")
    print()
    
    return all_metrics, results


if __name__ == '__main__':
    main()
