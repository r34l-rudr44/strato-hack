#!/usr/bin/env python3
"""
StratoHack 2.0 - Model Training Pipeline
Train anomaly detection models on OPSSAT or synthetic data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pickle
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.synthetic_generator import SpacecraftTelemetryGenerator
from src.models.statistical import ZScoreDetector, MADDetector, RollingZScoreDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.one_class_svm import OneClassSVMDetector
from src.models.ensemble import EnsembleDetector
from src.evaluation.metrics import compute_all_metrics


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_data(args, config):
    """Load and prepare data for training."""
    print("\n[1/4] Loading data...")
    
    if args.synthetic:
        print("      Using synthetic telemetry data")
        generator = SpacecraftTelemetryGenerator(
            n_timesteps=args.n_samples or 10000,
            anomaly_ratio=config['preprocessing']['contamination'],
            random_seed=config['training']['random_seed']
        )
        data, labels, anomaly_info = generator.generate()
    else:
        print(f"      Loading from: {args.data_path or config['data']['raw_dir']}")
        loader = DataLoader(config['data']['raw_dir'])
        data, labels = loader.load_opssat()
    
    print(f"      Total samples: {len(data)}")
    print(f"      Anomaly ratio: {labels.mean():.2%}")
    print(f"      Features: {data.shape[1]}")
    
    return data, labels


def preprocess_data(data, labels, config):
    """Preprocess data with feature engineering."""
    print("\n[2/4] Preprocessing data...")
    
    preprocessor = DataPreprocessor(
        window_size=config['preprocessing']['window_size'],
        normalization=config['preprocessing']['normalization'],
        add_rolling_features=True,
        add_rate_features=True
    )
    
    # Split data
    test_split = config['training']['test_split']
    val_split = config['training']['val_split']
    
    n = len(data)
    train_end = int(n * (1 - test_split - val_split))
    val_end = int(n * (1 - test_split))
    
    X_train = data.iloc[:train_end]
    y_train = labels[:train_end]
    X_val = data.iloc[train_end:val_end]
    y_val = labels[train_end:val_end]
    X_test = data.iloc[val_end:]
    y_test = labels[val_end:]
    
    # Fit and transform
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    # Align labels
    offset = preprocessor.window_size - 1
    y_train_aligned = y_train[offset:offset+len(X_train_proc)]
    y_val_aligned = y_val[offset:offset+len(X_val_proc)]
    y_test_aligned = y_test[offset:offset+len(X_test_proc)]
    
    print(f"      Train: {len(X_train_proc)} samples, {y_train_aligned.sum()} anomalies")
    print(f"      Val:   {len(X_val_proc)} samples, {y_val_aligned.sum()} anomalies")
    print(f"      Test:  {len(X_test_proc)} samples, {y_test_aligned.sum()} anomalies")
    print(f"      Engineered features: {X_train_proc.shape[1]}")
    
    return {
        'preprocessor': preprocessor,
        'X_train': X_train_proc,
        'y_train': y_train_aligned,
        'X_val': X_val_proc,
        'y_val': y_val_aligned,
        'X_test': X_test_proc,
        'y_test': y_test_aligned,
        'X_train_raw': X_train,
        'X_test_raw': X_test
    }


def train_models(data_dict, config):
    """Train all anomaly detection models."""
    print("\n[3/4] Training models...")
    
    X_train = data_dict['X_train'].values
    y_train = data_dict['y_train'].values
    X_val = data_dict['X_val'].values
    y_val = data_dict['y_val'].values
    
    contamination = config['preprocessing']['contamination']
    
    # Initialize detectors
    detectors = {
        'ZScore': ZScoreDetector(threshold=3.0),
        'MAD': MADDetector(threshold=3.5),
        'RollingZScore': RollingZScoreDetector(window_size=50, threshold=3.0),
        'IsolationForest': IsolationForestDetector(
            contamination=contamination,
            n_estimators=config['models']['isolation_forest']['n_estimators'],
            random_state=config['training']['random_seed']
        ),
        'OneClassSVM': OneClassSVMDetector(
            nu=config['models']['one_class_svm']['nu'],
            kernel=config['models']['one_class_svm']['kernel']
        )
    }
    
    trained_models = {}
    val_metrics = {}
    
    for name, detector in detectors.items():
        print(f"      Training {name}...")
        detector.fit(X_train, y_train)
        trained_models[name] = detector
        
        # Validate
        val_result = detector.predict(X_val)
        metrics = compute_all_metrics(y_val, val_result.predictions, val_result.scores)
        val_metrics[name] = metrics
        print(f"         Val F1: {metrics['f1_score']:.3f}, ROC-AUC: {metrics['roc_auc']:.3f}")
    
    # Create ensemble with learned weights
    print("      Creating Ensemble...")
    f1_scores = [val_metrics[name]['f1_score'] for name in detectors.keys()]
    weights = np.array(f1_scores) / sum(f1_scores)
    
    ensemble = EnsembleDetector(
        detectors=list(trained_models.values()),
        detector_names=list(trained_models.keys()),
        voting='soft',
        weights=weights
    )
    ensemble.fit(X_train, y_train)
    
    # Validate ensemble
    val_result = ensemble.predict(X_val)
    ensemble_metrics = compute_all_metrics(y_val, val_result.predictions, val_result.scores)
    print(f"         Ensemble Val F1: {ensemble_metrics['f1_score']:.3f}, ROC-AUC: {ensemble_metrics['roc_auc']:.3f}")
    
    trained_models['Ensemble'] = ensemble
    val_metrics['Ensemble'] = ensemble_metrics
    
    return trained_models, val_metrics


def evaluate_and_save(trained_models, data_dict, config, output_dir):
    """Evaluate models on test set and save."""
    print("\n[4/4] Evaluating and saving models...")
    
    X_test = data_dict['X_test'].values
    y_test = data_dict['y_test'].values
    
    test_metrics = {}
    
    for name, model in trained_models.items():
        result = model.predict(X_test)
        metrics = compute_all_metrics(y_test, result.predictions, result.scores)
        test_metrics[name] = metrics
        print(f"      {name}: F1={metrics['f1_score']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}, Acc={metrics['accuracy']:.3f}")
    
    # Save models
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    for name, model in trained_models.items():
        model_path = os.path.join(models_dir, f'{name.lower()}_detector.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    # Save preprocessor
    preprocessor_path = os.path.join(models_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(data_dict['preprocessor'], f)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, 'reports', 'training_metrics.csv')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    metrics_df = pd.DataFrame(test_metrics).T
    metrics_df.to_csv(metrics_path)
    
    print(f"\n      Models saved to: {models_dir}")
    print(f"      Metrics saved to: {metrics_path}")
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description='Train spacecraft anomaly detection models')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--data-path', type=str, help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--n-samples', type=int, help='Number of synthetic samples')
    args = parser.parse_args()
    
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Load config
    config_path = os.path.join(project_dir, args.config)
    config = load_config(config_path)
    
    output_dir = os.path.join(project_dir, args.output_dir)
    
    print("=" * 60)
    print("StratoHack 2.0 - Spacecraft Anomaly Detection Training")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    
    # Prepare data
    data, labels = prepare_data(args, config)
    
    # Preprocess
    data_dict = preprocess_data(data, labels, config)
    
    # Train models
    trained_models, val_metrics = train_models(data_dict, config)
    
    # Evaluate and save
    test_metrics = evaluate_and_save(trained_models, data_dict, config, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    best_model = max(test_metrics.items(), key=lambda x: x[1]['f1_score'])
    print(f"\nBest Model: {best_model[0]}")
    print(f"  F1-Score: {best_model[1]['f1_score']:.3f}")
    print(f"  ROC-AUC:  {best_model[1]['roc_auc']:.3f}")
    print(f"  Accuracy: {best_model[1]['accuracy']:.3f}")
    
    print("\nNext steps:")
    print("  1. Run: python scripts/evaluate.py")
    print("  2. View visualizations in outputs/figures/")
    
    return test_metrics


if __name__ == '__main__':
    main()
