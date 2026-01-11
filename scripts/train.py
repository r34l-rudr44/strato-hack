#!/usr/bin/env python3
"""
StratoHack 2.0 - Model Training Pipeline
Train anomaly detection models on OPSSAT or synthetic data.

Usage:
    python scripts/train.py                    # Train on OPSSAT data
    python scripts/train.py --synthetic        # Train on synthetic data
    python scripts/train.py --config my.yaml   # Use custom config
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.statistical import ZScoreDetector, MADDetector
from src.models.isolation_forest import IsolationForestDetector
from src.models.one_class_svm import OneClassSVMDetector
from src.models.ensemble import EnsembleDetector
from src.evaluation.metrics import compute_all_metrics


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_data(args, config: dict):
    """Load and split data for training."""
    print("\n[1/4] Loading data...")
    
    loader = DataLoader(data_dir="data")
    
    if args.synthetic:
        print("      Using synthetic telemetry data")
        synth_cfg = config.get('data', {}).get('synthetic', {})
        data, labels = loader.load_synthetic(
            n_timesteps=synth_cfg.get('n_timesteps', 5000),
            anomaly_ratio=synth_cfg.get('anomaly_ratio', 0.05)
        )
    else:
        print("      Loading OPSSAT-AD dataset")
        data, labels = loader.load_opssat()
    
    print(f"      Total samples: {len(data)}")
    print(f"      Features: {len(data.columns)}")
    print(f"      Anomaly ratio: {labels.mean():.2%}")
    
    return data, labels


def preprocess_data(data: pd.DataFrame, labels: np.ndarray, config: dict):
    """Preprocess data with scaling and optional feature engineering."""
    print("\n[2/4] Preprocessing data...")
    
    prep_cfg = config.get('preprocessing', {})
    train_cfg = config.get('training', {})
    
    # For OPSSAT dataset.csv, features are already extracted
    # Disable rolling/FFT by default for tabular data (not time series)
    feature_config = {
        "rolling_mean": False,
        "rolling_std": False,
        "rolling_min": False,
        "rolling_max": False,
        "rate_of_change": False,
        "fft_features": False,
    }
    
    preprocessor = DataPreprocessor(
        normalization=prep_cfg.get('normalization', 'standard'),
        window_size=1,  # No windowing for tabular data
        feature_config=feature_config
    )
    
    # Split data
    test_split = train_cfg.get('test_split', 0.2)
    val_split = train_cfg.get('val_split', 0.1)
    random_seed = train_cfg.get('random_seed', 42)
    
    n = len(data)
    
    # Shuffle indices
    np.random.seed(random_seed)
    indices = np.random.permutation(n)
    
    train_end = int(n * (1 - test_split - val_split))
    val_end = int(n * (1 - test_split))
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    X_train = data.iloc[train_idx]
    y_train = labels[train_idx]
    X_val = data.iloc[val_idx]
    y_val = labels[val_idx]
    X_test = data.iloc[test_idx]
    y_test = labels[test_idx]
    
    # Fit and transform
    X_train_proc = preprocessor.fit_transform(X_train, add_features=False)
    X_val_proc = preprocessor.transform(X_val, add_features=False)
    X_test_proc = preprocessor.transform(X_test, add_features=False)
    
    print(f"      Train: {len(X_train_proc)} samples, {y_train.sum()} anomalies")
    print(f"      Val:   {len(X_val_proc)} samples, {y_val.sum()} anomalies")
    print(f"      Test:  {len(X_test_proc)} samples, {y_test.sum()} anomalies")
    print(f"      Features: {X_train_proc.shape[1]}")
    
    return {
        'preprocessor': preprocessor,
        'X_train': X_train_proc,
        'y_train': y_train,
        'X_val': X_val_proc,
        'y_val': y_val,
        'X_test': X_test_proc,
        'y_test': y_test,
        'feature_names': list(X_train_proc.columns),
        'test_indices': test_idx,
    }


def train_models(data_dict: dict, config: dict):
    """Train all anomaly detection models."""
    print("\n[3/4] Training models...")
    
    X_train = data_dict['X_train'].values
    y_train = data_dict['y_train']
    X_val = data_dict['X_val'].values
    y_val = data_dict['y_val']
    
    models_cfg = config.get('models', {})
    train_cfg = config.get('training', {})
    
    # Get contamination from isolation_forest config
    contamination = models_cfg.get('isolation_forest', {}).get('contamination', 0.05)
    random_seed = train_cfg.get('random_seed', 42)
    
    # Initialize detectors
    detectors = {
        'ZScore': ZScoreDetector(
            threshold=models_cfg.get('zscore', {}).get('threshold', 3.0)
        ),
        'MAD': MADDetector(
            threshold=3.5
        ),
        'IsolationForest': IsolationForestDetector(
            n_estimators=models_cfg.get('isolation_forest', {}).get('n_estimators', 100),
            contamination=contamination,
            random_state=random_seed
        ),
        'OneClassSVM': OneClassSVMDetector(
            kernel=models_cfg.get('one_class_svm', {}).get('kernel', 'rbf'),
            nu=models_cfg.get('one_class_svm', {}).get('nu', 0.05)
        ),
    }
    
    trained_models = {}
    val_metrics = {}
    
    for name, detector in detectors.items():
        print(f"      Training {name}...")
        detector.fit(X_train, y_train)
        trained_models[name] = detector
        
        # Validate
        val_pred = detector.predict(X_val)
        val_scores = detector.score_samples(X_val)
        metrics = compute_all_metrics(y_val, val_pred, val_scores)
        val_metrics[name] = metrics
        print(f"         Val F1: {metrics['f1_score']:.3f}, ROC-AUC: {metrics['roc_auc']:.3f}")
    
    # Create ensemble with learned weights
    print("      Creating Ensemble...")
    f1_scores = [val_metrics[name]['f1_score'] for name in detectors.keys()]
    total_f1 = sum(f1_scores) + 1e-10
    weights = [f1 / total_f1 for f1 in f1_scores]
    
    ensemble = EnsembleDetector(
        detectors=list(trained_models.values()),
        voting=models_cfg.get('ensemble', {}).get('voting', 'soft'),
        weights=weights
    )
    ensemble.fit(X_train, y_train)
    
    # Validate ensemble
    val_pred = ensemble.predict(X_val)
    val_scores = ensemble.score_samples(X_val)
    ensemble_metrics = compute_all_metrics(y_val, val_pred, val_scores)
    print(f"         Ensemble Val F1: {ensemble_metrics['f1_score']:.3f}, ROC-AUC: {ensemble_metrics['roc_auc']:.3f}")
    
    trained_models['Ensemble'] = ensemble
    val_metrics['Ensemble'] = ensemble_metrics
    
    # Store learned weights for reference
    ensemble.detector_weights = dict(zip(detectors.keys(), weights))
    
    return trained_models, val_metrics


def evaluate_and_save(trained_models: dict, data_dict: dict, config: dict, output_dir: Path):
    """Evaluate models on test set and save artifacts."""
    print("\n[4/4] Evaluating and saving models...")
    
    X_test = data_dict['X_test'].values
    y_test = data_dict['y_test']
    feature_names = data_dict['feature_names']
    
    # Create output directories
    models_dir = output_dir / 'models'
    reports_dir = output_dir / 'reports'
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    test_metrics = {}
    all_predictions = {}
    
    for name, model in trained_models.items():
        pred = model.predict(X_test)
        scores = model.score_samples(X_test)
        metrics = compute_all_metrics(y_test, pred, scores)
        test_metrics[name] = metrics
        all_predictions[name] = {'predictions': pred, 'scores': scores}
        print(f"      {name}: F1={metrics['f1_score']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}, Acc={metrics['accuracy']:.3f}")
    
    # Save models
    for name, model in trained_models.items():
        model_path = models_dir / f'{name.lower()}_detector.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    # Save preprocessor
    preprocessor_path = models_dir / 'preprocessor.pkl'
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(data_dict['preprocessor'], f)
    
    # Save feature names
    with open(models_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # Save metrics
    metrics_df = pd.DataFrame(test_metrics).T
    metrics_df.to_csv(reports_dir / 'training_metrics.csv')
    
    # Save predictions for each model
    for name, preds in all_predictions.items():
        pred_df = pd.DataFrame({
            'y_true': y_test,
            'y_pred': preds['predictions'],
            'score': preds['scores']
        })
        pred_df.to_csv(reports_dir / f'test_predictions_{name.lower()}.csv', index=False)
    
    # Save test indices for reproducibility
    np.save(reports_dir / 'test_indices.npy', data_dict['test_indices'])
    
    print(f"\n      Models saved to: {models_dir}")
    print(f"      Metrics saved to: {reports_dir}")
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description='Train spacecraft anomaly detection models')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data instead of OPSSAT')
    args = parser.parse_args()
    
    # Get project root
    script_dir = Path(__file__).parent.absolute()
    project_dir = script_dir.parent
    
    # Load config
    config_path = project_dir / args.config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1
    
    config = load_config(str(config_path))
    output_dir = project_dir / args.output_dir
    
    print("=" * 60)
    print("StratoHack 2.0 - Spacecraft Anomaly Detection Training")
    print("=" * 60)
    print(f"Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    
    try:
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
        
        # Find best model
        best_name = max(test_metrics.items(), key=lambda x: x[1]['f1_score'])[0]
        best_metrics = test_metrics[best_name]
        
        print(f"\nBest Model: {best_name}")
        print(f"  Accuracy:  {best_metrics['accuracy']:.3f}")
        print(f"  Precision: {best_metrics['precision']:.3f}")
        print(f"  Recall:    {best_metrics['recall']:.3f}")
        print(f"  F1-Score:  {best_metrics['f1_score']:.3f}")
        print(f"  ROC-AUC:   {best_metrics['roc_auc']:.3f}")
        
        print("\nAll models test performance:")
        for name, m in sorted(test_metrics.items(), key=lambda x: -x[1]['f1_score']):
            print(f"  {name:15s}: F1={m['f1_score']:.3f}, ROC-AUC={m['roc_auc']:.3f}")
        
        print("\nNext steps:")
        print("  1. Run: python scripts/evaluate.py")
        print("  2. View visualizations in outputs/figures/")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease download the dataset first:")
        print("  python scripts/download_data.py")
        return 1
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
