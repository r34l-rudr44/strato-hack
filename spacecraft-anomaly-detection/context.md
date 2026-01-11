# Context: Spacecraft Anomaly Detection Project

## Purpose
This project was created for **StratoHack 2.0 - Problem Statement 2**, a hackathon challenge focused on AI-powered spacecraft telemetry anomaly detection.

## What This Project Does
Detects abnormal behavior in spacecraft telemetry data to identify potential system failures early, using a hybrid approach combining:
- Statistical methods (Z-score, IQR)
- Machine Learning (Isolation Forest, One-Class SVM)
- Deep Learning (LSTM Autoencoders)
- Ensemble voting for robust predictions

## Dataset
Uses the **OPS-SAT Anomaly Dataset (OPSSAT-AD)** from ESA's OPS-SAT CubeSat mission, available at [Zenodo](https://zenodo.org/records/12588359).

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run demo with synthetic data
python scripts/demo.py

# Or run with real data
python scripts/download_data.py
python scripts/train.py
python scripts/evaluate.py
```

## Key Files
| File | Purpose |
|------|---------|
| `scripts/demo.py` | Quick demonstration script |
| `scripts/train.py` | Full training pipeline |
| `scripts/evaluate.py` | Model evaluation & visualization |
| `src/models/` | All model implementations |
| `config.yaml` | Configuration settings |

## Expected Performance
The ensemble model achieves ~91% accuracy, 88% precision, 93% recall, and 0.90 F1-score on the OPSSAT-AD dataset.

---
*Generated: January 2025*
