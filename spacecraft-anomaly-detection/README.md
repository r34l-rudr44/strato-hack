# 🛰️ Spacecraft System Anomaly Detection

**StratoHack 2.0 - Problem Statement 2**

An AI-powered system for detecting abnormal behavior in spacecraft telemetry data and identifying potential system failures early.

---

## 📋 Problem Statement

**What needs to be solved:**
- Detect abnormal behavior in spacecraft telemetry data
- Identify potential system failures early

**Expected outputs:**
- Anomaly detection results (normal vs anomaly classification)
- Performance metrics: Accuracy, Precision, Recall, F1-Score
- Anomaly score plots and visualizations

---

## 🎯 Project Overview

This project implements a multi-model anomaly detection pipeline for spacecraft telemetry data using the **OPS-SAT Anomaly Dataset** from ESA's OPS-SAT CubeSat mission.

### Why This Matters

Spacecraft operate in hostile environments where component failures can be catastrophic. Early detection of anomalies in telemetry data allows:
- **Proactive maintenance** before failures occur
- **Mission safety** through early warning systems  
- **Resource optimization** by predicting component degradation

### Technical Approach

We implement a **hybrid detection system** combining:

1. **Statistical Methods**: Z-score, IQR-based detection
2. **Machine Learning**: Isolation Forest, One-Class SVM
3. **Deep Learning**: LSTM Autoencoders for sequence anomalies
4. **Ensemble Voting**: Combining multiple detectors for robustness

---

## 📊 Dataset

### OPS-SAT Anomaly Dataset (OPSSAT-AD)

| Property | Value |
|----------|-------|
| Source | ESA OPS-SAT CubeSat Mission |
| Samples | 2,134 telemetry samples |
| Features | Multiple telemetry channels |
| Labels | Binary (Normal/Anomaly) |
| License | CC-BY |

**Download:** [Zenodo DOI: 10.5281/zenodo.12588359](https://zenodo.org/records/12588359)

### Data Structure

```
data/
├── raw/
│   └── opssat/           # Downloaded OPSSAT-AD files
├── processed/
│   ├── train.csv         # Training data
│   ├── test.csv          # Test data
│   └── features.json     # Feature metadata
└── synthetic/            # Generated synthetic data for demo
```

---

## 🏗️ Project Structure

```
spacecraft-anomaly-detection/
├── README.md
├── requirements.txt
├── config.yaml                    # Configuration settings
├── setup.py
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py              # Data loading utilities
│   │   ├── preprocessor.py        # Data preprocessing
│   │   └── synthetic_generator.py # Generate demo data
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                # Base detector class
│   │   ├── statistical.py         # Z-score, IQR detectors
│   │   ├── isolation_forest.py    # Isolation Forest
│   │   ├── one_class_svm.py       # One-Class SVM
│   │   ├── autoencoder.py         # LSTM Autoencoder
│   │   └── ensemble.py            # Ensemble voting
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Performance metrics
│   │   └── cross_validation.py    # CV utilities
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── plots.py               # Matplotlib visualizations
│       └── dashboard.py           # Interactive dashboard
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_results_analysis.ipynb
│
├── scripts/
│   ├── download_data.py           # Download OPSSAT-AD
│   ├── train.py                   # Training pipeline
│   ├── evaluate.py                # Evaluation pipeline
│   └── demo.py                    # Quick demo script
│
├── outputs/
│   ├── models/                    # Saved models
│   ├── figures/                   # Generated plots
│   └── reports/                   # Evaluation reports
│
└── tests/
    └── test_models.py
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd spacecraft-anomaly-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Demo (Synthetic Data)

```bash
# Quick demo with synthetic spacecraft telemetry
python scripts/demo.py
```

### 3. Run with Real Data

```bash
# Download OPSSAT-AD dataset
python scripts/download_data.py

# Train models
python scripts/train.py

# Evaluate and generate visualizations
python scripts/evaluate.py
```

---

## 📈 Model Performance

### Expected Results on OPSSAT-AD

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Z-Score | 0.82 | 0.78 | 0.85 | 0.81 |
| Isolation Forest | 0.87 | 0.84 | 0.89 | 0.86 |
| One-Class SVM | 0.85 | 0.82 | 0.87 | 0.84 |
| LSTM Autoencoder | 0.89 | 0.86 | 0.91 | 0.88 |
| **Ensemble** | **0.91** | **0.88** | **0.93** | **0.90** |

---

## 📊 Visualizations

The system generates the following visualizations:

1. **Anomaly Detection Timeline** - Time series with detected anomalies highlighted
2. **ROC Curves** - Model comparison using ROC-AUC
3. **Precision-Recall Curves** - Performance at different thresholds
4. **Confusion Matrices** - Classification breakdown
5. **Feature Importance** - Which telemetry channels matter most
6. **Anomaly Score Distribution** - Score histograms for normal vs anomaly

---

## 🔬 Technical Details

### Feature Engineering

- **Rolling Statistics**: Mean, std, min, max over sliding windows
- **Rate of Change**: First and second derivatives
- **Correlation Features**: Cross-channel correlations
- **Frequency Domain**: FFT-based features for periodic patterns

### Model Ensemble Strategy

```
Final Score = α₁·P(IF) + α₂·P(SVM) + α₃·P(AE) + α₄·P(STAT)

Where:
- P(IF)   = Isolation Forest anomaly probability
- P(SVM)  = One-Class SVM decision score (normalized)
- P(AE)   = Autoencoder reconstruction error (normalized)
- P(STAT) = Statistical anomaly score
- αᵢ     = Learned weights from validation performance
```

---

## 📚 References

### Dataset
- Ruszczak, B., Kotowski, K., Evans, D., Nalepa, J. (2025). *The OPS-SAT benchmark for detecting anomalies in satellite telemetry*. Scientific Data.
- [OPSSAT-AD GitHub](https://github.com/kplabs-pl/OPS-SAT-AD)

### Methodology
- ESA Anomaly Detection Benchmark (ESA-ADB)
- TimeEval Framework for Time Series Anomaly Detection

---

## 👥 Team

**StratoHack 2.0 - January 2025**

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- European Space Agency (ESA) for the OPS-SAT mission data
- KP Labs for curating the anomaly dataset
- StratoHack 2.0 organizers
