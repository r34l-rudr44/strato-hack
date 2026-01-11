import streamlit as st
import pandas as pd
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Spacecraft Anomaly Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
FIGURES_DIR = Path('outputs/figures')
REPORTS_DIR = Path('outputs/reports')

def main():
    st.title("🛰️ Spacecraft Anomaly Detection System")
    st.markdown("""
    **StratoHack 2.0 - Problem Statement 2**
    
    An AI-powered system for detecting abnormal behavior in spacecraft telemetry data.
    """)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Methodology"])
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Methodology":
        show_methodology()

def show_dashboard():
    st.header("📊 Model Performance Dashboard")
    
    # Metrics Summary
    metrics_path = REPORTS_DIR / 'metrics_summary.csv'
    if metrics_path.exists():
        st.subheader("Key Metrics")
        df = pd.read_csv(metrics_path, index_col=0)
        st.dataframe(df.style.highlight_max(axis=0, color='lightgreen'))
    
    # Visualizations Gallery
    st.subheader("Visualizations")
    
    cols = st.columns(2)
    
    with cols[0]:
        st.image(str(FIGURES_DIR / 'dashboard.png'), caption="Project Dashboard", width="stretch")
        st.image(str(FIGURES_DIR / 'confusion_matrix.png'), caption="Confusion Matrix (Best Model)", width="stretch")
        st.image(str(FIGURES_DIR / 'anomaly_timeline.png'), caption="Anomaly Timeline", width="stretch")
        
    with cols[1]:
        st.image(str(FIGURES_DIR / 'model_comparison.png'), caption="Model Comparison", width="stretch")
        st.image(str(FIGURES_DIR / 'roc_curves.png'), caption="ROC Curves", width="stretch")
        st.image(str(FIGURES_DIR / 'score_distribution.png'), caption="Anomaly Score Distribution", width="stretch")

def show_methodology():
    st.header("Technical Methodology")
    st.markdown("""
    ### Approaches Implemented
    
    1.  **Statistical Methods**
        -   **Z-Score**: Simple outlier detection assuming Gaussian distribution.
        -   **MAD (Median Absolute Deviation)**: Robust statistical method.
        
    2.  **Machine Learning**
        -   **Isolation Forest**: Tree-based ensemble method.
        -   **One-Class SVM**: learns a decision function for novelty detection.
        
    3.  **Ensemble**
        -   Combines the outputs of all models using weighted voting based on validation performance.
        
    ### Project Structure
    
    The project is structured into modular components:
    -   `src.models`: Detector implementations
    -   `src.data`: Data loading and preprocessing
    -   `src.evaluation`: Performance metrics
    """)

if __name__ == "__main__":
    main()
