# Spacecraft Anomaly Detection (Temporal Autoencoder)

A deep learning pipeline and live monitoring dashboard built to detect anomalies in spacecraft telemetry.

Using an unsupervised Time-Series Autoencoder and the NASA Telemanom dataset (SMAP satellite and MSL rover), this project flags critical events dynamically. By modeling the normal operating distribution of the spacecraft, the system adapts to local noise floors and prevents false alarms without relying on static, hardcoded safety limits.

## Key Features

- **Advanced Neural Architecture:** Combines 1D-CNNs for feature extraction, LSTMs for sequential dependencies, Bahdanau-style temporal attention, and skip connections for high-fidelity reconstruction.
- **Mission Control Dashboard:** An interactive Streamlit UI featuring live system status metrics and a customizable multi-sensor grid matrix for real-time telemetry analysis.
- **Dynamic Thresholding:** Implements Non-parametric Dynamic Thresholding (NDT) to intelligently adapt to shifting noise environments.
- **Automated Evaluation Pipeline:** Built-in testing scripts that automatically generate confusion matrices, and performance reports

## Repository Structure

- `app.py` — The main Streamlit dashboard application (Mission Control UI).
- `model.py` — The PyTorch definitions for the autoencoder and its ablation variants.
- `data_loader.py` — Data preprocessing, chronological windowing, and normalization.
- `evaluation.py` — Batch processing script to test model variants and calculate anomaly scores.
- `metrics.py` — Mathematical definitions for NDT and standard classification metrics (F1, Precision, Recall).
- `confusion_matrix.py` — Visualization engine for generating final academic-grade heatmaps and bar charts.

## Getting Started

**1. Install Dependencies**
Ensure you have Python 3.12+ and your virtual environment activated, then install the required packages:

```bash
pip install -r requirements.txt
```

**2. Run the Dashboard**
Launch the interactive mission control center:
Bash

streamlit run app.py

**3. Run the Evaluation Suite**
Generate the final metrics and charts for the ablation study:
Bash

python evaluation.py
python confusion_matrix.py

(Results will be saved in the evaluation_results/ directory).

## Results Overview

Our ablation study demonstrates that the full architecture (full_lstm_cnn_attention_skip) achieves the highest F1-Score on the test set. The inclusion of residual skip connections proved strictly necessary for reconstructing multidimensional normal telemetry, while the attention mechanism allowed the model to correctly focus on pre-anomaly sensor spikes.
