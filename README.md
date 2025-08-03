# Nuclear-replay-attack-detection
Requirements
Python 3.8+
pandas
numpy
keras / tensorflow
matplotlib
joblib


# Replay Attack Detection using Autoencoder and SHAP

This repository provides a real-time anomaly detection and explainability framework for multivariate time series (MTS) data collected from a PUR-1 reactor. 
The core functionality includes detecting replay attacks using a trained LSTM Autoencoder and interpreting the anomalies using a windowed SHAP method.

## 📂 Repository Structure

├── datasets/
│ └── Shutdown_X_replay_vY.csv # Input replay datasets
├── models/
│ └── lstm_autoencoder_replay_vX.model # Pre-trained AE model
├── results/
│ └── *.csv, *.gif # Output CSV and GIF files
├── real_time_reactor_anomaly_detection_shap.py
├── plot_anomalies_real_time.py
└── README.md



## 🚀 How to Run

### Step 1 – Run Anomaly Detection with SHAP Analysis

Run the `real_time_reactor_anomaly_detection_shap.py` script with your selected replay dataset from the `datasets/` directory:

This script:

Loads the replay dataset and pre-trained autoencoder.

Evaluates reconstruction error at each time step.

Applies a windowed SHAP algorithm to calculate feature contributions during high-error windows.

Saves the results (signals, reconstruction errors, and SHAP values) into a .csv file (e.g., test_6_signals.csv).


### Step 2 – Run the plot_anomalies_real_time.py script using the .csv file generated in Step 1:

This script:

Plots the reactor signal values over time.

Highlights anomalous windows based on reconstruction error.

Displays signals identified by SHAP as contributing to the anomaly.


### 📁 Data Availability

The training datasets used to develop the autoencoder are not publicly available due to confidentiality restrictions. 
Access can be granted upon request for research or academic purposes.



### Contact
For questions, please contact: Konstantinos Vasili at vasilik@purdue.edu




