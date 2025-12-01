#!/usr/bin/env python3
"""
    Description:
        This script performs real-time anomaly detection on nuclear reactor data using a trained LSTM autoencoder,
        combined with a SHAP-based interpretability module (WindowSHAP). It processes a given dataset window-by-window,
        reconstructs the input, computes the reconstruction error, and when abnormal, explains the anomaly via SHAP values.

    Usage:
        python real_time_reactor_anomaly_detection_shap2.py
"""

import os
import csv
import sys
import datetime
import pandas as pd
import numpy as np
from keras.models import load_model
import joblib

from WindowSHAP.windowshap import StationaryWindowSHAP

# Configuration
CONFIG = {
    "model_path": "./models/lstm_autoencoder_replay_v4.model",
    "background_data": "./replayed_datasets/background.csv",
    "test_data": "./replayed_datasets/Shutdown_6_replay_v2.csv",
    "output_csv": "test_6_signals.csv",
    "anomaly_threshold": 0.1,
    "sub_window_size": 5,
    "window_size": 10,
    "start_time": 250,
    "end_time": 450
}

FEATURES = [
    "Channel 1 counts", "Channel 3 power", "Channel 4 flux",
    "rr-active-state", "RR position", "ss1-active-state",
    "SS1 position", "ss2-active-state", "SS2 position"
]
SHAP_FEATURES = [f'SHAP_{f}' for f in FEATURES]
CSV_HEADERS = FEATURES + ["reconstruction_error"] + SHAP_FEATURES


def to_sequences(df, seq_size):
    """Converts a DataFrame into overlapping sequences for LSTM input."""
    return np.array([df.iloc[i:i+seq_size].values for i in range(len(df) - seq_size)])


def normalize_data(train_df, val_df, scaler_path):
    """Normalizes train and validation DataFrames using a saved scaler."""
    scaler = joblib.load(scaler_path)
    return pd.DataFrame(scaler.transform(train_df)), pd.DataFrame(scaler.transform(val_df))


def array_to_dataframe(array, original_columns):
    """Reconstructs a flat DataFrame from a 3D array (LSTM sequences)."""
    rows = []
    for i, sequence in enumerate(array):
        if i == 0:
            rows.extend(sequence)
        else:
            rows.append(sequence[-1])
    return pd.DataFrame(rows, columns=original_columns)


class SHAPBinding:

    '''A binding class to compute SHAP values for LSTM sequences.'''

    def __init__(self, model, window_size, sub_window_size):
        self.model = model
        self.window_size = window_size
        self.sub_window_size = sub_window_size

    def prediction_function(self, input_data):
        """
        Computes the MAE reconstruction error for the input sequence.
        Expects input_data of shape: (1, window_size, num_features)
        """
        predictions = self.model.predict(input_data)
        mae = np.mean(np.abs(input_data - predictions), axis=(1, 2))
        return mae.reshape(-1, 1)

    def explain_anomaly(self, current_seq, background_seq):
        """
        Computes SHAP values for the current sequence if an anomaly is detected.
        Both current_seq and background_seq must be of shape (1, window_size, num_features)
        """
        explainer = StationaryWindowSHAP(
            model=self.prediction_function,
            window_len=self.sub_window_size,
            B_ts=background_seq,
            test_ts=current_seq,
            model_type='lstm'
        )
        return explainer.shap_values()


def initialize_csv(filename, headers):
    """Creates the CSV file with header if it doesn't already exist."""
    if not os.path.isfile(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()


def main():
    print("[INFO] Starting anomaly detection...")

    # Load model
    model = load_model(CONFIG["model_path"])
    print("[INFO] Model loaded.")

    # Load datasets
    df_test = pd.read_csv(
        CONFIG["test_data"],
        usecols=FEATURES,
        skiprows=lambda x: x != 0 and (
            x < CONFIG["start_time"] or x > CONFIG["end_time"])
    ).reset_index(drop=True)

    df_background = pd.read_csv(
        CONFIG["background_data"],
        usecols=FEATURES,
        skiprows=lambda x: x != 0 and (
            x < CONFIG["start_time"] or x > CONFIG["end_time"])
    ).reset_index(drop=True)

    print(f"[INFO] Test dataset shape: {df_test.shape}")
    initialize_csv(CONFIG["output_csv"], CSV_HEADERS)

    for step in range(CONFIG["window_size"], df_test.shape[0] + 1):
        row_data = {}

        # Slice current and background windows
        current_window = df_test.iloc[step - CONFIG["window_size"]:step, :]
        background_window = df_background.iloc[step -
                                               CONFIG["window_size"]:step, :]

        # Reshape for LSTM input
        current_seq = current_window.values.reshape(
            1, CONFIG["window_size"], len(FEATURES))

        background_seq = background_window.values.reshape(
            1, CONFIG["window_size"], len(FEATURES))

        # Initialize SHAP binding
        shap_binding = SHAPBinding(
            model, CONFIG["window_size"], CONFIG["sub_window_size"])

        # Calculate reconstruction error
        error = shap_binding.prediction_function(current_seq)[0][0]
        print(f"[{CONFIG['start_time'] + step}] Reconstruction error: {error:.4f}")

        # Add current row (last timestamp of the current sequence)
        current_row = df_test.iloc[step - 1].to_dict()
        row_data.update(current_row)
        row_data['reconstruction_error'] = error

        # SHAP analysis if anomaly is detected
        if error > CONFIG["anomaly_threshold"]:
            print("  --> Anomaly detected. Computing SHAP values...")

            shap_vals = shap_binding.explain_anomaly(
                current_seq, background_seq)

            mean_shap = np.mean(shap_vals, axis=1).flatten()

            row_data.update({
                k: mean_shap[i] if mean_shap[i] > 0.0001 or mean_shap[i] < 0 else 0
                for i, k in enumerate(SHAP_FEATURES)
            })

        else:
            row_data.update({k: None for k in SHAP_FEATURES})

        # Append to CSV
        with open(CONFIG["output_csv"], 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writerow(row_data)

    print("[INFO] Anomaly detection complete.")
    print(f"[INFO] Results saved to: {CONFIG['output_csv']}")


if __name__ == "__main__":
    main()
