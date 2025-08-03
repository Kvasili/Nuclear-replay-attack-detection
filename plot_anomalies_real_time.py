"""
    Replay Attack Visualization on Reactor Signals

    Description:
        This script animates the output of a trained autoencoder-based anomaly detector
        applied to multivariate reactor time-series data. Detected anomalies are
        highlighted, and contributing signals based on SHAP values are persistently tracked.


    Requirements:
        - pandas
        - matplotlib
        - Pillow (for saving GIFs)
        - numpy
        - keras (for loading the model)

    Output:
        - A GIF animation showing the the contributing replayed signals.

    Usage:
        python plot_anomalies_real_time.py

"""

import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# <---------------------------- Configuration -------------------------------->

# Add path to the CSV file containing the replay attack shap results
# This file should contain the signals and their SHAP values
DATA_PATH = "./test_6_signals.csv"
GIF_OUTPUT = "test_6_signals_legend.gif"

# Threshold for anomaly detection
ERROR_THRESHOLD = 0.10

# Start and end indices
WINDOW_SIZE = 10
START_INDEX = 250 + WINDOW_SIZE
END_INDEX = 450 + WINDOW_SIZE

# Signal columns of interest
SIGNALS = [
    "Channel 1 counts", "Channel 3 power", "Channel 4 flux",
    "RR position", "rr-active-state",
    "SS1 position", "ss1-active-state",
    "SS2 position", "ss2-active-state"
]

# Subset of signals to plot
PLOT_SIGNALS = [
    "Channel 1 counts", "Channel 3 power", "Channel 4 flux",
    "RR position", "SS1 position", "SS2 position"
]

# <---------------------------- Load Data ------------------------------------>

data = pd.read_csv(DATA_PATH)
print(f"Loaded data from {DATA_PATH} with shape {data.shape}")
print(data.head())

# <---------------------------- Animation Setup ------------------------------>

fig, ax = plt.subplots()
highlighted_segments = []  # Store persistent red lines (SHAP > 0)


def animate(i):
    ax.clear()

    current_idx = i
    t_end = START_INDEX + current_idx
    t_start = t_end - WINDOW_SIZE

    time_index = list(range(START_INDEX, t_end + 1))
    slice_data = data.iloc[:current_idx + 1]

    # Plot base signals
    for signal in PLOT_SIGNALS:
        ax.plot(time_index, slice_data[signal], label=signal)

    # Highlight abnormal (red) or normal (green) segments
    if data.loc[current_idx, 'reconstruction_error'] > ERROR_THRESHOLD:
        ax.axvspan(t_start, t_end, color='red', alpha=0.2)

        for signal in SIGNALS:
            shap_col = f"SHAP_{signal}"
            shap_val = data.loc[current_idx, shap_col]

            if pd.notna(shap_val) and shap_val > 0:
                time_seg = list(range(t_end - 1, t_end + 1))
                sig_seg = data[signal].iloc[current_idx -
                                            1:current_idx + 1].values
                highlighted_segments.append((time_seg, sig_seg, signal))

    elif current_idx > WINDOW_SIZE:
        ax.axvspan(t_start, t_end, color='green', alpha=0.2)

    # Re-plot previously detected red segments
    for time_seg, sig_seg, _ in highlighted_segments:
        ax.plot(time_seg, sig_seg, color='red', linewidth=2)

    # Configure plot appearance
    ax.set_title(f"Replay Attack Detection on PUR-1 Signals (t = {t_end}s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Values")
    ax.set_ylim(0, 1.5)
    ax.set_xlim(START_INDEX, END_INDEX)
    ax.legend(loc='upper right')
    plt.tight_layout()


# <---------------------------- Run & Save Animation ------------------------->

# Ensure 'gifs/' directory exists in the current script location
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'gifs')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define GIF output path
GIF_OUTPUT = os.path.join(RESULTS_DIR, "test_6_signals_legend.gif")


ani = FuncAnimation(fig, animate, frames=len(data), interval=1000)
ani.save(GIF_OUTPUT, writer='pillow', fps=5)

print(f"✅ Animation saved as: {GIF_OUTPUT}")
print("✅ Animation complete. Check the GIF file.")
