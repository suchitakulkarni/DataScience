# utils.py (Complete, Final Version)

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import os

def simulate_harmonic_oscillator(timesteps=1000, dt=0.01, omega=2.0, noise_std=0.0):
    """Simulates a simple harmonic oscillator with optional measurement noise."""
    x = np.zeros(timesteps)
    v = np.zeros(timesteps)
    x[0] = 1.0
    for t in range(1, timesteps):
        a = -omega**2 * x[t-1]
        v[t] = v[t-1] + a * dt
        x[t] = x[t-1] + v[t] * dt

    # Add Gaussian measurement noise
    if noise_std > 0:
        np.random.seed(123)  # Different seed from anomaly injection
        x += np.random.normal(0, noise_std, timesteps)

    return x

def inject_perturbations(x, num_anomalies=10, severity=2.0):
    """Injects random Gaussian noise anomalies into the signal."""
    x_anomalous = x.copy()
    np.random.seed(42) 
    anomaly_indices = np.random.choice(len(x), num_anomalies, replace=False)
    for idx in anomaly_indices:
        if idx < len(x) - 1:
            x_anomalous[idx] += severity * np.random.randn()
    return x_anomalous, anomaly_indices

def create_rolling_windows(data, window_size):
    """Converts a 1D time series into a 2D array of overlapping sequences."""
    X = []
    for i in range(len(data) - window_size + 1):
        X.append(data[i:i+window_size])
    return np.array(X)
    
def prepare_data(data, window_size, scaler=None):
    """Scales the data and creates rolling windows."""
    data_reshaped = data.reshape(-1, 1) 
    
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_reshaped).flatten()
    else:
        scaled_data = scaler.transform(data_reshaped).flatten()
        
    windows = create_rolling_windows(scaled_data, window_size)
    
    # Reshape for LSTM input: (N, seq_len, 1)
    windows_tensor = windows[:, :, np.newaxis] 
    
    return windows_tensor, scaler, scaled_data


def _get_full_reconstruction(reconstruction_windows, N_signal, window_size):
    """Helper to convert windowed reconstructions to a single signal."""
    N_windows = len(reconstruction_windows)
    reconstruction_full_pts = reconstruction_windows[:, window_size - 1, 0]
    
    reconstruction_padded = np.full(N_signal, np.nan)
    
    for i in range(N_windows):
        original_idx = i + window_size - 1
        if original_idx < N_signal:
            reconstruction_padded[original_idx] = reconstruction_full_pts[i]
    
    return reconstruction_padded

# --- Matplotlib Plotting Functions ---

def plot_physics_comparison_results(x_anomalous, errors_pinn, errors_standard, threshold_pinn, threshold_standard, window_size, anomaly_idxs, pinn_weight, filename="physics_anomaly_score_comparison.png"):
    """
    Compares the Anomaly Score (Reconstruction Error) of the PINN model
    vs. the Standard Autoencoder model using Matplotlib.
    """
    N_signal = len(x_anomalous)
    N_windows = len(errors_pinn)
    padding_length = window_size - 1
    time_steps = np.arange(N_signal)

    def pad_errors(errors):
        errors_padded = np.full(N_signal, np.nan)
        errors_padded[padding_length:padding_length + N_windows] = errors
        return errors_padded

    errors_pinn_padded = pad_errors(errors_pinn)
    errors_standard_padded = pad_errors(errors_standard)
    
    detected_idxs_pinn = np.where(errors_pinn_padded > threshold_pinn)[0]
    detected_idxs_standard = np.where(errors_standard_padded > threshold_standard)[0]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), sharex=True)
    
    # Subplot 1: Anomalous Signal
    ax1 = axes[0]
    ax1.plot(time_steps, x_anomalous, label='Anomalous Signal (Scaled)', color='#1F77B4', linewidth=1)
    if anomaly_idxs is not None:
        ax1.plot(anomaly_idxs, x_anomalous[anomaly_idxs], 'rx', markersize=8, label='True Anomalies', markeredgewidth=2)
    ax1.set_title('Anomalous Signal and True Anomaly Locations')
    ax1.set_ylabel('Amplitude (Scaled)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Subplot 2: PINN Anomaly Score
    ax2 = axes[1]
    ax2.plot(time_steps, errors_pinn_padded, label='PINN Anomaly Score', color='#2ECC71', linewidth=1)
    ax2.axhline(threshold_pinn, color='#2ECC71', linestyle='--', linewidth=2, label=f'PINN Threshold ({threshold_pinn:.4f})')
    ax2.plot(detected_idxs_pinn, errors_pinn_padded[detected_idxs_pinn], 'go', markersize=4, alpha=0.7)
    ax2.set_title(f'Anomaly Score: Physics-Informed Autoencoder (Weight={pinn_weight:.2f})')
    ax2.set_ylabel('Error Magnitude (MSE)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Subplot 3: Standard AE Anomaly Score
    ax3 = axes[2]
    ax3.plot(time_steps, errors_standard_padded, label='Standard AE Anomaly Score', color='#E74C3C', linewidth=1)
    ax3.axhline(threshold_standard, color='#E74C3C', linestyle='--', linewidth=2, label=f'Standard AE Threshold ({threshold_standard:.4f})')
    ax3.plot(detected_idxs_standard, errors_standard_padded[detected_idxs_standard], 'ro', markersize=4, alpha=0.7)
    ax3.set_title('Anomaly Score: Standard Autoencoder (Weight=0.0)')
    ax3.set_xlabel('Time Step Index')
    ax3.set_ylabel('Error Magnitude (MSE)')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle("Comparison of Anomaly Scores (PINN vs. Standard Autoencoder)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")


def plot_reconstruction_comparison(x_anomalous, recon_pinn_windows, recon_standard_windows, window_size, anomaly_idxs, filename="reconstruction_comparison.png"):
    """
    Plots the original signal and the reconstructed signals from both models.
    """
    N_signal = len(x_anomalous)
    time_steps = np.arange(N_signal)

    # Get full signal reconstructions
    recon_pinn_full = _get_full_reconstruction(recon_pinn_windows, N_signal, window_size)
    recon_standard_full = _get_full_reconstruction(recon_standard_windows, N_signal, window_size)

    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Original Signal
    ax.plot(time_steps, x_anomalous, label='Anomalous Signal (Scaled)', color='gray', linewidth=1.5, alpha=0.5)
    
    # PINN Reconstruction
    ax.plot(time_steps, recon_pinn_full, label='PINN Reconstruction', color='b', linestyle='-', linewidth=2, alpha=0.9)
    
    # Standard AE Reconstruction
    ax.plot(time_steps, recon_standard_full, label='Standard AE Reconstruction', color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Highlight true anomalies
    if anomaly_idxs is not None:
        ax.plot(anomaly_idxs, x_anomalous[anomaly_idxs], 'rx', markersize=8, label='True Anomalies', markeredgewidth=2)
        
    ax.set_title('Reconstruction Comparison: PINN vs. Standard Autoencoder')
    ax.set_xlabel('Time Step Index', fontsize=14)
    ax.set_ylabel('Amplitude (Scaled)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")

def plot_history(history, save_path="results/loss_curves.png"):

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(history['mse_loss'], label='MSE Loss')
    ax.plot(history['physics_loss'], label='Physics Loss')
    ax.plot(history['total_loss'], label='Total Loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True)
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(save_path)

    return fig


import matplotlib.pyplot as plt
import numpy as np
import os

def get_detected_true_anomalies(anomaly_idxs, model_detections, window_size):
    """
    Checks which true anomaly indices were covered by the model's detections.

    Args:
        anomaly_idxs (np.ndarray): Array of time indices where true anomalies exist (Red 'X's).
        model_detections (np.ndarray): Array of time indices of detected anomaly windows (e.g., Green Circles).
        window_size (int): The size of the sliding window used by the autoencoder.

    Returns:
        tuple: (detected_anomaly_idxs, missed_anomaly_idxs)
    """
    detected_idxs = []
    missed_idxs = []
    
    # 1. Group the anomaly_idxs into unique 'events' (clusters of adjacent 'X's)
    # Since anomaly_idxs are usually time steps, we can define an event as a cluster.
    # For simplicity, we'll treat each unique index in anomaly_idxs as a potential event
    # and check if it was covered.
    
    # In a typical time-series setting, 'True Anomalies' are defined by a *range*
    # (e.g., a window of 30 steps). For this code, we assume the true anomaly
    # is covered if a detection occurs within a small proximity (defined by window_size).
    
    # We will use a simple mapping: a true anomaly at index 'i' is detected if
    # the model flags *any* time step 'j' that is near 'i'.
    # Since the anomaly score is computed for a *window*, we check for simple overlap.
    
    # Define a set of all time steps flagged by the model for quick lookup
    model_detection_set = set(model_detections)
    
    # Iterate over the true anomaly indices
    for idx in anomaly_idxs:
        # Check a small window around the true anomaly index
        # A true anomaly at idx is detected if any of the model_detections is within 
        # the window corresponding to the anomaly score
        is_detected = False
        
        # We need to consider the window_size logic. If a model detects a window
        # centered at 'c', it covers the range [c - W/2, c + W/2].
        # A simple check: if the true anomaly index 'idx' is within the range
        # covered by a detected window, it's a TP.
        
        # Given how 'model_detections' are calculated (window centers), let's check
        # if the true anomaly index 'idx' is in the set of all time steps covered 
        # by the detection windows.
        
        # The easiest approach is to check if any model detection is "close enough"
        # to the true anomaly index. Let's use the window size as a tolerance.
        
        # A more robust check: An anomaly at 'idx' is detected if any model_detection 
        # index 'd' is close to 'idx'.
        if np.any(np.abs(model_detections - idx) <= window_size // 2):
            is_detected = True
        
        if is_detected:
            detected_idxs.append(idx)
        else:
            missed_idxs.append(idx)
            
    return np.array(detected_idxs), np.array(missed_idxs)


def plot_detected_anomalies_comparison(x_anomalous, errors_pinn, errors_standard, 
                                       threshold_pinn, threshold_standard, 
                                       window_size, anomaly_idxs, 
                                       filename="results/detected_anomalies_comparison_TP.png"):
    """
    Plots the anomalous signal with detected anomalies overlaid from both models,
    including specific markers for True Positives (TPs).
    """
    N_signal = len(x_anomalous)
    time_steps = np.arange(N_signal)
    
    # Calculate window centers
    # Note: len(errors) is N_signal - window_size + 1 (the number of windows)
    window_centers = np.arange(window_size // 2, len(errors_pinn) + window_size // 2)
    
    # Get all window centers that were flagged as anomalous
    pinn_detections_centers = window_centers[errors_pinn > threshold_pinn]
    standard_detections_centers = window_centers[errors_standard > threshold_standard]
    
    # ------------------------------------------------------------------
    # NEW LOGIC: Identify which True Anomalies ('X's) were detected (TP)
    # ------------------------------------------------------------------
    
    # PINN TP
    pinn_tp_idxs, _ = get_detected_true_anomalies(
        anomaly_idxs, pinn_detections_centers, window_size
    )
    # Standard AE TP
    standard_tp_idxs, _ = get_detected_true_anomalies(
        anomaly_idxs, standard_detections_centers, window_size
    )

    # Convert to sets to find unique TPs (where one model caught it and the other didn't)
    pinn_tp_set = set(pinn_tp_idxs)
    standard_tp_set = set(standard_tp_idxs)
    
    # Anomalies detected ONLY by PINN (The one extra TP)
    pinn_only_tp_idxs = np.array(list(pinn_tp_set - standard_tp_set))
    # Anomalies detected ONLY by Standard AE (If there were any)
    standard_only_tp_idxs = np.array(list(standard_tp_set - pinn_tp_set))
    # Anomalies detected by BOTH
    both_tp_idxs = np.array(list(pinn_tp_set.intersection(standard_tp_set)))
    
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot anomalous signal
    ax.plot(time_steps, x_anomalous, label='Anomalous Signal (Scaled)', 
            color='#1F77B4', linewidth=1.5, alpha=0.8)
    
    # 1. Plot all True Anomalies (Missed + Detected) as the base 'X'
    if anomaly_idxs is not None:
        ax.scatter(anomaly_idxs, x_anomalous[anomaly_idxs], 
                  color='red', s=100, marker='X', 
                  label='True Anomalies (Ground Truth)', zorder=5, edgecolors='black', linewidths=1.5)
    
    # 2. Plot PINN detections (Green Circles)
    pinn_y_vals = x_anomalous[pinn_detections_centers]
    ax.scatter(pinn_detections_centers, pinn_y_vals, 
              color='#2ECC71', s=80, marker='o', alpha=0.6,
              label=f'PINN Detections (FP + TP Sprawl, n={len(pinn_detections_centers)})', 
              zorder=4, edgecolors='darkgreen', linewidths=1)
    
    # 3. Plot Standard AE detections (Red Squares)
    standard_y_vals = x_anomalous[standard_detections_centers]
    ax.scatter(standard_detections_centers, standard_y_vals, 
              color='#E74C3C', s=60, marker='s', alpha=0.6,
              label=f'Standard AE Detections (FP + TP Sprawl, n={len(standard_detections_centers)})', 
              zorder=3, edgecolors='darkred', linewidths=1)
              
    # ------------------------------------------------------------------
    # 4. Plot True Positives (TP) markers on top of the 'X's
    # ------------------------------------------------------------------
    
    # Anomalies detected ONLY by PINN (The one that accounts for 28 vs 27)
    if len(pinn_only_tp_idxs) > 0:
        ax.scatter(pinn_only_tp_idxs, x_anomalous[pinn_only_tp_idxs], 
                   color='darkgreen', s=200, marker='*', 
                   label=f'TP - PINN ONLY (n={len(pinn_only_tp_idxs)})', 
                   zorder=6, edgecolors='white', linewidths=1)

    # Anomalies detected ONLY by Standard AE
    if len(standard_only_tp_idxs) > 0:
        ax.scatter(standard_only_tp_idxs, x_anomalous[standard_only_tp_idxs], 
                   color='darkred', s=200, marker='^', 
                   label=f'TP - Standard AE ONLY (n={len(standard_only_tp_idxs)})', 
                   zorder=6, edgecolors='white', linewidths=1)

    # Anomalies detected by BOTH (Majority)
    if len(both_tp_idxs) > 0:
        ax.scatter(both_tp_idxs, x_anomalous[both_tp_idxs], 
                   color='purple', s=200, marker='P', 
                   label=f'TP - BOTH Models (n={len(both_tp_idxs)})', 
                   zorder=6, edgecolors='white', linewidths=1)


    # Update the title and labels for clarity
    ax.set_title('Detected Anomalies: PINN vs. Standard Autoencoder (with TP Highlighting)', fontsize=14)
    ax.set_xlabel('Time Step Index', fontsize=14)
    ax.set_ylabel('Amplitude (Scaled)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)

    ax.tick_params(axis='both', which='major', labelsize=12)
    #ax.tick_params(axis='both', which='minor', labelsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Plot saved to {filename}")