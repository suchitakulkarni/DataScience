import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import os

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluates the model and computes the reconstruction error and reconstructed windows.
    Returns: 
    - all_errors (1D numpy array of mean-per-window errors)
    - all_reconstructed_windows (3D numpy array, N x seq_len x 1)
    """
    model.eval()
    all_errors = []
    all_reconstructed_windows = []
    with torch.no_grad():
        for data in data_loader:
            data_window = data[0].to(device)
            reconstruction = model(data_window)
            
            loss_elementwise = criterion(reconstruction, data_window) 
            error_per_window = torch.mean(loss_elementwise, dim=[1, 2])
            
            all_errors.extend(error_per_window.cpu().numpy())
            all_reconstructed_windows.append(reconstruction.cpu().numpy())
            
    all_reconstructed_windows = np.concatenate(all_reconstructed_windows, axis=0)
    return np.array(all_errors), all_reconstructed_windows


def tune_threshold_f1(errors, true_anomaly_idxs, window_size, num_thresholds=100):
    thresholds = np.linspace(np.min(errors), np.max(errors), num_thresholds)
    results = []
    best_threshold = None
    best_f1 = -1
    best_metrics = {}

    window_centers = np.arange(window_size // 2, len(errors) + window_size // 2)

    for t in thresholds:
        preds = errors > t
        detected_idxs = window_centers[preds]

        # Match detected to true using proximity
        true_positives = [
            idx for idx in true_anomaly_idxs
            if any(abs(idx - d) <= window_size // 2 for d in detected_idxs)
        ]

        false_positives = [
            d for d in detected_idxs
            if not any(abs(d - idx) <= window_size // 2 for idx in true_anomaly_idxs)
        ]

        tp = len(true_positives)
        fp = len(false_positives)
        fn = len(true_anomaly_idxs) - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        results.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "detected_idxs": detected_idxs,
            "matched_true_idxs": true_positives # <--- NEW LINE
        })

        best = max(results, key=lambda x: x["f1"])

    return best, results