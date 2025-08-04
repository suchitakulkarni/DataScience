#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from math import radians, sin, cos, sqrt, atan2
import warnings
warnings.filterwarnings('ignore')

def evaluate_anomaly_detection(residuals, true_anomalies, method_name, threshold_sigma=2.5):
    """Evaluate anomaly detection using residuals"""
    # Convert to z-scores
    z_scores = (residuals - np.mean(residuals)) / np.std(residuals)
    anomaly_scores = np.abs(z_scores)
    predictions = anomaly_scores > threshold_sigma
    
    if np.sum(predictions) == 0:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'n_detected': 0}
    
    precision = precision_score(true_anomalies, predictions)
    recall = recall_score(true_anomalies, predictions)
    f1 = f1_score(true_anomalies, predictions)
    
    return {
        'precision': precision,
        'recall': recall, 
        'f1': f1,
        'n_detected': np.sum(predictions),
        'predictions': predictions,
        'scores': anomaly_scores
    }
