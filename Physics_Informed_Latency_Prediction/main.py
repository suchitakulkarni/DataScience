#!/usr/bin/env python
# coding: utf-8

"""
Simple Physics vs Data-Driven Comparison: Short Distance Bias

This simulation demonstrates physics-informed superiority by:
1. Training on SHORT DISTANCE data only (< 2000km)
2. Testing generalization to WORLDWIDE data (including long distances)
3. Showing data-driven models learn wrong baseline from biased training
4. Demonstrating physics-informed robustness to training bias
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from anomaly_detection import evaluate_anomaly_detection

import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Physics constants
FIBER_SPEED = 2e8  # speed of light in optical fiber (m/s)
TRUE_PHYSICS_SLOPE = 1000 / FIBER_SPEED * 1000  # ms/km = 0.005

train_data = pd.read_csv('data/enahnced_simulation_train_data.dat')
test_data = pd.read_csv('data/enahnced_simulation_test_data.dat')

print(f"Training data: {len(train_data)} samples")
print(f"  Distance range: {train_data['geo_distance_km'].min():.0f} - {train_data['geo_distance_km'].max():.0f} km")
print(f"  Anomalies: {train_data['is_anomaly'].sum()}")

print(f"Test data: {len(test_data)} samples") 
print(f"  Distance range: {test_data['geo_distance_km'].min():.0f} - {test_data['geo_distance_km'].max():.0f} km")
print(f"  Anomalies: {test_data['is_anomaly'].sum()}")
print()

# ============================================================================
# TRAIN MODELS AND COMPARE APPROACHES
# ============================================================================

print("TRAINING MODELS AND COMPARING APPROACHES")
print("=" * 45)

# Prepare training data
X_train = train_data[['geo_distance_km']].values
y_train = train_data['measured_latency_ms'].values

# Prepare test data  
X_test = test_data[['geo_distance_km']].values
y_test = test_data['measured_latency_ms'].values
y_true_anomalies = test_data['is_anomaly'].values

# APPROACH 1: PHYSICS-INFORMED (no training needed)
physics_predictions = test_data['physics_latency_ms'].values
physics_residuals = y_test - physics_predictions
physics_mse = mean_squared_error(y_test, physics_predictions)

print("1. PHYSICS-INFORMED APPROACH:")
print(f"   Uses: distance * {TRUE_PHYSICS_SLOPE:.6f} ms/km")
print(f"   Test MSE: {physics_mse:.2f}")
print(f"   No training needed - pure physics!")

# APPROACH 2: DATA-DRIVEN (trained on short distances only)
data_model = LinearRegression()
data_model.fit(X_train, y_train)
data_predictions = data_model.predict(X_test)
data_residuals = y_test - data_predictions
data_mse = mean_squared_error(y_test, data_predictions)

learned_slope = data_model.coef_[0]
learned_intercept = data_model.intercept_

print("\n2. DATA-DRIVEN APPROACH:")
print(f"   Learned slope: {learned_slope:.6f} ms/km")
print(f"   Learned intercept: {learned_intercept:.2f} ms")
print(f"   Test MSE: {data_mse:.2f}")

# ============================================================================
# ANALYZE LEARNING BIAS
# ============================================================================

print(f"\n" + "="*50)
print("BIAS ANALYSIS: WRONG BASELINE LEARNING")
print("="*50)

slope_error = abs(learned_slope - TRUE_PHYSICS_SLOPE)
slope_error_pct = (slope_error / TRUE_PHYSICS_SLOPE) * 100

print(f"TRUE PHYSICS SLOPE:    {TRUE_PHYSICS_SLOPE:.6f} ms/km")
print(f"DATA-DRIVEN SLOPE:     {learned_slope:.6f} ms/km") 
print(f"SLOPE ERROR:           {slope_error:.6f} ms/km ({slope_error_pct:.1f}%)")
print()

if learned_slope > TRUE_PHYSICS_SLOPE:
    print("DATA-DRIVEN OVERESTIMATES latency per km")
    print("   Reason: Short routes have more overhead per km")
    print("   Problem: Will underestimate long-distance latencies")
else:
    print("DATA-DRIVEN UNDERESTIMATES latency per km")
    print("   Problem: Will overestimate long-distance latencies")

# ============================================================================
# ANOMALY DETECTION COMPARISON
# ============================================================================

print(f"\n" + "="*50)
print("ANOMALY DETECTION PERFORMANCE")
print("="*50)

physics_ad = evaluate_anomaly_detection(physics_residuals, y_true_anomalies, "Physics")
data_ad = evaluate_anomaly_detection(data_residuals, y_true_anomalies, "Data-Driven")

print(f"{'Method':<15} {'Precision':<10} {'Recall':<8} {'F1-Score':<10} {'Detected':<10}")
print("-" * 55)
print(f"{'Physics':<15} {physics_ad['precision']:<10.3f} {physics_ad['recall']:<8.3f} {physics_ad['f1']:<10.3f} {physics_ad['n_detected']:<10}")
print(f"{'Data-Driven':<15} {data_ad['precision']:<10.3f} {data_ad['recall']:<8.3f} {data_ad['f1']:<10.3f} {data_ad['n_detected']:<10}")

best_f1 = max(physics_ad['f1'], data_ad['f1'])
if physics_ad['f1'] == best_f1:
    winner = "Physics-Informed"
else:
    winner = "Data-Driven"

print(f"\n BEST ANOMALY DETECTION: {winner} (F1 = {best_f1:.3f})")

# ============================================================================
# VISUALIZATION
# ============================================================================

print(f"\nCreating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Training vs Test Data Distribution
ax = axes[0, 0]
ax.hist(train_data['geo_distance_km'], bins=20, alpha=0.7, label='Training (Short)', color='blue')
ax.hist(test_data['geo_distance_km'], bins=20, alpha=0.7, label='Test (All)', color='orange')
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Frequency')
ax.set_title('Training vs Test Data Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Learned vs True Physics
ax = axes[0, 1]
distances = np.linspace(0, 20000, 100)
true_physics_line = distances * TRUE_PHYSICS_SLOPE
learned_line = distances * learned_slope + learned_intercept

ax.plot(distances, true_physics_line, 'g-', linewidth=3, label=f'True Physics ({TRUE_PHYSICS_SLOPE:.6f} ms/km)')
ax.plot(distances, learned_line, 'r--', linewidth=3, label=f'Data-Driven ({learned_slope:.6f} ms/km)')

# Mark training range
ax.axvspan(0, 2000, alpha=0.2, color='blue', label='Training Range')

ax.set_xlabel('Distance (km)')
ax.set_ylabel('Predicted Latency (ms)')
ax.set_title('Learned vs True Physics Baseline')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Prediction Accuracy by Distance
ax = axes[0, 2]
test_distances = test_data['geo_distance_km'].values

# Bin by distance for analysis
distance_bins = [0, 2000, 5000, 10000, 20000]
bin_labels = ['Short\n(<2km)', 'Medium\n(2-5km)', 'Long\n(5-10km)', 'Very Long\n(>10km)']

physics_mse_by_dist = []
data_mse_by_dist = []

for i in range(len(distance_bins)-1):
    mask = (test_distances >= distance_bins[i]) & (test_distances < distance_bins[i+1])
    if np.sum(mask) > 0:
        physics_mse_by_dist.append(mean_squared_error(y_test[mask], physics_predictions[mask]))
        data_mse_by_dist.append(mean_squared_error(y_test[mask], data_predictions[mask]))
    else:
        physics_mse_by_dist.append(0)
        data_mse_by_dist.append(0)

x_pos = np.arange(len(bin_labels))
width = 0.35

ax.bar(x_pos - width/2, physics_mse_by_dist, width, label='Physics', color='green', alpha=0.7)
ax.bar(x_pos + width/2, data_mse_by_dist, width, label='Data-Driven', color='red', alpha=0.7)

ax.set_xticks(x_pos)
ax.set_xticklabels(bin_labels)
ax.set_ylabel('MSE')
ax.set_title('Prediction Error by Distance Category')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Actual vs Predicted (Test Data)
ax = axes[1, 0]
normal_mask = test_data['is_anomaly'] == 0
anomaly_mask = test_data['is_anomaly'] == 1

ax.scatter(physics_predictions[normal_mask], y_test[normal_mask], alpha=0.6, s=20, 
          color='blue', label=f'Normal ({np.sum(normal_mask)})')
ax.scatter(physics_predictions[anomaly_mask], y_test[anomaly_mask], alpha=0.8, s=40,
          color='red', label=f'Anomalies ({np.sum(anomaly_mask)})')

# Perfect prediction line
min_val = min(np.min(physics_predictions), np.min(y_test))
max_val = max(np.max(physics_predictions), np.max(y_test))
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')

ax.set_xlabel('Physics Prediction (ms)')
ax.set_ylabel('Measured Latency (ms)')
ax.set_title('Physics-Informed: Predicted vs Actual')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Residuals Comparison
ax = axes[1, 1]
physics_resid_norm = (physics_residuals - np.mean(physics_residuals)) / np.std(physics_residuals)
data_resid_norm = (data_residuals - np.mean(data_residuals)) / np.std(data_residuals)

ax.hist(physics_resid_norm, bins=30, alpha=0.6, label='Physics-Informed', color='green', density=True)
ax.hist(data_resid_norm, bins=30, alpha=0.6, label='Data-Driven', color='red', density=True)

ax.axvline(-2.5, color='black', linestyle='--', alpha=0.7, label='±2.5σ threshold')
ax.axvline(2.5, color='black', linestyle='--', alpha=0.7)

ax.set_xlabel('Normalized Residuals')
ax.set_ylabel('Density')
ax.set_title('Residuals Distribution (Test Data)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Performance Summary
ax = axes[1, 2]
methods = ['Physics', 'Data-Driven']
f1_scores = [physics_ad['f1'], data_ad['f1']]
mse_scores = [physics_mse, data_mse]


# Normalize MSE for comparison (lower is better)
mse_normalized = [(max(mse_scores) - mse) / max(mse_scores) for mse in mse_scores]

x_pos = np.arange(len(methods))
width = 0.35

bars1 = ax.bar(x_pos - width/2, f1_scores, width, label='F1-Score', color='blue', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, mse_normalized, width, label='Accuracy (norm)', color='orange', alpha=0.7)

ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.set_ylabel('Score')
ax.set_title('Overall Performance Summary')
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels
for bar, score in zip(bars1, f1_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{score:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/simple_physics_vs_datadriven.pdf', dpi=300, bbox_inches='tight')
#plt.show()

# ============================================================================
# FINAL CONCLUSIONS
# ============================================================================

print(f"\n" + "="*60)
print("FINAL CONCLUSIONS")
print("="*60)

print(f"\n DEMONSTRATION SUCCESSFUL:")
print(f"   Training on short distances only leads to wrong baseline learning")
print(f"   Data-driven slope error: {slope_error_pct:.1f}%")

if physics_ad['f1'] >= data_ad['f1']:
    print(f"\n PHYSICS-INFORMED WINS:")
    print(f"   Better anomaly detection (F1: {physics_ad['f1']:.3f} vs {data_ad['f1']:.3f})")
    print(f"   No training bias dependency")
    print(f"   Interpretable residuals")
else:
    print(f"\n DATA-DRIVEN COMPETITIVE:")
    print(f"   Better prediction accuracy (MSE: {data_mse:.2f} vs {physics_mse:.2f})")
    print(f"   But learned wrong baseline from biased training")

print(f"\n KEY INSIGHTS:")
print(f"   1. Training data bias severely affects data-driven baselines")
print(f"   2. Physics-informed provides robust baseline regardless of training bias")
print(f"   3. Pure physics may not always win on accuracy, but wins on interpretability")

print(f"\n FILES SAVED:")
print(f"   results/simple_physics_vs_datadriven.pdf (visualization)")
