import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
from anomaly_detection import evaluate_anomaly_detection
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Physics vs Data-Driven Analysis", layout="wide")

st.title("Physics vs Data-Driven Latency Prediction")
st.markdown("Comparing physics-informed and data-driven approaches for network latency prediction")

# Physics constants
FIBER_SPEED = 2e8  # speed of light in optical fiber (m/s)
TRUE_PHYSICS_SLOPE = 1000 / FIBER_SPEED * 1000  # ms/km = 0.005

# Generate or load data
train_data = pd.read_csv('enahnced_simulation_train_data.dat')
test_data = pd.read_csv('enahnced_simulation_test_data.dat')
y_true_anomalies = test_data['is_anomaly'].values

# Sidebar for controls
st.sidebar.header("Simulation Parameters")
threshold = st.sidebar.slider("Anomaly Detection Threshold (σ)", 1.0, 4.0, 2.5, 0.1)
show_details = st.sidebar.checkbox("Show Detailed Analysis", True)

# Main analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Overview")
    st.write(f"**Training data:** {len(train_data)} samples")
    st.write(f"Distance range: {train_data['geo_distance_km'].min():.0f} - {train_data['geo_distance_km'].max():.0f} km")
    st.write(f"Anomalies: {train_data['is_anomaly'].sum()}")
    
    st.write(f"**Test data:** {len(test_data)} samples")
    st.write(f"Distance range: {test_data['geo_distance_km'].min():.0f} - {test_data['geo_distance_km'].max():.0f} km")
    st.write(f"Anomalies: {test_data['is_anomaly'].sum()}")

with col2:
    st.subheader("Model Training")
    
    # Prepare data
    X_train = train_data[['geo_distance_km']].values
    y_train = train_data['measured_latency_ms'].values
    X_test = test_data[['geo_distance_km']].values
    y_test = test_data['measured_latency_ms'].values
    
    # Physics-informed predictions
    physics_predictions = test_data['physics_latency_ms'].values
    physics_mse = mean_squared_error(y_test, physics_predictions)
    physics_residuals = y_test - physics_predictions
    
    # Data-driven model
    data_model = LinearRegression()
    data_model.fit(X_train, y_train)
    data_predictions = data_model.predict(X_test)
    data_mse = mean_squared_error(y_test, data_predictions)
    data_residuals = y_test - data_predictions
    
    learned_slope = data_model.coef_[0]
    learned_intercept = data_model.intercept_
    
    st.write(f"**Physics-informed:** MSE = {physics_mse:.2f}")
    st.write(f"Uses slope: {TRUE_PHYSICS_SLOPE:.6f} ms/km")
    
    st.write(f"**Data-driven:** MSE = {data_mse:.2f}")
    st.write(f"Learned slope: {learned_slope:.6f} ms/km")
    st.write(f"Learned intercept: {learned_intercept:.2f} ms")

# Bias Analysis
st.subheader("Learning Bias Analysis")
slope_error = abs(learned_slope - TRUE_PHYSICS_SLOPE)
slope_error_pct = (slope_error / TRUE_PHYSICS_SLOPE) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("True Physics Slope", f"{TRUE_PHYSICS_SLOPE:.6f} ms/km")
with col2:
    st.metric("Data-Driven Slope", f"{learned_slope:.6f} ms/km")
with col3:
    st.metric("Slope Error", f"{slope_error_pct:.1f}%")

if learned_slope > TRUE_PHYSICS_SLOPE:
    st.warning("Data-driven model OVERESTIMATES latency per km due to short-distance training bias")
else:
    st.warning("Data-driven model UNDERESTIMATES latency per km")

# Anomaly Detection
st.subheader("Anomaly Detection Performance")

physics_anomaly_pred = evaluate_anomaly_detection(physics_residuals, y_true_anomalies, "Physics", threshold_sigma = threshold)
data_anomaly_pred = evaluate_anomaly_detection(data_residuals,y_true_anomalies, "Data-Driven",threshold_sigma = threshold)

y_true_anomalies = test_data['is_anomaly'].values

def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, np.sum(y_pred)

physics_precision, physics_recall, physics_detected = calculate_metrics(y_true_anomalies, physics_anomaly_pred)
data_precision, data_recall, data_detected = calculate_metrics(y_true_anomalies, data_anomaly_pred)

col1, col2 = st.columns(2)
with col1:
    st.write("**Physics-Informed Results:**")
    st.write(f"Precision: {physics_anomaly_pred['precision']:.3f}")
    st.write(f"Recall: {physics_anomaly_pred['recall']:.3f}")
    st.write(f"F1-Score: {physics_anomaly_pred['f1']:.3f}")
    st.write(f"Detected: {physics_anomaly_pred['n_detected']}")

with col2:
    st.write("**Data-Driven Results:**")
    st.write(f"Precision: {data_anomaly_pred['precision']:.3f}")
    st.write(f"Recall: {data_anomaly_pred['recall']:.3f}")
    st.write(f"F1-Score: {data_anomaly_pred['f1']:.3f}")
    st.write(f"Detected: {data_anomaly_pred['n_detected']}")

#best_f1 = max(physics_f1, data_f1)
#winner = "Physics-Informed" if physics_f1 == best_f1 else "Data-Driven"
#st.success(f" **Best Anomaly Detection:** {winner} (F1 = {best_f1:.3f})")

best_f1 = max(physics_anomaly_pred['f1'], data_anomaly_pred['f1'])
if physics_anomaly_pred['f1'] == best_f1:
    winner = "Physics-Informed"
else:
    winner = "Data-Driven"

st.write(f"\n BEST ANOMALY DETECTION: {winner} (F1 = {best_f1:.3f})")

# Visualizations
st.subheader(" Visualizations")

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Data Distribution
ax = axes[0, 0]
ax.hist(train_data['geo_distance_km'], bins=20, alpha=0.7, label='Training (Short)', color='blue')
ax.hist(test_data['geo_distance_km'], bins=20, alpha=0.7, label='Test (All)', color='orange')
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Frequency')
ax.set_title('Training vs Test Data Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Model Comparison
ax = axes[0, 1]
distances = np.linspace(0, 20000, 100)
true_physics_line = distances * TRUE_PHYSICS_SLOPE
learned_line = distances * learned_slope + learned_intercept

ax.plot(distances, true_physics_line, 'g-', linewidth=3, label=f'True Physics')
ax.plot(distances, learned_line, 'r--', linewidth=3, label=f'Data-Driven')
ax.axvspan(0, 2000, alpha=0.2, color='blue', label='Training Range')
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Predicted Latency (ms)')
ax.set_title('Learned vs True Physics Baseline')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Predictions vs Actual
ax = axes[1, 0]
normal_mask = test_data['is_anomaly'] == 0
anomaly_mask = test_data['is_anomaly'] == 1

ax.scatter(physics_predictions[normal_mask], y_test[normal_mask], alpha=0.5, s=10, 
          color='blue', label=f'Normal')
ax.scatter(physics_predictions[anomaly_mask], y_test[anomaly_mask], alpha=0.8, s=20,
          color='red', label=f'Anomalies')

min_val = min(np.min(physics_predictions), np.min(y_test))
max_val = max(np.max(physics_predictions), np.max(y_test))
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')

ax.set_xlabel('Physics Prediction (ms)')
ax.set_ylabel('Measured Latency (ms)')
ax.set_title('Physics-Informed: Predicted vs Actual')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Performance Comparison
ax = axes[1, 1]
methods = ['Physics', 'Data-Driven']
f1_scores = [physics_anomaly_pred['f1'], data_anomaly_pred['f1']]
mse_scores = [physics_mse, data_mse]

# Normalize MSE for comparison (lower is better)
mse_normalized = [(max(mse_scores) - mse) / max(mse_scores) for mse in mse_scores]

x_pos = np.arange(len(methods))
width = 0.35

ax.bar(x_pos - width/2, f1_scores, width, label='F1-Score', color='blue', alpha=0.7)
ax.bar(x_pos + width/2, mse_normalized, width, label='Accuracy (norm)', color='orange', alpha=0.7)

ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.set_ylabel('Score')
ax.set_title('Overall Performance Summary')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)

# Conclusions
st.subheader(" Key Insights")

insights = [
    f"Training data bias severely affects data-driven baselines (slope error: {slope_error_pct:.1f}%)",
    "Physics-informed provides robust baseline regardless of training bias",
    "Pure physics may not always win on accuracy, but wins on interpretability",
]

for insight in insights:
    st.write(f" {insight}")

if show_details:
    st.subheader(" Detailed Results")
    
    # Show raw data samples
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Data Sample:**")
        st.dataframe(train_data.head())
    
    with col2:
        st.write("**Test Data Sample:**")
        st.dataframe(test_data.head())
