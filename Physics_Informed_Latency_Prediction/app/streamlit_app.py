import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Physics vs Data-Driven Analysis", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .key-insight {
        background-color: #f0f8ff;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"> Physics vs Data-Driven Network Latency Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Understanding Why Domain Knowledge Matters in Machine Learning</div>', unsafe_allow_html=True)

# Physics constants
FIBER_SPEED = 2e8  # speed of light in optical fiber (m/s)
TRUE_PHYSICS_SLOPE = 1000 / FIBER_SPEED * 1000  # ms/km = 0.005

@st.cache_data
def load_and_validate_data():
    """Load and validate the simulation data with proper error handling"""
    try:
        train_data = pd.read_csv('enahnced_simulation_train_data.dat')
        test_data = pd.read_csv('enahnced_simulation_test_data.dat')
        
        # Validate required columns
        required_cols = ['geo_distance_km', 'measured_latency_ms', 'physics_latency_ms', 'is_anomaly']
        for col in required_cols:
            if col not in train_data.columns or col not in test_data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return train_data, test_data
    except FileNotFoundError:
        st.error("Data files not found. Please ensure 'enahnced_simulation_train_data.dat' and 'enahnced_simulation_test_data.dat' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

def evaluate_anomaly_detection_simple(residuals, y_true, threshold_sigma=2.5):
    """Simple anomaly detection function that matches the expected return format"""
    # Normalize residuals
    normalized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    y_pred = np.abs(normalized_residuals) > threshold_sigma
    
    # Calculate metrics
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_detected': np.sum(y_pred),
        'predictions': y_pred
    }

# Problem Statement
st.markdown("---")
st.markdown("## The Problem: When Data-Driven Models Learn the Wrong Thing")

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
    **Network latency prediction** is critical for:
    - **Content Delivery Networks (CDNs)**: Routing traffic optimally
    - **Cloud Computing**: Selecting the best data centers
    - **Online Gaming**: Ensuring smooth multiplayer experiences
    - **Financial Trading**: Minimizing execution delays
    
    **The Challenge**: What happens when your training data is biased? 
    
    In this case study, we explore a common scenario where **training data contains only short-distance connections** 
    (< 2000km), but the model must predict latency for **worldwide connections** (up to 20,000km).
    """)

with col2:
    # Create a simple diagram
    fig = go.Figure()
    
    # Add training data points (short distances)
    fig.add_trace(go.Scatter(
        x=[500, 800, 1200, 1500, 1800],
        y=[2.5, 4, 6, 7.5, 9],
        mode='markers',
        marker=dict(size=12, color='blue'),
        name='Training Data (Short)',
        hovertemplate='Distance: %{x}km<br>Latency: %{y}ms'
    ))
    
    # Add test data points (long distances)
    fig.add_trace(go.Scatter(
        x=[5000, 8000, 12000, 15000, 18000],
        y=[25, 40, 60, 75, 90],
        mode='markers',
        marker=dict(size=12, color='red'),
        name='Test Data (Long)',
        hovertemplate='Distance: %{x}km<br>Latency: %{y}ms'
    ))
    
    fig.update_layout(
        title="Training vs Test Data Coverage",
        xaxis_title="Distance (km)",
        yaxis_title="Latency (ms)",
        height=300,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Load data
train_data, test_data = load_and_validate_data()

# Sidebar controls
st.sidebar.markdown("## Analysis Controls")
threshold = st.sidebar.slider("Anomaly Detection Threshold (σ)", 1.0, 4.0, 2.5, 0.1)
show_cv = st.sidebar.checkbox("Show Cross-Validation Results", True)
show_detailed = st.sidebar.checkbox("Show Detailed Analysis", False)

# Data Overview with insights
st.markdown("---")
st.markdown("## Dataset Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Training Samples",
        value=f"{len(train_data):,}",
        help="All training data is from short distances only"
    )
    st.metric(
        label="Distance Range (Training)",
        value=f"{train_data['geo_distance_km'].min():.0f} - {train_data['geo_distance_km'].max():.0f} km",
        help="Limited to short-distance connections"
    )

with col2:
    st.metric(
        label="Test Samples", 
        value=f"{len(test_data):,}",
        help="Test data includes worldwide connections"
    )
    st.metric(
        label="Distance Range (Test)",
        value=f"{test_data['geo_distance_km'].min():.0f} - {test_data['geo_distance_km'].max():.0f} km",
        help="Full range including long distances"
    )

with col3:
    st.metric(
        label="Training Anomalies",
        value=f"{train_data['is_anomaly'].sum():,} ({train_data['is_anomaly'].mean()*100:.1f}%)"
    )
    st.metric(
        label="Test Anomalies",
        value=f"{test_data['is_anomaly'].sum():,} ({test_data['is_anomaly'].mean()*100:.1f}%)"
    )

# Key insight box
st.markdown("""
<div class="key-insight">
<strong>🔍 Key Insight:</strong> The training data covers only 10% of the distance range we need to predict on. 
This is a classic example of <strong>distribution shift</strong> - a major challenge in real-world ML deployments.
</div>
""", unsafe_allow_html=True)

# Model Training and Comparison
st.markdown("---")
st.markdown("## Model Training & Comparison")

# Prepare data with proper train/validation split
X_train_full = train_data[['geo_distance_km']].values
y_train_full = train_data['measured_latency_ms'].values

# Create train/validation split for cross-validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

X_test = test_data[['geo_distance_km']].values
y_test = test_data['measured_latency_ms'].values
y_true_anomalies = test_data['is_anomaly'].values

# APPROACH 1: Physics-Informed (no training needed)
physics_predictions_test = test_data['physics_latency_ms'].values
physics_mse = mean_squared_error(y_test, physics_predictions_test)
physics_r2 = r2_score(y_test, physics_predictions_test)
physics_residuals = y_test - physics_predictions_test

# APPROACH 2: Data-Driven with Cross-Validation
data_model = LinearRegression()

# Perform cross-validation on training data
if show_cv:
    cv_scores = cross_val_score(data_model, X_train_full, y_train_full, cv=5, scoring='neg_mean_squared_error')
    cv_r2_scores = cross_val_score(data_model, X_train_full, y_train_full, cv=5, scoring='r2')

# Train on full training set
data_model.fit(X_train_full, y_train_full)
data_predictions_test = data_model.predict(X_test)
data_mse = mean_squared_error(y_test, data_predictions_test)
data_r2 = r2_score(y_test, data_predictions_test)
data_residuals = y_test - data_predictions_test

learned_slope = data_model.coef_[0]
learned_intercept = data_model.intercept_

# Display results
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Physics-Informed Approach")
    st.markdown(f"""
    **Methodology**: Uses fundamental physics equation
    ```
    Latency = Distance × (1/Speed_of_Light_in_Fiber)
    Slope = {TRUE_PHYSICS_SLOPE:.6f} ms/km
    ```
    
    **Test Performance**:
    - MSE: {physics_mse:.2f}
    - R²: {physics_r2:.3f}
    
    **Advantages**: No training needed, theoretically sound, robust to data bias
    """)

with col2:
    st.markdown("### Data-Driven Approach")
    st.markdown(f"""
    **Methodology**: Linear regression trained on available data
    ```
    Learned Slope = {learned_slope:.6f} ms/km
    Learned Intercept = {learned_intercept:.2f} ms
    ```
    
    **Test Performance**:
    - MSE: {data_mse:.2f}
    - R²: {data_r2:.3f}
    
    {f"**Cross-Validation** (5-fold):" if show_cv else ""}
    {f"- CV MSE: {-cv_scores.mean():.2f} (±{cv_scores.std():.2f})" if show_cv else ""}
    {f"- CV R²: {cv_r2_scores.mean():.3f} (±{cv_r2_scores.std():.3f})" if show_cv else ""}
    
    **Risk**: Learned from biased training data
    """)

# Bias Analysis - The Core Message
st.markdown("---")
st.markdown("## The Learning Bias Problem")

slope_error = abs(learned_slope - TRUE_PHYSICS_SLOPE)
slope_error_pct = (slope_error / TRUE_PHYSICS_SLOPE) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label="True Physics Slope", 
        value=f"{TRUE_PHYSICS_SLOPE:.6f} ms/km",
        help="Based on speed of light in optical fiber"
    )
with col2:
    st.metric(
        label="Data-Driven Slope", 
        value=f"{learned_slope:.6f} ms/km",
        delta=f"{slope_error:.6f} ms/km",
        delta_color="inverse"
    )
with col3:
    st.metric(
        label="Slope Error", 
        value=f"{slope_error_pct:.1f}%",
        help="Percentage difference from true physics"
    )

# Explain the bias
if learned_slope > TRUE_PHYSICS_SLOPE:
    bias_explanation = """
    <div class="warning-box">
    <strong> OVERESTIMATION BIAS DETECTED</strong><br>
    The data-driven model <strong>overestimates</strong> latency per kilometer because:
    <ul>
    <li><strong>Short routes have higher overhead per km</strong>: Connection setup, routing decisions, processing delays</li>
    <li><strong>The model learned this as the "normal" rate</strong></li>
    <li><strong>Problem</strong>: Will severely underestimate long-distance latencies</li>
    </ul>
    </div>
    """
else:
    bias_explanation = """
    <div class="warning-box">
    <strong> UNDERESTIMATION BIAS DETECTED</strong><br>
    The data-driven model <strong>underestimates</strong> latency per kilometer.
    <strong>Problem</strong>: Will overestimate long-distance latencies.
    </div>
    """

st.markdown(bias_explanation, unsafe_allow_html=True)

# Visualization: The Core Story
st.markdown("---")
st.markdown("## Visualizing the Bias Problem")

# Create interactive visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Training vs Test Data Distribution', 
                   'Model Predictions: Short vs Long Distance',
                   'Prediction Accuracy by Distance Range',
                   'Residuals Analysis'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# Plot 1: Data Distribution
fig.add_trace(
    go.Histogram(x=train_data['geo_distance_km'], name='Training Data', 
                opacity=0.7, marker_color='blue', nbinsx=20),
    row=1, col=1
)
fig.add_trace(
    go.Histogram(x=test_data['geo_distance_km'], name='Test Data', 
                opacity=0.7, marker_color='orange', nbinsx=20),
    row=1, col=1
)

# Plot 2: Model Predictions
distances = np.linspace(0, 20000, 100)
true_physics_line = distances * TRUE_PHYSICS_SLOPE
learned_line = distances * learned_slope + learned_intercept

fig.add_trace(
    go.Scatter(x=distances, y=true_physics_line, name='True Physics', 
              line=dict(color='green', width=3), mode='lines'),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=distances, y=learned_line, name='Data-Driven', 
              line=dict(color='red', width=3, dash='dash'), mode='lines'),
    row=1, col=2
)

# Add training range highlight
fig.add_vrect(x0=0, x1=2000, fillcolor="blue", opacity=0.2, 
              annotation_text="Training Range", row=1, col=2)

# Plot 3: Error by Distance Range
distance_bins = [0, 2000, 5000, 10000, 20000]
bin_labels = ['Short<br>(<2km)', 'Medium<br>(2-5km)', 'Long<br>(5-10km)', 'Very Long<br>(>10km)']

physics_mse_by_dist = []
data_mse_by_dist = []

for i in range(len(distance_bins)-1):
    mask = (X_test.flatten() >= distance_bins[i]) & (X_test.flatten() < distance_bins[i+1])
    if np.sum(mask) > 0:
        physics_mse_by_dist.append(mean_squared_error(y_test[mask], physics_predictions_test[mask]))
        data_mse_by_dist.append(mean_squared_error(y_test[mask], data_predictions_test[mask]))
    else:
        physics_mse_by_dist.append(0)
        data_mse_by_dist.append(0)

fig.add_trace(
    go.Bar(x=bin_labels, y=physics_mse_by_dist, name='Physics MSE', 
           marker_color='green', opacity=0.7),
    row=2, col=1
)
fig.add_trace(
    go.Bar(x=bin_labels, y=data_mse_by_dist, name='Data-Driven MSE', 
           marker_color='red', opacity=0.7),
    row=2, col=1
)

# Plot 4: Residuals
fig.add_trace(
    go.Histogram(x=physics_residuals, name='Physics Residuals', 
                opacity=0.6, marker_color='green', nbinsx=30),
    row=2, col=2
)
fig.add_trace(
    go.Histogram(x=data_residuals, name='Data-Driven Residuals', 
                opacity=0.6, marker_color='red', nbinsx=30),
    row=2, col=2
)

fig.update_layout(height=800, showlegend=True, title_text="Comprehensive Model Analysis")
fig.update_xaxes(title_text="Distance (km)", row=1, col=1)
fig.update_xaxes(title_text="Distance (km)", row=1, col=2)
fig.update_xaxes(title_text="Distance Range", row=2, col=1)
fig.update_xaxes(title_text="Residuals (ms)", row=2, col=2)
fig.update_yaxes(title_text="Frequency", row=1, col=1)
fig.update_yaxes(title_text="Latency (ms)", row=1, col=2)
fig.update_yaxes(title_text="MSE", row=2, col=1)
fig.update_yaxes(title_text="Frequency", row=2, col=2)

st.plotly_chart(fig, use_container_width=True)

# Anomaly Detection Performance
st.markdown("---")
st.markdown("## Anomaly Detection Performance")

st.markdown("""
**Why This Matters**: In network monitoring, we need to detect when latency is unusually high, 
indicating potential issues like:
- Network congestion
- Hardware failures  
- Routing problems
- Security attacks
""")

# Perform anomaly detection
physics_ad = evaluate_anomaly_detection_simple(physics_residuals, y_true_anomalies, threshold)
data_ad = evaluate_anomaly_detection_simple(data_residuals, y_true_anomalies, threshold)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Physics-Informed Results")
    col1a, col1b, col1c = st.columns(3)
    with col1a:
        st.metric("Precision", f"{physics_ad['precision']:.3f}")
    with col1b:
        st.metric("Recall", f"{physics_ad['recall']:.3f}")
    with col1c:
        st.metric("F1-Score", f"{physics_ad['f1']:.3f}")
    st.write(f"Detected: {physics_ad['n_detected']} anomalies")

with col2:
    st.markdown("### Data-Driven Results")
    col2a, col2b, col2c = st.columns(3)
    with col2a:
        st.metric("Precision", f"{data_ad['precision']:.3f}")
    with col2b:
        st.metric("Recall", f"{data_ad['recall']:.3f}")
    with col2c:
        st.metric("F1-Score", f"{data_ad['f1']:.3f}")
    st.write(f"Detected: {data_ad['n_detected']} anomalies")

# Winner determination
best_f1 = max(physics_ad['f1'], data_ad['f1'])
winner = "Physics-Informed" if physics_ad['f1'] == best_f1 else "Data-Driven"

if winner == "Physics-Informed":
    st.markdown(f"""
    <div class="success-box">
    <strong> Winner: Physics-Informed</strong> (F1 = {best_f1:.3f})<br>
    Physics-based models provide more reliable anomaly detection because they use the correct baseline,
    making deviations more meaningful and easier to detect.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="success-box">
    <strong> Winner: Data-Driven</strong> (F1 = {best_f1:.3f})<br>
    The data-driven model performed better in this case, possibly due to learning domain-specific patterns.
    </div>
    """, unsafe_allow_html=True)

# Key Insights and Takeaways
st.markdown("---")
st.markdown("## Key Insights & Takeaways")

insights = [
    f"**Training Bias Impact**: {slope_error_pct:.1f}% error in learned slope due to biased training data",
    "**Physics Provides Robustness**: Domain knowledge creates reliable baselines regardless of data bias",
    "**Interpretability Matters**: Physics-based residuals have clear meaning (deviation from expected)",
    "**Hybrid Approach Recommended**: Combine physics baseline with data-driven corrections for production systems"
]

for i, insight in enumerate(insights, 1):
    st.markdown(f"**{i}.** {insight}")

# Recommendations
st.markdown("---")
st.markdown("## Recommendations for Production Systems")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### **Do This**
    - **Start with physics/domain knowledge** as baseline
    - **Use data to learn corrections** from the physics baseline
    - **Monitor for data drift** and distribution shifts
    - **Validate on diverse test sets** covering full operational range
    - **Implement ensemble methods** combining multiple approaches
    """)

with col2:
    st.markdown("""
    ### **Avoid This**
    - **Pure black-box models** without domain knowledge
    - **Training only on convenient/available data**
    - **Ignoring distribution shift** between training and deployment
    - **Over-trusting in-sample validation** metrics
    - **Deploying without interpretability** mechanisms
    """)

# Detailed Results (Optional)
if show_detailed:
    st.markdown("---")
    st.markdown("## Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Training Data Sample")
        st.dataframe(train_data.head(10))
        
        st.markdown("### Model Coefficients")
        st.write(f"**Linear Regression Results:**")
        st.write(f"- Slope: {learned_slope:.6f} ms/km")
        st.write(f"- Intercept: {learned_intercept:.2f} ms")
        st.write(f"- Training R²: {data_model.score(X_train_full, y_train_full):.3f}")
    
    with col2:
        st.markdown("### Test Data Sample")
        st.dataframe(test_data.head(10))
        
        st.markdown("### Performance Summary")
        results_df = pd.DataFrame({
            'Method': ['Physics-Informed', 'Data-Driven'],
            'Test MSE': [physics_mse, data_mse],
            'Test R²': [physics_r2, data_r2],
            'Anomaly F1': [physics_ad['f1'], data_ad['f1']]
        })
        st.dataframe(results_df)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
<strong>Conclusion:</strong> This analysis demonstrates why domain knowledge and physics-informed approaches 
are crucial in machine learning, especially when dealing with biased training data and distribution shift.
<br><br>
</div>
""", unsafe_allow_html=True)
