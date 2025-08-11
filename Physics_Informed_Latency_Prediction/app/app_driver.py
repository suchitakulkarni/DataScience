import streamlit as st
import uncertainty_demo
import final_summary, basic_demo
import pandas as pd
from src.uncertainty_discovery import BlindUncertaintyEstimator, comprehensive_blind_analysis, \
    production_uncertainty_strategy
from src.anomaly_detection import evaluate_anomaly_detection, uncertainty_weighted_anomaly_detection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score



# Physics constants
FIBER_SPEED = 2e8  # speed of light in optical fiber (m/s)
TRUE_PHYSICS_SLOPE = 1000 / FIBER_SPEED * 1000  # ms/km = 0.005

#st.set_page_config(
#    page_title="Uncertainty Discovery in Latency Prediction",
#    layout="wide"
#)

# Custom CSS for better styling
import streamlit as st
import uncertainty_demo
import final_summary, basic_demo
import pandas as pd
from src.uncertainty_discovery import BlindUncertaintyEstimator, comprehensive_blind_analysis, \
    production_uncertainty_strategy
from src.anomaly_detection import evaluate_anomaly_detection, uncertainty_weighted_anomaly_detection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score



# Physics constants
FIBER_SPEED = 2e8  # speed of light in optical fiber (m/s)
TRUE_PHYSICS_SLOPE = 1000 / FIBER_SPEED * 1000  # ms/km = 0.005

st.set_page_config(
    page_title="Uncertainty Discovery in Latency Prediction",
    layout="wide"
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
    
    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #f8f9fa;
        padding: 8px 16px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 56px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 8px;
        color: #262730;
        font-size: 16px;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e3f2fd;
        border-color: #1f77b4;
        transform: translateY(-1px);
    }
    
    .stTabs [aria-selected="true"]:hover {
        background-color: #1565c0;
        transform: translateY(-2px);
    }
    
    /* Tab content styling */
    .stTabs [data-baseweb="tab-panel"] {
        padding: 24px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for parameters
st.sidebar.markdown("## Analysis Controls")
confidence_level = st.sidebar.slider("Confidence Level", 0.80, 0.99, 0.95)
threshold = st.sidebar.slider("Anomaly Detection Threshold ($\\sigma$)", 1.0, 4.0, 2.5, 0.1)
show_detailed = st.sidebar.checkbox("Show Detailed Analysis", False)

test_data = pd.read_csv('data/enahnced_simulation_test_data.dat')
train_data = pd.read_csv('data/enahnced_simulation_train_data.dat')

# Prepare training and test data
X_train = train_data[['geo_distance_km']].values
y_train = train_data['measured_latency_ms'].values
X_test = test_data[['geo_distance_km']].values
y_test = test_data['measured_latency_ms'].values
y_true_anomalies = test_data['is_anomaly'].values


basic_demo_dict = {}
# APPROACH 1: PHYSICS-INFORMED WITH UNCERTAINTY
physics_predictions = test_data['physics_latency_ms'].values
physics_mse = mean_squared_error(y_test, physics_predictions)
physics_r2 = r2_score(y_test, physics_predictions)
physics_residuals = y_test - physics_predictions

# APPROACH 2: DATA-DRIVEN WITH UNCERTAINTY
data_model = LinearRegression()
data_model.fit(X_train, y_train)
data_predictions = data_model.predict(X_test)
data_mse = mean_squared_error(y_test, data_predictions)
data_r2 = r2_score(y_test, data_predictions)
data_residuals = y_test - data_predictions

learned_slope = data_model.coef_[0]
learned_intercept = data_model.intercept_

# Standard anomaly detection
physics_ad = evaluate_anomaly_detection(physics_residuals, y_true_anomalies)
data_ad = evaluate_anomaly_detection(data_residuals, y_true_anomalies)


estimator = BlindUncertaintyEstimator(confidence_level=confidence_level)
estimator.discover_data_patterns(train_data, test_data)
# Get physics uncertainty
physics_uncertainty = estimator.adaptive_physics_uncertainty(
    X_test.flatten(), base_physics_slope=TRUE_PHYSICS_SLOPE
)
physics_weighted_ad = uncertainty_weighted_anomaly_detection(
    physics_residuals, physics_uncertainty['uncertainty'], y_true_anomalies
)

# Get data-driven uncertainty (extrapolation-aware)
data_uncertainty = estimator.extrapolation_aware_data_uncertainty(
    X_train, y_train, X_test, method='bootstrap'
)

data_weighted_ad = uncertainty_weighted_anomaly_detection(
    data_residuals, data_uncertainty['uncertainty'], y_true_anomalies
)

basic_demo_dict['data_mse'] = data_mse
basic_demo_dict['data_r2'] = data_r2
basic_demo_dict['data_residuals'] = data_residuals
basic_demo_dict['data_predictions'] = data_predictions
basic_demo_dict['data_anomalies'] = data_ad
basic_demo_dict['data_slope'] = learned_slope
basic_demo_dict['data_intercept'] = learned_intercept
basic_demo_dict['data_uncertainty'] = data_uncertainty
basic_demo_dict['data_weighted_anomalies'] = data_weighted_ad

basic_demo_dict['physics_mse'] = physics_mse
basic_demo_dict['physics_r2'] = physics_r2
basic_demo_dict['physics_uncertainty'] = physics_uncertainty
basic_demo_dict['physics_residuals'] = physics_residuals
basic_demo_dict['physics_predictions'] = physics_predictions
basic_demo_dict['physics_anomalies'] = physics_ad
basic_demo_dict['physics_slope'] = TRUE_PHYSICS_SLOPE
basic_demo_dict['physics_weighted_anomalies'] = physics_weighted_ad

patterns = estimator.discover_data_patterns(train_data, test_data)
strategy_result = production_uncertainty_strategy(train_data, test_data, confidence_level)
results = comprehensive_blind_analysis(train_data, test_data, confidence_level)

tab1, tab2, tab3 = st.tabs(["main", "Uncertainty", "Summary"])
with tab1:
    st.header("Main demo")
    basic_demo.main_demo(basic_demo_dict, show_detailed = show_detailed)

with tab2:
    st.header("Details of uncertainty analysis")
    uncertainty_demo.uncertainty_details(train_data, test_data, confidence_level, patterns, results)

with tab3:
    st.header("Final summary")
    final_summary.display_summary(strategy_result)
