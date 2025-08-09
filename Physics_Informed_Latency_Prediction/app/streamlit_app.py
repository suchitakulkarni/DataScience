#!/usr/bin/env python
"""
Streamlit App: Uncertainty Discovery in Latency Prediction

This app demonstrates the statistical tests and uncertainty estimation methods
from the blind uncertainty discovery code for latency prediction models.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

# Import the uncertainty estimation class
from src.clean_uncertainty_discovery import BlindUncertaintyEstimator, comprehensive_blind_analysis, \
    production_uncertainty_strategy

# Create results directory
os.makedirs('results', exist_ok=True)


def generate_synthetic_latency_data(n_samples=500, add_heteroscedasticity=True, add_nonlinearity=False, noise_level=2.0,
                                    seed=42):
    """Generate synthetic latency data similar to the original problem"""
    np.random.seed(seed)

    # Generate distances
    distances = np.random.uniform(100, 2000, n_samples)

    # Physics-based component (speed of light + routing overhead)
    physics_latency = distances * 0.005  # ~5ms per 1000km

    # Add measurement noise
    if add_heteroscedasticity:
        # Noise increases with distance
        noise_std = noise_level * (1 + distances / 5000)
    else:
        # Constant noise
        noise_std = np.full_like(distances, noise_level)

    measurement_noise = np.random.normal(0, noise_std)

    # Add non-linearity if requested
    if add_nonlinearity:
        nonlinear_component = 0.000001 * distances ** 2  # Quadratic term
    else:
        nonlinear_component = 0

    # Total latency
    measured_latency = physics_latency + nonlinear_component + measurement_noise + np.random.uniform(1, 3,
                                                                                                     n_samples)  # Base offset

    # Create DataFrame
    data = pd.DataFrame({
        'geo_distance_km': distances,
        'physics_latency_ms': physics_latency,
        'measured_latency_ms': measured_latency
    })

    return data


def plot_heteroscedasticity_test(data, estimator):
    """Visualize heteroscedasticity test results"""
    X = data[['geo_distance_km']].values
    y = data['measured_latency_ms'].values

    # Fit model and calculate residuals
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    residuals = y - predictions

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Scatter plot with residuals
    ax1.scatter(data['geo_distance_km'], data['measured_latency_ms'], alpha=0.6, label='Data')
    ax1.plot(data['geo_distance_km'], predictions, 'r-', label='Linear Fit')
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Data and Linear Fit')
    ax1.legend()

    # 2. Residuals vs Distance
    ax2.scatter(data['geo_distance_km'], residuals, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Residuals (ms)')
    ax2.set_title('Residuals vs Distance')

    # 3. Absolute residuals vs Distance
    abs_residuals = np.abs(residuals)
    ax3.scatter(data['geo_distance_km'], abs_residuals, alpha=0.6)
    # Fit line to show trend
    z = np.polyfit(data['geo_distance_km'], abs_residuals, 1)
    p = np.poly1d(z)
    ax3.plot(data['geo_distance_km'], p(data['geo_distance_km']), "r--", alpha=0.8)
    ax3.set_xlabel('Distance (km)')
    ax3.set_ylabel('|Residuals| (ms)')
    ax3.set_title('Absolute Residuals vs Distance')

    # 4. Histogram of residuals
    ax4.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Residuals (ms)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Residual Distribution')

    plt.tight_layout()
    plt.savefig('results/heteroscedasticity_analysis.png', dpi=150, bbox_inches='tight')
    return fig


def plot_uncertainty_comparison(results):
    """Compare different uncertainty estimation methods"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    X_test = results['test_data']['X'].flatten()
    y_test = results['test_data']['y']

    methods = ['adaptive_physics', 'extrapolation_data', 'conformal']
    method_names = ['Adaptive Physics', 'Extrapolation-Aware', 'Conformal Prediction']
    colors = ['blue', 'green', 'red']

    # Plot each method
    for i, (method, name, color) in enumerate(zip(methods, method_names, colors)):
        ax = [ax1, ax2, ax3][i]
        method_result = results['results'][method]

        # Sort by distance for better visualization
        sort_idx = np.argsort(X_test)
        X_sorted = X_test[sort_idx]
        pred_sorted = method_result['predictions'][sort_idx]
        lower_sorted = method_result['lower_bound'][sort_idx]
        upper_sorted = method_result['upper_bound'][sort_idx]
        y_sorted = y_test[sort_idx]

        # Plot uncertainty bands
        ax.fill_between(X_sorted, lower_sorted, upper_sorted, alpha=0.3, color=color, label=f'{name} CI')
        ax.plot(X_sorted, pred_sorted, color=color, linewidth=2, label=f'{name} Prediction')
        ax.scatter(X_test, y_test, alpha=0.6, s=20, color='black', label='True Values')

        ax.set_xlabel('Distance (km)')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(f'{name} Uncertainty')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Summary comparison
    coverage_data = []
    uncertainty_data = []
    method_labels = []

    for method, name in zip(methods, method_names):
        coverage_data.append(results['evaluation'][method]['coverage'])
        uncertainty_data.append(results['evaluation'][method]['avg_uncertainty'])
        method_labels.append(name)

    x_pos = np.arange(len(method_labels))

    ax4.bar(x_pos - 0.2, coverage_data, 0.4, label='Coverage', alpha=0.7)
    ax4_twin = ax4.twinx()
    ax4_twin.bar(x_pos + 0.2, uncertainty_data, 0.4, label='Avg Uncertainty', alpha=0.7, color='orange')

    ax4.set_xlabel('Method')
    ax4.set_ylabel('Coverage Rate')
    ax4_twin.set_ylabel('Average Uncertainty (ms)')
    ax4.set_title('Method Comparison')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(method_labels)
    ax4.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Target Coverage')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('results/uncertainty_comparison.png', dpi=150, bbox_inches='tight')
    return fig


def main():
    st.set_page_config(
        page_title="Uncertainty Discovery in Latency Prediction",
        layout="wide"
    )

    st.title("Uncertainty Discovery in Latency Prediction")
    st.markdown("---")

    # Sidebar for parameters
    st.sidebar.header("Data Generation Parameters")
    confidence_level = st.sidebar.slider("Confidence Level", 0.80, 0.99, 0.95)

    test_data = pd.read_csv('data/enahnced_simulation_test_data.dat')
    train_data = pd.read_csv('data/enahnced_simulation_train_data.dat')
    # Introduction
    st.header("Overview")
    st.markdown("""
    This app demonstrates **Blind Uncertainty Discovery** - techniques for estimating prediction uncertainty 
    when you don't know the underlying data generation process. This is the typical real-world scenario in machine learning.

    **Key Concepts:**
    - **Physics-Informed Uncertainty**: Uses domain knowledge (speed of light, routing) to constrain predictions
    - **Data-Driven Uncertainty**: Learns uncertainty patterns purely from observed data
    - **Statistical Tests**: Detect heteroscedasticity, non-linearity, and extrapolation risks
    """)

    # Data Overview
    st.header("Data Overview")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Data")
        st.write(f"**Samples:** {len(train_data)}")
        st.write(
            f"**Distance Range:** {train_data['geo_distance_km'].min():.0f} - {train_data['geo_distance_km'].max():.0f} km")
        st.write(f"**Mean Latency:** {train_data['measured_latency_ms'].mean():.1f} ms")

    with col2:
        st.subheader("Test Data")
        st.write(f"**Samples:** {len(test_data)}")
        st.write(
            f"**Distance Range:** {test_data['geo_distance_km'].min():.0f} - {test_data['geo_distance_km'].max():.0f} km")
        st.write(f"**Mean Latency:** {test_data['measured_latency_ms'].mean():.1f} ms")

    # Run comprehensive analysis
    estimator = BlindUncertaintyEstimator(confidence_level=confidence_level)

    # Pattern Discovery
    st.header("Pattern Discovery")
    st.markdown("The first step is to discover hidden patterns in the data without knowing the generation process.")

    patterns = estimator.discover_data_patterns(train_data, test_data)

    # Display pattern discovery results
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Heteroscedasticity Test")
        hetero_result = patterns['heteroscedasticity']
        st.write(f"**Detected:** {'Yes' if hetero_result['is_heteroscedastic'] else 'No'}")
        st.write(f"**Correlation with Distance:** {hetero_result['correlation_with_distance']:.3f}")
        st.write(f"**Correlation p-value:** {hetero_result['correlation_p_value']:.4f}")
        st.write(f"**Breusch-Pagan p-value:** {hetero_result['breusch_pagan_p_value']:.4f}")
        if hetero_result['levene_p_value'] is not None:
            st.write(f"**Levene p-value:** {hetero_result['levene_p_value']:.4f}")
        st.write(f"**Variance Ratio:** {hetero_result['variance_ratio']:.2f}")

        # Explain the test
        with st.expander("What is Heteroscedasticity?"):
            st.markdown("""
            **Heteroscedasticity** means the variance of residuals is not constant across different values of the predictor.

            **Tests Used:**
            1. **Correlation Test**: Correlation between distance and absolute residuals
            2. **Breusch-Pagan Test**: Regresses squared residuals on predictors
            3. **Levene Test**: Compares variance across distance bins

            **Why it Matters**: If present, uncertainty should increase with distance for better calibration.
            """)

    with col2:
        st.subheader("Linearity Test")
        linearity_result = patterns['linearity_test']
        st.write(f"**Non-linear:** {'Yes' if linearity_result['is_nonlinear'] else 'No'}")
        st.write(f"**Linear $R^2$:** {linearity_result['linear_r2']:.3f}")
        st.write(f"**Polynomial $R^2$:** {linearity_result['poly_r2']:.3f}")
        st.write(f"**Improvement:** {linearity_result['improvement']:.3f}")

        # Explain the test
        with st.expander("What is the Linearity Test?"):
            st.markdown("""
            **Linearity Test** compares linear vs polynomial models to detect non-linear relationships.

            **Method:**
            - Fit both linear and degree-2 polynomial models
            - Compare cross-validation R² scores
            - If polynomial improves R² by >5%, relationship is non-linear

            **Why it Matters**: Non-linearity adds model uncertainty, especially during extrapolation.
            """)

    # Visualize heteroscedasticity test
    st.subheader("Heteroscedasticity Analysis Visualization")
    fig_hetero = plot_heteroscedasticity_test(train_data, estimator)
    st.pyplot(fig_hetero)

    # Distance Effects Analysis
    if 'distance_effects' in patterns:
        st.subheader("Distance Effects Analysis")
        distance_effects = patterns['distance_effects']

        st.markdown("Analysis of how latency patterns change across distance ranges:")

        for range_name, range_data in distance_effects.items():
            with st.expander(f"Distance Range: {range_name}"):
                st.write(f"**Samples:** {range_data['n_samples']}")
                st.write(f"**Observed Slope:** {range_data['observed_slope']:.6f} ms/km")
                st.write(f"**Physics Slope:** {range_data['physics_slope']:.6f} ms/km")
                st.write(f"**Slope Ratio:** {range_data['slope_ratio']:.2f}")
                st.write(f"**R²:** {range_data['r_squared']:.3f}")

    # Comprehensive Analysis
    st.header("Uncertainty Method Comparison")
    st.markdown("Now we compare different uncertainty estimation methods:")

    with st.spinner("Running comprehensive analysis..."):
        results = comprehensive_blind_analysis(train_data, test_data, confidence_level)

    # Display results
    fig_comparison = plot_uncertainty_comparison(results)
    st.pyplot(fig_comparison)

    # Method Evaluation Table
    st.subheader("Method Evaluation")
    eval_df = pd.DataFrame(results['evaluation']).T
    eval_df['coverage_error'] = np.abs(eval_df['coverage'] - confidence_level)
    eval_df = eval_df.round(4)

    # Color code the coverage column
    def color_coverage(val):
        target = confidence_level
        if abs(val - target) < 0.02:  # Within 2%
            return 'background-color: lightgreen'
        elif abs(val - target) < 0.05:  # Within 5%
            return 'background-color: yellow'
        else:
            return 'background-color: lightcoral'

    st.dataframe(eval_df.style.applymap(color_coverage, subset=['coverage']))

    # Method Explanations
    st.header("Method Explanations")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Statistical Tests", "Adaptive Physics", "Extrapolation-Aware", "Conformal Prediction"])

    with tab1:
        st.markdown("""
        ### Statistical Tests for Pattern Discovery

        These tests help us understand the data characteristics without knowing the true generation process.
        """)

        st.subheader("1. Heteroscedasticity Tests")
        st.markdown(r"""
        **Purpose**: Detect if error variance changes with predictor values.

        **Correlation Test**:
        - Calculate correlation between distance $d_i$ and absolute residuals $|r_i|$
        - Test statistic: $\rho = \text{corr}(d_i, |r_i|)$
        - $H_0$: $\rho = 0$ (homoscedastic), $H_1$: $\rho \neq 0$ (heteroscedastic)

        **Breusch-Pagan Test**:
        1. Fit main model: $y_i = \beta_0 + \beta_1 d_i + \epsilon_i$
        2. Get squared residuals: $r_i^2$
        3. Regress: $r_i^2 = \gamma_0 + \gamma_1 d_i + u_i$
        4. Test statistic: $BP = n \cdot R^2$ ~ $\chi^2(1)$
        5. $H_0$: $\gamma_1 = 0$ (constant variance)

        **Levene Test**:
        1. Divide data into $k$ groups based on distance quantiles
        2. For each group $j$, calculate $Z_{ij} = |r_{ij} - \tilde{r}_j|$ where $\tilde{r}_j$ is group median
        3. Test statistic: $W = \frac{(n-k) \sum_{j=1}^k n_j (\bar{Z}_j - \bar{Z})^2}{(k-1) \sum_{j=1}^k \sum_{i=1}^{n_j} (Z_{ij} - \bar{Z}_j)^2}$ ~ $F(k-1, n-k)$

        **Decision Rule**: Evidence is heteroscedastic if ≥2 tests are significant at $\alpha$=0.05
        """)

        st.subheader("2. Linearity Test")
        st.markdown(r"""
        **Purpose**: Detect non-linear relationships that could affect extrapolation.

        **Method**:
        1. Fit linear model: $y_i = \beta_0 + \beta_1 d_i + \epsilon_i$
        2. Fit polynomial model: $y_i = \beta_0 + \beta_1 d_i + \beta_2 d_i^2 + \epsilon_i$
        3. Compare cross-validation $R^2$ scores: $\Delta R^2 = R^2_{poly} - R^2_{linear}$
        4. Decision: Non-linear if $\Delta R^2 > 0.05$

        **Why This Matters**: Non-linearity adds model uncertainty, especially during extrapolation where the linear approximation may break down.
        """)

        st.subheader("3. Distance Effects Analysis")
        st.markdown(r"""
        **Purpose**: Understand how latency-distance relationship varies across ranges.

        **Method**:
        1. Partition data into distance ranges: $[0,500), [500,1000), [1000,1500), [1500,2000)$
        2. For each range, fit: $y_i = \alpha_j + \beta_j d_i + \epsilon_i$
        3. Calculate slope ratio: $\rho_j = \frac{\beta_j}{\beta_{physics}}$ where $\beta_{physics} \approx 0.005$ ms/km
        4. Analyze variation: $\sigma_{slope} = \text{std}(\{\rho_j\})$

        **Interpretation**: Large $\sigma_{slope}$ indicates physics model may be inadequate across distance ranges.
        """)

    with tab1:
        # Show actual test results with equations
        st.subheader("Current Data Test Results")

        hetero_result = patterns['heteroscedasticity']
        st.markdown(f"""
        **Heteroscedasticity Results**:
        - Correlation: $\\rho = {hetero_result['correlation_with_distance']:.4f}$ (p = {hetero_result['correlation_p_value']:.4f})
        - Breusch-Pagan: $BP = {hetero_result['breusch_pagan_statistic']:.2f}$ (p = {hetero_result['breusch_pagan_p_value']:.4f})
        - Levene: $W = {hetero_result['levene_statistic']:.2f}$ (p = {hetero_result['levene_p_value']:.4f})
        - **Decision**: {'Heteroscedastic' if hetero_result['is_heteroscedastic'] else 'Homoscedastic'} (Evidence count: {hetero_result['evidence_count']}/3)
        """)

        linearity_result = patterns['linearity_test']
        st.markdown(f"""
        **Linearity Results**:
        - Linear $R^2$: ${linearity_result['linear_r2']:.4f}$
        - Polynomial $R^2$: ${linearity_result['poly_r2']:.4f}$
        - Improvement: $\\Delta R^2 = {linearity_result['improvement']:.4f}$
        - **Decision**: {'Non-linear' if linearity_result['is_nonlinear'] else 'Linear'} relationship detected
        """)

    with tab2:
        st.markdown("""
        ### Adaptive Physics Uncertainty

        **Concept**: Combines domain knowledge with data-driven adaptations.
        """)

        st.subheader("Mathematical Framework")
        st.markdown(r"""
        **Base Physics Model**:
        $\hat{y}_i = d_i \cdot \beta_{physics}$
        where $\beta_{physics} = 0.005$ ms/km (speed of light + routing overhead)

        **Total Uncertainty Decomposition**:
        $\sigma_{total}^2(d_i) = \sigma_{noise}^2(d_i) + \sigma_{physics}^2(d_i) + \sigma_{model}^2(d_i)$

        **1. Measurement Noise** (adapted from residual analysis):
        $\sigma_{noise}(d_i) = \begin{cases} 
        \sigma_{base} & \text{if homoscedastic} \\
        \sigma_{base} \cdot (1 + \frac{d_i}{10000} \cdot 0.5) & \text{if heteroscedastic}
        \end{cases}$
        where $\sigma_{base}$ is estimated from residual standard deviation.

        **2. Physics Parameter Uncertainty**:
        $\sigma_{physics}(d_i) = d_i \cdot \beta_{physics} \cdot \phi$
        where $\phi$ is the physics uncertainty percentage, starting at 5% and adapted based on slope variation:
        $\phi = 0.05 + \min(0.1 \cdot \sigma_{slope\_ratios}, 0.15)$

        **3. Model Uncertainty** (if non-linearity detected):
        $\sigma_{model}(d_i) = \begin{cases}
        0 & \text{if linear} \\
        d_i \cdot 0.0002 & \text{if non-linear detected}
        \end{cases}$

        **Confidence Interval**:
        $CI = \hat{y}_i \pm z_{\alpha/2} \cdot \sigma_{total}(d_i)$
        where $z_{\alpha/2}$ is the critical value for confidence level $\alpha$.
        """)

        # Show physics uncertainty components
        physics_result = results['results']['adaptive_physics']
        st.subheader("Uncertainty Components for Current Data")

        distances = results['test_data']['X'].flatten()
        components_df = pd.DataFrame({
            'Distance (km)': distances,
            'Noise $\sigma$_noise': physics_result['components']['noise'],
            'Physics $\sigma$_physics': physics_result['components']['physics'],
            'Model $\sigma$_model': physics_result['components']['model'],
            'Total $\sigma$_total': physics_result['uncertainty']
        })

        # Sample a few points for display
        sample_idx = np.linspace(0, len(distances) - 1, 10, dtype=int)
        st.dataframe(components_df.iloc[sample_idx].round(4))

        st.subheader("Key Insights")
        st.markdown(f"""
        - **Heteroscedasticity**: {'Applied distance-dependent scaling' if patterns['heteroscedasticity']['is_heteroscedastic'] else 'Using constant noise model'}
        - **Non-linearity**: {'Added model uncertainty component' if patterns['linearity_test']['is_nonlinear'] else 'No model uncertainty needed'}
        - **Physics Adaptation**: Uncertainty percentage adjusted based on distance effect variations
        """)

    with tab3:
        st.markdown("""
        ### Extrapolation-Aware Data Uncertainty

        **Concept**: Pure data-driven approach that automatically detects and penalizes risky predictions.
        """)

        st.subheader("Mathematical Framework")
        st.markdown(r"""
        **Base Uncertainty** (Bootstrap method):
        1. Generate $B$ bootstrap samples: $\{(\mathbf{X}^{(b)}, \mathbf{y}^{(b)})\}_{b=1}^B$
        2. Fit model on each: $\hat{f}^{(b)}(\mathbf{X}^{(b)}, \mathbf{y}^{(b)})$
        3. Predictions: $\hat{y}_i^{(b)} = \hat{f}^{(b)}(x_i)$
        4. Base uncertainty: $\sigma_{base}(x_i) = \text{std}(\{\hat{y}_i^{(b)}\}_{b=1}^B)$

        **Extrapolation Penalty**:
        $\sigma_{extrap}(x_i) = \begin{cases}
        0 & \text{if } x_{min}^{train} \leq x_i \leq x_{max}^{train} \\
        \frac{x_i - x_{max}^{train}}{1000} \cdot 2.0 & \text{if } x_i > x_{max}^{train} \\
        \frac{x_{min}^{train} - x_i}{1000} \cdot 2.0 & \text{if } x_i < x_{min}^{train}
        \end{cases}$

        **Model Complexity Penalty** (if non-linearity detected but using linear model):
        $\sigma_{complexity}(x_i) = \begin{cases}
        0 & \text{if linear relationship} \\
        \frac{|x_i - \bar{x}_{train}|}{5000} \cdot 1.0 & \text{if non-linear detected}
        \end{cases}$

        **Heteroscedasticity Adjustment**:
        $\text{adj}_{hetero}(x_i) = \begin{cases}
        1 & \text{if homoscedastic} \\
        1 + \frac{x_i}{10000} \cdot |\rho| & \text{if heteroscedastic}
        \end{cases}$
        where $\rho$ is the correlation between distance and absolute residuals.

        **Total Uncertainty**:
        $\sigma_{total}(x_i) = \sqrt{(\sigma_{base}(x_i) \cdot \text{adj}_{hetero}(x_i))^2 + \sigma_{extrap}^2(x_i) + \sigma_{complexity}^2(x_i)}$
        """)

        # Show extrapolation components
        extra_result = results['results']['extrapolation_data']
        st.subheader("Uncertainty Components for Current Data")

        components_df = pd.DataFrame({
            'Distance (km)': distances,
            'Base σ_base': extra_result['components']['base'],
            'Extrapolation σ_extrap': extra_result['components']['extrapolation'],
            'Complexity σ_complexity': extra_result['components']['complexity'],
            'Hetero Adj': extra_result['components']['heteroscedasticity'] + 1,  # Show as multiplier
            'Total σ_total': extra_result['uncertainty']
        })

        st.dataframe(components_df.iloc[sample_idx].round(4))

        # Extrapolation analysis
        train_max = patterns['train_distance_range'][1]
        test_max = patterns['test_distance_range'][1] if 'test_distance_range' in patterns else 0
        extrapolation_factor = patterns.get('distribution_shift', {}).get('extrapolation_factor', 1.0)

        st.subheader("Extrapolation Analysis")
        st.markdown(f"""
        - **Training Range**: {patterns['train_distance_range'][0]:.0f} - {train_max:.0f} km
        - **Test Range**: {patterns.get('test_distance_range', (0, 0))[0]:.0f} - {test_max:.0f} km
        - **Extrapolation Factor**: {extrapolation_factor:.2f}x beyond training
        - **Points Beyond Training**: {np.sum(distances > train_max)} / {len(distances)}
        """)

    with tab4:
        st.markdown("""
        ### Conformal Prediction

        **Concept**: Distribution-free method with mathematical guarantees for coverage probability.
        """)

        st.subheader("Mathematical Framework")
        st.markdown(r"""
        **Algorithm**:
        1. **Split Data**: $\mathcal{D} = \mathcal{D}_{train} \cup \mathcal{D}_{cal}$ where $|\mathcal{D}_{cal}| = n$

        2. **Train Model**: $\hat{f} = \text{fit}(\mathcal{D}_{train})$

        3. **Calibration**: For each $(x_i, y_i) \in \mathcal{D}_{cal}$:
           - Compute prediction: $\hat{y}_i = \hat{f}(x_i)$  
           - Compute conformity score: $R_i = |y_i - \hat{y}_i|$

        4. **Quantile Calculation**: 
           $\hat{q} = \text{Quantile}\left(\{R_i\}_{i=1}^n, \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$
           where $\alpha$ is the significance level.

        5. **Prediction Interval**: For new point $x_{new}$:
           $C(x_{new}) = [\hat{f}(x_{new}) - \hat{q}, \hat{f}(x_{new}) + \hat{q}]$

        **Coverage Guarantee**:
        $\mathbb{P}(Y_{new} \in C(X_{new})) \geq 1 - \alpha$

        This holds under the **exchangeability assumption**: $(X_1, Y_1), \ldots, (X_n, Y_n), (X_{new}, Y_{new})$ are exchangeable.

        **Key Properties**:
        - **Finite-sample validity**: Exact coverage for any finite sample size
        - **Distribution-free**: No assumptions about $P(Y|X)$ or model correctness
        - **Model-agnostic**: Works with any predictive model $\hat{f}$
        """)

        conformal_result = results['results']['conformal']
        st.subheader("Current Data Results")

        # Calculate some statistics about the calibration
        cal_size = int(0.3 * len(train_data))  # From train_test_split

        st.markdown(f"""
        **Calibration Setup**:
        - Calibration set size: $n = {cal_size}$
        - Significance level: $\\alpha = {1 - confidence_level:.3f}$
        - Target coverage: $1 - \\alpha = {confidence_level:.1%}$

        **Results**:
        - Quantile level: $\\frac{{\\lceil ({cal_size}+1) \\cdot {confidence_level:.3f} \\rceil}}{{{cal_size}}} = {((cal_size + 1) * confidence_level) / cal_size:.4f}$
        - Conformity score quantile: $\\hat{{q}} = {conformal_result['quantile_used']:.4f}$ ms
        - Prediction interval width: $2 \\times {conformal_result['quantile_used']:.4f} = {2 * conformal_result['quantile_used']:.4f}$ ms
        - Actual coverage: {results['evaluation']['conformal']['coverage']:.1%}
        """)

        st.subheader("Why Conformal Prediction Works")
        st.markdown(r"""
        **Intuition**: If the new point $(X_{new}, Y_{new})$ is "similar" to the calibration data, then its conformity score $R_{new} = |Y_{new} - \hat{f}(X_{new})|$ should be similar to the calibration conformity scores.

        **Mathematical Proof Sketch**:
        1. Under exchangeability, $R_{new}$ has the same distribution as calibration scores
        2. The probability that $R_{new}$ exceeds the $(1-\alpha)$-quantile of calibration scores is at most $\alpha$
        3. Therefore: $\mathbb{P}(R_{new} > \hat{q}) \leq \alpha$
        4. Which means: $\mathbb{P}(|Y_{new} - \hat{f}(X_{new})| \leq \hat{q}) \geq 1 - \alpha$

        **Practical Advantage**: Even if your model $\hat{f}$ is completely wrong, you still get valid coverage!
        """)

        # Show uniformity of conformal intervals
        st.subheader("Limitation: Uniform Uncertainty")
        uniform_width = conformal_result['quantile_used']
        st.markdown(f"""
        Conformal prediction gives the same uncertainty width ($\\pm {uniform_width:.3f}$ ms) everywhere, unlike adaptive methods that vary uncertainty based on:
        - Distance from training data
        - Local data density  
        - Detected heteroscedasticity

        **Trade-off**: Guaranteed coverage vs. adaptive uncertainty quantification
        """)

    # Production Strategy
    st.header("Production Strategy Recommendation")

    strategy_result = production_uncertainty_strategy(train_data, test_data, confidence_level)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Assessment")
        st.write(f"**Strategy**: {strategy_result['strategy']}")
        st.write(f"**Risk Score**: {strategy_result['risk_score']:.1f}/10")

        st.write("**Risk Factors Detected**:")
        for factor in strategy_result['risk_factors']:
            st.write(f"- {factor}")

    with col2:
        st.subheader("Recommendation")
        st.info(strategy_result['recommendation'])

        # Strategy-specific guidance
        if strategy_result['strategy'] == 'LOW_RISK':
            st.markdown("""
            **Implementation**:
            - Use bootstrap uncertainty estimation
            - Light calibration on holdout set
            - Monitor for distribution shift
            """)
        elif strategy_result['strategy'] == 'MEDIUM_RISK':
            st.markdown("""
            **Implementation**:
            - Use conformal prediction for guaranteed coverage
            - Add extrapolation penalties
            - Implement uncertainty-aware routing
            """)
        else:
            st.markdown("""
            **Implementation**:
            - Use conformal prediction + physics constraints
            - Ensemble multiple uncertainty methods
            - Conservative extrapolation handling
            - Extensive monitoring and recalibration
            """)

    # Key Takeaways
    st.header("Key Takeaways")
    st.markdown("""
    ### Statistical Discovery Without Ground Truth

    1. **Heteroscedasticity Testing** helps detect if uncertainty should vary with input features
    2. **Linearity Testing** reveals if simple models are sufficient or if model uncertainty is needed
    3. **Distance Effects Analysis** discovers how patterns change across feature ranges
    4. **Outlier Analysis** quantifies data quality and measurement noise levels

    ### Uncertainty Method Selection

    1. **Physics-Informed**: Best when you have strong domain knowledge and want interpretability
    2. **Data-Driven**: Most flexible, automatically adapts to data patterns
    3. **Conformal Prediction**: Best for guaranteed coverage, distribution-free guarantees

    ### Production Considerations

    - **Low Risk**: Standard methods work well
    - **Medium Risk**: Need extrapolation detection and conformal guarantees  
    - **High Risk**: Conservative ensemble approaches with extensive monitoring

    **Remember**: The goal is not just accurate predictions, but well-calibrated uncertainty estimates!
    """)


if __name__ == "__main__":
    main()