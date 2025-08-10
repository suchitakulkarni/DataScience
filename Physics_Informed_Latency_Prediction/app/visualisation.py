import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
def plot_heteroscedasticity_test(data):
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