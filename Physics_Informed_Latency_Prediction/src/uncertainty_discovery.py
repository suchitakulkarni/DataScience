#!/usr/bin/env python
"""
Blind Uncertainty Discovery: Clean Version for Integration

This demonstrates how to discover uncertainty patterns and biases
when you DON'T know the data generation process - the typical
real-world machine learning scenario.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from scipy import stats
import warnings
import copy
warnings.filterwarnings('ignore')

class BlindUncertaintyEstimator:
    """
    Uncertainty estimation without knowing the data generation process.
    Uses only the observed data to discover patterns and biases.
    """
    
    def __init__(self, confidence_level=0.95):
        """
        Initialize the estimator with configurable confidence level.
        
        Parameters:
        -----------
        confidence_level : float, default=0.95
            Confidence level for uncertainty intervals (e.g., 0.95 for 95% CI)
            Common values: 0.68 (1σ), 0.95 (1.96σ), 0.99 (2.58σ)
        """
        self.confidence_level = confidence_level
        self.z_score = self._confidence_to_z_score(confidence_level)
        self.patterns_discovered = {}
        self.calibration_factors = {}
    
    def _confidence_to_z_score(self, confidence_level):
        """Convert confidence level to corresponding z-score"""
        from scipy.stats import norm
        alpha = 1 - confidence_level
        return norm.ppf(1 - alpha/2)
        
    def discover_data_patterns(self, train_data, test_data=None):
        """
        Exploratory Data Analysis to discover hidden patterns
        """
        patterns = {}
        
        # 1. Distance distribution analysis
        train_distances = train_data['geo_distance_km']
        patterns['train_distance_range'] = (train_distances.min(), train_distances.max())
        patterns['train_distance_mean'] = train_distances.mean()
        patterns['train_distance_std'] = train_distances.std()
        
        if test_data is not None:
            test_distances = test_data['geo_distance_km']
            patterns['test_distance_range'] = (test_distances.min(), test_distances.max())
            patterns['distribution_shift'] = {
                'train_max': train_distances.max(),
                'test_max': test_distances.max(),
                'extrapolation_factor': test_distances.max() / train_distances.max()
            }
        
        # 2. Residual analysis to detect heteroscedasticity
        X = train_data[['geo_distance_km']].values
        y = train_data['measured_latency_ms'].values
        
        # Fit simple model
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        residuals = y - predictions
        
        # Check for heteroscedasticity (non-constant variance)
        patterns['heteroscedasticity'] = self._test_heteroscedasticity(X.flatten(), residuals)
        
        # 3. Non-linearity detection
        patterns['linearity_test'] = self._test_linearity(X, y)
        
        # 4. Distance-dependent patterns
        patterns['distance_effects'] = self._analyze_distance_effects(train_data)
        
        # 5. Outlier/anomaly patterns
        patterns['outlier_analysis'] = self._analyze_outliers(train_data)
        
        self.patterns_discovered = patterns
        return patterns
    
    def _test_heteroscedasticity(self, distances, residuals):
        """Test if residual variance changes with distance using multiple methods"""
        from scipy.stats import pearsonr, levene
        
        # 1. Correlation test with significance
        abs_residuals = np.abs(residuals)
        correlation, correlation_p_value = pearsonr(distances, abs_residuals)
        
        # 2. Breusch-Pagan test approximation
        # Regress squared residuals on distances
        from sklearn.linear_model import LinearRegression
        squared_residuals = residuals**2
        bp_model = LinearRegression()
        bp_model.fit(distances.reshape(-1, 1), squared_residuals)
        bp_predictions = bp_model.predict(distances.reshape(-1, 1))
        bp_r2 = bp_model.score(distances.reshape(-1, 1), squared_residuals)
        
        # Chi-square test statistic approximation: n * R^2
        n = len(residuals)
        bp_statistic = n * bp_r2
        # Critical value for chi-square with 1 df at 1-CL=0.05 is ~3.84
        bp_p_value_approx = 1 - stats.chi2.cdf(bp_statistic, df=1)
        
        # 3. Levene test (more robust) - divide into groups
        n_bins = min(5, max(3, len(distances) // 20))  # Adaptive number of bins
        try:
            distance_bins = np.quantile(distances, np.linspace(0, 1, n_bins + 1))
            bin_groups = []
            bin_variances = []
            
            for i in range(n_bins):
                mask = (distances >= distance_bins[i]) & (distances <= distance_bins[i + 1])
                if np.sum(mask) > 2:  # Need at least 3 points per bin
                    group_residuals = residuals[mask]
                    bin_groups.append(group_residuals)
                    bin_variances.append(np.var(group_residuals))
            
            if len(bin_groups) >= 2:
                levene_statistic, levene_p_value = levene(*bin_groups, center='median')
                variance_ratio = max(bin_variances) / min(bin_variances) if bin_variances else 1.0
            else:
                levene_statistic, levene_p_value = None, None
                variance_ratio = 1.0
                
        except Exception:
            levene_statistic, levene_p_value = None, None
            variance_ratio = 1.0
        
        # 4. Decision logic with statistical significance
        # Use multiple criteria with p-values
        significance_level = 0.05
        
        # Evidence for heteroscedasticity
        correlation_significant = correlation_p_value < significance_level and abs(correlation) > 0.1
        bp_significant = bp_p_value_approx < significance_level
        levene_significant = levene_p_value is not None and levene_p_value < significance_level
        variance_ratio_high = variance_ratio > 2.0  # Keep as secondary indicator
        
        # Combine evidence
        evidence_count = sum([correlation_significant, bp_significant, levene_significant])
        is_heteroscedastic = evidence_count >= 2 or (evidence_count >= 1 and variance_ratio_high)
        
        return {
            'correlation_with_distance': correlation,
            'correlation_p_value': correlation_p_value,
            'correlation_significant': correlation_significant,
            'variance_ratio': variance_ratio,
            'breusch_pagan_statistic': bp_statistic,
            'breusch_pagan_p_value': bp_p_value_approx,
            'breusch_pagan_significant': bp_significant,
            'levene_statistic': levene_statistic,
            'levene_p_value': levene_p_value,
            'levene_significant': levene_significant,
            'evidence_count': evidence_count,
            'is_heteroscedastic': is_heteroscedastic,
            'method': 'statistical_tests'
        }
    
    def _test_linearity(self, X, y):
        """Test if relationship is truly linear"""
        # Compare linear vs polynomial models
        linear_model = LinearRegression()
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        poly_model = LinearRegression()
        
        # Cross-validation scores
        linear_scores = cross_val_score(linear_model, X, y, cv=5, scoring='r2')
        poly_scores = cross_val_score(poly_model, X_poly, y, cv=5, scoring='r2')
        
        improvement = poly_scores.mean() - linear_scores.mean()
        
        return {
            'linear_r2': linear_scores.mean(),
            'poly_r2': poly_scores.mean(),
            'improvement': improvement,
            'is_nonlinear': improvement > 0.05  # 5% improvement threshold
        }
    
    def _analyze_distance_effects(self, train_data):
        """Analyze how latency patterns change with distance"""
        distances = train_data['geo_distance_km']
        latencies = train_data['measured_latency_ms']
        physics_latencies = train_data['physics_latency_ms']
        
        # Calculate latency per km in different distance ranges
        distance_ranges = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000)]
        range_analysis = {}
        
        for start, end in distance_ranges:
            mask = (distances >= start) & (distances < end)
            if np.sum(mask) > 5:  # Need enough samples
                range_latencies = latencies[mask]
                range_distances = distances[mask]
                range_physics = physics_latencies[mask]
                
                # Calculate effective slope (latency per km)
                if len(range_distances) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(range_distances, range_latencies)
                    physics_slope, _, _, _, _ = stats.linregress(range_distances, range_physics)
                    
                    range_analysis[f"{start}-{end}km"] = {
                        'n_samples': np.sum(mask),
                        'observed_slope': slope,
                        'physics_slope': physics_slope,
                        'slope_ratio': slope / physics_slope if physics_slope > 0 else 1,
                        'intercept': intercept,
                        'r_squared': r_value**2,
                        'avg_latency_per_km': np.mean(range_latencies / range_distances)
                    }
        
        return range_analysis
    
    def _analyze_outliers(self, train_data):
        """Analyze outlier patterns"""
        X = train_data[['geo_distance_km']].values
        y = train_data['measured_latency_ms'].values
        
        # Fit model and calculate residuals
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        residuals = y - predictions
        
        # Identify outliers using IQR method
        Q1 = np.percentile(residuals, 25)
        Q3 = np.percentile(residuals, 75)
        IQR = Q3 - Q1
        outlier_threshold = 1.5 * IQR
        
        outliers = (residuals < Q1 - outlier_threshold) | (residuals > Q3 + outlier_threshold)
        
        return {
            'outlier_rate': np.mean(outliers),
            'outlier_threshold': outlier_threshold,
            'residual_std': np.std(residuals),
            'residual_skewness': stats.skew(residuals)
        }
    
    def adaptive_physics_uncertainty(self, distances, base_physics_slope=0.005):
        """
        Physics uncertainty that adapts based on discovered patterns
        """
        distances = np.array(distances)
        base_predictions = distances * base_physics_slope
        
        # ADAPTIVE COMPONENTS based on discovered patterns

        # 1. Base measurement noise (adapted from residual analysis)
        if 'outlier_analysis' in self.patterns_discovered:
            #base_noise = self.patterns_discovered['outlier_analysis']['residual_std']
            base_noise = float(self.patterns_discovered['outlier_analysis']['residual_std'])
        else:
            base_noise = 2.0  # Default
        
        noise_std = np.full_like(distances, base_noise)
        
        # 2. Distance-dependent effects (if heteroscedasticity detected)
        if self.patterns_discovered.get('heteroscedasticity', {}).get('is_heteroscedastic', False):
            # Increase uncertainty with distance
            distance_factor = 1 + (distances / 10000) * 0.5
            noise_std *= distance_factor
        
        # 3. Physics parameter uncertainty (adaptive)
        physics_uncertainty_pct = 0.05  # Start with 5%
        
        # Adjust based on distance effects analysis
        if 'distance_effects' in self.patterns_discovered:
            slope_ratios = []
            for range_data in self.patterns_discovered['distance_effects'].values():
                slope_ratios.append(range_data['slope_ratio'])
            
            if slope_ratios:
                # If we see large variations in slope across distances, increase uncertainty
                slope_variation = np.std(slope_ratios)
                physics_uncertainty_pct += min(slope_variation * 0.1, 0.15)  # Cap at 20%
        
        physics_std = distances * base_physics_slope * physics_uncertainty_pct
        
        # 4. Model uncertainty (if non-linearity detected)
        model_uncertainty = np.zeros_like(distances)
        if self.patterns_discovered.get('linearity_test', {}).get('is_nonlinear', False):
            # Add model uncertainty for potential non-linearity
            model_uncertainty = distances * 0.0002  # Small but grows with distance
        
        # Total uncertainty
        total_std = np.sqrt(noise_std**2 + physics_std**2 + model_uncertainty**2)
        dict = {
            'predictions': base_predictions.copy(),
            'uncertainty': total_std.copy(),
            'lower_bound': (base_predictions - self.z_score * total_std).copy(),
            'upper_bound': (base_predictions + self.z_score * total_std).copy(),
            'confidence_level': self.confidence_level,
            'components': {
                'noise': (noise_std).copy(),
                'physics': (physics_std).copy(),
                'model': (model_uncertainty).copy()
            }
        }
        return copy.deepcopy(dict)
        #return dict
    
    def extrapolation_aware_data_uncertainty(self, X_train, y_train, X_test, method='bootstrap'):
        """
        Data-driven uncertainty that accounts for extrapolation without knowing the simulation
        """
        
        # Standard uncertainty
        if method == 'bootstrap':
            base_result = self._bootstrap_uncertainty(X_train, y_train, X_test)
        else:
            base_result = self._bayesian_uncertainty(X_train, y_train, X_test)
        
        # EXTRAPOLATION DETECTION AND PENALTY
        train_distances = X_train.flatten()
        test_distances = X_test.flatten()
        
        train_min, train_max = train_distances.min(), train_distances.max()
        
        # Extrapolation penalty
        extrapolation_penalty = np.zeros_like(test_distances)
        
        # For points outside training range
        beyond_max = test_distances > train_max
        beyond_min = test_distances < train_min
        
        # Penalty grows with distance from training range
        extrapolation_penalty[beyond_max] = (test_distances[beyond_max] - train_max) / 1000 * 2.0
        extrapolation_penalty[beyond_min] = (train_min - test_distances[beyond_min]) / 1000 * 2.0
        
        # MODEL COMPLEXITY PENALTY
        # If we detected non-linearity but used linear model, add uncertainty
        complexity_penalty = np.zeros_like(test_distances)
        if self.patterns_discovered.get('linearity_test', {}).get('is_nonlinear', False):
            # Higher penalty for distances far from training center
            train_center = np.mean(train_distances)
            distance_from_center = np.abs(test_distances - train_center)
            complexity_penalty = distance_from_center / 5000 * 1.0  # 1ms per 5000km from center
        
        # HETEROSCEDASTICITY ADJUSTMENT
        hetero_adjustment = np.ones_like(test_distances)
        if self.patterns_discovered.get('heteroscedasticity', {}).get('is_heteroscedastic', False):
            # Adjust uncertainty based on distance if heteroscedasticity detected
            correlation = self.patterns_discovered['heteroscedasticity']['correlation_with_distance']
            if correlation > 0:  # Uncertainty increases with distance
                hetero_adjustment = 1 + (test_distances / 10000) * abs(correlation)
        
        # Combine all uncertainties
        total_std = np.sqrt(
            (base_result['uncertainty'] * hetero_adjustment)**2 + 
            extrapolation_penalty**2 + 
            complexity_penalty**2
        )
        
        return {
            'predictions': base_result['predictions'],
            'uncertainty': total_std,
            'lower_bound': base_result['predictions'] - self.z_score * total_std,
            'upper_bound': base_result['predictions'] + self.z_score * total_std,
            'confidence_level': self.confidence_level,
            'components': {
                'base': base_result['uncertainty'],
                'extrapolation': extrapolation_penalty,
                'complexity': complexity_penalty,
                'heteroscedasticity': hetero_adjustment - 1
            }
        }
    
    def _bootstrap_uncertainty(self, X_train, y_train, X_test, n_bootstrap=200):
        """Standard bootstrap uncertainty"""
        n_samples = len(X_train)
        predictions = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]
            
            model = LinearRegression()
            model.fit(X_boot, y_boot)
            pred = model.predict(X_test)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        return {
            'predictions': np.mean(predictions, axis=0),
            'uncertainty': np.std(predictions, axis=0)
        }
    
    def _bayesian_uncertainty(self, X_train, y_train, X_test):
        """Standard Bayesian uncertainty"""
        model = BayesianRidge()
        model.fit(X_train, y_train)
        predictions, std_pred = model.predict(X_test, return_std=True)
        return {
            'predictions': predictions,
            'uncertainty': std_pred
        }
    
    def conformal_prediction_uncertainty(self, X_train, y_train, X_cal, y_cal, X_test, alpha=None):
        """
        Conformal prediction: distribution-free uncertainty quantification
        This works without knowing the data generation process!
        
        Parameters:
        -----------
        alpha : float, optional
            Significance level. If None, uses 1 - self.confidence_level
        """
        
        if alpha is None:
            alpha = 1 - self.confidence_level
        
        # Train model on training set
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Get predictions on calibration set
        cal_predictions = model.predict(X_cal)
        cal_residuals = np.abs(y_cal - cal_predictions)
        
        # Calculate quantile of absolute residuals
        quantile_level = (1 - alpha) * (1 + 1/len(cal_residuals))  # Adjusted for finite sample
        quantile = np.quantile(cal_residuals, quantile_level)
        
        # Predictions on test set
        test_predictions = model.predict(X_test)
        
        # Prediction intervals
        lower_bound = test_predictions - quantile
        upper_bound = test_predictions + quantile
        
        return {
            'predictions': test_predictions,
            'uncertainty': np.full_like(test_predictions, quantile),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': self.confidence_level,
            'alpha': alpha,
            'quantile_used': quantile
        }

def comprehensive_blind_analysis(train_data, test_data, confidence_level=0.95):
    """
    Complete analysis pipeline when you don't know the data generation process
    Returns structured results for further processing
    
    Parameters:
    -----------
    confidence_level : float, default=0.95
        Confidence level for uncertainty intervals
    """
    
    estimator = BlindUncertaintyEstimator(confidence_level=confidence_level)
    
    # STEP 1: Discover patterns
    patterns = estimator.discover_data_patterns(train_data, test_data)
    
    # STEP 2: Prepare data
    X_train_full = train_data[['geo_distance_km']].values
    y_train_full = train_data['measured_latency_ms'].values
    X_test = test_data[['geo_distance_km']].values
    y_test = test_data['measured_latency_ms'].values
    
    # Split training data for conformal prediction
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=0.3, random_state=42
    )
    
    # STEP 3: Compare uncertainty methods
    results = {}
    # Adaptive physics
    results['adaptive_physics'] = estimator.adaptive_physics_uncertainty(X_test.flatten())
    
    # Extrapolation-aware data
    results['extrapolation_data'] = estimator.extrapolation_aware_data_uncertainty(
        X_train_full, y_train_full, X_test
    )
    
    # Conformal prediction
    results['conformal'] = estimator.conformal_prediction_uncertainty(
        X_train, y_train, X_cal, y_cal, X_test
    )
    
    # STEP 4: Evaluate results
    def calculate_coverage(y_true, predictions, lower, upper):
        return np.mean((y_true >= lower) & (y_true <= upper))
    
    evaluation = {}
    for method_name, method_results in results.items():
        coverage = calculate_coverage(
            y_test, method_results['predictions'], 
            method_results['lower_bound'], method_results['upper_bound']
        )
        mse = mean_squared_error(y_test, method_results['predictions'])
        avg_uncertainty = np.mean(method_results['uncertainty'])
        
        evaluation[method_name] = {
            'coverage': coverage,
            'mse': mse,
            'avg_uncertainty': avg_uncertainty
        }
    
    return {
        'estimator': estimator,
        'results': results,
        'patterns': patterns,
        'evaluation': evaluation,
        'test_data': {'X': X_test, 'y': y_test}
    }

def production_uncertainty_strategy(train_data, test_data, confidence_level=0.95):
    """
    Production-ready uncertainty strategy for unknown data generation process
    Returns structured strategy recommendations
    
    Parameters:
    -----------
    confidence_level : float, default=0.95
        Confidence level for uncertainty intervals
    """
    
    estimator = BlindUncertaintyEstimator(confidence_level=confidence_level)
    
    # 1. DISCOVERY PHASE
    patterns = estimator.discover_data_patterns(train_data, test_data)
    
    # 2. RISK ASSESSMENT
    risk_factors = []
    risk_score = 0
    
    if 'distribution_shift' in patterns:
        extrapolation_factor = patterns['distribution_shift']['extrapolation_factor']
        if extrapolation_factor > 2:
            risk_factors.append(f"High extrapolation risk ({extrapolation_factor:.1f}x)")
            risk_score += min(extrapolation_factor, 10)  # Cap at 10
    
    if patterns['heteroscedasticity']['is_heteroscedastic']:
        risk_factors.append("Heteroscedastic residuals detected")
        risk_score += 3
    
    if patterns['linearity_test']['is_nonlinear']:
        risk_factors.append("Non-linear patterns detected")
        risk_score += 2
    
    outlier_rate = patterns['outlier_analysis']['outlier_rate']
    if outlier_rate > 0.1:
        risk_factors.append(f"High outlier rate ({outlier_rate:.1%})")
        risk_score += 2
    
    # 3. STRATEGY SELECTION based on risk
    if risk_score < 3:
        strategy = "LOW_RISK"
        recommendation = "Standard bootstrap uncertainty with light calibration"
    elif risk_score < 7:
        strategy = "MEDIUM_RISK" 
        recommendation = "Conformal prediction + extrapolation penalties"
    else:
        strategy = "HIGH_RISK"
        recommendation = "Conservative: Conformal + physics constraints + ensemble"
    
    return {
        'strategy': strategy,
        'risk_score': risk_score,
        'risk_factors': risk_factors,
        'recommendation': recommendation,
        'patterns': patterns,
        'estimator': estimator
    }

def get_pattern_summary(patterns):
    """
    Create a human-readable summary of discovered patterns
    """
    summary = {
        'data_range': {
            'train_min': patterns['train_distance_range'][0],
            'train_max': patterns['train_distance_range'][1],
            'train_mean': patterns['train_distance_mean'],
            'train_std': patterns['train_distance_std']
        },
        'data_quality': {
            'heteroscedastic': patterns['heteroscedasticity']['is_heteroscedastic'],
            'nonlinear': patterns['linearity_test']['is_nonlinear'],
            'outlier_rate': patterns['outlier_analysis']['outlier_rate'],
            'residual_std': patterns['outlier_analysis']['residual_std']
        }
    }
    
    if 'distribution_shift' in patterns:
        summary['extrapolation'] = {
            'test_max': patterns['test_distance_range'][1],
            'extrapolation_factor': patterns['distribution_shift']['extrapolation_factor']
        }
    
    if 'distance_effects' in patterns:
        summary['distance_effects'] = patterns['distance_effects']
    
    return summary