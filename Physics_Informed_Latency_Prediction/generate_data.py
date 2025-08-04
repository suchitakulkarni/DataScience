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
from math import radians, sin, cos, sqrt, atan2
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Physics constants
FIBER_SPEED = 2e8  # speed of light in optical fiber (m/s)
TRUE_PHYSICS_SLOPE = 1000 / FIBER_SPEED * 1000  # ms/km = 0.005

def haversine(lat1, lon1, lat2, lon2):
    """Calculate geodesic distance between two points on Earth"""
    R = 6371000  # Earth's radius in meters
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# City coordinates - simplified set
cities = {
    # Short distance pairs (for biased training)
    'London': (51.5074, -0.1278), 'Paris': (48.8566, 2.3522),
    'Frankfurt': (50.1109, 8.6821), 'Amsterdam': (52.3676, 4.9041),
    'Berlin': (52.5200, 13.4050), 'Zurich': (47.3769, 8.5417),
    'Vienna': (48.2082, 16.3738), 'Rome': (41.9028, 12.4964),
    
    # Long distance cities (for unbiased test)
    'New York': (40.7128, -74.0060), 'Los Angeles': (34.0522, -118.2437),
    'Tokyo': (35.6762, 139.6503), 'Sydney': (-33.8688, 151.2093),
    'Singapore': (1.3521, 103.8198), 'Mumbai': (19.0760, 72.8777),
    'São Paulo': (-23.5505, -46.6333), 'Dubai': (25.2048, 55.2708)
}

def generate_latency_data(city_pairs, n_samples, anomaly_rate=0.05):
    """Generate realistic latency data for given city pairs"""
    
    data = []
    
    for i in range(n_samples):
        # Sample a city pair
        city_a, city_b = city_pairs[i % len(city_pairs)]
        lat1, lon1 = cities[city_a]
        lat2, lon2 = cities[city_b]
        
        # Calculate geodesic distance
        geo_distance_m = haversine(lat1, lon1, lat2, lon2)
        geo_distance_km = geo_distance_m / 1000
        
        # REALISTIC LATENCY MODEL WITH HIDDEN COMPLEXITY
        
        # 1. True physics component
        physics_latency_ms = geo_distance_km * TRUE_PHYSICS_SLOPE
        
        # 2. Infrastructure complexity (hidden from models)
        if geo_distance_km < 1000:
            # Short routes: more hops, indirect routing
            route_factor = np.random.normal(2.2, 0.4)  # More overhead
            equipment_delay = np.random.normal(3.0, 0.8)
        elif geo_distance_km < 5000:
            # Medium routes: balanced
            route_factor = np.random.normal(1.4, 0.2)
            equipment_delay = np.random.normal(2.0, 0.4)
        else:
            # Long routes: direct fiber cables
            route_factor = np.random.normal(1.15, 0.05)  # More direct
            equipment_delay = np.random.normal(1.5, 0.3)
        
        route_factor = max(1.05, route_factor)
        equipment_delay = max(0.5, equipment_delay)
        
        # 3. Additional complexity
        actual_distance_km = geo_distance_km * route_factor
        actual_physics_ms = actual_distance_km * TRUE_PHYSICS_SLOPE
        
        # 4. Total expected latency
        expected_latency_ms = actual_physics_ms + equipment_delay
        
        # 5. Measurement noise
        noise_std = 0.5 + (geo_distance_km / 10000) * 1.0
        noise_ms = np.random.normal(0, noise_std)
        
        # 6. Anomalies
        is_anomaly = np.random.rand() < anomaly_rate
        anomaly_ms = 0
        
        if is_anomaly:
            # Different anomaly types
            anomaly_type = np.random.choice(['congestion', 'routing', 'equipment'])
            if anomaly_type == 'congestion':
                anomaly_ms = np.random.uniform(5, 20)
            elif anomaly_type == 'routing':
                anomaly_ms = np.random.uniform(15, 50)
            else:  # equipment
                anomaly_ms = np.random.uniform(-3, 25)
        
        # 7. Final measured latency
        measured_latency_ms = expected_latency_ms + noise_ms + anomaly_ms
        measured_latency_ms = max(0.1, measured_latency_ms)
        
        data.append({
            'source_city': city_a,
            'dest_city': city_b,
            'geo_distance_km': geo_distance_km,
            'measured_latency_ms': measured_latency_ms,
            'physics_latency_ms': geo_distance_km * TRUE_PHYSICS_SLOPE,  # Pure physics prediction
            'is_anomaly': int(is_anomaly)
        })
    return pd.DataFrame(data)

# SHORT DISTANCE CITY PAIRS (< 2000km) - BIASED TRAINING
european_cities = ['London', 'Paris', 'Frankfurt', 'Amsterdam', 'Berlin', 'Zurich', 'Vienna', 'Rome']
short_pairs = []
for i, city_a in enumerate(european_cities):
    for city_b in european_cities[i+1:]:
        dist = haversine(*cities[city_a], *cities[city_b]) / 1000
        if dist < 2000:  # Only short distances
            short_pairs.append((city_a, city_b))

print(f"SHORT DISTANCE PAIRS (training): {len(short_pairs)} pairs")
for pair in short_pairs[:5]:
    dist = haversine(*cities[pair[0]], *cities[pair[1]]) / 1000
    print(f"  {pair[0]} - {pair[1]}: {dist:.0f} km")
print(f"  ... and {len(short_pairs)-5} more")
print()

# ALL DISTANCE PAIRS - UNBIASED TEST
all_cities = list(cities.keys())
all_pairs = []
for i, city_a in enumerate(all_cities):
    for city_b in all_cities[i+1:]:
        all_pairs.append((city_a, city_b))

# Generate datasets
print("Generating datasets...")
train_data = generate_latency_data(short_pairs, n_samples=800, anomaly_rate=0.04)
test_data = generate_latency_data(all_pairs, n_samples=1200, anomaly_rate=0.05)
train_data.to_csv('enahnced_simulation_train_data.dat')
test_data.to_csv('enahnced_simulation_test_data.dat')

