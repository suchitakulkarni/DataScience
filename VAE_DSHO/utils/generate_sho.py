import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1. Generate Simple Harmonic Oscillator Data
# ============================================================================

def generate_train_sho_data(n_samples=1000, n_timesteps=50, dt=0.1):
    """
    Generate simple harmonic oscillator trajectories: x(t) = A*cos(ωt + φ)
    Physics: F = -kx, so ẍ = -ω²x where ω = sqrt(k/m)
    """
    data = []
    
    for _ in range(n_samples):
        # Random initial conditions
        A = np.random.uniform(0.5, 2.0)      # Amplitude
        omega = np.random.uniform(1.0, 3.0)   # Angular frequency
        phi = np.random.uniform(0, 2*np.pi)   # Phase

        #A = 1.0      # Amplitude
        #omega = 2.0   # Angular frequency
        #phi = 0   # Phase
        
        # Generate time series
        t = np.linspace(0, (n_timesteps-1)*dt, n_timesteps)
        x = A * np.cos(omega * t + phi)
        v = -A * omega * np.sin(omega * t + phi)  # velocity
        
        # Store as [position, velocity] pairs
        trajectory = np.stack([x, v], axis=1)
        data.append(trajectory)
    
    return np.array(data), {"dt": dt}


def generate_test_sho_data(n_samples=1000, n_timesteps=50, dt=0.1):
    """
    Generate simple harmonic oscillator trajectories: x(t) = A*cos(ωt + φ)
    Physics: F = -kx, so ẍ = -ω²x where ω = sqrt(k/m)
    """
    data = []

    for _ in range(n_samples):
        # Random initial conditions
        A = np.random.uniform(1, 5.0)  # Amplitude
        omega = np.random.uniform(3.0, 4.0)  # Angular frequency
        phi = np.random.uniform(0, 2 * np.pi)  # Phase

        # A = 1.0      # Amplitude
        # omega = 2.0   # Angular frequency
        # phi = 0   # Phase

        # Generate time series
        t = np.linspace(0, (n_timesteps - 1) * dt, n_timesteps)
        x = A * np.cos(omega * t + phi)
        v = -A * omega * np.sin(omega * t + phi)  # velocity

        # Store as [position, velocity] pairs
        trajectory = np.stack([x, v], axis=1)
        data.append(trajectory)

    return np.array(data), {"dt": dt}


'''def generate_sho_position_only(n_samples=1000, n_timesteps=50, dt=0.1):
    data = []
    for _ in range(n_samples):
        A, omega, phi = 1.0, 2.0, 0
        t = np.linspace(0, (n_timesteps - 1) * dt, n_timesteps)
        x = A * np.cos(omega * t + phi)

        # Only store position, not velocity
        trajectory = x.reshape(-1, 1)  # Shape: (timesteps, 1)
        data.append(trajectory)

    return np.array(data), {"dt": dt}'''


def generate_sho_position_only(n_samples=1000, n_timesteps=50, dt=0.1):
    data = []
    for _ in range(n_samples):
        #A, omega, phi = 1.0, 2.0, 0
        # Random initial conditions
        A = np.random.uniform(0.5, 2.0)      # Amplitude
        omega = np.random.uniform(1.0, 3.0)   # Angular frequency
        phi = np.random.uniform(0, 2*np.pi)   # Phase
        t = np.linspace(0, (n_timesteps - 1) * dt, n_timesteps)
        x = A * np.cos(omega * t + phi)

        # Pad with zeros to match original shape
        zeros = np.zeros_like(x)
        trajectory = np.stack([x, zeros], axis=1)  # (timesteps, 2)
        data.append(trajectory)

    return np.array(data), {"dt": dt}