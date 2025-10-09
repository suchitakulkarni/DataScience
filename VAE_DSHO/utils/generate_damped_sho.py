import numpy as np

def generate_damped_sho_data(n_samples=1000, n_timesteps=50, dt=0.1, noise_std=0.1):
    """
    Generate damped harmonic oscillator trajectories
    Physics: ẍ + 2γẋ + ω²x = 0
    
    Solutions depend on damping regime:
    - Underdamped (γ < ω): x(t) = A*e^(-γt)*cos(ω_d*t + φ) where ω_d = √(ω²-γ²)
    - Critically damped (γ = ω): x(t) = (A + Bt)*e^(-γt)  
    - Overdamped (γ > ω): x(t) = A*e^(r₁t) + B*e^(r₂t) where r₁,r₂ = -γ ± √(γ²-ω²)
    """
    data = []
    
    for _ in range(n_samples):
        # Random parameters
        A = np.random.uniform(0.5, 2.0)      # Amplitude/initial condition
        omega = np.random.uniform(1.0, 3.0)   # Natural frequency  
        gamma = np.random.uniform(0.1, 2.5)   # Damping coefficient
        phi = np.random.uniform(0, 2 * np.pi) # Phase
        
        # Generate time series
        t = np.linspace(0, (n_timesteps - 1) * dt, n_timesteps)
        
        if gamma < omega:  # Underdamped - oscillatory with decay
            wd = np.sqrt(omega ** 2 - gamma ** 2)  # Damped frequency
            x = A * np.exp(-gamma * t) * np.cos(wd * t + phi)
            # Velocity: v = dx/dt using product rule and chain rule
            v = A * np.exp(-gamma * t) * (-gamma * np.cos(wd * t + phi) - wd * np.sin(wd * t + phi))
            
        elif abs(gamma - omega) < 1e-10:  # Critically damped
            # Use initial conditions: x(0) = A, v(0) = 0 for simplicity
            # General form: x(t) = (A + Bt)*e^(-γt)
            # With v(0) = 0: B = γA, so x(t) = A(1 + γt)*e^(-γt)
            B = gamma * A
            x = (A + B * t) * np.exp(-gamma * t)
            v = (B - gamma * (A + B * t)) * np.exp(-gamma * t)
            
        else:  # Overdamped - exponential decay without oscillation
            discriminant = np.sqrt(gamma ** 2 - omega ** 2)
            r1 = -gamma + discriminant
            r2 = -gamma - discriminant
            
            # Use initial conditions to find coefficients
            # x(0) = A, v(0) = 0 gives us: C1 + C2 = A, r1*C1 + r2*C2 = 0
            # Solving: C1 = -r2*A/(r1-r2), C2 = r1*A/(r1-r2)
            C1 = -r2 * A / (r1 - r2)
            C2 = r1 * A / (r1 - r2)
            
            x = C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)
            v = C1 * r1 * np.exp(r1 * t) + C2 * r2 * np.exp(r2 * t)

        # Add Gaussian noise to both position and velocity
        if noise_std > 0.0:
            x += np.random.normal(scale=noise_std, size=x.shape)
            v += np.random.normal(scale=noise_std, size=v.shape)
        # Store as [position, velocity] pairs
        trajectory = np.stack([x, v], axis=1)
        data.append(trajectory)
    
    return np.array(data), {"dt": dt}


def generate_damped_sho_test_data(n_samples=1000, n_timesteps=50, dt=0.1, noise_std=0.0):
    """
    Generate damped harmonic oscillator trajectories
    Physics: ẍ + 2γẋ + ω²x = 0

    Solutions depend on damping regime:
    - Underdamped (γ < ω): x(t) = A*e^(-γt)*cos(ω_d*t + φ) where ω_d = √(ω²-γ²)
    - Critically damped (γ = ω): x(t) = (A + Bt)*e^(-γt)
    - Overdamped (γ > ω): x(t) = A*e^(r₁t) + B*e^(r₂t) where r₁,r₂ = -γ ± √(γ²-ω²)
    """
    data = []

    for _ in range(n_samples):
        # Random parameters
        A = np.random.uniform(0.5, 2.0)  # Amplitude/initial condition
        omega = np.random.uniform(1.0, 3.0)  # Natural frequency
        gamma = np.random.uniform(0.1, 2.5)  # Damping coefficient
        phi = np.random.uniform(0, 2 * np.pi)  # Phase

        # Generate time series
        t = np.linspace(0, (n_timesteps - 1) * dt, n_timesteps)

        if gamma < omega:  # Underdamped - oscillatory with decay
            wd = np.sqrt(omega ** 2 - gamma ** 2)  # Damped frequency
            x = A * np.exp(-gamma * t) * np.cos(wd * t + phi)
            # Velocity: v = dx/dt using product rule and chain rule
            v = A * np.exp(-gamma * t) * (-gamma * np.cos(wd * t + phi) - wd * np.sin(wd * t + phi))

        elif abs(gamma - omega) < 1e-10:  # Critically damped
            # Use initial conditions: x(0) = A, v(0) = 0 for simplicity
            # General form: x(t) = (A + Bt)*e^(-γt)
            # With v(0) = 0: B = γA, so x(t) = A(1 + γt)*e^(-γt)
            B = gamma * A
            x = (A + B * t) * np.exp(-gamma * t)
            v = (B - gamma * (A + B * t)) * np.exp(-gamma * t)

        else:  # Overdamped - exponential decay without oscillation
            discriminant = np.sqrt(gamma ** 2 - omega ** 2)
            r1 = -gamma + discriminant
            r2 = -gamma - discriminant

            # Use initial conditions to find coefficients
            # x(0) = A, v(0) = 0 gives us: C1 + C2 = A, r1*C1 + r2*C2 = 0
            # Solving: C1 = -r2*A/(r1-r2), C2 = r1*A/(r1-r2)
            C1 = -r2 * A / (r1 - r2)
            C2 = r1 * A / (r1 - r2)

            x = C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)
            v = C1 * r1 * np.exp(r1 * t) + C2 * r2 * np.exp(r2 * t)
        # Add Gaussian noise to both position and velocity
        if noise_std > 0.0:
            x += np.random.normal(scale=noise_std, size=x.shape)
            v += np.random.normal(scale=noise_std, size=v.shape)
        # Store as [position, velocity] pairs
        trajectory = np.stack([x, v], axis=1)
        data.append(trajectory)

    return np.array(data), {"dt": dt}


def generate_damped_sho_data_fixed(n_samples=1000, n_timesteps=50, dt=0.1):
    """
    Fixed parameter version for testing/debugging
    """
    data = []
    
    for _ in range(n_samples):
        # Fixed parameters for consistency
        A = 1.0       # Amplitude
        omega = 2.0   # Natural frequency
        gamma = 0.5   # Damping coefficient (underdamped since γ < ω)
        phi = 0       # Phase
        
        t = np.linspace(0, (n_timesteps - 1) * dt, n_timesteps)
        
        # Underdamped case
        wd = np.sqrt(omega ** 2 - gamma ** 2)
        x = A * np.exp(-gamma * t) * np.cos(wd * t + phi)
        v = A * np.exp(-gamma * t) * (-gamma * np.cos(wd * t + phi) - wd * np.sin(wd * t + phi))
        
        trajectory = np.stack([x, v], axis=1)
        data.append(trajectory)
    
    return np.array(data), {"dt": dt}


# Test function to verify the physics
def verify_damped_physics(data, dt=0.1):
    """
    Verify that generated data satisfies ẍ + 2γẋ + ω²x = 0
    """
    trajectory = data[0]  # Take first sample
    pos = trajectory[:, 0]
    vel = trajectory[:, 1]
    
    # Compute acceleration using finite differences
    accel = (pos[2:] - 2*pos[1:-1] + pos[:-2]) / (dt**2)
    pos_mid = pos[1:-1]
    vel_mid = vel[1:-1]
    
    # Try to fit the damped oscillator equation
    # accel + 2γ*vel + ω²*pos = 0
    # This is a least squares problem: [vel pos] @ [2γ, ω²] = -accel
    
    A_matrix = np.column_stack([vel_mid, pos_mid])
    b_vector = -accel
    
    # Solve least squares
    params, residuals, rank, s = np.linalg.lstsq(A_matrix, b_vector, rcond=None)
    two_gamma, omega_sq = params
    
    print(f"Estimated parameters: γ = {two_gamma/2:.3f}, ω² = {omega_sq:.3f}")
    print(f"Physics residual (should be ~0): {np.mean(residuals):.6f}")
    
    return two_gamma/2, omega_sq


if __name__ == "__main__":
    # Test the function
    data, metadata = generate_damped_sho_data(n_samples=5, n_timesteps=100)
    print(f"Generated data shape: {data.shape}")
    print(f"Metadata: {metadata}")
    
    # Verify physics
    gamma_est, omega_sq_est = verify_damped_physics(data, metadata["dt"])
    
    # Test fixed version
    data_fixed, _ = generate_damped_sho_data_fixed(n_samples=1, n_timesteps=100)
    print(f"\nFixed parameter test:")
    verify_damped_physics(data_fixed)