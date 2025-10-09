import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================================
# 1. Improved Physics-Informed VAE Architecture
# ============================================================================

class ImprovedPhysicsInformedVAE(nn.Module):
    def __init__(self, input_dim=100, latent_dim=4, hidden_dim=64, weight_decay=1e-4):
        super().__init__()
        self.latent_dim = latent_dim
        self.dt = 0.1  # time step

        # Encoder with BatchNorm and spectral normalization for stability
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim * 2)
        )

        # Decoder - deterministic for physics consistency
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim)
        )
        
        # Apply weight initialization for better convergence
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x):
        # Flatten trajectory: (batch, 50, 2) -> (batch, 100)
        x_flat = x.view(x.size(0), -1)
        h = self.encoder(x_flat)
        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        # Clamp logvar to prevent instability with noisy data
        logvar = torch.clamp(logvar, -10, 10)
        return mu, logvar
    
    def reparameterize(self, mu, logvar, training=True):
        """Enhanced reparameterization with optional deterministic mode"""
        if training and self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # Deterministic mode for inference - use mean only
            return mu
    
    def decode(self, z):
        h = self.decoder(z)
        # Reshape back to trajectory: (batch, 100) -> (batch, 50, 2)
        return h.view(h.size(0), -1, 2)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

# ============================================================================
# 2. Noise-Robust Physics Loss Functions
# ============================================================================

def smooth_finite_differences(pos, dt=0.1, window_size=3):
    """
    Apply simple moving average smoothing before computing derivatives
    This helps reduce noise amplification in finite differences
    """
    if window_size > 1:
        # Simple moving average
        kernel = torch.ones(1, 1, window_size, device=pos.device) / window_size
        # Pad to maintain sequence length
        pos_padded = torch.nn.functional.pad(pos.unsqueeze(1), 
                                           (window_size//2, window_size//2), 
                                           mode='reflect')
        pos_smooth = torch.nn.functional.conv1d(pos_padded, kernel, padding=0).squeeze(1)
    else:
        pos_smooth = pos
    
    # Compute derivatives on smoothed data
    vel = (pos_smooth[:, 2:] - pos_smooth[:, :-2]) / (2 * dt)
    accel = (pos_smooth[:, 2:] - 2*pos_smooth[:, 1:-1] + pos_smooth[:, :-2]) / (dt**2)
    pos_middle = pos_smooth[:, 1:-1]
    
    return vel, accel, pos_middle

def robust_damped_physics_loss(x_recon, dt=0.1, outlier_threshold=3.0):
    """
    Robust damped harmonic oscillator loss with outlier handling
    """
    pos = x_recon[:, :, 0]
    
    # Use smoothed finite differences
    vel, accel, pos_middle = smooth_finite_differences(pos, dt, window_size=3)
    
    batch_losses = []
    for b in range(pos.shape[0]):
        v_b = vel[b]
        x_b = pos_middle[b]
        a_b = accel[b]
        
        # Robust parameter estimation using Huber loss concept
        # Normal equation with outlier weighting
        vv = torch.sum(v_b * v_b)
        vx = torch.sum(v_b * x_b)
        xx = torch.sum(x_b * x_b)
        va = torch.sum(v_b * a_b)
        xa = torch.sum(x_b * a_b)
        
        # Solve for parameters with regularization
        det = vv * xx - vx * vx + 1e-6  # Increased regularization
        two_gamma = (-va * xx + xa * vx) / det
        omega_sq = (-xa * vv + va * vx) / det
        
        # Clamp parameters to reasonable ranges
        two_gamma = torch.clamp(two_gamma, 0.0, 2.0)  # Damping should be positive
        omega_sq = torch.clamp(omega_sq, 0.1, 100.0)  # Frequency should be positive
        
        # Physics residual
        residual = a_b + two_gamma * v_b + omega_sq * x_b
        
        # Apply robust loss (Huber-like)
        abs_residual = torch.abs(residual)
        quadratic = torch.minimum(abs_residual, torch.tensor(outlier_threshold))
        linear = abs_residual - quadratic
        huber_loss = 0.5 * quadratic**2 + outlier_threshold * linear
        
        batch_losses.append(torch.mean(huber_loss))
    
    return torch.stack(batch_losses).mean()

def consistency_loss(x_recon, x_original):
    """
    Temporal consistency loss to enforce smoothness
    Penalizes large jumps between consecutive time steps
    """
    # Compute differences between consecutive time steps
    pos_diff = torch.diff(x_recon[:, :, 0], dim=1)  # position differences
    vel_diff = torch.diff(x_recon[:, :, 1], dim=1)  # velocity differences
    
    # L1 loss on differences (promotes smoothness)
    pos_smoothness = torch.mean(torch.abs(pos_diff))
    vel_smoothness = torch.mean(torch.abs(vel_diff))
    
    return pos_smoothness + vel_smoothness

def noise_aware_vae_loss(x, x_recon, mu, logvar, beta=1.0, physics_weight=1.0, 
                        consistency_weight=0.1, dt=0.1, noise_estimate=0.1):
    """
    Enhanced VAE loss with noise awareness and multiple regularization terms
    """
    # Standard reconstruction loss with noise-aware weighting
    recon_loss = nn.MSELoss()(x_recon, x)
    
    # KL divergence with annealing for better training with noise
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # Robust physics constraint loss
    phys_loss = robust_damped_physics_loss(x_recon, dt=dt)
    
    # Temporal consistency loss
    consistency = consistency_loss(x_recon, x)
    
    # Noise-aware weighting: reduce physics weight when reconstruction error is high
    # This prevents physics from dominating when data is very noisy
    with torch.no_grad():
        noise_factor = torch.clamp(recon_loss / noise_estimate, 0.1, 2.0)
    
    # Total loss with adaptive weighting
    total_loss = (recon_loss + 
                 beta * kl_loss + 
                 (physics_weight / noise_factor) * phys_loss + 
                 consistency_weight * consistency)
    
    return total_loss, recon_loss, kl_loss, phys_loss, consistency

# ============================================================================
# 3. Training Utilities for Noisy Data
# ============================================================================

def add_training_noise(data, noise_level=0.05):
    """
    Add controlled noise during training for regularization
    """
    noise = torch.randn_like(data) * noise_level
    return data + noise

def curriculum_noise_schedule(epoch, max_epochs, initial_noise=0.1, final_noise=0.01):
    """
    Gradually reduce noise injection during training
    """
    progress = epoch / max_epochs
    noise_level = initial_noise * (1 - progress) + final_noise * progress
    return noise_level

# ============================================================================
# 4. Example Training Loop with Improvements
# ============================================================================

def train_improved_vae(model, train_loader, num_epochs=100, learning_rate=1e-3):
    """
    Training loop with better regularization (no dropout needed)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, 
                                 weight_decay=1e-4)  # L2 regularization instead of dropout
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                          patience=10, 
                                                          factor=0.5)
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        # Get current noise level for curriculum learning
        noise_level = curriculum_noise_schedule(epoch, num_epochs)
        
        for batch_idx, (data,) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Strategy: Handle already noisy data with optional augmentation
            if noise_level > 0.02:  # Only augment early in training
                # Add small amount of extra noise for regularization
                augmented_data = add_training_noise(data, noise_level * 0.5)
                x_recon, mu, logvar = model(augmented_data)
                # Loss against original (still noisy) data
                target_data = data
            else:
                # Later in training, work with data as-is
                x_recon, mu, logvar = model(data)
                target_data = data
            
            # Compute loss with robust handling
            loss, recon_loss, kl_loss, phys_loss, consistency = noise_aware_vae_loss(
                target_data, x_recon, mu, logvar,
                beta=min(1.0, epoch / 50),  # KL annealing
                physics_weight=1.0,
                consistency_weight=0.1,
                noise_estimate=noise_level
            )
            
            loss.backward()
            
            # Gradient clipping for stability (more important than dropout)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        # Update learning rate
        scheduler.step(epoch_loss / len(train_loader))
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {epoch_loss/len(train_loader):.4f}, '
                  f'Noise Level: {noise_level:.4f}')

# ============================================================================
# 5. Simple Denoising VAE Alternative (Bonus)
# ============================================================================

class DenoisingPhysicsVAE(ImprovedPhysicsInformedVAE):
    """
    Simple extension that explicitly trains on denoising task
    """
    def forward(self, x, noise_level=0.1):
        # Add noise during training
        if self.training:
            x_noisy = add_training_noise(x, noise_level)
            mu, logvar = self.encode(x_noisy)  # Encode noisy input
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            return x_recon, mu, logvar  # Reconstruct clean output
        else:
            return super().forward(x)
