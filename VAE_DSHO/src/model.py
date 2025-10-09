import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# ============================================================================
# 2. Physics-Informed VAE Architecture
# ============================================================================

class PhysicsInformedVAE(nn.Module):
    def __init__(self, input_dim=100, latent_dim=4, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.dt = 0.1  # time step

        '''self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),  # 256 if hidden_dim=128
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),  # 128
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 64
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),  # 64
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),  # 128
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),  # 256
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, input_dim)
        )'''
        # Encoder: gradual compression
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 100 → 64
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),  # 64 → 32
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # 32 → 16
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, latent_dim*2)  # 16 → 4 (mu & logvar)
        )

        # Decoder: mirror encoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 4),  # 4 → 16
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),  # 16 → 32,
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),  # 32 → 64
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # 64 → 100
        )
    
    def encode(self, x):
        # Flatten trajectory: (batch, 50, 2) -> (batch, 100)
        x_flat = x.view(x.size(0), -1)
        h = self.encoder(x_flat)
        mu = h[:, :self.latent_dim]
        logvar = h[:, self.latent_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
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
# 3. Physics-Informed Loss Function
# ============================================================================

def physics_loss(x_recon, dt=0.1):
    """
    Physics constraint: Simple Harmonic Oscillator equation
    ẍ + ω²x = 0, which means acceleration = -ω²*position

    We approximate derivatives using finite differences:
    velocity ≈ (x[t+1] - x[t-1]) / (2*dt)
    acceleration ≈ (x[t+1] - 2*x[t] + x[t-1]) / dt^2
    """
    pos = x_recon[:, :, 0]  # positions: (batch, timesteps)

    # Compute acceleration using finite differences (ignoring boundary points)
    # This is where you remember the structure of the dataset, it is
    # number of samples X time steps X 2; where 2 is position and velocity
    accel = (pos[:, 2:] - 2*pos[:, 1:-1] + pos[:, :-2]) / (dt**2)

    # For SHO: acceleration = -omega^2*position
    # We don't know ω, but we can enforce that accel = -k*pos for some k > 0
    # Minimize ||accel + k*pos||^2 and encourage k > 0

    pos_middle = pos[:, 1:-1]  # positions corresponding to computed accelerations

    # Find the best k for each trajectory by solving: accel = -k*pos
    # k = -mean(accel * pos) / mean(pos^2)
    numerator = torch.mean(accel * pos_middle, dim=1, keepdim=True)
    denominator = torch.mean(pos_middle**2, dim=1, keepdim=True) + 1e-8
    k_est = -numerator / denominator

    # Physics residual: how well does ẍ + k*x = 0 hold?
    physics_residual = accel + k_est * pos_middle

    return torch.mean(physics_residual**2)

def vae_loss(x, x_recon, mu, logvar, beta=1.0, physics_weight=1.0, latent_loss_weight = 1.0, dt=0.1):
    """
    Combined VAE + Physics loss
    """
    # Reconstruction loss
    recon_loss = nn.MSELoss()(x_recon, x)
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # Physics constraint loss
    #phys_loss = physics_loss(x_recon, dt=dt)
    phys_loss = damped_physics_loss(x_recon, dt=dt)

    # latent correlation loss
    corr_loss = latent_decorrelation_loss(mu)
    # Total loss
    #total_loss = recon_loss + beta * kl_loss + physics_weight * phys_loss
    total_loss = recon_loss + beta * kl_loss + physics_weight * phys_loss + latent_loss_weight*corr_loss
    #var_reg_loss = variance_regularization(mu)
    #total_loss += 0.02 * var_reg_loss  # Start with small weight

    return total_loss, recon_loss, kl_loss, phys_loss, corr_loss


def physics_loss_fixed(x_recon, omega=2.0, dt=0.1):
    pos = x_recon[:, :, 0]
    accel = (pos[:, 2:] - 2 * pos[:, 1:-1] + pos[:, :-2]) / (dt ** 2)
    pos_middle = pos[:, 1:-1]

    # Use correct k = ω² = 4.0
    physics_residual = accel + (omega ** 2) * pos_middle
    return torch.mean(physics_residual ** 2)



def damped_physics_loss(x_recon, dt=0.1):
    """
    Physics constraint: Damped Harmonic Oscillator equation
    ẍ + 2γẋ + ω²x = 0

    Simple approach: estimate γ and ω² using least squares in closed form
    """
    pos = x_recon[:, :, 0]  # positions: (batch, timesteps)

    # Finite differences
    vel = (pos[:, 2:] - pos[:, :-2]) / (2 * dt)  # velocity
    accel = (pos[:, 2:] - 2 * pos[:, 1:-1] + pos[:, :-2]) / (dt ** 2)  # acceleration
    pos_middle = pos[:, 1:-1]  # align with accel/vel

    # For equation: ẍ + 2γẋ + ω²x = 0
    # Rearrange: ẍ = -2γẋ - ω²x
    # This is: accel = -2γ*vel - ω²*pos

    # Solve for [2γ, ω²] using the normal equation approach
    # We want to minimize ||accel + 2γ*vel + ω²*pos||²

    batch_losses = []
    for b in range(pos.shape[0]):
        v_b = vel[b]  # [seq_len-2]
        x_b = pos_middle[b]  # [seq_len-2]
        a_b = accel[b]  # [seq_len-2]

        # Normal equation: [v·v  v·x] [2γ]   = -[v·a]
        #                  [x·v  p·x] [ω²]     [x·a]
        vv = torch.sum(v_b * v_b)
        vx = torch.sum(v_b * x_b)
        xx = torch.sum(x_b * x_b)
        va = torch.sum(v_b * a_b)
        xa = torch.sum(x_b * a_b)

        # 2x2 system solve with regularization for stability
        det = vv * xx - vx * vx + 1e-8
        two_gamma = (-va * xx + xa * vx) / det
        omega_sq = (-xa * vv + va * vx) / det

        # Physics residual: ẍ + 2γẋ + ω²x
        residual = a_b + two_gamma * v_b + omega_sq * x_b
        batch_losses.append(torch.mean(residual ** 2))

    return torch.stack(batch_losses).mean()


def damped_physics_loss_simple(x_recon, dt=0.1):
    """
    Even simpler version: assume moderate damping and estimate ω² only
    ẍ + 2γ₀ẋ + ω²x = 0 with fixed γ₀
    """
    pos = x_recon[:, :, 0]

    vel = (pos[:, 2:] - pos[:, :-2]) / (2 * dt)
    accel = (pos[:, 2:] - 2 * pos[:, 1:-1] + pos[:, :-2]) / (dt ** 2)
    pos_middle = pos[:, 1:-1]

    # Fixed damping coefficient (you can tune this)
    gamma_fixed = 0.1

    # Solve for ω²: accel = -2γ₀*vel - ω²*pos
    # ω² = -(accel + 2γ₀*vel) / pos (averaged)
    damped_accel = accel + 2 * gamma_fixed * vel

    numerator = torch.mean(damped_accel * pos_middle, dim=1, keepdim=True)
    denominator = torch.mean(pos_middle ** 2, dim=1, keepdim=True) + 1e-8
    omega_sq_est = -numerator / denominator

    # Physics residual
    physics_residual = accel + 2 * gamma_fixed * vel + omega_sq_est * pos_middle

    return torch.mean(physics_residual ** 2)

# separate vae loss to
# a. Avoid unnecessary physics computation
# b. Match the expected return signature
# c. Keep the code clean and explicit
def standard_vae_loss(x, x_recon, mu, logvar, beta=1.0):
    recon_loss = nn.MSELoss()(x_recon, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def smart_beta_schedule(epoch, reconstruction_loss, kl_loss):
    """
    Adjust β based on loss balance, not just epoch
    """
    # Your original base schedule
    base_beta = 0.001 + 0.009 * min(1.0, epoch / 100)

    # Calculate loss ratio (how much KL contributes vs reconstruction)
    loss_ratio = kl_loss / (reconstruction_loss + 1e-8)

    if loss_ratio < 0.01:  # KL too weak (high correlations likely)
        multiplier = 2.0
    elif loss_ratio > 0.1:  # KL too strong (collapse risk)
        multiplier = 0.5
    else:
        multiplier = 1.0  # Just right

    return base_beta * multiplier


def variance_regularization(mu):
    """
    Encourage different latent dims to capture orthogonal information
    """
    # Each dim should have different variance patterns across the batch
    variances = torch.var(mu, dim=0)  # [4] - variance of each latent dim

    # Want high diversity in variances (some dims high var, others low var)
    var_diversity = -torch.var(variances)

    # Prevent any dimension from collapsing to zero variance
    min_var_penalty = torch.sum(torch.relu(0.001 - variances))

    return var_diversity + min_var_penalty

import torch

def latent_decorrelation_loss(mu):
    """
    mu: torch.Tensor of shape (N, 4), batch of latent vectors
    returns: scalar loss penalizing correlation between latent dimensions
    We do not use np.corrcoef because we don't want to move data between devices
    One can use torch.corrcoef, here we are being explicit
    """
    # Center the latents
    # centering is essential since it is in the formula
    mu_centered = mu - mu.mean(dim=0, keepdim=True)  # N x 4

    # Compute covariance matrix (4 x 4)
    cov = (mu_centered.T @ mu_centered) / (mu.shape[0] - 1)

    # Compute standard deviations
    std = torch.sqrt(torch.diag(cov) + 1e-8)  # 4

    # Outer product of std to normalize covariance -> correlation matrix
    corr = cov / (std[:, None] * std[None, :])

    # Zero out diagonal terms
    corr_no_diag = corr - torch.diag(torch.diag(corr))

    # Loss = sum of squared off-diagonal correlations
    loss = (corr_no_diag ** 2).sum()

    return loss
