import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

def compute_physics_loss(x, z, t):
    """
    Physics loss for SHO: x(t) = A * sin(omega * t + phi)
    Compare reconstruction to analytic SHO form
    """
    batch_size = x.shape[0]

    # Extract physical parameters from latent space
    A = torch.abs(z[:, 0:1])  # amplitude (keep positive)
    omega = torch.abs(z[:, 1:2])  # frequency (keep positive)
    phi = z[:, 2:3]  # phase (can be any value)

    # Create analytic SHO signal
    t_expanded = t.unsqueeze(0).expand(batch_size, -1)  # [B, T]
    x_analytic = A * torch.sin(omega * t_expanded + phi)

    # Loss: reconstruction should match SHO dynamics
    return F.mse_loss(x, x_analytic)

def compute_intervention_losses(model, x_orig, x_inter, intervention_types):
    x_recon_orig, mu_orig, logvar_orig, z_orig = model(x_orig)
    x_recon_inter, mu_inter, logvar_inter, z_inter = model(x_inter)

    intervention_indices = torch.LongTensor([
        model.intervention_to_idx[it] for it in intervention_types
    ]).to(x_orig.device)

    masks = model.mask_network(intervention_indices)

    delta_z = z_inter - z_orig

    locality_loss = torch.mean((delta_z * (1 - masks)) ** 2)

    sparsity_loss = torch.mean(torch.sum(masks, dim=1))

    consistency_loss = torch.mean((delta_z * masks) ** 2)

    all_masks = torch.sigmoid(model.mask_network.masks)
    diversity_matrix = torch.mm(all_masks, all_masks.t())
    mask_norms = torch.norm(all_masks, dim=1, keepdim=True)
    diversity_matrix = diversity_matrix / (mask_norms * mask_norms.t() + 1e-8)
    diversity_matrix = diversity_matrix - torch.eye(3).to(x_orig.device)
    diversity_loss = torch.mean(diversity_matrix ** 2)

    return locality_loss, sparsity_loss, consistency_loss, diversity_loss

def get_intervention_loss_schedule(epoch):
    if epoch < 50:
        return {
            'recon': 1.0,
            'kl': 0.001,
            'physics': 0.0,
            'locality': 0.0,
            'sparsity': 0.0,
            'diversity': 0.0
        }
    elif epoch < 150:
        physics_weight = 0.1 + 0.9 * (epoch - 50) / 100
        return {
            'recon': 1.0,
            'kl': 0.001,
            'physics': physics_weight,
            'locality': 0.0,
            'sparsity': 0.0,
            'diversity': 0.0
        }
    else:
        # Gradual warmup + reduced weights
        progress = min((epoch - 150) / 100, 1.0)

        return {
            'recon': 1.0,
            'kl': 0.001,
            'physics': 1.0,
            'locality': 0.5 * progress,  # max 0.5 instead of 1.0
            'sparsity': 0.02 * progress,  # max 0.02 instead of 0.05
            'diversity': 0.05 * progress  # max 0.05 instead of 0.1
        }

def get_baseline_schedule(epoch):
    """Just VAE + Physics, no intervention losses"""
    if epoch < 50:
        return {
            'recon': 1.0,
            'kl': 0.001,
            'physics': 0.0,
            'locality': 0.0,
            'sparsity': 0.0,
            'diversity': 0.0
        }
    elif epoch < 150:
        progress = (epoch - 50) / 100
        physics_weight = 0.1 + 0.5 * progress
        return {
            'recon': 1,
            'kl': 0.005,
            'physics': physics_weight,
            'locality': 0.0,
            'sparsity': 0.0,
            'diversity': 0.0
        }
    else:
        return {
            'recon': 1,
            'kl': 0.005,
            'physics': 0.6,
            'locality': 0.0,
            'sparsity': 0.0,
            'diversity': 0.0
        }