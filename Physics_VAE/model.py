import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

import matplotlib.pyplot as plt
import os
from losses import *
from visualise import plot_model_reconstruction
from config import *

class InterventionEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class InterventionDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, z):
        return self.decoder(z)

class InterventionMaskNetwork(nn.Module):
    def __init__(self, latent_dim, n_intervention_types=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_intervention_types = n_intervention_types

        self.masks = nn.Parameter(torch.randn(n_intervention_types, latent_dim))

    def forward(self, intervention_idx):
        masks = torch.sigmoid(self.masks[intervention_idx])
        return masks

class InterventionPINNVAE(nn.Module):
    def __init__(self, input_dim=100, latent_dim=3):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = InterventionEncoder(input_dim, latent_dim)
        self.decoder = InterventionDecoder(latent_dim, input_dim)
        self.mask_network = InterventionMaskNetwork(latent_dim, n_intervention_types=3)

        self.intervention_to_idx = {
            'amplitude': 0,
            'frequency': 1,
            'phase': 2
        }

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

def train_model(model, train_loader, t, epochs=100, lr=1e-3, use_interventions=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = {
        'total': [], 'recon': [], 'kl': [], 'physics': [],
        'locality': [], 'sparsity': [], 'diversity': []
    }
    # the following fixed_idx only for making plots
    fixed_idx = np.random.choice(len(train_loader.dataset), size=3, replace=False)

    for epoch in range(epochs):
        model.train()
        epoch_losses = {key: 0 for key in losses.keys()}

        for batch in train_loader:
            x_orig = batch['x_original']
            x_inter = batch['x_intervened']
            intervention_types = batch['intervention_type']

            optimizer.zero_grad()

            x_recon_orig, mu_orig, logvar_orig, z_orig = model(x_orig)
            x_recon_inter, mu_inter, logvar_inter, z_inter = model(x_inter)

            recon_loss = F.mse_loss(x_recon_orig, x_orig) + F.mse_loss(x_recon_inter, x_inter)

            kl_loss = -0.5 * torch.sum(1 + logvar_orig - mu_orig.pow(2) - logvar_orig.exp())
            kl_loss += -0.5 * torch.sum(1 + logvar_inter - mu_inter.pow(2) - logvar_inter.exp())
            kl_loss /= (2 * x_orig.shape[0])

            physics_loss = compute_physics_loss(x_recon_orig, z_orig, t)
            physics_loss += compute_physics_loss(x_recon_inter, z_inter, t)
            physics_loss /= 2

            locality_loss, sparsity_loss, consistency_loss, diversity_loss = \
                compute_intervention_losses(model, x_orig, x_inter, intervention_types)

            if use_interventions:
                schedule = get_intervention_loss_schedule(epoch)
            else:
                schedule = get_baseline_schedule(epoch)

            beta_recon = schedule['recon']
            beta_kl = schedule['kl']  # encourage some structure, but not collapse
            beta_physics = schedule['physics']  # make physics ten times louder
            beta_locality = schedule['locality']  # push latent coordinates to move locally under small perturbations
            beta_sparsity = schedule['sparsity']  # avoid dense entanglement
            beta_diversity = schedule['diversity']  # promote distinct causal directions

            total_loss = (beta_recon * recon_loss +
                          beta_kl * kl_loss +
                          beta_physics * physics_loss +
                          beta_locality * locality_loss +
                          beta_sparsity * sparsity_loss +
                          beta_diversity * diversity_loss)

            total_loss.backward()
            optimizer.step()

            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += recon_loss.item()
            epoch_losses['kl'] += kl_loss.item()
            epoch_losses['physics'] += physics_loss.item()
            epoch_losses['locality'] += locality_loss.item()
            epoch_losses['sparsity'] += sparsity_loss.item()
            epoch_losses['diversity'] += diversity_loss.item()

        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
            losses[key].append(epoch_losses[key])

        if (epoch + 1) % 20 == 0:
            plot_model_reconstruction(model, train_loader.dataset, t, epoch + 1, fixed_idx)
            with torch.no_grad():
                # Sample batch
                x_sample = next(iter(train_loader))['x_original'][:100].to(device)
                mu, logvar = model.encoder(x_sample)

                print(f"\nEpoch {epoch} - Latent Statistics:")
                print(f"  Mean of μ: {mu.mean(dim=0)}")
                print(f"  Std of μ: {mu.std(dim=0)}")
                print(f"  Mean of σ: {torch.exp(0.5 * logvar).mean(dim=0)}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Total: {epoch_losses['total']:.4f}, Recon: {epoch_losses['recon']:.4f}")
            print(f"  KL: {epoch_losses['kl']:.4f}, Physics: {epoch_losses['physics']:.4f}")
            print(f"  Locality: {epoch_losses['locality']:.4f}, Sparsity: {epoch_losses['sparsity']:.4f}")

    return losses
