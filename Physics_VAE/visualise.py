import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from config import *

def plot_model_reconstruction(model, dataset, t, epoch, fixed_idx):
    """
    Plot reconstructions of the same fixed samples across epochs
    """
    model.eval()
    fig, axes = plt.subplots(len(fixed_idx), 1, figsize=(6, 1.5 * len(fixed_idx)), sharex=True)

    with torch.no_grad():
        for i, ax in enumerate(axes):
            sample = dataset[fixed_idx[i]]
            x = torch.FloatTensor(sample['x_original']).unsqueeze(0).to(next(model.parameters()).device)
            x_recon, mu, logvar, z = model(x)
            x = x.squeeze().cpu().numpy()
            x_recon = x_recon.squeeze().cpu().numpy()

            ax.plot(t.cpu().numpy(), x, label="True", linewidth=1)
            ax.plot(t.cpu().numpy(), x_recon, "--", label="Reconstructed", linewidth=1)
            ax.set_ylabel(f"Traj {fixed_idx[i]}")
            ax.legend(fontsize="small", loc="upper right")

    plt.tight_layout()
    fname = os.path.join(results_dir, f"reconstruction_progress_epoch_{epoch:04d}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")

def analyze_intervention_alignment(model, dataset, device='cpu'):
    model.eval()

    n_test = 500
    indices = np.random.choice(len(dataset), n_test, replace=False)

    latents = []
    true_params = []

    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            x = torch.FloatTensor(sample['x_original']).unsqueeze(0).to(device)

            mu, _ = model.encoder(x)
            latents.append(mu.cpu().numpy()[0])
            true_params.append(sample['params_original'])

    latents = np.array(latents)
    true_params = np.array(true_params)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    param_names = ['Amplitude', 'Frequency', 'Phase']

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            ax.scatter(true_params[:, i], latents[:, j], alpha=0.5, s=10)
            ax.set_xlabel(f'True {param_names[i]}')
            ax.set_ylabel(f'Latent dim {j}')

            corr = np.corrcoef(true_params[:, i], latents[:, j])[0, 1]
            ax.set_title(f'Correlation: {corr:.3f}')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/latent_alignment.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nLatent-Physical Parameter Correlations (Intervention-Based):")
    for i, param_name in enumerate(param_names):
        print(f"\n{param_name}:")
        for j in range(3):
            corr = np.corrcoef(true_params[:, i], latents[:, j])[0, 1]
            print(f"  Latent {j}: {corr:.3f}")

    print("\nLearned Intervention Masks:")
    with torch.no_grad():
        masks = torch.sigmoid(model.mask_network.masks).cpu().numpy()
        intervention_names = ['Amplitude', 'Frequency', 'Phase']

        for i, name in enumerate(intervention_names):
            print(f"\n{name} intervention affects:")
            for j in range(3):
                print(f"  Latent {j}: {masks[i, j]:.3f}")

def plot_losses(losses):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (key, values) in enumerate(losses.items()):
        if idx < len(axes):
            axes[idx].plot(values)
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Loss')
            axes[idx].set_title(f'{key.capitalize()} Loss')
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/training_losses.png', dpi=150, bbox_inches='tight')
    plt.close()