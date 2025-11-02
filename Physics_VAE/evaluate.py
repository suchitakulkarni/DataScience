import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

def comprehensive_evaluation(model, dataset, device):
    """Evaluate baseline without interventions"""

    # 1. Disentanglement score
    disentanglement = compute_disentanglement_score(model, dataset)

    # 2. Per-parameter best correlation
    latents, params = extract_latents_and_params(model, dataset)
    correlations = {}
    for i, param_name in enumerate(['amplitude', 'frequency', 'phase']):
        best_corr = max([
            abs(np.corrcoef(params[:, i], latents[:, j])[0, 1])
            for j in range(latents.shape[1])
        ])
        correlations[param_name] = best_corr

    # 3. Reconstruction quality
    recon_error = compute_reconstruction_error(model, dataset)

    # 4. Physics violation
    physics_error = compute_physics_violation(model, dataset)

    return {
        'disentanglement': disentanglement,
        'correlations': correlations,
        'reconstruction': recon_error,
        'physics': physics_error
    }

def compute_disentanglement_score(latents, true_params):
    """
    Compute disentanglement metric: how well each latent dimension
    aligns with a single physical parameter.

    Higher score = better disentanglement
    """
    n_latent = latents.shape[1]
    n_params = true_params.shape[1]

    corr_matrix = np.zeros((n_params, n_latent))

    for i in range(n_params):
        for j in range(n_latent):
            corr_matrix[i, j] = np.abs(np.corrcoef(true_params[:, i], latents[:, j])[0, 1])

    max_corr_per_latent = np.max(corr_matrix, axis=0)
    disentanglement_score = np.mean(max_corr_per_latent)

    assignment = np.argmax(corr_matrix, axis=0)

    return disentanglement_score, corr_matrix, assignment

def analyze_model(model, dataset, device, model_type):
    """Analyze trained model and extract latent representations"""
    model.eval()

    n_test = 500
    indices = np.random.choice(len(dataset), n_test, replace=False)

    latents = []
    true_params = []

    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]

            if model_type == "context_aware":
                x = torch.FloatTensor(sample['trajectory']).unsqueeze(0).to(device)
                context = torch.FloatTensor([
                    sample['amplitude'],
                    sample['frequency'],
                    sample['phase']
                ]).unsqueeze(0).to(device)

                mu, _ = model.encoder(x, context)
                latents.append(mu.cpu().numpy()[0])
                true_params.append([
                    sample['amplitude'],
                    sample['frequency'],
                    sample['phase']
                ])
            else:
                x = torch.FloatTensor(sample['x_original']).unsqueeze(0).to(device)
                mu, _ = model.encoder(x)
                latents.append(mu.cpu().numpy()[0])
                true_params.append(sample['params_original'])

    latents = np.array(latents)
    true_params = np.array(true_params)

    return latents, true_params