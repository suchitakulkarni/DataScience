import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset import *
from visualise import plot_losses, analyze_intervention_alignment
from model import train_model, InterventionPINNVAE
from config import *

if __name__ == "__main__":

    dataset = SHOInterventionDataset(n_samples=5000)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    t = torch.FloatTensor(dataset.t).to(device)
    model = InterventionPINNVAE(input_dim=100, latent_dim=3).to(device)

    # Baseline
    losses = train_model(
        model, train_loader, t, epochs = 200, lr = 1e-3, use_interventions = False
    )
    plot_losses(losses)

    # 2. Analyze disentanglement
    print("\nAnalyzing latent space alignment...")
    results_sho_baseline = analyze_intervention_alignment(
        model, dataset, device
    )

    # 3. Save results and model
    torch.save(model.state_dict(), 'results/sho_baseline.pth')
    #save_correlation_scores(results_sho_baseline, 'results/sho_baseline_scores.txt')

    #os.system("bash make_movie.sh")
    print("\nIntervention-Based PINN-VAE training complete!")
    print("Results saved to results/")