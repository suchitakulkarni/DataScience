import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.model import *
from utils.generate_damped_sho import generate_damped_sho_data,generate_damped_sho_test_data


def test_latent_stability(vae_model, test_data, device='cpu', n_trials=100):
    """
    Test if the model produces consistent latent codes for the same input
    Critical test for VAE stability
    """
    print("Testing latent stability...")

    # Take first trajectory as fixed test case
    fixed_trajectory = test_data[0:1]  # Keep batch dimension
    fixed_tensor = torch.FloatTensor(fixed_trajectory).to(device)

    latent_codes = []
    vae_model.eval()

    with torch.no_grad():
        for trial in range(n_trials):
            mu, logvar = vae_model.encode(fixed_tensor)
            z = vae_model.reparameterize(mu, logvar)
            latent_codes.append(z.cpu().numpy())

    latent_codes = np.array(latent_codes)  # [n_trials, 1, latent_dim]
    latent_codes = latent_codes[:, 0, :]  # Remove batch dim: [n_trials, latent_dim]

    print("\nLatent Stability Analysis:")
    print("=" * 50)

    unstable_dims = 0
    for dim in range(latent_codes.shape[1]):
        mean_val = np.mean(latent_codes[:, dim])
        std_val = np.std(latent_codes[:, dim])
        cv = std_val / abs(mean_val) if abs(mean_val) > 1e-8 else float('inf')

        status = "STABLE" if cv < 0.1 else "UNSTABLE"
        if cv >= 0.1:
            unstable_dims += 1

        print(f"z_{dim}: mean={mean_val:8.4f}, std={std_val:6.4f}, CV={cv:6.4f} [{status}]")

    print("=" * 50)
    if unstable_dims == 0:
        print("✓ Model is STABLE - all dimensions have consistent encodings")
    else:
        print(f"✗ Model is UNSTABLE - {unstable_dims}/{latent_codes.shape[1]} dimensions are unstable")
        print("  This suggests fundamental training issues.")

    return latent_codes

def analyze_vae_latents(vae_model, data_loader, device='cpu', n_samples=1000):
    """
    Comprehensive analysis of VAE latent variables
    
    Args:
        vae_model: Trained VAE model with encode() method
        data_loader: DataLoader with your oscillator data
        device: 'cpu' or 'cuda'
        n_samples: Number of samples to analyze
    
    Returns:
        latents: Encoded latent variables [n_samples, latent_dim]
        true_params: True physical parameters if available
    """
    vae_model.eval()

    with torch.no_grad():
        mu1, logvar1 = vae_model.encode(test_tensor)
        mu2, logvar2 = vae_model.encode(test_tensor)

    print(torch.allclose(mu1, mu2, atol=1e-8))

    latents = []
    reconstructions = []
    originals = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if len(latents) * batch[0].size(0) >= n_samples:
                break
                
            x = batch[0].to(device)  # Assuming batch[0] is your data
            
            # Encode to latent space
            mu, logvar = vae_model.encode(x)  # or however your VAE encodes
            #z = vae_model.reparameterize(mu, logvar)  # Sample from latent
            z = mu
            
            # Decode for reconstruction quality check
            x_recon = vae_model.decode(z)
            
            latents.append(z.cpu().numpy())
            reconstructions.append(x_recon.cpu().numpy())
            originals.append(x.cpu().numpy())
    
    latents = np.concatenate(latents, axis=0)[:n_samples]
    reconstructions = np.concatenate(reconstructions, axis=0)[:n_samples]
    originals = np.concatenate(originals, axis=0)[:n_samples]
    
    return latents, reconstructions, originals


def plot_latent_analysis(latents, true_params=None, reconstructions=None, originals=None):
    """
    Create comprehensive plots of latent space behavior
    
    Args:
        latents: [n_samples, latent_dim] encoded variables
        true_params: [n_samples, n_params] true physical parameters (γ, ω, A, φ)
        reconstructions: [n_samples, seq_len, features] reconstructed trajectories  
        originals: [n_samples, seq_len, features] original trajectories
    """
    n_samples, latent_dim = latents.shape
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Latent Distribution Analysis
    plt.subplot(3, 4, 1)
    plt.hist(latents.flatten(), bins=50, alpha=0.7, density=True)
    plt.axvline(0, color='red', linestyle='--', alpha=0.7, label='Zero')
    plt.title('Overall Latent Distribution')
    plt.xlabel('Latent Value')
    plt.ylabel('Density')
    plt.legend()
    
    # 2. Individual Latent Dimensions
    plt.subplot(3, 4, 2)
    for i in range(min(latent_dim, 6)):  # Show first 6 dimensions
        plt.hist(latents[:, i], bins=30, alpha=0.6, label=f'z_{i}')
    plt.title('Individual Latent Dimensions')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 3. Latent Pairwise Correlations
    plt.subplot(3, 4, 3)
    correlation_matrix = np.corrcoef(latents.T)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=[f'z_{i}' for i in range(latent_dim)],
                yticklabels=[f'z_{i}' for i in range(latent_dim)])
    plt.title('Latent Correlations')
    
    # 4. 2D Latent Space (first two dimensions)
    plt.subplot(3, 4, 4)
    if latent_dim >= 2:
        plt.scatter(latents[:, 0], latents[:, 1], alpha=0.6, s=20)
        plt.xlabel('z_0')
        plt.ylabel('z_1')
        plt.title('2D Latent Space (z_0 vs z_1)')
    
    # 5. PCA of Latent Space
    plt.subplot(3, 4, 5)
    if latent_dim > 2:
        pca = PCA(n_components=2)
        latents_pca = pca.fit_transform(latents)
        plt.scatter(latents_pca[:, 0], latents_pca[:, 1], alpha=0.6, s=20)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        plt.title('PCA of Latent Space')
    
    # 6. Latent vs True Parameters (if available)
    if true_params is not None:
        param_names = ['γ', 'ω', 'A', 'φ']
        n_params = min(true_params.shape[1], 4)
        
        plt.subplot(3, 4, 6)
        for i in range(min(latent_dim, 4)):
            for j in range(min(n_params, 2)):  # Show first 2 true params
                plt.scatter(true_params[:, j], latents[:, i], 
                           alpha=0.5, s=15, label=f'z_{i} vs {param_names[j]}')
        plt.xlabel('True Parameter Value')
        plt.ylabel('Latent Value')
        plt.title('Latent vs True Parameters')
        plt.legend()
    
    # 7. Latent Interpolation Visualization
    plt.subplot(3, 4, 7)
    if latent_dim >= 2:
        # Show interpolation between two random points
        idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
        z1, z2 = latents[idx1], latents[idx2]
        
        # Linear interpolation
        alphas = np.linspace(0, 1, 10)
        interpolated = np.array([alpha * z2 + (1 - alpha) * z1 for alpha in alphas])
        
        plt.plot(interpolated[:, 0], interpolated[:, 1], 'ro-', markersize=4)
        plt.scatter([z1[0], z2[0]], [z1[1], z2[1]], c=['blue', 'green'], s=100, 
                   marker='*', label=['Start', 'End'])
        plt.xlabel('z_0')
        plt.ylabel('z_1')
        plt.title('Latent Interpolation Path')
        plt.legend()
    
    # 8. Reconstruction Quality vs Latent Norm
    if reconstructions is not None and originals is not None:
        plt.subplot(3, 4, 8)
        # Compute reconstruction error
        recon_errors = np.mean((reconstructions - originals)**2, axis=(1, 2))
        latent_norms = np.linalg.norm(latents, axis=1)
        
        plt.scatter(latent_norms, recon_errors, alpha=0.6, s=20)
        plt.xlabel('Latent Norm ||z||')
        plt.ylabel('Reconstruction Error')
        plt.title('Reconstruction vs Latent Magnitude')
        
        # Fit trend line
        z = np.polyfit(latent_norms, recon_errors, 1)
        p = np.poly1d(z)
        plt.plot(np.sort(latent_norms), p(np.sort(latent_norms)), "r--", alpha=0.8)
    
    # 9. t-SNE visualization (if latent_dim > 2)
    if latent_dim > 2:
        plt.subplot(3, 4, 9)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples//4))
        latents_tsne = tsne.fit_transform(latents)
        plt.scatter(latents_tsne[:, 0], latents_tsne[:, 1], alpha=0.6, s=20)
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('t-SNE of Latent Space')
    
    # 10. Latent Activation Statistics
    plt.subplot(3, 4, 10)
    latent_means = np.mean(latents, axis=0)
    latent_stds = np.std(latents, axis=0)
    
    x_pos = np.arange(latent_dim)
    plt.bar(x_pos, latent_means, yerr=latent_stds, alpha=0.7, capsize=5)
    plt.xlabel('Latent Dimension')
    plt.ylabel('Mean ± Std')
    plt.title('Latent Statistics per Dimension')
    plt.xticks(x_pos, [f'z_{i}' for i in range(latent_dim)])
    
    # 11. Sample Trajectories in Latent Space
    if latent_dim >= 3:
        ax = fig.add_subplot(3, 4, 11, projection='3d')
        ax.scatter(latents[:, 0], latents[:, 1], latents[:, 2], alpha=0.6, s=20)
        ax.set_xlabel('z_0')
        ax.set_ylabel('z_1')
        ax.set_zlabel('z_2')
        ax.set_title('3D Latent Space')
    
    # 12. Latent vs Physical Parameter Correlation (if available)
    if true_params is not None:
        plt.subplot(3, 4, 12)
        param_names = ['γ', 'ω', 'A', 'φ']
        
        # Compute correlations between each latent dim and each true param
        correlations = np.zeros((latent_dim, true_params.shape[1]))
        for i in range(latent_dim):
            for j in range(true_params.shape[1]):
                correlations[i, j] = np.corrcoef(latents[:, i], true_params[:, j])[0, 1]
        
        sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0,
                   xticklabels=param_names[:true_params.shape[1]],
                   yticklabels=[f'z_{i}' for i in range(latent_dim)])
        plt.title('Latent-Parameter Correlations')
    
    plt.tight_layout()

    return fig

def plot_latent_traversal(vae_model, latents, device='cpu', n_steps=3):
    """
    Visualize what happens when you traverse individual latent dimensions
    """
    vae_model.eval()
    
    # Take a random sample as base
    #base_z = latents[np.random.randint(len(latents))]
    base_z = np.mean(latents, axis=0)
    latent_dim = len(base_z)
    
    fig, axes = plt.subplots(2, latent_dim//2 + latent_dim%2, figsize=(15, 8))
    axes = axes.flatten() if latent_dim > 2 else [axes]
    
    with torch.no_grad():
        for dim in range(latent_dim):
            # Traverse this dimension
            z_traverse = np.tile(base_z, (n_steps, 1))
            #z_traverse[:, dim] = np.linspace(-4, 4, n_steps)
            #z_min, z_max = latents[:, dim].min(), latents[:, dim].max()
            #z_traverse[:, dim] = np.linspace(z_min, z_max, n_steps)
            z_min, z_max = latents[:, dim].min(), latents[:, dim].max()
            z_traverse[:, dim] = np.linspace(z_min, z_max, n_steps)

            
            # Decode trajectories
            z_tensor = torch.tensor(z_traverse, dtype=torch.float32).to(device)
            decoded = vae_model.decode(z_tensor).cpu().numpy()
            
            # Plot trajectories
            ax = axes[dim] if dim < len(axes) else plt.gca()
            for i, traj in enumerate(decoded):
                alpha = 0.3 + 0.7 * i / (n_steps - 1)
                ax.plot(traj[:, 0], alpha=alpha, label=f'{z_traverse[i, dim]:.1f}' if i % 2 == 0 else '')
            
            ax.set_title(f'Latent z_{dim} Traversal')
            ax.set_xlabel('Time')
            ax.set_ylabel('Position')
            if dim == 0:
                ax.legend(title='z value', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig


# Example usage function
def full_latent_analysis(vae_model, test_loader, test_data_array=None, true_params=None, device='cpu'):
    """
    Complete workflow for analyzing VAE latent space
    """
    # Run stability test first
    if test_data_array is not None:
        stability_codes = test_latent_stability(vae_model, test_data_array, device)

        # If model is unstable, warn user
        cv_values = []
        for dim in range(stability_codes.shape[1]):
            mean_val = np.mean(stability_codes[:, dim])
            std_val = np.std(stability_codes[:, dim])
            cv = std_val / abs(mean_val) if abs(mean_val) > 1e-8 else float('inf')
            cv_values.append(cv)

        if any(cv > 0.1 for cv in cv_values):
            print("\n⚠️  WARNING: Model instability detected!")
            print("   Correlation matrices and traversals may be unreliable.")
            print("   Consider retraining with different hyperparameters.\n")

    print("Encoding test data...")
    latents, reconstructions, originals = analyze_vae_latents(vae_model, test_loader, device)
    
    print(f"Latent space shape: {latents.shape}")
    print(f"Latent statistics: mean={latents.mean():.3f}, std={latents.std():.3f}")
    
    print("Creating latent analysis plots...")
    fig_global = plot_latent_analysis(latents, true_params, reconstructions, originals)

    print("Creating latent traversal plots...")
    fig_traverse = plot_latent_traversal(vae_model, latents, device,n_steps=20)
    
    return fig_global, fig_traverse, latents, reconstructions, originals


if __name__ == "__main__":
    # Code here runs only when this script is executed directly
    # Load checkpoint
    checkpoint = torch.load("trained_physics_model.pth")

    # Retrieve hyperparameters
    hyperparams = checkpoint["hyperparams"]
    dt = 0.1
    #n_timesteps_train = 50
    n_timesteps_train = 200
    n_samples_train = 500
    batch_size = 32
    np.random.seed(42)

    vae_model = PhysicsInformedVAE(
                                    input_dim=hyperparams["input_dim"],
                                    latent_dim=hyperparams["latent_dim"],
                                    hidden_dim=hyperparams["hidden_dim"],
                                    )

    vae_model.load_state_dict(checkpoint["model_state_dict"])
    #train_data, metadata = generate_damped_sho_data(n_samples=n_samples_train,
    #                                                n_timesteps=n_timesteps_train, dt=dt,
    #                                                noise_std=0.0)

    if not os.path.exists('fixed_test_data.npy'):
        print('********** now generating new data')
        test_data, metadata = generate_damped_sho_data(n_samples=n_samples_train,
                                                       n_timesteps=n_timesteps_train, dt=dt,
                                                       noise_std=0.0)
        np.save('fixed_test_data.npy', test_data)

    test_data = np.load('fixed_test_data.npy')
    print(f"Loaded existing dataset with shape: {test_data.shape}")
    #train_tensor = torch.FloatTensor(train_data)
    test_tensor = torch.FloatTensor(test_data)

    # Create data loaders
    #train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
    #test_loader = DataLoader(TensorDataset(test_tensor), batch_size=batch_size, shuffle=False)
    subset = torch.utils.data.Subset(TensorDataset(test_tensor), range(n_samples_train))
    test_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    fig_global, fig_traverse, _, _, _ = full_latent_analysis(vae_model, test_loader, true_params=None, device='cpu')
    fig_traverse.savefig('results/latent_traverse.pdf')
    fig_global.savefig('results/latent_global.pdf')