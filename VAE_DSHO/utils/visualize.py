import matplotlib.pyplot as plt
import numpy as np
def plot_losses(losses):
    '''takes losses which is a dictionary and plots them on top of them'''
    # Plot training loss
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(losses['total'], label="Total Loss")
    ax.plot(losses['recon'], label="Reconstruction Loss")
    ax.plot(losses['kld'], label="KL loss")
    ax.plot(losses['physics'], label="physics")
    ax.plot(losses['latent_corr'], label="latent correlation")
    ax.set_title('Training Loss (Physics-Informed VAE)')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    return fig

def plot_signal(train_data_np, x):
    import numpy as np
    # Plot results
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle('All Training Data - Simulated SHO Signals')
    # plot first 100 trajectories
    n  = 100
    for i in range(n):
        ax.plot(x, train_data_np[i, :, 0], 'b-', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    #ax.set_title(f'All {len(train_data_np)} Training Trajectories')
    ax.set_title(f'First {n} Training Trajectories')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_reconstruction(x, true_data, reconstructed_data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Physics-Informed VAE: Simple Harmonic Oscillator Reconstruction train data')

    for i in range(4):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        ax.plot(x, true_data[i, :, 0], 'b-', label='True position', linewidth=2)
        ax.plot(x, reconstructed_data[i, :, 0], 'r--', label='Reconstructed', linewidth=2)

        ax.set_xlabel('Time')
        ax.set_ylabel('Position')
        ax.set_title(f'Trajectory {i + 1}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def compare_vae_phys(t, test_sample, recon_no_phys, recon_phys):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Comparison: Physics-Informed VAE vs Standard VAE')

    for i in range(4):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        ax.plot(t, test_sample[i, :, 0], 'b-', label='True', linewidth=2)
        ax.plot(t, recon_phys[i, :, 0], 'r--', label='Physics-Informed VAE', linewidth=2)
        ax.plot(t, recon_no_phys[i, :, 0], 'g:', label='Standard VAE', linewidth=2)

        ax.set_xlabel('Time')
        ax.set_ylabel('Position')
        ax.set_title(f'Trajectory {i + 1}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig