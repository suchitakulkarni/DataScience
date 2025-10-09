import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from utils.generate_sho import generate_train_sho_data, generate_test_sho_data, generate_sho_position_only
from utils.generate_damped_sho import generate_damped_sho_data,generate_damped_sho_test_data
from utils.debug import debug_kl, physics_compliance_test

from src.model import *
from utils.visualize import  plot_losses, plot_signal, plot_reconstruction, compare_vae_phys
import sys

case = 'DSHO'
train_only = False
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

n_samples_test = 200
dt = 0.1
# The network should see at least two full oscillations
#n_timesteps_train = 50
#n_timesteps_test = 50

hidden_dim = 64
batch_size = 32
n_timesteps_train = 200
n_timesteps_test = 200
#batch_size = 16

t = np.linspace(0, (n_timesteps_test - 1) * dt, n_timesteps_test)

if case == 'SHO':
    latent_dim = 4
    recommended_samples = latent_dim * 200  # Better performance
    n_samples_train = 100
    n_epochs = 500
    lr = 1E-3
    timesteps = train_data.shape[1]
    num_variables = train_data.shape[2]
    input_dim = timesteps * num_variables
    # Generate training data
    train_data, metadata = generate_train_sho_data(n_samples=n_samples_train, n_timesteps=n_timesteps_train)
    test_data, _ = generate_test_sho_data(n_samples=n_samples_test, n_timesteps=n_timesteps_test)

    #train_data, metadata = generate_sho_position_only(n_samples=n_samples_train, n_timesteps=50,dt=dt)
    #test_data, _ = generate_sho_position_only(n_samples=n_samples_test, n_timesteps=50,dt=dt)
if case == 'DSHO':
    latent_dim = 4
    n_epochs = 500
    recommended_samples = latent_dim * 200  # Better performance
    n_samples_train = recommended_samples
    #n_epochs = 10
    lr = 1E-3
    #train_data, metadata= generate_damped_sho_data(n_samples=n_samples_train, n_timesteps=n_timesteps_train, dt=dt)
    train_data, metadata = generate_damped_sho_data(n_samples=n_samples_train, n_timesteps=n_timesteps_train, dt=dt, noise_std=0.0)
    test_data, _ = generate_damped_sho_test_data(n_samples=n_samples_test, n_timesteps=n_timesteps_test, dt=dt, noise_std=0.01)
    timesteps = train_data.shape[1]
    num_variables = train_data.shape[2]
    input_dim = timesteps * num_variables


#train_data, metadata= generate_damped_sho_data_fixed(n_samples=n_samples_train, n_timesteps=50,dt=dt)
#test_data, _ = generate_damped_sho_data_fixed(n_samples=n_samples_test, n_timesteps=50,dt=dt)
print("="*60)
print(f"Training data shape: {train_data.shape}")  # (800, 50, 2)
print(f"Test data shape: {test_data.shape}")       # (200, 50, 2)
print(f"Shape is number of samples X number of timesteps X 2")
print(f"The 2 contains position and velocity")

# ============================================================================
# 4. Training
# ============================================================================

# Convert to PyTorch tensors
train_tensor = torch.FloatTensor(train_data)
test_tensor = torch.FloatTensor(test_data)

# Create data loaders
train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_tensor), batch_size=batch_size, shuffle=False)
dataset = train_loader.dataset
print("="*60)
print(len(dataset))             # total number of samples
print(dataset.tensors[0].shape) # full tensor shape
print("="*60)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PhysicsInformedVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
model.dt = metadata['dt']

#optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
    )

# Convert training data to numpy if it's not already
if isinstance(train_data, torch.Tensor):
    train_data_np = train_data.cpu().detach().numpy()
else:
    train_data_np = train_data

fig = plot_signal(train_data_np, t)
fig.savefig('results/simulated_signal_%s.pdf' %(case))
# Training loop
train_losses = {'total': [], 'recon': [], 'kld': [], 'physics': [], 'latent_corr': []}

KL_loss_weight = 0.001
physics_loss_weight = 0.001
prev_recon_loss = 1.0  # Initial guess
prev_kl_loss = 0.01    # Initial guess

print("Training Physics-Informed VAE...")
for epoch in range(n_epochs):
    if case == 'DSHO':
        # beta scheduling
        #KL_loss_weight = 0.001 + 0.009 * min(1.0, epoch / 100)
        #if epoch > 0:  # Skip first epoch
        #    KL_loss_weight = smart_beta_schedule(epoch, prev_recon_loss, prev_kl_loss)
        # lambda scheduling
        # physics_loss_weight = 0.001 * min(1.0, epoch / 50)
        # Keep KL simple and fixed
        #KL_loss_weight = 0.0001
        KL_loss_weight = 0.00005 # current working setup
        #KL_loss_weight = 0.001 + 0.009 * min(1.0, epoch / 100)

        # Make physics loss much stronger and ramp up faster
        physics_loss_weight = 0.01 * min(1.0, epoch / 20)  # Reaches 0.1 by epoch 20, current working setup

        # decorrelation scheduling
        #latent_corr_loss_weight = 0.1 * min(1.0,  epoch / 70)
        latent_corr_loss_weight = 2 * min(1.0, max(0, (epoch - 100) / 30)) # current working setup
        # Physics loss
        '''if epoch < 30:
            physics_weight = 0.0
        elif epoch < 80:
            physics_weight = 0.0002
        elif epoch < 150:
            physics_weight = 0.0005
        else:
            physics_weight = 0.001'''

        decorr_weight = 0
        # Decorrelation loss - aggressive stepping
        '''if epoch < 40:
            decorr_weight = 0.05
        elif epoch < 100:
            decorr_weight = 0.2
        elif epoch < 180:
            decorr_weight = 0.8
        else:
            decorr_weight = 1  # Very strong'''


    if case == 'SHO':
        # YOU MUST ALSO MODIFY THE LOSS FUNCTION
        physics_loss_weight = 0.001 * min(1.0, epoch / 50)
        KL_loss_weight = 0.01 * min(1.0, epoch / 30)  # 10x smaller

    model.train()
    epoch_loss = 0
    epoch_recon = 0
    epoch_kl = 0
    epoch_phys = 0
    epoch_corr = 0

    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(data)

        loss, recon_loss, kl_loss, phys_loss, corr_loss = vae_loss(
            data, x_recon, mu, logvar,
            beta=KL_loss_weight,           # Lower β for better reconstruction
            physics_weight=physics_loss_weight,  # Higher weight for physics constraint
            latent_loss_weight = decorr_weight,
            dt=dt
        )
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_recon += recon_loss.item()
        epoch_kl += kl_loss.item()
        epoch_phys += phys_loss.item()
        epoch_corr += corr_loss.item()
        prev_recon_loss = recon_loss.item()
        prev_kl_loss = kl_loss.item()

    avg_losses = {
        'total': epoch_loss / len(train_loader),
        'recon': epoch_recon / len(train_loader),
        'kld': epoch_kl / len(train_loader),
        'physics': epoch_phys / len(train_loader),
        'latent_corr': epoch_corr / len(train_loader)
    }
    #train_losses.append(avg_loss)
    for key, value in avg_losses.items():
        train_losses[key].append(value)
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch:3d}: Loss={avg_losses['total']:.4f}, '
              f'Recon={avg_losses['recon']:.4f}, KL={avg_losses['kld']:.4f}, Phys={avg_losses['physics']:.4f}')

print("Training completed!")
fig = plot_losses(train_losses)
fig.savefig('results/losses_%s.pdf' %(case))
# Save model, if you read this file in, you need to initialise the same architecture
# Create a hyperparameters dictionary
hyperparams = {
    "input_dim": input_dim,
    "latent_dim": latent_dim,
    "hidden_dim": hidden_dim,
    "learning_rate": lr,
    "batch_size": batch_size
}

# Save both state_dict and hyperparameters
checkpoint = {
    "model_state_dict": model.state_dict(),
    "hyperparams": hyperparams
}

torch.save(checkpoint, "trained_physics_model.pth")

# ============================================================================
# 5. Evaluation and Visualization
# ============================================================================

model.eval()
with torch.no_grad():
    # Test reconstruction
    test_sample = test_tensor[:4].to(device)  # First 4 test samples
    x_recon, mu, logvar = model(test_sample)

    # Move back to CPU for plotting
    test_sample_np = test_sample.cpu().detach().numpy()
    x_recon_np = x_recon.cpu().detach().numpy()

# Convert training data to numpy if it's not already
if isinstance(train_data, torch.Tensor):
    train_data_np = train_data.cpu().detach().numpy()
else:
    train_data_np = train_data

fig = plot_reconstruction(t, test_sample_np, x_recon_np)
fig.savefig('results/physics_VAE_test_%s.pdf' %(case))

if train_only == True: sys.exit()
# ============================================================================
# 6. Compare with Standard VAE (without physics)
# ============================================================================

print("\n" + "="*60)
print("COMPARISON: Training Standard VAE (without physics constraints)")
print("="*60)

# Standard VAE (same architecture, no physics loss, but we will train again)
model_standard = PhysicsInformedVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
optimizer_standard = optim.Adam(model_standard.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

train_losses_standard = []

for epoch in range(n_epochs):
    model_standard.train()
    epoch_loss = 0
    # beta scheduling
    KL_loss_weight = 0.001 + 0.009 * min(1.0, epoch / 100)
    
    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)
        
        optimizer_standard.zero_grad()
        x_recon, mu, logvar = model_standard(data)
        
        loss, recon_loss, kl_loss = standard_vae_loss(data, x_recon, mu, logvar, beta=KL_loss_weight)
        
        loss.backward()
        optimizer_standard.step()
        
        epoch_loss += loss.item()
    
    train_losses_standard.append(epoch_loss / len(train_loader))
    
    if epoch % 20 == 0:
        print(f'Standard VAE Epoch {epoch:3d}: Loss={epoch_loss/len(train_loader):.4f}')

# Compare reconstructions
model_standard.eval()
with torch.no_grad():
    # Convert back to tensor for standard VAE
    test_sample_tensor = torch.FloatTensor(test_sample_np).to(device)
    x_recon_standard, _, _ = model_standard(test_sample_tensor)
    x_recon_standard_np = x_recon_standard.cpu().detach().numpy()

test_physics_error = physics_compliance_test(x_recon)
print(f"Test physics compliance: {test_physics_error:.6f}")

# Final comparison plot
fig = compare_vae_phys(t, test_sample_np, x_recon_standard_np, x_recon_np)
fig.savefig('results/comparison_VAE_and_physics_%s.pdf' %(case))

#print("\nPhysics-Informed VAE completed!")
#print("Key insights:")
#print("1. Physics loss enforces SHO equation: ẍ + ω²x = 0")
#print("2. Better reconstruction quality due to physics constraints")
#print("3. Learned latent space respects physical laws")
