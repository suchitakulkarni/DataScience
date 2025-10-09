import torch

def debug_losses(model, data_loader):
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            x_recon, mu, logvar = model(batch)

            # Individual loss components
            recon_loss = F.mse_loss(x_recon, batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            phys_loss = physics_loss(x_recon)

            print(f"Reconstruction: {recon_loss.item():.6f}")
            print(f"KL Divergence: {kl_loss.item():.6f}")
            print(f"Physics: {phys_loss.item():.6f}")

            # Check for NaN/Inf
            print(f"Any NaN in recon? {torch.isnan(x_recon).any()}")
            print(f"Any Inf in mu? {torch.isinf(mu).any()}")
            break


def debug_physics_implementation(x_recon, dt=0.1):
    pos = x_recon[:, :, 0]

    # Check finite differences
    accel = (pos[:, 2:] - 2 * pos[:, 1:-1] + pos[:, :-2]) / (dt ** 2)
    pos_middle = pos[:, 1:-1]

    print(f"Position range: [{pos.min():.4f}, {pos.max():.4f}]")
    print(f"Acceleration range: [{accel.min():.4f}, {accel.max():.4f}]")

    # Check k estimation stability
    numerator = torch.mean(accel * pos_middle, dim=1, keepdim=True)
    denominator = torch.mean(pos_middle ** 2, dim=1, keepdim=True) + 1e-8
    k_est = -numerator / denominator

    print(f"k_est range: [{k_est.min():.4f}, {k_est.max():.4f}]")
    print(f"k_est std: {k_est.std():.4f}")

    # RED FLAG: if k_est varies by orders of magnitude
    if k_est.max() / k_est.min() > 100:
        print("WARNING: k_est is highly unstable!")

    return k_est


def debug_kl_behavior(mu, logvar):
    print(f"mu stats: mean={mu.mean():.4f}, std={mu.std():.4f}")
    print(f"logvar stats: mean={logvar.mean():.4f}, std={logvar.std():.4f}")
    print(f"sigma stats: mean={torch.exp(0.5 * logvar).mean():.4f}")

    # Check for posterior collapse
    if torch.exp(0.5 * logvar).mean() < 0.01:
        print("WARNING: Possible posterior collapse!")

    # Check for KL explosion
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    print(f"KL per dimension: {kl_per_dim.mean(0)}")


def debug_kl(mu, logvar):
    print(f"mu stats: mean={mu.mean():.4f}, std={mu.std():.4f}")
    print(f"logvar stats: mean={logvar.mean():.4f}, std={logvar.std():.4f}")
    print(f"sigma stats: mean={torch.exp(0.5 * logvar).mean():.4f}")

    if torch.exp(0.5 * logvar).mean() > 10:
        print("KL explosion - sigma too large!")
    if torch.exp(0.5 * logvar).mean() < 0.01:
        print("Posterior collapse - sigma too small!")


def physics_compliance_test(reconstructions, dt=0.1):
    pos = reconstructions[:, :, 0]
    vel = (pos[:, 2:] - pos[:, :-2]) / (2 * dt)
    accel = (pos[:, 2:] - 2 * pos[:, 1:-1] + pos[:, :-2]) / (dt ** 2)

    # Physics residual: ẍ + ω²x = 0
    pos_mid = pos[:, 1:-1]
    omega_sq_est = -torch.mean(accel * pos_mid) / torch.mean(pos_mid ** 2)
    residual = accel + omega_sq_est * pos_mid

    return torch.mean(residual ** 2)
