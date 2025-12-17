import torch
import torch.nn.functional as F

def vicreg_loss(z_a, z_b, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
    """
    Computes the VICReg loss.
    z_a, z_b: Embeddings from two different augmentations of the same image batch.
              Shape: [batch_size, embedding_dim]
    """
    batch_size = z_a.size(0)
    num_features = z_a.size(1)

    # 1. Invariance Loss (MSE)
    repr_loss = F.mse_loss(z_a, z_b)

    # 2. Variance Loss (Hinge Loss on Standard Deviation)
    z_a_std = torch.sqrt(z_a.var(dim=0) + 1e-04)
    z_b_std = torch.sqrt(z_b.var(dim=0) + 1e-04)
    std_loss = torch.mean(F.relu(1 - z_a_std)) / 2 + torch.mean(F.relu(1 - z_b_std)) / 2

    # 3. Covariance Loss (Decorrelation)
    z_a = z_a - z_a.mean(dim=0)
    z_b = z_b - z_b.mean(dim=0)

    cov_a = (z_a.T @ z_a) / (batch_size - 1)
    cov_b = (z_b.T @ z_b) / (batch_size - 1)

    # Calculate off-diagonal sum squared
    cov_loss = (
        (cov_a.flatten()[:-1].view(num_features - 1, num_features + 1)[:, 1:].flatten()).pow(2).sum() / num_features
        + (cov_b.flatten()[:-1].view(num_features - 1, num_features + 1)[:, 1:].flatten()).pow(2).sum() / num_features
    )

    loss = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss
    return loss