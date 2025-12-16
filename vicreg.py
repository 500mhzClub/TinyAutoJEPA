import torch
import torch.nn.functional as F

def variance_loss(x, gamma=1.0):
    """
    Forces the embeddings to differ across the batch.
    If the std of the batch is less than gamma (1.0), penalize it.
    """
    # x shape: [batch_size, dim]
    # Add epsilon for numerical stability
    std = torch.sqrt(x.var(dim=0) + 1e-04)
    loss = torch.mean(F.relu(gamma - std))
    return loss

def covariance_loss(x):
    """
    Forces the dimensions to be independent.
    The off-diagonal elements of the covariance matrix should be zero.
    """
    batch_size, dim = x.shape
    # Center the batch
    x = x - x.mean(dim=0)
    # Calculate covariance matrix
    cov = (x.T @ x) / (batch_size - 1)
    
    # Zero out the diagonal (we only care about off-diagonal)
    off_diag = cov.flatten()[:-1].view(dim - 1, dim + 1)[:, 1:].flatten()
    
    loss = off_diag.pow(2).sum() / dim
    return loss

def vicreg_loss(x, y):
    """
    Computes the full VICReg loss.
    x, y: Two different views (augmentations) of the same batch of images.
    """
    # 1. Invariance: The embeddings should be similar
    sim_loss = F.mse_loss(x, y)
    
    # 2. Variance: The embeddings should be "loud" (non-zero)
    std_loss = variance_loss(x) + variance_loss(y)
    
    # 3. Covariance: The embeddings should be efficient (decorrelated)
    cov_loss = covariance_loss(x) + covariance_loss(y)
    
    # Weights recommended by the paper (25, 25, 1)
    loss = (25.0 * sim_loss) + (25.0 * std_loss) + (1.0 * cov_loss)
    
    return loss