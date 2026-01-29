import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def vicreg_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4,
    stats_fp32: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    VICReg loss with optional fp32 stats (variance/covariance) for stability under AMP.

    Returns:
      loss: scalar tensor
      metrics: dict of scalar tensors (detached) for logging
    """
    assert z_a.shape == z_b.shape, "z_a and z_b must have same shape"
    bsz = z_a.size(0)
    dim = z_a.size(1)

    # 1) Invariance term
    repr_loss = F.mse_loss(z_a, z_b)

    # Stats terms optionally in fp32
    if stats_fp32:
        za = z_a.float()
        zb = z_b.float()
    else:
        za = z_a
        zb = z_b

    # 2) Variance term
    za = za - za.mean(dim=0)
    zb = zb - zb.mean(dim=0)

    std_za = torch.sqrt(za.var(dim=0, unbiased=False) + eps)
    std_zb = torch.sqrt(zb.var(dim=0, unbiased=False) + eps)
    std_loss = torch.mean(F.relu(1.0 - std_za)) + torch.mean(F.relu(1.0 - std_zb))

    # 3) Covariance term
    cov_za = (za.T @ za) / (bsz - 1)
    cov_zb = (zb.T @ zb) / (bsz - 1)

    eye = torch.eye(dim, device=cov_za.device, dtype=cov_za.dtype)
    cov_loss = (cov_za * (1.0 - eye)).pow(2).sum() / dim + (cov_zb * (1.0 - eye)).pow(2).sum() / dim

    loss = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss

    metrics = {
        "repr": repr_loss.detach(),
        "std": std_loss.detach(),
        "cov": cov_loss.detach(),
    }
    return loss, metrics
