import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class ELBOLoss(nn.Module):
    """
    Evidence Lower Bound (ELBO) loss for variational reconstruction.
    """
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        self.mse_loss = nn.MSELoss()
    
    def forward(self, 
                original: torch.Tensor, 
                reconstructed: torch.Tensor,
                mu: torch.Tensor, 
                logvar: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute ELBO loss.
        
        Args:
            original: Original features
            reconstructed: Reconstructed features
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Total ELBO loss and component losses
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = self.mse_loss(
