import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List

class ContrastiveLearning(nn.Module):
    """
    Contrastive learning module for disentangled representations.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def compute_similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between two tensors.
        
        Args:
            x1: First tensor [batch_size, dim]
            x2: Second tensor [batch_size, dim]
            
        Returns:
            Cosine similarity scores
        """
        return F.cosine_similarity(x1, x2, dim=-1)
    
    def create_positive_pairs(self, z_I: torch.Tensor, z_V: torch.Tensor) -> torch.Tensor:
        """
        Create positive pairs by adding sampled variance to content features.
        
        Args:
            z_I: Content features [batch_size, dim]
            z_V: Variance features [batch_size, dim]
            
        Returns:
            Augmented positive pairs
        """
        # Sample different variance vectors for augmentation
        batch_size = z_I.size(0)
        variance_dim = z_V.size(1)
        
        # Sample new variance vectors
        z_V_aug = torch.randn(batch_size, variance_dim, device=z_I.device)
        
        # Create positive pairs
        z_I_aug = z_I + z_V_aug
        
        return z_I_aug
    
    def create_negative_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Create mask for negative pairs based on MeSH label overlap.
        
        Args:
            labels: Multi-hot label tensor [batch_size, num_labels]
            
        Returns:
            Boolean mask where True indicates no label overlap (negative pair)
        """
        batch_size = labels.size(0)
        
        # Compute label overlap using dot product
        label_overlap = torch.mm(labels, labels.t())  # [batch_size, batch_size]
        
        # Create mask: True where no overlap exists (negative pairs)
        negative_mask = (label_overlap == 0)
        
        # Remove diagonal (self-similarity)
        negative_mask.fill_diagonal_(False)
        
        return negative_mask
    
    def forward(self, z_I: torch.Tensor, z_V: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            z_I: Content features [batch_size, dim]
            z_V: Variance features [batch_size, dim]
            labels: Multi-hot labels [batch_size, num_labels]
            
        Returns:
            Contrastive loss
        """
        batch_size = z_I.size(0)
        
        # Create positive pairs
        z_I_aug = self.create_positive_pairs(z_I, z_V)
        
        # Compute similarities between anchors and positive pairs
        pos_sim = self.compute_similarity(z_I, z_I_aug)  # [batch_size]
        pos_sim = pos_sim / self.temperature
        
        # Compute similarities between all pairs
        all_sim = torch.mm(z_I, z_I.t()) / self.temperature  # [batch_size, batch_size]
        
        # Create negative mask
        negative_mask = self.create_negative_mask(labels)
        
        # Compute contrastive loss
        # For each anchor, compute log probability of positive pair
        numerator = torch.exp(pos_sim)
        
        # Denominator includes positive pair + all negative pairs
        denominator = numerator.clone()
        
        for i in range(batch_size):
            # Add similarities with negative samples
            neg_sims = all_sim[i][negative_mask[i]]
            if len(neg_sims) > 0:
                denominator[i] += torch.sum(torch.exp(neg_sims))
        
        # Compute loss
        loss = -torch.log(numerator / denominator)
        
        return loss.mean()

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning loss adapted for multi-label classification.
    """
    
    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised contrastive loss for multi-label scenario.
        
        Args:
            features: Feature representations [batch_size, dim]
            labels: Multi-hot labels [batch_size, num_labels]
            
        Returns:
            Supervised contrastive loss
        """
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(features, features.t())
        
        # Create mask for positive pairs (samples with overlapping labels)
        label_overlap = torch.mm(labels, labels.t())  # [batch_size, batch_size]
        positive_mask = (label_overlap > 0).float()
        
        # Remove diagonal
        positive_mask.fill_diagonal_(0)
        
        # Apply temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log probabilities
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Compute mean log-likelihood over positive pairs
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / positive_mask.sum(1)
        
        # Handle case where no positive pairs exist
        mean_log_prob_pos[positive_mask.sum(1) == 0] = 0
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss

class StyleAugmentation(nn.Module):
    """
    Style-based data augmentation using variance components.
    """
    
    def __init__(self, augment_strength: float = 0.1):
        super().__init__()
        self.augment_strength = augment_strength
    
    def forward(self, z_I: torch.Tensor, z_V_params: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Generate style-augmented versions of content features.
        
        Args:
            z_I: Content features
            z_V_params: Tuple of (mu, logvar) for variance distribution
            
        Returns:
            Style-augmented features
        """
        mu, logvar = z_V_params
        
        # Sample multiple variance vectors
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_V_new = mu + eps * std * self.augment_strength
        
        # Create augmented features
        z_augmented = z_I + z_V_new
        
        return z_augmented

class ContrastiveTrainer:
    """
    Trainer class for contrastive learning with disentangled representations.
    """
    
    def __init__(self, 
                 temperature: float = 0.07,
                 augment_strength: float = 0.1,
                 use_supervised: bool = True):
        
        if use_supervised:
            self.contrastive_loss = SupConLoss(temperature=temperature)
        else:
            self.contrastive_loss = ContrastiveLearning(temperature=temperature)
            
        self.style_augmentation = StyleAugmentation(augment_strength=augment_strength)
        self.use_supervised = use_supervised
    
    def compute_contrastive_loss(self, 
                               z_I: torch.Tensor, 
                               z_V: torch.Tensor,
                               mu: torch.Tensor,
                               logvar: torch.Tensor,
                               labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive loss with style augmentation.
        
        Args:
            z_I: Content features
            z_V: Variance features
            mu: Mean of variance distribution
            logvar: Log variance of variance distribution
            labels: Multi-hot labels
            
        Returns:
            Dictionary of losses and metrics
        """
        # Generate style-augmented features
        z_augmented = self.style_augmentation(z_I, (mu, logvar))
        
        if self.use_supervised:
            # Supervised contrastive loss
            # Combine original and augmented features
            all_features = torch.cat([z_I, z_augmented], dim=0)
            all_labels = torch.cat([labels, labels], dim=0)
            
            contrastive_loss = self.contrastive_loss(all_features, all_labels)
        else:
            # Unsupervised contrastive loss
            contrastive_loss = self.contrastive_loss(z_I, z_V, labels)
        
        # Compute additional metrics
        with torch.no_grad():
            # Similarity between original and augmented
            sim_orig_aug = F.cosine_similarity(z_I, z_augmented, dim=1).mean()
            
            # Variance of features
            feature_var = torch.var(z_I, dim=0).mean()
        
        return {
            'contrastive_loss': contrastive_loss,
            'similarity_orig_aug': sim_orig_aug,
            'feature_variance': feature_var
        }

if __name__ == "__main__":
    # Example usage
    batch_size, dim = 32, 128
    num_labels = 50
    
    # Create dummy data
    z_I = torch.randn(batch_size, dim)
    z_V = torch.randn(batch_size, 64)
    mu = torch.randn(batch_size, 64)
    logvar = torch.randn(batch_size, 64)
    labels = torch.randint(0, 2, (batch_size, num_labels)).float()
    
    # Initialize trainer
    trainer = ContrastiveTrainer(temperature=0.07, use_supervised=True)
    
    # Compute losses
    loss_dict = trainer.compute_contrastive_loss(z_I, z_V, mu, logvar, labels)
    
    print("Contrastive losses:")
    for key, value in loss_dict.items():
        print(f"{key}: {value.item():.4f}")
