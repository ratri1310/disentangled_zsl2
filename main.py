import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from feature_extraction import BiomedicalFeatureExtractor
from contrastive_learning import ContrastiveTrainer
from loss_functions import IntegratedLoss
from zero_shot_classifier import ZeroShotClassifier

class DisentangledBiomedicalClassifier(nn.Module):
    """
    Main model combining feature disentanglement and zero-shot classification.
    """
    
    def __init__(self,
                 model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                 seen_mesh_codes: List[str] = None,
                 unseen_mesh_codes: List[str] = None,
                 d1: int = 512, d2: int = 256, d3: int = 256, d4: int = 128,
                 latent_dim: int = 64,
                 lambda_elbo: float = 1.0,
                 lambda_classification: float = 1.0,
                 lambda_contrastive: float = 0.5,
                 lambda_similarity: float = 0.3,
                 temperature: float = 0.07,
                 beta_vae: float = 1.0):
        super().__init__()
        
        self.seen_mesh_codes = seen_mesh_codes or []
        self.unseen_mesh_codes = unseen_mesh_codes or []
        self.all_mesh_codes = self.seen_mesh_codes + self.unseen_mesh_codes
        
        # Feature extraction and disentanglement
        self.feature_extractor = BiomedicalFeatureExtractor(
            model_name=model_name,
            mesh_codes=self.all_mesh_codes,
            d1=d1, d2=d2, d3=d3, d4=d4,
