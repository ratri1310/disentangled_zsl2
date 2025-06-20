import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Dict, Tuple, Optional

class PubMedBERTEncoder(nn.Module):
    """
    PubMedBERT encoder for extracting biomedical features.
    """
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through PubMedBERT.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Dictionary containing CLS token and full sequence representations
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract CLS token (first token) for abstract representation
        cls_representation = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Keep full sequence for reconstruction
        full_sequence = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        return {
            'cls_representation': cls_representation,
            'full_sequence': full_sequence,
            'attention_mask': attention_mask
        }

class SharedTrunkNetwork(nn.Module):
    """
    Shared trunk network for initial feature processing.
    """
    
    def __init__(self, input_dim: int, d1: int = 512, d2: int = 256):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, d1)
        self.layer2 = nn.Linear(d1, d2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared trunk.
        
        Args:
            x: Input features from PubMedBERT
            
        Returns:
            Shared representations
        """
        h1 = self.relu(self.layer1(x))
        h1 = self.dropout(h1)
        h_shared = self.relu(self.layer2(h1))
        h_shared = self.dropout(h_shared)
        
        return h_shared

class DiscriminativeContentHead(nn.Module):
    """
    Head for extracting discriminative biomedical content features.
    """
    
    def __init__(self, input_dim: int, d3: int = 256, d4: int = 128):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, d3)
        self.layer2 = nn.Linear(d3, d4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, h_shared: torch.Tensor) -> torch.Tensor:
        """
        Extract discriminative content features.
        
        Args:
            h_shared: Shared trunk output
            
        Returns:
            Discriminative content features zI
        """
        d1 = self.relu(self.layer1(h_shared))
        d1 = self.dropout(d1)
        z_I = self.layer2(d1)  # No activation for final layer
        
        return z_I

class VarianceModelingHead(nn.Module):
    """
    Head for modeling linguistic variance using VAE approach.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()
        self.mu_layer = nn.Linear(input_dim, latent_dim)
        self.logvar_layer = nn.Linear(input_dim, latent_dim)
        self.latent_dim = latent_dim
        
    def forward(self, h_shared: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate variance component using reparameterization trick.
        
        Args:
            h_shared: Shared trunk output
            
        Returns:
            Tuple of (z_V, mu, logvar)
        """
        mu = self.mu_layer(h_shared)
        logvar = self.logvar_layer(h_shared)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_V = mu + eps * std
        
        return z_V, mu, logvar

class ReconstructionDecoder(nn.Module):
    """
    Decoder for reconstructing original features from combined latent representation.
    """
    
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.layer1 = nn.Linear(latent_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, z_combined: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct original features.
        
        Args:
            z_combined: Combined latent representation
            
        Returns:
            Reconstructed features
        """
        h1 = self.relu(self.layer1(z_combined))
        h1 = self.dropout(h1)
        reconstructed = self.layer2(h1)
        
        return reconstructed

class MeSHAnchorEmbeddings(nn.Module):
    """
    Creates and manages MeSH class anchor embeddings.
    """
    
    def __init__(self, mesh_codes: list, embedding_dim: int, encoder: PubMedBERTEncoder):
        super().__init__()
        self.mesh_codes = mesh_codes
        self.embedding_dim = embedding_dim
        self.encoder = encoder
        self.anchor_embeddings = nn.Parameter(torch.randn(len(mesh_codes), embedding_dim))
        
        # Create code to index mapping
        self.code_to_idx = {code: idx for idx, code in enumerate(mesh_codes)}
        
    def create_anchors_from_definitions(self, mesh_definitions: Dict[str, torch.Tensor]):
        """
        Create anchor embeddings from MeSH definitions.
        
        Args:
            mesh_definitions: Tokenized MeSH definitions
        """
        with torch.no_grad():
            for code, tokenized_def in mesh_definitions.items():
                if code in self.code_to_idx:
                    idx = self.code_to_idx[code]
                    # Extract CLS representation for the definition
                    outputs = self.encoder(**tokenized_def)
                    self.anchor_embeddings[idx] = outputs['cls_representation'].squeeze()
    
    def get_anchor(self, mesh_code: str) -> torch.Tensor:
        """Get anchor embedding for a specific MeSH code."""
        idx = self.code_to_idx[mesh_code]
        return self.anchor_embeddings[idx]
    
    def get_all_anchors(self) -> torch.Tensor:
        """Get all anchor embeddings."""
        return self.anchor_embeddings

class BiomedicalFeatureExtractor(nn.Module):
    """
    Complete feature extraction module combining all components.
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                 mesh_codes: list = None,
                 d1: int = 512, d2: int = 256, d3: int = 256, d4: int = 128,
                 latent_dim: int = 64):
        super().__init__()
        
        # Initialize encoder
        self.encoder = PubMedBERTEncoder(model_name)
        
        # Initialize components
        self.shared_trunk = SharedTrunkNetwork(self.encoder.hidden_size, d1, d2)
        self.discriminative_head = DiscriminativeContentHead(d2, d3, d4)
        self.variance_head = VarianceModelingHead(d2, latent_dim)
        self.decoder = ReconstructionDecoder(d4 + latent_dim, self.encoder.hidden_size)
        
        # Initialize MeSH anchors if provided
        if mesh_codes:
            self.mesh_anchors = MeSHAnchorEmbeddings(mesh_codes, d4, self.encoder)
        else:
            self.mesh_anchors = None
            
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Dictionary containing all extracted features
        """
        # Extract base features
        encoder_outputs = self.encoder(input_ids, attention_mask)
        base_features = encoder_outputs['cls_representation']
        
        # Shared trunk processing
        h_shared = self.shared_trunk(base_features)
        
        # Discriminative content features
        z_I = self.discriminative_head(h_shared)
        
        # Variance modeling
        z_V, mu, logvar = self.variance_head(h_shared)
        
        # Combined representation
        z_combined = z_I + z_V
        
        # Reconstruction
        reconstructed = self.decoder(z_combined)
        
        return {
            'z_I': z_I,
            'z_V': z_V,
            'mu': mu,
            'logvar': logvar,
            'z_combined': z_combined,
            'reconstructed': reconstructed,
            'original_features': base_features,
            'full_sequence': encoder_outputs['full_sequence']
        }

if __name__ == "__main__":
    # Example usage
    model = BiomedicalFeatureExtractor()
    
    # Dummy input
    batch_size, seq_len = 2, 512
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    outputs = model(input_ids, attention_mask)
    
    print("Output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
