import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import clip
from typing import Dict, Optional, Any


class GeometricBridge(nn.Module):
    """
    Lightweight geometric adapter between CLIP modalities.
    Learns to predict and correct geometric transformations.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        dim = config['model']['bridge_dim']
        
        # Frozen CLIP encoders
        self.clip_model, self.preprocess = clip.load(config['model']['clip_model'], device='cuda')
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Get CLIP feature dimension based on model type
        self.clip_dim = self._get_clip_dimension(config['model']['clip_model'])
            
        # Geometric transformation layers (only these are trainable)
        self.image_proj = nn.Linear(self.clip_dim, dim)
        self.text_proj = nn.Linear(self.clip_dim, dim)
        
        # Cross-modal attention for geometric alignment
        self.cross_attention = nn.MultiheadAttention(
            dim, 
            config['model']['bridge_heads'],
            dropout=config['model']['dropout'],
            batch_first=True
        )
        
        # Geometric uncertainty predictor
        self.uncertainty_head = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(dim, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        # Output projection
        self.output_proj = nn.Linear(dim, self.clip_dim)
        
        # Learnable temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1) * config['model']['temperature'])
        
        # Cache for text prototypes
        self.text_prototype_cache = None
    
    def _get_clip_dimension(self, clip_model_name: str) -> int:
        """Get feature dimension for different CLIP models."""
        if 'ViT-B/32' in clip_model_name or 'ViT-B/16' in clip_model_name:
            return 512
        elif 'ViT-L/14' in clip_model_name:
            return 768  
        elif 'RN50' in clip_model_name:
            return 1024
        elif 'RN101' in clip_model_name:
            return 512
        else:
            # Default to 512 for unknown models
            return 512
        
    def forward(self, images: torch.Tensor, texts: Optional[torch.Tensor] = None, 
                return_uncertainty: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing adapted features and uncertainty.
        
        Args:
            images: Batch of images [B, 3, H, W]
            texts: Optional text tokens for training
            return_uncertainty: Whether to compute uncertainty
        """
        # Extract frozen CLIP features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = F.normalize(image_features, p=2, dim=-1)
            
            if texts is not None:
                text_features = self.clip_model.encode_text(texts)
                text_features = F.normalize(text_features, p=2, dim=-1)
            else:
                # During test-time, use cached prototypes
                text_features = self.get_cached_prototypes()
        
        # Project to bridge space
        img_bridge = self.image_proj(image_features)
        txt_bridge = self.text_proj(text_features)
        
        # Compute cross-modal attention (geometric alignment)
        aligned_img, attention_weights = self.cross_attention(
            img_bridge.unsqueeze(1),
            txt_bridge.unsqueeze(1),
            txt_bridge.unsqueeze(1)
        )
        aligned_img = aligned_img.squeeze(1)
        
        # Compute geometric uncertainty
        if return_uncertainty:
            # Concatenate original and aligned features
            uncertainty_input = torch.cat([img_bridge, aligned_img], dim=-1)
            uncertainty = self.uncertainty_head(uncertainty_input)
        else:
            uncertainty = None
            
        # Project back to CLIP space
        adapted_features = self.output_proj(aligned_img)
        adapted_features = F.normalize(adapted_features, p=2, dim=-1)
        
        return {
            'adapted_features': adapted_features,
            'uncertainty': uncertainty,
            'attention_weights': attention_weights,
            'original_image_features': image_features,
            'original_text_features': text_features if texts is not None else None
        }
    
    def get_cached_prototypes(self) -> torch.Tensor:
        """Get cached text prototypes for test-time adaptation."""
        if self.text_prototype_cache is None:
            # Create dummy prototypes - in practice these would be precomputed
            device = next(self.parameters()).device
            self.text_prototype_cache = torch.randn(1000, self.clip_dim, device=device)
            self.text_prototype_cache = F.normalize(self.text_prototype_cache, p=2, dim=-1)
        return self.text_prototype_cache
    
    def set_text_prototypes(self, prototypes: torch.Tensor):
        """Set text prototypes for test-time adaptation."""
        self.text_prototype_cache = F.normalize(prototypes, p=2, dim=-1)
    
    def get_text_prototypes(self) -> torch.Tensor:
        """Get text prototypes for classification."""
        return self.get_cached_prototypes()


def compute_logits_from_features(image_features: torch.Tensor, 
                                text_features: torch.Tensor) -> torch.Tensor:
    """Compute classification logits from features."""
    # Simple dot product in CLIP space
    logits = 100.0 * image_features @ text_features.T
    return logits