"""
Source-Only baseline (no adaptation)
Simple baseline for comparison in WACV 2026 submission.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class SourceOnly:
    """
    Source-Only baseline that performs no adaptation.
    Uses the pre-trained model as-is for inference.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        # Set model to eval mode
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Statistics
        self.num_samples = 0
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass without adaptation."""
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                outputs = self.model(x, return_uncertainty=True)
            else:
                outputs = {'adapted_features': self.model(x)}
            
            # Add logits if needed
            if 'logits' not in outputs:
                if hasattr(self.model, 'get_text_prototypes'):
                    text_protos = self.model.get_text_prototypes()
                    logits = 100.0 * outputs['adapted_features'] @ text_protos.T
                    outputs['logits'] = logits
                else:
                    # Create dummy logits
                    outputs['logits'] = torch.randn(x.shape[0], 1000, device=x.device)
        
        return outputs
    
    def adapt(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform no adaptation - just forward pass."""
        outputs = self.forward(x)
        
        # Add adaptation statistics (all zeros for source-only)
        outputs.update({
            'adaptation_loss': 0.0,
            'num_adapted_params': 0
        })
        
        self.num_samples += x.shape[0]
        return outputs
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions without adaptation."""
        return self.adapt(x)
    
    def reset(self):
        """Reset statistics (no model state to reset)."""
        self.num_samples = 0
    
    def get_adaptation_stats(self) -> Dict[str, float]:
        """Get adaptation statistics."""
        return {
            'adaptation_loss': 0.0,
            'num_adapted_params': 0,
            'total_samples': self.num_samples
        }


def create_source_only_baseline(model: nn.Module, config: Dict[str, Any]) -> SourceOnly:
    """
    Factory function to create Source-Only baseline.
    
    Args:
        model: Base model
        config: Configuration dictionary (unused for source-only)
    
    Returns:
        SourceOnly baseline instance
    """
    return SourceOnly(model, config)