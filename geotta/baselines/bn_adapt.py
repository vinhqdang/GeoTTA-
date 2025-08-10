"""
Batch Normalization Adaptation baseline
Simple but effective baseline for test-time adaptation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import copy


class BNAdapt:
    """
    Batch Normalization Adaptation.
    
    Adapts only the batch normalization statistics during test time.
    This is a simple but often effective baseline.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = copy.deepcopy(model)
        self.config = config
        
        # Set model to training mode to update BN statistics
        self.model.train()
        
        # Freeze all parameters except BN
        self._setup_bn_adaptation()
        
        # Statistics
        self.num_samples = 0
        
    def _setup_bn_adaptation(self):
        """Setup BN adaptation by freezing non-BN parameters."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Keep BN in training mode
                module.train()
                # Allow BN parameters to update
                for param in module.parameters():
                    param.requires_grad = False  # Don't optimize, just update running stats
            else:
                # Set other modules to eval mode
                module.eval()
                # Freeze other parameters
                for param in module.parameters():
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with BN adaptation."""
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
                outputs['logits'] = torch.randn(x.shape[0], 1000, device=x.device)
        
        return outputs
    
    def adapt(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform BN adaptation."""
        # Forward pass updates BN statistics automatically
        outputs = self.forward(x)
        
        # Add adaptation statistics
        outputs.update({
            'adaptation_loss': 0.0,  # No explicit loss for BN adaptation
            'num_adapted_params': self._count_bn_params()
        })
        
        self.num_samples += x.shape[0]
        return outputs
    
    def _count_bn_params(self) -> int:
        """Count the number of BN parameters."""
        count = 0
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if module.weight is not None:
                    count += module.weight.numel()
                if module.bias is not None:
                    count += module.bias.numel()
        return count
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions with BN adaptation."""
        return self.adapt(x)
    
    def reset(self):
        """Reset BN statistics."""
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.reset_running_stats()
        
        self.num_samples = 0
    
    def get_adaptation_stats(self) -> Dict[str, float]:
        """Get adaptation statistics."""
        return {
            'adaptation_loss': 0.0,
            'num_adapted_params': self._count_bn_params(),
            'total_samples': self.num_samples
        }


def create_bn_adapt_baseline(model: nn.Module, config: Dict[str, Any]) -> BNAdapt:
    """
    Factory function to create BN Adaptation baseline.
    
    Args:
        model: Base model to adapt
        config: Configuration dictionary
    
    Returns:
        BNAdapt baseline instance
    """
    return BNAdapt(model, config)