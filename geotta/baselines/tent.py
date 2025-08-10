"""
TENT: Test Entropy Minimization for Adaptation (ICLR 2021)
Implementation for fair comparison in WACV 2026 submission.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
import copy
from collections import deque


class TENT:
    """
    Test Entropy Minimization for Adaptation.
    
    Reference: Wang et al. "Tent: Fully test-time adaptation by entropy minimization" ICLR 2021.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.config = config
        self.base_model = model
        
        # Create adapted model
        self.model = copy.deepcopy(model)
        self.model.train()  # Keep in training mode for BN adaptation
        
        # Configure for TENT adaptation
        self._setup_tent_parameters()
        
        # Optimizer for tent parameters
        self.optimizer = torch.optim.Adam(
            self.tent_params,
            lr=config.get('tent_lr', 1e-3),
            weight_decay=config.get('tent_weight_decay', 0.0)
        )
        
        # Running statistics
        self.num_samples = 0
        self.total_entropy = 0.0
        self.adaptation_history = deque(maxlen=1000)
        
    def _setup_tent_parameters(self):
        """Setup parameters for TENT adaptation (typically BN parameters)."""
        self.tent_params = []
        
        for name, param in self.model.named_parameters():
            # Adapt normalization parameters
            if 'norm' in name.lower() or 'bn' in name.lower():
                param.requires_grad_(True)
                self.tent_params.append(param)
            else:
                param.requires_grad_(False)
        
        if len(self.tent_params) == 0:
            # If no norm layers, adapt all parameters (fallback)
            for param in self.model.parameters():
                param.requires_grad_(True)
                self.tent_params.append(param)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with TENT adaptation."""
        # Get model outputs
        if hasattr(self.model, 'forward'):
            outputs = self.model(x, return_uncertainty=True)
        else:
            # Fallback for standard models
            outputs = {'adapted_features': self.model(x)}
        
        # Compute logits if needed
        if 'logits' not in outputs:
            if hasattr(self.model, 'get_text_prototypes'):
                text_protos = self.model.get_text_prototypes()
                logits = 100.0 * outputs['adapted_features'] @ text_protos.T
                outputs['logits'] = logits
            else:
                # Create dummy logits for testing
                outputs['logits'] = torch.randn(x.shape[0], 1000, device=x.device)
        
        return outputs
    
    def adapt(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform TENT adaptation on input batch."""
        # Forward pass
        outputs = self.forward(x)
        logits = outputs['logits']
        
        # Compute entropy loss
        softmax_out = F.softmax(logits, dim=1)
        entropy = -(softmax_out * F.log_softmax(logits, dim=1)).sum(dim=1)
        entropy_loss = entropy.mean()
        
        # Adapt model
        self.optimizer.zero_grad()
        entropy_loss.backward()
        self.optimizer.step()
        
        # Update statistics
        self.num_samples += x.shape[0]
        self.total_entropy += entropy.sum().item()
        self.adaptation_history.append({
            'entropy_loss': entropy_loss.item(),
            'mean_entropy': entropy.mean().item(),
            'batch_size': x.shape[0]
        })
        
        # Return adapted outputs
        with torch.no_grad():
            adapted_outputs = self.forward(x)
            adapted_outputs['adaptation_loss'] = entropy_loss.item()
            adapted_outputs['mean_entropy'] = entropy.mean().item()
        
        return adapted_outputs
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions after adaptation."""
        return self.adapt(x)
    
    def reset(self):
        """Reset adaptation state."""
        # Restore original model parameters
        self.model.load_state_dict(self.base_model.state_dict())
        
        # Reset statistics
        self.num_samples = 0
        self.total_entropy = 0.0
        self.adaptation_history.clear()
        
        # Reset optimizer
        self.optimizer = torch.optim.Adam(
            self.tent_params,
            lr=self.config.get('tent_lr', 1e-3),
            weight_decay=self.config.get('tent_weight_decay', 0.0)
        )
    
    def get_adaptation_stats(self) -> Dict[str, float]:
        """Get adaptation statistics."""
        if self.num_samples == 0:
            return {'mean_entropy': 0.0, 'adaptation_steps': 0}
        
        return {
            'mean_entropy': self.total_entropy / self.num_samples,
            'adaptation_steps': len(self.adaptation_history),
            'total_samples': self.num_samples
        }


class TENTwithMomentum(TENT):
    """
    TENT with momentum updates for more stable adaptation.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        
        # Momentum coefficient
        self.momentum = config.get('tent_momentum', 0.9)
        
        # Store original parameters for momentum updates
        self.original_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.original_params[name] = param.data.clone()
    
    def adapt(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform TENT adaptation with momentum."""
        # Standard TENT adaptation
        outputs = super().adapt(x)
        
        # Apply momentum updates
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.original_params:
                    param.data = (self.momentum * self.original_params[name] + 
                                (1 - self.momentum) * param.data)
        
        return outputs


class AdaptiveTENT(TENT):
    """
    Adaptive TENT that adjusts learning rate based on entropy.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        
        self.base_lr = config.get('tent_lr', 1e-3)
        self.adaptive_factor = config.get('adaptive_factor', 0.1)
        self.entropy_threshold = config.get('entropy_threshold', 1.0)
        
    def adapt(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform adaptive TENT adaptation."""
        # Forward pass to get current entropy
        outputs = self.forward(x)
        logits = outputs['logits']
        
        # Compute entropy
        softmax_out = F.softmax(logits, dim=1)
        entropy = -(softmax_out * F.log_softmax(logits, dim=1)).sum(dim=1)
        mean_entropy = entropy.mean().item()
        
        # Adaptive learning rate based on entropy
        if mean_entropy > self.entropy_threshold:
            lr_multiplier = 1.0 + self.adaptive_factor
        else:
            lr_multiplier = 1.0 - self.adaptive_factor
        
        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr * lr_multiplier
        
        # Perform standard TENT adaptation
        return super().adapt(x)


class TENTEnsemble:
    """
    Ensemble of TENT models for improved robustness.
    """
    
    def __init__(self, models: List[nn.Module], config: Dict[str, Any]):
        self.tent_models = []
        for model in models:
            self.tent_models.append(TENT(model, config))
        
        self.ensemble_size = len(models)
        
    def adapt(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Adapt all models and return ensemble predictions."""
        all_outputs = []
        
        for tent_model in self.tent_models:
            outputs = tent_model.adapt(x)
            all_outputs.append(outputs)
        
        # Ensemble predictions
        ensemble_logits = torch.stack([out['logits'] for out in all_outputs])
        mean_logits = ensemble_logits.mean(dim=0)
        
        # Compute ensemble uncertainty
        ensemble_variance = ensemble_logits.var(dim=0).mean(dim=1)
        
        return {
            'logits': mean_logits,
            'adapted_features': all_outputs[0]['adapted_features'],  # Use first model's features
            'uncertainty': ensemble_variance,
            'ensemble_outputs': all_outputs
        }
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make ensemble predictions."""
        return self.adapt(x)
    
    def reset(self):
        """Reset all models."""
        for tent_model in self.tent_models:
            tent_model.reset()
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics from all models."""
        all_stats = []
        for i, tent_model in enumerate(self.tent_models):
            stats = tent_model.get_adaptation_stats()
            stats['model_id'] = i
            all_stats.append(stats)
        
        return {'individual_stats': all_stats}


def create_tent_baseline(model: nn.Module, config: Dict[str, Any]) -> TENT:
    """
    Factory function to create TENT baseline.
    
    Args:
        model: Base model to adapt
        config: Configuration dictionary
    
    Returns:
        TENT adapter instance
    """
    tent_type = config.get('tent_type', 'standard')
    
    if tent_type == 'standard':
        return TENT(model, config)
    elif tent_type == 'momentum':
        return TENTwithMomentum(model, config)
    elif tent_type == 'adaptive':
        return AdaptiveTENT(model, config)
    else:
        raise ValueError(f"Unknown TENT type: {tent_type}")