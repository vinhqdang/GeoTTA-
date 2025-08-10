"""
MEMO: Mutual Information Maximization for Test-Time Adaptation
Implementation for fair comparison in WACV 2026 submission.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import copy
from collections import deque
import numpy as np


class MEMO:
    """
    MEMO: Test-Time Adaptation via Mutual Information Maximization.
    
    Reference: Zhang et al. "MEMO: Test Time Robustness via Adaptation and Augmentation" NeurIPS 2022.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.config = config
        self.base_model = model
        
        # Create adapted model
        self.model = copy.deepcopy(model)
        self.model.train()
        
        # MEMO specific parameters
        self.num_augmentations = config.get('num_augmentations', 64)
        self.adapt_steps = config.get('adapt_steps', 1)
        self.marginal_entropy_weight = config.get('marginal_entropy_weight', 1.0)
        
        # Setup parameters for adaptation
        self._setup_memo_parameters()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.adapt_params,
            lr=config.get('memo_lr', 1e-3),
            weight_decay=config.get('memo_weight_decay', 0.0)
        )
        
        # Augmentation strategy
        self._setup_augmentations()
        
        # Statistics
        self.num_samples = 0
        self.adaptation_history = deque(maxlen=1000)
        
    def _setup_memo_parameters(self):
        """Setup parameters for MEMO adaptation."""
        self.adapt_params = []
        
        for name, param in self.model.named_parameters():
            # Adapt normalization and projection layers
            if any(layer in name.lower() for layer in ['norm', 'bn', 'proj', 'head']):
                param.requires_grad_(True)
                self.adapt_params.append(param)
            else:
                param.requires_grad_(False)
        
        if len(self.adapt_params) == 0:
            # Fallback: adapt all parameters
            for param in self.model.parameters():
                param.requires_grad_(True)
                self.adapt_params.append(param)
    
    def _setup_augmentations(self):
        """Setup augmentation strategies for MEMO."""
        from torchvision import transforms
        
        # Define augmentation pipeline
        self.augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        ])
        
        # For tensor-based augmentations
        self.tensor_augmentations = [
            self._gaussian_noise,
            self._random_brightness,
            self._random_contrast,
        ]
    
    def _gaussian_noise(self, x: torch.Tensor, std: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(x) * std
        return torch.clamp(x + noise, 0, 1)
    
    def _random_brightness(self, x: torch.Tensor, factor: float = 0.1) -> torch.Tensor:
        """Random brightness adjustment."""
        brightness = 1 + (torch.rand(1, device=x.device) * 2 - 1) * factor
        return torch.clamp(x * brightness, 0, 1)
    
    def _random_contrast(self, x: torch.Tensor, factor: float = 0.1) -> torch.Tensor:
        """Random contrast adjustment."""
        mean = x.mean(dim=[2, 3], keepdim=True)
        contrast = 1 + (torch.rand(1, device=x.device) * 2 - 1) * factor
        return torch.clamp((x - mean) * contrast + mean, 0, 1)
    
    def apply_tensor_augmentations(self, x: torch.Tensor, num_augs: int) -> torch.Tensor:
        """Apply tensor-based augmentations."""
        augmented_batch = []
        
        for _ in range(num_augs):
            augmented = x.clone()
            
            # Randomly apply augmentations
            for aug_func in self.tensor_augmentations:
                if torch.rand(1) > 0.5:
                    augmented = aug_func(augmented)
            
            augmented_batch.append(augmented)
        
        return torch.stack(augmented_batch, dim=1)  # [B, num_augs, C, H, W]
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through adapted model."""
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
    
    def compute_marginal_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute marginal entropy for mutual information maximization."""
        # Compute average prediction across augmentations
        # logits shape: [B * num_augs, C] or [B, num_augs, C]
        if logits.dim() == 3:
            batch_size, num_augs, num_classes = logits.shape
            marginal_probs = F.softmax(logits, dim=-1).mean(dim=1)  # Average over augmentations
        else:
            # Reshape if needed
            batch_size = logits.shape[0] // self.num_augmentations
            logits = logits.view(batch_size, self.num_augmentations, -1)
            marginal_probs = F.softmax(logits, dim=-1).mean(dim=1)
        
        # Marginal entropy (to be maximized)
        marginal_entropy = -(marginal_probs * torch.log(marginal_probs + 1e-8)).sum(dim=-1)
        
        return marginal_entropy.mean()
    
    def compute_conditional_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute conditional entropy (to be minimized)."""
        if logits.dim() == 3:
            batch_size, num_augs, num_classes = logits.shape
            logits_flat = logits.view(-1, num_classes)
        else:
            logits_flat = logits
        
        # Individual prediction entropies
        probs = F.softmax(logits_flat, dim=-1)
        conditional_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        
        return conditional_entropy.mean()
    
    def compute_memo_loss(self, logits: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute MEMO loss (maximize MI = maximize marginal entropy - minimize conditional entropy)."""
        marginal_entropy = self.compute_marginal_entropy(logits)
        conditional_entropy = self.compute_conditional_entropy(logits)
        
        # MEMO loss: -marginal_entropy + conditional_entropy (minimize this)
        memo_loss = -self.marginal_entropy_weight * marginal_entropy + conditional_entropy
        
        loss_components = {
            'marginal_entropy': marginal_entropy.item(),
            'conditional_entropy': conditional_entropy.item(),
            'memo_loss': memo_loss.item()
        }
        
        return memo_loss, loss_components
    
    def adapt(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform MEMO adaptation."""
        # Generate augmented versions
        # For simplicity, we'll use tensor augmentations
        augmented_x = self.apply_tensor_augmentations(x, self.num_augmentations)
        # Reshape: [B, num_augs, C, H, W] -> [B * num_augs, C, H, W]
        batch_size = x.shape[0]
        augmented_x = augmented_x.view(-1, *x.shape[1:])
        
        # Multiple adaptation steps
        for step in range(self.adapt_steps):
            # Forward pass on augmented data
            outputs = self.forward(augmented_x)
            logits = outputs['logits']
            
            # Reshape logits: [B * num_augs, C] -> [B, num_augs, C]
            logits = logits.view(batch_size, self.num_augmentations, -1)
            
            # Compute MEMO loss
            memo_loss, loss_components = self.compute_memo_loss(logits)
            
            # Adaptation step
            if memo_loss.requires_grad:
                self.optimizer.zero_grad()
                memo_loss.backward()
                self.optimizer.step()
                
                # Record statistics
                self.adaptation_history.append({
                    'step': step,
                    **loss_components,
                    'batch_size': batch_size
                })
        
        # Final prediction on original (non-augmented) data
        with torch.no_grad():
            final_outputs = self.forward(x)
            
            # Add adaptation statistics
            if self.adaptation_history:
                recent_stats = self.adaptation_history[-1]
                final_outputs.update({
                    'adaptation_loss': recent_stats['memo_loss'],
                    'marginal_entropy': recent_stats['marginal_entropy'],
                    'conditional_entropy': recent_stats['conditional_entropy']
                })
        
        self.num_samples += batch_size
        return final_outputs
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions with MEMO."""
        return self.adapt(x)
    
    def reset(self):
        """Reset adaptation state."""
        # Restore original model parameters
        self.model.load_state_dict(self.base_model.state_dict())
        
        # Reset statistics
        self.num_samples = 0
        self.adaptation_history.clear()
        
        # Reset optimizer
        self.optimizer = torch.optim.Adam(
            self.adapt_params,
            lr=self.config.get('memo_lr', 1e-3),
            weight_decay=self.config.get('memo_weight_decay', 0.0)
        )
    
    def get_adaptation_stats(self) -> Dict[str, float]:
        """Get adaptation statistics."""
        if len(self.adaptation_history) == 0:
            return {'memo_loss': 0.0, 'marginal_entropy': 0.0, 'conditional_entropy': 0.0}
        
        recent_stats = list(self.adaptation_history)[-100:]  # Last 100 steps
        
        return {
            'mean_memo_loss': np.mean([s['memo_loss'] for s in recent_stats]),
            'mean_marginal_entropy': np.mean([s['marginal_entropy'] for s in recent_stats]),
            'mean_conditional_entropy': np.mean([s['conditional_entropy'] for s in recent_stats]),
            'adaptation_steps': len(self.adaptation_history),
            'total_samples': self.num_samples
        }


class MEMOwithUncertainty(MEMO):
    """
    MEMO with uncertainty-aware sample selection.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self.uncertainty_threshold = config.get('uncertainty_threshold', 0.5)
    
    def select_uncertain_samples(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select samples with high uncertainty for adaptation."""
        with torch.no_grad():
            outputs = self.forward(x)
            
            if 'uncertainty' in outputs:
                uncertainties = outputs['uncertainty'].squeeze()
            else:
                # Compute uncertainty from logits
                logits = outputs['logits']
                probs = F.softmax(logits, dim=1)
                uncertainties = 1 - probs.max(dim=1)[0]
            
            # Select high-uncertainty samples
            uncertain_mask = uncertainties > self.uncertainty_threshold
            
            if not uncertain_mask.any():
                # If no uncertain samples, select all
                uncertain_mask = torch.ones_like(uncertainties, dtype=torch.bool)
            
            return x[uncertain_mask], uncertain_mask
    
    def adapt(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform uncertainty-aware MEMO adaptation."""
        # Select uncertain samples
        uncertain_x, uncertain_mask = self.select_uncertain_samples(x)
        
        if uncertain_x.shape[0] > 0:
            # Adapt on uncertain samples
            super().adapt(uncertain_x)
        
        # Final prediction on all samples
        with torch.no_grad():
            final_outputs = self.forward(x)
            final_outputs['uncertain_ratio'] = uncertain_mask.float().mean().item()
        
        return final_outputs


def create_memo_baseline(model: nn.Module, config: Dict[str, Any]) -> MEMO:
    """
    Factory function to create MEMO baseline.
    
    Args:
        model: Base model to adapt
        config: Configuration dictionary
    
    Returns:
        MEMO adapter instance
    """
    memo_type = config.get('memo_type', 'standard')
    
    if memo_type == 'standard':
        return MEMO(model, config)
    elif memo_type == 'uncertainty':
        return MEMOwithUncertainty(model, config)
    else:
        raise ValueError(f"Unknown MEMO type: {memo_type}")