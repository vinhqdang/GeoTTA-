"""
CoTTA: Continual Test-Time Adaptation (CVPR 2022)
Implementation for fair comparison in WACV 2026 submission.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
import copy
from collections import deque
import numpy as np


class CoTTA:
    """
    Continual Test-Time Adaptation via Self-Training and Stochastic Restoration.
    
    Reference: Wang et al. "Continual Test-Time Domain Adaptation" CVPR 2022.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.config = config
        self.base_model = model
        
        # Create adapted model and anchor model
        self.model = copy.deepcopy(model)
        self.anchor_model = copy.deepcopy(model)
        
        # CoTTA specific parameters
        self.restoration_factor = config.get('restoration_factor', 0.01)
        self.threshold_ent = config.get('threshold_ent', 0.4)
        self.threshold_cov = config.get('threshold_cov', 0.95)
        self.mt_alpha = config.get('mt_alpha', 0.99)  # Moving average coefficient
        
        # Setup parameters for adaptation
        self._setup_cotta_parameters()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.adapt_params,
            lr=config.get('cotta_lr', 1e-3),
            betas=(0.9, 0.999),
            weight_decay=config.get('cotta_weight_decay', 0.0)
        )
        
        # Statistics and buffers
        self.num_samples = 0
        self.adaptation_history = deque(maxlen=1000)
        self.prototype_memory = deque(maxlen=config.get('memory_size', 64))
        
        # For teacher model updates
        self._setup_teacher_model()
        
    def _setup_cotta_parameters(self):
        """Setup parameters for CoTTA adaptation."""
        self.adapt_params = []
        
        for name, param in self.model.named_parameters():
            # Adapt normalization layers
            if 'norm' in name.lower() or 'bn' in name.lower():
                param.requires_grad_(True)
                self.adapt_params.append(param)
            else:
                param.requires_grad_(False)
                
        if len(self.adapt_params) == 0:
            # Fallback: adapt all parameters
            for param in self.model.parameters():
                param.requires_grad_(True)
                self.adapt_params.append(param)
    
    def _setup_teacher_model(self):
        """Setup teacher model for self-training."""
        self.teacher_model = copy.deepcopy(self.model)
        
        # Initialize teacher with anchor weights
        self.update_teacher_model(init=True)
    
    def update_teacher_model(self, init: bool = False):
        """Update teacher model with exponential moving average."""
        if init:
            # Initialize teacher with current model
            self.teacher_model.load_state_dict(self.model.state_dict())
        else:
            # EMA update
            with torch.no_grad():
                for param_t, param_s in zip(self.teacher_model.parameters(), 
                                          self.model.parameters()):
                    param_t.data = (self.mt_alpha * param_t.data + 
                                  (1 - self.mt_alpha) * param_s.data)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through student model."""
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
    
    def teacher_forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through teacher model."""
        self.teacher_model.eval()
        
        with torch.no_grad():
            if hasattr(self.teacher_model, 'forward'):
                outputs = self.teacher_model(x, return_uncertainty=True)
            else:
                outputs = {'adapted_features': self.teacher_model(x)}
            
            # Add logits if needed
            if 'logits' not in outputs:
                if hasattr(self.teacher_model, 'get_text_prototypes'):
                    text_protos = self.teacher_model.get_text_prototypes()
                    logits = 100.0 * outputs['adapted_features'] @ text_protos.T
                    outputs['logits'] = logits
                else:
                    outputs['logits'] = torch.randn(x.shape[0], 1000, device=x.device)
        
        return outputs
    
    def compute_cotta_loss(self, student_outputs: Dict[str, torch.Tensor],
                          teacher_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute CoTTA loss combining entropy and consistency."""
        student_logits = student_outputs['logits']
        teacher_logits = teacher_outputs['logits']
        
        # Entropy loss
        softmax_out = F.softmax(student_logits, dim=1)
        entropy = -(softmax_out * F.log_softmax(student_logits, dim=1)).sum(dim=1)
        entropy_loss = entropy.mean()
        
        # Teacher predictions for consistency
        teacher_probs = F.softmax(teacher_logits, dim=1)
        
        # Filter reliable predictions based on entropy and confidence
        max_probs, _ = teacher_probs.max(dim=1)
        entropy_mask = entropy < self.threshold_ent
        confidence_mask = max_probs > self.threshold_cov
        reliable_mask = entropy_mask & confidence_mask
        
        # Consistency loss (only for reliable samples)
        consistency_loss = 0.0
        if reliable_mask.any():
            student_log_probs = F.log_softmax(student_logits[reliable_mask], dim=1)
            consistency_loss = F.kl_div(
                student_log_probs, 
                teacher_probs[reliable_mask], 
                reduction='mean'
            )
        
        # Combined loss
        total_loss = entropy_loss + consistency_loss
        
        return total_loss, entropy_loss, consistency_loss, reliable_mask.float().mean()
    
    def stochastic_restore(self):
        """Perform stochastic restoration to prevent error accumulation."""
        with torch.no_grad():
            for param_current, param_anchor in zip(self.model.parameters(), 
                                                 self.anchor_model.parameters()):
                # Stochastic restoration
                if torch.rand(1) < self.restoration_factor:
                    param_current.data = param_anchor.data.clone()
    
    def adapt(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform CoTTA adaptation."""
        # Set models to appropriate modes
        self.model.train()
        self.teacher_model.eval()
        
        # Forward passes
        student_outputs = self.forward(x)
        teacher_outputs = self.teacher_forward(x)
        
        # Compute losses
        total_loss, entropy_loss, consistency_loss, reliable_ratio = self.compute_cotta_loss(
            student_outputs, teacher_outputs
        )
        
        # Adaptation step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Update teacher model
        self.update_teacher_model()
        
        # Stochastic restoration
        self.stochastic_restore()
        
        # Update statistics
        self.num_samples += x.shape[0]
        self.adaptation_history.append({
            'total_loss': total_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'consistency_loss': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss,
            'reliable_ratio': reliable_ratio.item(),
            'batch_size': x.shape[0]
        })
        
        # Return final outputs
        with torch.no_grad():
            final_outputs = self.forward(x)
            final_outputs.update({
                'adaptation_loss': total_loss.item(),
                'entropy_loss': entropy_loss.item(),
                'consistency_loss': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss,
                'reliable_ratio': reliable_ratio.item()
            })
        
        return final_outputs
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions with CoTTA."""
        return self.adapt(x)
    
    def reset(self):
        """Reset adaptation state."""
        # Restore original model parameters
        self.model.load_state_dict(self.base_model.state_dict())
        self.anchor_model.load_state_dict(self.base_model.state_dict())
        
        # Reset teacher model
        self.update_teacher_model(init=True)
        
        # Reset statistics
        self.num_samples = 0
        self.adaptation_history.clear()
        self.prototype_memory.clear()
        
        # Reset optimizer
        self.optimizer = torch.optim.Adam(
            self.adapt_params,
            lr=self.config.get('cotta_lr', 1e-3),
            betas=(0.9, 0.999),
            weight_decay=self.config.get('cotta_weight_decay', 0.0)
        )
    
    def get_adaptation_stats(self) -> Dict[str, float]:
        """Get adaptation statistics."""
        if len(self.adaptation_history) == 0:
            return {'total_loss': 0.0, 'entropy_loss': 0.0, 'consistency_loss': 0.0}
        
        recent_stats = list(self.adaptation_history)[-100:]  # Last 100 batches
        
        return {
            'mean_total_loss': np.mean([s['total_loss'] for s in recent_stats]),
            'mean_entropy_loss': np.mean([s['entropy_loss'] for s in recent_stats]),
            'mean_consistency_loss': np.mean([s['consistency_loss'] for s in recent_stats]),
            'mean_reliable_ratio': np.mean([s['reliable_ratio'] for s in recent_stats]),
            'adaptation_steps': len(self.adaptation_history),
            'total_samples': self.num_samples
        }


class CoTTAwithMemory(CoTTA):
    """
    CoTTA with prototype memory for better consistency.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self.memory_size = config.get('memory_size', 64)
        self.memory_update_freq = config.get('memory_update_freq', 10)
        self.memory_counter = 0
        
    def update_memory(self, features: torch.Tensor, predictions: torch.Tensor):
        """Update prototype memory with reliable samples."""
        with torch.no_grad():
            probs = F.softmax(predictions, dim=1)
            max_probs, pred_labels = probs.max(dim=1)
            
            # Only store high-confidence samples
            confidence_threshold = 0.9
            reliable_mask = max_probs > confidence_threshold
            
            if reliable_mask.any():
                reliable_features = features[reliable_mask]
                reliable_labels = pred_labels[reliable_mask]
                
                for feat, label in zip(reliable_features, reliable_labels):
                    self.prototype_memory.append({
                        'feature': feat.cpu(),
                        'label': label.cpu(),
                        'confidence': max_probs[reliable_mask].mean().cpu()
                    })
    
    def memory_consistency_loss(self, current_features: torch.Tensor, 
                               current_predictions: torch.Tensor) -> torch.Tensor:
        """Compute consistency loss with memory prototypes."""
        if len(self.prototype_memory) == 0:
            return torch.tensor(0.0, device=current_features.device)
        
        # Sample from memory
        memory_size = min(len(self.prototype_memory), 32)
        memory_samples = np.random.choice(list(self.prototype_memory), memory_size, replace=False)
        
        memory_features = torch.stack([s['feature'] for s in memory_samples]).to(current_features.device)
        memory_labels = torch.stack([s['label'] for s in memory_samples]).to(current_features.device)
        
        # Compute similarities
        similarities = torch.cosine_similarity(
            current_features.unsqueeze(1), 
            memory_features.unsqueeze(0), 
            dim=2
        )
        
        # Find most similar memory samples
        max_similarities, similar_indices = similarities.max(dim=1)
        similar_labels = memory_labels[similar_indices]
        
        # Only apply consistency loss for highly similar samples
        similarity_threshold = 0.8
        similar_mask = max_similarities > similarity_threshold
        
        if similar_mask.any():
            current_logits = current_predictions[similar_mask]
            target_labels = similar_labels[similar_mask]
            
            consistency_loss = F.cross_entropy(current_logits, target_labels)
            return consistency_loss
        
        return torch.tensor(0.0, device=current_features.device)
    
    def adapt(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Adapt with memory-augmented CoTTA."""
        # Standard CoTTA adaptation
        outputs = super().adapt(x)
        
        # Update memory periodically
        self.memory_counter += 1
        if self.memory_counter % self.memory_update_freq == 0:
            self.update_memory(outputs['adapted_features'], outputs['logits'])
        
        # Add memory consistency loss
        memory_loss = self.memory_consistency_loss(
            outputs['adapted_features'], outputs['logits']
        )
        
        if memory_loss.item() > 0:
            # Additional adaptation step with memory loss
            self.optimizer.zero_grad()
            memory_loss.backward()
            self.optimizer.step()
            
            outputs['memory_loss'] = memory_loss.item()
        else:
            outputs['memory_loss'] = 0.0
        
        return outputs


def create_cotta_baseline(model: nn.Module, config: Dict[str, Any]) -> CoTTA:
    """
    Factory function to create CoTTA baseline.
    
    Args:
        model: Base model to adapt
        config: Configuration dictionary
    
    Returns:
        CoTTA adapter instance
    """
    cotta_type = config.get('cotta_type', 'standard')
    
    if cotta_type == 'standard':
        return CoTTA(model, config)
    elif cotta_type == 'memory':
        return CoTTAwithMemory(model, config)
    else:
        raise ValueError(f"Unknown CoTTA type: {cotta_type}")