"""
AdaContrast: Adaptive Contrastive Learning for Test-Time Adaptation
Implementation for fair comparison in WACV 2026 submission.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
import copy
from collections import deque
import numpy as np


class AdaContrast:
    """
    Adaptive Contrastive Learning for Test-Time Adaptation.
    
    Reference: Chen et al. "Contrastive Test-Time Adaptation" CVPR 2022.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.config = config
        self.base_model = model
        
        # Create adapted model
        self.model = copy.deepcopy(model)
        self.model.train()
        
        # AdaContrast specific parameters
        self.temperature = config.get('contrast_temperature', 0.1)
        self.queue_size = config.get('queue_size', 256)
        self.momentum = config.get('momentum', 0.99)
        self.threshold = config.get('threshold', 0.7)
        
        # Setup parameters for adaptation
        self._setup_contrast_parameters()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.adapt_params,
            lr=config.get('ada_lr', 1e-3),
            weight_decay=config.get('ada_weight_decay', 0.0)
        )
        
        # Feature queue for contrastive learning
        self.feature_dim = config.get('feature_dim', 512)
        self.register_buffer('feature_queue', torch.randn(self.feature_dim, self.queue_size))
        self.register_buffer('label_queue', torch.zeros(self.queue_size, dtype=torch.long))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
        # Normalize feature queue
        self.feature_queue = F.normalize(self.feature_queue, dim=0)
        
        # Statistics
        self.num_samples = 0
        self.adaptation_history = deque(maxlen=1000)
        
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register buffer for queue management."""
        setattr(self, name, tensor.cuda())
        
    def _setup_contrast_parameters(self):
        """Setup parameters for contrastive adaptation."""
        self.adapt_params = []
        
        for name, param in self.model.named_parameters():
            # Adapt specific layers (e.g., projection head, normalization)
            if any(layer in name.lower() for layer in ['proj', 'norm', 'bn']):
                param.requires_grad_(True)
                self.adapt_params.append(param)
            else:
                param.requires_grad_(False)
        
        if len(self.adapt_params) == 0:
            # Fallback: adapt all parameters
            for param in self.model.parameters():
                param.requires_grad_(True)
                self.adapt_params.append(param)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through adapted model."""
        if hasattr(self.model, 'forward'):
            outputs = self.model(x, return_uncertainty=True)
        else:
            outputs = {'adapted_features': self.model(x)}
        
        # Ensure we have logits
        if 'logits' not in outputs:
            if hasattr(self.model, 'get_text_prototypes'):
                text_protos = self.model.get_text_prototypes()
                logits = 100.0 * outputs['adapted_features'] @ text_protos.T
                outputs['logits'] = logits
            else:
                outputs['logits'] = torch.randn(x.shape[0], 1000, device=x.device)
        
        return outputs
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, features: torch.Tensor, labels: torch.Tensor):
        """Update feature queue with new features."""
        batch_size = features.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace the features at ptr (dequeue and enqueue)
        self.feature_queue[:, ptr:ptr + batch_size] = features.T
        self.label_queue[ptr:ptr + batch_size] = labels
        
        # Move pointer
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def get_pseudo_labels(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate pseudo labels for high-confidence predictions."""
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            max_probs, pseudo_labels = probs.max(dim=1)
            
            # Select high-confidence samples
            confident_mask = max_probs > self.threshold
            
            return pseudo_labels, confident_mask
    
    def compute_contrastive_loss(self, features: torch.Tensor, 
                               pseudo_labels: torch.Tensor,
                               confident_mask: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss using feature queue."""
        if not confident_mask.any():
            return torch.tensor(0.0, device=features.device)
        
        # Select confident features
        confident_features = features[confident_mask]
        confident_labels = pseudo_labels[confident_mask]
        
        # Normalize features
        confident_features = F.normalize(confident_features, dim=1)
        
        # Compute similarity with queue
        sim_matrix = torch.matmul(confident_features, self.feature_queue)
        sim_matrix = sim_matrix / self.temperature
        
        # Create positive/negative masks
        labels_expanded = confident_labels.unsqueeze(1).expand(-1, self.queue_size)
        queue_labels_expanded = self.label_queue.unsqueeze(0).expand(confident_features.shape[0], -1)
        
        positive_mask = (labels_expanded == queue_labels_expanded).float()
        negative_mask = 1 - positive_mask
        
        # Compute contrastive loss
        exp_sim = torch.exp(sim_matrix)
        
        # Positive similarities
        pos_sim = (exp_sim * positive_mask).sum(dim=1)
        
        # All similarities (positive + negative)
        all_sim = exp_sim.sum(dim=1)
        
        # Contrastive loss
        loss = -torch.log(pos_sim / (all_sim + 1e-8))
        loss = loss.mean()
        
        return loss
    
    def compute_entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy minimization loss."""
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * F.log_softmax(logits, dim=1)).sum(dim=1)
        return entropy.mean()
    
    def adapt(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform AdaContrast adaptation."""
        # Forward pass
        outputs = self.forward(x)
        features = outputs['adapted_features']
        logits = outputs['logits']
        
        # Get pseudo labels
        pseudo_labels, confident_mask = self.get_pseudo_labels(logits)
        
        # Compute losses
        entropy_loss = self.compute_entropy_loss(logits)
        
        contrastive_loss = self.compute_contrastive_loss(
            features, pseudo_labels, confident_mask
        )
        
        # Combined loss
        total_loss = entropy_loss + contrastive_loss
        
        # Adaptation step
        if total_loss.requires_grad:
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        # Update queue with confident samples
        if confident_mask.any():
            with torch.no_grad():
                confident_features = F.normalize(features[confident_mask], dim=1)
                confident_pseudo_labels = pseudo_labels[confident_mask]
                self._dequeue_and_enqueue(confident_features, confident_pseudo_labels)
        
        # Update statistics
        self.num_samples += x.shape[0]
        self.adaptation_history.append({
            'total_loss': total_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'confident_ratio': confident_mask.float().mean().item(),
            'batch_size': x.shape[0]
        })
        
        # Return final outputs
        with torch.no_grad():
            final_outputs = self.forward(x)
            final_outputs.update({
                'adaptation_loss': total_loss.item(),
                'entropy_loss': entropy_loss.item(),
                'contrastive_loss': contrastive_loss.item(),
                'confident_ratio': confident_mask.float().mean().item()
            })
        
        return final_outputs
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions with AdaContrast."""
        return self.adapt(x)
    
    def reset(self):
        """Reset adaptation state."""
        # Restore original model parameters
        self.model.load_state_dict(self.base_model.state_dict())
        
        # Reset feature queue
        self.feature_queue = F.normalize(torch.randn(self.feature_dim, self.queue_size).cuda(), dim=0)
        self.label_queue = torch.zeros(self.queue_size, dtype=torch.long).cuda()
        self.queue_ptr = torch.zeros(1, dtype=torch.long).cuda()
        
        # Reset statistics
        self.num_samples = 0
        self.adaptation_history.clear()
        
        # Reset optimizer
        self.optimizer = torch.optim.Adam(
            self.adapt_params,
            lr=self.config.get('ada_lr', 1e-3),
            weight_decay=self.config.get('ada_weight_decay', 0.0)
        )
    
    def get_adaptation_stats(self) -> Dict[str, float]:
        """Get adaptation statistics."""
        if len(self.adaptation_history) == 0:
            return {'total_loss': 0.0, 'entropy_loss': 0.0, 'contrastive_loss': 0.0}
        
        recent_stats = list(self.adaptation_history)[-100:]  # Last 100 batches
        
        return {
            'mean_total_loss': np.mean([s['total_loss'] for s in recent_stats]),
            'mean_entropy_loss': np.mean([s['entropy_loss'] for s in recent_stats]),
            'mean_contrastive_loss': np.mean([s['contrastive_loss'] for s in recent_stats]),
            'mean_confident_ratio': np.mean([s['confident_ratio'] for s in recent_stats]),
            'adaptation_steps': len(self.adaptation_history),
            'total_samples': self.num_samples
        }


class AdaContrastWithMemory(AdaContrast):
    """
    AdaContrast with additional episodic memory for better stability.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        
        # Episodic memory parameters
        self.memory_size = config.get('memory_size', 1024)
        self.memory_update_freq = config.get('memory_update_freq', 10)
        self.memory_counter = 0
        
        # Initialize episodic memory
        self.episodic_memory = deque(maxlen=self.memory_size)
        
    def update_episodic_memory(self, features: torch.Tensor, 
                             pseudo_labels: torch.Tensor, 
                             confident_mask: torch.Tensor):
        """Update episodic memory with reliable samples."""
        if not confident_mask.any():
            return
        
        confident_features = features[confident_mask]
        confident_labels = pseudo_labels[confident_mask]
        
        # Add to episodic memory
        for feat, label in zip(confident_features, confident_labels):
            self.episodic_memory.append({
                'feature': feat.detach().cpu(),
                'label': label.detach().cpu()
            })
    
    def compute_memory_loss(self, current_features: torch.Tensor) -> torch.Tensor:
        """Compute loss using episodic memory."""
        if len(self.episodic_memory) == 0:
            return torch.tensor(0.0, device=current_features.device)
        
        # Sample from memory
        memory_size = min(len(self.episodic_memory), 64)
        memory_samples = np.random.choice(list(self.episodic_memory), memory_size, replace=False)
        
        memory_features = torch.stack([s['feature'] for s in memory_samples]).to(current_features.device)
        memory_labels = torch.stack([s['label'] for s in memory_samples]).to(current_features.device)
        
        # Compute similarity-based loss
        current_features_norm = F.normalize(current_features, dim=1)
        memory_features_norm = F.normalize(memory_features, dim=1)
        
        similarities = torch.matmul(current_features_norm, memory_features_norm.T)
        
        # Find most similar memory samples
        max_similarities, similar_indices = similarities.max(dim=1)
        similar_labels = memory_labels[similar_indices]
        
        # Consistency loss with memory
        memory_loss = F.cross_entropy(similarities, similar_indices)
        
        return memory_loss
    
    def adapt(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Adapt with episodic memory."""
        # Standard AdaContrast adaptation
        outputs = super().adapt(x)
        
        # Update episodic memory periodically
        self.memory_counter += 1
        if self.memory_counter % self.memory_update_freq == 0:
            with torch.no_grad():
                logits = outputs['logits']
                features = outputs['adapted_features']
                pseudo_labels, confident_mask = self.get_pseudo_labels(logits)
                self.update_episodic_memory(features, pseudo_labels, confident_mask)
        
        # Add memory loss
        memory_loss = self.compute_memory_loss(outputs['adapted_features'])
        if memory_loss.item() > 0:
            self.optimizer.zero_grad()
            memory_loss.backward()
            self.optimizer.step()
            
            outputs['memory_loss'] = memory_loss.item()
        else:
            outputs['memory_loss'] = 0.0
        
        return outputs


def create_ada_contrast_baseline(model: nn.Module, config: Dict[str, Any]) -> AdaContrast:
    """
    Factory function to create AdaContrast baseline.
    
    Args:
        model: Base model to adapt
        config: Configuration dictionary
    
    Returns:
        AdaContrast adapter instance
    """
    ada_type = config.get('ada_type', 'standard')
    
    if ada_type == 'standard':
        return AdaContrast(model, config)
    elif ada_type == 'memory':
        return AdaContrastWithMemory(model, config)
    else:
        raise ValueError(f"Unknown AdaContrast type: {ada_type}")