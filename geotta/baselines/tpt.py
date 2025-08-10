"""
TPT: Test-time Prompt Tuning (NeurIPS 2022)
Implementation for fair comparison in WACV 2026 submission.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
import copy
import clip
from collections import deque
import numpy as np


class TPT:
    """
    Test-time Prompt Tuning for Vision-Language Models.
    
    Reference: Shu et al. "Test-Time Prompt Tuning for Zero-Shot Generalization in Vision-Language Models" NeurIPS 2022.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.config = config
        self.base_model = model
        
        # Create adapted model
        self.model = copy.deepcopy(model)
        
        # TPT specific parameters
        self.n_ctx = config.get('n_ctx', 4)  # Number of prompt tokens
        self.ctx_init = config.get('ctx_init', 'a photo of a')  # Initial prompt
        self.selection_p = config.get('selection_p', 0.1)  # Selection ratio
        self.tta_steps = config.get('tta_steps', 1)  # TTA steps per sample
        
        # Setup prompt learning
        self._setup_prompt_learning()
        
        # Optimizer for prompt tokens
        self.optimizer = torch.optim.AdamW(
            [self.ctx],
            lr=config.get('tpt_lr', 5e-3),
            weight_decay=config.get('tpt_weight_decay', 0.0)
        )
        
        # Statistics
        self.num_samples = 0
        self.adaptation_history = deque(maxlen=1000)
        
        # For confident sample selection
        self.confidence_threshold = config.get('confidence_threshold', 0.1)
        
    def _setup_prompt_learning(self):
        """Setup learnable prompt tokens."""
        # Get CLIP model
        if hasattr(self.model, 'clip_model'):
            clip_model = self.model.clip_model
        else:
            # Fallback: load CLIP model
            clip_model, _ = clip.load(self.config.get('clip_model', 'ViT-B/32'), device='cuda')
        
        # Get text encoder
        self.text_encoder = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        
        # Setup context tokens
        self.dtype = self.text_encoder.get_dtype() if hasattr(self.text_encoder, 'get_dtype') else torch.float32
        
        # Initialize context vectors
        if self.ctx_init:
            # Initialize with pre-defined prompt
            ctx_init = self.ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = self.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            self.n_ctx = n_ctx
        else:
            # Random initialization
            ctx_vectors = torch.empty(self.n_ctx, 512, dtype=self.dtype, device='cuda')
            nn.init.normal_(ctx_vectors, std=0.02)
        
        # Make context learnable
        self.ctx = nn.Parameter(ctx_vectors)
        
        # Class names and templates
        self.class_names = self._get_class_names()
        self.prompt_templates = [
            'a photo of a {}.',
            'a photograph of a {}.',
            'an image of a {}.',
            'this is a {}.',
            'a picture of a {}.',
        ]
        
    def _get_class_names(self) -> List[str]:
        """Get class names for the dataset."""
        if hasattr(self.model, 'get_text_prototypes'):
            # Try to get from model
            return [f'class_{i}' for i in range(1000)]  # Default ImageNet
        else:
            return [f'class_{i}' for i in range(1000)]
    
    def create_text_prompts(self) -> torch.Tensor:
        """Create text prompts with learnable context."""
        # Get class names
        prompts = []
        
        for class_name in self.class_names:
            for template in self.prompt_templates:
                prompt_text = template.format(class_name)
                prompts.append(prompt_text)
        
        # Tokenize prompts
        text_tokens = clip.tokenize(prompts).cuda()
        
        # Replace context tokens with learnable ones
        with torch.no_grad():
            text_embeddings = self.token_embedding(text_tokens).type(self.dtype)
        
        # Insert learnable context
        # This is a simplified version - in practice, you'd need to handle
        # the exact positioning of context tokens
        prefix = text_embeddings[:, :1, :]  # SOS token
        suffix = text_embeddings[:, 1 + self.n_ctx:, :]  # Class name + EOS
        
        # Expand context for all prompts
        ctx_expanded = self.ctx.unsqueeze(0).expand(text_embeddings.shape[0], -1, -1)
        
        # Concatenate
        prompts_embedding = torch.cat([prefix, ctx_expanded, suffix], dim=1)
        
        return prompts_embedding, text_tokens
    
    def encode_text_with_prompt(self) -> torch.Tensor:
        """Encode text with learnable prompts."""
        prompts_embedding, text_tokens = self.create_text_prompts()
        
        # Pass through transformer
        x = prompts_embedding + self.text_encoder.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # Take features from the EOS token
        x = self.ln_final(x).type(self.dtype)
        
        # Extract features at EOS token position
        eos_indices = text_tokens.argmax(dim=-1)
        text_features = x[torch.arange(x.shape[0]), eos_indices] @ self.text_projection
        
        return text_features
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive prompts."""
        # Get image features
        if hasattr(self.model, 'forward'):
            outputs = self.model(x, return_uncertainty=True)
            image_features = outputs['adapted_features']
        else:
            image_features = self.model(x)
            outputs = {'adapted_features': image_features}
        
        # Get text features with learnable prompts
        text_features = self.encode_text_with_prompt()
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute logits
        logits = 100.0 * image_features @ text_features.T
        
        outputs['logits'] = logits
        outputs['text_features'] = text_features
        
        return outputs
    
    def select_confident_samples(self, logits: torch.Tensor) -> torch.Tensor:
        """Select confident samples for adaptation."""
        with torch.no_grad():
            softmax_out = F.softmax(logits, dim=1)
            max_probs, _ = softmax_out.max(dim=1)
            
            # Select top-p confident samples
            num_select = max(1, int(self.selection_p * logits.shape[0]))
            _, indices = torch.topk(max_probs, num_select)
            
            return indices
    
    def compute_tpt_loss(self, logits: torch.Tensor, confident_indices: torch.Tensor) -> torch.Tensor:
        """Compute TPT loss using marginal entropy."""
        # Select confident samples
        confident_logits = logits[confident_indices]
        
        # Compute marginal entropy
        softmax_out = F.softmax(confident_logits, dim=1)
        marginal_dist = softmax_out.mean(dim=0)
        
        # Marginal entropy (to be maximized, so we minimize negative entropy)
        marginal_entropy = -(marginal_dist * torch.log(marginal_dist + 1e-10)).sum()
        loss = -marginal_entropy  # Negative because we want to maximize entropy
        
        return loss
    
    def adapt(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform TPT adaptation."""
        # Multiple adaptation steps
        for step in range(self.tta_steps):
            # Forward pass
            outputs = self.forward(x)
            logits = outputs['logits']
            
            # Select confident samples
            confident_indices = self.select_confident_samples(logits)
            
            if len(confident_indices) > 0:
                # Compute TPT loss
                tpt_loss = self.compute_tpt_loss(logits, confident_indices)
                
                # Adaptation step
                self.optimizer.zero_grad()
                tpt_loss.backward()
                self.optimizer.step()
                
                # Update statistics
                self.adaptation_history.append({
                    'tpt_loss': tpt_loss.item(),
                    'confident_samples': len(confident_indices),
                    'total_samples': x.shape[0],
                    'step': step
                })
        
        # Final forward pass
        with torch.no_grad():
            final_outputs = self.forward(x)
            
            # Add adaptation statistics
            if self.adaptation_history:
                recent_stats = self.adaptation_history[-1]
                final_outputs.update({
                    'adaptation_loss': recent_stats['tpt_loss'],
                    'confident_ratio': recent_stats['confident_samples'] / recent_stats['total_samples']
                })
        
        self.num_samples += x.shape[0]
        return final_outputs
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions with TPT."""
        return self.adapt(x)
    
    def reset(self):
        """Reset adaptation state."""
        # Reset prompt tokens
        self._setup_prompt_learning()
        
        # Reset optimizer
        self.optimizer = torch.optim.AdamW(
            [self.ctx],
            lr=self.config.get('tpt_lr', 5e-3),
            weight_decay=self.config.get('tpt_weight_decay', 0.0)
        )
        
        # Reset statistics
        self.num_samples = 0
        self.adaptation_history.clear()
    
    def get_adaptation_stats(self) -> Dict[str, float]:
        """Get adaptation statistics."""
        if len(self.adaptation_history) == 0:
            return {'tpt_loss': 0.0, 'confident_ratio': 0.0}
        
        recent_stats = list(self.adaptation_history)[-100:]  # Last 100 steps
        
        return {
            'mean_tpt_loss': np.mean([s['tpt_loss'] for s in recent_stats]),
            'mean_confident_ratio': np.mean([s['confident_samples'] / s['total_samples'] for s in recent_stats]),
            'adaptation_steps': len(self.adaptation_history),
            'total_samples': self.num_samples
        }


class TPTwithConsistency(TPT):
    """
    TPT with consistency regularization across different augmentations.
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self.consistency_weight = config.get('consistency_weight', 0.1)
        
        # Define augmentations
        from torchvision import transforms
        self.augmentations = [
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    
    def apply_augmentation(self, x: torch.Tensor, aug_idx: int) -> torch.Tensor:
        """Apply specific augmentation."""
        if aug_idx >= len(self.augmentations):
            return x
        
        aug = self.augmentations[aug_idx]
        # Note: This is simplified - you'd need to handle tensor/PIL conversion
        return aug(x)
    
    def compute_consistency_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute consistency loss across augmentations."""
        # Original prediction
        original_outputs = self.forward(x)
        original_logits = original_outputs['logits']
        
        consistency_losses = []
        
        # Predictions on augmented versions
        for aug_idx in range(len(self.augmentations)):
            try:
                augmented_x = self.apply_augmentation(x, aug_idx)
                aug_outputs = self.forward(augmented_x)
                aug_logits = aug_outputs['logits']
                
                # KL divergence consistency loss
                consistency_loss = F.kl_div(
                    F.log_softmax(aug_logits, dim=1),
                    F.softmax(original_logits, dim=1),
                    reduction='batchmean'
                )
                consistency_losses.append(consistency_loss)
            except:
                continue
        
        if consistency_losses:
            return sum(consistency_losses) / len(consistency_losses)
        else:
            return torch.tensor(0.0, device=x.device)
    
    def adapt(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform TPT adaptation with consistency."""
        for step in range(self.tta_steps):
            # Forward pass
            outputs = self.forward(x)
            logits = outputs['logits']
            
            # Select confident samples
            confident_indices = self.select_confident_samples(logits)
            
            # Compute losses
            tpt_loss = torch.tensor(0.0, device=x.device)
            if len(confident_indices) > 0:
                tpt_loss = self.compute_tpt_loss(logits, confident_indices)
            
            # Consistency loss
            consistency_loss = self.compute_consistency_loss(x)
            
            # Combined loss
            total_loss = tpt_loss + self.consistency_weight * consistency_loss
            
            if total_loss.requires_grad:
                # Adaptation step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                self.adaptation_history.append({
                    'tpt_loss': tpt_loss.item(),
                    'consistency_loss': consistency_loss.item(),
                    'total_loss': total_loss.item(),
                    'confident_samples': len(confident_indices),
                    'total_samples': x.shape[0]
                })
        
        # Final outputs
        with torch.no_grad():
            final_outputs = self.forward(x)
            
            if self.adaptation_history:
                recent_stats = self.adaptation_history[-1]
                final_outputs.update({
                    'adaptation_loss': recent_stats['total_loss'],
                    'tpt_loss': recent_stats['tpt_loss'],
                    'consistency_loss': recent_stats['consistency_loss']
                })
        
        self.num_samples += x.shape[0]
        return final_outputs


def create_tpt_baseline(model: nn.Module, config: Dict[str, Any]) -> TPT:
    """
    Factory function to create TPT baseline.
    
    Args:
        model: Base model to adapt
        config: Configuration dictionary
    
    Returns:
        TPT adapter instance
    """
    tpt_type = config.get('tpt_type', 'standard')
    
    if tpt_type == 'standard':
        return TPT(model, config)
    elif tpt_type == 'consistency':
        return TPTwithConsistency(model, config)
    else:
        raise ValueError(f"Unknown TPT type: {tpt_type}")