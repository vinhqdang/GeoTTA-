import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Tuple, Dict, Any


class TestTimeAdapter:
    """
    Efficient test-time adaptation using geometric uncertainty.
    No backpropagation required during test time.
    """
    def __init__(self, bridge_model: nn.Module, config: Dict[str, Any]):
        self.bridge = bridge_model
        self.config = config
        
        # Cache for prototypes
        self.prototype_cache = deque(maxlen=config['test_time']['cache_size'])
        self.uncertainty_cache = deque(maxlen=config['test_time']['cache_size'])
        
        # Running statistics for calibration
        self.running_mean_uncertainty = 0.0
        self.running_var_uncertainty = 0.0
        self.num_samples = 0
        
    @torch.no_grad()
    def adapt_single_sample(self, image: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Single-pass adaptation for one test sample.
        
        Args:
            image: Single image tensor [3, H, W]
            
        Returns:
            Tuple of (adapted_features, uncertainty)
        """
        # Get features and uncertainty
        output = self.bridge(image.unsqueeze(0), return_uncertainty=True)
        
        adapted_features = output['adapted_features']
        uncertainty = output['uncertainty'].item()
        
        # Update running statistics
        self.update_statistics(uncertainty)
        
        # Compute adaptation weight based on normalized uncertainty
        norm_uncertainty = self.normalize_uncertainty(uncertainty)
        adaptation_weight = torch.sigmoid(-norm_uncertainty)  # High confidence = high weight
        
        # Weighted combination with original features
        final_features = (adaptation_weight * adapted_features + 
                         (1 - adaptation_weight) * output['original_image_features'])
        
        # Update cache if confidence is high
        if adaptation_weight > 0.7:
            self.prototype_cache.append(final_features)
            self.uncertainty_cache.append(uncertainty)
            
        return final_features.squeeze(0), uncertainty
    
    @torch.no_grad()
    def adapt_batch(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch adaptation for multiple test samples.
        
        Args:
            images: Batch of images [B, 3, H, W]
            
        Returns:
            Tuple of (adapted_features [B, D], uncertainties [B])
        """
        batch_size = images.shape[0]
        adapted_features_list = []
        uncertainties_list = []
        
        for i in range(batch_size):
            features, uncertainty = self.adapt_single_sample(images[i])
            adapted_features_list.append(features)
            uncertainties_list.append(uncertainty)
        
        adapted_features = torch.stack(adapted_features_list)
        uncertainties = torch.tensor(uncertainties_list, device=images.device)
        
        return adapted_features, uncertainties
    
    def update_statistics(self, uncertainty: float):
        """Update running statistics for uncertainty normalization."""
        self.num_samples += 1
        delta = uncertainty - self.running_mean_uncertainty
        self.running_mean_uncertainty += delta / self.num_samples
        self.running_var_uncertainty += delta * (uncertainty - self.running_mean_uncertainty)
        
    def normalize_uncertainty(self, uncertainty: float) -> float:
        """Normalize uncertainty using running statistics."""
        if self.num_samples < 2:
            return uncertainty
        
        std = (self.running_var_uncertainty / (self.num_samples - 1)) ** 0.5
        if std < 1e-6:
            return 0.0
        
        return (uncertainty - self.running_mean_uncertainty) / std
    
    def get_prototype_statistics(self) -> Dict[str, float]:
        """Get statistics about cached prototypes."""
        if len(self.prototype_cache) == 0:
            return {'count': 0, 'mean_uncertainty': 0.0, 'std_uncertainty': 0.0}
        
        uncertainties = list(self.uncertainty_cache)
        return {
            'count': len(self.prototype_cache),
            'mean_uncertainty': sum(uncertainties) / len(uncertainties),
            'std_uncertainty': (sum((u - sum(uncertainties)/len(uncertainties))**2 for u in uncertainties) / len(uncertainties)) ** 0.5
        }
    
    def reset_adaptation_state(self):
        """Reset the adaptation state for new domain."""
        self.prototype_cache.clear()
        self.uncertainty_cache.clear()
        self.running_mean_uncertainty = 0.0
        self.running_var_uncertainty = 0.0
        self.num_samples = 0
    
    @torch.no_grad()
    def compute_domain_shift_score(self, images: torch.Tensor) -> float:
        """
        Compute a score indicating how much domain shift there is.
        
        Args:
            images: Batch of test images
            
        Returns:
            Domain shift score (higher = more shift)
        """
        outputs = self.bridge(images, return_uncertainty=True)
        uncertainties = outputs['uncertainty'].squeeze()
        
        # Use mean uncertainty as proxy for domain shift
        domain_shift_score = uncertainties.mean().item()
        
        return domain_shift_score
    
    @torch.no_grad()
    def predict_with_uncertainty(self, images: torch.Tensor, 
                                text_prototypes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            images: Input images [B, 3, H, W]
            text_prototypes: Text prototype features [C, D]
            
        Returns:
            Dictionary with predictions, uncertainties, and confidence
        """
        adapted_features, uncertainties = self.adapt_batch(images)
        
        # Compute logits
        logits = 100.0 * adapted_features @ text_prototypes.T
        
        # Get predictions
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        confidences = probs.max(dim=-1)[0]
        
        return {
            'predictions': predictions,
            'logits': logits,
            'probabilities': probs,
            'confidences': confidences,
            'uncertainties': uncertainties,
            'adapted_features': adapted_features
        }