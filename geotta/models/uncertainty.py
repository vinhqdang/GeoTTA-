import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any


class GeometricUncertainty:
    """
    Compute uncertainty based on geometric relationships in embedding space.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.geometric_weight = config['uncertainty']['geometric_weight']
        self.angular_weight = config['uncertainty']['angular_weight']
        
    def compute_uncertainty(self, image_features: torch.Tensor, 
                          text_features: torch.Tensor) -> torch.Tensor:
        """
        Compute uncertainty from cross-modal geometric distance.
        
        Args:
            image_features: [B, D] normalized image embeddings
            text_features: [B, D] or [K, D] normalized text embeddings
        """
        # Euclidean distance in normalized space
        if text_features.dim() == 3:  # [B, K, D] - multiple prototypes per image
            distances = torch.cdist(
                image_features.unsqueeze(1), 
                text_features, 
                p=2
            )
            min_distances, _ = distances.min(dim=1)
        else:
            distances = torch.norm(image_features - text_features, p=2, dim=-1)
            min_distances = distances
            
        # Angular distance (more robust to modality gap)
        cosine_sim = F.cosine_similarity(image_features, text_features, dim=-1)
        angular_distance = torch.acos(torch.clamp(cosine_sim, -1 + 1e-7, 1 - 1e-7))
        
        # Combined uncertainty
        uncertainty = (self.geometric_weight * min_distances + 
                      self.angular_weight * angular_distance)
        
        return uncertainty
    
    def calibrate_uncertainty(self, uncertainties: torch.Tensor, 
                            errors: torch.Tensor, 
                            num_bins: int = 15) -> float:
        """
        Calibrate uncertainty to actual errors (for training).
        
        Args:
            uncertainties: Predicted uncertainties
            errors: Actual prediction errors (0 or 1)
            num_bins: Number of bins for calibration
            
        Returns:
            Expected Calibration Error (ECE)
        """
        # Convert to numpy for calibration computation
        uncertainties_np = uncertainties.cpu().numpy()
        errors_np = errors.cpu().numpy()
        
        # Bin uncertainties
        bin_boundaries = np.linspace(0, uncertainties_np.max(), num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0  # Expected Calibration Error
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (uncertainties_np > bin_lower) & (uncertainties_np <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_uncertainty_in_bin = uncertainties_np[in_bin].mean()
                avg_error_in_bin = errors_np[in_bin].mean()
                ece += np.abs(avg_uncertainty_in_bin - avg_error_in_bin) * prop_in_bin
                
        return ece
    
    def compute_ensemble_uncertainty(self, features_list: list) -> torch.Tensor:
        """
        Compute uncertainty from ensemble of feature predictions.
        
        Args:
            features_list: List of feature tensors from different runs
            
        Returns:
            Uncertainty based on feature variance
        """
        if len(features_list) < 2:
            return torch.zeros(features_list[0].shape[0], device=features_list[0].device)
        
        # Stack features and compute variance
        features_stack = torch.stack(features_list, dim=0)  # [N_ensemble, B, D]
        feature_var = torch.var(features_stack, dim=0)  # [B, D]
        
        # Mean variance across dimensions as uncertainty measure
        uncertainty = torch.mean(feature_var, dim=-1)  # [B]
        
        return uncertainty
    
    def expected_calibration_error(self, confidences: torch.Tensor, 
                                 accuracies: torch.Tensor, 
                                 num_bins: int = 15) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            confidences: Model confidence scores [0, 1]
            accuracies: Binary accuracy (0 or 1)
            num_bins: Number of bins
            
        Returns:
            ECE value
        """
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Samples in bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()
    
    def compute_predictive_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute predictive entropy as uncertainty measure.
        
        Args:
            logits: Model logits [B, C]
            
        Returns:
            Entropy values [B]
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return entropy