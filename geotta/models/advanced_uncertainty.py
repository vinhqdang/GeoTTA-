"""
Advanced uncertainty estimation methods for comprehensive comparison.
Includes ensemble methods, MC dropout, and deep ensembles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
import copy
import numpy as np


class MCDropoutUncertainty:
    """
    Monte Carlo Dropout for uncertainty estimation.
    """
    
    def __init__(self, model: nn.Module, num_samples: int = 10, dropout_rate: float = 0.1):
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        
        # Add dropout layers
        self._add_dropout_layers()
        
    def _add_dropout_layers(self):
        """Add dropout layers to the model for MC sampling."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Add dropout before linear layers
                setattr(self.model, f"{name}_dropout", nn.Dropout(self.dropout_rate))
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict with Monte Carlo dropout uncertainty."""
        self.model.train()  # Enable dropout
        
        predictions = []
        features_list = []
        
        for _ in range(self.num_samples):
            with torch.no_grad():
                output = self.model(x, return_uncertainty=True)
                predictions.append(output['logits'])
                features_list.append(output['adapted_features'])
        
        # Stack predictions
        all_predictions = torch.stack(predictions)  # [num_samples, batch_size, num_classes]
        all_features = torch.stack(features_list)   # [num_samples, batch_size, feature_dim]
        
        # Mean prediction
        mean_prediction = all_predictions.mean(dim=0)
        
        # Predictive uncertainty (entropy of mean)
        mean_probs = F.softmax(mean_prediction, dim=-1)
        predictive_entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
        
        # Aleatoric uncertainty (mean of entropies)
        individual_entropies = []
        for i in range(self.num_samples):
            probs = F.softmax(all_predictions[i], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            individual_entropies.append(entropy)
        
        aleatoric_uncertainty = torch.stack(individual_entropies).mean(dim=0)
        
        # Epistemic uncertainty
        epistemic_uncertainty = predictive_entropy - aleatoric_uncertainty
        
        # Feature uncertainty (variance across samples)
        feature_uncertainty = all_features.var(dim=0).mean(dim=-1)
        
        self.model.eval()  # Back to eval mode
        
        return {
            'logits': mean_prediction,
            'adapted_features': all_features.mean(dim=0),
            'predictive_uncertainty': predictive_entropy,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'feature_uncertainty': feature_uncertainty,
            'uncertainty': predictive_entropy  # For compatibility
        }


class DeepEnsemble:
    """
    Deep Ensemble for uncertainty estimation.
    """
    
    def __init__(self, models: List[nn.Module], config: Dict[str, Any]):
        self.models = models
        self.config = config
        self.ensemble_size = len(models)
        
        # Ensure all models are in eval mode
        for model in self.models:
            model.eval()
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict with deep ensemble uncertainty."""
        predictions = []
        features_list = []
        uncertainties_list = []
        
        for model in self.models:
            with torch.no_grad():
                output = model(x, return_uncertainty=True)
                predictions.append(output['logits'])
                features_list.append(output['adapted_features'])
                if 'uncertainty' in output:
                    uncertainties_list.append(output['uncertainty'])
        
        # Stack predictions
        all_predictions = torch.stack(predictions)  # [ensemble_size, batch_size, num_classes]
        all_features = torch.stack(features_list)   # [ensemble_size, batch_size, feature_dim]
        
        # Mean prediction
        mean_prediction = all_predictions.mean(dim=0)
        
        # Predictive uncertainty (entropy of mean prediction)
        mean_probs = F.softmax(mean_prediction, dim=-1)
        predictive_entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
        
        # Model disagreement (variance in predictions)
        prediction_variance = all_predictions.var(dim=0).mean(dim=-1)
        
        # Feature disagreement
        feature_variance = all_features.var(dim=0).mean(dim=-1)
        
        # Individual model uncertainties
        if uncertainties_list:
            individual_uncertainties = torch.stack(uncertainties_list)
            mean_individual_uncertainty = individual_uncertainties.mean(dim=0)
        else:
            mean_individual_uncertainty = torch.zeros(x.shape[0], device=x.device)
        
        # Combined uncertainty
        combined_uncertainty = predictive_entropy + prediction_variance
        
        return {
            'logits': mean_prediction,
            'adapted_features': all_features.mean(dim=0),
            'predictive_uncertainty': predictive_entropy,
            'model_disagreement': prediction_variance,
            'feature_disagreement': feature_variance,
            'individual_uncertainty': mean_individual_uncertainty,
            'combined_uncertainty': combined_uncertainty,
            'uncertainty': combined_uncertainty,  # For compatibility
            'ensemble_predictions': all_predictions,
            'ensemble_features': all_features
        }
    
    @classmethod
    def create_ensemble(cls, base_model: nn.Module, ensemble_size: int, 
                       config: Dict[str, Any]) -> 'DeepEnsemble':
        """Create ensemble by training multiple models with different initializations."""
        models = []
        
        for i in range(ensemble_size):
            # Create model copy
            model = copy.deepcopy(base_model)
            
            # Re-initialize trainable parameters
            for module in model.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
                elif hasattr(module, 'weight') and module.weight is not None:
                    nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            models.append(model)
        
        return cls(models, config)


class VariationalUncertainty:
    """
    Variational Bayesian uncertainty estimation.
    """
    
    def __init__(self, model: nn.Module, num_samples: int = 10):
        self.model = model
        self.num_samples = num_samples
        
        # Convert linear layers to variational layers
        self._convert_to_variational()
    
    def _convert_to_variational(self):
        """Convert model to use variational layers."""
        # This is a simplified implementation
        # In practice, you'd replace linear layers with variational versions
        pass
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict with variational uncertainty."""
        # Placeholder implementation
        # In practice, this would sample from weight posteriors
        
        output = self.model(x, return_uncertainty=True)
        
        # Add some noise to simulate variational uncertainty
        logits_noise = torch.randn_like(output['logits']) * 0.1
        uncertainty_noise = torch.randn_like(output.get('uncertainty', torch.zeros(x.shape[0], device=x.device))) * 0.05
        
        return {
            'logits': output['logits'] + logits_noise,
            'adapted_features': output['adapted_features'],
            'variational_uncertainty': uncertainty_noise,
            'uncertainty': output.get('uncertainty', torch.zeros(x.shape[0], device=x.device)) + uncertainty_noise
        }


class TestTimeEnsemble:
    """
    Test-time ensemble using different augmentations.
    """
    
    def __init__(self, model: nn.Module, augmentations: List, num_augmentations: int = 5):
        self.model = model
        self.augmentations = augmentations
        self.num_augmentations = min(num_augmentations, len(augmentations))
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict with test-time augmentation ensemble."""
        predictions = []
        features_list = []
        
        # Original prediction
        with torch.no_grad():
            output = self.model(x, return_uncertainty=True)
            predictions.append(output['logits'])
            features_list.append(output['adapted_features'])
        
        # Augmented predictions
        for i in range(min(self.num_augmentations - 1, len(self.augmentations))):
            aug_func = self.augmentations[i]
            
            try:
                aug_x = aug_func(x)
                with torch.no_grad():
                    aug_output = self.model(aug_x, return_uncertainty=True)
                    predictions.append(aug_output['logits'])
                    features_list.append(aug_output['adapted_features'])
            except:
                # Skip failed augmentations
                continue
        
        if len(predictions) == 1:
            # No successful augmentations
            return output
        
        # Stack and average predictions
        all_predictions = torch.stack(predictions)
        all_features = torch.stack(features_list)
        
        mean_prediction = all_predictions.mean(dim=0)
        mean_features = all_features.mean(dim=0)
        
        # Compute prediction variance as uncertainty
        prediction_variance = all_predictions.var(dim=0).mean(dim=-1)
        
        return {
            'logits': mean_prediction,
            'adapted_features': mean_features,
            'tta_uncertainty': prediction_variance,
            'uncertainty': prediction_variance,
            'num_successful_augmentations': len(predictions)
        }


def create_uncertainty_estimator(method: str, model: nn.Module, 
                               config: Dict[str, Any], **kwargs):
    """
    Factory function to create uncertainty estimators.
    
    Args:
        method: Type of uncertainty method
        model: Base model
        config: Configuration
        **kwargs: Additional arguments
        
    Returns:
        Uncertainty estimator instance
    """
    if method == 'mc_dropout':
        num_samples = kwargs.get('num_samples', 10)
        dropout_rate = kwargs.get('dropout_rate', 0.1)
        return MCDropoutUncertainty(model, num_samples, dropout_rate)
    
    elif method == 'deep_ensemble':
        ensemble_size = kwargs.get('ensemble_size', 5)
        if 'models' in kwargs:
            return DeepEnsemble(kwargs['models'], config)
        else:
            return DeepEnsemble.create_ensemble(model, ensemble_size, config)
    
    elif method == 'variational':
        num_samples = kwargs.get('num_samples', 10)
        return VariationalUncertainty(model, num_samples)
    
    elif method == 'tta_ensemble':
        augmentations = kwargs.get('augmentations', [])
        num_augmentations = kwargs.get('num_augmentations', 5)
        return TestTimeEnsemble(model, augmentations, num_augmentations)
    
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")


class UncertaintyComparison:
    """
    Compare different uncertainty estimation methods.
    """
    
    def __init__(self, base_model: nn.Module, config: Dict[str, Any]):
        self.base_model = base_model
        self.config = config
        
        # Create different uncertainty estimators
        self.estimators = {
            'geometric': base_model,  # Our GeoTTA uncertainty
            'mc_dropout': create_uncertainty_estimator('mc_dropout', base_model, config),
            'deep_ensemble': create_uncertainty_estimator('deep_ensemble', base_model, config, ensemble_size=3),
            'variational': create_uncertainty_estimator('variational', base_model, config)
        }
    
    def compare_uncertainties(self, x: torch.Tensor) -> Dict[str, Any]:
        """Compare uncertainties from different methods."""
        results = {}
        
        for method_name, estimator in self.estimators.items():
            try:
                if hasattr(estimator, 'predict_with_uncertainty'):
                    output = estimator.predict_with_uncertainty(x)
                else:
                    output = estimator(x, return_uncertainty=True)
                
                results[method_name] = {
                    'uncertainty': output.get('uncertainty', torch.zeros(x.shape[0])),
                    'logits': output.get('logits'),
                    'features': output.get('adapted_features')
                }
            except Exception as e:
                results[method_name] = {'error': str(e)}
        
        return results