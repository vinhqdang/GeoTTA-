import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, calibration_curve
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor, 
                   uncertainties: torch.Tensor = None) -> Dict[str, float]:
    """
    Comprehensive evaluation including uncertainty quality.
    
    Args:
        predictions: Model predictions [N]
        labels: Ground truth labels [N]  
        uncertainties: Uncertainty estimates [N]
        
    Returns:
        Dictionary of computed metrics
    """
    # Convert to numpy for sklearn compatibility
    preds_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Basic accuracy
    accuracy = accuracy_score(labels_np, preds_np)
    
    metrics = {'accuracy': accuracy}
    
    if uncertainties is not None:
        uncertainties_np = uncertainties.cpu().numpy()
        errors = (preds_np != labels_np).astype(float)
        
        # Expected Calibration Error
        ece = expected_calibration_error(uncertainties, (predictions != labels).float())
        metrics['ece'] = ece
        
        # Uncertainty quality (AUROC for error detection)
        if len(np.unique(errors)) > 1:  # Need both correct and incorrect predictions
            auroc = roc_auc_score(errors, uncertainties_np)
            metrics['auroc'] = auroc
        else:
            metrics['auroc'] = 0.5  # Random performance when all predictions are same
    
    return metrics


def expected_calibration_error(confidences: torch.Tensor, 
                             accuracies: torch.Tensor, 
                             num_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        confidences: Model confidence scores [0, 1] - for uncertainties, use 1-uncertainty
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


def compute_calibration_metrics(predictions: torch.Tensor, labels: torch.Tensor, 
                              confidences: torch.Tensor, num_bins: int = 15) -> Dict[str, float]:
    """
    Compute comprehensive calibration metrics.
    
    Args:
        predictions: Model predictions [N]
        labels: Ground truth labels [N]
        confidences: Confidence scores [N]
        num_bins: Number of bins for calibration
        
    Returns:
        Dictionary with calibration metrics
    """
    accuracies = (predictions == labels).float()
    
    # Expected Calibration Error
    ece = expected_calibration_error(confidences, accuracies, num_bins)
    
    # Maximum Calibration Error
    mce = maximum_calibration_error(confidences, accuracies, num_bins)
    
    # Average Confidence
    avg_confidence = confidences.mean().item()
    
    # Average Accuracy
    avg_accuracy = accuracies.mean().item()
    
    return {
        'ece': ece,
        'mce': mce,
        'avg_confidence': avg_confidence,
        'avg_accuracy': avg_accuracy,
        'confidence_accuracy_gap': abs(avg_confidence - avg_accuracy)
    }


def maximum_calibration_error(confidences: torch.Tensor, 
                            accuracies: torch.Tensor, 
                            num_bins: int = 15) -> float:
    """Compute Maximum Calibration Error (MCE)."""
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    max_error = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            bin_error = torch.abs(avg_confidence_in_bin - accuracy_in_bin).item()
            max_error = max(max_error, bin_error)
    
    return max_error


def compute_uncertainty_quality_metrics(uncertainties: torch.Tensor, 
                                       errors: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics for uncertainty quality assessment.
    
    Args:
        uncertainties: Uncertainty estimates [N]
        errors: Binary error indicators [N]
        
    Returns:
        Dictionary with uncertainty quality metrics
    """
    uncertainties_np = uncertainties.cpu().numpy()
    errors_np = errors.cpu().numpy()
    
    metrics = {}
    
    # AUROC for error detection
    if len(np.unique(errors_np)) > 1:
        auroc = roc_auc_score(errors_np, uncertainties_np)
        metrics['uncertainty_auroc'] = auroc
    
    # Correlation between uncertainty and error
    correlation = np.corrcoef(uncertainties_np, errors_np)[0, 1]
    if not np.isnan(correlation):
        metrics['uncertainty_error_correlation'] = correlation
    
    # Uncertainty statistics
    metrics['mean_uncertainty'] = uncertainties_np.mean()
    metrics['std_uncertainty'] = uncertainties_np.std()
    
    return metrics


def plot_calibration_curve(confidences: torch.Tensor, accuracies: torch.Tensor, 
                          save_path: str = None, num_bins: int = 15):
    """Plot calibration curve."""
    conf_np = confidences.cpu().numpy()
    acc_np = accuracies.cpu().numpy()
    
    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        acc_np, conf_np, n_bins=num_bins
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_uncertainty_histogram(uncertainties: torch.Tensor, errors: torch.Tensor, 
                              save_path: str = None, bins: int = 50):
    """Plot histogram of uncertainties for correct vs incorrect predictions."""
    uncertainties_np = uncertainties.cpu().numpy()
    errors_np = errors.cpu().numpy()
    
    correct_uncertainties = uncertainties_np[errors_np == 0]
    incorrect_uncertainties = uncertainties_np[errors_np == 1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(correct_uncertainties, bins=bins, alpha=0.7, label='Correct', density=True)
    plt.hist(incorrect_uncertainties, bins=bins, alpha=0.7, label='Incorrect', density=True)
    plt.xlabel('Uncertainty')
    plt.ylabel('Density')
    plt.title('Uncertainty Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def evaluate_model_comprehensive(model, dataloader, device='cuda', 
                               save_plots: bool = False, plot_dir: str = None):
    """
    Comprehensive model evaluation with all metrics.
    
    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to run evaluation on
        save_plots: Whether to save calibration plots
        plot_dir: Directory to save plots
        
    Returns:
        Dictionary with all computed metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_confidences = []
    all_uncertainties = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            # Get model outputs
            if hasattr(model, 'predict_with_uncertainty'):
                # For TTA adapter
                text_prototypes = model.bridge.get_text_prototypes()
                outputs = model.predict_with_uncertainty(images, text_prototypes)
                preds = outputs['predictions']
                confidences = outputs['confidences']
                uncertainties = outputs['uncertainties']
            else:
                # For regular model
                outputs = model(images, return_uncertainty=True)
                logits = 100.0 * outputs['adapted_features'] @ model.get_text_prototypes().T
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)
                confidences = probs.max(dim=-1)[0]
                uncertainties = outputs['uncertainty'].squeeze()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_confidences.append(confidences.cpu())
            all_uncertainties.append(uncertainties.cpu())
    
    # Concatenate all results
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_confidences = torch.cat(all_confidences)
    all_uncertainties = torch.cat(all_uncertainties)
    
    # Compute all metrics
    basic_metrics = compute_metrics(all_preds, all_labels, all_uncertainties)
    
    calibration_metrics = compute_calibration_metrics(
        all_preds, all_labels, all_confidences
    )
    
    uncertainty_metrics = compute_uncertainty_quality_metrics(
        all_uncertainties, (all_preds != all_labels).float()
    )
    
    # Combine all metrics
    all_metrics = {**basic_metrics, **calibration_metrics, **uncertainty_metrics}
    
    # Save plots if requested
    if save_plots and plot_dir:
        import os
        os.makedirs(plot_dir, exist_ok=True)
        
        plot_calibration_curve(
            all_confidences, (all_preds == all_labels).float(),
            save_path=os.path.join(plot_dir, 'calibration_curve.png')
        )
        
        plot_uncertainty_histogram(
            all_uncertainties, (all_preds != all_labels).float(),
            save_path=os.path.join(plot_dir, 'uncertainty_histogram.png')
        )
    
    return all_metrics