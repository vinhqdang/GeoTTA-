import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Tuple
import os


def plot_training_curves(train_losses: List[float], val_losses: List[float],
                        val_accuracies: List[float], save_path: str = None):
    """Plot training curves for loss and accuracy."""
    epochs = range(len(train_losses))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(epochs, train_losses, label='Train Loss', marker='o')
    ax1.plot(epochs, val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs, val_accuracies, label='Val Accuracy', marker='o', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_attention_weights(attention_weights: torch.Tensor, 
                          save_path: str = None, max_examples: int = 4):
    """Plot attention weight heatmaps."""
    # attention_weights: [batch_size, num_heads, seq_len, seq_len]
    attention_weights = attention_weights.cpu().numpy()
    batch_size = min(attention_weights.shape[0], max_examples)
    num_heads = attention_weights.shape[1]
    
    fig, axes = plt.subplots(batch_size, num_heads, 
                           figsize=(num_heads * 3, batch_size * 3))
    
    if batch_size == 1 and num_heads == 1:
        axes = [[axes]]
    elif batch_size == 1:
        axes = [axes]
    elif num_heads == 1:
        axes = [[ax] for ax in axes]
    
    for i in range(batch_size):
        for j in range(num_heads):
            sns.heatmap(attention_weights[i, j], 
                       ax=axes[i][j], 
                       cmap='Blues', 
                       cbar=True)
            axes[i][j].set_title(f'Sample {i}, Head {j}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_uncertainty_distribution(uncertainties: torch.Tensor, 
                                errors: torch.Tensor,
                                save_path: str = None):
    """Plot distribution of uncertainties for correct vs incorrect predictions."""
    uncertainties_np = uncertainties.cpu().numpy()
    errors_np = errors.cpu().numpy()
    
    correct_uncertainties = uncertainties_np[errors_np == 0]
    incorrect_uncertainties = uncertainties_np[errors_np == 1]
    
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(correct_uncertainties, bins=50, alpha=0.7, 
            label=f'Correct ({len(correct_uncertainties)})', density=True)
    plt.hist(incorrect_uncertainties, bins=50, alpha=0.7, 
            label=f'Incorrect ({len(incorrect_uncertainties)})', density=True)
    
    # Add vertical lines for means
    plt.axvline(correct_uncertainties.mean(), color='blue', linestyle='--', 
               label=f'Correct Mean: {correct_uncertainties.mean():.3f}')
    plt.axvline(incorrect_uncertainties.mean(), color='orange', linestyle='--',
               label=f'Incorrect Mean: {incorrect_uncertainties.mean():.3f}')
    
    plt.xlabel('Uncertainty')
    plt.ylabel('Density')
    plt.title('Uncertainty Distribution: Correct vs Incorrect Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confidence_vs_accuracy(confidences: torch.Tensor, 
                               accuracies: torch.Tensor,
                               save_path: str = None, num_bins: int = 10):
    """Plot confidence vs accuracy calibration plot."""
    conf_np = confidences.cpu().numpy()
    acc_np = accuracies.cpu().numpy()
    
    # Create bins
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute accuracy for each bin
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(num_bins):
        mask = (conf_np >= bin_edges[i]) & (conf_np < bin_edges[i + 1])
        if i == num_bins - 1:  # Include the last edge for the last bin
            mask = (conf_np >= bin_edges[i]) & (conf_np <= bin_edges[i + 1])
        
        if mask.sum() > 0:
            bin_acc = acc_np[mask].mean()
            bin_conf = conf_np[mask].mean()
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(0)
            bin_confidences.append(bin_centers[i])
            bin_counts.append(0)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calibration plot
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax1.plot(bin_confidences, bin_accuracies, 'o-', label='Model')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Calibration Plot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bar plot showing counts per bin
    ax2.bar(bin_centers, bin_counts, width=0.8/num_bins, alpha=0.7)
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Sample Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_space_2d(features: torch.Tensor, labels: torch.Tensor,
                         save_path: str = None, method: str = 'pca'):
    """Plot 2D visualization of feature space."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
        features_2d = reducer.fit_transform(features_np)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        features_2d = reducer.fit_transform(features_np)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels_np, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'Feature Space Visualization ({method.upper()})')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_evaluation_dashboard(metrics: Dict[str, float], 
                              uncertainties: torch.Tensor,
                              errors: torch.Tensor,
                              confidences: torch.Tensor,
                              accuracies: torch.Tensor,
                              save_dir: str):
    """Create a comprehensive evaluation dashboard."""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Metrics summary
    fig, ax = plt.subplots(figsize=(10, 6))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax.bar(metric_names, metric_values)
    ax.set_ylabel('Value')
    ax.set_title('Model Performance Metrics')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_summary.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Uncertainty distribution
    plot_uncertainty_distribution(
        uncertainties, errors,
        save_path=os.path.join(save_dir, 'uncertainty_distribution.png')
    )
    
    # 3. Calibration plot
    plot_confidence_vs_accuracy(
        confidences, accuracies,
        save_path=os.path.join(save_dir, 'calibration_plot.png')
    )
    
    # 4. Uncertainty vs confidence scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(confidences.cpu(), uncertainties.cpu(), alpha=0.5)
    plt.xlabel('Confidence')
    plt.ylabel('Uncertainty')
    plt.title('Confidence vs Uncertainty')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'confidence_vs_uncertainty.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dashboard saved to {save_dir}")


def save_model_predictions(predictions: torch.Tensor, 
                          labels: torch.Tensor,
                          confidences: torch.Tensor,
                          uncertainties: torch.Tensor,
                          save_path: str):
    """Save model predictions to CSV for further analysis."""
    import pandas as pd
    
    df = pd.DataFrame({
        'prediction': predictions.cpu().numpy(),
        'label': labels.cpu().numpy(),
        'confidence': confidences.cpu().numpy(),
        'uncertainty': uncertainties.cpu().numpy(),
        'correct': (predictions == labels).cpu().numpy().astype(int)
    })
    
    df.to_csv(save_path, index=False)
    print(f"Predictions saved to {save_path}")


def plot_learning_curve(train_sizes: List[int], 
                       train_scores: List[float],
                       val_scores: List[float],
                       save_path: str = None):
    """Plot learning curve showing performance vs training set size."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', label='Training Score')
    plt.plot(train_sizes, val_scores, 'o-', label='Validation Score')
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()