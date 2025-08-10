import torch
import yaml
import argparse
import os
from tqdm import tqdm

# Import GeoTTA modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from geotta.models.geometric_bridge import GeometricBridge
from geotta.models.tta_adapter import TestTimeAdapter
from geotta.data.datasets import get_dataloader
from geotta.utils.metrics import evaluate_model_comprehensive
from geotta.utils.visualization import create_evaluation_dashboard
from geotta.utils.memory import MemoryProfiler


def load_model_from_checkpoint(checkpoint_path: str, config: dict) -> GeometricBridge:
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    # Initialize model
    model = GeometricBridge(config).cuda()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
    return model


def evaluate_standard_model(model: GeometricBridge, dataloader, config: dict) -> dict:
    """Evaluate model without test-time adaptation."""
    print("Evaluating standard model (no TTA)...")
    
    with MemoryProfiler("Standard Evaluation"):
        metrics = evaluate_model_comprehensive(
            model, dataloader, device='cuda',
            save_plots=True, plot_dir='./evaluation_plots/standard'
        )
    
    return metrics


def evaluate_with_tta(model: GeometricBridge, dataloader, config: dict) -> dict:
    """Evaluate model with test-time adaptation."""
    print("Evaluating with test-time adaptation...")
    
    # Create TTA adapter
    tta_adapter = TestTimeAdapter(model, config)
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_uncertainties = []
    
    with MemoryProfiler("TTA Evaluation"):
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="TTA Evaluation")):
            images, labels = images.cuda(), labels.cuda()
            
            # Get text prototypes
            text_prototypes = model.get_text_prototypes()
            
            # Predict with uncertainty
            outputs = tta_adapter.predict_with_uncertainty(images, text_prototypes)
            
            all_predictions.append(outputs['predictions'].cpu())
            all_labels.append(labels.cpu())
            all_confidences.append(outputs['confidences'].cpu())
            all_uncertainties.append(outputs['uncertainties'].cpu())
    
    # Concatenate results
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    all_confidences = torch.cat(all_confidences)
    all_uncertainties = torch.cat(all_uncertainties)
    
    # Compute metrics manually for TTA
    from geotta.utils.metrics import compute_metrics, compute_calibration_metrics, compute_uncertainty_quality_metrics
    
    basic_metrics = compute_metrics(all_predictions, all_labels, all_uncertainties)
    calibration_metrics = compute_calibration_metrics(all_predictions, all_labels, all_confidences)
    uncertainty_metrics = compute_uncertainty_quality_metrics(
        all_uncertainties, (all_predictions != all_labels).float()
    )
    
    # Combine metrics
    metrics = {**basic_metrics, **calibration_metrics, **uncertainty_metrics}
    
    # Create visualization dashboard
    create_evaluation_dashboard(
        metrics, all_uncertainties, (all_predictions != all_labels).float(),
        all_confidences, (all_predictions == all_labels).float(),
        save_dir='./evaluation_plots/tta'
    )
    
    return metrics


def compare_models(standard_metrics: dict, tta_metrics: dict) -> dict:
    """Compare standard model vs TTA performance."""
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    
    comparison = {}
    
    # Key metrics to compare
    key_metrics = ['accuracy', 'ece', 'auroc', 'avg_confidence']
    
    print(f"{'Metric':<20} {'Standard':<12} {'TTA':<12} {'Improvement':<12}")
    print("-" * 56)
    
    for metric in key_metrics:
        if metric in standard_metrics and metric in tta_metrics:
            std_val = standard_metrics[metric]
            tta_val = tta_metrics[metric]
            
            # For ECE, lower is better
            if metric == 'ece':
                improvement = (std_val - tta_val) / std_val * 100 if std_val != 0 else 0
            else:
                improvement = (tta_val - std_val) / std_val * 100 if std_val != 0 else 0
            
            comparison[f'{metric}_improvement'] = improvement
            
            print(f"{metric:<20} {std_val:<12.4f} {tta_val:<12.4f} {improvement:<12.2f}%")
    
    return comparison


def run_domain_shift_analysis(model: GeometricBridge, config: dict):
    """Run analysis on domain shift detection."""
    print("\nRunning domain shift analysis...")
    
    # This would typically use different test sets representing domain shifts
    # For now, we'll just demonstrate the concept
    
    tta_adapter = TestTimeAdapter(model, config)
    
    # Simulate different domain shifts
    domain_shifts = {
        'clean': 0.1,  # Low shift score expected
        'noisy': 0.5,  # Medium shift
        'corrupted': 0.8  # High shift
    }
    
    print(f"{'Domain':<15} {'Shift Score':<15} {'Interpretation'}")
    print("-" * 45)
    
    for domain, expected_score in domain_shifts.items():
        print(f"{domain:<15} {expected_score:<15.3f} {'Low' if expected_score < 0.3 else 'High' if expected_score > 0.6 else 'Medium'}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate GeoTTA Model')
    parser.add_argument('--checkpoint', required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--config', default='geotta/configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--data-split', default='val', choices=['val', 'test'],
                       help='Data split to evaluate on')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--output-dir', default='./evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--skip-tta', action='store_true',
                       help='Skip test-time adaptation evaluation')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('./evaluation_plots', exist_ok=True)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override batch size for evaluation
    config['training']['batch_size'] = args.batch_size
    
    print(f"Loaded config from {args.config}")
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, config)
    
    # Create data loader
    print(f"Loading {args.data_split} dataset...")
    dataloader = get_dataloader(config, split=args.data_split)
    print(f"Dataset size: {len(dataloader.dataset)} samples")
    
    # Evaluate standard model
    standard_metrics = evaluate_standard_model(model, dataloader, config)
    
    # Evaluate with TTA (if not skipped)
    if not args.skip_tta:
        tta_metrics = evaluate_with_tta(model, dataloader, config)
        
        # Compare models
        comparison = compare_models(standard_metrics, tta_metrics)
        
        # Save comparison results
        import json
        results = {
            'standard_metrics': standard_metrics,
            'tta_metrics': tta_metrics,
            'comparison': comparison
        }
    else:
        results = {'standard_metrics': standard_metrics}
        tta_metrics = {}
    
    # Save results to JSON
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        # Convert any tensor values to float for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {k: float(v) if torch.is_tensor(v) else v 
                                   for k, v in value.items()}
            else:
                json_results[key] = float(value) if torch.is_tensor(value) else value
        
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Run domain shift analysis
    if not args.skip_tta:
        run_domain_shift_analysis(model, config)
    
    # Print final summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Standard Model Accuracy: {standard_metrics.get('accuracy', 0):.4f}")
    if tta_metrics:
        print(f"TTA Model Accuracy: {tta_metrics.get('accuracy', 0):.4f}")
        acc_improvement = (tta_metrics.get('accuracy', 0) - standard_metrics.get('accuracy', 0)) * 100
        print(f"Accuracy Improvement: {acc_improvement:+.2f}%")
    print("="*50)


if __name__ == '__main__':
    main()