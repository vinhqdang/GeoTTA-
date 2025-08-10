"""
Comprehensive ablation studies for GeoTTA WACV 2026 paper.
Systematically evaluates each component's contribution.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple, Optional
import itertools
import numpy as np
from pathlib import Path
import logging
from copy import deepcopy

# Import GeoTTA components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from geotta.models.geometric_bridge import GeometricBridge
from geotta.models.tta_adapter import TestTimeAdapter
from experiments.experiment_manager import ExperimentManager


class AblationStudyManager:
    """
    Comprehensive ablation study manager for systematic component evaluation.
    """
    
    def __init__(self, base_config: Dict[str, Any], results_dir: str = './ablation_results'):
        self.base_config = base_config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        self._setup_logging()
        
        # Define ablation configurations
        self.ablation_configs = self._define_ablation_configs()
        
    def _setup_logging(self):
        """Setup logging for ablation studies."""
        log_file = self.results_dir / 'ablation_study.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def _define_ablation_configs(self) -> Dict[str, Dict[str, Any]]:
        """Define all ablation study configurations."""
        
        ablations = {
            # Full model (baseline)
            'full_model': {
                'description': 'Full GeoTTA model with all components',
                'modifications': {}
            },
            
            # Architecture ablations
            'no_cross_attention': {
                'description': 'Remove cross-modal attention mechanism',
                'modifications': {
                    'model.use_cross_attention': False
                }
            },
            
            'no_uncertainty_head': {
                'description': 'Remove uncertainty prediction head',
                'modifications': {
                    'model.use_uncertainty_head': False
                }
            },
            
            'linear_bridge': {
                'description': 'Replace bridge with simple linear layers',
                'modifications': {
                    'model.bridge_layers': 1,
                    'model.bridge_heads': 1,
                    'model.use_cross_attention': False
                }
            },
            
            # Geometric components
            'no_geometric_distance': {
                'description': 'Remove geometric distance in uncertainty',
                'modifications': {
                    'uncertainty.geometric_weight': 0.0,
                    'uncertainty.angular_weight': 1.0
                }
            },
            
            'no_angular_distance': {
                'description': 'Remove angular distance in uncertainty',
                'modifications': {
                    'uncertainty.geometric_weight': 1.0,
                    'uncertainty.angular_weight': 0.0
                }
            },
            
            'no_geometric_uncertainty': {
                'description': 'Remove all geometric uncertainty components',
                'modifications': {
                    'uncertainty.geometric_weight': 0.0,
                    'uncertainty.angular_weight': 0.0
                }
            },
            
            # Test-time adaptation ablations
            'no_prototype_cache': {
                'description': 'Disable prototype caching during TTA',
                'modifications': {
                    'test_time.cache_size': 0
                }
            },
            
            'no_uncertainty_weighting': {
                'description': 'Remove uncertainty-based adaptation weighting',
                'modifications': {
                    'test_time.use_uncertainty_weighting': False
                }
            },
            
            'multiple_adaptation_steps': {
                'description': 'Use multiple adaptation steps instead of single-pass',
                'modifications': {
                    'test_time.adaptation_steps': 5
                }
            },
            
            # Loss function ablations
            'only_entropy_loss': {
                'description': 'Use only entropy loss, remove consistency terms',
                'modifications': {
                    'training.consistency_weight': 0.0,
                    'training.calibration_weight': 0.0
                }
            },
            
            'no_calibration_loss': {
                'description': 'Remove uncertainty calibration loss',
                'modifications': {
                    'training.calibration_weight': 0.0
                }
            },
            
            'no_consistency_loss': {
                'description': 'Remove geometric consistency loss',
                'modifications': {
                    'training.consistency_weight': 0.0
                }
            },
            
            # Architectural variations
            'smaller_bridge': {
                'description': 'Reduce bridge dimension by half',
                'modifications': {
                    'model.bridge_dim': 256  # Half of default 512
                }
            },
            
            'larger_bridge': {
                'description': 'Double bridge dimension',
                'modifications': {
                    'model.bridge_dim': 1024  # Double of default 512
                }
            },
            
            'more_heads': {
                'description': 'Increase number of attention heads',
                'modifications': {
                    'model.bridge_heads': 16  # Double of default 8
                }
            },
            
            'fewer_heads': {
                'description': 'Reduce number of attention heads',
                'modifications': {
                    'model.bridge_heads': 4  # Half of default 8
                }
            },
            
            # Training strategy ablations
            'no_mixed_precision': {
                'description': 'Disable mixed precision training',
                'modifications': {
                    'training.mixed_precision': False
                }
            },
            
            'different_lr': {
                'description': 'Use different learning rate',
                'modifications': {
                    'test_time.adaptation_lr': 0.01  # 10x higher than default
                }
            }
        }
        
        return ablations
    
    def run_comprehensive_ablation(self, 
                                 datasets: List[str] = ['cifar10', 'cifar100'],
                                 num_seeds: int = 3,
                                 max_samples: int = 1000) -> Dict[str, Any]:
        """
        Run comprehensive ablation study across all components.
        
        Args:
            datasets: List of datasets to evaluate
            num_seeds: Number of random seeds per ablation
            max_samples: Maximum samples per dataset (for efficiency)
            
        Returns:
            Comprehensive ablation results
        """
        self.logger.info("Starting comprehensive ablation study")
        self.logger.info(f"Ablations: {len(self.ablation_configs)}")
        self.logger.info(f"Datasets: {datasets}")
        self.logger.info(f"Seeds per ablation: {num_seeds}")
        
        all_results = {}
        total_experiments = len(self.ablation_configs) * len(datasets) * num_seeds
        current_experiment = 0
        
        for ablation_name, ablation_config in self.ablation_configs.items():
            self.logger.info(f"Running ablation: {ablation_name}")
            self.logger.info(f"Description: {ablation_config['description']}")
            
            ablation_results = {}
            
            for dataset in datasets:
                dataset_results = []
                
                for seed in range(num_seeds):
                    current_experiment += 1
                    
                    self.logger.info(f"Experiment {current_experiment}/{total_experiments}")
                    self.logger.info(f"  Ablation: {ablation_name}, Dataset: {dataset}, Seed: {seed}")
                    
                    try:
                        # Create modified config
                        modified_config = self._apply_ablation_modifications(
                            self.base_config, ablation_config['modifications']
                        )
                        
                        # Run single experiment
                        result = self._run_single_ablation(
                            modified_config, dataset, seed, max_samples, ablation_name
                        )
                        
                        dataset_results.append(result)
                        
                        self.logger.info(f"  Completed. Accuracy: {result.get('accuracy', 0):.4f}")
                        
                    except Exception as e:
                        self.logger.error(f"  Failed with error: {str(e)}")
                        continue
                    
                    # Memory cleanup
                    torch.cuda.empty_cache()
                
                ablation_results[dataset] = dataset_results
            
            all_results[ablation_name] = {
                'config': ablation_config,
                'results': ablation_results
            }
        
        # Analyze and save results
        analysis_results = self._analyze_ablation_results(all_results)
        self._save_ablation_results(analysis_results)
        
        return analysis_results
    
    def _apply_ablation_modifications(self, base_config: Dict[str, Any], 
                                    modifications: Dict[str, Any]) -> Dict[str, Any]:
        """Apply modifications to base configuration."""
        config = deepcopy(base_config)
        
        for key, value in modifications.items():
            keys = key.split('.')
            current = config
            
            # Navigate to the correct nested dictionary
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Set the value
            current[keys[-1]] = value
        
        return config
    
    def _run_single_ablation(self, config: Dict[str, Any], dataset: str, 
                           seed: int, max_samples: int, ablation_name: str) -> Dict[str, Any]:
        """Run a single ablation experiment."""
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Create modified model based on ablation
        model = self._create_ablated_model(config, ablation_name)
        
        # Create TTA adapter
        tta_adapter = TestTimeAdapter(model, config)
        
        # Load dataset
        from geotta.data.benchmark_datasets import get_benchmark_dataloader
        dataloader = get_benchmark_dataloader(
            dataset_name=dataset,
            split='test',
            batch_size=1,
            clip_model=config['model']['clip_model'],
            shuffle=False
        )
        
        # Run evaluation (similar to experiment_manager)
        results = self._evaluate_ablated_model(tta_adapter, dataloader, max_samples)
        
        # Add metadata
        results.update({
            'ablation_name': ablation_name,
            'dataset': dataset,
            'seed': seed,
            'config': config
        })
        
        return results
    
    def _create_ablated_model(self, config: Dict[str, Any], ablation_name: str) -> GeometricBridge:
        """Create model with ablation modifications."""
        
        if ablation_name == 'no_cross_attention':
            return GeometricBridgeNoCrossAttention(config)
        elif ablation_name == 'no_uncertainty_head':
            return GeometricBridgeNoUncertainty(config)
        elif ablation_name == 'linear_bridge':
            return GeometricBridgeLinear(config)
        else:
            # Standard model with config modifications
            return GeometricBridge(config)
    
    def _evaluate_ablated_model(self, tta_adapter: TestTimeAdapter, 
                              dataloader, max_samples: int) -> Dict[str, Any]:
        """Evaluate ablated model."""
        
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        all_confidences = []
        
        sample_count = 0
        for batch in dataloader:
            if sample_count >= max_samples:
                break
                
            images, labels, metadata = batch
            images = images.cuda()
            
            # Predict with ablated model
            outputs = tta_adapter.predict(images)
            
            # Extract results
            logits = outputs.get('logits', torch.randn(images.shape[0], 1000).cuda())
            predictions = logits.argmax(dim=-1)
            confidences = torch.softmax(logits, dim=-1).max(dim=-1)[0]
            uncertainties = outputs.get('uncertainty', torch.zeros(images.shape[0]).cuda())
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels)
            all_confidences.append(confidences.cpu())
            all_uncertainties.append(uncertainties.cpu() if hasattr(uncertainties, 'cpu') else torch.tensor(uncertainties))
            
            sample_count += images.shape[0]
        
        # Concatenate results
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_confidences = torch.cat(all_confidences)
        all_uncertainties = torch.cat(all_uncertainties)
        
        # Compute metrics
        from geotta.utils.metrics import compute_metrics
        metrics = compute_metrics(all_predictions, all_labels, all_uncertainties)
        
        return metrics
    
    def _analyze_ablation_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ablation study results."""
        analysis = {
            'ablation_comparison': {},
            'component_importance': {},
            'statistical_significance': {}
        }
        
        # Extract results for comparison
        ablation_performance = {}
        
        for ablation_name, ablation_data in all_results.items():
            performance_scores = []
            
            for dataset, results_list in ablation_data['results'].items():
                for result in results_list:
                    accuracy = result.get('accuracy', 0)
                    performance_scores.append(accuracy)
            
            if performance_scores:
                ablation_performance[ablation_name] = {
                    'mean': np.mean(performance_scores),
                    'std': np.std(performance_scores),
                    'scores': performance_scores
                }
        
        # Compare to full model
        if 'full_model' in ablation_performance:
            full_model_performance = ablation_performance['full_model']['mean']
            
            for ablation_name, perf_data in ablation_performance.items():
                if ablation_name != 'full_model':
                    performance_drop = full_model_performance - perf_data['mean']
                    analysis['ablation_comparison'][ablation_name] = {
                        'performance_drop': performance_drop,
                        'relative_drop': performance_drop / full_model_performance * 100,
                        'mean_performance': perf_data['mean'],
                        'std_performance': perf_data['std']
                    }
        
        # Component importance ranking
        importance_scores = []
        for ablation_name, comparison_data in analysis['ablation_comparison'].items():
            importance_scores.append((
                ablation_name, 
                comparison_data['performance_drop'],
                comparison_data['relative_drop']
            ))
        
        # Sort by performance drop (importance)
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        analysis['component_importance']['ranking'] = importance_scores
        
        return analysis
    
    def _save_ablation_results(self, analysis_results: Dict[str, Any]):
        """Save ablation study results."""
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive results
        results_file = self.results_dir / f'ablation_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Create summary report
        self._create_ablation_report(analysis_results, timestamp)
        
        self.logger.info(f"Ablation results saved to {results_file}")
    
    def _create_ablation_report(self, analysis_results: Dict[str, Any], timestamp: str):
        """Create human-readable ablation report."""
        report_lines = []
        
        report_lines.append("GEOTTA ABLATION STUDY REPORT")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Component importance ranking
        if 'component_importance' in analysis_results and 'ranking' in analysis_results['component_importance']:
            report_lines.append("COMPONENT IMPORTANCE RANKING")
            report_lines.append("-" * 30)
            report_lines.append("(Higher performance drop = more important component)")
            report_lines.append("")
            
            for i, (ablation_name, abs_drop, rel_drop) in enumerate(analysis_results['component_importance']['ranking'], 1):
                report_lines.append(f"{i:2d}. {ablation_name}")
                report_lines.append(f"     Performance drop: {abs_drop:.4f} ({rel_drop:+.2f}%)")
                
                # Add description
                if ablation_name in self.ablation_configs:
                    desc = self.ablation_configs[ablation_name]['description']
                    report_lines.append(f"     Description: {desc}")
                
                report_lines.append("")
        
        # Detailed comparison
        if 'ablation_comparison' in analysis_results:
            report_lines.append("\nDETAILED ABLATION COMPARISON")
            report_lines.append("-" * 40)
            
            for ablation_name, data in analysis_results['ablation_comparison'].items():
                report_lines.append(f"\n{ablation_name}:")
                report_lines.append(f"  Mean Performance: {data['mean_performance']:.4f} ± {data['std_performance']:.4f}")
                report_lines.append(f"  Performance Drop: {data['performance_drop']:.4f} ({data['relative_drop']:+.2f}%)")
        
        # Key findings
        report_lines.append("\n\nKEY FINDINGS")
        report_lines.append("-" * 20)
        
        if 'component_importance' in analysis_results and 'ranking' in analysis_results['component_importance']:
            ranking = analysis_results['component_importance']['ranking']
            
            if ranking:
                most_important = ranking[0]
                report_lines.append(f"• Most important component: {most_important[0]}")
                report_lines.append(f"  (Performance drop: {most_important[1]:.4f})")
                
                least_important = ranking[-1]
                report_lines.append(f"• Least important component: {least_important[0]}")
                report_lines.append(f"  (Performance drop: {least_important[1]:.4f})")
                
                # Find components with large impact
                large_impact = [item for item in ranking if item[2] > 5.0]  # >5% relative drop
                if large_impact:
                    report_lines.append(f"• Components with large impact (>5% drop):")
                    for item in large_impact:
                        report_lines.append(f"  - {item[0]}: {item[2]:.2f}%")
        
        # Save report
        report_file = self.results_dir / f'ablation_report_{timestamp}.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))


# Ablated model variants
class GeometricBridgeNoCrossAttention(GeometricBridge):
    """GeometricBridge without cross-modal attention."""
    
    def __init__(self, config):
        super().__init__(config)
        # Remove cross-attention
        self.cross_attention = nn.Identity()
    
    def forward(self, images, texts=None, return_uncertainty=True):
        """Forward without cross-attention."""
        # Extract CLIP features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
            
            if texts is not None:
                text_features = self.clip_model.encode_text(texts)
                text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
            else:
                text_features = self.get_cached_prototypes()
        
        # Simple projection without cross-attention
        img_bridge = self.image_proj(image_features)
        
        # Compute uncertainty if needed
        uncertainty = None
        if return_uncertainty:
            txt_bridge = self.text_proj(text_features)
            uncertainty_input = torch.cat([img_bridge, txt_bridge[:img_bridge.shape[0]]], dim=-1)
            uncertainty = self.uncertainty_head(uncertainty_input)
        
        # Project back to CLIP space
        adapted_features = self.output_proj(img_bridge)
        adapted_features = torch.nn.functional.normalize(adapted_features, p=2, dim=-1)
        
        return {
            'adapted_features': adapted_features,
            'uncertainty': uncertainty,
            'attention_weights': None,
            'original_image_features': image_features,
            'original_text_features': text_features if texts is not None else None
        }


class GeometricBridgeNoUncertainty(GeometricBridge):
    """GeometricBridge without uncertainty head."""
    
    def __init__(self, config):
        super().__init__(config)
        # Remove uncertainty head
        self.uncertainty_head = None
    
    def forward(self, images, texts=None, return_uncertainty=True):
        """Forward without uncertainty prediction."""
        result = super().forward(images, texts, return_uncertainty=False)
        result['uncertainty'] = torch.zeros(images.shape[0], device=images.device)
        return result


class GeometricBridgeLinear(GeometricBridge):
    """GeometricBridge with only linear transformations."""
    
    def __init__(self, config):
        # Modify config for linear bridge
        linear_config = config.copy()
        linear_config['model']['bridge_heads'] = 1
        linear_config['model']['bridge_layers'] = 1
        
        super().__init__(linear_config)
        
        # Replace cross-attention with simple linear layer
        dim = config['model']['bridge_dim']
        self.cross_attention = nn.Linear(dim, dim)
    
    def forward(self, images, texts=None, return_uncertainty=True):
        """Forward with linear transformations only."""
        # Extract CLIP features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
            
            if texts is not None:
                text_features = self.clip_model.encode_text(texts)
                text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
            else:
                text_features = self.get_cached_prototypes()
        
        # Project to bridge space
        img_bridge = self.image_proj(image_features)
        
        # Simple linear transformation instead of attention
        aligned_img = self.cross_attention(img_bridge)
        
        # Compute uncertainty
        uncertainty = None
        if return_uncertainty:
            txt_bridge = self.text_proj(text_features)
            uncertainty_input = torch.cat([img_bridge, aligned_img], dim=-1)
            uncertainty = self.uncertainty_head(uncertainty_input)
        
        # Project back to CLIP space
        adapted_features = self.output_proj(aligned_img)
        adapted_features = torch.nn.functional.normalize(adapted_features, p=2, dim=-1)
        
        return {
            'adapted_features': adapted_features,
            'uncertainty': uncertainty,
            'attention_weights': None,
            'original_image_features': image_features,
            'original_text_features': text_features if texts is not None else None
        }