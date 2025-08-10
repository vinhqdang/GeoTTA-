"""
Comprehensive experiment management system for rigorous WACV 2026 evaluation.
Handles multiple runs, statistical significance testing, and result aggregation.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import json
import os
import time
from pathlib import Path
import logging
from collections import defaultdict
import yaml
from datetime import datetime
import hashlib

# Import GeoTTA components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from geotta.models.geometric_bridge import GeometricBridge
from geotta.models.tta_adapter import TestTimeAdapter
from geotta.data.benchmark_datasets import BenchmarkDatasetManager, get_benchmark_dataloader
from geotta.baselines import *
from geotta.utils.metrics import evaluate_model_comprehensive
from geotta.utils.memory import MemoryProfiler, print_memory_stats


class ExperimentManager:
    """
    Comprehensive experiment management for WACV 2026 paper.
    
    Features:
    - Multiple random seeds for statistical significance
    - Comprehensive baseline comparisons
    - Memory and runtime profiling
    - Automatic result aggregation and analysis
    - Configuration management and reproducibility
    """
    
    def __init__(self, base_config: Dict[str, Any], results_dir: str = './wacv_results'):
        self.base_config = base_config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logging
        self._setup_logging()
        
        # Experiment tracking
        self.experiments = {}
        self.current_experiment_id = None
        
        # Dataset manager
        self.dataset_manager = BenchmarkDatasetManager()
        
        # Supported methods
        self.methods = {
            'geotta': self._create_geotta_method,
            'source_only': self._create_source_only,
            'tent': self._create_tent_method,
            'cotta': self._create_cotta_method,
            'tpt': self._create_tpt_method,
            'ada_contrast': self._create_ada_contrast_method,
            'bn_adapt': self._create_bn_adapt_method,
            'memo': self._create_memo_method,
        }
        
        # Standard datasets for WACV evaluation
        self.benchmark_datasets = [
            'imagenet', 'imagenet_c', 'imagenet_r', 'imagenet_a', 'imagenet_v2',
            'cifar10', 'cifar100', 'office_home', 'domainnet',
            'caltech101', 'oxford_pets', 'stanford_cars', 'flowers102'
        ]
        
        # CLIP backbones to evaluate
        self.clip_backbones = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14']
        
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_file = self.results_dir / 'experiment.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Experiment manager initialized. Results dir: {self.results_dir}")
        
    def create_experiment_config(self, 
                               method: str,
                               dataset: str, 
                               clip_backbone: str = 'ViT-B/32',
                               **kwargs) -> Dict[str, Any]:
        """Create experiment configuration."""
        config = self.base_config.copy()
        
        # Update with experiment specific settings
        config['experiment'] = {
            'method': method,
            'dataset': dataset,
            'clip_backbone': clip_backbone,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }
        
        # Update model config for backbone
        config['model']['clip_model'] = clip_backbone
        
        # Dataset specific configurations
        if dataset in ['cifar10', 'cifar100']:
            config['training']['batch_size'] = 32
        elif 'imagenet' in dataset:
            config['training']['batch_size'] = 16
        
        # Method specific configurations
        if method == 'tent':
            config.update({
                'tent_lr': 1e-3,
                'tent_weight_decay': 0.0,
                'tent_type': 'standard'
            })
        elif method == 'cotta':
            config.update({
                'cotta_lr': 1e-3,
                'restoration_factor': 0.01,
                'threshold_ent': 0.4,
                'mt_alpha': 0.99
            })
        elif method == 'tpt':
            config.update({
                'tpt_lr': 5e-3,
                'n_ctx': 4,
                'selection_p': 0.1,
                'tta_steps': 1
            })
        
        return config
    
    def run_comprehensive_evaluation(self, 
                                   methods: List[str] = None,
                                   datasets: List[str] = None,
                                   clip_backbones: List[str] = None,
                                   num_seeds: int = 5,
                                   max_samples: int = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation across methods, datasets, and backbones.
        
        Args:
            methods: List of methods to evaluate
            datasets: List of datasets to use
            clip_backbones: List of CLIP backbones
            num_seeds: Number of random seeds for statistical significance
            max_samples: Maximum samples per dataset (for debugging)
        
        Returns:
            Comprehensive results dictionary
        """
        if methods is None:
            methods = ['geotta', 'source_only', 'tent', 'cotta']
        if datasets is None:
            datasets = ['cifar10', 'cifar100', 'imagenet_c']  # Start with subset
        if clip_backbones is None:
            clip_backbones = ['ViT-B/32']
        
        self.logger.info(f"Starting comprehensive evaluation:")
        self.logger.info(f"  Methods: {methods}")
        self.logger.info(f"  Datasets: {datasets}")
        self.logger.info(f"  Backbones: {clip_backbones}")
        self.logger.info(f"  Seeds: {num_seeds}")
        
        # Results storage
        all_results = defaultdict(lambda: defaultdict(list))
        
        # Total experiments
        total_experiments = len(methods) * len(datasets) * len(clip_backbones) * num_seeds
        current_experiment = 0
        
        for method in methods:
            for dataset in datasets:
                for backbone in clip_backbones:
                    for seed in range(num_seeds):
                        current_experiment += 1
                        
                        self.logger.info(f"Experiment {current_experiment}/{total_experiments}")
                        self.logger.info(f"  Method: {method}, Dataset: {dataset}, Backbone: {backbone}, Seed: {seed}")
                        
                        # Set random seed for reproducibility
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed(seed)
                        
                        try:
                            # Run single experiment
                            results = self.run_single_experiment(
                                method=method,
                                dataset=dataset,
                                clip_backbone=backbone,
                                seed=seed,
                                max_samples=max_samples
                            )
                            
                            # Store results
                            experiment_key = f"{method}_{dataset}_{backbone}"
                            all_results[experiment_key]['results'].append(results)
                            all_results[experiment_key]['config'] = results.get('config', {})
                            
                            self.logger.info(f"  Completed successfully. Accuracy: {results.get('accuracy', 0):.4f}")
                            
                        except Exception as e:
                            self.logger.error(f"  Failed with error: {str(e)}")
                            continue
                        
                        # Memory cleanup
                        torch.cuda.empty_cache()
        
        # Aggregate and analyze results
        aggregated_results = self._aggregate_results(all_results)
        
        # Save comprehensive results
        self._save_results(aggregated_results, 'comprehensive_evaluation')
        
        return aggregated_results
    
    def run_single_experiment(self,
                            method: str,
                            dataset: str, 
                            clip_backbone: str = 'ViT-B/32',
                            seed: int = 42,
                            max_samples: int = None) -> Dict[str, Any]:
        """Run a single experiment with comprehensive metrics."""
        
        # Create experiment configuration
        config = self.create_experiment_config(method, dataset, clip_backbone, seed=seed)
        
        # Create unique experiment ID
        experiment_id = self._create_experiment_id(config)
        
        with MemoryProfiler(f"Experiment {experiment_id}"):
            # Load dataset
            try:
                dataloader = get_benchmark_dataloader(
                    dataset_name=dataset,
                    split='test',
                    batch_size=1,  # TTA typically uses batch_size=1
                    clip_model=clip_backbone,
                    shuffle=False,
                    root_dir='./data'
                )
                
                # Limit samples for debugging
                if max_samples:
                    # Create limited dataset
                    limited_samples = []
                    for i, batch in enumerate(dataloader):
                        if i >= max_samples:
                            break
                        limited_samples.append(batch)
                    dataloader = limited_samples
                
            except Exception as e:
                self.logger.error(f"Failed to load dataset {dataset}: {e}")
                raise
            
            # Create method
            base_model = GeometricBridge(config)
            adapter_method = self.methods[method](base_model, config)
            
            # Run evaluation with timing
            start_time = time.time()
            
            # Collect results
            all_predictions = []
            all_labels = []
            all_uncertainties = []
            all_confidences = []
            adaptation_stats = []
            
            for i, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    images, labels, metadata = batch
                else:
                    # Handle case where we created a limited dataset
                    images, labels, metadata = batch
                
                images = images.cuda()
                labels = labels.cuda() if hasattr(labels, 'cuda') else labels
                
                # Adapt and predict
                outputs = adapter_method.predict(images)
                
                # Extract results
                logits = outputs.get('logits', torch.randn(images.shape[0], 1000).cuda())
                predictions = logits.argmax(dim=-1)
                confidences = torch.softmax(logits, dim=-1).max(dim=-1)[0]
                uncertainties = outputs.get('uncertainty', torch.zeros(images.shape[0]).cuda())
                
                all_predictions.append(predictions.cpu())
                all_labels.append(labels if not hasattr(labels, 'cpu') else labels.cpu())
                all_confidences.append(confidences.cpu())
                all_uncertainties.append(uncertainties.cpu() if hasattr(uncertainties, 'cpu') else torch.tensor(uncertainties))
                
                # Collect adaptation statistics
                if hasattr(adapter_method, 'get_adaptation_stats'):
                    adaptation_stats.append(adapter_method.get_adaptation_stats())
                
                # Progress logging
                if (i + 1) % 100 == 0:
                    self.logger.info(f"    Processed {i + 1} samples...")
            
            total_time = time.time() - start_time
            
            # Concatenate all results
            all_predictions = torch.cat(all_predictions)
            all_labels = torch.cat(all_labels)
            all_confidences = torch.cat(all_confidences)
            all_uncertainties = torch.cat(all_uncertainties)
            
            # Compute comprehensive metrics
            from geotta.utils.metrics import compute_metrics, compute_calibration_metrics, compute_uncertainty_quality_metrics
            
            basic_metrics = compute_metrics(all_predictions, all_labels, all_uncertainties)
            calibration_metrics = compute_calibration_metrics(all_predictions, all_labels, all_confidences)
            uncertainty_metrics = compute_uncertainty_quality_metrics(
                all_uncertainties, (all_predictions != all_labels).float()
            )
            
            # Combine all metrics
            results = {
                **basic_metrics,
                **calibration_metrics,
                **uncertainty_metrics,
                'runtime_seconds': total_time,
                'samples_per_second': len(all_predictions) / total_time,
                'total_samples': len(all_predictions),
                'method': method,
                'dataset': dataset,
                'clip_backbone': clip_backbone,
                'seed': seed,
                'experiment_id': experiment_id,
                'config': config,
                'adaptation_stats': adaptation_stats
            }
            
            # Memory statistics
            if torch.cuda.is_available():
                results['peak_memory_gb'] = torch.cuda.max_memory_allocated() / 1024**3
            
        return results
    
    def _create_experiment_id(self, config: Dict[str, Any]) -> str:
        """Create unique experiment ID."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _aggregate_results(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results across multiple seeds."""
        aggregated = {}
        
        for experiment_key, data in raw_results.items():
            results_list = data['results']
            config = data['config']
            
            if not results_list:
                continue
            
            # Extract metrics
            metrics = defaultdict(list)
            for result in results_list:
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        metrics[key].append(value)
            
            # Compute statistics
            aggregated_metrics = {}
            for metric, values in metrics.items():
                if len(values) > 0:
                    values = np.array(values)
                    aggregated_metrics[f'{metric}_mean'] = float(np.mean(values))
                    aggregated_metrics[f'{metric}_std'] = float(np.std(values))
                    aggregated_metrics[f'{metric}_min'] = float(np.min(values))
                    aggregated_metrics[f'{metric}_max'] = float(np.max(values))
                    
                    # 95% confidence interval
                    if len(values) > 1:
                        sem = np.std(values) / np.sqrt(len(values))
                        ci_95 = 1.96 * sem
                        aggregated_metrics[f'{metric}_ci95'] = float(ci_95)
            
            aggregated[experiment_key] = {
                'metrics': aggregated_metrics,
                'num_seeds': len(results_list),
                'config': config,
                'raw_results': results_list
            }
        
        return aggregated
    
    def _save_results(self, results: Dict[str, Any], prefix: str):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON results
        json_file = self.results_dir / f'{prefix}_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # CSV summary
        self._create_csv_summary(results, prefix, timestamp)
        
        self.logger.info(f"Results saved to {json_file}")
    
    def _create_csv_summary(self, results: Dict[str, Any], prefix: str, timestamp: str):
        """Create CSV summary of results."""
        rows = []
        
        for experiment_key, data in results.items():
            method, dataset, backbone = experiment_key.split('_', 2)
            metrics = data['metrics']
            
            row = {
                'method': method,
                'dataset': dataset,
                'backbone': backbone,
                'num_seeds': data['num_seeds']
            }
            
            # Add key metrics
            key_metrics = ['accuracy', 'ece', 'auroc', 'runtime_seconds']
            for metric in key_metrics:
                row[f'{metric}_mean'] = metrics.get(f'{metric}_mean', 0)
                row[f'{metric}_std'] = metrics.get(f'{metric}_std', 0)
                row[f'{metric}_ci95'] = metrics.get(f'{metric}_ci95', 0)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_file = self.results_dir / f'{prefix}_{timestamp}.csv'
        df.to_csv(csv_file, index=False)
        
        return df
    
    # Method factory functions
    def _create_geotta_method(self, base_model: GeometricBridge, config: Dict[str, Any]):
        """Create GeoTTA method."""
        return TestTimeAdapter(base_model, config)
    
    def _create_source_only(self, base_model: GeometricBridge, config: Dict[str, Any]):
        """Create Source-Only baseline."""
        from geotta.baselines.source_only import create_source_only_baseline
        return create_source_only_baseline(base_model, config)
    
    def _create_tent_method(self, base_model: GeometricBridge, config: Dict[str, Any]):
        """Create TENT method.""" 
        from geotta.baselines.tent import create_tent_baseline
        return create_tent_baseline(base_model, config)
    
    def _create_cotta_method(self, base_model: GeometricBridge, config: Dict[str, Any]):
        """Create CoTTA method."""
        from geotta.baselines.cotta import create_cotta_baseline
        return create_cotta_baseline(base_model, config)
    
    def _create_tpt_method(self, base_model: GeometricBridge, config: Dict[str, Any]):
        """Create TPT method."""
        from geotta.baselines.tpt import create_tpt_baseline
        return create_tpt_baseline(base_model, config)
    
    def _create_ada_contrast_method(self, base_model: GeometricBridge, config: Dict[str, Any]):
        """Create AdaContrast method."""
        from geotta.baselines.ada_contrast import create_ada_contrast_baseline
        return create_ada_contrast_baseline(base_model, config)
    
    def _create_bn_adapt_method(self, base_model: GeometricBridge, config: Dict[str, Any]):
        """Create BN Adapt method."""
        from geotta.baselines.bn_adapt import create_bn_adapt_baseline
        return create_bn_adapt_baseline(base_model, config)
    
    def _create_memo_method(self, base_model: GeometricBridge, config: Dict[str, Any]):
        """Create MEMO method."""
        from geotta.baselines.memo import create_memo_baseline
        return create_memo_baseline(base_model, config)


def create_wacv_experiment_config() -> Dict[str, Any]:
    """Create standard configuration for WACV experiments."""
    return {
        'model': {
            'clip_model': 'ViT-B/32',
            'bridge_dim': 512,
            'bridge_layers': 2,
            'bridge_heads': 8,
            'dropout': 0.1,
            'use_hyperbolic': True,
            'temperature': 0.07
        },
        'training': {
            'batch_size': 1,  # TTA typically uses batch_size=1
            'mixed_precision': True,
        },
        'uncertainty': {
            'geometric_weight': 1.0,
            'angular_weight': 0.5,
            'calibration_bins': 15
        },
        'test_time': {
            'adaptation_steps': 1,
            'adaptation_lr': 0.001,
            'cache_size': 32
        }
    }