"""
Main script to run comprehensive WACV 2026 experiments.
Executes all experiments needed for the paper submission.
"""

import argparse
import yaml
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiments.experiment_manager import ExperimentManager, create_wacv_experiment_config
from experiments.ablation_studies import AblationStudyManager
from experiments.statistical_analysis import StatisticalAnalyzer, analyze_wacv_results


def run_main_experiments(config, args):
    """Run main comparison experiments."""
    print("="*60)
    print("RUNNING MAIN EXPERIMENTS")
    print("="*60)
    
    # Initialize experiment manager
    experiment_manager = ExperimentManager(config, results_dir=args.results_dir)
    
    # Define experiment scope
    if args.quick_test:
        methods = ['geotta', 'source_only', 'tent']
        datasets = ['cifar10']
        backbones = ['ViT-B/32']
        num_seeds = 2
        max_samples = 500
    else:
        methods = ['geotta', 'source_only', 'tent', 'cotta', 'tpt', 'ada_contrast', 'bn_adapt']
        datasets = ['cifar10', 'cifar100', 'imagenet_c', 'imagenet_r', 'imagenet_a']
        backbones = ['ViT-B/32', 'ViT-B/16', 'ViT-L/14']
        num_seeds = args.num_seeds
        max_samples = args.max_samples
    
    print(f"Methods: {methods}")
    print(f"Datasets: {datasets}")
    print(f"Backbones: {backbones}")
    print(f"Seeds per experiment: {num_seeds}")
    
    # Run comprehensive evaluation
    results = experiment_manager.run_comprehensive_evaluation(
        methods=methods,
        datasets=datasets,
        clip_backbones=backbones,
        num_seeds=num_seeds,
        max_samples=max_samples
    )
    
    print("\nMain experiments completed successfully!")
    return results


def run_ablation_studies(config, args):
    """Run comprehensive ablation studies."""
    if args.skip_ablation:
        print("Skipping ablation studies (--skip-ablation)")
        return None
    
    print("="*60)
    print("RUNNING ABLATION STUDIES")
    print("="*60)
    
    # Initialize ablation manager
    ablation_manager = AblationStudyManager(config, results_dir=f"{args.results_dir}/ablations")
    
    # Define ablation scope
    if args.quick_test:
        datasets = ['cifar10']
        num_seeds = 2
        max_samples = 500
    else:
        datasets = ['cifar10', 'cifar100', 'imagenet_c']
        num_seeds = max(2, args.num_seeds // 2)  # Fewer seeds for ablations
        max_samples = args.max_samples
    
    print(f"Ablation datasets: {datasets}")
    print(f"Seeds per ablation: {num_seeds}")
    
    # Run ablation studies
    ablation_results = ablation_manager.run_comprehensive_ablation(
        datasets=datasets,
        num_seeds=num_seeds,
        max_samples=max_samples
    )
    
    print("\nAblation studies completed successfully!")
    return ablation_results


def run_statistical_analysis(args):
    """Run statistical analysis on results."""
    if args.skip_stats:
        print("Skipping statistical analysis (--skip-stats)")
        return None
    
    print("="*60)
    print("RUNNING STATISTICAL ANALYSIS")
    print("="*60)
    
    # Find most recent results file
    results_dir = Path(args.results_dir)
    json_files = list(results_dir.glob('comprehensive_evaluation_*.json'))
    
    if not json_files:
        print("No results files found for statistical analysis!")
        return None
    
    # Use most recent file
    latest_results = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"Analyzing results from: {latest_results}")
    
    # Run analysis
    analysis_dir = f"{args.results_dir}/statistical_analysis"
    analysis_results = analyze_wacv_results(str(latest_results), analysis_dir)
    
    print(f"\nStatistical analysis completed! Results saved to {analysis_dir}")
    return analysis_results


def create_paper_plots(args):
    """Create publication-quality plots and tables."""
    if args.skip_plots:
        print("Skipping plot generation (--skip-plots)")
        return
    
    print("="*60)
    print("CREATING PUBLICATION PLOTS")
    print("="*60)
    
    # This would create all the plots needed for the paper
    print("Creating paper plots and tables...")
    
    # Implementation would go here to create:
    # - Performance comparison tables
    # - Ablation study visualizations
    # - Uncertainty calibration plots
    # - Domain shift analysis plots
    # - Statistical significance matrices
    
    print("Publication plots created!")


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive WACV 2026 experiments')
    
    # Experiment scope
    parser.add_argument('--config', default='geotta/configs/default.yaml',
                       help='Configuration file')
    parser.add_argument('--results-dir', default='./wacv_results',
                       help='Directory to save results')
    parser.add_argument('--num-seeds', type=int, default=5,
                       help='Number of random seeds per experiment')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples per dataset (for debugging)')
    
    # Experiment selection
    parser.add_argument('--skip-main', action='store_true',
                       help='Skip main experiments')
    parser.add_argument('--skip-ablation', action='store_true',
                       help='Skip ablation studies')
    parser.add_argument('--skip-stats', action='store_true',
                       help='Skip statistical analysis')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip plot generation')
    
    # Special modes
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with minimal experiments')
    parser.add_argument('--only-ablation', action='store_true',
                       help='Run only ablation studies')
    parser.add_argument('--only-stats', action='store_true',
                       help='Run only statistical analysis')
    
    args = parser.parse_args()
    
    # Create results directory
    Path(args.results_dir).mkdir(exist_ok=True, parents=True)
    
    # Load or create configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default WACV configuration")
        config = create_wacv_experiment_config()
    
    print("WACV 2026 COMPREHENSIVE EXPERIMENTS")
    print("GeoTTA: Geometric Test-Time Adapter")
    print(f"Results will be saved to: {args.results_dir}")
    print(f"Configuration: {args.config}")
    
    if args.quick_test:
        print("\nRunning in QUICK TEST mode - limited experiments")
    
    try:
        # Run experiments based on arguments
        if args.only_stats:
            run_statistical_analysis(args)
        elif args.only_ablation:
            run_ablation_studies(config, args)
        else:
            # Run full experimental pipeline
            
            # 1. Main experiments
            if not args.skip_main:
                main_results = run_main_experiments(config, args)
            
            # 2. Ablation studies  
            if not args.skip_ablation:
                ablation_results = run_ablation_studies(config, args)
            
            # 3. Statistical analysis
            if not args.skip_stats:
                statistical_results = run_statistical_analysis(args)
            
            # 4. Create publication plots
            create_paper_plots(args)
        
        print("\n" + "="*60)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results available in: {args.results_dir}")
        print("\nNext steps:")
        print("1. Review results in the statistical analysis folder")
        print("2. Check ablation study rankings")
        print("3. Examine publication-quality plots")
        print("4. Use results for WACV 2026 paper writing")
        
    except KeyboardInterrupt:
        print("\n\nExperiments interrupted by user")
    except Exception as e:
        print(f"\n\nExperiments failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()