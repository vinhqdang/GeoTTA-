"""
Statistical analysis tools for WACV 2026 paper.
Includes significance tests, confidence intervals, and effect size calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import scipy.stats as stats
from scipy.stats import ttest_ind, wilcoxon, mannwhitneyu, friedmanchisquare
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for TTA method comparisons.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha  # Significance level
        
    def compare_methods(self, results_dict: Dict[str, List[float]], 
                       method_names: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple methods with statistical significance testing.
        
        Args:
            results_dict: Dictionary where keys are method names and values are lists of results
            method_names: Optional list to specify order of methods
            
        Returns:
            Dictionary with statistical test results
        """
        if method_names is None:
            method_names = list(results_dict.keys())
        
        results = {
            'method_names': method_names,
            'descriptive_stats': {},
            'pairwise_tests': {},
            'omnibus_test': None,
            'effect_sizes': {},
            'rankings': {}
        }
        
        # Descriptive statistics
        for method in method_names:
            if method in results_dict:
                values = np.array(results_dict[method])
                results['descriptive_stats'][method] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'n': len(values),
                    'sem': float(np.std(values) / np.sqrt(len(values))),
                    'ci95_lower': float(np.mean(values) - 1.96 * np.std(values) / np.sqrt(len(values))),
                    'ci95_upper': float(np.mean(values) + 1.96 * np.std(values) / np.sqrt(len(values)))
                }
        
        # Omnibus test (Friedman test for non-parametric)
        if len(method_names) > 2:
            values_lists = [results_dict[method] for method in method_names if method in results_dict]
            if len(values_lists) > 2:
                try:
                    stat, p_value = friedmanchisquare(*values_lists)
                    results['omnibus_test'] = {
                        'test': 'Friedman',
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'significant': p_value < self.alpha
                    }
                except:
                    results['omnibus_test'] = {'error': 'Could not compute omnibus test'}
        
        # Pairwise comparisons
        for i, method1 in enumerate(method_names):
            if method1 not in results_dict:
                continue
                
            for j, method2 in enumerate(method_names[i+1:], i+1):
                if method2 not in results_dict:
                    continue
                
                values1 = np.array(results_dict[method1])
                values2 = np.array(results_dict[method2])
                
                # Multiple statistical tests
                pairwise_result = self._pairwise_comparison(values1, values2, method1, method2)
                results['pairwise_tests'][f'{method1}_vs_{method2}'] = pairwise_result
        
        # Effect sizes
        results['effect_sizes'] = self._compute_effect_sizes(results_dict, method_names)
        
        # Rankings
        results['rankings'] = self._compute_rankings(results_dict, method_names)
        
        return results
    
    def _pairwise_comparison(self, values1: np.ndarray, values2: np.ndarray,
                           method1: str, method2: str) -> Dict[str, Any]:
        """Perform comprehensive pairwise comparison."""
        result = {
            'method1': method1,
            'method2': method2,
            'n1': len(values1),
            'n2': len(values2)
        }
        
        # T-test (parametric)
        try:
            t_stat, t_p = ttest_ind(values1, values2)
            result['t_test'] = {
                'statistic': float(t_stat),
                'p_value': float(t_p),
                'significant': t_p < self.alpha
            }
        except:
            result['t_test'] = {'error': 'Could not compute t-test'}
        
        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, u_p = mannwhitneyu(values1, values2, alternative='two-sided')
            result['mann_whitney'] = {
                'statistic': float(u_stat),
                'p_value': float(u_p),
                'significant': u_p < self.alpha
            }
        except:
            result['mann_whitney'] = {'error': 'Could not compute Mann-Whitney test'}
        
        # Wilcoxon signed-rank test (if paired)
        if len(values1) == len(values2):
            try:
                w_stat, w_p = wilcoxon(values1, values2)
                result['wilcoxon'] = {
                    'statistic': float(w_stat),
                    'p_value': float(w_p),
                    'significant': w_p < self.alpha
                }
            except:
                result['wilcoxon'] = {'error': 'Could not compute Wilcoxon test'}
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1) + 
                             (len(values2) - 1) * np.var(values2)) / 
                            (len(values1) + len(values2) - 2))
        if pooled_std > 0:
            cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std
            result['cohens_d'] = {
                'value': float(cohens_d),
                'interpretation': self._interpret_cohens_d(cohens_d)
            }
        
        return result
    
    def _compute_effect_sizes(self, results_dict: Dict[str, List[float]], 
                            method_names: List[str]) -> Dict[str, Any]:
        """Compute effect sizes between methods."""
        effect_sizes = {}
        
        # Find the baseline method (typically first one or source_only)
        baseline_method = None
        if 'source_only' in method_names:
            baseline_method = 'source_only'
        elif 'geotta' in method_names:
            baseline_method = method_names[0]  # Use first as baseline
        else:
            baseline_method = method_names[0]
        
        if baseline_method in results_dict:
            baseline_values = np.array(results_dict[baseline_method])
            
            for method in method_names:
                if method == baseline_method or method not in results_dict:
                    continue
                
                method_values = np.array(results_dict[method])
                
                # Cohen's d
                pooled_std = np.sqrt((np.var(baseline_values) + np.var(method_values)) / 2)
                if pooled_std > 0:
                    cohens_d = (np.mean(method_values) - np.mean(baseline_values)) / pooled_std
                    effect_sizes[f'{method}_vs_{baseline_method}'] = {
                        'cohens_d': float(cohens_d),
                        'interpretation': self._interpret_cohens_d(cohens_d)
                    }
        
        return effect_sizes
    
    def _compute_rankings(self, results_dict: Dict[str, List[float]], 
                        method_names: List[str]) -> Dict[str, Any]:
        """Compute method rankings."""
        rankings = {}
        
        # Mean ranking
        mean_values = []
        valid_methods = []
        
        for method in method_names:
            if method in results_dict:
                mean_val = np.mean(results_dict[method])
                mean_values.append(mean_val)
                valid_methods.append(method)
        
        # Sort by performance (assuming higher is better)
        sorted_indices = np.argsort(mean_values)[::-1]
        
        rankings['by_mean'] = [valid_methods[i] for i in sorted_indices]
        rankings['mean_values'] = [mean_values[i] for i in sorted_indices]
        
        # Median ranking
        median_values = []
        for method in valid_methods:
            median_val = np.median(results_dict[method])
            median_values.append(median_val)
        
        sorted_indices = np.argsort(median_values)[::-1]
        rankings['by_median'] = [valid_methods[i] for i in sorted_indices]
        rankings['median_values'] = [median_values[i] for i in sorted_indices]
        
        return rankings
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def multiple_comparison_correction(self, p_values: List[float], 
                                     method: str = 'bonferroni') -> List[float]:
        """Apply multiple comparison correction."""
        p_values = np.array(p_values)
        
        if method == 'bonferroni':
            corrected = p_values * len(p_values)
            corrected = np.minimum(corrected, 1.0)
        elif method == 'holm':
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            corrected_sorted = np.zeros_like(sorted_p)
            
            for i, p in enumerate(sorted_p):
                corrected_sorted[i] = p * (len(p_values) - i)
            
            # Enforce monotonicity
            for i in range(1, len(corrected_sorted)):
                corrected_sorted[i] = max(corrected_sorted[i], corrected_sorted[i-1])
            
            corrected = np.zeros_like(p_values)
            corrected[sorted_indices] = corrected_sorted
        elif method == 'fdr':
            # Benjamini-Hochberg procedure
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            corrected_sorted = np.zeros_like(sorted_p)
            
            for i in range(len(sorted_p) - 1, -1, -1):
                corrected_sorted[i] = sorted_p[i] * len(p_values) / (i + 1)
            
            # Enforce monotonicity
            for i in range(len(corrected_sorted) - 2, -1, -1):
                corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i+1])
            
            corrected = np.zeros_like(p_values)
            corrected[sorted_indices] = corrected_sorted
        else:
            corrected = p_values
        
        return np.minimum(corrected, 1.0).tolist()
    
    def create_statistical_report(self, comparison_results: Dict[str, Any], 
                                save_path: Optional[str] = None) -> str:
        """Create a comprehensive statistical report."""
        report_lines = []
        
        # Header
        report_lines.append("STATISTICAL ANALYSIS REPORT")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Descriptive statistics
        report_lines.append("DESCRIPTIVE STATISTICS")
        report_lines.append("-" * 30)
        
        for method, stats in comparison_results['descriptive_stats'].items():
            report_lines.append(f"\n{method}:")
            report_lines.append(f"  Mean ± SEM: {stats['mean']:.4f} ± {stats['sem']:.4f}")
            report_lines.append(f"  95% CI: [{stats['ci95_lower']:.4f}, {stats['ci95_upper']:.4f}]")
            report_lines.append(f"  Median: {stats['median']:.4f}")
            report_lines.append(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            report_lines.append(f"  N: {stats['n']}")
        
        # Omnibus test
        if comparison_results.get('omnibus_test'):
            report_lines.append("\n\nOMNIBUS TEST")
            report_lines.append("-" * 20)
            omnibus = comparison_results['omnibus_test']
            if 'error' not in omnibus:
                report_lines.append(f"Test: {omnibus['test']}")
                report_lines.append(f"Statistic: {omnibus['statistic']:.4f}")
                report_lines.append(f"P-value: {omnibus['p_value']:.6f}")
                report_lines.append(f"Significant: {omnibus['significant']}")
        
        # Pairwise comparisons
        report_lines.append("\n\nPAIRWISE COMPARISONS")
        report_lines.append("-" * 30)
        
        for comparison, results in comparison_results['pairwise_tests'].items():
            report_lines.append(f"\n{comparison}:")
            
            if 't_test' in results and 'error' not in results['t_test']:
                t_test = results['t_test']
                report_lines.append(f"  T-test: t = {t_test['statistic']:.4f}, p = {t_test['p_value']:.6f}, sig = {t_test['significant']}")
            
            if 'mann_whitney' in results and 'error' not in results['mann_whitney']:
                mw_test = results['mann_whitney']
                report_lines.append(f"  Mann-Whitney: U = {mw_test['statistic']:.4f}, p = {mw_test['p_value']:.6f}, sig = {mw_test['significant']}")
            
            if 'cohens_d' in results:
                cohens = results['cohens_d']
                report_lines.append(f"  Effect size: d = {cohens['value']:.4f} ({cohens['interpretation']})")
        
        # Rankings
        if 'rankings' in comparison_results:
            report_lines.append("\n\nMETHOD RANKINGS")
            report_lines.append("-" * 20)
            
            rankings = comparison_results['rankings']
            if 'by_mean' in rankings:
                report_lines.append("\nBy Mean Performance:")
                for i, (method, value) in enumerate(zip(rankings['by_mean'], rankings['mean_values']), 1):
                    report_lines.append(f"  {i}. {method}: {value:.4f}")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def create_significance_matrix(self, comparison_results: Dict[str, Any], 
                                 test_type: str = 't_test') -> pd.DataFrame:
        """Create a significance matrix showing p-values between all method pairs."""
        method_names = comparison_results['method_names']
        n_methods = len(method_names)
        
        # Initialize matrix
        matrix = np.ones((n_methods, n_methods))
        
        # Fill in p-values
        for comparison, results in comparison_results['pairwise_tests'].items():
            method1, method2 = comparison.split('_vs_')
            
            if method1 in method_names and method2 in method_names:
                i = method_names.index(method1)
                j = method_names.index(method2)
                
                if test_type in results and 'p_value' in results[test_type]:
                    p_value = results[test_type]['p_value']
                    matrix[i, j] = p_value
                    matrix[j, i] = p_value  # Symmetric
        
        return pd.DataFrame(matrix, index=method_names, columns=method_names)
    
    def plot_performance_comparison(self, results_dict: Dict[str, List[float]], 
                                  title: str = "Method Comparison",
                                  save_path: Optional[str] = None):
        """Create publication-quality comparison plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot
        methods = list(results_dict.keys())
        values = [results_dict[method] for method in methods]
        
        ax1.boxplot(values, labels=methods)
        ax1.set_title(f'{title} - Distribution Comparison')
        ax1.set_ylabel('Performance')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Bar plot with error bars
        means = [np.mean(vals) for vals in values]
        stds = [np.std(vals) for vals in values]
        
        ax2.bar(methods, means, yerr=stds, capsize=5, alpha=0.7)
        ax2.set_title(f'{title} - Mean ± Std')
        ax2.set_ylabel('Performance')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compute_confidence_intervals(self, values: np.ndarray, 
                                   confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence intervals for a set of values."""
        alpha = 1 - confidence
        mean = np.mean(values)
        sem = stats.sem(values)
        
        if len(values) > 30:
            # Use normal distribution for large samples
            z_score = stats.norm.ppf(1 - alpha/2)
            margin = z_score * sem
        else:
            # Use t-distribution for small samples
            df = len(values) - 1
            t_score = stats.t.ppf(1 - alpha/2, df)
            margin = t_score * sem
        
        return (mean - margin, mean + margin)
    
    def power_analysis(self, effect_size: float, alpha: float = 0.05, 
                      power: float = 0.8) -> int:
        """Estimate required sample size for detecting an effect."""
        # Simplified power analysis for two-sample t-test
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))


def analyze_wacv_results(results_file: str, output_dir: str = './statistical_analysis'):
    """
    Comprehensive statistical analysis of WACV experiment results.
    
    Args:
        results_file: Path to JSON results file
        output_dir: Directory to save analysis outputs
    """
    import json
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer()
    
    # Extract performance metrics by method
    methods_data = {}
    for experiment_key, experiment_data in results.items():
        method = experiment_key.split('_')[0]
        
        if method not in methods_data:
            methods_data[method] = []
        
        # Extract accuracy (main metric)
        accuracy_mean = experiment_data['metrics'].get('accuracy_mean', 0)
        methods_data[method].append(accuracy_mean)
    
    # Statistical comparison
    comparison_results = analyzer.compare_methods(methods_data)
    
    # Save statistical report
    report = analyzer.create_statistical_report(
        comparison_results, 
        save_path=f"{output_dir}/statistical_report.txt"
    )
    
    # Create significance matrix
    sig_matrix = analyzer.create_significance_matrix(comparison_results)
    sig_matrix.to_csv(f"{output_dir}/significance_matrix.csv")
    
    # Create comparison plots
    analyzer.plot_performance_comparison(
        methods_data, 
        title="WACV Method Comparison",
        save_path=f"{output_dir}/performance_comparison.png"
    )
    
    print(f"Statistical analysis completed. Results saved to {output_dir}")
    
    return comparison_results