"""
Results Collector Module

This module provides functionality for collecting, processing, and storing benchmark results.
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

class StorageBackend(ABC):
    """Abstract base class for result storage backends."""
    
    @abstractmethod
    def save(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Save results to storage.
        
        Args:
            results: Results to save
            output_path: Optional output path
            
        Returns:
            Path where results were saved
        """
        pass
        
    @abstractmethod
    def load(self, input_path: str) -> Dict[str, Any]:
        """
        Load results from storage.
        
        Args:
            input_path: Path to load results from
            
        Returns:
            Loaded results
        """
        pass


class JSONStorageBackend(StorageBackend):
    """JSON file storage backend for benchmark results."""
    
    def save(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Save results to JSON file.
        
        Args:
            results: Results to save
            output_path: Optional output path. If not provided, a timestamped file
                        in the current directory is used.
            
        Returns:
            Path where results were saved
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"benchmark_results_{timestamp}.json"
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {output_path}")
        return output_path
        
    def load(self, input_path: str) -> Dict[str, Any]:
        """
        Load results from JSON file.
        
        Args:
            input_path: Path to JSON file
            
        Returns:
            Loaded results
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        logger.info(f"Results loaded from {input_path}")
        return results


class ResultsProcessor:
    """Processes and analyzes benchmark results."""
    
    @staticmethod
    def calculate_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate statistics from benchmark results.
        
        Args:
            results: Benchmark results
            
        Returns:
            Dictionary with calculated statistics
        """
        stats = {
            'benchmarks_count': len(results.get('benchmarks', {})),
            'success_count': sum(1 for b in results.get('benchmarks', {}).values() 
                              if b.get('success', False)),
            'failure_count': sum(1 for b in results.get('benchmarks', {}).values() 
                              if not b.get('success', False)),
            'total_execution_time': sum(b.get('execution_time', 0) 
                                     for b in results.get('benchmarks', {}).values()),
        }
        
        return stats
        
    @staticmethod
    def compare_results(baseline: Dict[str, Any], current: Dict[str, Any], 
                      threshold: float = 0.05) -> Dict[str, Any]:
        """
        Compare benchmark results to detect regressions.
        
        Args:
            baseline: Baseline results
            current: Current results
            threshold: Regression threshold (fractional change)
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            'baseline_timestamp': baseline.get('timestamp', 'unknown'),
            'current_timestamp': current.get('timestamp', 'unknown'),
            'regressions': {},
            'improvements': {},
            'unchanged': {},
            'new_benchmarks': [],
            'removed_benchmarks': [],
            'has_significant_regression': False
        }
        
        # Get benchmark names from both results
        baseline_benchmarks = set(baseline.get('benchmarks', {}).keys())
        current_benchmarks = set(current.get('benchmarks', {}).keys())
        
        # Find new and removed benchmarks
        comparison['new_benchmarks'] = list(current_benchmarks - baseline_benchmarks)
        comparison['removed_benchmarks'] = list(baseline_benchmarks - current_benchmarks)
        
        # Compare common benchmarks
        for name in baseline_benchmarks & current_benchmarks:
            baseline_bench = baseline.get('benchmarks', {}).get(name, {})
            current_bench = current.get('benchmarks', {}).get(name, {})
            
            # Skip if either doesn't have metrics
            if 'metrics' not in baseline_bench or 'metrics' not in current_bench:
                continue
                
            baseline_metrics = baseline_bench['metrics']
            current_metrics = current_bench['metrics']
            
            # Compare each metric
            metric_comparisons = {}
            for metric_name in set(baseline_metrics.keys()) & set(current_metrics.keys()):
                baseline_value = baseline_metrics[metric_name]
                current_value = current_metrics[metric_name]
                
                # Skip non-numeric metrics
                if not isinstance(baseline_value, (int, float)) or not isinstance(current_value, (int, float)):
                    continue
                    
                # Calculate change
                if baseline_value != 0:
                    change = (current_value - baseline_value) / baseline_value
                else:
                    change = float('inf') if current_value > 0 else 0
                    
                metric_comparisons[metric_name] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'change': change,
                    'is_improvement': change < 0  # For metrics like latency, lower is better
                }
                
            # Determine if benchmark has regressed or improved
            significant_changes = {k: v for k, v in metric_comparisons.items() 
                                if abs(v['change']) >= threshold}
            
            if significant_changes:
                # Group by improvement/regression
                improvements = {k: v for k, v in significant_changes.items() if v['is_improvement']}
                regressions = {k: v for k, v in significant_changes.items() if not v['is_improvement']}
                
                if improvements and not regressions:
                    comparison['improvements'][name] = {'metrics': improvements}
                elif regressions:
                    comparison['regressions'][name] = {'metrics': regressions}
                    comparison['has_significant_regression'] = True
            else:
                comparison['unchanged'][name] = {'metrics': metric_comparisons}
                
        return comparison


class ResultsCollector:
    """
    Standardized result collection and storage.
    
    This class provides functionality for collecting, processing,
    and storing benchmark results.
    """
    
    def __init__(self, storage_backend: Optional[StorageBackend] = None):
        """
        Initialize results collector.
        
        Args:
            storage_backend: Storage backend to use. If not provided,
                            JSONStorageBackend is used.
        """
        self.storage = storage_backend or JSONStorageBackend()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'benchmarks': {},
            'summary': {},
            'metadata': {
                'version': '1.0',
                'platform': os.name,
                'python_version': os.getenv('PYTHON_VERSION', 'unknown')
            }
        }
        self.logger = logging.getLogger("ResultsCollector")
        
    def add_result(self, benchmark_name: str, result: Dict[str, Any]) -> None:
        """
        Add a benchmark result to the collection.
        
        Args:
            benchmark_name: Name of the benchmark
            result: Benchmark result
        """
        self.results['benchmarks'][benchmark_name] = result
        self.logger.debug(f"Added result for benchmark: {benchmark_name}")
        
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to the results.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.results['metadata'][key] = value
        
    def compute_summary(self) -> Dict[str, Any]:
        """
        Compute result summary.
        
        Returns:
            Summary statistics
        """
        summary = ResultsProcessor.calculate_statistics(self.results)
        self.results['summary'] = summary
        return summary
        
    def save_results(self, output_path: Optional[str] = None) -> str:
        """
        Save results using the configured storage backend.
        
        Args:
            output_path: Optional output path
            
        Returns:
            Path where results were saved
        """
        # Update timestamp and compute summary before saving
        self.results['timestamp'] = datetime.now().isoformat()
        self.compute_summary()
        
        return self.storage.save(self.results, output_path)
        
    def compare_with_baseline(self, baseline_path: str, threshold: float = 0.05) -> Dict[str, Any]:
        """
        Compare current results with baseline.
        
        Args:
            baseline_path: Path to baseline results
            threshold: Regression threshold
            
        Returns:
            Comparison results
        """
        baseline = self.storage.load(baseline_path)
        comparison = ResultsProcessor.compare_results(baseline, self.results, threshold)
        
        return comparison
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of collected results.
        
        Returns:
            Dictionary with result summary
        """
        # Ensure summary is up to date
        self.compute_summary()
        
        return {
            'timestamp': self.results['timestamp'],
            'benchmarks_count': len(self.results['benchmarks']),
            'summary': self.results['summary'],
            'metadata': self.results['metadata']
        }
        
    def generate_markdown_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate markdown report of benchmark results.
        
        Args:
            output_path: Optional output path for markdown report
            
        Returns:
            Path where report was saved
        """
        # Ensure summary is up to date
        self.compute_summary()
        
        # Generate report lines
        lines = [
            "# Benchmark Results Report",
            "",
            f"Generated: {self.results['timestamp']}",
            "",
            "## Summary",
            "",
            f"- **Total Benchmarks**: {self.results['summary']['benchmarks_count']}",
            f"- **Successful**: {self.results['summary']['success_count']}",
            f"- **Failed**: {self.results['summary']['failure_count']}",
            f"- **Total Execution Time**: {self.results['summary']['total_execution_time']:.2f} seconds",
            "",
            "## Benchmarks",
            ""
        ]
        
        # Add benchmark results
        for name, result in self.results['benchmarks'].items():
            success = result.get('success', False)
            status = "✅ Success" if success else "❌ Failure"
            
            lines.extend([
                f"### {name}",
                "",
                f"**Status**: {status}",
                f"**Execution Time**: {result.get('execution_time', 0):.2f} seconds",
                ""
            ])
            
            if not success:
                lines.extend([
                    f"**Error**: {result.get('error', 'Unknown error')}",
                    f"**Error Type**: {result.get('error_type', 'Unknown')}",
                    ""
                ])
                continue
                
            # Add metrics if available
            if 'metrics' in result:
                lines.append("**Metrics**:")
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")
                
                for metric, value in result['metrics'].items():
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                        
                    lines.append(f"| {metric} | {formatted_value} |")
                    
                lines.append("")
                
            # Add hardware info if available
            if 'hardware' in result:
                lines.append("**Hardware**:")
                lines.append("")
                
                for key, value in result['hardware'].items():
                    lines.append(f"- **{key}**: {value}")
                    
                lines.append("")
                
        # Save report if output path provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"benchmark_report_{timestamp}.md"
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
            
        self.logger.info(f"Markdown report saved to {output_path}")
        return output_path