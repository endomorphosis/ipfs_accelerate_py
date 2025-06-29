"""
Benchmark Runner Module

This module provides functionality for running benchmarks and benchmark suites.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple

from .registry import BenchmarkRegistry
from .hardware import HardwareManager
from .results import ResultsCollector
from .base import BenchmarkBase

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """
    Unified entry point for executing benchmarks.
    
    This class provides a unified interface for executing benchmarks and
    benchmark suites, handling result collection and reporting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize benchmark runner.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        self.results_collector = ResultsCollector()
        self.hardware_manager = HardwareManager()
        self.logger = logging.getLogger("BenchmarkRunner")
        
        # Initialize output directory
        self.output_dir = self.config.get('output_dir', './benchmark_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def execute(self, benchmark_name: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a single benchmark with given parameters.
        
        Args:
            benchmark_name: Name of benchmark to execute
            params: Optional parameters for benchmark
            
        Returns:
            Benchmark results
            
        Raises:
            ValueError: If benchmark not found or hardware not available
        """
        params = params or {}
        self.logger.info(f"Executing benchmark: {benchmark_name}")
        
        # Get benchmark class from registry
        benchmark_class = BenchmarkRegistry.get_benchmark(benchmark_name)
        if not benchmark_class:
            raise ValueError(f"Benchmark {benchmark_name} not found in registry")
            
        # Get hardware backend
        hardware_name = params.get('hardware', 'cpu')
        hardware = self.hardware_manager.get_hardware(hardware_name, params.get('hardware_config'))
        
        # Create and run benchmark
        benchmark = benchmark_class(
            hardware=hardware,
            config=params
        )
        
        start_time = time.time()
        try:
            result = benchmark.run()
            
            # Add metadata to result
            result.update({
                'name': benchmark_name,
                'hardware_info': hardware.get_info(),
                'params': params,
                'timestamp': time.time(),
                'execution_time': time.time() - start_time
            })
            
            # Collect result
            self.results_collector.add_result(benchmark_name, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing benchmark {benchmark_name}: {e}", exc_info=True)
            
            # Create error result
            error_result = {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'name': benchmark_name,
                'hardware_info': hardware.get_info(),
                'params': params,
                'timestamp': time.time(),
                'execution_time': time.time() - start_time
            }
            
            # Collect error result
            self.results_collector.add_result(benchmark_name, error_result)
            
            return error_result
            
    def execute_suite(self, suite_config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a benchmark suite.
        
        Args:
            suite_config: Suite configuration file path or dictionary
            
        Returns:
            Dictionary with suite results
        """
        # Load suite configuration if path provided
        if isinstance(suite_config, str):
            import json
            with open(suite_config, 'r') as f:
                suite = json.load(f)
        else:
            suite = suite_config
            
        suite_name = suite.get('name', 'Unnamed Suite')
        benchmarks = suite.get('benchmarks', [])
        
        self.logger.info(f"Executing benchmark suite: {suite_name} ({len(benchmarks)} benchmarks)")
        
        # Add suite metadata
        self.results_collector.add_metadata('suite_name', suite_name)
        self.results_collector.add_metadata('suite_description', suite.get('description', ''))
        
        # Execute each benchmark in suite
        results = {}
        for benchmark_config in benchmarks:
            benchmark_name = benchmark_config.get('name')
            if not benchmark_name:
                self.logger.warning("Skipping benchmark with no name")
                continue
                
            params = benchmark_config.get('params', {})
            
            try:
                result = self.execute(benchmark_name, params)
                results[benchmark_name] = result
                
            except Exception as e:
                self.logger.error(f"Error executing benchmark {benchmark_name}: {e}", exc_info=True)
                results[benchmark_name] = {
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                
        # Save suite results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suite_results_path = os.path.join(self.output_dir, f"{suite_name.replace(' ', '_')}_{timestamp}.json")
        self.results_collector.save_results(suite_results_path)
        
        # Generate markdown report
        report_path = os.path.join(self.output_dir, f"{suite_name.replace(' ', '_')}_{timestamp}.md")
        self.results_collector.generate_markdown_report(report_path)
        
        return {
            'suite_name': suite_name,
            'benchmarks_executed': len(results),
            'success_count': sum(1 for r in results.values() if r.get('success', False)),
            'failure_count': sum(1 for r in results.values() if not r.get('success', False)),
            'results_path': suite_results_path,
            'report_path': report_path
        }
        
    def compare_with_baseline(self, current_results_path: str, baseline_results_path: str, 
                           threshold: float = 0.05) -> Dict[str, Any]:
        """
        Compare current results with baseline results.
        
        Args:
            current_results_path: Path to current results
            baseline_results_path: Path to baseline results
            threshold: Regression threshold
            
        Returns:
            Comparison results
        """
        # Load results using collector's storage backend
        current_collector = ResultsCollector()
        comparison = current_collector.compare_with_baseline(baseline_results_path, threshold)
        
        # Save comparison report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        comparison_path = os.path.join(self.output_dir, f"comparison_{timestamp}.json")
        
        with open(comparison_path, 'w') as f:
            import json
            json.dump(comparison, f, indent=2)
            
        # Generate markdown comparison report
        report_path = os.path.join(self.output_dir, f"comparison_{timestamp}.md")
        
        with open(report_path, 'w') as f:
            f.write("# Benchmark Comparison Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- Baseline: {comparison['baseline_timestamp']}\n")
            f.write(f"- Current: {comparison['current_timestamp']}\n")
            f.write(f"- Regressions: {len(comparison['regressions'])}\n")
            f.write(f"- Improvements: {len(comparison['improvements'])}\n")
            f.write(f"- Unchanged: {len(comparison['unchanged'])}\n")
            f.write(f"- New benchmarks: {len(comparison['new_benchmarks'])}\n")
            f.write(f"- Removed benchmarks: {len(comparison['removed_benchmarks'])}\n\n")
            
            if comparison['has_significant_regression']:
                f.write("⚠️ **Significant regressions detected!**\n\n")
                
            if comparison['regressions']:
                f.write("## Regressions\n\n")
                for name, data in comparison['regressions'].items():
                    f.write(f"### {name}\n\n")
                    f.write("| Metric | Baseline | Current | Change |\n")
                    f.write("|--------|----------|---------|--------|\n")
                    
                    for metric, values in data['metrics'].items():
                        baseline = values['baseline']
                        current = values['current']
                        change = values['change'] * 100  # Convert to percentage
                        
                        f.write(f"| {metric} | {baseline:.4f} | {current:.4f} | {change:+.2f}% |\n")
                        
                    f.write("\n")
                    
            if comparison['improvements']:
                f.write("## Improvements\n\n")
                for name, data in comparison['improvements'].items():
                    f.write(f"### {name}\n\n")
                    f.write("| Metric | Baseline | Current | Change |\n")
                    f.write("|--------|----------|---------|--------|\n")
                    
                    for metric, values in data['metrics'].items():
                        baseline = values['baseline']
                        current = values['current']
                        change = values['change'] * 100  # Convert to percentage
                        
                        f.write(f"| {metric} | {baseline:.4f} | {current:.4f} | {change:+.2f}% |\n")
                        
                    f.write("\n")
                    
            if comparison['new_benchmarks']:
                f.write("## New Benchmarks\n\n")
                for name in comparison['new_benchmarks']:
                    f.write(f"- {name}\n")
                    
                f.write("\n")
                
            if comparison['removed_benchmarks']:
                f.write("## Removed Benchmarks\n\n")
                for name in comparison['removed_benchmarks']:
                    f.write(f"- {name}\n")
                    
                f.write("\n")
                
        logger.info(f"Comparison report saved to {report_path}")
        
        return {
            'comparison': comparison,
            'comparison_path': comparison_path,
            'report_path': report_path,
            'has_regression': comparison['has_significant_regression']
        }
        
    def save_results(self) -> str:
        """
        Save current results.
        
        Returns:
            Path to saved results
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.output_dir, f"benchmark_results_{timestamp}.json")
        
        return self.results_collector.save_results(results_path)
        
    def generate_report(self) -> str:
        """
        Generate markdown report of current results.
        
        Returns:
            Path to generated report
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"benchmark_report_{timestamp}.md")
        
        return self.results_collector.generate_markdown_report(report_path)
        
    def cleanup(self) -> None:
        """Clean up resources."""
        self.hardware_manager.cleanup()


def ci_entrypoint():
    """Command-line entry point for CI/CD integration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Framework CI/CD Entry Point")
    
    parser.add_argument("--benchmark", type=str, help="Specific benchmark to run")
    parser.add_argument("--suite", type=str, help="Benchmark suite configuration file")
    parser.add_argument("--output", type=str, default="./benchmark_results", help="Output directory")
    parser.add_argument("--hardware", type=str, default="cpu", help="Hardware backend to use")
    parser.add_argument("--compare", type=str, help="Previous benchmark run to compare against")
    parser.add_argument("--threshold", type=float, default=0.05, help="Regression threshold (0.05 = 5%)")
    parser.add_argument("--params", type=str, help="JSON string with benchmark parameters")
    
    args = parser.parse_args()
    
    # Create runner with configuration
    runner = BenchmarkRunner(config={
        "output_dir": args.output
    })
    
    try:
        # Parse parameters if provided
        params = {}
        if args.params:
            import json
            params = json.loads(args.params)
            
        # Add hardware to parameters
        params['hardware'] = args.hardware
        
        # Run benchmark or suite
        if args.benchmark:
            logger.info(f"Running benchmark: {args.benchmark}")
            result = runner.execute(args.benchmark, params)
            results_path = runner.save_results()
            report_path = runner.generate_report()
            
            logger.info(f"Benchmark complete. Results saved to {results_path}")
            logger.info(f"Report generated at {report_path}")
            
        elif args.suite:
            logger.info(f"Running benchmark suite: {args.suite}")
            suite_result = runner.execute_suite(args.suite)
            
            logger.info(f"Suite complete. {suite_result['benchmarks_executed']} benchmarks executed.")
            logger.info(f"Successful: {suite_result['success_count']}, Failed: {suite_result['failure_count']}")
            logger.info(f"Results saved to {suite_result['results_path']}")
            logger.info(f"Report generated at {suite_result['report_path']}")
            
        else:
            logger.error("Either --benchmark or --suite must be specified")
            return 1
            
        # Compare with baseline if requested
        if args.compare:
            logger.info(f"Comparing with baseline: {args.compare}")
            
            comparison_result = runner.compare_with_baseline(
                runner.results_collector.storage.get_latest_results_path(),
                args.compare,
                args.threshold
            )
            
            logger.info(f"Comparison report generated at {comparison_result['report_path']}")
            
            # Exit with failure if significant regression detected
            if comparison_result['has_regression']:
                logger.warning("Significant regressions detected!")
                return 1
                
        return 0
        
    except Exception as e:
        logger.error(f"Error in benchmark execution: {e}", exc_info=True)
        return 1
        
    finally:
        runner.cleanup()