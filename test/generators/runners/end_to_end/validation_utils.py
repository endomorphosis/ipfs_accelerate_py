#!/usr/bin/env python3
"""
Validation Utilities for End-to-End Testing Framework

This module provides utilities for validating simulations and comparing performance
across hardware platforms. It is a standalone version of the classes from enhanced_ci_cd_reports.py,
focusing on the core validation and comparison functionality.
"""

import os
import sys
import json
import time
import re
import logging
import datetime
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SIMULATION_VALIDATION_DIR = "simulation_validation"

# Hardware validation reference data
# These are the expected performance ratios between different hardware platforms
# based on real hardware testing. Used for simulation validation.
HARDWARE_PERFORMANCE_RATIOS = {
    # Format: (hw1, hw2): expected_ratio
    # A ratio of 2.0 means hw1 is expected to be 2x faster than hw2
    ('cuda', 'cpu'): 3.5,
    ('rocm', 'cpu'): 2.8,
    ('mps', 'cpu'): 2.2,
    ('openvino', 'cpu'): 1.5,
    ('qnn', 'cpu'): 2.5,
    ('webgpu', 'cpu'): 2.0,
    ('webnn', 'cpu'): 1.8,
    ('cuda', 'rocm'): 1.25,
    ('cuda', 'mps'): 1.6,
    ('cuda', 'webgpu'): 1.75,
    ('rocm', 'mps'): 1.3,
    ('webgpu', 'webnn'): 1.1,
}

# Define acceptable variance for simulation validation (as percentage)
SIMULATION_TOLERANCE = 0.25  # 25% tolerance for simulation vs real hardware

# Utility function to create directories if they don't exist
def ensure_dir_exists(directory_path):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    return directory_path


class SimulationValidator:
    """
    Validates hardware simulations against expected performance ratios.
    
    This class is responsible for determining if test results from simulated hardware
    are realistic compared to expected performance characteristics.
    """
    
    def __init__(self, 
                reference_ratios: Dict[Tuple[str, str], float] = None, 
                tolerance: float = SIMULATION_TOLERANCE):
        """
        Initialize the simulation validator.
        
        Args:
            reference_ratios: Dictionary mapping hardware platform pairs to expected performance ratios
            tolerance: Acceptable tolerance for simulation deviation as a percentage (0.25 = 25%)
        """
        self.reference_ratios = reference_ratios or HARDWARE_PERFORMANCE_RATIOS
        self.tolerance = tolerance
        
    def is_simulation(self, result: Dict[str, Any]) -> bool:
        """
        Determine if a test result is from a simulated environment.
        
        Args:
            result: Test result dictionary
            
        Returns:
            True if the result appears to be from a simulation, False otherwise
        """
        # Check for explicit simulation flags
        if result.get("simulation", False):
            return True
            
        # Check for simulation markers in metadata
        metadata = result.get("metadata", {})
        if metadata.get("simulation", False) or metadata.get("simulated", False):
            return True
            
        # Check for simulation indicators in the environment
        env = metadata.get("environment", {})
        if env.get("simulation", False) or "simulator" in env.get("platform", "").lower():
            return True
            
        # Check for missing hardware-specific data that would be present in real hardware
        hardware = result.get("hardware", "")
        if hardware in ["cuda", "rocm", "mps"]:
            # Check for missing GPU device info
            if "gpu_info" not in metadata and "device_info" not in metadata:
                return True
                
        elif hardware in ["webgpu", "webnn"]:
            # Check for missing browser info
            if "browser_info" not in metadata and "browser" not in metadata:
                return True
                
        return False
        
    def validate_performance(self, 
                          result1: Dict[str, Any], 
                          result2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that the performance ratio between two hardware platforms is realistic.
        
        Args:
            result1: Test result for first hardware platform
            result2: Test result for second hardware platform
            
        Returns:
            Dictionary with validation results
        """
        hw1 = result1.get("hardware", "")
        hw2 = result2.get("hardware", "")
        
        # Skip if we don't have a reference ratio for this pair
        if (hw1, hw2) not in self.reference_ratios and (hw2, hw1) not in self.reference_ratios:
            return {
                "valid": None,
                "reason": f"No reference ratio available for {hw1} vs {hw2}"
            }
            
        # Get reference ratio
        if (hw1, hw2) in self.reference_ratios:
            expected_ratio = self.reference_ratios[(hw1, hw2)]
            reverse = False
        else:
            expected_ratio = self.reference_ratios[(hw2, hw1)]
            reverse = True
            
        # Extract performance metrics
        perf1 = self._extract_performance_metrics(result1)
        perf2 = self._extract_performance_metrics(result2)
        
        # Skip if we don't have throughput or latency info
        if not perf1 or not perf2:
            return {
                "valid": None,
                "reason": "Missing performance metrics"
            }
            
        # Calculate actual ratios
        actual_ratios = {}
        
        # Throughput: higher is better
        if "throughput" in perf1 and "throughput" in perf2 and perf2["throughput"] > 0:
            throughput_ratio = perf1["throughput"] / perf2["throughput"]
            if reverse:
                throughput_ratio = 1 / throughput_ratio
            actual_ratios["throughput"] = throughput_ratio
            
        # Latency: lower is better
        if "latency" in perf1 and "latency" in perf2 and perf1["latency"] > 0:
            latency_ratio = perf2["latency"] / perf1["latency"]
            if reverse:
                latency_ratio = 1 / latency_ratio
            actual_ratios["latency"] = latency_ratio
            
        if not actual_ratios:
            return {
                "valid": None,
                "reason": "Could not calculate performance ratios"
            }
            
        # Validate ratios
        validations = {}
        for metric, actual_ratio in actual_ratios.items():
            lower_bound = expected_ratio * (1 - self.tolerance)
            upper_bound = expected_ratio * (1 + self.tolerance)
            
            valid = lower_bound <= actual_ratio <= upper_bound
            
            validations[metric] = {
                "valid": valid,
                "expected_ratio": expected_ratio,
                "actual_ratio": actual_ratio,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "deviation": abs(actual_ratio - expected_ratio) / expected_ratio,
                "within_tolerance": valid
            }
            
        # Overall validation
        valid_metrics = [v["valid"] for v in validations.values()]
        overall_valid = all(valid_metrics) if valid_metrics else None
        
        return {
            "valid": overall_valid,
            "hardware_pair": (hw1, hw2),
            "expected_ratio": expected_ratio,
            "metrics": validations,
            "is_simulation1": self.is_simulation(result1),
            "is_simulation2": self.is_simulation(result2)
        }
        
    def _extract_performance_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract performance metrics from a test result.
        
        Args:
            result: Test result dictionary
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        # Direct metrics from result
        if "metrics" in result:
            metrics.update(result["metrics"])
            return metrics
        
        # Try to extract from benchmark results
        if "benchmark_results" in result:
            benchmark = result["benchmark_results"]
            
            # Check for results by batch
            if "results_by_batch" in benchmark:
                batch_results = benchmark["results_by_batch"]
                if "1" in batch_results:  # Use batch size 1 if available
                    if "average_throughput_items_per_second" in batch_results["1"]:
                        metrics["throughput"] = float(batch_results["1"]["average_throughput_items_per_second"])
                    if "average_latency_ms" in batch_results["1"]:
                        metrics["latency"] = float(batch_results["1"]["average_latency_ms"])
                        
        # Check output
        if "output" in result and isinstance(result["output"], dict) and "metrics" in result["output"]:
            metrics.update(result["output"]["metrics"])
                
        return metrics
        
    def validate_results(self, results: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Validate test results across all models and hardware platforms.
        
        Args:
            results: Dictionary mapping models to hardware to test results
            
        Returns:
            Dictionary with validation results
        """
        validations = {}
        simulations = {}
        
        # Find all simulated results
        for model, hw_results in results.items():
            if model not in simulations:
                simulations[model] = {}
                
            for hw, result in hw_results.items():
                if self.is_simulation(result):
                    simulations[model][hw] = result
        
        # Validate each simulated result against non-simulated results
        for model, hw_results in simulations.items():
            if model not in validations:
                validations[model] = {}
                
            for sim_hw, sim_result in hw_results.items():
                validations[model][sim_hw] = {}
                
                # Find all real hardware results for this model
                real_results = {
                    hw: result for hw, result in results.get(model, {}).items()
                    if not self.is_simulation(result)
                }
                
                # Validate against each real hardware result
                for real_hw, real_result in real_results.items():
                    validation = self.validate_performance(sim_result, real_result)
                    validations[model][sim_hw][real_hw] = validation
        
        return {
            "validations": validations,
            "simulations": simulations
        }
        
    def generate_validation_report(self, 
                                  results: Dict[str, Dict[str, Dict[str, Any]]], 
                                  output_dir: str) -> Dict[str, str]:
        """
        Generate a simulation validation report.
        
        Args:
            results: Dictionary mapping models to hardware to test results
            output_dir: Directory to save the report
            
        Returns:
            Dictionary mapping report names to file paths
        """
        ensure_dir_exists(output_dir)
        validation_result = self.validate_results(results)
        
        # Generate HTML report
        html_report = os.path.join(output_dir, "simulation_validation_report.html")
        self._generate_html_report(validation_result, html_report)
        
        # Generate Markdown report
        md_report = os.path.join(output_dir, "simulation_validation_report.md")
        self._generate_markdown_report(validation_result, md_report)
        
        # Generate visualization
        viz_file = os.path.join(output_dir, "simulation_validation_visualization.png")
        self._generate_visualization(validation_result, viz_file)
        
        return {
            "html": html_report,
            "markdown": md_report,
            "visualization": viz_file
        }
        
    def _generate_html_report(self, 
                             validation_result: Dict[str, Any], 
                             output_path: str) -> None:
        """
        Generate an HTML report for simulation validation.
        
        Args:
            validation_result: Validation results
            output_path: Path to save the HTML report
        """
        validations = validation_result.get("validations", {})
        simulations = validation_result.get("simulations", {})
        
        # Count validations
        valid_count = 0
        invalid_count = 0
        unknown_count = 0
        
        for model, hw_validations in validations.items():
            for sim_hw, real_validations in hw_validations.items():
                for real_hw, validation in real_validations.items():
                    if validation.get("valid") is True:
                        valid_count += 1
                    elif validation.get("valid") is False:
                        invalid_count += 1
                    else:
                        unknown_count += 1
        
        total_validations = valid_count + invalid_count + unknown_count
        
        # Basic HTML structure
        with open(output_path, 'w') as f:
            f.write("<!DOCTYPE html><html><head><title>Simulation Validation Report</title>")
            f.write("<style>body{font-family:sans-serif;line-height:1.6;max-width:1200px;margin:0 auto;padding:1em}")
            f.write("h1,h2,h3{color:#333}table{border-collapse:collapse;width:100%}th,td{text-align:left;border:1px solid #ddd;padding:8px}")
            f.write("th{background:#f2f2f2}.success{color:green}.failure{color:red}.unknown{color:orange}</style></head><body>")
            
            # Report header
            f.write("<h1>Simulation Validation Report</h1>")
            f.write("<p>This report validates simulation accuracy against real hardware performance expectations.</p>")
            
            # Summary
            f.write("<h2>Summary</h2>")
            f.write(f"<p>Total Simulations: {len(simulations)}</p>")
            f.write(f"<p>Total Validations: {total_validations}</p>")
            f.write(f"<p>Valid Simulations: <span class='success'>{valid_count}</span></p>")
            f.write(f"<p>Invalid Simulations: <span class='failure'>{invalid_count}</span></p>")
            f.write(f"<p>Unknown Simulations: <span class='unknown'>{unknown_count}</span></p>")
            
            if total_validations > 0:
                validity_rate = (valid_count / total_validations) * 100
                f.write(f"<p>Validity Rate: <span class='{'success' if validity_rate >= 80 else 'failure'}'>{validity_rate:.1f}%</span></p>")
            
            # Visualization reference
            f.write("<h2>Validation Visualization</h2>")
            f.write("<img src='simulation_validation_visualization.png' alt='Validation Visualization' width='100%'>")
            
            # Detailed results
            f.write("<h2>Detailed Validation Results</h2>")
            
            for model, hw_validations in validations.items():
                f.write(f"<h3>Model: {model}</h3>")
                
                for sim_hw, real_validations in hw_validations.items():
                    f.write(f"<h4>Simulated Hardware: {sim_hw}</h4>")
                    
                    for real_hw, validation in real_validations.items():
                        valid = validation.get("valid")
                        status_class = "success" if valid is True else "failure" if valid is False else "unknown"
                        status_text = "VALID" if valid is True else "INVALID" if valid is False else "UNKNOWN"
                        
                        f.write(f"<h5>Compared to {real_hw}: <span class='{status_class}'>{status_text}</span></h5>")
                        
                        expected_ratio = validation.get("expected_ratio", "N/A")
                        f.write(f"<p>Expected Performance Ratio: {expected_ratio:.2f}x</p>")
                        
                        # Show metrics
                        metrics = validation.get("metrics", {})
                        if metrics:
                            f.write("<table><tr><th>Metric</th><th>Expected Ratio</th><th>Actual Ratio</th><th>Deviation</th><th>Status</th></tr>")
                            
                            for metric, metric_validation in metrics.items():
                                actual_ratio = metric_validation.get("actual_ratio", 0)
                                deviation = metric_validation.get("deviation", 0) * 100
                                within_tolerance = metric_validation.get("within_tolerance", False)
                                
                                status_class = "success" if within_tolerance else "failure"
                                status_text = "VALID" if within_tolerance else "INVALID"
                                
                                f.write(f"<tr><td>{metric.capitalize()}</td><td>{expected_ratio:.2f}x</td>")
                                f.write(f"<td>{actual_ratio:.2f}x</td><td>{deviation:.1f}%</td>")
                                f.write(f"<td class='{status_class}'>{status_text}</td></tr>")
                                
                            f.write("</table>")
            
            f.write("</body></html>")
        
    def _generate_markdown_report(self, 
                                validation_result: Dict[str, Any], 
                                output_path: str) -> None:
        """
        Generate a Markdown report for simulation validation.
        
        Args:
            validation_result: Validation results
            output_path: Path to save the Markdown report
        """
        validations = validation_result.get("validations", {})
        simulations = validation_result.get("simulations", {})
        
        # Count validations
        valid_count = 0
        invalid_count = 0
        unknown_count = 0
        
        for model, hw_validations in validations.items():
            for sim_hw, real_validations in hw_validations.items():
                for real_hw, validation in real_validations.items():
                    if validation.get("valid") is True:
                        valid_count += 1
                    elif validation.get("valid") is False:
                        invalid_count += 1
                    else:
                        unknown_count += 1
        
        total_validations = valid_count + invalid_count + unknown_count
        
        with open(output_path, 'w') as f:
            # Report header
            f.write("# Simulation Validation Report\n\n")
            f.write("This report validates simulation accuracy against real hardware performance expectations.\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Total Simulations:** {len(simulations)}\n")
            f.write(f"- **Total Validations:** {total_validations}\n")
            f.write(f"- **Valid Simulations:** {valid_count}\n")
            f.write(f"- **Invalid Simulations:** {invalid_count}\n")
            f.write(f"- **Unknown Simulations:** {unknown_count}\n")
            
            if total_validations > 0:
                validity_rate = (valid_count / total_validations) * 100
                f.write(f"- **Validity Rate:** {validity_rate:.1f}%\n")
            
            f.write("\n")
            
            # Visualization reference
            f.write("## Validation Visualization\n\n")
            f.write("![Simulation Validation Visualization](simulation_validation_visualization.png)\n\n")
            
            # Detailed results
            f.write("## Detailed Validation Results\n\n")
            
            for model, hw_validations in validations.items():
                f.write(f"### Model: {model}\n\n")
                
                for sim_hw, real_validations in hw_validations.items():
                    f.write(f"#### Simulated Hardware: {sim_hw}\n\n")
                    
                    for real_hw, validation in real_validations.items():
                        valid = validation.get("valid")
                        if valid is True:
                            status = "✅ VALID"
                        elif valid is False:
                            status = "❌ INVALID"
                        else:
                            status = "❓ UNKNOWN"
                            
                        f.write(f"##### Compared to {real_hw}: {status}\n\n")
                        
                        expected_ratio = validation.get("expected_ratio", "N/A")
                        f.write(f"- **Expected Performance Ratio:** {expected_ratio:.2f}x\n\n")
                        
                        # Show metrics
                        metrics = validation.get("metrics", {})
                        if metrics:
                            f.write("| Metric | Expected Ratio | Actual Ratio | Deviation | Status |\n")
                            f.write("|--------|---------------|--------------|-----------|--------|\n")
                            
                            for metric, metric_validation in metrics.items():
                                actual_ratio = metric_validation.get("actual_ratio", 0)
                                deviation = metric_validation.get("deviation", 0) * 100
                                within_tolerance = metric_validation.get("within_tolerance", False)
                                
                                status_text = "VALID" if within_tolerance else "INVALID"
                                
                                f.write(f"| {metric.capitalize()} | {expected_ratio:.2f}x | {actual_ratio:.2f}x | {deviation:.1f}% | {status_text} |\n")
                                
                            f.write("\n")
    
    def _generate_visualization(self, 
                               validation_result: Dict[str, Any], 
                               output_path: str) -> None:
        """
        Generate a visualization for simulation validation results.
        
        Args:
            validation_result: Validation results
            output_path: Path to save the visualization
        """
        validations = validation_result.get("validations", {})
        
        # Collect data for visualization
        models = []
        simulated_hw = []
        real_hw = []
        validity = []
        deviations = []
        
        for model, hw_validations in validations.items():
            for sim_hw, real_validations in hw_validations.items():
                for real_hw_name, validation in real_validations.items():
                    models.append(model)
                    simulated_hw.append(sim_hw)
                    real_hw.append(real_hw_name)
                    validity.append(validation.get("valid", None))
                    
                    # Calculate average deviation across metrics
                    metrics = validation.get("metrics", {})
                    if metrics:
                        avg_deviation = np.mean([m.get("deviation", 0) for m in metrics.values()])
                        deviations.append(avg_deviation * 100)  # Convert to percentage
                    else:
                        deviations.append(np.nan)
        
        # Create visualization based on available data
        if not models or not simulated_hw or not real_hw:
            # Create empty visualization if no data
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No simulation validation data available", 
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.gca().set_axis_off()
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            return
            
        # Get unique values for axes
        unique_models = sorted(list(set(models)))
        unique_sim_hw = sorted(list(set(simulated_hw)))
        
        # Create a figure for the visualization
        plt.figure(figsize=(max(10, len(unique_models) * 1.5), max(8, len(unique_sim_hw) * 0.6)))
        
        # Create a custom colormap for deviations (green to red)
        cmap = plt.cm.get_cmap('RdYlGn_r')
        
        # Create scatter plot
        scatter = plt.scatter(
            [unique_models.index(m) for m in models],
            [unique_sim_hw.index(s) for s in simulated_hw],
            c=deviations,
            cmap=cmap,
            s=100,
            vmin=0,
            vmax=50,  # Max deviation of 50%
            alpha=0.8
        )
        
        # Add validity markers
        for i, valid in enumerate(validity):
            if valid is True:
                plt.plot(unique_models.index(models[i]), unique_sim_hw.index(simulated_hw[i]), 'o', color='none', 
                       markeredgecolor='darkgreen', markeredgewidth=2, markersize=10)
            elif valid is False:
                plt.plot(unique_models.index(models[i]), unique_sim_hw.index(simulated_hw[i]), 'X', color='none', 
                       markeredgecolor='darkred', markeredgewidth=2, markersize=10)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Deviation (%)')
        
        # Set labels and title
        plt.xlabel('Model')
        plt.ylabel('Simulated Hardware')
        plt.title('Simulation Validation Results')
        
        # Set tick labels
        plt.xticks(range(len(unique_models)), unique_models, rotation=45, ha='right')
        plt.yticks(range(len(unique_sim_hw)), unique_sim_hw)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()


class CrossHardwareComparison:
    """
    Compares performance metrics across different hardware platforms.
    
    This class generates comprehensive cross-hardware performance comparison reports
    to help identify the optimal hardware for different model types.
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the cross-hardware comparison generator.
        
        Args:
            output_dir: Directory to save report files
        """
        self.output_dir = output_dir or "cross_hardware"
        ensure_dir_exists(self.output_dir)
    
    def generate_comparison(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive comparison of performance across hardware platforms.
        
        Args:
            results: Dictionary mapping models to hardware results
            
        Returns:
            Dictionary with comparison results
        """
        # Extract metrics for each model and hardware combination
        metrics = {}
        hardware_sets = set()
        model_families = {}
        
        for model, hw_results in results.items():
            metrics[model] = {}
            
            # Determine model family
            model_family = self._determine_model_family(model)
            model_families[model] = model_family
            
            for hw, result in hw_results.items():
                hardware_sets.add(hw)
                metrics[model][hw] = self._extract_performance_metrics(result)
        
        # Create comparison by hardware
        hardware_comparison = {}
        hardware_list = sorted(list(hardware_sets))
        
        for model, hw_metrics in metrics.items():
            if model not in hardware_comparison:
                hardware_comparison[model] = {}
            
            # Compare each hardware against others
            for hw1 in hardware_list:
                if hw1 not in hw_metrics:
                    continue
                    
                for hw2 in hardware_list:
                    if hw2 == hw1 or hw2 not in hw_metrics:
                        continue
                        
                    # Compare metrics
                    comparison = self._compare_metrics(hw_metrics[hw1], hw_metrics[hw2])
                    
                    if hw1 not in hardware_comparison[model]:
                        hardware_comparison[model][hw1] = {}
                        
                    hardware_comparison[model][hw1][hw2] = comparison
        
        # Create comparison by model family
        family_comparison = {}
        
        for family_name in set(model_families.values()):
            family_comparison[family_name] = {}
            family_models = [m for m, f in model_families.items() if f == family_name]
            
            # Compute average metrics by hardware for this family
            family_metrics = {}
            
            for hw in hardware_list:
                # Collect all metrics for this hardware across family models
                hw_family_metrics = []
                
                for model in family_models:
                    if model in metrics and hw in metrics[model]:
                        hw_family_metrics.append(metrics[model][hw])
                
                if hw_family_metrics:
                    # Compute average metrics
                    family_metrics[hw] = self._average_metrics(hw_family_metrics)
            
            # Compare each hardware against others for this family
            for hw1 in hardware_list:
                if hw1 not in family_metrics:
                    continue
                    
                family_comparison[family_name][hw1] = {}
                
                for hw2 in hardware_list:
                    if hw2 == hw1 or hw2 not in family_metrics:
                        continue
                        
                    # Compare metrics
                    comparison = self._compare_metrics(family_metrics[hw1], family_metrics[hw2])
                    family_comparison[family_name][hw1][hw2] = comparison
        
        # Find optimal hardware for each model and family
        optimal_hardware = {}
        
        for model, hw_metrics in metrics.items():
            if not hw_metrics:
                continue
                
            # Find hardware with best throughput
            best_hw_throughput = max(hw_metrics.items(), key=lambda x: x[1].get("throughput", 0) if "throughput" in x[1] else 0, default=(None, {}))
            
            # Find hardware with best latency
            best_hw_latency = min(hw_metrics.items(), key=lambda x: x[1].get("latency", float('inf')) if "latency" in x[1] else float('inf'), default=(None, {}))
            
            optimal_hardware[model] = {
                "best_throughput": best_hw_throughput[0] if best_hw_throughput[0] else None,
                "best_latency": best_hw_latency[0] if best_hw_latency[0] else None
            }
        
        # Find optimal hardware for each family
        family_optimal_hardware = {}
        
        for family, family_hw_metrics in family_comparison.items():
            family_optimal_hardware[family] = {
                "best_hardware": self._find_best_hardware_for_family(family_hw_metrics)
            }
        
        return {
            "model_metrics": metrics,
            "hardware_comparison": hardware_comparison,
            "family_comparison": family_comparison,
            "optimal_hardware": optimal_hardware,
            "family_optimal_hardware": family_optimal_hardware,
            "hardware_list": hardware_list,
            "model_families": model_families
        }
    
    def _determine_model_family(self, model_name: str) -> str:
        """
        Determine the model family based on model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model family name
        """
        # Check against known model patterns
        if any(keyword in model_name.lower() for keyword in ["bert", "roberta", "sentence", "embedding"]):
            return "text-embedding"
        elif any(keyword in model_name.lower() for keyword in ["gpt", "llama", "opt", "t5", "falcon"]):
            return "text-generation"
        elif any(keyword in model_name.lower() for keyword in ["vit", "resnet", "detr", "yolo"]):
            return "vision"
        elif any(keyword in model_name.lower() for keyword in ["whisper", "wav2vec", "clap", "audio"]):
            return "audio"
        elif any(keyword in model_name.lower() for keyword in ["clip", "llava", "flava", "blip"]):
            return "multimodal"
        
        # Default to unknown
        return "unknown"
    
    def _extract_performance_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract performance metrics from a test result.
        
        Args:
            result: Test result dictionary
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        # Direct metrics from result
        if "metrics" in result:
            metrics.update(result["metrics"])
            return metrics
        
        # Try to extract from benchmark results
        if "benchmark_results" in result:
            benchmark = result["benchmark_results"]
            
            # Check for results by batch
            if "results_by_batch" in benchmark:
                batch_results = benchmark["results_by_batch"]
                if "1" in batch_results:  # Use batch size 1 if available
                    if "average_throughput_items_per_second" in batch_results["1"]:
                        metrics["throughput"] = float(batch_results["1"]["average_throughput_items_per_second"])
                    if "average_latency_ms" in batch_results["1"]:
                        metrics["latency"] = float(batch_results["1"]["average_latency_ms"])
                
                # Also consider batch size 4 if available
                if "4" in batch_results:
                    if "average_throughput_items_per_second" in batch_results["4"]:
                        metrics["throughput_batch4"] = float(batch_results["4"]["average_throughput_items_per_second"])
                    if "average_latency_ms" in batch_results["4"]:
                        metrics["latency_batch4"] = float(batch_results["4"]["average_latency_ms"])
                        
        # Check output
        if "output" in result and isinstance(result["output"], dict) and "metrics" in result["output"]:
            metrics.update(result["output"]["metrics"])
                
        return metrics
    
    def _compare_metrics(self, metrics1: Dict[str, float], metrics2: Dict[str, float]) -> Dict[str, Any]:
        """
        Compare performance metrics between two hardware platforms.
        
        Args:
            metrics1: Performance metrics for first hardware
            metrics2: Performance metrics for second hardware
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {}
        
        # Compare throughput (higher is better)
        if "throughput" in metrics1 and "throughput" in metrics2 and metrics2["throughput"] > 0:
            throughput_ratio = metrics1["throughput"] / metrics2["throughput"]
            comparison["throughput_ratio"] = throughput_ratio
            comparison["throughput_percentage"] = (throughput_ratio - 1) * 100  # Percentage improvement
        
        # Compare latency (lower is better)
        if "latency" in metrics1 and "latency" in metrics2 and metrics1["latency"] > 0:
            latency_ratio = metrics2["latency"] / metrics1["latency"]
            comparison["latency_ratio"] = latency_ratio
            comparison["latency_percentage"] = (latency_ratio - 1) * 100  # Percentage improvement
        
        # Compare batch 4 metrics if available
        if "throughput_batch4" in metrics1 and "throughput_batch4" in metrics2 and metrics2["throughput_batch4"] > 0:
            throughput_b4_ratio = metrics1["throughput_batch4"] / metrics2["throughput_batch4"]
            comparison["throughput_batch4_ratio"] = throughput_b4_ratio
            comparison["throughput_batch4_percentage"] = (throughput_b4_ratio - 1) * 100
        
        # Compare memory usage if available (lower is better)
        if "memory" in metrics1 and "memory" in metrics2 and metrics1["memory"] > 0:
            memory_ratio = metrics2["memory"] / metrics1["memory"]
            comparison["memory_ratio"] = memory_ratio
            comparison["memory_percentage"] = (memory_ratio - 1) * 100
        
        # Compare power consumption if available (lower is better)
        if "power" in metrics1 and "power" in metrics2 and metrics1["power"] > 0:
            power_ratio = metrics2["power"] / metrics1["power"]
            comparison["power_ratio"] = power_ratio
            comparison["power_percentage"] = (power_ratio - 1) * 100
        
        # Overall performance score (weighted average of throughput and latency)
        if "throughput_ratio" in comparison and "latency_ratio" in comparison:
            comparison["overall_score"] = (comparison["throughput_ratio"] + comparison["latency_ratio"]) / 2
        
        return comparison
    
    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Compute average metrics across multiple results.
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Dictionary with averaged metrics
        """
        if not metrics_list:
            return {}
            
        avg_metrics = {}
        metric_keys = set()
        
        # Collect all metric keys
        for metrics in metrics_list:
            metric_keys.update(metrics.keys())
        
        # Compute average for each metric
        for key in metric_keys:
            values = [metrics[key] for metrics in metrics_list if key in metrics]
            if values:
                avg_metrics[key] = np.mean(values)
        
        return avg_metrics
    
    def _find_best_hardware_for_family(self, family_hw_metrics: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, str]:
        """
        Find the best hardware platform for a model family based on comparison metrics.
        
        Args:
            family_hw_metrics: Hardware comparison metrics for a model family
            
        Returns:
            Dictionary with best hardware recommendations
        """
        if not family_hw_metrics:
            return {}
        
        # Aggregate scores for each hardware
        hw_scores = {}
        
        for hw1, comparisons in family_hw_metrics.items():
            if hw1 not in hw_scores:
                hw_scores[hw1] = {
                    "win_count": 0,
                    "total_comparisons": 0,
                    "throughput_score": 0,
                    "latency_score": 0,
                    "overall_score": 0
                }
            
            for hw2, comparison in comparisons.items():
                hw_scores[hw1]["total_comparisons"] += 1
                
                # Count wins for throughput
                if "throughput_ratio" in comparison and comparison["throughput_ratio"] > 1:
                    hw_scores[hw1]["win_count"] += 1
                    hw_scores[hw1]["throughput_score"] += comparison["throughput_ratio"]
                
                # Count wins for latency
                if "latency_ratio" in comparison and comparison["latency_ratio"] > 1:
                    hw_scores[hw1]["win_count"] += 1
                    hw_scores[hw1]["latency_score"] += comparison["latency_ratio"]
                
                # Add overall score
                if "overall_score" in comparison:
                    hw_scores[hw1]["overall_score"] += comparison["overall_score"]
        
        # Normalize scores by number of comparisons
        for hw, scores in hw_scores.items():
            if scores["total_comparisons"] > 0:
                scores["win_rate"] = scores["win_count"] / (scores["total_comparisons"] * 2)  # Multiplied by 2 because we count throughput and latency separately
                scores["throughput_score"] /= scores["total_comparisons"]
                scores["latency_score"] /= scores["total_comparisons"]
                scores["overall_score"] /= scores["total_comparisons"]
        
        # Find best hardware for different metrics
        best_hardware = {}
        
        # Best overall (highest win rate)
        if hw_scores:
            best_overall = max(hw_scores.items(), key=lambda x: x[1]["win_rate"], default=(None, {}))[0]
            best_hardware["overall"] = best_overall
        
        # Best for throughput
        if hw_scores:
            best_throughput = max(hw_scores.items(), key=lambda x: x[1]["throughput_score"], default=(None, {}))[0]
            best_hardware["throughput"] = best_throughput
        
        # Best for latency
        if hw_scores:
            best_latency = max(hw_scores.items(), key=lambda x: x[1]["latency_score"], default=(None, {}))[0]
            best_hardware["latency"] = best_latency
        
        return best_hardware
    
    def generate_comparison_report(self, 
                                  results: Dict[str, Dict[str, Any]], 
                                  output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Generate a cross-hardware comparison report.
        
        Args:
            results: Dictionary mapping models to hardware results
            output_dir: Directory to save the report (defaults to self.output_dir)
            
        Returns:
            Dictionary mapping report names to file paths
        """
        report_dir = output_dir or self.output_dir
        ensure_dir_exists(report_dir)
        
        # Generate comparison data
        comparison_data = self.generate_comparison(results)
        
        # Generate HTML report
        html_report = os.path.join(report_dir, "cross_hardware_comparison.html")
        self._generate_html_report(comparison_data, html_report)
        
        # Generate Markdown report
        md_report = os.path.join(report_dir, "cross_hardware_comparison.md")
        self._generate_markdown_report(comparison_data, md_report)
        
        # Generate visualizations
        viz_dir = os.path.join(report_dir, "visualizations")
        ensure_dir_exists(viz_dir)
        
        # Generate performance heatmap
        heatmap_file = os.path.join(viz_dir, "performance_heatmap.png")
        self._generate_performance_heatmap(comparison_data, heatmap_file)
        
        # Generate family comparison chart
        family_chart_file = os.path.join(viz_dir, "family_comparison.png")
        self._generate_family_comparison_chart(comparison_data, family_chart_file)
        
        return {
            "html": html_report,
            "markdown": md_report,
            "heatmap": heatmap_file,
            "family_chart": family_chart_file
        }
    
    def _generate_html_report(self, 
                             comparison_data: Dict[str, Any], 
                             output_path: str) -> None:
        """
        Generate an HTML report for cross-hardware comparison.
        
        Args:
            comparison_data: Cross-hardware comparison data
            output_path: Path to save the HTML report
        """
        family_comparison = comparison_data.get("family_comparison", {})
        family_optimal_hardware = comparison_data.get("family_optimal_hardware", {})
        hardware_list = comparison_data.get("hardware_list", [])
        
        # Basic HTML structure
        with open(output_path, 'w') as f:
            f.write("<!DOCTYPE html><html><head><title>Cross-Hardware Performance Comparison</title>")
            f.write("<style>body{font-family:sans-serif;line-height:1.6;max-width:1200px;margin:0 auto;padding:1em}")
            f.write("h1,h2,h3{color:#333}table{border-collapse:collapse;width:100%}th,td{text-align:left;border:1px solid #ddd;padding:8px}")
            f.write("th{background:#f2f2f2}.success{color:green}.failure{color:red}.chart{width:100%;max-width:800px;margin:1em 0}</style></head><body>")
            
            # Report header
            f.write("<h1>Cross-Hardware Performance Comparison</h1>")
            f.write("<p>This report compares performance metrics across different hardware platforms to identify optimal hardware for each model type.</p>")
            
            # Visualizations
            f.write("<h2>Performance Visualization</h2>")
            f.write("<div><img src='visualizations/performance_heatmap.png' alt='Performance Heatmap' class='chart'>")
            f.write("<img src='visualizations/family_comparison.png' alt='Family Comparison Chart' class='chart'></div>")
            
            # Recommended hardware by model family
            f.write("<h2>Recommended Hardware by Model Family</h2>")
            f.write("<table><tr><th>Model Family</th><th>Best Overall</th><th>Best for Throughput</th><th>Best for Latency</th></tr>")
            
            for family, optimal in family_optimal_hardware.items():
                best_hw = optimal.get("best_hardware", {})
                best_overall = best_hw.get("overall", "N/A")
                best_throughput = best_hw.get("throughput", "N/A")
                best_latency = best_hw.get("latency", "N/A")
                
                f.write(f"<tr><td>{family}</td><td>{best_overall}</td><td>{best_throughput}</td><td>{best_latency}</td></tr>")
                
            f.write("</table>")
            
            # Detailed hardware comparison
            f.write("<h2>Detailed Hardware Comparison</h2>")
            
            for family, family_comparisons in family_comparison.items():
                f.write(f"<h3>{family}</h3>")
                
                # Create comparison table for this family
                if hardware_list:
                    f.write("<table><tr><th>Hardware</th>")
                    for hw in hardware_list:
                        f.write(f"<th>{hw}</th>")
                    f.write("</tr>")
                    
                    # Data rows
                    for hw1 in hardware_list:
                        if hw1 not in family_comparisons:
                            continue
                            
                        f.write(f"<tr><td><strong>{hw1}</strong></td>")
                        
                        for hw2 in hardware_list:
                            if hw1 == hw2:
                                f.write("<td>-</td>")
                            elif hw2 in family_comparisons.get(hw1, {}):
                                comparison = family_comparisons[hw1][hw2]
                                throughput_pct = comparison.get("throughput_percentage", 0)
                                latency_pct = comparison.get("latency_percentage", 0)
                                
                                # Format cell content
                                cell_class = "success" if throughput_pct > 0 and latency_pct > 0 else "failure" if throughput_pct < 0 and latency_pct < 0 else ""
                                cell_content = f"T: {throughput_pct:+.1f}%<br>L: {latency_pct:+.1f}%"
                                
                                f.write(f"<td class='{cell_class}'>{cell_content}</td>")
                            else:
                                f.write("<td>N/A</td>")
                                
                        f.write("</tr>")
                        
                    f.write("</table>")
            
            f.write("</body></html>")
    
    def _generate_markdown_report(self, 
                                comparison_data: Dict[str, Any], 
                                output_path: str) -> None:
        """
        Generate a Markdown report for cross-hardware comparison.
        
        Args:
            comparison_data: Cross-hardware comparison data
            output_path: Path to save the Markdown report
        """
        family_comparison = comparison_data.get("family_comparison", {})
        family_optimal_hardware = comparison_data.get("family_optimal_hardware", {})
        hardware_list = comparison_data.get("hardware_list", [])
        
        with open(output_path, 'w') as f:
            # Report header
            f.write("# Cross-Hardware Performance Comparison\n\n")
            f.write("This report compares performance metrics across different hardware platforms to identify optimal hardware for each model type.\n\n")
            
            # Visualizations
            f.write("## Performance Visualization\n\n")
            f.write("![Performance Heatmap](visualizations/performance_heatmap.png)\n\n")
            f.write("![Family Comparison Chart](visualizations/family_comparison.png)\n\n")
            
            # Recommended hardware by model family
            f.write("## Recommended Hardware by Model Family\n\n")
            f.write("| Model Family | Best Overall | Best for Throughput | Best for Latency |\n")
            f.write("|-------------|-------------|-------------------|----------------|\n")
            
            for family, optimal in family_optimal_hardware.items():
                best_hw = optimal.get("best_hardware", {})
                best_overall = best_hw.get("overall", "N/A")
                best_throughput = best_hw.get("throughput", "N/A")
                best_latency = best_hw.get("latency", "N/A")
                
                f.write(f"| {family} | {best_overall} | {best_throughput} | {best_latency} |\n")
                
            f.write("\n")
            
            # Detailed hardware comparison
            f.write("## Detailed Hardware Comparison\n\n")
            
            for family, family_comparisons in family_comparison.items():
                f.write(f"### {family}\n\n")
                
                # Create comparison table for this family
                if hardware_list:
                    # Header row with hardware names
                    f.write("| Hardware | " + " | ".join(hardware_list) + " |\n")
                    f.write("|----------|" + "|".join(["---" for _ in hardware_list]) + "|\n")
                    
                    # Data rows
                    for hw1 in hardware_list:
                        if hw1 not in family_comparisons:
                            continue
                            
                        row = [f"**{hw1}**"]
                        
                        for hw2 in hardware_list:
                            if hw1 == hw2:
                                row.append("-")
                            elif hw2 in family_comparisons.get(hw1, {}):
                                comparison = family_comparisons[hw1][hw2]
                                throughput_pct = comparison.get("throughput_percentage", 0)
                                latency_pct = comparison.get("latency_percentage", 0)
                                
                                # Format cell content
                                cell_content = f"T: {throughput_pct:+.1f}%<br>L: {latency_pct:+.1f}%"
                                row.append(cell_content)
                            else:
                                row.append("N/A")
                                
                        f.write("| " + " | ".join(row) + " |\n")
                        
                    f.write("\n")
    
    def _generate_performance_heatmap(self, 
                                     comparison_data: Dict[str, Any], 
                                     output_path: str) -> None:
        """
        Generate a performance heatmap visualization.
        
        Args:
            comparison_data: Cross-hardware comparison data
            output_path: Path to save the visualization
        """
        hardware_list = comparison_data.get("hardware_list", [])
        model_metrics = comparison_data.get("model_metrics", {})
        model_families = comparison_data.get("model_families", {})
        
        if not hardware_list or not model_metrics:
            # Create empty visualization if no data
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No performance data available", 
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.gca().set_axis_off()
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            return
        
        # Prepare data for heatmap
        throughput_data = np.zeros((len(model_metrics), len(hardware_list)))
        throughput_data[:] = np.nan  # Initialize with NaN
        
        # Group models by family for better visualization
        family_order = ["text-embedding", "text-generation", "vision", "audio", "multimodal", "unknown"]
        models_by_family = {family: [] for family in family_order}
        
        for model, family in model_families.items():
            models_by_family[family].append(model)
        
        # Flatten the ordered model list
        ordered_models = []
        for family in family_order:
            ordered_models.extend(sorted(models_by_family[family]))
        
        # Fill in throughput data
        for i, model in enumerate(ordered_models):
            if model not in model_metrics:
                continue
                
            for j, hw in enumerate(hardware_list):
                if hw in model_metrics[model] and "throughput" in model_metrics[model][hw]:
                    throughput_data[i, j] = model_metrics[model][hw]["throughput"]
        
        # Create the heatmap
        plt.figure(figsize=(max(12, len(hardware_list) * 1.5), max(8, len(ordered_models) * 0.4)))
        
        # Create custom colormap
        cmap = plt.cm.viridis
        
        # Create the heatmap
        ax = sns.heatmap(
            throughput_data, 
            cmap=cmap,
            linewidths=.5,
            cbar_kws={'label': 'Throughput (items/second)'},
            vmin=0,
            annot=True,
            fmt=".1f",
            mask=np.isnan(throughput_data)
        )
        
        # Set labels
        plt.title("Throughput Performance by Model and Hardware", fontsize=16)
        plt.ylabel("Model")
        plt.xlabel("Hardware Platform")
        
        # Set tick labels
        ax.set_yticklabels(ordered_models, rotation=0)
        ax.set_xticklabels(hardware_list, rotation=45, ha="right")
        
        # Add family separators
        current_pos = 0
        for family in family_order:
            family_count = len(models_by_family[family])
            if family_count > 0:
                if current_pos > 0:
                    plt.axhline(y=current_pos, color='black', linestyle='-', linewidth=1)
                current_pos += family_count
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def _generate_family_comparison_chart(self, 
                                        comparison_data: Dict[str, Any], 
                                        output_path: str) -> None:
        """
        Generate a chart comparing performance across model families.
        
        Args:
            comparison_data: Cross-hardware comparison data
            output_path: Path to save the visualization
        """
        family_comparison = comparison_data.get("family_comparison", {})
        hardware_list = comparison_data.get("hardware_list", [])
        family_optimal_hardware = comparison_data.get("family_optimal_hardware", {})
        
        if not family_comparison or not hardware_list:
            # Create empty visualization if no data
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "No family comparison data available", 
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.gca().set_axis_off()
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            return
        
        # Sort families for consistent display
        sorted_families = sorted(family_optimal_hardware.keys())
        
        # Create a simple bar chart showing best hardware by family
        plt.figure(figsize=(14, 8))
        
        # Define colors for hardware platforms
        hardware_colors = {}
        for i, hw in enumerate(hardware_list):
            hardware_colors[hw] = plt.cm.tab10(i % 10)
        
        # Prepare data
        bar_width = 0.25
        family_pos = np.arange(len(sorted_families))
        
        # Plot best overall hardware
        best_overall = []
        for family in sorted_families:
            best_hw = family_optimal_hardware.get(family, {}).get("best_hardware", {})
            best_overall.append(best_hw.get("overall"))
        
        # Create the bar plot
        plt.bar(
            family_pos - bar_width,
            [1] * len(sorted_families),
            width=bar_width,
            color=[hardware_colors.get(hw, 'gray') for hw in best_overall],
            label="Best Overall"
        )
        
        # Plot best for throughput
        best_throughput = []
        for family in sorted_families:
            best_hw = family_optimal_hardware.get(family, {}).get("best_hardware", {})
            best_throughput.append(best_hw.get("throughput"))
        
        plt.bar(
            family_pos,
            [1] * len(sorted_families),
            width=bar_width,
            color=[hardware_colors.get(hw, 'gray') for hw in best_throughput],
            label="Best for Throughput"
        )
        
        # Plot best for latency
        best_latency = []
        for family in sorted_families:
            best_hw = family_optimal_hardware.get(family, {}).get("best_hardware", {})
            best_latency.append(best_hw.get("latency"))
        
        plt.bar(
            family_pos + bar_width,
            [1] * len(sorted_families),
            width=bar_width,
            color=[hardware_colors.get(hw, 'gray') for hw in best_latency],
            label="Best for Latency"
        )
        
        # Add hardware name labels
        for i, family in enumerate(sorted_families):
            best_hw = family_optimal_hardware.get(family, {}).get("best_hardware", {})
            
            # Label best overall
            plt.text(
                i - bar_width,
                0.5,
                best_hw.get("overall", ""),
                ha="center",
                va="center",
                rotation=90,
                color="white",
                fontweight="bold"
            )
            
            # Label best for throughput
            plt.text(
                i,
                0.5,
                best_hw.get("throughput", ""),
                ha="center",
                va="center",
                rotation=90,
                color="white",
                fontweight="bold"
            )
            
            # Label best for latency
            plt.text(
                i + bar_width,
                0.5,
                best_hw.get("latency", ""),
                ha="center",
                va="center",
                rotation=90,
                color="white",
                fontweight="bold"
            )
        
        # Add legend
        plt.legend()
        
        # Set labels and ticks
        plt.title("Best Hardware by Model Family", fontsize=16)
        plt.xlabel("Model Family")
        plt.ylabel("")
        plt.xticks(family_pos, sorted_families)
        plt.yticks([])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Utility for validating simulations and comparing hardware performance"
    )
    
    parser.add_argument(
        "--input-dir", 
        default="collected_results",
        help="Directory containing test results"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="reports",
        help="Directory to save reports"
    )
    
    parser.add_argument(
        "--simulation-validation", 
        action="store_true",
        help="Generate simulation validation report"
    )
    
    parser.add_argument(
        "--cross-hardware-comparison", 
        action="store_true",
        help="Generate cross-hardware comparison report"
    )
    
    parser.add_argument(
        "--combined-report", 
        action="store_true",
        help="Generate both reports"
    )
    
    parser.add_argument(
        "--tolerance", 
        type=float,
        default=SIMULATION_TOLERANCE,
        help="Tolerance for simulation validation (as percentage)"
    )
    
    args = parser.parse_args()
    
    # Load test results
    results = {}
    
    # Example usage
    if args.simulation_validation or args.combined_report:
        validator = SimulationValidator(tolerance=args.tolerance)
        validator.generate_validation_report(results, os.path.join(args.output_dir, "simulation_validation"))
        
    if args.cross_hardware_comparison or args.combined_report:
        comparator = CrossHardwareComparison(output_dir=os.path.join(args.output_dir, "cross_hardware"))
        comparator.generate_comparison_report(results)