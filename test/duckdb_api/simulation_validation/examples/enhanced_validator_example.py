#!/usr/bin/env python3
"""
Example usage of the EnhancedStatisticalValidator.

This example demonstrates how to use the EnhancedStatisticalValidator to perform advanced
statistical analysis of simulation accuracy against real hardware measurements.
"""

import os
import sys
import datetime
import numpy as np
from pathlib import Path

# Add parent directories to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import necessary classes
from duckdb_api.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult
)

from duckdb_api.simulation_validation.statistical.enhanced_statistical_validator import (
    EnhancedStatisticalValidator,
    get_enhanced_statistical_validator_instance
)

def create_sample_results(n_samples=1):
    """
    Create sample simulation and hardware results for testing.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (sim_results, hw_results) lists
    """
    sim_results = []
    hw_results = []
    
    # Base values
    base_metrics = {
        "throughput_items_per_second": 95.0,
        "average_latency_ms": 105.0,
        "memory_peak_mb": 5250.0,
        "power_consumption_w": 210.0,
        "initialization_time_ms": 420.0,
        "warmup_time_ms": 210.0
    }
    
    hw_base_metrics = {
        "throughput_items_per_second": 100.0,
        "average_latency_ms": 100.0,
        "memory_peak_mb": 5000.0,
        "power_consumption_w": 200.0,
        "initialization_time_ms": 400.0,
        "warmup_time_ms": 200.0
    }
    
    for i in range(n_samples):
        # Add some randomness for multiple samples
        if n_samples > 1:
            sim_metrics = {k: v + np.random.normal(0, v * 0.02) for k, v in base_metrics.items()}
            hw_metrics = {k: v + np.random.normal(0, v * 0.02) for k, v in hw_base_metrics.items()}
        else:
            sim_metrics = base_metrics.copy()
            hw_metrics = hw_base_metrics.copy()
        
        # Create simulation result
        sim_result = SimulationResult(
            model_id="bert-base-uncased",
            hardware_id="rtx3080",
            metrics=sim_metrics,
            batch_size=32,
            precision="fp16",
            timestamp=datetime.datetime.now().isoformat(),
            simulation_version="v1.0",
            additional_metadata={"configuration": "default"}
        )
        
        # Create hardware result
        hw_result = HardwareResult(
            model_id="bert-base-uncased",
            hardware_id="rtx3080",
            metrics=hw_metrics,
            batch_size=32,
            precision="fp16",
            timestamp=datetime.datetime.now().isoformat(),
            hardware_details={"gpu": "RTX 3080"},
            test_environment={"driver_version": "456.71"},
            additional_metadata={"measurement_method": "avg_of_5_runs"}
        )
        
        sim_results.append(sim_result)
        hw_results.append(hw_result)
    
    return sim_results, hw_results

def demonstrate_basic_validation():
    """Demonstrate basic validation with the EnhancedStatisticalValidator."""
    print("\n=== Basic Validation ===\n")
    
    # Create single sample
    sim_results, hw_results = create_sample_results(1)
    
    # Create validator instance
    validator = get_enhanced_statistical_validator_instance()
    
    # Validate single result
    validation_result = validator.validate(sim_results[0], hw_results[0])
    
    # Print basic metrics
    print("Basic metrics:")
    for metric_name, comparison in validation_result.metrics_comparison.items():
        if "mape" in comparison:
            print(f"  {metric_name} MAPE: {comparison['mape']:.2f}%")
    
    # Print enhanced metrics
    print("\nEnhanced metrics:")
    if validation_result.additional_metrics and "enhanced_metrics" in validation_result.additional_metrics:
        for metric_name, metrics in validation_result.additional_metrics["enhanced_metrics"].items():
            print(f"  {metric_name}:")
            for metric_type, value in metrics.items():
                if not np.isnan(value):
                    print(f"    {metric_type}: {value:.4f}")
    
    # Print confidence intervals
    print("\nConfidence intervals:")
    if validation_result.additional_metrics and "confidence_intervals" in validation_result.additional_metrics:
        for metric_name, ci_data in validation_result.additional_metrics["confidence_intervals"].items():
            if "mape" in ci_data:
                ci = ci_data["mape"]
                print(f"  {metric_name} MAPE: {ci['value']:.2f}% (95% CI: {ci['lower_bound']:.2f}% - {ci['upper_bound']:.2f}%)")
    
    # Print Bland-Altman analysis
    print("\nBland-Altman analysis:")
    if validation_result.additional_metrics and "bland_altman" in validation_result.additional_metrics:
        for metric_name, ba_data in validation_result.additional_metrics["bland_altman"].items():
            print(f"  {metric_name}:")
            print(f"    Bias: {ba_data['bias']:.4f}")
            print(f"    Limits of Agreement: {ba_data['lower_loa']:.4f} to {ba_data['upper_loa']:.4f}")

def demonstrate_batch_validation():
    """Demonstrate batch validation with the EnhancedStatisticalValidator."""
    print("\n=== Batch Validation ===\n")
    
    # Create multiple samples
    n_samples = 10
    sim_results, hw_results = create_sample_results(n_samples)
    
    # Create validator instance
    validator = get_enhanced_statistical_validator_instance()
    
    # Validate batch of results
    validation_results = validator.validate_batch(sim_results, hw_results)
    
    # Create summary
    summary = validator.summarize_validation(validation_results)
    
    # Print overall metrics
    print(f"Number of validation results: {len(validation_results)}")
    print("\nOverall metrics:")
    if "overall" in summary and "mape" in summary["overall"]:
        mape = summary["overall"]["mape"]
        print(f"  Mean MAPE: {mape['mean']:.2f}%")
        print(f"  Median MAPE: {mape['median']:.2f}%")
        print(f"  Min MAPE: {mape['min']:.2f}%")
        print(f"  Max MAPE: {mape['max']:.2f}%")
    
    # Print status
    if "overall" in summary and "status" in summary["overall"]:
        print(f"  Status: {summary['overall']['status']}")
    
    # Print power analysis
    print("\nPower analysis:")
    if "enhanced" in summary and "power_analysis" in summary["enhanced"]:
        for effect_size, power_data in summary["enhanced"]["power_analysis"].items():
            print(f"  Effect size {effect_size}:")
            if "mean" in power_data:
                print(f"    Mean power: {power_data['mean']:.4f}")
            if "proportion_sufficient" in power_data:
                print(f"    Sufficient power: {power_data['proportion_sufficient'] * 100:.1f}%")
    
    # Print distribution tests
    if len(validation_results) > 0 and validation_results[0].additional_metrics and "distribution_comparison" in validation_results[0].additional_metrics:
        print("\nDistribution tests:")
        for metric_name, dist_data in validation_results[0].additional_metrics["distribution_comparison"].items():
            if "tests" in dist_data:
                print(f"  {metric_name}:")
                for test_name, test_data in dist_data["tests"].items():
                    print(f"    {test_name}:")
                    for key, value in test_data.items():
                        if isinstance(value, dict) and "is_normal" in value:
                            print(f"      {key}: {value['is_normal']}")
                        elif isinstance(value, dict) and "equal_variances" in value:
                            print(f"      {key}: {value['equal_variances']}")
                        elif isinstance(value, dict) and "same_distribution" in value:
                            print(f"      {key}: {value['same_distribution']}")

def main():
    """Main function to demonstrate the EnhancedStatisticalValidator."""
    print("Enhanced Statistical Validator Example")
    print("======================================")
    
    # Demonstrate basic validation
    demonstrate_basic_validation()
    
    # Demonstrate batch validation
    demonstrate_batch_validation()

if __name__ == "__main__":
    main()