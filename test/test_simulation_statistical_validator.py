#!/usr/bin/env python3
"""
Test script for the Basic Statistical Validator component of the Simulation Accuracy and Validation Framework.

This script tests the functionality of the BasicStatisticalValidator class for
validating simulation accuracy against real hardware measurements.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_statistical_validator")

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent))

# Import components
try:
    from data.duckdb.simulation_validation.core.base import (
        SimulationResult,
        HardwareResult,
        ValidationResult
    )
    from data.duckdb.simulation_validation.statistical.basic_validator import BasicStatisticalValidator
    HAS_COMPONENTS = True
except ImportError as e:
    logger.error(f"Error importing simulation validation components: {e}")
    HAS_COMPONENTS = False


def create_test_data():
    """Create test data for simulation and hardware results."""
    # Create simulation results
    simulation_results = [
        SimulationResult(
            model_id="bert-base-uncased",
            hardware_id="nvidia-a100",
            metrics={
                "throughput_items_per_second": 120.5,
                "average_latency_ms": 8.3,
                "memory_peak_mb": 8192,
                "power_consumption_w": 350.0
            },
            batch_size=8,
            precision="fp16",
            simulation_version="sim_v1.2"
        ),
        SimulationResult(
            model_id="bert-base-uncased",
            hardware_id="amd-mi250",
            metrics={
                "throughput_items_per_second": 95.7,
                "average_latency_ms": 10.5,
                "memory_peak_mb": 7168,
                "power_consumption_w": 320.0
            },
            batch_size=8,
            precision="fp16",
            simulation_version="sim_v1.2"
        ),
        SimulationResult(
            model_id="t5-base",
            hardware_id="nvidia-a100",
            metrics={
                "throughput_items_per_second": 65.2,
                "average_latency_ms": 15.3,
                "memory_peak_mb": 12288,
                "power_consumption_w": 380.0
            },
            batch_size=4,
            precision="fp16",
            simulation_version="sim_v1.2"
        )
    ]
    
    # Create hardware results
    hardware_results = [
        HardwareResult(
            model_id="bert-base-uncased",
            hardware_id="nvidia-a100",
            metrics={
                "throughput_items_per_second": 115.2,
                "average_latency_ms": 8.7,
                "memory_peak_mb": 8450,
                "power_consumption_w": 360.0
            },
            batch_size=8,
            precision="fp16",
            hardware_details={"gpu_memory": "40GB", "cuda_cores": 6912},
            test_environment={"driver_version": "470.82.01", "cuda_version": "11.4"}
        ),
        HardwareResult(
            model_id="bert-base-uncased",
            hardware_id="amd-mi250",
            metrics={
                "throughput_items_per_second": 88.4,
                "average_latency_ms": 11.3,
                "memory_peak_mb": 7552,
                "power_consumption_w": 335.0
            },
            batch_size=8,
            precision="fp16",
            hardware_details={"gpu_memory": "128GB", "compute_units": 220},
            test_environment={"driver_version": "22.20.3", "rocm_version": "5.2.0"}
        ),
        HardwareResult(
            model_id="t5-base",
            hardware_id="nvidia-a100",
            metrics={
                "throughput_items_per_second": 62.8,
                "average_latency_ms": 15.9,
                "memory_peak_mb": 12672,
                "power_consumption_w": 385.0
            },
            batch_size=4,
            precision="fp16",
            hardware_details={"gpu_memory": "40GB", "cuda_cores": 6912},
            test_environment={"driver_version": "470.82.01", "cuda_version": "11.4"}
        )
    ]
    
    return simulation_results, hardware_results


def test_basic_validator():
    """Test the basic statistical validator."""
    # Create test data
    simulation_results, hardware_results = create_test_data()
    
    # Create validator
    validator = BasicStatisticalValidator()
    
    # Run validation
    validation_results = validator.validate_batch(simulation_results, hardware_results)
    
    # Check that we got the expected number of validation results
    assert len(validation_results) == 3
    
    # Verify validation results
    for val_result in validation_results:
        # Check that metrics were compared
        assert "throughput_items_per_second" in val_result.metrics_comparison
        assert "average_latency_ms" in val_result.metrics_comparison
        assert "memory_peak_mb" in val_result.metrics_comparison
        assert "power_consumption_w" in val_result.metrics_comparison
        
        # Check that additional metrics were generated
        assert "overall_accuracy_score" in val_result.additional_metrics
        assert "is_acceptable" in val_result.additional_metrics
        assert "prediction_bias" in val_result.additional_metrics
        
        # Check validation methods
        for metric, comparison in val_result.metrics_comparison.items():
            assert "absolute_error" in comparison
            assert "relative_error" in comparison
            assert "mape" in comparison
            assert "percent_error" in comparison
    
    logger.info("Basic validator test passed successfully")
    return validation_results


def test_summary_generation(validation_results):
    """Test the summary generation functionality."""
    # Create validator
    validator = BasicStatisticalValidator()
    
    # Generate summary
    summary = validator.summarize_validation(validation_results)
    
    # Check summary structure
    assert "num_validations" in summary
    assert "metrics" in summary
    assert "models" in summary
    assert "hardware" in summary
    assert "overall" in summary
    
    # Check metrics summary
    metrics_summary = summary["metrics"]
    for metric in ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb", "power_consumption_w"]:
        assert metric in metrics_summary
        assert "mean_mape" in metrics_summary[metric]
        assert "median_mape" in metrics_summary[metric]
        assert "min_mape" in metrics_summary[metric]
        assert "max_mape" in metrics_summary[metric]
    
    # Check models summary
    models_summary = summary["models"]
    for model in ["bert-base-uncased", "t5-base"]:
        assert model in models_summary
        assert "mean_accuracy_score" in models_summary[model]
    
    # Check hardware summary
    hardware_summary = summary["hardware"]
    for hardware in ["nvidia-a100", "amd-mi250"]:
        assert hardware in hardware_summary
        assert "mean_accuracy_score" in hardware_summary[hardware]
    
    # Check overall summary
    overall_summary = summary["overall"]
    assert "mean_accuracy_score" in overall_summary
    assert "median_accuracy_score" in overall_summary
    assert "status" in overall_summary
    
    logger.info("Summary generation test passed successfully")
    return summary


def save_results(validation_results, summary, output_path):
    """Save validation results and summary to files."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save validation results
    val_results_data = [val_result.to_dict() for val_result in validation_results]
    with open(f"{output_path}_validation_results.json", "w") as f:
        json.dump(val_results_data, f, indent=2)
    
    # Save summary
    with open(f"{output_path}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {output_path}_validation_results.json and {output_path}_summary.json")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test the Basic Statistical Validator")
    parser.add_argument("--output", default="./simulation_validation_results",
                        help="Path prefix for output files")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving results to files")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed results to console")
    
    args = parser.parse_args()
    
    # Check if components are available
    if not HAS_COMPONENTS:
        logger.error("Simulation validation components not available.")
        logger.error("Please check if the components are properly installed.")
        return 1
    
    # Run validator test
    try:
        validation_results = test_basic_validator()
        logger.info(f"Validated {len(validation_results)} simulation results")
        
        # Generate summary
        summary = test_summary_generation(validation_results)
        
        # Print status to console
        logger.info(f"Overall validation status: {summary['overall']['status']}")
        logger.info(f"Mean accuracy score: {summary['overall']['mean_accuracy_score']:.2f}%")
        
        # Print detailed results if requested
        if args.verbose:
            print("\nDetailed validation results:")
            for i, val_result in enumerate(validation_results):
                model_id = val_result.simulation_result.model_id
                hardware_id = val_result.simulation_result.hardware_id
                accuracy_score = val_result.additional_metrics["overall_accuracy_score"]
                acceptable = val_result.additional_metrics["is_acceptable"]
                
                print(f"\nResult {i+1}: {model_id} on {hardware_id}")
                print(f"  Accuracy score: {accuracy_score:.2f}%")
                print(f"  Acceptable: {acceptable}")
                
                print("  Metrics comparison:")
                for metric, comparison in val_result.metrics_comparison.items():
                    mape = comparison["mape"]
                    percent_error = comparison["percent_error"]
                    print(f"    {metric}: MAPE = {mape:.2f}%, Error = {percent_error:.2f}%")
            
            print("\nSummary by model:")
            for model, model_summary in summary["models"].items():
                print(f"  {model}: {model_summary['mean_accuracy_score']:.2f}% (n={model_summary['count']})")
            
            print("\nSummary by hardware:")
            for hw, hw_summary in summary["hardware"].items():
                print(f"  {hw}: {hw_summary['mean_accuracy_score']:.2f}% (n={hw_summary['count']})")
        
        # Save results to files
        if not args.no_save:
            save_results(validation_results, summary, args.output)
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    logger.info("All tests completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())