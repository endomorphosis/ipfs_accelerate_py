#!/usr/bin/env python3
"""
Test script for the Basic Simulation Calibrator component of the Simulation Accuracy and Validation Framework.

This script tests the functionality of the BasicSimulationCalibrator class for
calibrating simulation parameters based on validation results.
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
logger = logging.getLogger("test_simulation_calibrator")

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent))

# Import components
try:
    from duckdb_api.simulation_validation.core.base import (
        SimulationResult,
        HardwareResult,
        ValidationResult
    )
    from duckdb_api.simulation_validation.statistical.basic_validator import BasicStatisticalValidator
    from duckdb_api.simulation_validation.calibration.basic_calibrator import BasicSimulationCalibrator
    HAS_COMPONENTS = True
except ImportError as e:
    logger.error(f"Error importing simulation validation components: {e}")
    HAS_COMPONENTS = False


def create_test_data():
    """Create test data for simulation and hardware results."""
    # Create simulation results
    simulation_results = [
        # BERT on NVIDIA A100 (overestimates throughput, underestimates latency)
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
        # BERT on NVIDIA A100 with different batch size
        SimulationResult(
            model_id="bert-base-uncased",
            hardware_id="nvidia-a100",
            metrics={
                "throughput_items_per_second": 140.2,
                "average_latency_ms": 11.4,
                "memory_peak_mb": 10240,
                "power_consumption_w": 380.0
            },
            batch_size=16,
            precision="fp16",
            simulation_version="sim_v1.2"
        ),
        # BERT on NVIDIA A100 with different precision
        SimulationResult(
            model_id="bert-base-uncased",
            hardware_id="nvidia-a100",
            metrics={
                "throughput_items_per_second": 95.3,
                "average_latency_ms": 10.5,
                "memory_peak_mb": 6144,
                "power_consumption_w": 330.0
            },
            batch_size=8,
            precision="int8",
            simulation_version="sim_v1.2"
        ),
        # BERT on AMD MI250 (overestimates throughput more significantly)
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
        # T5 on NVIDIA A100 (underestimates throughput slightly)
        SimulationResult(
            model_id="t5-base",
            hardware_id="nvidia-a100",
            metrics={
                "throughput_items_per_second": 62.8,
                "average_latency_ms": 16.0,
                "memory_peak_mb": 12288,
                "power_consumption_w": 375.0
            },
            batch_size=4,
            precision="fp16",
            simulation_version="sim_v1.2"
        )
    ]
    
    # Create hardware results
    hardware_results = [
        # BERT on NVIDIA A100
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
        # BERT on NVIDIA A100 with different batch size
        HardwareResult(
            model_id="bert-base-uncased",
            hardware_id="nvidia-a100",
            metrics={
                "throughput_items_per_second": 132.5,
                "average_latency_ms": 12.1,
                "memory_peak_mb": 10500,
                "power_consumption_w": 395.0
            },
            batch_size=16,
            precision="fp16",
            hardware_details={"gpu_memory": "40GB", "cuda_cores": 6912},
            test_environment={"driver_version": "470.82.01", "cuda_version": "11.4"}
        ),
        # BERT on NVIDIA A100 with different precision
        HardwareResult(
            model_id="bert-base-uncased",
            hardware_id="nvidia-a100",
            metrics={
                "throughput_items_per_second": 92.4,
                "average_latency_ms": 10.8,
                "memory_peak_mb": 6080,
                "power_consumption_w": 325.0
            },
            batch_size=8,
            precision="int8",
            hardware_details={"gpu_memory": "40GB", "cuda_cores": 6912},
            test_environment={"driver_version": "470.82.01", "cuda_version": "11.4"}
        ),
        # BERT on AMD MI250
        HardwareResult(
            model_id="bert-base-uncased",
            hardware_id="amd-mi250",
            metrics={
                "throughput_items_per_second": 80.4,
                "average_latency_ms": 12.4,
                "memory_peak_mb": 7552,
                "power_consumption_w": 345.0
            },
            batch_size=8,
            precision="fp16",
            hardware_details={"gpu_memory": "128GB", "compute_units": 220},
            test_environment={"driver_version": "22.20.3", "rocm_version": "5.2.0"}
        ),
        # T5 on NVIDIA A100
        HardwareResult(
            model_id="t5-base",
            hardware_id="nvidia-a100",
            metrics={
                "throughput_items_per_second": 65.3,
                "average_latency_ms": 15.3,
                "memory_peak_mb": 12100,
                "power_consumption_w": 385.0
            },
            batch_size=4,
            precision="fp16",
            hardware_details={"gpu_memory": "40GB", "cuda_cores": 6912},
            test_environment={"driver_version": "470.82.01", "cuda_version": "11.4"}
        )
    ]
    
    return simulation_results, hardware_results


def test_calibration_process():
    """Test the simulation calibration process."""
    # Create test data
    simulation_results, hardware_results = create_test_data()
    
    # Create validator for generating validation results
    validator = BasicStatisticalValidator()
    validation_results = validator.validate_batch(simulation_results, hardware_results)
    
    # Initialize calibrator
    calibrator = BasicSimulationCalibrator(
        config={
            "calibration_method": "linear_scaling",
            "min_samples_per_hardware": 1,
            "learning_rate": 1.0  # Use full learning rate for testing
        }
    )
    
    # Initial simulation parameters
    initial_parameters = {
        "simulation_version": "sim_v1.2",
        "correction_factors": {
            # Some pre-existing correction factors (should be updated)
            "nvidia-a100": {
                "bert-base-uncased": {
                    "throughput_items_per_second": 0.98,
                    "average_latency_ms": 1.02
                }
            }
        }
    }
    
    # Run calibration
    calibrated_parameters = calibrator.calibrate(validation_results, initial_parameters)
    
    # Check that calibration was performed
    assert "correction_factors" in calibrated_parameters
    assert "nvidia-a100" in calibrated_parameters["correction_factors"]
    assert "bert-base-uncased" in calibrated_parameters["correction_factors"]["nvidia-a100"]
    assert "t5-base" in calibrated_parameters["correction_factors"]["nvidia-a100"]
    assert "amd-mi250" in calibrated_parameters["correction_factors"]
    
    # Check that correction factors were updated
    bert_factors = calibrated_parameters["correction_factors"]["nvidia-a100"]["bert-base-uncased"]
    assert "throughput_items_per_second" in bert_factors
    assert "average_latency_ms" in bert_factors
    
    # Check calibration metadata
    assert "calibration_version" in calibrated_parameters
    assert "calibration_method" in calibrated_parameters
    assert "num_samples_used" in calibrated_parameters
    
    logger.info("Calibration process test passed successfully")
    return calibrated_parameters, validation_results


def test_apply_calibration(calibrated_parameters, original_data):
    """Test applying calibration to new simulation results."""
    # Extract original simulation results
    simulation_results, _ = original_data
    
    # Create calibrator
    calibrator = BasicSimulationCalibrator()
    
    # Apply calibration to each simulation result
    calibrated_results = []
    for sim_result in simulation_results:
        calibrated_result = calibrator.apply_calibration(sim_result, calibrated_parameters)
        calibrated_results.append(calibrated_result)
    
    # Check that calibration was applied
    for orig_result, calib_result in zip(simulation_results, calibrated_results):
        # Calibrated version should be updated
        assert calib_result.simulation_version.endswith("_calibrated")
        
        # Check that metrics were updated
        for metric in ["throughput_items_per_second", "average_latency_ms"]:
            if metric in orig_result.metrics and metric in calib_result.metrics:
                # Metrics should be different after calibration
                assert orig_result.metrics[metric] != calib_result.metrics[metric]
        
        # Check that additional metadata was added
        assert "calibration_applied" in calib_result.additional_metadata
    
    logger.info("Apply calibration test passed successfully")
    return calibrated_results


def test_calibration_evaluation(validation_results, calibrated_results, hardware_results):
    """Test evaluation of calibration effectiveness."""
    # Create validator
    validator = BasicStatisticalValidator()
    
    # Create validation results for calibrated simulation results
    calibrated_validation_results = validator.validate_batch(calibrated_results, hardware_results)
    
    # Create calibrator
    calibrator = BasicSimulationCalibrator()
    
    # Evaluate calibration
    evaluation = calibrator.evaluate_calibration(validation_results, calibrated_validation_results)
    
    # Check evaluation structure
    assert "metrics" in evaluation
    assert "overall" in evaluation
    
    # Check that metrics were evaluated
    for metric in ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb", "power_consumption_w"]:
        if metric in evaluation["metrics"]:
            assert "before_mape" in evaluation["metrics"][metric]
            assert "after_mape" in evaluation["metrics"][metric]
            assert "absolute_improvement" in evaluation["metrics"][metric]
            assert "relative_improvement" in evaluation["metrics"][metric]
    
    # Check overall evaluation
    assert "before_mape" in evaluation["overall"]
    assert "after_mape" in evaluation["overall"]
    assert "absolute_improvement" in evaluation["overall"]
    assert "relative_improvement" in evaluation["overall"]
    
    logger.info("Calibration evaluation test passed successfully")
    return evaluation


def save_results(calibrated_parameters, evaluation, output_path):
    """Save calibration parameters and evaluation to files."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save calibrated parameters
    with open(f"{output_path}_calibrated_parameters.json", "w") as f:
        json.dump(calibrated_parameters, f, indent=2)
    
    # Save evaluation
    with open(f"{output_path}_evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=2)
    
    logger.info(f"Results saved to {output_path}_calibrated_parameters.json and {output_path}_evaluation.json")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test the Basic Simulation Calibrator")
    parser.add_argument("--output", default="./simulation_calibration_results",
                        help="Path prefix for output files")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving results to files")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed results to console")
    parser.add_argument("--method", choices=["linear_scaling", "additive_adjustment", "regression"],
                        default="linear_scaling", help="Calibration method to use")
    
    args = parser.parse_args()
    
    # Check if components are available
    if not HAS_COMPONENTS:
        logger.error("Simulation validation components not available.")
        logger.error("Please check if the components are properly installed.")
        return 1
    
    # Run calibration tests
    try:
        # Create test data
        original_data = create_test_data()
        
        # Test calibration process
        calibrated_parameters, validation_results = test_calibration_process()
        
        # Test applying calibration
        calibrated_results = test_apply_calibration(calibrated_parameters, original_data)
        
        # Test calibration evaluation
        evaluation = test_calibration_evaluation(validation_results, calibrated_results, original_data[1])
        
        # Print results to console
        logger.info(f"Overall calibration improvement: {evaluation['overall']['relative_improvement']:.2f}%")
        logger.info(f"MAPE before: {evaluation['overall']['before_mape']:.2f}%, after: {evaluation['overall']['after_mape']:.2f}%")
        
        # Print detailed results if requested
        if args.verbose:
            print("\nCalibration improvements by metric:")
            for metric, metric_eval in evaluation["metrics"].items():
                print(f"  {metric}:")
                print(f"    Before MAPE: {metric_eval['before_mape']:.2f}%")
                print(f"    After MAPE: {metric_eval['after_mape']:.2f}%")
                print(f"    Improvement: {metric_eval['relative_improvement']:.2f}%")
            
            print("\nCorrection factors by hardware/model:")
            for hw_id, hw_factors in calibrated_parameters["correction_factors"].items():
                for model_id, model_factors in hw_factors.items():
                    print(f"  {hw_id}/{model_id}:")
                    for metric, factor in model_factors.items():
                        print(f"    {metric}: {factor:.4f}")
        
        # Save results to files
        if not args.no_save:
            save_results(calibrated_parameters, evaluation, args.output)
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    logger.info("All tests completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())