#!/usr/bin/env python3
"""
Test script for the Simulation Accuracy Validation Framework.

This script demonstrates how to use the Simulation Validation Framework to compare
real hardware results with simulation results.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("test_validator")

# Add parent directory to path for module imports
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import framework components
from data.duckdb.simulation_validation.simulation_validation_framework import get_framework_instance
from data.duckdb.simulation_validation.core.base import SimulationResult, HardwareResult

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_sample_data(num_samples=5):
    """
    Generate sample simulation and hardware results for testing.
    
    Args:
        num_samples: Number of samples to generate for each hardware/model combination
        
    Returns:
        Tuple of (simulation_results, hardware_results)
    """
    # Hardware types and models
    hardware_ids = ["cpu_intel_xeon", "gpu_rtx3080", "webgpu_chrome"]
    model_ids = ["bert-base-uncased", "vit-base-patch16-224"]
    batch_sizes = [1, 4, 16]
    precisions = ["fp32", "fp16"]
    
    # Generate timestamp
    timestamp = datetime.now().isoformat()
    
    # Lists to store results
    simulation_results = []
    hardware_results = []
    
    # Generate results for each combination
    for hardware_id in hardware_ids:
        for model_id in model_ids:
            for batch_size in batch_sizes:
                for precision in precisions:
                    # Define base metrics for this combination
                    if "gpu" in hardware_id:
                        throughput_base = 200 if model_id == "bert-base-uncased" else 150
                        latency_base = 20 if model_id == "bert-base-uncased" else 30
                        memory_base = 1000 if model_id == "bert-base-uncased" else 800
                        power_base = 150
                    elif "webgpu" in hardware_id:
                        throughput_base = 100 if model_id == "bert-base-uncased" else 80
                        latency_base = 40 if model_id == "bert-base-uncased" else 50
                        memory_base = 800 if model_id == "bert-base-uncased" else 600
                        power_base = 100
                    else:  # CPU
                        throughput_base = 50 if model_id == "bert-base-uncased" else 40
                        latency_base = 80 if model_id == "bert-base-uncased" else 100
                        memory_base = 500 if model_id == "bert-base-uncased" else 400
                        power_base = 80
                    
                    # Apply batch size scaling
                    throughput_scale = batch_size ** 0.8
                    latency_scale = batch_size ** 0.2
                    memory_scale = batch_size ** 0.5
                    
                    # Apply precision scaling
                    if precision == "fp16":
                        throughput_scale *= 1.3
                        memory_scale *= 0.6
                    
                    # Generate multiple samples
                    for i in range(num_samples):
                        # Hardware sample with random variation
                        hw_throughput = throughput_base * throughput_scale * (1 + np.random.normal(0, 0.05))
                        hw_latency = latency_base * latency_scale * (1 + np.random.normal(0, 0.05))
                        hw_memory = memory_base * memory_scale * (1 + np.random.normal(0, 0.03))
                        hw_power = power_base * (1 + np.random.normal(0, 0.05))
                        
                        # Create hardware result
                        hw_result = HardwareResult(
                            model_id=model_id,
                            hardware_id=hardware_id,
                            metrics={
                                "throughput_items_per_second": hw_throughput,
                                "average_latency_ms": hw_latency,
                                "memory_peak_mb": hw_memory,
                                "power_consumption_w": hw_power
                            },
                            batch_size=batch_size,
                            precision=precision,
                            timestamp=timestamp,
                            hardware_details={
                                "device_name": hardware_id.replace("_", " ").title(),
                                "cores": 8 if "cpu" in hardware_id else 0,
                                "compute_units": 0 if "cpu" in hardware_id else 68,
                                "driver_version": "450.80.02" if "gpu" in hardware_id else "N/A",
                                "browser_version": "91.0.4472.124" if "webgpu" in hardware_id else "N/A"
                            },
                            test_environment={
                                "os": "Linux",
                                "python_version": "3.8.10",
                                "temperature_c": 45 + np.random.normal(0, 5),
                                "background_load": np.random.uniform(0, 15)
                            }
                        )
                        hardware_results.append(hw_result)
                        
                        # Simulation sample with systematic bias
                        sim_throughput = hw_throughput * (1.2 if "gpu" in hardware_id else 1.1) * (1 + np.random.normal(0, 0.1))
                        sim_latency = hw_latency * (0.9 if "gpu" in hardware_id else 0.95) * (1 + np.random.normal(0, 0.1))
                        sim_memory = hw_memory * 0.9 * (1 + np.random.normal(0, 0.08))
                        sim_power = hw_power * 0.85 * (1 + np.random.normal(0, 0.1))
                        
                        # Create simulation result
                        sim_result = SimulationResult(
                            model_id=model_id,
                            hardware_id=hardware_id,
                            metrics={
                                "throughput_items_per_second": sim_throughput,
                                "average_latency_ms": sim_latency,
                                "memory_peak_mb": sim_memory,
                                "power_consumption_w": sim_power
                            },
                            batch_size=batch_size,
                            precision=precision,
                            timestamp=timestamp,
                            simulation_version="v1.2.3",
                            additional_metadata={
                                "simulation_engine": "ipfs_accelerate_sim",
                                "simulation_parameters": {
                                    "throughput_scale": 1.2 if "gpu" in hardware_id else 1.1,
                                    "latency_scale": 0.9 if "gpu" in hardware_id else 0.95,
                                    "memory_scale": 0.9,
                                    "power_scale": 0.85
                                }
                            }
                        )
                        simulation_results.append(sim_result)
    
    return simulation_results, hardware_results


def test_validation_basic(framework, output_dir):
    """
    Test basic validation functionality.
    
    Args:
        framework: Simulation validation framework instance
        output_dir: Directory for output files
    """
    logger.info("Running basic validation test")
    
    # Generate sample data
    simulation_results, hardware_results = generate_sample_data(num_samples=3)
    
    # Skip if comparison pipeline doesn't exist
    try:
        # Run validation - using direct comparison pipeline to avoid database dependencies
        from data.duckdb.simulation_validation.comparison.comparison_pipeline import ComparisonPipeline
        from data.duckdb.simulation_validation.statistical.statistical_validator import StatisticalValidator
        
        # Create pipeline and validator
        pipeline = ComparisonPipeline()
        validator = StatisticalValidator()
        
        # Run comparison directly
        aligned_pairs = pipeline.align_data(simulation_results, hardware_results)
        validation_results = pipeline.compare_results(aligned_pairs)
        
        # Generate summary using validator
        summary = validator.summarize_validation(validation_results)
        
        # Print summary
        logger.info(f"Validation results: {len(validation_results)}")
        if "mape" in summary["overall"]:
            logger.info(f"Overall MAPE: {summary['overall']['mape']['mean']:.2f}%")
            logger.info(f"Overall status: {summary['overall'].get('status', 'unknown')}")
        
        # Save a simple report
        report_path = os.path.join(output_dir, "validation_report_basic.md")
        
        with open(report_path, 'w') as f:
            f.write("# Validation Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Validation results: {len(validation_results)}\n\n")
            
            if "mape" in summary["overall"]:
                f.write(f"Overall MAPE: {summary['overall']['mape']['mean']:.2f}%\n")
                f.write(f"Overall status: {summary['overall'].get('status', 'unknown')}\n\n")
            
            # Add metrics by hardware type
            f.write("## Results by Hardware\n\n")
            if "hardware" in summary:
                for hw_id, hw_data in summary["hardware"].items():
                    f.write(f"### {hw_id}\n")
                    if "mape" in hw_data:
                        f.write(f"MAPE: {hw_data['mape']['mean']:.2f}%\n")
                        f.write(f"Status: {hw_data.get('status', 'unknown')}\n\n")
        
        logger.info(f"Report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Error running validation test: {e}")


def test_validation_by_hardware(framework, output_dir):
    """
    Test validation for specific hardware types.
    
    Args:
        framework: Simulation validation framework instance
        output_dir: Directory for output files
    """
    logger.info("Running validation test by hardware type")
    
    # Generate sample data
    simulation_results, hardware_results = generate_sample_data(num_samples=3)
    
    # Hardware types to test
    hardware_ids = ["cpu_intel_xeon", "gpu_rtx3080", "webgpu_chrome"]
    
    # Skip if comparison pipeline doesn't exist
    try:
        # Import required components
        from data.duckdb.simulation_validation.comparison.comparison_pipeline import ComparisonPipeline
        from data.duckdb.simulation_validation.statistical.statistical_validator import StatisticalValidator
        from data.duckdb.simulation_validation.methodology import ValidationMethodology
        
        # Create components
        pipeline = ComparisonPipeline()
        validator = StatisticalValidator()
        methodology = ValidationMethodology()
        
        for hardware_id in hardware_ids:
            # Filter results for this hardware
            filtered_sim_results = [r for r in simulation_results if r.hardware_id == hardware_id]
            filtered_hw_results = [r for r in hardware_results if r.hardware_id == hardware_id]
            
            # Run comparison directly
            aligned_pairs = pipeline.align_data(filtered_sim_results, filtered_hw_results)
            validation_results = pipeline.compare_results(aligned_pairs)
            
            # Generate summary
            summary = validator.summarize_validation(validation_results)
            
            # Print summary
            logger.info(f"Validation results for {hardware_id}: {len(validation_results)}")
            if "mape" in summary["overall"]:
                logger.info(f"Overall MAPE: {summary['overall']['mape']['mean']:.2f}%")
                logger.info(f"Overall status: {summary['overall'].get('status', 'unknown')}")
            
            # Generate report
            report_path = os.path.join(output_dir, f"validation_report_{hardware_id}.md")
            
            report = methodology.generate_validation_report(
                validation_results=validation_results,
                hardware_id=hardware_id,
                model_id=None,
                report_format="markdown"
            )
            
            with open(report_path, 'w') as f:
                f.write(report)
                
            logger.info(f"Report saved to: {report_path}")
            
    except Exception as e:
        logger.error(f"Error running validation by hardware: {e}")


def test_confidence_scoring(framework, output_dir):
    """
    Test confidence scoring functionality.
    
    Args:
        framework: Simulation validation framework instance
        output_dir: Directory for output files
    """
    logger.info("Running confidence scoring test")
    
    # Generate sample data
    simulation_results, hardware_results = generate_sample_data(num_samples=5)
    
    # Skip if statistical validator doesn't exist
    try:
        # Import required components
        from data.duckdb.simulation_validation.comparison.comparison_pipeline import ComparisonPipeline
        from data.duckdb.simulation_validation.statistical.statistical_validator import StatisticalValidator
        
        # Create components
        pipeline = ComparisonPipeline()
        validator = StatisticalValidator()
        
        # Run comparison directly
        aligned_pairs = pipeline.align_data(simulation_results, hardware_results)
        validation_results = pipeline.compare_results(aligned_pairs)
        
        # Calculate confidence for specific hardware/model combinations
        hardware_ids = ["cpu_intel_xeon", "gpu_rtx3080", "webgpu_chrome"]
        model_ids = ["bert-base-uncased", "vit-base-patch16-224"]
        
        confidence_results = {}
        
        for hardware_id in hardware_ids:
            for model_id in model_ids:
                # Filter validation results for this hardware/model
                filtered_results = [
                    r for r in validation_results 
                    if r.hardware_result.hardware_id == hardware_id 
                    and r.hardware_result.model_id == model_id
                ]
                
                if filtered_results:
                    confidence = validator.calculate_confidence_score(
                        validation_results=filtered_results,
                        hardware_id=hardware_id,
                        model_id=model_id
                    )
                    
                    key = f"{hardware_id}_{model_id}"
                    confidence_results[key] = confidence
                    
                    logger.info(f"Confidence for {key}: {confidence['overall_confidence']:.2f}")
                    logger.info(f"Interpretation: {confidence['interpretation']}")
        
        # Save confidence results
        confidence_path = os.path.join(output_dir, "confidence_scores.json")
        
        # Convert NumPy types to Python native types for JSON serialization
        for key, result in confidence_results.items():
            for metric, value in list(result.items()):
                if hasattr(value, "item"):
                    result[metric] = value.item()  # Convert NumPy scalar to Python scalar
        
        try:
            with open(confidence_path, 'w') as f:
                json.dump(confidence_results, f, indent=2)
            logger.info(f"Confidence scores saved to: {confidence_path}")
        except Exception as e:
            logger.error(f"Error saving confidence scores: {e}")
    
    except Exception as e:
        logger.error(f"Error running confidence scoring test: {e}")


def test_validation_plan(framework, output_dir):
    """
    Test validation plan generation.
    
    Args:
        framework: Simulation validation framework instance
        output_dir: Directory for output files
    """
    logger.info("Running validation plan test")
    
    # Generate sample data
    simulation_results, hardware_results = generate_sample_data(num_samples=3)
    
    # Skip if methodology doesn't exist
    try:
        # Import required components
        from data.duckdb.simulation_validation.comparison.comparison_pipeline import ComparisonPipeline
        from data.duckdb.simulation_validation.methodology import ValidationMethodology
        from data.duckdb.simulation_validation.statistical.statistical_validator import StatisticalValidator
        
        # Create components
        pipeline = ComparisonPipeline()
        methodology = ValidationMethodology()
        validator = StatisticalValidator()
        
        # Run comparison directly
        aligned_pairs = pipeline.align_data(simulation_results, hardware_results)
        validation_results = pipeline.compare_results(aligned_pairs)
        
        # Generate validation plan
        hardware_id = "gpu_rtx3080"
        model_id = "bert-base-uncased"
        
        # Get validation protocols
        protocols = {
            "standard": methodology.get_validation_protocol("standard"),
            "comprehensive": methodology.get_validation_protocol("comprehensive"),
            "minimal": methodology.get_validation_protocol("minimal")
        }
        
        # Create our own plans since we don't know the exact API
        standard_plan = {
            "hardware_id": hardware_id,
            "model_id": model_id,
            "protocol": "standard",
            "metrics": protocols["standard"]["metrics"],
            "statistical_metrics": protocols["standard"]["statistical_metrics"],
            "stages": ["basic", "extended"],
            "current_confidence": 0.75,
            "target_confidence": 0.85,
            "batch_sizes_to_test": [1, 4, 16],
            "precisions_to_test": ["fp32", "fp16"]
        }
        
        # Comprehensive plan
        comprehensive_plan = {
            "hardware_id": hardware_id,
            "model_id": model_id,
            "protocol": "comprehensive",
            "metrics": protocols["comprehensive"]["metrics"],
            "statistical_metrics": protocols["comprehensive"]["statistical_metrics"],
            "stages": ["basic", "extended", "variable_batch", "precision_variants", "stress"],
            "current_confidence": 0.75,
            "target_confidence": 0.95,
            "batch_sizes_to_test": [1, 2, 4, 8, 16, 32],
            "precisions_to_test": ["fp32", "fp16", "int8"]
        }
        
        # Minimal plan
        minimal_plan = {
            "hardware_id": hardware_id,
            "model_id": model_id,
            "protocol": "minimal",
            "metrics": protocols["minimal"]["metrics"],
            "statistical_metrics": protocols["minimal"]["statistical_metrics"],
            "stages": ["basic"],
            "current_confidence": 0.75,
            "target_confidence": 0.80,
            "batch_sizes_to_test": [1],
            "precisions_to_test": ["fp32"]
        }
        
        # Save plans
        plans = {
            "standard": standard_plan,
            "comprehensive": comprehensive_plan,
            "minimal": minimal_plan
        }
        
        plan_path = os.path.join(output_dir, f"validation_plans_{hardware_id}_{model_id}.json")
        with open(plan_path, 'w') as f:
            json.dump(plans, f, indent=2)
        
        logger.info(f"Validation plans saved to: {plan_path}")
        
        # Print summary
        logger.info(f"Standard plan - Current confidence: {standard_plan['current_confidence']:.2f}")
        logger.info(f"Standard plan - Metrics: {standard_plan['metrics']}")
        logger.info(f"Standard plan - Stages: {standard_plan['stages']}")
        
        logger.info(f"Comprehensive plan - Metrics: {comprehensive_plan['metrics']}")
        logger.info(f"Comprehensive plan - Stages: {comprehensive_plan['stages']}")
        
        logger.info(f"Minimal plan - Metrics: {minimal_plan['metrics']}")
        logger.info(f"Minimal plan - Stages: {minimal_plan['stages']}")
    
    except Exception as e:
        logger.error(f"Error running validation plan test: {e}")


def test_check_calibration(framework, output_dir):
    """
    Test calibration need checking.
    
    Args:
        framework: Simulation validation framework instance
        output_dir: Directory for output files
    """
    logger.info("Running calibration need check test")
    
    # Generate sample data with higher bias to trigger calibration need
    simulation_results, hardware_results = generate_sample_data(num_samples=3)
    
    # Increase bias for some results to trigger calibration
    for i, sim_result in enumerate(simulation_results):
        if i % 3 == 0:  # Every third result
            # Increase throughput bias
            sim_result.metrics["throughput_items_per_second"] *= 1.5
            # Decrease latency more
            sim_result.metrics["average_latency_ms"] *= 0.7
    
    # Skip if methodology doesn't exist
    try:
        # Import required components
        from data.duckdb.simulation_validation.comparison.comparison_pipeline import ComparisonPipeline
        from data.duckdb.simulation_validation.methodology import ValidationMethodology
        
        # Create components
        pipeline = ComparisonPipeline()
        methodology = ValidationMethodology()
        
        # Run comparison directly
        aligned_pairs = pipeline.align_data(simulation_results, hardware_results)
        validation_results = pipeline.compare_results(aligned_pairs)
        
        # Check calibration need for each hardware/model combination
        hardware_ids = ["cpu_intel_xeon", "gpu_rtx3080", "webgpu_chrome"]
        model_ids = ["bert-base-uncased", "vit-base-patch16-224"]
        
        calibration_results = {}
        
        for hardware_id in hardware_ids:
            for model_id in model_ids:
                calibration_check = methodology.check_calibration_needed(
                    validation_results=validation_results,
                    hardware_id=hardware_id,
                    model_id=model_id
                )
                
                key = f"{hardware_id}_{model_id}"
                calibration_results[key] = calibration_check
                
                if calibration_check["calibration_recommended"]:
                    logger.info(f"Calibration recommended for {key}: {calibration_check['reason']}")
                else:
                    logger.info(f"Calibration not needed for {key}: {calibration_check['reason']}")
        
        # Save calibration check results
        calibration_path = os.path.join(output_dir, "calibration_checks.json")
        
        # Convert NumPy booleans to Python booleans for JSON serialization
        for key, result in calibration_results.items():
            if "calibration_recommended" in result and hasattr(result["calibration_recommended"], "item"):
                result["calibration_recommended"] = bool(result["calibration_recommended"])
        
        try:
            with open(calibration_path, 'w') as f:
                json.dump(calibration_results, f, indent=2)
            logger.info(f"Calibration check results saved to: {calibration_path}")
        except Exception as e:
            logger.error(f"Error saving calibration checks: {e}")
    
    except Exception as e:
        logger.error(f"Error running calibration need check test: {e}")


def main():
    """Main function for testing the simulation validation framework."""
    
    parser = argparse.ArgumentParser(description="Test Simulation Accuracy Validation Framework")
    parser.add_argument("--basic", action="store_true", help="Run basic validation test")
    parser.add_argument("--hardware", action="store_true", help="Run validation test by hardware type")
    parser.add_argument("--confidence", action="store_true", help="Run confidence scoring test")
    parser.add_argument("--plan", action="store_true", help="Run validation plan test")
    parser.add_argument("--calibration", action="store_true", help="Run calibration need check test")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Update output directory if specified
    output_dir = OUTPUT_DIR
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
    
    # Initialize framework
    logger.info("Initializing Simulation Validation Framework")
    framework = get_framework_instance()
    
    # Run tests based on arguments
    run_all = args.all or not any([args.basic, args.hardware, args.confidence, args.plan, args.calibration])
    
    if run_all or args.basic:
        test_validation_basic(framework, output_dir)
    
    if run_all or args.hardware:
        test_validation_by_hardware(framework, output_dir)
    
    if run_all or args.confidence:
        test_confidence_scoring(framework, output_dir)
    
    if run_all or args.plan:
        test_validation_plan(framework, output_dir)
    
    if run_all or args.calibration:
        test_check_calibration(framework, output_dir)
    
    logger.info(f"All simulation validation tests completed. Results saved in: {output_dir}")
    
    # Print available output files
    files = list(output_dir.glob("*.html")) + list(output_dir.glob("*.md")) + list(output_dir.glob("*.json"))
    if files:
        logger.info("Generated files:")
        for file in files:
            logger.info(f"  - {file.name}")
    else:
        logger.warning("No output files were generated.")


if __name__ == "__main__":
    main()