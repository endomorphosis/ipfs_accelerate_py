#!/usr/bin/env python3
"""
Test script for the enhanced reporting functionality in the Simulation Accuracy and Validation Framework.

This script creates a set of simulated validation results and then uses the enhanced
reporting functionality to generate reports in various formats, testing the new
executive summary, statistical analysis, and visualization capabilities.
"""

import os
import sys
import datetime
import random
import json
import argparse
from typing import Dict, List, Any, Optional, Union, Tuple

# Import base classes
sys.path.append('/home/barberb/ipfs_accelerate_py/test')
from data.duckdb.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)

# Import the reporter
from data.duckdb.simulation_validation.visualization.validation_reporter import ValidationReporterImpl


def create_test_data(
    num_hardware_types: int = 3,
    num_model_types: int = 4,
    num_results_per_combination: int = 2,
    seed: int = 42
) -> List[ValidationResult]:
    """
    Create test data for the reporting system.
    
    Args:
        num_hardware_types: Number of hardware types to simulate
        num_model_types: Number of model types to simulate
        num_results_per_combination: Number of results per hardware-model combination
        seed: Random seed for reproducibility
        
    Returns:
        List of ValidationResult objects
    """
    random.seed(seed)
    
    # Define hardware types
    hardware_types = [f"hardware_{i}" for i in range(num_hardware_types)]
    
    # Define model types
    model_types = [f"model_{i}" for i in range(num_model_types)]
    
    # Define batch sizes and precisions
    batch_sizes = [1, 2, 4, 8, 16]
    precisions = ["fp32", "fp16", "int8"]
    
    # Define metrics
    metrics = ["throughput_items_per_second", "average_latency_ms", "memory_peak_mb", "power_consumption_w"]
    
    # Define base values for each metric
    base_values = {
        "throughput_items_per_second": 100.0,
        "average_latency_ms": 50.0,
        "memory_peak_mb": 1000.0,
        "power_consumption_w": 150.0
    }
    
    # Define hardware-specific multipliers
    hardware_multipliers = {}
    for hw in hardware_types:
        hardware_multipliers[hw] = {
            "throughput_items_per_second": random.uniform(0.8, 1.2),
            "average_latency_ms": random.uniform(0.8, 1.2),
            "memory_peak_mb": random.uniform(0.8, 1.2),
            "power_consumption_w": random.uniform(0.8, 1.2)
        }
    
    # Define model-specific multipliers
    model_multipliers = {}
    for model in model_types:
        model_multipliers[model] = {
            "throughput_items_per_second": random.uniform(0.8, 1.2),
            "average_latency_ms": random.uniform(0.8, 1.2),
            "memory_peak_mb": random.uniform(0.8, 1.2),
            "power_consumption_w": random.uniform(0.8, 1.2)
        }
    
    # Generate validation results
    validation_results = []
    
    timestamp_base = datetime.datetime.now() - datetime.timedelta(days=30)
    timestamp_increment = datetime.timedelta(hours=1)
    
    for hw_idx, hardware_type in enumerate(hardware_types):
        for model_idx, model_type in enumerate(model_types):
            for i in range(num_results_per_combination):
                # Choose batch size and precision
                batch_size = random.choice(batch_sizes)
                precision = random.choice(precisions)
                
                # Calculate timestamp
                timestamp = timestamp_base + timestamp_increment * (hw_idx * num_model_types * num_results_per_combination + 
                                                                  model_idx * num_results_per_combination + i)
                timestamp_str = timestamp.isoformat()
                
                # Generate hardware and simulation metrics
                hw_metrics = {}
                sim_metrics = {}
                metrics_comparison = {}
                
                for metric in metrics:
                    # Calculate true hardware value
                    hw_value = (base_values[metric] * 
                               hardware_multipliers[hardware_type][metric] * 
                               model_multipliers[model_type][metric] * 
                               (batch_size / 4.0))
                    
                    # Add some random variation
                    hw_value *= random.uniform(0.95, 1.05)
                    
                    # Calculate simulation value with error
                    error_pct = random.uniform(-20.0, 20.0)  # -20% to +20% error
                    sim_value = hw_value * (1.0 + error_pct / 100.0)
                    
                    # Store metrics
                    hw_metrics[metric] = hw_value
                    sim_metrics[metric] = sim_value
                    
                    # Calculate error metrics
                    abs_error = abs(sim_value - hw_value)
                    rel_error = abs_error / hw_value
                    mape = rel_error * 100.0
                    
                    metrics_comparison[metric] = {
                        "hardware_value": hw_value,
                        "simulation_value": sim_value,
                        "absolute_error": abs_error,
                        "relative_error": rel_error,
                        "mape": mape
                    }
                
                # Create hardware result
                hw_result = HardwareResult(
                    model_id=model_type,
                    hardware_id=hardware_type,
                    metrics=hw_metrics,
                    batch_size=batch_size,
                    precision=precision,
                    timestamp=timestamp_str,
                    hardware_details={"type": hardware_type},
                    test_environment={"temperature": random.uniform(20.0, 30.0)}
                )
                
                # Create simulation result
                sim_result = SimulationResult(
                    model_id=model_type,
                    hardware_id=hardware_type,
                    metrics=sim_metrics,
                    batch_size=batch_size,
                    precision=precision,
                    timestamp=timestamp_str,
                    simulation_version="v1.0"
                )
                
                # Create validation result
                validation_result = ValidationResult(
                    simulation_result=sim_result,
                    hardware_result=hw_result,
                    metrics_comparison=metrics_comparison,
                    validation_timestamp=timestamp_str,
                    validation_version="v1.0"
                )
                
                validation_results.append(validation_result)
    
    return validation_results


def test_enhanced_reporting():
    """Test the enhanced reporting functionality."""
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.getcwd(), "test_reports")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test data
    print("Creating test data...")
    validation_results = create_test_data(
        num_hardware_types=3,
        num_model_types=4,
        num_results_per_combination=3
    )
    print(f"Created {len(validation_results)} validation results.")
    
    # Create reporter
    print("Initializing reporter...")
    reporter = ValidationReporterImpl()
    
    # Generate reports in various formats
    print("Generating HTML report with executive summary and visualizations...")
    html_path = os.path.join(output_dir, "enhanced_report.html")
    reporter.export_report(
        validation_results=validation_results,
        output_path=html_path,
        format="html",
        include_visualizations=True,
        include_executive_summary=True,
        include_statistical_analysis=True,
        custom_title="Enhanced Simulation Validation Report"
    )
    print(f"HTML report saved to {html_path}")
    
    print("Generating Markdown report...")
    md_path = os.path.join(output_dir, "enhanced_report.md")
    reporter.export_report(
        validation_results=validation_results,
        output_path=md_path,
        format="markdown",
        include_executive_summary=True
    )
    print(f"Markdown report saved to {md_path}")
    
    print("Generating JSON report...")
    json_path = os.path.join(output_dir, "enhanced_report.json")
    reporter.export_report(
        validation_results=validation_results,
        output_path=json_path,
        format="json"
    )
    print(f"JSON report saved to {json_path}")
    
    print("Generating CSV report...")
    csv_path = os.path.join(output_dir, "enhanced_report.csv")
    reporter.export_report(
        validation_results=validation_results,
        output_path=csv_path,
        format="csv"
    )
    print(f"CSV report saved to {csv_path}")
    
    # Try to generate PDF if libraries are available
    try:
        import weasyprint
        print("Generating PDF report...")
        pdf_path = os.path.join(output_dir, "enhanced_report.pdf")
        reporter.export_report(
            validation_results=validation_results,
            output_path=pdf_path,
            format="pdf",
            include_visualizations=True,
            include_executive_summary=True,
            include_statistical_analysis=True
        )
        print(f"PDF report saved to {pdf_path}")
    except ImportError:
        print("weasyprint not available, skipping PDF generation.")
    
    # Test filtered reports
    print("Generating hardware-filtered report...")
    hw_filtered_path = os.path.join(output_dir, "hardware_filtered_report.html")
    reporter.export_report(
        validation_results=validation_results,
        output_path=hw_filtered_path,
        format="html",
        include_visualizations=True,
        hardware_filter="hardware_0",
        custom_title="Hardware-Filtered Report (hardware_0)"
    )
    print(f"Hardware-filtered report saved to {hw_filtered_path}")
    
    print("Generating model-filtered report...")
    model_filtered_path = os.path.join(output_dir, "model_filtered_report.html")
    reporter.export_report(
        validation_results=validation_results,
        output_path=model_filtered_path,
        format="html",
        include_visualizations=True,
        model_filter="model_0",
        custom_title="Model-Filtered Report (model_0)"
    )
    print(f"Model-filtered report saved to {model_filtered_path}")
    
    # Test custom section filtering
    print("Generating report with only executive summary and recommendations...")
    sections_filtered_path = os.path.join(output_dir, "sections_filtered_report.html")
    reporter.export_report(
        validation_results=validation_results,
        output_path=sections_filtered_path,
        format="html",
        include_visualizations=True,
        include_sections=["executive_summary", "recommendations"],
        custom_title="Sections-Filtered Report (Executive Summary & Recommendations)"
    )
    print(f"Sections-filtered report saved to {sections_filtered_path}")
    
    print("All reports generated successfully.")


def test_specialized_reports():
    """Test specialized report generation with different configurations."""
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "test_reports/specialized")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test data
    print("Creating test data for specialized reports...")
    validation_results = create_test_data(
        num_hardware_types=4,
        num_model_types=5,
        num_results_per_combination=4,
        seed=43  # Different seed for variety
    )
    print(f"Created {len(validation_results)} validation results.")
    
    # Create reporter with custom configuration
    print("Initializing reporter with custom configuration...")
    custom_config = {
        "report_theme": "dark",
        "visualization_width": 1000,
        "visualization_height": 600,
        "report_title_template": "Advanced Simulation Analysis Report - {timestamp}",
        "report_footer": "Simulation Accuracy and Validation Framework - Confidential",
        "report_sections": [
            "executive_summary",
            "statistical_analysis",
            "hardware_comparison",
            "recommendations"
        ]
    }
    reporter = ValidationReporterImpl(config=custom_config)
    
    # Generate time-series analysis report
    print("Generating time-series analysis report...")
    # Sort validation results by timestamp to simulate time series data
    sorted_results = sorted(validation_results, key=lambda vr: vr.validation_timestamp)
    
    time_series_path = os.path.join(output_dir, "time_series_report.html")
    reporter.export_report(
        validation_results=sorted_results,
        output_path=time_series_path,
        format="html",
        include_visualizations=True,
        include_executive_summary=True,
        include_statistical_analysis=True,
        custom_title="Time-Series Simulation Analysis Report"
    )
    print(f"Time-series report saved to {time_series_path}")
    
    # Generate hardware-focused report
    print("Generating hardware-focused report...")
    hardware_focus_path = os.path.join(output_dir, "hardware_focus_report.html")
    reporter.export_report(
        validation_results=validation_results,
        output_path=hardware_focus_path,
        format="html",
        include_visualizations=True,
        include_sections=["executive_summary", "hardware_comparison", "recommendations"],
        custom_title="Hardware Performance Analysis Report"
    )
    print(f"Hardware-focused report saved to {hardware_focus_path}")
    
    # Generate minimal executive summary report
    print("Generating minimal executive summary report...")
    exec_summary_path = os.path.join(output_dir, "executive_summary_report.html")
    reporter.export_report(
        validation_results=validation_results,
        output_path=exec_summary_path,
        format="html",
        include_visualizations=False,
        include_sections=["executive_summary", "recommendations"],
        custom_title="Executive Summary Report"
    )
    print(f"Executive summary report saved to {exec_summary_path}")
    
    # Generate statistical analysis focused report
    print("Generating statistical analysis report...")
    stats_path = os.path.join(output_dir, "statistical_analysis_report.html")
    reporter.export_report(
        validation_results=validation_results,
        output_path=stats_path,
        format="html",
        include_visualizations=True,
        include_sections=["executive_summary", "statistical_analysis"],
        custom_title="Statistical Validation Analysis Report"
    )
    print(f"Statistical analysis report saved to {stats_path}")
    
    print("All specialized reports generated successfully.")

def test_comparative_analysis():
    """Test comparative analysis between different validation result sets."""
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "test_reports/comparative")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create two different sets of validation results
    print("Creating baseline validation results...")
    baseline_results = create_test_data(
        num_hardware_types=3,
        num_model_types=4,
        num_results_per_combination=2,
        seed=44  # Different seed
    )
    
    # Create improved results with better accuracy
    print("Creating improved validation results...")
    improved_results = []
    
    # Use the same structure but improve the MAPE values
    for vr in baseline_results:
        # Create a copy of the validation result
        sim_result = SimulationResult(
            model_id=vr.simulation_result.model_id,
            hardware_id=vr.simulation_result.hardware_id,
            metrics=dict(vr.simulation_result.metrics),  # Copy metrics
            batch_size=vr.simulation_result.batch_size,
            precision=vr.simulation_result.precision,
            simulation_version="v1.1"  # New version
        )
        
        hw_result = HardwareResult(
            model_id=vr.hardware_result.model_id,
            hardware_id=vr.hardware_result.hardware_id,
            metrics=dict(vr.hardware_result.metrics),  # Copy metrics
            batch_size=vr.hardware_result.batch_size,
            precision=vr.hardware_result.precision,
            hardware_details=dict(vr.hardware_result.hardware_details),
            test_environment=dict(vr.hardware_result.test_environment)
        )
        
        # Improve the metrics_comparison with better MAPE values
        metrics_comparison = {}
        for metric, comparison in vr.metrics_comparison.items():
            # Reduce MAPE by 40-60%
            improvement_factor = random.uniform(0.4, 0.6)
            new_mape = comparison["mape"] * improvement_factor
            
            # Calculate new values based on improved MAPE
            hw_value = comparison["hardware_value"]
            new_sim_value = hw_value * (1 + (new_mape / 100.0) * (1 if hw_value < comparison["simulation_value"] else -1))
            new_abs_error = abs(new_sim_value - hw_value)
            new_rel_error = new_abs_error / hw_value
            
            metrics_comparison[metric] = {
                "hardware_value": hw_value,
                "simulation_value": new_sim_value,
                "absolute_error": new_abs_error,
                "relative_error": new_rel_error,
                "mape": new_mape
            }
        
        # Create a new validation result with improved metrics
        improved_result = ValidationResult(
            simulation_result=sim_result,
            hardware_result=hw_result,
            metrics_comparison=metrics_comparison,
            validation_timestamp=(datetime.datetime.now() - datetime.timedelta(days=7)).isoformat(),
            validation_version="v1.1"
        )
        
        improved_results.append(improved_result)
    
    # Create a reporter
    reporter = ValidationReporterImpl()
    
    # Generate reports for each set
    print("Generating baseline report...")
    baseline_path = os.path.join(output_dir, "baseline_report.html")
    reporter.export_report(
        validation_results=baseline_results,
        output_path=baseline_path,
        format="html",
        include_visualizations=True,
        custom_title="Baseline Validation Report (v1.0)"
    )
    print(f"Baseline report saved to {baseline_path}")
    
    print("Generating improved report...")
    improved_path = os.path.join(output_dir, "improved_report.html")
    reporter.export_report(
        validation_results=improved_results,
        output_path=improved_path,
        format="html",
        include_visualizations=True,
        custom_title="Improved Validation Report (v1.1)"
    )
    print(f"Improved report saved to {improved_path}")
    
    # Generate comparative report by combining both sets
    print("Generating comparative report...")
    # Add version information to distinguish the sets
    for vr in baseline_results:
        vr.additional_metrics = {"version": "v1.0", "group": "baseline"}
    
    for vr in improved_results:
        vr.additional_metrics = {"version": "v1.1", "group": "improved"}
    
    combined_results = baseline_results + improved_results
    
    comparative_path = os.path.join(output_dir, "comparative_report.html")
    reporter.export_report(
        validation_results=combined_results,
        output_path=comparative_path,
        format="html",
        include_visualizations=True,
        custom_title="Comparative Validation Analysis (v1.0 vs v1.1)"
    )
    print(f"Comparative report saved to {comparative_path}")
    
    print("All comparative reports generated successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test enhanced reporting functionality")
    parser.add_argument("--output-dir", type=str, default="test_reports",
                      help="Directory to save test reports")
    parser.add_argument("--test-type", type=str, choices=["basic", "specialized", "comparative", "all"], 
                      default="basic", help="Type of test to run")
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the appropriate test
    if args.test_type == "basic" or args.test_type == "all":
        test_enhanced_reporting()
        
    if args.test_type == "specialized" or args.test_type == "all":
        test_specialized_reports()
        
    if args.test_type == "comparative" or args.test_type == "all":
        test_comparative_analysis()