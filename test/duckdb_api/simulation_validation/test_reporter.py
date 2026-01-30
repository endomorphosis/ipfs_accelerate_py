#!/usr/bin/env python3
"""
Simple test script for the validation reporter.
"""

import os
import sys
import datetime
import random
import json
from typing import Dict, List, Any, Optional, Union, Tuple

# Add the parent directory to sys.path
sys.path.append('/home/barberb/ipfs_accelerate_py/test')

# Import necessary classes
from data.duckdb.simulation_validation.core.base import (
    SimulationResult,
    HardwareResult,
    ValidationResult
)
from data.duckdb.simulation_validation.visualization.validation_reporter import ValidationReporterImpl

def create_simple_test_data():
    """Create simple test data for the reporter."""
    # Create a simulation result
    sim_result = SimulationResult(
        model_id="test_model",
        hardware_id="test_hardware",
        metrics={
            "throughput_items_per_second": 100.0,
            "average_latency_ms": 50.0,
            "memory_peak_mb": 1000.0
        },
        batch_size=1,
        precision="fp32",
        simulation_version="v1.0"
    )
    
    # Create a hardware result
    hw_result = HardwareResult(
        model_id="test_model",
        hardware_id="test_hardware",
        metrics={
            "throughput_items_per_second": 110.0,
            "average_latency_ms": 45.0,
            "memory_peak_mb": 950.0
        },
        batch_size=1,
        precision="fp32",
        hardware_details={"type": "test_hardware"},
        test_environment={"temperature": 25.0}
    )
    
    # Create a validation result
    validation_result = ValidationResult(
        simulation_result=sim_result,
        hardware_result=hw_result,
        metrics_comparison={
            "throughput_items_per_second": {
                "hardware_value": 110.0,
                "simulation_value": 100.0,
                "absolute_error": 10.0,
                "relative_error": 0.09,
                "mape": 9.0
            },
            "average_latency_ms": {
                "hardware_value": 45.0,
                "simulation_value": 50.0,
                "absolute_error": 5.0,
                "relative_error": 0.11,
                "mape": 11.0
            },
            "memory_peak_mb": {
                "hardware_value": 950.0,
                "simulation_value": 1000.0,
                "absolute_error": 50.0,
                "relative_error": 0.05,
                "mape": 5.0
            }
        },
        validation_timestamp=datetime.datetime.now().isoformat(),
        validation_version="v1.0"
    )
    
    return [validation_result]

def main():
    """Run the validation reporter test."""
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test data
    print("Creating test data...")
    validation_results = create_simple_test_data()
    
    # Create reporter
    print("Initializing reporter...")
    reporter = ValidationReporterImpl()
    
    # Generate a simple report
    print("Generating simple report...")
    try:
        html_report = reporter.generate_report(
            validation_results=validation_results,
            format="html",
            include_visualizations=True
        )
        
        # Save the report
        html_path = os.path.join(output_dir, "reporter_test.html")
        with open(html_path, "w") as f:
            f.write(html_report)
        print(f"HTML report saved to {html_path}")
        
        # Save as JSON
        json_path = os.path.join(output_dir, "reporter_test.json")
        reporter.export_report(
            validation_results=validation_results,
            output_path=json_path,
            format="json"
        )
        print(f"JSON report saved to {json_path}")
        
        # Save as markdown
        md_path = os.path.join(output_dir, "reporter_test.md")
        reporter.export_report(
            validation_results=validation_results,
            output_path=md_path,
            format="markdown"
        )
        print(f"Markdown report saved to {md_path}")
    except Exception as e:
        print(f"Error generating report: {e}")
        raise

if __name__ == "__main__":
    main()