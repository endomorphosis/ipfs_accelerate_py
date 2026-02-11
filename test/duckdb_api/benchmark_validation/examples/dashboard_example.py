#!/usr/bin/env python3
"""
Example for using the Validation Dashboard component of the Benchmark Validation System.

This example demonstrates how to create, update, and manage dashboards for validation results.
It showcases various dashboard features, including:

1. Creating basic dashboards with validation results
2. Creating comparison dashboards with multiple sets of validation results
3. Updating existing dashboards with new data
4. Registering dashboards with the monitoring dashboard system
5. Listing available dashboards
6. Exporting dashboards to different formats
7. Managing dashboards (deletion, etc.)

Usage:
    python -m duckdb_api.benchmark_validation.examples.dashboard_example

    The script will display a menu with different options to demonstrate various
    dashboard features. You can run each example individually or run all examples
    in sequence.

Note:
    This example creates output in the ./output/dashboards/ directory.
    Some features require the Advanced Visualization System and the
    Distributed Testing Monitoring Dashboard to be available.
"""

import os
import sys
import logging
import uuid
import random
import datetime
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dashboard_example")

# Add parent directory to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Import base classes
from data.duckdb.benchmark_validation.core.base import (
    ValidationResult,
    BenchmarkResult,
    ValidationLevel,
    ValidationStatus,
    BenchmarkType
)

# Import dashboard component
from data.duckdb.benchmark_validation.visualization.dashboard import ValidationDashboard

def create_sample_validation_results(count: int = 50) -> List[ValidationResult]:
    """
    Create sample validation results for testing the dashboard.
    
    Args:
        count: Number of validation results to create
        
    Returns:
        List of sample validation results
    """
    validation_results = []
    
    # Define some sample data
    model_ids = ["bert-base-uncased", "vit-base-patch16-224", "whisper-small", "t5-base", "clip-vit-base-patch32"]
    hardware_ids = ["cpu_intel_xeon", "gpu_nvidia_rtx3080", "gpu_nvidia_rtx4090", "webgpu_chrome", "webgpu_firefox"]
    benchmark_types = [BenchmarkType.LATENCY, BenchmarkType.THROUGHPUT, BenchmarkType.MEMORY]
    validation_levels = [ValidationLevel.BASIC, ValidationLevel.ADVANCED, ValidationLevel.COMPREHENSIVE]
    validation_statuses = [ValidationStatus.VALID, ValidationStatus.INVALID, ValidationStatus.WARNING, ValidationStatus.ERROR]
    
    status_weights = {
        ValidationStatus.VALID: 0.7,     # 70% valid
        ValidationStatus.WARNING: 0.15,  # 15% warning
        ValidationStatus.INVALID: 0.1,   # 10% invalid
        ValidationStatus.ERROR: 0.05     # 5% error
    }
    
    # Generate random validation results
    for i in range(count):
        # Generate random model and hardware
        model_id = random.choice(model_ids)
        hardware_id = random.choice(hardware_ids)
        benchmark_type = random.choice(benchmark_types)
        validation_level = random.choice(validation_levels)
        
        # Generate random status based on weights
        status = random.choices(
            list(status_weights.keys()), 
            weights=list(status_weights.values()), 
            k=1
        )[0]
        
        # Generate confidence score based on status
        if status == ValidationStatus.VALID:
            confidence_score = random.uniform(0.8, 1.0)
        elif status == ValidationStatus.WARNING:
            confidence_score = random.uniform(0.5, 0.8)
        elif status == ValidationStatus.INVALID:
            confidence_score = random.uniform(0.2, 0.5)
        else:  # ERROR
            confidence_score = random.uniform(0.0, 0.2)
        
        # Create benchmark result
        benchmark_result = BenchmarkResult(
            id=f"bench_{uuid.uuid4().hex[:8]}",
            model_id=model_id,
            hardware_id=hardware_id,
            benchmark_type=benchmark_type,
            metrics={
                "latency_ms": random.uniform(10, 1000),
                "throughput_items_per_sec": random.uniform(1, 100),
                "memory_mb": random.uniform(100, 5000)
            },
            metadata={
                "batch_size": random.choice([1, 2, 4, 8, 16, 32]),
                "precision": random.choice(["fp32", "fp16", "int8", "int4"]),
                "runtime": random.choice(["pytorch", "tensorflow", "onnx"])
            }
        )
        
        # Generate random issues based on status
        issues = []
        if status != ValidationStatus.VALID:
            num_issues = random.randint(1, 3)
            issue_types = ["performance", "accuracy", "reliability", "hardware_compatibility"]
            
            for j in range(num_issues):
                issue_type = random.choice(issue_types)
                issues.append({
                    "type": issue_type,
                    "severity": random.choice(["low", "medium", "high"]),
                    "description": f"Sample {issue_type} issue with {model_id} on {hardware_id}"
                })
        
        # Generate random recommendations based on issues
        recommendations = []
        if issues:
            for issue in issues:
                recommendations.append({
                    "type": f"fix_{issue['type']}",
                    "priority": random.choice(["low", "medium", "high"]),
                    "description": f"Recommendation to fix {issue['type']} issue with {model_id} on {hardware_id}"
                })
        
        # Create validation result
        validation_result = ValidationResult(
            id=f"val_{uuid.uuid4().hex[:8]}",
            benchmark_result=benchmark_result,
            validation_level=validation_level,
            status=status,
            confidence_score=confidence_score,
            timestamp=datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 30)),
            issues=issues,
            recommendations=recommendations
        )
        
        validation_results.append(validation_result)
    
    return validation_results

def create_basic_dashboard():
    """Create a basic validation dashboard with sample data."""
    # Create dashboard instance
    dashboard = ValidationDashboard(config={
        "output_directory": "output",
        "dashboard_directory": "dashboards",
        "theme": "light"
    })
    
    # Generate sample validation results
    validation_results = create_sample_validation_results(count=100)
    logger.info(f"Generated {len(validation_results)} sample validation results")
    
    # Create dashboard
    dashboard_path = dashboard.create_dashboard(
        validation_results=validation_results,
        dashboard_name="sample_validation_dashboard",
        dashboard_title="Sample Validation Dashboard",
        dashboard_description="A dashboard with sample validation results for demonstration"
    )
    
    if dashboard_path:
        logger.info(f"Created dashboard at {dashboard_path}")
        
        # Display URL
        dashboard_url = dashboard.get_dashboard_url(
            dashboard_name="sample_validation_dashboard",
            base_url="http://localhost:8080"
        )
        logger.info(f"Dashboard URL: {dashboard_url}")
        
        # Export dashboard to different formats
        html_path = dashboard.export_dashboard(
            dashboard_name="sample_validation_dashboard",
            export_format="html"
        )
        logger.info(f"Exported dashboard as HTML to {html_path}")
        
        markdown_path = dashboard.export_dashboard(
            dashboard_name="sample_validation_dashboard",
            export_format="markdown"
        )
        logger.info(f"Exported dashboard as Markdown to {markdown_path}")
    else:
        logger.error("Failed to create dashboard")

def create_comparison_dashboard():
    """Create a comparison dashboard with multiple sets of sample data."""
    # Create dashboard instance
    dashboard = ValidationDashboard(config={
        "output_directory": "output",
        "dashboard_directory": "dashboards",
        "theme": "light"
    })
    
    # Generate different sets of validation results
    validation_sets = {
        "baseline": create_sample_validation_results(count=80),
        "experiment_1": create_sample_validation_results(count=80),
        "experiment_2": create_sample_validation_results(count=80)
    }
    
    logger.info(f"Generated validation result sets: {', '.join(validation_sets.keys())}")
    
    # Create comparison dashboard
    dashboard_path = dashboard.create_comparison_dashboard(
        validation_results_sets=validation_sets,
        dashboard_name="validation_comparison_dashboard",
        dashboard_title="Validation Comparison Dashboard",
        dashboard_description="Comparison of validation results across different experiments"
    )
    
    if dashboard_path:
        logger.info(f"Created comparison dashboard at {dashboard_path}")
        
        # Display URL
        dashboard_url = dashboard.get_dashboard_url(
            dashboard_name="validation_comparison_dashboard",
            base_url="http://localhost:8080"
        )
        logger.info(f"Comparison Dashboard URL: {dashboard_url}")
    else:
        logger.error("Failed to create comparison dashboard")

def update_dashboard():
    """Update an existing dashboard with new validation results."""
    # Create dashboard instance
    dashboard = ValidationDashboard(config={
        "output_directory": "output",
        "dashboard_directory": "dashboards"
    })
    
    # Check if dashboard exists
    dashboards = dashboard.list_dashboards()
    dashboard_exists = any(d.get("name") == "sample_validation_dashboard" for d in dashboards)
    
    if not dashboard_exists:
        logger.error("Sample dashboard doesn't exist. Creating it first.")
        create_basic_dashboard()
    
    # Generate new validation results
    new_validation_results = create_sample_validation_results(count=50)
    logger.info(f"Generated {len(new_validation_results)} new validation results for update")
    
    # Update the dashboard
    updated_path = dashboard.update_dashboard(
        dashboard_name="sample_validation_dashboard",
        validation_results=new_validation_results,
        dashboard_title="Updated Sample Validation Dashboard",
        dashboard_description="An updated dashboard with new validation results"
    )
    
    if updated_path:
        logger.info(f"Updated dashboard at {updated_path}")
    else:
        logger.error("Failed to update dashboard")

def register_with_monitoring():
    """Register a dashboard with the monitoring dashboard."""
    # Create dashboard instance
    dashboard = ValidationDashboard(config={
        "output_directory": "output",
        "dashboard_directory": "dashboards",
        "monitoring_integration": True
    })
    
    # Check if dashboard exists
    dashboards = dashboard.list_dashboards()
    dashboard_exists = any(d.get("name") == "sample_validation_dashboard" for d in dashboards)
    
    if not dashboard_exists:
        logger.error("Sample dashboard doesn't exist. Creating it first.")
        create_basic_dashboard()
    
    # Register with monitoring dashboard
    success = dashboard.register_with_monitoring_dashboard(
        dashboard_name="sample_validation_dashboard",
        page="validation",
        position="below"
    )
    
    if success:
        logger.info("Dashboard registered with monitoring system")
    else:
        logger.error("Failed to register dashboard with monitoring system")
        
        # Check for availability of monitoring integration
        try:
            from data.duckdb.distributed_testing.dashboard.monitoring_dashboard_visualization_integration import (
                VisualizationDashboardIntegration
            )
            logger.info("Monitoring integration is available but registration failed")
        except ImportError:
            logger.error("Monitoring integration is not available. Check if distributed_testing module is installed")

def list_dashboards():
    """List all available dashboards."""
    # Create dashboard instance
    dashboard = ValidationDashboard()
    
    # List dashboards
    dashboards = dashboard.list_dashboards()
    
    if dashboards:
        logger.info(f"Found {len(dashboards)} dashboards:")
        for i, dash in enumerate(dashboards):
            logger.info(f"  {i+1}. {dash.get('name')} - {dash.get('title')}")
            logger.info(f"     Path: {dash.get('path')}")
            logger.info(f"     Source: {dash.get('source')}")
            logger.info(f"     Last Updated: {dash.get('updated_at', 'Unknown')}")
    else:
        logger.info("No dashboards found")

def clean_up():
    """Clean up by deleting sample dashboards."""
    # Create dashboard instance
    dashboard = ValidationDashboard()
    
    # List of sample dashboards to delete
    sample_dashboards = ["sample_validation_dashboard", "validation_comparison_dashboard"]
    
    for dash_name in sample_dashboards:
        success = dashboard.delete_dashboard(dash_name)
        if success:
            logger.info(f"Deleted dashboard '{dash_name}'")
        else:
            logger.info(f"Dashboard '{dash_name}' not found or couldn't be deleted")

def main():
    """Main function to demonstrate dashboard functionality."""
    # Ensure output directories exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/dashboards", exist_ok=True)
    
    # Display available options
    print("\n=== Benchmark Validation Dashboard Example ===")
    print("1. Create a basic dashboard")
    print("2. Create a comparison dashboard")
    print("3. Update an existing dashboard")
    print("4. Register with monitoring dashboard")
    print("5. List all dashboards")
    print("6. Clean up (delete sample dashboards)")
    print("7. Run all examples")
    print("0. Exit")
    
    try:
        choice = int(input("\nEnter your choice (0-7): "))
        
        if choice == 1:
            create_basic_dashboard()
        elif choice == 2:
            create_comparison_dashboard()
        elif choice == 3:
            update_dashboard()
        elif choice == 4:
            register_with_monitoring()
        elif choice == 5:
            list_dashboards()
        elif choice == 6:
            clean_up()
        elif choice == 7:
            print("\n--- Running all examples ---")
            create_basic_dashboard()
            create_comparison_dashboard()
            update_dashboard()
            register_with_monitoring()
            list_dashboards()
        elif choice == 0:
            print("Exiting...")
        else:
            print("Invalid choice. Please enter a number between 0 and 7.")
    
    except ValueError:
        print("Invalid input. Please enter a number.")
    
    print("\nExample completed.")

if __name__ == "__main__":
    main()