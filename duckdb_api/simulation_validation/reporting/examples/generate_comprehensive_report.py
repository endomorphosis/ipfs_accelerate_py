#!/usr/bin/env python
"""
Example script for generating comprehensive reports using the Reporting System.

This script demonstrates how to create different types of reports with the
Simulation Validation Reporting System.
"""

import os
import sys
import json
import datetime
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from duckdb_api.simulation_validation.reporting.report_manager import ReportManager
from duckdb_api.simulation_validation.reporting.report_generator import ReportType, ReportFormat

def load_sample_results(filename: str) -> Dict[str, Any]:
    """Load sample validation results from a JSON file."""
    sample_path = os.path.join(os.path.dirname(__file__), 'sample_data', filename)
    
    if not os.path.exists(sample_path):
        # If sample file doesn't exist, generate and save sample data
        sample_data = generate_sample_results()
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        with open(sample_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        return sample_data
    
    with open(sample_path, 'r') as f:
        return json.load(f)

def generate_sample_results() -> Dict[str, Any]:
    """Generate sample validation results."""
    return {
        "overall": {
            "mape": {
                "mean": 5.32,
                "std_dev": 1.23,
                "min": 2.1,
                "max": 8.7,
                "sample_size": 50
            },
            "rmse": {
                "mean": 0.0245,
                "std_dev": 0.0078,
                "min": 0.0123,
                "max": 0.0456,
                "sample_size": 50
            },
            "correlation": {
                "mean": 0.92,
                "std_dev": 0.08,
                "min": 0.85,
                "max": 0.98,
                "sample_size": 50
            },
            "status": "pass"
        },
        "hardware_results": {
            "rtx3080": {
                "mape": {"mean": 4.21, "std_dev": 1.1},
                "rmse": {"mean": 0.0201, "std_dev": 0.005},
                "correlation": {"mean": 0.94, "std_dev": 0.06},
                "status": "pass"
            },
            "a100": {
                "mape": {"mean": 3.56, "std_dev": 0.9},
                "rmse": {"mean": 0.0189, "std_dev": 0.004},
                "correlation": {"mean": 0.96, "std_dev": 0.05},
                "status": "pass"
            }
        },
        "model_results": {
            "bert-base-uncased": {
                "mape": {"mean": 6.78, "std_dev": 1.5},
                "rmse": {"mean": 0.0312, "std_dev": 0.008},
                "correlation": {"mean": 0.91, "std_dev": 0.07},
                "status": "pass"
            },
            "vit-base-patch16-224": {
                "mape": {"mean": 5.92, "std_dev": 1.3},
                "rmse": {"mean": 0.0278, "std_dev": 0.007},
                "correlation": {"mean": 0.93, "std_dev": 0.06},
                "status": "pass"
            }
        },
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "1.0.0",
            "description": "Sample validation results"
        }
    }

def generate_sample_before_results() -> Dict[str, Any]:
    """Generate sample 'before' validation results."""
    return {
        "overall": {
            "mape": {
                "mean": 8.45,
                "std_dev": 1.85,
                "min": 3.2,
                "max": 12.1,
                "sample_size": 50
            },
            "rmse": {
                "mean": 0.0312,
                "std_dev": 0.0092,
                "min": 0.0145,
                "max": 0.0523,
                "sample_size": 50
            },
            "correlation": {
                "mean": 0.86,
                "std_dev": 0.09,
                "min": 0.74,
                "max": 0.95,
                "sample_size": 50
            },
            "status": "pass"
        },
        "hardware_results": {
            "rtx3080": {
                "mape": {"mean": 7.82, "std_dev": 1.7},
                "rmse": {"mean": 0.0287, "std_dev": 0.008},
                "correlation": {"mean": 0.88, "std_dev": 0.08},
                "status": "pass"
            },
            "a100": {
                "mape": {"mean": 6.91, "std_dev": 1.5},
                "rmse": {"mean": 0.0256, "std_dev": 0.007},
                "correlation": {"mean": 0.91, "std_dev": 0.07},
                "status": "pass"
            }
        },
        "model_results": {
            "bert-base-uncased": {
                "mape": {"mean": 9.12, "std_dev": 1.9},
                "rmse": {"mean": 0.0341, "std_dev": 0.009},
                "correlation": {"mean": 0.85, "std_dev": 0.08},
                "status": "pass"
            },
            "vit-base-patch16-224": {
                "mape": {"mean": 8.67, "std_dev": 1.8},
                "rmse": {"mean": 0.0321, "std_dev": 0.008},
                "correlation": {"mean": 0.87, "std_dev": 0.07},
                "status": "pass"
            }
        },
        "metadata": {
            "timestamp": (datetime.datetime.now() - datetime.timedelta(days=7)).isoformat(),
            "version": "0.9.0",
            "description": "Sample 'before' validation results"
        }
    }

def main():
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating reports in {output_dir}")
    
    # Create archive directory
    archive_dir = os.path.join(output_dir, 'archives')
    os.makedirs(archive_dir, exist_ok=True)
    
    # Create a report manager
    manager = ReportManager(
        output_dir=output_dir,
        archive_dir=archive_dir,
        default_format=ReportFormat.HTML,
        company_name="Example Corp"
    )
    
    # Load sample validation results
    validation_results = load_sample_results('sample_results.json')
    validation_results_before = generate_sample_before_results()
    
    # Generate different types of reports
    
    # 1. Comprehensive Report (HTML)
    comprehensive_report = manager.generate_report(
        validation_results=validation_results,
        report_type=ReportType.COMPREHENSIVE_REPORT,
        output_format=ReportFormat.HTML,
        title="Comprehensive Validation Report",
        description="Complete validation results for all hardware and models"
    )
    print(f"Generated comprehensive report (HTML): {os.path.basename(comprehensive_report['path'])}")
    
    # 2. Comprehensive Report (Markdown)
    comprehensive_md_report = manager.generate_report(
        validation_results=validation_results,
        report_type=ReportType.COMPREHENSIVE_REPORT,
        output_format=ReportFormat.MARKDOWN,
        title="Comprehensive Validation Report",
        description="Complete validation results for all hardware and models"
    )
    print(f"Generated comprehensive report (Markdown): {os.path.basename(comprehensive_md_report['path'])}")
    
    # 3. Executive Summary (HTML)
    executive_report = manager.generate_report(
        validation_results=validation_results,
        report_type=ReportType.EXECUTIVE_SUMMARY,
        output_format=ReportFormat.HTML,
        title="Executive Summary",
        description="High-level summary for executive review"
    )
    print(f"Generated executive summary (HTML): {os.path.basename(executive_report['path'])}")
    
    # 4. Technical Report (HTML)
    technical_report = manager.generate_report(
        validation_results=validation_results,
        report_type=ReportType.TECHNICAL_REPORT,
        output_format=ReportFormat.HTML,
        title="Technical Validation Report",
        description="Detailed technical analysis of validation results"
    )
    print(f"Generated technical report (HTML): {os.path.basename(technical_report['path'])}")
    
    # 5. Comparative Report (HTML)
    comparative_report = manager.generate_report(
        validation_results=validation_results,
        report_type=ReportType.COMPARATIVE_REPORT,
        output_format=ReportFormat.HTML,
        title="Comparative Validation Report",
        description="Comparison of validation results before and after calibration",
        comparative_data={
            "validation_results_before": validation_results_before
        }
    )
    print(f"Generated comparative report (HTML): {os.path.basename(comparative_report['path'])}")
    
    # Generate report catalog
    catalog_path = os.path.join(output_dir, manager.report_catalog_file)
    print(f"Generated report catalog: {catalog_path}")
    
    print("\nAll reports generated successfully.")
    print(f"View report catalog at: file://{os.path.abspath(catalog_path)}")

if __name__ == "__main__":
    main()