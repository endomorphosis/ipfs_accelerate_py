#\!/usr/bin/env python3
"""
Test script for the ValidationReporter implementation.

This script demonstrates how to use the ValidationReporter to generate reports
in different formats.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for module imports
parent_dir = str(Path(__file__).resolve().parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import framework components
from data.duckdb.simulation_validation.visualization.validation_reporter import ValidationReporterImpl
from data.duckdb.simulation_validation.test_validator import generate_sample_data
from data.duckdb.simulation_validation.comparison.comparison_pipeline import ComparisonPipeline

# Create output directory
OUTPUT_DIR = Path(parent_dir) / "duckdb_api" / "simulation_validation" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    """Main function for testing the ValidationReporter."""
    
    # Generate sample data
    simulation_results, hardware_results = generate_sample_data(num_samples=5)
    
    # Create comparison pipeline
    pipeline = ComparisonPipeline()
    
    # Align and compare results
    aligned_pairs = pipeline.align_data(simulation_results, hardware_results)
    validation_results = pipeline.compare_results(aligned_pairs)
    
    # Create validation reporter
    reporter = ValidationReporterImpl()
    
    # Generate reports in different formats
    formats = ["html", "markdown", "json", "text"]
    for format in formats:
        # Generate report
        report = reporter.generate_report(
            validation_results=validation_results,
            format=format,
            include_visualizations=(format == "html")
        )
        
        # Save to file
        output_path = os.path.join(OUTPUT_DIR, f"reporter_test.{format}")
        if format == "html":
            output_path = os.path.join(OUTPUT_DIR, "reporter_test.html")
        elif format == "markdown":
            output_path = os.path.join(OUTPUT_DIR, "reporter_test.md")
        elif format == "json":
            output_path = os.path.join(OUTPUT_DIR, "reporter_test.json")
        else:
            output_path = os.path.join(OUTPUT_DIR, "reporter_test.txt")
            
        with open(output_path, 'w') as f:
            f.write(report)
            
        print(f"Report saved to: {output_path}")

if __name__ == "__main__":
    main()
