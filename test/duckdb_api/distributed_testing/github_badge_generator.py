#!/usr/bin/env python3
"""
GitHub Badge Generator for Distributed Testing Framework

This script generates JSON files for GitHub repository badges that display test status.
It analyzes test results and creates badge JSON files in the .github/badges directory.

Usage:
    python -m duckdb_api.distributed_testing.github_badge_generator [--input-dir DIR] [--output-dir DIR]
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def parse_test_results(results_dir: str) -> Dict[str, Dict]:
    """
    Parse test results from the specified directory.
    
    Args:
        results_dir: Directory containing test result files
        
    Returns:
        Dictionary of test type to status information
    """
    status_by_type = {}
    
    # Find all JSON result files
    result_files = glob.glob(os.path.join(results_dir, "test-results-*/results.json"))
    
    for result_file in result_files:
        # Extract test type from filename pattern (test-results-{type}/results.json)
        test_type = os.path.basename(os.path.dirname(result_file)).replace("test-results-", "")
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                
            success = data.get("success", False)
            status_by_type[test_type] = {
                "success": success,
                "status": "passing" if success else "failing",
                "color": "green" if success else "red",
                "total_tests": data.get("total_tests", 0),
                "passed_tests": data.get("passed_tests", 0),
                "failed_tests": data.get("failed_tests", 0)
            }
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error parsing result file {result_file}: {e}")
            status_by_type[test_type] = {
                "success": False,
                "status": "error",
                "color": "yellow",
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "error": str(e)
            }
    
    return status_by_type

def generate_individual_badges(status_data: Dict[str, Dict], output_dir: str) -> None:
    """
    Generate individual badge JSON files for each test type.
    
    Args:
        status_data: Dictionary of test type to status information
        output_dir: Directory to write badge JSON files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate badge for each test type
    for test_type, status in status_data.items():
        badge_data = {
            "schemaVersion": 1,
            "label": f"{test_type} tests",
            "message": status["status"],
            "color": status["color"]
        }
        
        # Write badge JSON file
        badge_file = os.path.join(output_dir, f"{test_type}-status.json")
        with open(badge_file, 'w') as f:
            json.dump(badge_data, f, indent=2)
        
        print(f"Generated badge for {test_type} tests: {badge_file}")

def generate_combined_badge(status_data: Dict[str, Dict], output_dir: str) -> None:
    """
    Generate a combined badge JSON file for overall test status.
    
    Args:
        status_data: Dictionary of test type to status information
        output_dir: Directory to write badge JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine overall status (failing if any test type fails)
    all_passing = all(status.get("success", False) for status in status_data.values())
    
    badge_data = {
        "schemaVersion": 1,
        "label": "tests",
        "message": "passing" if all_passing else "failing",
        "color": "green" if all_passing else "red"
    }
    
    # Write badge JSON file
    badge_file = os.path.join(output_dir, "combined-status.json")
    with open(badge_file, 'w') as f:
        json.dump(badge_data, f, indent=2)
    
    print(f"Generated combined badge: {badge_file}")

def generate_coverage_badge(coverage_data: Dict[str, float], output_dir: str) -> None:
    """
    Generate a coverage badge JSON file based on coverage data.
    
    Args:
        coverage_data: Dictionary of test type to coverage percentage
        output_dir: Directory to write badge JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate average coverage
    if coverage_data:
        avg_coverage = sum(coverage_data.values()) / len(coverage_data)
    else:
        avg_coverage = 0.0
    
    # Determine color based on coverage percentage
    if avg_coverage >= 90:
        color = "brightgreen"
    elif avg_coverage >= 80:
        color = "green"
    elif avg_coverage >= 70:
        color = "yellowgreen"
    elif avg_coverage >= 60:
        color = "yellow"
    else:
        color = "red"
    
    badge_data = {
        "schemaVersion": 1,
        "label": "coverage",
        "message": f"{avg_coverage:.1f}%",
        "color": color
    }
    
    # Write badge JSON file
    badge_file = os.path.join(output_dir, "coverage.json")
    with open(badge_file, 'w') as f:
        json.dump(badge_data, f, indent=2)
    
    print(f"Generated coverage badge: {badge_file}")

def extract_coverage_data(results_dir: str) -> Dict[str, float]:
    """
    Extract coverage data from coverage XML files.
    
    Args:
        results_dir: Directory containing test result files
        
    Returns:
        Dictionary of test type to coverage percentage
    """
    coverage_by_type = {}
    
    # Find all coverage XML files
    coverage_files = glob.glob(os.path.join(results_dir, "test-results-*/coverage.xml"))
    
    for coverage_file in coverage_files:
        # Extract test type from filename pattern (test-results-{type}/coverage.xml)
        test_type = os.path.basename(os.path.dirname(coverage_file)).replace("test-results-", "")
        
        try:
            # Simple extraction of line coverage percentage
            with open(coverage_file, 'r') as f:
                content = f.read()
                # Look for line-rate attribute in the coverage tag
                import re
                match = re.search(r'line-rate="([0-9.]+)"', content)
                if match:
                    line_rate = float(match.group(1))
                    coverage_by_type[test_type] = line_rate * 100.0
        except IOError as e:
            print(f"Error reading coverage file {coverage_file}: {e}")
    
    return coverage_by_type

def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(description='GitHub Badge Generator')
    parser.add_argument('--input-dir', default='../../../test-results',
                        help='Directory containing test result files')
    parser.add_argument('--output-dir', default='.github/badges',
                        help='Directory to write badge JSON files')
    
    args = parser.parse_args()
    
    # Parse test results
    status_data = parse_test_results(args.input_dir)
    
    if not status_data:
        print("No test results found.")
        return 1
    
    # Generate individual badges
    generate_individual_badges(status_data, args.output_dir)
    
    # Generate combined badge
    generate_combined_badge(status_data, args.output_dir)
    
    # Extract coverage data
    coverage_data = extract_coverage_data(args.input_dir)
    
    # Generate coverage badge if data available
    if coverage_data:
        generate_coverage_badge(coverage_data, args.output_dir)
    
    print("Badge generation complete.")
    return 0

if __name__ == '__main__':
    sys.exit(main())