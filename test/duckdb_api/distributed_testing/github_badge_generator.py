#!/usr/bin/env python3
"""
GitHub Badge Generator for Distributed Testing Framework

This script generates JSON files for GitHub repository badges that display test status.
It analyzes test results and creates badge JSON files in the .github/badges directory.

Usage:
    python -m duckdb_api.distributed_testing.github_badge_generator [--input-dir DIR] [--output-dir DIR]
    python -m duckdb_api.distributed_testing.github_badge_generator --input ./drm_reports/test_report_*.json --output ./.github/badges/drm_status.json --type drm
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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

def generate_drm_badge(input_file: str, output_file: str, label: str = "DRM Tests") -> bool:
    """
    Generate a GitHub-compatible status badge JSON file from DRM test results.
    
    Args:
        input_file: JSON report file (can include glob pattern)
        output_file: Output JSON file for the badge
        label: Badge label
        
    Returns:
        True if successful
    """
    # Resolve input file(s)
    input_files = []
    if '*' in input_file:
        input_files = glob.glob(input_file)
    else:
        input_files = [input_file]
    
    if not input_files:
        print(f"Error: No input files found matching pattern '{input_file}'")
        return False
    
    # Aggregate results
    total_tasks = 0
    completed_tasks = 0
    failed_tasks = 0
    canceled_tasks = 0
    timeout_tasks = 0
    
    for file_path in input_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Extract summary statistics
                summary = data.get('summary', {})
                total_tasks += summary.get('total', 0)
                
                # Extract status counts
                status_counts = summary.get('status_counts', {})
                completed_tasks += status_counts.get('completed', 0)
                failed_tasks += status_counts.get('failed', 0)
                canceled_tasks += status_counts.get('cancelled', 0)
                timeout_tasks += status_counts.get('timeout', 0)
                
                # If no summary, try to extract directly from results
                if not summary and 'results' in data:
                    results = data.get('results', {})
                    total_tasks += len(results)
                    
                    for task_id, result in results.items():
                        status = result.get('status', '')
                        if status == 'completed':
                            completed_tasks += 1
                        elif status == 'failed':
                            failed_tasks += 1
                        elif status == 'cancelled':
                            canceled_tasks += 1
                        elif status == 'timeout':
                            timeout_tasks += 1
        
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error processing file {file_path}: {str(e)}")
    
    # Determine badge color and status text
    if total_tasks == 0:
        color = "lightgrey"
        status = "unknown"
    elif failed_tasks > 0 or timeout_tasks > 0:
        color = "red"
        status = "failing"
    elif canceled_tasks > 0:
        color = "yellow"
        status = f"passing ({canceled_tasks} canceled)"
    else:
        color = "brightgreen"
        status = f"passing ({completed_tasks}/{total_tasks})"
    
    # Create badge data
    badge_data = {
        "schemaVersion": 1,
        "label": label,
        "message": status,
        "color": color
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write badge data
    with open(output_file, 'w') as f:
        json.dump(badge_data, f, indent=2)
    
    print(f"Badge generated: {output_file}")
    print(f"Status: {label} - {status}")
    
    return True


def generate_drm_component_badges(input_file: str, output_dir: str) -> bool:
    """
    Generate DRM component-specific badges.
    
    Args:
        input_file: JSON report file
        output_dir: Output directory for badge files
        
    Returns:
        True if successful
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
            results = data.get('results', {})
            
            # Group results by component
            components = {
                'dynamic_resource_manager': {'total': 0, 'passed': 0, 'failed': 0},
                'resource_performance_predictor': {'total': 0, 'passed': 0, 'failed': 0},
                'cloud_provider_manager': {'total': 0, 'passed': 0, 'failed': 0},
                'resource_optimizer': {'total': 0, 'passed': 0, 'failed': 0},
                'drm_integration': {'total': 0, 'passed': 0, 'failed': 0},
                'e2e_test': {'total': 0, 'passed': 0, 'failed': 0}
            }
            
            # Analyze results
            for task_id, result in results.items():
                config = result.get('config', {})
                test_file = config.get('test_file', '')
                
                # Determine component from test_file
                component = None
                for c in components.keys():
                    if c in test_file:
                        component = c
                        break
                
                # Check for e2e test specifically
                if 'e2e' in test_file and 'run_e2e_drm_test' in test_file:
                    component = 'e2e_test'
                
                if component:
                    components[component]['total'] += 1
                    if result.get('status') == 'completed':
                        components[component]['passed'] += 1
                    else:
                        components[component]['failed'] += 1
            
            # Generate badges for each component
            os.makedirs(output_dir, exist_ok=True)
            
            for component, stats in components.items():
                if stats['total'] > 0:
                    # Determine badge color and status text
                    if stats['failed'] > 0:
                        color = "red"
                        status = f"failing ({stats['passed']}/{stats['total']})"
                    else:
                        color = "brightgreen"
                        status = f"passing ({stats['passed']}/{stats['total']})"
                    
                    # Create badge data
                    badge_data = {
                        "schemaVersion": 1,
                        "label": component.replace('_', ' ').title(),
                        "message": status,
                        "color": color
                    }
                    
                    # Write badge data
                    output_file = os.path.join(output_dir, f"drm_{component}_status.json")
                    with open(output_file, 'w') as f:
                        json.dump(badge_data, f, indent=2)
                    
                    print(f"Component badge generated: {output_file}")
        
        return True
    
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error processing file {input_file}: {str(e)}")
        return False


def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(description='GitHub Badge Generator')
    
    # Allow both traditional directory-based and file-based inputs
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--input-dir', help='Directory containing test result files')
    input_group.add_argument('--input', help='JSON report file (can include glob pattern)')
    
    parser.add_argument('--output-dir', help='Directory to write badge JSON files')
    parser.add_argument('--output', help='Output JSON file for the badge')
    parser.add_argument('--label', default='Tests', help='Badge label')
    parser.add_argument('--type', choices=['standard', 'drm'], default='standard', 
                      help='Type of badge to generate')
    parser.add_argument('--component-badges', action='store_true', 
                      help='Generate component-specific badges')
    
    args = parser.parse_args()
    
    # DRM badge generation
    if args.type == 'drm':
        if not args.input:
            print("Error: --input is required for DRM badge generation")
            return 1
        
        if not args.output:
            print("Error: --output is required for DRM badge generation")
            return 1
        
        # Generate main DRM badge
        success = generate_drm_badge(args.input, args.output, args.label or "DRM Tests")
        
        # Generate component badges if requested
        if args.component_badges and success:
            # Use the first file if the input is a glob pattern
            input_file = args.input
            if '*' in args.input:
                matching_files = glob.glob(args.input)
                if matching_files:
                    input_file = matching_files[0]
            
            output_dir = os.path.dirname(args.output)
            generate_drm_component_badges(input_file, output_dir)
        
        return 0 if success else 1
    
    # Standard badge generation
    else:
        input_dir = args.input_dir or '../../../test-results'
        output_dir = args.output_dir or '.github/badges'
        
        # Parse test results
        status_data = parse_test_results(input_dir)
        
        if not status_data:
            print("No test results found.")
            return 1
        
        # Generate individual badges
        generate_individual_badges(status_data, output_dir)
        
        # Generate combined badge
        generate_combined_badge(status_data, output_dir)
        
        # Extract coverage data
        coverage_data = extract_coverage_data(input_dir)
        
        # Generate coverage badge if data available
        if coverage_data:
            generate_coverage_badge(coverage_data, output_dir)
        
        print("Badge generation complete.")
        return 0

if __name__ == '__main__':
    sys.exit(main())