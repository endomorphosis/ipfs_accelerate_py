#!/usr/bin/env python3
"""
Test result analysis script for IPFS Accelerate.

This script analyzes test results from JUnit XML files and generates reports.
"""

import os
import sys
import json
import argparse
import logging
import glob
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import xml.etree.ElementTree as ET

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import visualization packages
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    logger.warning("Visualization packages not available. Install matplotlib, seaborn, pandas to enable visualizations.")
    VISUALIZATION_AVAILABLE = False


def parse_junit_xml(file_path: str) -> Dict[str, Any]:
    """
    Parse a JUnit XML file and extract test results.
    
    Args:
        file_path: Path to the JUnit XML file
        
    Returns:
        Dictionary with test results
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        results = {
            'total': 0,
            'failures': 0,
            'errors': 0,
            'skipped': 0,
            'passed': 0,
            'time': 0.0,
            'test_cases': []
        }
        
        # Process testsuite elements
        for testsuite in root.findall('.//testsuite'):
            results['total'] += int(testsuite.get('tests', 0))
            results['failures'] += int(testsuite.get('failures', 0))
            results['errors'] += int(testsuite.get('errors', 0))
            results['skipped'] += int(testsuite.get('skipped', 0))
            results['time'] += float(testsuite.get('time', 0))
            
            # Process testcase elements
            for testcase in testsuite.findall('testcase'):
                case_result = {
                    'name': testcase.get('name', ''),
                    'classname': testcase.get('classname', ''),
                    'time': float(testcase.get('time', 0)),
                    'status': 'passed',
                    'message': ''
                }
                
                # Check for failures
                failure = testcase.find('failure')
                if failure is not None:
                    case_result['status'] = 'failed'
                    case_result['message'] = failure.get('message', '')
                
                # Check for errors
                error = testcase.find('error')
                if error is not None:
                    case_result['status'] = 'error'
                    case_result['message'] = error.get('message', '')
                
                # Check for skipped
                skipped = testcase.find('skipped')
                if skipped is not None:
                    case_result['status'] = 'skipped'
                    case_result['message'] = skipped.get('message', '')
                
                results['test_cases'].append(case_result)
        
        # Calculate passed tests
        results['passed'] = results['total'] - results['failures'] - results['errors'] - results['skipped']
        
        return results
    
    except Exception as e:
        logger.error(f"Error parsing JUnit XML file {file_path}: {e}")
        return {
            'total': 0,
            'failures': 0,
            'errors': 0,
            'skipped': 0,
            'passed': 0,
            'time': 0.0,
            'test_cases': []
        }


def find_junit_xml_files(directory: str) -> List[str]:
    """
    Find JUnit XML files in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        List of file paths
    """
    # Check if the path exists
    if not os.path.exists(directory):
        logger.error(f"Directory {directory} does not exist")
        return []
    
    # Find XML files recursively
    xml_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('-results.xml') or file == 'test-results.xml':
                xml_files.append(os.path.join(root, file))
    
    return xml_files


def categorize_tests(test_cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    Categorize test cases by type and status.
    
    Args:
        test_cases: List of test case dictionaries
        
    Returns:
        Dictionary with test categories and counts
    """
    categories = {
        'model': defaultdict(int),
        'hardware': defaultdict(int),
        'api': defaultdict(int),
        'integration': defaultdict(int),
        'other': defaultdict(int)
    }
    
    for case in test_cases:
        classname = case['classname']
        status = case['status']
        
        # Determine category based on classname
        if 'model' in classname.lower() or any(model in classname.lower() for model in ['bert', 't5', 'gpt', 'vit', 'whisper']):
            category = 'model'
        elif 'hardware' in classname.lower() or any(hw in classname.lower() for hw in ['webgpu', 'webnn', 'cuda', 'rocm']):
            category = 'hardware'
        elif 'api' in classname.lower() or any(api in classname.lower() for api in ['openai', 'hf_tei', 'hf_tgi', 'ollama', 'vllm', 'claude']):
            category = 'api'
        elif 'integration' in classname.lower():
            category = 'integration'
        else:
            category = 'other'
        
        # Increment count for the category and status
        categories[category][status] += 1
        
    return categories


def analyze_test_duration(test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze test duration statistics.
    
    Args:
        test_cases: List of test case dictionaries
        
    Returns:
        Dictionary with duration statistics
    """
    if not test_cases:
        return {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'median': 0.0,
            'total': 0.0,
            'slowest_tests': []
        }
    
    # Extract duration values
    durations = [case['time'] for case in test_cases]
    
    # Calculate statistics
    stats = {
        'min': min(durations),
        'max': max(durations),
        'mean': sum(durations) / len(durations),
        'median': sorted(durations)[len(durations) // 2],
        'total': sum(durations)
    }
    
    # Find slowest tests
    sorted_cases = sorted(test_cases, key=lambda x: x['time'], reverse=True)
    stats['slowest_tests'] = sorted_cases[:10]  # Top 10 slowest tests
    
    return stats


def generate_text_report(results: Dict[str, Any], categories: Dict[str, Dict[str, int]], 
                        duration_stats: Dict[str, Any]) -> str:
    """
    Generate a text report of test results.
    
    Args:
        results: Test results dictionary
        categories: Test categories dictionary
        duration_stats: Test duration statistics
        
    Returns:
        Text report
    """
    report = [
        "# IPFS Accelerate Test Analysis Report",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overall Summary",
        "",
        f"- Total tests: {results['total']}",
        f"- Passed: {results['passed']} ({results['passed'] / results['total'] * 100:.1f}% if results['total'] > 0 else 0}%)",
        f"- Failed: {results['failures']} ({results['failures'] / results['total'] * 100:.1f}% if results['total'] > 0 else 0}%)",
        f"- Errors: {results['errors']} ({results['errors'] / results['total'] * 100:.1f}% if results['total'] > 0 else 0}%)",
        f"- Skipped: {results['skipped']} ({results['skipped'] / results['total'] * 100:.1f}% if results['total'] > 0 else 0}%)",
        f"- Total execution time: {duration_stats['total']:.2f} seconds",
        "",
        "## Test Categories",
        ""
    ]
    
    # Add category details
    for category, statuses in categories.items():
        total = sum(statuses.values())
        if total == 0:
            continue
            
        passed = statuses.get('passed', 0)
        report.append(f"### {category.capitalize()} Tests")
        report.append(f"- Total: {total}")
        report.append(f"- Passed: {passed} ({passed / total * 100:.1f}% if total > 0 else 0}%)")
        report.append(f"- Failed: {statuses.get('failed', 0)}")
        report.append(f"- Errors: {statuses.get('error', 0)}")
        report.append(f"- Skipped: {statuses.get('skipped', 0)}")
        report.append("")
    
    # Add duration statistics
    report.extend([
        "## Performance Statistics",
        "",
        f"- Mean execution time: {duration_stats['mean']:.4f} seconds",
        f"- Median execution time: {duration_stats['median']:.4f} seconds",
        f"- Minimum execution time: {duration_stats['min']:.4f} seconds",
        f"- Maximum execution time: {duration_stats['max']:.4f} seconds",
        "",
        "### Slowest Tests",
        ""
    ])
    
    # Add slowest tests
    for i, test in enumerate(duration_stats['slowest_tests']):
        report.append(f"{i+1}. {test['classname']}.{test['name']} - {test['time']:.4f} seconds")
    
    report.append("")
    
    # Add failure details if any
    failures = [case for case in results['test_cases'] if case['status'] == 'failed']
    if failures:
        report.extend([
            "## Test Failures",
            ""
        ])
        
        for i, test in enumerate(failures):
            report.append(f"{i+1}. {test['classname']}.{test['name']}")
            report.append(f"   Message: {test['message']}")
            report.append("")
    
    return "\n".join(report)


def generate_visualizations(results: Dict[str, Any], categories: Dict[str, Dict[str, int]], 
                           duration_stats: Dict[str, Any], output_dir: str) -> None:
    """
    Generate visualizations of test results.
    
    Args:
        results: Test results dictionary
        categories: Test categories dictionary
        duration_stats: Test duration statistics
        output_dir: Output directory for visualizations
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Skipping visualizations as required packages are not available")
        return
    
    try:
        # Set up plot style
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 10))
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Overall results pie chart
        labels = ['Passed', 'Failed', 'Errors', 'Skipped']
        sizes = [results['passed'], results['failures'], results['errors'], results['skipped']]
        colors = ['#4CAF50', '#F44336', '#FF9800', '#9E9E9E']
        
        if sum(sizes) > 0:  # Avoid division by zero
            axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Overall Test Results')
            axes[0, 0].axis('equal')
        
        # Plot 2: Results by category
        category_names = list(categories.keys())
        category_passed = [categories[cat].get('passed', 0) for cat in category_names]
        category_failed = [categories[cat].get('failed', 0) + categories[cat].get('error', 0) for cat in category_names]
        category_skipped = [categories[cat].get('skipped', 0) for cat in category_names]
        
        x = np.arange(len(category_names))
        width = 0.25
        
        axes[0, 1].bar(x - width, category_passed, width, label='Passed', color='#4CAF50')
        axes[0, 1].bar(x, category_failed, width, label='Failed/Errors', color='#F44336')
        axes[0, 1].bar(x + width, category_skipped, width, label='Skipped', color='#9E9E9E')
        
        axes[0, 1].set_title('Results by Category')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([cat.capitalize() for cat in category_names])
        axes[0, 1].legend()
        
        # Plot 3: Test duration histogram
        durations = [case['time'] for case in results['test_cases']]
        
        if durations:  # Check if there are any durations
            sns.histplot(durations, kde=True, color='#2196F3', ax=axes[1, 0])
            axes[1, 0].set_title('Test Duration Distribution')
            axes[1, 0].set_xlabel('Duration (seconds)')
            axes[1, 0].set_ylabel('Count')
        
        # Plot 4: Top 10 slowest tests
        if duration_stats['slowest_tests']:  # Check if there are any tests
            test_names = [f"{test['classname'].split('.')[-1]}.{test['name']}" for test in duration_stats['slowest_tests']]
            test_durations = [test['time'] for test in duration_stats['slowest_tests']]
            
            # Truncate long names
            test_names = [name[-30:] if len(name) > 30 else name for name in test_names]
            
            y_pos = np.arange(len(test_names))
            
            axes[1, 1].barh(y_pos, test_durations, color='#673AB7')
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels(test_names)
            axes[1, 1].invert_yaxis()  # Labels read top-to-bottom
            axes[1, 1].set_title('Top 10 Slowest Tests')
            axes[1, 1].set_xlabel('Duration (seconds)')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'test_performance_chart.png'), dpi=300)
        plt.close()
        
        logger.info(f"Visualizations saved to {os.path.join(output_dir, 'test_performance_chart.png')}")
    
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze test results from JUnit XML files')
    parser.add_argument('--reports-dir', default='test-reports', help='Directory containing test reports')
    parser.add_argument('--output-dir', default='.', help='Output directory for analysis files')
    parser.add_argument('--format', choices=['text', 'json', 'both'], default='both', help='Output format')
    
    args = parser.parse_args()
    
    # Find JUnit XML files
    logger.info(f"Searching for JUnit XML files in {args.reports_dir}")
    xml_files = find_junit_xml_files(args.reports_dir)
    
    if not xml_files:
        logger.error(f"No JUnit XML files found in {args.reports_dir}")
        return 1
    
    logger.info(f"Found {len(xml_files)} JUnit XML files")
    
    # Combine results from all files
    combined_results = {
        'total': 0,
        'failures': 0,
        'errors': 0,
        'skipped': 0,
        'passed': 0,
        'time': 0.0,
        'test_cases': []
    }
    
    for file_path in xml_files:
        logger.info(f"Parsing {file_path}")
        results = parse_junit_xml(file_path)
        
        combined_results['total'] += results['total']
        combined_results['failures'] += results['failures']
        combined_results['errors'] += results['errors']
        combined_results['skipped'] += results['skipped']
        combined_results['passed'] += results['passed']
        combined_results['time'] += results['time']
        combined_results['test_cases'].extend(results['test_cases'])
    
    # Analyze results
    logger.info("Analyzing test results")
    categories = categorize_tests(combined_results['test_cases'])
    duration_stats = analyze_test_duration(combined_results['test_cases'])
    
    # Generate reports
    if args.format in ['text', 'both']:
        logger.info("Generating text report")
        text_report = generate_text_report(combined_results, categories, duration_stats)
        
        output_path = os.path.join(args.output_dir, 'test_analysis_report.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        logger.info(f"Text report saved to {output_path}")
    
    if args.format in ['json', 'both']:
        logger.info("Generating JSON report")
        json_report = {
            'summary': {
                'total': combined_results['total'],
                'passed': combined_results['passed'],
                'failures': combined_results['failures'],
                'errors': combined_results['errors'],
                'skipped': combined_results['skipped'],
                'time': combined_results['time']
            },
            'categories': categories,
            'duration_stats': duration_stats
        }
        
        output_path = os.path.join(args.output_dir, 'test_analysis_report.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2)
        
        logger.info(f"JSON report saved to {output_path}")
    
    # Generate visualizations
    logger.info("Generating visualizations")
    generate_visualizations(combined_results, categories, duration_stats, args.output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())