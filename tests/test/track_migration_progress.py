#!/usr/bin/env python3
"""
Migration progress tracking script for IPFS Accelerate.

This script analyzes the migration progress and generates reports.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def find_test_files(directory: str) -> Dict[str, List[str]]:
    """
    Find test files in a directory.
    
    Args:
        directory: Directory to search
        
    Returns:
        Dictionary with test types and file lists
    """
    if not os.path.isdir(directory):
        logger.error(f"{directory} is not a directory")
        return {}
    
    test_files = {
        'model': [],
        'hardware': [],
        'api': [],
        'other': []
    }
    
    # Find model tests
    model_dirs = ['models', 'text', 'vision', 'audio', 'multimodal']
    hardware_dirs = ['hardware', 'webgpu', 'webnn', 'cuda', 'rocm']
    api_dirs = ['api', 'clients', 'endpoints']
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, directory)
                
                # Determine test type based on directory
                if any(model_dir in rel_path.lower() for model_dir in model_dirs):
                    test_files['model'].append(rel_path)
                elif any(hw_dir in rel_path.lower() for hw_dir in hardware_dirs):
                    test_files['hardware'].append(rel_path)
                elif any(api_dir in rel_path.lower() for api_dir in api_dirs):
                    test_files['api'].append(rel_path)
                else:
                    test_files['other'].append(rel_path)
    
    return test_files


def load_analysis_report(file_path: str) -> List[Dict[str, Any]]:
    """
    Load analysis report from a file.
    
    Args:
        file_path: Path to the analysis report
        
    Returns:
        Analysis results
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading analysis report: {e}")
        return []


def calculate_migration_progress(analysis_results: List[Dict[str, Any]], 
                                migrated_tests: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Calculate migration progress.
    
    Args:
        analysis_results: Analysis results
        migrated_tests: Dictionary with migrated test files
        
    Returns:
        Dictionary with migration progress statistics
    """
    total_tests = len(analysis_results)
    
    # Count tests by type
    type_counts = {
        'model': sum(1 for result in analysis_results if result['test_type'] == 'model'),
        'hardware': sum(1 for result in analysis_results if result['test_type'] == 'hardware'),
        'api': sum(1 for result in analysis_results if result['test_type'] == 'api'),
        'unknown': sum(1 for result in analysis_results if result['test_type'] == 'unknown')
    }
    
    # Count migrated tests
    migrated_counts = {
        'model': len(migrated_tests['model']),
        'hardware': len(migrated_tests['hardware']),
        'api': len(migrated_tests['api']),
        'other': len(migrated_tests['other'])
    }
    
    # Calculate percentages
    percentages = {}
    for test_type, count in type_counts.items():
        if test_type == 'unknown':
            continue
            
        if count > 0:
            percentages[test_type] = (migrated_counts[test_type] / count) * 100
        else:
            percentages[test_type] = 0
    
    # Calculate overall percentage
    total_known = sum(count for test_type, count in type_counts.items() if test_type != 'unknown')
    total_migrated = sum(migrated_counts.values())
    
    if total_known > 0:
        overall_percentage = (total_migrated / total_known) * 100
    else:
        overall_percentage = 0
    
    return {
        'total_tests': total_tests,
        'total_known': total_known,
        'total_migrated': total_migrated,
        'type_counts': type_counts,
        'migrated_counts': migrated_counts,
        'percentages': percentages,
        'overall_percentage': overall_percentage
    }


def generate_text_report(progress: Dict[str, Any]) -> str:
    """
    Generate a text report of migration progress.
    
    Args:
        progress: Migration progress dictionary
        
    Returns:
        Text report
    """
    report = [
        "# IPFS Accelerate Test Migration Progress Report",
        "",
        f"Total tests analyzed: {progress['total_tests']}",
        f"Total tests with recognized type: {progress['total_known']}",
        f"Total tests migrated: {progress['total_migrated']}",
        f"Overall migration progress: {progress['overall_percentage']:.1f}%",
        "",
        "## Test Types",
        ""
    ]
    
    # Add type details
    for test_type in ['model', 'hardware', 'api']:
        count = progress['type_counts'][test_type]
        migrated = progress['migrated_counts'][test_type]
        percentage = progress['percentages'][test_type]
        
        report.append(f"### {test_type.capitalize()} Tests")
        report.append(f"- Total: {count}")
        report.append(f"- Migrated: {migrated}")
        report.append(f"- Progress: {percentage:.1f}%")
        report.append("")
    
    # Add unknown tests
    unknown_count = progress['type_counts']['unknown']
    report.append(f"### Unknown Tests")
    report.append(f"- Total: {unknown_count}")
    report.append("")
    
    return "\n".join(report)


def generate_ascii_progress_bar(percentage: float, width: int = 50) -> str:
    """
    Generate an ASCII progress bar.
    
    Args:
        percentage: Percentage complete
        width: Width of the progress bar
        
    Returns:
        ASCII progress bar string
    """
    filled_width = int(width * percentage / 100)
    bar = '█' * filled_width + '░' * (width - filled_width)
    return f"[{bar}] {percentage:.1f}%"


def generate_ascii_report(progress: Dict[str, Any]) -> str:
    """
    Generate an ASCII report of migration progress.
    
    Args:
        progress: Migration progress dictionary
        
    Returns:
        ASCII report
    """
    bar_width = 50
    
    report = [
        "=== IPFS Accelerate Test Migration Progress ===",
        "",
        f"Total tests: {progress['total_tests']}",
        f"Recognized: {progress['total_known']}",
        f"Migrated:  {progress['total_migrated']}",
        "",
        "Overall progress:",
        generate_ascii_progress_bar(progress['overall_percentage'], bar_width),
        "",
        "Progress by type:",
        f"Model tests:    {generate_ascii_progress_bar(progress['percentages']['model'], bar_width)}",
        f"Hardware tests: {generate_ascii_progress_bar(progress['percentages']['hardware'], bar_width)}",
        f"API tests:      {generate_ascii_progress_bar(progress['percentages']['api'], bar_width)}",
        "",
        f"Unknown tests:  {progress['type_counts']['unknown']}"
    ]
    
    return "\n".join(report)


def save_report(report: str, output_path: str) -> None:
    """
    Save a report to a file.
    
    Args:
        report: Report content
        output_path: Path to save the report
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving report: {e}")


def main() -> None:
    """
    Main function to track migration progress.
    """
    parser = argparse.ArgumentParser(description='Track test migration progress')
    
    parser.add_argument('--analysis-report', required=True, help='Path to the analysis report')
    parser.add_argument('--migrated-dir', required=True, help='Directory with migrated tests')
    parser.add_argument('--output-report', default='migration_progress.md', help='Path to save the report')
    parser.add_argument('--ascii', action='store_true', help='Generate ASCII report instead of Markdown')
    
    args = parser.parse_args()
    
    # Load analysis report
    logger.info(f"Loading analysis report from {args.analysis_report}")
    analysis_results = load_analysis_report(args.analysis_report)
    
    if not analysis_results:
        logger.error("No analysis results found")
        return
    
    logger.info(f"Loaded {len(analysis_results)} analysis results")
    
    # Find migrated tests
    logger.info(f"Finding migrated tests in {args.migrated_dir}")
    migrated_tests = find_test_files(args.migrated_dir)
    
    # Calculate progress
    progress = calculate_migration_progress(analysis_results, migrated_tests)
    
    # Generate report
    if args.ascii:
        report = generate_ascii_report(progress)
    else:
        report = generate_text_report(progress)
    
    # Print report to console
    print(report)
    
    # Save report to file
    save_report(report, args.output_report)
    logger.info(f"Report saved to {args.output_report}")


if __name__ == "__main__":
    main()