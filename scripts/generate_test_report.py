#!/usr/bin/env python3
"""
Test Report Generator for Mojo CI/CD Pipeline

Aggregates test results from various test suites and generates
comprehensive reports in JSON and Markdown formats.
"""

import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

def parse_junit_xml(file_path):
    """Parse JUnit XML file and extract test statistics."""
    if not os.path.exists(file_path):
        return {'tests': 0, 'failures': 0, 'errors': 0, 'skipped': 0}
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        return {
            'tests': int(root.get('tests', 0)),
            'failures': int(root.get('failures', 0)),
            'errors': int(root.get('errors', 0)),
            'skipped': int(root.get('skipped', 0))
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return {'tests': 0, 'failures': 0, 'errors': 0, 'skipped': 0}

def parse_performance_results(file_path):
    """Parse performance test results."""
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error parsing performance results: {e}")
        return None

def parse_quality_logs(results_dir):
    """Parse code quality check logs."""
    quality_results = {}
    
    quality_files = {
        'black': 'black-check.log',
        'flake8': 'flake8-check.log',
        'mypy': 'mypy-check.log',
        'bandit': 'bandit-check.log',
        'safety': 'safety-check.log'
    }
    
    for tool, filename in quality_files.items():
        file_path = os.path.join(results_dir, filename)
        if os.path.exists(file_path):
            try:
                with open(file_path) as f:
                    content = f.read().strip()
                    # Simple heuristic: if file is empty or contains only OK messages, assume pass
                    has_issues = bool(content and 'error' in content.lower() or 'warning' in content.lower())
                    quality_results[tool] = {
                        'status': 'fail' if has_issues else 'pass',
                        'output_length': len(content)
                    }
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                quality_results[tool] = {'status': 'unknown', 'output_length': 0}
        else:
            quality_results[tool] = {'status': 'not_run', 'output_length': 0}
    
    return quality_results

def generate_test_report(results_dir='/app/test-results'):
    """Generate comprehensive test report."""
    print("Generating test report...")
    
    # Initialize report structure
    report = {
        'timestamp': datetime.now().isoformat(),
        'environment': 'docker-compose',
        'test_results': {},
        'quality_results': {},
        'performance_results': None
    }
    
    # Parse test results
    test_files = [
        ('unit-tests', os.path.join(results_dir, 'unit-tests.xml')),
        ('integration-tests', os.path.join(results_dir, 'integration-tests.xml')),
        ('e2e-tests', os.path.join(results_dir, 'e2e-tests.xml'))
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    
    for test_name, file_path in test_files:
        print(f"Parsing {test_name} results...")
        results = parse_junit_xml(file_path)
        report['test_results'][test_name] = results
        total_tests += results['tests']
        total_failures += results['failures']
        total_errors += results['errors']
        total_skipped += results['skipped']
    
    # Calculate summary
    report['summary'] = {
        'total_tests': total_tests,
        'total_failures': total_failures,
        'total_errors': total_errors,
        'total_skipped': total_skipped,
        'success_rate': round((total_tests - total_failures - total_errors) / max(total_tests, 1) * 100, 2)
    }
    
    # Parse quality results
    print("Parsing quality check results...")
    report['quality_results'] = parse_quality_logs(results_dir)
    
    # Parse performance results
    perf_file = os.path.join(results_dir, 'performance.json')
    if os.path.exists(perf_file):
        print("Parsing performance results...")
        report['performance_results'] = parse_performance_results(perf_file)
    
    # Save JSON report
    json_report_path = os.path.join(results_dir, 'test-report.json')
    with open(json_report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"JSON report saved to: {json_report_path}")
    
    # Generate markdown report
    print("Generating markdown report...")
    md_report = generate_markdown_report(report)
    
    md_report_path = os.path.join(results_dir, 'test-report.md')
    with open(md_report_path, 'w') as f:
        f.write(md_report)
    print(f"Markdown report saved to: {md_report_path}")
    
    # Print summary to console
    print_summary(report)
    
    return report

def generate_markdown_report(report):
    """Generate markdown test report."""
    md_report = f"""# Mojo Integration Test Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Environment:** {report['environment']}

## Summary

- **Total Tests:** {report['summary']['total_tests']}
- **Passed:** {report['summary']['total_tests'] - report['summary']['total_failures'] - report['summary']['total_errors']}
- **Failures:** {report['summary']['total_failures']}
- **Errors:** {report['summary']['total_errors']}
- **Skipped:** {report['summary']['total_skipped']}
- **Success Rate:** {report['summary']['success_rate']}%

## Test Results

"""
    
    # Add test results
    for test_name, results in report['test_results'].items():
        status = '✅ PASSED' if results['failures'] == 0 and results['errors'] == 0 else '❌ FAILED'
        md_report += f"### {test_name.replace('-', ' ').title()}\n"
        md_report += f"- **Status:** {status}\n"
        md_report += f"- **Tests:** {results['tests']}\n"
        md_report += f"- **Failures:** {results['failures']}\n"
        md_report += f"- **Errors:** {results['errors']}\n"
        md_report += f"- **Skipped:** {results['skipped']}\n\n"
    
    # Add quality results
    if report['quality_results']:
        md_report += "## Code Quality\n\n"
        for tool, result in report['quality_results'].items():
            status_emoji = {'pass': '✅', 'fail': '❌', 'unknown': '❓', 'not_run': '⏭️'}
            status = status_emoji.get(result['status'], '❓')
            md_report += f"- **{tool.title()}:** {status} {result['status'].upper()}\n"
        md_report += "\n"
    
    # Add performance results
    if report['performance_results']:
        md_report += "## Performance Results\n\n"
        if 'benchmarks' in report['performance_results']:
            for benchmark in report['performance_results']['benchmarks']:
                name = benchmark.get('name', 'Unknown')
                stats = benchmark.get('stats', {})
                md_report += f"### {name}\n"
                md_report += f"- **Mean:** {stats.get('mean', 0):.4f}s\n"
                md_report += f"- **Min:** {stats.get('min', 0):.4f}s\n"
                md_report += f"- **Max:** {stats.get('max', 0):.4f}s\n\n"
        else:
            md_report += "Performance data format not recognized.\n\n"
    
    md_report += "---\n"
    md_report += f"*Report generated on {report['timestamp']}*\n"
    
    return md_report

def print_summary(report):
    """Print test summary to console."""
    print("\n" + "="*60)
    print("🚀 MOJO INTEGRATION TEST SUMMARY")
    print("="*60)
    
    summary = report['summary']
    print(f"Total Tests:    {summary['total_tests']}")
    print(f"Passed:         {summary['total_tests'] - summary['total_failures'] - summary['total_errors']}")
    print(f"Failures:       {summary['total_failures']}")
    print(f"Errors:         {summary['total_errors']}")
    print(f"Skipped:        {summary['total_skipped']}")
    print(f"Success Rate:   {summary['success_rate']}%")
    
    # Overall status
    if summary['total_failures'] == 0 and summary['total_errors'] == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    # Quality summary
    if report['quality_results']:
        print("\nCode Quality:")
        for tool, result in report['quality_results'].items():
            status = result['status'].upper()
            print(f"  {tool}: {status}")
    
    print("="*60)

if __name__ == "__main__":
    import sys
    
    results_dir = sys.argv[1] if len(sys.argv) > 1 else '/app/test-results'
    
    try:
        report = generate_test_report(results_dir)
        sys.exit(0 if report['summary']['total_failures'] == 0 and report['summary']['total_errors'] == 0 else 1)
    except Exception as e:
        print(f"Error generating test report: {e}")
        sys.exit(1)
