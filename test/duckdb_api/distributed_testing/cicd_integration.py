#!/usr/bin/env python3
"""
CI/CD Integration for Distributed Testing Framework

This module provides integration between the Distributed Testing Framework
and common CI/CD systems (GitHub Actions, GitLab CI, Jenkins). It enables:

1. Automatic test discovery and submission to the coordinator
2. Webhook-based result reporting
3. Status reporting back to CI/CD systems
4. Report generation for CI/CD artifacts
5. Detection of required hardware for tests

Usage examples:
    # Submit tests from GitHub Actions
    python -m duckdb_api.distributed_testing.cicd_integration --provider github \
        --test-dir ./tests --coordinator http://coordinator-url:8080 --api-key KEY
        
    # Submit tests from GitLab CI
    python -m duckdb_api.distributed_testing.cicd_integration --provider gitlab \
        --test-pattern "test_*.py" --coordinator http://coordinator-url:8080 --api-key KEY

    # Submit tests from Jenkins
    python -m duckdb_api.distributed_testing.cicd_integration --provider jenkins \
        --test-files test_file1.py test_file2.py --coordinator http://coordinator-url:8080 --api-key KEY
"""

import argparse
import glob
import json
import os
import re
import requests
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

# Local imports
try:
    from duckdb_api.distributed_testing.run_test import Client
except ImportError:
    # Handle case when running directly
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from duckdb_api.distributed_testing.run_test import Client


class CICDIntegration:
    """
    Integration layer between CI/CD systems and the Distributed Testing Framework.
    Handles test discovery, submission, and result reporting.
    """
    
    def __init__(
        self,
        coordinator_url: str,
        api_key: str,
        provider: str = 'generic',
        timeout: int = 3600,
        poll_interval: int = 15,
        verbose: bool = False
    ):
        """
        Initialize the CI/CD integration.
        
        Args:
            coordinator_url: URL of the coordinator server
            api_key: API key for authentication
            provider: CI/CD provider (github, gitlab, jenkins, generic)
            timeout: Maximum time to wait for test completion (seconds)
            poll_interval: How often to poll for results (seconds)
            verbose: Enable verbose logging
        """
        self.coordinator_url = coordinator_url
        self.api_key = api_key
        self.provider = provider.lower()
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.verbose = verbose
        self.client = Client(coordinator_url, api_key)
        
        # Validate provider
        valid_providers = ['github', 'gitlab', 'jenkins', 'generic']
        if self.provider not in valid_providers:
            raise ValueError(f"Provider '{provider}' not supported. Use one of: {', '.join(valid_providers)}")
        
        # Set up provider-specific settings
        self._setup_provider_context()
    
    def _setup_provider_context(self):
        """Set up CI/CD provider-specific context and environment variables"""
        self.build_id = str(uuid.uuid4())[:8]  # Default build ID
        self.repo_name = "unknown"
        self.branch = "unknown"
        self.commit_sha = "unknown"
        
        # GitHub Actions
        if self.provider == 'github':
            if os.environ.get('GITHUB_ACTIONS') == 'true':
                self.build_id = os.environ.get('GITHUB_RUN_ID', self.build_id)
                self.repo_name = os.environ.get('GITHUB_REPOSITORY', self.repo_name)
                self.branch = os.environ.get('GITHUB_REF_NAME', self.branch)
                self.commit_sha = os.environ.get('GITHUB_SHA', self.commit_sha)
        
        # GitLab CI
        elif self.provider == 'gitlab':
            if os.environ.get('GITLAB_CI') == 'true':
                self.build_id = os.environ.get('CI_JOB_ID', self.build_id)
                self.repo_name = os.environ.get('CI_PROJECT_PATH', self.repo_name)
                self.branch = os.environ.get('CI_COMMIT_REF_NAME', self.branch)
                self.commit_sha = os.environ.get('CI_COMMIT_SHA', self.commit_sha)
        
        # Jenkins
        elif self.provider == 'jenkins':
            if os.environ.get('JENKINS_URL'):
                self.build_id = os.environ.get('BUILD_ID', self.build_id)
                self.repo_name = os.environ.get('JOB_NAME', self.repo_name)
                self.branch = os.environ.get('GIT_BRANCH', '').split('/')[-1] or self.branch
                self.commit_sha = os.environ.get('GIT_COMMIT', self.commit_sha)
    
    def discover_tests(
        self, 
        test_dir: Optional[str] = None,
        test_pattern: Optional[str] = None,
        test_files: Optional[List[str]] = None
    ) -> List[str]:
        """
        Discover test files based on directory, pattern, or explicit list.
        
        Args:
            test_dir: Directory to search for tests
            test_pattern: Glob pattern for test files
            test_files: Explicit list of test files
            
        Returns:
            List of test file paths
        """
        discovered_tests = []
        
        # Option 1: Explicit list of test files
        if test_files:
            for test_file in test_files:
                if os.path.exists(test_file):
                    discovered_tests.append(os.path.abspath(test_file))
                else:
                    if self.verbose:
                        print(f"Warning: Test file not found: {test_file}")
        
        # Option 2: Test directory with default pattern
        elif test_dir:
            pattern = test_pattern or "test_*.py"
            search_path = os.path.join(test_dir, pattern)
            discovered_tests = [os.path.abspath(p) for p in glob.glob(search_path)]
        
        # Option 3: Global pattern
        elif test_pattern:
            discovered_tests = [os.path.abspath(p) for p in glob.glob(test_pattern)]
        
        # Sort for deterministic ordering
        discovered_tests.sort()
        
        if self.verbose:
            print(f"Discovered {len(discovered_tests)} test files")
            for test in discovered_tests:
                print(f"  - {test}")
        
        return discovered_tests
    
    def analyze_test_requirements(self, test_file: str) -> Dict[str, Union[str, List[str]]]:
        """
        Analyze a test file to determine hardware requirements.
        
        Args:
            test_file: Path to the test file
            
        Returns:
            Dictionary of hardware requirements
        """
        requirements = {
            'hardware_type': [],
            'browser': None,
            'platform': None,
            'min_memory_mb': None,
            'priority': 5,  # Default priority
        }
        
        # Don't analyze if file doesn't exist
        if not os.path.exists(test_file):
            return requirements
        
        # Read the file content
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Detect hardware types
        hardware_patterns = {
            'cuda': r'(?:--hardware\s+[\'"]?cuda|cuda.*?=\s*True|\bcuda\b)',
            'rocm': r'(?:--hardware\s+[\'"]?rocm|rocm.*?=\s*True|\brocm\b)',
            'mps': r'(?:--hardware\s+[\'"]?mps|mps.*?=\s*True|\bmps\b)',
            'cpu': r'(?:--hardware\s+[\'"]?cpu|cpu.*?=\s*True|\bcpu\b)',
            'openvino': r'(?:--hardware\s+[\'"]?openvino|openvino.*?=\s*True|\bopenvino\b)',
            'qualcomm': r'(?:--hardware\s+[\'"]?qualcomm|qualcomm.*?=\s*True|\bqualcomm\b|\bqnn\b)',
            'webnn': r'(?:--hardware\s+[\'"]?webnn|webnn.*?=\s*True|\bwebnn\b)',
            'webgpu': r'(?:--hardware\s+[\'"]?webgpu|webgpu.*?=\s*True|\bwebgpu\b)',
        }
        
        for hw_type, pattern in hardware_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                requirements['hardware_type'].append(hw_type)
        
        # If no specific hardware detected, default to CPU
        if not requirements['hardware_type']:
            requirements['hardware_type'].append('cpu')
        
        # Detect browser requirements
        browser_pattern = r'(?:--browser\s+[\'"]?(\w+)|browser\s*=\s*[\'"](\w+)[\'"])'
        browser_match = re.search(browser_pattern, content, re.IGNORECASE)
        if browser_match:
            # Use the first non-None group
            browser = next((g for g in browser_match.groups() if g), None)
            if browser:
                requirements['browser'] = browser.lower()
        
        # Detect platform requirements (if any)
        platform_pattern = r'(?:--platform\s+[\'"]?(\w+)|platform\s*=\s*[\'"](\w+)[\'"])'
        platform_match = re.search(platform_pattern, content, re.IGNORECASE)
        if platform_match:
            # Use the first non-None group
            platform = next((g for g in platform_match.groups() if g), None)
            if platform:
                requirements['platform'] = platform.lower()
        
        # Detect memory requirements (if any)
        memory_pattern = r'(?:--min[-_]memory\s+(\d+)|min[-_]memory\s*=\s*(\d+))'
        memory_match = re.search(memory_pattern, content, re.IGNORECASE)
        if memory_match:
            # Use the first non-None group
            memory = next((g for g in memory_match.groups() if g), None)
            if memory:
                requirements['min_memory_mb'] = int(memory)
        
        # Detect priority (if any)
        priority_pattern = r'(?:--priority\s+(\d+)|priority\s*=\s*(\d+))'
        priority_match = re.search(priority_pattern, content, re.IGNORECASE)
        if priority_match:
            # Use the first non-None group
            priority = next((g for g in priority_match.groups() if g), None)
            if priority:
                requirements['priority'] = int(priority)
        
        return requirements
    
    def submit_test(self, test_file: str, requirements: Dict[str, Union[str, List[str]]]) -> str:
        """
        Submit a test to the distributed testing framework.
        
        Args:
            test_file: Path to the test file
            requirements: Dictionary of test requirements
            
        Returns:
            Task ID for the submitted test
        """
        # Prepare task data
        task_data = {
            'type': 'test',
            'config': {
                'test_file': test_file,
                'source': self.provider,
                'build_id': self.build_id,
                'repo': self.repo_name,
                'branch': self.branch,
                'commit': self.commit_sha
            },
            'requirements': requirements,
            'priority': requirements.get('priority', 5)
        }
        
        # Submit the task
        task_id = self.client.submit_task(task_data)
        
        if self.verbose:
            print(f"Submitted test {os.path.basename(test_file)} as task {task_id}")
            print(f"  Hardware: {', '.join(requirements.get('hardware_type', []))}")
            if requirements.get('browser'):
                print(f"  Browser: {requirements['browser']}")
            if requirements.get('platform'):
                print(f"  Platform: {requirements['platform']}")
            if requirements.get('min_memory_mb'):
                print(f"  Min Memory: {requirements['min_memory_mb']} MB")
            print(f"  Priority: {requirements.get('priority', 5)}")
        
        return task_id
    
    def wait_for_results(self, task_ids: List[str]) -> Dict[str, Dict]:
        """
        Wait for all tasks to complete and collect results.
        
        Args:
            task_ids: List of task IDs to monitor
            
        Returns:
            Dictionary mapping task IDs to results
        """
        if not task_ids:
            return {}
        
        results = {}
        pending_tasks = set(task_ids)
        start_time = time.time()
        
        if self.verbose:
            print(f"Waiting for {len(task_ids)} tasks to complete (timeout: {self.timeout}s)")
        
        # Poll until all tasks complete or timeout
        while pending_tasks and (time.time() - start_time) < self.timeout:
            tasks_to_remove = set()
            
            for task_id in pending_tasks:
                status = self.client.get_task_status(task_id)
                
                if status['status'] in ('completed', 'failed', 'cancelled', 'timeout'):
                    # Task is done, get full results
                    result = self.client.get_task_results(task_id)
                    results[task_id] = result
                    tasks_to_remove.add(task_id)
                    
                    if self.verbose:
                        test_file = status.get('config', {}).get('test_file', 'Unknown')
                        status_str = status['status'].upper()
                        print(f"Task {task_id} ({os.path.basename(test_file)}): {status_str}")
            
            # Remove completed tasks
            pending_tasks -= tasks_to_remove
            
            # If tasks still pending, wait before next poll
            if pending_tasks:
                time.sleep(self.poll_interval)
        
        # Check for timeout
        if pending_tasks:
            if self.verbose:
                print(f"Warning: {len(pending_tasks)} tasks did not complete within timeout ({self.timeout}s)")
                
            # Get current status for remaining tasks
            for task_id in pending_tasks:
                status = self.client.get_task_status(task_id)
                results[task_id] = {
                    'status': 'timeout',
                    'original_status': status.get('status', 'unknown'),
                    'error': f"Task did not complete within timeout ({self.timeout}s)"
                }
        
        return results
    
    def generate_report(
        self, 
        results: Dict[str, Dict], 
        output_dir: Optional[str] = None,
        formats: List[str] = ['json', 'md']
    ) -> Dict[str, str]:
        """
        Generate report files for CI/CD artifacts.
        
        Args:
            results: Dictionary of task results
            output_dir: Directory to write reports (defaults to current dir)
            formats: List of output formats (json, md, html)
            
        Returns:
            Dictionary mapping format to report file path
        """
        if not output_dir:
            output_dir = os.getcwd()
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_files = {}
        
        # Prepare summary statistics
        total_tasks = len(results)
        status_counts = {
            'completed': 0,
            'failed': 0,
            'cancelled': 0,
            'timeout': 0
        }
        
        for task_id, result in results.items():
            status = result.get('status', 'unknown')
            if status in status_counts:
                status_counts[status] += 1
            else:
                status_counts[status] = 1
        
        # Generate reports in requested formats
        for fmt in formats:
            if fmt.lower() == 'json':
                # JSON Report
                report_file = os.path.join(output_dir, f"test_report_{timestamp}.json")
                with open(report_file, 'w') as f:
                    json.dump({
                        'timestamp': timestamp,
                        'provider': self.provider,
                        'build_id': self.build_id,
                        'repo': self.repo_name,
                        'branch': self.branch,
                        'commit': self.commit_sha,
                        'summary': {
                            'total': total_tasks,
                            'status_counts': status_counts
                        },
                        'results': results
                    }, f, indent=2)
                report_files['json'] = report_file
            
            elif fmt.lower() == 'md':
                # Markdown Report
                report_file = os.path.join(output_dir, f"test_report_{timestamp}.md")
                with open(report_file, 'w') as f:
                    f.write(f"# Distributed Testing Report\n\n")
                    f.write(f"- **Timestamp:** {timestamp}\n")
                    f.write(f"- **Provider:** {self.provider}\n")
                    f.write(f"- **Build ID:** {self.build_id}\n")
                    f.write(f"- **Repository:** {self.repo_name}\n")
                    f.write(f"- **Branch:** {self.branch}\n")
                    f.write(f"- **Commit:** {self.commit_sha}\n\n")
                    
                    f.write(f"## Summary\n\n")
                    f.write(f"- **Total Tasks:** {total_tasks}\n")
                    f.write(f"- **Completed:** {status_counts.get('completed', 0)}\n")
                    f.write(f"- **Failed:** {status_counts.get('failed', 0)}\n")
                    f.write(f"- **Cancelled:** {status_counts.get('cancelled', 0)}\n")
                    f.write(f"- **Timeout:** {status_counts.get('timeout', 0)}\n\n")
                    
                    f.write(f"## Detailed Results\n\n")
                    f.write(f"| Task ID | Test File | Status | Duration | Hardware | Details |\n")
                    f.write(f"|---------|-----------|--------|----------|----------|--------|\n")
                    
                    for task_id, result in results.items():
                        status = result.get('status', 'unknown')
                        config = result.get('config', {})
                        test_file = os.path.basename(config.get('test_file', 'Unknown'))
                        duration = result.get('duration', 'N/A')
                        hardware = ', '.join(result.get('hardware_type', ['Unknown']))
                        
                        # Format details based on status
                        if status == 'completed':
                            details = "✅ Success"
                        elif status == 'failed':
                            error = result.get('error', 'Unknown error')
                            details = f"❌ {error}"
                        elif status == 'timeout':
                            details = "⏱️ Test timed out"
                        else:
                            details = status.capitalize()
                        
                        f.write(f"| {task_id} | {test_file} | {status.upper()} | {duration} | {hardware} | {details} |\n")
                        
                report_files['md'] = report_file
            
            elif fmt.lower() == 'html':
                # HTML Report (simplistic version)
                report_file = os.path.join(output_dir, f"test_report_{timestamp}.html")
                with open(report_file, 'w') as f:
                    f.write("<!DOCTYPE html>\n<html>\n<head>\n")
                    f.write("<title>Distributed Testing Report</title>\n")
                    f.write("<style>\n")
                    f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
                    f.write("table { border-collapse: collapse; width: 100%; }\n")
                    f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
                    f.write("th { background-color: #f2f2f2; }\n")
                    f.write("tr:nth-child(even) { background-color: #f9f9f9; }\n")
                    f.write(".completed { color: green; }\n")
                    f.write(".failed { color: red; }\n")
                    f.write(".timeout { color: orange; }\n")
                    f.write(".cancelled { color: gray; }\n")
                    f.write("</style>\n</head>\n<body>\n")
                    
                    f.write("<h1>Distributed Testing Report</h1>\n")
                    f.write("<p><strong>Timestamp:</strong> " + timestamp + "</p>\n")
                    f.write("<p><strong>Provider:</strong> " + self.provider + "</p>\n")
                    f.write("<p><strong>Build ID:</strong> " + self.build_id + "</p>\n")
                    f.write("<p><strong>Repository:</strong> " + self.repo_name + "</p>\n")
                    f.write("<p><strong>Branch:</strong> " + self.branch + "</p>\n")
                    f.write("<p><strong>Commit:</strong> " + self.commit_sha + "</p>\n")
                    
                    f.write("<h2>Summary</h2>\n")
                    f.write("<ul>\n")
                    f.write(f"<li><strong>Total Tasks:</strong> {total_tasks}</li>\n")
                    f.write(f"<li><strong>Completed:</strong> {status_counts.get('completed', 0)}</li>\n")
                    f.write(f"<li><strong>Failed:</strong> {status_counts.get('failed', 0)}</li>\n")
                    f.write(f"<li><strong>Cancelled:</strong> {status_counts.get('cancelled', 0)}</li>\n")
                    f.write(f"<li><strong>Timeout:</strong> {status_counts.get('timeout', 0)}</li>\n")
                    f.write("</ul>\n")
                    
                    f.write("<h2>Detailed Results</h2>\n")
                    f.write("<table>\n")
                    f.write("<tr><th>Task ID</th><th>Test File</th><th>Status</th><th>Duration</th>")
                    f.write("<th>Hardware</th><th>Details</th></tr>\n")
                    
                    for task_id, result in results.items():
                        status = result.get('status', 'unknown')
                        config = result.get('config', {})
                        test_file = os.path.basename(config.get('test_file', 'Unknown'))
                        duration = result.get('duration', 'N/A')
                        hardware = ', '.join(result.get('hardware_type', ['Unknown']))
                        
                        # Format details based on status
                        if status == 'completed':
                            details = "✅ Success"
                        elif status == 'failed':
                            error = result.get('error', 'Unknown error')
                            details = f"❌ {error}"
                        elif status == 'timeout':
                            details = "⏱️ Test timed out"
                        else:
                            details = status.capitalize()
                        
                        f.write(f"<tr>\n")
                        f.write(f"<td>{task_id}</td>\n")
                        f.write(f"<td>{test_file}</td>\n")
                        f.write(f"<td class='{status}'>{status.upper()}</td>\n")
                        f.write(f"<td>{duration}</td>\n")
                        f.write(f"<td>{hardware}</td>\n")
                        f.write(f"<td>{details}</td>\n")
                        f.write(f"</tr>\n")
                    
                    f.write("</table>\n")
                    f.write("</body>\n</html>")
                    
                report_files['html'] = report_file
        
        if self.verbose:
            for fmt, file_path in report_files.items():
                print(f"Generated {fmt.upper()} report: {file_path}")
        
        return report_files
    
    def report_to_ci_system(self, results: Dict[str, Dict], report_files: Dict[str, str]) -> bool:
        """
        Report results back to the CI/CD system (where supported).
        
        Args:
            results: Dictionary of task results
            report_files: Dictionary of generated report files
            
        Returns:
            True if reporting was successful
        """
        # Calculate success/failure
        failed_count = sum(1 for r in results.values() if r.get('status') not in ('completed',))
        success = failed_count == 0
        
        # GitHub Actions reporting
        if self.provider == 'github' and os.environ.get('GITHUB_ACTIONS') == 'true':
            summary_file = os.environ.get('GITHUB_STEP_SUMMARY')
            if summary_file:
                # Read the markdown report content
                md_report = report_files.get('md')
                if md_report and os.path.exists(md_report):
                    with open(md_report, 'r') as f:
                        report_content = f.read()
                    
                    # Write to GitHub step summary
                    with open(summary_file, 'a') as f:
                        f.write(report_content)
                        
                    if self.verbose:
                        print("Added test results to GitHub step summary")
        
        # GitLab CI reporting
        elif self.provider == 'gitlab' and os.environ.get('GITLAB_CI') == 'true':
            # For GitLab, results are reported through artifacts
            # Just ensure report files are created in the correct location
            if self.verbose:
                print("GitLab CI will collect report artifacts from output directory")
        
        # Jenkins reporting
        elif self.provider == 'jenkins' and os.environ.get('JENKINS_URL'):
            # For Jenkins, results are reported through artifacts and test reports
            # Just ensure report files are created in the correct location
            if self.verbose:
                print("Jenkins will collect report artifacts from output directory")
        
        return success
    
    def run(
        self, 
        test_dir: Optional[str] = None,
        test_pattern: Optional[str] = None,
        test_files: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        report_formats: List[str] = ['json', 'md']
    ) -> int:
        """
        Run the full CI/CD integration workflow.
        
        Args:
            test_dir: Directory to search for tests
            test_pattern: Glob pattern for test files
            test_files: Explicit list of test files
            output_dir: Directory to write reports
            report_formats: List of output formats
            
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        # 1. Discover tests
        discovered_tests = self.discover_tests(test_dir, test_pattern, test_files)
        if not discovered_tests:
            print("No test files discovered.")
            return 1
        
        # 2. Submit tests and gather task IDs
        task_ids = []
        for test_file in discovered_tests:
            # Analyze requirements
            requirements = self.analyze_test_requirements(test_file)
            
            # Submit test
            task_id = self.submit_test(test_file, requirements)
            task_ids.append(task_id)
        
        # 3. Wait for results
        results = self.wait_for_results(task_ids)
        
        # 4. Generate reports
        report_files = self.generate_report(results, output_dir, report_formats)
        
        # 5. Report back to CI/CD system
        success = self.report_to_ci_system(results, report_files)
        
        # 6. Return appropriate exit code
        return 0 if success else 1


def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(description='CI/CD Integration for Distributed Testing Framework')
    
    # Coordinator connection options
    parser.add_argument('--coordinator', required=True, help='Coordinator URL')
    parser.add_argument('--api-key', required=True, help='API key for authentication')
    
    # CI/CD provider options
    parser.add_argument('--provider', default='generic', 
                        choices=['github', 'gitlab', 'jenkins', 'generic'],
                        help='CI/CD provider')
    
    # Test discovery options (mutually exclusive)
    test_group = parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument('--test-dir', help='Directory to search for tests')
    test_group.add_argument('--test-pattern', help='Glob pattern for test files')
    test_group.add_argument('--test-files', nargs='+', help='Explicit list of test files')
    
    # Report options
    parser.add_argument('--output-dir', help='Directory to write reports')
    parser.add_argument('--report-formats', nargs='+', default=['json', 'md'],
                        choices=['json', 'md', 'html'], 
                        help='Report formats to generate')
    
    # Execution options
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Maximum time to wait for test completion (seconds)')
    parser.add_argument('--poll-interval', type=int, default=15,
                        help='How often to poll for results (seconds)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Initialize CI/CD integration
    integration = CICDIntegration(
        coordinator_url=args.coordinator,
        api_key=args.api_key,
        provider=args.provider,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        verbose=args.verbose
    )
    
    # Run the integration workflow
    exit_code = integration.run(
        test_dir=args.test_dir,
        test_pattern=args.test_pattern,
        test_files=args.test_files,
        output_dir=args.output_dir,
        report_formats=args.report_formats
    )
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()