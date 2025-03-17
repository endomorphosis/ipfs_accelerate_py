#!/usr/bin/env python3
"""
Enhanced Example showing how to use the TestResultReporter with the artifact URL retrieval system.

This example demonstrates advanced usage of the artifact URL retrieval system:
1. Parallel URL retrieval with the get_artifact_urls method
2. Automatic URL retrieval in collect_and_upload_artifacts
3. Enhanced reports with artifact URLs
4. PR comments with artifact URLs
5. Custom formatting of artifact URLs in reports
6. Error handling and graceful degradation
7. Cross-provider compatibility
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import argparse
from typing import Dict, Any, Optional, List

# Ensure parent directory is in the path
sys.path.append('/home/barberb/ipfs_accelerate_py/test')

# Import CI system components
from distributed_testing.ci.api_interface import CIProviderFactory, TestRunResult
from distributed_testing.ci.result_reporter import TestResultReporter
from distributed_testing.ci.register_providers import register_all_providers

# Import URL validator if available
try:
    from distributed_testing.ci.url_validator import (
        get_validator,
        validate_url,
        validate_urls,
        generate_health_report,
        close_validator
    )
    URL_VALIDATOR_AVAILABLE = True
except ImportError:
    URL_VALIDATOR_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Colors for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

class EnhancedReporterDemo:
    """
    Enhanced demo for TestResultReporter with artifact URL retrieval.
    """
    
    def __init__(self, args):
        """Initialize the demo with command line arguments."""
        self.args = args
        self.ci_provider = None
        self.reporter = None
        self.report_dir = args.report_dir
        self.artifact_dir = args.artifact_dir
        self.test_run_id = None
        self.artifacts = []
        
        # Create directories if they don't exist
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.artifact_dir, exist_ok=True)
    
    async def initialize(self):
        """Set up the demo environment."""
        print(f"{BOLD}{BLUE}Initializing Enhanced Reporter Artifact URL Demo{RESET}")
        
        # Register all providers
        register_all_providers()
        
        # Create a CI provider based on arguments
        if self.args.provider:
            try:
                # Load provider config from file if specified
                if self.args.config:
                    with open(self.args.config, 'r') as f:
                        config = json.load(f)
                else:
                    # Use minimal config for testing
                    config = {
                        "token": self.args.token or "test_token",
                        "repository": self.args.repository or "test_user/test_repo"
                    }
                
                self.ci_provider = await CIProviderFactory.create_provider(self.args.provider, config)
                print(f"{GREEN}Successfully created {self.args.provider} CI provider{RESET}")
            except Exception as e:
                print(f"{RED}Failed to create CI provider: {str(e)}{RESET}")
                print(f"{YELLOW}Continuing without CI provider for demo purposes{RESET}")
        
        # Create the test result reporter
        self.reporter = TestResultReporter(
            ci_provider=self.ci_provider,
            report_dir=self.report_dir,
            artifact_dir=self.artifact_dir
        )
        
        # Create a unique test run ID
        import uuid
        self.test_run_id = f"enhanced-demo-{uuid.uuid4().hex[:8]}"
        
        # Create a test run in the CI system if a provider is available
        if self.ci_provider:
            try:
                test_run = await self.ci_provider.create_test_run({
                    "name": "Enhanced Reporter Artifact URL Demo",
                    "project": "distributed-testing",
                    "commit_sha": self.args.commit_sha or "abcdef1234567890"
                })
                
                if test_run and "id" in test_run:
                    self.test_run_id = test_run["id"]
                    print(f"{GREEN}Created test run with ID: {self.test_run_id}{RESET}")
            except Exception as e:
                print(f"{RED}Failed to create test run: {str(e)}{RESET}")
    
    async def create_test_artifacts(self):
        """Create test artifacts for the demo."""
        print(f"{BLUE}Creating test artifacts...{RESET}")
        
        # Define artifacts to create
        artifacts_to_create = [
            {"name": "test_results.json", "content": json.dumps({
                "tests": [
                    {"name": "test_api", "result": "pass", "duration": 0.5},
                    {"name": "test_integration", "result": "pass", "duration": 1.2},
                    {"name": "test_performance", "result": "fail", "duration": 2.3,
                     "error": "Performance threshold exceeded"}
                ]
            }, indent=2)},
            {"name": "performance_metrics.csv", "content": 
             "metric,value\nthroughput,125.4\nlatency_ms,7.9\nmemory_mb,256.5\ncpu_utilization,32.1\ndisk_io_mb_s,45.7"},
            {"name": "test.log", "content": 
             "2025-03-16 10:15:32 INFO: Starting tests\n" +
             "2025-03-16 10:15:33 INFO: Running test_api\n" +
             "2025-03-16 10:15:34 INFO: test_api passed\n" +
             "2025-03-16 10:15:35 INFO: Running test_integration\n" +
             "2025-03-16 10:15:36 INFO: test_integration passed\n" +
             "2025-03-16 10:15:37 INFO: Running test_performance\n" +
             "2025-03-16 10:15:39 ERROR: test_performance failed: Performance threshold exceeded\n" +
             "2025-03-16 10:15:40 INFO: Tests completed\n"},
            {"name": "coverage_report.html", "content":
             "<!DOCTYPE html>\n<html>\n<head>\n  <title>Coverage Report</title>\n</head>\n<body>\n" +
             "  <h1>Coverage Report</h1>\n  <p>Overall coverage: 87.5%</p>\n" +
             "  <table>\n    <tr><th>Module</th><th>Coverage</th></tr>\n" +
             "    <tr><td>core</td><td>92.3%</td></tr>\n" +
             "    <tr><td>api</td><td>89.7%</td></tr>\n" +
             "    <tr><td>utils</td><td>78.4%</td></tr>\n  </table>\n" +
             "</body>\n</html>"},
            {"name": "system_info.txt", "content":
             "Platform: Linux\nPython: 3.9.10\nCPU Cores: 4\nMemory: 8GB\n" +
             "Disk: 50GB\nDistributed Testing Version: 1.5.0\n"},
        ]
        
        # Create the artifacts
        self.artifact_files = []
        for artifact in artifacts_to_create:
            file_path = os.path.join(self.artifact_dir, artifact["name"])
            with open(file_path, "w") as f:
                f.write(artifact["content"])
            self.artifact_files.append(file_path)
            print(f"  Created artifact: {artifact['name']}")
        
        print(f"{GREEN}Created {len(self.artifact_files)} test artifacts{RESET}")
    
    async def demonstrate_url_validation(self):
        """Demonstrate the URL validation functionality."""
        print(f"\n{BOLD}{BLUE}Demonstrating URL Validation{RESET}")
        
        if not URL_VALIDATOR_AVAILABLE:
            print(f"{YELLOW}Skipping URL validation demo (URL validator not available){RESET}")
            return False
        
        # Create some example URLs to validate (both valid and invalid)
        example_urls = [
            "https://github.com",
            "https://gitlab.com",
            "https://example.com/non-existent-page",
            "https://invalid-domain-that-does-not-exist.example"
        ]
        
        print(f"Validating {len(example_urls)} URLs...")
        validation_results = await validate_urls(example_urls)
        
        # Display results
        for url, (is_valid, status_code, error_message) in validation_results.items():
            status = f"{GREEN}Valid{RESET}" if is_valid else f"{RED}Invalid{RESET}"
            print(f"  URL: {url}")
            print(f"    Status: {status} (Code: {status_code or 'N/A'})")
            if error_message:
                print(f"    Error: {error_message}")
            print()
        
        # Generate a health report
        print("Generating URL health report...")
        report = await generate_health_report(format="markdown")
        
        # Save the report to a file
        report_path = os.path.join(self.report_dir, "url_health_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"URL health report saved to: {report_path}")
        
        # Generate HTML report
        html_report = await generate_health_report(format="html")
        html_report_path = os.path.join(self.report_dir, "url_health_report.html")
        with open(html_report_path, "w") as f:
            f.write(html_report)
        
        print(f"HTML health report saved to: {html_report_path}")
        
        return True
        
    async def demonstrate_bulk_url_retrieval(self):
        """Demonstrate the bulk URL retrieval feature."""
        print(f"\n{BOLD}{BLUE}Demonstrating Bulk URL Retrieval{RESET}")
        
        if not self.ci_provider:
            print(f"{YELLOW}Skipping bulk URL retrieval demo (no CI provider available){RESET}")
            return False
        
        # Upload artifacts manually
        print("Uploading artifacts manually...")
        artifact_names = []
        for file_path in self.artifact_files:
            artifact_name = os.path.basename(file_path)
            artifact_names.append(artifact_name)
            
            success = await self.ci_provider.upload_artifact(
                test_run_id=self.test_run_id,
                artifact_path=file_path,
                artifact_name=artifact_name
            )
            
            if success:
                print(f"  Uploaded: {artifact_name}")
            else:
                print(f"{RED}  Failed to upload: {artifact_name}{RESET}")
        
        # Sequential retrieval for comparison
        print("\nRetrieving URLs sequentially (for comparison)...")
        sequential_start = time.time()
        
        sequential_urls = {}
        for name in artifact_names:
            try:
                url = await self.ci_provider.get_artifact_url(self.test_run_id, name)
                sequential_urls[name] = url
                print(f"  Retrieved URL for: {name}")
            except Exception as e:
                print(f"{RED}  Error retrieving URL for {name}: {str(e)}{RESET}")
                sequential_urls[name] = None
        
        sequential_time = time.time() - sequential_start
        print(f"{GREEN}Sequential retrieval completed in {sequential_time:.4f} seconds{RESET}")
        
        # Parallel retrieval using get_artifact_urls
        print("\nRetrieving URLs in parallel...")
        parallel_start = time.time()
        
        parallel_urls = await self.reporter.get_artifact_urls(
            test_run_id=self.test_run_id,
            artifact_names=artifact_names
        )
        
        parallel_time = time.time() - parallel_start
        print(f"{GREEN}Parallel retrieval completed in {parallel_time:.4f} seconds{RESET}")
        
        # Calculate speedup
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        print(f"\n{BOLD}Performance comparison:{RESET}")
        print(f"  Sequential: {sequential_time:.4f} seconds")
        print(f"  Parallel:   {parallel_time:.4f} seconds")
        print(f"  Speedup:    {speedup:.2f}x")
        
        # Verify results match
        mismatches = 0
        for name in artifact_names:
            if sequential_urls[name] != parallel_urls[name]:
                print(f"{RED}URL mismatch for {name}:{RESET}")
                print(f"  Sequential: {sequential_urls[name]}")
                print(f"  Parallel:   {parallel_urls[name]}")
                mismatches += 1
        
        if mismatches == 0:
            print(f"{GREEN}All URLs match between sequential and parallel retrieval{RESET}")
        
        return len(parallel_urls) > 0
    
    async def demonstrate_automatic_url_retrieval(self):
        """Demonstrate the automatic URL retrieval in collect_and_upload_artifacts."""
        print(f"\n{BOLD}{BLUE}Demonstrating Automatic URL Retrieval{RESET}")
        
        if not self.ci_provider:
            print(f"{YELLOW}Skipping automatic URL retrieval demo (no CI provider available){RESET}")
            return False
        
        # Collect and upload artifacts with automatic URL retrieval and validation
        print("Collecting and uploading artifacts with automatic URL retrieval and validation...")
        self.artifacts = await self.reporter.collect_and_upload_artifacts(
            test_run_id=self.test_run_id,
            artifact_patterns=self.artifact_files,
            validate_urls=URL_VALIDATOR_AVAILABLE,
            include_health_info=URL_VALIDATOR_AVAILABLE
        )
        
        # Display the artifacts with URLs
        if self.artifacts:
            print(f"{GREEN}Successfully collected and uploaded {len(self.artifacts)} artifacts{RESET}")
            print("\nArtifacts with URLs:")
            
            for artifact in self.artifacts:
                name = artifact.get("name", "Unknown")
                size = artifact.get("size_bytes", 0)
                url = artifact.get("url", "No URL")
                
                # Show validation status if available
                validation_info = ""
                if URL_VALIDATOR_AVAILABLE:
                    if "url_validated" in artifact and artifact["url_validated"]:
                        is_valid = artifact.get("url_valid", False)
                        validation_status = f"{GREEN}Valid{RESET}" if is_valid else f"{RED}Invalid{RESET}"
                        validation_info = f" - Status: {validation_status}"
                        
                        # Show error if not valid
                        if not is_valid and "url_validation_error" in artifact:
                            validation_info += f" ({artifact['url_validation_error']})"
                        
                        # Show health info if available
                        if "url_health" in artifact:
                            availability = artifact["url_health"].get("availability", 0)
                            validation_info += f" - Availability: {availability:.1f}%"
                
                size_str = self._format_size(size)
                print(f"  {name} ({size_str}): {url}{validation_info}")
            
            return True
        else:
            print(f"{RED}No artifacts were collected and uploaded{RESET}")
            return False
    
    async def demonstrate_enhanced_reports(self):
        """Demonstrate enhanced reports with artifact URLs."""
        print(f"\n{BOLD}{BLUE}Demonstrating Enhanced Reports with Artifact URLs{RESET}")
        
        # Create a test result
        test_result = TestRunResult(
            test_run_id=self.test_run_id,
            status="failure",
            total_tests=10,
            passed_tests=8,
            failed_tests=1,
            skipped_tests=1,
            duration_seconds=45.6,
            metadata={
                "pr_number": self.args.pr_number or "123",
                "performance_metrics": {
                    "average_throughput": 125.4,
                    "average_latency_ms": 7.9,
                    "memory_usage_mb": 256,
                    "cpu_utilization_percent": 32.1,
                    "response_time_p95_ms": 15.3
                },
                "environment": {
                    "platform": "linux",
                    "python_version": "3.9.10",
                    "cpu_cores": 4,
                    "memory_gb": 8,
                    "disk_space_gb": 50
                },
                "test_details": {
                    "failed_tests": [
                        {
                            "name": "test_performance",
                            "error": "Performance threshold exceeded",
                            "duration_seconds": 2.3
                        }
                    ],
                    "passed_tests": [
                        {"name": "test_api", "duration_seconds": 0.5},
                        {"name": "test_integration", "duration_seconds": 1.2},
                        {"name": "test_utils", "duration_seconds": 0.8},
                        {"name": "test_core", "duration_seconds": 1.5},
                        {"name": "test_validation", "duration_seconds": 1.1},
                        {"name": "test_reporting", "duration_seconds": 0.9},
                        {"name": "test_database", "duration_seconds": 1.7},
                        {"name": "test_api_client", "duration_seconds": 0.6}
                    ]
                }
            }
        )
        
        # Add artifacts to test result metadata if available
        if self.artifacts:
            test_result.metadata["artifacts"] = self.artifacts
        
        # Generate reports
        print("Generating reports with artifact URLs...")
        report_files = await self.reporter.report_test_result(
            test_result,
            formats=["markdown", "html", "json"]
        )
        
        # Display the report file paths
        if report_files:
            print(f"{GREEN}Successfully generated reports:{RESET}")
            for fmt, file_path in report_files.items():
                print(f"  {fmt.upper()} report: {file_path}")
            
            # Add report artifacts to the list if available
            if "artifacts" in test_result.metadata:
                report_artifacts = [a for a in test_result.metadata["artifacts"] if a.get("type") == "report"]
                if report_artifacts:
                    print(f"\n{GREEN}Report artifacts with URLs:{RESET}")
                    for artifact in report_artifacts:
                        name = artifact.get("name", "Unknown")
                        url = artifact.get("url", "No URL")
                        print(f"  {name}: {url}")
            
            # Read and display the artifacts section from the markdown report
            if "markdown" in report_files:
                with open(report_files["markdown"], "r") as f:
                    content = f.read()
                    
                    artifacts_section = self._extract_section(content, "## Artifacts", "##")
                    if artifacts_section:
                        print(f"\n{BOLD}Excerpt from Markdown report:{RESET}")
                        print(artifacts_section)
            
            return True
        else:
            print(f"{RED}No reports were generated{RESET}")
            return False
    
    async def demonstrate_pr_comments(self):
        """Demonstrate PR comments with artifact URLs."""
        print(f"\n{BOLD}{BLUE}Demonstrating PR Comments with Artifact URLs{RESET}")
        
        if not self.ci_provider:
            print(f"{YELLOW}Skipping PR comment demo (no CI provider available){RESET}")
            return False
        
        # Create a test result with PR number
        test_result = TestRunResult(
            test_run_id=self.test_run_id,
            status="failure",
            total_tests=10,
            passed_tests=8,
            failed_tests=1,
            skipped_tests=1,
            duration_seconds=45.6,
            metadata={
                "pr_number": self.args.pr_number or "123",
                "performance_metrics": {
                    "average_throughput": 125.4,
                    "average_latency_ms": 7.9
                }
            }
        )
        
        # Add artifacts to test result metadata if available
        if self.artifacts:
            test_result.metadata["artifacts"] = self.artifacts
        
        # Mock the add_pr_comment method to capture the comment
        pr_comment = None
        
        original_add_pr_comment = None
        if hasattr(self.ci_provider, 'add_pr_comment'):
            original_add_pr_comment = self.ci_provider.add_pr_comment
            
            async def mock_add_pr_comment(pr_number, comment):
                nonlocal pr_comment
                pr_comment = comment
                print(f"{GREEN}PR comment would be added to PR #{pr_number}{RESET}")
                return True
            
            self.ci_provider.add_pr_comment = mock_add_pr_comment
        
        # Generate report
        print("Generating report with PR comment...")
        await self.reporter.report_test_result(
            test_result,
            formats=["markdown"]
        )
        
        # Restore original method if it was replaced
        if original_add_pr_comment:
            self.ci_provider.add_pr_comment = original_add_pr_comment
        
        # Display the PR comment if it was generated
        if pr_comment:
            print(f"\n{BOLD}PR comment content:{RESET}")
            print(pr_comment[:500] + "..." if len(pr_comment) > 500 else pr_comment)
            
            # Check for artifacts section
            artifacts_section = self._extract_section(pr_comment, "## Artifacts", "##")
            if artifacts_section:
                print(f"\n{BOLD}Artifacts section from PR comment:{RESET}")
                print(artifacts_section)
            
            return True
        else:
            print(f"{RED}No PR comment was generated{RESET}")
            return False
    
    async def demonstrate_error_handling(self):
        """Demonstrate error handling in URL retrieval."""
        print(f"\n{BOLD}{BLUE}Demonstrating Error Handling in URL Retrieval{RESET}")
        
        if not self.ci_provider:
            print(f"{YELLOW}Skipping error handling demo (no CI provider available){RESET}")
            return False
        
        # Try to get URLs for non-existent artifacts
        print("Retrieving URLs for non-existent artifacts...")
        urls = await self.reporter.get_artifact_urls(
            test_run_id=self.test_run_id,
            artifact_names=["non_existent_1.json", "non_existent_2.log"]
        )
        
        # Display the results
        if urls:
            print(f"{GREEN}URL retrieval completed with graceful error handling:{RESET}")
            for name, url in urls.items():
                status = f"{RED}Not found{RESET}" if url is None else f"{GREEN}Found{RESET}"
                print(f"  {name}: {status}")
            
            # Verify that we have entries for all artifacts
            if len(urls) == 2 and all(name in urls for name in ["non_existent_1.json", "non_existent_2.log"]):
                print(f"{GREEN}All artifact entries present with appropriate null values{RESET}")
                return True
        
        print(f"{RED}Error handling did not work as expected{RESET}")
        return False
    
    async def demonstrate_dtf_integration(self):
        """Demonstrate integration with the Distributed Testing Framework."""
        print(f"\n{BOLD}{BLUE}Demonstrating Integration with Distributed Testing Framework{RESET}")
        
        # Check if Distributed Testing Framework is available
        try:
            from distributed_testing.coordinator import DistributedTestingCoordinator
        except ImportError:
            print(f"{YELLOW}Skipping DTF integration (coordinator not available){RESET}")
            return False
        
        # Create a temporary directory for the coordinator database
        import tempfile
        db_dir = tempfile.mkdtemp()
        db_path = os.path.join(db_dir, "coordinator.db")
        
        try:
            # Create a coordinator
            print("Creating coordinator with batch processing...")
            coordinator = DistributedTestingCoordinator(
                db_path=db_path,
                enable_batch_processing=True,
                batch_size_limit=5
            )
            
            # Create test run ID
            import uuid
            test_run_id = f"dtf-demo-{uuid.uuid4().hex[:8]}"
            
            # Register a task with the coordinator
            print("Registering task with coordinator...")
            task_id = await coordinator.register_task({
                "name": "DTF Integration Demo",
                "type": "test",
                "priority": 1,
                "parameters": {
                    "test_file": "test_integration.py",
                    "timeout": 30
                },
                "metadata": {
                    "test_run_id": test_run_id
                }
            })
            
            print(f"{GREEN}Registered task with ID: {task_id}{RESET}")
            
            # Update task status to running
            await coordinator.update_task_status(task_id, "running")
            
            # Upload artifacts and collect URLs
            if self.ci_provider:
                print("Uploading artifacts with automatic URL retrieval and validation...")
                artifacts = await self.reporter.collect_and_upload_artifacts(
                    test_run_id=test_run_id,
                    artifact_patterns=self.artifact_files,
                    validate_urls=URL_VALIDATOR_AVAILABLE,
                    include_health_info=URL_VALIDATOR_AVAILABLE
                )
                
                if artifacts:
                    print(f"{GREEN}Uploaded {len(artifacts)} artifacts with URLs:{RESET}")
                    for artifact in artifacts:
                        name = artifact.get("name", "Unknown")
                        url = artifact.get("url", "No URL")
                        
                        # Display validation info if available
                        validation_info = ""
                        if URL_VALIDATOR_AVAILABLE:
                            if "url_validated" in artifact and artifact["url_validated"]:
                                is_valid = artifact.get("url_valid", False)
                                validation_status = f"{GREEN}Valid{RESET}" if is_valid else f"{RED}Invalid{RESET}"
                                validation_info = f" - Status: {validation_status}"
                                
                                # Show error if not valid
                                if not is_valid and "url_validation_error" in artifact:
                                    validation_info += f" ({artifact['url_validation_error']})"
                                
                                # Show health info if available
                                if "url_health" in artifact:
                                    availability = artifact["url_health"].get("availability", 0)
                                    validation_info += f" - Availability: {availability:.1f}%"
                        
                        print(f"  {name}: {url}{validation_info}")
            else:
                print(f"{YELLOW}Skipping artifact upload (no CI provider){RESET}")
                artifacts = []
            
            # Create a test result with artifact URLs
            print("Creating test result with artifact URLs...")
            test_result = TestRunResult(
                test_run_id=test_run_id,
                status="success",
                total_tests=10,
                passed_tests=9,
                failed_tests=1,
                skipped_tests=0,
                duration_seconds=15.5,
                metadata={
                    "task_id": task_id,
                    "pr_number": self.args.pr_number or "123",
                    "performance_metrics": {
                        "average_throughput": 125.4,
                        "average_latency_ms": 7.9,
                        "memory_usage_mb": 256
                    }
                }
            )
            
            # Add artifacts to test result metadata
            if artifacts:
                test_result.metadata["artifacts"] = artifacts
            
            # Send test result to coordinator
            print("Sending test result to coordinator...")
            await coordinator.process_test_result(test_result)
            
            # Update task status to completed
            await coordinator.update_task_status(task_id, "completed", {
                "status": "success",
                "total_tests": test_result.total_tests,
                "passed_tests": test_result.passed_tests,
                "failed_tests": test_result.failed_tests,
                "skipped_tests": test_result.skipped_tests,
                "duration_seconds": test_result.duration_seconds
            })
            
            # Get task details from coordinator
            print("Retrieving task details from coordinator...")
            task = await coordinator.get_task(task_id)
            
            # Display task details
            print(f"\n{BOLD}Task Details from Coordinator:{RESET}")
            print(f"  Task ID: {task['id']}")
            print(f"  Name: {task['name']}")
            print(f"  Status: {task['status']}")
            
            # Display result metadata
            if "result_metadata" in task:
                print(f"\n{BOLD}Result Metadata:{RESET}")
                
                metrics = task["result_metadata"].get("performance_metrics", {})
                if metrics:
                    print("  Performance Metrics:")
                    for key, value in metrics.items():
                        print(f"    {key}: {value}")
                
                # Display artifacts with URLs
                if "artifacts" in task["result_metadata"]:
                    task_artifacts = task["result_metadata"]["artifacts"]
                    print(f"\n  {BOLD}Artifacts from Coordinator:{RESET}")
                    
                    for artifact in task_artifacts:
                        name = artifact.get("name", "Unknown")
                        url = artifact.get("url", "No URL")
                        size = artifact.get("size_bytes", 0)
                        size_str = self._format_size(size)
                        print(f"    {name} ({size_str}): {url}")
            
            # Generate a dashboard report
            print("\nGenerating dashboard report from coordinator data...")
            
            dashboard_items = await coordinator.get_dashboard_items(limit=5)
            
            if dashboard_items:
                print(f"\n{BOLD}Dashboard Report:{RESET}")
                for item in dashboard_items:
                    status_color = GREEN if item.get("status") == "success" else RED
                    print(f"  Task: {item.get('name')} - Status: {status_color}{item.get('status')}{RESET}")
                    
                    # If this task has URLs, display them
                    if item.get("id") == task_id and "artifacts" in task["result_metadata"]:
                        print("    Artifacts:")
                        for artifact in task["result_metadata"]["artifacts"]:
                            name = artifact.get("name", "Unknown")
                            url = artifact.get("url", "No URL")
                            
                            # Display validation info if available
                            validation_info = ""
                            if URL_VALIDATOR_AVAILABLE:
                                if "url_validated" in artifact and artifact["url_validated"]:
                                    is_valid = artifact.get("url_valid", False)
                                    validation_status = f"{GREEN}Valid{RESET}" if is_valid else f"{RED}Invalid{RESET}"
                                    validation_info = f" - Status: {validation_status}"
                                    
                                    # Show health info if available
                                    if "url_health" in artifact:
                                        availability = artifact["url_health"].get("availability", 0)
                                        validation_info += f" - Availability: {availability:.1f}%"
                            
                            print(f"      {name}: {url}{validation_info}")
            
            print(f"\n{GREEN}DTF integration demonstration completed successfully{RESET}")
            return True
            
        except Exception as e:
            print(f"{RED}Error in DTF integration: {str(e)}{RESET}")
            return False
            
        finally:
            # Clean up temporary directory
            import shutil
            if os.path.exists(db_dir):
                shutil.rmtree(db_dir)

    async def run_demo(self):
        """Run the complete demonstration."""
        await self.initialize()
        await self.create_test_artifacts()
        
        # Run all demonstrations
        results = {}
        
        results["url_validation"] = await self.demonstrate_url_validation()
        results["bulk_url_retrieval"] = await self.demonstrate_bulk_url_retrieval()
        results["automatic_url_retrieval"] = await self.demonstrate_automatic_url_retrieval()
        results["enhanced_reports"] = await self.demonstrate_enhanced_reports()
        results["pr_comments"] = await self.demonstrate_pr_comments()
        results["error_handling"] = await self.demonstrate_error_handling()
        results["dtf_integration"] = await self.demonstrate_dtf_integration()
        
        # Display summary
        print(f"\n{BOLD}{BLUE}Demo Summary{RESET}")
        
        for demo, success in results.items():
            status = f"{GREEN}Completed{RESET}" if success else f"{RED}Skipped/Failed{RESET}"
            print(f"  {demo.replace('_', ' ').title()}: {status}")
        
        # Clean up
        if self.ci_provider:
            await self.ci_provider.close()
            
        # Close the URL validator if available
        if URL_VALIDATOR_AVAILABLE:
            try:
                await close_validator()
                print("URL validator closed")
            except Exception as e:
                print(f"{RED}Error closing URL validator: {str(e)}{RESET}")
        
        print(f"\n{BOLD}{GREEN}Enhanced Reporter Artifact URL Demo Completed{RESET}")
    
    def _extract_section(self, content, start_marker, end_marker):
        """Extract a section from a string between two markers."""
        start_idx = content.find(start_marker)
        if start_idx == -1:
            return None
        
        # Find the next occurrence of end_marker after start_idx
        end_idx = content.find(end_marker, start_idx + len(start_marker))
        if end_idx == -1:
            # If there's no end marker, take everything to the end
            return content[start_idx:]
        
        # Return the section between start and end markers
        return content[start_idx:end_idx].strip()
    
    def _format_size(self, size_bytes):
        """Format file size in bytes to a human-readable string."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes/1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes/(1024*1024):.2f} MB"
        else:
            return f"{size_bytes/(1024*1024*1024):.2f} GB"


def main():
    """Parse arguments and run the demo."""
    parser = argparse.ArgumentParser(
        description="Enhanced Test Reporter with Artifact URL Retrieval Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # CI provider options
    parser.add_argument("--provider", choices=["github", "gitlab", "jenkins", "circleci", "azure", "bitbucket", "teamcity", "travis"], 
                        help="CI provider to use")
    parser.add_argument("--token", help="Token for CI provider authentication")
    parser.add_argument("--repository", help="Repository name (e.g. user/repo)")
    parser.add_argument("--config", help="Path to CI provider configuration file")
    parser.add_argument("--commit-sha", help="Commit SHA for test run")
    parser.add_argument("--pr-number", help="Pull request number for comments")
    
    # Reporter options
    parser.add_argument("--report-dir", default="./reports", help="Directory for reports")
    parser.add_argument("--artifact-dir", default="./artifacts", help="Directory for artifacts")
    
    args = parser.parse_args()
    
    # Run the demo
    demo = EnhancedReporterDemo(args)
    asyncio.run(demo.run_demo())


if __name__ == "__main__":
    main()