#!/usr/bin/env python3
"""
Standardized Test Result Reporter for CI/CD Integrations

This module provides a standardized way to report test results to CI/CD systems
using different formatters (Markdown, HTML, JSON) and integration with CI providers.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set

import aiohttp

# Import CI interfaces
from ci.api_interface import CIProviderInterface, TestRunResult, CIProviderFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResultFormatter:
    """
    Formats test results for different output formats.
    """
    
    @staticmethod
    def format_markdown(result: TestRunResult) -> str:
        """
        Format test results as Markdown.
        
        Args:
            result: Test result data
            
        Returns:
            Formatted Markdown string
        """
        # Create header with summary
        header = f"# Test Run Report: {result.test_run_id}\n\n"
        summary = f"**Status:** {result.status.upper()}\n\n"
        stats = (
            f"**Summary:**\n"
            f"- Total Tests: {result.total_tests}\n"
            f"- Passed: {result.passed_tests} ({_percentage(result.passed_tests, result.total_tests)}%)\n"
            f"- Failed: {result.failed_tests} ({_percentage(result.failed_tests, result.total_tests)}%)\n"
            f"- Skipped: {result.skipped_tests} ({_percentage(result.skipped_tests, result.total_tests)}%)\n"
            f"- Duration: {_format_duration(result.duration_seconds)}\n\n"
        )
        
        # Add test details if available
        details = ""
        if "test_details" in result.metadata:
            details += "## Test Details\n\n"
            
            # Add failed tests first if any
            if result.failed_tests > 0 and "failed_tests" in result.metadata:
                details += "### Failed Tests\n\n"
                details += "| Test | Error | Duration |\n"
                details += "|------|-------|----------|\n"
                
                for test in result.metadata["failed_tests"]:
                    test_name = test.get("name", "Unknown")
                    error = test.get("error", "Unknown error")
                    # Truncate long error messages
                    error = error if len(error) < 50 else error[:47] + "..."
                    duration = _format_duration(test.get("duration_seconds", 0))
                    details += f"| {test_name} | {error} | {duration} |\n"
                
                details += "\n"
            
            # Add passed tests summary
            if result.passed_tests > 0:
                details += f"### Passed Tests: {result.passed_tests}\n\n"
                
                # List passed tests if available and not too many
                if "passed_tests" in result.metadata and len(result.metadata["passed_tests"]) <= 20:
                    details += "| Test | Duration |\n"
                    details += "|------|----------|\n"
                    
                    for test in result.metadata["passed_tests"]:
                        test_name = test.get("name", "Unknown")
                        duration = _format_duration(test.get("duration_seconds", 0))
                        details += f"| {test_name} | {duration} |\n"
                    
                    details += "\n"
        
        # Add performance metrics if available
        performance = ""
        if "performance_metrics" in result.metadata:
            metrics = result.metadata["performance_metrics"]
            performance = "## Performance Metrics\n\n"
            
            if isinstance(metrics, dict):
                performance += "| Metric | Value |\n"
                performance += "|--------|-------|\n"
                
                for metric, value in metrics.items():
                    # Format the metric name for better readability
                    metric_name = metric.replace("_", " ").title()
                    
                    # Format the value based on its type
                    if isinstance(value, float):
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    
                    performance += f"| {metric_name} | {formatted_value} |\n"
            
            performance += "\n"
        
        # Add execution environment information if available
        environment = ""
        if "environment" in result.metadata:
            env_data = result.metadata["environment"]
            environment = "## Execution Environment\n\n"
            
            if isinstance(env_data, dict):
                environment += "| Parameter | Value |\n"
                environment += "|-----------|-------|\n"
                
                for param, value in env_data.items():
                    environment += f"| {param} | {value} |\n"
            
            environment += "\n"
        
        # Add artifacts if available
        artifacts = ""
        if "artifacts" in result.metadata and result.metadata["artifacts"]:
            artifacts = "## Artifacts\n\n"
            
            for artifact in result.metadata["artifacts"]:
                name = artifact.get("name", "Unknown")
                url = artifact.get("url", "#")
                size = artifact.get("size_bytes", 0)
                size_str = _format_size(size)
                
                artifacts += f"- [{name}]({url}) ({size_str})\n"
            
            artifacts += "\n"
        
        # Add timestamp
        timestamp = f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        # Combine all sections
        return f"{header}{summary}{stats}{performance}{details}{environment}{artifacts}{timestamp}"
    
    @staticmethod
    def format_html(result: TestRunResult) -> str:
        """
        Format test results as HTML.
        
        Args:
            result: Test result data
            
        Returns:
            Formatted HTML string
        """
        # Start with HTML boilerplate and styling
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Run Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .status-success {
            color: #27ae60;
            font-weight: bold;
        }
        .status-failure {
            color: #e74c3c;
            font-weight: bold;
        }
        .status-skipped {
            color: #f39c12;
        }
        .summary-box {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .summary-item {
            flex: 1;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .total {
            background-color: #f0f0f0;
        }
        .passed {
            background-color: #d5f5e3;
        }
        .failed {
            background-color: #fadbd8;
        }
        .skipped {
            background-color: #fef9e7;
        }
        .duration {
            background-color: #ebf5fb;
        }
        .progress-bar-container {
            height: 20px;
            background-color: #f1f1f1;
            border-radius: 10px;
            margin-bottom: 20px;
            overflow: hidden;
        }
        .progress-bar {
            height: 100%;
            border-radius: 10px;
            display: flex;
        }
        .progress-passed {
            background-color: #27ae60;
        }
        .progress-failed {
            background-color: #e74c3c;
        }
        .progress-skipped {
            background-color: #f39c12;
        }
        .footer {
            margin-top: 30px;
            font-size: 0.8em;
            color: #7f8c8d;
            text-align: center;
        }
    </style>
</head>
<body>
"""
        
        # Add header and summary
        status_class = "status-success" if result.status.lower() == "success" else "status-failure"
        html += f"<h1>Test Run Report: {result.test_run_id}</h1>\n"
        html += f"<p><span class=\"{status_class}\">Status: {result.status.upper()}</span></p>\n"
        
        # Add summary boxes
        html += "<div class=\"summary-box\">\n"
        html += f"  <div class=\"summary-item total\"><h3>Total Tests</h3><p>{result.total_tests}</p></div>\n"
        html += f"  <div class=\"summary-item passed\"><h3>Passed</h3><p>{result.passed_tests} ({_percentage(result.passed_tests, result.total_tests)}%)</p></div>\n"
        html += f"  <div class=\"summary-item failed\"><h3>Failed</h3><p>{result.failed_tests} ({_percentage(result.failed_tests, result.total_tests)}%)</p></div>\n"
        html += f"  <div class=\"summary-item skipped\"><h3>Skipped</h3><p>{result.skipped_tests} ({_percentage(result.skipped_tests, result.total_tests)}%)</p></div>\n"
        html += f"  <div class=\"summary-item duration\"><h3>Duration</h3><p>{_format_duration(result.duration_seconds)}</p></div>\n"
        html += "</div>\n"
        
        # Add progress bar
        if result.total_tests > 0:
            passed_pct = result.passed_tests / result.total_tests * 100
            failed_pct = result.failed_tests / result.total_tests * 100
            skipped_pct = result.skipped_tests / result.total_tests * 100
            
            html += "<div class=\"progress-bar-container\">\n"
            html += "  <div class=\"progress-bar\">\n"
            
            if passed_pct > 0:
                html += f"    <div class=\"progress-passed\" style=\"width: {passed_pct}%\"></div>\n"
            if failed_pct > 0:
                html += f"    <div class=\"progress-failed\" style=\"width: {failed_pct}%\"></div>\n"
            if skipped_pct > 0:
                html += f"    <div class=\"progress-skipped\" style=\"width: {skipped_pct}%\"></div>\n"
            
            html += "  </div>\n"
            html += "</div>\n"
        
        # Add performance metrics if available
        if "performance_metrics" in result.metadata:
            metrics = result.metadata["performance_metrics"]
            html += "<h2>Performance Metrics</h2>\n"
            
            if isinstance(metrics, dict):
                html += "<table>\n"
                html += "  <tr><th>Metric</th><th>Value</th></tr>\n"
                
                for metric, value in metrics.items():
                    # Format the metric name for better readability
                    metric_name = metric.replace("_", " ").title()
                    
                    # Format the value based on its type
                    if isinstance(value, float):
                        formatted_value = f"{value:.2f}"
                    else:
                        formatted_value = str(value)
                    
                    html += f"  <tr><td>{metric_name}</td><td>{formatted_value}</td></tr>\n"
                
                html += "</table>\n"
        
        # Add test details if available
        if "test_details" in result.metadata:
            html += "<h2>Test Details</h2>\n"
            
            # Add failed tests first if any
            if result.failed_tests > 0 and "failed_tests" in result.metadata:
                html += "<h3>Failed Tests</h3>\n"
                html += "<table>\n"
                html += "  <tr><th>Test</th><th>Error</th><th>Duration</th></tr>\n"
                
                for test in result.metadata["failed_tests"]:
                    test_name = test.get("name", "Unknown")
                    error = test.get("error", "Unknown error")
                    # Truncate long error messages
                    error = error if len(error) < 100 else error[:97] + "..."
                    duration = _format_duration(test.get("duration_seconds", 0))
                    html += f"  <tr><td>{test_name}</td><td>{error}</td><td>{duration}</td></tr>\n"
                
                html += "</table>\n"
            
            # Add passed tests summary
            if result.passed_tests > 0:
                html += f"<h3>Passed Tests: {result.passed_tests}</h3>\n"
                
                # List passed tests if available and not too many
                if "passed_tests" in result.metadata and len(result.metadata["passed_tests"]) <= 20:
                    html += "<table>\n"
                    html += "  <tr><th>Test</th><th>Duration</th></tr>\n"
                    
                    for test in result.metadata["passed_tests"]:
                        test_name = test.get("name", "Unknown")
                        duration = _format_duration(test.get("duration_seconds", 0))
                        html += f"  <tr><td>{test_name}</td><td>{duration}</td></tr>\n"
                    
                    html += "</table>\n"
        
        # Add execution environment information if available
        if "environment" in result.metadata:
            env_data = result.metadata["environment"]
            html += "<h2>Execution Environment</h2>\n"
            
            if isinstance(env_data, dict):
                html += "<table>\n"
                html += "  <tr><th>Parameter</th><th>Value</th></tr>\n"
                
                for param, value in env_data.items():
                    html += f"  <tr><td>{param}</td><td>{value}</td></tr>\n"
                
                html += "</table>\n"
        
        # Add artifacts if available
        if "artifacts" in result.metadata and result.metadata["artifacts"]:
            html += "<h2>Artifacts</h2>\n"
            html += "<ul>\n"
            
            for artifact in result.metadata["artifacts"]:
                name = artifact.get("name", "Unknown")
                url = artifact.get("url", "#")
                size = artifact.get("size_bytes", 0)
                size_str = _format_size(size)
                
                html += f"  <li><a href=\"{url}\">{name}</a> ({size_str})</li>\n"
            
            html += "</ul>\n"
        
        # Add footer with timestamp
        html += "<div class=\"footer\">\n"
        html += f"  Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        html += "</div>\n"
        
        # Close HTML tags
        html += "</body>\n</html>"
        
        return html
    
    @staticmethod
    def format_json(result: TestRunResult) -> str:
        """
        Format test results as JSON.
        
        Args:
            result: Test result data
            
        Returns:
            Formatted JSON string
        """
        return json.dumps(result.to_dict(), indent=2)


class TestResultReporter:
    """
    Reports test results to various CI/CD systems and generates reports.
    """
    
    def __init__(
        self,
        ci_provider: Optional[CIProviderInterface] = None,
        report_dir: Optional[str] = None,
        artifact_dir: Optional[str] = None
    ):
        """
        Initialize the test result reporter.
        
        Args:
            ci_provider: CI provider interface
            report_dir: Directory to save reports
            artifact_dir: Directory to save artifacts
        """
        self.ci_provider = ci_provider
        self.report_dir = report_dir or os.environ.get("REPORT_DIR", "reports")
        self.artifact_dir = artifact_dir or os.environ.get("ARTIFACT_DIR", "artifacts")
        
        # Create directories if they don't exist
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.artifact_dir, exist_ok=True)
        
        logger.info(f"Test result reporter initialized with report_dir={self.report_dir}, artifact_dir={self.artifact_dir}")
    
    async def get_artifact_urls(self, test_run_id: str, artifact_names: List[str], validate: bool = False) -> Dict[str, Optional[str]]:
        """
        Retrieve URLs for multiple artifacts in bulk.
        
        This method efficiently retrieves URLs for multiple artifacts in a single operation,
        which is more efficient than retrieving them one by one.
        
        Args:
            test_run_id: Test run ID
            artifact_names: List of artifact names
            validate: Whether to validate URL accessibility
            
        Returns:
            Dictionary mapping artifact names to their URLs (or None if not found)
        """
        if not self.ci_provider or not hasattr(self.ci_provider, 'get_artifact_url'):
            logger.warning("CI provider doesn't support get_artifact_url method")
            return {name: None for name in artifact_names}
        
        # Create tasks for retrieving URLs in parallel
        tasks = []
        for name in artifact_names:
            task = asyncio.create_task(self.ci_provider.get_artifact_url(test_run_id, name))
            tasks.append((name, task))
        
        # Wait for all tasks to complete
        urls = {}
        for name, task in tasks:
            try:
                url = await task
                urls[name] = url
            except Exception as e:
                logger.error(f"Error retrieving artifact URL for {name}: {str(e)}")
                urls[name] = None
        
        # Validate URLs if requested
        if validate and urls:
            try:
                # Import the URL validator
                from ci.url_validator import validate_urls
                
                # Get valid URLs (skip None values)
                valid_urls = {name: url for name, url in urls.items() if url is not None}
                
                if valid_urls:
                    # Run validation on valid URLs
                    validation_results = await validate_urls(list(valid_urls.values()))
                    
                    # Log validation results
                    for url, (is_valid, status_code, error_message) in validation_results.items():
                        # Find the artifact name for this URL
                        name = next((n for n, u in valid_urls.items() if u == url), None)
                        
                        if name is not None:
                            if is_valid:
                                logger.debug(f"Validated artifact URL for {name}: {url} (Status: {status_code})")
                            else:
                                logger.warning(f"Artifact URL for {name} ({url}) is not accessible: {error_message}")
            except ImportError:
                logger.warning("URL validator not available, skipping URL validation")
            except Exception as e:
                logger.error(f"Error validating artifact URLs: {str(e)}")
        
        return urls
    
    async def report_test_result(self, result: TestRunResult, formats: List[str] = None) -> Dict[str, str]:
        """
        Report test results in various formats and to CI system.
        
        Args:
            result: Test result data
            formats: List of formats to generate (markdown, html, json)
            
        Returns:
            Dictionary mapping format to report file path
        """
        formats = formats or ["markdown", "html", "json"]
        report_files = {}
        
        # Generate reports in each format
        for fmt in formats:
            if fmt.lower() == "markdown":
                report_content = TestResultFormatter.format_markdown(result)
                report_path = os.path.join(self.report_dir, f"{result.test_run_id}_report.md")
            elif fmt.lower() == "html":
                report_content = TestResultFormatter.format_html(result)
                report_path = os.path.join(self.report_dir, f"{result.test_run_id}_report.html")
            elif fmt.lower() == "json":
                report_content = TestResultFormatter.format_json(result)
                report_path = os.path.join(self.report_dir, f"{result.test_run_id}_report.json")
            else:
                logger.warning(f"Unsupported format: {fmt}")
                continue
            
            # Write report to file
            with open(report_path, "w") as f:
                f.write(report_content)
            
            report_files[fmt] = report_path
            logger.info(f"Generated {fmt} report at {report_path}")
        
        # Upload reports as artifacts if CI provider is available
        if self.ci_provider and result.test_run_id:
            report_artifact_names = {}
            
            # Upload all report formats as artifacts
            for fmt, file_path in report_files.items():
                try:
                    artifact_name = f"{result.test_run_id}_report.{fmt}"
                    success = await self.ci_provider.upload_artifact(result.test_run_id, file_path, artifact_name)
                    if success:
                        report_artifact_names[fmt] = artifact_name
                        logger.info(f"Uploaded {fmt} report as artifact")
                except Exception as e:
                    logger.error(f"Failed to upload {fmt} report as artifact: {str(e)}")
            
            # Retrieve URLs for all uploaded report artifacts
            if report_artifact_names and hasattr(self.ci_provider, 'get_artifact_url'):
                try:
                    # Bulk retrieve URLs for all reports
                    report_urls = await self.get_artifact_urls(
                        result.test_run_id, 
                        list(report_artifact_names.values())
                    )
                    
                    # Add report URLs to result metadata
                    if "artifacts" not in result.metadata:
                        result.metadata["artifacts"] = []
                    
                    for fmt, artifact_name in report_artifact_names.items():
                        if artifact_name in report_urls and report_urls[artifact_name]:
                            report_size = os.path.getsize(report_files[fmt])
                            result.metadata["artifacts"].append({
                                "name": f"Report ({fmt.upper()})",
                                "path": report_files[fmt],
                                "size_bytes": report_size,
                                "url": report_urls[artifact_name],
                                "type": "report"
                            })
                            logger.info(f"Added {fmt} report URL to result metadata: {report_urls[artifact_name]}")
                except Exception as e:
                    logger.error(f"Failed to retrieve report artifact URLs: {str(e)}")
            
            # Update test run status in CI system
            try:
                await self.ci_provider.update_test_run(result.test_run_id, {
                    "status": result.status,
                    "total_tests": result.total_tests,
                    "passed_tests": result.passed_tests,
                    "failed_tests": result.failed_tests,
                    "skipped_tests": result.skipped_tests,
                    "duration_seconds": result.duration_seconds
                })
                logger.info(f"Updated test run status in CI system")
            except Exception as e:
                logger.error(f"Failed to update test run status in CI system: {str(e)}")
            
            # Add comment to PR if available
            if "pr_number" in result.metadata:
                pr_number = result.metadata["pr_number"]
                try:
                    # Use markdown format for PR comments
                    comment = TestResultFormatter.format_markdown(result)
                    await self.ci_provider.add_pr_comment(pr_number, comment)
                    logger.info(f"Added test result comment to PR #{pr_number}")
                except Exception as e:
                    logger.error(f"Failed to add comment to PR #{pr_number}: {str(e)}")
        
        return report_files
    
    async def collect_and_upload_artifacts(
        self, 
        test_run_id: str, 
        artifact_patterns: List[str],
        validate_urls: bool = False,
        include_health_info: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Collect and upload artifacts matching the given patterns.
        
        Args:
            test_run_id: Test run ID
            artifact_patterns: List of glob patterns for artifacts
            validate_urls: Whether to validate URL accessibility
            include_health_info: Whether to include health information in artifact metadata
            
        Returns:
            List of artifact information dictionaries
        """
        import glob
        
        artifacts = []
        
        # Collect artifacts matching patterns
        for pattern in artifact_patterns:
            for file_path in glob.glob(pattern, recursive=True):
                # Get file info
                file_stat = os.stat(file_path)
                file_size = file_stat.st_size
                file_name = os.path.basename(file_path)
                
                # Copy to artifact directory
                artifact_path = os.path.join(self.artifact_dir, file_name)
                with open(file_path, "rb") as src, open(artifact_path, "wb") as dst:
                    dst.write(src.read())
                
                # Upload to CI system if available
                artifact_url = None
                if self.ci_provider:
                    try:
                        # Upload the artifact
                        success = await self.ci_provider.upload_artifact(test_run_id, artifact_path, file_name)
                        
                        if success:
                            # Try to get the actual URL using get_artifact_url
                            try:
                                if hasattr(self.ci_provider, 'get_artifact_url'):
                                    artifact_url = await self.ci_provider.get_artifact_url(test_run_id, file_name)
                                    if artifact_url:
                                        logger.info(f"Retrieved artifact URL for {file_name}: {artifact_url}")
                                    else:
                                        # Fallback to a placeholder URL if get_artifact_url returned None
                                        artifact_url = f"ci://artifacts/{test_run_id}/{file_name}"
                                        logger.warning(f"Could not retrieve URL for artifact {file_name}, using placeholder")
                                else:
                                    # Fallback if get_artifact_url method doesn't exist
                                    artifact_url = f"ci://artifacts/{test_run_id}/{file_name}"
                                    logger.warning(f"CI provider doesn't support get_artifact_url, using placeholder URL")
                            except Exception as e:
                                # Fallback if get_artifact_url fails
                                artifact_url = f"ci://artifacts/{test_run_id}/{file_name}"
                                logger.warning(f"Error retrieving artifact URL for {file_name}: {str(e)}, using placeholder")
                            
                            logger.info(f"Uploaded artifact {file_name}")
                    except Exception as e:
                        logger.error(f"Failed to upload artifact {file_name}: {str(e)}")
                
                # Create base artifact info
                artifact_info = {
                    "name": file_name,
                    "path": artifact_path,
                    "size_bytes": file_size,
                    "url": artifact_url
                }
                
                # Validate URL if requested and URL exists
                if validate_urls and artifact_url and not artifact_url.startswith("ci://artifacts/"):
                    try:
                        # Import the URL validator
                        from ci.url_validator import validate_url
                        
                        # Validate the URL
                        is_valid, status_code, error_message = await validate_url(artifact_url)
                        
                        # Add validation info to artifact
                        artifact_info["url_validated"] = True
                        artifact_info["url_valid"] = is_valid
                        
                        if not is_valid:
                            logger.warning(f"Artifact URL for {file_name} ({artifact_url}) is not accessible: {error_message}")
                            artifact_info["url_validation_error"] = error_message
                        
                        # Include health info if requested
                        if include_health_info:
                            try:
                                from ci.url_validator import get_validator
                                validator = await get_validator()
                                health_info = validator.get_url_health(artifact_url)
                                artifact_info["url_health"] = health_info
                            except Exception as e:
                                logger.error(f"Failed to get URL health info for {file_name}: {str(e)}")
                        
                    except ImportError:
                        logger.warning("URL validator not available, skipping URL validation")
                    except Exception as e:
                        logger.error(f"Error validating artifact URL for {file_name}: {str(e)}")
                
                # Add to artifacts list
                artifacts.append(artifact_info)
        
        return artifacts


def _percentage(value: int, total: int) -> float:
    """Calculate percentage with handling for zero total."""
    return round((value / total) * 100, 1) if total > 0 else 0.0


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def _format_size(size_bytes: int) -> str:
    """Format file size in bytes to a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.2f} MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.2f} GB"


async def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Result Reporter")
    
    # Test result information
    parser.add_argument("--test-run-id", required=True, help="Test run ID")
    parser.add_argument("--status", required=True, choices=["success", "failure", "running"], help="Test run status")
    parser.add_argument("--total", type=int, required=True, help="Total number of tests")
    parser.add_argument("--passed", type=int, required=True, help="Number of passed tests")
    parser.add_argument("--failed", type=int, required=True, help="Number of failed tests")
    parser.add_argument("--skipped", type=int, required=True, help="Number of skipped tests")
    parser.add_argument("--duration", type=float, required=True, help="Duration in seconds")
    
    # Report options
    parser.add_argument("--formats", default="markdown,html,json", help="Comma-separated list of output formats")
    parser.add_argument("--report-dir", default="reports", help="Directory to save reports")
    parser.add_argument("--artifact-dir", default="artifacts", help="Directory to save artifacts")
    parser.add_argument("--collect-artifacts", help="Comma-separated list of glob patterns for artifacts to collect")
    
    # CI provider options
    parser.add_argument("--ci-provider", help="CI provider type (github, gitlab, jenkins, etc.)")
    parser.add_argument("--ci-config", help="Path to CI provider configuration file")
    
    args = parser.parse_args()
    
    # Parse formats
    formats = [fmt.strip() for fmt in args.formats.split(",")]
    
    # Create test result
    result = TestRunResult(
        test_run_id=args.test_run_id,
        status=args.status,
        total_tests=args.total,
        passed_tests=args.passed,
        failed_tests=args.failed,
        skipped_tests=args.skipped,
        duration_seconds=args.duration,
        metadata={}
    )
    
    # Initialize CI provider if specified
    ci_provider = None
    if args.ci_provider:
        try:
            # Load CI provider configuration
            if args.ci_config:
                with open(args.ci_config, "r") as f:
                    ci_config = json.load(f)
            else:
                ci_config = {}
            
            # Create CI provider
            ci_provider = await CIProviderFactory.create_provider(args.ci_provider, ci_config)
            logger.info(f"Initialized CI provider: {args.ci_provider}")
        except Exception as e:
            logger.error(f"Failed to initialize CI provider: {str(e)}")
    
    # Create reporter
    reporter = TestResultReporter(
        ci_provider=ci_provider,
        report_dir=args.report_dir,
        artifact_dir=args.artifact_dir
    )
    
    # Collect artifacts if specified
    if args.collect_artifacts:
        artifact_patterns = [p.strip() for p in args.collect_artifacts.split(",")]
        artifacts = await reporter.collect_and_upload_artifacts(args.test_run_id, artifact_patterns)
        
        if artifacts:
            result.metadata["artifacts"] = artifacts
            logger.info(f"Collected {len(artifacts)} artifacts")
    
    # Generate and upload reports
    report_files = await reporter.report_test_result(result, formats)
    
    # Print report file paths
    for fmt, file_path in report_files.items():
        print(f"{fmt.upper()} report: {file_path}")
    
    # Clean up
    if ci_provider:
        await ci_provider.close()


if __name__ == "__main__":
    asyncio.run(main())