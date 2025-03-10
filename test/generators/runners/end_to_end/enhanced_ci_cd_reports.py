#!/usr/bin/env python3
"""
Enhanced CI/CD Reports Generator for End-to-End Testing Framework

This script generates comprehensive CI/CD reports for the end-to-end testing framework,
including detailed test results, performance metrics, hardware compatibility matrices,
and trend analysis. It can be used both within CI/CD pipelines and locally.

Features:
- Comprehensive test status reports with detailed metrics
- Hardware compatibility matrix generation
- Trend analysis of test results over time
- Performance comparison across hardware platforms
- Badge generation for status dashboards
- HTML and Markdown report formats
- GitHub integration

Usage:
    # Generate reports from a single test run
    python enhanced_ci_cd_reports.py --input-dir /path/to/collected_results
    
    # Generate comprehensive reports with historical data
    python enhanced_ci_cd_reports.py --historical --days 30 --output-dir reports
    
    # Generate reports for CI/CD with badges
    python enhanced_ci_cd_reports.py --ci --badge-only

    # Generate reports and upload to GitHub Pages
    python enhanced_ci_cd_reports.py --github-pages
"""

import os
import sys
import json
import time
import argparse
import logging
import datetime
import tempfile
import re
import numpy as np
import glob
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union, Set

# Add parent directory to path so we can import project modules
script_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.abspath(os.path.join(script_dir, "../../../"))
sys.path.append(test_dir)

# Import utilities
from simple_utils import setup_logging, ensure_dir_exists

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Constants
COLLECTED_RESULTS_DIR = os.path.join(os.path.dirname(script_dir), "collected_results")
EXPECTED_RESULTS_DIR = os.path.join(os.path.dirname(script_dir), "expected_results")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(script_dir), "reports")

# Define styles for HTML reports
HTML_STYLE = """
<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 1200px;
    margin: 0 auto;
    padding: 2em;
}
h1, h2, h3, h4 {
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    color: #0366d6;
}
h1 {
    border-bottom: 1px solid #eaecef;
    padding-bottom: 0.3em;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
}
th, td {
    border: 1px solid #dfe2e5;
    padding: 0.5em 1em;
    text-align: left;
}
th {
    background-color: #f6f8fa;
    font-weight: 600;
}
tr:nth-child(even) {
    background-color: #f6f8fa;
}
.success {
    color: #22863a;
    font-weight: bold;
}
.failure {
    color: #cb2431;
    font-weight: bold;
}
.error {
    color: #e36209;
    font-weight: bold;
}
.matrix-cell {
    text-align: center;
    font-weight: bold;
}
.success-bg {
    background-color: #dcffe4;
}
.failure-bg {
    background-color: #ffdce0;
}
.unknown-bg {
    background-color: #f1f8ff;
}
.badge {
    display: inline-block;
    padding: 0.25em 0.6em;
    font-size: 0.75em;
    font-weight: 700;
    line-height: 1;
    text-align: center;
    white-space: nowrap;
    vertical-align: baseline;
    border-radius: 0.25em;
    margin-right: 0.5em;
}
.badge-success {
    color: white;
    background-color: #28a745;
}
.badge-failure {
    color: white;
    background-color: #dc3545;
}
.badge-unknown {
    color: white;
    background-color: #6c757d;
}
.chart {
    width: 100%;
    height: 400px;
    margin: 1em 0;
    border: 1px solid #dfe2e5;
}
.metrics-box {
    border: 1px solid #dfe2e5;
    border-radius: 3px;
    background-color: #f6f8fa;
    padding: 1em;
    margin: 1em 0;
}
.summary-box {
    border-left: 4px solid #0366d6;
    padding-left: 1em;
    margin: 1em 0;
}
.grid-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    grid-gap: 1em;
    margin: 1em 0;
}
.grid-item {
    border: 1px solid #dfe2e5;
    border-radius: 3px;
    padding: 1em;
}
</style>
"""

class StatusBadgeGenerator:
    """Generates status badges for CI/CD dashboards."""
    
    @staticmethod
    def generate_svg_badge(label: str, status: str, color: str) -> str:
        """
        Generate an SVG badge with the given label, status, and color.
        
        Args:
            label: The label for the badge (left side)
            status: The status text (right side)
            color: The color for the status part (hex code or named color)
            
        Returns:
            SVG badge as a string
        """
        # Estimate width based on text length
        label_width = max(len(label) * 6, 40)
        status_width = max(len(status) * 6, 40)
        total_width = label_width + status_width
        
        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <mask id="a">
    <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
  </mask>
  <g mask="url(#a)">
    <path fill="#555" d="M0 0h{label_width}v20H0z"/>
    <path fill="{color}" d="M{label_width} 0h{status_width}v20H{label_width}z"/>
    <path fill="url(#b)" d="M0 0h{total_width}v20H0z"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="{label_width/2}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{label_width/2}" y="14">{label}</text>
    <text x="{label_width + status_width/2}" y="15" fill="#010101" fill-opacity=".3">{status}</text>
    <text x="{label_width + status_width/2}" y="14">{status}</text>
  </g>
</svg>"""
    
    @staticmethod
    def generate_status_badge(test_results: Dict[str, Any], output_path: str) -> str:
        """
        Generate a status badge for the test results.
        
        Args:
            test_results: Dictionary with test results
            output_path: Path to save the badge
            
        Returns:
            Path to the generated badge
        """
        # Calculate success rate
        total = test_results.get("summary", {}).get("total", 0)
        success = test_results.get("summary", {}).get("success", 0)
        
        if total == 0:
            status = "unknown"
            color = "#6c757d"  # gray
        elif success == total:
            status = "passing"
            color = "#28a745"  # green
        elif success >= total * 0.8:
            status = "partial"
            color = "#f0ad4e"  # orange
        else:
            status = "failing"
            color = "#dc3545"  # red
        
        # Add success rate if not unknown
        if status != "unknown" and total > 0:
            success_rate = (success / total) * 100
            status = f"{status} ({success_rate:.0f}%)"
        
        # Generate SVG badge
        svg_content = StatusBadgeGenerator.generate_svg_badge("E2E Tests", status, color)
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(svg_content)
        
        return output_path
    
    @staticmethod
    def generate_model_badges(model_results: Dict[str, Dict[str, Any]], output_dir: str) -> Dict[str, str]:
        """
        Generate status badges for individual models.
        
        Args:
            model_results: Dictionary with model-specific results
            output_dir: Directory to save the badges
            
        Returns:
            Dictionary mapping model names to badge paths
        """
        ensure_dir_exists(output_dir)
        badges = {}
        
        for model, hw_results in model_results.items():
            # Calculate success rate for this model
            total = len(hw_results)
            success = sum(1 for r in hw_results.values() if r.get("status") == "success")
            
            if total == 0:
                status = "unknown"
                color = "#6c757d"  # gray
            elif success == total:
                status = "passing"
                color = "#28a745"  # green
            elif success >= total * 0.8:
                status = "partial"
                color = "#f0ad4e"  # orange
            else:
                status = "failing"
                color = "#dc3545"  # red
            
            # Add success rate if not unknown
            if status != "unknown" and total > 0:
                success_rate = (success / total) * 100
                status = f"{status} ({success_rate:.0f}%)"
            
            # Generate SVG badge
            model_short = model.split("/")[-1]  # Use short model name for label
            svg_content = StatusBadgeGenerator.generate_svg_badge(model_short, status, color)
            
            # Save to file
            badge_path = os.path.join(output_dir, f"{model.replace('/', '_')}_badge.svg")
            with open(badge_path, 'w') as f:
                f.write(svg_content)
            
            badges[model] = badge_path
        
        return badges
    
    @staticmethod
    def generate_hardware_badges(model_results: Dict[str, Dict[str, Any]], output_dir: str) -> Dict[str, str]:
        """
        Generate status badges for hardware platforms.
        
        Args:
            model_results: Dictionary with model-specific results
            output_dir: Directory to save the badges
            
        Returns:
            Dictionary mapping hardware platforms to badge paths
        """
        ensure_dir_exists(output_dir)
        badges = {}
        
        # Organize results by hardware
        hardware_results = {}
        
        for model, hw_results in model_results.items():
            for hw, result in hw_results.items():
                if hw not in hardware_results:
                    hardware_results[hw] = []
                hardware_results[hw].append(result)
        
        # Generate badges for each hardware platform
        for hw, results in hardware_results.items():
            # Calculate success rate for this hardware
            total = len(results)
            success = sum(1 for r in results if r.get("status") == "success")
            
            if total == 0:
                status = "unknown"
                color = "#6c757d"  # gray
            elif success == total:
                status = "passing"
                color = "#28a745"  # green
            elif success >= total * 0.8:
                status = "partial"
                color = "#f0ad4e"  # orange
            else:
                status = "failing"
                color = "#dc3545"  # red
            
            # Add success rate if not unknown
            if status != "unknown" and total > 0:
                success_rate = (success / total) * 100
                status = f"{status} ({success_rate:.0f}%)"
            
            # Generate SVG badge
            svg_content = StatusBadgeGenerator.generate_svg_badge(hw, status, color)
            
            # Save to file
            badge_path = os.path.join(output_dir, f"{hw}_badge.svg")
            with open(badge_path, 'w') as f:
                f.write(svg_content)
            
            badges[hw] = badge_path
        
        return badges


class ReportGenerator:
    """Generates comprehensive CI/CD reports for the end-to-end testing framework."""
    
    def __init__(self, args):
        """Initialize the report generator with command-line arguments."""
        self.args = args
        self.output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
        self.input_dir = args.input_dir or COLLECTED_RESULTS_DIR
        self.days = args.days or 30
        self.historical = args.historical
        self.format = args.format or "html"
        self.badge_only = args.badge_only
        self.ci = args.ci
        self.github_pages = args.github_pages
        self.test_results = {}
        
        # Create output directory if it doesn't exist
        ensure_dir_exists(self.output_dir)
    
    def collect_results(self) -> Dict[str, Any]:
        """
        Collect test results from the input directory.
        
        Returns:
            Dictionary with collected test results
        """
        logger.info(f"Collecting test results from {self.input_dir}")
        
        # Find all summary files
        summary_dir = os.path.join(self.input_dir, "summary")
        if not os.path.exists(summary_dir):
            logger.warning(f"Summary directory not found: {summary_dir}")
            summary_dir = self.input_dir
        
        # Find all summary files
        summary_files = glob.glob(os.path.join(summary_dir, "summary_*.json"))
        if not summary_files:
            logger.warning("No summary files found")
            return {}
        
        # Sort by timestamp (newest first)
        summary_files.sort(reverse=True)
        
        # Use the newest summary file by default, or collect historical data if requested
        if self.historical:
            # Collect data from multiple summary files within the specified time range
            cutoff_time = datetime.datetime.now() - datetime.timedelta(days=self.days)
            collected_data = []
            
            for summary_file in summary_files:
                try:
                    # Extract timestamp from filename
                    timestamp_match = re.search(r'summary_(\d{8}_\d{6})\.json', os.path.basename(summary_file))
                    if not timestamp_match:
                        continue
                    
                    timestamp_str = timestamp_match.group(1)
                    timestamp = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    # Skip if older than cutoff
                    if timestamp < cutoff_time:
                        continue
                    
                    # Load summary file
                    with open(summary_file, 'r') as f:
                        summary_data = json.load(f)
                    
                    # Add to collection
                    collected_data.append(summary_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing summary file {summary_file}: {str(e)}")
            
            if not collected_data:
                logger.warning(f"No data found within the last {self.days} days")
                return {}
            
            # Process historical data
            return self._process_historical_data(collected_data)
        else:
            # Use just the newest summary file
            newest_summary = summary_files[0]
            logger.info(f"Using newest summary file: {newest_summary}")
            
            try:
                with open(newest_summary, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading summary file {newest_summary}: {str(e)}")
                return {}
    
    def _process_historical_data(self, collected_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process historical data to create a combined result set.
        
        Args:
            collected_data: List of summary data dictionaries
            
        Returns:
            Combined results dictionary with historical data
        """
        if not collected_data:
            return {}
        
        # Use the newest data as the base
        newest_data = collected_data[0]
        result = {
            "timestamp": newest_data.get("timestamp", ""),
            "summary": newest_data.get("summary", {}),
            "results": newest_data.get("results", {}),
            "historical": {
                "timestamps": [],
                "success_rates": [],
                "total_tests": [],
                "model_success_rates": {},
                "hardware_success_rates": {}
            }
        }
        
        # Process each data point for historical trends
        for data in collected_data:
            timestamp = data.get("timestamp", "")
            summary = data.get("summary", {})
            results = data.get("results", {})
            
            # Add to historical data
            result["historical"]["timestamps"].append(timestamp)
            
            # Calculate success rate
            total = summary.get("total", 0)
            success = summary.get("success", 0)
            success_rate = (success / total) * 100 if total > 0 else 0
            
            result["historical"]["success_rates"].append(success_rate)
            result["historical"]["total_tests"].append(total)
            
            # Track model-specific success rates
            for model, hw_results in results.items():
                if model not in result["historical"]["model_success_rates"]:
                    result["historical"]["model_success_rates"][model] = {
                        "timestamps": [],
                        "success_rates": []
                    }
                
                model_total = len(hw_results)
                model_success = sum(1 for r in hw_results.values() if r.get("status") == "success")
                model_success_rate = (model_success / model_total) * 100 if model_total > 0 else 0
                
                result["historical"]["model_success_rates"][model]["timestamps"].append(timestamp)
                result["historical"]["model_success_rates"][model]["success_rates"].append(model_success_rate)
            
            # Track hardware-specific success rates
            hw_results_dict = {}
            for model, hw_results in results.items():
                for hw, hw_result in hw_results.items():
                    if hw not in hw_results_dict:
                        hw_results_dict[hw] = []
                    hw_results_dict[hw].append(hw_result)
            
            for hw, hw_results in hw_results_dict.items():
                if hw not in result["historical"]["hardware_success_rates"]:
                    result["historical"]["hardware_success_rates"][hw] = {
                        "timestamps": [],
                        "success_rates": []
                    }
                
                hw_total = len(hw_results)
                hw_success = sum(1 for r in hw_results if r.get("status") == "success")
                hw_success_rate = (hw_success / hw_total) * 100 if hw_total > 0 else 0
                
                result["historical"]["hardware_success_rates"][hw]["timestamps"].append(timestamp)
                result["historical"]["hardware_success_rates"][hw]["success_rates"].append(hw_success_rate)
        
        return result
    
    def generate_compatibility_matrix(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
        """
        Generate a compatibility matrix from test results.
        
        Args:
            results: Dictionary with test results by model and hardware
            
        Returns:
            Dictionary representing the compatibility matrix
        """
        matrix = {}
        
        # Get all models and hardware platforms
        all_models = list(results.keys())
        all_hardware = set()
        
        for model, hw_results in results.items():
            all_hardware.update(hw_results.keys())
        
        all_hardware = sorted(list(all_hardware))
        
        # Build the matrix
        for model in all_models:
            matrix[model] = {}
            
            for hw in all_hardware:
                if hw in results[model]:
                    status = results[model][hw].get("status", "unknown")
                    matrix[model][hw] = status
                else:
                    matrix[model][hw] = "untested"
        
        return matrix
    
    def collect_performance_metrics(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Collect performance metrics from test results.
        
        Args:
            results: Dictionary with test results by model and hardware
            
        Returns:
            Dictionary with performance metrics by model and hardware
        """
        metrics = {}
        
        for model, hw_results in results.items():
            metrics[model] = {}
            
            for hw, result in hw_results.items():
                # Check if there are metrics in the result
                if "result_path" in result:
                    result_path = result["result_path"]
                    result_file = os.path.join(result_path, "result.json")
                    
                    if os.path.exists(result_file):
                        try:
                            with open(result_file, 'r') as f:
                                result_data = json.load(f)
                            
                            # Extract metrics
                            if "metrics" in result_data:
                                metrics[model][hw] = result_data["metrics"]
                            elif "output" in result_data and "metrics" in result_data["output"]:
                                metrics[model][hw] = result_data["output"]["metrics"]
                        except Exception as e:
                            logger.warning(f"Error loading result file {result_file}: {str(e)}")
        
        return metrics
    
    def generate_report(self) -> Dict[str, str]:
        """
        Generate comprehensive CI/CD reports.
        
        Returns:
            Dictionary mapping report names to file paths
        """
        # Collect test results
        self.test_results = self.collect_results()
        
        if not self.test_results:
            logger.error("No test results found")
            return {}
        
        # Generate reports based on format
        if self.badge_only:
            return self._generate_badges()
        elif self.format == "html":
            return self._generate_html_reports()
        elif self.format == "markdown":
            return self._generate_markdown_reports()
        else:
            logger.error(f"Unsupported format: {self.format}")
            return {}
    
    def _generate_badges(self) -> Dict[str, str]:
        """
        Generate status badges for CI/CD dashboards.
        
        Returns:
            Dictionary mapping badge names to file paths
        """
        badges = {}
        
        # Create badges directory
        badges_dir = os.path.join(self.output_dir, "badges")
        ensure_dir_exists(badges_dir)
        
        # Generate overall status badge
        badge_path = os.path.join(badges_dir, "status_badge.svg")
        badges["status"] = StatusBadgeGenerator.generate_status_badge(self.test_results, badge_path)
        
        # Generate model-specific badges if we have results
        if "results" in self.test_results:
            model_badges = StatusBadgeGenerator.generate_model_badges(
                self.test_results["results"], badges_dir)
            badges.update(model_badges)
        
            # Generate hardware-specific badges
            hw_badges = StatusBadgeGenerator.generate_hardware_badges(
                self.test_results["results"], badges_dir)
            badges.update(hw_badges)
        
        return badges
    
    def _generate_html_reports(self) -> Dict[str, str]:
        """
        Generate HTML reports for test results.
        
        Returns:
            Dictionary mapping report names to file paths
        """
        reports = {}
        
        # Create main report directory
        reports_dir = self.output_dir
        ensure_dir_exists(reports_dir)
        
        # Generate overall status badge for inclusion in reports
        badges_dir = os.path.join(reports_dir, "badges")
        ensure_dir_exists(badges_dir)
        badge_path = os.path.join(badges_dir, "status_badge.svg")
        status_badge = StatusBadgeGenerator.generate_status_badge(self.test_results, badge_path)
        
        # Extract data for reports
        summary = self.test_results.get("summary", {})
        results = self.test_results.get("results", {})
        timestamp = self.test_results.get("timestamp", "")
        historical = self.test_results.get("historical", {})
        
        # Generate compatibility matrix
        matrix = self.generate_compatibility_matrix(results) if results else {}
        
        # Collect performance metrics
        metrics = self.collect_performance_metrics(results) if results else {}
        
        # Generate main report
        main_report = os.path.join(reports_dir, "e2e_test_report.html")
        with open(main_report, 'w') as f:
            # Write HTML header
            f.write("<!DOCTYPE html>\n")
            f.write("<html lang=\"en\">\n")
            f.write("<head>\n")
            f.write("  <meta charset=\"UTF-8\">\n")
            f.write("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n")
            f.write(f"  <title>End-to-End Test Report - {timestamp}</title>\n")
            f.write(HTML_STYLE)
            f.write("</head>\n")
            f.write("<body>\n")
            
            # Header and status badge
            f.write(f"<h1>End-to-End Test Report - {timestamp}</h1>\n")
            f.write(f'<p><img src="badges/status_badge.svg" alt="Test Status" /></p>\n')
            
            # Summary section
            f.write("<h2>Summary</h2>\n")
            f.write("<div class=\"summary-box\">\n")
            total = summary.get("total", 0)
            success = summary.get("success", 0)
            failure = summary.get("failure", 0)
            error = summary.get("error", 0)
            
            success_rate = (success / total) * 100 if total > 0 else 0
            
            f.write(f"<p><strong>Total Tests:</strong> {total}</p>\n")
            f.write(f"<p><strong>Successful:</strong> <span class=\"success\">{success}</span></p>\n")
            f.write(f"<p><strong>Failed:</strong> <span class=\"failure\">{failure}</span></p>\n")
            f.write(f"<p><strong>Errors:</strong> <span class=\"error\">{error}</span></p>\n")
            f.write(f"<p><strong>Success Rate:</strong> <span class=\"success\">{success_rate:.1f}%</span></p>\n")
            f.write("</div>\n")
            
            # Historical trends if available
            if historical and "success_rates" in historical and historical["success_rates"]:
                f.write("<h2>Historical Trends</h2>\n")
                
                # Simple trend visualization using a table
                f.write("<table>\n")
                f.write("  <tr><th>Date</th><th>Success Rate</th><th>Total Tests</th></tr>\n")
                
                for i, timestamp in enumerate(historical["timestamps"]):
                    success_rate = historical["success_rates"][i]
                    total_tests = historical["total_tests"][i]
                    
                    # Format date for display
                    date_str = timestamp
                    try:
                        date_obj = datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                        date_str = date_obj.strftime("%Y-%m-%d %H:%M")
                    except:
                        pass
                    
                    f.write(f"  <tr><td>{date_str}</td><td>{success_rate:.1f}%</td><td>{total_tests}</td></tr>\n")
                
                f.write("</table>\n")
            
            # Compatibility matrix
            if matrix:
                f.write("<h2>Compatibility Matrix</h2>\n")
                
                # Get all hardware platforms
                hardware_platforms = set()
                for model, hw_results in matrix.items():
                    hardware_platforms.update(hw_results.keys())
                
                hardware_platforms = sorted(list(hardware_platforms))
                
                # Generate table
                f.write("<table>\n")
                
                # Header row
                f.write("  <tr><th>Model</th>")
                for hw in hardware_platforms:
                    f.write(f"<th>{hw}</th>")
                f.write("</tr>\n")
                
                # Data rows
                for model, hw_results in matrix.items():
                    f.write(f"  <tr><td>{model}</td>")
                    
                    for hw in hardware_platforms:
                        status = hw_results.get(hw, "untested")
                        
                        # Determine cell styling
                        if status == "success":
                            cell_class = "success-bg matrix-cell"
                            cell_text = "✅"
                        elif status == "failure":
                            cell_class = "failure-bg matrix-cell"
                            cell_text = "❌"
                        elif status == "error":
                            cell_class = "failure-bg matrix-cell"
                            cell_text = "⚠️"
                        else:
                            cell_class = "unknown-bg matrix-cell"
                            cell_text = "⚪"
                        
                        f.write(f"<td class=\"{cell_class}\">{cell_text}</td>")
                    
                    f.write("</tr>\n")
                
                f.write("</table>\n")
            
            # Performance metrics
            if metrics:
                f.write("<h2>Performance Metrics</h2>\n")
                
                # Get all hardware platforms with metrics
                hardware_with_metrics = set()
                for model, hw_metrics in metrics.items():
                    hardware_with_metrics.update(hw_metrics.keys())
                
                hardware_with_metrics = sorted(list(hardware_with_metrics))
                
                # Create grid layout for metrics
                f.write("<div class=\"grid-container\">\n")
                
                for model, hw_metrics in metrics.items():
                    if not hw_metrics:
                        continue
                    
                    f.write(f"<div class=\"grid-item\">\n")
                    f.write(f"<h3>{model}</h3>\n")
                    
                    for hw, metric in hw_metrics.items():
                        f.write(f"<h4>{hw}</h4>\n")
                        f.write("<div class=\"metrics-box\">\n")
                        
                        # Display all metrics
                        for metric_name, metric_value in metric.items():
                            f.write(f"<p><strong>{metric_name}:</strong> {metric_value}</p>\n")
                        
                        f.write("</div>\n")
                    
                    f.write("</div>\n")
                
                f.write("</div>\n")
            
            # Detailed results
            if results:
                f.write("<h2>Detailed Results</h2>\n")
                
                for model, hw_results in results.items():
                    f.write(f"<h3>{model}</h3>\n")
                    
                    for hw, result in hw_results.items():
                        status = result.get("status", "unknown")
                        status_class = "success" if status == "success" else "failure" if status == "failure" else "error"
                        
                        f.write(f"<div class=\"metrics-box\">\n")
                        f.write(f"<h4>{hw}</h4>\n")
                        f.write(f"<p><strong>Status:</strong> <span class=\"{status_class}\">{status.upper()}</span></p>\n")
                        
                        if "error" in result:
                            f.write(f"<p><strong>Error:</strong> {result['error']}</p>\n")
                        
                        if "comparison" in result and "differences" in result["comparison"]:
                            f.write("<p><strong>Differences:</strong></p>\n")
                            f.write("<ul>\n")
                            
                            for key, diff in result["comparison"]["differences"].items():
                                f.write(f"<li>{key}: {json.dumps(diff)}</li>\n")
                            
                            f.write("</ul>\n")
                        
                        if "result_path" in result:
                            f.write(f"<p><strong>Results:</strong> {result['result_path']}</p>\n")
                        
                        f.write("</div>\n")
            
            # HTML footer
            f.write("</body>\n")
            f.write("</html>\n")
        
        reports["main"] = main_report
        
        # Generate GitHub Pages index if requested
        if self.github_pages:
            index_path = os.path.join(reports_dir, "index.html")
            shutil.copy(main_report, index_path)
            reports["index"] = index_path
        
        return reports
    
    def _generate_markdown_reports(self) -> Dict[str, str]:
        """
        Generate Markdown reports for test results.
        
        Returns:
            Dictionary mapping report names to file paths
        """
        reports = {}
        
        # Create main report directory
        reports_dir = self.output_dir
        ensure_dir_exists(reports_dir)
        
        # Extract data for reports
        summary = self.test_results.get("summary", {})
        results = self.test_results.get("results", {})
        timestamp = self.test_results.get("timestamp", "")
        historical = self.test_results.get("historical", {})
        
        # Generate compatibility matrix
        matrix = self.generate_compatibility_matrix(results) if results else {}
        
        # Collect performance metrics
        metrics = self.collect_performance_metrics(results) if results else {}
        
        # Generate main report
        main_report = os.path.join(reports_dir, "e2e_test_report.md")
        with open(main_report, 'w') as f:
            f.write(f"# End-to-End Test Report - {timestamp}\n\n")
            
            # Summary section
            f.write("## Summary\n\n")
            total = summary.get("total", 0)
            success = summary.get("success", 0)
            failure = summary.get("failure", 0)
            error = summary.get("error", 0)
            
            success_rate = (success / total) * 100 if total > 0 else 0
            
            f.write(f"- **Total Tests**: {total}\n")
            f.write(f"- **Successful**: {success}\n")
            f.write(f"- **Failed**: {failure}\n")
            f.write(f"- **Errors**: {error}\n")
            f.write(f"- **Success Rate**: {success_rate:.1f}%\n\n")
            
            # Historical trends if available
            if historical and "success_rates" in historical and historical["success_rates"]:
                f.write("## Historical Trends\n\n")
                
                # Table for trends
                f.write("| Date | Success Rate | Total Tests |\n")
                f.write("|------|--------------|-------------|\n")
                
                for i, timestamp in enumerate(historical["timestamps"]):
                    success_rate = historical["success_rates"][i]
                    total_tests = historical["total_tests"][i]
                    
                    # Format date for display
                    date_str = timestamp
                    try:
                        date_obj = datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                        date_str = date_obj.strftime("%Y-%m-%d %H:%M")
                    except:
                        pass
                    
                    f.write(f"| {date_str} | {success_rate:.1f}% | {total_tests} |\n")
                
                f.write("\n")
            
            # Compatibility matrix
            if matrix:
                f.write("## Compatibility Matrix\n\n")
                
                # Get all hardware platforms
                hardware_platforms = set()
                for model, hw_results in matrix.items():
                    hardware_platforms.update(hw_results.keys())
                
                hardware_platforms = sorted(list(hardware_platforms))
                
                # Generate table
                f.write("| Model | " + " | ".join(hardware_platforms) + " |\n")
                f.write("|-------|" + "|".join(["---" for _ in hardware_platforms]) + "|\n")
                
                # Data rows
                for model, hw_results in matrix.items():
                    row = [model]
                    
                    for hw in hardware_platforms:
                        status = hw_results.get(hw, "untested")
                        
                        # Determine cell text
                        if status == "success":
                            cell_text = "✅"
                        elif status == "failure":
                            cell_text = "❌"
                        elif status == "error":
                            cell_text = "⚠️"
                        else:
                            cell_text = "⚪"
                        
                        row.append(cell_text)
                    
                    f.write("| " + " | ".join(row) + " |\n")
                
                f.write("\n")
            
            # Performance metrics
            if metrics:
                f.write("## Performance Metrics\n\n")
                
                for model, hw_metrics in metrics.items():
                    if not hw_metrics:
                        continue
                    
                    f.write(f"### {model}\n\n")
                    
                    for hw, metric in hw_metrics.items():
                        f.write(f"#### {hw}\n\n")
                        
                        # Display all metrics
                        for metric_name, metric_value in metric.items():
                            f.write(f"- **{metric_name}**: {metric_value}\n")
                        
                        f.write("\n")
            
            # Detailed results
            if results:
                f.write("## Detailed Results\n\n")
                
                for model, hw_results in results.items():
                    f.write(f"### {model}\n\n")
                    
                    for hw, result in hw_results.items():
                        status = result.get("status", "unknown")
                        status_icon = "✅" if status == "success" else "❌" if status == "failure" else "⚠️"
                        
                        f.write(f"- {status_icon} **{hw}**: {status.upper()}\n")
                        
                        if "error" in result:
                            f.write(f"  - Error: {result['error']}\n")
                        
                        if "comparison" in result and "differences" in result["comparison"]:
                            f.write("  - Differences found:\n")
                            for key, diff in result["comparison"]["differences"].items():
                                f.write(f"    - {key}: {json.dumps(diff)}\n")
                        
                        if "result_path" in result:
                            f.write(f"  - Results: {result['result_path']}\n")
                        
                        f.write("\n")
        
        reports["main"] = main_report
        return reports


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate enhanced CI/CD reports for end-to-end testing")
    
    # Input/output options
    parser.add_argument("--input-dir", help="Directory containing test results")
    parser.add_argument("--output-dir", help="Directory to save reports")
    
    # Report format options
    parser.add_argument("--format", choices=["html", "markdown"], default="html",
                       help="Report format (default: html)")
    
    # Historical data options
    parser.add_argument("--historical", action="store_true",
                       help="Include historical trend data in the report")
    parser.add_argument("--days", type=int, default=30,
                       help="Number of days to include in historical data (default: 30)")
    
    # Special report options
    parser.add_argument("--badge-only", action="store_true",
                       help="Generate only status badges (no full reports)")
    parser.add_argument("--ci", action="store_true",
                       help="Generate reports for CI/CD integration")
    parser.add_argument("--github-pages", action="store_true",
                       help="Generate reports for GitHub Pages")
    
    # Logging options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Set log level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Generate reports
    generator = ReportGenerator(args)
    reports = generator.generate_report()
    
    # Print report paths
    if reports:
        logger.info("Reports generated:")
        for report_name, report_path in reports.items():
            logger.info(f"- {report_name}: {report_path}")
    else:
        logger.error("No reports generated")


if __name__ == "__main__":
    main()