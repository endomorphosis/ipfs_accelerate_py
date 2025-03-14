#!/usr/bin/env python3
"""
Validation Reporter implementation for the Simulation Accuracy and Validation Framework.

This module provides a concrete implementation of the ValidationReporter interface
that generates reports and visualizations of validation results.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("validation_reporter")

# Import base classes
from duckdb_api.simulation_validation.core.base import (
    ValidationResult,
    ValidationReporter
)


class ValidationReporterImpl(ValidationReporter):
    """
    Implementation of a validation reporter that generates reports and visualizations.
    
    This reporter supports HTML, Markdown, and JSON formats for reports and can
    include visualizations of validation results.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the validation reporter.
        
        Args:
            config: Configuration options for the reporter
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            "report_formats": ["html", "markdown", "json"],
            "include_visualizations": True,
            "visualization_types": ["error_distribution", "trend_chart", "metric_heatmap"],
            "max_results_per_page": 20,
            "output_directory": "output",
            "report_title_template": "Simulation Validation Report - {timestamp}",
            "css_style_path": None,
            "html_template_path": None
        }
        
        # Apply default config values if not specified
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
                
    def generate_report(
        self,
        validation_results: List[ValidationResult],
        format: str = "html",
        include_visualizations: bool = True
    ) -> str:
        """
        Generate a validation report.
        
        Args:
            validation_results: List of validation results
            format: Output format (html, markdown, json, etc.)
            include_visualizations: Whether to include visualizations
            
        Returns:
            Report content as a string
        """
        if not validation_results:
            return "No validation results to report"
        
        if format not in self.config["report_formats"]:
            format = "html"  # Default to HTML if invalid format
        
        # Common report data
        report_data = self._prepare_report_data(validation_results)
        
        # Generate report based on format
        if format == "html":
            return self._generate_html_report(report_data, include_visualizations)
        elif format == "markdown":
            return self._generate_markdown_report(report_data, include_visualizations)
        elif format == "json":
            return self._generate_json_report(report_data)
        else:
            return self._generate_text_report(report_data)
    
    def export_report(
        self,
        validation_results: List[ValidationResult],
        output_path: str,
        format: str = "html",
        include_visualizations: bool = True
    ) -> str:
        """
        Export a validation report to a file.
        
        Args:
            validation_results: List of validation results
            output_path: Path to save the report
            format: Output format (html, markdown, json, etc.)
            include_visualizations: Whether to include visualizations
            
        Returns:
            Path to the saved report
        """
        # Generate report content
        report_content = self.generate_report(validation_results, format, include_visualizations)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Write report to file
        try:
            with open(output_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Report saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving report to {output_path}: {e}")
            return f"Error saving report: {e}"
    
    def _prepare_report_data(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Prepare data for the report.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            Dictionary with report data
        """
        # Group validation results by hardware and model
        hardware_model_results = {}
        
        for val_result in validation_results:
            hw_id = val_result.hardware_result.hardware_id
            model_id = val_result.hardware_result.model_id
            
            key = (hw_id, model_id)
            
            if key not in hardware_model_results:
                hardware_model_results[key] = []
            
            hardware_model_results[key].append(val_result)
        
        # Calculate summary statistics
        overall_mape_values = []
        hardware_stats = {}
        model_stats = {}
        hardware_model_stats = {}
        
        for (hw_id, model_id), results in hardware_model_results.items():
            # Initialize stats dictionaries if needed
            if hw_id not in hardware_stats:
                hardware_stats[hw_id] = {"mape_values": [], "count": 0}
            
            if model_id not in model_stats:
                model_stats[model_id] = {"mape_values": [], "count": 0}
            
            if (hw_id, model_id) not in hardware_model_stats:
                hardware_model_stats[(hw_id, model_id)] = {"mape_values": [], "count": 0}
            
            # Process each validation result
            for val_result in results:
                # Skip results with no metrics comparison
                if not val_result.metrics_comparison:
                    continue
                
                # Calculate mean MAPE for this result
                result_mape_values = []
                
                for metric, comparison in val_result.metrics_comparison.items():
                    if "mape" in comparison:
                        mape = comparison["mape"]
                        result_mape_values.append(mape)
                
                if result_mape_values:
                    mean_mape = sum(result_mape_values) / len(result_mape_values)
                    
                    overall_mape_values.append(mean_mape)
                    hardware_stats[hw_id]["mape_values"].append(mean_mape)
                    model_stats[model_id]["mape_values"].append(mean_mape)
                    hardware_model_stats[(hw_id, model_id)]["mape_values"].append(mean_mape)
            
            # Update count
            hardware_stats[hw_id]["count"] += len(results)
            model_stats[model_id]["count"] += len(results)
            hardware_model_stats[(hw_id, model_id)]["count"] += len(results)
        
        # Calculate mean MAPE for each group
        if overall_mape_values:
            overall_mean_mape = sum(overall_mape_values) / len(overall_mape_values)
        else:
            overall_mean_mape = None
        
        for hw_id, stats in hardware_stats.items():
            if stats["mape_values"]:
                stats["mean_mape"] = sum(stats["mape_values"]) / len(stats["mape_values"])
                stats["status"] = self._interpret_mape(stats["mean_mape"])
        
        for model_id, stats in model_stats.items():
            if stats["mape_values"]:
                stats["mean_mape"] = sum(stats["mape_values"]) / len(stats["mape_values"])
                stats["status"] = self._interpret_mape(stats["mean_mape"])
        
        for (hw_id, model_id), stats in hardware_model_stats.items():
            if stats["mape_values"]:
                stats["mean_mape"] = sum(stats["mape_values"]) / len(stats["mape_values"])
                stats["status"] = self._interpret_mape(stats["mean_mape"])
        
        # Prepare report data
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report_data = {
            "timestamp": timestamp,
            "title": self.config["report_title_template"].format(timestamp=timestamp),
            "total_results": len(validation_results),
            "overall_mape": overall_mean_mape,
            "overall_status": self._interpret_mape(overall_mean_mape) if overall_mean_mape is not None else "unknown",
            "hardware_stats": hardware_stats,
            "model_stats": model_stats,
            "hardware_model_stats": hardware_model_stats,
            "validation_results": validation_results,
            "grouped_results": hardware_model_results
        }
        
        return report_data
    
    def _generate_html_report(self, report_data: Dict[str, Any], include_visualizations: bool) -> str:
        """
        Generate an HTML report.
        
        Args:
            report_data: Report data
            include_visualizations: Whether to include visualizations
            
        Returns:
            HTML report as a string
        """
        # Basic HTML template
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3, h4 {{
                    color: #2c3e50;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .status-excellent {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .status-good {{
                    color: #2ecc71;
                }}
                .status-acceptable {{
                    color: #f39c12;
                }}
                .status-problematic {{
                    color: #e67e22;
                }}
                .status-poor {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .summary-box {{
                    background-color: #f8f9fa;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 15px;
                    margin-bottom: 20px;
                }}
                .metric-value {{
                    font-weight: bold;
                }}
                .visualization {{
                    margin-top: 20px;
                    margin-bottom: 20px;
                    border: 1px solid #ddd;
                    padding: 10px;
                    border-radius: 4px;
                }}
                .visualization-placeholder {{
                    background-color: #f8f9fa;
                    height: 300px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-style: italic;
                    color: #777;
                }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Generated on: {timestamp}</p>
            
            <div class="summary-box">
                <h2>Summary</h2>
                <p>Total validation results: <span class="metric-value">{total_results}</span></p>
                <p>Overall MAPE: <span class="metric-value">{overall_mape:.2f}%</span></p>
                <p>Overall status: <span class="status-{overall_status_class}">{overall_status}</span></p>
            </div>
            
            {hardware_summary}
            
            {model_summary}
            
            {hardware_model_summary}
            
            {visualizations}
            
            {detailed_results}
            
            <p><em>Report generated by the Simulation Accuracy and Validation Framework</em></p>
        </body>
        </html>
        """
        
        # Generate hardware summary
        hardware_summary = "<h2>Results by Hardware</h2>"
        hardware_summary += "<table>"
        hardware_summary += "<tr><th>Hardware</th><th>Count</th><th>MAPE</th><th>Status</th></tr>"
        
        for hw_id, stats in report_data["hardware_stats"].items():
            if "mean_mape" in stats:
                status_class = self._status_class(stats["status"])
                hardware_summary += f"<tr>"
                hardware_summary += f"<td>{hw_id}</td>"
                hardware_summary += f"<td>{stats['count']}</td>"
                hardware_summary += f"<td>{stats['mean_mape']:.2f}%</td>"
                hardware_summary += f"<td class='status-{status_class}'>{stats['status']}</td>"
                hardware_summary += f"</tr>"
        
        hardware_summary += "</table>"
        
        # Generate model summary
        model_summary = "<h2>Results by Model</h2>"
        model_summary += "<table>"
        model_summary += "<tr><th>Model</th><th>Count</th><th>MAPE</th><th>Status</th></tr>"
        
        for model_id, stats in report_data["model_stats"].items():
            if "mean_mape" in stats:
                status_class = self._status_class(stats["status"])
                model_summary += f"<tr>"
                model_summary += f"<td>{model_id}</td>"
                model_summary += f"<td>{stats['count']}</td>"
                model_summary += f"<td>{stats['mean_mape']:.2f}%</td>"
                model_summary += f"<td class='status-{status_class}'>{stats['status']}</td>"
                model_summary += f"</tr>"
        
        model_summary += "</table>"
        
        # Generate hardware-model summary
        hardware_model_summary = "<h2>Results by Hardware and Model</h2>"
        hardware_model_summary += "<table>"
        hardware_model_summary += "<tr><th>Hardware</th><th>Model</th><th>Count</th><th>MAPE</th><th>Status</th></tr>"
        
        for (hw_id, model_id), stats in report_data["hardware_model_stats"].items():
            if "mean_mape" in stats:
                status_class = self._status_class(stats["status"])
                hardware_model_summary += f"<tr>"
                hardware_model_summary += f"<td>{hw_id}</td>"
                hardware_model_summary += f"<td>{model_id}</td>"
                hardware_model_summary += f"<td>{stats['count']}</td>"
                hardware_model_summary += f"<td>{stats['mean_mape']:.2f}%</td>"
                hardware_model_summary += f"<td class='status-{status_class}'>{stats['status']}</td>"
                hardware_model_summary += f"</tr>"
        
        hardware_model_summary += "</table>"
        
        # Generate visualizations section
        visualizations = ""
        if include_visualizations:
            visualizations = "<h2>Visualizations</h2>"
            visualizations += "<div class='visualization'>"
            visualizations += "<h3>Error Distribution</h3>"
            visualizations += "<div class='visualization-placeholder'>Visualization placeholder: Error distribution chart would be shown here</div>"
            visualizations += "</div>"
            
            visualizations += "<div class='visualization'>"
            visualizations += "<h3>Error by Metric</h3>"
            visualizations += "<div class='visualization-placeholder'>Visualization placeholder: Error by metric chart would be shown here</div>"
            visualizations += "</div>"
            
            visualizations += "<div class='visualization'>"
            visualizations += "<h3>Hardware Comparison</h3>"
            visualizations += "<div class='visualization-placeholder'>Visualization placeholder: Hardware comparison chart would be shown here</div>"
            visualizations += "</div>"
        
        # Generate detailed results section (limited to max_results_per_page)
        detailed_results = "<h2>Detailed Results</h2>"
        detailed_results += "<p>Showing up to {0} of {1} results</p>".format(
            min(self.config["max_results_per_page"], report_data["total_results"]),
            report_data["total_results"]
        )
        
        detailed_results += "<table>"
        detailed_results += "<tr>"
        detailed_results += "<th>Hardware</th>"
        detailed_results += "<th>Model</th>"
        detailed_results += "<th>Batch Size</th>"
        detailed_results += "<th>Precision</th>"
        detailed_results += "<th>Throughput MAPE</th>"
        detailed_results += "<th>Latency MAPE</th>"
        detailed_results += "<th>Memory MAPE</th>"
        detailed_results += "<th>Power MAPE</th>"
        detailed_results += "</tr>"
        
        for i, val_result in enumerate(report_data["validation_results"]):
            if i >= self.config["max_results_per_page"]:
                break
                
            hw_result = val_result.hardware_result
            sim_result = val_result.simulation_result
            
            # Extract MAPE values for each metric
            throughput_mape = val_result.metrics_comparison.get("throughput_items_per_second", {}).get("mape", "N/A")
            latency_mape = val_result.metrics_comparison.get("average_latency_ms", {}).get("mape", "N/A")
            memory_mape = val_result.metrics_comparison.get("memory_peak_mb", {}).get("mape", "N/A")
            power_mape = val_result.metrics_comparison.get("power_consumption_w", {}).get("mape", "N/A")
            
            detailed_results += "<tr>"
            detailed_results += f"<td>{hw_result.hardware_id}</td>"
            detailed_results += f"<td>{hw_result.model_id}</td>"
            detailed_results += f"<td>{hw_result.batch_size}</td>"
            detailed_results += f"<td>{hw_result.precision}</td>"
            detailed_results += f"<td>{throughput_mape if throughput_mape == 'N/A' else f'{throughput_mape:.2f}%'}</td>"
            detailed_results += f"<td>{latency_mape if latency_mape == 'N/A' else f'{latency_mape:.2f}%'}</td>"
            detailed_results += f"<td>{memory_mape if memory_mape == 'N/A' else f'{memory_mape:.2f}%'}</td>"
            detailed_results += f"<td>{power_mape if power_mape == 'N/A' else f'{power_mape:.2f}%'}</td>"
            detailed_results += "</tr>"
        
        detailed_results += "</table>"
        
        # Format overall MAPE
        overall_mape_str = f"{report_data['overall_mape']:.2f}%" if report_data['overall_mape'] is not None else "N/A"
        
        # Render HTML template
        html_report = html_template.format(
            title=report_data["title"],
            timestamp=report_data["timestamp"],
            total_results=report_data["total_results"],
            overall_mape=report_data["overall_mape"] if report_data["overall_mape"] is not None else 0,
            overall_status=report_data["overall_status"],
            overall_status_class=self._status_class(report_data["overall_status"]),
            hardware_summary=hardware_summary,
            model_summary=model_summary,
            hardware_model_summary=hardware_model_summary,
            visualizations=visualizations,
            detailed_results=detailed_results
        )
        
        return html_report
    
    def _generate_markdown_report(self, report_data: Dict[str, Any], include_visualizations: bool) -> str:
        """
        Generate a Markdown report.
        
        Args:
            report_data: Report data
            include_visualizations: Whether to include visualizations
            
        Returns:
            Markdown report as a string
        """
        # Markdown template
        markdown = f"# {report_data['title']}\n\n"
        markdown += f"Generated on: {report_data['timestamp']}\n\n"
        
        # Summary
        markdown += "## Summary\n\n"
        markdown += f"Total validation results: **{report_data['total_results']}**\n\n"
        
        if report_data["overall_mape"] is not None:
            markdown += f"Overall MAPE: **{report_data['overall_mape']:.2f}%**\n\n"
            markdown += f"Overall status: **{report_data['overall_status']}**\n\n"
        else:
            markdown += "Overall MAPE: **N/A**\n\n"
            markdown += "Overall status: **unknown**\n\n"
        
        # Hardware summary
        markdown += "## Results by Hardware\n\n"
        markdown += "| Hardware | Count | MAPE | Status |\n"
        markdown += "| --- | --- | --- | --- |\n"
        
        for hw_id, stats in report_data["hardware_stats"].items():
            if "mean_mape" in stats:
                markdown += f"| {hw_id} | {stats['count']} | {stats['mean_mape']:.2f}% | {stats['status']} |\n"
        
        markdown += "\n"
        
        # Model summary
        markdown += "## Results by Model\n\n"
        markdown += "| Model | Count | MAPE | Status |\n"
        markdown += "| --- | --- | --- | --- |\n"
        
        for model_id, stats in report_data["model_stats"].items():
            if "mean_mape" in stats:
                markdown += f"| {model_id} | {stats['count']} | {stats['mean_mape']:.2f}% | {stats['status']} |\n"
        
        markdown += "\n"
        
        # Hardware-model summary
        markdown += "## Results by Hardware and Model\n\n"
        markdown += "| Hardware | Model | Count | MAPE | Status |\n"
        markdown += "| --- | --- | --- | --- | --- |\n"
        
        for (hw_id, model_id), stats in report_data["hardware_model_stats"].items():
            if "mean_mape" in stats:
                markdown += f"| {hw_id} | {model_id} | {stats['count']} | {stats['mean_mape']:.2f}% | {stats['status']} |\n"
        
        markdown += "\n"
        
        # Visualizations section
        if include_visualizations:
            markdown += "## Visualizations\n\n"
            markdown += "_Note: Visualizations are not available in Markdown format. Please use HTML format to view visualizations._\n\n"
        
        # Detailed results section
        markdown += "## Detailed Results\n\n"
        markdown += f"Showing up to {min(self.config['max_results_per_page'], report_data['total_results'])} of {report_data['total_results']} results\n\n"
        
        markdown += "| Hardware | Model | Batch Size | Precision | Throughput MAPE | Latency MAPE | Memory MAPE | Power MAPE |\n"
        markdown += "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        
        for i, val_result in enumerate(report_data["validation_results"]):
            if i >= self.config["max_results_per_page"]:
                break
                
            hw_result = val_result.hardware_result
            sim_result = val_result.simulation_result
            
            # Extract MAPE values for each metric
            throughput_mape = val_result.metrics_comparison.get("throughput_items_per_second", {}).get("mape", "N/A")
            latency_mape = val_result.metrics_comparison.get("average_latency_ms", {}).get("mape", "N/A")
            memory_mape = val_result.metrics_comparison.get("memory_peak_mb", {}).get("mape", "N/A")
            power_mape = val_result.metrics_comparison.get("power_consumption_w", {}).get("mape", "N/A")
            
            markdown += f"| {hw_result.hardware_id} | {hw_result.model_id} | {hw_result.batch_size} | {hw_result.precision} "
            markdown += f"| {throughput_mape if throughput_mape == 'N/A' else f'{throughput_mape:.2f}%'} "
            markdown += f"| {latency_mape if latency_mape == 'N/A' else f'{latency_mape:.2f}%'} "
            markdown += f"| {memory_mape if memory_mape == 'N/A' else f'{memory_mape:.2f}%'} "
            markdown += f"| {power_mape if power_mape == 'N/A' else f'{power_mape:.2f}%'} |\n"
        
        markdown += "\n"
        markdown += "_Report generated by the Simulation Accuracy and Validation Framework_"
        
        return markdown
    
    def _generate_json_report(self, report_data: Dict[str, Any]) -> str:
        """
        Generate a JSON report.
        
        Args:
            report_data: Report data
            
        Returns:
            JSON report as a string
        """
        # Filter out data that can't be serialized to JSON
        json_data = {
            "title": report_data["title"],
            "timestamp": report_data["timestamp"],
            "total_results": report_data["total_results"],
            "overall_mape": report_data["overall_mape"],
            "overall_status": report_data["overall_status"],
            "hardware_stats": report_data["hardware_stats"],
            "model_stats": report_data["model_stats"]
        }
        
        # Convert hardware_model_stats keys to strings for JSON
        hw_model_stats = {}
        for (hw_id, model_id), stats in report_data["hardware_model_stats"].items():
            key = f"{hw_id}__{model_id}"
            hw_model_stats[key] = stats
        
        json_data["hardware_model_stats"] = hw_model_stats
        
        # Add limited detailed results
        detailed_results = []
        for i, val_result in enumerate(report_data["validation_results"]):
            if i >= self.config["max_results_per_page"]:
                break
                
            hw_result = val_result.hardware_result
            sim_result = val_result.simulation_result
            
            # Extract MAPE values for each metric
            throughput_mape = val_result.metrics_comparison.get("throughput_items_per_second", {}).get("mape", None)
            latency_mape = val_result.metrics_comparison.get("average_latency_ms", {}).get("mape", None)
            memory_mape = val_result.metrics_comparison.get("memory_peak_mb", {}).get("mape", None)
            power_mape = val_result.metrics_comparison.get("power_consumption_w", {}).get("mape", None)
            
            detailed_results.append({
                "hardware_id": hw_result.hardware_id,
                "model_id": hw_result.model_id,
                "batch_size": hw_result.batch_size,
                "precision": hw_result.precision,
                "throughput_mape": throughput_mape,
                "latency_mape": latency_mape,
                "memory_mape": memory_mape,
                "power_mape": power_mape
            })
        
        json_data["detailed_results"] = detailed_results
        
        # Convert to JSON string
        try:
            return json.dumps(json_data, indent=2)
        except Exception as e:
            logger.error(f"Error generating JSON report: {e}")
            return json.dumps({"error": f"Error generating JSON report: {e}"})
    
    def _generate_text_report(self, report_data: Dict[str, Any]) -> str:
        """
        Generate a plain text report.
        
        Args:
            report_data: Report data
            
        Returns:
            Text report as a string
        """
        # Text template
        text = f"{report_data['title']}\n"
        text += f"Generated on: {report_data['timestamp']}\n\n"
        
        # Summary
        text += "SUMMARY\n"
        text += "=======\n"
        text += f"Total validation results: {report_data['total_results']}\n"
        
        if report_data["overall_mape"] is not None:
            text += f"Overall MAPE: {report_data['overall_mape']:.2f}%\n"
            text += f"Overall status: {report_data['overall_status']}\n\n"
        else:
            text += "Overall MAPE: N/A\n"
            text += "Overall status: unknown\n\n"
        
        # Hardware summary
        text += "RESULTS BY HARDWARE\n"
        text += "===================\n"
        text += f"{'Hardware':<20} {'Count':<8} {'MAPE':<10} {'Status':<15}\n"
        text += "-" * 60 + "\n"
        
        for hw_id, stats in report_data["hardware_stats"].items():
            if "mean_mape" in stats:
                text += f"{hw_id:<20} {stats['count']:<8} {stats['mean_mape']:.2f}%{'':5} {stats['status']:<15}\n"
        
        text += "\n"
        
        # Model summary
        text += "RESULTS BY MODEL\n"
        text += "================\n"
        text += f"{'Model':<25} {'Count':<8} {'MAPE':<10} {'Status':<15}\n"
        text += "-" * 60 + "\n"
        
        for model_id, stats in report_data["model_stats"].items():
            if "mean_mape" in stats:
                text += f"{model_id:<25} {stats['count']:<8} {stats['mean_mape']:.2f}%{'':5} {stats['status']:<15}\n"
        
        text += "\n"
        
        # Hardware-model summary
        text += "RESULTS BY HARDWARE AND MODEL\n"
        text += "=============================\n"
        text += f"{'Hardware':<20} {'Model':<25} {'Count':<8} {'MAPE':<10} {'Status':<15}\n"
        text += "-" * 80 + "\n"
        
        for (hw_id, model_id), stats in report_data["hardware_model_stats"].items():
            if "mean_mape" in stats:
                text += f"{hw_id:<20} {model_id:<25} {stats['count']:<8} {stats['mean_mape']:.2f}%{'':5} {stats['status']:<15}\n"
        
        text += "\n"
        
        # Detailed results section
        text += "DETAILED RESULTS\n"
        text += "===============\n"
        text += f"Showing up to {min(self.config['max_results_per_page'], report_data['total_results'])} of {report_data['total_results']} results\n\n"
        
        col_widths = {
            "hardware": 20,
            "model": 25,
            "batch": 8,
            "precision": 8,
            "throughput": 12,
            "latency": 12,
            "memory": 12,
            "power": 12
        }
        
        # Header
        text += f"{'Hardware':<{col_widths['hardware']}} {'Model':<{col_widths['model']}} {'Batch':<{col_widths['batch']}} {'Prec':<{col_widths['precision']}} "
        text += f"{'Throughput':<{col_widths['throughput']}} {'Latency':<{col_widths['latency']}} {'Memory':<{col_widths['memory']}} {'Power':<{col_widths['power']}}\n"
        text += "-" * (sum(col_widths.values()) + len(col_widths)) + "\n"
        
        for i, val_result in enumerate(report_data["validation_results"]):
            if i >= self.config["max_results_per_page"]:
                break
                
            hw_result = val_result.hardware_result
            sim_result = val_result.simulation_result
            
            # Extract MAPE values for each metric
            throughput_mape = val_result.metrics_comparison.get("throughput_items_per_second", {}).get("mape", "N/A")
            latency_mape = val_result.metrics_comparison.get("average_latency_ms", {}).get("mape", "N/A")
            memory_mape = val_result.metrics_comparison.get("memory_peak_mb", {}).get("mape", "N/A")
            power_mape = val_result.metrics_comparison.get("power_consumption_w", {}).get("mape", "N/A")
            
            # Format MAPE values
            throughput_str = "N/A" if throughput_mape == "N/A" else f"{throughput_mape:.2f}%"
            latency_str = "N/A" if latency_mape == "N/A" else f"{latency_mape:.2f}%"
            memory_str = "N/A" if memory_mape == "N/A" else f"{memory_mape:.2f}%"
            power_str = "N/A" if power_mape == "N/A" else f"{power_mape:.2f}%"
            
            text += f"{hw_result.hardware_id:<{col_widths['hardware']}} {hw_result.model_id:<{col_widths['model']}} {hw_result.batch_size:<{col_widths['batch']}} {hw_result.precision:<{col_widths['precision']}} "
            text += f"{throughput_str:<{col_widths['throughput']}} {latency_str:<{col_widths['latency']}} {memory_str:<{col_widths['memory']}} {power_str:<{col_widths['power']}}\n"
        
        text += "\n"
        text += "Report generated by the Simulation Accuracy and Validation Framework"
        
        return text
    
    def _interpret_mape(self, mape: Optional[float]) -> str:
        """
        Interpret MAPE value as a status.
        
        Args:
            mape: Mean Absolute Percentage Error value
            
        Returns:
            Status string based on MAPE
        """
        if mape is None:
            return "unknown"
        
        if mape < 5:
            return "excellent"
        elif mape < 10:
            return "good"
        elif mape < 15:
            return "acceptable"
        elif mape < 25:
            return "problematic"
        else:
            return "poor"
    
    def _status_class(self, status: str) -> str:
        """
        Convert status to CSS class name.
        
        Args:
            status: Status string
            
        Returns:
            CSS class name
        """
        if status == "excellent":
            return "excellent"
        elif status == "good":
            return "good"
        elif status == "acceptable":
            return "acceptable"
        elif status == "problematic":
            return "problematic"
        elif status == "poor":
            return "poor"
        else:
            return "unknown"