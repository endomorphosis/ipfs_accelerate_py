#!/usr/bin/env python3
"""
Validation Reporter implementation for the Benchmark Validation System.

This module provides a concrete implementation of the ValidationReporter interface
that generates reports and visualizations of validation results.
"""

import os
import logging
import json
import datetime
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("benchmark_validation_reporter")

# Import base classes
from data.duckdb.benchmark_validation.core.base import (
    ValidationLevel,
    ValidationStatus,
    BenchmarkResult,
    ValidationResult,
    ValidationReporter
)

# Import Advanced Visualization System components if available
try:
    from data.duckdb.visualization.advanced_visualization.viz_heatmap import HardwareHeatmapVisualization
    from data.duckdb.visualization.advanced_visualization.viz_time_series import TimeSeriesVisualization
    from data.duckdb.visualization.advanced_visualization.viz_3d import ThreeDVisualization
    from data.duckdb.visualization.advanced_visualization.export_integration import ExportIntegration
    ADVANCED_VIZ_AVAILABLE = True
except ImportError:
    logger.warning("Advanced visualization components not available. Falling back to basic visualizations.")
    ADVANCED_VIZ_AVAILABLE = False

# Import other visualization dependencies if available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("Pandas not available. Some visualization features will be limited.")
    PANDAS_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not available. Install with: pip install plotly")
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not available. Install with: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False


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
            "visualization_types": ["confidence_distribution", "metric_comparison", "validation_heatmap"],
            "max_results_per_page": 20,
            "output_directory": "output",
            "report_title_template": "Benchmark Validation Report - {timestamp}",
            "css_style_path": None,
            "html_template_path": None,
            "theme": "light"
        }
        
        # Apply default config values if not specified
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Initialize visualization components if available
        self.viz_components = {}
        if ADVANCED_VIZ_AVAILABLE:
            self.viz_components["heatmap"] = HardwareHeatmapVisualization(theme=self.config["theme"])
            self.viz_components["time_series"] = TimeSeriesVisualization(theme=self.config["theme"])
            self.viz_components["3d"] = ThreeDVisualization(theme=self.config["theme"])
            # Create export integration
            self.export_integration = ExportIntegration()
            self.export_integration.initialize_export_manager(self.config["output_directory"])
        
        logger.info("Validation Reporter initialized")
                
    def generate_report(
        self,
        validation_results: List[ValidationResult],
        report_format: str = "html",
        include_visualizations: bool = True
    ) -> str:
        """
        Generate a validation report.
        
        Args:
            validation_results: List of validation results
            report_format: Output format (html, markdown, json, etc.)
            include_visualizations: Whether to include visualizations
            
        Returns:
            Report content as a string
        """
        if not validation_results:
            return "No validation results to report"
        
        if report_format not in self.config["report_formats"]:
            report_format = "html"  # Default to HTML if invalid format
        
        # Common report data
        report_data = self._prepare_report_data(validation_results)
        
        # Generate report based on format
        if report_format == "html":
            return self._generate_html_report(report_data, include_visualizations)
        elif report_format == "markdown":
            return self._generate_markdown_report(report_data, include_visualizations)
        elif report_format == "json":
            return self._generate_json_report(report_data)
        else:
            return self._generate_text_report(report_data)
    
    def export_report(
        self,
        validation_results: List[ValidationResult],
        output_path: str,
        report_format: str = "html",
        include_visualizations: bool = True
    ) -> str:
        """
        Export a validation report to a file.
        
        Args:
            validation_results: List of validation results
            output_path: Path to save the report
            report_format: Output format (html, markdown, json, etc.)
            include_visualizations: Whether to include visualizations
            
        Returns:
            Path to the saved report
        """
        # Generate report content
        report_content = self.generate_report(validation_results, report_format, include_visualizations)
        
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
        # Group validation results by benchmark type and validation level
        benchmark_type_results = {}
        validation_level_results = {}
        
        for val_result in validation_results:
            # Group by benchmark type
            benchmark_type = val_result.benchmark_result.benchmark_type.name
            if benchmark_type not in benchmark_type_results:
                benchmark_type_results[benchmark_type] = []
            benchmark_type_results[benchmark_type].append(val_result)
            
            # Group by validation level
            validation_level = val_result.validation_level.name
            if validation_level not in validation_level_results:
                validation_level_results[validation_level] = []
            validation_level_results[validation_level].append(val_result)
        
        # Calculate summary statistics
        status_counts = {
            "VALID": 0,
            "INVALID": 0,
            "WARNING": 0,
            "ERROR": 0,
            "PENDING": 0
        }
        
        confidence_scores = []
        benchmark_type_stats = {}
        validation_level_stats = {}
        
        # Process each validation result
        for val_result in validation_results:
            # Count by status
            status_name = val_result.status.name
            status_counts[status_name] = status_counts.get(status_name, 0) + 1
            
            # Collect confidence scores
            confidence_scores.append(val_result.confidence_score)
            
            # Stats by benchmark type
            benchmark_type = val_result.benchmark_result.benchmark_type.name
            if benchmark_type not in benchmark_type_stats:
                benchmark_type_stats[benchmark_type] = {
                    "confidence_scores": [],
                    "status_counts": {"VALID": 0, "INVALID": 0, "WARNING": 0, "ERROR": 0, "PENDING": 0},
                    "count": 0
                }
            
            benchmark_type_stats[benchmark_type]["confidence_scores"].append(val_result.confidence_score)
            benchmark_type_stats[benchmark_type]["status_counts"][status_name] += 1
            benchmark_type_stats[benchmark_type]["count"] += 1
            
            # Stats by validation level
            validation_level = val_result.validation_level.name
            if validation_level not in validation_level_stats:
                validation_level_stats[validation_level] = {
                    "confidence_scores": [],
                    "status_counts": {"VALID": 0, "INVALID": 0, "WARNING": 0, "ERROR": 0, "PENDING": 0},
                    "count": 0
                }
            
            validation_level_stats[validation_level]["confidence_scores"].append(val_result.confidence_score)
            validation_level_stats[validation_level]["status_counts"][status_name] += 1
            validation_level_stats[validation_level]["count"] += 1
        
        # Calculate mean confidence score
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Calculate mean confidence scores for each group
        for benchmark_type, stats in benchmark_type_stats.items():
            scores = stats["confidence_scores"]
            stats["mean_confidence"] = sum(scores) / len(scores) if scores else 0
            
        for validation_level, stats in validation_level_stats.items():
            scores = stats["confidence_scores"]
            stats["mean_confidence"] = sum(scores) / len(scores) if scores else 0
        
        # Prepare report data
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report_data = {
            "timestamp": timestamp,
            "title": self.config["report_title_template"].format(timestamp=timestamp),
            "total_results": len(validation_results),
            "status_counts": status_counts,
            "overall_confidence": overall_confidence,
            "benchmark_type_stats": benchmark_type_stats,
            "validation_level_stats": validation_level_stats,
            "benchmark_type_results": benchmark_type_results,
            "validation_level_results": validation_level_results,
            "validation_results": validation_results
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
                .status-VALID {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .status-INVALID {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .status-WARNING {{
                    color: #f39c12;
                }}
                .status-ERROR {{
                    color: #e74c3c;
                }}
                .status-PENDING {{
                    color: #3498db;
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
                .confidence {{
                    display: inline-block;
                    width: 100px;
                    height: 20px;
                    background: linear-gradient(to right, #e74c3c, #f39c12, #2ecc71);
                    border-radius: 10px;
                    position: relative;
                }}
                .confidence-marker {{
                    position: absolute;
                    width: 10px;
                    height: 20px;
                    background-color: #2c3e50;
                    border-radius: 5px;
                }}
                .issues-list {{
                    background-color: #fff3cd;
                    border: 1px solid #ffeeba;
                    border-radius: 4px;
                    padding: 15px;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Generated on: {timestamp}</p>
            
            <div class="summary-box">
                <h2>Summary</h2>
                <p>Total validation results: <span class="metric-value">{total_results}</span></p>
                <p>Overall confidence score: <span class="metric-value">{overall_confidence:.2f}</span></p>
                <p>
                    Status breakdown: 
                    <span class="status-VALID">Valid: {status_valid}</span>, 
                    <span class="status-WARNING">Warning: {status_warning}</span>, 
                    <span class="status-INVALID">Invalid: {status_invalid}</span>, 
                    <span class="status-ERROR">Error: {status_error}</span>, 
                    <span class="status-PENDING">Pending: {status_pending}</span>
                </p>
            </div>
            
            {benchmark_type_summary}
            
            {validation_level_summary}
            
            {visualizations}
            
            {detailed_results}
            
            <p><em>Report generated by the Benchmark Validation System</em></p>
        </body>
        </html>
        """
        
        # Generate benchmark type summary
        benchmark_type_summary = "<h2>Results by Benchmark Type</h2>"
        benchmark_type_summary += "<table>"
        benchmark_type_summary += "<tr><th>Benchmark Type</th><th>Count</th><th>Mean Confidence</th><th>Valid</th><th>Warning</th><th>Invalid</th><th>Error</th><th>Pending</th></tr>"
        
        for benchmark_type, stats in report_data["benchmark_type_stats"].items():
            benchmark_type_summary += f"<tr>"
            benchmark_type_summary += f"<td>{benchmark_type}</td>"
            benchmark_type_summary += f"<td>{stats['count']}</td>"
            benchmark_type_summary += f"<td>{stats['mean_confidence']:.2f}</td>"
            benchmark_type_summary += f"<td class='status-VALID'>{stats['status_counts']['VALID']}</td>"
            benchmark_type_summary += f"<td class='status-WARNING'>{stats['status_counts']['WARNING']}</td>"
            benchmark_type_summary += f"<td class='status-INVALID'>{stats['status_counts']['INVALID']}</td>"
            benchmark_type_summary += f"<td class='status-ERROR'>{stats['status_counts']['ERROR']}</td>"
            benchmark_type_summary += f"<td class='status-PENDING'>{stats['status_counts']['PENDING']}</td>"
            benchmark_type_summary += f"</tr>"
        
        benchmark_type_summary += "</table>"
        
        # Generate validation level summary
        validation_level_summary = "<h2>Results by Validation Level</h2>"
        validation_level_summary += "<table>"
        validation_level_summary += "<tr><th>Validation Level</th><th>Count</th><th>Mean Confidence</th><th>Valid</th><th>Warning</th><th>Invalid</th><th>Error</th><th>Pending</th></tr>"
        
        for validation_level, stats in report_data["validation_level_stats"].items():
            validation_level_summary += f"<tr>"
            validation_level_summary += f"<td>{validation_level}</td>"
            validation_level_summary += f"<td>{stats['count']}</td>"
            validation_level_summary += f"<td>{stats['mean_confidence']:.2f}</td>"
            validation_level_summary += f"<td class='status-VALID'>{stats['status_counts']['VALID']}</td>"
            validation_level_summary += f"<td class='status-WARNING'>{stats['status_counts']['WARNING']}</td>"
            validation_level_summary += f"<td class='status-INVALID'>{stats['status_counts']['INVALID']}</td>"
            validation_level_summary += f"<td class='status-ERROR'>{stats['status_counts']['ERROR']}</td>"
            validation_level_summary += f"<td class='status-PENDING'>{stats['status_counts']['PENDING']}</td>"
            validation_level_summary += f"</tr>"
        
        validation_level_summary += "</table>"
        
        # Generate visualizations section
        visualizations = ""
        if include_visualizations and (PLOTLY_AVAILABLE or MATPLOTLIB_AVAILABLE or ADVANCED_VIZ_AVAILABLE):
            visualizations = "<h2>Visualizations</h2>"
            
            # Generate confidence score distribution visualization
            visualizations += "<div class='visualization'>"
            visualizations += "<h3>Confidence Score Distribution</h3>"
            
            if PLOTLY_AVAILABLE and PANDAS_AVAILABLE:
                # Create confidence distribution plot
                confidence_scores = [val_result.confidence_score for val_result in report_data["validation_results"]]
                confidence_statuses = [val_result.status.name for val_result in report_data["validation_results"]]
                
                df = pd.DataFrame({
                    'confidence_score': confidence_scores,
                    'status': confidence_statuses
                })
                
                fig = px.histogram(df, x="confidence_score", color="status", 
                                   title="Confidence Score Distribution",
                                   labels={"confidence_score": "Confidence Score", "count": "Count"},
                                   barmode="group",
                                   color_discrete_map={
                                       "VALID": "#27ae60",
                                       "WARNING": "#f39c12",
                                       "INVALID": "#e74c3c",
                                       "ERROR": "#e74c3c",
                                       "PENDING": "#3498db"
                                   })
                
                visualizations += fig.to_html(full_html=False, include_plotlyjs='cdn')
            else:
                visualizations += "<div class='visualization-placeholder'>Plotly and pandas required for confidence distribution chart</div>"
            
            visualizations += "</div>"
            
            # Generate validation status by benchmark type visualization
            visualizations += "<div class='visualization'>"
            visualizations += "<h3>Validation Status by Benchmark Type</h3>"
            
            if PLOTLY_AVAILABLE:
                # Extract data
                benchmark_types = list(report_data["benchmark_type_stats"].keys())
                valid_counts = [stats["status_counts"]["VALID"] for stats in report_data["benchmark_type_stats"].values()]
                warning_counts = [stats["status_counts"]["WARNING"] for stats in report_data["benchmark_type_stats"].values()]
                invalid_counts = [stats["status_counts"]["INVALID"] for stats in report_data["benchmark_type_stats"].values()]
                error_counts = [stats["status_counts"]["ERROR"] for stats in report_data["benchmark_type_stats"].values()]
                pending_counts = [stats["status_counts"]["PENDING"] for stats in report_data["benchmark_type_stats"].values()]
                
                # Create grouped bar chart
                fig = go.Figure(data=[
                    go.Bar(name='Valid', x=benchmark_types, y=valid_counts, marker_color='#27ae60'),
                    go.Bar(name='Warning', x=benchmark_types, y=warning_counts, marker_color='#f39c12'),
                    go.Bar(name='Invalid', x=benchmark_types, y=invalid_counts, marker_color='#e74c3c'),
                    go.Bar(name='Error', x=benchmark_types, y=error_counts, marker_color='#c0392b'),
                    go.Bar(name='Pending', x=benchmark_types, y=pending_counts, marker_color='#3498db')
                ])
                
                # Update layout
                fig.update_layout(
                    title='Validation Status by Benchmark Type',
                    xaxis_title='Benchmark Type',
                    yaxis_title='Count',
                    barmode='stack',
                    template='plotly_white'
                )
                
                visualizations += fig.to_html(full_html=False, include_plotlyjs='cdn')
            else:
                visualizations += "<div class='visualization-placeholder'>Plotly required for benchmark type chart</div>"
            
            visualizations += "</div>"
            
            # Generate mean confidence by validation level visualization
            visualizations += "<div class='visualization'>"
            visualizations += "<h3>Mean Confidence by Validation Level</h3>"
            
            if PLOTLY_AVAILABLE:
                # Extract data
                validation_levels = list(report_data["validation_level_stats"].keys())
                mean_confidences = [stats["mean_confidence"] for stats in report_data["validation_level_stats"].values()]
                
                # Create bar chart
                fig = go.Figure(data=[
                    go.Bar(x=validation_levels, y=mean_confidences, 
                           marker_color='#3498db',
                           text=[f"{conf:.2f}" for conf in mean_confidences],
                           textposition='auto')
                ])
                
                # Update layout
                fig.update_layout(
                    title='Mean Confidence by Validation Level',
                    xaxis_title='Validation Level',
                    yaxis_title='Mean Confidence Score',
                    template='plotly_white',
                    yaxis=dict(range=[0, 1])
                )
                
                visualizations += fig.to_html(full_html=False, include_plotlyjs='cdn')
            else:
                visualizations += "<div class='visualization-placeholder'>Plotly required for validation level chart</div>"
            
            visualizations += "</div>"
        
        # Generate detailed results section
        detailed_results = "<h2>Detailed Results</h2>"
        detailed_results += "<p>Showing up to {0} of {1} results</p>".format(
            min(self.config["max_results_per_page"], report_data["total_results"]),
            report_data["total_results"]
        )
        
        detailed_results += "<table>"
        detailed_results += "<tr>"
        detailed_results += "<th>ID</th>"
        detailed_results += "<th>Model</th>"
        detailed_results += "<th>Hardware</th>"
        detailed_results += "<th>Benchmark Type</th>"
        detailed_results += "<th>Validation Level</th>"
        detailed_results += "<th>Status</th>"
        detailed_results += "<th>Confidence</th>"
        detailed_results += "<th>Issues</th>"
        detailed_results += "</tr>"
        
        for i, val_result in enumerate(report_data["validation_results"]):
            if i >= self.config["max_results_per_page"]:
                break
                
            benchmark_result = val_result.benchmark_result
            
            # Create confidence score visual indicator
            confidence_marker = val_result.confidence_score * 100 - 5  # Adjust for marker width
            confidence_viz = f"""
            <div class="confidence">
                <div class="confidence-marker" style="left: {confidence_marker}px;"></div>
            </div>
            """
            
            # Format issues
            issues_text = ""
            if val_result.issues:
                issues_text = "<ul>"
                for issue in val_result.issues[:2]:  # Limit to first two issues
                    if isinstance(issue, dict) and "description" in issue:
                        issues_text += f"<li>{issue['description']}</li>"
                    else:
                        issues_text += f"<li>{str(issue)}</li>"
                
                if len(val_result.issues) > 2:
                    issues_text += f"<li>... {len(val_result.issues) - 2} more issue(s)</li>"
                
                issues_text += "</ul>"
            
            detailed_results += "<tr>"
            detailed_results += f"<td>{val_result.id[:8]}</td>"
            detailed_results += f"<td>{benchmark_result.model_id or 'N/A'}</td>"
            detailed_results += f"<td>{benchmark_result.hardware_id or 'N/A'}</td>"
            detailed_results += f"<td>{benchmark_result.benchmark_type.name}</td>"
            detailed_results += f"<td>{val_result.validation_level.name}</td>"
            detailed_results += f"<td class='status-{val_result.status.name}'>{val_result.status.name}</td>"
            detailed_results += f"<td>{confidence_viz} {val_result.confidence_score:.2f}</td>"
            detailed_results += f"<td>{issues_text}</td>"
            detailed_results += "</tr>"
        
        detailed_results += "</table>"
        
        # Render HTML template
        html_report = html_template.format(
            title=report_data["title"],
            timestamp=report_data["timestamp"],
            total_results=report_data["total_results"],
            overall_confidence=report_data["overall_confidence"],
            status_valid=report_data["status_counts"]["VALID"],
            status_warning=report_data["status_counts"]["WARNING"],
            status_invalid=report_data["status_counts"]["INVALID"],
            status_error=report_data["status_counts"]["ERROR"],
            status_pending=report_data["status_counts"]["PENDING"],
            benchmark_type_summary=benchmark_type_summary,
            validation_level_summary=validation_level_summary,
            visualizations=visualizations if include_visualizations else "",
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
        markdown += f"Overall confidence score: **{report_data['overall_confidence']:.2f}**\n\n"
        markdown += "Status breakdown:\n\n"
        markdown += f"- Valid: **{report_data['status_counts']['VALID']}**\n"
        markdown += f"- Warning: **{report_data['status_counts']['WARNING']}**\n"
        markdown += f"- Invalid: **{report_data['status_counts']['INVALID']}**\n"
        markdown += f"- Error: **{report_data['status_counts']['ERROR']}**\n"
        markdown += f"- Pending: **{report_data['status_counts']['PENDING']}**\n\n"
        
        # Benchmark type summary
        markdown += "## Results by Benchmark Type\n\n"
        markdown += "| Benchmark Type | Count | Mean Confidence | Valid | Warning | Invalid | Error | Pending |\n"
        markdown += "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        
        for benchmark_type, stats in report_data["benchmark_type_stats"].items():
            markdown += f"| {benchmark_type} | {stats['count']} | {stats['mean_confidence']:.2f} | "
            markdown += f"{stats['status_counts']['VALID']} | {stats['status_counts']['WARNING']} | "
            markdown += f"{stats['status_counts']['INVALID']} | {stats['status_counts']['ERROR']} | "
            markdown += f"{stats['status_counts']['PENDING']} |\n"
        
        markdown += "\n"
        
        # Validation level summary
        markdown += "## Results by Validation Level\n\n"
        markdown += "| Validation Level | Count | Mean Confidence | Valid | Warning | Invalid | Error | Pending |\n"
        markdown += "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        
        for validation_level, stats in report_data["validation_level_stats"].items():
            markdown += f"| {validation_level} | {stats['count']} | {stats['mean_confidence']:.2f} | "
            markdown += f"{stats['status_counts']['VALID']} | {stats['status_counts']['WARNING']} | "
            markdown += f"{stats['status_counts']['INVALID']} | {stats['status_counts']['ERROR']} | "
            markdown += f"{stats['status_counts']['PENDING']} |\n"
        
        markdown += "\n"
        
        # Visualizations section
        if include_visualizations:
            markdown += "## Visualizations\n\n"
            markdown += "_Note: Visualizations are not available in Markdown format. Please use HTML format to view visualizations._\n\n"
        
        # Detailed results section
        markdown += "## Detailed Results\n\n"
        markdown += f"Showing up to {min(self.config['max_results_per_page'], report_data['total_results'])} of {report_data['total_results']} results\n\n"
        
        markdown += "| ID | Model | Hardware | Benchmark Type | Validation Level | Status | Confidence | Issues |\n"
        markdown += "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        
        for i, val_result in enumerate(report_data["validation_results"]):
            if i >= self.config["max_results_per_page"]:
                break
                
            benchmark_result = val_result.benchmark_result
            
            # Format issues
            issues_text = ""
            if val_result.issues:
                issues = []
                for issue in val_result.issues[:2]:  # Limit to first two issues
                    if isinstance(issue, dict) and "description" in issue:
                        issues.append(issue["description"])
                    else:
                        issues.append(str(issue))
                
                if len(val_result.issues) > 2:
                    issues.append(f"... {len(val_result.issues) - 2} more issue(s)")
                
                issues_text = "<br>".join(issues)
            
            markdown += f"| {val_result.id[:8]} | {benchmark_result.model_id or 'N/A'} | "
            markdown += f"{benchmark_result.hardware_id or 'N/A'} | {benchmark_result.benchmark_type.name} | "
            markdown += f"{val_result.validation_level.name} | {val_result.status.name} | "
            markdown += f"{val_result.confidence_score:.2f} | {issues_text} |\n"
        
        markdown += "\n"
        markdown += "_Report generated by the Benchmark Validation System_"
        
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
            "overall_confidence": report_data["overall_confidence"],
            "status_counts": report_data["status_counts"],
            "benchmark_type_stats": report_data["benchmark_type_stats"],
            "validation_level_stats": report_data["validation_level_stats"]
        }
        
        # Add detailed results
        detailed_results = []
        for i, val_result in enumerate(report_data["validation_results"]):
            if i >= self.config["max_results_per_page"]:
                break
                
            benchmark_result = val_result.benchmark_result
            
            detailed_results.append({
                "id": val_result.id,
                "model_id": benchmark_result.model_id,
                "hardware_id": benchmark_result.hardware_id,
                "benchmark_type": benchmark_result.benchmark_type.name,
                "validation_level": val_result.validation_level.name,
                "status": val_result.status.name,
                "confidence_score": val_result.confidence_score,
                "issues": val_result.issues,
                "recommendations": val_result.recommendations
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
        text += f"Overall confidence score: {report_data['overall_confidence']:.2f}\n"
        text += "Status breakdown:\n"
        text += f"  Valid: {report_data['status_counts']['VALID']}\n"
        text += f"  Warning: {report_data['status_counts']['WARNING']}\n"
        text += f"  Invalid: {report_data['status_counts']['INVALID']}\n"
        text += f"  Error: {report_data['status_counts']['ERROR']}\n"
        text += f"  Pending: {report_data['status_counts']['PENDING']}\n\n"
        
        # Benchmark type summary
        text += "RESULTS BY BENCHMARK TYPE\n"
        text += "=========================\n"
        text += f"{'Benchmark Type':<20} {'Count':<8} {'Confidence':<12} {'Valid':<6} {'Warning':<8} {'Invalid':<8} {'Error':<6} {'Pending':<8}\n"
        text += "-" * 80 + "\n"
        
        for benchmark_type, stats in report_data["benchmark_type_stats"].items():
            text += f"{benchmark_type:<20} {stats['count']:<8} {stats['mean_confidence']:.2f}{'':<6} "
            text += f"{stats['status_counts']['VALID']:<6} {stats['status_counts']['WARNING']:<8} "
            text += f"{stats['status_counts']['INVALID']:<8} {stats['status_counts']['ERROR']:<6} "
            text += f"{stats['status_counts']['PENDING']:<8}\n"
        
        text += "\n"
        
        # Validation level summary
        text += "RESULTS BY VALIDATION LEVEL\n"
        text += "===========================\n"
        text += f"{'Validation Level':<20} {'Count':<8} {'Confidence':<12} {'Valid':<6} {'Warning':<8} {'Invalid':<8} {'Error':<6} {'Pending':<8}\n"
        text += "-" * 80 + "\n"
        
        for validation_level, stats in report_data["validation_level_stats"].items():
            text += f"{validation_level:<20} {stats['count']:<8} {stats['mean_confidence']:.2f}{'':<6} "
            text += f"{stats['status_counts']['VALID']:<6} {stats['status_counts']['WARNING']:<8} "
            text += f"{stats['status_counts']['INVALID']:<8} {stats['status_counts']['ERROR']:<6} "
            text += f"{stats['status_counts']['PENDING']:<8}\n"
        
        text += "\n"
        
        # Detailed results section
        text += "DETAILED RESULTS\n"
        text += "===============\n"
        text += f"Showing up to {min(self.config['max_results_per_page'], report_data['total_results'])} of {report_data['total_results']} results\n\n"
        
        col_widths = {
            "id": 10,
            "model": 15,
            "hardware": 15,
            "type": 15,
            "level": 12,
            "status": 10,
            "confidence": 12,
            "issues": 30
        }
        
        # Header
        text += f"{'ID':<{col_widths['id']}} {'Model':<{col_widths['model']}} {'Hardware':<{col_widths['hardware']}} "
        text += f"{'Type':<{col_widths['type']}} {'Level':<{col_widths['level']}} {'Status':<{col_widths['status']}} "
        text += f"{'Confidence':<{col_widths['confidence']}} {'Issues':<{col_widths['issues']}}\n"
        text += "-" * (sum(col_widths.values()) + len(col_widths)) + "\n"
        
        for i, val_result in enumerate(report_data["validation_results"]):
            if i >= self.config["max_results_per_page"]:
                break
                
            benchmark_result = val_result.benchmark_result
            
            # Format issues
            issues_text = ""
            if val_result.issues:
                # Take the first issue
                if isinstance(val_result.issues[0], dict) and "description" in val_result.issues[0]:
                    issues_text = val_result.issues[0]["description"]
                else:
                    issues_text = str(val_result.issues[0])
                
                # Truncate if too long
                if len(issues_text) > col_widths["issues"] - 5:
                    issues_text = issues_text[:col_widths["issues"]-8] + "..."
                
                # Indicate if there are more issues
                if len(val_result.issues) > 1:
                    issues_text += f" (+{len(val_result.issues)-1})"
            
            text += f"{val_result.id[:8]:<{col_widths['id']}} "
            text += f"{(benchmark_result.model_id or 'N/A')[:col_widths['model']]:<{col_widths['model']}} "
            text += f"{(benchmark_result.hardware_id or 'N/A')[:col_widths['hardware']]:<{col_widths['hardware']}} "
            text += f"{benchmark_result.benchmark_type.name:<{col_widths['type']}} "
            text += f"{val_result.validation_level.name:<{col_widths['level']}} "
            text += f"{val_result.status.name:<{col_widths['status']}} "
            text += f"{val_result.confidence_score:.2f}{'':<{col_widths['confidence']-5}} "
            text += f"{issues_text:<{col_widths['issues']}}\n"
        
        text += "\n"
        text += "Report generated by the Benchmark Validation System"
        
        return text
    
    def create_visualization(
        self,
        validation_results: List[ValidationResult],
        visualization_type: str = "confidence_distribution",
        output_path: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Create a specific visualization for validation results.
        
        Args:
            validation_results: List of validation results
            visualization_type: Type of visualization (confidence_distribution, validation_heatmap, etc.)
            output_path: Path to save the visualization
            title: Title for the visualization
            **kwargs: Additional parameters for the visualization
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        if not validation_results:
            logger.error("No validation results to visualize")
            return None
        
        # Generate default title if not provided
        if title is None:
            title = f"Benchmark Validation - {visualization_type.replace('_', ' ').title()}"
        
        # Check if we should use advanced visualization system
        if ADVANCED_VIZ_AVAILABLE and visualization_type in ["validation_heatmap", "benchmark_comparison"]:
            return self._create_advanced_visualization(validation_results, visualization_type, output_path, title, **kwargs)
        elif PLOTLY_AVAILABLE:
            return self._create_plotly_visualization(validation_results, visualization_type, output_path, title, **kwargs)
        elif MATPLOTLIB_AVAILABLE:
            return self._create_matplotlib_visualization(validation_results, visualization_type, output_path, title, **kwargs)
        else:
            logger.error("No visualization libraries available")
            return None
    
    def _create_advanced_visualization(
        self,
        validation_results: List[ValidationResult],
        visualization_type: str,
        output_path: Optional[str],
        title: str,
        **kwargs
    ) -> Optional[str]:
        """
        Create a visualization using the Advanced Visualization System.
        
        Args:
            validation_results: List of validation results
            visualization_type: Type of visualization
            output_path: Path to save the visualization
            title: Title for the visualization
            **kwargs: Additional parameters for the visualization
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        if visualization_type == "validation_heatmap":
            # Prepare data for heatmap (model x hardware with confidence scores)
            data = self._prepare_validation_heatmap_data(validation_results)
            
            # Create heatmap visualization
            metric = kwargs.get("metric", "confidence_score")
            return self.viz_components["heatmap"].create_hardware_heatmap(
                data=data,
                metric=metric,
                output_path=output_path,
                title=title,
                **kwargs
            )
        elif visualization_type == "benchmark_comparison":
            # Create a comparison of validation results across benchmark types
            # This would be a custom visualization
            pass
        else:
            logger.error(f"Unsupported advanced visualization type: {visualization_type}")
            return None
    
    def _create_plotly_visualization(
        self,
        validation_results: List[ValidationResult],
        visualization_type: str,
        output_path: Optional[str],
        title: str,
        **kwargs
    ) -> Optional[str]:
        """
        Create a visualization using Plotly.
        
        Args:
            validation_results: List of validation results
            visualization_type: Type of visualization
            output_path: Path to save the visualization
            title: Title for the visualization
            **kwargs: Additional parameters for the visualization
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        try:
            if visualization_type == "confidence_distribution":
                # Extract confidence scores and statuses
                confidence_scores = [val_result.confidence_score for val_result in validation_results]
                statuses = [val_result.status.name for val_result in validation_results]
                
                # Create a DataFrame for the data
                if PANDAS_AVAILABLE:
                    df = pd.DataFrame({
                        'confidence_score': confidence_scores,
                        'status': statuses
                    })
                    
                    # Create histogram
                    fig = px.histogram(df, x="confidence_score", color="status", 
                                      title=title,
                                      labels={"confidence_score": "Confidence Score", "count": "Count"},
                                      barmode="group",
                                      color_discrete_map={
                                          "VALID": "#27ae60",
                                          "WARNING": "#f39c12",
                                          "INVALID": "#e74c3c",
                                          "ERROR": "#e74c3c",
                                          "PENDING": "#3498db"
                                      })
                else:
                    # Create figure manually if pandas isn't available
                    fig = go.Figure()
                    
                    # Group by status
                    status_groups = {}
                    for score, status in zip(confidence_scores, statuses):
                        if status not in status_groups:
                            status_groups[status] = []
                        status_groups[status].append(score)
                    
                    # Add a trace for each status
                    colors = {
                        "VALID": "#27ae60",
                        "WARNING": "#f39c12",
                        "INVALID": "#e74c3c",
                        "ERROR": "#e74c3c",
                        "PENDING": "#3498db"
                    }
                    
                    for status, scores in status_groups.items():
                        fig.add_trace(go.Histogram(
                            x=scores,
                            name=status,
                            marker_color=colors.get(status, "#7f8c8d")
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title=title,
                        xaxis_title="Confidence Score",
                        yaxis_title="Count",
                        barmode='group'
                    )
                
                # Save figure if output path is provided
                if output_path:
                    fig.write_html(output_path)
                    logger.info(f"Saved confidence distribution visualization to {output_path}")
                    return output_path
                else:
                    # Return as HTML string if no path provided
                    return fig.to_html()
                
            elif visualization_type == "validation_metrics":
                # Create a scatter plot of validation metrics
                pass
            else:
                logger.error(f"Unsupported Plotly visualization type: {visualization_type}")
                return None
        except Exception as e:
            logger.error(f"Error creating Plotly visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _create_matplotlib_visualization(
        self,
        validation_results: List[ValidationResult],
        visualization_type: str,
        output_path: Optional[str],
        title: str,
        **kwargs
    ) -> Optional[str]:
        """
        Create a visualization using Matplotlib.
        
        Args:
            validation_results: List of validation results
            visualization_type: Type of visualization
            output_path: Path to save the visualization
            title: Title for the visualization
            **kwargs: Additional parameters for the visualization
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        try:
            if visualization_type == "confidence_distribution":
                # Extract confidence scores and statuses
                confidence_scores = [val_result.confidence_score for val_result in validation_results]
                statuses = [val_result.status.name for val_result in validation_results]
                
                # Create figure and axis
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Define status colors
                status_colors = {
                    "VALID": "#27ae60",
                    "WARNING": "#f39c12",
                    "INVALID": "#e74c3c",
                    "ERROR": "#e74c3c",
                    "PENDING": "#3498db"
                }
                
                # Group by status
                status_groups = {}
                for score, status in zip(confidence_scores, statuses):
                    if status not in status_groups:
                        status_groups[status] = []
                    status_groups[status].append(score)
                
                # Create histogram for each status
                for status, scores in status_groups.items():
                    ax.hist(scores, bins=10, alpha=0.7, label=status, 
                           color=status_colors.get(status, "#7f8c8d"))
                
                # Add labels and title
                ax.set_xlabel('Confidence Score')
                ax.set_ylabel('Count')
                ax.set_title(title)
                ax.legend()
                
                # Save figure if output path is provided
                if output_path:
                    plt.savefig(output_path, dpi=100, bbox_inches="tight")
                    plt.close(fig)
                    logger.info(f"Saved confidence distribution visualization to {output_path}")
                    return output_path
                else:
                    plt.show()
                    return "Displayed visualization"
                
            elif visualization_type == "validation_metrics":
                # Create a scatter plot of validation metrics
                pass
            else:
                logger.error(f"Unsupported Matplotlib visualization type: {visualization_type}")
                return None
        except Exception as e:
            logger.error(f"Error creating Matplotlib visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _prepare_validation_heatmap_data(self, validation_results: List[ValidationResult]) -> pd.DataFrame:
        """
        Prepare data for validation heatmap visualization.
        
        Args:
            validation_results: List of validation results
            
        Returns:
            DataFrame with data for the heatmap
        """
        # Extract model, hardware, and confidence scores
        rows = []
        for val_result in validation_results:
            row = {
                "model_name": val_result.benchmark_result.model_id or "Unknown",
                "model_family": "Unknown",  # Could be determined based on model name conventions
                "hardware_type": val_result.benchmark_result.hardware_id or "Unknown",
                "confidence_score": val_result.confidence_score,
                "validation_level": val_result.validation_level.name,
                "status": val_result.status.name,
                "is_valid": val_result.status == ValidationStatus.VALID
            }
            rows.append(row)
        
        # Create DataFrame
        if PANDAS_AVAILABLE:
            return pd.DataFrame(rows)
        else:
            return rows  # Return as list of dictionaries if pandas not available