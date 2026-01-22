#!/usr/bin/env python3
"""
Dashboard Generator for Distributed Testing Framework

This module implements the dashboard generator for the distributed testing framework.
It creates interactive dashboards for visualizing test results and performance metrics.

Features:
- HTML dashboard generation
- Interactive charts and visualizations
- Performance trend analysis
- Regression detection
- Multi-dimensional data analysis
"""

import os
import sys
import json
import logging
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path
import tempfile

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("dashboard_generator")

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import the visualization engine
try:
    from duckdb_api.distributed_testing.dashboard.visualization import VisualizationEngine
    VISUALIZATION_ENGINE_AVAILABLE = True
except ImportError:
    logger.warning("VisualizationEngine not available. Using basic dashboard generation.")
    VISUALIZATION_ENGINE_AVAILABLE = False

class DashboardGenerator:
    """Dashboard generator for the distributed testing framework."""
    
    def __init__(self, result_aggregator=None, output_dir: str = "./dashboards"):
        """Initialize the dashboard generator.
        
        Args:
            result_aggregator: Result aggregator for accessing result data
            output_dir: Directory to save dashboards
        """
        self.result_aggregator = result_aggregator
        self.output_dir = output_dir
        
        # Create visualization engine if available
        self.visualization_engine = None
        if VISUALIZATION_ENGINE_AVAILABLE:
            self.visualization_engine = VisualizationEngine(
                result_aggregator=result_aggregator,
                output_dir=os.path.join(output_dir, "visualizations")
            )
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Configuration
        self.config = {
            "theme": "light",  # light or dark
            "refresh_interval": 60,  # Auto-refresh interval in seconds (0 to disable)
            "include_performance_charts": True,
            "include_regression_detection": True,
            "include_dimension_analysis": True,
            "include_test_details": True,
            "include_worker_details": True,
            "max_items_per_section": 10,  # Maximum number of items to show in each section
            "embed_images": True,  # Embed images as base64 instead of linking files
            "chart_width": 800,
            "chart_height": 400,
        }
        
        # Define colors based on theme
        self.colors = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "warning": "#d6b117",
            "danger": "#d62728",
            "light": "#f8f9fa",
            "dark": "#343a40",
            "background": "#ffffff" if self.config["theme"] == "light" else "#222222",
            "text": "#333333" if self.config["theme"] == "light" else "#f8f9fa",
            "border": "#dee2e6" if self.config["theme"] == "light" else "#495057",
        }
        
        logger.info("Dashboard generator initialized")
    
    def configure(self, config_updates: Dict[str, Any]) -> None:
        """Update the dashboard generator configuration.
        
        Args:
            config_updates: Dictionary of configuration updates
        """
        self.config.update(config_updates)
        
        # Update colors based on theme
        if "theme" in config_updates:
            self.colors["background"] = "#ffffff" if self.config["theme"] == "light" else "#222222"
            self.colors["text"] = "#333333" if self.config["theme"] == "light" else "#f8f9fa"
            self.colors["border"] = "#dee2e6" if self.config["theme"] == "light" else "#495057"
            
            # Update visualization engine theme if available
            if self.visualization_engine:
                self.visualization_engine.configure({"theme": self.config["theme"]})
        
        logger.info(f"Dashboard generator configuration updated: {config_updates}")
    
    def generate_dashboard(self, data: Optional[Dict[str, Any]] = None, 
                          output_path: Optional[str] = None) -> Optional[str]:
        """Generate a comprehensive dashboard.
        
        Args:
            data: Optional data to use for the dashboard (uses result_aggregator if None)
            output_path: Optional path for the dashboard HTML file
            
        Returns:
            Path to the generated dashboard, or None if generation failed
        """
        # Get data from result aggregator if not provided
        if not data and self.result_aggregator:
            data = {
                "overall_status": self.result_aggregator.get_overall_status(),
                "test_analysis": self.result_aggregator.get_test_analysis(),
                "worker_analysis": self.result_aggregator.worker_analysis,
                "task_type_analysis": self.result_aggregator.task_type_analysis,
                "dimension_analysis": self.result_aggregator.get_dimension_analysis(),
                "regression_results": self.result_aggregator.get_regressions(),
                "historical_performance": getattr(self.result_aggregator, 'historical_performance', {})
            }
        
        # Ensure we have data to work with
        if not data:
            logger.error("No data provided and no result aggregator available")
            return None
        
        # Generate default output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"dashboard_{timestamp}.html")
        
        try:
            # Start building the HTML
            html_content = self._generate_dashboard_html(data)
            
            # Write the HTML to the output file
            with open(output_path, "w") as f:
                f.write(html_content)
            
            logger.info(f"Dashboard generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating dashboard: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _generate_dashboard_html(self, data: Dict[str, Any]) -> str:
        """Generate the HTML content for the dashboard.
        
        Args:
            data: Dashboard data
            
        Returns:
            HTML content as string
        """
        # Define page title
        title = "Distributed Testing Dashboard"
        
        # Start HTML document
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary-color: {self.colors["primary"]};
            --secondary-color: {self.colors["secondary"]};
            --success-color: {self.colors["success"]};
            --warning-color: {self.colors["warning"]};
            --danger-color: {self.colors["danger"]};
            --background-color: {self.colors["background"]};
            --text-color: {self.colors["text"]};
            --border-color: {self.colors["border"]};
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: var(--background-color);
            color: var(--text-color);
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .dashboard-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .dashboard-title {{
            font-size: 24px;
            font-weight: bold;
        }}
        
        .dashboard-timestamp {{
            font-size: 14px;
            opacity: 0.8;
        }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background-color: {self.colors["light"] if self.config["theme"] == "light" else self.colors["dark"]};
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        .card-title {{
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .card-value {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .card-details {{
            font-size: 14px;
        }}
        
        .section {{
            margin-bottom: 30px;
        }}
        
        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .section-title {{
            font-size: 20px;
            font-weight: bold;
        }}
        
        .section-actions {{
            display: flex;
            gap: 10px;
        }}
        
        .tabs {{
            display: flex;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 20px;
        }}
        
        .tab {{
            padding: 10px 20px;
            cursor: pointer;
            border-bottom: 2px solid transparent;
        }}
        
        .tab.active {{
            border-bottom: 2px solid var(--primary-color);
            font-weight: bold;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        
        table th, table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        table th {{
            background-color: {self.colors["light"] if self.config["theme"] == "light" else self.colors["dark"]};
            font-weight: bold;
        }}
        
        table tr:hover {{
            background-color: {self.colors["light"] if self.config["theme"] == "light" else "#3a3a3a"};
        }}
        
        .chart-container {{
            margin-bottom: 20px;
            border-radius: 8px;
            overflow: hidden;
            background-color: {self.colors["light"] if self.config["theme"] == "light" else self.colors["dark"]};
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        
        .chart-title {{
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .progress {{
            height: 8px;
            background-color: {self.colors["light"] if self.config["theme"] == "light" else "#3a3a3a"};
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .progress-bar {{
            height: 100%;
            background-color: var(--primary-color);
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        
        .badge-success {{
            background-color: var(--success-color);
            color: white;
        }}
        
        .badge-warning {{
            background-color: var(--warning-color);
            color: white;
        }}
        
        .badge-danger {{
            background-color: var(--danger-color);
            color: white;
        }}
        
        .badge-primary {{
            background-color: var(--primary-color);
            color: white;
        }}
        
        .badge-secondary {{
            background-color: var(--secondary-color);
            color: white;
        }}
        
        .collapsible {{
            cursor: pointer;
        }}
        
        .collapsible-content {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
            text-align: center;
            font-size: 14px;
            opacity: 0.8;
        }}

        /* Responsive adjustments */
        @media (max-width: 768px) {{
            .summary-cards {{
                grid-template-columns: 1fr;
            }}
            
            .tabs {{
                flex-direction: column;
            }}
            
            .tab {{
                padding: 10px;
            }}
        }}
    </style>
"""

        # Add auto-refresh meta tag if enabled
        if self.config["refresh_interval"] > 0:
            html += f"""    <meta http-equiv="refresh" content="{self.config['refresh_interval']}">
"""

        # Add JavaScript libraries if needed
        html += """    <script>
        // Tab switching functionality
        function switchTab(tabId, groupId) {
            // Hide all tab contents in the group
            var tabContents = document.querySelectorAll(`#${groupId} .tab-content`);
            tabContents.forEach(function(tab) {
                tab.classList.remove('active');
            });
            
            // Deactivate all tabs in the group
            var tabs = document.querySelectorAll(`#${groupId} .tab`);
            tabs.forEach(function(tab) {
                tab.classList.remove('active');
            });
            
            // Activate selected tab and content
            document.getElementById(tabId).classList.add('active');
            document.querySelector(`[onclick="switchTab('${tabId}', '${groupId}')"]`).classList.add('active');
        }
        
        // Collapsible sections
        function toggleCollapsible(id) {
            var content = document.getElementById(id);
            if (content.style.maxHeight) {
                content.style.maxHeight = null;
            } else {
                content.style.maxHeight = content.scrollHeight + "px";
            }
        }
    </script>
</head>
<body>
    <div class="container">
"""

        # Add dashboard header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html += f"""        <div class="dashboard-header">
            <div class="dashboard-title">{title}</div>
            <div class="dashboard-timestamp">Generated: {timestamp}</div>
        </div>
"""

        # Add summary cards
        html += self._generate_summary_cards(data)
        
        # Add performance trends section
        if self.config["include_performance_charts"]:
            html += self._generate_performance_section(data)
        
        # Add regression detection section
        if self.config["include_regression_detection"]:
            html += self._generate_regression_section(data)
        
        # Add dimension analysis section
        if self.config["include_dimension_analysis"]:
            html += self._generate_dimension_section(data)
        
        # Add test details section
        if self.config["include_test_details"]:
            html += self._generate_test_section(data)
        
        # Add worker details section
        if self.config["include_worker_details"]:
            html += self._generate_worker_section(data)
        
        # Add footer
        html += """        <div class="footer">
            Distributed Testing Framework - Advanced Dashboard
        </div>
    </div>
</body>
</html>"""

        return html
    
    def _generate_summary_cards(self, data: Dict[str, Any]) -> str:
        """Generate HTML for summary cards.
        
        Args:
            data: Dashboard data
            
        Returns:
            HTML content as string
        """
        # Extract overall status
        overall_status = data.get("overall_status", {})
        regression_results = data.get("regression_results", {})
        
        # Calculate summary metrics
        test_count = overall_status.get("test_count", 0)
        worker_count = overall_status.get("worker_count", 0)
        task_type_count = overall_status.get("task_type_count", 0)
        total_executions = overall_status.get("total_executions", 0)
        regression_count = overall_status.get("regression_count", 0)
        significant_regression_count = overall_status.get("significant_regression_count", 0)
        
        # Generate HTML
        html = """        <div class="summary-cards">
"""

        # Tests card
        html += f"""            <div class="card">
                <div class="card-title">Tests</div>
                <div class="card-value">{test_count}</div>
                <div class="card-details">
                    <div>{total_executions} total executions</div>
                    <div>{task_type_count} task types</div>
                </div>
            </div>
"""

        # Workers card
        html += f"""            <div class="card">
                <div class="card-title">Workers</div>
                <div class="card-value">{worker_count}</div>
                <div class="card-details">
                    <div>Active execution nodes</div>
                </div>
            </div>
"""

        # Regressions card
        html += f"""            <div class="card">
                <div class="card-title">Regressions</div>
                <div class="card-value">{regression_count}</div>
                <div class="card-details">
                    <div>{significant_regression_count} significant regressions</div>
                </div>
            </div>
"""

        # Performance card - show aggregated metrics if available
        if "aggregated_metrics" in overall_status:
            aggregated_metrics = overall_status["aggregated_metrics"]
            metrics_html = ""
            
            # Find key metrics
            for metric in ["throughput_mean", "latency_mean", "memory_usage_mean", "success_rate_mean"]:
                if metric in aggregated_metrics:
                    # Format the metric value
                    value = aggregated_metrics[metric]
                    formatted_value = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                    
                    # Add to metrics HTML
                    metric_name = metric.replace("_mean", "").replace("_", " ").title()
                    metrics_html += f"<div>{metric_name}: {formatted_value}</div>"
            
            html += f"""            <div class="card">
                <div class="card-title">Performance</div>
                <div class="card-details">
                    {metrics_html}
                </div>
            </div>
"""

        html += """        </div>
"""
        return html
    
    def _generate_performance_section(self, data: Dict[str, Any]) -> str:
        """Generate HTML for performance trends section.
        
        Args:
            data: Dashboard data
            
        Returns:
            HTML content as string
        """
        html = """        <div class="section">
            <div class="section-header">
                <div class="section-title">Performance Trends</div>
            </div>
"""

        # Check if we have a visualization engine and historical performance data
        if self.visualization_engine and "historical_performance" in data and data["historical_performance"]:
            # Create temp directory for visualizations if embedding
            if self.config["embed_images"]:
                temp_dir = tempfile.mkdtemp()
                visualization_path = os.path.join(temp_dir, "performance_trend.png")
            else:
                visualization_path = os.path.join(self.output_dir, "visualizations", "performance_trend.png")
            
            # Get metrics to visualize
            metrics = ["throughput", "latency", "memory_usage"]
            available_metrics = set()
            
            # Check which metrics are available in the data
            for test_id, dates in data["historical_performance"].items():
                for date, date_data in dates.items():
                    for metric in list(metrics):
                        if metric in date_data.get("metrics", {}):
                            available_metrics.add(metric)
            
            if available_metrics:
                # Create visualization data
                visualization_data = {}
                
                # Add time series data for each test and metric
                for test_id, dates in data["historical_performance"].items():
                    for metric in available_metrics:
                        # Create time series for this test and metric
                        time_series = []
                        
                        for date_str, date_data in sorted(dates.items()):
                            if metric in date_data.get("metrics", {}):
                                metric_data = date_data["metrics"][metric]
                                if isinstance(metric_data, dict) and "mean" in metric_data:
                                    date = datetime.strptime(date_str, "%Y-%m-%d")
                                    time_series.append((date, metric_data["mean"]))
                        
                        if time_series:
                            if "time_series" not in visualization_data:
                                visualization_data["time_series"] = {}
                            
                            visualization_data["time_series"][f"{test_id} - {metric}"] = time_series
                
                if visualization_data:
                    # Create visualization
                    visualization_data["metric"] = list(available_metrics)[0]
                    visualization_data["title"] = "Performance Trends Over Time"
                    
                    visualization_path = self.visualization_engine._create_time_series_visualization(
                        visualization_data, visualization_path
                    )
                    
                    if visualization_path:
                        # Add visualization to HTML
                        if self.config["embed_images"]:
                            # Embed image as base64
                            with open(visualization_path, "rb") as img_file:
                                img_data = base64.b64encode(img_file.read()).decode("utf-8")
                            
                            html += f"""            <div class="chart-container">
                <div class="chart-title">Performance Trends Over Time</div>
                <img src="data:image/png;base64,{img_data}" alt="Performance Trends" width="100%">
            </div>
"""
                        else:
                            # Link to image file
                            rel_path = os.path.relpath(visualization_path, os.path.dirname(self.output_dir))
                            html += f"""            <div class="chart-container">
                <div class="chart-title">Performance Trends Over Time</div>
                <img src="{rel_path}" alt="Performance Trends" width="100%">
            </div>
"""
            else:
                html += """            <div class="chart-container">
                <div class="chart-title">Performance Trends</div>
                <p>No performance trend data available.</p>
            </div>
"""
        else:
            html += """            <div class="chart-container">
                <div class="chart-title">Performance Trends</div>
                <p>No performance trend data available or visualization engine not available.</p>
            </div>
"""

        html += """        </div>
"""
        return html
    
    def _generate_regression_section(self, data: Dict[str, Any]) -> str:
        """Generate HTML for regression detection section.
        
        Args:
            data: Dashboard data
            
        Returns:
            HTML content as string
        """
        html = """        <div class="section">
            <div class="section-header">
                <div class="section-title">Regression Detection</div>
            </div>
"""

        # Check if we have regression data
        regression_results = data.get("regression_results", {})
        
        if regression_results:
            # Create regression visualization if visualization engine is available
            if self.visualization_engine:
                # Create temp directory for visualizations if embedding
                if self.config["embed_images"]:
                    temp_dir = tempfile.mkdtemp()
                    visualization_path = os.path.join(temp_dir, "regression_analysis.png")
                else:
                    visualization_path = os.path.join(self.output_dir, "visualizations", "regression_analysis.png")
                
                # Create visualization data
                visualization_data = {"regressions": regression_results}
                
                visualization_path = self.visualization_engine._create_regression_visualization(
                    visualization_data, visualization_path
                )
                
                if visualization_path:
                    # Add visualization to HTML
                    if self.config["embed_images"]:
                        # Embed image as base64
                        with open(visualization_path, "rb") as img_file:
                            img_data = base64.b64encode(img_file.read()).decode("utf-8")
                        
                        html += f"""            <div class="chart-container">
                <div class="chart-title">Regression Analysis</div>
                <img src="data:image/png;base64,{img_data}" alt="Regression Analysis" width="100%">
            </div>
"""
                    else:
                        # Link to image file
                        rel_path = os.path.relpath(visualization_path, os.path.dirname(self.output_dir))
                        html += f"""            <div class="chart-container">
                <div class="chart-title">Regression Analysis</div>
                <img src="{rel_path}" alt="Regression Analysis" width="100%">
            </div>
"""
            
            # Add regression table
            html += """            <table>
                <thead>
                    <tr>
                        <th>Test ID</th>
                        <th>Metric</th>
                        <th>Change</th>
                        <th>Significance</th>
                        <th>Baseline</th>
                        <th>Current</th>
                    </tr>
                </thead>
                <tbody>
"""

            # Add rows for significant regressions first, then non-significant
            rows = []
            
            for test_id, regression_info in regression_results.items():
                for metric, metric_info in regression_info.get("metrics", {}).items():
                    if metric_info.get("is_regression", False):
                        is_significant = metric_info.get("is_significant", False)
                        percent_change = metric_info.get("percent_change", 0)
                        baseline = metric_info.get("baseline_mean", 0)
                        current = metric_info.get("current_mean", 0)
                        
                        # Create row with appropriate styling
                        badge_class = "badge-danger" if is_significant else "badge-warning"
                        significance_text = "Significant" if is_significant else "Not significant"
                        
                        row = {
                            "test_id": test_id,
                            "metric": metric,
                            "percent_change": percent_change,
                            "is_significant": is_significant,
                            "baseline": baseline,
                            "current": current,
                            "html": f"""                    <tr>
                        <td>{test_id}</td>
                        <td>{metric}</td>
                        <td>{percent_change:.2f}%</td>
                        <td><span class="badge {badge_class}">{significance_text}</span></td>
                        <td>{baseline:.2f}</td>
                        <td>{current:.2f}</td>
                    </tr>
"""
                        }
                        
                        rows.append(row)
            
            # Sort rows by significance, then by percent change
            rows.sort(key=lambda x: (-int(x["is_significant"]), abs(x["percent_change"])), reverse=True)
            
            # Add rows to HTML, limited by max_items
            max_items = self.config["max_items_per_section"]
            for i, row in enumerate(rows):
                if i >= max_items:
                    break
                html += row["html"]
            
            html += """                </tbody>
            </table>
"""
        else:
            html += """            <div class="chart-container">
                <div class="chart-title">Regression Analysis</div>
                <p>No regression data available.</p>
            </div>
"""

        html += """        </div>
"""
        return html
    
    def _generate_dimension_section(self, data: Dict[str, Any]) -> str:
        """Generate HTML for dimension analysis section.
        
        Args:
            data: Dashboard data
            
        Returns:
            HTML content as string
        """
        html = """        <div class="section">
            <div class="section-header">
                <div class="section-title">Dimension Analysis</div>
            </div>
            
            <div id="dimension-tabs" class="tabs">
"""

        # Get dimensions
        dimension_analysis = data.get("dimension_analysis", {})
        
        if dimension_analysis:
            # Create tabs for each dimension
            dimensions = list(dimension_analysis.keys())
            
            for i, dimension in enumerate(dimensions):
                active_class = "active" if i == 0 else ""
                html += f"""                <div class="tab {active_class}" onclick="switchTab('dimension-{dimension}', 'dimension-tabs')">{dimension.replace('_', ' ').title()}</div>
"""
            
            html += """            </div>
"""

            # Create tab content for each dimension
            for i, dimension in enumerate(dimensions):
                active_class = "active" if i == 0 else ""
                html += f"""            <div id="dimension-{dimension}" class="tab-content {active_class}">
"""
                
                # Create dimension visualization if visualization engine is available
                if self.visualization_engine:
                    # Create temp directory for visualizations if embedding
                    if self.config["embed_images"]:
                        temp_dir = tempfile.mkdtemp()
                        visualization_path = os.path.join(temp_dir, f"dimension_{dimension}.png")
                    else:
                        visualization_path = os.path.join(self.output_dir, "visualizations", f"dimension_{dimension}.png")
                    
                    # Get a metric to visualize
                    metric = None
                    # Look for throughput, latency, or memory_usage in the first value
                    first_value = next(iter(dimension_analysis[dimension].values()), {})
                    for possible_metric in ["throughput", "latency", "memory_usage"]:
                        if f"{possible_metric}_mean" in first_value:
                            metric = possible_metric
                            break
                    
                    if metric:
                        # Create visualization data
                        values = {}
                        for value, metrics in dimension_analysis[dimension].items():
                            mean_key = f"{metric}_mean"
                            if mean_key in metrics:
                                values[value] = metrics[mean_key]
                        
                        visualization_data = {
                            "dimension": dimension,
                            "metric": metric,
                            "values": values,
                            "title": f"{metric.replace('_', ' ').title()} by {dimension.replace('_', ' ').title()}"
                        }
                        
                        visualization_path = self.visualization_engine._create_dimension_comparison_visualization(
                            visualization_data, visualization_path
                        )
                        
                        if visualization_path:
                            # Add visualization to HTML
                            if self.config["embed_images"]:
                                # Embed image as base64
                                with open(visualization_path, "rb") as img_file:
                                    img_data = base64.b64encode(img_file.read()).decode("utf-8")
                                
                                html += f"""                <div class="chart-container">
                    <div class="chart-title">{metric.replace('_', ' ').title()} by {dimension.replace('_', ' ').title()}</div>
                    <img src="data:image/png;base64,{img_data}" alt="Dimension Analysis" width="100%">
                </div>
"""
                            else:
                                # Link to image file
                                rel_path = os.path.relpath(visualization_path, os.path.dirname(self.output_dir))
                                html += f"""                <div class="chart-container">
                    <div class="chart-title">{metric.replace('_', ' ').title()} by {dimension.replace('_', ' ').title()}</div>
                    <img src="{rel_path}" alt="Dimension Analysis" width="100%">
                </div>
"""
                
                # Add dimension data table
                html += """                <table>
                    <thead>
                        <tr>
                            <th>Value</th>
"""

                # Find metrics to include in table
                metrics = set()
                for value_data in dimension_analysis[dimension].values():
                    for key in value_data.keys():
                        if key.endswith("_mean"):
                            metrics.add(key)
                
                # Add metric columns
                for metric in sorted(metrics):
                    metric_name = metric.replace("_mean", "").replace("_", " ").title()
                    html += f"""                            <th>{metric_name}</th>
"""
                
                html += """                        </tr>
                    </thead>
                    <tbody>
"""

                # Add rows for each value
                for value, value_data in dimension_analysis[dimension].items():
                    html += f"""                        <tr>
                            <td>{value}</td>
"""
                    
                    for metric in sorted(metrics):
                        if metric in value_data:
                            metric_value = value_data[metric]
                            formatted_value = f"{metric_value:.2f}" if isinstance(metric_value, (int, float)) else str(metric_value)
                            html += f"""                            <td>{formatted_value}</td>
"""
                        else:
                            html += """                            <td>-</td>
"""
                    
                    html += """                        </tr>
"""
                
                html += """                    </tbody>
                </table>
            </div>
"""
        else:
            html += """                <div class="tab active">No Data</div>
            </div>
            
            <div id="dimension-no-data" class="tab-content active">
                <p>No dimension analysis data available.</p>
            </div>
"""

        html += """        </div>
"""
        return html
    
    def _generate_test_section(self, data: Dict[str, Any]) -> str:
        """Generate HTML for test details section.
        
        Args:
            data: Dashboard data
            
        Returns:
            HTML content as string
        """
        html = """        <div class="section">
            <div class="section-header">
                <div class="section-title">Test Details</div>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Test ID</th>
                        <th>Executions</th>
                        <th>Success Rate</th>
                        <th>Avg Duration</th>
                        <th>Last Execution</th>
                    </tr>
                </thead>
                <tbody>
"""

        # Add rows for each test
        test_analysis = data.get("test_analysis", {})
        
        if test_analysis:
            # Sort tests by execution count
            sorted_tests = sorted(
                test_analysis.items(),
                key=lambda x: x[1].get("execution_count", 0),
                reverse=True
            )
            
            # Add rows, limited by max_items
            max_items = self.config["max_items_per_section"]
            for i, (test_id, analysis) in enumerate(sorted_tests):
                if i >= max_items:
                    break
                
                # Extract metrics
                execution_count = analysis.get("execution_count", 0)
                success_rate = analysis.get("success_rate", 0) * 100
                avg_duration = analysis.get("average_duration", 0)
                last_execution = analysis.get("last_execution", datetime.now())
                
                # Format last execution
                if isinstance(last_execution, datetime):
                    last_execution_str = last_execution.strftime("%Y-%m-%d %H:%M")
                else:
                    last_execution_str = str(last_execution)
                
                # Determine success rate badge class
                if success_rate >= 90:
                    badge_class = "badge-success"
                elif success_rate >= 70:
                    badge_class = "badge-warning"
                else:
                    badge_class = "badge-danger"
                
                html += f"""                <tr>
                    <td>{test_id}</td>
                    <td>{execution_count}</td>
                    <td><span class="badge {badge_class}">{success_rate:.1f}%</span></td>
                    <td>{avg_duration:.2f}s</td>
                    <td>{last_execution_str}</td>
                </tr>
"""
        else:
            html += """                <tr>
                    <td colspan="5">No test data available.</td>
                </tr>
"""

        html += """            </tbody>
            </table>
        </div>
"""
        return html
    
    def _generate_worker_section(self, data: Dict[str, Any]) -> str:
        """Generate HTML for worker details section.
        
        Args:
            data: Dashboard data
            
        Returns:
            HTML content as string
        """
        html = """        <div class="section">
            <div class="section-header">
                <div class="section-title">Worker Details</div>
            </div>
            
            <table>
                <thead>
                    <tr>
                        <th>Worker ID</th>
                        <th>Executions</th>
                        <th>Success Rate</th>
                        <th>Avg Duration</th>
                        <th>Task Distribution</th>
                    </tr>
                </thead>
                <tbody>
"""

        # Add rows for each worker
        worker_analysis = data.get("worker_analysis", {})
        
        if worker_analysis:
            # Sort workers by execution count
            sorted_workers = sorted(
                worker_analysis.items(),
                key=lambda x: x[1].get("execution_count", 0),
                reverse=True
            )
            
            # Add rows, limited by max_items
            max_items = self.config["max_items_per_section"]
            for i, (worker_id, analysis) in enumerate(sorted_workers):
                if i >= max_items:
                    break
                
                # Extract metrics
                execution_count = analysis.get("execution_count", 0)
                success_rate = analysis.get("success_rate", 0) * 100
                avg_duration = analysis.get("average_duration", 0)
                
                # Get task type distribution
                task_distribution = analysis.get("task_type_distribution", {})
                task_distribution_html = ""
                
                if task_distribution and "type_percentages" in task_distribution:
                    task_percentages = task_distribution["type_percentages"]
                    
                    for task_type, percentage in sorted(task_percentages.items(), key=lambda x: x[1], reverse=True)[:3]:
                        task_distribution_html += f"""
                        <div>{task_type}: {percentage:.1f}%</div>
"""
                else:
                    task_distribution_html = "No data"
                
                # Determine success rate badge class
                if success_rate >= 90:
                    badge_class = "badge-success"
                elif success_rate >= 70:
                    badge_class = "badge-warning"
                else:
                    badge_class = "badge-danger"
                
                html += f"""                <tr>
                    <td>{worker_id}</td>
                    <td>{execution_count}</td>
                    <td><span class="badge {badge_class}">{success_rate:.1f}%</span></td>
                    <td>{avg_duration:.2f}s</td>
                    <td>{task_distribution_html}</td>
                </tr>
"""
        else:
            html += """                <tr>
                    <td colspan="5">No worker data available.</td>
                </tr>
"""

        html += """            </tbody>
            </table>
        </div>
"""
        return html
    
    def generate_report(self, report_type: str, data: Optional[Dict[str, Any]] = None, 
                       output_path: Optional[str] = None) -> Optional[str]:
        """Generate a specific type of report.
        
        Args:
            report_type: Type of report to generate
            data: Optional data to use for the report
            output_path: Optional path for the report
            
        Returns:
            Path to the generated report, or None if generation failed
        """
        # Get data from result aggregator if not provided
        if not data and self.result_aggregator:
            data = {
                "overall_status": self.result_aggregator.get_overall_status(),
                "test_analysis": self.result_aggregator.get_test_analysis(),
                "worker_analysis": self.result_aggregator.worker_analysis,
                "task_type_analysis": self.result_aggregator.task_type_analysis,
                "dimension_analysis": self.result_aggregator.get_dimension_analysis(),
                "regression_results": self.result_aggregator.get_regressions(),
                "historical_performance": getattr(self.result_aggregator, 'historical_performance', {})
            }
        
        # Ensure we have data to work with
        if not data:
            logger.error("No data provided and no result aggregator available")
            return None
        
        # Generate default output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"{report_type}_report_{timestamp}.html")
        
        # Generate the requested report
        if report_type == "regression":
            if self.visualization_engine:
                return self.visualization_engine.create_regression_report(data, output_path)
            else:
                logger.error("Visualization engine not available for regression report")
                return None
        
        elif report_type == "performance":
            # Generate performance report
            if self.visualization_engine:
                return self.visualization_engine.create_performance_dashboard(data, output_path)
            else:
                logger.error("Visualization engine not available for performance report")
                return None
        
        else:
            logger.error(f"Unknown report type: {report_type}")
            return None