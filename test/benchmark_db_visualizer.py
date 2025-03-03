#!/usr/bin/env python3
"""
Benchmark Database Visualizer for the IPFS Accelerate Python Framework.

This module provides visualization and reporting tools for benchmark data stored
in the DuckDB database. It generates interactive HTML reports, comparative plots,
and trend analysis visualizations.

Usage:
    python benchmark_db_visualizer.py --report performance --format html --output report.html
    python benchmark_db_visualizer.py --model bert-base-uncased --compare-hardware --output bert_comparison.png
    python benchmark_db_visualizer.py --compare-models --hardware cuda --metric throughput --output model_comparison.png
"""

import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

try:
    import duckdb
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
except ImportError:
    print("Error: Required visualization packages not installed. Please install with:")
    print("pip install duckdb pandas numpy matplotlib seaborn")
    sys.exit(1)

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    HAS_JINJA = True
except ImportError:
    HAS_JINJA = False
    print("Warning: Jinja2 not installed. HTML reports will be in basic format.")
    print("Install with: pip install jinja2")

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import the database models
try:
    import benchmark_db_models
    from benchmark_db_models import BenchmarkDB
    HAS_DB_MODELS = True
except ImportError:
    HAS_DB_MODELS = False
    logger.warning("Database models not found. Some features will be limited.")

class BenchmarkDBVisualizer:
    """
    Visualization and reporting tools for benchmark data stored in the database.
    """
    
    def __init__(self, db_path: str = "./benchmark_db.duckdb", debug: bool = False):
        """
        Initialize the visualizer.
        
        Args:
            db_path: Path to the DuckDB database
            debug: Enable debug logging
        """
        self.db_path = db_path
        
        # Set up logging
        if debug:
            logger.setLevel(logging.DEBUG)
            
        # Connect to database
        self.db_conn = None
        try:
            if HAS_DB_MODELS:
                self.db_conn = BenchmarkDB(db_path=db_path, read_only=True)
                logger.info(f"Connected to database using models: {db_path}")
            else:
                self.db_conn = duckdb.connect(db_path, read_only=True)
                logger.info(f"Connected to database using DuckDB: {db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            sys.exit(1)
            
        # Set up Jinja2 environment for templates if available
        self.jinja_env = None
        if HAS_JINJA:
            template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
            if not os.path.exists(template_dir):
                template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_templates")
                if not os.path.exists(template_dir):
                    os.makedirs(template_dir)
                    
            self.jinja_env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=select_autoescape(['html', 'xml'])
            )
            
    def __del__(self):
        """Clean up database connection on exit"""
        if self.db_conn:
            self.db_conn.close()
            
    def generate_performance_report(self, format: str = "html", output: str = None) -> Optional[str]:
        """
        Generate a comprehensive performance report.
        
        Args:
            format: Output format ('html', 'md', 'json')
            output: Output file path (or None for stdout)
            
        Returns:
            Generated report content (if output is None) or None
        """
        logger.info("Generating performance report")
        
        # Query for performance data
        if HAS_DB_MODELS:
            # Use ORM models
            performance_data = self._query_performance_data_orm()
        else:
            # Use direct SQL
            performance_data = self._query_performance_data_sql()
            
        if not performance_data or not performance_data.get('models'):
            logger.error("No performance data found in database")
            return None
            
        # Generate report in requested format
        if format == "html":
            return self._generate_html_report(performance_data, output)
        elif format == "md":
            return self._generate_markdown_report(performance_data, output)
        elif format == "json":
            return self._generate_json_report(performance_data, output)
        else:
            logger.error(f"Unsupported format: {format}")
            return None
            
    def _query_performance_data_orm(self) -> Dict[str, Any]:
        """
        Query performance data using ORM models.
        
        Returns:
            Dictionary of performance data
        """
        try:
            # Get latest performance results
            latest_results = self.db_conn.get_all_performance_results(limit=1000)
            
            # Group by model and hardware
            performance_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'models': {}
            }
            
            for result in latest_results:
                # Get model and hardware information
                model = self.db_conn.get_model_by_model_id(result.model_id)
                hardware = self.db_conn.get_hardware_platform_by_hardware_id(result.hardware_id)
                
                if not model or not hardware:
                    continue
                    
                model_name = model.model_name
                hardware_type = hardware.hardware_type
                
                # Initialize model entry if needed
                if model_name not in performance_data['models']:
                    performance_data['models'][model_name] = {
                        'model_family': model.model_family,
                        'hardware_results': {}
                    }
                    
                # Initialize hardware entry if needed
                if hardware_type not in performance_data['models'][model_name]['hardware_results']:
                    performance_data['models'][model_name]['hardware_results'][hardware_type] = []
                    
                # Add performance result
                performance_data['models'][model_name]['hardware_results'][hardware_type].append({
                    'batch_size': result.batch_size,
                    'precision': result.precision,
                    'latency_ms': result.average_latency_ms,
                    'throughput': result.throughput_items_per_second,
                    'memory_mb': result.memory_peak_mb,
                    'test_case': result.test_case,
                    'timestamp': result.created_at.isoformat() if hasattr(result.created_at, 'isoformat') else str(result.created_at)
                })
                
            return performance_data
            
        except Exception as e:
            logger.error(f"Error querying performance data with ORM: {e}")
            return {}
            
    def _query_performance_data_sql(self) -> Dict[str, Any]:
        """
        Query performance data using direct SQL.
        
        Returns:
            Dictionary of performance data
        """
        try:
            # Query for latest performance results
            query = """
            SELECT 
                pr.result_id, pr.batch_size, pr.precision, pr.average_latency_ms, 
                pr.throughput_items_per_second, pr.memory_peak_mb, pr.test_case,
                pr.created_at, m.model_id, m.model_name, m.model_family,
                hp.hardware_id, hp.hardware_type
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON pr.hardware_id = hp.hardware_id
            ORDER BY 
                pr.created_at DESC
            LIMIT 1000
            """
            
            results = self.db_conn.execute(query).fetchdf()
            
            # Group by model and hardware
            performance_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'models': {}
            }
            
            for _, row in results.iterrows():
                model_name = row['model_name']
                hardware_type = row['hardware_type']
                
                # Initialize model entry if needed
                if model_name not in performance_data['models']:
                    performance_data['models'][model_name] = {
                        'model_family': row['model_family'],
                        'hardware_results': {}
                    }
                    
                # Initialize hardware entry if needed
                if hardware_type not in performance_data['models'][model_name]['hardware_results']:
                    performance_data['models'][model_name]['hardware_results'][hardware_type] = []
                    
                # Add performance result
                performance_data['models'][model_name]['hardware_results'][hardware_type].append({
                    'batch_size': row['batch_size'],
                    'precision': row['precision'],
                    'latency_ms': row['average_latency_ms'],
                    'throughput': row['throughput_items_per_second'],
                    'memory_mb': row['memory_peak_mb'],
                    'test_case': row['test_case'],
                    'timestamp': row['created_at'].isoformat() if hasattr(row['created_at'], 'isoformat') else str(row['created_at'])
                })
                
            return performance_data
            
        except Exception as e:
            logger.error(f"Error querying performance data with SQL: {e}")
            return {}
            
    def _generate_html_report(self, data: Dict[str, Any], output: Optional[str]) -> Optional[str]:
        """
        Generate an HTML report from performance data.
        
        Args:
            data: Performance data dictionary
            output: Output file path (or None for return)
            
        Returns:
            HTML content (if output is None) or None
        """
        if not data or not data.get('models'):
            logger.error("No data available for HTML report")
            return None
            
        # Use Jinja2 if available, otherwise basic HTML
        if self.jinja_env:
            try:
                # Check if performance template exists
                try:
                    template = self.jinja_env.get_template('performance_report.html')
                except:
                    # Create a basic template if not found
                    template_path = os.path.join(
                        os.path.dirname(self.jinja_env.loader.searchpath[0]), 
                        "templates", 
                        "performance_report.html"
                    )
                    os.makedirs(os.path.dirname(template_path), exist_ok=True)
                    
                    with open(template_path, 'w') as f:
                        f.write(self._get_default_html_template())
                        
                    template = self.jinja_env.get_template('performance_report.html')
                
                # Prepare data for template
                template_data = {
                    'title': 'Performance Benchmark Report',
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'performance_data': data,
                    'models': sorted(data['models'].keys())
                }
                
                # Render template
                html_content = template.render(**template_data)
                
                # Save or return
                if output:
                    with open(output, 'w') as f:
                        f.write(html_content)
                    logger.info(f"HTML report saved to {output}")
                    return None
                else:
                    return html_content
                    
            except Exception as e:
                logger.error(f"Error generating HTML report with Jinja2: {e}")
                return self._generate_basic_html_report(data, output)
        else:
            return self._generate_basic_html_report(data, output)
            
    def _generate_basic_html_report(self, data: Dict[str, Any], output: Optional[str]) -> Optional[str]:
        """
        Generate a basic HTML report without using Jinja2.
        
        Args:
            data: Performance data dictionary
            output: Output file path (or None for return)
            
        Returns:
            HTML content (if output is None) or None
        """
        # Start building HTML
        html = ["<!DOCTYPE html>", "<html>", "<head>",
                "<title>Performance Benchmark Report</title>",
                "<style>",
                "body { font-family: Arial, sans-serif; margin: 20px; }",
                "table { border-collapse: collapse; width: 100%; }",
                "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
                "th { background-color: #f2f2f2; }",
                "tr:nth-child(even) { background-color: #f9f9f9; }",
                "h1, h2, h3 { color: #333; }",
                "</style>",
                "</head>",
                "<body>",
                f"<h1>Performance Benchmark Report</h1>",
                f"<p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"]
        
        # Add model results
        for model_name, model_data in sorted(data['models'].items()):
            html.append(f"<h2>Model: {model_name}</h2>")
            html.append(f"<p>Family: {model_data.get('model_family', 'Unknown')}</p>")
            
            # Hardware results table
            html.append("<h3>Hardware Performance</h3>")
            html.append("<table>")
            html.append("<tr><th>Hardware</th><th>Batch Size</th><th>Precision</th><th>Latency (ms)</th><th>Throughput</th><th>Memory (MB)</th></tr>")
            
            for hw_type, hw_results in sorted(model_data['hardware_results'].items()):
                # Sort by batch size
                hw_results = sorted(hw_results, key=lambda x: x.get('batch_size', 0))
                
                for result in hw_results:
                    html.append("<tr>")
                    html.append(f"<td>{hw_type}</td>")
                    html.append(f"<td>{result.get('batch_size', 'N/A')}</td>")
                    html.append(f"<td>{result.get('precision', 'N/A')}</td>")
                    html.append(f"<td>{result.get('latency_ms', 'N/A'):.2f}</td>")
                    html.append(f"<td>{result.get('throughput', 'N/A'):.2f}</td>")
                    html.append(f"<td>{result.get('memory_mb', 'N/A'):.2f}</td>")
                    html.append("</tr>")
                    
            html.append("</table>")
            
        # Close HTML
        html.append("</body>")
        html.append("</html>")
        
        # Join and output
        html_content = "\n".join(html)
        
        if output:
            with open(output, 'w') as f:
                f.write(html_content)
            logger.info(f"Basic HTML report saved to {output}")
            return None
        else:
            return html_content
            
    def _get_default_html_template(self) -> str:
        """Get a default Jinja2 HTML template for performance reports"""
        return """<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        h1, h2, h3 { color: #333; }
        .section { margin-bottom: 30px; }
        .card { border: 1px solid #ddd; border-radius: 4px; padding: 15px; margin-bottom: 20px; }
        .card-header { background-color: #f2f2f2; padding: 10px; margin: -15px -15px 15px -15px; border-radius: 4px 4px 0 0; }
        .tabs { display: flex; border-bottom: 1px solid #ddd; margin-bottom: 15px; }
        .tab { padding: 10px 15px; cursor: pointer; }
        .tab.active { border: 1px solid #ddd; border-bottom: none; border-radius: 4px 4px 0 0; background-color: #fff; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
    <script>
        function showTab(tabId, linkElement) {
            // Hide all tab contents
            var tabContents = document.getElementsByClassName('tab-content');
            for (var i = 0; i < tabContents.length; i++) {
                tabContents[i].classList.remove('active');
            }
            
            // Deactivate all tab links
            var tabLinks = document.getElementsByClassName('tab');
            for (var i = 0; i < tabLinks.length; i++) {
                tabLinks[i].classList.remove('active');
            }
            
            // Show the selected tab content and activate the link
            document.getElementById(tabId).classList.add('active');
            linkElement.classList.add('active');
        }
    </script>
</head>
<body>
    <h1>{{ title }}</h1>
    <p>Generated: {{ timestamp }}</p>
    
    <div class="section">
        <h2>Performance Summary</h2>
        <p>This report contains performance data for {{ models|length }} models across different hardware platforms.</p>
        
        <div class="tabs">
            <div class="tab active" onclick="showTab('tab-models', this)">By Model</div>
            <div class="tab" onclick="showTab('tab-hardware', this)">By Hardware</div>
        </div>
        
        <div id="tab-models" class="tab-content active">
            {% for model_name in models %}
            {% set model_data = performance_data.models[model_name] %}
            <div class="card">
                <div class="card-header">
                    <h3>{{ model_name }}</h3>
                    <p>Family: {{ model_data.model_family }}</p>
                </div>
                
                <table>
                    <tr>
                        <th>Hardware</th>
                        <th>Batch Size</th>
                        <th>Precision</th>
                        <th>Latency (ms)</th>
                        <th>Throughput</th>
                        <th>Memory (MB)</th>
                    </tr>
                    {% for hw_type, hw_results in model_data.hardware_results.items() %}
                        {% for result in hw_results %}
                        <tr>
                            <td>{{ hw_type }}</td>
                            <td>{{ result.batch_size }}</td>
                            <td>{{ result.precision }}</td>
                            <td>{{ "%.2f"|format(result.latency_ms) }}</td>
                            <td>{{ "%.2f"|format(result.throughput) }}</td>
                            <td>{{ "%.2f"|format(result.memory_mb) if result.memory_mb else 'N/A' }}</td>
                        </tr>
                        {% endfor %}
                    {% endfor %}
                </table>
            </div>
            {% endfor %}
        </div>
        
        <div id="tab-hardware" class="tab-content">
            {% set hardware_types = [] %}
            {% for model_name, model_data in performance_data.models.items() %}
                {% for hw_type in model_data.hardware_results.keys() %}
                    {% if hw_type not in hardware_types %}
                        {% do hardware_types.append(hw_type) %}
                    {% endif %}
                {% endfor %}
            {% endfor %}
            
            {% for hw_type in hardware_types|sort %}
            <div class="card">
                <div class="card-header">
                    <h3>{{ hw_type }}</h3>
                </div>
                
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Family</th>
                        <th>Batch Size</th>
                        <th>Latency (ms)</th>
                        <th>Throughput</th>
                    </tr>
                    {% for model_name, model_data in performance_data.models.items() %}
                        {% if hw_type in model_data.hardware_results %}
                            {% for result in model_data.hardware_results[hw_type] %}
                            <tr>
                                <td>{{ model_name }}</td>
                                <td>{{ model_data.model_family }}</td>
                                <td>{{ result.batch_size }}</td>
                                <td>{{ "%.2f"|format(result.latency_ms) }}</td>
                                <td>{{ "%.2f"|format(result.throughput) }}</td>
                            </tr>
                            {% endfor %}
                        {% endif %}
                    {% endfor %}
                </table>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""
            
    def _generate_markdown_report(self, data: Dict[str, Any], output: Optional[str]) -> Optional[str]:
        """
        Generate a markdown report from performance data.
        
        Args:
            data: Performance data dictionary
            output: Output file path (or None for return)
            
        Returns:
            Markdown content (if output is None) or None
        """
        if not data or not data.get('models'):
            logger.error("No data available for markdown report")
            return None
            
        # Build markdown content
        md = []
        md.append("# Performance Benchmark Report")
        md.append("")
        md.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append("")
        
        # Group by model family
        models_by_family = {}
        for model_name, model_data in data['models'].items():
            family = model_data.get('model_family', 'Unknown')
            if family not in models_by_family:
                models_by_family[family] = []
            models_by_family[family].append((model_name, model_data))
        
        # Process by family
        for family, models in sorted(models_by_family.items()):
            md.append(f"## {family.title()} Models")
            md.append("")
            
            for model_name, model_data in sorted(models):
                md.append(f"### {model_name}")
                md.append("")
                
                # Hardware results table
                md.append("#### Hardware Performance")
                md.append("")
                md.append("| Hardware | Batch Size | Precision | Latency (ms) | Throughput | Memory (MB) |")
                md.append("|----------|------------|-----------|--------------|------------|-------------|")
                
                for hw_type, hw_results in sorted(model_data['hardware_results'].items()):
                    # Sort by batch size
                    hw_results = sorted(hw_results, key=lambda x: x.get('batch_size', 0))
                    
                    for result in hw_results:
                        md.append(f"| {hw_type} | {result.get('batch_size', 'N/A')} | {result.get('precision', 'N/A')} | {result.get('latency_ms', 0):.2f} | {result.get('throughput', 0):.2f} | {result.get('memory_mb', 0):.2f} |")
                
                md.append("")
        
        # Join and output
        md_content = "\n".join(md)
        
        if output:
            with open(output, 'w') as f:
                f.write(md_content)
            logger.info(f"Markdown report saved to {output}")
            return None
        else:
            return md_content
            
    def _generate_json_report(self, data: Dict[str, Any], output: Optional[str]) -> Optional[str]:
        """
        Generate a JSON report from performance data.
        
        Args:
            data: Performance data dictionary
            output: Output file path (or None for return)
            
        Returns:
            JSON content (if output is None) or None
        """
        # Add report metadata
        report_data = {
            "report_type": "performance",
            "generated_at": datetime.datetime.now().isoformat(),
            "data": data
        }
        
        # Convert to JSON
        json_content = json.dumps(report_data, indent=2)
        
        if output:
            with open(output, 'w') as f:
                f.write(json_content)
            logger.info(f"JSON report saved to {output}")
            return None
        else:
            return json_content
            
    def compare_hardware_for_model(self, model_name: str, metric: str = "throughput", 
                                output: Optional[str] = None) -> Optional[str]:
        """
        Compare different hardware platforms for a specific model.
        
        Args:
            model_name: Name of the model to compare
            metric: Performance metric to compare ('throughput', 'latency', 'memory')
            output: Output plot file path
            
        Returns:
            Path to the generated plot file
        """
        logger.info(f"Comparing hardware platforms for model: {model_name}")
        
        # Query for model data
        if HAS_DB_MODELS:
            # Use ORM models
            model_data = self._query_model_performance_orm(model_name)
        else:
            # Use direct SQL
            model_data = self._query_model_performance_sql(model_name)
            
        if not model_data:
            logger.error(f"No performance data found for model: {model_name}")
            return None
        
        # Create the comparison plot
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        hardware_types = []
        metric_values = []
        error_values = []
        
        # Map metric name to its key in the data
        metric_key = {
            "throughput": "throughput_items_per_second", 
            "latency": "average_latency_ms",
            "memory": "memory_peak_mb"
        }.get(metric.lower(), "throughput_items_per_second")
        
        # Group by hardware type
        for hw_type, hw_results in model_data.items():
            if not hw_results:
                continue
                
            hardware_types.append(hw_type)
            
            # Extract metric values
            values = [result.get(metric_key, 0) for result in hw_results]
            
            # Calculate mean and standard deviation
            if values:
                metric_values.append(np.mean(values))
                error_values.append(np.std(values) if len(values) > 1 else 0)
            else:
                metric_values.append(0)
                error_values.append(0)
        
        # Check if we have data to plot
        if not hardware_types:
            logger.error(f"No hardware platforms found for model: {model_name}")
            return None
        
        # Create bar plot
        bars = plt.bar(hardware_types, metric_values, yerr=error_values, capsize=10)
        
        # Add labels and title
        plt.title(f"{metric.title()} Comparison for {model_name}")
        plt.xlabel("Hardware Platform")
        
        if metric.lower() == "throughput":
            plt.ylabel("Throughput (items/second)")
        elif metric.lower() == "latency":
            plt.ylabel("Latency (ms)")
        elif metric.lower() == "memory":
            plt.ylabel("Memory Usage (MB)")
        else:
            plt.ylabel(metric.title())
        
        # Add value labels on top of bars
        for i, v in enumerate(metric_values):
            plt.text(i, v + (max(metric_values) * 0.02), f'{v:.2f}', ha='center')
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot if output path provided
        if output:
            plt.savefig(output)
            logger.info(f"Comparison plot saved to {output}")
            plt.close()
            return output
        else:
            # Show the plot
            plt.show()
            plt.close()
            return None
            
    def _query_model_performance_orm(self, model_name: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query performance data for a specific model using ORM models.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of performance data grouped by hardware type
        """
        try:
            # Get model by name
            models = self.db_conn.get_models_by_name(model_name)
            if not models:
                return {}
                
            model_id = models[0].model_id
            
            # Get performance results for this model
            query = f"""
            SELECT pr.*, hp.hardware_type, hp.device_name
            FROM performance_results pr
            JOIN hardware_platforms hp ON pr.hardware_id = hp.hardware_id
            WHERE pr.model_id = {model_id}
            """
            results = self.db_conn.conn.execute(query).fetchdf()
            
            # Group by hardware type
            performance_data = {}
            
            for _, row in results.iterrows():
                hw_type = row['hardware_type']
                
                if hw_type not in performance_data:
                    performance_data[hw_type] = []
                
                performance_data[hw_type].append({
                    'batch_size': row['batch_size'],
                    'precision': row['precision'],
                    'average_latency_ms': row['average_latency_ms'],
                    'throughput_items_per_second': row['throughput_items_per_second'],
                    'memory_peak_mb': row['memory_peak_mb'],
                    'test_case': row['test_case']
                })
                
            return performance_data
            
        except Exception as e:
            logger.error(f"Error querying model performance with ORM: {e}")
            return {}
            
    def _query_model_performance_sql(self, model_name: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query performance data for a specific model using direct SQL.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of performance data grouped by hardware type
        """
        try:
            # Query for model performance data
            query = f"""
            SELECT 
                pr.batch_size, pr.precision, pr.average_latency_ms, 
                pr.throughput_items_per_second, pr.memory_peak_mb, pr.test_case,
                hp.hardware_type, hp.device_name
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON pr.hardware_id = hp.hardware_id
            WHERE 
                m.model_name LIKE '%{model_name}%'
            """
            
            results = self.db_conn.execute(query).fetchdf()
            
            # Group by hardware type
            performance_data = {}
            
            for _, row in results.iterrows():
                hw_type = row['hardware_type']
                
                if hw_type not in performance_data:
                    performance_data[hw_type] = []
                
                performance_data[hw_type].append({
                    'batch_size': row['batch_size'],
                    'precision': row['precision'],
                    'average_latency_ms': row['average_latency_ms'],
                    'throughput_items_per_second': row['throughput_items_per_second'],
                    'memory_peak_mb': row['memory_peak_mb'],
                    'test_case': row['test_case']
                })
                
            return performance_data
            
        except Exception as e:
            logger.error(f"Error querying model performance with SQL: {e}")
            return {}
            
    def compare_models_on_hardware(self, hardware_type: str, metric: str = "throughput", 
                                model_family: Optional[str] = None,
                                output: Optional[str] = None) -> Optional[str]:
        """
        Compare different models on a specific hardware platform.
        
        Args:
            hardware_type: Type of hardware to compare on
            metric: Performance metric to compare ('throughput', 'latency', 'memory')
            model_family: Optional filter for model family
            output: Output plot file path
            
        Returns:
            Path to the generated plot file
        """
        logger.info(f"Comparing models on hardware: {hardware_type}")
        
        # Query for hardware data
        if HAS_DB_MODELS:
            # Use ORM models
            hardware_data = self._query_hardware_performance_orm(hardware_type, model_family)
        else:
            # Use direct SQL
            hardware_data = self._query_hardware_performance_sql(hardware_type, model_family)
            
        if not hardware_data:
            logger.error(f"No performance data found for hardware: {hardware_type}")
            return None
        
        # Create the comparison plot
        plt.figure(figsize=(14, 8))
        
        # Prepare data for plotting
        model_names = []
        metric_values = []
        error_values = []
        bar_colors = []
        
        # Map metric name to its key in the data
        metric_key = {
            "throughput": "throughput_items_per_second", 
            "latency": "average_latency_ms",
            "memory": "memory_peak_mb"
        }.get(metric.lower(), "throughput_items_per_second")
        
        # Group models by family for coloring
        family_colors = {}
        unique_families = set()
        
        for model_name, model_results in hardware_data.items():
            if not model_results:
                continue
                
            # Get model family
            family = model_results[0].get('model_family', 'unknown')
            unique_families.add(family)
        
        # Generate colors for each family
        colormap = plt.cm.get_cmap('tab10', len(unique_families))
        for i, family in enumerate(sorted(unique_families)):
            family_colors[family] = colormap(i)
        
        # Process each model
        for model_name, model_results in sorted(hardware_data.items()):
            if not model_results:
                continue
                
            model_names.append(model_name)
            
            # Get model family for coloring
            family = model_results[0].get('model_family', 'unknown')
            bar_colors.append(family_colors.get(family, 'blue'))
            
            # Extract metric values
            values = [result.get(metric_key, 0) for result in model_results]
            
            # Calculate mean and standard deviation
            if values:
                metric_values.append(np.mean(values))
                error_values.append(np.std(values) if len(values) > 1 else 0)
            else:
                metric_values.append(0)
                error_values.append(0)
        
        # Check if we have data to plot
        if not model_names:
            logger.error(f"No models found for hardware: {hardware_type}")
            return None
        
        # Create bar plot
        bars = plt.bar(model_names, metric_values, yerr=error_values, capsize=10, color=bar_colors)
        
        # Add labels and title
        plt.title(f"{metric.title()} Comparison on {hardware_type}")
        plt.xlabel("Model")
        plt.xticks(rotation=45, ha='right')
        
        if metric.lower() == "throughput":
            plt.ylabel("Throughput (items/second)")
        elif metric.lower() == "latency":
            plt.ylabel("Latency (ms)")
        elif metric.lower() == "memory":
            plt.ylabel("Memory Usage (MB)")
        else:
            plt.ylabel(metric.title())
        
        # Add legend for model families
        legend_handles = []
        for family, color in sorted(family_colors.items()):
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=family))
        
        plt.legend(handles=legend_handles, title="Model Family")
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot if output path provided
        if output:
            plt.savefig(output)
            logger.info(f"Comparison plot saved to {output}")
            plt.close()
            return output
        else:
            # Show the plot
            plt.show()
            plt.close()
            return None
            
    def _query_hardware_performance_orm(self, hardware_type: str, 
                                       model_family: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query performance data for a specific hardware platform using ORM models.
        
        Args:
            hardware_type: Type of hardware
            model_family: Optional filter for model family
            
        Returns:
            Dictionary of performance data grouped by model name
        """
        try:
            # Get hardware platforms of this type
            query = f"""
            SELECT hardware_id, device_name
            FROM hardware_platforms
            WHERE hardware_type = '{hardware_type}'
            """
            hw_results = self.db_conn.conn.execute(query).fetchdf()
            
            if hw_results.empty:
                return {}
                
            # Extract hardware IDs
            hardware_ids = hw_results['hardware_id'].tolist()
            hardware_ids_str = ', '.join(map(str, hardware_ids))
            
            # Get performance results for these hardware platforms
            model_filter = f"AND m.model_family = '{model_family}'" if model_family else ""
            
            query = f"""
            SELECT 
                pr.*, m.model_name, m.model_family
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            WHERE 
                pr.hardware_id IN ({hardware_ids_str})
                {model_filter}
            """
            
            results = self.db_conn.conn.execute(query).fetchdf()
            
            # Group by model name
            performance_data = {}
            
            for _, row in results.iterrows():
                model_name = row['model_name']
                
                if model_name not in performance_data:
                    performance_data[model_name] = []
                
                performance_data[model_name].append({
                    'batch_size': row['batch_size'],
                    'precision': row['precision'],
                    'average_latency_ms': row['average_latency_ms'],
                    'throughput_items_per_second': row['throughput_items_per_second'],
                    'memory_peak_mb': row['memory_peak_mb'],
                    'test_case': row['test_case'],
                    'model_family': row['model_family']
                })
                
            return performance_data
            
        except Exception as e:
            logger.error(f"Error querying hardware performance with ORM: {e}")
            return {}
            
    def _query_hardware_performance_sql(self, hardware_type: str, 
                                       model_family: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query performance data for a specific hardware platform using direct SQL.
        
        Args:
            hardware_type: Type of hardware
            model_family: Optional filter for model family
            
        Returns:
            Dictionary of performance data grouped by model name
        """
        try:
            # Query for hardware performance data
            model_filter = f"AND m.model_family = '{model_family}'" if model_family else ""
            
            query = f"""
            SELECT 
                pr.batch_size, pr.precision, pr.average_latency_ms, 
                pr.throughput_items_per_second, pr.memory_peak_mb, pr.test_case,
                m.model_name, m.model_family
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON pr.hardware_id = hp.hardware_id
            WHERE 
                hp.hardware_type = '{hardware_type}'
                {model_filter}
            """
            
            results = self.db_conn.execute(query).fetchdf()
            
            # Group by model name
            performance_data = {}
            
            for _, row in results.iterrows():
                model_name = row['model_name']
                
                if model_name not in performance_data:
                    performance_data[model_name] = []
                
                performance_data[model_name].append({
                    'batch_size': row['batch_size'],
                    'precision': row['precision'],
                    'average_latency_ms': row['average_latency_ms'],
                    'throughput_items_per_second': row['throughput_items_per_second'],
                    'memory_peak_mb': row['memory_peak_mb'],
                    'test_case': row['test_case'],
                    'model_family': row['model_family']
                })
                
            return performance_data
            
        except Exception as e:
            logger.error(f"Error querying hardware performance with SQL: {e}")
            return {}
            
    def plot_performance_trend(self, model_name: str, hardware_type: str, 
                             metric: str = "throughput", 
                             output: Optional[str] = None) -> Optional[str]:
        """
        Plot performance trend over time for a specific model and hardware.
        
        Args:
            model_name: Name of the model
            hardware_type: Type of hardware
            metric: Performance metric to plot ('throughput', 'latency', 'memory')
            output: Output plot file path
            
        Returns:
            Path to the generated plot file
        """
        logger.info(f"Plotting performance trend for {model_name} on {hardware_type}")
        
        # Query for trend data
        trend_data = self._query_performance_trend(model_name, hardware_type, metric)
        
        if not trend_data:
            logger.error(f"No trend data found for {model_name} on {hardware_type}")
            return None
        
        # Create the trend plot
        plt.figure(figsize=(12, 6))
        
        # Sort by timestamp
        trend_data.sort(key=lambda x: x['timestamp'])
        
        # Extract data for plotting
        timestamps = [point['timestamp'] for point in trend_data]
        values = [point['value'] for point in trend_data]
        
        # Plot the trend line
        plt.plot(timestamps, values, marker='o', linestyle='-', linewidth=2)
        
        # Add labels and title
        metric_label = {
            "throughput": "Throughput (items/second)",
            "latency": "Latency (ms)",
            "memory": "Memory Usage (MB)"
        }.get(metric.lower(), metric.title())
        
        plt.title(f"{metric_label} Trend for {model_name} on {hardware_type}")
        plt.xlabel("Date")
        plt.ylabel(metric_label)
        
        # Format the x-axis for dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot if output path provided
        if output:
            plt.savefig(output)
            logger.info(f"Trend plot saved to {output}")
            plt.close()
            return output
        else:
            # Show the plot
            plt.show()
            plt.close()
            return None
            
    def _query_performance_trend(self, model_name: str, hardware_type: str, 
                                metric: str = "throughput") -> List[Dict[str, Any]]:
        """
        Query performance trend data for a specific model and hardware.
        
        Args:
            model_name: Name of the model
            hardware_type: Type of hardware
            metric: Performance metric to query
            
        Returns:
            List of trend data points (timestamp and value)
        """
        try:
            # Map metric name to its column in the database
            metric_column = {
                "throughput": "throughput_items_per_second", 
                "latency": "average_latency_ms",
                "memory": "memory_peak_mb"
            }.get(metric.lower(), "throughput_items_per_second")
            
            # Query for trend data
            query = f"""
            SELECT 
                pr.{metric_column} as value, pr.created_at as timestamp
            FROM 
                performance_results pr
            JOIN 
                models m ON pr.model_id = m.model_id
            JOIN 
                hardware_platforms hp ON pr.hardware_id = hp.hardware_id
            WHERE 
                m.model_name LIKE '%{model_name}%'
                AND hp.hardware_type = '{hardware_type}'
            ORDER BY 
                pr.created_at
            """
            
            results = self.db_conn.execute(query).fetchdf()
            
            if results.empty:
                return []
                
            # Convert to list of dictionaries
            trend_data = []
            for _, row in results.iterrows():
                trend_data.append({
                    'timestamp': row['timestamp'],
                    'value': row['value']
                })
                
            return trend_data
            
        except Exception as e:
            logger.error(f"Error querying performance trend: {e}")
            return []

def main():
    """Command-line interface for the benchmark database visualizer."""
    parser = argparse.ArgumentParser(description="Benchmark Database Visualizer")
    parser.add_argument("--db", default="./benchmark_db.duckdb",
                       help="Path to the DuckDB database")
    
    # Report options
    parser.add_argument("--report", choices=["performance", "hardware", "compatibility", "all"],
                       help="Generate a comprehensive report")
    parser.add_argument("--format", choices=["html", "md", "json"], default="html",
                       help="Output format for the report")
    
    # Comparison options
    parser.add_argument("--model", type=str,
                       help="Model name for hardware comparison")
    parser.add_argument("--hardware", type=str,
                       help="Hardware type for model comparison")
    parser.add_argument("--model-family", type=str,
                       help="Filter models by family (for --hardware comparison)")
    parser.add_argument("--compare-hardware", action="store_true",
                       help="Compare hardware platforms for the specified model")
    parser.add_argument("--compare-models", action="store_true",
                       help="Compare models on the specified hardware platform")
    parser.add_argument("--metric", choices=["throughput", "latency", "memory"], default="throughput",
                       help="Performance metric to compare")
    
    # Trend options
    parser.add_argument("--plot-trend", action="store_true",
                       help="Plot performance trend over time for model and hardware")
    
    # Output options
    parser.add_argument("--output", type=str,
                       help="Output file path")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Check if database exists
    if not os.path.exists(args.db):
        print(f"Error: Database file not found: {args.db}")
        return
    
    # Create visualizer
    visualizer = BenchmarkDBVisualizer(db_path=args.db, debug=args.debug)
    
    # Generate reports
    if args.report:
        if args.report == "performance" or args.report == "all":
            visualizer.generate_performance_report(args.format, args.output)
    
    # Compare hardware for a model
    if args.model and args.compare_hardware:
        visualizer.compare_hardware_for_model(args.model, args.metric, args.output)
    
    # Compare models on a hardware platform
    if args.hardware and args.compare_models:
        visualizer.compare_models_on_hardware(args.hardware, args.metric, args.model_family, args.output)
    
    # Plot performance trend
    if args.plot_trend and args.model and args.hardware:
        visualizer.plot_performance_trend(args.model, args.hardware, args.metric, args.output)
    
    # No specific action, print help
    if not any([args.report, 
               (args.model and args.compare_hardware),
               (args.hardware and args.compare_models),
               (args.plot_trend and args.model and args.hardware)]):
        parser.print_help()

if __name__ == "__main__":
    main()