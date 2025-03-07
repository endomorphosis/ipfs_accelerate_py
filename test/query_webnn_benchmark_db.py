#!/usr/bin/env python3
"""
WebNN and WebGPU Benchmark Database Query Tool

This script provides a utility for querying the WebNN and WebGPU benchmark database
to analyze test results, generate reports, and visualize performance data.

Usage:
    python query_webnn_benchmark_db.py --list-browsers
    python query_webnn_benchmark_db.py --list-models
    python query_webnn_benchmark_db.py --report capabilities --format markdown
    python query_webnn_benchmark_db.py --report performance --format html
    python query_webnn_benchmark_db.py --sql "SELECT * FROM browser_capabilities"
    python query_webnn_benchmark_db.py --model whisper-tiny --browser firefox --optimization compute-shaders

Version: March 2025
"""

import os
import sys
import time
import json
import argparse
import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Default database path (can be overridden)
DEFAULT_DB_PATH = os.environ.get("BENCHMARK_DB_PATH", "./webnn_benchmark.duckdb")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WebNN and WebGPU Benchmark Database Query Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Database options
    db_group = parser.add_argument_group('Database options')
    db_group.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH,
                          help="Path to the benchmark database")
    
    # Query options
    query_group = parser.add_argument_group('Query options')
    query_group.add_argument("--sql", type=str,
                            help="Run a custom SQL query")
    query_group.add_argument("--list-browsers", action="store_true",
                            help="List all browsers in the database")
    query_group.add_argument("--list-models", action="store_true",
                            help="List all models in the database")
    query_group.add_argument("--browser", type=str,
                            help="Filter by browser")
    query_group.add_argument("--model", type=str,
                            help="Filter by model")
    query_group.add_argument("--optimization", type=str,
                            choices=["compute-shaders", "parallel-loading", "shader-precompile"],
                            help="Filter by optimization type")
    query_group.add_argument("--days", type=int, default=30,
                            help="Filter results by days (default: last 30 days)")
    
    # Report options
    report_group = parser.add_argument_group('Report options')
    report_group.add_argument("--report", type=str,
                             choices=["capabilities", "performance", "optimizations", "browsers", "combined"],
                             help="Generate a specific report")
    report_group.add_argument("--format", type=str, default="markdown",
                             choices=["markdown", "html", "csv", "json", "chart"],
                             help="Report output format")
    report_group.add_argument("--output", type=str,
                             help="Output file path (defaults to stdout)")
    report_group.add_argument("--title", type=str,
                             help="Custom report title")
    
    # Visualization options
    visualization_group = parser.add_argument_group('Visualization options')
    visualization_group.add_argument("--chart-type", type=str, default="bar",
                                    choices=["bar", "line", "scatter", "pie", "heatmap"],
                                    help="Chart type for visualization")
    visualization_group.add_argument("--width", type=int, default=10,
                                    help="Chart width in inches")
    visualization_group.add_argument("--height", type=int, default=6,
                                    help="Chart height in inches")
    
    return parser.parse_args()

def connect_to_database(db_path: str):
    """Connect to the benchmark database.
    
    Args:
        db_path: Path to the database file.
        
    Returns:
        DuckDB connection object.
    """
    try:
        import duckdb
        return duckdb.connect(db_path)
    except ImportError:
        print("Error: DuckDB not installed. Install with 'pip install duckdb'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

def list_browsers(con):
    """List all browsers in the database.
    
    Args:
        con: DuckDB connection object.
    """
    try:
        # Query for browsers and their capabilities
        result = con.execute("""
        SELECT DISTINCT 
            browser, 
            COUNT(*) as test_count,
            MAX(timestamp) as last_test,
            SUM(CASE WHEN webnn_available THEN 1 ELSE 0 END) > 0 as webnn_available,
            SUM(CASE WHEN webgpu_available THEN 1 ELSE 0 END) > 0 as webgpu_available,
            SUM(CASE WHEN hardware_acceleration THEN 1 ELSE 0 END) > 0 as hardware_acceleration
        FROM browser_capabilities
        GROUP BY browser
        ORDER BY browser
        """).fetchall()
        
        # Print results as a table
        print("\nBrowsers in Database:\n")
        print("| Browser | Test Count | Last Test | WebNN | WebGPU | HW Accel |")
        print("|---------|------------|-----------|-------|--------|----------|")
        
        for row in result:
            browser, test_count, last_test, webnn, webgpu, hw_accel = row
            last_test_str = last_test.strftime("%Y-%m-%d %H:%M") if last_test else "N/A"
            webnn_str = "✅" if webnn else "❌"
            webgpu_str = "✅" if webgpu else "❌"
            hw_accel_str = "✅" if hw_accel else "❌"
            
            print(f"| {browser} | {test_count} | {last_test_str} | {webnn_str} | {webgpu_str} | {hw_accel_str} |")
        
        print("\n")
        
    except Exception as e:
        print(f"Error listing browsers: {e}")

def list_models(con):
    """List all models in the database.
    
    Args:
        con: DuckDB connection object.
    """
    try:
        # First, gather unique models from WebNN benchmarks
        webnn_models = con.execute("""
        SELECT DISTINCT model, model_type, COUNT(*) as test_count
        FROM webnn_benchmarks
        GROUP BY model, model_type
        """).fetchall()
        
        # Then, gather unique models from WebGPU benchmarks
        webgpu_models = con.execute("""
        SELECT DISTINCT model, model_type, COUNT(*) as test_count
        FROM webgpu_benchmarks
        GROUP BY model, model_type
        """).fetchall()
        
        # Combine and deduplicate
        models = {}
        for model, model_type, count in webnn_models:
            if model not in models:
                models[model] = {"model": model, "model_type": model_type, "webnn_count": count, "webgpu_count": 0}
            else:
                models[model]["webnn_count"] = count
        
        for model, model_type, count in webgpu_models:
            if model not in models:
                models[model] = {"model": model, "model_type": model_type, "webnn_count": 0, "webgpu_count": count}
            else:
                models[model]["webgpu_count"] = count
                if not models[model]["model_type"]:
                    models[model]["model_type"] = model_type
        
        # Print results as a table
        print("\nModels in Database:\n")
        print("| Model | Type | WebNN Tests | WebGPU Tests | Total Tests |")
        print("|-------|------|-------------|--------------|-------------|")
        
        for model_data in sorted(models.values(), key=lambda x: x["model"]):
            model = model_data["model"]
            model_type = model_data["model_type"] or "unknown"
            webnn_count = model_data["webnn_count"]
            webgpu_count = model_data["webgpu_count"]
            total = webnn_count + webgpu_count
            
            print(f"| {model} | {model_type} | {webnn_count} | {webgpu_count} | {total} |")
        
        print("\n")
        
    except Exception as e:
        print(f"Error listing models: {e}")

def generate_capabilities_report(con, format="markdown", output=None, title=None):
    """Generate a report on browser capabilities.
    
    Args:
        con: DuckDB connection object.
        format: Report format (markdown, html, csv, json).
        output: Output file path.
        title: Custom report title.
    """
    try:
        # Query for detailed browser capabilities
        result = con.execute("""
        SELECT DISTINCT 
            browser,
            webnn_available,
            webgpu_available,
            hardware_acceleration,
            vendor,
            architecture,
            webnn_backends,
            webgpu_adapter,
            MAX(timestamp) as last_test
        FROM browser_capabilities
        GROUP BY 
            browser,
            webnn_available,
            webgpu_available,
            hardware_acceleration,
            vendor,
            architecture,
            webnn_backends,
            webgpu_adapter
        ORDER BY browser
        """).fetchall()
        
        # Generate the report based on format
        if format == "markdown":
            report = generate_markdown_capabilities_report(result, title)
        elif format == "html":
            report = generate_html_capabilities_report(result, title)
        elif format == "csv":
            report = generate_csv_capabilities_report(result)
        elif format == "json":
            report = generate_json_capabilities_report(result)
        else:
            print(f"Unsupported format: {format}")
            return
        
        # Output the report
        if output:
            with open(output, 'w') as f:
                f.write(report)
            print(f"Capabilities report saved to: {output}")
        else:
            print(report)
        
    except Exception as e:
        print(f"Error generating capabilities report: {e}")

def generate_markdown_capabilities_report(result, title=None):
    """Generate a Markdown report for browser capabilities.
    
    Args:
        result: Query result with browser capabilities.
        title: Custom report title.
        
    Returns:
        Markdown report as a string.
    """
    title = title or "Browser Capabilities Report"
    report = f"# {title}\n\n"
    report += f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report += "## WebNN and WebGPU Support by Browser\n\n"
    report += "| Browser | WebNN | WebGPU | Hardware Acceleration | GPU Vendor | Architecture |\n"
    report += "|---------|-------|--------|----------------------|------------|---------------|\n"
    
    for row in result:
        browser, webnn, webgpu, hw_accel, vendor, arch, webnn_backends, webgpu_adapter, last_test = row
        
        webnn_str = "✅" if webnn else "❌"
        webgpu_str = "✅" if webgpu else "❌"
        hw_accel_str = "✅" if hw_accel else "❌"
        
        report += f"| {browser} | {webnn_str} | {webgpu_str} | {hw_accel_str} | {vendor} | {arch} |\n"
    
    report += "\n## Detailed Backend Information\n\n"
    
    for row in result:
        browser, webnn, webgpu, hw_accel, vendor, arch, webnn_backends, webgpu_adapter, last_test = row
        
        report += f"### {browser}\n\n"
        report += f"Last tested: {last_test.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        if webnn:
            report += "#### WebNN Backends\n\n"
            try:
                backends = json.loads(webnn_backends)
                if backends:
                    for backend in backends:
                        report += f"- {backend}\n"
                else:
                    report += "- No WebNN backends detected\n"
            except:
                report += "- Error parsing WebNN backends\n"
            report += "\n"
        
        if webgpu:
            report += "#### WebGPU Adapter\n\n"
            try:
                adapter = json.loads(webgpu_adapter)
                if adapter:
                    for key, value in adapter.items():
                        report += f"- {key}: {value}\n"
                else:
                    report += "- No WebGPU adapter details available\n"
            except:
                report += "- Error parsing WebGPU adapter\n"
            report += "\n"
    
    return report

def generate_html_capabilities_report(result, title=None):
    """Generate an HTML report for browser capabilities.
    
    Args:
        result: Query result with browser capabilities.
        title: Custom report title.
        
    Returns:
        HTML report as a string.
    """
    title = title or "Browser Capabilities Report"
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ text-align: left; padding: 12px; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .success {{ color: green; font-weight: bold; }}
        .failure {{ color: red; }}
        .details {{ margin-left: 20px; margin-bottom: 20px; padding: 10px; background-color: #f8f9fa; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>WebNN and WebGPU Support by Browser</h2>
    <table>
        <tr>
            <th>Browser</th>
            <th>WebNN</th>
            <th>WebGPU</th>
            <th>Hardware Acceleration</th>
            <th>GPU Vendor</th>
            <th>Architecture</th>
        </tr>
"""
    
    for row in result:
        browser, webnn, webgpu, hw_accel, vendor, arch, webnn_backends, webgpu_adapter, last_test = row
        
        webnn_str = '<span class="success">✓</span>' if webnn else '<span class="failure">✗</span>'
        webgpu_str = '<span class="success">✓</span>' if webgpu else '<span class="failure">✗</span>'
        hw_accel_str = '<span class="success">✓</span>' if hw_accel else '<span class="failure">✗</span>'
        
        html += f"""
        <tr>
            <td>{browser}</td>
            <td>{webnn_str}</td>
            <td>{webgpu_str}</td>
            <td>{hw_accel_str}</td>
            <td>{vendor}</td>
            <td>{arch}</td>
        </tr>
"""
    
    html += """
    </table>
    
    <h2>Detailed Backend Information</h2>
"""
    
    for row in result:
        browser, webnn, webgpu, hw_accel, vendor, arch, webnn_backends, webgpu_adapter, last_test = row
        
        html += f"""
    <h3>{browser}</h3>
    <p>Last tested: {last_test.strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
        
        if webnn:
            html += """
    <h4>WebNN Backends</h4>
    <div class="details">
"""
            try:
                backends = json.loads(webnn_backends)
                if backends:
                    for backend in backends:
                        html += f"        <p>• {backend}</p>\n"
                else:
                    html += "        <p>• No WebNN backends detected</p>\n"
            except:
                html += "        <p>• Error parsing WebNN backends</p>\n"
            
            html += "    </div>\n"
        
        if webgpu:
            html += """
    <h4>WebGPU Adapter</h4>
    <div class="details">
"""
            try:
                adapter = json.loads(webgpu_adapter)
                if adapter:
                    for key, value in adapter.items():
                        html += f"        <p>• {key}: {value}</p>\n"
                else:
                    html += "        <p>• No WebGPU adapter details available</p>\n"
            except:
                html += "        <p>• Error parsing WebGPU adapter</p>\n"
            
            html += "    </div>\n"
    
    html += """
</body>
</html>
"""
    
    return html

def generate_csv_capabilities_report(result):
    """Generate a CSV report for browser capabilities.
    
    Args:
        result: Query result with browser capabilities.
        
    Returns:
        CSV report as a string.
    """
    csv = "Browser,WebNN,WebGPU,HardwareAcceleration,Vendor,Architecture,LastTest\n"
    
    for row in result:
        browser, webnn, webgpu, hw_accel, vendor, arch, webnn_backends, webgpu_adapter, last_test = row
        
        # Convert to CSV format
        csv += f"{browser},{webnn},{webgpu},{hw_accel},{vendor},{arch},{last_test}\n"
    
    return csv

def generate_json_capabilities_report(result):
    """Generate a JSON report for browser capabilities.
    
    Args:
        result: Query result with browser capabilities.
        
    Returns:
        JSON report as a string.
    """
    data = []
    
    for row in result:
        browser, webnn, webgpu, hw_accel, vendor, arch, webnn_backends, webgpu_adapter, last_test = row
        
        # Parse JSON fields
        try:
            webnn_backends_parsed = json.loads(webnn_backends)
        except:
            webnn_backends_parsed = []
            
        try:
            webgpu_adapter_parsed = json.loads(webgpu_adapter)
        except:
            webgpu_adapter_parsed = {}
        
        # Create browser entry
        browser_data = {
            "browser": browser,
            "capabilities": {
                "webnn_available": bool(webnn),
                "webgpu_available": bool(webgpu),
                "hardware_acceleration": bool(hw_accel),
                "vendor": vendor,
                "architecture": arch
            },
            "webnn_backends": webnn_backends_parsed,
            "webgpu_adapter": webgpu_adapter_parsed,
            "last_test": last_test.strftime("%Y-%m-%d %H:%M:%S") if last_test else None
        }
        
        data.append(browser_data)
    
    return json.dumps({"browsers": data}, indent=2)

def generate_performance_report(con, browser=None, model=None, days=30, format="markdown", output=None, title=None):
    """Generate a performance report for WebNN and WebGPU benchmarks.
    
    Args:
        con: DuckDB connection object.
        browser: Filter by browser.
        model: Filter by model.
        days: Filter by days.
        format: Report format.
        output: Output file path.
        title: Custom report title.
    """
    try:
        # Calculate cutoff date
        cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        
        # Build WHERE clause
        where_clause = f"WHERE timestamp >= '{cutoff_date}'"
        if browser:
            where_clause += f" AND browser = '{browser}'"
        if model:
            where_clause += f" AND model = '{model}'"
        
        # Query for WebNN benchmarks
        webnn_query = f"""
        SELECT 
            browser,
            model,
            model_type,
            batch_size,
            AVG(cpu_time_ms) as avg_cpu_time,
            AVG(webnn_time_ms) as avg_webnn_time,
            AVG(speedup) as avg_speedup,
            COUNT(*) as test_count,
            MAX(timestamp) as last_test
        FROM webnn_benchmarks
        {where_clause}
        GROUP BY browser, model, model_type, batch_size
        ORDER BY model_type, model, browser, batch_size
        """
        
        webnn_results = con.execute(webnn_query).fetchall()
        
        # Query for WebGPU benchmarks
        webgpu_query = f"""
        SELECT 
            browser,
            model,
            model_type,
            batch_size,
            AVG(inference_time_ms) as avg_inference_time,
            AVG(first_inference_time_ms) as avg_first_inference_time,
            AVG(loading_time_ms) as avg_loading_time,
            AVG(memory_usage_mb) as avg_memory_usage,
            COUNT(*) as test_count,
            MAX(timestamp) as last_test
        FROM webgpu_benchmarks
        {where_clause}
        GROUP BY browser, model, model_type, batch_size
        ORDER BY model_type, model, browser, batch_size
        """
        
        webgpu_results = con.execute(webgpu_query).fetchall()
        
        # Generate the report based on format
        if format == "markdown":
            report = generate_markdown_performance_report(webnn_results, webgpu_results, title, days, browser, model)
        elif format == "html":
            report = generate_html_performance_report(webnn_results, webgpu_results, title, days, browser, model)
        elif format == "csv":
            report = generate_csv_performance_report(webnn_results, webgpu_results)
        elif format == "json":
            report = generate_json_performance_report(webnn_results, webgpu_results)
        elif format == "chart":
            # Generate chart and save to file
            if not output:
                output = f"performance_chart_{int(time.time())}.png"
            generate_performance_chart(webnn_results, webgpu_results, output, browser, model)
            print(f"Performance chart saved to: {output}")
            return
        else:
            print(f"Unsupported format: {format}")
            return
        
        # Output the report
        if output:
            with open(output, 'w') as f:
                f.write(report)
            print(f"Performance report saved to: {output}")
        else:
            print(report)
        
    except Exception as e:
        print(f"Error generating performance report: {e}")
        import traceback
        traceback.print_exc()

def generate_markdown_performance_report(webnn_results, webgpu_results, title=None, days=30, browser=None, model=None):
    """Generate a Markdown report for performance benchmarks.
    
    Args:
        webnn_results: Query results for WebNN benchmarks.
        webgpu_results: Query results for WebGPU benchmarks.
        title: Custom report title.
        days: Number of days to include in report.
        browser: Browser filter.
        model: Model filter.
        
    Returns:
        Markdown report as a string.
    """
    title = title or "WebNN and WebGPU Performance Report"
    report = f"# {title}\n\n"
    report += f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add filters to report
    filters = []
    if days:
        filters.append(f"Last {days} days")
    if browser:
        filters.append(f"Browser: {browser}")
    if model:
        filters.append(f"Model: {model}")
    
    if filters:
        report += "Filters: " + ", ".join(filters) + "\n\n"
    
    # Group models by type
    model_types = {}
    
    # Process WebNN results
    for row in webnn_results:
        browser, model, model_type, batch_size, avg_cpu, avg_webnn, avg_speedup, test_count, last_test = row
        
        if model_type not in model_types:
            model_types[model_type] = {"models": set(), "webnn": {}, "webgpu": {}}
            
        model_types[model_type]["models"].add(model)
        
        if model not in model_types[model_type]["webnn"]:
            model_types[model_type]["webnn"][model] = []
            
        model_types[model_type]["webnn"][model].append({
            "browser": browser,
            "batch_size": batch_size,
            "avg_cpu": avg_cpu,
            "avg_webnn": avg_webnn,
            "avg_speedup": avg_speedup,
            "test_count": test_count,
            "last_test": last_test
        })
    
    # Process WebGPU results
    for row in webgpu_results:
        browser, model, model_type, batch_size, avg_inference, avg_first, avg_loading, avg_memory, test_count, last_test = row
        
        if model_type not in model_types:
            model_types[model_type] = {"models": set(), "webnn": {}, "webgpu": {}}
            
        model_types[model_type]["models"].add(model)
        
        if model not in model_types[model_type]["webgpu"]:
            model_types[model_type]["webgpu"][model] = []
            
        model_types[model_type]["webgpu"][model].append({
            "browser": browser,
            "batch_size": batch_size,
            "avg_inference": avg_inference,
            "avg_first": avg_first,
            "avg_loading": avg_loading,
            "avg_memory": avg_memory,
            "test_count": test_count,
            "last_test": last_test
        })
    
    # Generate report for each model type
    for model_type, data in sorted(model_types.items()):
        report += f"## {model_type.title()} Models\n\n"
        
        # WebNN section
        if any(data["webnn"].values()):
            report += "### WebNN Performance\n\n"
            report += "| Model | Browser | Batch Size | CPU Time (ms) | WebNN Time (ms) | Speedup | Tests |\n"
            report += "|-------|---------|------------|---------------|-----------------|---------|-------|\n"
            
            for model in sorted(data["models"]):
                if model in data["webnn"]:
                    for result in sorted(data["webnn"][model], key=lambda x: (x["browser"], x["batch_size"])):
                        browser = result["browser"]
                        batch_size = result["batch_size"]
                        avg_cpu = f"{result['avg_cpu']:.2f}" if result["avg_cpu"] is not None else "N/A"
                        avg_webnn = f"{result['avg_webnn']:.2f}" if result["avg_webnn"] is not None else "N/A"
                        avg_speedup = f"{result['avg_speedup']:.2f}x" if result["avg_speedup"] is not None else "N/A"
                        test_count = result["test_count"]
                        
                        report += f"| {model} | {browser} | {batch_size} | {avg_cpu} | {avg_webnn} | {avg_speedup} | {test_count} |\n"
            
            report += "\n"
        
        # WebGPU section
        if any(data["webgpu"].values()):
            report += "### WebGPU Performance\n\n"
            report += "| Model | Browser | Batch Size | Inference Time (ms) | First Inference (ms) | Loading Time (ms) | Memory (MB) | Tests |\n"
            report += "|-------|---------|------------|---------------------|---------------------|-------------------|------------|-------|\n"
            
            for model in sorted(data["models"]):
                if model in data["webgpu"]:
                    for result in sorted(data["webgpu"][model], key=lambda x: (x["browser"], x["batch_size"])):
                        browser = result["browser"]
                        batch_size = result["batch_size"]
                        avg_inference = f"{result['avg_inference']:.2f}" if result["avg_inference"] is not None else "N/A"
                        avg_first = f"{result['avg_first']:.2f}" if result["avg_first"] is not None else "N/A"
                        avg_loading = f"{result['avg_loading']:.2f}" if result["avg_loading"] is not None else "N/A"
                        avg_memory = f"{result['avg_memory']:.2f}" if result["avg_memory"] is not None else "N/A"
                        test_count = result["test_count"]
                        
                        report += f"| {model} | {browser} | {batch_size} | {avg_inference} | {avg_first} | {avg_loading} | {avg_memory} | {test_count} |\n"
            
            report += "\n"
    
    # Generate summary
    report += "## Performance Summary\n\n"
    
    # WebNN summary
    if webnn_results:
        webnn_summary = {}
        for row in webnn_results:
            browser, model, model_type, batch_size, avg_cpu, avg_webnn, avg_speedup, test_count, last_test = row
            
            if browser not in webnn_summary:
                webnn_summary[browser] = {"speedups": [], "models": set()}
                
            if avg_speedup is not None:
                webnn_summary[browser]["speedups"].append(avg_speedup)
                webnn_summary[browser]["models"].add(model)
        
        report += "### WebNN Speedup by Browser\n\n"
        report += "| Browser | Average Speedup | Max Speedup | Models Tested |\n"
        report += "|---------|----------------|-------------|---------------|\n"
        
        for browser, data in sorted(webnn_summary.items()):
            if data["speedups"]:
                avg_speedup = sum(data["speedups"]) / len(data["speedups"])
                max_speedup = max(data["speedups"])
                model_count = len(data["models"])
                
                report += f"| {browser} | {avg_speedup:.2f}x | {max_speedup:.2f}x | {model_count} |\n"
        
        report += "\n"
    
    # WebGPU summary
    if webgpu_results:
        webgpu_summary = {}
        for row in webgpu_results:
            browser, model, model_type, batch_size, avg_inference, avg_first, avg_loading, avg_memory, test_count, last_test = row
            
            if browser not in webgpu_summary:
                webgpu_summary[browser] = {
                    "inference_times": [],
                    "first_inference_times": [],
                    "loading_times": [],
                    "memory_usages": [],
                    "models": set()
                }
                
            if avg_inference is not None:
                webgpu_summary[browser]["inference_times"].append(avg_inference)
            if avg_first is not None:
                webgpu_summary[browser]["first_inference_times"].append(avg_first)
            if avg_loading is not None:
                webgpu_summary[browser]["loading_times"].append(avg_loading)
            if avg_memory is not None:
                webgpu_summary[browser]["memory_usages"].append(avg_memory)
                
            webgpu_summary[browser]["models"].add(model)
        
        report += "### WebGPU Performance by Browser\n\n"
        report += "| Browser | Avg Inference (ms) | Avg First Inference (ms) | Avg Loading (ms) | Avg Memory (MB) | Models Tested |\n"
        report += "|---------|-------------------|-----------------------|----------------|---------------|---------------|\n"
        
        for browser, data in sorted(webgpu_summary.items()):
            avg_inference = sum(data["inference_times"]) / len(data["inference_times"]) if data["inference_times"] else None
            avg_first = sum(data["first_inference_times"]) / len(data["first_inference_times"]) if data["first_inference_times"] else None
            avg_loading = sum(data["loading_times"]) / len(data["loading_times"]) if data["loading_times"] else None
            avg_memory = sum(data["memory_usages"]) / len(data["memory_usages"]) if data["memory_usages"] else None
            model_count = len(data["models"])
            
            avg_inference_str = f"{avg_inference:.2f}" if avg_inference is not None else "N/A"
            avg_first_str = f"{avg_first:.2f}" if avg_first is not None else "N/A"
            avg_loading_str = f"{avg_loading:.2f}" if avg_loading is not None else "N/A"
            avg_memory_str = f"{avg_memory:.2f}" if avg_memory is not None else "N/A"
            
            report += f"| {browser} | {avg_inference_str} | {avg_first_str} | {avg_loading_str} | {avg_memory_str} | {model_count} |\n"
    
    return report

def generate_html_performance_report(webnn_results, webgpu_results, title=None, days=30, browser=None, model=None):
    """Generate an HTML report for performance benchmarks.
    
    Args:
        webnn_results: Query results for WebNN benchmarks.
        webgpu_results: Query results for WebGPU benchmarks.
        title: Custom report title.
        days: Number of days to include in report.
        browser: Browser filter.
        model: Model filter.
        
    Returns:
        HTML report as a string.
    """
    # This is a simplified implementation - in a real-world scenario, this would be more comprehensive
    title = title or "WebNN and WebGPU Performance Report"
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ text-align: left; padding: 12px; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .model-type {{ background-color: #f8f9fa; padding: 10px; margin-top: 20px; border-radius: 4px; }}
        .speedup {{ color: green; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
    
    # Add filters to report
    filters = []
    if days:
        filters.append(f"Last {days} days")
    if browser:
        filters.append(f"Browser: {browser}")
    if model:
        filters.append(f"Model: {model}")
    
    if filters:
        html += f"    <p>Filters: {', '.join(filters)}</p>\n"
    
    # Group models by type
    model_types = {}
    
    # Process WebNN results
    for row in webnn_results:
        browser, model, model_type, batch_size, avg_cpu, avg_webnn, avg_speedup, test_count, last_test = row
        
        if model_type not in model_types:
            model_types[model_type] = {"models": set(), "webnn": {}, "webgpu": {}}
            
        model_types[model_type]["models"].add(model)
        
        if model not in model_types[model_type]["webnn"]:
            model_types[model_type]["webnn"][model] = []
            
        model_types[model_type]["webnn"][model].append({
            "browser": browser,
            "batch_size": batch_size,
            "avg_cpu": avg_cpu,
            "avg_webnn": avg_webnn,
            "avg_speedup": avg_speedup,
            "test_count": test_count,
            "last_test": last_test
        })
    
    # Process WebGPU results
    for row in webgpu_results:
        browser, model, model_type, batch_size, avg_inference, avg_first, avg_loading, avg_memory, test_count, last_test = row
        
        if model_type not in model_types:
            model_types[model_type] = {"models": set(), "webnn": {}, "webgpu": {}}
            
        model_types[model_type]["models"].add(model)
        
        if model not in model_types[model_type]["webgpu"]:
            model_types[model_type]["webgpu"][model] = []
            
        model_types[model_type]["webgpu"][model].append({
            "browser": browser,
            "batch_size": batch_size,
            "avg_inference": avg_inference,
            "avg_first": avg_first,
            "avg_loading": avg_loading,
            "avg_memory": avg_memory,
            "test_count": test_count,
            "last_test": last_test
        })
    
    # Generate report for each model type
    for model_type, data in sorted(model_types.items()):
        html += f"""
    <div class="model-type">
        <h2>{model_type.title()} Models</h2>
"""
        
        # WebNN section
        if any(data["webnn"].values()):
            html += """
        <h3>WebNN Performance</h3>
        <table>
            <tr>
                <th>Model</th>
                <th>Browser</th>
                <th>Batch Size</th>
                <th>CPU Time (ms)</th>
                <th>WebNN Time (ms)</th>
                <th>Speedup</th>
                <th>Tests</th>
            </tr>
"""
            
            for model in sorted(data["models"]):
                if model in data["webnn"]:
                    for result in sorted(data["webnn"][model], key=lambda x: (x["browser"], x["batch_size"])):
                        browser = result["browser"]
                        batch_size = result["batch_size"]
                        avg_cpu = f"{result['avg_cpu']:.2f}" if result["avg_cpu"] is not None else "N/A"
                        avg_webnn = f"{result['avg_webnn']:.2f}" if result["avg_webnn"] is not None else "N/A"
                        avg_speedup = f"{result['avg_speedup']:.2f}x" if result["avg_speedup"] is not None else "N/A"
                        test_count = result["test_count"]
                        
                        html += f"""
            <tr>
                <td>{model}</td>
                <td>{browser}</td>
                <td>{batch_size}</td>
                <td>{avg_cpu}</td>
                <td>{avg_webnn}</td>
                <td class="speedup">{avg_speedup}</td>
                <td>{test_count}</td>
            </tr>
"""
            
            html += """
        </table>
"""
        
        # WebGPU section
        if any(data["webgpu"].values()):
            html += """
        <h3>WebGPU Performance</h3>
        <table>
            <tr>
                <th>Model</th>
                <th>Browser</th>
                <th>Batch Size</th>
                <th>Inference Time (ms)</th>
                <th>First Inference (ms)</th>
                <th>Loading Time (ms)</th>
                <th>Memory (MB)</th>
                <th>Tests</th>
            </tr>
"""
            
            for model in sorted(data["models"]):
                if model in data["webgpu"]:
                    for result in sorted(data["webgpu"][model], key=lambda x: (x["browser"], x["batch_size"])):
                        browser = result["browser"]
                        batch_size = result["batch_size"]
                        avg_inference = f"{result['avg_inference']:.2f}" if result["avg_inference"] is not None else "N/A"
                        avg_first = f"{result['avg_first']:.2f}" if result["avg_first"] is not None else "N/A"
                        avg_loading = f"{result['avg_loading']:.2f}" if result["avg_loading"] is not None else "N/A"
                        avg_memory = f"{result['avg_memory']:.2f}" if result["avg_memory"] is not None else "N/A"
                        test_count = result["test_count"]
                        
                        html += f"""
            <tr>
                <td>{model}</td>
                <td>{browser}</td>
                <td>{batch_size}</td>
                <td>{avg_inference}</td>
                <td>{avg_first}</td>
                <td>{avg_loading}</td>
                <td>{avg_memory}</td>
                <td>{test_count}</td>
            </tr>
"""
            
            html += """
        </table>
"""
        
        html += """
    </div>
"""
    
    # Generate summary
    html += """
    <h2>Performance Summary</h2>
"""
    
    # WebNN summary
    if webnn_results:
        webnn_summary = {}
        for row in webnn_results:
            browser, model, model_type, batch_size, avg_cpu, avg_webnn, avg_speedup, test_count, last_test = row
            
            if browser not in webnn_summary:
                webnn_summary[browser] = {"speedups": [], "models": set()}
                
            if avg_speedup is not None:
                webnn_summary[browser]["speedups"].append(avg_speedup)
                webnn_summary[browser]["models"].add(model)
        
        html += """
    <h3>WebNN Speedup by Browser</h3>
    <table>
        <tr>
            <th>Browser</th>
            <th>Average Speedup</th>
            <th>Max Speedup</th>
            <th>Models Tested</th>
        </tr>
"""
        
        for browser, data in sorted(webnn_summary.items()):
            if data["speedups"]:
                avg_speedup = sum(data["speedups"]) / len(data["speedups"])
                max_speedup = max(data["speedups"])
                model_count = len(data["models"])
                
                html += f"""
        <tr>
            <td>{browser}</td>
            <td class="speedup">{avg_speedup:.2f}x</td>
            <td class="speedup">{max_speedup:.2f}x</td>
            <td>{model_count}</td>
        </tr>
"""
        
        html += """
    </table>
"""
    
    # WebGPU summary
    if webgpu_results:
        webgpu_summary = {}
        for row in webgpu_results:
            browser, model, model_type, batch_size, avg_inference, avg_first, avg_loading, avg_memory, test_count, last_test = row
            
            if browser not in webgpu_summary:
                webgpu_summary[browser] = {
                    "inference_times": [],
                    "first_inference_times": [],
                    "loading_times": [],
                    "memory_usages": [],
                    "models": set()
                }
                
            if avg_inference is not None:
                webgpu_summary[browser]["inference_times"].append(avg_inference)
            if avg_first is not None:
                webgpu_summary[browser]["first_inference_times"].append(avg_first)
            if avg_loading is not None:
                webgpu_summary[browser]["loading_times"].append(avg_loading)
            if avg_memory is not None:
                webgpu_summary[browser]["memory_usages"].append(avg_memory)
                
            webgpu_summary[browser]["models"].add(model)
        
        html += """
    <h3>WebGPU Performance by Browser</h3>
    <table>
        <tr>
            <th>Browser</th>
            <th>Avg Inference (ms)</th>
            <th>Avg First Inference (ms)</th>
            <th>Avg Loading (ms)</th>
            <th>Avg Memory (MB)</th>
            <th>Models Tested</th>
        </tr>
"""
        
        for browser, data in sorted(webgpu_summary.items()):
            avg_inference = sum(data["inference_times"]) / len(data["inference_times"]) if data["inference_times"] else None
            avg_first = sum(data["first_inference_times"]) / len(data["first_inference_times"]) if data["first_inference_times"] else None
            avg_loading = sum(data["loading_times"]) / len(data["loading_times"]) if data["loading_times"] else None
            avg_memory = sum(data["memory_usages"]) / len(data["memory_usages"]) if data["memory_usages"] else None
            model_count = len(data["models"])
            
            avg_inference_str = f"{avg_inference:.2f}" if avg_inference is not None else "N/A"
            avg_first_str = f"{avg_first:.2f}" if avg_first is not None else "N/A"
            avg_loading_str = f"{avg_loading:.2f}" if avg_loading is not None else "N/A"
            avg_memory_str = f"{avg_memory:.2f}" if avg_memory is not None else "N/A"
            
            html += f"""
        <tr>
            <td>{browser}</td>
            <td>{avg_inference_str}</td>
            <td>{avg_first_str}</td>
            <td>{avg_loading_str}</td>
            <td>{avg_memory_str}</td>
            <td>{model_count}</td>
        </tr>
"""
        
        html += """
    </table>
"""
    
    html += """
</body>
</html>
"""
    
    return html

def generate_csv_performance_report(webnn_results, webgpu_results):
    """Generate a CSV report for performance benchmarks.
    
    Args:
        webnn_results: Query results for WebNN benchmarks.
        webgpu_results: Query results for WebGPU benchmarks.
        
    Returns:
        CSV report as a string.
    """
    # WebNN CSV
    webnn_csv = "Type,Browser,Model,ModelType,BatchSize,CPUTime,WebNNTime,Speedup,TestCount\n"
    
    for row in webnn_results:
        browser, model, model_type, batch_size, avg_cpu, avg_webnn, avg_speedup, test_count, last_test = row
        webnn_csv += f"WebNN,{browser},{model},{model_type},{batch_size},{avg_cpu},{avg_webnn},{avg_speedup},{test_count}\n"
    
    # WebGPU CSV
    webgpu_csv = "Type,Browser,Model,ModelType,BatchSize,InferenceTime,FirstInferenceTime,LoadingTime,MemoryUsage,TestCount\n"
    
    for row in webgpu_results:
        browser, model, model_type, batch_size, avg_inference, avg_first, avg_loading, avg_memory, test_count, last_test = row
        webgpu_csv += f"WebGPU,{browser},{model},{model_type},{batch_size},{avg_inference},{avg_first},{avg_loading},{avg_memory},{test_count}\n"
    
    return webnn_csv + "\n" + webgpu_csv

def generate_json_performance_report(webnn_results, webgpu_results):
    """Generate a JSON report for performance benchmarks.
    
    Args:
        webnn_results: Query results for WebNN benchmarks.
        webgpu_results: Query results for WebGPU benchmarks.
        
    Returns:
        JSON report as a string.
    """
    data = {
        "webnn": [],
        "webgpu": [],
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Process WebNN results
    for row in webnn_results:
        browser, model, model_type, batch_size, avg_cpu, avg_webnn, avg_speedup, test_count, last_test = row
        
        data["webnn"].append({
            "browser": browser,
            "model": model,
            "model_type": model_type,
            "batch_size": batch_size,
            "avg_cpu_time_ms": avg_cpu,
            "avg_webnn_time_ms": avg_webnn,
            "avg_speedup": avg_speedup,
            "test_count": test_count,
            "last_test": last_test.strftime("%Y-%m-%d %H:%M:%S") if last_test else None
        })
    
    # Process WebGPU results
    for row in webgpu_results:
        browser, model, model_type, batch_size, avg_inference, avg_first, avg_loading, avg_memory, test_count, last_test = row
        
        data["webgpu"].append({
            "browser": browser,
            "model": model,
            "model_type": model_type,
            "batch_size": batch_size,
            "avg_inference_time_ms": avg_inference,
            "avg_first_inference_time_ms": avg_first,
            "avg_loading_time_ms": avg_loading,
            "avg_memory_usage_mb": avg_memory,
            "test_count": test_count,
            "last_test": last_test.strftime("%Y-%m-%d %H:%M:%S") if last_test else None
        })
    
    return json.dumps(data, indent=2)

def generate_performance_chart(webnn_results, webgpu_results, output_file, browser=None, model=None):
    """Generate a performance chart and save to file.
    
    Args:
        webnn_results: Query results for WebNN benchmarks.
        webgpu_results: Query results for WebGPU benchmarks.
        output_file: Output file path.
        browser: Browser filter.
        model: Model filter.
    """
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Set up the chart based on available data
    if webnn_results and webgpu_results:
        # Comparison of WebNN and WebGPU by browser
        browsers = set()
        for row in webnn_results:
            browsers.add(row[0])  # browser is the first column
        for row in webgpu_results:
            browsers.add(row[0])  # browser is the first column
        
        browsers = sorted(browsers)
        
        # Calculate average speedup for WebNN and average inference time for WebGPU
        webnn_speedups = {}
        for row in webnn_results:
            browser, model, model_type, batch_size, avg_cpu, avg_webnn, avg_speedup, test_count, last_test = row
            
            if browser not in webnn_speedups:
                webnn_speedups[browser] = []
                
            if avg_speedup is not None:
                webnn_speedups[browser].append(avg_speedup)
        
        webgpu_inference_times = {}
        for row in webgpu_results:
            browser, model, model_type, batch_size, avg_inference, avg_first, avg_loading, avg_memory, test_count, last_test = row
            
            if browser not in webgpu_inference_times:
                webgpu_inference_times[browser] = []
                
            if avg_inference is not None:
                webgpu_inference_times[browser].append(avg_inference)
        
        # Prepare data for plotting
        webnn_avg_speedups = []
        webgpu_avg_inference = []
        
        for browser in browsers:
            if browser in webnn_speedups and webnn_speedups[browser]:
                webnn_avg_speedups.append(sum(webnn_speedups[browser]) / len(webnn_speedups[browser]))
            else:
                webnn_avg_speedups.append(0)
                
            if browser in webgpu_inference_times and webgpu_inference_times[browser]:
                webgpu_avg_inference.append(sum(webgpu_inference_times[browser]) / len(webgpu_inference_times[browser]))
            else:
                webgpu_avg_inference.append(0)
        
        # Create a two-panel chart
        plt.subplot(1, 2, 1)
        plt.bar(browsers, webnn_avg_speedups, color='#4CAF50')
        plt.title('WebNN Average Speedup by Browser')
        plt.xlabel('Browser')
        plt.ylabel('Speedup (x)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(1, 2, 2)
        plt.bar(browsers, webgpu_avg_inference, color='#3F51B5')
        plt.title('WebGPU Average Inference Time by Browser')
        plt.xlabel('Browser')
        plt.ylabel('Inference Time (ms)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
    elif webnn_results:
        # Just WebNN data available
        browsers = sorted(set(row[0] for row in webnn_results))
        
        # Calculate average speedup for WebNN
        webnn_speedups = {}
        for row in webnn_results:
            browser, model, model_type, batch_size, avg_cpu, avg_webnn, avg_speedup, test_count, last_test = row
            
            if browser not in webnn_speedups:
                webnn_speedups[browser] = []
                
            if avg_speedup is not None:
                webnn_speedups[browser].append(avg_speedup)
        
        # Prepare data for plotting
        webnn_avg_speedups = []
        
        for browser in browsers:
            if browser in webnn_speedups and webnn_speedups[browser]:
                webnn_avg_speedups.append(sum(webnn_speedups[browser]) / len(webnn_speedups[browser]))
            else:
                webnn_avg_speedups.append(0)
        
        # Create a single panel chart
        plt.bar(browsers, webnn_avg_speedups, color='#4CAF50')
        plt.title('WebNN Average Speedup by Browser')
        plt.xlabel('Browser')
        plt.ylabel('Speedup (x)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
    elif webgpu_results:
        # Just WebGPU data available
        browsers = sorted(set(row[0] for row in webgpu_results))
        
        # Calculate average inference time for WebGPU
        webgpu_inference_times = {}
        for row in webgpu_results:
            browser, model, model_type, batch_size, avg_inference, avg_first, avg_loading, avg_memory, test_count, last_test = row
            
            if browser not in webgpu_inference_times:
                webgpu_inference_times[browser] = []
                
            if avg_inference is not None:
                webgpu_inference_times[browser].append(avg_inference)
        
        # Prepare data for plotting
        webgpu_avg_inference = []
        
        for browser in browsers:
            if browser in webgpu_inference_times and webgpu_inference_times[browser]:
                webgpu_avg_inference.append(sum(webgpu_inference_times[browser]) / len(webgpu_inference_times[browser]))
            else:
                webgpu_avg_inference.append(0)
        
        # Create a single panel chart
        plt.bar(browsers, webgpu_avg_inference, color='#3F51B5')
        plt.title('WebGPU Average Inference Time by Browser')
        plt.xlabel('Browser')
        plt.ylabel('Inference Time (ms)')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add a title
    title = "WebNN and WebGPU Performance Comparison"
    if browser:
        title += f" - {browser}"
    if model:
        title += f" - {model}"
    
    plt.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the chart
    plt.savefig(output_file)
    print(f"Chart saved to: {output_file}")

def run_custom_sql_query(con, sql, format="markdown", output=None):
    """Run a custom SQL query against the database.
    
    Args:
        con: DuckDB connection object.
        sql: SQL query to run.
        format: Output format.
        output: Output file path.
    """
    try:
        # Run the query
        result = con.execute(sql).fetchall()
        
        # Get column names
        column_names = [col[0] for col in con.description()]
        
        # Format the output
        if format == "markdown":
            # Create markdown table
            md_table = "| " + " | ".join(column_names) + " |\n"
            md_table += "| " + " | ".join(["---"] * len(column_names)) + " |\n"
            
            for row in result:
                md_table += "| " + " | ".join(str(col) for col in row) + " |\n"
            
            output_content = md_table
            
        elif format == "html":
            # Create HTML table
            html_table = "<table>\n<tr>\n"
            html_table += "".join(f"<th>{col}</th>" for col in column_names)
            html_table += "\n</tr>\n"
            
            for row in result:
                html_table += "<tr>\n"
                html_table += "".join(f"<td>{col}</td>" for col in row)
                html_table += "\n</tr>\n"
            
            html_table += "</table>"
            
            output_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>SQL Query Result</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ text-align: left; padding: 12px; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>SQL Query Result</h1>
    <p>Query: <code>{sql}</code></p>
    {html_table}
</body>
</html>
"""
            
        elif format == "csv":
            # Create CSV
            csv_content = ",".join(column_names) + "\n"
            
            for row in result:
                csv_content += ",".join(str(col).replace(",", ";") for col in row) + "\n"
            
            output_content = csv_content
            
        elif format == "json":
            # Create JSON
            data = []
            
            for row in result:
                row_dict = {}
                for i, col in enumerate(column_names):
                    row_dict[col] = row[i]
                data.append(row_dict)
            
            output_content = json.dumps(data, indent=2)
            
        elif format == "chart":
            # Create chart (supported only for certain queries)
            if len(column_names) >= 2:
                plt.figure(figsize=(10, 6))
                
                # Assume first column is x-axis, second is y-axis
                x = [row[0] for row in result]
                y = [row[1] for row in result]
                
                plt.bar(x, y)
                plt.xlabel(column_names[0])
                plt.ylabel(column_names[1])
                plt.title("SQL Query Result")
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                if not output:
                    output = "sql_query_chart.png"
                
                plt.savefig(output)
                print(f"Chart saved to: {output}")
                return
            else:
                raise ValueError("Chart format requires at least 2 columns in the result")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Output the result
        if output:
            with open(output, 'w') as f:
                f.write(output_content)
            print(f"SQL query result saved to: {output}")
        else:
            print(output_content)
            
    except Exception as e:
        print(f"Error running SQL query: {e}")

def main():
    """Main function."""
    args = parse_args()
    
    # Connect to the database
    con = connect_to_database(args.db_path)
    
    # Handle the request
    if args.list_browsers:
        list_browsers(con)
    elif args.list_models:
        list_models(con)
    elif args.report == "capabilities":
        generate_capabilities_report(con, args.format, args.output, args.title)
    elif args.report == "performance":
        generate_performance_report(con, args.browser, args.model, args.days, args.format, args.output, args.title)
    elif args.sql:
        run_custom_sql_query(con, args.sql, args.format, args.output)
    else:
        # Default action: show usage
        print("Please specify an action. Run with --help for usage information.")

if __name__ == "__main__":
    main()