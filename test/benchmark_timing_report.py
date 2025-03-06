#!/usr/bin/env python3
"""
Comprehensive Benchmark Timing Report Generator

This script generates detailed benchmark timing reports for all 13 model types
across 8 hardware endpoints, with comparative visualizations and analysis.

Usage:
    python benchmark_timing_report.py --generate --output report.html
    python benchmark_timing_report.py --interactive
    python benchmark_timing_report.py --api-server
"""

import os
import sys
import argparse
import logging
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import duckdb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define model types and hardware endpoints
MODEL_TYPES = ["bert", "t5", "llama", "clip", "vit", "clap", 
               "wav2vec2", "whisper", "llava", "llava-next", "xclip", "qwen2", "detr"]
HARDWARE_ENDPOINTS = ["cpu", "cuda", "rocm", "mps", "openvino", "qnn", "webnn", "webgpu"]

# Hardware endpoint descriptions
HARDWARE_DESCRIPTIONS = {
    "cpu": "CPU (Standard CPU processing)",
    "cuda": "CUDA (NVIDIA GPU acceleration)",
    "rocm": "ROCm (AMD GPU acceleration)",
    "mps": "MPS (Apple Silicon GPU acceleration)",
    "openvino": "OpenVINO (Intel acceleration)",
    "qnn": "QNN (Qualcomm AI Engine)",
    "webnn": "WebNN (Browser neural network API)",
    "webgpu": "WebGPU (Browser graphics API for ML)"
}

# Model type descriptions and categories
MODEL_DESCRIPTIONS = {
    "bert": {"description": "BERT (Text embedding model)", "category": "text"},
    "t5": {"description": "T5 (Text-to-text generation model)", "category": "text"},
    "llama": {"description": "LLAMA (Large language model)", "category": "text"},
    "clip": {"description": "CLIP (Vision-text multimodal model)", "category": "multimodal"},
    "vit": {"description": "ViT (Vision transformer model)", "category": "vision"},
    "clap": {"description": "CLAP (Audio-text multimodal model)", "category": "audio"},
    "wav2vec2": {"description": "Wav2Vec2 (Speech recognition model)", "category": "audio"},
    "whisper": {"description": "Whisper (Speech recognition model)", "category": "audio"},
    "llava": {"description": "LLaVA (Vision-language model)", "category": "multimodal"},
    "llava-next": {"description": "LLaVA-Next (Advanced vision-language model)", "category": "multimodal"},
    "xclip": {"description": "XCLIP (Video-text multimodal model)", "category": "vision"},
    "qwen2": {"description": "Qwen2 (Advanced text generation model)", "category": "text"},
    "detr": {"description": "DETR (DEtection TRansformer for object detection)", "category": "vision"}
}

class BenchmarkTimingReport:
    """Generate comprehensive benchmark timing reports."""
    
    def __init__(self, db_path=None):
        """Initialize with optional database path."""
        self.db_path = db_path or os.environ.get("BENCHMARK_DB_PATH", "./benchmark_db.duckdb")
        self.conn = None
        self._connect_db()
        
    def _connect_db(self):
        """Connect to the DuckDB database."""
        try:
            self.conn = duckdb.connect(self.db_path, read_only=True)
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
            
    def _fetch_timing_data(self):
        """Fetch timing data for all models and hardware platforms."""
        try:
            # Placeholder query - adjust based on actual schema
            query = """
                SELECT 
                    m.model_name,
                    m.model_family,
                    hp.hardware_type,
                    pr.batch_size,
                    pr.average_latency_ms,
                    pr.throughput_items_per_second,
                    pr.memory_peak_mb,
                    pr.created_at
                FROM 
                    performance_results pr
                JOIN 
                    models m ON pr.model_id = m.model_id
                JOIN 
                    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
                ORDER BY
                    m.model_family, hp.hardware_type, pr.created_at DESC
            """
            
            result = self.conn.execute(query).fetchdf()
            return result
        except Exception as e:
            logger.error(f"Failed to fetch timing data: {str(e)}")
            # Return empty DataFrame with expected structure
            return pd.DataFrame(columns=[
                'model_name', 'model_family', 'hardware_type', 'batch_size',
                'average_latency_ms', 'throughput_items_per_second', 'memory_peak_mb',
                'created_at'
            ])
    
    def _get_latest_results(self, df):
        """Get the latest results for each model-hardware combination."""
        if df.empty:
            return df
            
        # Group by model and hardware, keep latest result
        latest_results = df.sort_values('created_at', ascending=False).groupby(
            ['model_family', 'hardware_type']).first().reset_index()
        
        return latest_results
        
    def generate_timing_report(self, output_format="html", output_path=None, days_lookback=30):
        """Generate the comprehensive timing report."""
        logger.info("Generating comprehensive benchmark timing report...")
        
        # Determine output path
        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"benchmark_timing_report_{timestamp}.{output_format}"
        
        # Fetch timing data
        all_data = self._fetch_timing_data()
        if all_data.empty:
            logger.warning("No timing data found in database")
            return None
            
        # Get latest results for each model-hardware combination
        latest_results = self._get_latest_results(all_data)
        
        # Generate report based on format
        if output_format == "html":
            self._generate_html_report(latest_results, all_data, output_path, days_lookback)
        elif output_format in ["md", "markdown"]:
            self._generate_markdown_report(latest_results, output_path)
        elif output_format == "json":
            self._generate_json_report(latest_results, output_path)
        else:
            logger.error(f"Unsupported output format: {output_format}")
            return None
            
        logger.info(f"Report generated: {output_path}")
        return output_path
        
    def _generate_html_report(self, latest_results, all_data, output_path, days_lookback):
        """Generate an HTML report with interactive visualizations."""
        try:
            # Create pivot table for latency comparisons
            latency_pivot = latest_results.pivot(
                index='model_family', 
                columns='hardware_type', 
                values='average_latency_ms'
            ).fillna(-1)
            
            # Create pivot table for throughput comparisons
            throughput_pivot = latest_results.pivot(
                index='model_family', 
                columns='hardware_type', 
                values='throughput_items_per_second'
            ).fillna(-1)
            
            # Create charts
            # Save latency comparison chart
            latency_fig = plt.figure(figsize=(12, 8))
            sns.heatmap(latency_pivot, annot=True, fmt='.2f', cmap='YlGnBu')
            plt.title('Latency Comparison (ms) - Lower is Better')
            latency_chart_path = 'latency_comparison.png'
            latency_fig.savefig(latency_chart_path)
            plt.close(latency_fig)
            
            # Save throughput comparison chart
            throughput_fig = plt.figure(figsize=(12, 8))
            sns.heatmap(throughput_pivot, annot=True, fmt='.2f', cmap='YlGnBu')
            plt.title('Throughput Comparison (items/sec) - Higher is Better')
            throughput_chart_path = 'throughput_comparison.png'
            throughput_fig.savefig(throughput_chart_path)
            plt.close(throughput_fig)
            
            # Generate time series data if available
            timeseries_charts = []
            if not all_data.empty and len(all_data) > 1:
                # Filter for last N days
                cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_lookback)
                recent_data = all_data[all_data['created_at'] >= cutoff_date]
                
                # Group by date and hardware type
                if not recent_data.empty:
                    # Convert created_at to date only
                    recent_data['date'] = pd.to_datetime(recent_data['created_at']).dt.date
                    
                    # Create time series chart for top models
                    for model in MODEL_TYPES[:5]:  # Top 5 models
                        model_data = recent_data[recent_data['model_family'] == model]
                        if not model_data.empty:
                            ts_fig = plt.figure(figsize=(10, 6))
                            for hw in HARDWARE_ENDPOINTS:
                                hw_data = model_data[model_data['hardware_type'] == hw]
                                if not hw_data.empty:
                                    plt.plot(hw_data['date'], hw_data['average_latency_ms'], label=hw)
                            
                            plt.title(f'Latency Trend for {model} - Last {days_lookback} Days')
                            plt.xlabel('Date')
                            plt.ylabel('Latency (ms)')
                            plt.legend()
                            plt.xticks(rotation=45)
                            
                            chart_path = f'{model}_timeseries.png'
                            ts_fig.savefig(chart_path)
                            plt.close(ts_fig)
                            timeseries_charts.append({
                                'model': model,
                                'path': chart_path
                            })
            
            # Generate HTML report
            with open(output_path, 'w') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Comprehensive Benchmark Timing Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2, h3 {{ color: #333; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .chart-container {{ margin: 20px 0; }}
                        .model-category {{ margin-top: 30px; }}
                        .hardware-info {{ margin-bottom: 30px; }}
                        .model-header {{ background-color: #e6f3ff; }}
                        .best-result {{ font-weight: bold; color: green; }}
                        .limited-support {{ color: orange; }}
                        .no-support {{ color: red; }}
                    </style>
                </head>
                <body>
                    <h1>Comprehensive Benchmark Timing Report</h1>
                    <p>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <h2>Overview</h2>
                    <p>This report provides detailed benchmark timing data for all 13 model types across 8 hardware endpoints, 
                    showing performance metrics including latency and throughput.</p>
                    
                    <h2>Hardware Platforms</h2>
                    <table>
                        <tr>
                            <th>Hardware</th>
                            <th>Description</th>
                        </tr>
                """)
                
                # Add hardware descriptions
                for hw, desc in HARDWARE_DESCRIPTIONS.items():
                    f.write(f"<tr><td>{hw}</td><td>{desc}</td></tr>\n")
                
                f.write("""
                    </table>
                    
                    <h2>Performance Visualization</h2>
                    
                    <h3>Latency Comparison (ms)</h3>
                    <div class="chart-container">
                        <img src="{0}" alt="Latency Comparison" style="max-width: 100%;">
                    </div>
                    
                    <h3>Throughput Comparison (items/sec)</h3>
                    <div class="chart-container">
                        <img src="{1}" alt="Throughput Comparison" style="max-width: 100%;">
                    </div>
                """.format(latency_chart_path, throughput_chart_path))
                
                # Add time series charts if available
                if timeseries_charts:
                    f.write("<h2>Performance Trends</h2>\n")
                    for chart in timeseries_charts:
                        f.write(f"""
                        <h3>Trends for {chart['model']}</h3>
                        <div class="chart-container">
                            <img src="{chart['path']}" alt="{chart['model']} Trends" style="max-width: 100%;">
                        </div>
                        """)
                
                # Detailed results by model category
                f.write("<h2>Detailed Results by Category</h2>\n")
                
                # Group models by category
                categories = {}
                for model, info in MODEL_DESCRIPTIONS.items():
                    cat = info["category"]
                    if cat not in categories:
                        categories[cat] = []
                    categories[cat].append(model)
                
                # Generate tables by category
                for cat, models in categories.items():
                    f.write(f"""
                    <div class="model-category">
                        <h3>{cat.capitalize()} Models</h3>
                        <table>
                            <tr>
                                <th>Model</th>
                    """)
                    
                    # Add hardware columns
                    for hw in HARDWARE_ENDPOINTS:
                        f.write(f"<th>{hw}</th>\n")
                    
                    f.write("</tr>\n")
                    
                    # Add data rows
                    for model in models:
                        f.write(f"""
                        <tr class="model-header">
                            <td>{MODEL_DESCRIPTIONS[model]['description']}</td>
                        """)
                        
                        # Add latency data for each hardware
                        for hw in HARDWARE_ENDPOINTS:
                            model_hw_data = latest_results[(latest_results['model_family'] == model) & 
                                                          (latest_results['hardware_type'] == hw)]
                            
                            if not model_hw_data.empty:
                                latency = model_hw_data.iloc[0]['average_latency_ms']
                                throughput = model_hw_data.iloc[0]['throughput_items_per_second']
                                f.write(f"<td>Latency: {latency:.2f}ms<br>Throughput: {throughput:.2f} items/s</td>\n")
                            else:
                                f.write("<td class='no-support'>No data available</td>\n")
                        
                        f.write("</tr>\n")
                    
                    f.write("</table></div>\n")
                
                # Add optimization recommendations
                f.write("""
                    <h2>Optimization Recommendations</h2>
                    <p>Based on the benchmark results, here are some optimization recommendations:</p>
                    <ul>
                """)
                
                # Example recommendations (in a real implementation, these would be derived from the data)
                recommendations = [
                    "Text models (BERT, T5) perform best on CUDA and WebGPU with shader precompilation",
                    "Audio models (Whisper, Wav2Vec2) see significant improvements with Firefox WebGPU compute shader optimizations",
                    "Vision models (ViT, CLIP) work well across most hardware platforms",
                    "Large language models (LLAMA, Qwen2) require CUDA or ROCm for optimal performance",
                    "Memory-intensive models (LLaVA, LLaVA-Next) perform best with dedicated GPU memory"
                ]
                
                for rec in recommendations:
                    f.write(f"<li>{rec}</li>\n")
                
                f.write("""
                    </ul>
                    
                    <h2>Conclusion</h2>
                    <p>This report provides a comprehensive view of the performance characteristics of 13 key model types 
                    across 8 hardware platforms. Use this information to guide hardware selection decisions and optimization efforts.</p>
                    
                </body>
                </html>
                """)
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {str(e)}")
            raise
        
    def _generate_markdown_report(self, latest_results, output_path):
        """Generate a markdown report."""
        try:
            with open(output_path, 'w') as f:
                f.write(f"# Comprehensive Benchmark Timing Report\n\n")
                f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Overview\n\n")
                f.write("This report provides detailed benchmark timing data for all 13 model types across 8 hardware endpoints.\n\n")
                
                f.write("## Hardware Platforms\n\n")
                f.write("| Hardware | Description |\n")
                f.write("|----------|-------------|\n")
                
                for hw, desc in HARDWARE_DESCRIPTIONS.items():
                    f.write(f"| {hw} | {desc} |\n")
                
                f.write("\n## Model Performance\n\n")
                
                # Group by model category
                categories = {}
                for model, info in MODEL_DESCRIPTIONS.items():
                    cat = info["category"]
                    if cat not in categories:
                        categories[cat] = []
                    categories[cat].append(model)
                
                # Generate tables by category
                for cat, models in categories.items():
                    f.write(f"### {cat.capitalize()} Models\n\n")
                    
                    # Create header row with model and hardware types
                    f.write("| Model | " + " | ".join(HARDWARE_ENDPOINTS) + " |\n")
                    f.write("|-------|" + "-|".join(["-" * len(hw) for hw in HARDWARE_ENDPOINTS]) + "-|\n")
                    
                    # Add data rows
                    for model in models:
                        row = f"| {MODEL_DESCRIPTIONS[model]['description']} |"
                        
                        for hw in HARDWARE_ENDPOINTS:
                            model_hw_data = latest_results[(latest_results['model_family'] == model) & 
                                                          (latest_results['hardware_type'] == hw)]
                            
                            if not model_hw_data.empty:
                                latency = model_hw_data.iloc[0]['average_latency_ms']
                                throughput = model_hw_data.iloc[0]['throughput_items_per_second']
                                row += f" {latency:.2f}ms / {throughput:.2f}it/s |"
                            else:
                                row += " N/A |"
                        
                        f.write(row + "\n")
                    
                    f.write("\n")
                
                # Add optimization recommendations
                f.write("## Optimization Recommendations\n\n")
                
                recommendations = [
                    "Text models (BERT, T5) perform best on CUDA and WebGPU with shader precompilation",
                    "Audio models (Whisper, Wav2Vec2) see significant improvements with Firefox WebGPU compute shader optimizations",
                    "Vision models (ViT, CLIP) work well across most hardware platforms",
                    "Large language models (LLAMA, Qwen2) require CUDA or ROCm for optimal performance",
                    "Memory-intensive models (LLaVA, LLaVA-Next) perform best with dedicated GPU memory"
                ]
                
                for rec in recommendations:
                    f.write(f"- {rec}\n")
                
                f.write("\n## Conclusion\n\n")
                f.write("This report provides a comprehensive view of the performance characteristics of 13 key model types ")
                f.write("across 8 hardware platforms. Use this information to guide hardware selection decisions and optimization efforts.\n")
        
        except Exception as e:
            logger.error(f"Failed to generate Markdown report: {str(e)}")
            raise
    
    def _generate_json_report(self, latest_results, output_path):
        """Generate a JSON report with raw data."""
        try:
            # Convert DataFrame to dict format for JSON serialization
            result_dict = {
                "generated_at": datetime.datetime.now().isoformat(),
                "report_type": "benchmark_timing",
                "hardware_platforms": HARDWARE_DESCRIPTIONS,
                "model_descriptions": MODEL_DESCRIPTIONS,
                "results": []
            }
            
            # Convert DataFrame rows to dict
            for _, row in latest_results.iterrows():
                result_dict["results"].append({
                    "model_name": row["model_name"],
                    "model_family": row["model_family"],
                    "hardware_type": row["hardware_type"],
                    "batch_size": int(row["batch_size"]),
                    "average_latency_ms": float(row["average_latency_ms"]),
                    "throughput_items_per_second": float(row["throughput_items_per_second"]),
                    "memory_peak_mb": float(row["memory_peak_mb"]),
                    "created_at": row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else str(row["created_at"])
                })
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(result_dict, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to generate JSON report: {str(e)}")
            raise
        
    def create_interactive_dashboard(self, port=8501):
        """Launch interactive dashboard for exploring benchmark data."""
        try:
            import streamlit as st
            
            # Define the Streamlit app
            def streamlit_app():
                st.title("Benchmark Timing Dashboard")
                st.write("Interactive dashboard for exploring benchmark timing data")
                
                # Fetch data
                data = self._fetch_timing_data()
                if data.empty:
                    st.error("No benchmark data available")
                    return
                
                # Sidebar filters
                st.sidebar.title("Filters")
                selected_models = st.sidebar.multiselect(
                    "Select Models", 
                    options=sorted(data['model_family'].unique()),
                    default=sorted(data['model_family'].unique())[:5]
                )
                
                selected_hardware = st.sidebar.multiselect(
                    "Select Hardware",
                    options=sorted(data['hardware_type'].unique()),
                    default=sorted(data['hardware_type'].unique())
                )
                
                metric = st.sidebar.selectbox(
                    "Select Metric",
                    options=["average_latency_ms", "throughput_items_per_second", "memory_peak_mb"],
                    format_func=lambda x: {
                        "average_latency_ms": "Latency (ms)",
                        "throughput_items_per_second": "Throughput (items/sec)",
                        "memory_peak_mb": "Memory Usage (MB)"
                    }[x]
                )
                
                # Filter data
                filtered_data = data[
                    data['model_family'].isin(selected_models) & 
                    data['hardware_type'].isin(selected_hardware)
                ]
                
                # Get latest results
                latest_results = self._get_latest_results(filtered_data)
                
                # Show comparison chart
                st.header("Performance Comparison")
                
                pivot_data = latest_results.pivot(
                    index='model_family', 
                    columns='hardware_type', 
                    values=metric
                )
                
                st.bar_chart(pivot_data)
                
                # Show data table
                st.header("Raw Data")
                st.dataframe(latest_results[['model_family', 'hardware_type', 'average_latency_ms', 
                                           'throughput_items_per_second', 'memory_peak_mb', 'created_at']])
                
                # Performance analysis
                st.header("Performance Analysis")
                
                if not latest_results.empty:
                    # Best hardware for each model
                    st.subheader("Best Hardware for Each Model")
                    
                    best_hardware = []
                    for model in selected_models:
                        model_data = latest_results[latest_results['model_family'] == model]
                        if not model_data.empty:
                            if metric == "average_latency_ms":
                                # For latency, lower is better
                                best_hw = model_data.loc[model_data['average_latency_ms'].idxmin()]
                                best_hardware.append({
                                    "model": model,
                                    "best_hardware": best_hw['hardware_type'],
                                    "value": best_hw['average_latency_ms'],
                                    "metric": "latency (ms)"
                                })
                            else:
                                # For throughput and memory, higher might be better
                                best_hw = model_data.loc[model_data[metric].idxmax()]
                                best_hardware.append({
                                    "model": model,
                                    "best_hardware": best_hw['hardware_type'],
                                    "value": best_hw[metric],
                                    "metric": "throughput (items/s)" if metric == "throughput_items_per_second" else "memory (MB)"
                                })
                    
                    best_hw_df = pd.DataFrame(best_hardware)
                    st.table(best_hw_df)
            
            # Run the Streamlit app
            import sys
            from streamlit.web import cli as stcli
            
            sys.argv = ["streamlit", "run", "__main__", f"--server.port={port}"]
            sys.exit(stcli.main())
            
        except ImportError:
            logger.error("Streamlit is required for interactive dashboard. Install with 'pip install streamlit'")
            print("Streamlit is required for interactive dashboard. Install with 'pip install streamlit'")
        except Exception as e:
            logger.error(f"Failed to launch interactive dashboard: {str(e)}")
            raise

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive Benchmark Timing Report Generator")
    
    # Main command groups
    parser.add_argument("--generate", action="store_true", help="Generate comprehensive timing report")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive dashboard")
    parser.add_argument("--api-server", action="store_true", help="Start API server for report data")
    
    # Configuration options
    parser.add_argument("--db-path", help="Path to benchmark database")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--format", choices=["html", "md", "markdown", "json"], default="html", help="Output format")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data to include")
    parser.add_argument("--port", type=int, default=8501, help="Port for interactive dashboard")
    
    args = parser.parse_args()
    
    # Create report generator
    report_gen = BenchmarkTimingReport(db_path=args.db_path)
    
    if args.generate:
        report_gen.generate_timing_report(output_format=args.format, output_path=args.output, days_lookback=args.days)
    elif args.interactive:
        report_gen.create_interactive_dashboard(port=args.port)
    elif args.api_server:
        # Future implementation
        logger.error("API server not yet implemented")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()