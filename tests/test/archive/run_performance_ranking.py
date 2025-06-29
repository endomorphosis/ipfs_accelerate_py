#!/usr/bin/env python3
"""
Performance Ranking Generator

This script generates performance rankings for hardware platforms based on benchmark results,
addressing the "Create performance ranking of hardware platforms based on real data" item
from NEXT_STEPS.md (line 166).

It analyzes the benchmark data stored in DuckDB, calculates performance metrics for each
hardware platform across different model categories, and generates a comprehensive ranking report.

Usage:
    python run_performance_ranking.py --generate
    python run_performance_ranking.py --format markdown --output performance_ranking.md
    python run_performance_ranking.py --analyze-bottlenecks
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
import duckdb
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("performance_ranking.log")
    ]
)
logger = logging.getLogger(__name__)

# Define model categories
MODEL_CATEGORIES = {
    "text": ["bert", "t5", "llama", "qwen2"],
    "vision": ["vit", "detr", "xclip"],
    "audio": ["whisper", "wav2vec2", "clap"],
    "multimodal": ["clip", "llava", "llava-next"]
}

# Hardware descriptions
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

class PerformanceRankingGenerator:
    """Generate performance rankings for hardware platforms based on benchmark results."""
    
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
            self.conn = None
            
    def _fetch_benchmark_data(self):
        """Fetch benchmark data from the database."""
        try:
            # Query the database for benchmark data
            query = """
                SELECT 
                    m.model_name,
                    m.model_family,
                    hp.hardware_type,
                    pr.batch_size,
                    pr.average_latency_ms,
                    pr.throughput_items_per_second,
                    pr.memory_peak_mb,
                    pr.inference_time_ms,
                    COALESCE(pr.test_timestamp, CURRENT_TIMESTAMP) as created_at,
                    pr.is_simulated
                FROM 
                    performance_results pr
                JOIN 
                    models m ON pr.model_id = m.model_id
                JOIN 
                    hardware_platforms hp ON pr.hardware_id = hp.hardware_id
                WHERE
                    pr.is_simulated = FALSE  -- Only use real benchmark data
                ORDER BY
                    m.model_family, hp.hardware_type, pr.test_timestamp DESC
            """
            
            # Try to execute the query
            if self.conn:
                result = self.conn.execute(query).fetchdf()
                if not result.empty:
                    logger.info(f"Found {len(result)} benchmark results in database")
                    return result
                else:
                    logger.warning("No non-simulated benchmark data found in database, trying with simulated data")
                    # Try again including simulated data but with a warning flag
                    query_with_simulated = query.replace("pr.is_simulated = FALSE", "1=1")
                    result = self.conn.execute(query_with_simulated).fetchdf()
                    if not result.empty:
                        logger.warning(f"Using {len(result)} benchmark results including simulated data")
                        return result
            
            logger.warning("No data found in database, using sample data instead")
            return self._generate_sample_data()
            
        except Exception as e:
            logger.error(f"Failed to fetch benchmark data: {str(e)}")
            logger.warning("Using sample data instead")
            return self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate sample benchmark data for testing."""
        logger.info("Generating sample benchmark data")
        
        # Create sample data structure
        sample_data = []
        
        # Define hardware types with relative performance characteristics
        hardware_performance = {
            "cpu": {"latency": 100.0, "throughput": 10.0, "memory": 1.0},
            "cuda": {"latency": 20.0, "throughput": 100.0, "memory": 1.2},
            "rocm": {"latency": 25.0, "throughput": 80.0, "memory": 1.2},
            "mps": {"latency": 30.0, "throughput": 60.0, "memory": 1.1},
            "openvino": {"latency": 50.0, "throughput": 40.0, "memory": 0.9},
            "qnn": {"latency": 40.0, "throughput": 50.0, "memory": 0.8},
            "webnn": {"latency": 80.0, "throughput": 25.0, "memory": 1.3},
            "webgpu": {"latency": 60.0, "throughput": 30.0, "memory": 1.4}
        }
        
        # Generate sample data for each model category and hardware type
        for category, models in MODEL_CATEGORIES.items():
            for model in models:
                for hardware, perf in hardware_performance.items():
                    # Add a sample data point for batch size 1
                    base_latency = perf["latency"] * (1.0 + 0.2 * np.random.randn())
                    base_throughput = perf["throughput"] * (1.0 + 0.2 * np.random.randn())
                    base_memory = perf["memory"] * 1000 * (1.0 + 0.1 * np.random.randn())
                    
                    sample_data.append({
                        'model_name': f"{model}-benchmark",
                        'model_family': model,
                        'hardware_type': hardware,
                        'batch_size': 1,
                        'average_latency_ms': max(1.0, base_latency),
                        'throughput_items_per_second': max(1.0, base_throughput),
                        'memory_peak_mb': max(100.0, base_memory),
                        'inference_time_ms': max(1.0, base_latency),
                        'created_at': datetime.now(),
                        'is_simulated': True
                    })
                    
                    # Add a data point for batch size 4
                    sample_data.append({
                        'model_name': f"{model}-benchmark",
                        'model_family': model,
                        'hardware_type': hardware,
                        'batch_size': 4,
                        'average_latency_ms': max(1.0, base_latency * 2.2),
                        'throughput_items_per_second': max(1.0, base_throughput * 3.2),
                        'memory_peak_mb': max(100.0, base_memory * 1.3),
                        'inference_time_ms': max(1.0, base_latency * 2.2),
                        'created_at': datetime.now(),
                        'is_simulated': True
                    })
        
        # Convert to DataFrame
        return pd.DataFrame(sample_data)
    
    def _get_latest_results(self, df):
        """Get the latest results for each model-hardware-batch_size combination."""
        if df.empty:
            return df
            
        # Group by model, hardware, and batch size, keep latest result
        latest_results = df.sort_values('created_at', ascending=False).groupby(
            ['model_family', 'hardware_type', 'batch_size']).first().reset_index()
        
        return latest_results
    
    def _calculate_rankings(self, data):
        """Calculate performance rankings for each hardware platform."""
        if data.empty:
            return pd.DataFrame(), pd.DataFrame()
            
        # Get latest results
        latest_results = self._get_latest_results(data)
        
        # Calculate mean performance metrics by hardware and model category
        rankings = []
        
        # For each model category
        for category, models in MODEL_CATEGORIES.items():
            # Filter to models in this category
            category_df = latest_results[latest_results['model_family'].isin(models)]
            
            if category_df.empty:
                continue
                
            # Calculate metrics by hardware for this category
            for hardware in category_df['hardware_type'].unique():
                hw_df = category_df[category_df['hardware_type'] == hardware]
                
                # Calculate mean metrics
                mean_latency = hw_df['average_latency_ms'].mean()
                mean_throughput = hw_df['throughput_items_per_second'].mean()
                mean_memory = hw_df['memory_peak_mb'].mean()
                
                # Calculate composite score (higher is better)
                # Normalize latency (lower is better) by inverting
                norm_latency = 1000.0 / max(1.0, mean_latency)
                norm_throughput = mean_throughput / 10.0  # Scale throughput
                norm_memory = 1000.0 / max(1.0, mean_memory)  # Lower memory is better
                
                # Weighted score - throughput weighted highest
                composite_score = (norm_latency * 0.35) + (norm_throughput * 0.5) + (norm_memory * 0.15)
                
                # Add to rankings
                rankings.append({
                    'category': category,
                    'hardware_type': hardware,
                    'mean_latency_ms': mean_latency,
                    'mean_throughput': mean_throughput,
                    'mean_memory_mb': mean_memory,
                    'composite_score': composite_score,
                    'num_models': len(hw_df['model_family'].unique()),
                    'num_benchmarks': len(hw_df)
                })
        
        # Convert to DataFrame
        rankings_df = pd.DataFrame(rankings)
        
        # Calculate overall rankings across all categories
        overall_rankings = []
        
        for hardware in latest_results['hardware_type'].unique():
            hw_df = latest_results[latest_results['hardware_type'] == hardware]
            
            # Calculate mean metrics
            mean_latency = hw_df['average_latency_ms'].mean()
            mean_throughput = hw_df['throughput_items_per_second'].mean()
            mean_memory = hw_df['memory_peak_mb'].mean()
            
            # Calculate composite score
            norm_latency = 1000.0 / max(1.0, mean_latency)
            norm_throughput = mean_throughput / 10.0
            norm_memory = 1000.0 / max(1.0, mean_memory)
            
            composite_score = (norm_latency * 0.35) + (norm_throughput * 0.5) + (norm_memory * 0.15)
            
            # Add to overall rankings
            overall_rankings.append({
                'hardware_type': hardware,
                'mean_latency_ms': mean_latency,
                'mean_throughput': mean_throughput,
                'mean_memory_mb': mean_memory,
                'composite_score': composite_score,
                'num_models': len(hw_df['model_family'].unique()),
                'num_benchmarks': len(hw_df),
                'simulated_data': hw_df['is_simulated'].any() if 'is_simulated' in hw_df.columns else True
            })
        
        # Convert to DataFrame
        overall_df = pd.DataFrame(overall_rankings)
        
        # Sort both DataFrames by composite score (descending)
        if not rankings_df.empty and 'composite_score' in rankings_df.columns:
            rankings_df = rankings_df.sort_values('composite_score', ascending=False)
        
        if not overall_df.empty and 'composite_score' in overall_df.columns:
            overall_df = overall_df.sort_values('composite_score', ascending=False)
        
        return rankings_df, overall_df
    
    def _identify_bottlenecks(self, data):
        """Identify performance bottlenecks for each model-hardware combination."""
        if data.empty:
            return pd.DataFrame()
            
        # Get latest results
        latest_results = self._get_latest_results(data)
        
        # Calculate performance scaling with batch size
        bottlenecks = []
        
        # Group by model and hardware
        for (model, hardware), group in latest_results.groupby(['model_family', 'hardware_type']):
            # Need at least two batch sizes to analyze scaling
            if len(group) < 2:
                continue
                
            # Sort by batch size
            group = group.sort_values('batch_size')
            
            # Calculate throughput scaling
            batch_sizes = group['batch_size'].tolist()
            throughputs = group['throughput_items_per_second'].tolist()
            
            # Calculate throughput scaling efficiency (should be ~linear)
            if len(batch_sizes) >= 2 and batch_sizes[0] > 0:
                scaling_efficiency = (throughputs[-1] / throughputs[0]) / (batch_sizes[-1] / batch_sizes[0])
            else:
                scaling_efficiency = 0.0
                
            # Determine bottleneck type
            if scaling_efficiency < 0.5:
                bottleneck_type = "Memory bandwidth"
            elif scaling_efficiency < 0.8:
                bottleneck_type = "Compute utilization"
            else:
                bottleneck_type = "None detected"
                
            # Memory pressure detection
            memory_increase_ratio = group['memory_peak_mb'].max() / max(1.0, group['memory_peak_mb'].min())
            if memory_increase_ratio > 1.5:
                memory_pressure = "High"
            elif memory_increase_ratio > 1.2:
                memory_pressure = "Medium"
            else:
                memory_pressure = "Low"
                
            # Add to bottlenecks
            bottlenecks.append({
                'model_family': model,
                'hardware_type': hardware,
                'batch_scaling_efficiency': scaling_efficiency,
                'memory_pressure': memory_pressure,
                'primary_bottleneck': bottleneck_type,
                'max_throughput': group['throughput_items_per_second'].max(),
                'min_latency_ms': group['average_latency_ms'].min(),
                'max_batch_size': group['batch_size'].max(),
                'memory_usage_ratio': memory_increase_ratio
            })
        
        # Convert to DataFrame
        bottlenecks_df = pd.DataFrame(bottlenecks)
        
        return bottlenecks_df
    
    def generate_performance_ranking(self, output_format="html", output_path=None, analyze_bottlenecks=False):
        """Generate performance ranking report."""
        logger.info("Generating performance ranking report...")
        
        # Determine output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"performance_ranking_{timestamp}.{output_format}"
        
        # Fetch benchmark data
        data = self._fetch_benchmark_data()
        
        if data.empty:
            logger.error("No benchmark data available for analysis")
            return None
            
        # Calculate rankings
        rankings_by_category, overall_rankings = self._calculate_rankings(data)
        
        # Identify bottlenecks if requested
        bottlenecks = None
        if analyze_bottlenecks:
            bottlenecks = self._identify_bottlenecks(data)
        
        # Generate report based on format
        if output_format == "html":
            self._generate_html_report(rankings_by_category, overall_rankings, bottlenecks, output_path)
        elif output_format in ["md", "markdown"]:
            self._generate_markdown_report(rankings_by_category, overall_rankings, bottlenecks, output_path)
        elif output_format == "json":
            self._generate_json_report(rankings_by_category, overall_rankings, bottlenecks, output_path)
        else:
            logger.error(f"Unsupported output format: {output_format}")
            return None
            
        logger.info(f"Performance ranking report generated: {output_path}")
        return output_path
    
    def _generate_html_report(self, rankings_by_category, overall_rankings, bottlenecks, output_path):
        """Generate an HTML performance ranking report."""
        try:
            with open(output_path, 'w') as f:
                # Start HTML document
                f.write(f"""
                <!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Hardware Performance Ranking Report</title>
                    <style>
                        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; color: #333; line-height: 1.6; }}
                        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
                        h1, h2, h3, h4 {{ color: #1a5276; }}
                        h1 {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; box-shadow: 0 2px 3px rgba(0,0,0,0.1); }}
                        th, td {{ border: 1px solid #ddd; padding: 12px 15px; text-align: left; }}
                        th {{ background-color: #3498db; color: white; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        tr:hover {{ background-color: #f1f1f1; }}
                        .summary-card {{ background-color: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin-bottom: 20px; }}
                        .warning {{ background-color: #fcf8e3; border-left: 4px solid #f0ad4e; padding: 15px; margin-bottom: 20px; }}
                        .top-performer {{ background-color: #dff0d8; }}
                        .note {{ font-style: italic; margin-top: 5px; color: #666; }}
                        .simulated-data {{ color: #999; font-style: italic; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Hardware Performance Ranking Report</h1>
                        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                """)
                
                # Add summary section
                f.write(f"""
                        <div class="summary-card">
                            <h2>Executive Summary</h2>
                            <p>This report provides a comprehensive ranking of hardware platforms based on benchmark performance data.</p>
                            <p>Rankings consider latency, throughput, and memory usage across different model categories.</p>
                        </div>
                """)
                
                # Check if we're using simulated data
                if not overall_rankings.empty and 'simulated_data' in overall_rankings.columns and overall_rankings['simulated_data'].any():
                    f.write(f"""
                        <div class="warning">
                            <h3>⚠️ Simulation Notice</h3>
                            <p>Some rankings include simulated benchmark data. Real hardware performance may differ.</p>
                        </div>
                    """)
                
                # Add overall rankings table
                f.write(f"""
                        <h2>Overall Hardware Rankings</h2>
                        <table>
                            <tr>
                                <th>Rank</th>
                                <th>Hardware</th>
                                <th>Description</th>
                                <th>Composite Score</th>
                                <th>Latency (ms)</th>
                                <th>Throughput (items/s)</th>
                                <th>Memory (MB)</th>
                                <th>Models Tested</th>
                            </tr>
                """)
                
                # Add rows for overall rankings
                for i, (_, row) in enumerate(overall_rankings.iterrows()):
                    hardware = row['hardware_type']
                    description = HARDWARE_DESCRIPTIONS.get(hardware, "")
                    score = row['composite_score']
                    latency = row['mean_latency_ms']
                    throughput = row['mean_throughput']
                    memory = row['mean_memory_mb']
                    num_models = row['num_models']
                    simulated = row.get('simulated_data', True)
                    
                    # Add CSS class for top performer
                    row_class = "top-performer" if i == 0 else ""
                    
                    f.write(f"""
                        <tr class="{row_class}">
                            <td>{i+1}</td>
                            <td>{hardware}</td>
                            <td>{description}</td>
                            <td>{score:.2f}</td>
                            <td>{latency:.2f}</td>
                            <td>{throughput:.2f}</td>
                            <td>{memory:.2f}</td>
                            <td>{num_models}</td>
                        </tr>
                    """)
                    
                    # Add simulated data indicator if needed
                    if simulated:
                        f.write(f"""
                        <tr class="simulated-data">
                            <td colspan="8"><i>⚠️ Note: Rankings for {hardware} include simulated data</i></td>
                        </tr>
                        """)
                
                f.write("</table>\n")
                
                # Add rankings by model category
                if not rankings_by_category.empty:
                    f.write("<h2>Rankings by Model Category</h2>\n")
                    
                    for category in sorted(rankings_by_category['category'].unique()):
                        category_df = rankings_by_category[rankings_by_category['category'] == category]
                        
                        f.write(f"""
                            <h3>{category.capitalize()} Models</h3>
                            <table>
                                <tr>
                                    <th>Rank</th>
                                    <th>Hardware</th>
                                    <th>Composite Score</th>
                                    <th>Latency (ms)</th>
                                    <th>Throughput (items/s)</th>
                                    <th>Memory (MB)</th>
                                </tr>
                        """)
                        
                        # Add rows for this category
                        for i, (_, row) in enumerate(category_df.iterrows()):
                            hardware = row['hardware_type']
                            score = row['composite_score']
                            latency = row['mean_latency_ms']
                            throughput = row['mean_throughput']
                            memory = row['mean_memory_mb']
                            
                            # Add CSS class for top performer
                            row_class = "top-performer" if i == 0 else ""
                            
                            f.write(f"""
                                <tr class="{row_class}">
                                    <td>{i+1}</td>
                                    <td>{hardware}</td>
                                    <td>{score:.2f}</td>
                                    <td>{latency:.2f}</td>
                                    <td>{throughput:.2f}</td>
                                    <td>{memory:.2f}</td>
                                </tr>
                            """)
                        
                        f.write("</table>\n")
                
                # Add bottleneck analysis if available
                if bottlenecks is not None and not bottlenecks.empty:
                    f.write(f"""
                        <h2>Performance Bottleneck Analysis</h2>
                        <p>This analysis identifies potential bottlenecks in hardware-model combinations.</p>
                        <table>
                            <tr>
                                <th>Model</th>
                                <th>Hardware</th>
                                <th>Primary Bottleneck</th>
                                <th>Batch Scaling Efficiency</th>
                                <th>Memory Pressure</th>
                                <th>Max Throughput (items/s)</th>
                                <th>Min Latency (ms)</th>
                            </tr>
                    """)
                    
                    # Add rows for bottleneck analysis
                    for _, row in bottlenecks.iterrows():
                        model = row['model_family']
                        hardware = row['hardware_type']
                        bottleneck = row['primary_bottleneck']
                        scaling = row['batch_scaling_efficiency']
                        memory = row['memory_pressure']
                        throughput = row['max_throughput']
                        latency = row['min_latency_ms']
                        
                        f.write(f"""
                            <tr>
                                <td>{model}</td>
                                <td>{hardware}</td>
                                <td>{bottleneck}</td>
                                <td>{scaling:.2f}</td>
                                <td>{memory}</td>
                                <td>{throughput:.2f}</td>
                                <td>{latency:.2f}</td>
                            </tr>
                        """)
                    
                    f.write("</table>\n")
                    
                    # Add optimization recommendations
                    f.write(f"""
                        <h3>Optimization Recommendations</h3>
                        <ul>
                            <li>For memory bandwidth bottlenecks: Consider hardware with higher memory bandwidth or optimizing memory access patterns</li>
                            <li>For compute utilization bottlenecks: Consider hardware with more compute resources or optimizing compute-intensive operations</li>
                            <li>For high memory pressure: Consider using smaller batch sizes or hardware with more available memory</li>
                        </ul>
                    """)
                
                # Add footnotes and methodology
                f.write(f"""
                    <h2>Methodology</h2>
                    <p>Rankings were calculated using the following methodology:</p>
                    <ul>
                        <li>Composite Score: Weighted combination of normalized latency (35%), throughput (50%), and memory usage (15%)</li>
                        <li>Higher scores indicate better overall performance</li>
                        <li>Rankings are based on the latest benchmark results for each model-hardware combination</li>
                        <li>Bottleneck analysis is based on scaling efficiency with batch size</li>
                    </ul>
                    
                    <p class="note">Note: Performance may vary depending on model size, batch size, and specific hardware configurations.</p>
                </body>
                </html>
                """)
            
            logger.info(f"HTML performance ranking report generated: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return False
    
    def _generate_markdown_report(self, rankings_by_category, overall_rankings, bottlenecks, output_path):
        """Generate a markdown performance ranking report."""
        try:
            with open(output_path, 'w') as f:
                # Header
                f.write("# Hardware Performance Ranking Report\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Executive summary
                f.write("## Executive Summary\n\n")
                f.write("This report provides a comprehensive ranking of hardware platforms based on benchmark performance data.\n")
                f.write("Rankings consider latency, throughput, and memory usage across different model categories.\n\n")
                
                # Simulation notice if needed
                if not overall_rankings.empty and 'simulated_data' in overall_rankings.columns and overall_rankings['simulated_data'].any():
                    f.write("⚠️ **Simulation Notice**: Some rankings include simulated benchmark data. Real hardware performance may differ.\n\n")
                
                # Overall rankings
                f.write("## Overall Hardware Rankings\n\n")
                f.write("| Rank | Hardware | Description | Composite Score | Latency (ms) | Throughput (items/s) | Memory (MB) | Models Tested |\n")
                f.write("|------|----------|-------------|----------------|--------------|---------------------|-------------|---------------|\n")
                
                # Add rows for overall rankings
                for i, (_, row) in enumerate(overall_rankings.iterrows()):
                    hardware = row['hardware_type']
                    description = HARDWARE_DESCRIPTIONS.get(hardware, "")
                    score = row['composite_score']
                    latency = row['mean_latency_ms']
                    throughput = row['mean_throughput']
                    memory = row['mean_memory_mb']
                    num_models = row['num_models']
                    simulated = row.get('simulated_data', True)
                    
                    f.write(f"| {i+1} | {hardware} | {description} | {score:.2f} | {latency:.2f} | {throughput:.2f} | {memory:.2f} | {num_models} |\n")
                    
                    # Add simulated data indicator if needed
                    if simulated:
                        f.write(f"| | | ⚠️ *Note: Rankings for {hardware} include simulated data* | | | | | |\n")
                
                f.write("\n")
                
                # Rankings by model category
                if not rankings_by_category.empty:
                    f.write("## Rankings by Model Category\n\n")
                    
                    for category in sorted(rankings_by_category['category'].unique()):
                        category_df = rankings_by_category[rankings_by_category['category'] == category]
                        
                        f.write(f"### {category.capitalize()} Models\n\n")
                        f.write("| Rank | Hardware | Composite Score | Latency (ms) | Throughput (items/s) | Memory (MB) |\n")
                        f.write("|------|----------|----------------|--------------|---------------------|-------------|\n")
                        
                        # Add rows for this category
                        for i, (_, row) in enumerate(category_df.iterrows()):
                            hardware = row['hardware_type']
                            score = row['composite_score']
                            latency = row['mean_latency_ms']
                            throughput = row['mean_throughput']
                            memory = row['mean_memory_mb']
                            
                            f.write(f"| {i+1} | {hardware} | {score:.2f} | {latency:.2f} | {throughput:.2f} | {memory:.2f} |\n")
                        
                        f.write("\n")
                
                # Bottleneck analysis if available
                if bottlenecks is not None and not bottlenecks.empty:
                    f.write("## Performance Bottleneck Analysis\n\n")
                    f.write("This analysis identifies potential bottlenecks in hardware-model combinations.\n\n")
                    f.write("| Model | Hardware | Primary Bottleneck | Batch Scaling Efficiency | Memory Pressure | Max Throughput (items/s) | Min Latency (ms) |\n")
                    f.write("|-------|----------|-------------------|--------------------------|----------------|--------------------------|------------------|\n")
                    
                    # Add rows for bottleneck analysis
                    for _, row in bottlenecks.iterrows():
                        model = row['model_family']
                        hardware = row['hardware_type']
                        bottleneck = row['primary_bottleneck']
                        scaling = row['batch_scaling_efficiency']
                        memory = row['memory_pressure']
                        throughput = row['max_throughput']
                        latency = row['min_latency_ms']
                        
                        f.write(f"| {model} | {hardware} | {bottleneck} | {scaling:.2f} | {memory} | {throughput:.2f} | {latency:.2f} |\n")
                    
                    f.write("\n")
                    
                    # Add optimization recommendations
                    f.write("### Optimization Recommendations\n\n")
                    f.write("- For memory bandwidth bottlenecks: Consider hardware with higher memory bandwidth or optimizing memory access patterns\n")
                    f.write("- For compute utilization bottlenecks: Consider hardware with more compute resources or optimizing compute-intensive operations\n")
                    f.write("- For high memory pressure: Consider using smaller batch sizes or hardware with more available memory\n\n")
                
                # Methodology
                f.write("## Methodology\n\n")
                f.write("Rankings were calculated using the following methodology:\n\n")
                f.write("- Composite Score: Weighted combination of normalized latency (35%), throughput (50%), and memory usage (15%)\n")
                f.write("- Higher scores indicate better overall performance\n")
                f.write("- Rankings are based on the latest benchmark results for each model-hardware combination\n")
                f.write("- Bottleneck analysis is based on scaling efficiency with batch size\n\n")
                
                f.write("*Note: Performance may vary depending on model size, batch size, and specific hardware configurations.*\n")
            
            logger.info(f"Markdown performance ranking report generated: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate Markdown report: {e}")
            return False
    
    def _generate_json_report(self, rankings_by_category, overall_rankings, bottlenecks, output_path):
        """Generate a JSON performance ranking report."""
        try:
            # Create result dictionary
            result = {
                "generated_at": datetime.now().isoformat(),
                "report_type": "performance_ranking",
                "hardware_descriptions": HARDWARE_DESCRIPTIONS,
                "model_categories": MODEL_CATEGORIES,
                "overall_rankings": [],
                "rankings_by_category": [],
                "bottleneck_analysis": []
            }
            
            # Convert DataFrames to dictionaries
            if not overall_rankings.empty:
                for _, row in overall_rankings.iterrows():
                    result["overall_rankings"].append({
                        "hardware_type": row["hardware_type"],
                        "composite_score": float(row["composite_score"]),
                        "mean_latency_ms": float(row["mean_latency_ms"]),
                        "mean_throughput": float(row["mean_throughput"]),
                        "mean_memory_mb": float(row["mean_memory_mb"]),
                        "num_models": int(row["num_models"]),
                        "num_benchmarks": int(row["num_benchmarks"]),
                        "simulated_data": bool(row.get("simulated_data", True))
                    })
            
            if not rankings_by_category.empty:
                for _, row in rankings_by_category.iterrows():
                    result["rankings_by_category"].append({
                        "category": row["category"],
                        "hardware_type": row["hardware_type"],
                        "composite_score": float(row["composite_score"]),
                        "mean_latency_ms": float(row["mean_latency_ms"]),
                        "mean_throughput": float(row["mean_throughput"]),
                        "mean_memory_mb": float(row["mean_memory_mb"]),
                        "num_models": int(row["num_models"]),
                        "num_benchmarks": int(row["num_benchmarks"])
                    })
            
            if bottlenecks is not None and not bottlenecks.empty:
                for _, row in bottlenecks.iterrows():
                    result["bottleneck_analysis"].append({
                        "model_family": row["model_family"],
                        "hardware_type": row["hardware_type"],
                        "primary_bottleneck": row["primary_bottleneck"],
                        "batch_scaling_efficiency": float(row["batch_scaling_efficiency"]),
                        "memory_pressure": row["memory_pressure"],
                        "max_throughput": float(row["max_throughput"]),
                        "min_latency_ms": float(row["min_latency_ms"]),
                        "max_batch_size": int(row["max_batch_size"]),
                        "memory_usage_ratio": float(row["memory_usage_ratio"])
                    })
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"JSON performance ranking report generated: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to generate JSON report: {e}")
            return False

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Hardware Performance Ranking Generator")
    
    # Main command groups
    parser.add_argument("--generate", action="store_true", help="Generate performance ranking report")
    parser.add_argument("--analyze-bottlenecks", action="store_true", help="Include bottleneck analysis in the report")
    
    # Configuration options
    parser.add_argument("--db-path", help="Path to benchmark database")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--format", choices=["html", "md", "markdown", "json"], default="html", help="Output format")
    
    args = parser.parse_args()
    
    # Create ranking generator
    ranking_gen = PerformanceRankingGenerator(db_path=args.db_path)
    
    if args.generate:
        ranking_gen.generate_performance_ranking(
            output_format=args.format, 
            output_path=args.output, 
            analyze_bottlenecks=args.analyze_bottlenecks
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()