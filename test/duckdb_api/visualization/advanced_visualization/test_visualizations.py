#!/usr/bin/env python3
"""
Test script for the Advanced Visualization System.

This script demonstrates how to use the visualization components
to create different types of visualizations.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("test_visualizations")

# Add parent directory to path for module imports
parent_dir = str(Path(__file__).resolve().parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import visualization components
from duckdb_api.visualization.advanced_visualization.base import BaseVisualization, PLOTLY_AVAILABLE, MATPLOTLIB_AVAILABLE
from duckdb_api.visualization.advanced_visualization.viz_heatmap import HardwareHeatmapVisualization
from duckdb_api.visualization.advanced_visualization.viz_time_series import TimeSeriesVisualization

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_sample_data():
    """Generate sample performance data for testing visualizations."""
    
    # Hardware types and models
    hardware_types = ["CPU", "GPU", "WebGPU", "WebNN", "MPS"]
    model_families = {
        "Text": ["BERT", "LLAMA", "T5"],
        "Vision": ["ViT", "CLIP"],
        "Audio": ["Whisper"]
    }
    
    # Metric values
    throughput_base = {
        "CPU": 50,
        "GPU": 200,
        "WebGPU": 180,
        "WebNN": 150,
        "MPS": 120
    }
    
    latency_base = {
        "CPU": 100,
        "GPU": 20,
        "WebGPU": 30,
        "WebNN": 40,
        "MPS": 50
    }
    
    memory_base = {
        "CPU": 500,
        "GPU": 1200,
        "WebGPU": 1000,
        "WebNN": 800,
        "MPS": 900
    }
    
    # Generate hardware comparison data
    rows = []
    for family, models in model_families.items():
        for model in models:
            for hw in hardware_types:
                # Generate metric values with some randomness
                throughput = throughput_base[hw] * (1.0 + np.random.normal(0, 0.15))
                latency = latency_base[hw] * (1.0 + np.random.normal(0, 0.15))
                memory = memory_base[hw] * (1.0 + np.random.normal(0, 0.1))
                
                # Add row
                rows.append({
                    "model_name": model,
                    "model_family": family,
                    "hardware_type": hw,
                    "batch_size": 1,
                    "precision": "fp32",
                    "throughput_items_per_second": throughput,
                    "average_latency_ms": latency,
                    "memory_peak_mb": memory,
                    "is_simulated": np.random.random() > 0.7  # Random simulation flag
                })
    
    # Convert to DataFrame
    hardware_df = pd.DataFrame(rows)
    
    # Generate time series data
    time_rows = []
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=90)
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
    
    for date in date_range:
        for hw in ["CPU", "GPU", "WebGPU"]:
            # Base value depends on hardware
            throughput_base_value = 100 if hw == "GPU" else 50
            latency_base_value = 20 if hw == "GPU" else 40
            
            # Add trend over time
            days_since_start = (date - start_dt).days
            throughput_trend = 1.0 + days_since_start / 180.0
            latency_trend = 1.0 - days_since_start / 400.0
            
            # Calculate values with trends and noise
            throughput = throughput_base_value * throughput_trend * (1 + np.random.normal(0, 0.1))
            latency = latency_base_value * latency_trend * (1 + np.random.normal(0, 0.15))
            memory = 500 + np.random.normal(0, 50)
            
            # Add some outliers
            if np.random.random() < 0.03:  # 3% chance of outlier
                throughput *= 1.5 if np.random.random() < 0.5 else 0.5
                latency *= 1.5 if np.random.random() < 0.5 else 0.5
            
            time_rows.append({
                "timestamp": date,
                "model_name": "BERT",
                "hardware_type": hw,
                "batch_size": 1,
                "throughput_items_per_second": throughput,
                "average_latency_ms": latency,
                "memory_peak_mb": memory,
            })
    
    # Convert to DataFrame
    time_series_df = pd.DataFrame(time_rows)
    
    return hardware_df, time_series_df


def test_hardware_heatmap(data=None):
    """Test the hardware heatmap visualization component."""
    
    logger.info("Testing Hardware Heatmap Visualization")
    
    # Create visualization component
    heatmap_viz = HardwareHeatmapVisualization(debug=True)
    
    # Generate data if not provided
    if data is None:
        data, _ = generate_sample_data()
    
    # Create throughput heatmap
    output_path = os.path.join(OUTPUT_DIR, "hardware_throughput_heatmap.html")
    logger.info(f"Creating throughput heatmap: {output_path}")
    heatmap_viz.create_hardware_heatmap(
        data=data,
        metric="throughput",
        output_path=output_path,
        title="Hardware Throughput Comparison by Model Family"
    )
    
    # Create latency heatmap
    output_path = os.path.join(OUTPUT_DIR, "hardware_latency_heatmap.html")
    logger.info(f"Creating latency heatmap: {output_path}")
    heatmap_viz.create_hardware_heatmap(
        data=data,
        metric="latency",
        output_path=output_path,
        title="Hardware Latency Comparison by Model Family"
    )
    
    # Create memory heatmap
    output_path = os.path.join(OUTPUT_DIR, "hardware_memory_heatmap.html")
    logger.info(f"Creating memory heatmap: {output_path}")
    heatmap_viz.create_hardware_heatmap(
        data=data,
        metric="memory",
        output_path=output_path,
        title="Hardware Memory Usage Comparison by Model Family"
    )
    
    # Create heatmap with different configuration options
    output_path = os.path.join(OUTPUT_DIR, "hardware_custom_heatmap.html")
    logger.info(f"Creating custom heatmap: {output_path}")
    heatmap_viz.create_hardware_heatmap(
        data=data,
        metric="throughput",
        hardware_types=["CPU", "GPU", "WebGPU"],
        model_families=["Text", "Vision"],
        output_path=output_path,
        title="Custom Hardware Comparison (Text & Vision Models Only)",
        show_values=True,
        normalize_by_column=True,
        mark_simulated=True
    )
    
    logger.info("Hardware Heatmap Visualization tests completed")


def test_time_series(data=None):
    """Test the time series visualization component."""
    
    logger.info("Testing Time Series Visualization")
    
    # Create visualization component
    time_series_viz = TimeSeriesVisualization(debug=True)
    
    # Generate data if not provided
    if data is None:
        _, data = generate_sample_data()
    
    # Create throughput time series
    output_path = os.path.join(OUTPUT_DIR, "throughput_time_series.html")
    logger.info(f"Creating throughput time series: {output_path}")
    time_series_viz.create_time_series(
        data=data,
        metric="throughput",
        model_name="BERT",
        group_by="hardware_type",
        output_path=output_path,
        title="BERT Throughput Over Time by Hardware Type"
    )
    
    # Create latency time series
    output_path = os.path.join(OUTPUT_DIR, "latency_time_series.html")
    logger.info(f"Creating latency time series: {output_path}")
    time_series_viz.create_time_series(
        data=data,
        metric="latency",
        model_name="BERT",
        group_by="hardware_type",
        output_path=output_path,
        title="BERT Latency Over Time by Hardware Type"
    )
    
    # Create time series for a single hardware type
    output_path = os.path.join(OUTPUT_DIR, "gpu_throughput_time_series.html")
    logger.info(f"Creating GPU throughput time series: {output_path}")
    time_series_viz.create_time_series(
        data=data,
        metric="throughput",
        model_name="BERT",
        hardware_type="GPU",
        output_path=output_path,
        title="BERT Throughput Over Time on GPU",
        show_trend=True,
        show_anomalies=True,
        include_regression=True,
        trend_window=7
    )
    
    # Create time series with different time range
    # Select last 30 days
    end_date = data["timestamp"].max().strftime("%Y-%m-%d")
    start_date = (data["timestamp"].max() - timedelta(days=30)).strftime("%Y-%m-%d")
    output_path = os.path.join(OUTPUT_DIR, "recent_throughput_time_series.html")
    logger.info(f"Creating recent throughput time series: {output_path}")
    time_series_viz.create_time_series(
        data=data,
        metric="throughput",
        model_name="BERT",
        group_by="hardware_type",
        start_date=start_date,
        end_date=end_date,
        output_path=output_path,
        title="Recent BERT Throughput (Last 30 Days)"
    )
    
    logger.info("Time Series Visualization tests completed")


def main():
    """Main function for testing visualizations."""
    
    parser = argparse.ArgumentParser(description="Test Advanced Visualization System")
    parser.add_argument("--heatmap", action="store_true", help="Test hardware heatmap visualizations")
    parser.add_argument("--time-series", action="store_true", help="Test time series visualizations")
    parser.add_argument("--all", action="store_true", help="Test all visualization types")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Update output directory if specified
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Generate sample data
    hardware_df, time_series_df = generate_sample_data()
    
    # Run tests based on arguments
    if args.all or (not args.heatmap and not args.time_series):
        # Run all tests
        test_hardware_heatmap(hardware_df)
        test_time_series(time_series_df)
    else:
        # Run specific tests
        if args.heatmap:
            test_hardware_heatmap(hardware_df)
        if args.time_series:
            test_time_series(time_series_df)
    
    logger.info(f"All visualization tests completed. Results saved in: {OUTPUT_DIR}")
    
    # Print available output files
    files = list(OUTPUT_DIR.glob("*.html"))
    if files:
        logger.info("Generated visualization files:")
        for file in files:
            logger.info(f"  - {file.name}")
    else:
        logger.warning("No visualization files were generated.")


if __name__ == "__main__":
    main()