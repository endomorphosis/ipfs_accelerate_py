#!/usr/bin/env python3
"""
Test script for the Animated Time Series Visualization Component.

This script demonstrates how to use the animated time series visualization
component to create interactive animations for tracking performance metrics over time.
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
logger = logging.getLogger("test_animated_time_series")

# Add parent directory to path for module imports
parent_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import visualization components
from duckdb_api.visualization.advanced_visualization.base import BaseVisualization, PLOTLY_AVAILABLE, MATPLOTLIB_AVAILABLE
from duckdb_api.visualization.advanced_visualization.viz_animated_time_series import AnimatedTimeSeriesVisualization

# Create output directory
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_sample_data(days=90, time_interval="day"):
    """Generate sample performance data for testing animated time series visualizations."""
    
    # Generate date range
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    
    # Generate date range based on time interval
    if time_interval == "hour":
        freq = "H"
    elif time_interval == "day":
        freq = "D"
    elif time_interval == "week":
        freq = "W"
    else:
        freq = "M"  # month
    
    date_range = pd.date_range(start=start_dt, end=end_dt, freq=freq)
    
    # Hardware types and models
    hardware_types = ["CPU", "GPU", "WebGPU", "WebNN"]
    model_families = ["Text", "Vision", "Audio"]
    
    # Generate sample data
    rows = []
    
    # Progressive trend factors - allow values to change over time with trends
    hw_trend_factors = {
        "CPU": 1.0,
        "GPU": 1.2,
        "WebGPU": 1.15,
        "WebNN": 1.1
    }
    
    family_trend_factors = {
        "Text": 1.0,
        "Vision": 1.05,
        "Audio": 0.95
    }
    
    # Generate data for each date
    for date in date_range:
        # Calculate days from start for trend progression
        days_progression = (date - start_dt).days / max(1, (end_dt - start_dt).days)
        
        for family in model_families:
            for hw in hardware_types:
                # Base throughput value (higher for GPU/WebGPU)
                base_throughput = 100 if hw in ["GPU", "WebGPU"] else 50
                # Add progressive improvement over time (hardware gets faster)
                improvement = 1.0 + days_progression * 0.5 * hw_trend_factors[hw] * family_trend_factors[family]
                throughput = base_throughput * improvement
                # Add noise and some weekly pattern
                day_of_week_factor = 1.0 + 0.05 * (date.weekday() % 7) / 6.0
                throughput *= day_of_week_factor * (1 + np.random.normal(0, 0.05))
                
                # Base latency value (lower for GPU/WebGPU)
                base_latency = 20 if hw in ["GPU", "WebGPU"] else 40
                # Latency improves (decreases) over time
                improvement = 1.0 - days_progression * 0.3 * hw_trend_factors[hw] * family_trend_factors[family]
                latency = base_latency * max(0.5, improvement)
                # Add noise and some weekly pattern
                day_of_week_factor = 1.0 + 0.05 * (date.weekday() % 7) / 6.0
                latency *= day_of_week_factor * (1 + np.random.normal(0, 0.08))
                
                # Memory usage increases slightly over time
                base_memory = 500 if hw in ["GPU", "WebGPU"] else 300
                memory_trend = 1.0 + days_progression * 0.2  # Memory usage tends to grow
                memory = base_memory * memory_trend * (1 + np.random.normal(0, 0.05))
                
                # Power consumption metrics
                base_power = 50 if hw in ["GPU", "WebGPU"] else 30
                power = base_power * (1 + np.random.normal(0, 0.1))
                
                # Add occasional anomalies
                if np.random.random() < 0.02:  # 2% chance of an anomaly
                    if np.random.random() < 0.3:
                        # 30% chance of positive throughput anomaly
                        throughput *= 1.5 + np.random.random() * 0.5  # 1.5-2x increase
                    else:
                        # 70% chance of negative throughput anomaly
                        throughput *= 0.4 + np.random.random() * 0.3  # 40-70% decrease
                    
                    if np.random.random() < 0.7:
                        # 70% chance of negative latency anomaly (higher value)
                        latency *= 1.5 + np.random.random() * 1.0  # 1.5-2.5x increase
                    else:
                        # 30% chance of positive latency anomaly (lower value)
                        latency *= 0.5 + np.random.random() * 0.3  # 50-80% decrease
                
                # Add model-specific data
                if family == "Text":
                    model_names = ["BERT", "T5", "LLAMA"]
                elif family == "Vision":
                    model_names = ["ViT", "CLIP", "DETR"]
                else:
                    model_names = ["Whisper", "CLAP"]
                
                for model_name in model_names:
                    # Add row with base metrics plus model-specific variation
                    row = {
                        "timestamp": date,
                        "model_name": model_name,
                        "model_family": family,
                        "hardware_type": hw,
                        "batch_size": 1,
                        "throughput_items_per_second": throughput * (1 + np.random.normal(0, 0.05)),
                        "average_latency_ms": latency * (1 + np.random.normal(0, 0.05)),
                        "memory_peak_mb": memory * (1 + np.random.normal(0, 0.05)),
                        "power_consumption_w": power * (1 + np.random.normal(0, 0.05)),
                        "efficiency": throughput / (power + 0.1)  # Efficiency as throughput per watt
                    }
                    rows.append(row)
    
    # Convert to DataFrame
    time_series_df = pd.DataFrame(rows)
    
    return time_series_df


def test_animated_time_series(data=None):
    """Test the animated time series visualization component."""
    
    logger.info("Testing Animated Time Series Visualization")
    
    # Create visualization component
    animated_ts_viz = AnimatedTimeSeriesVisualization(debug=True)
    
    # Generate data if not provided
    if data is None:
        data = generate_sample_data(days=90, time_interval="day")
    
    # Create basic animated time series visualization
    output_path = os.path.join(OUTPUT_DIR, "animated_throughput_time_series.html")
    logger.info(f"Creating animated throughput time series: {output_path}")
    animated_ts_viz.create_animated_time_series(
        data=data,
        metric="throughput",
        dimensions=["hardware_type"],
        time_interval="day",
        output_path=output_path,
        title="Throughput Over Time by Hardware Type"
    )
    
    # Create animated time series with trend analysis and anomaly detection
    output_path = os.path.join(OUTPUT_DIR, "animated_latency_with_trends_and_anomalies.html")
    logger.info(f"Creating animated latency time series with trends and anomalies: {output_path}")
    animated_ts_viz.create_animated_time_series(
        data=data,
        metric="latency",
        dimensions=["hardware_type"],
        time_interval="day",
        show_trend=True,
        show_anomalies=True,
        trend_window=7,
        output_path=output_path,
        title="Latency Over Time with Trend Analysis and Anomaly Detection"
    )
    
    # Create animated time series with event markers
    output_path = os.path.join(OUTPUT_DIR, "animated_throughput_with_events.html")
    logger.info(f"Creating animated throughput time series with events: {output_path}")
    animated_ts_viz.create_animated_time_series(
        data=data,
        metric="throughput",
        dimensions=["model_family"],
        time_interval="day",
        events=[
            {"date": (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d"), "label": "v2.0 Release", "color": "green"},
            {"date": (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"), "label": "Config Update", "color": "blue"},
            {"date": (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d"), "label": "Hardware Upgrade", "color": "purple"}
        ],
        output_path=output_path,
        title="Throughput Over Time with Significant Events"
    )
    
    # Create animated time series with multi-dimensional grouping
    output_path = os.path.join(OUTPUT_DIR, "animated_memory_multidimensional.html")
    logger.info(f"Creating animated multi-dimensional memory time series: {output_path}")
    animated_ts_viz.create_animated_time_series(
        data=data,
        metric="memory",
        dimensions=["model_family", "hardware_type"],
        time_interval="week",  # Aggregate by week for clearer trends
        output_path=output_path,
        title="Memory Usage Over Time by Model Family and Hardware Type"
    )
    
    # Create animated time series with different time intervals
    output_path = os.path.join(OUTPUT_DIR, "animated_efficiency_monthly.html")
    logger.info(f"Creating animated efficiency time series with monthly intervals: {output_path}")
    animated_ts_viz.create_animated_time_series(
        data=data,
        metric="efficiency",
        dimensions=["hardware_type"],
        time_interval="month",  # Aggregate by month
        output_path=output_path,
        title="Monthly Efficiency Trends by Hardware Type"
    )
    
    # Create animated time series with progressive display disabled
    output_path = os.path.join(OUTPUT_DIR, "animated_throughput_non_progressive.html")
    logger.info(f"Creating animated throughput time series without progressive display: {output_path}")
    animated_ts_viz.create_animated_time_series(
        data=data,
        metric="throughput",
        dimensions=["hardware_type"],
        time_interval="day",
        progressive_display=False,  # Only show current date's data
        output_path=output_path,
        title="Throughput Over Time (Non-Progressive)"
    )
    
    # Create animated time series with filters
    output_path = os.path.join(OUTPUT_DIR, "animated_throughput_filtered.html")
    logger.info(f"Creating animated throughput time series with filters: {output_path}")
    animated_ts_viz.create_animated_time_series(
        data=data,
        metric="throughput",
        dimensions=["model_name"],
        time_interval="day",
        filters={"hardware_type": ["GPU", "WebGPU"], "model_family": "Text"},
        output_path=output_path,
        title="Throughput Over Time for Text Models on GPU/WebGPU"
    )
    
    logger.info("Animated Time Series Visualization tests completed")


def main():
    """Main function for testing the animated time series visualization component."""
    
    parser = argparse.ArgumentParser(description="Test Animated Time Series Visualization Component")
    parser.add_argument("--days", type=int, default=90, help="Number of days of data to generate")
    parser.add_argument("--interval", type=str, default="day", choices=["hour", "day", "week", "month"], 
                        help="Time interval for aggregation")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for visualizations")
    parser.add_argument("--export-mp4", action="store_true", help="Export an example as MP4")
    parser.add_argument("--export-gif", action="store_true", help="Export an example as GIF")
    
    args = parser.parse_args()
    
    # Update output directory if specified
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
        OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Generate sample data
    data = generate_sample_data(days=args.days, time_interval=args.interval)
    
    # Run tests
    test_animated_time_series(data)
    
    # Export animations if requested
    if args.export_mp4 or args.export_gif:
        animated_ts_viz = AnimatedTimeSeriesVisualization(debug=True)
        animated_ts_viz.create_animated_time_series(
            data=data,
            metric="throughput",
            dimensions=["hardware_type"],
            time_interval=args.interval,
            output_path=os.path.join(OUTPUT_DIR, "animated_throughput_for_export.html"),
            title="Throughput Over Time by Hardware Type"
        )
        
        if args.export_mp4:
            mp4_path = os.path.join(OUTPUT_DIR, "animated_throughput.mp4")
            logger.info(f"Exporting animation as MP4: {mp4_path}")
            animated_ts_viz.export_animation(output_format="mp4", output_path=mp4_path)
        
        if args.export_gif:
            gif_path = os.path.join(OUTPUT_DIR, "animated_throughput.gif")
            logger.info(f"Exporting animation as GIF: {gif_path}")
            animated_ts_viz.export_animation(output_format="gif", output_path=gif_path)
    
    logger.info(f"All tests completed. Results saved in: {OUTPUT_DIR}")
    
    # Print available output files
    # When Plotly is not available, we generate PNG files instead of HTML
    if PLOTLY_AVAILABLE:
        files = list(OUTPUT_DIR.glob("*.html"))
    else:
        files = list(OUTPUT_DIR.glob("*.png"))
        
    if files:
        logger.info("Generated visualization files:")
        for file in files:
            logger.info(f"  - {file.name}")
    else:
        logger.warning("No visualization files were generated.")


if __name__ == "__main__":
    main()