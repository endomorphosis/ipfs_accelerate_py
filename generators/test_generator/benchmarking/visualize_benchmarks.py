#!/usr/bin/env python3
"""
Visualize benchmark results for IPFS Accelerate Python framework.

This script generates visualizations from benchmark results
stored in the DuckDB database or from JSON result files.
"""

import os
import sys
import json
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.warning("DuckDB not available. Database visualizations will not work.")

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib/Pandas not available. Visualizations will be limited to text output.")

# Hardware-specific colors for consistent visualization
HARDWARE_COLORS = {
    "cpu": "#1f77b4",    # Blue
    "cuda": "#ff7f0e",   # Orange 
    "rocm": "#2ca02c",   # Green
    "mps": "#d62728",    # Red
    "openvino": "#9467bd", # Purple
    "qnn": "#8c564b"     # Brown
}

# Architecture-specific markers for consistent visualization
ARCHITECTURE_MARKERS = {
    "encoder-only": "o",      # Circle
    "decoder-only": "s",      # Square
    "encoder-decoder": "^",   # Triangle up
    "vision": "D",            # Diamond
    "vision-encoder-text-decoder": "v", # Triangle down
    "speech": "P",            # Plus (filled)
    "multimodal": "*",        # Star
    "diffusion": "X",         # X (filled)
    "mixture-of-experts": "h", # Hexagon
    "state-space": "p",       # Pentagon
    "rag": "H"                # Hexagon 2
}


class BenchmarkVisualizer:
    """Benchmark visualization class."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            db_path: Path to benchmark database
        """
        self.db_path = db_path
        self.db_connection = None
        
        # Connect to database if path provided and DuckDB is available
        if db_path and DUCKDB_AVAILABLE:
            try:
                self.db_connection = duckdb.connect(db_path)
                logger.info(f"Connected to database at {db_path}")
            except Exception as e:
                logger.error(f"Error connecting to database: {e}")
                self.db_connection = None
    
    def load_from_json(self, json_path: str) -> Dict[str, Any]:
        """
        Load benchmark results from JSON file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Benchmark results dictionary
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            logger.info(f"Loaded benchmark results from {json_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            return {}
            
    def load_from_db(self, query: str) -> pd.DataFrame:
        """
        Load benchmark results from database.
        
        Args:
            query: SQL query to execute
            
        Returns:
            DataFrame with query results
        """
        if not DUCKDB_AVAILABLE:
            logger.error("DuckDB not available. Cannot load from database.")
            return pd.DataFrame()
            
        if not self.db_connection:
            logger.error("No database connection. Provide db_path in initialization.")
            return pd.DataFrame()
            
        try:
            # Execute query and convert to DataFrame
            result = self.db_connection.execute(query).fetchdf()
            return result
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def visualize_json_result(self, data: Dict[str, Any], output_dir: str = "benchmark_visualizations"):
        """
        Visualize benchmark results from a JSON result.
        
        Args:
            data: Benchmark results dictionary
            output_dir: Directory to save visualizations
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Pandas not available. Generating text summary only.")
            self._text_summary(data)
            return
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract model and device info
        model_id = data.get("model_id", "unknown_model")
        device = data.get("device", "unknown_device")
        architecture_type = data.get("architecture_type", "unknown_architecture")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a safe filename base
        model_name = model_id.replace("/", "_").replace(":", "_")
        base_filename = f"{model_name}_{device}_{timestamp}"
        
        # Create DataFrame for batch results
        batch_results = []
        
        for batch_key, result in data.get("batch_results", {}).items():
            if not result.get("success", False):
                continue
                
            # Parse batch key to get batch size and sequence length
            # Format: b{batch_size}_s{seq_len}
            parts = batch_key.split("_")
            batch_size = int(parts[0][1:])  # Remove 'b' prefix
            seq_len = int(parts[1][1:])     # Remove 's' prefix
            
            batch_results.append({
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "latency_ms": result.get("latency_mean_ms", 0),
                "throughput": result.get("throughput_samples_per_sec", 0),
                "memory_mb": result.get("memory_usage_mb", 0)
            })
        
        # If no successful results, return
        if not batch_results:
            logger.warning("No successful benchmark results to visualize")
            return
            
        # Convert to DataFrame
        df = pd.DataFrame(batch_results)
        
        # 1. Plot latency vs batch size
        plt.figure(figsize=(10, 6))
        for seq_len in df["sequence_length"].unique():
            subset = df[df["sequence_length"] == seq_len]
            plt.plot(subset["batch_size"], subset["latency_ms"], 
                    marker='o', label=f"Sequence Length: {seq_len}")
            
        plt.xlabel("Batch Size")
        plt.ylabel("Latency (ms)")
        plt.title(f"Latency vs Batch Size for {model_id} on {device}")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot
        latency_path = os.path.join(output_dir, f"{base_filename}_latency.png")
        plt.savefig(latency_path)
        logger.info(f"Latency plot saved to {latency_path}")
        
        # 2. Plot throughput vs batch size
        plt.figure(figsize=(10, 6))
        for seq_len in df["sequence_length"].unique():
            subset = df[df["sequence_length"] == seq_len]
            plt.plot(subset["batch_size"], subset["throughput"], 
                    marker='o', label=f"Sequence Length: {seq_len}")
            
        plt.xlabel("Batch Size")
        plt.ylabel("Throughput (samples/sec)")
        plt.title(f"Throughput vs Batch Size for {model_id} on {device}")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot
        throughput_path = os.path.join(output_dir, f"{base_filename}_throughput.png")
        plt.savefig(throughput_path)
        logger.info(f"Throughput plot saved to {throughput_path}")
        
        # 3. Plot memory usage vs batch size
        plt.figure(figsize=(10, 6))
        for seq_len in df["sequence_length"].unique():
            subset = df[df["sequence_length"] == seq_len]
            plt.plot(subset["batch_size"], subset["memory_mb"], 
                    marker='o', label=f"Sequence Length: {seq_len}")
            
        plt.xlabel("Batch Size")
        plt.ylabel("Memory Usage (MB)")
        plt.title(f"Memory Usage vs Batch Size for {model_id} on {device}")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save plot
        memory_path = os.path.join(output_dir, f"{base_filename}_memory.png")
        plt.savefig(memory_path)
        logger.info(f"Memory usage plot saved to {memory_path}")
        
        # Generate text summary
        summary_path = os.path.join(output_dir, f"{base_filename}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Benchmark Summary for {model_id} on {device} ({architecture_type})\n")
            f.write("="*80 + "\n\n")
            
            # Load time and memory
            if "load_results" in data and data["load_results"]["success"]:
                f.write(f"Load Time: {data['load_results']['load_time_seconds']:.2f} seconds\n")
                f.write(f"Load Memory: {data['load_results']['memory_usage_mb']:.2f} MB\n\n")
            
            # Summary metrics
            if "summary" in data:
                f.write("Summary Metrics:\n")
                f.write(f"- Average Latency: {data['summary'].get('average_latency_ms', 0):.2f} ms\n")
                f.write(f"- Best Throughput: {data['summary'].get('best_throughput_samples_per_sec', 0):.2f} samples/sec\n")
                f.write(f"  (Config: {data['summary'].get('best_throughput_config', 'N/A')})\n")
                f.write(f"- Best Latency: {data['summary'].get('best_latency_ms', 0):.2f} ms\n")
                f.write(f"  (Config: {data['summary'].get('best_latency_config', 'N/A')})\n\n")
            
            # Detailed results
            f.write("Detailed Results:\n")
            f.write(f"{'Batch Size':<10} {'Seq Length':<10} {'Latency (ms)':<15} {'Throughput (samples/s)':<25} {'Memory (MB)':<15}\n")
            f.write("-"*80 + "\n")
            
            for _, row in df.sort_values(["sequence_length", "batch_size"]).iterrows():
                f.write(f"{row['batch_size']:<10} {row['sequence_length']:<10} {row['latency_ms']:<15.2f} {row['throughput']:<25.2f} {row['memory_mb']:<15.2f}\n")
            
        logger.info(f"Summary saved to {summary_path}")
    
    def visualize_hardware_comparison(self, model_id: str, output_dir: str = "benchmark_visualizations"):
        """
        Visualize hardware comparison for a specific model.
        
        Args:
            model_id: Model ID to visualize
            output_dir: Directory to save visualizations
        """
        if not DUCKDB_AVAILABLE:
            logger.error("DuckDB not available. Cannot generate hardware comparison.")
            return
            
        if not self.db_connection:
            logger.error("No database connection. Provide db_path in initialization.")
            return
            
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Pandas not available. Generating text summary only.")
            return
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Query for model results across hardware
            query = f"""
            SELECT 
                r.model_id, r.device, r.precision, r.architecture_type,
                b.batch_size, b.sequence_length, 
                b.latency_mean_ms, b.throughput_samples_per_sec, b.memory_usage_mb
            FROM 
                benchmark_runs r
            JOIN 
                benchmark_results b ON r.id = b.run_id
            WHERE 
                r.model_id = '{model_id}'
            ORDER BY 
                r.device, b.batch_size, b.sequence_length
            """
            
            df = self.load_from_db(query)
            
            if df.empty:
                logger.warning(f"No benchmark results found for model {model_id}")
                return
                
            # Create a safe filename base
            model_name = model_id.replace("/", "_").replace(":", "_")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{model_name}_hardware_comparison_{timestamp}"
            
            # Create plots for different metrics
            
            # 1. Latency comparison across hardware (batch size = 1)
            plt.figure(figsize=(12, 6))
            
            for device in df["device"].unique():
                subset = df[(df["device"] == device) & (df["batch_size"] == 1)]
                if not subset.empty:
                    plt.plot(subset["sequence_length"], subset["latency_mean_ms"], 
                            marker='o', label=device, 
                            color=HARDWARE_COLORS.get(device, None))
            
            plt.xlabel("Sequence Length")
            plt.ylabel("Latency (ms)")
            plt.title(f"Hardware Latency Comparison for {model_id} (batch size=1)")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Save plot
            latency_path = os.path.join(output_dir, f"{base_filename}_latency_comparison.png")
            plt.savefig(latency_path)
            logger.info(f"Latency comparison plot saved to {latency_path}")
            
            # 2. Throughput comparison across hardware
            plt.figure(figsize=(12, 6))
            
            for device in df["device"].unique():
                subset = df[(df["device"] == device) & (df["batch_size"] == 1)]
                if not subset.empty:
                    plt.plot(subset["sequence_length"], subset["throughput_samples_per_sec"], 
                            marker='o', label=device,
                            color=HARDWARE_COLORS.get(device, None))
            
            plt.xlabel("Sequence Length")
            plt.ylabel("Throughput (samples/sec)")
            plt.title(f"Hardware Throughput Comparison for {model_id} (batch size=1)")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Save plot
            throughput_path = os.path.join(output_dir, f"{base_filename}_throughput_comparison.png")
            plt.savefig(throughput_path)
            logger.info(f"Throughput comparison plot saved to {throughput_path}")
            
            # 3. Memory usage comparison across hardware
            plt.figure(figsize=(12, 6))
            
            for device in df["device"].unique():
                subset = df[(df["device"] == device) & (df["batch_size"] == 1)]
                if not subset.empty:
                    plt.plot(subset["sequence_length"], subset["memory_usage_mb"], 
                            marker='o', label=device,
                            color=HARDWARE_COLORS.get(device, None))
            
            plt.xlabel("Sequence Length")
            plt.ylabel("Memory Usage (MB)")
            plt.title(f"Hardware Memory Usage Comparison for {model_id} (batch size=1)")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Save plot
            memory_path = os.path.join(output_dir, f"{base_filename}_memory_comparison.png")
            plt.savefig(memory_path)
            logger.info(f"Memory usage comparison plot saved to {memory_path}")
            
            # 4. Bar chart for batch=1, seq_len=128 comparison
            plt.figure(figsize=(12, 6))
            
            compare_df = df[(df["batch_size"] == 1) & (df["sequence_length"] == 128)]
            if not compare_df.empty:
                devices = compare_df["device"].tolist()
                latencies = compare_df["latency_mean_ms"].tolist()
                throughputs = compare_df["throughput_samples_per_sec"].tolist()
                
                x = np.arange(len(devices))
                width = 0.4
                
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                # Latency bars
                bars1 = ax1.bar(x - width/2, latencies, width, label='Latency (ms)',
                               color=[HARDWARE_COLORS.get(d, "#333333") for d in devices])
                ax1.set_xlabel('Hardware')
                ax1.set_ylabel('Latency (ms)')
                ax1.set_xticks(x)
                ax1.set_xticklabels(devices)
                
                # Throughput line (secondary y-axis)
                ax2 = ax1.twinx()
                bars2 = ax2.bar(x + width/2, throughputs, width, label='Throughput (samples/sec)',
                               color=[HARDWARE_COLORS.get(d, "#333333") for d in devices], alpha=0.7)
                ax2.set_ylabel('Throughput (samples/sec)')
                
                # Add legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                plt.title(f"Hardware Performance Comparison for {model_id} (batch=1, seq_len=128)")
                plt.tight_layout()
                
                # Save plot
                bar_path = os.path.join(output_dir, f"{base_filename}_performance_comparison.png")
                plt.savefig(bar_path)
                logger.info(f"Performance comparison plot saved to {bar_path}")
            
            # Generate text summary
            summary_path = os.path.join(output_dir, f"{base_filename}_hardware_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(f"Hardware Comparison for {model_id}\n")
                f.write("="*80 + "\n\n")
                
                architecture_type = df["architecture_type"].iloc[0]
                f.write(f"Model Architecture: {architecture_type}\n\n")
                
                # Single batch, sequence_length=128 comparison
                f.write("Performance for Batch Size=1, Sequence Length=128:\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Device':<10} {'Precision':<10} {'Latency (ms)':<15} {'Throughput (samples/s)':<25} {'Memory (MB)':<15}\n")
                f.write("-"*80 + "\n")
                
                for device in df["device"].unique():
                    subset = df[(df["device"] == device) & 
                               (df["batch_size"] == 1) & 
                               (df["sequence_length"] == 128)]
                    if not subset.empty:
                        row = subset.iloc[0]
                        f.write(f"{device:<10} {row['precision']:<10} {row['latency_mean_ms']:<15.2f} {row['throughput_samples_per_sec']:<25.2f} {row['memory_usage_mb']:<15.2f}\n")
                
                f.write("\n\n")
                
                # Best throughput by hardware
                f.write("Best Throughput by Hardware:\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Device':<10} {'Batch Size':<10} {'Seq Length':<10} {'Throughput (samples/s)':<25} {'Latency (ms)':<15}\n")
                f.write("-"*80 + "\n")
                
                for device in df["device"].unique():
                    subset = df[df["device"] == device]
                    if not subset.empty:
                        # Find configuration with best throughput
                        best_idx = subset["throughput_samples_per_sec"].idxmax()
                        row = subset.loc[best_idx]
                        f.write(f"{device:<10} {row['batch_size']:<10} {row['sequence_length']:<10} {row['throughput_samples_per_sec']:<25.2f} {row['latency_mean_ms']:<15.2f}\n")
            
            logger.info(f"Hardware comparison summary saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Error visualizing hardware comparison: {e}")
    
    def visualize_architecture_comparison(self, output_dir: str = "benchmark_visualizations"):
        """
        Visualize architecture comparison across hardware backends.
        
        Args:
            output_dir: Directory to save visualizations
        """
        if not DUCKDB_AVAILABLE:
            logger.error("DuckDB not available. Cannot generate architecture comparison.")
            return
            
        if not self.db_connection:
            logger.error("No database connection. Provide db_path in initialization.")
            return
            
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Pandas not available. Generating text summary only.")
            return
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Query for architecture results
            query = """
            SELECT 
                architecture_type,
                device,
                AVG(throughput_samples_per_sec) as avg_throughput,
                COUNT(DISTINCT model_id) as model_count
            FROM 
                v_throughput_by_architecture
            GROUP BY 
                architecture_type, device
            ORDER BY 
                architecture_type, avg_throughput DESC
            """
            
            df = self.load_from_db(query)
            
            if df.empty:
                logger.warning("No architecture comparison data found")
                return
                
            # Create timestamp for filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"architecture_comparison_{timestamp}"
            
            # Plot architecture comparison
            plt.figure(figsize=(14, 8))
            
            # Group by architecture
            arch_types = df["architecture_type"].unique()
            
            # Set up bars
            x = np.arange(len(arch_types))
            width = 0.15
            devices = df["device"].unique()
            n_devices = len(devices)
            
            # Plot bars for each device
            for i, device in enumerate(devices):
                device_data = []
                
                for arch in arch_types:
                    subset = df[(df["architecture_type"] == arch) & (df["device"] == device)]
                    if not subset.empty:
                        device_data.append(float(subset["avg_throughput"].iloc[0]))
                    else:
                        device_data.append(0)
                
                offset = width * (i - n_devices/2 + 0.5)
                plt.bar(x + offset, device_data, width, label=device, 
                       color=HARDWARE_COLORS.get(device, None))
            
            plt.xlabel("Architecture Type")
            plt.ylabel("Average Throughput (samples/sec)")
            plt.title("Architecture Performance Comparison by Hardware")
            plt.xticks(x, arch_types, rotation=45, ha="right")
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            arch_path = os.path.join(output_dir, f"{base_filename}_throughput.png")
            plt.savefig(arch_path)
            logger.info(f"Architecture comparison plot saved to {arch_path}")
            
            # Plot model count by architecture
            plt.figure(figsize=(14, 8))
            
            model_counts = []
            for arch in arch_types:
                subset = df[df["architecture_type"] == arch]
                if not subset.empty:
                    # Take the first device's count since it should be the same for all devices
                    model_counts.append(int(subset["model_count"].iloc[0]))
                else:
                    model_counts.append(0)
            
            plt.bar(arch_types, model_counts, color="#1f77b4")
            plt.xlabel("Architecture Type")
            plt.ylabel("Number of Models")
            plt.title("Model Count by Architecture")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            # Save plot
            count_path = os.path.join(output_dir, f"{base_filename}_model_counts.png")
            plt.savefig(count_path)
            logger.info(f"Model count plot saved to {count_path}")
            
            # Generate text summary
            summary_path = os.path.join(output_dir, f"{base_filename}_summary.txt")
            with open(summary_path, 'w') as f:
                f.write("Architecture Performance Comparison\n")
                f.write("="*80 + "\n\n")
                
                # Summarize by architecture
                f.write("Average Throughput (samples/sec) by Architecture:\n")
                f.write("-"*80 + "\n")
                f.write(f"{'Architecture':<20} {'Device':<10} {'Avg Throughput':<20} {'Model Count':<15}\n")
                f.write("-"*80 + "\n")
                
                for arch in arch_types:
                    first = True
                    for device in devices:
                        subset = df[(df["architecture_type"] == arch) & (df["device"] == device)]
                        if not subset.empty:
                            row = subset.iloc[0]
                            if first:
                                f.write(f"{arch:<20} {device:<10} {row['avg_throughput']:<20.2f} {row['model_count']:<15}\n")
                                first = False
                            else:
                                f.write(f"{'':<20} {device:<10} {row['avg_throughput']:<20.2f} {'':<15}\n")
                    f.write("\n")
            
            logger.info(f"Architecture comparison summary saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Error visualizing architecture comparison: {e}")
    
    def _text_summary(self, data: Dict[str, Any]):
        """
        Generate text summary of benchmark results.
        
        Args:
            data: Benchmark results dictionary
        """
        print("\nBenchmark Summary:")
        print("="*80)
        
        print(f"Model: {data.get('model_id', 'Unknown')}")
        print(f"Device: {data.get('device', 'Unknown')}")
        print(f"Architecture: {data.get('architecture_type', 'Unknown')}")
        print(f"Precision: {data.get('precision', 'Unknown')}")
        print("-"*80)
        
        # Load time and memory
        if "load_results" in data and data["load_results"].get("success", False):
            print(f"Load Time: {data['load_results']['load_time_seconds']:.2f} seconds")
            print(f"Load Memory: {data['load_results'].get('memory_usage_mb', 0):.2f} MB")
        print("-"*80)
        
        # Summary metrics
        if "summary" in data:
            print("Summary Metrics:")
            print(f"- Average Latency: {data['summary'].get('average_latency_ms', 0):.2f} ms")
            print(f"- Best Throughput: {data['summary'].get('best_throughput_samples_per_sec', 0):.2f} samples/sec")
            print(f"  (Config: {data['summary'].get('best_throughput_config', 'N/A')})")
            print(f"- Best Latency: {data['summary'].get('best_latency_ms', 0):.2f} ms")
            print(f"  (Config: {data['summary'].get('best_latency_config', 'N/A')})")
        print("-"*80)
        
        # Detailed results
        print("Detailed Results:")
        print(f"{'Batch Size':<10} {'Seq Length':<12} {'Latency (ms)':<15} {'Throughput (samples/s)':<25} {'Memory (MB)':<15}")
        print("-"*80)
        
        for batch_key, result in sorted(data.get("batch_results", {}).items()):
            if not result.get("success", False):
                continue
                
            # Parse batch key to get batch size and sequence length
            # Format: b{batch_size}_s{seq_len}
            parts = batch_key.split("_")
            batch_size = int(parts[0][1:])  # Remove 'b' prefix
            seq_len = int(parts[1][1:])     # Remove 's' prefix
            
            print(f"{batch_size:<10} {seq_len:<12} {result.get('latency_mean_ms', 0):<15.2f} {result.get('throughput_samples_per_sec', 0):<25.2f} {result.get('memory_usage_mb', 0):<15.2f}")
        
        print("="*80)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    
    parser.add_argument("--json-path", type=str,
                        help="Path to JSON benchmark result file")
    
    parser.add_argument("--db-path", type=str,
                        help="Path to benchmark database")
    
    parser.add_argument("--model-id", type=str,
                        help="Model ID for hardware comparison visualization")
    
    parser.add_argument("--output-dir", type=str, default="benchmark_visualizations",
                        help="Directory to save visualizations")
    
    parser.add_argument("--architecture-comparison", action="store_true",
                        help="Generate architecture comparison visualization")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.json_path is None and args.db_path is None:
        logger.error("Must provide either --json-path or --db-path")
        return 1
        
    if args.model_id is not None and args.db_path is None:
        logger.error("Hardware comparison requires --db-path")
        return 1
        
    if args.architecture_comparison and args.db_path is None:
        logger.error("Architecture comparison requires --db-path")
        return 1
        
    # Create visualizer
    visualizer = BenchmarkVisualizer(args.db_path)
    
    # Process based on arguments
    if args.json_path:
        data = visualizer.load_from_json(args.json_path)
        if data:
            visualizer.visualize_json_result(data, args.output_dir)
    
    if args.model_id and args.db_path:
        visualizer.visualize_hardware_comparison(args.model_id, args.output_dir)
    
    if args.architecture_comparison and args.db_path:
        visualizer.visualize_architecture_comparison(args.output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())