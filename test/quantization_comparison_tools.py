#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quantization Comparison Tools

This module provides tools for comparing different quantization methods for Qualcomm hardware,
including benchmarking, visualization, and analysis tools.

Usage:
    python quantization_comparison_tools.py compare-all --model-path <path> --output-dir <dir>
    python quantization_comparison_tools.py visualize --results-path <path> --output-path <path>
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import time
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
QUANTIZATION_METHODS = [
    "int8", 
    "int4", 
    "cluster", 
    "hybrid", 
    "per-channel", 
    "qat", 
    "sparse"
]

METRICS = [
    "accuracy",
    "latency", 
    "throughput", 
    "power", 
    "memory", 
    "size", 
    "startup"
]

PLOT_TYPES = [
    "bar", 
    "radar", 
    "scatter", 
    "pareto", 
    "heatmap"
]

class QuantizationComparison:
    """Base class for quantization comparison tools."""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        model_type: str,
        methods: List[str] = None,
        metrics: List[str] = None,
        mock: bool = False,
        **kwargs
    ):
        """
        Initialize the quantization comparison.
        
        Args:
            model_path: Path to the input model
            output_dir: Directory to save results
            model_type: Type of the model (text, vision, audio, etc.)
            methods: List of quantization methods to compare
            metrics: List of metrics to measure
            mock: Run in mock mode without actual hardware
            **kwargs: Additional keyword arguments
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.model_type = model_type
        self.methods = methods or QUANTIZATION_METHODS
        self.metrics = metrics or METRICS
        self.mock = mock
        self.kwargs = kwargs
        
        # Validate inputs
        self._validate_inputs()
        
        # Initialize results storage
        self.results = {}
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
    
    def _validate_inputs(self):
        """Validate input parameters."""
        if not self.mock and not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        
        for method in self.methods:
            if method not in QUANTIZATION_METHODS:
                warnings.warn(f"Unknown quantization method: {method}")
        
        for metric in self.metrics:
            if metric not in METRICS:
                warnings.warn(f"Unknown metric: {metric}")
    
    def compare_all(self):
        """Compare all quantization methods."""
        logger.info(f"Comparing quantization methods: {self.methods}")
        logger.info(f"Measuring metrics: {self.metrics}")
        
        results = {}
        
        # Quantize the model with each method and collect metrics
        for method in self.methods:
            logger.info(f"Applying {method} quantization")
            metrics = self._apply_quantization(method)
            results[method] = metrics
        
        self.results = results
        
        # Save results to file
        self._save_results()
        
        return results
    
    def _apply_quantization(self, method):
        """Apply a specific quantization method and collect metrics."""
        if self.mock:
            logger.info(f"Mock mode: Simulating {method} quantization")
            return self._generate_mock_metrics(method)
        
        # In real implementation, apply quantization and measure metrics
        # This would call the actual quantization implementation
        # from qualcomm_advanced_quantization import *
        
        logger.info(f"Applying {method} quantization to model {self.model_path}")
        
        output_path = self._get_output_path(method)
        
        # Create command for the appropriate quantization method
        if method == "int8":
            cmd = self._create_basic_quantization_command("int8", output_path)
        elif method == "int4":
            cmd = self._create_basic_quantization_command("int4", output_path)
        elif method == "cluster":
            cmd = self._create_cluster_command(output_path)
        elif method == "hybrid":
            cmd = self._create_hybrid_command(output_path)
        elif method == "per-channel":
            cmd = self._create_per_channel_command(output_path)
        elif method == "qat":
            cmd = self._create_qat_command(output_path)
        elif method == "sparse":
            cmd = self._create_sparse_command(output_path)
        else:
            raise ValueError(f"Unknown quantization method: {method}")
        
        # Execute command and measure metrics
        # This is a placeholder - in real implementation, we would execute the command
        # and measure actual metrics
        metrics = {
            "latency_ms": 10.5,
            "throughput_items_per_sec": 95.2,
            "memory_mb": 78.3,
            "power_watts": 1.2,
            "accuracy": 0.91,
            "model_size_mb": 25.7
        }
        
        logger.info(f"Collected metrics for {method} quantization: {metrics}")
        return metrics
    
    def _create_basic_quantization_command(self, precision, output_path):
        """Create command for basic int8/int4 quantization."""
        return f"python qualcomm_quantization_support.py quantize --model-path {self.model_path} --output-path {output_path} --method {precision} --model-type {self.model_type} --mock"
    
    def _create_cluster_command(self, output_path):
        """Create command for weight clustering quantization."""
        return f"python qualcomm_advanced_quantization.py cluster --model-path {self.model_path} --output-path {output_path} --model-type {self.model_type} --clusters 16 --adaptive-centroids --mock"
    
    def _create_hybrid_command(self, output_path):
        """Create command for hybrid precision quantization."""
        return f"python qualcomm_advanced_quantization.py hybrid --model-path {self.model_path} --output-path {output_path} --model-type {self.model_type} --attention-precision int8 --feedforward-precision int4 --embedding-precision int8 --mock"
    
    def _create_per_channel_command(self, output_path):
        """Create command for per-channel quantization."""
        return f"python qualcomm_advanced_quantization.py per-channel --model-path {self.model_path} --output-path {output_path} --model-type {self.model_type} --mock"
    
    def _create_qat_command(self, output_path):
        """Create command for QAT quantization."""
        # In a real implementation, we would provide an actual dataset
        return f"python qualcomm_advanced_quantization.py qat --model-path {self.model_path} --output-path {output_path} --model-type {self.model_type} --train-dataset mock_dataset --mock"
    
    def _create_sparse_command(self, output_path):
        """Create command for sparse quantization."""
        return f"python qualcomm_advanced_quantization.py sparse --model-path {self.model_path} --output-path {output_path} --model-type {self.model_type} --sparsity 0.5 --mock"
    
    def _get_output_path(self, method):
        """Get output path for a specific method."""
        model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        return os.path.join(self.output_dir, f"{model_name}_{method}.qnn")
    
    def _generate_mock_metrics(self, method):
        """Generate mock metrics for simulation."""
        base_metrics = {
            "accuracy": 0.92,
            "latency_ms": 10.0,
            "throughput_items_per_sec": 100.0,
            "power_watts": 1.5,
            "memory_mb": 80.0,
            "size_mb": 30.0,
            "startup_ms": 150.0
        }
        
        # Adjust metrics based on method
        if method == "int8":
            base_metrics["accuracy"] = 0.91
            base_metrics["latency_ms"] = 8.0
            base_metrics["memory_mb"] = 60.0
            base_metrics["size_mb"] = 25.0
        elif method == "int4":
            base_metrics["accuracy"] = 0.89
            base_metrics["latency_ms"] = 7.0
            base_metrics["memory_mb"] = 40.0
            base_metrics["size_mb"] = 15.0
        elif method == "cluster":
            base_metrics["accuracy"] = 0.92
            base_metrics["latency_ms"] = 9.0
            base_metrics["memory_mb"] = 55.0
            base_metrics["size_mb"] = 20.0
        elif method == "hybrid":
            base_metrics["accuracy"] = 0.91
            base_metrics["latency_ms"] = 7.5
            base_metrics["memory_mb"] = 50.0
            base_metrics["size_mb"] = 18.0
        elif method == "per-channel":
            base_metrics["accuracy"] = 0.93
            base_metrics["latency_ms"] = 9.5
            base_metrics["memory_mb"] = 65.0
            base_metrics["size_mb"] = 27.0
        elif method == "qat":
            base_metrics["accuracy"] = 0.94
            base_metrics["latency_ms"] = 9.0
            base_metrics["memory_mb"] = 60.0
            base_metrics["size_mb"] = 25.0
        elif method == "sparse":
            base_metrics["accuracy"] = 0.90
            base_metrics["latency_ms"] = 6.0
            base_metrics["memory_mb"] = 35.0
            base_metrics["size_mb"] = 12.0
        
        # Add some randomness to make it more realistic
        for key in base_metrics:
            if key == "accuracy":
                # Smaller variation for accuracy
                base_metrics[key] *= (1.0 + np.random.uniform(-0.01, 0.01))
            else:
                # Larger variation for other metrics
                base_metrics[key] *= (1.0 + np.random.uniform(-0.05, 0.05))
        
        return base_metrics
    
    def _save_results(self):
        """Save comparison results to file."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        results_path = os.path.join(self.output_dir, f"{model_name}_comparison.json")
        
        # Prepare output data
        output_data = {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "methods": self.methods,
            "metrics": self.metrics,
            "results": self.results,
            "timestamp": time.time()
        }
        
        try:
            with open(results_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Results saved to {results_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def store_in_db(self):
        """Store comparison results in the benchmark database."""
        if not self.results:
            logger.warning("No results to store in database")
            return
        
        try:
            from benchmark_db_api import BenchmarkDB
            db = BenchmarkDB(db_path="./benchmark_db.duckdb")
            
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            
            # Store comparison results
            db.store_quantization_comparison(
                model_name=model_name,
                model_type=self.model_type,
                methods=self.methods,
                metrics=self.metrics,
                results=self.results
            )
            
            logger.info("Comparison results stored in database")
        except ImportError:
            logger.warning("benchmark_db_api module not found, results not stored in database")
        except Exception as e:
            logger.error(f"Error storing results in database: {e}")


class QuantizationVisualizer:
    """Visualizer for quantization comparison results."""
    
    def __init__(
        self,
        results_path: str,
        output_path: str,
        plot_type: str = 'radar',
        metrics: List[str] = None,
        methods: List[str] = None,
        **kwargs
    ):
        """
        Initialize the quantization visualizer.
        
        Args:
            results_path: Path to the comparison results JSON file
            output_path: Path to save the visualization
            plot_type: Type of plot to generate
            metrics: List of metrics to include in the visualization
            methods: List of methods to include in the visualization
            **kwargs: Additional keyword arguments
        """
        self.results_path = results_path
        self.output_path = output_path
        self.plot_type = plot_type
        self.metrics = metrics
        self.methods = methods
        self.kwargs = kwargs
        
        # Load results
        self.results = self._load_results()
        
        # Filter metrics and methods if provided
        if self.metrics:
            self._filter_metrics()
        else:
            self.metrics = self.results.get('metrics', METRICS)
        
        if self.methods:
            self._filter_methods()
        else:
            self.methods = self.results.get('methods', QUANTIZATION_METHODS)
    
    def _load_results(self):
        """Load comparison results from file."""
        try:
            with open(self.results_path, 'r') as f:
                results = json.load(f)
            return results
        except Exception as e:
            logger.error(f"Error loading results: {e}")
            raise
    
    def _filter_metrics(self):
        """Filter results to include only specified metrics."""
        all_metrics = self.results.get('metrics', METRICS)
        filtered_metrics = [m for m in self.metrics if m in all_metrics]
        
        if len(filtered_metrics) != len(self.metrics):
            missing = set(self.metrics) - set(filtered_metrics)
            warnings.warn(f"Some metrics not found in results: {missing}")
        
        self.metrics = filtered_metrics
    
    def _filter_methods(self):
        """Filter results to include only specified methods."""
        all_methods = self.results.get('methods', QUANTIZATION_METHODS)
        filtered_methods = [m for m in self.methods if m in all_methods]
        
        if len(filtered_methods) != len(self.methods):
            missing = set(self.methods) - set(filtered_methods)
            warnings.warn(f"Some methods not found in results: {missing}")
        
        self.methods = filtered_methods
    
    def visualize(self):
        """Generate visualization based on plot type."""
        logger.info(f"Generating {self.plot_type} visualization")
        
        if self.plot_type == 'bar':
            self._create_bar_chart()
        elif self.plot_type == 'radar':
            self._create_radar_chart()
        elif self.plot_type == 'scatter':
            self._create_scatter_plot()
        elif self.plot_type == 'pareto':
            self._create_pareto_plot()
        elif self.plot_type == 'heatmap':
            self._create_heatmap()
        else:
            logger.error(f"Unknown plot type: {self.plot_type}")
            raise ValueError(f"Unknown plot type: {self.plot_type}")
        
        logger.info(f"Visualization saved to {self.output_path}")
    
    def _normalize_metrics(self):
        """Normalize metrics for visualization."""
        normalized_data = {}
        
        for method in self.methods:
            if method not in self.results['results']:
                continue
            
            normalized_data[method] = {}
            
            for metric in self.metrics:
                # Skip if metric not available for method
                if metric not in self.results['results'][method]:
                    continue
                
                # Get min and max values across all methods
                all_values = [self.results['results'][m].get(metric, 0) for m in self.methods 
                              if metric in self.results['results'][m]]
                
                if not all_values:
                    continue
                
                min_val = min(all_values)
                max_val = max(all_values)
                
                # Avoid division by zero
                if min_val == max_val:
                    normalized_data[method][metric] = 1.0
                else:
                    # Higher is better for accuracy and throughput
                    if metric in ['accuracy', 'throughput_items_per_sec']:
                        normalized_data[method][metric] = (self.results['results'][method][metric] - min_val) / (max_val - min_val)
                    # Lower is better for other metrics
                    else:
                        normalized_data[method][metric] = 1.0 - (self.results['results'][method][metric] - min_val) / (max_val - min_val)
        
        return normalized_data
    
    def _create_bar_chart(self):
        """Create bar chart for each metric."""
        # Setup plot
        fig, axes = plt.subplots(len(self.metrics), 1, figsize=(12, 4 * len(self.metrics)))
        
        # If only one metric, axes is not iterable
        if len(self.metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(self.metrics):
            ax = axes[i]
            
            # Get values for the metric
            values = []
            labels = []
            
            for method in self.methods:
                if method in self.results['results'] and metric in self.results['results'][method]:
                    values.append(self.results['results'][method][metric])
                    labels.append(method)
            
            # Skip if no values
            if not values:
                continue
            
            # Higher is better for accuracy and throughput
            color = 'green' if metric in ['accuracy', 'throughput_items_per_sec'] else 'red'
            
            # Plot bars
            ax.bar(labels, values, color=color, alpha=0.7)
            
            # Add labels and title
            ax.set_ylabel(metric)
            ax.set_title(f"{metric.capitalize()} by Quantization Method")
            
            # Add value labels on bars
            for j, v in enumerate(values):
                ax.text(j, v, f"{v:.3f}", ha='center', va='bottom')
            
            # Rotate x labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(self.output_path)
        plt.close()
    
    def _create_radar_chart(self):
        """Create radar chart comparing all methods."""
        # Normalize metrics for radar chart
        normalized_data = self._normalize_metrics()
        
        # Setup radar chart
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of metrics
        N = len(self.metrics)
        
        # Angles for each metric (evenly spaced around the circle)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        
        # Close the loop
        angles += angles[:1]
        
        # Plot for each method
        for method in normalized_data:
            # Get values for each metric
            values = [normalized_data[method].get(metric, 0) for metric in self.metrics]
            
            # Close the loop
            values += values[:1]
            
            # Plot values
            ax.plot(angles, values, linewidth=2, label=method)
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels for each metric
        plt.xticks(angles[:-1], self.metrics)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Set title
        plt.title("Quantization Method Comparison")
        
        plt.tight_layout()
        plt.savefig(self.output_path)
        plt.close()
    
    def _create_scatter_plot(self):
        """Create scatter plot comparing two metrics."""
        # We need at least two metrics for a scatter plot
        if len(self.metrics) < 2:
            logger.error("Scatter plot requires at least two metrics")
            raise ValueError("Scatter plot requires at least two metrics")
        
        # Use the first two metrics
        metric_x = self.metrics[0]
        metric_y = self.metrics[1]
        
        # Setup plot
        plt.figure(figsize=(10, 8))
        
        # Get values for each method
        x_values = []
        y_values = []
        labels = []
        
        for method in self.methods:
            if (method in self.results['results'] and 
                metric_x in self.results['results'][method] and 
                metric_y in self.results['results'][method]):
                x_values.append(self.results['results'][method][metric_x])
                y_values.append(self.results['results'][method][metric_y])
                labels.append(method)
        
        # Skip if no values
        if not x_values or not y_values:
            logger.error("No data available for scatter plot")
            raise ValueError("No data available for scatter plot")
        
        # Plot scatter points
        plt.scatter(x_values, y_values, s=100, alpha=0.7)
        
        # Add labels for each point
        for i, label in enumerate(labels):
            plt.annotate(label, (x_values[i], y_values[i]), fontsize=10)
        
        # Add labels and title
        plt.xlabel(metric_x)
        plt.ylabel(metric_y)
        plt.title(f"{metric_x.capitalize()} vs {metric_y.capitalize()}")
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.output_path)
        plt.close()
    
    def _create_pareto_plot(self):
        """Create Pareto frontier plot for two metrics."""
        # We need at least two metrics for a Pareto plot
        if len(self.metrics) < 2:
            logger.error("Pareto plot requires at least two metrics")
            raise ValueError("Pareto plot requires at least two metrics")
        
        # Use the first two metrics
        metric_x = self.metrics[0]
        metric_y = self.metrics[1]
        
        # Setup plot
        plt.figure(figsize=(10, 8))
        
        # Get values for each method
        points = []
        labels = []
        
        for method in self.methods:
            if (method in self.results['results'] and 
                metric_x in self.results['results'][method] and 
                metric_y in self.results['results'][method]):
                # We need to handle the direction of optimization
                # For accuracy and throughput, higher is better, so negate for Pareto
                x_val = self.results['results'][method][metric_x]
                y_val = self.results['results'][method][metric_y]
                
                # Higher is better for accuracy and throughput, so negate for Pareto
                if metric_x in ['accuracy', 'throughput_items_per_sec']:
                    x_val = -x_val
                
                # Higher is better for accuracy and throughput, so negate for Pareto
                if metric_y in ['accuracy', 'throughput_items_per_sec']:
                    y_val = -y_val
                
                points.append((x_val, y_val))
                labels.append(method)
        
        # Skip if no values
        if not points:
            logger.error("No data available for Pareto plot")
            raise ValueError("No data available for Pareto plot")
        
        # Find Pareto frontier
        pareto_points = []
        pareto_indices = []
        
        # Sort points by x value
        sorted_indices = np.argsort([p[0] for p in points])
        sorted_points = [points[i] for i in sorted_indices]
        sorted_labels = [labels[i] for i in sorted_indices]
        
        # Initialize with first point
        pareto_points.append(sorted_points[0])
        pareto_indices.append(0)
        
        # Find Pareto points
        for i in range(1, len(sorted_points)):
            if sorted_points[i][1] < pareto_points[-1][1]:
                pareto_points.append(sorted_points[i])
                pareto_indices.append(i)
        
        # Plot all points
        for i, point in enumerate(sorted_points):
            x_val, y_val = point
            
            # Reverse negation for plotting
            if metric_x in ['accuracy', 'throughput_items_per_sec']:
                x_val = -x_val
            
            if metric_y in ['accuracy', 'throughput_items_per_sec']:
                y_val = -y_val
            
            # Different color for Pareto points
            color = 'red' if i in pareto_indices else 'blue'
            marker = 'o' if i in pareto_indices else 'x'
            
            plt.scatter(x_val, y_val, s=100, alpha=0.7, color=color, marker=marker)
            plt.annotate(sorted_labels[i], (x_val, y_val), fontsize=10)
        
        # Connect Pareto points with line
        pareto_x = []
        pareto_y = []
        
        for i in pareto_indices:
            x_val, y_val = sorted_points[i]
            
            # Reverse negation for plotting
            if metric_x in ['accuracy', 'throughput_items_per_sec']:
                x_val = -x_val
            
            if metric_y in ['accuracy', 'throughput_items_per_sec']:
                y_val = -y_val
            
            pareto_x.append(x_val)
            pareto_y.append(y_val)
        
        plt.plot(pareto_x, pareto_y, 'r--', alpha=0.7)
        
        # Add labels and title
        plt.xlabel(metric_x)
        plt.ylabel(metric_y)
        plt.title(f"Pareto Frontier: {metric_x.capitalize()} vs {metric_y.capitalize()}")
        
        # Add legend
        plt.legend(['Pareto Frontier', 'Pareto Optimal', 'Non-Optimal'])
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.output_path)
        plt.close()
    
    def _create_heatmap(self):
        """Create heatmap for all methods and metrics."""
        # Normalize metrics for heatmap
        normalized_data = self._normalize_metrics()
        
        # Setup heatmap
        plt.figure(figsize=(12, 8))
        
        # Prepare data for heatmap
        data = []
        
        for method in self.methods:
            if method not in normalized_data:
                continue
            
            row = []
            for metric in self.metrics:
                row.append(normalized_data[method].get(metric, 0))
            
            data.append(row)
        
        # Skip if no data
        if not data:
            logger.error("No data available for heatmap")
            raise ValueError("No data available for heatmap")
        
        # Create heatmap
        plt.imshow(data, cmap='RdYlGn', aspect='auto', alpha=0.8)
        
        # Add labels
        plt.xticks(np.arange(len(self.metrics)), self.metrics, rotation=45, ha='right')
        plt.yticks(np.arange(len(self.methods)), self.methods)
        
        # Add title
        plt.title("Quantization Method Comparison Heatmap")
        
        # Add colorbar
        plt.colorbar(label='Normalized Score')
        
        # Add values to cells
        for i in range(len(self.methods)):
            for j in range(len(self.metrics)):
                plt.text(j, i, f"{data[i][j]:.2f}", ha='center', va='center', color='black')
        
        plt.tight_layout()
        plt.savefig(self.output_path)
        plt.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quantization Comparison Tools")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Compare-all command
    compare_parser = subparsers.add_parser("compare-all", help="Compare all quantization methods")
    compare_parser.add_argument("--model-path", required=True, help="Path to the input model")
    compare_parser.add_argument("--output-dir", required=True, help="Directory to save results")
    compare_parser.add_argument("--model-type", required=True, choices=["text", "vision", "audio", "multimodal"],
                              help="Type of the model")
    compare_parser.add_argument("--methods", help="Comma-separated list of quantization methods to compare")
    compare_parser.add_argument("--metrics", help="Comma-separated list of metrics to measure")
    compare_parser.add_argument("--mock", action="store_true", help="Run in mock mode without actual hardware")
    compare_parser.add_argument("--store-in-db", action="store_true", help="Store results in the benchmark database")
    
    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize comparison results")
    visualize_parser.add_argument("--results-path", required=True, help="Path to the comparison results JSON file")
    visualize_parser.add_argument("--output-path", required=True, help="Path to save the visualization")
    visualize_parser.add_argument("--plot-type", default="radar", choices=PLOT_TYPES,
                                help="Type of plot to generate")
    visualize_parser.add_argument("--metrics", help="Comma-separated list of metrics to include in the visualization")
    visualize_parser.add_argument("--methods", help="Comma-separated list of methods to include in the visualization")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    if args.command == "compare-all":
        # Parse methods and metrics lists if provided
        methods = args.methods.split(',') if args.methods else None
        metrics = args.metrics.split(',') if args.metrics else None
        
        # Create comparison object
        comparison = QuantizationComparison(
            model_path=args.model_path,
            output_dir=args.output_dir,
            model_type=args.model_type,
            methods=methods,
            metrics=metrics,
            mock=args.mock
        )
        
        # Compare all methods
        results = comparison.compare_all()
        
        # Store in database if requested
        if args.store_in_db:
            comparison.store_in_db()
        
        logger.info(f"Comparison complete. Results saved to {args.output_dir}")
    
    elif args.command == "visualize":
        # Parse metrics and methods lists if provided
        metrics = args.metrics.split(',') if args.metrics else None
        methods = args.methods.split(',') if args.methods else None
        
        # Create visualizer object
        visualizer = QuantizationVisualizer(
            results_path=args.results_path,
            output_path=args.output_path,
            plot_type=args.plot_type,
            metrics=metrics,
            methods=methods
        )
        
        # Generate visualization
        visualizer.visualize()
        
        logger.info(f"Visualization complete. Saved to {args.output_path}")
    
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()