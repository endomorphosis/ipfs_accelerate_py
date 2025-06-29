#!/usr/bin/env python3
"""
Visualization tools for model validation and benchmark results.

This module provides various visualization capabilities for:
1. Comparing model performance across hardware platforms
2. Visualizing functionality test results
3. Creating interactive dashboards for hardware-model compatibility
4. Generating recommendation charts for optimal hardware selection
"""

import os
import json
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import visualization libraries with graceful degradation
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    logger.warning("Matplotlib not available. Static plots will be disabled.")
    HAS_MATPLOTLIB = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    logger.warning("Plotly not available. Interactive plots will be disabled.")
    HAS_PLOTLY = False

# Define constants
DEFAULT_OUTPUT_DIR = "./visualization_results"
DEFAULT_BENCHMARK_DIR = "./benchmark_results"
DEFAULT_FUNCTIONALITY_DIR = "./functionality_reports"
HARDWARE_COLORS = {
    "cpu": "#1f77b4",  # blue
    "cuda": "#ff7f0e",  # orange
    "mps": "#2ca02c",   # green
    "rocm": "#d62728",  # red
    "openvino": "#9467bd",  # purple
    "webnn": "#8c564b",  # brown
    "webgpu": "#e377c2"  # pink
}
MODEL_FAMILIES = ["embedding", "text_generation", "vision", "audio", "multimodal"]

class ModelValidationVisualizer:
    """
    A class to create visualizations from model validation and benchmark results.
    """
    
    def __init__(self, 
                 benchmark_dir: str = DEFAULT_BENCHMARK_DIR,
                 functionality_dir: str = DEFAULT_FUNCTIONALITY_DIR,
                 output_dir: str = DEFAULT_OUTPUT_DIR):
        """
        Initialize the visualizer.
        
        Args:
            benchmark_dir: Directory containing benchmark results
            functionality_dir: Directory containing functionality test results
            output_dir: Directory to save visualizations
        """
        self.benchmark_dir = Path(benchmark_dir)
        self.functionality_dir = Path(functionality_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load data
        self.benchmark_data = self._load_benchmark_data()
        self.functionality_data = self._load_functionality_data()
        
        # Prepare data frames
        self.benchmark_df = self._prepare_benchmark_dataframe()
        self.functionality_df = self._prepare_functionality_dataframe()
        
        # Set plot styling
        if HAS_MATPLOTLIB:
            sns.set_theme(style="whitegrid")
            sns.set_context("paper", font_scale=1.2)
    
    def _load_benchmark_data(self) -> List[Dict]:
        """
        Load benchmark data from benchmark results directory.
        
        Returns:
            List of benchmark result dictionaries
        """
        benchmark_data = []
        
        if not self.benchmark_dir.exists():
            logger.warning(f"Benchmark directory {self.benchmark_dir} does not exist.")
            return benchmark_data
        
        # Look for benchmark result JSON files
        benchmark_files = list(self.benchmark_dir.glob("**/benchmark_results*.json"))
        
        for file_path in benchmark_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    benchmark_data.append(data)
                    logger.info(f"Loaded benchmark data from {file_path}")
            except Exception as e:
                logger.error(f"Error loading benchmark data from {file_path}: {e}")
        
        logger.info(f"Loaded {len(benchmark_data)} benchmark result sets")
        return benchmark_data
    
    def _load_functionality_data(self) -> List[Dict]:
        """
        Load functionality test data from functionality reports directory.
        
        Returns:
            List of functionality test result dictionaries
        """
        functionality_data = []
        
        if not self.functionality_dir.exists():
            logger.warning(f"Functionality directory {self.functionality_dir} does not exist.")
            return functionality_data
        
        # Look for functionality result JSON files
        functionality_files = list(self.functionality_dir.glob("**/model_functionality_*.json"))
        
        for file_path in functionality_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    functionality_data.append(data)
                    logger.info(f"Loaded functionality data from {file_path}")
            except Exception as e:
                logger.error(f"Error loading functionality data from {file_path}: {e}")
        
        logger.info(f"Loaded {len(functionality_data)} functionality result sets")
        return functionality_data
    
    def _prepare_benchmark_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Prepare a consolidated DataFrame from benchmark data.
        
        Returns:
            Pandas DataFrame with benchmark results or None if no data
        """
        if not self.benchmark_data:
            return None
        
        records = []
        
        for result_set in self.benchmark_data:
            timestamp = result_set.get("timestamp", "unknown")
            system_info = result_set.get("system_info", {})
            
            if "benchmarks" not in result_set:
                continue
                
            for family, models in result_set["benchmarks"].items():
                for model_name, hw_results in models.items():
                    for hw_type, metrics in hw_results.items():
                        if metrics.get("status") != "completed" or "performance_summary" not in metrics:
                            continue
                            
                        perf = metrics["performance_summary"]
                        
                        # Extract key metrics
                        record = {
                            "timestamp": timestamp,
                            "family": family,
                            "model": model_name,
                            "hardware": hw_type,
                            "platform": system_info.get("platform", "unknown"),
                            "success": True
                        }
                        
                        # Add latency metrics if available
                        if "latency" in perf:
                            for metric in ["min", "max", "mean", "median"]:
                                if metric in perf["latency"]:
                                    record[f"latency_{metric}"] = perf["latency"][metric] * 1000  # convert to ms
                        
                        # Add throughput metrics if available
                        if "throughput" in perf:
                            for metric in ["min", "max", "mean", "median"]:
                                if metric in perf["throughput"]:
                                    record[f"throughput_{metric}"] = perf["throughput"][metric]
                        
                        # Add memory metrics if available
                        if "memory" in perf:
                            for metric in ["min_allocated", "max_allocated", "mean_allocated"]:
                                if metric in perf["memory"]:
                                    record[f"memory_{metric}"] = perf["memory"][metric]
                        
                        # Add model load time if available
                        if "model_load_time" in metrics:
                            record["model_load_time"] = metrics["model_load_time"]
                        
                        records.append(record)
        
        if not records:
            logger.warning("No valid benchmark records found")
            return None
            
        return pd.DataFrame(records)
    
    def _prepare_functionality_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Prepare a consolidated DataFrame from functionality test data.
        
        Returns:
            Pandas DataFrame with functionality results or None if no data
        """
        if not self.functionality_data:
            return None
        
        records = []
        
        for result_set in self.functionality_data:
            if "detailed_results" not in result_set:
                continue
                
            for result in result_set["detailed_results"]:
                record = {
                    "model": result.get("model", "unknown"),
                    "hardware": result.get("hardware", "unknown"),
                    "success": result.get("success", False),
                    "return_code": result.get("return_code", None),
                    "timestamp": result_set.get("stats", {}).get("timestamp", "unknown")
                }
                
                if "error" in result and result["error"]:
                    record["error"] = result["error"]
                    
                records.append(record)
        
        if not records:
            logger.warning("No valid functionality records found")
            return None
            
        return pd.DataFrame(records)
    
    def plot_hardware_comparison(self, metric="latency_mean", output_format="png"):
        """
        Create bar plots comparing hardware platforms for each model family.
        
        Args:
            metric: The metric to compare (latency_mean, throughput_mean, etc.)
            output_format: Output file format (png, pdf, svg, html)
        
        Returns:
            List of output file paths
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available. Cannot create static plots.")
            return []
            
        if self.benchmark_df is None:
            logger.error("No benchmark data available for plotting.")
            return []
        
        output_files = []
        
        # Determine metric label and direction (lower or higher is better)
        metric_label = " ".join(word.capitalize() for word in metric.split("_"))
        if "latency" in metric:
            metric_unit = "ms"
            is_lower_better = True
        elif "throughput" in metric:
            metric_unit = "items/sec"
            is_lower_better = False
        elif "memory" in metric:
            metric_unit = "MB"
            is_lower_better = True
        else:
            metric_unit = ""
            is_lower_better = False
        
        # Plot for each model family
        for family in self.benchmark_df["family"].unique():
            df_family = self.benchmark_df[self.benchmark_df["family"] == family].copy()
            
            plt.figure(figsize=(12, 8))
            
            # Create grouped bar chart
            ax = sns.barplot(x="model", y=metric, hue="hardware", data=df_family, 
                           palette=HARDWARE_COLORS, errorbar=None)
            
            # Set title and labels
            plt.title(f"{metric_label} by Hardware Platform for {family.title()} Models")
            plt.xlabel("Model")
            plt.ylabel(f"{metric_label} ({metric_unit})")
            
            # Handle x-axis labels for readability
            if len(df_family["model"].unique()) > 5:
                plt.xticks(rotation=45, ha="right")
            
            # Add a note about which direction is better
            better_text = "Lower is better" if is_lower_better else "Higher is better"
            plt.figtext(0.99, 0.01, better_text, horizontalalignment="right", 
                      fontsize=10, fontstyle="italic")
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the figure
            output_file = self.output_dir / f"{family}_{metric}_hardware_comparison.{output_format}"
            plt.savefig(output_file)
            plt.close()
            
            output_files.append(str(output_file))
            logger.info(f"Created hardware comparison plot for {family}: {output_file}")
        
        # Create overall comparison across all families
        try:
            plt.figure(figsize=(14, 10))
            
            # Create grouped bar chart with family as y-axis
            df_grouped = self.benchmark_df.groupby(["family", "hardware"])[metric].mean().reset_index()
            
            ax = sns.barplot(x=metric, y="family", hue="hardware", data=df_grouped, 
                           palette=HARDWARE_COLORS, errorbar=None, orient="h")
            
            # Set title and labels
            plt.title(f"Average {metric_label} by Model Family and Hardware Platform")
            plt.xlabel(f"{metric_label} ({metric_unit})")
            plt.ylabel("Model Family")
            
            # Add a note about which direction is better
            better_text = "Lower is better" if is_lower_better else "Higher is better"
            plt.figtext(0.99, 0.01, better_text, horizontalalignment="right", 
                      fontsize=10, fontstyle="italic")
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the figure
            output_file = self.output_dir / f"overall_{metric}_hardware_comparison.{output_format}"
            plt.savefig(output_file)
            plt.close()
            
            output_files.append(str(output_file))
            logger.info(f"Created overall hardware comparison plot: {output_file}")
        except Exception as e:
            logger.error(f"Error creating overall comparison plot: {e}")
        
        return output_files
    
    def plot_functionality_results(self, output_format="png"):
        """
        Create plots showing functionality test results.
        
        Args:
            output_format: Output file format (png, pdf, svg, html)
        
        Returns:
            List of output file paths
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available. Cannot create static plots.")
            return []
            
        if self.functionality_df is None:
            logger.error("No functionality data available for plotting.")
            return []
        
        output_files = []
        
        # Plot success rate by hardware platform
        plt.figure(figsize=(10, 6))
        
        # Calculate success rates
        success_rates = self.functionality_df.groupby("hardware")["success"].mean() * 100
        
        # Create horizontal bar chart
        ax = success_rates.plot(kind="barh", color=[HARDWARE_COLORS.get(hw, "#aaaaaa") for hw in success_rates.index])
        
        # Set title and labels
        plt.title("Model Functionality Success Rate by Hardware Platform")
        plt.xlabel("Success Rate (%)")
        plt.ylabel("Hardware Platform")
        
        # Add value labels to the bars
        for i, v in enumerate(success_rates):
            ax.text(v + 1, i, f"{v:.1f}%", va="center")
        
        # Set x-axis range to 0-100%
        plt.xlim(0, 100)
        
        # Add grid lines
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        
        # Save the figure
        output_file = self.output_dir / f"functionality_success_by_hardware.{output_format}"
        plt.savefig(output_file)
        plt.close()
        
        output_files.append(str(output_file))
        logger.info(f"Created functionality success rate plot: {output_file}")
        
        # Plot success rate by model
        plt.figure(figsize=(12, 8))
        
        # Calculate success rates
        model_success_rates = self.functionality_df.groupby("model")["success"].mean() * 100
        
        # Sort by success rate
        model_success_rates = model_success_rates.sort_values(ascending=False)
        
        # Create bar chart
        ax = model_success_rates.plot(kind="bar", color="#1f77b4")
        
        # Set title and labels
        plt.title("Model Functionality Success Rate by Model")
        plt.xlabel("Model")
        plt.ylabel("Success Rate (%)")
        
        # Handle x-axis labels for readability
        if len(model_success_rates) > 5:
            plt.xticks(rotation=45, ha="right")
        
        # Add value labels to the bars
        for i, v in enumerate(model_success_rates):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center")
        
        # Set y-axis range to 0-100%
        plt.ylim(0, 100)
        
        # Add grid lines
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        # Save the figure
        output_file = self.output_dir / f"functionality_success_by_model.{output_format}"
        plt.savefig(output_file)
        plt.close()
        
        output_files.append(str(output_file))
        logger.info(f"Created functionality success by model plot: {output_file}")
        
        # Create heatmap of success by model and hardware
        try:
            # Pivot data to create model x hardware matrix
            pivot_df = self.functionality_df.pivot_table(
                index="model", 
                columns="hardware", 
                values="success",
                aggfunc=lambda x: x.mean() * 100
            )
            
            plt.figure(figsize=(12, len(pivot_df) * 0.4 + 2))
            
            # Create heatmap
            sns.heatmap(pivot_df, annot=True, cmap="RdYlGn", vmin=0, vmax=100, fmt=".0f",
                      cbar_kws={"label": "Success Rate (%)"})
            
            # Set title
            plt.title("Model Functionality Success Rate by Model and Hardware")
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the figure
            output_file = self.output_dir / f"functionality_heatmap.{output_format}"
            plt.savefig(output_file)
            plt.close()
            
            output_files.append(str(output_file))
            logger.info(f"Created functionality heatmap: {output_file}")
        except Exception as e:
            logger.error(f"Error creating functionality heatmap: {e}")
        
        return output_files
    
    def plot_interactive_dashboard(self):
        """
        Create an interactive HTML dashboard with Plotly.
        
        Returns:
            Path to the HTML dashboard
        """
        if not HAS_PLOTLY:
            logger.error("Plotly not available. Cannot create interactive dashboard.")
            return None
            
        if self.benchmark_df is None and self.functionality_df is None:
            logger.error("No data available for plotting.")
            return None
        
        # Create subplots for the dashboard
        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "heatmap", "colspan": 2}, None],
                [{"type": "scatter", "colspan": 2}, None]
            ],
            subplot_titles=(
                "Latency by Hardware Platform",
                "Throughput by Hardware Platform",
                "Functionality Success Rate by Model and Hardware",
                "Performance vs Functionality"
            )
        )
        
        # Add benchmark data if available
        if self.benchmark_df is not None:
            # Latency by hardware
            latency_data = self.benchmark_df.groupby(["hardware", "family"])["latency_mean"].mean().reset_index()
            for hardware in latency_data["hardware"].unique():
                hw_data = latency_data[latency_data["hardware"] == hardware]
                fig.add_trace(
                    go.Bar(
                        x=hw_data["family"],
                        y=hw_data["latency_mean"],
                        name=hardware,
                        marker_color=HARDWARE_COLORS.get(hardware, "#aaaaaa")
                    ),
                    row=1, col=1
                )
            
            # Throughput by hardware
            throughput_data = self.benchmark_df.groupby(["hardware", "family"])["throughput_mean"].mean().reset_index()
            for hardware in throughput_data["hardware"].unique():
                hw_data = throughput_data[throughput_data["hardware"] == hardware]
                fig.add_trace(
                    go.Bar(
                        x=hw_data["family"],
                        y=hw_data["throughput_mean"],
                        name=hardware,
                        marker_color=HARDWARE_COLORS.get(hardware, "#aaaaaa"),
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # Add functionality data if available
        if self.functionality_df is not None:
            try:
                # Create pivot table for heatmap
                pivot_df = self.functionality_df.pivot_table(
                    index="model", 
                    columns="hardware", 
                    values="success",
                    aggfunc=lambda x: x.mean() * 100
                )
                
                # Add heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=pivot_df.values,
                        x=pivot_df.columns,
                        y=pivot_df.index,
                        colorscale="RdYlGn",
                        showscale=True,
                        zmin=0,
                        zmax=100,
                        colorbar=dict(title="Success Rate (%)"),
                        hovertemplate="Model: %{y}<br>Hardware: %{x}<br>Success Rate: %{z:.1f}%<extra></extra>"
                    ),
                    row=2, col=1
                )
            except Exception as e:
                logger.error(f"Error adding functionality heatmap to dashboard: {e}")
        
        # Add combined performance vs functionality plot if both are available
        if self.benchmark_df is not None and self.functionality_df is not None:
            try:
                # Join benchmark and functionality data
                perf_func_data = pd.merge(
                    self.benchmark_df[["model", "hardware", "latency_mean"]],
                    self.functionality_df.groupby(["model", "hardware"])["success"].mean().reset_index(),
                    on=["model", "hardware"],
                    how="inner"
                )
                
                # Create scatter plot
                for hardware in perf_func_data["hardware"].unique():
                    hw_data = perf_func_data[perf_func_data["hardware"] == hardware]
                    fig.add_trace(
                        go.Scatter(
                            x=hw_data["latency_mean"],
                            y=hw_data["success"] * 100,
                            mode="markers",
                            name=hardware,
                            marker=dict(
                                color=HARDWARE_COLORS.get(hardware, "#aaaaaa"),
                                size=10
                            ),
                            text=hw_data["model"],
                            hovertemplate="Model: %{text}<br>Latency: %{x:.2f}ms<br>Success Rate: %{y:.1f}%<extra></extra>"
                        ),
                        row=3, col=1
                    )
            except Exception as e:
                logger.error(f"Error adding performance vs functionality plot to dashboard: {e}")
        
        # Update layout
        fig.update_layout(
            title_text="Model Validation and Performance Dashboard",
            height=1200,
            width=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes titles
        fig.update_xaxes(title_text="Model Family", row=1, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
        fig.update_xaxes(title_text="Model Family", row=1, col=2)
        fig.update_yaxes(title_text="Throughput (items/sec)", row=1, col=2)
        fig.update_xaxes(title_text="Hardware Platform", row=2, col=1)
        fig.update_yaxes(title_text="Model", row=2, col=1)
        fig.update_xaxes(title_text="Latency (ms)", row=3, col=1)
        fig.update_yaxes(title_text="Success Rate (%)", row=3, col=1)
        
        # Save the dashboard
        output_file = self.output_dir / "model_validation_dashboard.html"
        fig.write_html(output_file)
        
        logger.info(f"Created interactive dashboard: {output_file}")
        return str(output_file)
    
    def generate_hardware_recommendations(self, output_format="html"):
        """
        Generate hardware recommendations based on benchmark and functionality results.
        
        Args:
            output_format: Output format for recommendations (html, json, md)
            
        Returns:
            Path to the recommendations file
        """
        if self.benchmark_df is None and self.functionality_df is None:
            logger.error("No data available for generating recommendations.")
            return None
        
        recommendations = {
            "timestamp": datetime.now().isoformat(),
            "model_families": {},
            "general_recommendations": []
        }
        
        # Generate recommendations based on benchmark data
        if self.benchmark_df is not None:
            # Group by family and hardware
            family_hw_latency = self.benchmark_df.groupby(["family", "hardware"])["latency_mean"].mean().reset_index()
            family_hw_throughput = self.benchmark_df.groupby(["family", "hardware"])["throughput_mean"].mean().reset_index()
            
            # Determine best hardware for each family based on latency and throughput
            for family in self.benchmark_df["family"].unique():
                family_latency = family_hw_latency[family_hw_latency["family"] == family]
                family_throughput = family_hw_throughput[family_hw_throughput["family"] == family]
                
                if family_latency.empty or family_throughput.empty:
                    continue
                
                # Get best hardware by latency (lower is better)
                best_latency_hw = family_latency.loc[family_latency["latency_mean"].idxmin()]
                
                # Get best hardware by throughput (higher is better)
                best_throughput_hw = family_throughput.loc[family_throughput["throughput_mean"].idxmax()]
                
                # Store recommendations
                recommendations["model_families"][family] = {
                    "best_for_latency": {
                        "hardware": best_latency_hw["hardware"],
                        "latency_ms": best_latency_hw["latency_mean"]
                    },
                    "best_for_throughput": {
                        "hardware": best_throughput_hw["hardware"],
                        "throughput": best_throughput_hw["throughput_mean"]
                    },
                    "all_hardware": {}
                }
                
                # Add data for all hardware platforms
                for _, row in family_latency.iterrows():
                    hw = row["hardware"]
                    latency = row["latency_mean"]
                    throughput = family_throughput[family_throughput["hardware"] == hw]["throughput_mean"].values[0]
                    
                    recommendations["model_families"][family]["all_hardware"][hw] = {
                        "latency_ms": latency,
                        "throughput": throughput
                    }
        
        # Add functionality data if available
        if self.functionality_df is not None:
            # Calculate success rates by family and hardware
            family_hw_success = {}
            
            # Map models to families
            model_family_map = {}
            if self.benchmark_df is not None:
                for _, row in self.benchmark_df[["model", "family"]].drop_duplicates().iterrows():
                    model_family_map[row["model"]] = row["family"]
            
            # Group by model and hardware
            model_hw_success = self.functionality_df.groupby(["model", "hardware"])["success"].mean().reset_index()
            
            # Map to families
            for _, row in model_hw_success.iterrows():
                model = row["model"]
                hardware = row["hardware"]
                success = row["success"]
                
                # Try to determine family
                family = model_family_map.get(model)
                if not family:
                    # Try to infer from model name
                    for potential_family in MODEL_FAMILIES:
                        if potential_family in model.lower():
                            family = potential_family
                            break
                
                if not family:
                    # Default to model name as family
                    family = model
                
                if family not in family_hw_success:
                    family_hw_success[family] = {}
                    
                if hardware not in family_hw_success[family]:
                    family_hw_success[family][hardware] = []
                    
                family_hw_success[family][hardware].append(success)
            
            # Calculate average success rates
            for family, hw_data in family_hw_success.items():
                for hw, success_list in hw_data.items():
                    avg_success = sum(success_list) / len(success_list) if success_list else 0
                    
                    if family not in recommendations["model_families"]:
                        recommendations["model_families"][family] = {
                            "all_hardware": {}
                        }
                        
                    if hw not in recommendations["model_families"][family]["all_hardware"]:
                        recommendations["model_families"][family]["all_hardware"][hw] = {}
                        
                    recommendations["model_families"][family]["all_hardware"][hw]["success_rate"] = avg_success * 100
                    
                    # Determine best hardware for functionality
                    if "best_for_functionality" not in recommendations["model_families"][family]:
                        recommendations["model_families"][family]["best_for_functionality"] = {
                            "hardware": hw,
                            "success_rate": avg_success * 100
                        }
                    elif avg_success > recommendations["model_families"][family]["best_for_functionality"]["success_rate"] / 100:
                        recommendations["model_families"][family]["best_for_functionality"] = {
                            "hardware": hw,
                            "success_rate": avg_success * 100
                        }
        
        # Add general recommendations
        recommendations["general_recommendations"] = [
            "For embedding models, prefer GPU acceleration when available",
            "For text generation models, high memory GPUs are recommended",
            "For vision models, balance between CPU and GPU based on batch size",
            "For audio models, CUDA is generally the best option",
            "For multimodal models, high memory GPUs provide the best performance"
        ]
        
        # Save recommendations
        if output_format == "json":
            output_file = self.output_dir / "hardware_recommendations.json"
            with open(output_file, 'w') as f:
                json.dump(recommendations, f, indent=2)
        elif output_format == "md":
            output_file = self.output_dir / "hardware_recommendations.md"
            with open(output_file, 'w') as f:
                f.write("# Hardware Recommendations for Model Deployment\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## General Recommendations\n\n")
                for rec in recommendations["general_recommendations"]:
                    f.write(f"- {rec}\n")
                
                f.write("\n## Model Family Recommendations\n\n")
                for family, rec in recommendations["model_families"].items():
                    f.write(f"### {family.title()} Models\n\n")
                    
                    if "best_for_latency" in rec:
                        f.write(f"- **Best for Latency**: {rec['best_for_latency']['hardware']} ({rec['best_for_latency']['latency_ms']:.2f} ms)\n")
                    
                    if "best_for_throughput" in rec:
                        f.write(f"- **Best for Throughput**: {rec['best_for_throughput']['hardware']} ({rec['best_for_throughput']['throughput']:.2f} items/sec)\n")
                    
                    if "best_for_functionality" in rec:
                        f.write(f"- **Best for Reliability**: {rec['best_for_functionality']['hardware']} ({rec['best_for_functionality']['success_rate']:.1f}% success rate)\n")
                    
                    f.write("\n**Performance Across Hardware Platforms**:\n\n")
                    f.write("| Hardware | Latency (ms) | Throughput (items/sec) | Success Rate (%) |\n")
                    f.write("|----------|-------------|------------------------|------------------|\n")
                    
                    for hw, metrics in rec["all_hardware"].items():
                        latency = metrics.get("latency_ms", "N/A")
                        throughput = metrics.get("throughput", "N/A")
                        success_rate = metrics.get("success_rate", "N/A")
                        
                        latency_str = f"{latency:.2f}" if isinstance(latency, (int, float)) else latency
                        throughput_str = f"{throughput:.2f}" if isinstance(throughput, (int, float)) else throughput
                        success_rate_str = f"{success_rate:.1f}" if isinstance(success_rate, (int, float)) else success_rate
                        
                        f.write(f"| {hw} | {latency_str} | {throughput_str} | {success_rate_str} |\n")
                    
                    f.write("\n")
        else:
            # Default to HTML format
            output_file = self.output_dir / "hardware_recommendations.html"
            
            # Generate HTML output
            html_content = [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                "    <meta charset='utf-8'>",
                "    <meta name='viewport' content='width=device-width, initial-scale=1'>",
                "    <title>Hardware Recommendations for Model Deployment</title>",
                "    <style>",
                "        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }",
                "        h1 { color: #333; }",
                "        .section { margin-bottom: 30px; }",
                "        .card { border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 20px; }",
                "        .card h3 { margin-top: 0; color: #2c3e50; }",
                "        .recommendation { display: flex; margin-bottom: 10px; }",
                "        .recommendation .label { font-weight: bold; min-width: 150px; }",
                "        table { border-collapse: collapse; width: 100%; }",
                "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
                "        th { background-color: #f2f2f2; }",
                "        .best { font-weight: bold; color: #27ae60; }",
                "    </style>",
                "</head>",
                "<body>",
                "    <h1>Hardware Recommendations for Model Deployment</h1>",
                f"    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
                "    <div class='section'>",
                "        <h2>General Recommendations</h2>",
                "        <ul>"
            ]
            
            # Add general recommendations
            for rec in recommendations["general_recommendations"]:
                html_content.append(f"            <li>{rec}</li>")
            
            html_content.append("        </ul>")
            html_content.append("    </div>")
            
            # Add model family recommendations
            html_content.append("    <div class='section'>")
            html_content.append("        <h2>Model Family Recommendations</h2>")
            
            for family, rec in recommendations["model_families"].items():
                html_content.append(f"        <div class='card'>")
                html_content.append(f"            <h3>{family.title()} Models</h3>")
                
                # Add best recommendations
                if "best_for_latency" in rec:
                    html_content.append("            <div class='recommendation'>")
                    html_content.append("                <div class='label'>Best for Latency:</div>")
                    html_content.append(f"                <div class='value'>{rec['best_for_latency']['hardware']} ({rec['best_for_latency']['latency_ms']:.2f} ms)</div>")
                    html_content.append("            </div>")
                
                if "best_for_throughput" in rec:
                    html_content.append("            <div class='recommendation'>")
                    html_content.append("                <div class='label'>Best for Throughput:</div>")
                    html_content.append(f"                <div class='value'>{rec['best_for_throughput']['hardware']} ({rec['best_for_throughput']['throughput']:.2f} items/sec)</div>")
                    html_content.append("            </div>")
                
                if "best_for_functionality" in rec:
                    html_content.append("            <div class='recommendation'>")
                    html_content.append("                <div class='label'>Best for Reliability:</div>")
                    html_content.append(f"                <div class='value'>{rec['best_for_functionality']['hardware']} ({rec['best_for_functionality']['success_rate']:.1f}% success rate)</div>")
                    html_content.append("            </div>")
                
                # Add performance table
                html_content.append("            <h4>Performance Across Hardware Platforms</h4>")
                html_content.append("            <table>")
                html_content.append("                <thead>")
                html_content.append("                    <tr>")
                html_content.append("                        <th>Hardware</th>")
                html_content.append("                        <th>Latency (ms)</th>")
                html_content.append("                        <th>Throughput (items/sec)</th>")
                html_content.append("                        <th>Success Rate (%)</th>")
                html_content.append("                    </tr>")
                html_content.append("                </thead>")
                html_content.append("                <tbody>")
                
                for hw, metrics in rec["all_hardware"].items():
                    latency = metrics.get("latency_ms", "N/A")
                    throughput = metrics.get("throughput", "N/A")
                    success_rate = metrics.get("success_rate", "N/A")
                    
                    # Determine if this is the best option
                    is_best_latency = "best_for_latency" in rec and rec["best_for_latency"]["hardware"] == hw
                    is_best_throughput = "best_for_throughput" in rec and rec["best_for_throughput"]["hardware"] == hw
                    is_best_functionality = "best_for_functionality" in rec and rec["best_for_functionality"]["hardware"] == hw
                    
                    latency_class = " class='best'" if is_best_latency else ""
                    throughput_class = " class='best'" if is_best_throughput else ""
                    success_class = " class='best'" if is_best_functionality else ""
                    
                    latency_str = f"{latency:.2f}" if isinstance(latency, (int, float)) else latency
                    throughput_str = f"{throughput:.2f}" if isinstance(throughput, (int, float)) else throughput
                    success_rate_str = f"{success_rate:.1f}" if isinstance(success_rate, (int, float)) else success_rate
                    
                    html_content.append("                    <tr>")
                    html_content.append(f"                        <td>{hw}</td>")
                    html_content.append(f"                        <td{latency_class}>{latency_str}</td>")
                    html_content.append(f"                        <td{throughput_class}>{throughput_str}</td>")
                    html_content.append(f"                        <td{success_class}>{success_rate_str}</td>")
                    html_content.append("                    </tr>")
                
                html_content.append("                </tbody>")
                html_content.append("            </table>")
                html_content.append("        </div>")
            
            html_content.append("    </div>")
            html_content.append("</body>")
            html_content.append("</html>")
            
            # Write HTML to file
            with open(output_file, 'w') as f:
                f.write("\n".join(html_content))
        
        logger.info(f"Generated hardware recommendations: {output_file}")
        return str(output_file)
    
    def generate_complete_report(self):
        """
        Generate a complete report with all visualizations and recommendations.
        
        Returns:
            Dictionary with all output paths
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "hardware_comparison_plots": [],
            "functionality_plots": [],
            "dashboard": None,
            "recommendations": None
        }
        
        # Generate all visualizations
        logger.info("Generating hardware comparison plots...")
        report["hardware_comparison_plots"] = self.plot_hardware_comparison()
        
        logger.info("Generating functionality plots...")
        report["functionality_plots"] = self.plot_functionality_results()
        
        logger.info("Generating interactive dashboard...")
        report["dashboard"] = self.plot_interactive_dashboard()
        
        logger.info("Generating hardware recommendations...")
        report["recommendations"] = self.generate_hardware_recommendations()
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for model validation and benchmark results")
    parser.add_argument("--benchmark-dir", default=DEFAULT_BENCHMARK_DIR, help="Directory containing benchmark results")
    parser.add_argument("--functionality-dir", default=DEFAULT_FUNCTIONALITY_DIR, help="Directory containing functionality test results")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save visualizations")
    parser.add_argument("--plot-hardware", action="store_true", help="Generate hardware comparison plots")
    parser.add_argument("--plot-functionality", action="store_true", help="Generate functionality plots")
    parser.add_argument("--dashboard", action="store_true", help="Generate interactive dashboard")
    parser.add_argument("--recommendations", action="store_true", help="Generate hardware recommendations")
    parser.add_argument("--all", action="store_true", help="Generate all visualizations")
    parser.add_argument("--format", default="html", help="Output format for recommendations (html, json, md)")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ModelValidationVisualizer(
        benchmark_dir=args.benchmark_dir,
        functionality_dir=args.functionality_dir,
        output_dir=args.output_dir
    )
    
    if args.all:
        report = visualizer.generate_complete_report()
        logger.info(f"Generated complete report:")
        for key, value in report.items():
            if isinstance(value, list):
                logger.info(f"  {key}: {len(value)} files")
            else:
                logger.info(f"  {key}: {value}")
    else:
        if args.plot_hardware:
            visualizer.plot_hardware_comparison()
        
        if args.plot_functionality:
            visualizer.plot_functionality_results()
        
        if args.dashboard:
            visualizer.plot_interactive_dashboard()
        
        if args.recommendations:
            visualizer.generate_hardware_recommendations(output_format=args.format)

if __name__ == "__main__":
    main()