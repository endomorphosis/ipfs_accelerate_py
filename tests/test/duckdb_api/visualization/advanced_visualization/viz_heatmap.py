"""
Hardware Heatmap Visualization Component for the Advanced Visualization System.

This module provides specialized heatmap visualization capabilities for comparing
performance across hardware platforms and model families. It extends the
BaseVisualization class with hardware-specific heatmap generation.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger("hardware_heatmap")

# Import base visualization class
from duckdb_api.visualization.advanced_visualization.base import BaseVisualization, PLOTLY_AVAILABLE, MATPLOTLIB_AVAILABLE

# Check for plotly
if PLOTLY_AVAILABLE:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

# Check for matplotlib
if MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors


class HardwareHeatmapVisualization(BaseVisualization):
    """
    Hardware Heatmap Visualization Component.
    
    This component creates heatmap visualizations for comparing performance
    metrics across hardware platforms and model families.
    """
    
    def __init__(self, db_connection=None, theme="light", debug=False):
        """Initialize the hardware heatmap visualization component."""
        super().__init__(db_connection, theme, debug)
        
        # Additional configuration specific to heatmap visualizations
        self.heatmap_config = {
            "cell_height": 30,          # Height of each cell in pixels
            "cell_width": 80,           # Width of each cell in pixels
            "show_values": True,        # Show values in cells
            "show_colorbar": True,      # Show colorbar
            "colorbar_title": "",       # Title for colorbar
            "group_by_family": True,    # Group models by family
            "mark_simulated": True,     # Mark simulated results
            "simulated_marker": "S",    # Marker for simulated results
            "normalize_by_column": False,  # Normalize values by column
            "normalize_by_row": False,  # Normalize values by row
            "row_height_factor": 25,    # Factor for calculating row height
            "max_rows_per_family": 20,  # Maximum rows per family before truncating
        }
        
        # Metric-specific colormap selections
        self.metric_colormaps = {
            "throughput": "viridis",     # Higher is better
            "latency": "viridis_r",      # Lower is better
            "memory": "viridis_r",       # Lower is better
            "power": "viridis_r",        # Lower is better
            "efficiency": "viridis",     # Higher is better
        }
        
        logger.info("Hardware Heatmap Visualization component initialized")
    
    def create_visualization(self, data=None, **kwargs):
        """
        Create a hardware heatmap visualization.
        
        This is a wrapper for the more specific create_hardware_heatmap method.
        
        Args:
            data: Performance data
            **kwargs: Additional arguments passed to create_hardware_heatmap
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        return self.create_hardware_heatmap(data, **kwargs)
    
    def create_hardware_heatmap(self,
                               data=None,
                               metric="throughput",
                               model_families=None,
                               hardware_types=None,
                               batch_size=1,
                               precision="fp32",
                               output_path=None,
                               title=None,
                               **kwargs):
        """
        Create a hardware comparison heatmap grouped by model family.
        
        Args:
            data: Performance data (DataFrame, dict, or path to file)
            metric: Metric to visualize ("throughput", "latency", "memory", etc.)
            model_families: Optional list of model families to include
            hardware_types: Optional list of hardware types to include
            batch_size: Batch size to use for comparison
            precision: Precision format to use for comparison
            output_path: Optional path for saving the visualization
            title: Custom title for the visualization
            **kwargs: Additional configuration parameters
            
        Returns:
            Path to the saved heatmap visualization, or None if creation failed
        """
        # Update configuration with any provided kwargs
        heatmap_config = self.heatmap_config.copy()
        heatmap_config.update(kwargs)
        
        # Determine visualization title
        if title is None:
            title = f"{metric.capitalize()} Comparison by Hardware and Model Family (Batch Size: {batch_size})"
        
        # Load data if provided or use placeholder
        if data is None:
            # Create sample data for demonstration purposes
            # In a real implementation, this would fetch from the database
            if self.debug:
                logger.info("No data provided, using sample data for demonstration")
            
            # Sample data with hardware types and model families
            hardware_list = hardware_types or ["CPU", "GPU", "WebGPU", "WebNN", "MPS"]
            model_list = ["BERT", "ViT", "LLAMA", "Whisper", "CLIP", "T5"]
            family_list = model_families or ["Text", "Vision", "Audio"]
            
            # Create sample dataframe
            rows = []
            for family in family_list:
                # Select models for this family
                if family == "Text":
                    models = ["BERT", "LLAMA", "T5"]
                elif family == "Vision":
                    models = ["ViT", "CLIP"]
                else:
                    models = ["Whisper"]
                
                for model in models:
                    for hw in hardware_list:
                        # Generate sample metric value
                        if metric == "throughput":
                            # Higher values for GPU/WebGPU
                            base_value = 100 if hw in ["GPU", "WebGPU"] else 50
                            value = base_value * (1 + np.random.random() * 0.5)
                        elif metric == "latency":
                            # Lower values for GPU/WebGPU
                            base_value = 50 if hw in ["GPU", "WebGPU"] else 100
                            value = base_value * (0.8 + np.random.random() * 0.4)
                        else:
                            value = 50 + np.random.random() * 50
                        
                        rows.append({
                            "model_name": model,
                            "model_family": family,
                            "hardware_type": hw,
                            "batch_size": batch_size,
                            "precision": precision,
                            "throughput_items_per_second": value if metric == "throughput" else np.random.random() * 100,
                            "average_latency_ms": value if metric == "latency" else np.random.random() * 100,
                            "memory_peak_mb": value if metric == "memory" else np.random.random() * 100,
                            "is_simulated": np.random.random() > 0.7  # Random simulation flag
                        })
            
            df = pd.DataFrame(rows)
        else:
            # Load data from the provided source
            df = self.load_data(data)
            
            if df.empty:
                logger.error("Failed to load data for visualization")
                return None
        
        # Determine the metric column
        metric_column = None
        if metric == "throughput":
            metric_column = "throughput_items_per_second"
            metric_title = "Throughput (items/sec)"
        elif metric == "latency":
            metric_column = "average_latency_ms"
            metric_title = "Latency (ms)"
        elif metric == "memory":
            metric_column = "memory_peak_mb"
            metric_title = "Memory Usage (MB)"
        else:
            # Check if the metric is directly available in the dataframe
            if metric in df.columns:
                metric_column = metric
                metric_title = metric.replace("_", " ").title()
            else:
                logger.error(f"Unknown metric: {metric}")
                return None
        
        # Check if required columns are available
        required_columns = ["model_name", "hardware_type", metric_column]
        if heatmap_config["group_by_family"]:
            required_columns.append("model_family")
        if heatmap_config["mark_simulated"]:
            required_columns.append("is_simulated")
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Create the heatmap based on available visualization libraries
        if PLOTLY_AVAILABLE:
            return self._create_interactive_heatmap(
                df, metric, metric_column, metric_title, 
                output_path, title, heatmap_config
            )
        elif MATPLOTLIB_AVAILABLE:
            return self._create_static_heatmap(
                df, metric, metric_column, metric_title, 
                output_path, title, heatmap_config
            )
        else:
            logger.error("Neither Plotly nor Matplotlib is available. Cannot create visualization.")
            return None
    
    def _create_interactive_heatmap(self, df, metric, metric_column, metric_title, 
                                   output_path, title, heatmap_config):
        """
        Create an interactive heatmap using Plotly.
        
        Args:
            df: DataFrame with performance data
            metric: Metric name
            metric_column: Column name in DataFrame for the metric
            metric_title: Title for the metric (for display)
            output_path: Path to save the visualization
            title: Title for the visualization
            heatmap_config: Configuration for the heatmap
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        try:
            # Get unique model families if grouping by family
            model_families = []
            if heatmap_config["group_by_family"] and "model_family" in df.columns:
                model_families = df["model_family"].unique()
            else:
                # Use a placeholder "All Models" family
                model_families = ["All Models"]
                df["model_family"] = "All Models"
            
            # Create a subplot for each model family
            fig = make_subplots(
                rows=len(model_families),
                cols=1,
                subplot_titles=[f"{family} Models" for family in model_families],
                vertical_spacing=0.05
            )
            
            # Add annotations to show simulated results
            simulated_annotations = []
            
            # Get color scale based on metric
            if metric.lower() in self.metric_colormaps:
                color_scale = self.metric_colormaps[metric.lower()]
            else:
                color_scale = "viridis"
                # For metrics where lower is better, reverse the colorscale
                if metric.lower() in ["latency", "memory", "power"]:
                    color_scale = "viridis_r"
            
            # For each model family, create a heatmap
            for i, family in enumerate(model_families):
                family_df = df[df["model_family"] == family]
                
                # If no data for this family, skip
                if len(family_df) == 0:
                    continue
                
                # Get unique models in this family
                models = family_df["model_name"].unique()
                
                # Limit the number of models if too many
                if len(models) > heatmap_config["max_rows_per_family"]:
                    logger.warning(f"Too many models in family {family}, showing top {heatmap_config['max_rows_per_family']}")
                    # Get top models by median metric value
                    model_medians = []
                    for model in models:
                        model_df = family_df[family_df["model_name"] == model]
                        model_medians.append((model, model_df[metric_column].median()))
                    
                    # Sort based on metric (higher is better for throughput, lower for others)
                    reverse_sort = metric.lower() in ["throughput", "efficiency"]
                    model_medians.sort(key=lambda x: x[1], reverse=reverse_sort)
                    
                    # Get top models
                    models = [m[0] for m in model_medians[:heatmap_config["max_rows_per_family"]]]
                
                # Get unique hardware types
                hardware_types = family_df["hardware_type"].unique()
                
                # Create a new dataframe with all hardware types and models
                heatmap_data = np.full((len(models), len(hardware_types)), np.nan)
                
                # Fill in available data
                for j, model in enumerate(models):
                    model_df = family_df[family_df["model_name"] == model]
                    for k, hw in enumerate(hardware_types):
                        hw_df = model_df[model_df["hardware_type"] == hw]
                        if len(hw_df) > 0:
                            heatmap_data[j, k] = hw_df[metric_column].values[0]
                            
                            # Mark simulated results
                            if heatmap_config["mark_simulated"] and "is_simulated" in hw_df.columns and hw_df["is_simulated"].values[0]:
                                simulated_annotations.append(
                                    dict(
                                        x=k,
                                        y=j,
                                        xref=f"x{i+1}" if i > 0 else "x",
                                        yref=f"y{i+1}" if i > 0 else "y",
                                        text=heatmap_config["simulated_marker"],
                                        showarrow=False,
                                        font=dict(color="white", size=10),
                                        bgcolor="rgba(0,0,0,0.5)",
                                        bordercolor="white",
                                        borderwidth=1,
                                        borderpad=1,
                                        opacity=0.8
                                    )
                                )
                
                # Create hover texts for each cell
                hover_texts = []
                for j in range(len(models)):
                    hover_texts.append([])
                    for k in range(len(hardware_types)):
                        value = heatmap_data[j, k]
                        if np.isnan(value):
                            hover_texts[j].append("No data")
                        else:
                            hover_texts[j].append(
                                f"<b>Model:</b> {models[j]}<br>"
                                f"<b>Hardware:</b> {hardware_types[k]}<br>"
                                f"<b>{metric_title}:</b> {value:.2f}"
                            )
                
                # Add heatmap
                fig.add_trace(
                    go.Heatmap(
                        z=heatmap_data,
                        x=hardware_types,
                        y=models,
                        colorscale=color_scale,
                        showscale=heatmap_config["show_colorbar"],
                        colorbar=dict(
                            title=metric_title if heatmap_config["colorbar_title"] == "" else heatmap_config["colorbar_title"],
                            len=1/len(model_families),
                            y=(1-(2*i+1)/(2*len(model_families)))
                        ),
                        text=hover_texts,
                        hoverinfo="text",
                        name=family
                    ),
                    row=i+1, col=1
                )
            
            # Calculate appropriate height based on number of rows
            total_rows = sum(len(df[df["model_family"] == family]["model_name"].unique()) for family in model_families)
            height = max(600, total_rows * heatmap_config["row_height_factor"])
            
            # Add annotations for cell values if configured
            if heatmap_config["show_values"]:
                # Iterate through each family
                for i, family in enumerate(model_families):
                    family_df = df[df["model_family"] == family]
                    models = family_df["model_name"].unique()
                    hardware_types = family_df["hardware_type"].unique()
                    
                    for j, model in enumerate(models):
                        model_df = family_df[family_df["model_name"] == model]
                        for k, hw in enumerate(hardware_types):
                            hw_df = model_df[model_df["hardware_type"] == hw]
                            if len(hw_df) > 0:
                                value = hw_df[metric_column].values[0]
                                if not np.isnan(value):
                                    fig.add_annotation(
                                        x=hw,
                                        y=model,
                                        text=f"{value:.2f}",
                                        showarrow=False,
                                        font=dict(color="black" if value < 0.7 * np.nanmax(family_df[metric_column]) else "white"),
                                        xref=f"x{i+1}" if i > 0 else "x",
                                        yref=f"y{i+1}" if i > 0 else "y"
                                    )
            
            # Update layout
            fig.update_layout(
                title=title,
                height=height,
                width=1000,
                template="plotly_white" if self.theme == "light" else "plotly_dark",
                annotations=simulated_annotations if heatmap_config["mark_simulated"] else None,
                margin=dict(l=50, r=50, t=50, b=50),
            )
            
            # Update y-axis properties for each subplot
            for i in range(1, len(model_families) + 1):
                fig.update_yaxes(
                    title="Model",
                    autorange="reversed",
                    row=i, col=1
                )
            
            # Only add x-axis title to the bottom subplot
            fig.update_xaxes(
                title="Hardware Platform",
                row=len(model_families), col=1
            )
            
            # Export visualization or show it
            if output_path:
                self.figure = fig
                self.export(output_path, format="html")
                return output_path
            else:
                self.figure = fig
                return self.show()
            
        except Exception as e:
            logger.error(f"Error creating interactive heatmap: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _create_static_heatmap(self, df, metric, metric_column, metric_title, 
                              output_path, title, heatmap_config):
        """
        Create a static heatmap using Matplotlib.
        
        Args:
            df: DataFrame with performance data
            metric: Metric name
            metric_column: Column name in DataFrame for the metric
            metric_title: Title for the metric (for display)
            output_path: Path to save the visualization
            title: Title for the visualization
            heatmap_config: Configuration for the heatmap
            
        Returns:
            Path to the saved visualization, or None if creation failed
        """
        try:
            # Get unique model families if grouping by family
            model_families = []
            if heatmap_config["group_by_family"] and "model_family" in df.columns:
                model_families = df["model_family"].unique()
            else:
                # Use a placeholder "All Models" family
                model_families = ["All Models"]
                df["model_family"] = "All Models"
            
            # Calculate total rows and figure size
            total_rows = sum(len(df[df["model_family"] == family]["model_name"].unique()) for family in model_families)
            fig_height = max(6, total_rows * 0.4 + len(model_families) * 1.0)  # Base height + rows + space for titles
            
            # Create figure with subplots
            fig, axes = plt.subplots(
                len(model_families), 1,
                figsize=(10, fig_height),
                squeeze=False
            )
            
            # Get colormap based on metric
            if metric.lower() in self.metric_colormaps:
                cmap_name = self.metric_colormaps[metric.lower()]
            else:
                cmap_name = "viridis"
                # For metrics where lower is better, reverse the colormap
                if metric.lower() in ["latency", "memory", "power"]:
                    cmap_name = "viridis_r"
            
            cmap = plt.get_cmap(cmap_name)
            
            # Process each model family
            for i, family in enumerate(model_families):
                ax = axes[i, 0]
                family_df = df[df["model_family"] == family]
                
                # If no data for this family, skip
                if len(family_df) == 0:
                    ax.text(0.5, 0.5, f"No data for {family}", 
                          ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{family} Models")
                    continue
                
                # Get unique models in this family
                models = family_df["model_name"].unique()
                
                # Limit the number of models if too many
                if len(models) > heatmap_config["max_rows_per_family"]:
                    logger.warning(f"Too many models in family {family}, showing top {heatmap_config['max_rows_per_family']}")
                    # Get top models by median metric value
                    model_medians = []
                    for model in models:
                        model_df = family_df[family_df["model_name"] == model]
                        model_medians.append((model, model_df[metric_column].median()))
                    
                    # Sort based on metric (higher is better for throughput, lower for others)
                    reverse_sort = metric.lower() in ["throughput", "efficiency"]
                    model_medians.sort(key=lambda x: x[1], reverse=reverse_sort)
                    
                    # Get top models
                    models = [m[0] for m in model_medians[:heatmap_config["max_rows_per_family"]]]
                
                # Get unique hardware types
                hardware_types = family_df["hardware_type"].unique()
                
                # Create a new dataframe with all hardware types and models
                heatmap_data = np.full((len(models), len(hardware_types)), np.nan)
                simulated_mask = np.zeros((len(models), len(hardware_types)), dtype=bool)
                
                # Fill in available data
                for j, model in enumerate(models):
                    model_df = family_df[family_df["model_name"] == model]
                    for k, hw in enumerate(hardware_types):
                        hw_df = model_df[model_df["hardware_type"] == hw]
                        if len(hw_df) > 0:
                            heatmap_data[j, k] = hw_df[metric_column].values[0]
                            
                            # Mark simulated results
                            if heatmap_config["mark_simulated"] and "is_simulated" in hw_df.columns:
                                simulated_mask[j, k] = hw_df["is_simulated"].values[0]
                
                # Create heatmap
                im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', interpolation='nearest')
                
                # Add colorbar if requested
                if heatmap_config["show_colorbar"]:
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label(metric_title if heatmap_config["colorbar_title"] == "" else heatmap_config["colorbar_title"])
                
                # Set axis labels and ticks
                ax.set_xticks(np.arange(len(hardware_types)))
                ax.set_yticks(np.arange(len(models)))
                ax.set_xticklabels(hardware_types)
                ax.set_yticklabels(models)
                
                # Rotate x-axis labels for better readability
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                # Add title for this family
                ax.set_title(f"{family} Models")
                
                # Add values in cells if requested
                if heatmap_config["show_values"]:
                    for j in range(len(models)):
                        for k in range(len(hardware_types)):
                            value = heatmap_data[j, k]
                            if not np.isnan(value):
                                # Determine text color based on cell color
                                cell_color = im.cmap(im.norm(value))
                                text_color = "white" if np.mean(cell_color[:3]) < 0.5 else "black"
                                
                                # Add value text
                                ax.text(k, j, f"{value:.2f}", 
                                      ha="center", va="center", color=text_color, fontsize=8)
                
                # Add markers for simulated results if requested
                if heatmap_config["mark_simulated"]:
                    for j in range(len(models)):
                        for k in range(len(hardware_types)):
                            if simulated_mask[j, k]:
                                ax.text(k, j, heatmap_config["simulated_marker"], 
                                      ha="right", va="top", color="white", fontsize=8,
                                      bbox=dict(boxstyle="round,pad=0.1", fc="rgba(0,0,0,0.5)", ec="none"))
            
            # Set x-axis label on the bottom subplot
            axes[-1, 0].set_xlabel("Hardware Platform")
            
            # Set common y-axis label
            fig.text(0.04, 0.5, "Model", va='center', rotation='vertical')
            
            # Add overall title
            fig.suptitle(title, fontsize=14)
            
            # Add legend for simulated results if needed
            if heatmap_config["mark_simulated"] and np.any(simulated_mask):
                fig.text(0.95, 0.02, f"{heatmap_config['simulated_marker']} = Simulated Result", 
                       ha="right", va="bottom", fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
            
            # Adjust layout
            plt.tight_layout(rect=[0.05, 0.02, 0.95, 0.95])  # Adjust for overall title and y-label
            
            # Export visualization or show it
            if output_path:
                plt.savefig(output_path, dpi=100, bbox_inches="tight")
                plt.close(fig)
                return output_path
            else:
                self.figure = fig
                plt.show()
                return True
            
        except Exception as e:
            logger.error(f"Error creating static heatmap: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None