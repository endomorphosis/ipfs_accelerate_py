#!/usr/bin/env python3
"""
Advanced Visualization Tools for the Predictive Performance System.

This module provides sophisticated visualization capabilities for the Predictive
Performance System, including 3D visualizations, interactive dashboards,
time-series performance tracking, and more.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from datetime import datetime

# Optional imports for interactive visualizations
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Define constants
DEFAULT_STYLE = "whitegrid"
DEFAULT_CONTEXT = "paper"
DEFAULT_PALETTE = "viridis"
DEFAULT_FIGURE_SIZE = (12, 8)
DEFAULT_DPI = 100
DEFAULT_FORMAT = "png"
DEFAULT_METRICS = ["throughput", "latency_mean", "memory_usage"]
DEFAULT_3D_METRICS = ["throughput", "latency_mean", "memory_usage"]

class AdvancedVisualization:
    """Advanced visualization tools for the Predictive Performance System."""
    
    def __init__(
        self,
        style: str = DEFAULT_STYLE,
        context: str = DEFAULT_CONTEXT,
        palette: str = DEFAULT_PALETTE,
        figure_size: Tuple[int, int] = DEFAULT_FIGURE_SIZE,
        dpi: int = DEFAULT_DPI,
        output_format: str = DEFAULT_FORMAT,
        output_dir: Optional[str] = None,
        interactive: bool = True
    ):
        """
        Initialize the visualization system.
        
        Args:
            style: Seaborn style to use for static plots
            context: Seaborn context to use for static plots
            palette: Color palette to use for plots
            figure_size: Default figure size for static plots
            dpi: DPI for static plots
            output_format: Format to save static plots ('png', 'pdf', 'svg', etc.)
            output_dir: Directory to save output files
            interactive: Whether to use interactive visualizations when possible
        """
        self.style = style
        self.context = context
        self.palette = palette
        self.figure_size = figure_size
        self.dpi = dpi
        self.output_format = output_format
        self.output_dir = output_dir or Path("./visualizations")
        self.interactive = interactive and PLOTLY_AVAILABLE
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up matplotlib style
        sns.set_style(self.style)
        sns.set_context(self.context)
        plt.rcParams["figure.figsize"] = self.figure_size
        plt.rcParams["figure.dpi"] = self.dpi
    
    def _prepare_data(self, data: Union[pd.DataFrame, Dict, str]) -> pd.DataFrame:
        """
        Prepare data for visualization.
        
        Args:
            data: Data to visualize (DataFrame, dict, or path to JSON/CSV file)
            
        Returns:
            Prepared DataFrame
        """
        if isinstance(data, str):
            # Load from file
            path = Path(data)
            if path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    data = json.load(f)
                    return pd.json_normalize(data)
            elif path.suffix.lower() in ['.csv', '.tsv']:
                return pd.read_csv(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        elif isinstance(data, dict):
            # Convert dict to DataFrame
            return pd.DataFrame.from_dict(data)
        
        elif isinstance(data, pd.DataFrame):
            # Already a DataFrame
            return data.copy()
        
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def create_3d_visualization(
        self,
        data: Union[pd.DataFrame, Dict, str],
        x_metric: str = "batch_size",
        y_metric: str = "model_name",
        z_metric: str = "throughput",
        color_metric: Optional[str] = "hardware",
        size_metric: Optional[str] = "confidence",
        title: str = "3D Performance Visualization",
        output_file: Optional[str] = None,
        interactive: Optional[bool] = None
    ) -> str:
        """
        Create a 3D visualization of performance data.
        
        Args:
            data: Performance data to visualize
            x_metric: Metric to use for x-axis
            y_metric: Metric to use for y-axis
            z_metric: Metric to use for z-axis
            color_metric: Metric to use for point color
            size_metric: Metric to use for point size
            title: Plot title
            output_file: Output file path (generated if None)
            interactive: Whether to create an interactive plot (overrides instance setting)
            
        Returns:
            Path to output file
        """
        # Prepare data
        df = self._prepare_data(data)
        
        # Determine if we should use interactive visualization
        use_interactive = interactive if interactive is not None else self.interactive
        
        if use_interactive and PLOTLY_AVAILABLE:
            # Create interactive 3D plot with Plotly
            fig = px.scatter_3d(
                df,
                x=x_metric,
                y=y_metric,
                z=z_metric,
                color=color_metric,
                size=size_metric,
                title=title,
                opacity=0.7
            )
            
            # Update layout
            fig.update_layout(
                scene = dict(
                    xaxis_title=x_metric,
                    yaxis_title=y_metric,
                    zaxis_title=z_metric
                ),
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            # Determine output file
            if output_file is None:
                output_file = f"3d_visualization_{x_metric}_{y_metric}_{z_metric}.html"
            output_path = Path(self.output_dir) / output_file
            
            # Save figure
            fig.write_html(str(output_path))
            
        else:
            # Create static 3D plot with Matplotlib
            fig = plt.figure(figsize=self.figure_size)
            ax = fig.add_subplot(111, projection='3d')
            
            # If color_metric is specified, use it for coloring points
            if color_metric and color_metric in df.columns:
                # Create a colormap for categorical data
                categories = df[color_metric].unique()
                colors = cm.viridis(np.linspace(0, 1, len(categories)))
                colormap = dict(zip(categories, colors))
                point_colors = [colormap[val] for val in df[color_metric]]
            else:
                point_colors = 'blue'
            
            # If size_metric is specified, use it for point sizes
            if size_metric and size_metric in df.columns:
                # Normalize sizes between 20 and 200
                min_size = 20
                max_size = 200
                sizes = df[size_metric].values
                if sizes.max() > sizes.min():
                    sizes = min_size + (max_size - min_size) * (sizes - sizes.min()) / (sizes.max() - sizes.min())
                else:
                    sizes = np.full(len(sizes), (min_size + max_size) / 2)
            else:
                sizes = 100
            
            # Create the scatter plot
            scatter = ax.scatter(
                df[x_metric],
                df[y_metric],
                df[z_metric],
                c=point_colors,
                s=sizes,
                alpha=0.7
            )
            
            # Add labels and title
            ax.set_xlabel(x_metric)
            ax.set_ylabel(y_metric)
            ax.set_zlabel(z_metric)
            ax.set_title(title)
            
            # Add a colorbar if using color_metric
            if color_metric and color_metric in df.columns:
                # Create a legend for categorical data
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  label=cat, markerfacecolor=colormap[cat], markersize=10)
                          for cat in categories]
                ax.legend(handles=legend_elements, title=color_metric)
            
            # Determine output file
            if output_file is None:
                output_file = f"3d_visualization_{x_metric}_{y_metric}_{z_metric}.{self.output_format}"
            output_path = Path(self.output_dir) / output_file
            
            # Save figure
            plt.tight_layout()
            plt.savefig(str(output_path), dpi=self.dpi)
            plt.close()
        
        return str(output_path)
    
    def create_performance_dashboard(
        self,
        data: Union[pd.DataFrame, Dict, str],
        metrics: List[str] = DEFAULT_METRICS,
        groupby: List[str] = ["model_name", "hardware"],
        title: str = "Performance Dashboard",
        output_file: Optional[str] = None,
        interactive: Optional[bool] = None
    ) -> str:
        """
        Create a comprehensive performance dashboard with multiple visualizations.
        
        Args:
            data: Performance data to visualize
            metrics: Performance metrics to include in the dashboard
            groupby: Columns to group by for comparison
            title: Dashboard title
            output_file: Output file path (generated if None)
            interactive: Whether to create an interactive dashboard (overrides instance setting)
            
        Returns:
            Path to output file
        """
        # Prepare data
        df = self._prepare_data(data)
        
        # Determine if we should use interactive visualization
        use_interactive = interactive if interactive is not None else self.interactive
        
        if use_interactive and PLOTLY_AVAILABLE:
            # Create interactive dashboard with Plotly
            num_metrics = len(metrics)
            
            # Create subplots grid
            fig = make_subplots(
                rows=num_metrics,
                cols=1,
                subplot_titles=[f"{metric.replace('_', ' ').title()} by {' and '.join(groupby)}" for metric in metrics],
                vertical_spacing=0.1
            )
            
            # Add traces for each metric
            for i, metric in enumerate(metrics):
                # Group by specified columns and calculate mean values
                grouped_df = df.groupby(groupby)[metric].mean().reset_index()
                
                # Create bars for this metric
                if len(groupby) >= 2:
                    primary_groups = grouped_df[groupby[0]].unique()
                    secondary_groups = grouped_df[groupby[1]].unique()
                    
                    for j, secondary_value in enumerate(secondary_groups):
                        filtered_df = grouped_df[grouped_df[groupby[1]] == secondary_value]
                        
                        fig.add_trace(
                            go.Bar(
                                x=filtered_df[groupby[0]],
                                y=filtered_df[metric],
                                name=f"{groupby[1]}={secondary_value}",
                                legendgroup=secondary_value,
                                showlegend=(i == 0)  # Only show in legend for first metric
                            ),
                            row=i+1,
                            col=1
                        )
                else:
                    # Simple bar chart for single groupby
                    fig.add_trace(
                        go.Bar(
                            x=grouped_df[groupby[0]],
                            y=grouped_df[metric],
                            name=metric
                        ),
                        row=i+1,
                        col=1
                    )
            
            # Update layout
            fig.update_layout(
                title_text=title,
                height=300 * num_metrics,
                width=1000,
                barmode='group'
            )
            
            # Determine output file
            if output_file is None:
                output_file = f"performance_dashboard_{'_'.join(groupby)}.html"
            output_path = Path(self.output_dir) / output_file
            
            # Save figure
            fig.write_html(str(output_path))
            
        else:
            # Create static dashboard with Matplotlib
            num_metrics = len(metrics)
            
            # Create figure and axes grid
            fig, axes = plt.subplots(num_metrics, 1, figsize=(self.figure_size[0], self.figure_size[1] * num_metrics / 2), 
                                      dpi=self.dpi, constrained_layout=True)
            
            # Convert to array if only one metric
            if num_metrics == 1:
                axes = np.array([axes])
                
            # Add title to the figure
            fig.suptitle(title, fontsize=16)
            
            # Create plots for each metric
            for i, metric in enumerate(metrics):
                ax = axes[i]
                
                # Group by specified columns and calculate mean values
                grouped_df = df.groupby(groupby)[metric].mean().reset_index()
                
                # Create bars for this metric
                if len(groupby) >= 2:
                    # Create a pivot table for grouped bar chart
                    pivot_df = grouped_df.pivot(index=groupby[0], columns=groupby[1], values=metric)
                    pivot_df.plot(kind='bar', ax=ax, rot=45)
                    ax.set_title(f"{metric.replace('_', ' ').title()} by {' and '.join(groupby)}")
                    ax.set_ylabel(metric)
                    ax.legend(title=groupby[1])
                else:
                    # Simple bar chart for single groupby
                    sns.barplot(x=groupby[0], y=metric, data=grouped_df, ax=ax)
                    ax.set_title(f"{metric.replace('_', ' ').title()} by {groupby[0]}")
                    ax.set_ylabel(metric)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Adjust layout
            fig.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
            
            # Determine output file
            if output_file is None:
                output_file = f"performance_dashboard_{'_'.join(groupby)}.{self.output_format}"
            output_path = Path(self.output_dir) / output_file
            
            # Save figure
            plt.savefig(str(output_path), dpi=self.dpi)
            plt.close()
        
        return str(output_path)
    
    def create_time_series_visualization(
        self,
        data: Union[pd.DataFrame, Dict, str],
        time_column: str = "timestamp",
        metric: str = "throughput",
        groupby: List[str] = ["model_name", "hardware"],
        include_trend: bool = True,
        window_size: int = 5,
        title: str = "Performance Over Time",
        output_file: Optional[str] = None,
        interactive: Optional[bool] = None
    ) -> str:
        """
        Create a time-series visualization showing performance trends over time.
        
        Args:
            data: Performance data to visualize, must include a timestamp column
            time_column: Name of the column containing timestamps
            metric: Performance metric to visualize
            groupby: Columns to group by for comparison
            include_trend: Whether to include trend lines
            window_size: Window size for rolling average trend line
            title: Plot title
            output_file: Output file path (generated if None)
            interactive: Whether to create an interactive plot (overrides instance setting)
            
        Returns:
            Path to output file
        """
        # Prepare data
        df = self._prepare_data(data)
        
        # Ensure time column exists and convert to datetime if needed
        if time_column not in df.columns:
            raise ValueError(f"Time column '{time_column}' not found in data")
        
        if not pd.api.types.is_datetime64_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(df[time_column])
        
        # Sort by time
        df = df.sort_values(time_column)
        
        # Determine if we should use interactive visualization
        use_interactive = interactive if interactive is not None else self.interactive
        
        if use_interactive and PLOTLY_AVAILABLE:
            # Create interactive time-series with Plotly
            fig = go.Figure()
            
            # Group by specified columns
            for group_name, group_df in df.groupby(groupby):
                # Create group label
                if isinstance(group_name, tuple):
                    # Multiple groupby columns
                    label = " - ".join([f"{col}={val}" for col, val in zip(groupby, group_name)])
                else:
                    # Single groupby column
                    label = f"{groupby[0]}={group_name}"
                
                # Add scatter plot for this group
                fig.add_trace(
                    go.Scatter(
                        x=group_df[time_column],
                        y=group_df[metric],
                        mode='markers+lines',
                        name=label,
                        line=dict(width=1),
                        marker=dict(size=8)
                    )
                )
                
                # Add trend line if requested
                if include_trend and len(group_df) >= window_size:
                    # Calculate rolling average
                    group_df = group_df.copy()
                    group_df[f"{metric}_trend"] = group_df[metric].rolling(window=window_size, min_periods=1).mean()
                    
                    # Add trend line
                    fig.add_trace(
                        go.Scatter(
                            x=group_df[time_column],
                            y=group_df[f"{metric}_trend"],
                            mode='lines',
                            name=f"{label} (Trend)",
                            line=dict(width=3, dash='dash')
                        )
                    )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=time_column,
                yaxis_title=metric,
                legend_title_text=" & ".join(groupby),
                width=1000,
                height=600
            )
            
            # Determine output file
            if output_file is None:
                output_file = f"time_series_{metric}_{'_'.join(groupby)}.html"
            output_path = Path(self.output_dir) / output_file
            
            # Save figure
            fig.write_html(str(output_path))
            
        else:
            # Create static time-series with Matplotlib
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Generate color palette
            group_count = df.groupby(groupby).ngroups
            palette = sns.color_palette(self.palette, n_colors=group_count)
            
            # Plot each group
            for i, (group_name, group_df) in enumerate(df.groupby(groupby)):
                # Create group label
                if isinstance(group_name, tuple):
                    # Multiple groupby columns
                    label = " - ".join([f"{col}={val}" for col, val in zip(groupby, group_name)])
                else:
                    # Single groupby column
                    label = f"{groupby[0]}={group_name}"
                
                # Plot scatter + line for this group
                ax.plot(
                    group_df[time_column],
                    group_df[metric],
                    marker='o',
                    linestyle='-',
                    label=label,
                    alpha=0.7,
                    color=palette[i]
                )
                
                # Add trend line if requested
                if include_trend and len(group_df) >= window_size:
                    # Calculate rolling average
                    group_df = group_df.copy()
                    group_df[f"{metric}_trend"] = group_df[metric].rolling(window=window_size, min_periods=1).mean()
                    
                    # Plot trend line
                    ax.plot(
                        group_df[time_column],
                        group_df[f"{metric}_trend"],
                        linestyle='--',
                        linewidth=2,
                        color=palette[i],
                        alpha=0.9
                    )
            
            # Set labels and title
            ax.set_xlabel(time_column)
            ax.set_ylabel(metric)
            ax.set_title(title)
            
            # Format x-axis to look nice with dates
            plt.gcf().autofmt_xdate()
            
            # Add legend
            ax.legend(title=" & ".join(groupby))
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Determine output file
            if output_file is None:
                output_file = f"time_series_{metric}_{'_'.join(groupby)}.{self.output_format}"
            output_path = Path(self.output_dir) / output_file
            
            # Save figure
            plt.savefig(str(output_path), dpi=self.dpi)
            plt.close()
        
        return str(output_path)
    
    def create_power_efficiency_visualization(
        self,
        data: Union[pd.DataFrame, Dict, str],
        performance_metric: str = "throughput",
        power_metric: str = "power_consumption",
        groupby: List[str] = ["model_name", "hardware"],
        title: str = "Power Efficiency Analysis",
        output_file: Optional[str] = None,
        interactive: Optional[bool] = None
    ) -> str:
        """
        Create a Sankey diagram or other visualization for power efficiency analysis.
        
        Args:
            data: Performance data to visualize
            performance_metric: Name of the performance metric column
            power_metric: Name of the power consumption metric column
            groupby: Columns to group by for comparison
            title: Plot title
            output_file: Output file path (generated if None)
            interactive: Whether to create an interactive plot (overrides instance setting)
            
        Returns:
            Path to output file
        """
        # Prepare data
        df = self._prepare_data(data)
        
        # Calculate efficiency = performance / power
        if power_metric in df.columns and performance_metric in df.columns:
            df["efficiency"] = df[performance_metric] / df[power_metric]
        else:
            raise ValueError(f"Performance metric '{performance_metric}' or power metric '{power_metric}' not found in data")
        
        # Determine if we should use interactive visualization
        use_interactive = interactive if interactive is not None else self.interactive
        
        if use_interactive and PLOTLY_AVAILABLE:
            # Create interactive power efficiency visualization with Plotly
            # Group by specified columns and calculate mean values
            grouped_df = df.groupby(groupby)[["efficiency", performance_metric, power_metric]].mean().reset_index()
            
            # Create scatter plot with efficiency
            fig = px.scatter(
                grouped_df,
                x=power_metric,
                y=performance_metric,
                size="efficiency",
                color=groupby[0] if len(groupby) > 0 else None,
                symbol=groupby[1] if len(groupby) > 1 else None,
                hover_data=groupby + ["efficiency"],
                log_x=True,
                log_y=True,
                title=title
            )
            
            # Add efficiency contour lines
            power_range = np.logspace(
                np.log10(grouped_df[power_metric].min() * 0.8),
                np.log10(grouped_df[power_metric].max() * 1.2),
                100
            )
            
            # Add multiple efficiency contours
            eff_values = np.percentile(grouped_df["efficiency"], [10, 25, 50, 75, 90])
            for eff in eff_values:
                perf_values = eff * power_range
                fig.add_trace(
                    go.Scatter(
                        x=power_range,
                        y=perf_values,
                        mode="lines",
                        line=dict(dash="dash", width=1),
                        name=f"Efficiency = {eff:.2f}",
                        hoverinfo="name",
                        showlegend=True
                    )
                )
            
            # Update layout
            fig.update_layout(
                xaxis_title=f"{power_metric} (log scale)",
                yaxis_title=f"{performance_metric} (log scale)",
                width=1000,
                height=700
            )
            
            # Determine output file
            if output_file is None:
                output_file = f"power_efficiency_{'_'.join(groupby)}.html"
            output_path = Path(self.output_dir) / output_file
            
            # Save figure
            fig.write_html(str(output_path))
            
        else:
            # Create static power efficiency visualization with Matplotlib
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Group by specified columns and calculate mean values
            grouped_df = df.groupby(groupby)[["efficiency", performance_metric, power_metric]].mean().reset_index()
            
            # Generate color palette based on first groupby column
            if len(groupby) > 0:
                first_groups = grouped_df[groupby[0]].unique()
                color_palette = sns.color_palette(self.palette, n_colors=len(first_groups))
                color_map = dict(zip(first_groups, color_palette))
                
                # Generate markers based on second groupby column (if available)
                markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', 'p']
                if len(groupby) > 1:
                    second_groups = grouped_df[groupby[1]].unique()
                    marker_map = dict(zip(second_groups, markers[:len(second_groups)]))
                else:
                    marker_map = {'all': 'o'}
                
                # Create scatter plot
                for (first_val, second_val), group in grouped_df.groupby(groupby[:2]) if len(groupby) > 1 else grouped_df.groupby(groupby[0]):
                    marker = marker_map[second_val] if len(groupby) > 1 else marker_map['all']
                    label = f"{groupby[0]}={first_val}" + (f", {groupby[1]}={second_val}" if len(groupby) > 1 else "")
                    
                    # Plot point with size proportional to efficiency
                    sizes = group["efficiency"] / grouped_df["efficiency"].max() * 300 + 50
                    
                    ax.scatter(
                        group[power_metric],
                        group[performance_metric],
                        s=sizes,
                        color=color_map[first_val],
                        marker=marker,
                        alpha=0.7,
                        edgecolor='black',
                        linewidth=1,
                        label=label
                    )
            else:
                # Simple scatter plot if no groupby
                sizes = grouped_df["efficiency"] / grouped_df["efficiency"].max() * 300 + 50
                ax.scatter(
                    grouped_df[power_metric],
                    grouped_df[performance_metric],
                    s=sizes,
                    alpha=0.7,
                    edgecolor='black',
                    linewidth=1
                )
            
            # Add efficiency contour lines
            power_range = np.logspace(
                np.log10(grouped_df[power_metric].min() * 0.8),
                np.log10(grouped_df[power_metric].max() * 1.2),
                100
            )
            
            # Add multiple efficiency contours
            eff_values = np.percentile(grouped_df["efficiency"], [10, 25, 50, 75, 90])
            for eff in eff_values:
                perf_values = eff * power_range
                ax.plot(
                    power_range,
                    perf_values,
                    'k--',
                    alpha=0.5,
                    linewidth=1
                )
                # Add text label at the middle of the line
                middle_idx = len(power_range) // 2
                ax.text(
                    power_range[middle_idx],
                    perf_values[middle_idx],
                    f" Eff = {eff:.2f}",
                    fontsize=8,
                    alpha=0.7
                )
            
            # Set scales to logarithmic
            ax.set_xscale('log')
            ax.set_yscale('log')
            
            # Set labels and title
            ax.set_xlabel(f"{power_metric} (log scale)")
            ax.set_ylabel(f"{performance_metric} (log scale)")
            ax.set_title(title)
            
            # Add grid
            ax.grid(True, alpha=0.3, which='both')
            
            # Add legend
            if len(groupby) > 0:
                ax.legend(title=" & ".join(groupby), loc='upper left', bbox_to_anchor=(1, 1))
            
            # Adjust layout to make room for legend
            plt.tight_layout(rect=[0, 0, 0.85, 1] if len(groupby) > 0 else None)
            
            # Determine output file
            if output_file is None:
                output_file = f"power_efficiency_{'_'.join(groupby)}.{self.output_format}"
            output_path = Path(self.output_dir) / output_file
            
            # Save figure
            plt.savefig(str(output_path), dpi=self.dpi)
            plt.close()
        
        return str(output_path)
    
    def create_dimension_reduction_visualization(
        self,
        data: Union[pd.DataFrame, Dict, str],
        features: List[str],
        target: str = "throughput",
        method: str = "pca",
        n_components: int = 2,
        groupby: str = "model_type",
        title: str = "Feature Importance Visualization",
        output_file: Optional[str] = None,
        interactive: Optional[bool] = None
    ) -> str:
        """
        Create a dimension reduction visualization (PCA or t-SNE) to show feature importance.
        
        Args:
            data: Performance data to visualize
            features: List of feature columns to include
            target: Target metric for coloring points
            method: Dimension reduction method ('pca' or 'tsne')
            n_components: Number of components for dimension reduction
            groupby: Column to group by for coloring
            title: Plot title
            output_file: Output file path (generated if None)
            interactive: Whether to create an interactive plot (overrides instance setting)
            
        Returns:
            Path to output file
        """
        # Prepare data
        df = self._prepare_data(data)
        
        # Ensure all features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found in data: {missing_features}")
        
        # Extract features and convert categorical columns to numeric
        X = pd.get_dummies(df[features], drop_first=True)
        
        # Apply dimension reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components)
            reducer_name = 'PCA'
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
            reducer_name = 't-SNE'
        else:
            raise ValueError(f"Unknown dimension reduction method: {method}")
        
        # Apply transformation
        X_reduced = reducer.fit_transform(X)
        
        # Create a new DataFrame with reduced dimensions
        reduced_df = pd.DataFrame(X_reduced, columns=[f"{reducer_name}{i+1}" for i in range(n_components)])
        
        # Add target and groupby columns
        reduced_df[target] = df[target].values if target in df.columns else np.zeros(len(reduced_df))
        if groupby in df.columns:
            reduced_df[groupby] = df[groupby].values
        
        # Determine if we should use interactive visualization
        use_interactive = interactive if interactive is not None else self.interactive
        
        if use_interactive and PLOTLY_AVAILABLE:
            # Create interactive dimension reduction plot with Plotly
            if n_components == 2:
                # 2D scatter plot
                fig = px.scatter(
                    reduced_df,
                    x=f"{reducer_name}1",
                    y=f"{reducer_name}2",
                    color=groupby if groupby in reduced_df.columns else None,
                    size=target if target in reduced_df.columns else None,
                    hover_data=[groupby, target] if groupby in reduced_df.columns and target in reduced_df.columns else None,
                    title=title
                )
                
                # Add feature projection vectors if using PCA
                if method.lower() == 'pca':
                    # Get PCA components
                    pca_components = reducer.components_
                    feature_names = X.columns
                    
                    # Scale the vectors for visibility
                    scale_factor = 3
                    
                    # Add vectors
                    for i, feature in enumerate(feature_names):
                        fig.add_annotation(
                            x=0,
                            y=0,
                            ax=pca_components[0, i] * scale_factor,
                            ay=pca_components[1, i] * scale_factor,
                            xref="x", yref="y",
                            axref="x", ayref="y",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor="black"
                        )
                        fig.add_annotation(
                            x=pca_components[0, i] * scale_factor,
                            y=pca_components[1, i] * scale_factor,
                            text=feature,
                            showarrow=False,
                            font=dict(size=10)
                        )
                
            elif n_components == 3:
                # 3D scatter plot
                fig = px.scatter_3d(
                    reduced_df,
                    x=f"{reducer_name}1",
                    y=f"{reducer_name}2",
                    z=f"{reducer_name}3",
                    color=groupby if groupby in reduced_df.columns else None,
                    size=target if target in reduced_df.columns else None,
                    hover_data=[groupby, target] if groupby in reduced_df.columns and target in reduced_df.columns else None,
                    title=title
                )
            else:
                raise ValueError(f"Interactive visualization only supports 2 or 3 components, got {n_components}")
            
            # Update layout
            fig.update_layout(
                width=1000,
                height=700
            )
            
            # Determine output file
            if output_file is None:
                output_file = f"dimension_reduction_{method}_{n_components}d.html"
            output_path = Path(self.output_dir) / output_file
            
            # Save figure
            fig.write_html(str(output_path))
            
        else:
            # Create static dimension reduction plot with Matplotlib
            if n_components == 2:
                # 2D scatter plot
                fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
                
                # Create scatter plot with groupby as color
                if groupby in reduced_df.columns:
                    # Group points by category
                    for group_name, group_df in reduced_df.groupby(groupby):
                        ax.scatter(
                            group_df[f"{reducer_name}1"],
                            group_df[f"{reducer_name}2"],
                            label=group_name,
                            alpha=0.7,
                            s=group_df[target] / reduced_df[target].max() * 100 + 30 if target in reduced_df.columns else 50
                        )
                    ax.legend(title=groupby)
                else:
                    # Simple scatter plot
                    scatter = ax.scatter(
                        reduced_df[f"{reducer_name}1"],
                        reduced_df[f"{reducer_name}2"],
                        c=reduced_df[target] if target in reduced_df.columns else 'blue',
                        alpha=0.7,
                        s=50,
                        cmap='viridis'
                    )
                    if target in reduced_df.columns:
                        plt.colorbar(scatter, label=target)
                
                # Add feature projection vectors if using PCA
                if method.lower() == 'pca':
                    # Get PCA components
                    pca_components = reducer.components_
                    feature_names = X.columns
                    
                    # Scale the vectors for visibility
                    scale_factor = 3
                    
                    # Add vectors
                    for i, feature in enumerate(feature_names):
                        ax.arrow(0, 0, pca_components[0, i] * scale_factor, pca_components[1, i] * scale_factor,
                                head_width=0.2, head_length=0.2, fc='black', ec='black', alpha=0.7)
                        ax.text(pca_components[0, i] * scale_factor * 1.1, pca_components[1, i] * scale_factor * 1.1,
                               feature, fontsize=9)
                
                # Set labels and title
                ax.set_xlabel(f"{reducer_name} Component 1")
                ax.set_ylabel(f"{reducer_name} Component 2")
                ax.set_title(title)
                
                # Add grid
                ax.grid(True, alpha=0.3)
                
            elif n_components == 3:
                # 3D scatter plot
                fig = plt.figure(figsize=self.figure_size, dpi=self.dpi)
                ax = fig.add_subplot(111, projection='3d')
                
                # Create scatter plot with groupby as color
                if groupby in reduced_df.columns:
                    # Group points by category
                    for group_name, group_df in reduced_df.groupby(groupby):
                        ax.scatter(
                            group_df[f"{reducer_name}1"],
                            group_df[f"{reducer_name}2"],
                            group_df[f"{reducer_name}3"],
                            label=group_name,
                            alpha=0.7,
                            s=group_df[target] / reduced_df[target].max() * 100 + 30 if target in reduced_df.columns else 50
                        )
                    ax.legend(title=groupby)
                else:
                    # Simple scatter plot
                    scatter = ax.scatter(
                        reduced_df[f"{reducer_name}1"],
                        reduced_df[f"{reducer_name}2"],
                        reduced_df[f"{reducer_name}3"],
                        c=reduced_df[target] if target in reduced_df.columns else 'blue',
                        alpha=0.7,
                        s=50,
                        cmap='viridis'
                    )
                    if target in reduced_df.columns:
                        plt.colorbar(scatter, label=target)
                
                # Set labels and title
                ax.set_xlabel(f"{reducer_name} Component 1")
                ax.set_ylabel(f"{reducer_name} Component 2")
                ax.set_zlabel(f"{reducer_name} Component 3")
                ax.set_title(title)
                
            else:
                # For more than 3 components, create a matrix of scatter plots
                n_plots = min(6, n_components)  # Limit to 6 components for visibility
                fig, axes = plt.subplots(n_plots, n_plots, figsize=(15, 15), dpi=self.dpi)
                
                # Flatten axes array for easier indexing
                axes = axes.flatten()
                
                # Create scatter plots for each pair of components
                for i in range(n_plots):
                    for j in range(n_plots):
                        # Skip diagonal plots
                        if i == j:
                            axes[i * n_plots + j].set_visible(False)
                            continue
                        
                        ax = axes[i * n_plots + j]
                        
                        # Create scatter plot with groupby as color
                        if groupby in reduced_df.columns:
                            # Group points by category
                            for group_name, group_df in reduced_df.groupby(groupby):
                                ax.scatter(
                                    group_df[f"{reducer_name}{i+1}"],
                                    group_df[f"{reducer_name}{j+1}"],
                                    label=group_name if i == 0 and j == 1 else None,  # Only show legend once
                                    alpha=0.7,
                                    s=30
                                )
                        else:
                            # Simple scatter plot
                            ax.scatter(
                                reduced_df[f"{reducer_name}{i+1}"],
                                reduced_df[f"{reducer_name}{j+1}"],
                                c=reduced_df[target] if target in reduced_df.columns else 'blue',
                                alpha=0.7,
                                s=30,
                                cmap='viridis'
                            )
                        
                        # Set labels
                        ax.set_xlabel(f"Component {i+1}")
                        ax.set_ylabel(f"Component {j+1}")
                        
                        # Remove tick labels for cleaner appearance
                        ax.tick_params(axis='both', which='major', labelsize=8)
                
                # Add a legend if using groupby
                if groupby in reduced_df.columns:
                    axes[1].legend(title=groupby, loc='upper right', bbox_to_anchor=(1.5, 1))
                
                # Add overall title
                fig.suptitle(title, fontsize=16)
                fig.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
                
            # Determine output file
            if output_file is None:
                output_file = f"dimension_reduction_{method}_{n_components}d.{self.output_format}"
            output_path = Path(self.output_dir) / output_file
            
            # Save figure
            plt.savefig(str(output_path), dpi=self.dpi)
            plt.close()
        
        return str(output_path)
    
    def export_visualization(
        self,
        visualization_path: str,
        output_format: str = 'png',
        output_file: Optional[str] = None,
        dpi: Optional[int] = None
    ) -> str:
        """
        Export an interactive visualization to a static image format.
        
        Args:
            visualization_path: Path to the interactive visualization file
            output_format: Output format ('png', 'pdf', 'svg', etc.)
            output_file: Output file path (generated from input path if None)
            dpi: DPI for output image (uses instance default if None)
            
        Returns:
            Path to output file
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for exporting visualizations")
        
        # Check if input file exists
        input_path = Path(visualization_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Visualization file not found: {visualization_path}")
        
        # Determine output file
        if output_file is None:
            output_file = input_path.stem + f".{output_format}"
        output_path = Path(self.output_dir) / output_file
        
        # Use instance default DPI if not specified
        if dpi is None:
            dpi = self.dpi
        
        # Load the interactive figure
        with open(input_path, 'r') as f:
            html_content = f.read()
        
        # Extract the JSON data from the HTML
        import re
        json_match = re.search(r'Plotly\.newPlot\((.*?),\s*({.*}),\s*({.*})', html_content, re.DOTALL)
        
        if json_match:
            # Create a Plotly figure from JSON
            figure_id = json_match.group(1).strip('"\'')
            figure_data = json.loads(json_match.group(2))
            figure_layout = json.loads(json_match.group(3))
            
            fig = go.Figure(data=figure_data, layout=figure_layout)
            
            # Export to static image
            fig.write_image(str(output_path), format=output_format, scale=dpi/100)
        else:
            raise ValueError("Failed to extract Plotly figure from HTML file")
        
        return str(output_path)
    
    def create_prediction_confidence_visualization(
        self,
        data: Union[pd.DataFrame, Dict, str],
        metric: str = "throughput",
        confidence_column: Optional[str] = None,
        groupby: List[str] = ["model_name", "hardware"],
        title: str = "Prediction Confidence Visualization",
        output_file: Optional[str] = None,
        interactive: Optional[bool] = None
    ) -> str:
        """
        Create a visualization showing prediction values with confidence intervals.
        
        Args:
            data: Performance data to visualize
            metric: The performance metric to visualize
            confidence_column: Column with confidence scores (0-1)
            groupby: Columns to group by
            title: Plot title
            output_file: Output file path (generated if None)
            interactive: Whether to create an interactive plot (overrides instance setting)
            
        Returns:
            Path to output file
        """
        # Prepare data
        df = self._prepare_data(data)
        
        # Check if metric exists
        if metric not in df.columns:
            raise ValueError(f"Metric column '{metric}' not found in data")
        
        # Check if confidence column exists, or try to infer it
        if confidence_column is None:
            # Try common patterns for confidence columns
            candidates = [f"{metric}_confidence", f"confidence_{metric}", "confidence", "conf"]
            for candidate in candidates:
                if candidate in df.columns:
                    confidence_column = candidate
                    break
        
        # Check if we have lower and upper bounds
        has_bounds = (f"{metric}_lower_bound" in df.columns and f"{metric}_upper_bound" in df.columns)
        
        # If no confidence or bounds, we'll visualize without error bars
        has_confidence = confidence_column in df.columns if confidence_column else False
        
        # Determine if we should use interactive visualization
        use_interactive = interactive if interactive is not None else self.interactive
        
        if use_interactive and PLOTLY_AVAILABLE:
            # Group by specified columns and calculate mean values
            grouped_df = df.groupby(groupby)[metric].mean().reset_index()
            
            # Add confidence to grouped data if available
            if has_confidence:
                grouped_df[confidence_column] = df.groupby(groupby)[confidence_column].mean().values
            
            # Add bounds to grouped data if available
            if has_bounds:
                grouped_df[f"{metric}_lower_bound"] = df.groupby(groupby)[f"{metric}_lower_bound"].mean().values
                grouped_df[f"{metric}_upper_bound"] = df.groupby(groupby)[f"{metric}_upper_bound"].mean().values
            
            # Create interactive plot with Plotly
            if len(groupby) >= 2:
                # Create grouped bar chart for two groupby columns
                fig = px.bar(
                    grouped_df,
                    x=groupby[0],
                    y=metric,
                    color=groupby[1],
                    barmode="group",
                    title=title,
                    error_y=None if not has_bounds else dict(
                        type="data",
                        symmetric=False,
                        array=grouped_df[f"{metric}_upper_bound"] - grouped_df[metric],
                        arrayminus=grouped_df[metric] - grouped_df[f"{metric}_lower_bound"]
                    )
                )
                
                # Add confidence as color intensity if available
                if has_confidence:
                    # Create custom hover text
                    hover_template = (
                        f"{groupby[0]}: %{{x}}<br>"
                        f"{groupby[1]}: %{{color}}<br>"
                        f"{metric}: %{{y:.2f}}<br>"
                        f"Confidence: %{{customdata[0]:.2%}}"
                    )
                    
                    for i, trace in enumerate(fig.data):
                        # Extract data for this trace
                        trace_group = trace.name
                        trace_data = grouped_df[grouped_df[groupby[1]] == trace_group]
                        
                        # Update opacity based on confidence
                        trace.marker.opacity = trace_data[confidence_column].values
                        
                        # Add confidence to hover text
                        trace.customdata = np.column_stack([trace_data[confidence_column]])
                        trace.hovertemplate = hover_template
            else:
                # Simple bar chart for single groupby
                fig = px.bar(
                    grouped_df,
                    x=groupby[0],
                    y=metric,
                    title=title,
                    error_y=None if not has_bounds else dict(
                        type="data",
                        symmetric=False,
                        array=grouped_df[f"{metric}_upper_bound"] - grouped_df[metric],
                        arrayminus=grouped_df[metric] - grouped_df[f"{metric}_lower_bound"]
                    )
                )
                
                # Add confidence as color intensity if available
                if has_confidence:
                    # Update opacity based on confidence
                    fig.data[0].marker.opacity = grouped_df[confidence_column].values
                    
                    # Create custom hover text
                    hover_template = (
                        f"{groupby[0]}: %{{x}}<br>"
                        f"{metric}: %{{y:.2f}}<br>"
                        f"Confidence: %{{customdata[0]:.2%}}"
                    )
                    
                    # Add confidence to hover text
                    fig.data[0].customdata = np.column_stack([grouped_df[confidence_column]])
                    fig.data[0].hovertemplate = hover_template
            
            # Update layout
            fig.update_layout(
                yaxis_title=metric,
                xaxis_title=groupby[0],
                width=1000,
                height=600
            )
            
            # Add color legend for confidence
            if has_confidence:
                fig.update_layout(
                    coloraxis_colorbar=dict(
                        title="Confidence",
                        tickvals=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        ticktext=["50%", "60%", "70%", "80%", "90%", "100%"]
                    )
                )
            
            # Determine output file
            if output_file is None:
                output_file = f"confidence_{metric}_{'_'.join(groupby)}.html"
            output_path = Path(self.output_dir) / output_file
            
            # Save figure
            fig.write_html(str(output_path))
            
        else:
            # Group by specified columns and calculate mean values
            grouped_df = df.groupby(groupby)[metric].mean().reset_index()
            
            # Add confidence to grouped data if available
            if has_confidence:
                grouped_df[confidence_column] = df.groupby(groupby)[confidence_column].mean().values
            
            # Add bounds to grouped data if available
            if has_bounds:
                grouped_df[f"{metric}_lower_bound"] = df.groupby(groupby)[f"{metric}_lower_bound"].mean().values
                grouped_df[f"{metric}_upper_bound"] = df.groupby(groupby)[f"{metric}_upper_bound"].mean().values
            
            # Create static plot with Matplotlib
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            if len(groupby) >= 2:
                # Create a pivot table for grouped bar chart
                pivot_df = grouped_df.pivot(index=groupby[0], columns=groupby[1], values=metric)
                
                # Get the number of groups and bar positions
                n_groups = len(pivot_df.index)
                n_bars = len(pivot_df.columns)
                bar_width = 0.8 / n_bars
                
                # Define colors for each group
                colors = sns.color_palette(self.palette, n_colors=n_bars)
                
                # Get x positions for bars
                x_pos = np.arange(n_groups)
                
                # Plot each group
                for i, col in enumerate(pivot_df.columns):
                    # Get data for this group
                    values = pivot_df[col].values
                    
                    # Calculate positions for this group's bars
                    group_pos = x_pos + i * bar_width - (n_bars - 1) * bar_width / 2
                    
                    # Create bars
                    bars = ax.bar(
                        group_pos,
                        values,
                        bar_width,
                        label=col,
                        color=colors[i],
                        edgecolor='black',
                        linewidth=0.5
                    )
                    
                    # Add error bars if bounds are available
                    if has_bounds:
                        # Get bounds for this group
                        lower = grouped_df[
                            (grouped_df[groupby[0]].isin(pivot_df.index)) & 
                            (grouped_df[groupby[1]] == col)
                        ][f"{metric}_lower_bound"].values
                        
                        upper = grouped_df[
                            (grouped_df[groupby[0]].isin(pivot_df.index)) & 
                            (grouped_df[groupby[1]] == col)
                        ][f"{metric}_upper_bound"].values
                        
                        # Add error bars
                        ax.errorbar(
                            group_pos,
                            values,
                            yerr=[values - lower, upper - values],
                            fmt='none',
                            ecolor='black',
                            capsize=3
                        )
                    
                    # Modify bar colors based on confidence if available
                    if has_confidence:
                        # Get confidence for this group
                        conf_values = grouped_df[
                            (grouped_df[groupby[0]].isin(pivot_df.index)) & 
                            (grouped_df[groupby[1]] == col)
                        ][confidence_column].values
                        
                        # Apply varying opacity based on confidence
                        for j, bar in enumerate(bars):
                            if j < len(conf_values):
                                # Apply confidence as alpha
                                bar.set_alpha(conf_values[j])
                
                # Set x-axis labels and position
                ax.set_xticks(x_pos)
                ax.set_xticklabels(pivot_df.index)
                
            else:
                # Simple bar chart for single groupby
                bars = ax.bar(
                    grouped_df[groupby[0]],
                    grouped_df[metric],
                    color=sns.color_palette(self.palette, n_colors=1)[0],
                    edgecolor='black',
                    linewidth=0.5
                )
                
                # Add error bars if bounds are available
                if has_bounds:
                    # Add error bars
                    ax.errorbar(
                        grouped_df[groupby[0]],
                        grouped_df[metric],
                        yerr=[
                            grouped_df[metric] - grouped_df[f"{metric}_lower_bound"],
                            grouped_df[f"{metric}_upper_bound"] - grouped_df[metric]
                        ],
                        fmt='none',
                        ecolor='black',
                        capsize=3
                    )
                
                # Modify bar colors based on confidence if available
                if has_confidence:
                    # Apply varying opacity based on confidence
                    for i, bar in enumerate(bars):
                        bar.set_alpha(grouped_df[confidence_column].iloc[i])
            
            # Set labels and title
            ax.set_xlabel(groupby[0])
            ax.set_ylabel(metric)
            ax.set_title(title)
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add legend for multiple groups
            if len(groupby) >= 2:
                ax.legend(title=groupby[1])
            
            # Add confidence legend if available
            if has_confidence:
                # Create a custom legend for confidence
                from matplotlib.patches import Patch
                confidence_levels = [0.5, 0.7, 0.9]
                legend_elements = [
                    Patch(facecolor=colors[0] if len(groupby) >= 2 else sns.color_palette(self.palette, n_colors=1)[0],
                          alpha=conf,
                          label=f"{conf:.0%} Confidence")
                    for conf in confidence_levels
                ]
                
                # Add a second legend for confidence levels
                ax2 = plt.gca().twinx()
                ax2.set_yticks([])
                ax2.legend(handles=legend_elements, title="Confidence", loc='upper right')
            
            # Rotate x-tick labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Determine output file
            if output_file is None:
                output_file = f"confidence_{metric}_{'_'.join(groupby)}.{self.output_format}"
            output_path = Path(self.output_dir) / output_file
            
            # Save figure
            plt.savefig(str(output_path), dpi=self.dpi)
            plt.close()
        
        return str(output_path)
    
    def create_batch_visualizations(
        self,
        data: Union[pd.DataFrame, Dict, str],
        metrics: List[str] = DEFAULT_METRICS,
        groupby: List[str] = ["model_name", "hardware"],
        output_dir: Optional[str] = None,
        interactive: Optional[bool] = None,
        include_3d: bool = True,
        include_time_series: bool = True,
        include_power_efficiency: bool = True,
        include_dimension_reduction: bool = True,
        include_confidence: bool = True
    ) -> Dict[str, List[str]]:
        """
        Create a batch of visualizations for the given data.
        
        Args:
            data: Performance data to visualize
            metrics: List of metrics to visualize
            groupby: Columns to group by for comparisons
            output_dir: Output directory for visualizations (uses instance default if None)
            interactive: Whether to create interactive visualizations (overrides instance setting)
            include_3d: Whether to include 3D visualizations
            include_time_series: Whether to include time-series visualizations
            include_power_efficiency: Whether to include power efficiency visualizations
            include_dimension_reduction: Whether to include dimension reduction visualizations
            include_confidence: Whether to include confidence visualizations
            
        Returns:
            Dictionary mapping visualization types to lists of output file paths
        """
        # Set output directory
        if output_dir is not None:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Prepare data
        df = self._prepare_data(data)
        
        # Track output files
        output_files = {
            "dashboard": [],
            "3d": [],
            "time_series": [],
            "power_efficiency": [],
            "dimension_reduction": [],
            "confidence": []
        }
        
        # Create dashboards for each metric
        for metric in metrics:
            if metric in df.columns:
                output_file = self.create_performance_dashboard(
                    df,
                    metrics=[metric],
                    groupby=groupby,
                    title=f"{metric.replace('_', ' ').title()} Performance Dashboard",
                    interactive=interactive
                )
                output_files["dashboard"].append(output_file)
        
        # Create 3D visualizations if requested
        if include_3d and len(DEFAULT_3D_METRICS) >= 3:
            # Check if we have enough metrics
            available_metrics = [m for m in DEFAULT_3D_METRICS if m in df.columns]
            
            if len(available_metrics) >= 3:
                # Use the first three available metrics
                x_metric, y_metric, z_metric = available_metrics[:3]
                
                output_file = self.create_3d_visualization(
                    df,
                    x_metric=x_metric,
                    y_metric=y_metric,
                    z_metric=z_metric,
                    color_metric=groupby[0] if len(groupby) > 0 else None,
                    title="3D Performance Visualization",
                    interactive=interactive
                )
                output_files["3d"].append(output_file)
        
        # Create time-series visualizations if requested
        if include_time_series and "timestamp" in df.columns:
            for metric in metrics:
                if metric in df.columns:
                    output_file = self.create_time_series_visualization(
                        df,
                        time_column="timestamp",
                        metric=metric,
                        groupby=groupby,
                        title=f"{metric.replace('_', ' ').title()} Over Time",
                        interactive=interactive
                    )
                    output_files["time_series"].append(output_file)
        
        # Create power efficiency visualizations if requested
        if include_power_efficiency and "power_consumption" in df.columns:
            for metric in metrics:
                if metric in df.columns and metric != "power_consumption":
                    output_file = self.create_power_efficiency_visualization(
                        df,
                        performance_metric=metric,
                        power_metric="power_consumption",
                        groupby=groupby,
                        title=f"{metric.replace('_', ' ').title()} Power Efficiency",
                        interactive=interactive
                    )
                    output_files["power_efficiency"].append(output_file)
        
        # Create dimension reduction visualizations if requested
        if include_dimension_reduction:
            # Get numerical features (exclude metrics and categorical columns)
            potential_features = [col for col in df.columns 
                                 if pd.api.types.is_numeric_dtype(df[col]) 
                                 and col not in metrics
                                 and col not in groupby]
            
            if len(potential_features) >= 5:  # Ensure we have enough features
                # Use PCA for dimension reduction
                for metric in metrics:
                    if metric in df.columns:
                        output_file = self.create_dimension_reduction_visualization(
                            df,
                            features=potential_features[:10],  # Use up to 10 features
                            target=metric,
                            method="pca",
                            n_components=2,
                            groupby=groupby[0] if len(groupby) > 0 else None,
                            title=f"PCA Analysis for {metric.replace('_', ' ').title()}",
                            interactive=interactive
                        )
                        output_files["dimension_reduction"].append(output_file)
        
        # Create confidence visualizations if requested
        if include_confidence:
            for metric in metrics:
                # Check if metric and confidence columns exist
                if metric in df.columns:
                    confidence_col = None
                    for candidate in [f"{metric}_confidence", f"confidence_{metric}", "confidence"]:
                        if candidate in df.columns:
                            confidence_col = candidate
                            break
                    
                    if confidence_col or f"{metric}_lower_bound" in df.columns:
                        output_file = self.create_prediction_confidence_visualization(
                            df,
                            metric=metric,
                            confidence_column=confidence_col,
                            groupby=groupby,
                            title=f"{metric.replace('_', ' ').title()} with Confidence",
                            interactive=interactive
                        )
                        output_files["confidence"].append(output_file)
        
        # Return all output files
        return output_files

def create_visualization_report(
    visualization_files: Dict[str, List[str]],
    title: str = "Performance Visualization Report",
    output_file: str = "visualization_report.html",
    output_dir: Optional[str] = None
) -> str:
    """
    Create an HTML report with all generated visualizations.
    
    Args:
        visualization_files: Dictionary mapping visualization types to lists of output file paths
        title: Report title
        output_file: Output file name
        output_dir: Output directory (uses current directory if None)
        
    Returns:
        Path to output file
    """
    # Set output directory
    if output_dir is None:
        output_dir = "."
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output path
    output_path = Path(output_dir) / output_file
    
    # Prepare HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                padding: 20px;
                max-width: 1200px;
                margin: 0 auto;
                color: #333;
            }}
            h1, h2, h3 {{
                color: #444;
            }}
            .visualization-section {{
                margin-bottom: 40px;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
            }}
            .visualization-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                grid-gap: 20px;
                margin-top: 20px;
            }}
            .visualization-item {{
                border: 1px solid #ddd;
                border-radius: 5px;
                overflow: hidden;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .visualization-item h4 {{
                margin: 0;
                padding: 10px;
                background-color: #f8f8f8;
                border-bottom: 1px solid #ddd;
            }}
            .visualization-content {{
                padding: 10px;
                text-align: center;
            }}
            .visualization-content img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto;
            }}
            .visualization-content iframe {{
                width: 100%;
                height: 500px;
                border: none;
            }}
            .timestamp {{
                text-align: right;
                color: #888;
                font-size: 0.8em;
                margin-top: 40px;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """
    
    # Add sections for each visualization type
    for vis_type, files in visualization_files.items():
        if files:
            html_content += f"""
            <div class="visualization-section">
                <h2>{vis_type.replace('_', ' ').title()} Visualizations</h2>
                <div class="visualization-grid">
            """
            
            for file_path in files:
                # Get file name and extension
                file = Path(file_path)
                file_name = file.stem
                extension = file.suffix.lower()
                
                # Create relative path
                relative_path = os.path.relpath(file_path, output_dir)
                
                # Create visualization item based on file type
                html_content += f"""
                <div class="visualization-item">
                    <h4>{file_name.replace('_', ' ').title()}</h4>
                    <div class="visualization-content">
                """
                
                if extension == ".html":
                    # For HTML (interactive visualization), embed as iframe
                    html_content += f"""
                        <iframe src="{relative_path}" frameborder="0"></iframe>
                    """
                else:
                    # For images, embed as img
                    html_content += f"""
                        <img src="{relative_path}" alt="{file_name}">
                    """
                
                html_content += """
                    </div>
                </div>
                """
            
            html_content += """
                </div>
            </div>
            """
    
    # Add timestamp and close HTML
    html_content += f"""
        <div class="timestamp">Generated by Predictive Performance System on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_path, "w") as f:
        f.write(html_content)
    
    return str(output_path)

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate advanced visualizations for performance data")
    parser.add_argument("--data", required=True, help="Path to data file (JSON or CSV)")
    parser.add_argument("--output-dir", default="./visualizations", help="Output directory for visualizations")
    parser.add_argument("--interactive", action="store_true", help="Generate interactive visualizations")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, help="Metrics to visualize")
    parser.add_argument("--groupby", nargs="+", default=["model_name", "hardware"], help="Columns to group by")
    parser.add_argument("--no-3d", action="store_true", help="Skip 3D visualizations")
    parser.add_argument("--no-time-series", action="store_true", help="Skip time-series visualizations")
    parser.add_argument("--no-power-efficiency", action="store_true", help="Skip power efficiency visualizations")
    parser.add_argument("--no-dimension-reduction", action="store_true", help="Skip dimension reduction visualizations")
    parser.add_argument("--no-confidence", action="store_true", help="Skip confidence visualizations")
    parser.add_argument("--report", action="store_true", help="Generate HTML report with all visualizations")
    parser.add_argument("--report-title", default="Performance Visualization Report", help="Report title")
    parser.add_argument("--report-file", default="visualization_report.html", help="Report output file")
    
    args = parser.parse_args()
    
    # Create visualization system
    vis = AdvancedVisualization(
        output_dir=args.output_dir,
        interactive=args.interactive
    )
    
    # Generate visualizations
    visualization_files = vis.create_batch_visualizations(
        data=args.data,
        metrics=args.metrics,
        groupby=args.groupby,
        include_3d=not args.no_3d,
        include_time_series=not args.no_time_series,
        include_power_efficiency=not args.no_power_efficiency,
        include_dimension_reduction=not args.no_dimension_reduction,
        include_confidence=not args.no_confidence
    )
    
    # Generate report if requested
    if args.report:
        report_path = create_visualization_report(
            visualization_files=visualization_files,
            title=args.report_title,
            output_file=args.report_file,
            output_dir=args.output_dir
        )
        print(f"Generated report: {report_path}")
    
    # Print summary
    total_visualizations = sum(len(files) for files in visualization_files.values())
    print(f"Generated {total_visualizations} visualizations in {args.output_dir}")
    
    for vis_type, files in visualization_files.items():
        if files:
            print(f"  - {len(files)} {vis_type} visualizations")