#!/usr/bin/env python3
"""
Distributed Testing Framework - Dynamic Resource Management Visualization

This module implements visualization capabilities for the Dynamic Resource Management
component of the Distributed Testing Framework. It provides comprehensive visualizations
for resource allocation, scaling decisions, workload patterns, and cloud resource 
utilization.

The visualizations help in understanding resource allocation patterns, scaling 
effectiveness, and identifying optimization opportunities.

Usage:
    # Import the module
    from data.duckdb.distributed_testing.dynamic_resource_management_visualization import DRMVisualization
    
    # Create a visualization instance with a reference to the DRM
    visualization = DRMVisualization(dynamic_resource_manager)
    
    # Generate a resource utilization heatmap
    visualization.create_resource_utilization_heatmap()
    
    # Generate a scaling history visualization
    visualization.create_scaling_history_visualization()
    
    # Generate a complete resource dashboard
    visualization.create_resource_dashboard()
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from collections import defaultdict

# For interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    
# For web dashboard
try:
    import tornado.web
    import tornado.ioloop
    import tornado.websocket
    TORNADO_AVAILABLE = True
except ImportError:
    TORNADO_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules from parent
import sys
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Try to import from the local path
try:
    from dynamic_resource_manager import DynamicResourceManager, ScalingDecision
    DRM_AVAILABLE = True
except ImportError:
    try:
        # Try relative import
        from test.tests.api.duckdb_api.distributed_testing.dynamic_resource_manager import DynamicResourceManager, ScalingDecision
        DRM_AVAILABLE = True
    except ImportError:
        logger.warning("DynamicResourceManager not available, some features will be limited")
        DRM_AVAILABLE = False

try:
    from cloud_provider_manager import CloudProviderManager
    CPM_AVAILABLE = True
except ImportError:
    try:
        # Try relative import
        from test.tests.api.duckdb_api.distributed_testing.cloud_provider_manager import CloudProviderManager
        CPM_AVAILABLE = True
    except ImportError:
        logger.warning("CloudProviderManager not available, some features will be limited")
        CPM_AVAILABLE = False

try:
    from resource_optimization import ResourceOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    try:
        # Try relative import
        from test.tests.api.duckdb_api.distributed_testing.resource_optimization import ResourceOptimizer
        OPTIMIZER_AVAILABLE = True
    except ImportError:
        logger.warning("ResourceOptimizer not available, some features will be limited")
        OPTIMIZER_AVAILABLE = False

class DRMVisualization:
    """
    Dynamic Resource Management Visualization
    
    This class provides visualization capabilities for the Dynamic Resource Management
    system, offering insights into resource utilization, scaling decisions, and 
    optimization opportunities.
    """
    
    def __init__(self, 
                 dynamic_resource_manager=None, 
                 cloud_provider_manager=None,
                 resource_optimizer=None,
                 output_dir=None,
                 dashboard_port=8889,
                 data_retention_days=30,
                 update_interval=300,
                 interactive=True):
        """
        Initialize the DRM visualization system.
        
        Args:
            dynamic_resource_manager: Optional DRM instance
            cloud_provider_manager: Optional CPM instance
            resource_optimizer: Optional ResourceOptimizer instance
            output_dir: Directory for output files
            dashboard_port: Port for web dashboard
            data_retention_days: Days of history to keep
            update_interval: Seconds between data updates
            interactive: Use interactive Plotly visualizations instead of static Matplotlib
        """
        self.drm = dynamic_resource_manager
        self.cpm = cloud_provider_manager
        self.optimizer = resource_optimizer
        
        # Configuration
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "visualizations"
        )
        self.dashboard_port = dashboard_port
        self.data_retention_days = data_retention_days
        self.update_interval = update_interval
        self.interactive = interactive and PLOTLY_AVAILABLE
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Historical data
        self.resource_history = []
        self.scaling_history = []
        self.worker_history = defaultdict(list)
        self.cloud_usage_history = defaultdict(lambda: defaultdict(list))
        
        # Dashboard components
        self.dashboard_running = False
        self.dashboard_app = None
        self.dashboard_thread = None
        self.dashboard_clients = set()
        
        # Update thread
        self.update_thread = None
        self.update_stop_event = threading.Event()
        
        logger.info(f"DRM Visualization initialized with output dir: {self.output_dir}")
        
        # Start data collection if DRM is available
        if self.drm:
            self._start_data_collection()
        
    def _start_data_collection(self):
        """Start the data collection thread."""
        if self.update_thread and self.update_thread.is_alive():
            return
            
        self.update_stop_event.clear()
        self.update_thread = threading.Thread(
            target=self._data_collection_loop,
            daemon=True
        )
        self.update_thread.start()
        logger.info("Data collection started")
        
    def _stop_data_collection(self):
        """Stop the data collection thread."""
        if not self.update_thread or not self.update_thread.is_alive():
            return
            
        self.update_stop_event.set()
        self.update_thread.join(timeout=5.0)
        logger.info("Data collection stopped")
        
    def _data_collection_loop(self):
        """Background thread for collecting data."""
        while not self.update_stop_event.is_set():
            try:
                # Collect resource data
                self._collect_resource_data()
                
                # Prune old data
                self._prune_old_data()
                
                # Update any active dashboard
                if self.dashboard_running:
                    self._update_dashboard_clients()
                    
            except Exception as e:
                logger.error(f"Error in data collection: {e}")
                
            # Wait for next update
            self.update_stop_event.wait(self.update_interval)
            
    def _collect_resource_data(self):
        """Collect resource data from DRM and related components."""
        if not self.drm:
            return
            
        timestamp = datetime.now()
        
        # Get worker statistics
        worker_stats = self.drm.get_worker_statistics()
        
        # Get overall utilization
        overall_utilization = worker_stats.get("overall_utilization", {})
        
        # Create resource snapshot
        resource_snapshot = {
            "timestamp": timestamp,
            "worker_count": worker_stats.get("total_workers", 0),
            "active_tasks": worker_stats.get("active_tasks", 0),
            "resource_reservations": worker_stats.get("resource_reservations", 0),
            "overall_utilization": overall_utilization,
            "workers": worker_stats.get("workers", {})
        }
        
        # Add to resource history
        self.resource_history.append(resource_snapshot)
        
        # Update worker history
        for worker_id, worker_data in worker_stats.get("workers", {}).items():
            self.worker_history[worker_id].append({
                "timestamp": timestamp,
                "utilization": worker_data.get("utilization", {}),
                "tasks": worker_data.get("tasks", 0),
                "resources": worker_data.get("resources", {})
            })
            
        # Get cloud provider data if available
        if self.cpm:
            for provider in self.cpm.providers:
                # Get resources for provider
                try:
                    resources = self.cpm.get_available_resources(provider)
                    
                    # Add to cloud usage history
                    self.cloud_usage_history[provider]["resources"].append({
                        "timestamp": timestamp,
                        "resources": resources
                    })
                    
                    # Get active workers for provider
                    active_workers = self.cpm.get_active_workers(provider)
                    
                    # Add to cloud usage history
                    self.cloud_usage_history[provider]["workers"].append({
                        "timestamp": timestamp,
                        "count": len(active_workers),
                        "workers": active_workers
                    })
                    
                    # Get cost data if available
                    if hasattr(self.cpm, "get_cost_estimate"):
                        cost = self.cpm.get_cost_estimate(provider)
                        
                        # Add to cloud usage history
                        self.cloud_usage_history[provider]["cost"].append({
                            "timestamp": timestamp,
                            "cost": cost
                        })
                except Exception as e:
                    logger.error(f"Error getting cloud provider data for {provider}: {e}")
                    
        # Get scaling decision if available
        if hasattr(self.drm, "last_scaling_decision"):
            scaling_decision = self.drm.last_scaling_decision
            
            # Add to scaling history if available
            if scaling_decision:
                self.scaling_history.append({
                    "timestamp": timestamp,
                    "decision": scaling_decision
                })
                
    def _prune_old_data(self):
        """Prune old data beyond retention period."""
        if self.data_retention_days <= 0:
            return
            
        cutoff_time = datetime.now() - timedelta(days=self.data_retention_days)
        
        # Prune resource history
        self.resource_history = [
            snapshot for snapshot in self.resource_history
            if snapshot["timestamp"] >= cutoff_time
        ]
        
        # Prune scaling history
        self.scaling_history = [
            entry for entry in self.scaling_history
            if entry["timestamp"] >= cutoff_time
        ]
        
        # Prune worker history
        for worker_id in list(self.worker_history.keys()):
            self.worker_history[worker_id] = [
                entry for entry in self.worker_history[worker_id]
                if entry["timestamp"] >= cutoff_time
            ]
            
            # Remove empty workers
            if not self.worker_history[worker_id]:
                del self.worker_history[worker_id]
                
        # Prune cloud usage history
        for provider in list(self.cloud_usage_history.keys()):
            for data_type in list(self.cloud_usage_history[provider].keys()):
                self.cloud_usage_history[provider][data_type] = [
                    entry for entry in self.cloud_usage_history[provider][data_type]
                    if entry["timestamp"] >= cutoff_time
                ]
                
                # Remove empty data types
                if not self.cloud_usage_history[provider][data_type]:
                    del self.cloud_usage_history[provider][data_type]
                    
            # Remove empty providers
            if not self.cloud_usage_history[provider]:
                del self.cloud_usage_history[provider]
                
    def create_resource_utilization_heatmap(self, 
                                          output_path=None, 
                                          show_plot=False, 
                                          interactive=None):
        """
        Create a resource utilization heatmap visualization.
        
        This visualization shows resource utilization across workers over time,
        allowing identification of utilization patterns and potential bottlenecks.
        
        Args:
            output_path: Path to save the visualization
            show_plot: Whether to display the plot
            interactive: Override instance interactive setting
            
        Returns:
            Path to the generated visualization file
        """
        interactive = self.interactive if interactive is None else interactive
        
        # Check if we have data
        if not self.worker_history:
            logger.warning("No worker history data available for heatmap")
            return None
            
        # Prepare data
        worker_ids = list(self.worker_history.keys())
        timestamps = []
        
        # Get list of all timestamps across all workers
        for worker_data in self.worker_history.values():
            timestamps.extend([entry["timestamp"] for entry in worker_data])
            
        # Get unique timestamps sorted
        timestamps = sorted(set(timestamps))
        
        # Create data structure for heatmap
        cpu_data = np.zeros((len(worker_ids), len(timestamps)))
        memory_data = np.zeros((len(worker_ids), len(timestamps)))
        gpu_data = np.zeros((len(worker_ids), len(timestamps)))
        
        # Fill data arrays
        for i, worker_id in enumerate(worker_ids):
            worker_data = self.worker_history[worker_id]
            
            # Create mapping of timestamps to entries
            entry_map = {entry["timestamp"]: entry for entry in worker_data}
            
            for j, timestamp in enumerate(timestamps):
                if timestamp in entry_map:
                    entry = entry_map[timestamp]
                    utilization = entry.get("utilization", {})
                    cpu_data[i, j] = utilization.get("cpu", 0) * 100
                    memory_data[i, j] = utilization.get("memory", 0) * 100
                    gpu_data[i, j] = utilization.get("gpu", 0) * 100
                    
        # Output path
        if not output_path:
            filename = f"resource_utilization_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if interactive and PLOTLY_AVAILABLE:
                output_path = os.path.join(self.output_dir, f"{filename}.html")
            else:
                output_path = os.path.join(self.output_dir, f"{filename}.png")
                
        # Create visualization
        if interactive and PLOTLY_AVAILABLE:
            # Create plotly figure
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=("CPU Utilization (%)", "Memory Utilization (%)", "GPU Utilization (%)"),
                vertical_spacing=0.1
            )
            
            # Format timestamps for display
            timestamp_texts = [ts.strftime("%H:%M:%S") for ts in timestamps]
            
            # Add CPU heatmap
            fig.add_trace(
                go.Heatmap(
                    z=cpu_data,
                    x=timestamp_texts,
                    y=worker_ids,
                    colorscale="Viridis",
                    colorbar=dict(title="CPU %", x=1.02, y=0.83, len=0.25),
                    zmin=0,
                    zmax=100
                ),
                row=1, col=1
            )
            
            # Add Memory heatmap
            fig.add_trace(
                go.Heatmap(
                    z=memory_data,
                    x=timestamp_texts,
                    y=worker_ids,
                    colorscale="Viridis",
                    colorbar=dict(title="Memory %", x=1.02, y=0.5, len=0.25),
                    zmin=0,
                    zmax=100
                ),
                row=2, col=1
            )
            
            # Add GPU heatmap
            fig.add_trace(
                go.Heatmap(
                    z=gpu_data,
                    x=timestamp_texts,
                    y=worker_ids,
                    colorscale="Viridis",
                    colorbar=dict(title="GPU %", x=1.02, y=0.17, len=0.25),
                    zmin=0,
                    zmax=100
                ),
                row=3, col=1
            )
            
            # Update layout
            fig.update_layout(
                title="Resource Utilization Heatmap",
                height=800,
                width=1200,
                showlegend=False
            )
            
            # Save figure
            fig.write_html(output_path)
            
            # Show figure if requested
            if show_plot:
                fig.show()
                
        else:
            # Create matplotlib figure
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            
            # Plot CPU heatmap
            im0 = axes[0].imshow(
                cpu_data, 
                aspect='auto', 
                cmap='viridis',
                vmin=0, 
                vmax=100
            )
            axes[0].set_title("CPU Utilization (%)")
            axes[0].set_yticks(range(len(worker_ids)))
            axes[0].set_yticklabels(worker_ids)
            fig.colorbar(im0, ax=axes[0])
            
            # Plot Memory heatmap
            im1 = axes[1].imshow(
                memory_data, 
                aspect='auto', 
                cmap='viridis',
                vmin=0, 
                vmax=100
            )
            axes[1].set_title("Memory Utilization (%)")
            axes[1].set_yticks(range(len(worker_ids)))
            axes[1].set_yticklabels(worker_ids)
            fig.colorbar(im1, ax=axes[1])
            
            # Plot GPU heatmap
            im2 = axes[2].imshow(
                gpu_data, 
                aspect='auto', 
                cmap='viridis',
                vmin=0, 
                vmax=100
            )
            axes[2].set_title("GPU Utilization (%)")
            axes[2].set_yticks(range(len(worker_ids)))
            axes[2].set_yticklabels(worker_ids)
            fig.colorbar(im2, ax=axes[2])
            
            # Set x-axis labels (timestamps)
            if len(timestamps) > 10:
                # Too many timestamps, show subset
                idx = np.linspace(0, len(timestamps) - 1, 10, dtype=int)
                axes[2].set_xticks(idx)
                axes[2].set_xticklabels([timestamps[i].strftime("%H:%M:%S") for i in idx], rotation=45)
            else:
                axes[2].set_xticks(range(len(timestamps)))
                axes[2].set_xticklabels([ts.strftime("%H:%M:%S") for ts in timestamps], rotation=45)
                
            # Add overall title
            fig.suptitle("Resource Utilization Heatmap", fontsize=16)
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            # Save figure
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
            # Show figure if requested
            if show_plot:
                plt.show()
            
            plt.close()
            
        logger.info(f"Resource utilization heatmap saved to {output_path}")
        return output_path
        
    def create_scaling_history_visualization(self, 
                                           output_path=None, 
                                           show_plot=False, 
                                           interactive=None):
        """
        Create a visualization of scaling decisions over time.
        
        This visualization shows scaling decisions (scale up, scale down, maintain)
        and their impact on resource utilization.
        
        Args:
            output_path: Path to save the visualization
            show_plot: Whether to display the plot
            interactive: Override instance interactive setting
            
        Returns:
            Path to the generated visualization file
        """
        interactive = self.interactive if interactive is None else interactive
        
        # Check if we have data
        if not self.scaling_history or not self.resource_history:
            logger.warning("No scaling history or resource history data available")
            return None
            
        # Prepare data
        timestamps = [entry["timestamp"] for entry in self.resource_history]
        worker_counts = [snapshot["worker_count"] for snapshot in self.resource_history]
        utilizations = [snapshot["overall_utilization"].get("overall", 0) * 100 
                       for snapshot in self.resource_history]
        
        # Prepare scaling events
        scale_up_times = []
        scale_down_times = []
        maintain_times = []
        
        for entry in self.scaling_history:
            timestamp = entry["timestamp"]
            decision = entry["decision"]
            
            if isinstance(decision, dict):
                action = decision.get("action", "maintain")
            else:
                # Assume ScalingDecision object
                action = decision.action
                
            if action == "scale_up":
                scale_up_times.append(timestamp)
            elif action == "scale_down":
                scale_down_times.append(timestamp)
            elif action == "maintain":
                maintain_times.append(timestamp)
                
        # Output path
        if not output_path:
            filename = f"scaling_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if interactive and PLOTLY_AVAILABLE:
                output_path = os.path.join(self.output_dir, f"{filename}.html")
            else:
                output_path = os.path.join(self.output_dir, f"{filename}.png")
                
        # Create visualization
        if interactive and PLOTLY_AVAILABLE:
            # Create plotly figure
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Worker Count & Scaling Events", "Resource Utilization"),
                vertical_spacing=0.15,
                shared_xaxes=True
            )
            
            # Plot worker count
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=worker_counts,
                    mode='lines+markers',
                    name='Worker Count',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
            
            # Add scaling events
            for timestamp in scale_up_times:
                fig.add_vline(
                    x=timestamp, 
                    line=dict(color="green", width=2, dash="dash"),
                    row=1, col=1
                )
                fig.add_annotation(
                    x=timestamp,
                    y=max(worker_counts) * 1.1,
                    text="Scale Up",
                    showarrow=False,
                    textangle=90,
                    font=dict(color="green"),
                    row=1, col=1
                )
                
            for timestamp in scale_down_times:
                fig.add_vline(
                    x=timestamp, 
                    line=dict(color="red", width=2, dash="dash"),
                    row=1, col=1
                )
                fig.add_annotation(
                    x=timestamp,
                    y=max(worker_counts) * 1.1,
                    text="Scale Down",
                    showarrow=False,
                    textangle=90,
                    font=dict(color="red"),
                    row=1, col=1
                )
                
            # Plot utilization
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=utilizations,
                    mode='lines',
                    name='Utilization (%)',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
            
            # Add threshold references if available
            if self.drm:
                try:
                    scale_up_threshold = self.drm.scale_up_threshold * 100
                    scale_down_threshold = self.drm.scale_down_threshold * 100
                    target_utilization = self.drm.target_utilization * 100
                    
                    # Add threshold lines
                    fig.add_hline(
                        y=scale_up_threshold,
                        line=dict(color="green", width=2, dash="dot"),
                        annotation_text="Scale Up Threshold",
                        annotation_position="right",
                        row=2, col=1
                    )
                    
                    fig.add_hline(
                        y=scale_down_threshold,
                        line=dict(color="red", width=2, dash="dot"),
                        annotation_text="Scale Down Threshold",
                        annotation_position="right",
                        row=2, col=1
                    )
                    
                    fig.add_hline(
                        y=target_utilization,
                        line=dict(color="blue", width=2, dash="dot"),
                        annotation_text="Target Utilization",
                        annotation_position="right",
                        row=2, col=1
                    )
                except Exception as e:
                    logger.warning(f"Could not add threshold lines: {e}")
                    
            # Update layout
            fig.update_layout(
                title="Scaling History and Resource Utilization",
                height=700,
                width=1200,
                showlegend=True,
                legend=dict(orientation="h", y=1.1),
                xaxis2=dict(title="Time"),
                yaxis=dict(title="Worker Count"),
                yaxis2=dict(title="Utilization (%)")
            )
            
            # Save figure
            fig.write_html(output_path)
            
            # Show figure if requested
            if show_plot:
                fig.show()
                
        else:
            # Create matplotlib figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Plot worker count
            ax1.plot(timestamps, worker_counts, 'bo-', linewidth=2, markersize=6, label='Worker Count')
            
            # Add scaling events
            ymin, ymax = ax1.get_ylim()
            
            for timestamp in scale_up_times:
                ax1.axvline(x=timestamp, color='green', linestyle='--', alpha=0.7)
                ax1.text(timestamp, ymax * 0.95, "Scale Up", rotation=90, color='green',
                       ha='right', va='top', alpha=0.9)
                
            for timestamp in scale_down_times:
                ax1.axvline(x=timestamp, color='red', linestyle='--', alpha=0.7)
                ax1.text(timestamp, ymax * 0.95, "Scale Down", rotation=90, color='red',
                       ha='right', va='top', alpha=0.9)
                
            # Configure worker count axis
            ax1.set_title("Worker Count & Scaling Events")
            ax1.set_ylabel("Worker Count")
            ax1.grid(True, alpha=0.3)
            
            # Plot utilization
            ax2.plot(timestamps, utilizations, 'purple', linewidth=2, label='Utilization (%)')
            
            # Add threshold references if available
            if self.drm:
                try:
                    scale_up_threshold = self.drm.scale_up_threshold * 100
                    scale_down_threshold = self.drm.scale_down_threshold * 100
                    target_utilization = self.drm.target_utilization * 100
                    
                    # Add threshold lines
                    ax2.axhline(y=scale_up_threshold, color='green', linestyle=':', linewidth=2)
                    ax2.text(timestamps[0], scale_up_threshold, "Scale Up Threshold", 
                           color='green', va='bottom', ha='left')
                    
                    ax2.axhline(y=scale_down_threshold, color='red', linestyle=':', linewidth=2)
                    ax2.text(timestamps[0], scale_down_threshold, "Scale Down Threshold", 
                           color='red', va='top', ha='left')
                    
                    ax2.axhline(y=target_utilization, color='blue', linestyle=':', linewidth=2)
                    ax2.text(timestamps[0], target_utilization, "Target Utilization", 
                           color='blue', va='bottom', ha='left')
                except Exception as e:
                    logger.warning(f"Could not add threshold lines: {e}")
                    
            # Configure utilization axis
            ax2.set_title("Resource Utilization")
            ax2.set_ylabel("Utilization (%)")
            ax2.set_xlabel("Time")
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis
            plt.xticks(rotation=45)
            fig.autofmt_xdate()
            
            # Add overall title
            fig.suptitle("Scaling History and Resource Utilization", fontsize=16)
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            # Save figure
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
            # Show figure if requested
            if show_plot:
                plt.show()
            
            plt.close()
            
        logger.info(f"Scaling history visualization saved to {output_path}")
        return output_path
        
    def create_cloud_resource_visualization(self, 
                                          output_path=None, 
                                          show_plot=False, 
                                          interactive=None):
        """
        Create a visualization of cloud resource usage.
        
        This visualization shows resource usage across different cloud providers,
        including worker counts, resource consumption, and cost information if available.
        
        Args:
            output_path: Path to save the visualization
            show_plot: Whether to display the plot
            interactive: Override instance interactive setting
            
        Returns:
            Path to the generated visualization file
        """
        interactive = self.interactive if interactive is None else interactive
        
        # Check if we have data
        if not self.cloud_usage_history:
            logger.warning("No cloud usage history data available")
            return None
            
        # Prepare data
        providers = list(self.cloud_usage_history.keys())
        
        # Output path
        if not output_path:
            filename = f"cloud_resource_usage_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if interactive and PLOTLY_AVAILABLE:
                output_path = os.path.join(self.output_dir, f"{filename}.html")
            else:
                output_path = os.path.join(self.output_dir, f"{filename}.png")
                
        # Create visualization
        if interactive and PLOTLY_AVAILABLE:
            # Create plotly figure
            n_providers = len(providers)
            fig = make_subplots(
                rows=n_providers, cols=2,
                subplot_titles=[f"{provider} - Workers" for provider in providers] +
                              [f"{provider} - Cost" for provider in providers],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            # For each provider, plot worker count and cost if available
            for i, provider in enumerate(providers):
                # Worker count over time
                if "workers" in self.cloud_usage_history[provider]:
                    worker_data = self.cloud_usage_history[provider]["workers"]
                    timestamps = [entry["timestamp"] for entry in worker_data]
                    counts = [entry["count"] for entry in worker_data]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=counts,
                            mode='lines+markers',
                            name=f"{provider} Workers",
                            line=dict(color='blue', width=2),
                            marker=dict(size=8)
                        ),
                        row=i+1, col=1
                    )
                    
                # Cost over time if available
                if "cost" in self.cloud_usage_history[provider]:
                    cost_data = self.cloud_usage_history[provider]["cost"]
                    timestamps = [entry["timestamp"] for entry in cost_data]
                    costs = [entry["cost"] for entry in cost_data]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=costs,
                            mode='lines+markers',
                            name=f"{provider} Cost",
                            line=dict(color='red', width=2),
                            marker=dict(size=8)
                        ),
                        row=i+1, col=2
                    )
                    
            # Update layout
            fig.update_layout(
                title="Cloud Resource Usage by Provider",
                height=300 * n_providers,
                width=1200,
                showlegend=True
            )
            
            # Update x and y axes titles
            for i in range(1, n_providers + 1):
                fig.update_yaxes(title_text="Worker Count", row=i, col=1)
                fig.update_yaxes(title_text="Cost", row=i, col=2)
                
            fig.update_xaxes(title_text="Time", row=n_providers, col=1)
            fig.update_xaxes(title_text="Time", row=n_providers, col=2)
            
            # Save figure
            fig.write_html(output_path)
            
            # Show figure if requested
            if show_plot:
                fig.show()
                
        else:
            # Create matplotlib figure
            n_providers = len(providers)
            fig, axes = plt.subplots(n_providers, 2, figsize=(14, 5 * n_providers))
            
            # Handle case with only one provider
            if n_providers == 1:
                axes = np.array([axes])
                
            # For each provider, plot worker count and cost if available
            for i, provider in enumerate(providers):
                # Worker count over time
                if "workers" in self.cloud_usage_history[provider]:
                    worker_data = self.cloud_usage_history[provider]["workers"]
                    timestamps = [entry["timestamp"] for entry in worker_data]
                    counts = [entry["count"] for entry in worker_data]
                    
                    axes[i, 0].plot(timestamps, counts, 'bo-', linewidth=2, markersize=6)
                    axes[i, 0].set_title(f"{provider} - Workers")
                    axes[i, 0].set_ylabel("Worker Count")
                    if i == n_providers - 1:
                        axes[i, 0].set_xlabel("Time")
                        
                    axes[i, 0].grid(True, alpha=0.3)
                    
                # Cost over time if available
                if "cost" in self.cloud_usage_history[provider]:
                    cost_data = self.cloud_usage_history[provider]["cost"]
                    timestamps = [entry["timestamp"] for entry in cost_data]
                    costs = [entry["cost"] for entry in cost_data]
                    
                    axes[i, 1].plot(timestamps, costs, 'ro-', linewidth=2, markersize=6)
                    axes[i, 1].set_title(f"{provider} - Cost")
                    axes[i, 1].set_ylabel("Cost")
                    if i == n_providers - 1:
                        axes[i, 1].set_xlabel("Time")
                        
                    axes[i, 1].grid(True, alpha=0.3)
                    
            # Format x-axis dates
            for i in range(n_providers):
                for j in range(2):
                    plt.setp(axes[i, j].xaxis.get_majorticklabels(), rotation=45)
                    axes[i, j].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    
            # Add overall title
            fig.suptitle("Cloud Resource Usage by Provider", fontsize=16)
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            
            # Save figure
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
            # Show figure if requested
            if show_plot:
                plt.show()
            
            plt.close()
            
        logger.info(f"Cloud resource visualization saved to {output_path}")
        return output_path
        
    def create_resource_allocation_visualization(self, 
                                               output_path=None, 
                                               show_plot=False, 
                                               interactive=None):
        """
        Create a visualization of resource allocation across workers.
        
        This visualization shows how different resource types (CPU, memory, GPU)
        are allocated across workers.
        
        Args:
            output_path: Path to save the visualization
            show_plot: Whether to display the plot
            interactive: Override instance interactive setting
            
        Returns:
            Path to the generated visualization file
        """
        interactive = self.interactive if interactive is None else interactive
        
        # Check if we have data
        if not self.worker_history:
            logger.warning("No worker history data available")
            return None
            
        # Get the most recent data point for each worker
        worker_data = {}
        for worker_id, history in self.worker_history.items():
            if history:
                worker_data[worker_id] = history[-1]
                
        if not worker_data:
            logger.warning("No current worker data available")
            return None
            
        # Extract resource allocation
        worker_ids = list(worker_data.keys())
        cpu_allocations = []
        memory_allocations = []
        gpu_allocations = []
        
        for worker_id in worker_ids:
            data = worker_data[worker_id]
            resources = data.get("resources", {})
            
            # Calculate allocated resources (total - available)
            cpu_total = resources.get("cpu", {}).get("cores", 0)
            cpu_available = resources.get("cpu", {}).get("available_cores", 0)
            cpu_allocated = cpu_total - cpu_available
            cpu_allocations.append(cpu_allocated)
            
            memory_total = resources.get("memory", {}).get("total_mb", 0)
            memory_available = resources.get("memory", {}).get("available_mb", 0)
            memory_allocated = memory_total - memory_available
            memory_allocations.append(memory_allocated)
            
            if "gpu" in resources:
                gpu_total = resources.get("gpu", {}).get("memory_mb", 0)
                gpu_available = resources.get("gpu", {}).get("available_memory_mb", 0)
                gpu_allocated = gpu_total - gpu_available
                gpu_allocations.append(gpu_allocated)
            else:
                gpu_allocations.append(0)
                
        # Output path
        if not output_path:
            filename = f"resource_allocation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if interactive and PLOTLY_AVAILABLE:
                output_path = os.path.join(self.output_dir, f"{filename}.html")
            else:
                output_path = os.path.join(self.output_dir, f"{filename}.png")
                
        # Create visualization
        if interactive and PLOTLY_AVAILABLE:
            # Create plotly figure
            fig = go.Figure()
            
            # Add CPU allocation
            fig.add_trace(
                go.Bar(
                    x=worker_ids,
                    y=cpu_allocations,
                    name='CPU Cores',
                    marker_color='blue'
                )
            )
            
            # Add Memory allocation (convert to GB for display)
            fig.add_trace(
                go.Bar(
                    x=worker_ids,
                    y=[mem / 1024 for mem in memory_allocations],
                    name='Memory (GB)',
                    marker_color='green'
                )
            )
            
            # Add GPU allocation (convert to GB for display)
            fig.add_trace(
                go.Bar(
                    x=worker_ids,
                    y=[gpu / 1024 for gpu in gpu_allocations],
                    name='GPU Memory (GB)',
                    marker_color='red'
                )
            )
            
            # Update layout
            fig.update_layout(
                title="Resource Allocation by Worker",
                xaxis_title="Worker ID",
                yaxis_title="Allocated Resources",
                barmode='group',
                height=600,
                width=1200
            )
            
            # Save figure
            fig.write_html(output_path)
            
            # Show figure if requested
            if show_plot:
                fig.show()
                
        else:
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Set up bar positions
            x = np.arange(len(worker_ids))
            width = 0.25
            
            # Plot bars
            cpu_bars = ax.bar(x - width, cpu_allocations, width, label='CPU Cores', color='blue')
            mem_bars = ax.bar(x, [mem / 1024 for mem in memory_allocations], width, 
                             label='Memory (GB)', color='green')
            gpu_bars = ax.bar(x + width, [gpu / 1024 for gpu in gpu_allocations], width,
                             label='GPU Memory (GB)', color='red')
            
            # Add labels and title
            ax.set_xlabel('Worker ID')
            ax.set_ylabel('Allocated Resources')
            ax.set_title('Resource Allocation by Worker')
            ax.set_xticks(x)
            ax.set_xticklabels(worker_ids, rotation=45)
            ax.legend()
            
            # Add grid
            ax.grid(True, axis='y', alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
            # Show figure if requested
            if show_plot:
                plt.show()
            
            plt.close()
            
        logger.info(f"Resource allocation visualization saved to {output_path}")
        return output_path
        
    def create_resource_efficiency_visualization(self, 
                                               output_path=None, 
                                               show_plot=False, 
                                               interactive=None):
        """
        Create a visualization of resource allocation efficiency.
        
        This visualization shows the efficiency of resource allocation,
        comparing allocated resources to actual resource usage.
        
        Args:
            output_path: Path to save the visualization
            show_plot: Whether to display the plot
            interactive: Override instance interactive setting
            
        Returns:
            Path to the generated visualization file
        """
        interactive = self.interactive if interactive is None else interactive
        
        # Check if we have data
        if not self.worker_history:
            logger.warning("No worker history data available")
            return None
            
        # Get the most recent data point for each worker
        worker_data = {}
        for worker_id, history in self.worker_history.items():
            if history:
                worker_data[worker_id] = history[-1]
                
        if not worker_data:
            logger.warning("No current worker data available")
            return None
            
        # Extract resource allocation and utilization
        worker_ids = list(worker_data.keys())
        cpu_efficiency = []
        memory_efficiency = []
        gpu_efficiency = []
        
        for worker_id in worker_ids:
            data = worker_data[worker_id]
            resources = data.get("resources", {})
            utilization = data.get("utilization", {})
            
            # Calculate efficiency (utilization / allocation)
            cpu_util = utilization.get("cpu", 0)
            cpu_total = resources.get("cpu", {}).get("cores", 0)
            cpu_available = resources.get("cpu", {}).get("available_cores", 0)
            cpu_allocated = (cpu_total - cpu_available) / max(1, cpu_total)
            
            # Avoid division by zero
            if cpu_allocated > 0:
                cpu_eff = min(1.0, cpu_util / cpu_allocated)
            else:
                cpu_eff = 0
                
            cpu_efficiency.append(cpu_eff * 100)  # Convert to percentage
            
            memory_util = utilization.get("memory", 0)
            memory_total = resources.get("memory", {}).get("total_mb", 0)
            memory_available = resources.get("memory", {}).get("available_mb", 0)
            memory_allocated = (memory_total - memory_available) / max(1, memory_total)
            
            if memory_allocated > 0:
                memory_eff = min(1.0, memory_util / memory_allocated)
            else:
                memory_eff = 0
                
            memory_efficiency.append(memory_eff * 100)
            
            gpu_util = utilization.get("gpu", 0)
            if "gpu" in resources:
                gpu_total = resources.get("gpu", {}).get("memory_mb", 0)
                gpu_available = resources.get("gpu", {}).get("available_memory_mb", 0)
                gpu_allocated = (gpu_total - gpu_available) / max(1, gpu_total)
                
                if gpu_allocated > 0:
                    gpu_eff = min(1.0, gpu_util / gpu_allocated)
                else:
                    gpu_eff = 0
            else:
                gpu_eff = 0
                
            gpu_efficiency.append(gpu_eff * 100)
            
        # Output path
        if not output_path:
            filename = f"resource_efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if interactive and PLOTLY_AVAILABLE:
                output_path = os.path.join(self.output_dir, f"{filename}.html")
            else:
                output_path = os.path.join(self.output_dir, f"{filename}.png")
                
        # Create visualization
        if interactive and PLOTLY_AVAILABLE:
            # Create plotly figure
            fig = go.Figure()
            
            # Add CPU efficiency
            fig.add_trace(
                go.Bar(
                    x=worker_ids,
                    y=cpu_efficiency,
                    name='CPU Efficiency',
                    marker_color='blue'
                )
            )
            
            # Add Memory efficiency
            fig.add_trace(
                go.Bar(
                    x=worker_ids,
                    y=memory_efficiency,
                    name='Memory Efficiency',
                    marker_color='green'
                )
            )
            
            # Add GPU efficiency
            fig.add_trace(
                go.Bar(
                    x=worker_ids,
                    y=gpu_efficiency,
                    name='GPU Efficiency',
                    marker_color='red'
                )
            )
            
            # Update layout
            fig.update_layout(
                title="Resource Allocation Efficiency by Worker",
                xaxis_title="Worker ID",
                yaxis_title="Efficiency (%)",
                barmode='group',
                height=600,
                width=1200,
                yaxis=dict(range=[0, 100])
            )
            
            # Add target line at 100%
            fig.add_shape(
                type="line",
                x0=-0.5,
                y0=100,
                x1=len(worker_ids) - 0.5,
                y1=100,
                line=dict(
                    color="black",
                    width=2,
                    dash="dash",
                )
            )
            
            # Save figure
            fig.write_html(output_path)
            
            # Show figure if requested
            if show_plot:
                fig.show()
                
        else:
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Set up bar positions
            x = np.arange(len(worker_ids))
            width = 0.25
            
            # Plot bars
            cpu_bars = ax.bar(x - width, cpu_efficiency, width, label='CPU Efficiency', color='blue')
            mem_bars = ax.bar(x, memory_efficiency, width, label='Memory Efficiency', color='green')
            gpu_bars = ax.bar(x + width, gpu_efficiency, width, label='GPU Efficiency', color='red')
            
            # Add optimal line
            ax.axhline(y=100, color='black', linestyle='--', linewidth=2, label='Optimal')
            
            # Add labels and title
            ax.set_xlabel('Worker ID')
            ax.set_ylabel('Efficiency (%)')
            ax.set_title('Resource Allocation Efficiency by Worker')
            ax.set_xticks(x)
            ax.set_xticklabels(worker_ids, rotation=45)
            ax.set_ylim(0, 105)
            ax.legend()
            
            # Add grid
            ax.grid(True, axis='y', alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
            # Show figure if requested
            if show_plot:
                plt.show()
            
            plt.close()
            
        logger.info(f"Resource efficiency visualization saved to {output_path}")
        return output_path
        
    def create_resource_dashboard(self, output_dir=None):
        """
        Create a comprehensive resource dashboard with multiple visualizations.
        
        This dashboard includes multiple visualizations of resource utilization,
        scaling history, resource allocation, and efficiency.
        
        Args:
            output_dir: Output directory for the dashboard
            
        Returns:
            Path to the generated dashboard HTML file
        """
        if not output_dir:
            output_dir = os.path.join(self.output_dir, 
                                    f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        heatmap_path = self.create_resource_utilization_heatmap(
            output_path=os.path.join(output_dir, "resource_heatmap.html"),
            interactive=True
        )
        
        scaling_path = self.create_scaling_history_visualization(
            output_path=os.path.join(output_dir, "scaling_history.html"),
            interactive=True
        )
        
        allocation_path = self.create_resource_allocation_visualization(
            output_path=os.path.join(output_dir, "resource_allocation.html"),
            interactive=True
        )
        
        efficiency_path = self.create_resource_efficiency_visualization(
            output_path=os.path.join(output_dir, "resource_efficiency.html"),
            interactive=True
        )
        
        cloud_path = None
        if self.cloud_usage_history:
            cloud_path = self.create_cloud_resource_visualization(
                output_path=os.path.join(output_dir, "cloud_resources.html"),
                interactive=True
            )
            
        # Create dashboard HTML
        dashboard_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Dynamic Resource Management Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }}
        
        .header {{
            background-color: #3f51b5;
            color: white;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        
        .header h1 {{
            margin: 0;
            font-weight: 300;
        }}
        
        .container {{
            width: 90%;
            margin: 0 auto;
            margin-bottom: 40px;
        }}
        
        .dashboard-section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 4px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        .dashboard-iframe {{
            width: 100%;
            height: 650px;
            border: none;
        }}
        
        .dashboard-row {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .half-width {{
            width: calc(50% - 10px);
        }}
        
        .footer {{
            background-color: #3f51b5;
            color: white;
            text-align: center;
            padding: 10px;
            position: fixed;
            bottom: 0;
            width: 100%;
        }}
        
        @media (max-width: 768px) {{
            .dashboard-row {{
                flex-direction: column;
            }}
            
            .half-width {{
                width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Dynamic Resource Management Dashboard</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="container">
        <div class="dashboard-section">
            <h2>Resource Utilization Heatmap</h2>
            <iframe src="{os.path.basename(heatmap_path) if heatmap_path else ''}" class="dashboard-iframe"></iframe>
        </div>
        
        <div class="dashboard-section">
            <h2>Scaling History</h2>
            <iframe src="{os.path.basename(scaling_path) if scaling_path else ''}" class="dashboard-iframe"></iframe>
        </div>
        
        <div class="dashboard-row">
            <div class="dashboard-section half-width">
                <h2>Resource Allocation</h2>
                <iframe src="{os.path.basename(allocation_path) if allocation_path else ''}" class="dashboard-iframe"></iframe>
            </div>
            
            <div class="dashboard-section half-width">
                <h2>Resource Efficiency</h2>
                <iframe src="{os.path.basename(efficiency_path) if efficiency_path else ''}" class="dashboard-iframe"></iframe>
            </div>
        </div>
        
        {f"""
        <div class="dashboard-section">
            <h2>Cloud Resource Usage</h2>
            <iframe src="{os.path.basename(cloud_path)}" class="dashboard-iframe"></iframe>
        </div>
        """ if cloud_path else ""}
    </div>
    
    <div class="footer">
        <p>Dynamic Resource Management - Distributed Testing Framework</p>
    </div>
</body>
</html>
"""
        
        # Write dashboard HTML
        dashboard_path = os.path.join(output_dir, "index.html")
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)
            
        logger.info(f"Resource dashboard created at {dashboard_path}")
        return dashboard_path
        
    def start_dashboard_server(self, port=None, background=True):
        """
        Start a web server to serve the dashboard visualizations.
        
        This starts a Tornado web server that serves the dashboard visualizations
        and provides real-time updates via WebSockets.
        
        Args:
            port: Port to listen on (default: self.dashboard_port)
            background: Whether to run in a background thread
            
        Returns:
            Dashboard URL if started, None otherwise
        """
        if not TORNADO_AVAILABLE:
            logger.error("Tornado is not available, cannot start dashboard server")
            return None
            
        if self.dashboard_running:
            logger.warning("Dashboard server is already running")
            current_port = getattr(self.dashboard_app, "port", self.dashboard_port)
            return f"http://localhost:{current_port}"
            
        port = port or self.dashboard_port
        
        try:
            # Create app
            self.dashboard_app = DRMDashboardApp(self, port)
            
            if background:
                # Start in background thread
                self.dashboard_thread = threading.Thread(
                    target=self.dashboard_app.start,
                    daemon=True
                )
                self.dashboard_thread.start()
            else:
                # Start in current thread (blocking)
                self.dashboard_app.start()
                
            self.dashboard_running = True
            logger.info(f"Dashboard server started at http://localhost:{port}")
            return f"http://localhost:{port}"
            
        except Exception as e:
            logger.error(f"Failed to start dashboard server: {e}")
            return None
            
    def stop_dashboard_server(self):
        """Stop the dashboard server."""
        if not self.dashboard_running:
            return
            
        try:
            if self.dashboard_app:
                self.dashboard_app.stop()
                
            if self.dashboard_thread and self.dashboard_thread.is_alive():
                self.dashboard_thread.join(timeout=5.0)
                
            self.dashboard_running = False
            logger.info("Dashboard server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping dashboard server: {e}")
            
    def _update_dashboard_clients(self):
        """Send updates to connected dashboard clients."""
        if not self.dashboard_running or not self.dashboard_app:
            return
            
        # Collect data for update
        update_data = {
            "timestamp": datetime.now().isoformat(),
            "resource_data": None,
            "scaling_data": None
        }
        
        # Add resource data if available
        if self.resource_history:
            latest = self.resource_history[-1]
            update_data["resource_data"] = {
                "timestamp": latest["timestamp"].isoformat(),
                "worker_count": latest["worker_count"],
                "active_tasks": latest["active_tasks"],
                "overall_utilization": latest["overall_utilization"]
            }
            
        # Add scaling data if available
        if self.scaling_history:
            latest = self.scaling_history[-1]
            update_data["scaling_data"] = {
                "timestamp": latest["timestamp"].isoformat(),
                "action": latest["decision"].get("action", "unknown") 
                        if isinstance(latest["decision"], dict) 
                        else latest["decision"].action
            }
            
        # Send update
        self.dashboard_app.broadcast_update(update_data)
        
    def cleanup(self):
        """Clean up resources used by the visualization system."""
        self._stop_data_collection()
        self.stop_dashboard_server()
        logger.info("DRM Visualization resources cleaned up")


class DRMDashboardApp:
    """
    Web application for the Dynamic Resource Management Dashboard.
    
    This class provides a Tornado web application that serves the DRM dashboard
    and handles WebSocket connections for real-time updates.
    """
    
    def __init__(self, visualization, port=8889):
        """
        Initialize the dashboard application.
        
        Args:
            visualization: DRMVisualization instance
            port: Port to listen on
        """
        self.visualization = visualization
        self.port = port
        self.app = None
        self.server = None
        self.io_loop = None
        
    def start(self):
        """Start the dashboard server."""
        # Create Tornado application
        self.app = tornado.web.Application([
            (r"/", MainHandler, {"visualization": self.visualization}),
            (r"/ws", DashboardWebSocketHandler, {"visualization": self.visualization}),
            (r"/visualizations/(.*)", tornado.web.StaticFileHandler, {"path": self.visualization.output_dir}),
            (r"/updates", UpdatesHandler, {"visualization": self.visualization}),
            (r"/data", DataHandler, {"visualization": self.visualization})
        ])
        
        # Start server
        self.server = self.app.listen(self.port)
        self.io_loop = tornado.ioloop.IOLoop.current()
        
        try:
            self.io_loop.start()
        except KeyboardInterrupt:
            self.stop()
            
    def stop(self):
        """Stop the dashboard server."""
        if self.server:
            self.server.stop()
            
        if self.io_loop:
            self.io_loop.add_callback(self.io_loop.stop)
            
    def broadcast_update(self, data):
        """
        Broadcast an update to all connected WebSocket clients.
        
        Args:
            data: Update data
        """
        # Get WebSocket handler
        for handler in tornado.web.Application.handlers[0][1]:
            if isinstance(handler[0], tornado.web.URLSpec) and \
               handler[0].name == "websocket":
                # Get the WebSocket handler class
                ws_handler_class = handler[1]
                # Broadcast to all clients
                ws_handler_class.broadcast(json.dumps(data))
                break


class MainHandler(tornado.web.RequestHandler):
    """Handler for the main dashboard page."""
    
    def initialize(self, visualization):
        """Initialize with visualization instance."""
        self.visualization = visualization
        
    def get(self):
        """Handle GET request."""
        # Generate dashboard HTML
        dashboard_path = self.visualization.create_resource_dashboard()
        
        # Redirect to dashboard
        self.redirect(os.path.relpath(dashboard_path, self.visualization.output_dir))


class DashboardWebSocketHandler(tornado.websocket.WebSocketHandler):
    """Handler for WebSocket connections."""
    
    # Class variable to keep track of clients
    clients = set()
    
    def initialize(self, visualization):
        """Initialize with visualization instance."""
        self.visualization = visualization
        
    def open(self):
        """Handle WebSocket connection opened."""
        # Add client to set
        DashboardWebSocketHandler.clients.add(self)
        
    def on_close(self):
        """Handle WebSocket connection closed."""
        # Remove client from set
        DashboardWebSocketHandler.clients.discard(self)
        
    def on_message(self, message):
        """Handle WebSocket message."""
        # Process message if needed
        pass
        
    @classmethod
    def broadcast(cls, message):
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: Message to broadcast
        """
        for client in cls.clients:
            try:
                client.write_message(message)
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")


class UpdatesHandler(tornado.web.RequestHandler):
    """Handler for receiving updates about live visualizations."""
    
    def initialize(self, visualization):
        """Initialize with visualization instance."""
        self.visualization = visualization
        
    def get(self):
        """Handle GET request."""
        # Get list of available visualizations
        visualizations = []
        
        # Resource utilization heatmap
        vis_path = self.visualization.create_resource_utilization_heatmap(interactive=True)
        if vis_path:
            visualizations.append({
                "id": "heatmap",
                "title": "Resource Utilization Heatmap",
                "path": os.path.relpath(vis_path, self.visualization.output_dir)
            })
            
        # Scaling history
        vis_path = self.visualization.create_scaling_history_visualization(interactive=True)
        if vis_path:
            visualizations.append({
                "id": "scaling",
                "title": "Scaling History",
                "path": os.path.relpath(vis_path, self.visualization.output_dir)
            })
            
        # Resource allocation
        vis_path = self.visualization.create_resource_allocation_visualization(interactive=True)
        if vis_path:
            visualizations.append({
                "id": "allocation",
                "title": "Resource Allocation",
                "path": os.path.relpath(vis_path, self.visualization.output_dir)
            })
            
        # Resource efficiency
        vis_path = self.visualization.create_resource_efficiency_visualization(interactive=True)
        if vis_path:
            visualizations.append({
                "id": "efficiency",
                "title": "Resource Efficiency",
                "path": os.path.relpath(vis_path, self.visualization.output_dir)
            })
            
        # Cloud resources
        if self.visualization.cloud_usage_history:
            vis_path = self.visualization.create_cloud_resource_visualization(interactive=True)
            if vis_path:
                visualizations.append({
                    "id": "cloud",
                    "title": "Cloud Resource Usage",
                    "path": os.path.relpath(vis_path, self.visualization.output_dir)
                })
                
        # Return as JSON
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps({
            "visualizations": visualizations,
            "timestamp": datetime.now().isoformat()
        }))


class DataHandler(tornado.web.RequestHandler):
    """Handler for fetching raw data."""
    
    def initialize(self, visualization):
        """Initialize with visualization instance."""
        self.visualization = visualization
        
    def get(self):
        """Handle GET request."""
        # Get data type from query parameter
        data_type = self.get_argument("type", "summary")
        
        # Get data based on type
        if data_type == "summary":
            data = self._get_summary_data()
        elif data_type == "resource":
            data = self._get_resource_data()
        elif data_type == "scaling":
            data = self._get_scaling_data()
        elif data_type == "workers":
            data = self._get_worker_data()
        elif data_type == "cloud":
            data = self._get_cloud_data()
        else:
            data = {"error": f"Unknown data type: {data_type}"}
            
        # Return as JSON
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(data))
        
    def _get_summary_data(self):
        """Get summary data."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_workers": 0,
            "active_tasks": 0,
            "utilization": {
                "cpu": 0,
                "memory": 0,
                "gpu": 0,
                "overall": 0
            }
        }
        
        # Get latest resource snapshot if available
        if self.visualization.resource_history:
            latest = self.visualization.resource_history[-1]
            summary["total_workers"] = latest["worker_count"]
            summary["active_tasks"] = latest["active_tasks"]
            summary["utilization"] = latest["overall_utilization"]
            
        return summary
        
    def _get_resource_data(self):
        """Get resource history data."""
        # Get time range from query parameters
        hours = float(self.get_argument("hours", 24))
        
        # Get resource history for the time range
        cutoff_time = datetime.now() - timedelta(hours=hours)
        filtered_history = [
            {
                "timestamp": snapshot["timestamp"].isoformat(),
                "worker_count": snapshot["worker_count"],
                "active_tasks": snapshot["active_tasks"],
                "overall_utilization": snapshot["overall_utilization"]
            }
            for snapshot in self.visualization.resource_history
            if snapshot["timestamp"] >= cutoff_time
        ]
        
        return {
            "resource_history": filtered_history,
            "timestamp": datetime.now().isoformat()
        }
        
    def _get_scaling_data(self):
        """Get scaling history data."""
        # Get time range from query parameters
        hours = float(self.get_argument("hours", 24))
        
        # Get scaling history for the time range
        cutoff_time = datetime.now() - timedelta(hours=hours)
        filtered_history = []
        
        for entry in self.visualization.scaling_history:
            if entry["timestamp"] >= cutoff_time:
                # Extract action from decision
                decision = entry["decision"]
                if isinstance(decision, dict):
                    action = decision.get("action", "unknown")
                    reason = decision.get("reason", "")
                    count = decision.get("count", 0)
                    worker_ids = decision.get("worker_ids", [])
                else:
                    # Assume ScalingDecision object
                    action = decision.action
                    reason = decision.reason
                    count = decision.count
                    worker_ids = decision.worker_ids or []
                    
                filtered_history.append({
                    "timestamp": entry["timestamp"].isoformat(),
                    "action": action,
                    "reason": reason,
                    "count": count,
                    "worker_ids": worker_ids
                })
                
        return {
            "scaling_history": filtered_history,
            "timestamp": datetime.now().isoformat()
        }
        
    def _get_worker_data(self):
        """Get worker data."""
        # Get time range from query parameters
        hours = float(self.get_argument("hours", 24))
        
        # Get worker IDs from query parameters
        worker_id = self.get_argument("worker_id", None)
        
        # Get worker history for the time range
        cutoff_time = datetime.now() - timedelta(hours=hours)
        result = {}
        
        if worker_id:
            # Get history for a specific worker
            if worker_id in self.visualization.worker_history:
                filtered_history = [
                    {
                        "timestamp": entry["timestamp"].isoformat(),
                        "utilization": entry["utilization"],
                        "tasks": entry["tasks"]
                    }
                    for entry in self.visualization.worker_history[worker_id]
                    if entry["timestamp"] >= cutoff_time
                ]
                
                result[worker_id] = filtered_history
        else:
            # Get history for all workers
            for worker_id, history in self.visualization.worker_history.items():
                filtered_history = [
                    {
                        "timestamp": entry["timestamp"].isoformat(),
                        "utilization": entry["utilization"],
                        "tasks": entry["tasks"]
                    }
                    for entry in history
                    if entry["timestamp"] >= cutoff_time
                ]
                
                result[worker_id] = filtered_history
                
        return {
            "worker_history": result,
            "timestamp": datetime.now().isoformat()
        }
        
    def _get_cloud_data(self):
        """Get cloud usage data."""
        # Get time range from query parameters
        hours = float(self.get_argument("hours", 24))
        
        # Get provider from query parameters
        provider = self.get_argument("provider", None)
        
        # Get cloud usage history for the time range
        cutoff_time = datetime.now() - timedelta(hours=hours)
        result = {}
        
        if provider:
            # Get history for a specific provider
            if provider in self.visualization.cloud_usage_history:
                provider_data = {}
                
                for data_type, history in self.visualization.cloud_usage_history[provider].items():
                    filtered_history = [
                        {k: v.isoformat() if isinstance(v, datetime) else v 
                         for k, v in entry.items()}
                        for entry in history
                        if entry["timestamp"] >= cutoff_time
                    ]
                    
                    provider_data[data_type] = filtered_history
                    
                result[provider] = provider_data
        else:
            # Get history for all providers
            for provider, provider_data in self.visualization.cloud_usage_history.items():
                result[provider] = {}
                
                for data_type, history in provider_data.items():
                    filtered_history = [
                        {k: v.isoformat() if isinstance(v, datetime) else v 
                         for k, v in entry.items()}
                        for entry in history
                        if entry["timestamp"] >= cutoff_time
                    ]
                    
                    result[provider][data_type] = filtered_history
                
        return {
            "cloud_history": result,
            "timestamp": datetime.now().isoformat()
        }


# Main entry point
if __name__ == "__main__":
    import argparse
    import sys
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Dynamic Resource Management Visualization")
    parser.add_argument("--drm", help="Path to DynamicResourceManager instance file")
    parser.add_argument("--output-dir", help="Output directory for visualizations")
    parser.add_argument("--dashboard", action="store_true", help="Start dashboard server")
    parser.add_argument("--port", type=int, default=8889, help="Dashboard port")
    
    args = parser.parse_args()
    
    # Load DRM if provided
    drm = None
    if args.drm:
        try:
            # Import module
            import importlib.util
            spec = importlib.util.spec_from_file_location("drm_module", args.drm)
            drm_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(drm_module)
            
            # Get DRM instance
            for attr_name in dir(drm_module):
                attr = getattr(drm_module, attr_name)
                if not attr_name.startswith("_") and isinstance(attr, DynamicResourceManager):
                    drm = attr
                    break
                    
            if not drm:
                raise ValueError("No DynamicResourceManager instance found in the provided file")
                
        except Exception as e:
            print(f"Error loading DRM: {e}")
            sys.exit(1)
            
    # Create visualization
    visualization = DRMVisualization(
        dynamic_resource_manager=drm,
        output_dir=args.output_dir,
        dashboard_port=args.port
    )
    
    # Generate visualizations
    print("Generating visualizations...")
    visualization.create_resource_utilization_heatmap()
    visualization.create_scaling_history_visualization()
    visualization.create_resource_allocation_visualization()
    visualization.create_resource_efficiency_visualization()
    if visualization.cloud_usage_history:
        visualization.create_cloud_resource_visualization()
        
    # Create dashboard
    print("Creating dashboard...")
    dashboard_path = visualization.create_resource_dashboard()
    print(f"Dashboard created at: {dashboard_path}")
    
    # Start dashboard server if requested
    if args.dashboard:
        print(f"Starting dashboard server on port {args.port}...")
        url = visualization.start_dashboard_server(port=args.port, background=False)
        print(f"Dashboard server started at: {url}")
    else:
        # Clean up resources
        visualization.cleanup()