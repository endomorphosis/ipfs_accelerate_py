#!/usr/bin/env python3
"""
Hardware-Aware Scheduler Visualization Module

This module provides visualization tools for the Hardware-Aware Workload Management
system and its integration with the Load Balancer, allowing for better understanding
of scheduling decisions, hardware efficiency, and system performance.
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import json
import math
import random

# Try to import visualization libraries, with graceful fallback
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Import components
from distributed_testing.hardware_workload_management import (
    HardwareWorkloadManager, WorkloadProfile, WorkloadType, WorkloadProfileMetric,
    HardwareTaxonomy, WorkloadExecutionPlan
)
from distributed_testing.hardware_aware_scheduler import HardwareAwareScheduler
from duckdb_api.distributed_testing.hardware_taxonomy import (
    HardwareCapabilityProfile, HardwareClass, SoftwareBackend, PrecisionType
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("hardware_aware_visualization")


class HardwareSchedulingVisualizer:
    """
    Visualization tools for hardware-aware scheduling.
    
    This class provides methods to visualize various aspects of the hardware-aware
    scheduling system, including:
    - Hardware efficiency metrics
    - Workload-to-hardware matching results
    - System performance over time
    - Thermal state visualization
    - Resource utilization
    """
    
    def __init__(self, output_dir: str = None, file_format: str = "png"):
        """
        Initialize the hardware scheduling visualizer.
        
        Args:
            output_dir: Directory to save visualization files (default: current directory)
            file_format: File format for saved visualizations (png, pdf, svg)
        """
        # Check if visualization libraries are available
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib is not available. Visualizations will be limited.")
        
        if not PANDAS_AVAILABLE:
            logger.warning("Pandas is not available. Data processing capabilities will be limited.")
        
        # Set output directory
        self.output_dir = output_dir or os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set file format
        self.file_format = file_format
        
        # History tracking for time-series visualizations
        self.history = {
            "assignments": [],
            "efficiency_scores": {},
            "thermal_states": {},
            "resource_utilization": {},
            "execution_times": {}
        }
    
    def visualize_hardware_efficiency(self, 
                                     hardware_profiles: List[HardwareCapabilityProfile],
                                     workload_profile: WorkloadProfile,
                                     efficiency_scores: Dict[str, float],
                                     filename: str = "hardware_efficiency") -> str:
        """
        Visualize hardware efficiency scores for a specific workload.
        
        Args:
            hardware_profiles: List of hardware capability profiles
            workload_profile: Workload profile being matched
            efficiency_scores: Dictionary of hardware_id -> efficiency score
            filename: Base filename for the visualization
            
        Returns:
            Path to the saved visualization file
        """
        if not MATPLOTLIB_AVAILABLE:
            return "Matplotlib is required for this visualization"
        
        # Prepare data
        hardware_ids = list(efficiency_scores.keys())
        scores = [efficiency_scores[hw_id] for hw_id in hardware_ids]
        
        # Get hardware classes for coloring
        hardware_classes = {}
        for profile in hardware_profiles:
            # Extract worker_id from profile (assuming format: worker_id_profile_name)
            hw_id = None
            for candidate_id in hardware_ids:
                if profile.model_name in candidate_id:
                    hw_id = candidate_id
                    break
            
            if hw_id:
                hardware_classes[hw_id] = profile.hardware_class.value
        
        # Define colors for hardware classes
        class_colors = {
            "cpu": "blue",
            "gpu": "green",
            "tpu": "purple",
            "npu": "orange",
            "hybrid": "red",
            "unknown": "gray"
        }
        
        # Assign colors based on hardware class
        colors = [class_colors.get(hardware_classes.get(hw_id, "unknown"), "gray") for hw_id in hardware_ids]
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Create bar chart
        bars = plt.bar(hardware_ids, scores, color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Add title and labels
        plt.title(f'Hardware Efficiency Scores for {workload_profile.workload_type.value} Workload')
        plt.xlabel('Hardware ID')
        plt.ylabel('Efficiency Score (0-1)')
        
        # Add workload details
        details = (f"Workload Profile: {workload_profile.workload_id}\n"
                  f"Type: {workload_profile.workload_type.value}\n"
                  f"Memory: {workload_profile.min_memory_bytes / (1024*1024*1024):.2f} GB\n"
                  f"Compute Units: {workload_profile.min_compute_units}")
        
        plt.figtext(0.02, 0.02, details, wrap=True, fontsize=8)
        
        # Add legend for hardware classes
        legend_elements = [plt.Rectangle((0,0), 1, 1, color=color, label=hw_class) 
                          for hw_class, color in class_colors.items()
                          if hw_class in hardware_classes.values()]
        
        plt.legend(handles=legend_elements, title="Hardware Class", loc="upper right")
        
        # Adjust layout and save
        plt.tight_layout()
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1.1)  # Set y-axis limit from 0 to 1.1
        
        # Save visualization
        output_path = os.path.join(self.output_dir, f"{filename}.{self.file_format}")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def visualize_workload_distribution(self, 
                                      worker_assignments: Dict[str, List[str]],
                                      worker_types: Dict[str, str],
                                      filename: str = "workload_distribution") -> str:
        """
        Visualize the distribution of workloads across workers.
        
        Args:
            worker_assignments: Dictionary of worker_id -> list of workload_ids
            worker_types: Dictionary of worker_id -> worker type
            filename: Base filename for the visualization
            
        Returns:
            Path to the saved visualization file
        """
        if not MATPLOTLIB_AVAILABLE:
            return "Matplotlib is required for this visualization"
        
        # Prepare data
        worker_ids = list(worker_assignments.keys())
        assignment_counts = [len(assignments) for assignments in worker_assignments.values()]
        
        # Define colors for worker types
        type_colors = {
            "generic": "gray",
            "cpu": "blue",
            "gpu": "green",
            "tpu": "purple",
            "browser": "red",
            "mobile": "orange"
        }
        
        # Assign colors based on worker type
        colors = [type_colors.get(worker_types.get(worker_id, "generic"), "gray") for worker_id in worker_ids]
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Create bar chart
        bars = plt.bar(worker_ids, assignment_counts, color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Add title and labels
        plt.title('Workload Distribution Across Workers')
        plt.xlabel('Worker ID')
        plt.ylabel('Number of Assigned Workloads')
        
        # Add legend for worker types
        legend_elements = [plt.Rectangle((0,0), 1, 1, color=color, label=worker_type) 
                          for worker_type, color in type_colors.items()
                          if worker_type in worker_types.values()]
        
        plt.legend(handles=legend_elements, title="Worker Type", loc="upper right")
        
        # Adjust layout and save
        plt.tight_layout()
        plt.xticks(rotation=45, ha="right")
        
        # Save visualization
        output_path = os.path.join(self.output_dir, f"{filename}.{self.file_format}")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def visualize_thermal_states(self, 
                               thermal_states: Dict[str, Dict[str, float]],
                               filename: str = "thermal_states") -> str:
        """
        Visualize the thermal states of workers.
        
        Args:
            thermal_states: Dictionary of worker_id -> thermal state data
            filename: Base filename for the visualization
            
        Returns:
            Path to the saved visualization file
        """
        if not MATPLOTLIB_AVAILABLE:
            return "Matplotlib is required for this visualization"
        
        # Prepare data
        worker_ids = list(thermal_states.keys())
        temperatures = [state.get("temperature", 0.0) for state in thermal_states.values()]
        warming_states = [1 if state.get("warming_state", False) else 0 for state in thermal_states.values()]
        cooling_states = [1 if state.get("cooling_state", False) else 0 for state in thermal_states.values()]
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Create temperature bar chart
        ax1 = plt.subplot(111)
        temp_bars = ax1.bar(worker_ids, temperatures, color='red', alpha=0.6, label='Temperature')
        
        # Add value labels on top of bars
        for bar in temp_bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        # Add title and labels
        ax1.set_title('Worker Thermal States')
        ax1.set_xlabel('Worker ID')
        ax1.set_ylabel('Temperature (0-1 scale)')
        ax1.set_ylim(0, 1.1)  # Set y-axis limit from 0 to 1.1
        
        # Plot warming/cooling indicators on secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(worker_ids, warming_states, 'ro', label='Warming', alpha=0.7, markersize=10)
        ax2.plot(worker_ids, cooling_states, 'bo', label='Cooling', alpha=0.7, markersize=10)
        ax2.set_ylabel('Thermal State')
        ax2.set_ylim(-0.1, 1.1)  # Set y-axis limit from 0 to 1.1
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Off', 'On'])
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        
        # Adjust layout and save
        plt.tight_layout()
        plt.xticks(rotation=45, ha="right")
        
        # Save visualization
        output_path = os.path.join(self.output_dir, f"{filename}.{self.file_format}")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def visualize_resource_utilization(self, 
                                     worker_loads: Dict[str, Dict[str, float]],
                                     filename: str = "resource_utilization") -> str:
        """
        Visualize resource utilization across workers.
        
        Args:
            worker_loads: Dictionary of worker_id -> resource utilization metrics
            filename: Base filename for the visualization
            
        Returns:
            Path to the saved visualization file
        """
        if not MATPLOTLIB_AVAILABLE:
            return "Matplotlib is required for this visualization"
        
        # Prepare data
        worker_ids = list(worker_loads.keys())
        cpu_util = [load.get("cpu_utilization", 0.0) for load in worker_loads.values()]
        memory_util = [load.get("memory_utilization", 0.0) for load in worker_loads.values()]
        gpu_util = [load.get("gpu_utilization", 0.0) for load in worker_loads.values()]
        io_util = [load.get("io_utilization", 0.0) for load in worker_loads.values()]
        network_util = [load.get("network_utilization", 0.0) for load in worker_loads.values()]
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot each resource utilization as a separate group of bars
        bar_width = 0.15
        index = np.arange(len(worker_ids))
        
        plt.bar(index, cpu_util, bar_width, label='CPU', color='blue')
        plt.bar(index + bar_width, memory_util, bar_width, label='Memory', color='green')
        plt.bar(index + 2 * bar_width, gpu_util, bar_width, label='GPU', color='red')
        plt.bar(index + 3 * bar_width, io_util, bar_width, label='I/O', color='purple')
        plt.bar(index + 4 * bar_width, network_util, bar_width, label='Network', color='orange')
        
        # Add title and labels
        plt.title('Resource Utilization Across Workers')
        plt.xlabel('Worker ID')
        plt.ylabel('Utilization (%)')
        plt.xticks(index + 2 * bar_width, worker_ids, rotation=45, ha="right")
        plt.ylim(0, 100)  # Set y-axis limit from 0 to 100%
        
        # Add legend
        plt.legend(loc="upper right")
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.output_dir, f"{filename}.{self.file_format}")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def visualize_execution_times(self, 
                                execution_data: Dict[str, Dict[str, Any]],
                                filename: str = "execution_times") -> str:
        """
        Visualize execution times compared to estimates.
        
        Args:
            execution_data: Dictionary of workload_id -> execution data
            filename: Base filename for the visualization
            
        Returns:
            Path to the saved visualization file
        """
        if not MATPLOTLIB_AVAILABLE:
            return "Matplotlib is required for this visualization"
        
        # Prepare data
        workload_ids = list(execution_data.keys())
        estimated_times = [data.get("estimated_time", 0.0) for data in execution_data.values()]
        actual_times = [data.get("actual_time", 0.0) for data in execution_data.values()]
        workload_types = [data.get("workload_type", "unknown") for data in execution_data.values()]
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Define colors for workload types
        type_colors = {
            "VISION": "blue",
            "NLP": "green",
            "AUDIO": "purple",
            "MULTIMODAL": "red",
            "TRAINING": "orange",
            "INFERENCE": "cyan",
            "EMBEDDING": "magenta",
            "CONVERSATIONAL": "yellow",
            "MIXED": "gray",
            "unknown": "black"
        }
        
        # Assign colors based on workload type
        colors = [type_colors.get(wl_type, "gray") for wl_type in workload_types]
        
        # Create scatter plot of estimated vs actual times
        plt.scatter(estimated_times, actual_times, c=colors, alpha=0.7, s=50)
        
        # Add diagonal line (perfect prediction)
        max_time = max(max(estimated_times), max(actual_times)) * 1.1
        plt.plot([0, max_time], [0, max_time], 'k--', alpha=0.5)
        
        # Add title and labels
        plt.title('Estimated vs Actual Execution Times')
        plt.xlabel('Estimated Time (seconds)')
        plt.ylabel('Actual Time (seconds)')
        
        # Add legend for workload types
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=wl_type) 
                          for wl_type, color in type_colors.items()
                          if wl_type in workload_types]
        
        plt.legend(handles=legend_elements, title="Workload Type", loc="upper left")
        
        # Add annotations for outliers
        for i, (est, act, wl_id) in enumerate(zip(estimated_times, actual_times, workload_ids)):
            if abs(est - act) / max(est, 1.0) > 0.3:  # If actual time differs by more than 30% from estimate
                plt.annotate(wl_id, (est, act), fontsize=8, alpha=0.7)
        
        # Adjust layout and save
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.output_dir, f"{filename}.{self.file_format}")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def visualize_history(self, history_data: Dict[str, Any], filename_prefix: str = "history") -> List[str]:
        """
        Visualize historical data over time.
        
        Args:
            history_data: Historical data to visualize
            filename_prefix: Prefix for filenames
            
        Returns:
            List of paths to saved visualization files
        """
        if not MATPLOTLIB_AVAILABLE:
            return ["Matplotlib is required for this visualization"]
        
        output_paths = []
        
        # Visualize efficiency scores over time
        if "efficiency_scores" in history_data and history_data["efficiency_scores"]:
            eff_filename = f"{filename_prefix}_efficiency_scores"
            eff_path = self._visualize_efficiency_history(history_data["efficiency_scores"], eff_filename)
            output_paths.append(eff_path)
        
        # Visualize thermal states over time
        if "thermal_states" in history_data and history_data["thermal_states"]:
            thermal_filename = f"{filename_prefix}_thermal_states"
            thermal_path = self._visualize_thermal_history(history_data["thermal_states"], thermal_filename)
            output_paths.append(thermal_path)
        
        # Visualize resource utilization over time
        if "resource_utilization" in history_data and history_data["resource_utilization"]:
            resource_filename = f"{filename_prefix}_resource_utilization"
            resource_path = self._visualize_resource_history(history_data["resource_utilization"], resource_filename)
            output_paths.append(resource_path)
        
        return output_paths
    
    def _visualize_efficiency_history(self, efficiency_history: Dict[str, List[Tuple[datetime, float]]],
                                    filename: str) -> str:
        """Visualize efficiency scores over time."""
        plt.figure(figsize=(12, 6))
        
        for hardware_id, data_points in efficiency_history.items():
            if not data_points:
                continue
                
            timestamps, scores = zip(*data_points)
            plt.plot(timestamps, scores, label=hardware_id, marker='o', markersize=4, alpha=0.7)
        
        plt.title('Hardware Efficiency Scores Over Time')
        plt.xlabel('Time')
        plt.ylabel('Efficiency Score (0-1)')
        plt.legend(loc="upper right")
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis date labels
        plt.gcf().autofmt_xdate()
        
        # Save visualization
        output_path = os.path.join(self.output_dir, f"{filename}.{self.file_format}")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def _visualize_thermal_history(self, thermal_history: Dict[str, List[Tuple[datetime, Dict[str, Any]]]],
                                 filename: str) -> str:
        """Visualize thermal states over time."""
        plt.figure(figsize=(12, 6))
        
        for worker_id, data_points in thermal_history.items():
            if not data_points:
                continue
                
            timestamps = [dp[0] for dp in data_points]
            temperatures = [dp[1].get("temperature", 0.0) for dp in data_points]
            
            plt.plot(timestamps, temperatures, label=f"{worker_id} Temp", marker='o', markersize=4, alpha=0.7)
            
            # Plot warming/cooling events
            warming_events = [(ts, 0.05) for ts, state in data_points if state.get("warming_state", False)]
            cooling_events = [(ts, 0.1) for ts, state in data_points if state.get("cooling_state", False)]
            
            if warming_events:
                w_timestamps, w_markers = zip(*warming_events)
                plt.scatter(w_timestamps, w_markers, marker='^', color='red', label=f"{worker_id} Warming")
            
            if cooling_events:
                c_timestamps, c_markers = zip(*cooling_events)
                plt.scatter(c_timestamps, c_markers, marker='v', color='blue', label=f"{worker_id} Cooling")
        
        plt.title('Thermal States Over Time')
        plt.xlabel('Time')
        plt.ylabel('Temperature (0-1 scale)')
        plt.legend(loc="upper right")
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis date labels
        plt.gcf().autofmt_xdate()
        
        # Save visualization
        output_path = os.path.join(self.output_dir, f"{filename}.{self.file_format}")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def _visualize_resource_history(self, resource_history: Dict[str, List[Tuple[datetime, Dict[str, float]]]],
                                  filename: str) -> str:
        """Visualize resource utilization over time."""
        # We'll create separate plots for each resource type
        resource_types = ["cpu_utilization", "memory_utilization", "gpu_utilization"]
        
        plt.figure(figsize=(12, 10))
        
        for i, resource_type in enumerate(resource_types):
            plt.subplot(3, 1, i+1)
            
            for worker_id, data_points in resource_history.items():
                if not data_points:
                    continue
                    
                timestamps = [dp[0] for dp in data_points]
                utilization = [dp[1].get(resource_type, 0.0) for dp in data_points]
                
                plt.plot(timestamps, utilization, label=worker_id, marker='o', markersize=3, alpha=0.7)
            
            plt.title(f'{resource_type.replace("_", " ").title()} Over Time')
            plt.xlabel('Time')
            plt.ylabel('Utilization (%)')
            plt.legend(loc="upper right")
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            
            # Format x-axis date labels
            plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(self.output_dir, f"{filename}.{self.file_format}")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def record_assignment(self, 
                         workload_id: str, 
                         worker_id: str, 
                         efficiency_score: float,
                         workload_type: str,
                         timestamp: datetime = None) -> None:
        """
        Record a workload assignment for historical tracking.
        
        Args:
            workload_id: ID of the workload
            worker_id: ID of the worker the workload was assigned to
            efficiency_score: Efficiency score for this assignment
            workload_type: Type of workload
            timestamp: Time of assignment (defaults to now)
        """
        timestamp = timestamp or datetime.now()
        
        # Record assignment
        self.history["assignments"].append({
            "timestamp": timestamp,
            "workload_id": workload_id,
            "worker_id": worker_id,
            "efficiency_score": efficiency_score,
            "workload_type": workload_type
        })
        
        # Record efficiency score
        if worker_id not in self.history["efficiency_scores"]:
            self.history["efficiency_scores"][worker_id] = []
            
        self.history["efficiency_scores"][worker_id].append((timestamp, efficiency_score))
    
    def record_thermal_state(self, 
                           worker_id: str, 
                           temperature: float,
                           warming_state: bool,
                           cooling_state: bool,
                           timestamp: datetime = None) -> None:
        """
        Record a thermal state update for historical tracking.
        
        Args:
            worker_id: ID of the worker
            temperature: Temperature value (0-1 scale)
            warming_state: Whether the worker is in warming state
            cooling_state: Whether the worker is in cooling state
            timestamp: Time of update (defaults to now)
        """
        timestamp = timestamp or datetime.now()
        
        # Record thermal state
        if worker_id not in self.history["thermal_states"]:
            self.history["thermal_states"][worker_id] = []
            
        self.history["thermal_states"][worker_id].append((
            timestamp, 
            {
                "temperature": temperature,
                "warming_state": warming_state,
                "cooling_state": cooling_state
            }
        ))
    
    def record_resource_utilization(self, 
                                  worker_id: str, 
                                  utilization: Dict[str, float],
                                  timestamp: datetime = None) -> None:
        """
        Record resource utilization for historical tracking.
        
        Args:
            worker_id: ID of the worker
            utilization: Dictionary of resource utilization metrics
            timestamp: Time of update (defaults to now)
        """
        timestamp = timestamp or datetime.now()
        
        # Record resource utilization
        if worker_id not in self.history["resource_utilization"]:
            self.history["resource_utilization"][worker_id] = []
            
        self.history["resource_utilization"][worker_id].append((timestamp, utilization))
    
    def record_execution_time(self, 
                           workload_id: str, 
                           estimated_time: float,
                           actual_time: float,
                           workload_type: str,
                           worker_id: str,
                           timestamp: datetime = None) -> None:
        """
        Record execution time comparison for historical tracking.
        
        Args:
            workload_id: ID of the workload
            estimated_time: Estimated execution time in seconds
            actual_time: Actual execution time in seconds
            workload_type: Type of workload
            worker_id: ID of the worker that executed the workload
            timestamp: Time of completion (defaults to now)
        """
        timestamp = timestamp or datetime.now()
        
        # Record execution time
        self.history["execution_times"][workload_id] = {
            "timestamp": timestamp,
            "estimated_time": estimated_time,
            "actual_time": actual_time,
            "workload_type": workload_type,
            "worker_id": worker_id
        }
    
    def save_history(self, filename: str = "scheduling_history.json") -> str:
        """
        Save historical data to a JSON file.
        
        Args:
            filename: Filename for the JSON file
            
        Returns:
            Path to the saved file
        """
        # Prepare data for serialization (convert datetime objects to strings)
        serializable_history = {
            "assignments": [
                {**assignment, "timestamp": assignment["timestamp"].isoformat()}
                for assignment in self.history["assignments"]
            ],
            "efficiency_scores": {
                worker_id: [(ts.isoformat(), score) for ts, score in scores]
                for worker_id, scores in self.history["efficiency_scores"].items()
            },
            "thermal_states": {
                worker_id: [(ts.isoformat(), state) for ts, state in states]
                for worker_id, states in self.history["thermal_states"].items()
            },
            "resource_utilization": {
                worker_id: [(ts.isoformat(), util) for ts, util in utils]
                for worker_id, utils in self.history["resource_utilization"].items()
            },
            "execution_times": {
                workload_id: {**data, "timestamp": data["timestamp"].isoformat()}
                for workload_id, data in self.history["execution_times"].items()
            }
        }
        
        # Save to file
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
            
        return output_path
    
    def load_history(self, filename: str) -> bool:
        """
        Load historical data from a JSON file.
        
        Args:
            filename: Path to the JSON file
            
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            # Load JSON data
            with open(filename, 'r') as f:
                data = json.load(f)
                
            # Parse datetime strings back to datetime objects
            parsed_history = {
                "assignments": [
                    {**assignment, "timestamp": datetime.fromisoformat(assignment["timestamp"])}
                    for assignment in data.get("assignments", [])
                ],
                "efficiency_scores": {
                    worker_id: [(datetime.fromisoformat(ts), score) for ts, score in scores]
                    for worker_id, scores in data.get("efficiency_scores", {}).items()
                },
                "thermal_states": {
                    worker_id: [(datetime.fromisoformat(ts), state) for ts, state in states]
                    for worker_id, states in data.get("thermal_states", {}).items()
                },
                "resource_utilization": {
                    worker_id: [(datetime.fromisoformat(ts), util) for ts, util in utils]
                    for worker_id, utils in data.get("resource_utilization", {}).items()
                },
                "execution_times": {
                    workload_id: {**data_item, "timestamp": datetime.fromisoformat(data_item["timestamp"])}
                    for workload_id, data_item in data.get("execution_times", {}).items()
                }
            }
            
            # Update history
            self.history = parsed_history
            
            return True
        except Exception as e:
            logger.error(f"Error loading history from {filename}: {e}")
            return False
    
    def generate_summary_report(self, 
                              filename: str = "scheduling_summary_report.html",
                              include_visualizations: bool = True) -> str:
        """
        Generate a comprehensive HTML report summarizing the scheduling history.
        
        Args:
            filename: Filename for the HTML report
            include_visualizations: Whether to include visualizations in the report
            
        Returns:
            Path to the generated report file
        """
        # Prepare report data
        total_assignments = len(self.history["assignments"])
        workers_with_assignments = set(assignment["worker_id"] for assignment in self.history["assignments"])
        workload_types = set(assignment["workload_type"] for assignment in self.history["assignments"])
        
        # Calculate average efficiency scores
        worker_avg_efficiency = {}
        for worker_id, scores in self.history["efficiency_scores"].items():
            if scores:
                worker_avg_efficiency[worker_id] = sum(score for _, score in scores) / len(scores)
        
        # Calculate efficiency by workload type
        workload_type_efficiency = {}
        for assignment in self.history["assignments"]:
            workload_type = assignment["workload_type"]
            if workload_type not in workload_type_efficiency:
                workload_type_efficiency[workload_type] = []
            workload_type_efficiency[workload_type].append(assignment["efficiency_score"])
        
        workload_type_avg_efficiency = {
            wl_type: sum(scores) / len(scores) if scores else 0.0
            for wl_type, scores in workload_type_efficiency.items()
        }
        
        # Calculate execution time accuracy
        execution_time_accuracy = {}
        for workload_id, data in self.history["execution_times"].items():
            est_time = data["estimated_time"]
            act_time = data["actual_time"]
            
            if est_time > 0:
                accuracy = act_time / est_time
                execution_time_accuracy[workload_id] = accuracy
        
        avg_time_accuracy = sum(execution_time_accuracy.values()) / len(execution_time_accuracy) if execution_time_accuracy else 0.0
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Hardware-Aware Scheduling Summary Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .visualization {{ margin: 20px 0; text-align: center; }}
                .visualization img {{ max-width: 100%; height: auto; }}
                .summary-box {{ background-color: #f0f0f0; border-radius: 5px; padding: 15px; margin-bottom: 20px; }}
                .good {{ color: green; }}
                .warning {{ color: orange; }}
                .poor {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Hardware-Aware Scheduling Summary Report</h1>
            <div class="summary-box">
                <h2>Summary Statistics</h2>
                <p>Total Workload Assignments: <strong>{total_assignments}</strong></p>
                <p>Number of Workers: <strong>{len(workers_with_assignments)}</strong></p>
                <p>Workload Types: <strong>{', '.join(workload_types)}</strong></p>
                <p>Average Execution Time Accuracy: <strong class="{self._get_accuracy_class(avg_time_accuracy)}">{avg_time_accuracy:.2f}</strong></p>
            </div>
            
            <h2>Worker Efficiency</h2>
            <table>
                <tr>
                    <th>Worker ID</th>
                    <th>Average Efficiency Score</th>
                </tr>
        """
        
        # Add worker efficiency rows
        for worker_id, avg_eff in sorted(worker_avg_efficiency.items(), key=lambda x: x[1], reverse=True):
            eff_class = "good" if avg_eff >= 0.7 else "warning" if avg_eff >= 0.4 else "poor"
            html_content += f"""
                <tr>
                    <td>{worker_id}</td>
                    <td class="{eff_class}">{avg_eff:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Workload Type Efficiency</h2>
            <table>
                <tr>
                    <th>Workload Type</th>
                    <th>Average Efficiency Score</th>
                </tr>
        """
        
        # Add workload type efficiency rows
        for wl_type, avg_eff in sorted(workload_type_avg_efficiency.items(), key=lambda x: x[1], reverse=True):
            eff_class = "good" if avg_eff >= 0.7 else "warning" if avg_eff >= 0.4 else "poor"
            html_content += f"""
                <tr>
                    <td>{wl_type}</td>
                    <td class="{eff_class}">{avg_eff:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Execution Time Accuracy</h2>
            <table>
                <tr>
                    <th>Workload ID</th>
                    <th>Estimated Time (s)</th>
                    <th>Actual Time (s)</th>
                    <th>Accuracy Ratio</th>
                </tr>
        """
        
        # Add execution time accuracy rows
        for workload_id, data in sorted(self.history["execution_times"].items()):
            est_time = data["estimated_time"]
            act_time = data["actual_time"]
            accuracy = act_time / est_time if est_time > 0 else 0.0
            accuracy_class = self._get_accuracy_class(accuracy)
            
            html_content += f"""
                <tr>
                    <td>{workload_id}</td>
                    <td>{est_time:.2f}</td>
                    <td>{act_time:.2f}</td>
                    <td class="{accuracy_class}">{accuracy:.2f}</td>
                </tr>
            """
        
        html_content += """
            </table>
        """
        
        # Add visualizations if requested
        if include_visualizations and MATPLOTLIB_AVAILABLE:
            # Generate visualizations
            visualization_paths = []
            
            # Resource utilization
            if self.history["resource_utilization"]:
                worker_loads = {
                    worker_id: utils[-1][1] if utils else {}
                    for worker_id, utils in self.history["resource_utilization"].items()
                }
                resource_viz_path = self.visualize_resource_utilization(worker_loads, "report_resource_utilization")
                visualization_paths.append(("Resource Utilization", resource_viz_path))
            
            # Thermal states
            if self.history["thermal_states"]:
                thermal_states = {
                    worker_id: states[-1][1] if states else {}
                    for worker_id, states in self.history["thermal_states"].items()
                }
                thermal_viz_path = self.visualize_thermal_states(thermal_states, "report_thermal_states")
                visualization_paths.append(("Thermal States", thermal_viz_path))
            
            # Execution times
            if self.history["execution_times"]:
                execution_data = {
                    workload_id: {
                        "estimated_time": data["estimated_time"],
                        "actual_time": data["actual_time"],
                        "workload_type": data["workload_type"]
                    }
                    for workload_id, data in self.history["execution_times"].items()
                }
                execution_viz_path = self.visualize_execution_times(execution_data, "report_execution_times")
                visualization_paths.append(("Execution Times", execution_viz_path))
            
            # Add visualizations to report
            for title, path in visualization_paths:
                filename = os.path.basename(path)
                html_content += f"""
                <div class="visualization">
                    <h3>{title}</h3>
                    <img src="{filename}" alt="{title}">
                </div>
                """
        
        # Close HTML document
        html_content += """
        </body>
        </html>
        """
        
        # Save report
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        return output_path
    
    def _get_accuracy_class(self, accuracy: float) -> str:
        """Get CSS class for accuracy value."""
        if 0.8 <= accuracy <= 1.2:
            return "good"  # Within 20% of estimate
        elif 0.5 <= accuracy < 0.8 or 1.2 < accuracy <= 1.5:
            return "warning"  # Within 50% of estimate
        else:
            return "poor"  # More than 50% off
            

# Utility function to create a visualizer instance
def create_visualizer(output_dir: str = None, file_format: str = "png") -> HardwareSchedulingVisualizer:
    """
    Create and return a hardware scheduling visualizer.
    
    Args:
        output_dir: Directory to save visualization files
        file_format: File format for saved visualizations
        
    Returns:
        HardwareSchedulingVisualizer instance
    """
    return HardwareSchedulingVisualizer(output_dir, file_format)