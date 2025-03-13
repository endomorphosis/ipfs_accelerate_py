#!/usr/bin/env python3
"""
Visualization tools for Load Balancer performance results.

This script generates visualizations from the JSON results produced by 
the load balancer stress tests and benchmarks, including:
1. Throughput and latency comparison charts
2. Worker utilization heat maps 
3. Scalability analysis
4. Resource efficiency visualization
5. Time series performance visualization for load spikes
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from datetime import datetime
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def load_results(filepath: str) -> Dict[str, Any]:
    """Load test results from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_single_test_results(results: Dict[str, Any], output_dir: str) -> None:
    """Generate plots for a single stress test result."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract configuration and metrics
    config = results.get("configuration", {})
    metrics = results.get("metrics", {})
    
    # Plot title with key configuration
    title_base = f"Load Balancer Test: {config.get('workers', 'N/A')} workers, {config.get('tests', 'N/A')} tests"
    title_suffix = ""
    if config.get("burst_mode"):
        title_suffix += " (Burst Mode)"
    if config.get("dynamic_workers"):
        title_suffix += " (Dynamic Workers)"
        
    # Timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Performance Summary Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_to_plot = ['success_rate', 'avg_latency', 'peak_throughput', 'worker_utilization']
    labels = ['Success Rate (%)', 'Avg Latency (s)', 'Peak Throughput (tests/s)', 'Worker Utilization (%)']
    
    # Normalize values for chart
    normalized_values = []
    for metric, label in zip(metrics_to_plot, labels):
        value = metrics.get(metric, 0)
        if metric == 'success_rate':
            normalized_values.append(value / 100)  # Already percentage
        elif metric == 'avg_latency':
            normalized_values.append(1.0 - min(value / 10.0, 1.0))  # Invert latency, lower is better
        elif metric == 'peak_throughput':
            normalized_values.append(min(value / 100.0, 1.0))  # Normalize to 0-1 range
        elif metric == 'worker_utilization':
            normalized_values.append(value)  # Already 0-1
    
    # Create bar chart
    bars = ax.bar(labels, normalized_values, color=['#4CAF50', '#2196F3', '#FFC107', '#9C27B0'])
    
    # Add value labels on bars
    for i, (metric, bar) in enumerate(zip(metrics_to_plot, bars)):
        value = metrics.get(metric, 0)
        if metric == 'success_rate' or metric == 'worker_utilization':
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{value:.1f}%",
                    ha='center', va='bottom', fontweight='bold')
        elif metric == 'avg_latency':
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{metrics.get(metric, 0):.2f}s",
                    ha='center', va='bottom', fontweight='bold')
        else:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{int(value)}",
                    ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylim(0, 1.3)
    ax.set_title(f"Performance Summary\n{title_base}{title_suffix}")
    ax.set_ylabel('Normalized Score (higher is better)')
    
    # Save figure
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"performance_summary_{timestamp}.png"), dpi=300)
    plt.close(fig)
    
    # 2. Latency Distribution (if available in detailed results)
    if "time_series" in results and "latency" in results["time_series"]:
        latency_data = results["time_series"]["latency"]
        if latency_data:
            times, latencies = zip(*latency_data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(times, latencies, 'b-', linewidth=2)
            ax.set_title(f"Latency Over Time\n{title_base}{title_suffix}")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Latency (s)')
            ax.grid(True)
            
            # Save figure
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"latency_time_series_{timestamp}.png"), dpi=300)
            plt.close(fig)
    
    # 3. Throughput Distribution (if available in detailed results)
    if "time_series" in results and "throughput" in results["time_series"]:
        throughput_data = results["time_series"]["throughput"]
        if throughput_data:
            times, throughputs = zip(*throughput_data)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(times, throughputs, 'g-', linewidth=2)
            ax.set_title(f"Throughput Over Time\n{title_base}{title_suffix}")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Throughput (tests/s)')
            ax.grid(True)
            
            # Save figure
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"throughput_time_series_{timestamp}.png"), dpi=300)
            plt.close(fig)
    
    # 4. Create combined dashboard
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 4.1 Success Rate & Latency
    ax1 = axs[0, 0]
    metrics_to_plot = ['success_rate', 'avg_latency']
    values = [metrics.get(m, 0) for m in metrics_to_plot]
    colors = ['#4CAF50', '#2196F3']
    
    bars1 = ax1.bar(['Success Rate', 'Avg Latency (s)'], values, color=colors)
    ax1.set_title('Success Rate & Latency')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 4.2 Throughput
    ax2 = axs[0, 1]
    metrics_to_plot = ['peak_throughput', 'avg_throughput']
    values = [metrics.get(m, 0) for m in metrics_to_plot]
    colors = ['#FFC107', '#FF5722']
    
    bars2 = ax2.bar(['Peak Throughput', 'Avg Throughput'], values, color=colors)
    ax2.set_title('Throughput (tests/s)')
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 4.3 Worker Distribution
    ax3 = axs[1, 0]
    metrics_to_plot = ['worker_assignment_stddev', 'worker_assignment_range', 'avg_scheduling_attempts']
    values = [metrics.get(m, 0) for m in metrics_to_plot]
    colors = ['#9C27B0', '#E91E63', '#3F51B5']
    
    bars3 = ax3.bar(['Assignment StdDev', 'Assignment Range', 'Avg Scheduling Attempts'], values, color=colors)
    ax3.set_title('Worker Distribution')
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 4.4 Resource Utilization
    ax4 = axs[1, 1]
    worker_count = config.get('workers', 0)
    final_worker_count = metrics.get('final_worker_count', worker_count)
    workers_data = [worker_count, final_worker_count, worker_count * metrics.get('worker_utilization', 0)]
    
    bars4 = ax4.bar(['Initial Workers', 'Final Workers', 'Effective Workers'], workers_data, color=['#009688', '#607D8B', '#8BC34A'])
    ax4.set_title('Worker Utilization')
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    # Set title and layout
    fig.suptitle(f"Load Balancer Performance Dashboard\n{title_base}{title_suffix}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save combined dashboard
    fig.savefig(os.path.join(output_dir, f"performance_dashboard_{timestamp}.png"), dpi=300)
    plt.close(fig)


def plot_benchmark_results(results: Dict[str, Any], output_dir: str) -> None:
    """Generate plots for benchmark suite results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract benchmark results
    benchmark_results = results.get("benchmark_results", [])
    if not benchmark_results:
        print("No benchmark results found")
        return
        
    # Extract timestamp 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert results to pandas DataFrame for easier manipulation
    df = pd.DataFrame(benchmark_results)
    
    # Extract metrics into separate columns
    for metric in ['success_rate', 'avg_latency', 'peak_throughput', 'worker_utilization']:
        df[metric] = df['metrics'].apply(lambda x: x.get(metric, 0))
    
    # 1. 3D Scalability Plot - workers, tests, throughput
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique worker and test counts for grid
    worker_counts = sorted(df['workers'].unique())
    test_counts = sorted(df['tests'].unique())
    
    # Create coordinate matrices
    X, Y = np.meshgrid(worker_counts, test_counts)
    
    # Create empty Z matrix for throughput 
    Z = np.zeros_like(X, dtype=float)
    
    # Fill Z matrix with throughput values
    for i, workers in enumerate(worker_counts):
        for j, tests in enumerate(test_counts):
            mask = (df['workers'] == workers) & (df['tests'] == tests)
            if mask.any():
                Z[j, i] = df.loc[mask, 'peak_throughput'].values[0]
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    
    # Add color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Peak Throughput (tests/s)')
    
    # Set labels
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Number of Tests')
    ax.set_zlabel('Peak Throughput (tests/s)')
    ax.set_title('Load Balancer Scalability Analysis')
    
    # Save figure
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"scalability_3d_{timestamp}.png"), dpi=300)
    plt.close(fig)
    
    # 2. Heatmap of worker utilization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create empty matrix for worker utilization
    worker_util = np.zeros_like(X, dtype=float)
    
    # Fill matrix with worker utilization values
    for i, workers in enumerate(worker_counts):
        for j, tests in enumerate(test_counts):
            mask = (df['workers'] == workers) & (df['tests'] == tests)
            if mask.any():
                worker_util[j, i] = df.loc[mask, 'worker_utilization'].values[0] * 100  # Convert to percentage
    
    # Create heatmap
    im = ax.imshow(worker_util, cmap='YlGnBu', interpolation='nearest')
    
    # Add color bar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Worker Utilization (%)')
    
    # Show all ticks and label them
    ax.set_xticks(np.arange(len(worker_counts)))
    ax.set_yticks(np.arange(len(test_counts)))
    ax.set_xticklabels(worker_counts)
    ax.set_yticklabels(test_counts)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(len(test_counts)):
        for j in range(len(worker_counts)):
            text = ax.text(j, i, f"{worker_util[i, j]:.1f}%",
                          ha="center", va="center", color="black" if worker_util[i, j] < 70 else "white")
    
    ax.set_title("Worker Utilization Heatmap")
    ax.set_xlabel("Number of Workers")
    ax.set_ylabel("Number of Tests")
    
    # Save figure
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"worker_utilization_heatmap_{timestamp}.png"), dpi=300)
    plt.close(fig)
    
    # 3. Line chart of latency by worker count
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by workers and tests, then calculate mean latency
    for test_count in test_counts:
        test_data = df[df['tests'] == test_count]
        if not test_data.empty:
            test_data = test_data.sort_values('workers')
            ax.plot(test_data['workers'], test_data['avg_latency'], 
                    marker='o', linewidth=2, label=f'{test_count} tests')
    
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Average Latency (s)')
    ax.set_title('Latency Scaling with Worker Count')
    ax.grid(True)
    ax.legend()
    
    # Save figure
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"latency_scaling_{timestamp}.png"), dpi=300)
    plt.close(fig)
    
    # 4. Worker efficiency chart (throughput per worker)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate throughput per worker
    df['throughput_per_worker'] = df['peak_throughput'] / df['workers']
    
    # Group by workers and calculate mean throughput per worker
    for test_count in test_counts:
        test_data = df[df['tests'] == test_count]
        if not test_data.empty:
            test_data = test_data.sort_values('workers')
            ax.plot(test_data['workers'], test_data['throughput_per_worker'], 
                    marker='o', linewidth=2, label=f'{test_count} tests')
    
    ax.set_xlabel('Number of Workers')
    ax.set_ylabel('Throughput per Worker (tests/s/worker)')
    ax.set_title('Worker Efficiency Analysis')
    ax.grid(True)
    ax.legend()
    
    # Save figure
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"worker_efficiency_{timestamp}.png"), dpi=300)
    plt.close(fig)
    
    # 5. Combined dashboard of key metrics
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # 5.1 Success Rate by Configuration
    ax1 = axs[0, 0]
    pivot_success = df.pivot_table(values='success_rate', index='tests', columns='workers')
    sns_plot = sns.heatmap(pivot_success, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax1)
    ax1.set_title('Success Rate (%)')
    
    # 5.2 Latency by Configuration
    ax2 = axs[0, 1]
    pivot_latency = df.pivot_table(values='avg_latency', index='tests', columns='workers')
    sns_plot = sns.heatmap(pivot_latency, annot=True, fmt='.2f', cmap='YlOrRd_r', ax=ax2)
    ax2.set_title('Average Latency (s)')
    
    # 5.3 Throughput by Configuration
    ax3 = axs[1, 0]
    pivot_throughput = df.pivot_table(values='peak_throughput', index='tests', columns='workers')
    sns_plot = sns.heatmap(pivot_throughput, annot=True, fmt='.1f', cmap='viridis', ax=ax3)
    ax3.set_title('Peak Throughput (tests/s)')
    
    # 5.4 Worker Utilization by Configuration
    ax4 = axs[1, 1]
    # Convert worker_utilization to percentage
    df['worker_utilization_pct'] = df['worker_utilization'] * 100
    pivot_util = df.pivot_table(values='worker_utilization_pct', index='tests', columns='workers')
    sns_plot = sns.heatmap(pivot_util, annot=True, fmt='.1f', cmap='YlGnBu', ax=ax4)
    ax4.set_title('Worker Utilization (%)')
    
    # Set overall title
    fig.suptitle('Load Balancer Benchmark Results', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save the dashboard
    fig.savefig(os.path.join(output_dir, f"benchmark_dashboard_{timestamp}.png"), dpi=300)
    plt.close(fig)


def plot_spike_simulation_results(results: Dict[str, Any], output_dir: str) -> None:
    """Generate plots for load spike simulation results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract configuration and metrics
    config = results.get("configuration", {})
    metrics = results.get("metrics", {})
    
    # Check if we have time series data
    if "time_series" not in results:
        print("No time series data found for spike visualization")
        return
        
    time_series = results["time_series"]
    
    # Extract timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Throughput Over Time
    if "throughput" in time_series and time_series["throughput"]:
        throughput_data = time_series["throughput"]
        times, throughputs = zip(*throughput_data)
        
        # Normalize times to start from 0
        start_time = times[0]
        times = [t - start_time for t in times]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(times, throughputs, 'g-', linewidth=2)
        ax.set_title(f"Throughput During Load Spike\n{config.get('initial_workers', 'N/A')} initial workers, {config.get('tests', 'N/A')} tests")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Throughput (tests/s)')
        ax.grid(True)
        
        # Add horizontal line for average throughput
        avg_throughput = metrics.get('avg_throughput', 0)
        ax.axhline(y=avg_throughput, color='r', linestyle='--', label=f'Avg: {avg_throughput:.1f} tests/s')
        
        # Add vertical lines for potential spike points (find sudden increases)
        if len(throughputs) > 5:
            # Calculate moving average differences
            diffs = [throughputs[i+1] - throughputs[i] for i in range(len(throughputs)-1)]
            spike_threshold = max(1.0, np.std(diffs) * 2)
            
            spike_times = []
            for i, diff in enumerate(diffs):
                if diff > spike_threshold:
                    spike_times.append(times[i])
                    
            # Add vertical lines for spikes
            for spike_time in spike_times[:5]:  # Limit to first 5 spikes to avoid clutter
                ax.axvline(x=spike_time, color='orange', linestyle='-', alpha=0.5)
                ax.text(spike_time, max(throughputs) * 0.9, f"Spike", rotation=90,
                        ha='right', va='top', color='orange')
        
        ax.legend()
        
        # Save figure
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"spike_throughput_{timestamp}.png"), dpi=300)
        plt.close(fig)
    
    # 2. Latency Over Time
    if "latency" in time_series and time_series["latency"]:
        latency_data = time_series["latency"]
        times, latencies = zip(*latency_data)
        
        # Normalize times to start from 0
        start_time = times[0]
        times = [t - start_time for t in times]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(times, latencies, 'b-', linewidth=2)
        ax.set_title(f"Latency During Load Spike\n{config.get('initial_workers', 'N/A')} initial workers, {config.get('tests', 'N/A')} tests")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Latency (s)')
        ax.grid(True)
        
        # Add horizontal line for average latency
        avg_latency = metrics.get('avg_latency', 0)
        ax.axhline(y=avg_latency, color='r', linestyle='--', label=f'Avg: {avg_latency:.2f}s')
        
        # Add vertical lines for potential spike points (find sudden increases)
        if len(latencies) > 5:
            # Calculate moving average differences
            diffs = [latencies[i+1] - latencies[i] for i in range(len(latencies)-1)]
            spike_threshold = max(0.1, np.std(diffs) * 2)
            
            spike_times = []
            for i, diff in enumerate(diffs):
                if diff > spike_threshold:
                    spike_times.append(times[i])
                    
            # Add vertical lines for spikes
            for spike_time in spike_times[:5]:  # Limit to first 5 spikes
                ax.axvline(x=spike_time, color='orange', linestyle='-', alpha=0.5)
                ax.text(spike_time, max(latencies) * 0.9, f"Spike", rotation=90,
                        ha='right', va='top', color='orange')
        
        ax.legend()
        
        # Save figure
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"spike_latency_{timestamp}.png"), dpi=300)
        plt.close(fig)
    
    # 3. Combined Throughput and Latency
    if "throughput" in time_series and time_series["throughput"] and "latency" in time_series and time_series["latency"]:
        throughput_data = time_series["throughput"]
        latency_data = time_series["latency"]
        
        # Ensure we have matching timestamps
        throughput_dict = dict(throughput_data)
        latency_dict = dict(latency_data)
        
        # Use intersection of timestamps
        common_times = sorted(set(throughput_dict.keys()).intersection(set(latency_dict.keys())))
        
        if common_times:
            # Extract aligned data
            times = common_times
            throughputs = [throughput_dict[t] for t in times]
            latencies = [latency_dict[t] for t in times]
            
            # Normalize times to start from 0
            start_time = times[0]
            times = [t - start_time for t in times]
            
            # Normalize values for dual axis
            max_throughput = max(throughputs) if throughputs else 1
            max_latency = max(latencies) if latencies else 1
            
            # Create combined plot
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Throughput on primary axis
            color = 'tab:green'
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Throughput (tests/s)', color=color)
            ax1.plot(times, throughputs, color=color, linewidth=2)
            ax1.tick_params(axis='y', labelcolor=color)
            
            # Latency on secondary axis
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Latency (s)', color=color)
            ax2.plot(times, latencies, color=color, linewidth=2)
            ax2.tick_params(axis='y', labelcolor=color)
            
            # Add grid and title
            ax1.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.suptitle(f"Throughput vs Latency During Load Spike\n{config.get('initial_workers', 'N/A')} initial workers, {config.get('tests', 'N/A')} tests")
            
            # Add vertical lines for correlation points (where throughput increases and latency also changes)
            if len(times) > 5:
                throughput_changes = [throughputs[i+1] - throughputs[i] for i in range(len(throughputs)-1)]
                latency_changes = [latencies[i+1] - latencies[i] for i in range(len(latencies)-1)]
                
                # Find points where throughput increases significantly and latency changes
                correlation_points = []
                throughput_threshold = max(1.0, np.std(throughput_changes) * 1.5)
                
                for i, (t_change, l_change) in enumerate(zip(throughput_changes, latency_changes)):
                    if t_change > throughput_threshold and abs(l_change) > 0.1:
                        correlation_points.append((times[i], t_change, l_change))
                
                # Mark correlation points
                for i, (time_point, t_change, l_change) in enumerate(correlation_points[:5]):
                    ax1.axvline(x=time_point, color='red', linestyle='--', alpha=0.5)
                    direction = "↑" if l_change > 0 else "↓"
                    ax1.text(time_point, max_throughput * 0.9, f"Correlation {direction}", rotation=90,
                            ha='right', va='top', color='red')
            
            # Save figure
            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            fig.savefig(os.path.join(output_dir, f"spike_correlation_{timestamp}.png"), dpi=300)
            plt.close(fig)
    
    # 4. Create heat map over time (if we have enough data points)
    if ("throughput" in time_series and len(time_series["throughput"]) > 10 and 
            "latency" in time_series and len(time_series["latency"]) > 10):
        
        # Extract data
        throughput_data = dict(time_series["throughput"])
        latency_data = dict(time_series["latency"])
        
        # Find common timestamps
        common_times = sorted(set(throughput_data.keys()).intersection(set(latency_data.keys())))
        
        if len(common_times) > 10:
            # Normalize times
            start_time = common_times[0]
            norm_times = np.array([t - start_time for t in common_times])
            
            # Bin the time series data into time windows
            num_bins = min(20, len(common_times) // 5)  # Create reasonable number of bins
            time_bins = np.linspace(0, max(norm_times), num_bins)
            
            # Create empty matrices for heatmap
            throughput_matrix = np.zeros((num_bins-1,))
            latency_matrix = np.zeros((num_bins-1,))
            
            # Fill matrices with average values in each time bin
            for i in range(num_bins-1):
                bin_start = time_bins[i]
                bin_end = time_bins[i+1]
                
                # Find data points in this bin
                bin_mask = (norm_times >= bin_start) & (norm_times < bin_end)
                bin_times = [common_times[j] for j, is_in_bin in enumerate(bin_mask) if is_in_bin]
                
                # Calculate averages for this bin
                if bin_times:
                    bin_throughputs = [throughput_data[t] for t in bin_times]
                    bin_latencies = [latency_data[t] for t in bin_times]
                    
                    throughput_matrix[i] = np.mean(bin_throughputs)
                    latency_matrix[i] = np.mean(bin_latencies)
            
            # Create heat map
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Create a 2D matrix for the heatmap
            # Use throughput for x-axis and latency for y-axis values
            heatmap_data = np.zeros((2, num_bins-1))
            heatmap_data[0, :] = throughput_matrix
            heatmap_data[1, :] = latency_matrix
            
            # Create heatmap
            im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.set_label('Value')
            
            # Add labels
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Throughput', 'Latency'])
            
            # X-axis labels - show time bins
            bin_centers = [(time_bins[i] + time_bins[i+1])/2 for i in range(num_bins-1)]
            ax.set_xticks(np.arange(len(bin_centers)))
            ax.set_xticklabels([f"{x:.1f}s" for x in bin_centers], rotation=45)
            
            # Add title
            ax.set_title(f"Performance Heat Map Over Time\n{config.get('initial_workers', 'N/A')} initial workers, {config.get('tests', 'N/A')} tests")
            
            # Save figure
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"spike_heatmap_{timestamp}.png"), dpi=300)
            plt.close(fig)


def main():
    """Main function to parse arguments and generate visualizations."""
    parser = argparse.ArgumentParser(description="Load Balancer Performance Visualization Tool")
    
    parser.add_argument("input", type=str, help="Input JSON file with test results")
    parser.add_argument("--output-dir", "-o", type=str, default="./visualizations",
                       help="Output directory for visualization files")
    parser.add_argument("--type", "-t", type=str, choices=["stress", "benchmark", "spike"], 
                       help="Type of test results (auto-detected if not specified)")
    
    args = parser.parse_args()
    
    # Load results from JSON file
    try:
        results = load_results(args.input)
    except Exception as e:
        print(f"Error loading results file: {e}")
        sys.exit(1)
    
    # Auto-detect result type if not specified
    result_type = args.type
    if not result_type:
        if "benchmark_results" in results:
            result_type = "benchmark"
        elif "configuration" in results and "burst_mode" in results["configuration"]:
            result_type = "stress"
        elif "time_series" in results:
            result_type = "spike"
        else:
            result_type = "stress"  # Default to stress test
    
    # Generate visualizations based on result type
    print(f"Generating {result_type} visualizations from {args.input}...")
    
    try:
        if result_type == "stress":
            plot_single_test_results(results, args.output_dir)
        elif result_type == "benchmark":
            plot_benchmark_results(results, args.output_dir)
        elif result_type == "spike":
            plot_spike_simulation_results(results, args.output_dir)
        
        print(f"Visualizations saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Check if matplotlib and pandas are installed
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import pandas as pd
        import numpy as np
        import seaborn as sns
    except ImportError:
        print("Error: Required visualization libraries not found.")
        print("Please install required packages with:")
        print("pip install matplotlib pandas numpy seaborn")
        sys.exit(1)
        
    main()