#!/usr/bin/env python3
"""
Visualization script for Distributed Testing Framework test results.

This script generates visualizations from test execution results, including:
- Performance metrics over time
- Resource usage graphs
- Task distribution visualizations
- Success rate graphs
- Component interaction visualizations

Usage:
    python visualize_test_results.py --input-dir path/to/input --output-dir path/to/output
"""

import argparse
import json
import os
import glob
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dt_visualizer')

# Custom color schemes
COLORS = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#d35400', '#c0392b']
SUCCESS_COLOR = '#2ecc71'  # Green
FAILURE_COLOR = '#e74c3c'  # Red
WARNING_COLOR = '#f39c12'  # Orange
INFO_COLOR = '#3498db'     # Blue

# Visualization layout constants
FIG_WIDTH = 12
FIG_HEIGHT = 8
TITLE_FONTSIZE = 16
AXIS_FONTSIZE = 12
TICK_FONTSIZE = 10
DPI = 100
SMALL_FIG_SIZE = (10, 6)
LARGE_FIG_SIZE = (16, 9)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate visualizations from test results')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing test result artifacts')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save visualization outputs')
    parser.add_argument('--format', type=str, default='png',
                        help='Output image format (png, pdf, svg)')
    parser.add_argument('--style', type=str, default='seaborn-v0_8-whitegrid',
                        help='Matplotlib style to use')
    return parser.parse_args()

def find_test_result_files(input_dir):
    """Find all test result JSON files in the input directory."""
    logger.info(f"Searching for test result files in {input_dir}")
    
    # Recursive search for result files
    json_files = []
    
    # Check different directory structures based on test type
    for test_type in ['e2e', 'component', 'integration']:
        # Check for direct result files
        pattern = os.path.join(input_dir, f"test-results-{test_type}", "**", "*.json")
        files = glob.glob(pattern, recursive=True)
        logger.info(f"Found {len(files)} result files for {test_type} tests with pattern {pattern}")
        json_files.extend(files)
        
        # Check for nested result files
        pattern = os.path.join(input_dir, f"test-results-{test_type}", "**", "**", "*.json")
        files = glob.glob(pattern, recursive=True)
        logger.info(f"Found {len(files)} nested result files for {test_type} tests")
        json_files.extend(files)
    
    # Check for direct result files in the artifact directory
    pattern = os.path.join(input_dir, "**", "results.json")
    files = glob.glob(pattern, recursive=True)
    logger.info(f"Found {len(files)} direct result files with pattern {pattern}")
    json_files.extend(files)
    
    # Deduplicate
    json_files = list(set(json_files))
    logger.info(f"Found {len(json_files)} total unique result files")
    
    return json_files

def load_test_results(json_files):
    """Load test results from JSON files."""
    all_results = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Add source file info
                if isinstance(data, dict):
                    data['source_file'] = json_file
                    all_results.append(data)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            item['source_file'] = json_file
                    all_results.extend(data)
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    logger.info(f"Loaded {len(all_results)} result records")
    return all_results

def create_performance_summary(results, output_dir, fmt='png'):
    """Create summary performance visualization."""
    logger.info("Generating performance summary visualization")
    
    # Extract execution time data
    exec_times = []
    test_names = []
    test_statuses = []
    
    for result in results:
        if 'execution_time' in result and 'test_name' in result:
            exec_times.append(float(result['execution_time']))
            test_names.append(result['test_name'])
            test_statuses.append(result.get('status', 'unknown'))
    
    if not exec_times:
        logger.warning("No execution time data found for performance summary")
        return
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'test_name': test_names,
        'execution_time': exec_times,
        'status': test_statuses
    })
    
    # Group by test name and calculate mean execution time
    grouped = df.groupby('test_name').agg({
        'execution_time': ['mean', 'min', 'max', 'count'],
        'status': lambda x: (x == 'pass').mean()  # Success rate
    })
    
    grouped.columns = ['mean_time', 'min_time', 'max_time', 'count', 'success_rate']
    grouped = grouped.reset_index()
    
    # Sort by mean execution time
    grouped = grouped.sort_values('mean_time', ascending=False)
    
    # Take top 15 tests by execution time for readability
    plot_df = grouped.head(15)
    
    # Create visualization
    plt.figure(figsize=LARGE_FIG_SIZE, dpi=DPI)
    
    # Create a custom colormap based on success rate
    colors = [FAILURE_COLOR, WARNING_COLOR, SUCCESS_COLOR]
    n_bins = 100
    cmap_name = 'success_rate_cmap'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    
    # Create horizontal bar chart
    bars = plt.barh(plot_df['test_name'], plot_df['mean_time'], 
                   color=cm(plot_df['success_rate']))
    
    # Add error bars for min/max
    plt.errorbar(plot_df['mean_time'], plot_df['test_name'], 
                xerr=[(plot_df['mean_time'] - plot_df['min_time']), 
                      (plot_df['max_time'] - plot_df['mean_time'])],
                fmt='none', ecolor='black', capsize=5, alpha=0.5)
    
    # Add count and success rate as text
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        plt.text(row['mean_time'] + 0.05, i, 
                f"n={int(row['count'])}, {row['success_rate']*100:.0f}% pass", 
                va='center')
    
    # Formatting
    plt.title('Test Execution Performance Summary', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Mean Execution Time (seconds)', fontsize=AXIS_FONTSIZE)
    plt.ylabel('Test Name', fontsize=AXIS_FONTSIZE)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'performance_summary.{fmt}')
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved performance summary to {output_path}")
    
    return output_path

def create_component_interaction_graph(results, output_dir, fmt='png'):
    """Create visualization of component interactions."""
    logger.info("Generating component interaction visualization")
    
    # Count interactions between components
    interactions = {}
    
    for result in results:
        if 'component_interactions' in result:
            for interaction in result['component_interactions']:
                source = interaction.get('source', 'unknown')
                target = interaction.get('target', 'unknown')
                count = interaction.get('count', 1)
                success = interaction.get('success', True)
                
                key = (source, target)
                if key in interactions:
                    interactions[key]['count'] += count
                    interactions[key]['success_count'] += (count if success else 0)
                else:
                    interactions[key] = {
                        'count': count,
                        'success_count': (count if success else 0)
                    }
    
    if not interactions:
        logger.warning("No component interaction data found")
        return
    
    # Create network graph visualization
    try:
        import networkx as nx
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for (source, target), data in interactions.items():
            if source not in G:
                G.add_node(source)
            if target not in G:
                G.add_node(target)
            
            # Calculate success rate
            success_rate = data['success_count'] / data['count']
            
            # Set edge properties
            G.add_edge(source, target, 
                      weight=data['count'],
                      success_rate=success_rate,
                      width=np.sqrt(data['count']))
        
        # Prepare visualization
        plt.figure(figsize=LARGE_FIG_SIZE, dpi=DPI)
        
        # Use spring layout for node positioning
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color=INFO_COLOR, alpha=0.8)
        
        # Draw edges with colors based on success rate
        for u, v, data in G.edges(data=True):
            success_rate = data['success_rate']
            if success_rate >= 0.95:
                color = SUCCESS_COLOR
            elif success_rate >= 0.8:
                color = WARNING_COLOR
            else:
                color = FAILURE_COLOR
            
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=data['width'],
                                 edge_color=color, alpha=0.7, 
                                 arrowsize=10+data['width'])
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12)
        
        # Formatting
        plt.title('Component Interaction Graph', fontsize=TITLE_FONTSIZE)
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'component_interaction_graph.{fmt}')
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved component interaction graph to {output_path}")
        
    except ImportError:
        logger.warning("networkx library not available, skipping component interaction graph")
        return None
    
    return output_path

def create_resource_usage_visualization(results, output_dir, fmt='png'):
    """Create visualization of resource usage patterns."""
    logger.info("Generating resource usage visualization")
    
    # Extract resource metrics
    time_points = []
    cpu_usage = []
    memory_usage = []
    task_count = []
    worker_count = []
    
    for result in results:
        if 'resource_metrics' in result:
            metrics = result['resource_metrics']
            for point in metrics:
                # Check if we have timestamps and metrics
                if 'timestamp' in point and 'cpu_percent' in point and 'memory_percent' in point:
                    time_points.append(pd.to_datetime(point['timestamp']))
                    cpu_usage.append(float(point['cpu_percent']))
                    memory_usage.append(float(point['memory_percent']))
                    task_count.append(int(point.get('active_tasks', 0)))
                    worker_count.append(int(point.get('active_workers', 0)))
    
    if not time_points:
        logger.warning("No resource metrics found")
        return
    
    # Create DataFrame for time series
    df = pd.DataFrame({
        'timestamp': time_points,
        'cpu_percent': cpu_usage,
        'memory_percent': memory_usage,
        'active_tasks': task_count,
        'active_workers': worker_count
    })
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Create visualization
    fig = plt.figure(figsize=LARGE_FIG_SIZE, dpi=DPI)
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 1])
    
    # CPU and Memory subplot
    ax1 = plt.subplot(gs[0])
    ax1.plot(df['timestamp'], df['cpu_percent'], 'b-', label='CPU Usage (%)')
    ax1.plot(df['timestamp'], df['memory_percent'], 'r-', label='Memory Usage (%)')
    ax1.set_ylabel('Usage (%)', fontsize=AXIS_FONTSIZE)
    ax1.legend(loc='upper right')
    ax1.set_title('Resource Usage Over Time', fontsize=TITLE_FONTSIZE)
    ax1.grid(True, alpha=0.3)
    
    # Task count subplot
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(df['timestamp'], df['active_tasks'], 'g-', label='Active Tasks')
    ax2.set_ylabel('Task Count', fontsize=AXIS_FONTSIZE)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Worker count subplot
    ax3 = plt.subplot(gs[2], sharex=ax1)
    ax3.plot(df['timestamp'], df['active_workers'], 'm-', label='Active Workers')
    ax3.set_ylabel('Worker Count', fontsize=AXIS_FONTSIZE)
    ax3.set_xlabel('Time', fontsize=AXIS_FONTSIZE)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis date
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'resource_usage.{fmt}')
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved resource usage visualization to {output_path}")
    
    return output_path

def create_scaling_analysis(results, output_dir, fmt='png'):
    """Create visualization analyzing the scaling behavior."""
    logger.info("Generating scaling analysis visualization")
    
    # Extract scaling events
    scaling_events = []
    
    for result in results:
        if 'scaling_events' in result:
            for event in result['scaling_events']:
                scaling_events.append({
                    'timestamp': pd.to_datetime(event.get('timestamp', '1970-01-01')),
                    'workers_before': int(event.get('workers_before', 0)),
                    'workers_after': int(event.get('workers_after', 0)),
                    'reason': event.get('reason', 'unknown'),
                    'queue_depth': int(event.get('queue_depth', 0)),
                    'strategy': event.get('scaling_strategy', 'unknown')
                })
    
    if not scaling_events:
        logger.warning("No scaling events found")
        return
    
    # Create DataFrame
    df = pd.DataFrame(scaling_events)
    df = df.sort_values('timestamp')
    
    # Calculate change in worker count
    df['worker_delta'] = df['workers_after'] - df['workers_before']
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=LARGE_FIG_SIZE, dpi=DPI, sharex=True)
    
    # Plot worker changes over time
    colors = np.where(df['worker_delta'] > 0, SUCCESS_COLOR, 
                     np.where(df['worker_delta'] < 0, FAILURE_COLOR, WARNING_COLOR))
    
    ax1.bar(df['timestamp'], df['worker_delta'], color=colors, alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_ylabel('Worker Change', fontsize=AXIS_FONTSIZE)
    ax1.set_title('Dynamic Resource Scaling Analysis', fontsize=TITLE_FONTSIZE)
    
    # Annotate with reason for significant changes
    for idx, row in df.iterrows():
        if abs(row['worker_delta']) > 2:  # Only label significant changes
            ax1.annotate(row['reason'][:20] + '...' if len(row['reason']) > 20 else row['reason'],
                       (row['timestamp'], row['worker_delta']),
                       textcoords="offset points",
                       xytext=(0, 10 if row['worker_delta'] > 0 else -20),
                       ha='center',
                       fontsize=8,
                       arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))
    
    # Plot queue depth as a line
    ax2.plot(df['timestamp'], df['queue_depth'], 'm-', label='Queue Depth')
    ax2.set_ylabel('Queue Depth', fontsize=AXIS_FONTSIZE)
    ax2.set_xlabel('Time', fontsize=AXIS_FONTSIZE)
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis date
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'scaling_analysis.{fmt}')
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved scaling analysis visualization to {output_path}")
    
    # Create a second figure for scaling strategy distribution
    if 'strategy' in df.columns:
        plt.figure(figsize=SMALL_FIG_SIZE, dpi=DPI)
        strategy_counts = df['strategy'].value_counts()
        
        colors = [INFO_COLOR, SUCCESS_COLOR, WARNING_COLOR, FAILURE_COLOR]
        plt.pie(strategy_counts, labels=strategy_counts.index, autopct='%1.1f%%',
               startangle=90, colors=colors[:len(strategy_counts)])
        plt.axis('equal')
        plt.title('Scaling Strategy Distribution', fontsize=TITLE_FONTSIZE)
        
        # Save figure
        output_path2 = os.path.join(output_dir, f'scaling_strategies.{fmt}')
        plt.savefig(output_path2)
        plt.close()
        logger.info(f"Saved scaling strategies visualization to {output_path2}")
    
    return output_path

def create_test_results_summary(results, output_dir, fmt='png'):
    """Create visualization summarizing test results by type."""
    logger.info("Generating test results summary visualization")
    
    # Extract test results
    test_types = []
    test_results = []
    execution_times = []
    
    for result in results:
        if 'test_type' in result and 'status' in result:
            test_types.append(result['test_type'])
            test_results.append(result['status'])
            if 'execution_time' in result:
                execution_times.append(float(result['execution_time']))
            else:
                execution_times.append(0.0)
    
    if not test_types:
        logger.warning("No test results found")
        return
    
    # Create DataFrame
    df = pd.DataFrame({
        'test_type': test_types,
        'status': test_results,
        'execution_time': execution_times
    })
    
    # Convert status to pass/fail
    df['passed'] = df['status'].apply(lambda x: x == 'pass')
    
    # Group by test type
    result_summary = df.groupby('test_type').agg({
        'passed': ['mean', 'count'],
        'execution_time': ['mean', 'sum']
    })
    
    result_summary.columns = ['pass_rate', 'count', 'avg_time', 'total_time']
    result_summary = result_summary.reset_index()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=LARGE_FIG_SIZE, dpi=DPI)
    
    # Pass rate by test type
    ax1.bar(result_summary['test_type'], result_summary['pass_rate'] * 100,
           color=[SUCCESS_COLOR if rate > 0.8 else WARNING_COLOR if rate > 0.5 else FAILURE_COLOR 
                  for rate in result_summary['pass_rate']])
    
    # Add count as text
    for i, (idx, row) in enumerate(result_summary.iterrows()):
        ax1.text(i, row['pass_rate'] * 50, f"n={int(row['count'])}", 
                ha='center', va='center')
    
    ax1.set_ylabel('Pass Rate (%)', fontsize=AXIS_FONTSIZE)
    ax1.set_title('Test Pass Rate by Type', fontsize=TITLE_FONTSIZE)
    ax1.set_ylim(0, 100)
    ax1.grid(True, alpha=0.3)
    
    # Execution time by test type
    ax2.bar(result_summary['test_type'], result_summary['avg_time'],
           color=INFO_COLOR, alpha=0.7)
    
    # Add total time as text
    for i, (idx, row) in enumerate(result_summary.iterrows()):
        ax2.text(i, row['avg_time'] / 2, f"total: {row['total_time']:.1f}s", 
                ha='center', va='center')
    
    ax2.set_ylabel('Average Execution Time (s)', fontsize=AXIS_FONTSIZE)
    ax2.set_title('Test Execution Time by Type', fontsize=TITLE_FONTSIZE)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f'test_results_summary.{fmt}')
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved test results summary to {output_path}")
    
    return output_path

def create_dashboard_html(visualizations, output_dir):
    """Create an HTML dashboard with all visualizations."""
    logger.info("Generating HTML dashboard")
    
    if not visualizations:
        logger.warning("No visualizations to include in dashboard")
        return
    
    # Generate HTML content
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Distributed Testing Framework - Test Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background-color: #333; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .header h1 {{ margin: 0; }}
        .timestamp {{ font-size: 0.8em; margin-top: 10px; }}
        .visualization {{ background-color: white; border-radius: 5px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .visualization h2 {{ margin-top: 0; color: #333; }}
        img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Distributed Testing Framework - Test Results</h1>
            <div class="timestamp">Report generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
"""
    
    # Add each visualization
    for vis_type, path in visualizations.items():
        if path and os.path.exists(path):
            # Get relative path
            rel_path = os.path.basename(path)
            title = vis_type.replace('_', ' ').title()
            
            html_content += f"""
        <div class="visualization">
            <h2>{title}</h2>
            <img src="{rel_path}" alt="{title}">
        </div>
"""
    
    # Close HTML
    html_content += """
    </div>
</body>
</html>
"""
    
    # Write HTML file
    output_path = os.path.join(output_dir, 'dashboard.html')
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Saved HTML dashboard to {output_path}")
    return output_path

def main():
    """Main execution function."""
    args = parse_args()
    
    # Check that input directory exists
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory {args.input_dir} does not exist")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set matplotlib style
    try:
        plt.style.use(args.style)
    except Exception as e:
        logger.warning(f"Could not set matplotlib style {args.style}: {e}")
    
    # Find test result files
    json_files = find_test_result_files(args.input_dir)
    
    if not json_files:
        logger.error("No result files found")
        return 1
    
    # Load results data
    results = load_test_results(json_files)
    
    if not results:
        logger.error("No results could be loaded from files")
        return 1
    
    # Create visualizations
    visualizations = {}
    
    visualizations['performance_summary'] = create_performance_summary(results, args.output_dir, args.format)
    visualizations['component_interaction'] = create_component_interaction_graph(results, args.output_dir, args.format)
    visualizations['resource_usage'] = create_resource_usage_visualization(results, args.output_dir, args.format)
    visualizations['scaling_analysis'] = create_scaling_analysis(results, args.output_dir, args.format)
    visualizations['test_results'] = create_test_results_summary(results, args.output_dir, args.format)
    
    # Create HTML dashboard
    dashboard_path = create_dashboard_html(visualizations, args.output_dir)
    
    logger.info("Visualization generation complete")
    return 0

if __name__ == "__main__":
    exit(main())