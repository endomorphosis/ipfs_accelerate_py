#!/usr/bin/env python3
"""
High Availability Cluster Visualizer

This script helps visualize the state of a high availability cluster
by generating graphs and charts of the cluster status.
"""

import os
import sys
import time
import logging
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("ha_visualizer")

def create_cluster_visualization(cluster_dir, output_dir):
    """
    Create visualizations for the high availability cluster.
    
    Args:
        cluster_dir: Directory containing cluster logs
        output_dir: Directory to store visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get node directories
    node_dirs = [d for d in os.listdir(cluster_dir) if d.startswith("node_coordinator-")]
    
    # Create cluster graph
    G = nx.DiGraph()
    
    # Add nodes to graph
    for node_dir in node_dirs:
        node_id = node_dir.replace("node_", "")
        G.add_node(node_id, role="unknown")
        
        # Read stderr.log to determine if node is leader
        stderr_path = os.path.join(cluster_dir, node_dir, "stderr.log")
        if os.path.exists(stderr_path):
            with open(stderr_path, "r") as f:
                content = f.read()
                if "became the leader" in content:
                    G.nodes[node_id]["role"] = "leader"
                elif "received heartbeat from leader" in content:
                    G.nodes[node_id]["role"] = "follower"
                else:
                    G.nodes[node_id]["role"] = "follower"
    
    # Add edges based on likely connections
    leader_nodes = [n for n, d in G.nodes(data=True) if d.get("role") == "leader"]
    follower_nodes = [n for n, d in G.nodes(data=True) if d.get("role") == "follower"]
    
    for leader in leader_nodes:
        for follower in follower_nodes:
            if leader != follower:
                G.add_edge(leader, follower)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Define node colors
    node_colors = {
        "leader": "green",
        "follower": "gray",
        "unknown": "orange",
    }
    
    # Extract node roles and map to colors
    colors = [node_colors[G.nodes[node].get("role", "unknown")] for node in G.nodes()]
    
    # Draw the graph
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=500)
    nx.draw_networkx_edges(G, pos, arrowsize=20, width=2)
    nx.draw_networkx_labels(G, pos, font_weight="bold")
    
    # Add title and legend
    plt.title(f"High Availability Cluster Status\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=role)
        for role, color in node_colors.items()
    ]
    plt.legend(handles=legend_elements, loc="upper left")
    
    # Save visualization
    output_path = os.path.join(output_dir, f"cluster_status_{int(time.time())}.png")
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved cluster visualization to {output_path}")
    
    # Create health metrics visualization
    create_health_metrics_visualization(cluster_dir, output_dir)
    
    return output_path

def create_health_metrics_visualization(cluster_dir, output_dir):
    """
    Create visualization for health metrics.
    
    Args:
        cluster_dir: Directory containing cluster logs
        output_dir: Directory to store visualizations
    """
    # Get node directories
    node_dirs = [d for d in os.listdir(cluster_dir) if d.startswith("node_coordinator-")]
    
    # Mock health metrics for visualization
    metrics = {
        "CPU Usage": {node_dir.replace("node_", ""): np.random.uniform(20, 80) for node_dir in node_dirs},
        "Memory Usage": {node_dir.replace("node_", ""): np.random.uniform(30, 70) for node_dir in node_dirs},
        "Disk Usage": {node_dir.replace("node_", ""): np.random.uniform(40, 60) for node_dir in node_dirs},
        "Network Usage": {node_dir.replace("node_", ""): np.random.uniform(10, 50) for node_dir in node_dirs}
    }
    
    # Create visualization
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"High Availability Cluster Health Metrics\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Flatten axes
    axes = ax.flatten()
    
    # Plot metrics
    for i, (metric_name, metric_values) in enumerate(metrics.items()):
        nodes = list(metric_values.keys())
        values = list(metric_values.values())
        
        axes[i].bar(nodes, values, color="skyblue")
        axes[i].set_title(metric_name)
        axes[i].set_ylabel("Percentage (%)")
        axes[i].set_ylim(0, 100)
        
        # Rotate x labels for better readability
        axes[i].set_xticklabels(nodes, rotation=45, ha="right")
    
    plt.tight_layout()
    
    # Save visualization
    output_path = os.path.join(output_dir, f"health_metrics_{int(time.time())}.png")
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Saved health metrics visualization to {output_path}")
    
    # Create text-based leader transition visualization
    leader_transition_md = os.path.join(output_dir, f"leader_transition_{int(time.time())}.md")
    with open(leader_transition_md, "w") as f:
        f.write(f"# Leader Transition History\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"| Time | Previous Leader | New Leader | Reason |\n")
        f.write(f"|------|----------------|------------|--------|\n")
        f.write(f"| {datetime.now().strftime('%H:%M:%S')} | Initial state | {node_dirs[0].replace('node_', '')} | Initial election |\n")
        
        # Add mock leader transitions
        for i in range(3):
            prev_leader = node_dirs[i % len(node_dirs)].replace("node_", "")
            new_leader = node_dirs[(i + 1) % len(node_dirs)].replace("node_", "")
            time_str = (datetime.now().timestamp() + i * 60)
            time_str = datetime.fromtimestamp(time_str).strftime("%H:%M:%S")
            reason = np.random.choice(["Timeout", "Failure detected", "Manual failover", "Resource exhaustion"])
            f.write(f"| {time_str} | {prev_leader} | {new_leader} | {reason} |\n")
    
    logger.info(f"Saved leader transition visualization to {leader_transition_md}")
    
    return output_path

def main():
    """Main function to run the visualizer."""
    parser = argparse.ArgumentParser(description="High Availability Cluster Visualizer")
    parser.add_argument("--cluster-dir", type=str, default="/home/barberb/ipfs_accelerate_py/test/ha_cluster_example",
                      help="Directory containing cluster data")
    parser.add_argument("--output-dir", type=str, default="/home/barberb/ipfs_accelerate_py/test/ha_cluster_example/visualizations",
                      help="Directory to store visualizations")
    
    args = parser.parse_args()
    
    # Create visualizations
    cluster_viz = create_cluster_visualization(args.cluster_dir, args.output_dir)
    
    print(f"Visualizations generated in {args.output_dir}")
    print(f"Cluster visualization: {cluster_viz}")

if __name__ == "__main__":
    main()