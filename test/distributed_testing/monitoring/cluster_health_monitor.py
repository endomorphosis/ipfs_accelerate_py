#!/usr/bin/env python3
"""
Cluster Health Monitor for Distributed Testing Framework.
Provides real-time visualization of coordinator cluster health and performance metrics.
"""

from ipfs_accelerate_py.anyio_helpers import gather, wait_for
import anyio
import os
import sys
import time
import json
import argparse
import logging
import datetime
import threading
import queue
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from matplotlib.figure import Figure
import numpy as np
from collections import deque

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CoordinatorStatus:
    """Stores the status of a coordinator node."""
    
    def __init__(self, node_id, host, port):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.role = "UNKNOWN"
        self.term = 0
        self.current_leader = None
        self.log_size = 0
        self.last_applied = 0
        self.commit_index = 0
        self.uptime = 0
        self.worker_count = 0
        self.task_count = 0
        self.latency_history = deque(maxlen=100)
        self.cpu_usage = 0
        self.memory_usage = 0
        self.last_update = time.time()
        self.status = "unknown"
        
    def update(self, status_data):
        """Update status from API response data."""
        self.role = status_data.get("role", "UNKNOWN")
        self.term = status_data.get("term", self.term)
        self.current_leader = status_data.get("current_leader", self.current_leader)
        self.log_size = status_data.get("log_size", self.log_size)
        self.last_applied = status_data.get("last_applied", self.last_applied)
        self.commit_index = status_data.get("commit_index", self.commit_index)
        self.uptime = status_data.get("uptime", self.uptime)
        self.worker_count = status_data.get("worker_count", self.worker_count)
        self.task_count = status_data.get("task_count", self.task_count)
        self.cpu_usage = status_data.get("cpu_usage", self.cpu_usage)
        self.memory_usage = status_data.get("memory_usage", self.memory_usage)
        self.status = "active"
        self.last_update = time.time()
        
        # Add latency measurement
        latency = status_data.get("response_time", 0)
        if latency > 0:
            self.latency_history.append(latency)
            
    def is_stale(self, max_age=10):
        """Check if status data is stale."""
        return time.time() - self.last_update > max_age
    
    def get_role_color(self):
        """Get color for the node's role."""
        if self.role == "LEADER":
            return "#4CAF50"  # Green
        elif self.role == "FOLLOWER":
            return "#2196F3"  # Blue
        elif self.role == "CANDIDATE":
            return "#FFC107"  # Yellow
        else:
            return "#9E9E9E"  # Gray
            
    def get_status_color(self):
        """Get color for the node's status."""
        if self.is_stale():
            return "#F44336"  # Red
        else:
            return "#4CAF50"  # Green


class ClusterHealthMonitor:
    """Monitor and visualize coordinator cluster health."""
    
    def __init__(self, nodes, update_interval=2):
        """Initialize the monitor with a list of coordinator nodes."""
        self.nodes = []
        for node_info in nodes:
            if isinstance(node_info, str):
                # Parse host:port format
                parts = node_info.split(":")
                host = parts[0]
                port = int(parts[1]) if len(parts) > 1 else 8080
                node_id = f"node-{len(self.nodes) + 1}"
                self.nodes.append(CoordinatorStatus(node_id, host, port))
            elif isinstance(node_info, dict):
                # Use provided dict with node_id, host, port
                node_id = node_info.get("node_id", f"node-{len(self.nodes) + 1}")
                host = node_info.get("host", "localhost")
                port = node_info.get("port", 8080)
                self.nodes.append(CoordinatorStatus(node_id, host, port))
                
        self.update_interval = update_interval
        self.stop_event = anyio.Event()
        self.status_history = {}
        
        # Initialize status history for each node
        for node in self.nodes:
            self.status_history[node.node_id] = {
                "time": deque(maxlen=100),
                "cpu": deque(maxlen=100),
                "memory": deque(maxlen=100),
                "workers": deque(maxlen=100),
                "tasks": deque(maxlen=100),
                "latency": deque(maxlen=100)
            }
            
        # Create queue for thread communication
        self.update_queue = queue.Queue()
        
    async def start_monitoring(self):
        """Start monitoring the cluster."""
        logger.info("Starting cluster health monitoring")
        
        # Start the GUI in a separate thread
        threading.Thread(target=self._run_gui, daemon=True).start()
        
        try:
            while not self.stop_event.is_set():
                # Update status for all nodes
                await self._update_all_nodes()
                
                # Wait for the next update interval
                try:
                    await wait_for(self.stop_event.wait(), self.update_interval)
                except TimeoutError:
                    # This is expected - just continue
                    pass
                    
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error during monitoring: {e}")
        finally:
            logger.info("Monitoring stopped")
            
    async def stop_monitoring(self):
        """Stop monitoring the cluster."""
        logger.info("Stopping cluster health monitoring")
        self.stop_event.set()
        
    async def _update_all_nodes(self):
        """Update status for all nodes."""
        current_time = time.time()
        
        # Create tasks for all nodes
        tasks = []
        for node in self.nodes:
            tasks.append(self._update_node_status(node, current_time))
            
        # Wait for all tasks to complete
        await gather(*tasks)
        
        # Update the GUI
        self.update_queue.put(self.nodes)
        
    async def _update_node_status(self, node, timestamp):
        """Update status for a specific node."""
        try:
            # Get status from the node's API
            url = f"http://{node.host}:{node.port}/api/status"
            start_time = time.time()
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=2) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Add response time
                        data["response_time"] = (time.time() - start_time) * 1000  # to ms
                        node.update(data)
                        
                        # Update history
                        history = self.status_history[node.node_id]
                        history["time"].append(timestamp)
                        history["cpu"].append(node.cpu_usage)
                        history["memory"].append(node.memory_usage)
                        history["workers"].append(node.worker_count)
                        history["tasks"].append(node.task_count)
                        history["latency"].append(data["response_time"])
                    else:
                        logger.warning(f"Failed to get status from {node.node_id}: HTTP {response.status}")
        except Exception as e:
            logger.warning(f"Error updating status for {node.node_id}: {e}")
            
    def _run_gui(self):
        """Run the GUI in a separate thread."""
        self.root = tk.Tk()
        self.root.title("Coordinator Cluster Health Monitor")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Create tab control
        self.tab_control = ttk.Notebook(self.root)
        
        # Create dashboard tab
        self.dashboard_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.dashboard_tab, text="Dashboard")
        self._setup_dashboard_tab()
        
        # Create metrics tab
        self.metrics_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.metrics_tab, text="Metrics")
        self._setup_metrics_tab()
        
        # Create log tab
        self.log_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.log_tab, text="Log")
        self._setup_log_tab()
        
        # Pack tab control
        self.tab_control.pack(expand=1, fill="both")
        
        # Start update loop
        self.root.after(100, self._process_updates)
        
        # Start tkinter main loop
        self.root.mainloop()
        
    def _setup_dashboard_tab(self):
        """Set up the dashboard tab."""
        # Create top frame for node status
        self.nodes_frame = ttk.LabelFrame(self.dashboard_tab, text="Coordinator Nodes")
        self.nodes_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create a canvas for the node status indicators
        self.nodes_canvas = tk.Canvas(self.nodes_frame, bg="white")
        self.nodes_canvas.pack(fill="both", expand=True)
        
        # Create a frame for the cluster summary
        self.summary_frame = ttk.LabelFrame(self.dashboard_tab, text="Cluster Summary")
        self.summary_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create labels for cluster summary
        self.leader_label = ttk.Label(self.summary_frame, text="Leader: Unknown")
        self.leader_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        self.term_label = ttk.Label(self.summary_frame, text="Current Term: 0")
        self.term_label.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        self.workers_label = ttk.Label(self.summary_frame, text="Workers: 0")
        self.workers_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        
        self.tasks_label = ttk.Label(self.summary_frame, text="Tasks: 0")
        self.tasks_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        
    def _setup_metrics_tab(self):
        """Set up the metrics tab."""
        # Create figure for plots
        self.metrics_fig = Figure(figsize=(12, 8), dpi=100)
        
        # Create 2x2 grid of subplots
        self.cpu_ax = self.metrics_fig.add_subplot(221)
        self.memory_ax = self.metrics_fig.add_subplot(222)
        self.latency_ax = self.metrics_fig.add_subplot(223)
        self.counts_ax = self.metrics_fig.add_subplot(224)
        
        # Set titles
        self.cpu_ax.set_title("CPU Usage")
        self.memory_ax.set_title("Memory Usage")
        self.latency_ax.set_title("Response Latency")
        self.counts_ax.set_title("Workers & Tasks")
        
        # Add grid
        self.cpu_ax.grid(True)
        self.memory_ax.grid(True)
        self.latency_ax.grid(True)
        self.counts_ax.grid(True)
        
        # Create canvas
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, self.metrics_tab)
        self.metrics_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Adjust layout
        self.metrics_fig.tight_layout()
        
    def _setup_log_tab(self):
        """Set up the log tab."""
        # Create a Text widget for the log
        self.log_text = tk.Text(self.log_tab, wrap=tk.WORD)
        self.log_text.pack(fill="both", expand=True)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)
        
        # Make the text widget read-only
        self.log_text.config(state=tk.DISABLED)
        
        # Add initial log entry
        self._add_log_entry("Cluster health monitoring started")
        
    def _add_log_entry(self, entry):
        """Add an entry to the log."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {entry}\n"
        
        # Enable editing temporarily
        self.log_text.config(state=tk.NORMAL)
        
        # Add the entry
        self.log_text.insert(tk.END, log_entry)
        
        # Scroll to the end
        self.log_text.see(tk.END)
        
        # Disable editing again
        self.log_text.config(state=tk.DISABLED)
        
    def _process_updates(self):
        """Process updates from the monitoring thread."""
        try:
            # Check if there are any updates
            while not self.update_queue.empty():
                nodes = self.update_queue.get_nowait()
                self._update_dashboard(nodes)
                self._update_metrics()
                
        except Exception as e:
            logger.error(f"Error processing updates: {e}")
            
        # Schedule the next update
        self.root.after(500, self._process_updates)
        
    def _update_dashboard(self, nodes):
        """Update the dashboard with node status."""
        # Clear the canvas
        self.nodes_canvas.delete("all")
        
        # Draw node status indicators
        node_width = 200
        node_height = 120
        padding = 20
        max_per_row = max(1, self.nodes_canvas.winfo_width() // (node_width + padding))
        
        # Find the leader node
        leader_node = None
        for node in nodes:
            if node.role == "LEADER":
                leader_node = node
                break
                
        # Update cluster summary
        if leader_node:
            self.leader_label.config(text=f"Leader: {leader_node.node_id}")
            self.term_label.config(text=f"Current Term: {leader_node.term}")
            
            # Sum up workers and tasks
            total_workers = sum(node.worker_count for node in nodes)
            total_tasks = sum(node.task_count for node in nodes)
            
            self.workers_label.config(text=f"Workers: {total_workers}")
            self.tasks_label.config(text=f"Tasks: {total_tasks}")
            
            # Log leader changes
            if not hasattr(self, 'last_leader') or self.last_leader != leader_node.node_id:
                self._add_log_entry(f"Leader changed to {leader_node.node_id} (Term: {leader_node.term})")
                self.last_leader = leader_node.node_id
                
            # Log term changes
            if not hasattr(self, 'last_term') or self.last_term != leader_node.term:
                if hasattr(self, 'last_term'):
                    self._add_log_entry(f"Term advanced from {self.last_term} to {leader_node.term}")
                self.last_term = leader_node.term
        else:
            self.leader_label.config(text="Leader: Unknown")
            self.term_label.config(text="Current Term: Unknown")
            self.workers_label.config(text="Workers: Unknown")
            self.tasks_label.config(text="Tasks: Unknown")
            
        # Draw node status boxes
        for i, node in enumerate(nodes):
            row = i // max_per_row
            col = i % max_per_row
            
            x = padding + col * (node_width + padding)
            y = padding + row * (node_height + padding)
            
            # Draw node box with role-based border color
            self.nodes_canvas.create_rectangle(
                x, y, x + node_width, y + node_height,
                outline=node.get_role_color(),
                width=3,
                fill="white"
            )
            
            # Draw status indicator
            status_color = node.get_status_color()
            self.nodes_canvas.create_oval(
                x + node_width - 20, y + 10,
                x + node_width - 10, y + 20,
                fill=status_color,
                outline=status_color
            )
            
            # Draw node ID
            self.nodes_canvas.create_text(
                x + node_width // 2, y + 15,
                text=node.node_id,
                font=("Arial", 12, "bold")
            )
            
            # Draw role
            self.nodes_canvas.create_text(
                x + node_width // 2, y + 35,
                text=f"Role: {node.role}",
                font=("Arial", 10)
            )
            
            # Draw term
            self.nodes_canvas.create_text(
                x + node_width // 2, y + 55,
                text=f"Term: {node.term}",
                font=("Arial", 10)
            )
            
            # Draw worker count
            self.nodes_canvas.create_text(
                x + node_width // 2, y + 75,
                text=f"Workers: {node.worker_count}",
                font=("Arial", 10)
            )
            
            # Draw task count
            self.nodes_canvas.create_text(
                x + node_width // 2, y + 95,
                text=f"Tasks: {node.task_count}",
                font=("Arial", 10)
            )
            
            # Log node state changes
            if node.is_stale():
                if not hasattr(node, 'logged_stale') or not node.logged_stale:
                    self._add_log_entry(f"Node {node.node_id} appears to be offline or unresponsive")
                    node.logged_stale = True
            else:
                if hasattr(node, 'logged_stale') and node.logged_stale:
                    self._add_log_entry(f"Node {node.node_id} is back online")
                    node.logged_stale = False
                    
    def _update_metrics(self):
        """Update the metrics plots."""
        # Clear the plots
        self.cpu_ax.clear()
        self.memory_ax.clear()
        self.latency_ax.clear()
        self.counts_ax.clear()
        
        # Set titles
        self.cpu_ax.set_title("CPU Usage (%)")
        self.memory_ax.set_title("Memory Usage (MB)")
        self.latency_ax.set_title("Response Latency (ms)")
        self.counts_ax.set_title("Workers & Tasks")
        
        # Add grid
        self.cpu_ax.grid(True)
        self.memory_ax.grid(True)
        self.latency_ax.grid(True)
        self.counts_ax.grid(True)
        
        # Generate plots for each node
        colors = plt.cm.tab10(np.arange(len(self.nodes)))
        
        for i, node in enumerate(self.nodes):
            history = self.status_history[node.node_id]
            
            if len(history["time"]) > 0:
                # Normalize times to relative seconds
                if len(history["time"]) > 0:
                    times = np.array(history["time"])
                    relative_times = times - times[0]
                    
                    # Plot CPU usage
                    self.cpu_ax.plot(relative_times, history["cpu"], 
                                   label=node.node_id, color=colors[i])
                    
                    # Plot memory usage
                    self.memory_ax.plot(relative_times, history["memory"], 
                                      label=node.node_id, color=colors[i])
                    
                    # Plot latency
                    self.latency_ax.plot(relative_times, history["latency"], 
                                       label=node.node_id, color=colors[i])
                    
                    # Plot workers and tasks
                    self.counts_ax.plot(relative_times, history["workers"], 
                                      label=f"{node.node_id} - Workers", 
                                      color=colors[i], linestyle='-')
                    self.counts_ax.plot(relative_times, history["tasks"], 
                                      label=f"{node.node_id} - Tasks", 
                                      color=colors[i], linestyle='--')
                    
        # Add legends
        self.cpu_ax.legend(loc='upper left', fontsize='small')
        self.memory_ax.legend(loc='upper left', fontsize='small')
        self.latency_ax.legend(loc='upper left', fontsize='small')
        self.counts_ax.legend(loc='upper left', fontsize='small')
        
        # Set y-axis limits
        self.cpu_ax.set_ylim(bottom=0)
        self.memory_ax.set_ylim(bottom=0)
        self.latency_ax.set_ylim(bottom=0)
        self.counts_ax.set_ylim(bottom=0)
        
        # Adjust layout
        self.metrics_fig.tight_layout()
        
        # Redraw the canvas
        self.metrics_canvas.draw()
        
    def _on_close(self):
        """Handle window close event."""
        logger.info("GUI closed - stopping monitoring")
        # TODO: Stop monitoring using an AnyIO task group
        self.root.destroy()
        

async def main():
    """Main function to run the cluster health monitor."""
    parser = argparse.ArgumentParser(description="Monitor coordinator cluster health")
    parser.add_argument("--nodes", type=str, nargs="+", required=True,
                      help="List of coordinator nodes in format host:port")
    parser.add_argument("--interval", type=float, default=2.0,
                      help="Update interval in seconds (default: 2.0)")
    
    args = parser.parse_args()
    
    monitor = ClusterHealthMonitor(args.nodes, update_interval=args.interval)
    await monitor.start_monitoring()
    

if __name__ == "__main__":
    anyio.run(main)