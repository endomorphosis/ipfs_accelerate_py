#!/usr/bin/env python3
"""
High Availability Cluster Example

This example demonstrates how to set up a high availability coordinator cluster
using the AutoRecoverySystem for advanced fault tolerance.

The example includes:
1. Creating multiple coordinator instances in separate processes
2. Automatic leader election
3. Fault injection to test failover
4. Visualization of cluster state
5. WebNN/WebGPU capability detection and integration

Usage:
    python high_availability_cluster.py --nodes 3 --fault-injection

This will create a cluster of 3 coordinator nodes and periodically inject faults to test
the automatic failover capabilities.
"""

import os
import sys
import time
import uuid
import argparse
import threading
import random
import signal
import subprocess
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import modules from parent
parent_dir = str(Path(__file__).parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from duckdb_api.distributed_testing.auto_recovery import (
    AutoRecoverySystem, 
    COORDINATOR_STATUS_LEADER,
    COORDINATOR_STATUS_FOLLOWER
)

class ClusterNode:
    """Represents a single node in the high availability cluster."""
    
    def __init__(self, node_id, port, coordinator_addresses, db_path, visualization_path):
        """Initialize a cluster node.
        
        Args:
            node_id: Unique identifier for this node
            port: Port to listen on
            coordinator_addresses: List of other coordinator addresses
            db_path: Path to database
            visualization_path: Path to store visualizations
        """
        self.node_id = node_id
        self.port = port
        self.coordinator_addresses = coordinator_addresses
        self.db_path = db_path
        self.visualization_path = visualization_path
        
        # Create directories
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(visualization_path, exist_ok=True)
        
        # Create auto recovery system
        self.auto_recovery_system = AutoRecoverySystem(
            coordinator_id=node_id,
            coordinator_addresses=coordinator_addresses,
            db_path=db_path,
            auto_leader_election=True,
            visualization_path=visualization_path
        )
        
        # Running flag
        self.running = False
        self.is_leader = False
        
        # Status update thread
        self.status_thread = None
        self.status_stop_event = threading.Event()
        
    def start(self):
        """Start the cluster node."""
        if self.running:
            print(f"Node {self.node_id} is already running")
            return
        
        # Start auto recovery system
        self.auto_recovery_system.start()
        
        # Start status update thread
        self.status_thread = threading.Thread(target=self._status_update_loop)
        self.status_thread.daemon = True
        self.status_thread.start()
        
        self.running = True
        print(f"Node {self.node_id} started")
        
    def stop(self):
        """Stop the cluster node."""
        if not self.running:
            print(f"Node {self.node_id} is not running")
            return
        
        # Stop status update thread
        self.status_stop_event.set()
        if self.status_thread:
            self.status_thread.join(timeout=2.0)
        
        # Stop auto recovery system
        self.auto_recovery_system.stop()
        
        self.running = False
        print(f"Node {self.node_id} stopped")
        
    def _status_update_loop(self):
        """Status update thread function."""
        while not self.status_stop_event.is_set():
            # Update status
            status = self.auto_recovery_system.get_status()
            
            # Check if we're the leader
            is_leader = self.auto_recovery_system.is_leader()
            if is_leader != self.is_leader:
                self.is_leader = is_leader
                status_str = "LEADER" if is_leader else "FOLLOWER"
                print(f"Node {self.node_id} is now {status_str}")
            
            # Wait for next update
            self.status_stop_event.wait(1.0)

def start_node_process(node_id, port, other_ports, base_dir):
    """Start a node in a separate process.
    
    Args:
        node_id: Node ID
        port: Port for this node
        other_ports: Ports of other nodes
        base_dir: Base directory for node data
        
    Returns:
        Subprocess object
    """
    # Create coordinator addresses for other nodes
    coordinator_addresses = [f"localhost:{p}" for p in other_ports]
    
    # Create node directory
    node_dir = os.path.join(base_dir, f"node_{node_id}")
    os.makedirs(node_dir, exist_ok=True)
    
    # Create database and visualization paths
    db_path = os.path.join(node_dir, "node.duckdb")
    visualization_path = os.path.join(node_dir, "visualizations")
    
    # Create command
    cmd = [
        sys.executable,
        os.path.join(parent_dir, "duckdb_api/distributed_testing/run_integrated_system.py"),
        "--host", "localhost",
        "--port", str(port),
        "--dashboard-port", str(port + 1000),
        "--db-path", db_path,
        "--visualization-path", visualization_path,
        "--high-availability",
        "--coordinator-id", node_id,
        "--coordinator-addresses", ",".join(coordinator_addresses),
        "--auto-leader-election",
        "--mock-workers", "2",  # Start 2 mock workers
        "--enhanced-hardware-taxonomy"  # Enable WebNN/WebGPU support
    ]
    
    # Redirect output to files
    stdout = open(os.path.join(node_dir, "stdout.log"), "w")
    stderr = open(os.path.join(node_dir, "stderr.log"), "w")
    
    # Start process
    process = subprocess.Popen(
        cmd,
        stdout=stdout,
        stderr=stderr,
        preexec_fn=os.setsid  # Create a new process group
    )
    
    return process

def run_cluster_example(num_nodes=3, base_port=8080, fault_injection=False, runtime=60):
    """Run the high availability cluster example.
    
    Args:
        num_nodes: Number of nodes to start
        base_port: Base port number (each node uses base_port + node_index)
        fault_injection: Whether to inject faults
        runtime: Runtime in seconds
    """
    print(f"Starting high availability cluster with {num_nodes} nodes")
    
    # Create base directory
    base_dir = os.path.join(os.getcwd(), "ha_cluster_example")
    os.makedirs(base_dir, exist_ok=True)
    
    # Create node processes
    node_processes = []
    node_ids = [f"coordinator-{uuid.uuid4().hex[:8]}" for _ in range(num_nodes)]
    node_ports = [base_port + i for i in range(num_nodes)]
    
    # Start processes
    for i in range(num_nodes):
        other_ports = [p for j, p in enumerate(node_ports) if j != i]
        
        process = start_node_process(
            node_id=node_ids[i],
            port=node_ports[i],
            other_ports=other_ports,
            base_dir=base_dir
        )
        
        node_processes.append({
            "index": i,
            "node_id": node_ids[i],
            "port": node_ports[i],
            "process": process
        })
        
        print(f"Started node {i} (ID: {node_ids[i]}) on port {node_ports[i]}")
        
        # Wait a bit between starting nodes
        time.sleep(2)
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("Shutting down cluster...")
        for node in node_processes:
            try:
                # Kill process group
                os.killpg(os.getpgid(node["process"].pid), signal.SIGTERM)
                print(f"Stopped node {node['index']} (ID: {node['node_id']})")
            except Exception as e:
                print(f"Error stopping node {node['index']}: {e}")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run cluster for specified time
    start_time = time.time()
    fault_inject_interval = 15  # seconds
    last_fault_time = start_time
    
    try:
        while time.time() - start_time < runtime:
            # Check if any nodes have crashed
            for node in node_processes:
                if node["process"].poll() is not None:
                    print(f"Node {node['index']} (ID: {node['node_id']}) has crashed, restarting...")
                    
                    # Restart node
                    other_ports = [p for j, p in enumerate(node_ports) if j != node["index"]]
                    
                    process = start_node_process(
                        node_id=node["node_id"],
                        port=node["port"],
                        other_ports=other_ports,
                        base_dir=base_dir
                    )
                    
                    node["process"] = process
                    print(f"Restarted node {node['index']} (ID: {node['node_id']})")
            
            # Inject faults if enabled
            if fault_injection and time.time() - last_fault_time > fault_inject_interval:
                # Choose a random node to kill
                node_index = random.randint(0, num_nodes - 1)
                node = node_processes[node_index]
                
                print(f"Injecting fault: Killing node {node['index']} (ID: {node['node_id']})")
                
                try:
                    # Kill the process group
                    os.killpg(os.getpgid(node["process"].pid), signal.SIGTERM)
                    
                    # Wait a bit to let the process terminate
                    time.sleep(1)
                    
                    # Restart node
                    other_ports = [p for j, p in enumerate(node_ports) if j != node["index"]]
                    
                    process = start_node_process(
                        node_id=node["node_id"],
                        port=node["port"],
                        other_ports=other_ports,
                        base_dir=base_dir
                    )
                    
                    node["process"] = process
                    print(f"Restarted node {node['index']} (ID: {node['node_id']})")
                    
                except Exception as e:
                    print(f"Error injecting fault: {e}")
                
                last_fault_time = time.time()
            
            # Wait a bit
            time.sleep(1)
            
            # Calculate remaining time
            elapsed = time.time() - start_time
            remaining = runtime - elapsed
            
            if int(elapsed) % 10 == 0:  # Print status every 10 seconds
                print(f"Cluster running for {int(elapsed)}s, {int(remaining)}s remaining")
                
    finally:
        # Shutdown all nodes
        print("Shutting down cluster...")
        for node in node_processes:
            try:
                # Kill process group
                os.killpg(os.getpgid(node["process"].pid), signal.SIGTERM)
                print(f"Stopped node {node['index']} (ID: {node['node_id']})")
            except:
                pass
    
    print(f"Cluster shutdown complete. Results stored in {base_dir}")
    print(f"Visualizations are available in {base_dir}/node_*/visualizations")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="High Availability Cluster Example")
    parser.add_argument("--nodes", type=int, default=3, help="Number of nodes to start")
    parser.add_argument("--base-port", type=int, default=8080, help="Base port number")
    parser.add_argument("--fault-injection", action="store_true", help="Enable fault injection")
    parser.add_argument("--runtime", type=int, default=60, help="Runtime in seconds")
    
    args = parser.parse_args()
    
    run_cluster_example(
        num_nodes=args.nodes,
        base_port=args.base_port,
        fault_injection=args.fault_injection,
        runtime=args.runtime
    )

if __name__ == "__main__":
    main()