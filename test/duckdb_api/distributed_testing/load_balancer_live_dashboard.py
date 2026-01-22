#!/usr/bin/env python3
"""
Live Monitoring Dashboard for the Adaptive Load Balancer.

This script provides a real-time dashboard for monitoring load balancer performance,
worker utilization, and test distribution during stress tests and normal operation.

Features:
1. Real-time throughput and latency monitoring
2. Worker load visualization with dynamic updates
3. Test distribution across workers visualization
4. System resource utilization monitoring
5. Alert system for performance anomalies
6. Test queue monitoring with priority visualization
"""

import os
import sys
import json
import time
import argparse
import threading
import datetime
import queue
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import socket
import logging

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import load balancer components
from duckdb_api.distributed_testing.load_balancer import (
    LoadBalancerService,
    WorkerCapabilities,
    WorkerLoad,
    TestRequirements,
    WorkerAssignment
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("load_balancer_live_dashboard")

# Constants for terminal-based dashboard
DASHBOARD_WIDTH = 100
DASHBOARD_HEIGHT = 40
REFRESH_RATE = 1.0  # seconds

class DashboardMetricsCollector:
    """Collect and store metrics for the dashboard."""
    
    def __init__(self, max_history: int = 60):
        """Initialize metrics collector with history limit."""
        self.lock = threading.RLock()
        self.max_history = max_history
        
        # Time series metrics
        self.throughput_history = []  # (timestamp, value) pairs
        self.latency_history = []  # (timestamp, value) pairs
        self.worker_count_history = []  # (timestamp, value) pairs
        self.pending_tests_history = []  # (timestamp, value) pairs
        
        # Current state metrics
        self.active_workers = set()
        self.worker_loads = {}  # worker_id -> WorkerLoad
        self.test_assignments = {}  # test_id -> WorkerAssignment
        self.pending_tests = {}  # test_id -> TestRequirements
        self.pending_count = 0  # Count of pending tests
        
        # Performance metrics
        self.last_minute_throughput = 0
        self.last_minute_latency = 0
        self.total_tests_processed = 0
        self.total_tests_failed = 0
        self.start_time = datetime.datetime.now()
        
    def add_throughput_point(self, value: float) -> None:
        """Add a throughput data point."""
        with self.lock:
            now = datetime.datetime.now()
            self.throughput_history.append((now, value))
            
            # Trim history if needed
            if len(self.throughput_history) > self.max_history:
                self.throughput_history.pop(0)
                
    def add_latency_point(self, value: float) -> None:
        """Add a latency data point."""
        with self.lock:
            now = datetime.datetime.now()
            self.latency_history.append((now, value))
            
            # Trim history if needed
            if len(self.latency_history) > self.max_history:
                self.latency_history.pop(0)
                
    def update_worker_count(self, count: int) -> None:
        """Update the worker count history."""
        with self.lock:
            now = datetime.datetime.now()
            self.worker_count_history.append((now, count))
            
            # Trim history if needed
            if len(self.worker_count_history) > self.max_history:
                self.worker_count_history.pop(0)
                
    def update_pending_tests_history(self, count: int) -> None:
        """Update the pending tests history."""
        with self.lock:
            now = datetime.datetime.now()
            self.pending_tests_history.append((now, count))
            
            # Trim history if needed
            if len(self.pending_tests_history) > self.max_history:
                self.pending_tests_history.pop(0)
                
    def update_worker_loads(self, worker_loads: Dict[str, WorkerLoad]) -> None:
        """Update the worker load information."""
        with self.lock:
            self.worker_loads = worker_loads
            
    def update_active_workers(self, active_workers: set) -> None:
        """Update the set of active workers."""
        with self.lock:
            self.active_workers = active_workers
            
    def update_test_assignments(self, assignments: Dict[str, WorkerAssignment]) -> None:
        """Update the test assignment information."""
        with self.lock:
            self.test_assignments = assignments
            
    def update_pending_tests(self, pending: Dict[str, TestRequirements]) -> None:
        """Update the pending tests information."""
        with self.lock:
            self.pending_tests = pending
            self.pending_count = len(pending)
            
    def increment_total_processed(self) -> None:
        """Increment the total tests processed counter."""
        with self.lock:
            self.total_tests_processed += 1
            
    def increment_total_failed(self) -> None:
        """Increment the total tests failed counter."""
        with self.lock:
            self.total_tests_failed += 1
            
    def update_last_minute_metrics(self) -> None:
        """Update the last minute performance metrics."""
        with self.lock:
            # Calculate throughput for the last minute
            now = datetime.datetime.now()
            one_minute_ago = now - datetime.timedelta(minutes=1)
            
            # Filter throughput history for last minute
            last_minute_points = [
                value for timestamp, value in self.throughput_history
                if timestamp >= one_minute_ago
            ]
            
            self.last_minute_throughput = sum(last_minute_points) / len(last_minute_points) if last_minute_points else 0
            
            # Filter latency history for last minute
            last_minute_latency = [
                value for timestamp, value in self.latency_history
                if timestamp >= one_minute_ago
            ]
            
            self.last_minute_latency = sum(last_minute_latency) / len(last_minute_latency) if last_minute_latency else 0
            
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        with self.lock:
            # Update last minute metrics before returning
            self.update_last_minute_metrics()
            
            return {
                "throughput": {
                    "current": self.throughput_history[-1][1] if self.throughput_history else 0,
                    "last_minute_avg": self.last_minute_throughput,
                    "history": self.throughput_history
                },
                "latency": {
                    "current": self.latency_history[-1][1] if self.latency_history else 0,
                    "last_minute_avg": self.last_minute_latency,
                    "history": self.latency_history
                },
                "workers": {
                    "count": len(self.active_workers),
                    "active": self.active_workers,
                    "loads": self.worker_loads,
                    "history": self.worker_count_history
                },
                "tests": {
                    "total_processed": self.total_tests_processed,
                    "total_failed": self.total_tests_failed,
                    "current_pending": self.pending_count,
                    "current_assigned": len(self.test_assignments),
                    "pending_history": self.pending_tests_history
                },
                "runtime": {
                    "uptime": (datetime.datetime.now() - self.start_time).total_seconds(),
                    "start_time": self.start_time.isoformat()
                }
            }


class TerminalDashboard:
    """Terminal-based dashboard for load balancer monitoring."""
    
    def __init__(self, metrics_collector: DashboardMetricsCollector):
        """Initialize the terminal dashboard."""
        self.metrics = metrics_collector
        self.stop_event = threading.Event()
        self.update_thread = None
        
    def start(self) -> None:
        """Start the dashboard."""
        # Clear the terminal
        self._clear_screen()
        
        # Start update thread
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self.update_thread.start()
        
    def stop(self) -> None:
        """Stop the dashboard."""
        self.stop_event.set()
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=2)
            
    def _update_loop(self) -> None:
        """Main update loop for the dashboard."""
        while not self.stop_event.is_set():
            try:
                # Update the display
                self._update_display()
                
                # Wait for refresh interval
                self.stop_event.wait(REFRESH_RATE)
                
            except Exception as e:
                logger.error(f"Error updating dashboard: {e}")
                time.sleep(1)  # Avoid rapid error loops
                
    def _clear_screen(self) -> None:
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
        
    def _update_display(self) -> None:
        """Update the terminal display with current metrics."""
        # Get current metrics
        summary = self.metrics.get_summary()
        
        # Clear screen
        self._clear_screen()
        
        # Print header
        self._print_header()
        
        # Print key metrics
        self._print_key_metrics(summary)
        
        # Print throughput graph
        self._print_throughput_graph(summary)
        
        # Print worker utilization
        self._print_worker_utilization(summary)
        
        # Print pending tests
        self._print_pending_tests(summary)
        
        # Print footer
        self._print_footer(summary)
        
    def _print_header(self) -> None:
        """Print dashboard header."""
        header = "Load Balancer Live Monitoring Dashboard"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("=" * DASHBOARD_WIDTH)
        print(f"{header:^{DASHBOARD_WIDTH}}")
        print(f"Time: {timestamp:^{DASHBOARD_WIDTH-6}}")
        print("=" * DASHBOARD_WIDTH)
        
    def _print_key_metrics(self, summary: Dict[str, Any]) -> None:
        """Print key performance metrics."""
        # Extract metrics
        current_throughput = summary["throughput"]["current"]
        avg_throughput = summary["throughput"]["last_minute_avg"]
        current_latency = summary["latency"]["current"]
        avg_latency = summary["latency"]["last_minute_avg"]
        worker_count = summary["workers"]["count"]
        pending_tests = summary["tests"]["current_pending"]
        assigned_tests = summary["tests"]["current_assigned"]
        total_processed = summary["tests"]["total_processed"]
        total_failed = summary["tests"]["total_failed"]
        uptime = summary["runtime"]["uptime"]
        
        # Format uptime
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Print metrics section
        print("\n=== Key Metrics ===\n")
        
        # First row
        print(f"Throughput: {current_throughput:.2f} tests/s (Avg: {avg_throughput:.2f})", end="  |  ")
        print(f"Latency: {current_latency:.3f}s (Avg: {avg_latency:.3f}s)", end="  |  ")
        print(f"Uptime: {uptime_str}")
        
        # Second row
        print(f"Workers: {worker_count}", end="  |  ")
        print(f"Pending Tests: {pending_tests}", end="  |  ")
        print(f"Assigned Tests: {assigned_tests}", end="  |  ")
        print(f"Processed: {total_processed}")
        
        # Success rate
        if total_processed > 0:
            success_rate = ((total_processed - total_failed) / total_processed) * 100
            print(f"Success Rate: {success_rate:.2f}%")
        
    def _print_throughput_graph(self, summary: Dict[str, Any]) -> None:
        """Print throughput time series graph."""
        throughput_history = summary["throughput"]["history"]
        
        if not throughput_history:
            return
            
        # Extract values for graph
        values = [value for _, value in throughput_history]
        max_value = max(values) if values else 1.0
        graph_height = 5
        graph_width = min(len(values), 50)
        
        # Print graph title
        print("\n=== Throughput (tests/s) ===\n")
        
        # Generate graph
        for i in range(graph_height, 0, -1):
            threshold = max_value * (i / graph_height)
            line = ""
            for value in values[-graph_width:]:
                if value >= threshold:
                    line += "█"
                else:
                    line += " "
            
            # Print with y-axis label
            y_label = f"{threshold:5.2f} |"
            print(f"{y_label} {line}")
            
        # Print x-axis
        x_axis = "-" * graph_width
        print(f"       |{x_axis}")
        
        # Print x-axis labels
        if len(throughput_history) >= 2:
            first_time = throughput_history[-graph_width][0] if len(throughput_history) >= graph_width else throughput_history[0][0]
            last_time = throughput_history[-1][0]
            first_label = first_time.strftime("%H:%M:%S")
            last_label = last_time.strftime("%H:%M:%S")
            
            time_axis = f"{first_label}{' ' * (graph_width - len(first_label) - len(last_label))}{last_label}"
            print(f"        {time_axis}")
        
    def _print_worker_utilization(self, summary: Dict[str, Any]) -> None:
        """Print worker utilization information."""
        worker_loads = summary["workers"]["loads"]
        
        if not worker_loads:
            return
            
        # Print section title
        print("\n=== Worker Utilization ===\n")
        
        # Headers
        print(f"{'Worker ID':<20} {'CPU %':<8} {'Memory %':<10} {'GPU %':<8} {'Status':<10}")
        print("-" * 60)
        
        # Print top 5 workers by CPU utilization
        sorted_workers = sorted(
            worker_loads.items(),
            key=lambda x: x[1].cpu_utilization,
            reverse=True
        )
        
        for worker_id, load in sorted_workers[:5]:
            # Shortened worker ID
            short_id = worker_id[:12] + "..." if len(worker_id) > 15 else worker_id
            
            # Get status
            status = "Normal"
            if hasattr(load, 'is_warming') and load.is_warming:
                status = "Warming"
            elif hasattr(load, 'is_cooling') and load.is_cooling:
                status = "Cooling"
                
            cpu_util = load.cpu_utilization
            mem_util = load.memory_utilization
            gpu_util = load.gpu_utilization
            
            print(f"{short_id:<20} {cpu_util:<8.1f} {mem_util:<10.1f} {gpu_util:<8.1f} {status:<10}")
            
        # If more workers, show count
        if len(worker_loads) > 5:
            print(f"\n... and {len(worker_loads) - 5} more workers")
        
    def _print_pending_tests(self, summary: Dict[str, Any]) -> None:
        """Print pending tests information."""
        pending_count = summary["tests"]["current_pending"]
        
        # Print section title
        print("\n=== Pending Tests ===\n")
        
        if pending_count == 0:
            print("No pending tests.")
            return
            
        print(f"Total Pending: {pending_count}")
        
        # Get test distribution by priority if available
        if hasattr(self.metrics, 'pending_tests') and self.metrics.pending_tests:
            priority_counts = {}
            for test_req in self.metrics.pending_tests.values():
                priority = test_req.priority
                if priority not in priority_counts:
                    priority_counts[priority] = 0
                priority_counts[priority] += 1
                
            # Print priority distribution
            if priority_counts:
                print("\nPriority Distribution:")
                for priority in sorted(priority_counts.keys()):
                    count = priority_counts[priority]
                    percentage = (count / pending_count) * 100
                    print(f"  Priority {priority}: {count} tests ({percentage:.1f}%)")
        
    def _print_footer(self, summary: Dict[str, Any]) -> None:
        """Print dashboard footer."""
        hostname = socket.gethostname()
        uptime = summary["runtime"]["uptime"]
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        print("\n" + "=" * DASHBOARD_WIDTH)
        print(f"Host: {hostname} | Uptime: {uptime_str} | Press Ctrl+C to quit")
        print("=" * DASHBOARD_WIDTH)


class LoadBalancerMonitor:
    """Monitor for a load balancer service."""
    
    def __init__(self, load_balancer: LoadBalancerService):
        """Initialize the monitor with a load balancer service."""
        self.load_balancer = load_balancer
        self.metrics = DashboardMetricsCollector()
        self.dashboard = TerminalDashboard(self.metrics)
        self.stop_event = threading.Event()
        self.monitor_thread = None
        
    def start(self) -> None:
        """Start the monitor."""
        # Register callback on the load balancer
        self.load_balancer.register_assignment_callback(self._assignment_callback)
        
        # Start monitor thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        # Start dashboard
        self.dashboard.start()
        
        logger.info("Load balancer monitor started")
        
    def stop(self) -> None:
        """Stop the monitor."""
        self.stop_event.set()
        
        # Stop dashboard
        self.dashboard.stop()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
            
        logger.info("Load balancer monitor stopped")
        
    def _assignment_callback(self, assignment: WorkerAssignment) -> None:
        """Callback for test assignments."""
        # Record metrics
        if assignment.status == "completed" or assignment.status == "running":
            latency = (assignment.assigned_at - assignment.submitted_at).total_seconds()
            self.metrics.add_latency_point(latency)
            self.metrics.increment_total_processed()
        elif assignment.status == "failed":
            self.metrics.increment_total_failed()
            
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        last_throughput_time = time.time()
        throughput_count = 0
        
        while not self.stop_event.is_set():
            try:
                # Get current state from load balancer
                with self.load_balancer.lock:
                    active_workers = set(self.load_balancer.workers.keys())
                    worker_loads = {
                        worker_id: load for worker_id, load in self.load_balancer.worker_loads.items()
                    }
                    test_assignments = self.load_balancer.test_assignments.copy()
                    pending_tests = self.load_balancer.pending_tests.copy()
                    
                    # Count newly assigned tests for throughput
                    new_assignments = len(test_assignments) - throughput_count
                    throughput_count = len(test_assignments)
                    
                # Update metric collector
                self.metrics.update_active_workers(active_workers)
                self.metrics.update_worker_loads(worker_loads)
                self.metrics.update_test_assignments(test_assignments)
                self.metrics.update_pending_tests(pending_tests)
                self.metrics.update_worker_count(len(active_workers))
                self.metrics.update_pending_tests_history(len(pending_tests))
                
                # Calculate throughput (assignments per second)
                now = time.time()
                elapsed = now - last_throughput_time
                
                if elapsed >= 1.0:  # Calculate throughput every second
                    throughput = new_assignments / elapsed
                    self.metrics.add_throughput_point(throughput)
                    
                    # Reset throughput tracking
                    last_throughput_time = now
                    new_assignments = 0
                
                # Sleep briefly
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(1)  # Avoid rapid error loops


def monitor_running_load_balancer(args: argparse.Namespace) -> None:
    """Attach to a running load balancer service."""
    # Create load balancer service
    load_balancer = LoadBalancerService()
    
    # Create monitor
    monitor = LoadBalancerMonitor(load_balancer)
    
    try:
        # Start monitor
        monitor.start()
        
        # Wait for keyboard interrupt
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down monitor...")
        
    finally:
        # Stop monitor
        monitor.stop()


def run_stress_test_with_monitor(args: argparse.Namespace) -> None:
    """Run a stress test with monitoring."""
    from duckdb_api.distributed_testing.test_load_balancer_stress import LoadBalancerStressTest
    
    # Create load balancer service
    load_balancer = LoadBalancerService()
    
    # Create monitor
    monitor = LoadBalancerMonitor(load_balancer)
    
    # Create stress test
    stress_test = LoadBalancerStressTest(
        num_workers=args.workers,
        num_tests=args.tests,
        duration=args.duration,
        burst_mode=args.burst,
        dynamic_workers=args.dynamic
    )
    
    # Override load_balancer with the one we're monitoring
    stress_test.load_balancer = load_balancer
    
    try:
        # Start load balancer
        load_balancer.start()
        
        # Start monitor
        monitor.start()
        
        # Run stress test in a separate thread
        stress_thread = threading.Thread(
            target=stress_test.run,
            daemon=True
        )
        stress_thread.start()
        
        # Wait for stress test to complete
        while stress_thread.is_alive():
            time.sleep(1)
            
        # Keep dashboard running for a moment to see final state
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("\nShutting down test and monitor...")
        
    finally:
        # Stop monitor
        monitor.stop()
        
        # Stop load balancer
        load_balancer.stop()


def run_scenario_with_monitor(args: argparse.Namespace) -> None:
    """Run a specific scenario with monitoring."""
    from duckdb_api.distributed_testing.test_load_balancer_stress import LoadBalancerStressTest, load_config, get_scenario_configuration
    
    # Load configuration
    config = load_config(args.config)
    
    # Get scenario configuration
    scenario_config = get_scenario_configuration(config, args.scenario)
    if not scenario_config:
        logger.error(f"Failed to load configuration for scenario {args.scenario}")
        return
    
    # Create load balancer service
    load_balancer = LoadBalancerService()
    
    # Create monitor
    monitor = LoadBalancerMonitor(load_balancer)
    
    # Create stress test
    stress_test = LoadBalancerStressTest(
        num_workers=scenario_config.get('workers'),
        num_tests=scenario_config.get('tests'),
        duration=scenario_config.get('duration'),
        worker_memory=scenario_config.get('worker_memory'),
        worker_cuda=scenario_config.get('worker_cuda'),
        test_memory=scenario_config.get('test_memory'),
        test_cuda=scenario_config.get('test_cuda'),
        burst_mode=scenario_config.get('burst_mode', False),
        dynamic_workers=scenario_config.get('dynamic_workers', False)
    )
    
    # Override load_balancer with the one we're monitoring
    stress_test.load_balancer = load_balancer
    
    try:
        # Start load balancer
        load_balancer.start()
        
        # Start monitor
        monitor.start()
        
        # Run stress test in a separate thread
        stress_thread = threading.Thread(
            target=stress_test.run,
            daemon=True
        )
        stress_thread.start()
        
        # Wait for stress test to complete
        while stress_thread.is_alive():
            time.sleep(1)
            
        # Keep dashboard running for a moment to see final state
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("\nShutting down test and monitor...")
        
    finally:
        # Stop monitor
        monitor.stop()
        
        # Stop load balancer
        load_balancer.stop()


def main():
    """Main function to parse arguments and start the dashboard."""
    parser = argparse.ArgumentParser(description="Load Balancer Live Monitoring Dashboard")
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Dashboard mode")
    
    # Monitor mode - attach to a running load balancer
    monitor_parser = subparsers.add_parser("monitor", help="Attach to a running load balancer")
    
    # Stress test mode - run a stress test with monitoring
    stress_parser = subparsers.add_parser("stress", help="Run a stress test with monitoring")
    stress_parser.add_argument("--workers", type=int, default=20,
                              help="Number of workers (default: 20)")
    stress_parser.add_argument("--tests", type=int, default=100,
                              help="Number of tests (default: 100)")
    stress_parser.add_argument("--duration", type=int, default=60,
                              help="Test duration in seconds (default: 60)")
    stress_parser.add_argument("--burst", action="store_true",
                              help="Submit tests in bursts rather than evenly distributed")
    stress_parser.add_argument("--dynamic", action="store_true",
                              help="Add/remove workers during the test")
    
    # Scenario mode - run a specific scenario with monitoring
    scenario_parser = subparsers.add_parser("scenario", help="Run a specific scenario with monitoring")
    scenario_parser.add_argument("scenario", type=str,
                                help="Name of the scenario to run")
    scenario_parser.add_argument("--config", type=str, default="",
                                help="Path to configuration file (JSON format)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Default to monitor mode if no mode specified
    if not args.mode:
        args.mode = "monitor"
    
    try:
        # Run appropriate mode
        if args.mode == "monitor":
            monitor_running_load_balancer(args)
        elif args.mode == "stress":
            run_stress_test_with_monitor(args)
        elif args.mode == "scenario":
            run_scenario_with_monitor(args)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())