#!/usr/bin/env python3
"""
Performance benchmark for coordinator redundancy in the Distributed Testing Framework.
Measures performance impact of different cluster configurations and workloads.
"""

import anyio
import os
import sys
import time
import json
import argparse
import logging
import statistics
import uuid
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RedundancyBenchmark:
    """Benchmark for coordinator redundancy performance."""
    
    def __init__(self, cluster_sizes=[1, 3, 5], base_port=8080, 
                 operations_per_run=1000, runs=3):
        """Initialize the benchmark."""
        self.cluster_sizes = cluster_sizes
        self.base_port = base_port
        self.operations_per_run = operations_per_run
        self.runs = runs
        self.results = {}
        
    async def run_benchmarks(self):
        """Run all configured benchmarks."""
        for cluster_size in self.cluster_sizes:
            logger.info(f"Running benchmark with cluster size {cluster_size}")
            
            # Store results for this cluster size
            self.results[cluster_size] = {
                "register_worker": [],
                "submit_task": [],
                "update_status": [],
                "query_results": [],
                "cluster_size": cluster_size,
                "operations_per_run": self.operations_per_run,
                "runs": self.runs
            }
            
            for run in range(self.runs):
                logger.info(f"Run {run+1}/{self.runs}")
                
                try:
                    # Start a cluster with the specified size
                    await self._start_cluster(cluster_size)
                    
                    # Wait for the cluster to stabilize
                    await anyio.sleep(5)
                    
                    # Benchmark worker registration
                    register_times = await self._benchmark_register_worker(cluster_size)
                    self.results[cluster_size]["register_worker"].append(register_times)
                    
                    # Benchmark task submission
                    submit_times = await self._benchmark_submit_task(cluster_size)
                    self.results[cluster_size]["submit_task"].append(submit_times)
                    
                    # Benchmark status updates
                    update_times = await self._benchmark_update_status(cluster_size)
                    self.results[cluster_size]["update_status"].append(update_times)
                    
                    # Benchmark result queries
                    query_times = await self._benchmark_query_results(cluster_size)
                    self.results[cluster_size]["query_results"].append(query_times)
                    
                except Exception as e:
                    logger.error(f"Error during benchmark: {e}")
                    
                finally:
                    # Stop the cluster
                    await self._stop_cluster()
                    
        # Generate benchmark report
        self._generate_report()
                    
    async def _start_cluster(self, size):
        """Start a cluster with the specified size."""
        logger.info(f"Starting cluster with {size} nodes")
        
        self.processes = []
        self.temp_dirs = [f"/tmp/benchmark_redundancy_{i}" for i in range(size)]
        
        # Create temporary directories
        for temp_dir in self.temp_dirs:
            os.makedirs(temp_dir, exist_ok=True)
            
        # Start each node
        for i in range(size):
            node_id = f"node-{i+1}"
            port = self.base_port + i
            
            # Create peers list
            peers = ",".join([f"localhost:{self.base_port+j}" for j in range(size) if j != i])
            
            # Create command for starting a coordinator node
            cmd = [
                sys.executable,
                "-m", "distributed_testing.coordinator",
                "--id", node_id,
                "--port", str(port),
                "--db-path", os.path.join(self.temp_dirs[i], "coordinator.duckdb"),
                "--data-dir", self.temp_dirs[i],
                "--enable-redundancy",
                "--peers", peers,
                "--log-level", "ERROR"  # Use ERROR level to reduce logging noise
            ]
            
            # Start the process
            import subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.processes.append(process)
            
        # Wait for cluster to stabilize
        await anyio.sleep(5)
        
    async def _stop_cluster(self):
        """Stop the cluster."""
        logger.info("Stopping cluster")
        
        # Send termination signal to all processes
        for process in self.processes:
            try:
                import signal
                process.send_signal(signal.SIGTERM)
            except Exception as e:
                logger.warning(f"Error stopping process: {e}")
                
        # Wait for processes to exit
        for process in self.processes:
            try:
                process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error waiting for process: {e}")
                
        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Error cleaning up temp dir: {e}")
                
    async def _benchmark_register_worker(self, cluster_size):
        """Benchmark worker registration performance."""
        logger.info("Benchmarking worker registration")
        
        # Choose a random node to interact with
        port = self.base_port + (cluster_size // 2)  # Middle node
        url = f"http://localhost:{port}/api/workers/register"
        
        times = []
        
        # Register workers in parallel for better throughput measurement
        async def register_worker(worker_id):
            try:
                start_time = time.time()
                
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json={
                        "worker_id": worker_id,
                        "host": f"worker-host-{worker_id}",
                        "port": 9000 + int(worker_id.split('-')[1]),
                        "capabilities": ["cpu", "cuda", "rocm"],
                        "status": "idle"
                    }) as response:
                        if response.status != 200:
                            logger.warning(f"Failed to register worker: HTTP {response.status}")
                            return None
                        await response.json()
                        
                end_time = time.time()
                return end_time - start_time
            except Exception as e:
                logger.warning(f"Error registering worker: {e}")
                return None
                
        # Register workers in batches to avoid overwhelming the server
        batch_size = 50
        for batch in range(0, self.operations_per_run, batch_size):
            tasks = []
            for i in range(batch, min(batch + batch_size, self.operations_per_run)):
                worker_id = f"worker-{i}"
                tasks.append(register_worker(worker_id))
                
            batch_times = await # TODO: Replace with task group - asyncio.gather(*tasks)
            times.extend([t for t in batch_times if t is not None])
            
        return times
    
    async def _benchmark_submit_task(self, cluster_size):
        """Benchmark task submission performance."""
        logger.info("Benchmarking task submission")
        
        # Choose a random node to interact with
        port = self.base_port + (cluster_size // 2)  # Middle node
        url = f"http://localhost:{port}/api/tasks/submit"
        
        times = []
        
        # Submit tasks in parallel for better throughput measurement
        async def submit_task(task_id):
            try:
                start_time = time.time()
                
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json={
                        "task_id": task_id,
                        "type": "benchmark",
                        "model": "bert-base-uncased",
                        "hardware": "cuda",
                        "batch_sizes": [1, 2, 4, 8],
                        "priority": 1,
                        "worker_requirements": {
                            "capabilities": ["cuda"]
                        }
                    }) as response:
                        if response.status != 200:
                            logger.warning(f"Failed to submit task: HTTP {response.status}")
                            return None
                        await response.json()
                        
                end_time = time.time()
                return end_time - start_time
            except Exception as e:
                logger.warning(f"Error submitting task: {e}")
                return None
                
        # Submit tasks in batches to avoid overwhelming the server
        batch_size = 50
        for batch in range(0, self.operations_per_run, batch_size):
            tasks = []
            for i in range(batch, min(batch + batch_size, self.operations_per_run)):
                task_id = f"task-{i}"
                tasks.append(submit_task(task_id))
                
            batch_times = await # TODO: Replace with task group - asyncio.gather(*tasks)
            times.extend([t for t in batch_times if t is not None])
            
        return times
    
    async def _benchmark_update_status(self, cluster_size):
        """Benchmark status update performance."""
        logger.info("Benchmarking status updates")
        
        # Choose a random node to interact with
        port = self.base_port + (cluster_size // 2)  # Middle node
        url = f"http://localhost:{port}/api/workers/status"
        
        times = []
        
        # Update status in parallel for better throughput measurement
        async def update_status(worker_id):
            try:
                start_time = time.time()
                
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json={
                        "worker_id": worker_id,
                        "status": "running",
                        "current_task": f"task-{worker_id.split('-')[1]}",
                        "cpu_usage": 80.5,
                        "memory_usage": 4096.5,
                        "gpu_usage": 90.2
                    }) as response:
                        if response.status != 200:
                            logger.warning(f"Failed to update status: HTTP {response.status}")
                            return None
                        await response.json()
                        
                end_time = time.time()
                return end_time - start_time
            except Exception as e:
                logger.warning(f"Error updating status: {e}")
                return None
                
        # Register some workers first
        await self._benchmark_register_worker(cluster_size)
        
        # Update status in batches to avoid overwhelming the server
        batch_size = 50
        for batch in range(0, self.operations_per_run, batch_size):
            tasks = []
            for i in range(batch, min(batch + batch_size, self.operations_per_run)):
                worker_id = f"worker-{i}"
                tasks.append(update_status(worker_id))
                
            batch_times = await # TODO: Replace with task group - asyncio.gather(*tasks)
            times.extend([t for t in batch_times if t is not None])
            
        return times
    
    async def _benchmark_query_results(self, cluster_size):
        """Benchmark result query performance."""
        logger.info("Benchmarking result queries")
        
        # Choose a random node to interact with
        port = self.base_port + (cluster_size // 2)  # Middle node
        url = f"http://localhost:{port}/api/tasks"
        
        times = []
        
        # Query results in parallel for better throughput measurement
        async def query_results():
            try:
                start_time = time.time()
                
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status != 200:
                            logger.warning(f"Failed to query results: HTTP {response.status}")
                            return None
                        await response.json()
                        
                end_time = time.time()
                return end_time - start_time
            except Exception as e:
                logger.warning(f"Error querying results: {e}")
                return None
                
        # Submit some tasks first
        await self._benchmark_submit_task(cluster_size)
        
        # Query results in batches
        batch_size = 50
        for batch in range(0, self.operations_per_run, batch_size):
            tasks = []
            for i in range(batch, min(batch + batch_size, self.operations_per_run)):
                tasks.append(query_results())
                
            batch_times = await # TODO: Replace with task group - asyncio.gather(*tasks)
            times.extend([t for t in batch_times if t is not None])
            
        return times
    
    def _generate_report(self):
        """Generate a benchmark report."""
        logger.info("Generating benchmark report")
        
        # Calculate statistics for each cluster size and operation type
        stats = {}
        
        for cluster_size, data in self.results.items():
            stats[cluster_size] = {}
            
            for op_type in ["register_worker", "submit_task", "update_status", "query_results"]:
                # Flatten the list of lists
                all_times = [time for run_times in data[op_type] for time in run_times]
                
                if all_times:
                    stats[cluster_size][op_type] = {
                        "count": len(all_times),
                        "min": min(all_times) * 1000,  # Convert to ms
                        "max": max(all_times) * 1000,  # Convert to ms
                        "mean": statistics.mean(all_times) * 1000,  # Convert to ms
                        "median": statistics.median(all_times) * 1000,  # Convert to ms
                        "p95": np.percentile(all_times, 95) * 1000,  # Convert to ms
                        "p99": np.percentile(all_times, 99) * 1000,  # Convert to ms
                        "throughput": len(all_times) / sum(all_times)  # Operations per second
                    }
                else:
                    stats[cluster_size][op_type] = {
                        "count": 0,
                        "error": "No valid timing data collected"
                    }
                    
        # Write the report to a JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"redundancy_benchmark_{timestamp}.json"
        
        with open(report_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "cluster_sizes": self.cluster_sizes,
                "operations_per_run": self.operations_per_run,
                "runs": self.runs,
                "stats": stats
            }, f, indent=2)
            
        logger.info(f"Benchmark report saved to {report_file}")
        
        # Generate plots
        self._generate_plots(stats, timestamp)
        
    def _generate_plots(self, stats, timestamp):
        """Generate plots from the benchmark results."""
        logger.info("Generating benchmark plots")
        
        # Set up the figure
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Coordinator Redundancy Performance Benchmark", fontsize=16)
        
        # Colors for different cluster sizes
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.cluster_sizes)))
        
        # Plot types
        plot_types = [
            ("Latency (median ms)", lambda s: s["median"]),
            ("Latency (p95 ms)", lambda s: s["p95"]),
            ("Latency (p99 ms)", lambda s: s["p99"]),
            ("Throughput (ops/sec)", lambda s: s["throughput"])
        ]
        
        # Operation types
        op_types = ["register_worker", "submit_task", "update_status", "query_results"]
        
        # Human-readable operation names
        op_names = {
            "register_worker": "Register Worker",
            "submit_task": "Submit Task",
            "update_status": "Update Status",
            "query_results": "Query Results"
        }
        
        # Generate one plot for each statistic
        for i, (plot_name, stat_func) in enumerate(plot_types):
            ax = axs[i // 2, i % 2]
            ax.set_title(plot_name)
            
            # X positions for the bars
            x = np.arange(len(op_types))
            width = 0.8 / len(self.cluster_sizes)  # Width of the bars
            
            # Plot data for each cluster size
            for j, cluster_size in enumerate(self.cluster_sizes):
                values = []
                for op_type in op_types:
                    if op_type in stats[cluster_size] and "mean" in stats[cluster_size][op_type]:
                        values.append(stat_func(stats[cluster_size][op_type]))
                    else:
                        values.append(0)
                        
                ax.bar(x + j * width - 0.4 + width/2, values, width, 
                      label=f"{cluster_size} nodes" if i == 0 else "", 
                      color=colors[j])
                
            # Set up the axes
            ax.set_xlabel("Operation Type")
            ax.set_xticks(x)
            ax.set_xticklabels([op_names[op] for op in op_types])
            
            if "Throughput" in plot_name:
                ax.set_ylabel("Operations per Second")
            else:
                ax.set_ylabel("Milliseconds")
                
            if i == 0:
                ax.legend()
                
        plt.tight_layout()
        
        # Save the figure
        plot_file = f"redundancy_benchmark_{timestamp}.png"
        plt.savefig(plot_file)
        logger.info(f"Benchmark plots saved to {plot_file}")
        
        # Close the figure to free memory
        plt.close(fig)
        
        # Generate a comparison plot showing scaling behavior
        self._generate_scaling_plot(stats, timestamp)
        
    def _generate_scaling_plot(self, stats, timestamp):
        """Generate a plot showing how performance scales with cluster size."""
        logger.info("Generating scaling plot")
        
        # Set up the figure
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle("Coordinator Redundancy Scaling Behavior", fontsize=16)
        
        # Colors for different operation types
        colors = plt.cm.tab10(np.arange(4))
        
        # Operation types
        op_types = ["register_worker", "submit_task", "update_status", "query_results"]
        
        # Human-readable operation names
        op_names = {
            "register_worker": "Register Worker",
            "submit_task": "Submit Task",
            "update_status": "Update Status",
            "query_results": "Query Results"
        }
        
        # Plot latency scaling
        ax = axs[0]
        ax.set_title("Latency Scaling (median ms)")
        
        for i, op_type in enumerate(op_types):
            values = []
            for cluster_size in self.cluster_sizes:
                if op_type in stats[cluster_size] and "median" in stats[cluster_size][op_type]:
                    values.append(stats[cluster_size][op_type]["median"])
                else:
                    values.append(0)
                    
            ax.plot(self.cluster_sizes, values, 'o-', label=op_names[op_type], color=colors[i])
            
        ax.set_xlabel("Cluster Size (nodes)")
        ax.set_ylabel("Median Latency (ms)")
        ax.legend()
        
        # Plot throughput scaling
        ax = axs[1]
        ax.set_title("Throughput Scaling (ops/sec)")
        
        for i, op_type in enumerate(op_types):
            values = []
            for cluster_size in self.cluster_sizes:
                if op_type in stats[cluster_size] and "throughput" in stats[cluster_size][op_type]:
                    values.append(stats[cluster_size][op_type]["throughput"])
                else:
                    values.append(0)
                    
            ax.plot(self.cluster_sizes, values, 'o-', label=op_names[op_type], color=colors[i])
            
        ax.set_xlabel("Cluster Size (nodes)")
        ax.set_ylabel("Throughput (ops/sec)")
        ax.legend()
        
        plt.tight_layout()
        
        # Save the figure
        plot_file = f"redundancy_scaling_{timestamp}.png"
        plt.savefig(plot_file)
        logger.info(f"Scaling plot saved to {plot_file}")
        
        # Close the figure to free memory
        plt.close(fig)
        

async def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark coordinator redundancy performance")
    parser.add_argument("--cluster-sizes", type=int, nargs="+", default=[1, 3, 5],
                      help="Cluster sizes to benchmark (default: 1 3 5)")
    parser.add_argument("--base-port", type=int, default=18080,
                      help="Base port for coordinator nodes (default: 18080)")
    parser.add_argument("--operations", type=int, default=1000,
                      help="Number of operations per run (default: 1000)")
    parser.add_argument("--runs", type=int, default=3,
                      help="Number of runs for each configuration (default: 3)")
    
    args = parser.parse_args()
    
    benchmark = RedundancyBenchmark(
        cluster_sizes=args.cluster_sizes,
        base_port=args.base_port,
        operations_per_run=args.operations,
        runs=args.runs
    )
    
    await benchmark.run_benchmarks()
    

if __name__ == "__main__":
    anyio.run(main())