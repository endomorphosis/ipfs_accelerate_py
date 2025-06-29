#!/usr/bin/env python3
"""
Benchmark API Client

This script provides a command-line client for interacting with the Benchmark API Server.
It demonstrates how to use the API to start benchmarks, monitor progress, and retrieve results.
"""

import os
import sys
import json
import time
import argparse
import requests
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("Warning: websocket-client package not installed. WebSocket monitoring will not be available.")
    print("To install: pip install websocket-client")
from typing import Dict, Any, Optional, List

class BenchmarkClient:
    """Client for interacting with the Benchmark API Server."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the benchmark API server
        """
        self.base_url = base_url.rstrip('/')
        self.api_base = f"{self.base_url}/api/benchmark"
        
    def start_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start a new benchmark run.
        
        Args:
            config: Benchmark configuration
            
        Returns:
            Response from the server
        """
        url = f"{self.api_base}/run"
        response = requests.post(url, json=config)
        response.raise_for_status()
        return response.json()
    
    def get_status(self, run_id: str) -> Dict[str, Any]:
        """
        Get the status of a benchmark run.
        
        Args:
            run_id: ID of the benchmark run
            
        Returns:
            Status information
        """
        url = f"{self.api_base}/status/{run_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_results(self, run_id: str) -> Dict[str, Any]:
        """
        Get the results of a completed benchmark run.
        
        Args:
            run_id: ID of the benchmark run
            
        Returns:
            Benchmark results
        """
        url = f"{self.api_base}/results/{run_id}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models.
        
        Returns:
            List of model information
        """
        url = f"{self.api_base}/models"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def list_hardware(self) -> List[Dict[str, Any]]:
        """
        Get a list of available hardware platforms.
        
        Returns:
            List of hardware information
        """
        url = f"{self.api_base}/hardware"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def list_reports(self) -> List[Dict[str, Any]]:
        """
        Get a list of available benchmark reports.
        
        Returns:
            List of report information
        """
        url = f"{self.api_base}/reports"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def query_results(self, 
                     model: Optional[str] = None, 
                     hardware: Optional[str] = None,
                     batch_size: Optional[int] = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query benchmark results.
        
        Args:
            model: Filter by model name
            hardware: Filter by hardware type
            batch_size: Filter by batch size
            limit: Maximum number of results to return
            
        Returns:
            List of benchmark results
        """
        url = f"{self.api_base}/query"
        params = {}
        
        if model:
            params["model"] = model
        if hardware:
            params["hardware"] = hardware
        if batch_size:
            params["batch_size"] = batch_size
        
        params["limit"] = limit
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def monitor_progress(self, run_id: str, callback: callable, exit_on_complete: bool = True):
        """
        Monitor the progress of a benchmark run using WebSockets.
        
        Args:
            run_id: ID of the benchmark run
            callback: Function to call with status updates
            exit_on_complete: Whether to exit when the benchmark completes
        """
        if not WEBSOCKET_AVAILABLE:
            print("Cannot monitor progress: websocket-client package not installed.")
            print("Falling back to polling API endpoint...")
            self._poll_progress(run_id, callback, exit_on_complete)
            return
            
        ws_url = f"ws://{self.base_url.split('://', 1)[1]}/api/benchmark/ws/{run_id}"
        
        def on_message(ws, message):
            data = json.loads(message)
            callback(data)
            
            # Exit if the benchmark is complete or failed
            if exit_on_complete and data.get("status") in ["completed", "failed"]:
                ws.close()
                return True
            
            return False
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            print("WebSocket connection closed")
        
        def on_open(ws):
            print(f"WebSocket connection opened for run {run_id}")
        
        # Create and run the WebSocket client
        ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        ws.run_forever()
        
    def _poll_progress(self, run_id: str, callback: callable, exit_on_complete: bool = True):
        """
        Poll the status API endpoint to monitor progress.
        
        Args:
            run_id: ID of the benchmark run
            callback: Function to call with status updates
            exit_on_complete: Whether to exit when the benchmark completes
        """
        print(f"Polling status for run {run_id}...")
        
        try:
            while True:
                # Get status
                status = self.get_status(run_id)
                
                # Call callback
                callback(status)
                
                # Exit if the benchmark is complete or failed
                if exit_on_complete and status.get("status") in ["completed", "failed"]:
                    break
                
                # Wait before polling again
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user.")
        except Exception as e:
            print(f"\nError polling status: {e}")

def print_progress_bar(progress, width=50):
    """Print a progress bar."""
    filled_width = int(width * progress)
    bar = '█' * filled_width + '-' * (width - filled_width)
    print(f"\r[{bar}] {progress:.1%}", end='')

def monitor_callback(data):
    """Callback for handling WebSocket status updates."""
    # Clear the line
    print("\r" + " " * 80, end="\r")
    
    # Print progress information
    status = data.get("status", "unknown")
    progress = data.get("progress", 0)
    current_step = data.get("current_step", "")
    completed = data.get("completed_models", 0)
    total = data.get("total_models", 0)
    
    # Print progress bar
    print_progress_bar(progress)
    
    # Print status information
    print(f" {status.upper()} - {current_step}")
    print(f" Models: {completed}/{total}")
    
    # If there's an error, print it
    if "error" in data:
        print(f"\nError: {data['error']}")
    
    # If the benchmark is complete, print a message
    if status == "completed":
        print("\nBenchmark completed successfully!")
    elif status == "failed":
        print("\nBenchmark failed!")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark API Client")
    
    # Server connection options
    parser.add_argument("--server", default="http://localhost:8000", help="Benchmark API server URL")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Start benchmark command
    start_parser = subparsers.add_parser("start", help="Start a new benchmark run")
    start_parser.add_argument("--priority", default="high", choices=["critical", "high", "medium", "all"], 
                            help="Benchmark priority")
    start_parser.add_argument("--hardware", nargs="+", default=["cpu"], help="Hardware backends to use")
    start_parser.add_argument("--models", nargs="+", help="Specific models to benchmark")
    start_parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8], help="Batch sizes to benchmark")
    start_parser.add_argument("--precision", default="fp32", help="Precision to use (fp32, fp16, etc.)")
    start_parser.add_argument("--progressive-mode", action="store_true", help="Use progressive complexity mode")
    start_parser.add_argument("--incremental", action="store_true", help="Run only missing or outdated benchmarks")
    start_parser.add_argument("--monitor", action="store_true", help="Monitor progress after starting")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Get the status of a benchmark run")
    status_parser.add_argument("run_id", help="ID of the benchmark run")
    
    # Results command
    results_parser = subparsers.add_parser("results", help="Get the results of a completed benchmark run")
    results_parser.add_argument("run_id", help="ID of the benchmark run")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor the progress of a benchmark run")
    monitor_parser.add_argument("run_id", help="ID of the benchmark run")
    
    # Models command
    subparsers.add_parser("models", help="List available models")
    
    # Hardware command
    subparsers.add_parser("hardware", help="List available hardware platforms")
    
    # Reports command
    subparsers.add_parser("reports", help="List available benchmark reports")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query benchmark results")
    query_parser.add_argument("--model", help="Filter by model name")
    query_parser.add_argument("--hardware", help="Filter by hardware type")
    query_parser.add_argument("--batch-size", type=int, help="Filter by batch size")
    query_parser.add_argument("--limit", type=int, default=100, help="Maximum number of results to return")
    
    args = parser.parse_args()
    
    # Create client
    client = BenchmarkClient(args.server)
    
    # Execute command
    try:
        if args.command == "start":
            # Create benchmark configuration
            config = {
                "priority": args.priority,
                "hardware": args.hardware,
                "batch_sizes": args.batch_sizes,
                "precision": args.precision,
                "progressive_mode": args.progressive_mode,
                "incremental": args.incremental
            }
            
            if args.models:
                config["models"] = args.models
            
            # Start benchmark
            print(f"Starting benchmark with configuration: {json.dumps(config, indent=2)}")
            response = client.start_benchmark(config)
            print(f"Benchmark started with run ID: {response['run_id']}")
            
            # Monitor progress if requested
            if args.monitor:
                print("Monitoring benchmark progress...")
                client.monitor_progress(response["run_id"], monitor_callback)
            
        elif args.command == "status":
            status = client.get_status(args.run_id)
            print(json.dumps(status, indent=2))
            
        elif args.command == "results":
            results = client.get_results(args.run_id)
            print(json.dumps(results, indent=2))
            
        elif args.command == "monitor":
            print(f"Monitoring benchmark run {args.run_id}...")
            client.monitor_progress(args.run_id, monitor_callback)
            
        elif args.command == "models":
            models = client.list_models()
            print(json.dumps(models, indent=2))
            
        elif args.command == "hardware":
            hardware = client.list_hardware()
            print(json.dumps(hardware, indent=2))
            
        elif args.command == "reports":
            reports = client.list_reports()
            print(json.dumps(reports, indent=2))
            
        elif args.command == "query":
            results = client.query_results(
                model=args.model,
                hardware=args.hardware,
                batch_size=args.batch_size,
                limit=args.limit
            )
            print(json.dumps(results, indent=2))
            
        else:
            parser.print_help()
            
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()