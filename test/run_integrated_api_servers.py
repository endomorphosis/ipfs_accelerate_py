#!/usr/bin/env python3
"""
Run Integrated API Servers

This script runs the Predictive Performance API server and the Unified API server
with the integrated Predictive Performance API support.
"""

import os
import sys
import time
import signal
import argparse
import logging
import subprocess
import atexit
from pathlib import Path

# Add parent directory to path to allow importing from project root
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_integrated_servers")

class ServerManager:
    """Manager for running API servers."""
    
    def __init__(self):
        """Initialize the server manager."""
        self.processes = {}
        self.running = False
        
        # Register cleanup handler
        atexit.register(self.stop_all)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle signals to ensure clean shutdown."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop_all()
        sys.exit(0)
    
    def start_unified_api(self, gateway_port=8080, predictive_port=8500):
        """Start the Unified API server.
        
        Args:
            gateway_port: Port for the API gateway
            predictive_port: Port for the Predictive Performance API
            
        Returns:
            True if the server was started successfully, False otherwise
        """
        # Find the unified_api_server.py file
        file_paths = [
            Path(__file__).resolve().parent / "unified_api_server.py",
            parent_dir / "test" / "unified_api_server.py"
        ]
        
        unified_api_path = None
        for path in file_paths:
            if path.exists():
                unified_api_path = path
                break
        
        if unified_api_path is None:
            logger.error("Could not find unified_api_server.py")
            return False
        
        # Prepare command
        cmd = [
            sys.executable,
            str(unified_api_path),
            "--gateway-port", str(gateway_port),
            "--predictive-performance-api-port", str(predictive_port)
        ]
        
        logger.info(f"Starting Unified API server with command: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Store the process
            self.processes["unified_api"] = process
            
            # Start output monitoring
            self._start_output_monitoring("unified_api", process)
            
            # Wait a moment to check if process started correctly
            time.sleep(2)
            if process.poll() is not None:
                logger.error(f"Unified API server failed to start. Return code: {process.poll()}")
                return False
            
            logger.info(f"Unified API server started with gateway on port {gateway_port}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting Unified API server: {e}")
            return False
    
    def start_predictive_api(self, host="127.0.0.1", port=8500, db_path=None):
        """Start the Predictive Performance API server directly.
        
        Args:
            host: Host to bind the server
            port: Port to bind the server
            db_path: Path to DuckDB database file
            
        Returns:
            True if the server was started successfully, False otherwise
        """
        # Find the predictive_performance_api_server.py file
        file_paths = [
            Path(__file__).resolve().parent / "api_server" / "predictive_performance_api_server.py",
            parent_dir / "test" / "api_server" / "predictive_performance_api_server.py"
        ]
        
        api_path = None
        for path in file_paths:
            if path.exists():
                api_path = path
                break
        
        if api_path is None:
            logger.error("Could not find predictive_performance_api_server.py")
            return False
        
        # Prepare command
        cmd = [
            sys.executable,
            str(api_path),
            "--host", host,
            "--port", str(port)
        ]
        
        if db_path:
            cmd.extend(["--db", db_path])
        
        logger.info(f"Starting Predictive Performance API server with command: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Store the process
            self.processes["predictive_api"] = process
            
            # Start output monitoring
            self._start_output_monitoring("predictive_api", process)
            
            # Wait a moment to check if process started correctly
            time.sleep(2)
            if process.poll() is not None:
                logger.error(f"Predictive Performance API server failed to start. Return code: {process.poll()}")
                return False
            
            logger.info(f"Predictive Performance API server started on {host}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting Predictive Performance API server: {e}")
            return False
    
    def _start_output_monitoring(self, server_name, process):
        """Start monitoring the output of a process.
        
        Args:
            server_name: Name of the server
            process: The process object
        """
        import threading
        
        def monitor_output(stream, prefix):
            for line in iter(stream.readline, ''):
                if line:
                    logger.info(f"{prefix}: {line.strip()}")
        
        # Start monitoring stdout
        stdout_thread = threading.Thread(
            target=monitor_output,
            args=(process.stdout, f"{server_name.upper()}-STDOUT"),
            daemon=True
        )
        stdout_thread.start()
        
        # Start monitoring stderr
        stderr_thread = threading.Thread(
            target=monitor_output,
            args=(process.stderr, f"{server_name.upper()}-STDERR"),
            daemon=True
        )
        stderr_thread.start()
    
    def stop_server(self, server_name):
        """Stop a server.
        
        Args:
            server_name: Name of the server to stop
            
        Returns:
            True if the server was stopped successfully, False otherwise
        """
        if server_name not in self.processes or not self.processes[server_name]:
            logger.info(f"{server_name} not running")
            return True
        
        process = self.processes[server_name]
        
        try:
            # Try to terminate the process gracefully
            logger.info(f"Stopping {server_name}...")
            process.terminate()
            
            # Wait for the process to terminate
            for _ in range(10):  # Wait up to 5 seconds
                if process.poll() is not None:
                    break
                time.sleep(0.5)
            
            # If the process is still running, kill it
            if process.poll() is None:
                logger.warning(f"{server_name} did not terminate gracefully, killing...")
                process.kill()
                process.wait()
            
            logger.info(f"{server_name} stopped")
            self.processes[server_name] = None
            return True
            
        except Exception as e:
            logger.error(f"Error stopping {server_name}: {e}")
            return False
    
    def stop_all(self):
        """Stop all running servers."""
        if not self.running:
            return
        
        logger.info("Stopping all servers...")
        
        for server_name in list(self.processes.keys()):
            self.stop_server(server_name)
        
        self.running = False
        logger.info("All servers stopped")
    
    def run_both_servers(self, gateway_port=8080, predictive_port=8500, db_path=None):
        """Run both the Predictive Performance API server and the Unified API server.
        
        Args:
            gateway_port: Port for the API gateway
            predictive_port: Port for the Predictive Performance API
            db_path: Path to DuckDB database file
            
        Returns:
            True if both servers were started successfully, False otherwise
        """
        # Start the Predictive Performance API server
        if not self.start_predictive_api(port=predictive_port, db_path=db_path):
            logger.error("Failed to start Predictive Performance API server")
            return False
        
        # Give it a moment to fully initialize
        time.sleep(2)
        
        # Start the Unified API server
        if not self.start_unified_api(gateway_port=gateway_port, predictive_port=predictive_port):
            logger.error("Failed to start Unified API server")
            self.stop_server("predictive_api")
            return False
        
        self.running = True
        
        # Print success message with URLs
        print("\n=== API SERVERS RUNNING ===")
        print(f"Unified API Gateway: http://localhost:{gateway_port}")
        print(f"Predictive Performance API: http://localhost:{predictive_port}/api/predictive-performance")
        print(f"API Documentation: http://localhost:{gateway_port}/docs")
        print("Press Ctrl+C to stop the servers...")
        
        return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Integrated API Servers")
    parser.add_argument("--gateway-port", type=int, default=8080, help="Port for the API gateway")
    parser.add_argument("--predictive-port", type=int, default=8500, help="Port for the Predictive Performance API")
    parser.add_argument("--db-path", type=str, help="Path to DuckDB database file")
    parser.add_argument("--predictive-only", action="store_true", help="Run only the Predictive Performance API server")
    parser.add_argument("--unified-only", action="store_true", help="Run only the Unified API server")
    
    args = parser.parse_args()
    
    manager = ServerManager()
    
    try:
        if args.predictive_only:
            # Run only the Predictive Performance API server
            if not manager.start_predictive_api(port=args.predictive_port, db_path=args.db_path):
                return 1
            
            print(f"\nPredictive Performance API running on: http://localhost:{args.predictive_port}/api/predictive-performance")
            print("Press Ctrl+C to stop the server...")
            
        elif args.unified_only:
            # Run only the Unified API server
            if not manager.start_unified_api(gateway_port=args.gateway_port, predictive_port=args.predictive_port):
                return 1
            
            print(f"\nUnified API Gateway running on: http://localhost:{args.gateway_port}")
            print(f"API Documentation: http://localhost:{args.gateway_port}/docs")
            print("Press Ctrl+C to stop the server...")
            
        else:
            # Run both servers
            if not manager.run_both_servers(
                gateway_port=args.gateway_port,
                predictive_port=args.predictive_port,
                db_path=args.db_path
            ):
                return 1
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        manager.stop_all()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())