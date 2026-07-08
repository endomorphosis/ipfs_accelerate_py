#!/usr/bin/env python3
"""
Hardware Optimization Dashboard

This script runs the Hardware Optimization Dashboard, integrating the hardware optimization
recommendations with a web interface for easy access.
"""

import os
import sys
import logging
import argparse
import webbrowser
import subprocess
import time
from pathlib import Path
from threading import Thread

# Add parent directory to path to allow importing from project root
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hardware_optimization_dashboard")

def check_server_status(host: str, port: int, max_retries: int = 30, retry_interval: float = 1.0) -> bool:
    """
    Check if the API server is running.
    
    Args:
        host: Server host
        port: Server port
        max_retries: Maximum number of retries
        retry_interval: Interval between retries in seconds
        
    Returns:
        True if server is running, False otherwise
    """
    import socket
    
    for _ in range(max_retries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((host, port))
                return True
        except:
            time.sleep(retry_interval)
    
    return False

def start_api_server(
    host: str = "localhost",
    port: int = 8080,
    benchmark_db_path: str = "benchmark_db.duckdb",
    api_key: str = None,
    sync_benchmark_data: bool = True
) -> subprocess.Popen:
    """
    Start the Unified API Server with Hardware Optimization integration.
    
    Args:
        host: Server host
        port: Server port
        benchmark_db_path: Path to benchmark database
        api_key: Optional API key
        sync_benchmark_data: Whether to synchronize benchmark data
        
    Returns:
        Subprocess handle
    """
    try:
        # First, update the unified API server
        update_script = parent_dir / "test" / "api_server" / "update_hardware_optimization.py"
        
        if update_script.exists():
            logger.info("Updating Unified API Server with Hardware Optimization integration...")
            
            update_cmd = [
                sys.executable,
                str(update_script),
                "--benchmark-db", benchmark_db_path,
                "--api-url", f"http://{host}:{port}"
            ]
            
            if api_key:
                update_cmd.extend(["--api-key", api_key])
            
            result = subprocess.run(update_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error updating Unified API Server: {result.stderr}")
                logger.info("Proceeding anyway, in case server is already updated")
        else:
            logger.warning(f"Update script not found: {update_script}")
            
        # Update the server with export optimization integration
        export_update_script = parent_dir / "test" / "api_server" / "update_export_optimization.py"
        
        if export_update_script.exists():
            logger.info("Updating Unified API Server with Export Optimization integration...")
            
            update_cmd = [
                sys.executable,
                str(export_update_script),
                "--benchmark-db", benchmark_db_path,
                "--api-url", f"http://{host}:{port}"
            ]
            
            if api_key:
                update_cmd.extend(["--api-key", api_key])
            
            result = subprocess.run(update_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error updating Unified API Server with Export integration: {result.stderr}")
                logger.info("Proceeding anyway, in case server is already updated")
        else:
            logger.warning(f"Export update script not found: {export_update_script}")
        
        # Start the Unified API Server
        server_script = parent_dir / "test" / "api_server" / "unified_api_server.py"
        
        if not server_script.exists():
            logger.error(f"Server script not found: {server_script}")
            return None
        
        logger.info(f"Starting Unified API Server on {host}:{port}...")
        
        server_cmd = [
            sys.executable,
            str(server_script),
            "--host", host,
            "--port", str(port),
            "--database", benchmark_db_path
        ]
        
        if api_key:
            server_cmd.extend(["--api-key", api_key])
        
        server_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        if not check_server_status(host, port):
            logger.error("Failed to start Unified API Server")
            server_process.terminate()
            return None
        
        logger.info(f"Unified API Server started on http://{host}:{port}")
        
        # Synchronize benchmark data if requested
        if sync_benchmark_data:
            logger.info("Synchronizing benchmark data...")
            
            bridge_script = parent_dir / "test" / "integration" / "benchmark_predictive_performance_bridge.py"
            
            if bridge_script.exists():
                sync_cmd = [
                    sys.executable,
                    str(bridge_script),
                    "--benchmark-db", benchmark_db_path,
                    "--api-url", f"http://{host}:{port}",
                    "--limit", "100"
                ]
                
                if api_key:
                    sync_cmd.extend(["--api-key", api_key])
                
                # Run synchronization in a separate thread to not block
                def run_sync():
                    try:
                        result = subprocess.run(sync_cmd, capture_output=True, text=True)
                        if result.returncode == 0:
                            logger.info("Benchmark data synchronization complete")
                        else:
                            logger.warning(f"Error synchronizing benchmark data: {result.stderr}")
                    except Exception as e:
                        logger.warning(f"Error running synchronization: {e}")
                
                Thread(target=run_sync).start()
            else:
                logger.warning(f"Bridge script not found: {bridge_script}")
        
        return server_process
    
    except Exception as e:
        logger.error(f"Error starting API server: {e}")
        return None

def serve_dashboard(
    host: str = "localhost",
    port: int = 8000,
    dashboard_path: str = None
) -> subprocess.Popen:
    """
    Serve the hardware optimization dashboard using a simple HTTP server.
    
    Args:
        host: Server host
        port: Server port
        dashboard_path: Path to dashboard HTML file
        
    Returns:
        Subprocess handle
    """
    try:
        # Find dashboard file
        if dashboard_path is None:
            dashboard_path = parent_dir / "test" / "web_interface" / "hardware_optimization_dashboard.html"
        else:
            dashboard_path = Path(dashboard_path)
        
        if not dashboard_path.exists():
            logger.error(f"Dashboard file not found: {dashboard_path}")
            return None
        
        # Create a directory for serving
        serve_dir = dashboard_path.parent
        
        # Start a simple HTTP server
        logger.info(f"Starting dashboard server on {host}:{port}...")
        
        # Use Python's http.server module
        server_cmd = [
            sys.executable,
            "-m", "http.server",
            str(port),
            "-b", host,
            "-d", str(serve_dir)
        ]
        
        server_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Dashboard server started on http://{host}:{port}")
        
        return server_process
    
    except Exception as e:
        logger.error(f"Error starting dashboard server: {e}")
        return None

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hardware Optimization Dashboard")
    parser.add_argument("--benchmark-db", type=str, default="benchmark_db.duckdb",
                      help="Path to benchmark database")
    parser.add_argument("--api-host", type=str, default="localhost",
                      help="API server host")
    parser.add_argument("--api-port", type=int, default=8080,
                      help="API server port")
    parser.add_argument("--dashboard-host", type=str, default="localhost",
                      help="Dashboard server host")
    parser.add_argument("--dashboard-port", type=int, default=8000,
                      help="Dashboard server port")
    parser.add_argument("--api-key", type=str, help="Optional API key")
    parser.add_argument("--no-sync", action="store_true",
                      help="Skip benchmark data synchronization")
    parser.add_argument("--no-browser", action="store_true",
                      help="Do not open browser automatically")
    
    args = parser.parse_args()
    
    # Start API server
    api_server = start_api_server(
        host=args.api_host,
        port=args.api_port,
        benchmark_db_path=args.benchmark_db,
        api_key=args.api_key,
        sync_benchmark_data=not args.no_sync
    )
    
    if api_server is None:
        logger.error("Failed to start API server")
        return 1
    
    # Start dashboard server
    dashboard_server = serve_dashboard(
        host=args.dashboard_host,
        port=args.dashboard_port
    )
    
    if dashboard_server is None:
        logger.error("Failed to start dashboard server")
        api_server.terminate()
        return 1
    
    # Open browser if requested
    if not args.no_browser:
        dashboard_url = f"http://{args.dashboard_host}:{args.dashboard_port}/hardware_optimization_dashboard.html"
        logger.info(f"Opening dashboard in browser: {dashboard_url}")
        webbrowser.open(dashboard_url)
    
    try:
        logger.info("Press Ctrl+C to stop servers")
        api_server.wait()
    except KeyboardInterrupt:
        logger.info("Stopping servers...")
    finally:
        api_server.terminate()
        dashboard_server.terminate()
        logger.info("Servers stopped")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())