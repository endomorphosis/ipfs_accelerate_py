#!/usr/bin/env python3
"""
Predictive Performance Integration Module

This module provides the integration between the Predictive Performance API
and the Unified API Server for IPFS Accelerate.
"""

import os
import sys
import logging
import subprocess
import time
import signal
import atexit
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("predictive_performance_integration")

class PredictivePerformanceServiceManager:
    """Manager for the Predictive Performance API service."""
    
    def __init__(self, host="127.0.0.1", port=8500, db_path=None, benchmark_dir=None, benchmark_db=None):
        """Initialize the service manager.
        
        Args:
            host: Host to bind the service to
            port: Port to bind the service to
            db_path: Path to DuckDB database file
            benchmark_dir: Path to benchmark results directory
            benchmark_db: Path to benchmark database
        """
        self.host = host
        self.port = port
        self.db_path = db_path
        self.benchmark_dir = benchmark_dir
        self.benchmark_db = benchmark_db
        self.process = None
        
        # Find module path
        self.module_path = self._find_module_path()
        
        # Register cleanup handler
        atexit.register(self.stop)
    
    def _find_module_path(self):
        """Find the path to the predictive_performance_api_server.py file."""
        # Try relative path
        current_dir = Path(__file__).resolve().parent.parent
        module_path = current_dir / "predictive_performance_api_server.py"
        
        if module_path.exists():
            return str(module_path)
        
        # Try repository root
        repo_root = current_dir.parent.parent
        module_path = repo_root / "test" / "api_server" / "predictive_performance_api_server.py"
        
        if module_path.exists():
            return str(module_path)
        
        # Return default path and let it fail if not found
        return str(current_dir / "predictive_performance_api_server.py")
    
    def start(self):
        """Start the Predictive Performance API service.
        
        Returns:
            True if the service was started successfully, False otherwise
        """
        # If already running, return
        if self.is_running():
            logger.info(f"Predictive Performance API already running on {self.host}:{self.port}")
            return True
        
        # Prepare command
        cmd = [
            sys.executable,
            self.module_path,
            "--host", str(self.host),
            "--port", str(self.port)
        ]
        
        # Add optional arguments
        if self.db_path:
            cmd.extend(["--db", str(self.db_path)])
        
        if self.benchmark_dir:
            cmd.extend(["--benchmark-dir", str(self.benchmark_dir)])
        
        if self.benchmark_db:
            cmd.extend(["--benchmark-db", str(self.benchmark_db)])
        
        logger.info(f"Starting Predictive Performance API with command: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Start output monitoring
            self._start_output_monitoring()
            
            # Wait a moment to check if process started correctly
            time.sleep(2)
            if not self.is_running():
                logger.error(f"Predictive Performance API failed to start. Return code: {self.process.poll()}")
                return False
            
            logger.info(f"Predictive Performance API started on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting Predictive Performance API: {e}")
            return False
    
    def _start_output_monitoring(self):
        """Start monitoring the output of the process."""
        import threading
        
        def monitor_output(stream, prefix):
            for line in iter(stream.readline, ''):
                if line:
                    logger.info(f"{prefix}: {line.strip()}")
        
        # Start monitoring stdout
        stdout_thread = threading.Thread(
            target=monitor_output,
            args=(self.process.stdout, "PREDICTIVE-API"),
            daemon=True
        )
        stdout_thread.start()
        
        # Start monitoring stderr
        stderr_thread = threading.Thread(
            target=monitor_output,
            args=(self.process.stderr, "PREDICTIVE-API-ERR"),
            daemon=True
        )
        stderr_thread.start()
    
    def stop(self):
        """Stop the Predictive Performance API service.
        
        Returns:
            True if the service was stopped successfully, False otherwise
        """
        if not self.process:
            logger.info("Predictive Performance API not running")
            return True
        
        try:
            # Try to terminate the process gracefully
            logger.info("Stopping Predictive Performance API...")
            self.process.terminate()
            
            # Wait for the process to terminate
            for _ in range(10):  # Wait up to 5 seconds
                if self.process.poll() is not None:
                    break
                time.sleep(0.5)
            
            # If the process is still running, kill it
            if self.process.poll() is None:
                logger.warning("Predictive Performance API did not terminate gracefully, killing...")
                self.process.kill()
                self.process.wait()
            
            logger.info("Predictive Performance API stopped")
            self.process = None
            return True
            
        except Exception as e:
            logger.error(f"Error stopping Predictive Performance API: {e}")
            return False
    
    def is_running(self):
        """Check if the Predictive Performance API service is running.
        
        Returns:
            True if the service is running, False otherwise
        """
        return self.process is not None and self.process.poll() is None
    
    def get_status(self):
        """Get the status of the Predictive Performance API service.
        
        Returns:
            Dict with service status information
        """
        running = self.is_running()
        
        return {
            "running": running,
            "exit_code": self.process.poll() if self.process and not running else None,
            "host": self.host,
            "port": self.port,
            "url": f"http://{self.host}:{self.port}/api/predictive-performance"
        }

def integrate_with_unified_api(config=None):
    """Add Predictive Performance API configuration to unified API config.
    
    Args:
        config: Unified API configuration dict to update
        
    Returns:
        Updated configuration dict
    """
    if config is None:
        config = {}
    
    # Add Predictive Performance API to config
    config["predictive_performance_api"] = {
        "enabled": True,
        "port": 8500,
        "host": "127.0.0.1",
        "module": "api_server.predictive_performance_api_server",
        "args": []
    }
    
    return config

def get_router_code():
    """Get the code to add to the API gateway for Predictive Performance API routing.
    
    Returns:
        String containing Python code for API gateway routing
    """
    code = """
# Predictive Performance API routes
@app.api_route("/api/predictive-performance{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"])
async def predictive_performance_api_route(request: Request, path: str):
    """Route requests to the Predictive Performance API."""
    url = f"{PREDICTIVE_PERFORMANCE_API_URL}/api/predictive-performance{path}"
    return await forward_request(url, request)

@app.websocket("/api/predictive-performance/ws/{task_id}")
async def predictive_performance_api_websocket(websocket: WebSocket, task_id: str):
    """WebSocket connection for the Predictive Performance API."""
    await websocket_forward(websocket, f"{PREDICTIVE_PERFORMANCE_API_URL}/api/predictive-performance/ws/{task_id}")
"""
    return code

def get_gateway_url_code():
    """Get the code to add to the API gateway for Predictive Performance API URL configuration.
    
    Returns:
        String containing Python code for API gateway URL configuration
    """
    code = """
# Configure Predictive Performance API endpoint
PREDICTIVE_PERFORMANCE_API_URL = "http://{predictive_performance_api_host}:{predictive_performance_api_port}"
"""
    return code

def get_root_endpoint_addition():
    """Get the code to add to the API gateway's root endpoint for Predictive Performance API.
    
    Returns:
        String containing Python dictionary entry for root endpoint
    """
    code = """
            {"name": "Predictive Performance API", "url": f"{PREDICTIVE_PERFORMANCE_API_URL}/api/predictive-performance"},
"""
    return code

def get_db_endpoint_addition():
    """Get the code to add to the API gateway's database endpoints for Predictive Performance API.
    
    Returns:
        String containing Python dictionary entry for database endpoints
    """
    code = """
            {"name": "Predictive Performance Database API", "url": "/api/db/predictive-performance"},
"""
    return code

def get_db_routes_code():
    """Get the code to add to the API gateway for Predictive Performance database API routing.
    
    Returns:
        String containing Python code for API gateway database routing
    """
    code = """
# Database API routes - Predictive Performance
@app.api_route("/api/db/predictive-performance{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def predictive_performance_db_route(request: Request, path: str, api_key: str = Depends(get_api_key)):
    """
    Route requests to the Predictive Performance Database API.
    
    All database operations require API key authentication.
    """
    url = f"{PREDICTIVE_PERFORMANCE_API_URL}/api/predictive-performance/db{path}"
    return await forward_request(url, request)
"""
    return code

def get_cross_component_additions():
    """Get the code to add to the API gateway for cross-component database operations involving Predictive Performance.
    
    Returns:
        String containing Python code for cross-component operations
    """
    code = """
# Get Predictive Performance statistics
predictive_stats = await client.get(f"{PREDICTIVE_PERFORMANCE_API_URL}/api/predictive-performance/stats")
predictive_data = predictive_stats.json() if predictive_stats.status_code == 200 else {"error": "Failed to fetch predictive performance stats"}

# Add to response
response_data["predictive_stats"] = predictive_data
"""
    return code

def create_patch_files():
    """Create patch files for integrating Predictive Performance API with Unified API.
    
    Returns:
        Dict of filenames to patch content
    """
    patches = {}
    
    # Config patch
    patches["config_patch.py"] = """
# Add to DEFAULT_CONFIG
"predictive_performance_api": {
    "enabled": True,
    "port": 8500,
    "host": "0.0.0.0",
    "module": "test.api_server.predictive_performance_api_server",
    "args": []
},
"""
    
    # Gateway URL patch
    patches["gateway_url_patch.py"] = """
# Configure Predictive Performance API endpoint
PREDICTIVE_PERFORMANCE_API_URL = "http://{predictive_performance_api_host}:{predictive_performance_api_port}"
"""
    
    # Router patch
    patches["router_patch.py"] = """
# Predictive Performance API routes
@app.api_route("/api/predictive-performance{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"])
async def predictive_performance_api_route(request: Request, path: str):
    """Route requests to the Predictive Performance API."""
    url = f"{PREDICTIVE_PERFORMANCE_API_URL}/api/predictive-performance{path}"
    return await forward_request(url, request)

@app.websocket("/api/predictive-performance/ws/{task_id}")
async def predictive_performance_api_websocket(websocket: WebSocket, task_id: str):
    """WebSocket connection for the Predictive Performance API."""
    await websocket_forward(websocket, f"{PREDICTIVE_PERFORMANCE_API_URL}/api/predictive-performance/ws/{task_id}")
"""
    
    # Root endpoint patch
    patches["root_endpoint_patch.py"] = """
{"name": "Predictive Performance API", "url": f"{PREDICTIVE_PERFORMANCE_API_URL}/api/predictive-performance"},
"""
    
    # DB endpoint patch
    patches["db_endpoint_patch.py"] = """
{"name": "Predictive Performance Database API", "url": "/api/db/predictive-performance"},
"""
    
    # DB routes patch
    patches["db_routes_patch.py"] = """
# Database API routes - Predictive Performance
@app.api_route("/api/db/predictive-performance{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def predictive_performance_db_route(request: Request, path: str, api_key: str = Depends(get_api_key)):
    """
    Route requests to the Predictive Performance Database API.
    
    All database operations require API key authentication.
    """
    url = f"{PREDICTIVE_PERFORMANCE_API_URL}/api/predictive-performance/db{path}"
    return await forward_request(url, request)
"""
    
    # Cross-component patch
    patches["cross_component_patch.py"] = """
# Get Predictive Performance statistics
predictive_stats = await client.get(f"{PREDICTIVE_PERFORMANCE_API_URL}/api/predictive-performance/stats")
predictive_data = predictive_stats.json() if predictive_stats.status_code == 200 else {"error": "Failed to fetch predictive performance stats"}

# Add to response
response_data["predictive_stats"] = predictive_data
"""
    
    return patches