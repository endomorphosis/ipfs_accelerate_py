#!/usr/bin/env python3
"""
Unified API Server for IPFS Accelerate

This script launches a unified API server that combines the Test Suite API,
Generator API, and Benchmark API into a single cohesive interface.

The server provides a consistent API layer for all IPFS Accelerate components,
reducing code duplication and improving the user experience.
"""

import os
import sys
import logging
import argparse
import subprocess
import time
import signal
import atexit
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("unified_api_server.log")
    ]
)
logger = logging.getLogger("unified_api_server")

# Server configuration
DEFAULT_CONFIG = {
    "test_api": {
        "enabled": True,
        "port": 8000,
        "host": "0.0.0.0",
        "module": "refactored_test_suite.integration.test_api_integration",
        "args": ["--server"]
    },
    "generator_api": {
        "enabled": True,
        "port": 8001,
        "host": "0.0.0.0",
        "module": "refactored_generator_suite.generator_api_server",
        "args": []
    },
    "benchmark_api": {
        "enabled": True,
        "port": 8002,
        "host": "0.0.0.0",
        "module": "refactored_benchmark_suite.benchmark_api_server",
        "args": []
    },
    "gateway": {
        "enabled": True,
        "port": 8080,
        "host": "0.0.0.0"
    }
}

class ServiceManager:
    """Manager for handling the API services."""
    
    def __init__(self, config=None):
        """Initialize the service manager.
        
        Args:
            config: Optional configuration dictionary. If not provided, default config is used.
        """
        self.config = config or DEFAULT_CONFIG
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
    
    def start_service(self, service_name):
        """Start a specific service.
        
        Args:
            service_name: Name of the service to start (test_api, generator_api, benchmark_api, gateway)
            
        Returns:
            True if the service was started successfully, False otherwise
        """
        if service_name not in self.config:
            logger.error(f"Unknown service: {service_name}")
            return False
        
        service_config = self.config[service_name]
        if not service_config.get("enabled", True):
            logger.info(f"Service {service_name} is disabled in the configuration")
            return False
        
        # If service is already running, don't start it again
        if service_name in self.processes and self.processes[service_name] and self.processes[service_name].poll() is None:
            logger.info(f"Service {service_name} is already running")
            return True
        
        # If this is the gateway service
        if service_name == "gateway":
            return self._start_gateway_service(service_config)
        
        # For component APIs
        module = service_config.get("module")
        if not module:
            logger.error(f"No module specified for service {service_name}")
            return False
        
        # Prepare command
        host = service_config.get("host", "0.0.0.0")
        port = service_config.get("port", 8000)
        args = service_config.get("args", [])
        
        cmd = [
            sys.executable, "-m", module,
            "--host", str(host),
            "--port", str(port)
        ] + args
        
        logger.info(f"Starting {service_name} with command: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Store the process
            self.processes[service_name] = process
            
            # Start output monitoring
            self._start_output_monitoring(service_name, process)
            
            # Wait a moment to check if process started correctly
            time.sleep(1)
            if process.poll() is not None:
                logger.error(f"Service {service_name} failed to start. Return code: {process.poll()}")
                return False
            
            logger.info(f"Service {service_name} started on {host}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting service {service_name}: {e}")
            return False
    
    def _start_gateway_service(self, service_config):
        """Start the API gateway service.
        
        Args:
            service_config: Configuration for the gateway service
            
        Returns:
            True if the gateway was started successfully, False otherwise
        """
        try:
            # Check if FastAPI and uvicorn are available
            import fastapi
            import uvicorn
        except ImportError:
            logger.error("FastAPI or uvicorn not installed. Cannot start API gateway.")
            return False
        
        # Create a temporary gateway file
        gateway_file = Path("temp_api_gateway.py")
        self._generate_gateway_file(gateway_file, service_config)
        
        host = service_config.get("host", "0.0.0.0")
        port = service_config.get("port", 8080)
        
        cmd = [
            sys.executable, "-m", "uvicorn",
            "temp_api_gateway:app",
            "--host", str(host),
            "--port", str(port)
        ]
        
        logger.info(f"Starting API gateway with command: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Store the process
            self.processes["gateway"] = process
            
            # Start output monitoring
            self._start_output_monitoring("gateway", process)
            
            # Wait a moment to check if process started correctly
            time.sleep(1)
            if process.poll() is not None:
                logger.error(f"API gateway failed to start. Return code: {process.poll()}")
                return False
            
            logger.info(f"API gateway started on {host}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting API gateway: {e}")
            return False
    
    def _generate_gateway_file(self, file_path, service_config):
        """Generate a temporary gateway file.
        
        Args:
            file_path: Path to write the gateway file
            service_config: Configuration for the gateway service
        """
        # Get service configurations
        test_api_config = self.config.get("test_api", {})
        generator_api_config = self.config.get("generator_api", {})
        benchmark_api_config = self.config.get("benchmark_api", {})
        
        test_api_host = test_api_config.get("host", "0.0.0.0")
        test_api_port = test_api_config.get("port", 8000)
        generator_api_host = generator_api_config.get("host", "0.0.0.0")
        generator_api_port = generator_api_config.get("port", 8001)
        benchmark_api_host = benchmark_api_config.get("host", "0.0.0.0")
        benchmark_api_port = benchmark_api_config.get("port", 8002)
        
        # Create gateway file content
        content = f'''
#!/usr/bin/env python3
"""
API Gateway for IPFS Accelerate

This module provides a gateway for the unified API, routing requests to the appropriate
service based on the path. Includes support for database operations across components.
"""

import os
import json
import httpx
import logging
from typing import Dict, List, Any, Optional, Union
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_gateway")

# Create the API gateway
app = FastAPI(
    title="IPFS Accelerate API Gateway",
    description="Gateway for the unified IPFS Accelerate API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure service endpoints
TEST_API_URL = "http://{test_api_host}:{test_api_port}"
GENERATOR_API_URL = "http://{generator_api_host}:{generator_api_port}"
BENCHMARK_API_URL = "http://{benchmark_api_host}:{benchmark_api_port}"

# Define API key security
API_KEY_NAME = "X-API-Key"
API_KEY = APIKeyHeader(name=API_KEY_NAME)

# API key validation function
async def get_api_key(api_key: str = Depends(API_KEY)):
    """Validate API key for protected endpoints."""
    # In production, this would validate against a database or secure storage
    # For now, simply check if the key is provided
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key",
        )
    return api_key

# HTTP client for forwarding requests
client = httpx.AsyncClient()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    await client.aclose()

@app.get("/", include_in_schema=False)
async def read_root():
    """Root endpoint providing basic API information."""
    return {{
        "message": "IPFS Accelerate Unified API Gateway",
        "version": "1.0.0",
        "services": [
            {{"name": "Test API", "url": f"{{TEST_API_URL}}/api/test"}},
            {{"name": "Generator API", "url": f"{{GENERATOR_API_URL}}/api/generator"}},
            {{"name": "Benchmark API", "url": f"{{BENCHMARK_API_URL}}/api/benchmark"}}
        ],
        "database_endpoints": [
            {{"name": "Test Database API", "url": "/api/db/test"}},
            {{"name": "Generator Database API", "url": "/api/db/generator"}},
            {{"name": "Benchmark Database API", "url": "/api/db/benchmark"}}
        ],
        "docs_url": "/docs"
    }}

# Component API routes
@app.api_route("/api/test{{path:path}}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"])
async def test_api_route(request: Request, path: str):
    """Route requests to the Test API."""
    url = f"{{TEST_API_URL}}/api/test{{path}}"
    return await forward_request(url, request)

@app.api_route("/api/generator{{path:path}}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"])
async def generator_api_route(request: Request, path: str):
    """Route requests to the Generator API."""
    url = f"{{GENERATOR_API_URL}}/api/generator{{path}}"
    return await forward_request(url, request)

@app.api_route("/api/benchmark{{path:path}}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"])
async def benchmark_api_route(request: Request, path: str):
    """Route requests to the Benchmark API."""
    url = f"{{BENCHMARK_API_URL}}/api/benchmark{{path}}"
    return await forward_request(url, request)

# WebSocket routes
@app.websocket("/api/test/ws/{{task_id}}")
async def test_api_websocket(websocket: WebSocket, task_id: str):
    """WebSocket connection for the Test API."""
    await websocket_forward(websocket, f"{{TEST_API_URL}}/api/test/ws/{{task_id}}")

@app.websocket("/api/generator/ws/{{task_id}}")
async def generator_api_websocket(websocket: WebSocket, task_id: str):
    """WebSocket connection for the Generator API."""
    await websocket_forward(websocket, f"{{GENERATOR_API_URL}}/api/generator/ws/{{task_id}}")

@app.websocket("/api/benchmark/ws/{{task_id}}")
async def benchmark_api_websocket(websocket: WebSocket, task_id: str):
    """WebSocket connection for the Benchmark API."""
    await websocket_forward(websocket, f"{{BENCHMARK_API_URL}}/api/benchmark/ws/{{task_id}}")

# Database API routes - Test Suite
@app.api_route("/api/db/test{{path:path}}", methods=["GET", "POST", "PUT", "DELETE"])
async def test_db_route(request: Request, path: str, api_key: str = Depends(get_api_key)):
    """
    Route requests to the Test Suite Database API.
    
    All database operations require API key authentication.
    """
    url = f"{{TEST_API_URL}}/api/test/db{{path}}"
    return await forward_request(url, request)

# Database API routes - Generator
@app.api_route("/api/db/generator{{path:path}}", methods=["GET", "POST", "PUT", "DELETE"])
async def generator_db_route(request: Request, path: str, api_key: str = Depends(get_api_key)):
    """
    Route requests to the Generator Database API.
    
    All database operations require API key authentication.
    """
    url = f"{{GENERATOR_API_URL}}/api/generator/db{{path}}"
    return await forward_request(url, request)

# Database API routes - Benchmark
@app.api_route("/api/db/benchmark{{path:path}}", methods=["GET", "POST", "PUT", "DELETE"])
async def benchmark_db_route(request: Request, path: str, api_key: str = Depends(get_api_key)):
    """
    Route requests to the Benchmark Database API.
    
    All database operations require API key authentication.
    """
    url = f"{{BENCHMARK_API_URL}}/api/benchmark/db{{path}}"
    return await forward_request(url, request)

# Cross-component database operations
@app.get("/api/db/overview", dependencies=[Depends(get_api_key)])
async def get_db_overview():
    """
    Get a unified overview of all database components.
    
    Returns:
        Overview of all database components with statistics
    """
    try:
        # Gather data from all component databases
        test_stats = await client.get(f"{{TEST_API_URL}}/api/test/db/models/stats")
        generator_stats = await client.get(f"{{GENERATOR_API_URL}}/api/generator/db/models/stats")
        benchmark_stats = await client.get(f"{{BENCHMARK_API_URL}}/api/benchmark/db/models/stats")
        
        # Process responses
        test_data = test_stats.json() if test_stats.status_code == 200 else {{"error": "Failed to fetch test stats"}}
        generator_data = generator_stats.json() if generator_stats.status_code == 200 else {{"error": "Failed to fetch generator stats"}}
        benchmark_data = benchmark_stats.json() if benchmark_stats.status_code == 200 else {{"error": "Failed to fetch benchmark stats"}}
        
        # Combine data
        return {{
            "test_stats": test_data,
            "generator_stats": generator_data,
            "benchmark_stats": benchmark_data,
            "timestamp": import_datetime.datetime.now().isoformat()
        }}
    except Exception as e:
        logger.error(f"Error fetching database overview: {{e}}")
        return JSONResponse(
            status_code=500,
            content={{"error": f"Error fetching database overview: {{str(e)}}"}}
        )

@app.get("/api/db/model/{{model_name}}", dependencies=[Depends(get_api_key)])
async def get_model_unified_data(model_name: str):
    """
    Get unified data for a specific model across all components.
    
    Parameters:
        model_name: The name of the model to retrieve data for
        
    Returns:
        Unified view of the model data across test, generator, and benchmark components
    """
    try:
        # Gather model-specific data from all components
        async with httpx.AsyncClient() as client:
            # Use asyncio.gather to make concurrent requests
            import asyncio
            test_res, gen_res, bench_res = await asyncio.gather(
                client.get(f"{{TEST_API_URL}}/api/test/db/runs", params={{"model_name": model_name, "limit": 10}}),
                client.get(f"{{GENERATOR_API_URL}}/api/generator/db/tasks", params={{"model_name": model_name, "limit": 10}}),
                client.get(f"{{BENCHMARK_API_URL}}/api/benchmark/db/runs", params={{"model_name": model_name, "limit": 10}})
            )
        
        # Process responses
        test_data = test_res.json() if test_res.status_code == 200 else []
        generator_data = gen_res.json() if gen_res.status_code == 200 else []
        benchmark_data = bench_res.json() if bench_res.status_code == 200 else []
        
        # Calculate aggregate statistics
        total_test_runs = len(test_data)
        total_generator_tasks = len(generator_data)
        total_benchmark_runs = len(benchmark_data)
        
        # Calculate success rates
        test_success_rate = sum(1 for t in test_data if t.get('status') == 'completed') / max(total_test_runs, 1)
        generator_success_rate = sum(1 for t in generator_data if t.get('status') == 'completed') / max(total_generator_tasks, 1)
        benchmark_success_rate = sum(1 for t in benchmark_data if t.get('status') == 'completed') / max(total_benchmark_runs, 1)
        
        # Combine data
        return {{
            "model_name": model_name,
            "overview": {{
                "total_test_runs": total_test_runs,
                "total_generator_tasks": total_generator_tasks,
                "total_benchmark_runs": total_benchmark_runs,
                "test_success_rate": test_success_rate,
                "generator_success_rate": generator_success_rate,
                "benchmark_success_rate": benchmark_success_rate
            }},
            "recent_test_runs": test_data,
            "recent_generator_tasks": generator_data,
            "recent_benchmark_runs": benchmark_data,
            "timestamp": import_datetime.datetime.now().isoformat()
        }}
    except Exception as e:
        logger.error(f"Error fetching unified model data: {{e}}")
        return JSONResponse(
            status_code=500,
            content={{"error": f"Error fetching unified model data: {{str(e)}}"}}
        )

@app.post("/api/db/search", dependencies=[Depends(get_api_key)])
async def unified_search(query: str = Query(...), limit: int = Query(100, gt=0, le=1000)):
    """
    Search across all database components.
    
    Parameters:
        query: The search query
        limit: Maximum number of results per component
        
    Returns:
        Combined search results across all components
    """
    try:
        # Search across all component databases
        async with httpx.AsyncClient() as client:
            # Use asyncio.gather to make concurrent requests
            import asyncio
            test_res, gen_res, bench_res = await asyncio.gather(
                client.post(
                    f"{{TEST_API_URL}}/api/test/db/search", 
                    json={{"query": query, "limit": limit}}
                ),
                client.post(
                    f"{{GENERATOR_API_URL}}/api/generator/db/search", 
                    json={{"query": query, "limit": limit}}
                ),
                client.post(
                    f"{{BENCHMARK_API_URL}}/api/benchmark/db/search", 
                    json={{"query": query, "limit": limit}}
                )
            )
        
        # Process responses
        test_data = test_res.json() if test_res.status_code == 200 else []
        generator_data = gen_res.json() if gen_res.status_code == 200 else []
        benchmark_data = bench_res.json() if bench_res.status_code == 200 else []
        
        # Combine data
        return {{
            "query": query,
            "test_results": test_data,
            "generator_results": generator_data,
            "benchmark_results": benchmark_data,
            "result_counts": {{
                "test": len(test_data),
                "generator": len(generator_data),
                "benchmark": len(benchmark_data),
                "total": len(test_data) + len(generator_data) + len(benchmark_data)
            }},
            "timestamp": import_datetime.datetime.now().isoformat()
        }}
    except Exception as e:
        logger.error(f"Error performing unified search: {{e}}")
        return JSONResponse(
            status_code=500,
            content={{"error": f"Error performing unified search: {{str(e)}}"}}
        )

# Helper function to forward HTTP requests
async def forward_request(url: str, request: Request):
    """Forward an HTTP request to the specified URL.
    
    Args:
        url: The target URL
        request: The original request
        
    Returns:
        The forwarded response
    """
    try:
        # Get request body
        body = b""
        async for chunk in request.stream():
            body += chunk
        
        # Forward the request, keeping the original headers including api key
        response = await client.request(
            method=request.method,
            url=url,
            headers=[
                (k, v) for k, v in request.headers.items()
                if k.lower() not in ("host", "content-length")
            ],
            content=body,
            timeout=60.0
        )
        
        # Create response
        return StreamingResponse(
            content=response.aiter_bytes(),
            status_code=response.status_code,
            headers=dict(response.headers)
        )
    except Exception as e:
        logger.error(f"Error forwarding request to {url}: {e}")
        return JSONResponse(
            status_code=500,
            content={{"error": f"Error forwarding request: {{str(e)}}"}}
        )

# Helper function to forward WebSocket connections
async def websocket_forward(websocket: WebSocket, target_url: str):
    """Forward WebSocket connections.
    
    Args:
        websocket: The original WebSocket connection
        target_url: The target WebSocket URL
    """
    await websocket.accept()
    
    try:
        # Connect to the target WebSocket
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.websocket_connect(target_url) as ws:
                # Create tasks for bidirectional communication
                sender_task = None
                receiver_task = None
                
                # Forward messages from client to target
                async def sender():
                    try:
                        while True:
                            data = await websocket.receive_text()
                            await ws.send_text(data)
                    except WebSocketDisconnect:
                        pass
                    except Exception as e:
                        logger.error(f"Sender error: {{e}}")
                
                # Forward messages from target to client
                async def receiver():
                    try:
                        while True:
                            data = await ws.receive_text()
                            await websocket.send_text(data)
                    except WebSocketDisconnect:
                        pass
                    except Exception as e:
                        logger.error(f"Receiver error: {{e}}")
                
                import asyncio
                import datetime as import_datetime
                sender_task = asyncio.create_task(sender())
                receiver_task = asyncio.create_task(receiver())
                
                # Wait for either task to complete
                await asyncio.gather(sender_task, receiver_task, return_exceptions=True)
    except Exception as e:
        logger.error(f"WebSocket forward error: {{e}}")
        await websocket.close(code=1011, reason=f"Error connecting to target: {{str(e)}}")
'''
        
        # Write content to file
        with open(file_path, "w") as f:
            f.write(content)
        
        logger.info(f"Generated API gateway file at {file_path}")
    
    def _start_output_monitoring(self, service_name, process):
        """Start monitoring the output of a service process.
        
        Args:
            service_name: Name of the service
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
            args=(process.stdout, f"{service_name} (stdout)"),
            daemon=True
        )
        stdout_thread.start()
        
        # Start monitoring stderr
        stderr_thread = threading.Thread(
            target=monitor_output,
            args=(process.stderr, f"{service_name} (stderr)"),
            daemon=True
        )
        stderr_thread.start()
    
    def start_all(self):
        """Start all enabled services."""
        logger.info("Starting all services...")
        
        # Start component APIs
        for service_name in ["test_api", "generator_api", "benchmark_api"]:
            self.start_service(service_name)
        
        # Start API gateway
        self.start_service("gateway")
        
        self.running = True
        logger.info("All services started")
    
    def stop_service(self, service_name):
        """Stop a specific service.
        
        Args:
            service_name: Name of the service to stop
            
        Returns:
            True if the service was stopped successfully, False otherwise
        """
        if service_name not in self.processes or not self.processes[service_name]:
            logger.info(f"Service {service_name} is not running")
            return True
        
        process = self.processes[service_name]
        
        try:
            # Try to terminate the process gracefully
            logger.info(f"Stopping service {service_name}...")
            process.terminate()
            
            # Wait for the process to terminate
            for _ in range(10):  # Wait up to 5 seconds
                if process.poll() is not None:
                    break
                time.sleep(0.5)
            
            # If the process is still running, kill it
            if process.poll() is None:
                logger.warning(f"Service {service_name} did not terminate gracefully, killing...")
                process.kill()
                process.wait()
            
            logger.info(f"Service {service_name} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping service {service_name}: {e}")
            return False
    
    def stop_all(self):
        """Stop all running services."""
        if not self.running:
            return
        
        logger.info("Stopping all services...")
        
        # Stop in reverse order (gateway first, then component APIs)
        for service_name in ["gateway", "benchmark_api", "generator_api", "test_api"]:
            self.stop_service(service_name)
        
        # Clean up temp files
        if os.path.exists("temp_api_gateway.py"):
            try:
                os.remove("temp_api_gateway.py")
            except:
                pass
        
        self.running = False
        logger.info("All services stopped")
    
    def check_status(self):
        """Check the status of all services.
        
        Returns:
            Dict with service status information
        """
        status = {}
        
        for service_name in ["test_api", "generator_api", "benchmark_api", "gateway"]:
            if service_name in self.processes and self.processes[service_name]:
                process = self.processes[service_name]
                running = process.poll() is None
                status[service_name] = {
                    "running": running,
                    "exit_code": process.poll() if not running else None,
                    "host": self.config[service_name].get("host", "0.0.0.0"),
                    "port": self.config[service_name].get("port", 8000),
                    "url": f"http://{self.config[service_name].get('host', '0.0.0.0')}:{self.config[service_name].get('port', 8000)}"
                }
            else:
                status[service_name] = {
                    "running": False,
                    "exit_code": None,
                    "host": self.config[service_name].get("host", "0.0.0.0"),
                    "port": self.config[service_name].get("port", 8000),
                    "url": f"http://{self.config[service_name].get('host', '0.0.0.0')}:{self.config[service_name].get('port', 8000)}"
                }
        
        return status

def main():
    """Main entry point when run directly."""
    parser = argparse.ArgumentParser(description="Unified API Server for IPFS Accelerate")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--gateway-port", type=int, default=8080, help="Port for the API gateway")
    parser.add_argument("--test-api-port", type=int, default=8000, help="Port for the Test API")
    parser.add_argument("--generator-api-port", type=int, default=8001, help="Port for the Generator API")
    parser.add_argument("--benchmark-api-port", type=int, default=8002, help="Port for the Benchmark API")
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG
    
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config.update(json.load(f))
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
            return 1
    
    # Override ports from command line arguments
    config["gateway"]["port"] = args.gateway_port
    config["test_api"]["port"] = args.test_api_port
    config["generator_api"]["port"] = args.generator_api_port
    config["benchmark_api"]["port"] = args.benchmark_api_port
    
    # Create service manager
    manager = ServiceManager(config)
    
    try:
        # Start all services
        manager.start_all()
        
        # Show status
        status = manager.check_status()
        print("\nUnified API Server Status:")
        print("-------------------------")
        
        for service_name, service_status in status.items():
            status_text = "RUNNING" if service_status["running"] else "STOPPED"
            if not service_status["running"] and service_status["exit_code"] is not None:
                status_text += f" (Exit code: {service_status['exit_code']})"
                
            print(f"{service_name.upper()}: {status_text}")
            print(f"  URL: {service_status['url']}")
        
        print("\nAPI Gateway URL:")
        print(f"  http://localhost:{config['gateway']['port']}")
        print(f"  API Documentation: http://localhost:{config['gateway']['port']}/docs")
        print("\nPress Ctrl+C to stop the server...")
        
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