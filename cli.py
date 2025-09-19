#!/usr/bin/env python3
"""
IPFS Accelerate CLI Tool

This is the main CLI tool for IPFS Accelerate that provides a unified interface
for all functionality including MCP server management, inference operations,
file operations, and more.

Usage:
    ipfs-accelerate mcp start               # Start MCP server
    ipfs-accelerate mcp dashboard           # Start MCP server dashboard
    ipfs-accelerate mcp status              # Check MCP server status
    ipfs-accelerate inference generate      # Run text generation
    ipfs-accelerate files add               # Add files to IPFS
    ipfs-accelerate network status          # Check network status
    ipfs-accelerate models list             # List available models
    ipfs-accelerate --help                  # Show help for all commands
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import signal
import subprocess
import time
import webbrowser
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ipfs_accelerate_cli")

# Defer heavy imports until needed - global variables for lazy loading
HAVE_CORE = None
shared_core = None
inference_ops = None
file_ops = None
model_ops = None
network_ops = None
queue_ops = None
test_ops = None
IPFSAccelerateMCPServer = None

def _load_heavy_imports():
    """Load heavy imports only when needed for actual command execution"""
    global HAVE_CORE, shared_core, inference_ops, file_ops, model_ops, network_ops, queue_ops, test_ops, IPFSAccelerateMCPServer
    
    if HAVE_CORE is not None:
        return  # Already loaded
    
    try:
        from ipfs_accelerate_py.mcp.server import IPFSAccelerateMCPServer as _IPFSAccelerateMCPServer
        from shared import SharedCore, InferenceOperations, FileOperations, ModelOperations, NetworkOperations, QueueOperations, TestOperations
        
        IPFSAccelerateMCPServer = _IPFSAccelerateMCPServer
        HAVE_CORE = True
        
        # Initialize core components
        shared_core = SharedCore()
        inference_ops = InferenceOperations(shared_core)
        file_ops = FileOperations(shared_core)
        model_ops = ModelOperations(shared_core)
        network_ops = NetworkOperations(shared_core)
        queue_ops = QueueOperations(shared_core)
        test_ops = TestOperations(shared_core)
        
    except ImportError as e:
        logger.warning(f"Core modules not available: {e}")
        try:
            # Try alternative import paths
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from shared import SharedCore, InferenceOperations, FileOperations, ModelOperations, NetworkOperations, QueueOperations, TestOperations
            from ipfs_accelerate_py.mcp.server import IPFSAccelerateMCPServer as _IPFSAccelerateMCPServer
            
            IPFSAccelerateMCPServer = _IPFSAccelerateMCPServer
            HAVE_CORE = True
            
            # Initialize core components
            shared_core = SharedCore()
            inference_ops = InferenceOperations(shared_core)
            file_ops = FileOperations(shared_core)
            model_ops = ModelOperations(shared_core)
            network_ops = NetworkOperations(shared_core)
            queue_ops = QueueOperations(shared_core)
            test_ops = TestOperations(shared_core)
            
        except ImportError as e2:
            logger.warning(f"Alternative import also failed: {e2}")
            HAVE_CORE = False
            
            # Fallback shared core for when imports fail
            class SharedCore:
                def __init__(self):
                    pass
                def get_status(self):
                    return {"error": "Core not available", "fallback": True}
            
            shared_core = SharedCore()
            inference_ops = None
            file_ops = None
            model_ops = None
            network_ops = None
            queue_ops = None
            test_ops = None

class IPFSAccelerateCLI:
    """Main CLI class for IPFS Accelerate"""
    
    def __init__(self):
        self.mcp_process = None
        self.dashboard_process = None
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            self.cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.mcp_process:
            self.mcp_process.terminate()
            self.mcp_process = None
        if self.dashboard_process:
            self.dashboard_process.terminate()
            self.dashboard_process = None
    
    def run_mcp_start(self, args):
        """Start MCP server with integrated dashboard, model manager, and queue monitoring"""
        logger.info("Starting IPFS Accelerate MCP Server with integrated dashboard...")
        
        # Load heavy imports only when needed
        _load_heavy_imports()
        
        # Always enable dashboard integration
        args.dashboard = True
        
        try:
            # Start the integrated server with dashboard on the same port
            return self._start_integrated_mcp_server(args)
                
        except KeyboardInterrupt:
            logger.info("MCP server stopped by user")
            return 0
        except Exception as e:
            logger.error(f"Error starting MCP server: {e}")
            return 1
    
    def _start_integrated_mcp_server(self, args):
        """Start the integrated MCP server with dashboard, model manager, and queue monitoring"""
        import asyncio
        import threading
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json
        
        logger.info(f"Starting integrated MCP server on port {args.port}")
        logger.info("Integrated components: MCP Server, Web Dashboard, Model Manager, Queue Monitor")
        
        # Create the integrated dashboard handler
        class IntegratedMCPHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/' or self.path == '/dashboard':
                    self._serve_dashboard()
                elif self.path.startswith('/api/mcp/'):
                    self._handle_mcp_api()
                elif self.path.startswith('/api/models/'):
                    self._handle_model_api()
                elif self.path.startswith('/api/queue/'):
                    self._handle_queue_api()
                elif self.path.startswith('/static/'):
                    self._serve_static()
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def do_POST(self):
                if self.path.startswith('/api/'):
                    self._handle_post_api()
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def _serve_dashboard(self):
                """Serve the integrated dashboard"""
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                dashboard_html = self._get_integrated_dashboard_html()
                self.wfile.write(dashboard_html.encode())
            
            def _handle_mcp_api(self):
                """Handle MCP-related API calls"""
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                # Mock MCP status for now
                response = {
                    "status": "running",
                    "server": "IPFS Accelerate MCP",
                    "port": args.port,
                    "components": ["mcp_server", "dashboard", "model_manager", "queue_monitor"]
                }
                self.wfile.write(json.dumps(response).encode())
            
            def _handle_model_api(self):
                """Handle model manager API calls"""
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                # Try to get actual model data if model manager is available
                try:
                    if hasattr(self, '_model_manager') and self._model_manager:
                        models = self._model_manager.list_models()
                        response = {"models": [asdict(model) for model in models]}
                    else:
                        response = {"models": [], "status": "Model manager not initialized"}
                except Exception as e:
                    response = {"error": str(e), "models": []}
                
                self.wfile.write(json.dumps(response).encode())
            
            def _handle_queue_api(self):
                """Handle queue monitoring API calls"""
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                # Mock queue status
                response = {
                    "queue_status": "active",
                    "pending_jobs": 0,
                    "completed_jobs": 0,
                    "failed_jobs": 0,
                    "workers": 1
                }
                self.wfile.write(json.dumps(response).encode())
            
            def _serve_static(self):
                """Serve static files"""
                # For now, return a 404 - could be enhanced later
                self.send_response(404)
                self.end_headers()
            
            def _handle_post_api(self):
                """Handle POST API requests"""
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                response = {"status": "received", "message": "API endpoint not yet implemented"}
                self.wfile.write(json.dumps(response).encode())
            
            def _get_integrated_dashboard_html(self):
                """Get the integrated dashboard HTML"""
                return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IPFS Accelerate MCP Server Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
        }}
        .header {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        .header p {{
            color: #7f8c8d;
            font-size: 1.1rem;
        }}
        .status-bar {{
            background: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-around;
            align-items: center;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        }}
        .status-item {{
            text-align: center;
        }}
        .status-value {{
            font-size: 1.8rem;
            font-weight: bold;
            color: #27ae60;
        }}
        .status-label {{
            color: #7f8c8d;
            font-size: 0.9rem;
            margin-top: 5px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        }}
        .card h3 {{
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
        }}
        .metric:last-child {{
            border-bottom: none;
        }}
        .btn {{
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            margin: 5px;
            transition: transform 0.2s;
        }}
        .btn:hover {{
            transform: translateY(-2px);
        }}
        .log-container {{
            background: #2c3e50;
            color: #ecf0f1;
            border-radius: 8px;
            padding: 20px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9rem;
            max-height: 300px;
            overflow-y: auto;
        }}
        .online {{ color: #27ae60; }}
        .offline {{ color: #e74c3c; }}
        
        @media (max-width: 768px) {{
            .container {{ padding: 10px; }}
            .grid {{ grid-template-columns: 1fr; }}
            .status-bar {{ flex-direction: column; gap: 15px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 IPFS Accelerate MCP Server</h1>
            <p>Integrated Dashboard • Model Manager • Queue Monitor</p>
        </div>
        
        <div class="status-bar">
            <div class="status-item">
                <div class="status-value online" id="server-status">●</div>
                <div class="status-label">Server Status</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="port-number">{args.port}</div>
                <div class="status-label">Port</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="active-connections">1</div>
                <div class="status-label">Active Connections</div>
            </div>
            <div class="status-item">
                <div class="status-value" id="uptime">0s</div>
                <div class="status-label">Uptime</div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>🔧 MCP Server</h3>
                <div class="metric">
                    <span>Status:</span>
                    <span class="online">Running</span>
                </div>
                <div class="metric">
                    <span>Transport:</span>
                    <span>HTTP</span>
                </div>
                <div class="metric">
                    <span>Tools:</span>
                    <span id="tool-count">Loading...</span>
                </div>
                <button class="btn" onclick="refreshMCPStatus()">Refresh Status</button>
            </div>
            
            <div class="card">
                <h3>🤖 Model Manager</h3>
                <div class="metric">
                    <span>Available Models:</span>
                    <span id="model-count">Loading...</span>
                </div>
                <div class="metric">
                    <span>Model Types:</span>
                    <span>Text, Audio, Vision, Multimodal</span>
                </div>
                <div class="metric">
                    <span>Storage:</span>
                    <span>JSON Backend</span>
                </div>
                <button class="btn" onclick="refreshModels()">Refresh Models</button>
            </div>
            
            <div class="card">
                <h3>📊 Queue Monitor</h3>
                <div class="metric">
                    <span>Pending Jobs:</span>
                    <span id="pending-jobs">0</span>
                </div>
                <div class="metric">
                    <span>Completed:</span>
                    <span id="completed-jobs">0</span>
                </div>
                <div class="metric">
                    <span>Workers:</span>
                    <span id="active-workers">1</span>
                </div>
                <button class="btn" onclick="refreshQueue()">Refresh Queue</button>
            </div>
            
            <div class="card">
                <h3>📈 Performance</h3>
                <div class="metric">
                    <span>CPU Usage:</span>
                    <span id="cpu-usage">-</span>
                </div>
                <div class="metric">
                    <span>Memory:</span>
                    <span id="memory-usage">-</span>
                </div>
                <div class="metric">
                    <span>Requests/min:</span>
                    <span id="requests-per-min">0</span>
                </div>
                <button class="btn" onclick="showLogs()">View Logs</button>
            </div>
        </div>
        
        <div class="card">
            <h3>📝 System Logs</h3>
            <div class="log-container" id="logs">
                <div>🚀 IPFS Accelerate MCP Server started on port {args.port}</div>
                <div>✅ Integrated dashboard initialized</div>
                <div>🔧 Model manager ready</div>
                <div>📊 Queue monitor active</div>
                <div>🌐 Server accessible at http://{args.host}:{args.port}</div>
            </div>
        </div>
    </div>
    
    <script>
        let startTime = Date.now();
        
        function updateUptime() {{
            const now = Date.now();
            const uptimeMs = now - startTime;
            const uptimeSeconds = Math.floor(uptimeMs / 1000);
            
            let uptimeStr;
            if (uptimeSeconds < 60) {{
                uptimeStr = uptimeSeconds + 's';
            }} else if (uptimeSeconds < 3600) {{
                const minutes = Math.floor(uptimeSeconds / 60);
                uptimeStr = minutes + 'm';
            }} else {{
                const hours = Math.floor(uptimeSeconds / 3600);
                const minutes = Math.floor((uptimeSeconds % 3600) / 60);
                uptimeStr = hours + 'h ' + minutes + 'm';
            }}
            
            document.getElementById('uptime').textContent = uptimeStr;
        }}
        
        function refreshMCPStatus() {{
            fetch('/api/mcp/status')
                .then(response => response.json())
                .then(data => {{
                    console.log('MCP Status:', data);
                    addLog('MCP status refreshed: ' + data.status);
                }})
                .catch(error => {{
                    console.error('Error:', error);
                    addLog('Error refreshing MCP status');
                }});
        }}
        
        function refreshModels() {{
            fetch('/api/models/list')
                .then(response => response.json())
                .then(data => {{
                    const count = data.models ? data.models.length : 0;
                    document.getElementById('model-count').textContent = count;
                    addLog('Models refreshed: ' + count + ' available');
                }})
                .catch(error => {{
                    console.error('Error:', error);
                    document.getElementById('model-count').textContent = 'Error';
                    addLog('Error refreshing models');
                }});
        }}
        
        function refreshQueue() {{
            fetch('/api/queue/status')
                .then(response => response.json())
                .then(data => {{
                    document.getElementById('pending-jobs').textContent = data.pending_jobs || 0;
                    document.getElementById('completed-jobs').textContent = data.completed_jobs || 0;
                    document.getElementById('active-workers').textContent = data.workers || 1;
                    addLog('Queue status refreshed');
                }})
                .catch(error => {{
                    console.error('Error:', error);
                    addLog('Error refreshing queue status');
                }});
        }}
        
        function showLogs() {{
            const logContainer = document.getElementById('logs');
            logContainer.scrollTop = logContainer.scrollHeight;
        }}
        
        function addLog(message) {{
            const logContainer = document.getElementById('logs');
            const timestamp = new Date().toLocaleTimeString();
            const logEntry = document.createElement('div');
            logEntry.textContent = `[${{timestamp}}] ${{message}}`;
            logContainer.appendChild(logEntry);
            logContainer.scrollTop = logContainer.scrollHeight;
        }}
        
        // Initialize dashboard
        setInterval(updateUptime, 1000);
        
        // Auto-refresh components every 30 seconds
        setInterval(() => {{
            refreshMCPStatus();
            refreshModels();
            refreshQueue();
        }}, 30000);
        
        // Initial load
        setTimeout(() => {{
            refreshMCPStatus();
            refreshModels();
            refreshQueue();
        }}, 1000);
        
        addLog('Dashboard initialized successfully');
    </script>
</body>
</html>
                """
        
        # Start the integrated server
        server = HTTPServer((args.host, args.port), IntegratedMCPHandler)
        
        logger.info(f"Integrated MCP Server + Dashboard started at http://{args.host}:{args.port}")
        logger.info("Dashboard accessible at http://{args.host}:{args.port}/dashboard")
        
        if args.open_browser:
            import webbrowser
            webbrowser.open(f"http://{args.host}:{args.port}")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
            server.shutdown()
            return 0
        except Exception as e:
            logger.error(f"Server error: {e}")
            return 1
    
    def run_mcp_status(self, args):
        """Check MCP server status"""
        import requests
        
        try:
            url = f"http://{args.host}:{args.port}/health"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                status_data = response.json()
                if hasattr(args, 'output_json') and args.output_json:
                    print(json.dumps(status_data, indent=2))
                else:
                    print(f"✅ MCP Server Status: {status_data.get('status', 'Running')}")
                    print(f"   Server: http://{args.host}:{args.port}")
                    print(f"   Uptime: {status_data.get('uptime', 'Unknown')}")
                return 0
            else:
                if hasattr(args, 'output_json') and args.output_json:
                    print(json.dumps({"status": "error", "message": f"HTTP {response.status_code}"}))
                else:
                    print(f"❌ MCP Server Error: HTTP {response.status_code}")
                return 1
                
        except requests.exceptions.ConnectionError:
            if hasattr(args, 'output_json') and args.output_json:
                print(json.dumps({"status": "offline", "message": "Connection refused"}))
            else:
                print(f"❌ MCP Server Status: Offline (Connection refused)")
                print(f"   Attempted: http://{args.host}:{args.port}")
            return 1
        except Exception as e:
            logger.error(f"Error checking server status: {e}")
            if hasattr(args, 'output_json') and args.output_json:
                print(json.dumps({"status": "error", "message": str(e)}))
            else:
                print(f"❌ Error checking server status: {e}")
            return 1

    def run_mcp_dashboard(self, args):
        """Start MCP server dashboard with advanced features"""
        logger.info("Starting Advanced MCP Server Dashboard with HuggingFace Model Manager...")
        
        # Load heavy imports only when needed
        _load_heavy_imports()
        
        try:
            # Use the advanced dashboard with model manager
            logger.info("Using advanced dashboard with HuggingFace model manager and test fixtures")
            self._create_advanced_dashboard(args)
            
            # Keep the dashboard running
            if hasattr(args, 'keep_running') and args.keep_running:
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Dashboard stopped by user")
                    
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            logger.info("Falling back to simple dashboard")
            self._create_simple_dashboard(args)
    
    def _create_advanced_dashboard(self, args):
        """Create the advanced enterprise dashboard with HuggingFace model manager"""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import threading
            import json
            
            class AdvancedDashboardHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/':
                        # Serve the enhanced dashboard with embedded CSS
                        self.send_response(200)  
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        
                        # Use self-contained dashboard HTML instead of external template
                        html_content = self._get_self_contained_dashboard()
                        self.wfile.write(html_content.encode())
                    
                    elif self.path.startswith('/static/'):
                        # Serve static files (CSS, JS)
                        file_path = self.path[1:]  # Remove leading /
                        full_path = os.path.join(os.path.dirname(__file__), file_path)
                        
                        if os.path.exists(full_path):
                            self.send_response(200)
                            
                            # Set content type based on file extension
                            if file_path.endswith('.js'):
                                self.send_header('Content-type', 'application/javascript')
                            elif file_path.endswith('.css'):
                                self.send_header('Content-type', 'text/css')
                            else:
                                self.send_header('Content-type', 'text/plain')
                            
                            self.end_headers()
                            
                            with open(full_path, 'rb') as f:
                                self.wfile.write(f.read())
                        else:
                            self.send_response(404)
                            self.end_headers()
                            
                    elif self.path == '/api/status':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        
                        # Get comprehensive status including hardware info
                        status_data = shared_core.get_status()
                        
                        # Add hardware detection if available
                        try:
                            from .hardware_detection import HardwareDetector
                            hw_detector = HardwareDetector()
                            hardware_info = hw_detector.detect_all()
                            status_data['hardware'] = hardware_info
                        except Exception:
                            status_data['hardware'] = {"error": "Hardware detection not available"}
                        
                        self.wfile.write(json.dumps(status_data).encode())
                    
                    elif self.path == '/api/queue':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        
                        # Get queue status from shared operations
                        if queue_ops:
                            queue_data = queue_ops.get_queue_status()
                        else:
                            queue_data = {"error": "Queue operations not available"}
                        
                        self.wfile.write(json.dumps(queue_data).encode())
                    
                    elif self.path == '/api/models/search':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        
                        # Get model search results
                        models_data = self._get_model_search_results()
                        self.wfile.write(json.dumps(models_data).encode())
                    
                    elif self.path == '/jsonrpc':
                        # Handle JSON-RPC requests for advanced features
                        self._handle_jsonrpc_request()
                    
                    else:
                        self.send_response(404)
                        self.end_headers()
                        self.wfile.write(b'Not Found')
                
                def do_POST(self):
                    if self.path == '/jsonrpc':
                        self._handle_jsonrpc_request()
                    else:
                        self.send_response(404)
                        self.end_headers()
                
                def _handle_jsonrpc_request(self):
                    """Handle JSON-RPC requests for MCP compatibility"""
                    try:
                        content_length = int(self.headers.get('Content-Length', 0))
                        if content_length > 0:
                            request_data = self.rfile.read(content_length).decode('utf-8')
                            request_json = json.loads(request_data)
                        else:
                            request_json = {}
                        
                        # Process JSON-RPC request
                        response = self._process_jsonrpc(request_json)
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        
                        self.wfile.write(json.dumps(response).encode())
                        
                    except Exception as e:
                        logger.error(f"JSON-RPC error: {e}")
                        error_response = {
                            "jsonrpc": "2.0",
                            "error": {"code": -32603, "message": str(e)},
                            "id": None
                        }
                        
                        self.send_response(500)
                        self.send_header('Content-type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        
                        self.wfile.write(json.dumps(error_response).encode())
                
                def _process_jsonrpc(self, request):
                    """Process JSON-RPC requests using shared operations"""
                    method = request.get('method', '')
                    params = request.get('params', {})
                    request_id = request.get('id')
                    
                    try:
                        if method == 'searchModels':
                            query = params.get('query', '')
                            limit = params.get('limit', 50)
                            result = self._search_models(query, limit)
                        
                        elif method == 'getQueueStatus':
                            result = queue_ops.get_queue_status() if queue_ops else {"error": "Queue ops not available"}
                        
                        elif method == 'getModelDetails':
                            model_id = params.get('model_id', '')
                            result = self._get_model_details(model_id)
                        
                        elif method == 'runInference':
                            result = self._run_inference(params)
                        
                        elif method == 'runModelTest':
                            category = params.get('category', '')
                            test_type = params.get('test_type', '')
                            test_id = params.get('test_id', '')
                            result = test_ops.run_model_test(category, test_type, test_id) if test_ops else {"error": "Test ops not available"}
                        
                        elif method == 'runBatchTest':
                            batch_type = params.get('batch_type', '')
                            model_filter = params.get('model_filter', '')
                            test_id = params.get('test_id', '')
                            result = test_ops.run_batch_test(batch_type, model_filter, test_id) if test_ops else {"error": "Test ops not available"}
                        
                        else:
                            raise Exception(f"Unknown method: {method}")
                        
                        return {
                            "jsonrpc": "2.0",
                            "result": result,
                            "id": request_id
                        }
                        
                    except Exception as e:
                        return {
                            "jsonrpc": "2.0", 
                            "error": {"code": -32603, "message": str(e)},
                            "id": request_id
                        }
                
                def _search_models(self, query, limit):
                    """Search models using HuggingFace model manager if available"""
                    try:
                        # Try to use the HuggingFace model search
                        from .tools.huggingface_model_search import HuggingFaceModelSearch
                        
                        searcher = HuggingFaceModelSearch()
                        results = searcher.search(query, limit=limit)
                        
                        return {
                            "models": results,
                            "total": len(results),
                            "query": query,
                            "source": "huggingface"
                        }
                        
                    except Exception as e:
                        logger.warning(f"HuggingFace search not available: {e}")
                        
                        # Fallback to basic model list
                        if model_ops:
                            model_result = model_ops.list_models()
                            models = model_result.get('models', [])
                            
                            # Simple text search
                            if query:
                                filtered = []
                                query_lower = query.lower()
                                for model in models:
                                    model_text = json.dumps(model).lower()
                                    if query_lower in model_text:
                                        filtered.append(model)
                                models = filtered[:limit]
                            
                            return {
                                "models": models,
                                "total": len(models),
                                "query": query,
                                "source": "fallback"
                            }
                        
                        return {"models": [], "total": 0, "query": query, "source": "none"}
                
                def _get_model_details(self, model_id):
                    """Get detailed model information"""
                    if model_ops:
                        return model_ops.get_model_info(model_id)
                    return {"error": "Model operations not available"}
                
                def _run_inference(self, params):
                    """Run inference using shared operations"""
                    if inference_ops:
                        return inference_ops.run_text_generation(
                            model=params.get('model', 'gpt2'),
                            prompt=params.get('prompt', ''),
                            max_length=params.get('max_length', 100),
                            temperature=params.get('temperature', 0.7)
                        )
                    return {"error": "Inference operations not available"}
                
                def _get_model_search_results(self):
                    """Get model search results for API endpoint"""
                    return self._search_models("", 100)
                
                def _get_self_contained_dashboard(self):
                    """Generate unified self-contained dashboard HTML with embedded CSS/JS and MCP SDK integration"""
                    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IPFS Accelerate MCP Dashboard - Unified Platform</title>
    
    <style>
        /* Reset and base styles */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        :root {{
            --primary-color: #6366f1;
            --secondary-color: #8b5cf6;
            --accent-color: #06b6d4;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --error-color: #ef4444;
            --bg-color: #ffffff;
            --surface-color: #f8fafc;
            --text-color: #1e293b;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: var(--text-color);
            line-height: 1.6;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .dashboard-header {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: var(--shadow-lg);
            text-align: center;
        }}
        
        .dashboard-header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }}
        
        .dashboard-header .subtitle {{
            color: var(--text-muted);
            font-size: 1.1rem;
            margin-bottom: 20px;
        }}
        
        .status-indicator {{
            display: inline-block;
            padding: 8px 16px;
            background: var(--success-color);
            color: white;
            border-radius: 20px;
            font-weight: 600;
            margin-top: 15px;
        }}
        
        /* Tab Navigation */
        .tab-navigation {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 10px;
            margin-bottom: 30px;
            box-shadow: var(--shadow);
            display: flex;
            gap: 5px;
            overflow-x: auto;
        }}
        
        .tab-button {{
            padding: 12px 20px;
            border: none;
            background: transparent;
            color: var(--text-muted);
            border-radius: 10px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
            white-space: nowrap;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .tab-button:hover {{
            background: rgba(99, 102, 241, 0.1);
            color: var(--primary-color);
        }}
        
        .tab-button.active {{
            background: var(--primary-color);
            color: white;
            transform: translateY(-2px);
            box-shadow: var(--shadow);
        }}
        
        .tab-icon {{
            font-size: 1.2em;
        }}
        
        /* Main Content Area */
        .dashboard-content {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            box-shadow: var(--shadow-lg);
            min-height: 600px;
        }}
        
        .tab-pane {{
            display: none;
        }}
        
        .tab-pane.active {{
            display: block;
        }}
        
        /* Metrics Grid */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.05), rgba(139, 92, 246, 0.05));
            border: 1px solid rgba(99, 102, 241, 0.1);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            transition: all 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: var(--shadow-lg);
            border-color: var(--primary-color);
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 8px;
        }}
        
        .metric-label {{
            color: var(--text-muted);
            font-size: 0.9rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        /* Content Sections */
        .content-section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--text-color);
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .section-icon {{
            width: 24px;
            height: 24px;
            background: var(--primary-color);
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 14px;
        }}
        
        /* Action Cards */
        .action-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .action-card {{
            background: rgba(255, 255, 255, 0.7);
            border: 1px solid var(--border-color);
            border-radius: 15px;
            padding: 25px;
            transition: all 0.3s ease;
        }}
        
        .action-card:hover {{
            transform: translateY(-3px);
            box-shadow: var(--shadow-lg);
            border-color: var(--primary-color);
        }}
        
        .action-card h3 {{
            color: var(--text-color);
            font-size: 1.2rem;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .action-card p {{
            color: var(--text-muted);
            margin-bottom: 20px;
            line-height: 1.5;
        }}
        
        /* Buttons */
        .btn {{
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            text-decoration: none;
            font-size: 14px;
        }}
        
        .btn-primary {{
            background: var(--primary-color);
            color: white;
        }}
        
        .btn-primary:hover {{
            background: var(--secondary-color);
            transform: translateY(-2px);
            box-shadow: var(--shadow);
        }}
        
        .btn-secondary {{
            background: rgba(99, 102, 241, 0.1);
            color: var(--primary-color);
            border: 1px solid rgba(99, 102, 241, 0.2);
        }}
        
        .btn-secondary:hover {{
            background: rgba(99, 102, 241, 0.2);
            transform: translateY(-2px);
        }}
        
        .btn-success {{
            background: var(--success-color);
            color: white;
        }}
        
        .btn-success:hover {{
            background: #059669;
            transform: translateY(-2px);
        }}
        
        .btn-danger {{
            background: var(--error-color);
            color: white;
        }}
        
        .btn-danger:hover {{
            background: #dc2626;
            transform: translateY(-2px);
        }}
        
        .btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }}
        
        /* Form Elements */
        .form-group {{
            margin-bottom: 20px;
        }}
        
        .form-label {{
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: var(--text-color);
        }}
        
        .form-control {{
            width: 100%;
            padding: 12px 16px;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            font-size: 14px;
            transition: all 0.3s ease;
        }}
        
        .form-control:focus {{
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }}
        
        textarea.form-control {{
            resize: vertical;
            min-height: 100px;
        }}
        
        /* Results Display */
        .results-container {{
            background: rgba(248, 250, 252, 0.8);
            border: 1px solid var(--border-color);
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
            min-height: 200px;
        }}
        
        .results-title {{
            font-weight: 600;
            margin-bottom: 15px;
            color: var(--text-color);
        }}
        
        .result-item {{
            background: white;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
        }}
        
        .result-item:last-child {{
            margin-bottom: 0;
        }}
        
        /* Loading States */
        .loading {{
            display: inline-flex;
            align-items: center;
            gap: 10px;
        }}
        
        .spinner {{
            width: 20px;
            height: 20px;
            border: 2px solid var(--border-color);
            border-top: 2px solid var(--primary-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        /* Status Indicators */
        .status {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }}
        
        .status-running {{
            background: rgba(16, 185, 129, 0.1);
            color: var(--success-color);
        }}
        
        .status-error {{
            background: rgba(239, 68, 68, 0.1);
            color: var(--error-color);
        }}
        
        .status-warning {{
            background: rgba(245, 158, 11, 0.1);
            color: var(--warning-color);
        }}
        
        /* Test Grid */
        .test-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }}
        
        .test-category {{
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.05), rgba(139, 92, 246, 0.05));
            border: 1px solid rgba(99, 102, 241, 0.1);
            border-radius: 15px;
            padding: 25px;
            transition: all 0.3s ease;
        }}
        
        .test-category:hover {{
            transform: translateY(-3px);
            box-shadow: var(--shadow-lg);
            border-color: var(--primary-color);
        }}
        
        .test-category h3 {{
            font-size: 1.3rem;
            margin-bottom: 10px;
            color: var(--text-color);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .test-category-icon {{
            font-size: 1.5em;
        }}
        
        .test-category p {{
            color: var(--text-muted);
            margin-bottom: 20px;
            line-height: 1.5;
        }}
        
        .test-buttons {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        
        .test-btn {{
            padding: 8px 16px;
            background: var(--success-color);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .test-btn:hover {{
            background: #059669;
            transform: translateY(-1px);
        }}
        
        .test-btn:disabled {{
            background: var(--text-muted);
            cursor: not-allowed;
            transform: none;
        }}
        
        .test-btn.running {{
            background: var(--warning-color);
        }}
        
        /* Responsive Design */
        @media (max-width: 768px) {{
            .container {{
                padding: 15px;
            }}
            
            .dashboard-header {{
                padding: 20px;
            }}
            
            .dashboard-header h1 {{
                font-size: 2rem;
            }}
            
            .metrics-grid {{
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            }}
            
            .action-cards {{
                grid-template-columns: 1fr;
            }}
            
            .test-grid {{
                grid-template-columns: 1fr;
            }}
            
            .tab-navigation {{
                padding: 8px;
            }}
            
            .tab-button {{
                padding: 10px 16px;
                font-size: 14px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Dashboard Header -->
        <div class="dashboard-header">
            <h1>🚀 IPFS Accelerate MCP Dashboard</h1>
            <p class="subtitle">Unified AI Model Inference Platform with MCP Protocol Integration</p>
            <div class="status-indicator" id="serverStatus">Server Running</div>
        </div>
        
        <!-- Tab Navigation -->
        <div class="tab-navigation">
            <button class="tab-button active" data-tab="overview">
                <span class="tab-icon">📊</span>
                System Overview
            </button>
            <button class="tab-button" data-tab="inference">
                <span class="tab-icon">🤖</span>
                Text Generation
            </button>
            <button class="tab-button" data-tab="models">
                <span class="tab-icon">🎯</span>
                Model Manager
            </button>
            <button class="tab-button" data-tab="testing">
                <span class="tab-icon">🧪</span>
                Model Testing
            </button>
            <button class="tab-button" data-tab="queue">
                <span class="tab-icon">📈</span>
                Queue Monitor
            </button>
            <button class="tab-button" data-tab="network">
                <span class="tab-icon">🌐</span>
                Network Status
            </button>
            <button class="tab-button" data-tab="files">
                <span class="tab-icon">📁</span>
                File Manager
            </button>
        </div>
        
        <!-- Main Dashboard Content -->
        <div class="dashboard-content">
            <!-- System Overview Tab -->
            <div class="tab-pane active" id="overview">
                <!-- Live Metrics -->
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value" id="uptime">Loading...</div>
                        <div class="metric-label">Uptime</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="activeEndpoints">0</div>
                        <div class="metric-label">Active Endpoints</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="queueSize">0</div>
                        <div class="metric-label">Queue Size</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="processingTasks">0</div>
                        <div class="metric-label">Processing Tasks</div>
                    </div>
                </div>
                
                <!-- System Information -->
                <div class="content-section">
                    <h2 class="section-title">
                        <div class="section-icon">📋</div>
                        System Information
                    </h2>
                    <div class="action-cards">
                        <div class="action-card">
                            <h3>🏃‍♂️ Server Status</h3>
                            <p>MCP server is running and accepting connections via JSON-RPC protocol.</p>
                            <div id="serverInfo">
                                <p><strong>Status:</strong> <span class="status status-running">● Running</span></p>
                                <p><strong>Protocol:</strong> MCP (Model Context Protocol)</p>
                                <p><strong>Transport:</strong> HTTP JSON-RPC</p>
                                <p><strong>Started:</strong> <span id="startTime">Loading...</span></p>
                            </div>
                        </div>
                        <div class="action-card">
                            <h3>🔧 Available Tools</h3>
                            <p>MCP tools available through the JavaScript SDK for model inference and management.</p>
                            <div id="toolsList">
                                <p>• Text Generation & Analysis</p>
                                <p>• Model Search & Information</p>
                                <p>• Queue Management & Monitoring</p>
                                <p>• File Operations (IPFS)</p>
                                <p>• Network Status & Diagnostics</p>
                                <p>• Performance Testing & Validation</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Quick Actions -->
                <div class="content-section">
                    <h2 class="section-title">
                        <div class="section-icon">⚡</div>
                        Quick Actions
                    </h2>
                    <div class="action-cards">
                        <div class="action-card">
                            <h3>🔄 Refresh Data</h3>
                            <p>Update all dashboard metrics and system information.</p>
                            <button class="btn btn-primary" onclick="refreshDashboard()">
                                <span>🔄</span> Refresh Now
                            </button>
                        </div>
                        <div class="action-card">
                            <h3>📊 View Logs</h3>
                            <p>Access server logs and diagnostic information.</p>
                            <button class="btn btn-secondary" onclick="showLogs()">
                                <span>📋</span> View Logs
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Text Generation Tab -->
            <div class="tab-pane" id="inference">
                <div class="content-section">
                    <h2 class="section-title">
                        <div class="section-icon">🤖</div>
                        Advanced Text Generation
                    </h2>
                    <div class="action-card">
                        <h3>✨ Text Prompt</h3>
                        <div class="form-group">
                            <label class="form-label" for="promptInput">Enter your creative prompt here...</label>
                            <textarea class="form-control" id="promptInput" rows="4" placeholder="Write a story about AI and the future..."></textarea>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
                            <div class="form-group">
                                <label class="form-label" for="maxLength">Max Length</label>
                                <input type="number" class="form-control" id="maxLength" value="100" min="10" max="2048">
                            </div>
                            <div class="form-group">
                                <label class="form-label" for="temperature">Temperature</label>
                                <input type="number" class="form-control" id="temperature" value="0.7" min="0.1" max="2.0" step="0.1">
                            </div>
                        </div>
                        <button class="btn btn-primary" onclick="generateText()" id="generateBtn">
                            <span>✨</span> Generate Text
                        </button>
                        
                        <div class="results-container" id="textResults" style="display: none;">
                            <div class="results-title">Generated Text</div>
                            <div id="generatedContent"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Model Manager Tab -->
            <div class="tab-pane" id="models">
                <div class="content-section">
                    <h2 class="section-title">
                        <div class="section-icon">🎯</div>
                        HuggingFace Model Manager
                    </h2>
                    <div class="action-card">
                        <h3>🔍 Model Search</h3>
                        <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                            <input type="text" class="form-control" id="modelSearchInput" placeholder="Search models (e.g., 'text generation', 'bert', 'gpt')..." style="flex: 1;">
                            <button class="btn btn-primary" onclick="searchModels()">
                                <span>🔍</span> Search
                            </button>
                        </div>
                        
                        <div class="results-container" id="modelResults">
                            <div class="results-title">Popular Models</div>
                            <div id="modelList">
                                <p>Use the search above to find specific models, or browse popular models below.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Model Testing Tab -->
            <div class="tab-pane" id="testing">
                <div class="content-section">
                    <h2 class="section-title">
                        <div class="section-icon">🧪</div>
                        Model Testing & Validation
                    </h2>
                    <p style="margin-bottom: 30px; color: var(--text-muted);">
                        Test different categories of AI models with predefined test cases and validation scenarios.
                    </p>
                    
                    <div class="test-grid">
                        <!-- Text Generation Models -->
                        <div class="test-category">
                            <h3><span class="test-category-icon">📝</span> Text Generation Models</h3>
                            <p>Test language models with creative writing, code generation, and conversation prompts.</p>
                            <div class="test-buttons">
                                <button class="test-btn" onclick="runTest('text-generation', 'creative-writing')">Creative Writing Test</button>
                                <button class="test-btn" onclick="runTest('text-generation', 'code-generation')">Code Generation Test</button>
                                <button class="test-btn" onclick="runTest('text-generation', 'conversation')">Conversational Test</button>
                                <button class="test-btn" onclick="runTest('text-generation', 'summary')">Text Summary Test</button>
                            </div>
                        </div>
                        
                        <!-- Classification & Analysis -->
                        <div class="test-category">
                            <h3><span class="test-category-icon">🔍</span> Classification & Analysis</h3>
                            <p>Test models for sentiment analysis, text classification, and content analysis.</p>
                            <div class="test-buttons">
                                <button class="test-btn" onclick="runTest('classification', 'sentiment')">Sentiment Analysis Test</button>
                                <button class="test-btn" onclick="runTest('classification', 'topic')">Topic Classification Test</button>
                                <button class="test-btn" onclick="runTest('classification', 'language')">Language Detection Test</button>
                                <button class="test-btn" onclick="runTest('classification', 'safety')">Content Safety Test</button>
                            </div>
                        </div>
                        
                        <!-- Embedding Models -->
                        <div class="test-category">
                            <h3><span class="test-category-icon">🧮</span> Embedding Models</h3>
                            <p>Test vector embedding models for similarity, search, and semantic understanding.</p>
                            <div class="test-buttons">
                                <button class="test-btn" onclick="runTest('embedding', 'similarity')">Text Similarity Test</button>
                                <button class="test-btn" onclick="runTest('embedding', 'search')">Semantic Search Test</button>
                                <button class="test-btn" onclick="runTest('embedding', 'clustering')">Document Clustering Test</button>
                                <button class="test-btn" onclick="runTest('embedding', 'retrieval')">Information Retrieval Test</button>
                            </div>
                        </div>
                        
                        <!-- Code Generation Models -->
                        <div class="test-category">
                            <h3><span class="test-category-icon">💻</span> Code Generation Models</h3>
                            <p>Test specialized coding models for different programming languages and tasks.</p>
                            <div class="test-buttons">
                                <button class="test-btn" onclick="runTest('code', 'python')">Python Code Test</button>
                                <button class="test-btn" onclick="runTest('code', 'javascript')">JavaScript Code Test</button>
                                <button class="test-btn" onclick="runTest('code', 'sql')">SQL Query Test</button>
                                <button class="test-btn" onclick="runTest('code', 'debug')">Code Debug Test</button>
                            </div>
                        </div>
                        
                        <!-- Performance Testing -->
                        <div class="test-category">
                            <h3><span class="test-category-icon">⚡</span> Performance Testing</h3>
                            <p>Test model performance, latency, throughput, and resource utilization.</p>
                            <div class="test-buttons">
                                <button class="test-btn" onclick="runTest('performance', 'latency')">Latency Benchmark</button>
                                <button class="test-btn" onclick="runTest('performance', 'throughput')">Throughput Test</button>
                                <button class="test-btn" onclick="runTest('performance', 'memory')">Memory Usage Test</button>
                                <button class="test-btn" onclick="runTest('performance', 'concurrent')">Concurrent Load Test</button>
                            </div>
                        </div>
                        
                        <!-- Multimodal Models -->
                        <div class="test-category">
                            <h3><span class="test-category-icon">🎨</span> Multimodal Models</h3>
                            <p>Test models that handle multiple input types like text, images, and audio.</p>
                            <div class="test-buttons">
                                <button class="test-btn" onclick="runTest('multimodal', 'image-caption')">Image Captioning Test</button>
                                <button class="test-btn" onclick="runTest('multimodal', 'visual-qa')">Visual Q&A Test</button>
                                <button class="test-btn" onclick="runTest('multimodal', 'ocr')">OCR & Text Extract Test</button>
                                <button class="test-btn" onclick="runTest('multimodal', 'audio')">Audio Transcription Test</button>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Test Results -->
                    <div class="results-container" id="testResults" style="display: none;">
                        <div class="results-title">🧪 Test Results</div>
                        <p style="margin-bottom: 15px; color: var(--text-muted);">Select a test scenario above to start testing models</p>
                        <div id="testOutput">Test results will show model performance, accuracy metrics, and detailed analysis.</div>
                    </div>
                </div>
            </div>
            
            <!-- Queue Monitor Tab -->
            <div class="tab-pane" id="queue">
                <div class="content-section">
                    <h2 class="section-title">
                        <div class="section-icon">📈</div>
                        Queue Monitor & Analytics
                    </h2>
                    
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value" id="totalEndpoints">4</div>
                            <div class="metric-label">Total Endpoints</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" id="activeQueues">3</div>
                            <div class="metric-label">Active Queues</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" id="completedTasks">127</div>
                            <div class="metric-label">Completed Tasks</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value" id="avgProcessingTime">2.3s</div>
                            <div class="metric-label">Avg Processing Time</div>
                        </div>
                    </div>
                    
                    <div class="action-cards">
                        <div class="action-card">
                            <h3>📊 Queue Status</h3>
                            <p>View current queue status across all endpoints and model types.</p>
                            <button class="btn btn-primary" onclick="getQueueStatus()">
                                <span>📊</span> View Queue Status
                            </button>
                        </div>
                        <div class="action-card">
                            <h3>🎯 Model Queues</h3>
                            <p>Filter queue information by specific model types.</p>
                            <div style="display: flex; gap: 10px; margin-bottom: 15px;">
                                <select class="form-control" id="modelTypeFilter" style="flex: 1;">
                                    <option value="">All Model Types</option>
                                    <option value="text-generation">Text Generation</option>
                                    <option value="embedding">Embedding</option>
                                    <option value="classification">Classification</option>
                                    <option value="code-generation">Code Generation</option>
                                </select>
                                <button class="btn btn-secondary" onclick="getModelQueues()">
                                    <span>🎯</span> Filter
                                </button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="results-container" id="queueResults">
                        <div class="results-title">Queue Information</div>
                        <div id="queueData">
                            <p>Click the buttons above to view queue status and analytics.</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Network Status Tab -->
            <div class="tab-pane" id="network">
                <div class="content-section">
                    <h2 class="section-title">
                        <div class="section-icon">🌐</div>
                        Network Status & Diagnostics
                    </h2>
                    
                    <div class="action-cards">
                        <div class="action-card">
                            <h3>🌐 Network Overview</h3>
                            <p>Check IPFS network connectivity and peer status.</p>
                            <button class="btn btn-primary" onclick="getNetworkStatus()">
                                <span>🌐</span> Check Network
                            </button>
                        </div>
                        <div class="action-card">
                            <h3>👥 Peer Information</h3>
                            <p>View connected peers and distributed network topology.</p>
                            <button class="btn btn-secondary" onclick="getPeerInfo()">
                                <span>👥</span> View Peers
                            </button>
                        </div>
                    </div>
                    
                    <div class="results-container" id="networkResults">
                        <div class="results-title">Network Information</div>
                        <div id="networkData">
                            <p>Click the buttons above to check network status and peer information.</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- File Manager Tab -->
            <div class="tab-pane" id="files">
                <div class="content-section">
                    <h2 class="section-title">
                        <div class="section-icon">📁</div>
                        IPFS File Manager
                    </h2>
                    
                    <div class="action-cards">
                        <div class="action-card">
                            <h3>📄 File Operations</h3>
                            <p>Add files to IPFS network and manage content.</p>
                            <div class="form-group">
                                <label class="form-label" for="fileContent">File Content:</label>
                                <textarea class="form-control" id="fileContent" rows="4" placeholder="Enter file content or paste text here..."></textarea>
                            </div>
                            <div class="form-group">
                                <label class="form-label" for="fileName">File Name:</label>
                                <input type="text" class="form-control" id="fileName" placeholder="example.txt">
                            </div>
                            <button class="btn btn-primary" onclick="addFile()">
                                <span>📄</span> Add to IPFS
                            </button>
                        </div>
                        <div class="action-card">
                            <h3>📋 File Listing</h3>
                            <p>Browse and manage files stored in IPFS.</p>
                            <button class="btn btn-secondary" onclick="listFiles()">
                                <span>📋</span> List Files
                            </button>
                        </div>
                    </div>
                    
                    <div class="results-container" id="fileResults">
                        <div class="results-title">File Operations</div>
                        <div id="fileData">
                            <p>Use the forms above to add files to IPFS or browse existing content.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // MCP JSON-RPC Client Implementation
        class MCPClient {{
            constructor(endpoint = '/mcp') {{
                this.endpoint = endpoint;
                this.requestId = 1;
            }}
            
            async callTool(toolName, params = {{}}) {{
                try {{
                    const response = await fetch(this.endpoint, {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{
                            jsonrpc: '2.0',
                            method: 'call_tool',
                            params: {{
                                name: toolName,
                                arguments: params
                            }},
                            id: this.requestId++
                        }})
                    }});
                    
                    const data = await response.json();
                    
                    if (data.error) {{
                        throw new Error(data.error.message || 'MCP tool call failed');
                    }}
                    
                    return data.result;
                }} catch (error) {{
                    console.warn('MCP tool call failed, using fallback:', error);
                    return this.getFallbackResponse(toolName, params);
                }}
            }}
            
            getFallbackResponse(toolName, params) {{
                // Provide realistic fallback responses
                const fallbacks = {{
                    'run_inference': {{
                        success: true,
                        generated_text: 'This is a simulated text generation response. The actual MCP server would provide real AI-generated content based on your prompt.',
                        model: 'fallback-model',
                        processing_time: Math.random() * 2 + 1,
                        fallback: true
                    }},
                    'search_models': {{
                        success: true,
                        models: [
                            {{ id: 'gpt2', name: 'GPT-2', type: 'text-generation', downloads: 45000000, description: 'A transformer-based language model' }},
                            {{ id: 'bert-base-uncased', name: 'BERT Base', type: 'fill-mask', downloads: 15000000, description: 'Bidirectional transformer for language understanding' }},
                            {{ id: 'distilbert-base-uncased', name: 'DistilBERT', type: 'classification', downloads: 8000000, description: 'Lightweight version of BERT' }}
                        ],
                        total: 3,
                        fallback: true
                    }},
                    'get_queue_status': {{
                        success: true,
                        summary: {{
                            total_endpoints: 4,
                            active_endpoints: 3,
                            total_queue_size: 8,
                            processing_tasks: 3
                        }},
                        endpoints: [
                            {{ id: 'local_gpu_1', status: 'active', queue_size: 3, model_type: 'text-generation' }},
                            {{ id: 'distributed_peer_1', status: 'active', queue_size: 2, model_type: 'embedding' }},
                            {{ id: 'cloud_endpoint_1', status: 'active', queue_size: 3, model_type: 'classification' }}
                        ],
                        fallback: true
                    }},
                    'get_network_status': {{
                        success: true,
                        status: 'connected',
                        peers: 12,
                        network_info: {{
                            peer_id: 'QmFallbackPeerId123...',
                            addresses: ['/ip4/127.0.0.1/tcp/4001'],
                            protocol_version: 'ipfs/0.1.0'
                        }},
                        fallback: true
                    }},
                    'add_file': {{
                        success: true,
                        cid: 'QmFallbackCid' + Math.random().toString(36).substr(2, 9),
                        size: params.content ? params.content.length : 100,
                        name: params.filename || 'fallback-file.txt',
                        fallback: true
                    }},
                    'run_model_test': {{
                        success: true,
                        test_type: params.test_type || 'unknown',
                        test_name: params.test_name || 'unknown',
                        results: {{
                            accuracy: Math.random() * 0.3 + 0.7, // 70-100%
                            latency: Math.random() * 2 + 0.5, // 0.5-2.5s
                            throughput: Math.random() * 50 + 10, // 10-60 tokens/sec
                            success_rate: Math.random() * 0.2 + 0.8 // 80-100%
                        }},
                        details: 'This is a simulated test result. The actual MCP server would run comprehensive model validation.',
                        fallback: true
                    }}
                }};
                
                return fallbacks[toolName] || {{ error: 'Tool not found', fallback: true }};
            }}
        }}
        
        // Initialize MCP client
        const mcpClient = new MCPClient();
        
        // Dashboard State Management
        let dashboardData = {{
            uptime: 0,
            startTime: new Date(),
            activeEndpoints: 4,
            queueSize: 8,
            processingTasks: 3
        }};
        
        // Tab Management
        function initializeTabs() {{
            const tabButtons = document.querySelectorAll('.tab-button');
            const tabPanes = document.querySelectorAll('.tab-pane');
            
            tabButtons.forEach(button => {{
                button.addEventListener('click', () => {{
                    const targetTab = button.getAttribute('data-tab');
                    
                    // Update button states
                    tabButtons.forEach(btn => btn.classList.remove('active'));
                    button.classList.add('active');
                    
                    // Update pane visibility
                    tabPanes.forEach(pane => {{
                        pane.classList.remove('active');
                        if (pane.id === targetTab) {{
                            pane.classList.add('active');
                        }}
                    }});
                }});
            }});
        }}
        
        // Dashboard Functions
        async function refreshDashboard() {{
            console.log('Refreshing dashboard data...');
            updateMetrics();
            
            // Show loading state
            const refreshBtn = document.querySelector('button[onclick="refreshDashboard()"]');
            if (refreshBtn) {{
                const originalText = refreshBtn.innerHTML;
                refreshBtn.innerHTML = '<div class="spinner"></div> Refreshing...';
                refreshBtn.disabled = true;
                
                setTimeout(() => {{
                    refreshBtn.innerHTML = originalText;
                    refreshBtn.disabled = false;
                }}, 2000);
            }}
        }}
        
        function updateMetrics() {{
            dashboardData.uptime += 1;
            
            document.getElementById('uptime').textContent = formatUptime(dashboardData.uptime);
            document.getElementById('activeEndpoints').textContent = dashboardData.activeEndpoints;
            document.getElementById('queueSize').textContent = dashboardData.queueSize;
            document.getElementById('processingTasks').textContent = dashboardData.processingTasks;
            document.getElementById('startTime').textContent = dashboardData.startTime.toLocaleString();
        }}
        
        function formatUptime(seconds) {{
            if (seconds < 60) return seconds + 's';
            if (seconds < 3600) return Math.floor(seconds / 60) + 'm';
            if (seconds < 86400) return Math.floor(seconds / 3600) + 'h';
            return Math.floor(seconds / 86400) + 'd';
        }}
        
        // Text Generation
        async function generateText() {{
            const prompt = document.getElementById('promptInput').value;
            const maxLength = document.getElementById('maxLength').value;
            const temperature = document.getElementById('temperature').value;
            const generateBtn = document.getElementById('generateBtn');
            const resultsContainer = document.getElementById('textResults');
            const contentDiv = document.getElementById('generatedContent');
            
            if (!prompt.trim()) {{
                alert('Please enter a prompt');
                return;
            }}
            
            // Show loading state
            generateBtn.innerHTML = '<div class="spinner"></div> Generating...';
            generateBtn.disabled = true;
            
            try {{
                const result = await mcpClient.callTool('run_inference', {{
                    prompt: prompt,
                    max_length: parseInt(maxLength),
                    temperature: parseFloat(temperature)
                }});
                
                resultsContainer.style.display = 'block';
                contentDiv.innerHTML = `
                    <div class="result-item">
                        <p><strong>Generated Text:</strong></p>
                        <p style="font-style: italic; line-height: 1.6;">${{result.generated_text}}</p>
                        <div style="margin-top: 15px; font-size: 12px; color: var(--text-muted);">
                            <span>Model: ${{result.model || 'default'}}</span> • 
                            <span>Processing Time: ${{result.processing_time ? result.processing_time.toFixed(2) + 's' : 'N/A'}}</span>
                            ${{result.fallback ? ' • <span style="color: var(--warning-color);">Fallback Mode</span>' : ''}}
                        </div>
                    </div>
                `;
            }} catch (error) {{
                contentDiv.innerHTML = `
                    <div class="result-item" style="border-color: var(--error-color);">
                        <p style="color: var(--error-color);"><strong>Error:</strong> ${{error.message}}</p>
                    </div>
                `;
                resultsContainer.style.display = 'block';
            }}
            
            // Reset button
            generateBtn.innerHTML = '<span>✨</span> Generate Text';
            generateBtn.disabled = false;
        }}
        
        // Model Search
        async function searchModels() {{
            const query = document.getElementById('modelSearchInput').value;
            const resultsContainer = document.getElementById('modelResults');
            const listDiv = document.getElementById('modelList');
            
            if (!query.trim()) {{
                alert('Please enter a search query');
                return;
            }}
            
            try {{
                const result = await mcpClient.callTool('search_models', {{
                    query: query,
                    limit: 10
                }});
                
                if (result.success && result.models.length > 0) {{
                    listDiv.innerHTML = result.models.map(model => `
                        <div class="result-item">
                            <h4>${{model.name}} <span style="font-size: 0.8em; color: var(--text-muted);">(${{model.id}})</span></h4>
                            <p><strong>Type:</strong> ${{model.type}}</p>
                            <p><strong>Downloads:</strong> ${{model.downloads.toLocaleString()}}</p>
                            <p>${{model.description}}</p>
                            ${{result.fallback ? '<p style="color: var(--warning-color); font-size: 0.9em;">Using fallback data</p>' : ''}}
                        </div>
                    `).join('');
                }} else {{
                    listDiv.innerHTML = '<p>No models found for your search query.</p>';
                }}
            }} catch (error) {{
                listDiv.innerHTML = `<p style="color: var(--error-color);">Error searching models: ${{error.message}}</p>`;
            }}
        }}
        
        // Model Testing
        async function runTest(category, testName) {{
            const testBtn = event.target;
            const resultsContainer = document.getElementById('testResults');
            const outputDiv = document.getElementById('testOutput');
            
            // Show loading state
            testBtn.classList.add('running');
            testBtn.innerHTML = '<div class="spinner"></div> Running...';
            testBtn.disabled = true;
            
            try {{
                const result = await mcpClient.callTool('run_model_test', {{
                    test_type: category,
                    test_name: testName
                }});
                
                resultsContainer.style.display = 'block';
                outputDiv.innerHTML = `
                    <div class="result-item">
                        <h4>${{category.charAt(0).toUpperCase() + category.slice(1)}} - ${{testName.replace('-', ' ').replace(/\\b\\w/g, l => l.toUpperCase())}}</h4>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0;">
                            <div><strong>Accuracy:</strong> ${{(result.results.accuracy * 100).toFixed(1)}}%</div>
                            <div><strong>Latency:</strong> ${{result.results.latency.toFixed(2)}}s</div>
                            <div><strong>Throughput:</strong> ${{result.results.throughput.toFixed(1)}} tokens/sec</div>
                            <div><strong>Success Rate:</strong> ${{(result.results.success_rate * 100).toFixed(1)}}%</div>
                        </div>
                        <p>${{result.details}}</p>
                        ${{result.fallback ? '<p style="color: var(--warning-color); font-size: 0.9em; margin-top: 10px;">Using simulated test results</p>' : ''}}
                    </div>
                `;
            }} catch (error) {{
                outputDiv.innerHTML = `
                    <div class="result-item" style="border-color: var(--error-color);">
                        <p style="color: var(--error-color);"><strong>Test Failed:</strong> ${{error.message}}</p>
                    </div>
                `;
                resultsContainer.style.display = 'block';
            }}
            
            // Reset button
            testBtn.classList.remove('running');
            testBtn.innerHTML = testBtn.textContent;
            testBtn.disabled = false;
        }}
        
        // Queue Management
        async function getQueueStatus() {{
            const resultsContainer = document.getElementById('queueResults');
            const dataDiv = document.getElementById('queueData');
            
            try {{
                const result = await mcpClient.callTool('get_queue_status');
                
                if (result.success) {{
                    dataDiv.innerHTML = `
                        <div class="result-item">
                            <h4>Queue Summary</h4>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0;">
                                <div><strong>Total Endpoints:</strong> ${{result.summary.total_endpoints}}</div>
                                <div><strong>Active Endpoints:</strong> ${{result.summary.active_endpoints}}</div>
                                <div><strong>Queue Size:</strong> ${{result.summary.total_queue_size}}</div>
                                <div><strong>Processing Tasks:</strong> ${{result.summary.processing_tasks}}</div>
                            </div>
                        </div>
                        ${{result.endpoints.map(endpoint => `
                            <div class="result-item">
                                <h5>${{endpoint.id}}</h5>
                                <p><strong>Status:</strong> <span class="status status-${{endpoint.status === 'active' ? 'running' : 'warning'}}">● ${{endpoint.status}}</span></p>
                                <p><strong>Queue Size:</strong> ${{endpoint.queue_size}}</p>
                                <p><strong>Model Type:</strong> ${{endpoint.model_type}}</p>
                            </div>
                        `).join('')}}
                        ${{result.fallback ? '<p style="color: var(--warning-color); margin-top: 15px;">Using simulated queue data</p>' : ''}}
                    `;
                }}
            }} catch (error) {{
                dataDiv.innerHTML = `<p style="color: var(--error-color);">Error getting queue status: ${{error.message}}</p>`;
            }}
        }}
        
        async function getModelQueues() {{
            const modelType = document.getElementById('modelTypeFilter').value;
            const resultsContainer = document.getElementById('queueResults');
            const dataDiv = document.getElementById('queueData');
            
            try {{
                const result = await mcpClient.callTool('get_model_queues', {{
                    model_type: modelType || undefined
                }});
                
                if (result.success) {{
                    dataDiv.innerHTML = `
                        <div class="result-item">
                            <h4>Model Queues ${{modelType ? `(${{{modelType}}})` : '(All Types)'}}</h4>
                            <p>Filtered queue information based on model type selection.</p>
                        </div>
                    `;
                }}
            }} catch (error) {{
                dataDiv.innerHTML = `<p style="color: var(--error-color);">Error getting model queues: ${{error.message}}</p>`;
            }}
        }}
        
        // Network Status
        async function getNetworkStatus() {{
            const resultsContainer = document.getElementById('networkResults');
            const dataDiv = document.getElementById('networkData');
            
            try {{
                const result = await mcpClient.callTool('get_network_status');
                
                if (result.success) {{
                    dataDiv.innerHTML = `
                        <div class="result-item">
                            <h4>Network Status</h4>
                            <p><strong>Status:</strong> <span class="status status-${{result.status === 'connected' ? 'running' : 'warning'}}">● ${{result.status}}</span></p>
                            <p><strong>Connected Peers:</strong> ${{result.peers}}</p>
                            <p><strong>Peer ID:</strong> ${{result.network_info.peer_id}}</p>
                            ${{result.fallback ? '<p style="color: var(--warning-color); margin-top: 10px;">Using simulated network data</p>' : ''}}
                        </div>
                    `;
                }}
            }} catch (error) {{
                dataDiv.innerHTML = `<p style="color: var(--error-color);">Error getting network status: ${{error.message}}</p>`;
            }}
        }}
        
        async function getPeerInfo() {{
            const resultsContainer = document.getElementById('networkResults');
            const dataDiv = document.getElementById('networkData');
            
            dataDiv.innerHTML = `
                <div class="result-item">
                    <h4>Peer Information</h4>
                    <p>Detailed peer information and network topology would be displayed here.</p>
                    <p style="color: var(--text-muted);">This feature uses the MCP network tools to gather peer data.</p>
                </div>
            `;
        }}
        
        // File Operations
        async function addFile() {{
            const content = document.getElementById('fileContent').value;
            const filename = document.getElementById('fileName').value;
            const resultsContainer = document.getElementById('fileResults');
            const dataDiv = document.getElementById('fileData');
            
            if (!content.trim()) {{
                alert('Please enter file content');
                return;
            }}
            
            try {{
                const result = await mcpClient.callTool('add_file', {{
                    content: content,
                    filename: filename || 'untitled.txt'
                }});
                
                if (result.success) {{
                    dataDiv.innerHTML = `
                        <div class="result-item">
                            <h4>File Added Successfully</h4>
                            <p><strong>CID:</strong> ${{result.cid}}</p>
                            <p><strong>Name:</strong> ${{result.name}}</p>
                            <p><strong>Size:</strong> ${{result.size}} bytes</p>
                            ${{result.fallback ? '<p style="color: var(--warning-color); margin-top: 10px;">Using simulated file operation</p>' : ''}}
                        </div>
                    `;
                }}
            }} catch (error) {{
                dataDiv.innerHTML = `<p style="color: var(--error-color);">Error adding file: ${{error.message}}</p>`;
            }}
        }}
        
        async function listFiles() {{
            const resultsContainer = document.getElementById('fileResults');
            const dataDiv = document.getElementById('fileData');
            
            dataDiv.innerHTML = `
                <div class="result-item">
                    <h4>IPFS File Listing</h4>
                    <p>File listing functionality would display files stored in the IPFS network.</p>
                    <p style="color: var(--text-muted);">This feature uses MCP file tools to browse IPFS content.</p>
                </div>
            `;
        }}
        
        function showLogs() {{
            alert('Log viewing functionality would open server logs and diagnostic information via MCP tools.');
        }}
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {{
            initializeTabs();
            updateMetrics();
            
            // Auto-refresh metrics every 30 seconds
            setInterval(updateMetrics, 30000);
            
                        console.log('IPFS Accelerate MCP Dashboard initialized');
                        console.log('Using JavaScript MCP SDK for tool integration');
                    }});
                </script>
            </body>
            </html>
            """
            
            # Start the dashboard server
            port = getattr(args, 'port', 8001)
            host = getattr(args, 'host', 'localhost')
            
            server = HTTPServer((host, port), AdvancedDashboardHandler)
            
            logger.info(f"Advanced MCP Dashboard started at http://{host}:{port}")
            logger.info("Dashboard features: HuggingFace model manager, queue monitoring, model testing")
            
            if getattr(args, 'open_browser', False):
                import webbrowser
                webbrowser.open(f"http://{host}:{port}")
            
            # Start server in a separate thread to keep it non-blocking
            import threading
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            
            return server
            
        except Exception as e:
            logger.error(f"Error creating advanced dashboard: {e}")
            raise


def main():
    """Main entry point for the CLI"""
    try:
        # Create argument parser
        parser = argparse.ArgumentParser(
            description="IPFS Accelerate CLI - Unified interface for AI inference and IPFS operations",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  ipfs-accelerate mcp start --dashboard --open-browser
  ipfs-accelerate mcp status
  ipfs-accelerate inference generate --prompt "Hello world"
  ipfs-accelerate models list --output-json
  ipfs-accelerate queue status
  ipfs-accelerate network status
            """
        )
        
        # Add global arguments
        parser.add_argument('--output-json', action='store_true', help='Output results in JSON format')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        
        # Create subparsers for different command categories
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # MCP commands
        mcp_parser = subparsers.add_parser('mcp', help='MCP server management')
        mcp_subparsers = mcp_parser.add_subparsers(dest='mcp_command', help='MCP commands')
        
        # MCP start command
        start_parser = mcp_subparsers.add_parser('start', help='Start MCP server')
        start_parser.add_argument('--name', default='ipfs-accelerate', help='Server name')
        start_parser.add_argument('--host', default='localhost', help='Host to bind to')
        start_parser.add_argument('--port', type=int, default=3000, help='Port to bind to (default: 3000)')
        start_parser.add_argument('--dashboard', action='store_true', help='Enable web dashboard')
        start_parser.add_argument('--open-browser', action='store_true', help='Open browser automatically')
        start_parser.add_argument('--keep-running', action='store_true', help='Keep server running')
        
        # MCP dashboard command
        dashboard_parser = mcp_subparsers.add_parser('dashboard', help='Start dashboard only')
        dashboard_parser.add_argument('--host', default='localhost', help='Host to bind to')
        dashboard_parser.add_argument('--port', type=int, default=8001, help='Port to bind to')
        dashboard_parser.add_argument('--open-browser', action='store_true', help='Open browser automatically')
        
        # MCP status command
        status_parser = mcp_subparsers.add_parser('status', help='Check MCP server status')
        status_parser.add_argument('--host', default='localhost', help='Server host')
        status_parser.add_argument('--port', type=int, default=3000, help='Server port (default: 3000)')
        
        # Parse arguments
        args = parser.parse_args()
        
        # Set debug logging if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        
        # Handle commands
        if not args.command:
            parser.print_help()
            return 0
            
        cli = IPFSAccelerateCLI()
        
        if args.command == 'mcp':
            if args.mcp_command == 'start':
                return cli.run_mcp_start(args)
            elif args.mcp_command == 'dashboard':
                return cli.run_mcp_dashboard(args)
            elif args.mcp_command == 'status':
                return cli.run_mcp_status(args)
            else:
                mcp_parser.print_help()
                return 1
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        logger.info("CLI interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"CLI error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
