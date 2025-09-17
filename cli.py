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

# Import shared functionality
try:
    from ipfs_accelerate_py.mcp.server import IPFSAccelerateMCPServer
    from .shared import SharedCore, InferenceOperations, FileOperations, ModelOperations, NetworkOperations, QueueOperations
    HAVE_CORE = True
except ImportError as e:
    logger.warning(f"Core modules not available: {e}")
    try:
        # Try alternative import paths
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from shared import SharedCore, InferenceOperations, FileOperations, ModelOperations, NetworkOperations, QueueOperations
        from ipfs_accelerate_py.mcp.server import IPFSAccelerateMCPServer
        HAVE_CORE = True
    except ImportError as e2:
        logger.warning(f"Alternative import also failed: {e2}")
        HAVE_CORE = False
        
        # Fallback shared core for when imports fail
        class SharedCore:
            def __init__(self):
                pass
            def get_status(self):
                return {"error": "Core not available", "fallback": True}

# Global shared core instance
if HAVE_CORE:
    shared_core = SharedCore()
    inference_ops = InferenceOperations(shared_core)
    file_ops = FileOperations(shared_core)
    model_ops = ModelOperations(shared_core)
    network_ops = NetworkOperations(shared_core)
    queue_ops = QueueOperations(shared_core)
else:
    shared_core = SharedCore()
    inference_ops = None
    file_ops = None
    model_ops = None
    network_ops = None
    queue_ops = None

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
        """Start MCP server"""
        logger.info("Starting IPFS Accelerate MCP Server...")
        
        try:
            if HAVE_CORE:
                # Use the built-in MCP server
                server = IPFSAccelerateMCPServer(
                    name=args.name,
                    host=args.host,
                    port=args.port,
                    debug=args.debug
                )
                
                server.setup()
                logger.info(f"MCP Server started at http://{args.host}:{args.port}")
                
                if args.dashboard:
                    # Also start dashboard in a separate process
                    self.run_mcp_dashboard(args)
                
                # Run the server
                server.run()
            else:
                logger.error("MCP server core not available")
                return 1
                
        except KeyboardInterrupt:
            logger.info("MCP server stopped by user")
            return 0
        except Exception as e:
            logger.error(f"Error starting MCP server: {e}")
            return 1
    
    def run_mcp_dashboard(self, args):
        """Start MCP server dashboard"""
        logger.info("Starting MCP Server Dashboard...")
        
        try:
            # Always use the simple dashboard since dependencies aren't available
            logger.info("Using simple dashboard implementation")
            self._create_simple_dashboard(args)
            
            # Keep the dashboard running
            if hasattr(args, 'keep_running') and args.keep_running:
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    logger.info("Dashboard stopped by user")
                    
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
    
    def _create_simple_dashboard(self, args):
        """Create a simple dashboard status page"""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import threading
            import json
            
            class DashboardHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/':
                        self.send_response(200)  
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        
                        html = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <title>IPFS Accelerate MCP Dashboard</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                                .status {{ color: green; font-weight: bold; }}
                                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                                .metric {{ margin: 10px 0; }}
                                .refresh {{ margin: 20px 0; }}
                            </style>
                            <script>
                                function refreshData() {{
                                    fetch('/api/status')
                                        .then(response => response.json())
                                        .then(data => {{
                                            document.getElementById('timestamp').textContent = new Date(data.timestamp * 1000).toLocaleString();
                                            document.getElementById('uptime').textContent = Math.round(data.uptime) + 's';
                                            document.getElementById('core_available').textContent = data.core_available ? 'Yes' : 'No';
                                        }})
                                        .catch(error => console.log('Status error:', error));
                                    
                                    // Fetch queue status
                                    fetch('/api/queue')
                                        .then(response => response.json())
                                        .then(data => {{
                                            if (data.summary) {{
                                                document.getElementById('total_endpoints').textContent = data.summary.total_endpoints || 0;
                                                document.getElementById('active_endpoints').textContent = data.summary.active_endpoints || 0;
                                                document.getElementById('total_queue_size').textContent = data.summary.total_queue_size || 0;
                                                document.getElementById('processing_tasks').textContent = data.summary.total_processing || 0;
                                            }}
                                        }})
                                        .catch(error => console.log('Queue error:', error));
                                }}
                                
                                setInterval(refreshData, 5000);
                                window.onload = refreshData;
                            </script>
                        </head>
                        <body>
                        <h1>🚀 IPFS Accelerate MCP Server Dashboard</h1>
                        
                        <div class="section">
                            <h2>Server Status</h2>
                            <div class="metric">Status: <span class="status">Running</span></div>
                            <div class="metric">Server: <a href="http://{args.host}:{args.port}">http://{args.host}:{args.port}</a></div>
                            <div class="metric">Started: {time.strftime('%Y-%m-%d %H:%M:%S')}</div>
                            <div class="metric">Last Updated: <span id="timestamp">Loading...</span></div>
                        </div>
                        
                        <div class="section">
                            <h2>System Information</h2>
                            <div class="metric">Uptime: <span id="uptime">Loading...</span></div>
                            <div class="metric">Core Available: <span id="core_available">Loading...</span></div>
                        </div>
                        
                        <div class="section">
                            <h2>Available Commands</h2>
                            <div class="metric">• Text Generation: <code>ipfs-accelerate inference generate --prompt "Hello world"</code></div>
                            <div class="metric">• List Models: <code>ipfs-accelerate models list</code></div>
                            <div class="metric">• Network Status: <code>ipfs-accelerate network status</code></div>
                            <div class="metric">• Add File: <code>ipfs-accelerate files add /path/to/file</code></div>
                            <div class="metric">• Queue Status: <code>ipfs-accelerate queue status</code></div>
                            <div class="metric">• Queue by Model: <code>ipfs-accelerate queue models --model-type text-generation</code></div>
                            <div class="metric">• Endpoint Details: <code>ipfs-accelerate queue endpoints</code></div>
                        </div>
                        
                        <div class="section">
                            <h2>Queue Status</h2>
                            <div class="metric">Total Endpoints: <span id="total_endpoints">Loading...</span></div>
                            <div class="metric">Active Endpoints: <span id="active_endpoints">Loading...</span></div>
                            <div class="metric">Total Queue Size: <span id="total_queue_size">Loading...</span></div>
                            <div class="metric">Processing Tasks: <span id="processing_tasks">Loading...</span></div>
                        </div>
                        
                        <div class="refresh">
                            <button onclick="refreshData()">🔄 Refresh Data</button>
                        </div>
                        </body>
                        </html>
                        """
                        self.wfile.write(html.encode())
                        
                    elif self.path == '/api/status':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        
                        # Get status from shared core
                        status_data = shared_core.get_status()
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
                        
                    else:
                        self.send_response(404)
                        self.end_headers()
                        self.wfile.write(b'Not Found')
                
                def log_message(self, format, *args):
                    # Suppress request logs
                    pass
            
            server = HTTPServer((args.dashboard_host, args.dashboard_port), DashboardHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            
            logger.info(f"Simple dashboard started at http://{args.dashboard_host}:{args.dashboard_port}")
            
            if args.open_browser:
                time.sleep(1)
                webbrowser.open(f"http://{args.dashboard_host}:{args.dashboard_port}")
                
        except Exception as e:
            logger.error(f"Error creating simple dashboard: {e}")
    
    def run_mcp_status(self, args):
        """Check MCP server status"""
        try:
            import requests
            
            url = f"http://{args.host}:{args.port}/health"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                print(f"✅ MCP Server is running at http://{args.host}:{args.port}")
                return 0
            else:
                print(f"❌ MCP Server returned status {response.status_code}")
                return 1
                
        except requests.exceptions.ConnectionError:
            print(f"❌ MCP Server is not running at http://{args.host}:{args.port}")
            return 1
        except Exception as e:
            print(f"❌ Error checking server status: {e}")
            return 1
    
    def run_inference_generate(self, args):
        """Run text generation inference"""
        logger.info(f"Running text generation with model: {args.model}")
        
        if inference_ops:
            result = inference_ops.run_text_generation(
                model=args.model,
                prompt=args.prompt,
                max_length=args.max_length,
                temperature=args.temperature
            )
        else:
            result = {"error": "Inference operations not available", "fallback": True}
        
        if args.output_json:
            print(json.dumps(result, indent=2))
        else:
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return 1
            else:
                output = result.get('output') or result.get('result', 'No output')
                print(f"✅ Generated text: {output}")
        
        return 0
    
    def run_files_add(self, args):
        """Add file to IPFS"""
        logger.info(f"Adding file to IPFS: {args.file_path}")
        
        if not os.path.exists(args.file_path):
            print(f"❌ File not found: {args.file_path}")
            return 1
        
        if file_ops:
            result = file_ops.add_file(args.file_path)
        else:
            result = {"error": "File operations not available", "fallback": True}
        
        if args.output_json:
            print(json.dumps(result, indent=2))
        else:
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return 1
            else:
                cid = result.get('cid') or result.get('result', 'No CID')
                print(f"✅ File added: {cid}")
        
        return 0
    
    def run_models_list(self, args):
        """List available models"""
        logger.info("Listing available models")
        
        if model_ops:
            result = model_ops.list_models()
        else:
            result = {"error": "Model operations not available", "fallback": True}
        
        if args.output_json:
            print(json.dumps(result, indent=2))
        else:
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return 1
            else:
                models = result.get('models', [])
                print(f"✅ Available models ({len(models)}):")
                for model in models:
                    if isinstance(model, dict):
                        print(f"  - {model.get('id', 'unknown')} ({model.get('type', 'unknown type')})")
                    else:
                        print(f"  - {model}")
        
        return 0
    
    def run_network_status(self, args):
        """Check network status"""
        logger.info("Checking network status")
        
        if network_ops:
            result = network_ops.get_network_status()
        else:
            result = {"error": "Network operations not available", "fallback": True}
        
        if args.output_json:
            print(json.dumps(result, indent=2))
        else:
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return 1
            else:
                status = result.get('status', 'unknown')
                peers = result.get('peers', 0)
                print(f"✅ Network status: {status}")
                print(f"   Connected peers: {peers}")
        
        return 0
    
    def run_queue_status(self, args):
        """Show queue status for all endpoints"""
        logger.info("Getting queue status")
        
        if queue_ops:
            result = queue_ops.get_queue_status()
        else:
            result = {"error": "Queue operations not available", "fallback": True}
        
        if args.output_json:
            print(json.dumps(result, indent=2))
        else:
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return 1
            else:
                self._display_queue_status(result)
        
        return 0
    
    def run_queue_history(self, args):
        """Show queue performance history"""
        logger.info("Getting queue history")
        
        if queue_ops:
            result = queue_ops.get_queue_history()
        else:
            result = {"error": "Queue operations not available", "fallback": True}
        
        if args.output_json:
            print(json.dumps(result, indent=2))
        else:
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return 1
            else:
                self._display_queue_history(result)
        
        return 0
    
    def run_queue_models(self, args):
        """Show queues by model type"""
        logger.info(f"Getting model queues{' for ' + args.model_type if args.model_type else ''}")
        
        if queue_ops:
            result = queue_ops.get_model_queues(model_type=args.model_type)
        else:
            result = {"error": "Queue operations not available", "fallback": True}
        
        if args.output_json:
            print(json.dumps(result, indent=2))
        else:
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return 1
            else:
                self._display_model_queues(result, args.model_type)
        
        return 0
    
    def run_queue_endpoints(self, args):
        """Show endpoint details"""
        logger.info(f"Getting endpoint details{' for ' + args.endpoint_id if args.endpoint_id else ''}")
        
        if queue_ops:
            result = queue_ops.get_endpoint_details(endpoint_id=args.endpoint_id)
        else:
            result = {"error": "Queue operations not available", "fallback": True}
        
        if args.output_json:
            print(json.dumps(result, indent=2))
        else:
            if "error" in result:
                print(f"❌ Error: {result['error']}")
                return 1
            else:
                self._display_endpoint_details(result, args.endpoint_id)
        
        return 0
    
    def _display_queue_status(self, result):
        """Display queue status in human-readable format"""
        print("📊 Queue Status Summary")
        print("=" * 50)
        
        # Global queue stats
        global_queue = result.get('global_queue', {})
        print(f"🌐 Global Queue:")
        print(f"   Total tasks: {global_queue.get('total_tasks', 0)}")
        print(f"   Pending: {global_queue.get('pending_tasks', 0)}")
        print(f"   Processing: {global_queue.get('processing_tasks', 0)}")
        print(f"   Completed: {global_queue.get('completed_tasks', 0)}")
        print(f"   Failed: {global_queue.get('failed_tasks', 0)}")
        
        # Summary
        summary = result.get('summary', {})
        if summary:
            print(f"\n📈 Summary:")
            print(f"   Total endpoints: {summary.get('total_endpoints', 0)}")
            print(f"   Active endpoints: {summary.get('active_endpoints', 0)}")
            print(f"   Total queue size: {summary.get('total_queue_size', 0)}")
            print(f"   Total processing: {summary.get('total_processing', 0)}")
            
            endpoint_types = summary.get('endpoint_types', {})
            if endpoint_types:
                print(f"   Endpoint types:")
                for ep_type, count in endpoint_types.items():
                    print(f"     - {ep_type}: {count}")
        
        # Endpoint details
        endpoint_queues = result.get('endpoint_queues', {})
        if endpoint_queues:
            print(f"\n🖥️  Endpoint Details:")
            for endpoint_id, endpoint in endpoint_queues.items():
                status_icon = "🟢" if endpoint.get('status') == 'active' else "🟡" if endpoint.get('status') == 'idle' else "🔴"
                print(f"   {status_icon} {endpoint_id} ({endpoint.get('endpoint_type', 'unknown')})")
                print(f"      Queue: {endpoint.get('queue_size', 0)}, Processing: {endpoint.get('processing', 0)}")
                print(f"      Models: {', '.join(endpoint.get('model_types', []))}")
                if endpoint.get('current_task'):
                    task = endpoint['current_task']
                    print(f"      Current: {task.get('model', 'unknown')} ({task.get('task_type', 'unknown')})")
    
    def _display_queue_history(self, result):
        """Display queue history in human-readable format"""
        print("📈 Queue Performance History")
        print("=" * 50)
        
        # Model type statistics
        model_stats = result.get('model_type_stats', {})
        if model_stats:
            print("🤖 Model Type Statistics:")
            for model_type, stats in model_stats.items():
                print(f"   {model_type}:")
                print(f"      Requests: {stats.get('total_requests', 0)}")
                print(f"      Avg time: {stats.get('avg_time', 0):.1f}s")
                print(f"      Success rate: {stats.get('success_rate', 0):.1f}%")
        
        # Endpoint performance
        endpoint_perf = result.get('endpoint_performance', {})
        if endpoint_perf:
            print(f"\n🎯 Endpoint Performance:")
            for endpoint_id, perf in endpoint_perf.items():
                print(f"   {endpoint_id}:")
                print(f"      Uptime: {perf.get('uptime', 0):.1f}%")
                print(f"      Success rate: {perf.get('success_rate', 0):.1f}%")
                print(f"      Avg response: {perf.get('avg_response_time', 0):.1f}s")
        
        # Time series data summary
        time_series = result.get('time_series', {})
        if time_series and time_series.get('queue_sizes'):
            recent_queue = time_series['queue_sizes'][-1] if time_series['queue_sizes'] else 0
            recent_processing = time_series['processing_tasks'][-1] if time_series['processing_tasks'] else 0
            print(f"\n⏰ Current Status:")
            print(f"   Queue size: {recent_queue}")
            print(f"   Processing: {recent_processing}")
    
    def _display_model_queues(self, result, model_type):
        """Display model-specific queue information"""
        if model_type:
            print(f"🤖 Queue Status for Model Type: {model_type}")
            print("=" * 50)
            
            matching_endpoints = result.get('matching_endpoints', {})
            if not matching_endpoints:
                print(f"❌ No endpoints found for model type '{model_type}'")
                return
            
            print(f"📊 Summary:")
            print(f"   Matching endpoints: {result.get('total_matching', 0)}")
            print(f"   Total queue size: {result.get('total_queue_size', 0)}")
            print(f"   Total processing: {result.get('total_processing', 0)}")
            
            print(f"\n🖥️  Endpoints:")
            for endpoint_id, endpoint in matching_endpoints.items():
                status_icon = "🟢" if endpoint.get('status') == 'active' else "🟡" if endpoint.get('status') == 'idle' else "🔴"
                print(f"   {status_icon} {endpoint_id}")
                print(f"      Type: {endpoint.get('endpoint_type', 'unknown')}")
                print(f"      Queue: {endpoint.get('queue_size', 0)}, Processing: {endpoint.get('processing', 0)}")
        else:
            print("🤖 Queue Status by Model Type")
            print("=" * 50)
            
            model_type_queues = result.get('model_type_queues', {})
            if not model_type_queues:
                print("❌ No model type queues found")
                return
            
            for mt, queue_info in model_type_queues.items():
                endpoints = queue_info.get('endpoints', [])
                total_queue = queue_info.get('total_queue', 0)
                total_processing = queue_info.get('total_processing', 0)
                
                print(f"\n📋 {mt}:")
                print(f"   Endpoints: {len(endpoints)}")
                print(f"   Total queue: {total_queue}")
                print(f"   Total processing: {total_processing}")
                
                # Show top endpoints
                for endpoint in endpoints[:3]:  # Show first 3
                    status_icon = "🟢" if endpoint.get('status') == 'active' else "🟡" if endpoint.get('status') == 'idle' else "🔴"
                    print(f"     {status_icon} {endpoint.get('endpoint_id', 'unknown')}: queue={endpoint.get('queue_size', 0)}")
    
    def _display_endpoint_details(self, result, endpoint_id):
        """Display detailed endpoint information"""
        if endpoint_id:
            print(f"🖥️  Endpoint Details: {endpoint_id}")
            print("=" * 50)
            
            details = result.get('details', {})
            if not details:
                print("❌ No details found")
                return
            
            print(f"Type: {details.get('endpoint_type', 'unknown')}")
            print(f"Status: {details.get('status', 'unknown')}")
            print(f"Queue size: {details.get('queue_size', 0)}")
            print(f"Processing: {details.get('processing', 0)}")
            print(f"Avg processing time: {details.get('avg_processing_time', 0):.1f}s")
            print(f"Model types: {', '.join(details.get('model_types', []))}")
            
            if details.get('device'):
                print(f"Device: {details['device']}")
            if details.get('peer_id'):
                print(f"Peer ID: {details['peer_id']}")
            if details.get('provider'):
                print(f"Provider: {details['provider']}")
            if details.get('network_latency'):
                print(f"Network latency: {details['network_latency']}ms")
            
            current_task = details.get('current_task')
            if current_task:
                print(f"\n🔄 Current Task:")
                print(f"   Task ID: {current_task.get('task_id', 'unknown')}")
                print(f"   Model: {current_task.get('model', 'unknown')}")
                print(f"   Type: {current_task.get('task_type', 'unknown')}")
                print(f"   Estimated completion: {current_task.get('estimated_completion', 'unknown')}")
            
        else:
            print("🖥️  All Endpoints")
            print("=" * 50)
            
            endpoints = result.get('endpoints', {})
            if not endpoints:
                print("❌ No endpoints found")
                return
            
            print(f"Total endpoints: {result.get('total_endpoints', 0)}")
            
            for endpoint_id, endpoint in endpoints.items():
                status_icon = "🟢" if endpoint.get('status') == 'active' else "🟡" if endpoint.get('status') == 'idle' else "🔴"
                print(f"\n{status_icon} {endpoint_id}")
                print(f"   Type: {endpoint.get('endpoint_type', 'unknown')}")
                print(f"   Queue: {endpoint.get('queue_size', 0)}, Processing: {endpoint.get('processing', 0)}")
                print(f"   Models: {', '.join(endpoint.get('model_types', []))}")


def create_parser():
    """Create the argument parser"""
    parser = argparse.ArgumentParser(
        prog="ipfs-accelerate",
        description="IPFS Accelerate CLI - Unified interface for hardware-accelerated ML inference with IPFS"
    )
    
    parser.add_argument("--version", action="version", version="ipfs-accelerate 0.0.45")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--output-json", action="store_true", help="Output results as JSON")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # MCP commands
    mcp_parser = subparsers.add_parser("mcp", help="MCP server management")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", help="MCP commands")
    
    # MCP start command
    start_parser = mcp_subparsers.add_parser("start", help="Start MCP server")
    start_parser.add_argument("--name", default="ipfs-accelerate", help="Server name")
    start_parser.add_argument("--host", default="localhost", help="Host to bind to")
    start_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    start_parser.add_argument("--dashboard", action="store_true", help="Also start dashboard")
    start_parser.add_argument("--dashboard-host", default="localhost", help="Dashboard host")
    start_parser.add_argument("--dashboard-port", type=int, default=8001, help="Dashboard port")
    start_parser.add_argument("--open-browser", action="store_true", help="Open dashboard in browser")
    
    # MCP dashboard command
    dashboard_parser = mcp_subparsers.add_parser("dashboard", help="Start MCP server dashboard")
    dashboard_parser.add_argument("--host", default="localhost", help="MCP server host")
    dashboard_parser.add_argument("--port", type=int, default=8000, help="MCP server port")
    dashboard_parser.add_argument("--dashboard-host", default="localhost", help="Dashboard host")
    dashboard_parser.add_argument("--dashboard-port", type=int, default=8001, help="Dashboard port")
    dashboard_parser.add_argument("--open-browser", action="store_true", help="Open dashboard in browser")
    dashboard_parser.add_argument("--keep-running", action="store_true", help="Keep dashboard running (for testing)")
    
    # MCP status command
    status_parser = mcp_subparsers.add_parser("status", help="Check MCP server status")
    status_parser.add_argument("--host", default="localhost", help="Server host")
    status_parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    # Inference commands
    inference_parser = subparsers.add_parser("inference", help="AI inference operations")
    inference_subparsers = inference_parser.add_subparsers(dest="inference_command", help="Inference commands")
    
    # Inference generate command
    generate_parser = inference_subparsers.add_parser("generate", help="Generate text")
    generate_parser.add_argument("--model", default="gpt2", help="Model to use")
    generate_parser.add_argument("--prompt", required=True, help="Input prompt")
    generate_parser.add_argument("--max-length", type=int, default=100, help="Maximum length")
    generate_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    generate_parser.add_argument("--output-json", action="store_true", help="Output as JSON")
    
    # Files commands
    files_parser = subparsers.add_parser("files", help="IPFS file operations")
    files_subparsers = files_parser.add_subparsers(dest="files_command", help="File commands")
    
    # Files add command
    add_parser = files_subparsers.add_parser("add", help="Add file to IPFS")
    add_parser.add_argument("file_path", help="Path to file to add")
    add_parser.add_argument("--output-json", action="store_true", help="Output as JSON")
    
    # Models commands
    models_parser = subparsers.add_parser("models", help="Model management")
    models_subparsers = models_parser.add_subparsers(dest="models_command", help="Model commands")
    
    # Models list command
    list_parser = models_subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--output-json", action="store_true", help="Output as JSON")
    
    # Network commands
    network_parser = subparsers.add_parser("network", help="Network operations")
    network_subparsers = network_parser.add_subparsers(dest="network_command", help="Network commands")
    
    # Network status command
    net_status_parser = network_subparsers.add_parser("status", help="Check network status")
    net_status_parser.add_argument("--output-json", action="store_true", help="Output as JSON")
    
    # Queue commands
    queue_parser = subparsers.add_parser("queue", help="Queue management and monitoring")
    queue_subparsers = queue_parser.add_subparsers(dest="queue_command", help="Queue commands")
    
    # Queue status command
    queue_status_parser = queue_subparsers.add_parser("status", help="Show overall queue status")
    queue_status_parser.add_argument("--output-json", action="store_true", help="Output as JSON")
    
    # Queue history command
    queue_history_parser = queue_subparsers.add_parser("history", help="Show queue performance history")
    queue_history_parser.add_argument("--output-json", action="store_true", help="Output as JSON")
    
    # Queue models command
    queue_models_parser = queue_subparsers.add_parser("models", help="Show queues by model type")
    queue_models_parser.add_argument("--model-type", help="Filter by specific model type")
    queue_models_parser.add_argument("--output-json", action="store_true", help="Output as JSON")
    
    # Queue endpoints command
    queue_endpoints_parser = queue_subparsers.add_parser("endpoints", help="Show endpoint details")
    queue_endpoints_parser.add_argument("--endpoint-id", help="Get details for specific endpoint")
    queue_endpoints_parser.add_argument("--output-json", action="store_true", help="Output as JSON")
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Create CLI instance
    cli = IPFSAccelerateCLI()
    cli.setup_signal_handlers()
    
    try:
        # Handle commands
        if args.command == "mcp":
            if args.mcp_command == "start":
                return cli.run_mcp_start(args)
            elif args.mcp_command == "dashboard":
                return cli.run_mcp_dashboard(args)
            elif args.mcp_command == "status":
                return cli.run_mcp_status(args)
            else:
                parser.print_help()
                return 1
        
        elif args.command == "inference":
            if args.inference_command == "generate":
                return cli.run_inference_generate(args)
            else:
                parser.print_help()
                return 1
        
        elif args.command == "files":
            if args.files_command == "add":
                return cli.run_files_add(args)
            else:
                parser.print_help()
                return 1
        
        elif args.command == "models":
            if args.models_command == "list":
                return cli.run_models_list(args)
            else:
                parser.print_help()
                return 1
        
        elif args.command == "network":
            if args.network_command == "status":
                return cli.run_network_status(args)
            else:
                parser.print_help()
                return 1
        
        elif args.command == "queue":
            if args.queue_command == "status":
                return cli.run_queue_status(args)
            elif args.queue_command == "history":
                return cli.run_queue_history(args)
            elif args.queue_command == "models":
                return cli.run_queue_models(args)
            elif args.queue_command == "endpoints":
                return cli.run_queue_endpoints(args)
            else:
                parser.print_help()
                return 1
        
        else:
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        cli.cleanup()


if __name__ == "__main__":
    sys.exit(main())