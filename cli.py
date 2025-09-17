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
    from .shared import SharedCore, InferenceOperations, FileOperations, ModelOperations, NetworkOperations
    HAVE_CORE = True
except ImportError as e:
    logger.warning(f"Core modules not available: {e}")
    try:
        # Try alternative import paths
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from shared import SharedCore, InferenceOperations, FileOperations, ModelOperations, NetworkOperations
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
else:
    shared_core = SharedCore()
    inference_ops = None
    file_ops = None
    model_ops = None
    network_ops = None

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
            # Look for dashboard servers in the test directory
            dashboard_paths = [
                "/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test/duckdb_api/distributed_testing/load_balancer/monitoring/dashboard_server.py",
                "/home/runner/work/ipfs_accelerate_py/ipfs_accelerate_py/test/duckdb_api/distributed_testing/dashboard_server.py"
            ]
            
            dashboard_path = None
            for path in dashboard_paths:
                if os.path.exists(path):
                    dashboard_path = path
                    break
            
            if dashboard_path:
                # Start dashboard server
                cmd = [
                    sys.executable, dashboard_path,
                    "--host", args.dashboard_host,
                    "--port", str(args.dashboard_port),
                    "--coordinator-url", f"http://{args.host}:{args.port}"
                ]
                
                self.dashboard_process = subprocess.Popen(cmd)
                logger.info(f"Dashboard started at http://{args.dashboard_host}:{args.dashboard_port}")
                
                # Open in browser if requested
                if args.open_browser:
                    time.sleep(2)  # Give server time to start
                    webbrowser.open(f"http://{args.dashboard_host}:{args.dashboard_port}")
            else:
                logger.warning("Dashboard server not found, creating simple status page")
                # Create a simple status endpoint
                self._create_simple_dashboard(args)
                
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
    
    def _create_simple_dashboard(self, args):
        """Create a simple dashboard status page"""
        try:
            from http.server import HTTPServer, SimpleHTTPRequestHandler
            import threading
            
            class DashboardHandler(SimpleHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/':
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        
                        html = f"""
                        <!DOCTYPE html>
                        <html>
                        <head><title>IPFS Accelerate MCP Dashboard</title></head>
                        <body>
                        <h1>IPFS Accelerate MCP Server</h1>
                        <p>Status: <span style="color: green;">Running</span></p>
                        <p>Server: <a href="http://{args.host}:{args.port}">http://{args.host}:{args.port}</a></p>
                        <p>Started: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                        </body>
                        </html>
                        """
                        self.wfile.write(html.encode())
                    else:
                        super().do_GET()
            
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
    
    # Files commands
    files_parser = subparsers.add_parser("files", help="IPFS file operations")
    files_subparsers = files_parser.add_subparsers(dest="files_command", help="File commands")
    
    # Files add command
    add_parser = files_subparsers.add_parser("add", help="Add file to IPFS")
    add_parser.add_argument("file_path", help="Path to file to add")
    
    # Models commands
    models_parser = subparsers.add_parser("models", help="Model management")
    models_subparsers = models_parser.add_subparsers(dest="models_command", help="Model commands")
    
    # Models list command
    list_parser = models_subparsers.add_parser("list", help="List available models")
    
    # Network commands
    network_parser = subparsers.add_parser("network", help="Network operations")
    network_subparsers = network_parser.add_subparsers(dest="network_command", help="Network commands")
    
    # Network status command
    net_status_parser = network_subparsers.add_parser("status", help="Check network status")
    
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