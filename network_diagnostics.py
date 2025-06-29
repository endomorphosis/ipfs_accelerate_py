#!/usr/bin/env python3
"""
Network diagnostic tool for MCP servers

This script checks network connectivity and server availability.
"""

import os
import sys
import socket
import requests
import logging
import argparse
import subprocess
import time
from typing import Dict, Any, Optional, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("network_diagnostics.log", mode='w')
    ]
)
logger = logging.getLogger("network_diagnostics")

def check_port_in_use(host: str, port: int) -> bool:
    """
    Check if a port is in use
    
    Args:
        host: Host name or IP address
        port: Port number
    
    Returns:
        bool: True if the port is in use, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect((host, port))
            return True
    except (socket.timeout, ConnectionRefusedError):
        return False

def check_listening_ports() -> List[Tuple[str, int]]:
    """
    Check all listening ports on the system
    
    Returns:
        List[Tuple[str, int]]: List of (protocol, port) tuples
    """
    try:
        # Use netstat to check listening ports
        result = subprocess.run(
            ["netstat", "-tuln"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            logger.error(f"Error running netstat: {result.stderr}")
            return []
        
        # Parse the output
        lines = result.stdout.split("\n")
        listening_ports = []
        
        for line in lines:
            if "LISTEN" in line:
                parts = line.split()
                if len(parts) >= 4:
                    local_address = parts[3]
                    if ":" in local_address:
                        host, port_str = local_address.rsplit(":", 1)
                        try:
                            port = int(port_str)
                            proto = "tcp" if "tcp" in parts[0].lower() else "udp"
                            listening_ports.append((proto, port))
                        except ValueError:
                            pass
        
        return listening_ports
    except Exception as e:
        logger.error(f"Error checking listening ports: {str(e)}")
        return []

def check_server_status(url: str) -> Dict[str, Any]:
    """
    Check if a server is running and responding
    
    Args:
        url: URL to check
    
    Returns:
        Dict[str, Any]: Status information
    """
    try:
        response = requests.get(url, timeout=5)
        return {
            "url": url,
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "content_type": response.headers.get("Content-Type"),
            "content_length": len(response.content),
            "elapsed": response.elapsed.total_seconds()
        }
    except requests.exceptions.ConnectionError:
        return {
            "url": url,
            "status_code": None,
            "success": False,
            "error": "Connection error"
        }
    except requests.exceptions.Timeout:
        return {
            "url": url,
            "status_code": None,
            "success": False,
            "error": "Timeout"
        }
    except Exception as e:
        return {
            "url": url,
            "status_code": None,
            "success": False,
            "error": str(e)
        }

def run_simple_server(port: int, timeout: int = 30) -> bool:
    """
    Run a simple HTTP server to test if the port is available
    
    Args:
        port: Port to listen on
        timeout: Timeout in seconds
    
    Returns:
        bool: True if the server started successfully, False otherwise
    """
    try:
        from http.server import HTTPServer, BaseHTTPRequestHandler
        
        class SimpleHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Simple HTTP server is working!")
            
            def log_message(self, format, *args):
                # Silence the logging
                pass
        
        def start_server():
            try:
                server = HTTPServer(("0.0.0.0", port), SimpleHandler)
                server.timeout = timeout
                logger.info(f"Started simple HTTP server on port {port}")
                
                # Handle a single request then exit
                server.handle_request()
                logger.info("Simple HTTP server received a request, shutting down")
                return True
            except Exception as e:
                logger.error(f"Error starting simple HTTP server: {str(e)}")
                return False
        
        # Start the server in a separate process
        import multiprocessing
        process = multiprocessing.Process(target=start_server)
        process.start()
        
        # Wait for the server to start
        time.sleep(1)
        
        # Check if the server is running
        is_running = check_port_in_use("localhost", port)
        
        if is_running:
            # Test the server
            try:
                response = requests.get(f"http://localhost:{port}")
                success = response.status_code == 200
                logger.info(f"Simple HTTP server test: {'Success' if success else 'Failed'}")
            except Exception:
                success = False
                logger.error("Error connecting to simple HTTP server")
        else:
            success = False
            logger.error(f"Simple HTTP server did not start on port {port}")
        
        # Wait for the process to finish or kill it
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
        
        return success
    except Exception as e:
        logger.error(f"Error in run_simple_server: {str(e)}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Network diagnostic tool for MCP servers")
    parser.add_argument("--host", type=str, default="localhost", help="Host to check")
    parser.add_argument("--port", type=int, default=8002, help="Port to check")
    parser.add_argument("--test-port", action="store_true", help="Run a test server on the port")
    args = parser.parse_args()
    
    print("=" * 80)
    print("Network Diagnostics for MCP Servers")
    print("=" * 80)
    print()
    
    # Check if the port is in use
    port_in_use = check_port_in_use(args.host, args.port)
    print(f"Port {args.port} on {args.host}: {'In use' if port_in_use else 'Not in use'}")
    
    # Check all listening ports
    print("\nListening ports on the system:")
    listening_ports = check_listening_ports()
    if listening_ports:
        for proto, port in listening_ports:
            print(f"  {proto.upper()}: {port}")
    else:
        print("  No ports found or error retrieving ports")
    
    # Check MCP server status
    print("\nChecking MCP server status:")
    urls_to_check = [
        f"http://{args.host}:{args.port}/",
        f"http://{args.host}:{args.port}/mcp",
        f"http://{args.host}:{args.port}/mcp/manifest"
    ]
    
    for url in urls_to_check:
        status = check_server_status(url)
        if status["success"]:
            print(f"  ✓ {url} - Status: {status['status_code']} OK")
        else:
            print(f"  ✗ {url} - Error: {status.get('error') or f'Status: {status.get(\"status_code\")}'}")
    
    # Run a test server if requested
    if args.test_port and not port_in_use:
        print("\nRunning a test HTTP server:")
        success = run_simple_server(args.port)
        if success:
            print(f"  ✓ Successfully started and accessed a test server on port {args.port}")
        else:
            print(f"  ✗ Failed to start or access a test server on port {args.port}")
    
    # Print recommendations
    print("\nRecommendations:")
    if port_in_use:
        print("  - The port is in use, but the MCP server may not be responding correctly")
        print("  - Check the server logs for errors")
        print("  - Make sure the server is properly configured to listen on all interfaces (0.0.0.0)")
    else:
        print("  - The port is not in use, the MCP server is not running")
        print("  - Start the MCP server with: ./fixed_standards_mcp_server.py --host 0.0.0.0 --port 8002")
    
    print("\nDiagnostic complete")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
