#!/usr/bin/env python3
"""
Fix WebSocket Bridge for WebNN/WebGPU Implementation

This script modifies the WebSocket bridging approach in the implement_real_webnn_webgpu.py
file to resolve connection issues between Python and the browser.

Key fixes:
1. Updated WebSocket server to use more permissive settings
2. Added proper HTTP server for serving the HTML file directly
3. Modified security settings for Chrome/Firefox to allow connections
4. Added better error handling and diagnostics

Usage:
    python fix_websocket_bridge.py --check           # Check current implementation
    python fix_websocket_bridge.py --apply-fix       # Apply the fixes
    python fix_websocket_bridge.py --test-connection # Test connection after fix
"""

import os
import sys
import re
import json
import time
import asyncio
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the implementation file
IMPL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "implement_real_webnn_webgpu.py")

# Try to import websockets
try:
    import websockets
    print("Successfully imported websockets")
except ImportError as e:
    print(f"Failed to import websockets: {e}")
    print("Install with: pip install websockets")
    sys.exit(1)

# Try to import selenium if needed
if '--test-connection' in sys.argv:
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service as ChromeService
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        print("Successfully imported selenium")
    except ImportError as e:
        print(f"Failed to import selenium: {e}")
        print("Install with: pip install selenium")
        sys.exit(1)

# HTTP server for serving the HTML file
import http.server
import socketserver
import threading

def start_http_server(html_path, port=8000):
    """Start HTTP server for serving the HTML file.
    
    Args:
        html_path: Path to the HTML file
        port: Port to use for HTTP server
        
    Returns:
        Tuple of (server, server_thread)
    """
    # Create a simple HTTP server
    class SimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/' or self.path == '/index.html':
                self.path = html_path
            return http.server.SimpleHTTPRequestHandler.do_GET(self)
        
        def log_message(self, format, *args):
            # Suppress logging
            pass
    
    # Create server
    httpd = socketserver.TCPServer(("", port), SimpleHTTPRequestHandler)
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    logger.info(f"HTTP server started on port {port}")
    return httpd, server_thread

def stop_http_server(httpd):
    """Stop HTTP server.
    
    Args:
        httpd: HTTP server instance
    """
    httpd.shutdown()
    httpd.server_close()
    logger.info("HTTP server stopped")

def check_implementation():
    """Check the current WebSocket implementation in the file."""
    
    # Check if file exists
    if not os.path.exists(IMPL_FILE):
        logger.error(f"Implementation file not found: {IMPL_FILE}")
        return False
    
    # Read file
    with open(IMPL_FILE, "r") as f:
        content = f.read()
    
    # Check for WebSocket server implementation
    websocket_server_class = re.search(r"class\s+WebBridgeServer\s*:", content)
    if not websocket_server_class:
        logger.error("WebBridgeServer class not found in implementation file")
        return False
    
    # Check for browser manager implementation
    browser_manager_class = re.search(r"class\s+BrowserManager\s*:", content)
    if not browser_manager_class:
        logger.error("BrowserManager class not found in implementation file")
        return False
    
    # Look for WebSocket server start method
    websocket_server_start = re.search(r"async\s+def\s+start\s*\(\s*self\s*\):", content)
    if not websocket_server_start:
        logger.error("WebBridgeServer.start method not found")
        return False
    
    # Check for websockets.serve usage
    websockets_serve = re.search(r"self\.server\s*=\s*await\s+websockets\.serve\s*\(", content)
    if not websockets_serve:
        logger.error("websockets.serve not found in WebBridgeServer.start method")
        return False
    
    # Check for browser HTML template
    browser_html_template = re.search(r"BROWSER_HTML_TEMPLATE\s*=\s*\"\"\"", content)
    if not browser_html_template:
        logger.error("BROWSER_HTML_TEMPLATE not found in implementation file")
        return False
    
    logger.info("Implementation file contains expected WebSocket and browser components")
    
    # Check for potential issues
    issues = []
    
    # Check for WebSocket server settings
    if "ping_interval" not in content:
        issues.append("WebSocket server doesn't configure ping_interval")
    
    if "max_size" not in content:
        issues.append("WebSocket server doesn't configure max_size")
    
    # Check for browser options
    if "options.add_argument(\"--allow-file-access-from-files\")" not in content:
        issues.append("Chrome options don't include --allow-file-access-from-files")
    
    if "options.add_argument(\"--disable-web-security\")" not in content:
        issues.append("Chrome options don't include --disable-web-security")
    
    # HTML file loading approach
    if "file://" in content and "http://" not in content:
        issues.append("Uses file:// protocol instead of http:// for HTML loading")
    
    # Check for HTTP server
    if "http.server" not in content:
        issues.append("No HTTP server implementation found")
    
    if issues:
        logger.warning("Potential issues found in implementation:")
        for issue in issues:
            logger.warning(f"- {issue}")
    else:
        logger.info("No potential issues found in implementation")
    
    return True

def apply_fix():
    """Apply fixes to the WebSocket implementation."""
    
    # Check if file exists
    if not os.path.exists(IMPL_FILE):
        logger.error(f"Implementation file not found: {IMPL_FILE}")
        return False
    
    # Read file
    with open(IMPL_FILE, "r") as f:
        content = f.read()
    
    # Create backup
    backup_file = f"{IMPL_FILE}.bak"
    with open(backup_file, "w") as f:
        f.write(content)
    logger.info(f"Created backup file: {backup_file}")
    
    # Apply fixes
    
    # 1. Fix WebSocket server implementation
    old_server_code = re.search(r"async\s+def\s+start\s*\(\s*self\s*\):\s*.*?self\.server\s*=\s*await\s+websockets\.serve\s*\(.*?\)", content, re.DOTALL)
    if old_server_code:
        new_server_code = """async def start(self):
        \"\"\"Start WebSocket server.\"\"\"
        try:
            # Use more permissive settings for better compatibility
            self.server = await websockets.serve(
                self.handle_connection,
                "localhost",
                self.port,
                ping_interval=None,  # Disable ping to test raw connection
                max_size=10_000_000,  # Allow large messages
                max_queue=64          # Allow more queued messages
            )
            logger.info(f"WebSocket server started on port {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            return False"""
        
        content = content.replace(old_server_code.group(0), new_server_code)
        logger.info("Updated WebSocket server implementation")
    else:
        logger.warning("Could not find WebSocket server start method to update")
    
    # 2. Fix browser options
    chrome_options_pattern = r"(options\s*=\s*ChromeOptions\(\).*?)(if\s+self\.headless)"
    chrome_options_match = re.search(chrome_options_pattern, content, re.DOTALL)
    if chrome_options_match:
        chrome_options_code = chrome_options_match.group(1)
        new_chrome_options = chrome_options_code
        
        # Add browser security options
        if "options.add_argument(\"--allow-file-access-from-files\")" not in new_chrome_options:
            new_chrome_options += "\n                options.add_argument(\"--allow-file-access-from-files\")"
        
        if "options.add_argument(\"--disable-web-security\")" not in new_chrome_options:
            new_chrome_options += "\n                options.add_argument(\"--disable-web-security\")"
        
        if "options.add_argument(\"--disable-site-isolation-trials\")" not in new_chrome_options:
            new_chrome_options += "\n                options.add_argument(\"--disable-site-isolation-trials\")"
        
        content = content.replace(chrome_options_code, new_chrome_options)
        logger.info("Updated Chrome options")
    else:
        logger.warning("Could not find Chrome options to update")
    
    # 3. Add HTTP server implementation
    if "import http.server" not in content:
        imports_section = re.search(r"import.*?from typing", content, re.DOTALL)
        if imports_section:
            new_imports = imports_section.group(0)
            new_imports += "\nimport http.server\nimport socketserver\nimport threading\n"
            content = content.replace(imports_section.group(0), new_imports)
            logger.info("Added HTTP server imports")
        else:
            logger.warning("Could not find imports section to update")
    
    # Add HTTP server method to BrowserManager class
    browser_manager_class = re.search(r"class\s+BrowserManager\s*:.*?def\s+__init__", content, re.DOTALL)
    if browser_manager_class and "start_http_server" not in content:
        http_server_method = """    def start_http_server(self, html_path, port=8000):
        \"\"\"Start HTTP server for serving the HTML file.
        
        Args:
            html_path: Path to the HTML file
            port: Port to use for HTTP server
            
        Returns:
            Tuple of (server, server_thread)
        \"\"\"
        # Create a simple HTTP request handler
        class SimpleHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/' or self.path == '/index.html':
                    self.path = html_path
                return http.server.SimpleHTTPRequestHandler.do_GET(self)
            
            def log_message(self, format, *args):
                # Suppress logging
                pass
        
        # Find an available port if the specified one is in use
        server_port = port
        max_attempts = 10
        for _ in range(max_attempts):
            try:
                httpd = socketserver.TCPServer(("", server_port), SimpleHTTPRequestHandler)
                break
            except OSError:
                logger.warning(f"Port {server_port} is in use, trying another port")
                server_port += 1
        else:
            logger.error(f"Could not find an available port after {max_attempts} attempts")
            return None, None
        
        # Start server in a separate thread
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        logger.info(f"HTTP server started on port {server_port}")
        self.html_server_port = server_port
        self.html_server = httpd
        self.html_server_thread = server_thread
        return httpd, server_thread
    
    def stop_http_server(self):
        \"\"\"Stop HTTP server.\"\"\"
        if hasattr(self, 'html_server') and self.html_server:
            self.html_server.shutdown()
            self.html_server.server_close()
            logger.info("HTTP server stopped")
            self.html_server = None
            self.html_server_thread = None
    
    def """
        
        new_browser_manager = browser_manager_class.group(0).replace("    def __init__", http_server_method + "__init__")
        content = content.replace(browser_manager_class.group(0), new_browser_manager)
        logger.info("Added HTTP server methods to BrowserManager class")
    else:
        logger.warning("Could not find BrowserManager class to update or HTTP server methods already exist")
    
    # 4. Update the start_browser method to use HTTP instead of file protocol
    start_browser_method = re.search(r"def\s+start_browser\s*\(\s*self\s*\):.*?return\s+(?:True|False)", content, re.DOTALL)
    if start_browser_method:
        old_browser_code = start_browser_method.group(0)
        
        if "file_url = f\"file://{self.html_file}" in old_browser_code:
            # Start HTTP server and use HTTP URL instead of file URL
            new_browser_code = old_browser_code.replace(
                "file_url = f\"file://{self.html_file}?port={self.bridge_port}\"",
                "# Start HTTP server for serving HTML file\n" +
                "            self.start_http_server(self.html_file)\n" +
                "            http_port = self.html_server_port\n" +
                "            file_url = f\"http://localhost:{http_port}?port={self.bridge_port}\""
            )
            
            # Add cleanup for HTTP server in the error handler
            if "self.driver.quit()" in new_browser_code:
                new_browser_code = new_browser_code.replace(
                    "self.driver.quit()",
                    "self.driver.quit()\n                self.stop_http_server()"
                )
            
            content = content.replace(old_browser_code, new_browser_code)
            logger.info("Updated start_browser method to use HTTP server")
        else:
            logger.warning("Could not find file URL code in start_browser method")
    else:
        logger.warning("Could not find start_browser method to update")
    
    # 5. Update browser HTML template to add better error handling
    if "BROWSER_HTML_TEMPLATE" in content:
        browser_html_pattern = r"BROWSER_HTML_TEMPLATE\s*=\s*\"\"\"\s*<!DOCTYPE html>.*?</html>\s*\"\"\""
        browser_html_match = re.search(browser_html_pattern, content, re.DOTALL)
        
        if browser_html_match:
            # Find WebSocket connection code in the template
            ws_connect_pattern = r"function\s+connectWebSocket\s*\(\s*port\s*\)\s*{.*?socket\s*=\s*new\s*WebSocket\s*\(\s*wsUrl\s*\)\s*;"
            ws_connect_match = re.search(ws_connect_pattern, browser_html_match.group(0), re.DOTALL)
            
            if ws_connect_match:
                old_ws_connect = ws_connect_match.group(0)
                
                # Add retry logic and better error handling
                new_ws_connect = old_ws_connect.replace(
                    "socket = new WebSocket(wsUrl);",
                    """// Log connection attempt
                    log('Attempting to connect to: ' + wsUrl);
                    
                    // Add retry logic
                    let retryCount = 0;
                    const maxRetries = 5;
                    const retryInterval = 1000; // ms
                    
                    function tryConnect() {
                        try {
                            socket = new WebSocket(wsUrl);
                            log('WebSocket created, establishing connection...');
                        } catch (e) {
                            log('Error creating WebSocket: ' + e.message, 'error');
                            retryConnection();
                            return;
                        }
                    }
                    
                    function retryConnection() {
                        if (retryCount < maxRetries) {
                            retryCount++;
                            log('Retrying connection in ' + (retryInterval/1000) + ' seconds... (Attempt ' + retryCount + ' of ' + maxRetries + ')', 'info');
                            setTimeout(tryConnect, retryInterval);
                        } else {
                            log('Max retry attempts reached. Could not connect to WebSocket server.', 'error');
                            showError('Failed to connect to WebSocket server after ' + maxRetries + ' attempts. Please check if the server is running.');
                        }
                    }
                    
                    // Initial connection attempt
                    tryConnect();"""
                )
                
                content = content.replace(old_ws_connect, new_ws_connect)
                logger.info("Updated WebSocket connection code in HTML template")
            else:
                logger.warning("Could not find WebSocket connection code in HTML template")
        else:
            logger.warning("Could not find browser HTML template")
    
    # Write fixed content back to file
    with open(IMPL_FILE, "w") as f:
        f.write(content)
    
    logger.info(f"Applied fixes to {IMPL_FILE}")
    return True

async def test_connection():
    """Test the WebSocket connection after fixes."""
    logger.info("Testing WebSocket connection")
    
    # Create a simple HTML file for testing
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WebSocket Test</title>
    </head>
    <body>
        <h1>WebSocket Connection Test</h1>
        <div id="status">Connecting...</div>
        <div id="log"></div>
        
        <script>
            // Log element
            const logElement = document.getElementById('log');
            const statusElement = document.getElementById('status');
            
            // Add log message
            function log(message) {
                const div = document.createElement('div');
                div.textContent = message;
                logElement.appendChild(div);
                console.log(message);
            }
            
            // Update status
            function updateStatus(message, isError = false) {
                statusElement.textContent = message;
                statusElement.style.color = isError ? 'red' : 'green';
            }
            
            // Connect to WebSocket server
            log('Connecting to WebSocket server...');
            const socket = new WebSocket('ws://localhost:8765');
            
            socket.onopen = function() {
                log('Connected to WebSocket server');
                updateStatus('Connected');
                
                // Send a test message
                const message = JSON.stringify({
                    type: 'test',
                    message: 'Hello from browser',
                    timestamp: Date.now()
                });
                
                log('Sending message: ' + message);
                socket.send(message);
            };
            
            socket.onclose = function(event) {
                log('Disconnected from WebSocket server: Code: ' + event.code + ', Reason: ' + event.reason);
                updateStatus('Disconnected', true);
            };
            
            socket.onerror = function(error) {
                log('WebSocket error: ' + error);
                updateStatus('Error connecting to WebSocket server', true);
            };
            
            socket.onmessage = function(event) {
                log('Received message: ' + event.data);
                
                try {
                    const data = JSON.parse(event.data);
                    log('Parsed message type: ' + data.type);
                } catch (e) {
                    log('Not a JSON message: ' + e);
                }
            };
        </script>
    </body>
    </html>
    """
    
    # Create a temporary HTML file
    fd, html_path = tempfile.mkstemp(suffix=".html")
    with os.fdopen(fd, "w") as f:
        f.write(html_content)
    
    logger.info(f"Created test HTML file: {html_path}")
    
    # Start WebSocket server
    async def echo_handler(websocket):
        try:
            logger.info(f"Client connected: {websocket.remote_address}")
            
            # Send an initial message
            await websocket.send(json.dumps({
                "type": "hello",
                "message": "Hello from Python WebSocket server!"
            }))
            
            # Echo messages
            async for message in websocket:
                logger.info(f"Received message: {message}")
                
                # Try to parse as JSON
                try:
                    data = json.loads(message)
                    # Add a timestamp and echo back
                    data["timestamp"] = time.time()
                    data["echo"] = True
                    await websocket.send(json.dumps(data))
                except json.JSONDecodeError:
                    # Not JSON, just echo as text
                    await websocket.send(f"Echo: {message}")
                
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"Connection closed: {e}")
        except Exception as e:
            logger.error(f"Error in echo handler: {e}")
    
    # Start HTTP server
    httpd, server_thread = start_http_server(html_path, port=8000)
    
    # Start WebSocket server
    websocket_server = None
    try:
        logger.info("Starting WebSocket server on port 8765")
        websocket_server = await websockets.serve(
            echo_handler, 
            "localhost", 
            8765,
            ping_interval=None,
            max_size=10_000_000,
            max_queue=64
        )
        
        logger.info("WebSocket server started")
        
        # Start browser
        options = ChromeOptions()
        options.add_argument("--allow-file-access-from-files")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-site-isolation-trials")
        
        service = ChromeService()
        driver = webdriver.Chrome(service=service, options=options)
        
        # Navigate to HTML file
        url = "http://localhost:8000"
        logger.info(f"Opening browser with URL: {url}")
        driver.get(url)
        
        # Wait for test to complete
        logger.info("Waiting for WebSocket test to complete...")
        
        # Wait for connection
        time.sleep(3)
        
        # Check status
        status_element = driver.find_element("id", "status")
        status_text = status_element.text
        
        if "Connected" in status_text:
            logger.info("WebSocket connection successful!")
            success = True
        else:
            logger.error(f"WebSocket connection failed: {status_text}")
            success = False
        
        # Close browser
        driver.quit()
        
        return success
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        return False
    finally:
        # Clean up
        if websocket_server:
            websocket_server.close()
            await websocket_server.wait_closed()
            logger.info("WebSocket server stopped")
        
        # Stop HTTP server
        stop_http_server(httpd)
        
        # Remove temporary file
        try:
            os.unlink(html_path)
        except:
            pass

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fix WebSocket Bridge for WebNN/WebGPU Implementation")
    parser.add_argument("--check", action="store_true", help="Check current implementation")
    parser.add_argument("--apply-fix", action="store_true", help="Apply the fixes")
    parser.add_argument("--test-connection", action="store_true", help="Test connection after fix")
    
    args = parser.parse_args()
    
    # If no specific command is given, print help
    if not (args.check or args.apply_fix or args.test_connection):
        parser.print_help()
        return 1
    
    if args.check:
        check_implementation()
    
    if args.apply_fix:
        apply_fix()
    
    if args.test_connection:
        success = await test_connection()
        return 0 if success else 1
    
    return 0

if __name__ == "__main__":
    asyncio.run(main())