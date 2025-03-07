#!/usr/bin/env python3
"""
WebSocket Diagnostic Tool

This script helps diagnose WebSocket connection issues between Python and the browser.
It tests both server and client sides of the WebSocket connection.

Usage:
    python diagnose_websocket.py --server  # Run WebSocket server
    python diagnose_websocket.py --client  # Run WebSocket client test
    python diagnose_websocket.py --browser # Run browser WebSocket test
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import websockets
try:
    import websockets
    print("Successfully imported websockets")
except ImportError as e:
    print(f"Failed to import websockets: {e}")
    print("Install with: pip install websockets")
    sys.exit(1)

# Try to import selenium if needed for browser test
if '--browser' in sys.argv:
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service as ChromeService
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        print("Successfully imported selenium")
    except ImportError as e:
        print(f"Failed to import selenium: {e}")
        print("Install with: pip install selenium")
        sys.exit(1)

# WebSocket server for diagnostics
async def run_websocket_server(host='localhost', port=8765):
    """Run a WebSocket echo server for diagnostic purposes."""
    
    logger.info(f"Starting WebSocket server on {host}:{port}")
    
    async def echo_handler(websocket):
        try:
            logger.info(f"Client connected: {websocket.remote_address}")
            
            # Send an initial message
            await websocket.send(json.dumps({
                "type": "hello",
                "message": "WebSocket connection established successfully!"
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
    
    try:
        # Use more permissive settings
        async with websockets.serve(
            echo_handler, 
            host, 
            port,
            ping_interval=None,  # Disable ping to test raw connection
            max_size=10_000_000,  # Allow large messages
            max_queue=64          # Allow more queued messages
        ):
            logger.info(f"WebSocket server running at ws://{host}:{port}")
            logger.info("Press Ctrl+C to stop")
            await asyncio.Future()  # Run forever
    except Exception as e:
        logger.error(f"Failed to start WebSocket server: {e}")
        return 1
    
    return 0

# WebSocket client for diagnostics
async def run_websocket_client(host='localhost', port=8765):
    """Run a WebSocket client for diagnostic purposes."""
    
    uri = f"ws://{host}:{port}"
    logger.info(f"Connecting to WebSocket server at {uri}")
    
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to WebSocket server")
            
            # Receive initial message
            response = await websocket.recv()
            logger.info(f"Received: {response}")
            
            # Send a test message
            test_message = json.dumps({
                "type": "test",
                "message": "This is a test message",
                "timestamp": time.time()
            })
            
            logger.info(f"Sending: {test_message}")
            await websocket.send(test_message)
            
            # Receive echo response
            response = await websocket.recv()
            logger.info(f"Received: {response}")
            
            # Send a few more messages
            for i in range(3):
                test_message = json.dumps({
                    "type": "test",
                    "message": f"Test message {i+1}",
                    "timestamp": time.time()
                })
                
                logger.info(f"Sending: {test_message}")
                await websocket.send(test_message)
                
                # Receive echo response
                response = await websocket.recv()
                logger.info(f"Received: {response}")
                
                # Small delay between messages
                await asyncio.sleep(0.5)
            
            logger.info("WebSocket test completed successfully")
            
    except ConnectionRefusedError:
        logger.error(f"Connection refused. Make sure the server is running at {uri}")
        return 1
    except Exception as e:
        logger.error(f"WebSocket client error: {e}")
        return 1
    
    return 0

# Browser WebSocket test
def run_browser_websocket_test(host='localhost', port=8765, browser_name='chrome'):
    """Create a simple browser page that tests WebSocket connections."""
    
    # Create HTML file with WebSocket test code
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>WebSocket Test</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .logs {{ height: 300px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; background-color: #f8f8f8; font-family: monospace; }}
            .log-entry {{ margin-bottom: 5px; }}
            .log-info {{ color: #333; }}
            .log-error {{ color: #d9534f; }}
            .log-success {{ color: #5cb85c; }}
            .status {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .status.success {{ background-color: #dff0d8; color: #3c763d; }}
            .status.error {{ background-color: #f2dede; color: #a94442; }}
            .status.info {{ background-color: #d9edf7; color: #31708f; }}
            button {{ padding: 10px; margin: 5px; cursor: pointer; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>WebSocket Test</h1>
            
            <div id="status" class="status info">
                Initializing WebSocket test...
            </div>
            
            <div>
                <button id="connect">Connect</button>
                <button id="send">Send Test Message</button>
                <button id="disconnect">Disconnect</button>
            </div>
            
            <h2>Logs:</h2>
            <div id="logs" class="logs">
                <!-- Logs will be added here -->
            </div>
        </div>

        <script>
            // WebSocket variables
            let socket = null;
            let connected = false;
            
            // DOM Elements
            const logsContainer = document.getElementById('logs');
            const statusContainer = document.getElementById('status');
            const connectButton = document.getElementById('connect');
            const sendButton = document.getElementById('send');
            const disconnectButton = document.getElementById('disconnect');
            
            // Logging function
            function log(message, level = 'info') {{
                const logEntry = document.createElement('div');
                logEntry.className = 'log-entry log-' + level;
                logEntry.textContent = `[${{new Date().toLocaleTimeString()}}] ${{message}}`;
                logsContainer.appendChild(logEntry);
                logsContainer.scrollTop = logsContainer.scrollHeight;
                
                console.log(`[${{level}}] ${{message}}`);
            }}
            
            // Update status function
            function updateStatus(message, type = 'info') {{
                statusContainer.className = 'status ' + type;
                statusContainer.textContent = message;
            }}
            
            // Connect to WebSocket server
            function connectWebSocket() {{
                if (connected) {{
                    log('Already connected', 'info');
                    return;
                }}
                
                try {{
                    updateStatus('Connecting to WebSocket server...', 'info');
                    log('Connecting to WebSocket server at ws://{host}:{port}');
                    
                    socket = new WebSocket('ws://{host}:{port}');
                    
                    socket.onopen = function() {{
                        connected = true;
                        log('Connected to WebSocket server', 'success');
                        updateStatus('Connected to WebSocket server', 'success');
                    }};
                    
                    socket.onclose = function(event) {{
                        connected = false;
                        log(`Disconnected from WebSocket server: Code: ${{event.code}}, Reason: ${{event.reason || 'No reason provided'}}`, 'info');
                        updateStatus('Disconnected from WebSocket server', 'info');
                    }};
                    
                    socket.onerror = function(error) {{
                        log(`WebSocket error: ${{error}}`, 'error');
                        updateStatus('WebSocket connection error', 'error');
                    }};
                    
                    socket.onmessage = function(event) {{
                        log(`Received message: ${{event.data}}`, 'info');
                        
                        try {{
                            const data = JSON.parse(event.data);
                            log(`Parsed message type: ${{data.type}}`, 'info');
                        }} catch (e) {{
                            log(`Not a JSON message: ${{e}}`, 'error');
                        }}
                    }};
                    
                }} catch (e) {{
                    log(`Error creating WebSocket: ${{e}}`, 'error');
                    updateStatus(`Error creating WebSocket: ${{e}}`, 'error');
                }}
            }}
            
            // Send test message
            function sendTestMessage() {{
                if (!connected || !socket) {{
                    log('Not connected to WebSocket server', 'error');
                    updateStatus('Not connected to WebSocket server', 'error');
                    return;
                }}
                
                try {{
                    const message = {{
                        type: 'test',
                        message: 'Browser test message',
                        timestamp: Date.now()
                    }};
                    
                    log(`Sending message: ${{JSON.stringify(message)}}`, 'info');
                    socket.send(JSON.stringify(message));
                    
                }} catch (e) {{
                    log(`Error sending message: ${{e}}`, 'error');
                }}
            }}
            
            // Disconnect from WebSocket server
            function disconnectWebSocket() {{
                if (!connected || !socket) {{
                    log('Not connected to WebSocket server', 'info');
                    return;
                }}
                
                try {{
                    log('Closing WebSocket connection', 'info');
                    socket.close();
                    connected = false;
                    updateStatus('Disconnected from WebSocket server', 'info');
                    
                }} catch (e) {{
                    log(`Error closing WebSocket: ${{e}}`, 'error');
                }}
            }}
            
            // Wire up buttons
            connectButton.addEventListener('click', connectWebSocket);
            sendButton.addEventListener('click', sendTestMessage);
            disconnectButton.addEventListener('click', disconnectWebSocket);
            
            // Initialize
            log('WebSocket test page loaded', 'info');
            updateStatus('WebSocket test ready. Click "Connect" to begin.', 'info');
            
            // Detailed browser information
            log(`Browser: ${{navigator.userAgent}}`, 'info');
            log(`WebSocket supported: $${{typeof WebSocket !== 'undefined'}}`, 'info');
            
            // Show detailed WebSocket support
            if (typeof WebSocket !== 'undefined') {{
                log('WebSocket constructor is available', 'success');
                
                // Check for WebSocket protocol support
                const protocols = [];
                if (WebSocket.prototype.CONNECTING === 0) protocols.push('CONNECTING');
                if (WebSocket.prototype.OPEN === 1) protocols.push('OPEN');
                if (WebSocket.prototype.CLOSING === 2) protocols.push('CLOSING');
                if (WebSocket.prototype.CLOSED === 3) protocols.push('CLOSED');
                
                log(`WebSocket supports protocols: ${{protocols.join(', ')}}`, 'info');
            }} else {{
                log('WebSocket is not supported in this browser!', 'error');
            }}
        </script>
    </body>
    </html>
    """
    
    # Create a temporary HTML file
    fd, html_path = tempfile.mkstemp(suffix=".html")
    with os.fdopen(fd, "w") as f:
        f.write(html_content)
    
    logger.info(f"Created WebSocket test page at {html_path}")
    
    try:
        # Initialize browser
        if browser_name == 'chrome':
            options = ChromeOptions()
            service = ChromeService()
            driver = webdriver.Chrome(service=service, options=options)
        else:
            logger.error(f"Unsupported browser: {browser_name}")
            return 1
        
        file_url = f"file://{html_path}"
        logger.info(f"Opening WebSocket test page: {file_url}")
        driver.get(file_url)
        
        # Wait for user to manually test
        logger.info("Browser opened with WebSocket test page")
        logger.info("Use the browser to test WebSocket connections")
        logger.info("Press Enter to close the browser and finish the test")
        input()
        
        driver.quit()
        logger.info("Browser closed")
        
    except Exception as e:
        logger.error(f"Error in browser test: {e}")
        return 1
    finally:
        # Clean up temporary file
        try:
            os.unlink(html_path)
        except:
            pass
    
    return 0

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="WebSocket Diagnostic Tool")
    parser.add_argument("--server", action="store_true", help="Run WebSocket server")
    parser.add_argument("--client", action="store_true", help="Run WebSocket client")
    parser.add_argument("--browser", action="store_true", help="Run browser WebSocket test")
    parser.add_argument("--host", type=str, default="localhost", help="WebSocket host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--browser-name", type=str, default="chrome", help="Browser to use for test")
    
    args = parser.parse_args()
    
    # If no specific command is given, print help
    if not (args.server or args.client or args.browser):
        parser.print_help()
        return 1
    
    # Run requested test
    if args.server:
        return await run_websocket_server(args.host, args.port)
    
    if args.client:
        return await run_websocket_client(args.host, args.port)
    
    if args.browser:
        import tempfile
        return run_browser_websocket_test(args.host, args.port, args.browser_name)
    
    return 0

if __name__ == "__main__":
    asyncio.run(main())