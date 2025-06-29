#!/usr/bin/env python
"""
IPFS Accelerate MCP SSE Client Example

This module demonstrates how to use Server-Sent Events (SSE) to receive real-time updates
from the MCP server.
"""

import os
import sys
import json
import logging
import time
import asyncio
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def connect_to_sse(host: str = "localhost", port: int = 8002):
    """
    Connect to the SSE endpoint and print updates
    
    Args:
        host: Server host
        port: Server port
    """
    try:
        # Import aiohttp
        import aiohttp
    except ImportError:
        logger.error("This example requires aiohttp. Install it with: pip install aiohttp")
        return
    
    url = f"http://{host}:{port}/sse"
    logger.info(f"Connecting to SSE endpoint: {url}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                logger.info(f"Connected to SSE endpoint, status: {response.status}")
                
                if response.status != 200:
                    logger.error(f"Error connecting to SSE endpoint: {await response.text()}")
                    return
                
                print("Waiting for SSE updates... (Press Ctrl+C to exit)")
                print("----------------------------------------------------")
                
                # Process SSE events
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Process SSE data
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        try:
                            data = json.loads(data_str)
                            
                            # Format the timestamp
                            timestamp = data.get('timestamp', '')
                            if timestamp:
                                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
                            
                            # Print the event type
                            event_type = data.get('type', 'unknown')
                            print(f"\n[{timestamp}] Event Type: {event_type}")
                            
                            # Format based on event type
                            if event_type == 'hardware_update':
                                hardware_data = data.get('data', {})
                                print("Hardware Update:")
                                
                                # CPU usage
                                cpu_usage = hardware_data.get('cpu_usage')
                                if cpu_usage is not None:
                                    print(f"  CPU Usage: {cpu_usage:.1f}%")
                                
                                # Memory usage
                                memory = hardware_data.get('memory', {})
                                if memory:
                                    used = memory.get('used', 0)
                                    total = memory.get('total', 1)
                                    used_gb = used / (1024 ** 3)
                                    total_gb = total / (1024 ** 3)
                                    percent = memory.get('percent', 0)
                                    print(f"  Memory: {used_gb:.1f} GB / {total_gb:.1f} GB ({percent:.1f}%)")
                                
                                # GPU usage
                                gpu = hardware_data.get('gpu', {})
                                if gpu:
                                    print(f"  GPU: {gpu.get('name', 'Unknown')}")
                                    gpu_usage = gpu.get('usage')
                                    if gpu_usage is not None:
                                        print(f"  GPU Usage: {gpu_usage:.1f}%")
                            
                            elif event_type == 'system_update':
                                system_data = data.get('data', {})
                                print("System Update:")
                                for key, value in system_data.items():
                                    print(f"  {key}: {value}")
                            
                            elif event_type == 'model_update':
                                model_data = data.get('data', {})
                                print("Model Update:")
                                model_name = model_data.get('name', 'Unknown')
                                status = model_data.get('status', 'Unknown')
                                print(f"  Model: {model_name}")
                                print(f"  Status: {status}")
                                if 'progress' in model_data:
                                    print(f"  Progress: {model_data['progress']:.1f}%")
                            
                            else:
                                # Generic print for other events
                                print("Data:")
                                print(json.dumps(data.get('data', {}), indent=2))
                        
                        except json.JSONDecodeError:
                            print(f"Received non-JSON data: {data_str}")
                    
                    # Process event type
                    elif line.startswith('event: '):
                        event_type = line[7:]  # Remove 'event: ' prefix
                        logger.debug(f"Event type: {event_type}")
                    
                    # Process event ID
                    elif line.startswith('id: '):
                        event_id = line[4:]  # Remove 'id: ' prefix
                        logger.debug(f"Event ID: {event_id}")
                    
                    # Handle reconnection time
                    elif line.startswith('retry: '):
                        retry = int(line[7:])  # Remove 'retry: ' prefix
                        logger.debug(f"Retry after: {retry} ms")
    
    except aiohttp.ClientError as e:
        logger.error(f"Error connecting to SSE endpoint: {e}")
    
    except asyncio.CancelledError:
        logger.info("SSE client cancelled")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    
    finally:
        logger.info("Disconnected from SSE endpoint")

def check_server():
    """Check if the MCP server is running and start it if needed"""
    try:
        # Add parent directory to path if needed
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from mcp.client import is_server_running, start_server
        
        # Check if the server is running
        port = 8002
        if not is_server_running(port=port):
            logger.info(f"MCP server is not running on port {port}, starting...")
            success, port = start_server(port=port, wait=2)
            if not success:
                logger.error("Failed to start MCP server")
                return False
            
            logger.info(f"MCP server started on port {port}")
        else:
            logger.info(f"MCP server is already running on port {port}")
        
        return True
    
    except ImportError as e:
        logger.error(f"Could not import MCP client: {e}")
        return False
    
    except Exception as e:
        logger.error(f"Error checking/starting server: {e}")
        return False

def main():
    """Main function"""
    # Check if the server is running and start it if needed
    if not check_server():
        sys.exit(1)
    
    # Connect to SSE and print updates
    try:
        asyncio.run(connect_to_sse())
    except KeyboardInterrupt:
        logger.info("SSE client stopped by user")

if __name__ == "__main__":
    main()
