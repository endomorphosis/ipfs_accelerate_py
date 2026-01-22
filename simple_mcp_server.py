#!/usr/bin/env python3
"""
Simple MCP Server for VS Code Integration

This is a minimal MCP server that works with VS Code's MCP extension.
It provides basic functionality without complex dependencies.
"""

import asyncio
import json
import logging
import sys
import os
from typing import Any, Dict, List, Optional

# Add the parent directory to the path so we can import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Try to import mcp, fall back to simple implementation if not available
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleMCPServer:
    """Simple MCP server for VS Code integration."""
    
    def __init__(self):
        self.tools = []
        self._register_tools()
    
    def _register_tools(self):
        """Register available tools."""
        self.tools = [
            {
                "name": "list_models",
                "description": "List available AI models",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_server_info", 
                "description": "Get information about the MCP server",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "test_inference",
                "description": "Test AI inference with sample data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "model_type": {
                            "type": "string",
                            "description": "Type of model to test (text, image, audio)"
                        },
                        "input_text": {
                            "type": "string", 
                            "description": "Input text for processing"
                        }
                    }
                }
            }
        ]
    
    async def handle_list_tools(self) -> List[Dict]:
        """Handle list_tools request."""
        return self.tools
    
    async def handle_call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle call_tool request."""
        if name == "list_models":
            return [TextContent(
                type="text",
                text=json.dumps({
                    "models": [
                        {"id": "text-generator", "type": "text", "description": "Text generation model"},
                        {"id": "image-classifier", "type": "vision", "description": "Image classification model"},
                        {"id": "audio-transcriber", "type": "audio", "description": "Audio transcription model"}
                    ],
                    "total": 3
                }, indent=2)
            )]
        
        elif name == "get_server_info":
            return [TextContent(
                type="text", 
                text=json.dumps({
                    "name": "IPFS Accelerate MCP Server",
                    "version": "1.0.0",
                    "description": "AI inference server with IPFS acceleration",
                    "capabilities": ["text_processing", "image_processing", "audio_processing"],
                    "status": "running"
                }, indent=2)
            )]
        
        elif name == "test_inference":
            model_type = arguments.get("model_type", "text")
            input_text = arguments.get("input_text", "Hello, world!")
            
            result = {
                "model_type": model_type,
                "input": input_text,
                "output": f"Processed '{input_text}' using {model_type} model",
                "confidence": 0.95,
                "processing_time": "0.5s"
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        else:
            raise ValueError(f"Unknown tool: {name}")

async def main():
    """Main entry point for the MCP server."""
    
    if not MCP_AVAILABLE:
        logger.error("MCP library not available. Please install: pip install mcp")
        sys.exit(1)
    
    # Create the simple server
    simple_server = SimpleMCPServer()
    
    # Create MCP server
    server = Server("ipfs-accelerate-simple")
    
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available tools."""
        tools_data = await simple_server.handle_list_tools()
        return [Tool(**tool) for tool in tools_data]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Call a tool."""
        return await simple_server.handle_call_tool(name, arguments)
    
    # Run the server
    logger.info("Starting simple MCP server for VS Code...")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, 
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    # Handle the case where asyncio loop might already be running
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # If we're in a running loop, just run the coroutine directly
            import asyncio
            loop = asyncio.get_event_loop()
            loop.create_task(main())
        else:
            raise