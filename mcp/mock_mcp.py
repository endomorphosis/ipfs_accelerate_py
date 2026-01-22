"""
Mock implementation of the FastMCP server.

This module provides a mock implementation of the FastMCP server interface
for use when the actual FastMCP package is not available. This enables
testing and development without the full MCP infrastructure.
"""

import asyncio
import inspect
import json
import logging
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type, TypeVar, Union, cast

# Configure logging
logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')
LifespanResultT = TypeVar('LifespanResultT')
ToolFuncT = Callable[..., Awaitable[Any]]
LifespanStartFuncT = Callable[['Context'], Awaitable[LifespanResultT]]
LifespanStopFuncT = Callable[['Context', LifespanResultT], Awaitable[None]]


class TransportType(str, Enum):
    """Available transport types for MCP server."""
    STDIO = "stdio"
    SSE = "sse"


class Context:
    """Mock context object for tool execution.
    
    This simulates the MCP context that's passed to tools for tracking state
    and reporting progress.
    """
    
    def __init__(self):
        """Initialize a new context."""
        self.request_context = type('RequestContext', (), {
            'lifespan_context': type('LifespanContext', (), {
                'ipfs_context': None
            })
        })
        self.start_time = time.time()
        self._progress = 0.0
        self._total = 1.0
    
    async def info(self, message: str) -> None:
        """Log an informational message."""
        logger.info(message)
    
    async def error(self, message: str) -> None:
        """Log an error message."""
        logger.error(message)
    
    async def warning(self, message: str) -> None:
        """Log a warning message."""
        logger.warning(message)
    
    async def report_progress(self, progress: float, total: float = 1.0) -> None:
        """Report tool execution progress."""
        self._progress = progress
        self._total = total
        percentage = (progress / total) * 100 if total > 0 else 0
        logger.info(f"Progress: {percentage:.1f}% ({progress}/{total})")
    
    async def get_progress(self) -> Dict[str, float]:
        """Get the current progress."""
        return {
            "progress": self._progress,
            "total": self._total,
            "percentage": (self._progress / self._total) * 100 if self._total > 0 else 0
        }


class FastMCP:
    """Mock implementation of the FastMCP server.
    
    This class simulates the behavior of a FastMCP server for testing and
    development when the actual FastMCP package is not available.
    """
    
    def __init__(self, name: str = "Mock MCP Server", description: str = ""):
        """Initialize a new mock MCP server.
        
        Args:
            name: Server name
            description: Server description
        """
        self.name = name
        self.description = description or f"Mock implementation of {name}"
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.lifespan_start_handler: Optional[LifespanStartFuncT] = None
        self.lifespan_stop_handler: Optional[LifespanStopFuncT] = None
        self.lifespan_context: Any = None
        logger.info(f"Initialized mock MCP server: {name}")
    
    def tool(self, 
             name: Optional[str] = None, 
             description: Optional[str] = None) -> Callable[[ToolFuncT], ToolFuncT]:
        """Decorator to register a tool with the server.
        
        Args:
            name: Tool name (defaults to function name)
            description: Tool description (defaults to function docstring)
            
        Returns:
            Decorator function
        """
        def decorator(func: ToolFuncT) -> ToolFuncT:
            tool_name = name or func.__name__
            tool_desc = description or inspect.getdoc(func) or f"Tool: {tool_name}"
            
            # Generate schema from function signature
            sig = inspect.signature(func)
            params = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                if param_name == 'ctx' or param_name == 'context':
                    continue  # Skip context parameter
                
                param_info = {
                    "title": param_name,
                    "type": "string"  # Simplified - we'd normally infer type
                }
                
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)
                
                params[param_name] = param_info
            
            # Register the tool
            self.tools[tool_name] = {
                "name": tool_name,
                "description": tool_desc,
                "function": func,
                "schema": {
                    "type": "object",
                    "properties": params,
                    "required": required,
                    "title": f"{tool_name}Arguments"
                }
            }
            
            logger.info(f"Registered tool: {tool_name}")
            return func
        
        return decorator
    
    def on_lifespan_start(self) -> Callable[[LifespanStartFuncT], LifespanStartFuncT]:
        """Decorator to register a lifespan start handler.
        
        Returns:
            Decorator function
        """
        def decorator(func: LifespanStartFuncT) -> LifespanStartFuncT:
            self.lifespan_start_handler = func
            logger.info("Registered lifespan start handler")
            return func
        
        return decorator
    
    def on_lifespan_stop(self) -> Callable[[LifespanStopFuncT], LifespanStopFuncT]:
        """Decorator to register a lifespan stop handler.
        
        Returns:
            Decorator function
        """
        def decorator(func: LifespanStopFuncT) -> LifespanStopFuncT:
            self.lifespan_stop_handler = func
            logger.info("Registered lifespan stop handler")
            return func
        
        return decorator
    
    async def start(self) -> Any:
        """Start the server and invoke the lifespan start handler.
        
        Returns:
            The result of the lifespan start handler, or None if not defined
        """
        logger.info(f"Starting mock MCP server: {self.name}")
        
        # Create a context
        ctx = Context()
        
        # Call lifespan start handler if defined
        if self.lifespan_start_handler:
            logger.info("Invoking lifespan start handler")
            self.lifespan_context = await self.lifespan_start_handler(ctx)
            return self.lifespan_context
        
        return None
    
    async def stop(self) -> None:
        """Stop the server and invoke the lifespan stop handler."""
        logger.info(f"Stopping mock MCP server: {self.name}")
        
        # Create a context
        ctx = Context()
        
        # Call lifespan stop handler if defined
        if self.lifespan_stop_handler and self.lifespan_context is not None:
            logger.info("Invoking lifespan stop handler")
            await self.lifespan_stop_handler(ctx, self.lifespan_context)
    
    async def run(self, 
                  transport: Optional[Union[TransportType, str]] = None,
                  host: str = "127.0.0.1",
                  port: int = 8000) -> None:
        """Run the server using the specified transport.
        
        Args:
            transport: Transport type to use
            host: Host to bind to (for network transports)
            port: Port to bind to (for network transports)
        """
        logger.info(f"Running mock MCP server with transport: {transport}")
        
        # Start the server
        await self.start()
        
        try:
            # In a real server, we'd enter a request handling loop here
            # For the mock, we just log and return immediately
            logger.info(f"Mock MCP server ready: {self.name}")
            logger.info(f"Available tools: {', '.join(self.tools.keys())}")
            
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Server interrupted")
        finally:
            # Stop the server
            await self.stop()
    
    def get_tool_schemas(self) -> Dict[str, Any]:
        """Get schemas for all registered tools.
        
        Returns:
            Dictionary of tool schemas
        """
        schemas = {}
        for name, tool in self.tools.items():
            schemas[name] = tool["schema"]
        
        return schemas
    
    async def invoke_tool(self, 
                          name: str, 
                          args: Dict[str, Any]) -> Any:
        """Invoke a tool by name with the provided arguments.
        
        Args:
            name: Tool name
            args: Tool arguments
            
        Returns:
            Tool result
            
        Raises:
            ValueError: If the tool is not found
        """
        if name not in self.tools:
            raise ValueError(f"Tool not found: {name}")
        
        tool = self.tools[name]
        func = tool["function"]
        
        # Create a context
        ctx = Context()
        
        # Add context to args
        args["ctx"] = ctx
        
        # Invoke the tool
        logger.info(f"Invoking tool: {name}")
        start_time = time.time()
        result = await func(**args)
        duration = time.time() - start_time
        logger.info(f"Tool completed in {duration:.2f}s: {name}")
        
        return result
