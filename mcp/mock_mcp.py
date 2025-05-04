"""
Mock implementation of MCP (Model Context Protocol) components.

This module provides mock implementations of the FastMCP server and Context objects
to allow the IPFS Accelerate MCP integration to run without external dependencies.
"""

import asyncio
import json
import sys
import inspect
from typing import Any, Dict, List, Optional, Callable, AsyncGenerator, TypeVar, cast
from dataclasses import dataclass
from contextlib import asynccontextmanager


class Context:
    """Mock implementation of MCP Context.
    
    This class simulates the Context object used by FastMCP to provide
    context for tool executions and information about the current request.
    """
    
    def __init__(self, request_id: str = "mock-request-id"):
        """Initialize a mock Context.
        
        Args:
            request_id: A unique identifier for the request
        """
        self.request_id = request_id
        self.request_context = MockRequestContext()
    
    async def info(self, message: str) -> None:
        """Log an informational message.
        
        Args:
            message: The message to log
        """
        print(f"[INFO] {message}")
    
    async def error(self, message: str) -> None:
        """Log an error message.
        
        Args:
            message: The error message to log
        """
        print(f"[ERROR] {message}")
    
    async def report_progress(self, current: int, total: int) -> None:
        """Report progress for a long-running operation.
        
        Args:
            current: Current progress value
            total: Total progress value
        """
        percentage = 100 * current / total
        print(f"[PROGRESS] {percentage:.1f}% ({current}/{total})")


class MockRequestContext:
    """Mock implementation of MCP RequestContext.
    
    This class simulates the RequestContext object that provides access to
    the lifespan context and other request-specific information.
    """
    
    def __init__(self):
        """Initialize a mock RequestContext."""
        self.lifespan_context = {}


class MockLifespan:
    """A simple mock async context manager for use as a lifespan."""
    
    def __init__(self, context_obj: Any = None):
        """Initialize with optional context object."""
        self.context_obj = context_obj or {}
        
    async def __aenter__(self):
        """Enter the context manager and return the context."""
        return self.context_obj
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        pass


class FastMCP:
    """Mock implementation of FastMCP server.
    
    This class simulates the FastMCP server that provides the MCP protocol
    implementation and handles tool registration and execution.
    """
    
    def __init__(self, name: str, dependencies: Optional[List[str]] = None, 
                 lifespan: Any = None):
        """Initialize a mock FastMCP server.
        
        Args:
            name: The name of the MCP server
            dependencies: Optional list of dependencies
            lifespan: Optional lifespan context manager
        """
        self.name = name
        self.dependencies = dependencies or []
        self.lifespan = lifespan
        self.tools: Dict[str, Callable] = {}
        self.resources: Dict[str, Callable] = {}
        self.prompts: Dict[str, Callable] = {}
    
    def tool(self, **kwargs):
        """Decorator for registering a tool with the MCP server.
        
        Args:
            **kwargs: Optional additional arguments for tool registration
            
        Returns:
            Decorator function for registering tools
        """
        def decorator(func):
            tool_name = func.__name__
            self.tools[tool_name] = func
            print(f"Registered tool: {tool_name}")
            return func
        return decorator
    
    def resource(self, uri: str, **kwargs):
        """Decorator for registering a resource with the MCP server.
        
        Args:
            uri: The URI of the resource
            **kwargs: Optional additional arguments for resource registration
            
        Returns:
            Decorator function for registering resources
        """
        def decorator(func):
            self.resources[uri] = func
            print(f"Registered resource: {uri}")
            return func
        return decorator
    
    def prompt(self, **kwargs):
        """Decorator for registering a prompt with the MCP server.
        
        Args:
            **kwargs: Optional additional arguments for prompt registration
            
        Returns:
            Decorator function for registering prompts
        """
        def decorator(func):
            prompt_name = func.__name__
            self.prompts[prompt_name] = func
            print(f"Registered prompt: {prompt_name}")
            return func
        return decorator
    
    def run(self, transport: str = "stdio", **kwargs):
        """Run the MCP server with the specified transport.
        
        Args:
            transport: The transport protocol to use
            **kwargs: Additional arguments for the transport
        """
        print(f"Running mock MCP server '{self.name}' with {transport} transport")
        print(f"Additional arguments: {kwargs}")
        print(f"Registered {len(self.tools)} tools, {len(self.resources)} resources, and {len(self.prompts)} prompts")
        
        for tool_name in self.tools:
            print(f"- Tool: {tool_name}")


def create_mock_lifespan(context_obj: Any = None):
    """Create a mock lifespan context manager.
    
    Args:
        context_obj: Optional object to use as the context
        
    Returns:
        A mock lifespan context manager
    """
    return MockLifespan(context_obj)
