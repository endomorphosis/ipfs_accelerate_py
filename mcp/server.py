"""
Main MCP server implementation for IPFS Accelerate.

This module provides the core FastMCP server that exposes IPFS Accelerate
functionality to LLM clients through the Model Context Protocol.
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, cast, Callable

# Try to import FastMCP and Context from the actual packages
# Fall back to mock implementations if not available
try:
    from mcp.server.fastmcp import FastMCP, Context
except ImportError:
    # Fall back to the standalone FastMCP if MCP SDK is not installed
    try:
        from fastmcp import FastMCP, Context
    except ImportError:
        # Provide mock implementations directly
        print("Using mock MCP implementation as neither 'mcp' nor 'fastmcp' packages are installed.")
        print("Install with 'pip install mcp[cli]' or 'pip install fastmcp' for full functionality.")
        
        # Mock Context class
        class Context:
            """Mock implementation of MCP Context."""
            
            def __init__(self, request_id: str = "mock-request-id"):
                """Initialize a mock Context."""
                self.request_id = request_id
                self.request_context = MockRequestContext()
            
            async def info(self, message: str) -> None:
                """Log an informational message."""
                print(f"[INFO] {message}")
            
            async def error(self, message: str) -> None:
                """Log an error message."""
                print(f"[ERROR] {message}")
            
            async def report_progress(self, current: int, total: int) -> None:
                """Report progress for a long-running operation."""
                percentage = 100 * current / total
                print(f"[PROGRESS] {percentage:.1f}% ({current}/{total})")


        class MockRequestContext:
            """Mock implementation of MCP RequestContext."""
            
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
            """Mock implementation of FastMCP server."""
            
            def __init__(self, name: str, dependencies: Optional[List[str]] = None, 
                        lifespan: Any = None):
                """Initialize a mock FastMCP server."""
                self.name = name
                self.dependencies = dependencies or []
                self.lifespan = lifespan
                self.tools: Dict[str, Callable] = {}
                self.resources: Dict[str, Callable] = {}
                self.prompts: Dict[str, Callable] = {}
            
            def tool(self, **kwargs):
                """Decorator for registering a tool with the MCP server."""
                def decorator(func):
                    tool_name = func.__name__
                    self.tools[tool_name] = func
                    print(f"Registered tool: {tool_name}")
                    return func
                return decorator
            
            def resource(self, uri: str, **kwargs):
                """Decorator for registering a resource with the MCP server."""
                def decorator(func):
                    self.resources[uri] = func
                    print(f"Registered resource: {uri}")
                    return func
                return decorator
            
            def prompt(self, **kwargs):
                """Decorator for registering a prompt with the MCP server."""
                def decorator(func):
                    prompt_name = func.__name__
                    self.prompts[prompt_name] = func
                    print(f"Registered prompt: {prompt_name}")
                    return func
                return decorator
            
            def run(self, transport: str = "stdio", **kwargs):
                """Run the MCP server with the specified transport."""
                print(f"Running mock MCP server '{self.name}' with {transport} transport")
                print(f"Additional arguments: {kwargs}")
                print(f"Registered {len(self.tools)} tools, {len(self.resources)} resources, and {len(self.prompts)} prompts")
                
                for tool_name in self.tools:
                    print(f"- Tool: {tool_name}")


# Try to import ipfs_accelerate_py, but don't fail if it's not available
try:
    import ipfs_accelerate_py
except ImportError:
    # Create a placeholder module
    class DummyModule:
        __version__ = "0.0.0-mock"
    ipfs_accelerate_py = DummyModule()


# Define IPFSAccelerateContext here to avoid import issues
@dataclass
class IPFSAccelerateContext:
    """Context object for the IPFS Accelerate MCP server."""
    config: Dict[str, Any]
    ipfs_client: Any = None


@asynccontextmanager
async def ipfs_accelerate_lifespan(server: FastMCP) -> AsyncIterator[IPFSAccelerateContext]:
    """Manage the lifecycle of IPFS Accelerate resources and connections.
    
    This context manager initializes necessary resources when the server starts
    and ensures they're properly cleaned up when the server shuts down.
    """
    # Initialize IPFS client and acceleration context
    config = {
        "ipfs_api_endpoint": "/ip4/127.0.0.1/tcp/5001",
        "acceleration_enabled": True,
        # Additional configuration options
    }
    
    # TODO: Initialize actual IPFS client and connections
    ipfs_client = None  # Replace with actual initialization
    
    try:
        # Create and yield the context
        context = IPFSAccelerateContext(
            config=config,
            ipfs_client=ipfs_client,
        )
        yield context
    finally:
        # Cleanup resources
        if ipfs_client:
            # Close connections, release resources, etc.
            pass


def create_ipfs_mcp_server(name: str = "IPFS Accelerate", 
                          dependencies: Optional[List[str]] = None) -> FastMCP:
    """Create and configure the IPFS Accelerate MCP server.
    
    Args:
        name: The name of the MCP server
        dependencies: Optional list of additional dependencies
        
    Returns:
        Configured FastMCP server instance
    """
    if dependencies is None:
        dependencies = []
    
    # Ensure required dependencies are included
    all_dependencies = list(set(dependencies + [
        "ipfs-accelerate-py",  # Replace with actual package name if different
        "aiohttp",
        "pydantic>=2.0.0",
    ]))
    
    # Create the server
    mcp = FastMCP(
        name, 
        dependencies=all_dependencies,
        lifespan=ipfs_accelerate_lifespan
    )
    
    # Register tools directly
    _register_tools(mcp)
    _register_resources(mcp)
    _register_prompts(mcp)
    
    return mcp


def _register_tools(mcp: FastMCP) -> None:
    """Register IPFS Accelerate tools with the MCP server.
    
    This is a placeholder that will be expanded as we implement tool modules.
    """
    # Basic IPFS file tool examples - these will be moved to dedicated modules
    
    @mcp.tool()
    async def ipfs_status(ctx: Context) -> Dict[str, Any]:
        """Get the current status of the IPFS node.
        
        Returns:
            Dictionary with IPFS node status information
        """
        # Access the lifespan context
        ipfs_ctx = cast(IPFSAccelerateContext, ctx.request_context.lifespan_context)
        
        # Example implementation
        return {
            "status": "online",
            "peer_count": 0,  # Will be implemented with actual peer count
            "acceleration_enabled": ipfs_ctx.config["acceleration_enabled"],
            "version": ipfs_accelerate_py.__version__,
        }
    
    
    @mcp.tool()
    async def ipfs_add(path: str, ctx: Context) -> Dict[str, Any]:
        """Add a file to IPFS.
        
        Args:
            path: Path to the file to add
            
        Returns:
            Dictionary with CID and size information
        """
        # Placeholder implementation
        await ctx.info(f"Adding file: {path}")
        
        # This will be replaced with actual implementation
        return {
            "cid": "QmExample...",
            "size": 1024,
            "name": path.split("/")[-1]
        }


    @mcp.tool()
    async def ipfs_files_ls(path: str, ctx: Context) -> List[Dict[str, Any]]:
        """List files in a directory in the IPFS MFS.
        
        Args:
            path: Path to the directory to list
            
        Returns:
            List of file information dictionaries
        """
        # Placeholder implementation
        await ctx.info(f"Listing directory: {path}")
        
        # This will be replaced with actual implementation
        return [
            {"name": "example.txt", "type": "file", "size": 1024, "cid": "QmExample1..."},
            {"name": "subdir", "type": "directory", "size": 0, "cid": "QmExample2..."}
        ]


    @mcp.tool()
    async def ipfs_accelerate_model(cid: str, ctx: Context) -> Dict[str, Any]:
        """Accelerate an AI model stored on IPFS.
        
        Args:
            cid: Content identifier of the model to accelerate
            
        Returns:
            Status of the acceleration operation
        """
        await ctx.info(f"Accelerating model with CID: {cid}")
        
        # This will be replaced with actual implementation
        return {
            "cid": cid,
            "accelerated": True,
            "device": "GPU",
            "status": "Acceleration successfully applied"
        }


def _register_resources(mcp: FastMCP) -> None:
    """Register IPFS Accelerate resources with the MCP server.
    
    This is a placeholder that will be expanded as we implement resource modules.
    """
    # Basic IPFS resource examples - these will be moved to dedicated modules
    
    @mcp.resource("ipfs://status")
    async def get_ipfs_status(ctx: Context) -> str:
        """Get the current status of the IPFS node as a formatted string.
        
        Returns:
            Formatted status information
        """
        # Access the lifespan context
        ipfs_ctx = cast(IPFSAccelerateContext, ctx.request_context.lifespan_context)
        
        # Example implementation
        return f"""
        IPFS Accelerate Status
        ---------------------
        Version: {ipfs_accelerate_py.__version__}
        Acceleration: {'Enabled' if ipfs_ctx.config['acceleration_enabled'] else 'Disabled'}
        IPFS API: {ipfs_ctx.config['ipfs_api_endpoint']}
        """


def _register_prompts(mcp: FastMCP) -> None:
    """Register IPFS Accelerate prompts with the MCP server.
    
    This is a placeholder that will be expanded as we implement prompt modules.
    """
    # Basic IPFS prompt examples - these will be moved to dedicated modules
    
    @mcp.prompt()
    def ipfs_help(topic: str = "general") -> str:
        """Generate a help prompt for IPFS operations.
        
        Args:
            topic: The topic to get help with
            
        Returns:
            A formatted help prompt
        """
        topics = {
            "general": "IPFS is a peer-to-peer hypermedia protocol designed to make the web faster, safer, and more open. With IPFS Accelerate, you can leverage hardware acceleration for AI models stored on IPFS.",
            "add": "To add a file to IPFS, you can use the ipfs_add tool with the path to your file.",
            "pin": "Pinning ensures that files are kept in your local IPFS repository and not garbage collected."
        }
        
        return f"""
        # IPFS Help: {topic}
        
        {topics.get(topic.lower(), "Topic not found. Try 'general', 'add', or 'pin'.")}
        
        What specific IPFS operation would you like help with?
        """


# Convenience instance for direct usage
default_server = create_ipfs_mcp_server()


if __name__ == "__main__":
    # Run the server directly when this module is executed as a script
    default_server.run()
