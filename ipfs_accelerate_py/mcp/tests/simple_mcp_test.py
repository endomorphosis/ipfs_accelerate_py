"""
Simple MCP Server Test

This script creates a minimal MCP server to diagnose component registration issues.
"""

import sys
from fastmcp import FastMCP

# Create a simple MCP server
server = FastMCP(
    "Test Server",
    description="A simple test server to diagnose MCP registration issues",
)

# Register a simple tool
@server.tool()
def test_tool(message: str = "Hello") -> str:
    """A simple test tool that echoes a message"""
    return f"Echo: {message}"

# Register a simple resource
@server.resource("test://resource")
def test_resource() -> dict:
    """A simple test resource"""
    return {"status": "ok", "message": "This is a test resource"}

# Register a simple prompt
@server.prompt()
def test_prompt() -> str:
    """A simple test prompt"""
    return "This is a test prompt. Please respond with 'Hello, world!'"

# Print server information for debugging
def print_server_info():
    """Print information about registered components"""
    print("\n===== SERVER INFO =====")
    
    # Try to access tools
    print("\n--- Tools ---")
    if hasattr(server, "_tools"):
        print(f"Tools attribute type: {type(server._tools)}")
        print(f"Number of tools: {len(server._tools)}")
        for i, tool in enumerate(server._tools):
            print(f"Tool {i}: {tool}")
            # Try to extract tool name
            try:
                print(f"  Name: {tool.name}")
            except Exception as e:
                print(f"  Error accessing name: {e}")
    else:
        print("No tools attribute found")
    
    # Try to access resources
    print("\n--- Resources ---")
    if hasattr(server, "_resources"):
        print(f"Resources attribute type: {type(server._resources)}")
        print(f"Number of resources: {len(server._resources)}")
        for i, resource in enumerate(server._resources):
            print(f"Resource {i}: {resource}")
            # Try to extract resource URI
            try:
                print(f"  URI: {resource.uri}")
            except Exception as e:
                print(f"  Error accessing uri: {e}")
    else:
        print("No resources attribute found")
    
    # Try to access prompts
    print("\n--- Prompts ---")
    if hasattr(server, "_prompts"):
        print(f"Prompts attribute type: {type(server._prompts)}")
        print(f"Number of prompts: {len(server._prompts)}")
        for i, prompt in enumerate(server._prompts):
            print(f"Prompt {i}: {prompt}")
            # Try to extract prompt name
            try:
                print(f"  Name: {prompt.name}")
            except Exception as e:
                print(f"  Error accessing name: {e}")
    else:
        print("No prompts attribute found")

# Print registered components for debugging
print_server_info()

# Attempt to use the mock MCP verification client
def test_mcp_client():
    """Test mock client operations to diagnose the issue"""
    try:
        from mcp.server.lowlevel.types import (
            ListToolsRequest, ListToolsResponse,
            ListResourcesRequest, ListResourcesResponse,
            ListPromptsRequest, ListPromptsResponse
        )
        
        print("\n===== MOCK CLIENT TEST =====")
        
        # Test listing tools
        print("\nTesting ListToolsRequest/Response:")
        tools_request = ListToolsRequest()
        tools_response = server.process_request(tools_request)
        print(f"Response type: {type(tools_response)}")
        if isinstance(tools_response, ListToolsResponse):
            print(f"Number of tools: {len(tools_response.tools)}")
            for i, tool in enumerate(tools_response.tools):
                print(f"Tool {i} type: {type(tool)}")
                try:
                    print(f"  Name: {tool.name}")
                except Exception as e:
                    print(f"  Error accessing name: {e}")
        
        # Test listing resources
        print("\nTesting ListResourcesRequest/Response:")
        resources_request = ListResourcesRequest()
        resources_response = server.process_request(resources_request)
        print(f"Response type: {type(resources_response)}")
        if isinstance(resources_response, ListResourcesResponse):
            print(f"Number of resources: {len(resources_response.resources)}")
            for i, resource in enumerate(resources_response.resources):
                print(f"Resource {i} type: {type(resource)}")
                try:
                    print(f"  URI: {resource.uri}")
                except Exception as e:
                    print(f"  Error accessing uri: {e}")
        
        # Test listing prompts
        print("\nTesting ListPromptsRequest/Response:")
        prompts_request = ListPromptsRequest()
        prompts_response = server.process_request(prompts_request)
        print(f"Response type: {type(prompts_response)}")
        if isinstance(prompts_response, ListPromptsResponse):
            print(f"Number of prompts: {len(prompts_response.prompts)}")
            for i, prompt in enumerate(prompts_response.prompts):
                print(f"Prompt {i} type: {type(prompt)}")
                try:
                    print(f"  Name: {prompt.name}")
                except Exception as e:
                    print(f"  Error accessing name: {e}")
    
    except ImportError as e:
        print(f"Error importing MCP types: {e}")
    except Exception as e:
        print(f"Error testing MCP client: {e}")

# Run the MCP client test
test_mcp_client()

# Print the MCP library versions for debugging
print("\n===== MCP LIBRARY VERSIONS =====")
print(f"Python version: {sys.version}")
try:
    import fastmcp
    print(f"fastmcp version: {fastmcp.__version__ if hasattr(fastmcp, '__version__') else 'unknown'}")
except ImportError:
    print("fastmcp not installed")

try:
    import mcp
    print(f"mcp version: {mcp.__version__ if hasattr(mcp, '__version__') else 'unknown'}")
except ImportError:
    print("mcp not installed")

# Only run the server if executed directly
if __name__ == "__main__":
    print("\nStarting server (Ctrl+C to exit)...")
    server.run()
