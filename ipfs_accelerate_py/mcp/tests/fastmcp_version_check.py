"""
Check FastMCP Version and API

This script checks the FastMCP version and API structure to help debug issues.
"""

import inspect
from fastmcp import FastMCP

# Create test server
server = FastMCP("Test")

# Test tool registration
@server.tool()
def test_tool():
    """Test tool"""
    return "Test"

# Test resource registration
@server.resource("test://resource")
def test_resource():
    """Test resource"""
    return {"status": "ok"}

# Print FastMCP information
print("=== FastMCP Version Info ===")
import fastmcp
print(f"FastMCP version: {getattr(fastmcp, '__version__', 'unknown')}")

# Inspect FastMCP
print("\n=== FastMCP Class Structure ===")
print(f"FastMCP methods: {[m for m in dir(server) if not m.startswith('_')]}")

# Check internal storage
print("\n=== Internal Component Storage ===")
for attr in dir(server):
    if attr.startswith('_') and not attr.startswith('__'):
        value = getattr(server, attr, None)
        if value is not None:
            print(f"{attr}: {type(value)} - {value}")

# Print specific attributes we expect to use
attrs_to_check = ["_tools", "_resources", "_prompts"]
print("\n=== Expected Attributes ===")
for attr in attrs_to_check:
    value = getattr(server, attr, None)
    print(f"{attr}: {type(value)}")
    if value is not None:
        if isinstance(value, (list, tuple)):
            print(f"  Length: {len(value)}")
            if len(value) > 0:
                print(f"  First item type: {type(value[0])}")
                print(f"  First item attributes: {dir(value[0]) if hasattr(value[0], '__dir__') else 'No attributes'}")
        elif isinstance(value, dict):
            print(f"  Keys: {list(value.keys())}")

# Check FastMCP server module
print("\n=== FastMCP Server Module ===")
try:
    from fastmcp.server import server as server_module
    print(f"Server module attributes: {dir(server_module)}")
except ImportError as e:
    print(f"Error importing server module: {e}")

# Check processing methods
print("\n=== Processing Methods ===")
if hasattr(server, "process_request"):
    print(f"process_request signature: {inspect.signature(server.process_request)}")
else:
    print("No process_request method found")

# Check list methods
print("\n=== List Methods ===")
for method_name in ["list_tools", "list_resources", "list_prompts"]:
    if hasattr(server, method_name):
        method = getattr(server, method_name)
        print(f"{method_name} signature: {inspect.signature(method)}")
        try:
            result = method()
            print(f"{method_name} result type: {type(result)}")
            if isinstance(result, (list, tuple)):
                print(f"{method_name} result length: {len(result)}")
                if len(result) > 0:
                    print(f"{method_name} first item type: {type(result[0])}")
        except Exception as e:
            print(f"Error calling {method_name}: {e}")
    else:
        print(f"No {method_name} method found")
