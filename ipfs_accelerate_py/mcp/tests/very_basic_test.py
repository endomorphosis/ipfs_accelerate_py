#!/usr/bin/env python3
"""
Very basic test with print statements
"""
import sys

print("Starting very basic test...")

try:
    print("Importing FastMCP...")
    from fastmcp import FastMCP
    print("Successfully imported FastMCP")
    
    print("Creating FastMCP instance...")
    mcp = FastMCP("Basic Test")
    print(f"Created MCP server: {mcp.name}")
    
    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    print("Added a tool")
    print(f"Tool count: {len(mcp.tools)}")
    
    result = add(1, 2)
    print(f"Tool result: {result}")
    
    print("Test passed!")

except ImportError as e:
    print(f"Import error: {e}", file=sys.stderr)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
