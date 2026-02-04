"""
Test MCP Installation

This script tests if the MCP packages are correctly installed and accessible.
"""

def main():
    print("Testing MCP installation...")
    
    # Try to import MCP
    try:
        import mcp
        print(f"✓ MCP installed (version: {mcp.__version__})")
    except ImportError as e:
        print(f"✗ Failed to import MCP: {e}")
    
    # Try to import FastMCP
    try:
        import fastmcp
        print(f"✓ FastMCP installed (version: {fastmcp.__version__})")
    except ImportError as e:
        print(f"✗ Failed to import FastMCP: {e}")
    
    # Try to import uvicorn
    try:
        import uvicorn
        print(f"✓ Uvicorn installed")
    except ImportError as e:
        print(f"✗ Failed to import uvicorn: {e}")
    
    print("\nDone testing MCP installation.")

if __name__ == "__main__":
    main()
