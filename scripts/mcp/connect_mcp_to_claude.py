#!/usr/bin/env python
"""
Connect MCP Server to Claude

This script verifies the MCP server is running and provides instructions
on how to register it with Claude for tool use access.
"""

import json
import requests
import sys
import logging
import os
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def check_mcp_server(port=8002):
    """Check if the MCP server is running and get its manifest"""
    url = f"http://localhost:{port}/mcp/manifest"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            manifest = response.json()
            return True, manifest
        else:
            logger.error(f"Failed to get manifest: {response.status_code}")
            return False, None
    except Exception as e:
        logger.error(f"Error connecting to MCP server: {e}")
        return False, None

def test_get_hardware_info(port=8002):
    """Test the get_hardware_info tool"""
    url = f"http://localhost:{port}/mcp/tool/get_hardware_info"
    try:
        response = requests.post(url, json={}, timeout=10)
        if response.status_code == 200:
            hardware_info = response.json()
            logger.info("Hardware info successfully retrieved")
            
            # Print a summary of the hardware
            if "system" in hardware_info:
                system = hardware_info["system"]
                print(f"\nSystem: {system.get('os', 'Unknown')} {system.get('distribution', '')}")
                print(f"CPU: {system.get('cpu', {}).get('cores_logical', 'Unknown')} cores")
                print(f"Memory: {system.get('memory_total', 'Unknown')} GB")
            
            if "accelerators" in hardware_info:
                accelerators = hardware_info["accelerators"]
                print("\nAccelerators:")
                for acc_name, acc_info in accelerators.items():
                    if acc_info.get("available", False):
                        devices = acc_info.get("devices", [])
                        if devices:
                            for device in devices:
                                print(f"  - {acc_name.upper()}: {device.get('name', 'Unknown')}")
                        else:
                            print(f"  - {acc_name.upper()}: Available")
            
            return True
        else:
            logger.error(f"Failed to get hardware info: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error testing get_hardware_info: {e}")
        return False

def get_vscode_settings_path():
    """Get the path to VSCode settings based on the platform"""
    home = os.path.expanduser("~")
    
    if platform.system() == "Windows":
        return os.path.join(home, "AppData", "Roaming", "Code", "User", "settings.json")
    elif platform.system() == "Darwin":  # macOS
        return os.path.join(home, "Library", "Application Support", "Code", "User", "settings.json")
    else:  # Linux and others
        return os.path.join(home, ".config", "Code", "User", "settings.json")

def main():
    """Main entry point"""
    port = 8002
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            logger.error(f"Invalid port: {sys.argv[1]}")
            return 1
    
    print("=" * 50)
    print("IPFS Accelerate MCP Server to Claude Connection Tool")
    print("=" * 50)
    print("\nChecking MCP server status...")
    
    # Check if the server is running
    server_running, manifest = check_mcp_server(port)
    
    if not server_running:
        print("\n❌ MCP server is not running on port", port)
        print("   Please start the server first with:")
        print(f"   bash restart_mcp_server.sh --port {port}")
        return 1
    
    # Server is running, test the hardware info tool
    print("\n✅ MCP server is running on port", port)
    print("\nTesting the get_hardware_info tool...")
    tool_working = test_get_hardware_info(port)
    
    if not tool_working:
        print("\n❌ get_hardware_info tool is not working properly")
        return 1
    
    print("\n✅ get_hardware_info tool is working properly")
    
    # Print available tools from the manifest
    if manifest and "tools" in manifest:
        tools = manifest["tools"]
        print("\nAvailable MCP tools:")
        for tool_name in tools:
            print(f"  - {tool_name}")
    
    # Print configuration instructions for Claude integration
    print("\n" + "=" * 50)
    print("Claude Integration Instructions")
    print("=" * 50)
    print("\nTo enable the MCP server with Claude, you need to:")
    
    print("\n1. Open VSCode settings (File > Preferences > Settings)")
    print("2. Search for 'claude mcp'")
    print("3. Add the following server to the list:")
    print(f"""
   {{
     "name": "ipfs-accelerate-mcp",
     "url": "http://localhost:{port}",
     "auth": {{
       "type": "none"
     }}
   }}
""")
    
    print("4. Alternatively, edit your settings.json directly by adding:")
    print(f"""
   "claude-dev.mcp.servers": [
     {{
       "name": "ipfs-accelerate-mcp",
       "url": "http://localhost:{port}",
       "auth": {{
         "type": "none"
       }}
     }}
   ]
""")
    
    settings_path = get_vscode_settings_path()
    print(f"   The settings file is typically at: {settings_path}")
    
    print("\n5. After configuring, you can use MCP tools in Claude like this:")
    print("""
   <use_mcp_tool>
   <server_name>ipfs-accelerate-mcp</server_name>
   <tool_name>get_hardware_info</tool_name>
   <arguments>
   {}
   </arguments>
   </use_mcp_tool>
""")
    
    print("\n6. Restart VSCode if needed to apply the changes")
    
    print("\n" + "=" * 50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
