#!/usr/bin/env python3
"""
MCP Server Additional Tools Fixer

This script adds additional missing tools to the unified_mcp_server.py file.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_ipfs_get_hardware_info_tool():
    """Add the missing ipfs_get_hardware_info tool to the unified_mcp_server.py file."""
    server_file = "unified_mcp_server.py"
    backup_file = "unified_mcp_server.py.bak2"

    # Check if file exists
    if not os.path.exists(server_file):
        logger.error(f"File {server_file} not found")
        return False
    
    # Create backup if it doesn't exist
    if not os.path.exists(backup_file):
        try:
            with open(server_file, 'r') as src, open(backup_file, 'w') as dst:
                dst.write(src.read())
            logger.info(f"Created backup at {backup_file}")
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            return False
    
    # Read the file
    with open(server_file, 'r') as f:
        content = f.read()
    
    # Check if the tool is already there
    if "@register_tool(\"ipfs_get_hardware_info\")" in content:
        logger.info("ipfs_get_hardware_info tool already exists")
        return True
    
    # Find where to add the tool - after the get_hardware_info function
    get_hardware_info_code = "@register_tool(\"get_hardware_info\")\n@register_tool(\"get_hardware_info\")\ndef get_hardware_info():\n    \"\"\"Get hardware information.\"\"\"\n    return accelerate_bridge.get_hardware_info()"
    
    if get_hardware_info_code not in content:
        logger.error("Could not find get_hardware_info function")
        return False
    
    # Add the ipfs_get_hardware_info tool right after get_hardware_info
    ipfs_get_hardware_info_code = """

@register_tool("ipfs_get_hardware_info")
def ipfs_get_hardware_info():
    """Get hardware information through IPFS."""
    return accelerate_bridge.get_hardware_info()
"""

    modified_content = content.replace(
        get_hardware_info_code,
        get_hardware_info_code + ipfs_get_hardware_info_code
    )
    
    # Write the modified content
    try:
        with open(server_file, 'w') as f:
            f.write(modified_content)
        logger.info("Successfully added ipfs_get_hardware_info tool")
        return True
    except Exception as e:
        logger.error(f"Error writing modified file: {str(e)}")
        
        # Restore from backup
        try:
            with open(backup_file, 'r') as src, open(server_file, 'w') as dst:
                dst.write(src.read())
            logger.info(f"Restored from backup")
        except Exception as e:
            logger.error(f"Error restoring backup: {str(e)}")
        
        return False

def fix_ipfs_gateway_tool():
    """Fix the ipfs_gateway_url tool implementation."""
    server_file = "unified_mcp_server.py"
    
    # Read the file
    with open(server_file, 'r') as f:
        content = f.read()
    
    # Find the ipfs_gateway_url function
    gateway_url_pattern = "@register_tool(\"ipfs_gateway_url\")\ndef ipfs_gateway_url(cid: str, gateway: str = \"https://ipfs.io\"):"
    
    if gateway_url_pattern not in content:
        logger.error("Could not find ipfs_gateway_url function")
        return False
    
    # Define test functions for ipfs gateway tools
    test_functions_code = """
# Test tools for verified unit tests
@register_tool("ipfs_get_gateway_url")
def ipfs_get_gateway_url(cid: str, gateway: str = "https://ipfs.io"):
    """Get a gateway URL for an IPFS CID."""
    return {
        "cid": cid,
        "url": f"{gateway}/ipfs/{cid}",
        "success": True
    }

@register_tool("ipfs_get_gateway")
def ipfs_get_gateway(cid: str):
    """Get default gateway info for an IPFS CID."""
    return {
        "cid": cid,
        "gateway": "https://ipfs.io",
        "url": f"https://ipfs.io/ipfs/{cid}",
        "success": True
    }

@register_tool("ipfs_gateway_info")
def ipfs_gateway_info():
    """Get information about available IPFS gateways."""
    return {
        "default": "https://ipfs.io",
        "gateways": [
            {"name": "IPFS.io", "url": "https://ipfs.io", "public": True},
            {"name": "Cloudflare", "url": "https://cloudflare-ipfs.com", "public": True},
            {"name": "Pinata", "url": "https://gateway.pinata.cloud", "public": True},
            {"name": "Infura", "url": "https://ipfs.infura.io", "public": True},
            {"name": "Local", "url": "http://localhost:8080", "public": False}
        ],
        "success": True
    }
"""
    
    # Add the test functions at the end of the tools section
    gateway_section_end = "def accelerated_inference(model, input_data, use_ipfs=True):"
    
    if gateway_section_end in content:
        modified_content = content.replace(
            f"@register_tool(\"accelerated_inference\")\n{gateway_section_end}",
            f"@register_tool(\"accelerated_inference\")\n{gateway_section_end}\n{test_functions_code}\n"
        )
    
        # Write the modified content
        try:
            with open(server_file, 'w') as f:
                f.write(modified_content)
            logger.info("Successfully added gateway test tools")
            return True
        except Exception as e:
            logger.error(f"Error writing modified file: {str(e)}")
            return False
    else:
        logger.error("Could not find position to add gateway test tools")
        return False

def fix_server_routes():
    """Fix the server routes for proper tool listing."""
    server_file = "unified_mcp_server.py"
    
    # Read the file
    with open(server_file, 'r') as f:
        content = f.read()
    
    # Fix the tools route to return proper tool information
    tools_route = '@app.route("/tools", methods=["GET"])\ndef list_tools():\n    """List all available tools."""\n    return jsonify({"tools": list(MCP_TOOLS.keys())})'
    
    improved_tools_route = '''@app.route("/tools", methods=["GET"])
def list_tools():
    """List all available tools."""
    tools_dict = {}
    for name, func in MCP_TOOLS.items():
        tools_dict[name] = {
            "description": MCP_TOOL_DESCRIPTIONS.get(name, ""),
            "schema": MCP_TOOL_SCHEMAS.get(name, {})
        }
    return jsonify(tools_dict)'''
    
    if tools_route in content:
        modified_content = content.replace(tools_route, improved_tools_route)
    
        # Write the modified content
        try:
            with open(server_file, 'w') as f:
                f.write(modified_content)
            logger.info("Successfully improved tools route")
            return True
        except Exception as e:
            logger.error(f"Error writing modified file: {str(e)}")
            return False
    else:
        logger.warning("Could not find tools route to improve")
        # Not a critical error
        return True

def verify_tools_added():
    """Verify that all required tools were added successfully."""
    server_file = "unified_mcp_server.py"
    
    with open(server_file, 'r') as f:
        content = f.read()
    
    tools_to_check = [
        "@register_tool(\"ipfs_get_hardware_info\")",
        "@register_tool(\"ipfs_gateway_url\")",
        "@register_tool(\"ipfs_get_gateway\")",
        "@register_tool(\"ipfs_get_gateway_url\")"
    ]
    
    missing_tools = []
    
    for tool in tools_to_check:
        if tool not in content:
            missing_tools.append(tool)
    
    if missing_tools:
        logger.error(f"❌ Missing tools: {missing_tools}")
        return False
    else:
        logger.info("✅ All required tools were added successfully")
        return True

def main():
    """Main function to add missing tools."""
    logger.info("Adding additional missing tools to unified_mcp_server.py...")
    
    hardware_info_success = add_ipfs_get_hardware_info_tool()
    gateway_success = fix_ipfs_gateway_tool()
    routes_success = fix_server_routes()
    
    if hardware_info_success and gateway_success and routes_success:
        logger.info("All fixes applied successfully")
        return verify_tools_added()
    else:
        logger.error("Failed to apply some fixes")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
