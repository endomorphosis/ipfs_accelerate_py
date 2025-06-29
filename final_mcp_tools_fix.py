#!/usr/bin/env python3
"""
Final MCP Server Tools Fix

This script ensures that all the tools required by the tests are properly 
implemented in the unified_mcp_server.py file, with the exact signatures
and parameter names expected by the tests.
"""

import os
import sys
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def make_final_fixes():
    """Make final fixes to the MCP server implementation."""
    server_file = "unified_mcp_server.py"
    backup_file = f"{server_file}.bak_final"
    
    # Check if file exists
    if not os.path.exists(server_file):
        logger.error(f"File {server_file} not found")
        return False
    
    # Create backup
    try:
        with open(server_file, 'r') as src, open(backup_file, 'w') as dst:
            dst.write(src.read())
        logger.info(f"Created backup at {backup_file}")
    except Exception as e:
        logger.error(f"Error creating backup: {str(e)}")
        return False
    
    # Read the file content
    with open(server_file, 'r') as f:
        content = f.read()
    
    # 1. Fix the ipfs_gateway_url function to handle 'ipfs_hash' parameter
    if '@register_tool("ipfs_gateway_url")' in content:
        old_gateway_url_function = re.search(
            r'@register_tool\("ipfs_gateway_url"\)\s*def\s+ipfs_gateway_url\([^)]*\):[^}]+}',
            content, re.DOTALL
        )
        
        if old_gateway_url_function:
            new_gateway_url_function = '''@register_tool("ipfs_gateway_url")
def ipfs_gateway_url(ipfs_hash: str = None, cid: str = None, gateway: str = "https://ipfs.io"):
    """Get a gateway URL for an IPFS CID."""
    # Handle different parameter names (ipfs_hash or cid)
    hash_value = ipfs_hash if ipfs_hash is not None else cid
    if hash_value is None:
        return {"error": "No CID or IPFS hash provided", "success": False}
    
    return {
        "cid": hash_value,
        "url": f"{gateway}/ipfs/{hash_value}",
        "success": True
    }'''
            
            content = content.replace(old_gateway_url_function.group(0), new_gateway_url_function)
            logger.info("Fixed ipfs_gateway_url function")
    
    # 2. Add ipfs_get_hardware_info tool with proper implementation
    if '@register_tool("ipfs_get_hardware_info")' not in content:
        # Find location after get_hardware_info tool
        get_hardware_info_pattern = '@register_tool("get_hardware_info")\n@register_tool("get_hardware_info")\ndef get_hardware_info():'
        
        if get_hardware_info_pattern in content:
            ipfs_get_hardware_info_tool = '''

@register_tool("ipfs_get_hardware_info")
def ipfs_get_hardware_info():
    """Get hardware information through IPFS."""
    return accelerate_bridge.get_hardware_info()
'''
            content = content.replace(
                get_hardware_info_pattern,
                get_hardware_info_pattern + ipfs_get_hardware_info_tool
            )
            logger.info("Added ipfs_get_hardware_info tool")
    
    # 3. Update server endpoint for tool calls
    if '@app.route("/mcp/tool/<tool_name>", methods=["POST"])' in content:
        old_call_tool_function = re.search(
            r'@app\.route\("/mcp/tool/<tool_name>", methods=\["POST"\]\)\s*def\s+call_tool\([^}]+}', 
            content, re.DOTALL
        )
        
        if old_call_tool_function:
            new_call_tool_function = '''@app.route("/mcp/tool/<tool_name>", methods=["POST"])
def call_tool(tool_name):
    """Call a tool with arguments."""
    if tool_name not in MCP_TOOLS:
        # Special case for specific tools
        if tool_name == "ipfs_get_hardware_info" and "get_hardware_info" in MCP_TOOLS:
            logger.info(f"Redirecting {tool_name} to get_hardware_info")
            return MCP_TOOLS["get_hardware_info"](**request.json or {})
        
        return jsonify({"error": f"Tool not found: {tool_name}"}), 404
    
    try:
        arguments = request.json or {}
        result = MCP_TOOLS[tool_name](**arguments)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error calling {tool_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500'''
            
            content = content.replace(old_call_tool_function.group(0), new_call_tool_function)
            logger.info("Updated call_tool function to handle special cases")
    
    # Write the modified content
    try:
        with open(server_file, 'w') as f:
            f.write(content)
        logger.info("Successfully made all final fixes")
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

def verify_fixes():
    """Verify that all fixes were applied correctly."""
    server_file = "unified_mcp_server.py"
    
    with open(server_file, 'r') as f:
        content = f.read()
    
    # Check gateway URL function has ipfs_hash parameter
    if 'def ipfs_gateway_url(ipfs_hash: str = None, cid: str = None' not in content:
        logger.error("ipfs_gateway_url function does not handle ipfs_hash parameter")
        return False
    
    # Check ipfs_get_hardware_info is defined
    if '@register_tool("ipfs_get_hardware_info")' not in content:
        logger.error("ipfs_get_hardware_info tool is not defined")
        return False
    
    # Check call_tool function has special case handling
    if 'if tool_name == "ipfs_get_hardware_info" and "get_hardware_info" in MCP_TOOLS:' not in content:
        logger.error("call_tool function does not handle special cases")
        return False
    
    logger.info("All fixes verified successfully")
    return True

def main():
    """Main function to make final fixes."""
    logger.info("Making final fixes to unified_mcp_server.py...")
    
    if make_final_fixes():
        return verify_fixes()
    else:
        logger.error("Failed to apply final fixes")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
