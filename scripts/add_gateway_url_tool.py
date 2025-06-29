#!/usr/bin/env python3
"""
MCP Server Tool Fixer

This script adds the missing ipfs_gateway_url tool to the unified_mcp_server.py file.
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

def add_gateway_url_tool():
    """Add the missing ipfs_gateway_url tool to the unified_mcp_server.py file."""
    server_file = "unified_mcp_server.py"
    backup_file = "unified_mcp_server.py.bak"

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
        lines = f.readlines()
    
    # Find the position to insert the gateway_url tool
    ipfs_cat_end_idx = -1
    ipfs_files_write_idx = -1
    
    for i, line in enumerate(lines):
        # Find where ipfs_cat function ends and where ipfs_files_write starts
        if "@register_tool(\"ipfs_cat\")" in line:
            for j in range(i, len(lines)):
                if "return accelerate_bridge.cat" in lines[j]:
                    ipfs_cat_end_idx = j
                    break
        
        if "@register_tool(\"ipfs_files_write\")" in line:
            ipfs_files_write_idx = i
            break
    
    if ipfs_cat_end_idx == -1 or ipfs_files_write_idx == -1:
        logger.error("Could not find position to insert ipfs_gateway_url tool")
        return False
    
    # The ipfs_gateway_url tool code to insert
    gateway_url_code = [
        "\n",
        "@register_tool(\"ipfs_gateway_url\")\n",
        "def ipfs_gateway_url(cid: str, gateway: str = \"https://ipfs.io\"):\n",
        "    \"\"\"Get a gateway URL for an IPFS CID.\"\"\"\n",
        "    return {\n",
        "        \"cid\": cid,\n",
        "        \"url\": f\"{gateway}/ipfs/{cid}\",\n",
        "        \"success\": True\n",
        "    }\n",
    ]
    
    # Insert the code
    modified_lines = lines[:ipfs_files_write_idx]
    modified_lines.extend(gateway_url_code)
    modified_lines.extend(lines[ipfs_files_write_idx:])
    
    # Write the modified file
    try:
        with open(server_file, 'w') as f:
            f.writelines(modified_lines)
        logger.info("Successfully added ipfs_gateway_url tool")
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

def verify_tool_added():
    """Verify that the ipfs_gateway_url tool was added successfully."""
    server_file = "unified_mcp_server.py"
    
    with open(server_file, 'r') as f:
        content = f.read()
    
    if "@register_tool(\"ipfs_gateway_url\")" in content:
        logger.info("✅ ipfs_gateway_url tool was added successfully")
        return True
    else:
        logger.error("❌ ipfs_gateway_url tool was not added")
        return False

def main():
    """Main function to add the missing ipfs_gateway_url tool."""
    logger.info("Adding missing ipfs_gateway_url tool to unified_mcp_server.py...")
    
    if add_gateway_url_tool():
        return verify_tool_added()
    else:
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
