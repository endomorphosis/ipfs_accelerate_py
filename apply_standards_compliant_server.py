#!/usr/bin/env python3
"""
Script to install the standards-compliant IPFS Accelerate MCP server in Claude.
This script:
1. Copies the standards MCP settings to Claude's configuration directory
2. Runs the 'mcp install' command to properly register the server with Claude
"""

import os
import sys
import shutil
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("apply_standards_mcp")

def update_claude_settings():
    """Update Claude's MCP settings with the standards-compliant configuration"""
    logger.info("Updating Claude MCP settings...")
    
    # Source settings file
    source_settings = Path(__file__).parent / "standards_mcp_settings.json"
    
    if not source_settings.exists():
        logger.error(f"Settings file not found: {source_settings}")
        return False
    
    # Possible target locations for Claude settings
    target_paths = [
        Path.home() / ".config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json",
        Path.home() / ".config/Code - Insiders/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json",
    ]
    
    # Try to find and update the settings file
    success = False
    for target_path in target_paths:
        if target_path.parent.exists():
            try:
                # Make backup of original settings if it exists
                if target_path.exists():
                    backup_path = target_path.with_suffix('.json.backup')
                    shutil.copy2(target_path, backup_path)
                    logger.info(f"Backed up original settings to {backup_path}")
                
                # Copy new settings
                shutil.copy2(source_settings, target_path)
                logger.info(f"Updated Claude MCP settings at {target_path}")
                success = True
                break
            except Exception as e:
                logger.error(f"Error updating settings at {target_path}: {e}")
    
    if not success:
        logger.error("Could not find Claude settings directory")
        return False
    
    return True

def install_with_mcp_cli():
    """Install the server using the official MCP CLI tool"""
    logger.info("Installing server with MCP CLI tool...")
    
    server_script = Path(__file__).parent / "standards_compliant_mcp_server.py"
    
    if not server_script.exists():
        logger.error(f"Server script not found: {server_script}")
        return False
    
    try:
        # Run the 'mcp install' command
        result = subprocess.run(
            ["mcp", "install", str(server_script), "--name", "IPFS Accelerate MCP Server"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("Successfully installed server with MCP CLI")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"Error installing server: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error running MCP CLI: {e}")
        return False

def main():
    """Main function to apply standards-compliant MCP server"""
    logger.info("Starting setup for standards-compliant MCP server...")
    
    # Update Claude settings
    if not update_claude_settings():
        logger.error("Failed to update Claude settings")
        return 1
    
    # Install with MCP CLI
    if not install_with_mcp_cli():
        logger.error("Failed to install server with MCP CLI")
        # Continue anyway, manual settings might work
    
    logger.info("""
    ✅ Standards-compliant MCP server setup completed!
    
    To complete the setup:
    1. Ensure the standards_compliant_mcp_server.py is running
    2. Close VSCode completely
    3. Restart VSCode
    4. After VSCode restarts, open the Command Palette (Ctrl+Shift+P)
    5. Type and select "Developer: Reload Window"
    6. Start a new conversation with Claude
    
    Claude should now be able to see and use all IPFS Accelerate tools!
    """)
    return 0

if __name__ == "__main__":
    sys.exit(main())