#!/usr/bin/env python3
"""
Script to fix the connection between Claude and the IPFS Accelerate MCP server.
This script:
1. Stops any running MCP servers
2. Copies the fixed settings to Claude's configuration directory
3. Starts the enhanced MCP server in a separate process
"""

import os
import sys
import time
import json
import shutil
import signal
import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fix_claude_mcp")

def stop_all_mcp_servers():
    """Stop all running MCP server processes"""
    logger.info("Stopping all running MCP server processes...")
    
    try:
        # Find all python processes containing "mcp_server"
        result = subprocess.run(
            ["pgrep", "-f", "python.*mcp_server"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            logger.info(f"Found {len(pids)} MCP server processes running")
            
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    logger.info(f"Sent SIGTERM to process {pid}")
                except (ProcessLookupError, ValueError) as e:
                    logger.error(f"Could not terminate process {pid}: {e}")
        else:
            logger.info("No MCP server processes found running")
    except Exception as e:
        logger.error(f"Error stopping MCP servers: {e}")

def update_claude_settings():
    """Update Claude's MCP settings with the fixed configuration"""
    logger.info("Updating Claude MCP settings...")
    
    # Source settings file
    source_settings = Path(__file__).parent / "fixed_mcp_settings.json"
    
    if not source_settings.exists():
        logger.error(f"Fixed settings file not found: {source_settings}")
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

def start_enhanced_mcp_server():
    """Start the enhanced MCP server in a separate process"""
    logger.info("Starting Enhanced MCP server...")
    
    server_script = Path(__file__).parent / "enhanced_mcp_server.py"
    
    if not server_script.exists():
        logger.error(f"MCP server script not found: {server_script}")
        return False
    
    try:
        # Start server as a detached process
        server_log = open("mcp_server_fixed.log", "w")
        process = subprocess.Popen(
            [sys.executable, str(server_script), "--port", "8002"],
            stdout=server_log,
            stderr=server_log,
            start_new_session=True
        )
        
        logger.info(f"Started Enhanced MCP server (PID: {process.pid})")
        logger.info(f"Server logs will be written to mcp_server_fixed.log")
        
        # Wait a moment to ensure server starts up
        time.sleep(2)
        
        # Check if server is running
        if process.poll() is not None:
            logger.error(f"Server exited immediately with code {process.returncode}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error starting MCP server: {e}")
        return False

def verify_server_connection():
    """Verify that the MCP server is running and accessible"""
    import requests
    logger.info("Verifying server connection...")
    
    try:
        # Check manifest endpoint
        manifest_response = requests.get("http://localhost:8002/manifest", timeout=5)
        if manifest_response.status_code == 200:
            manifest_data = manifest_response.json()
            tool_count = len(manifest_data.get("tools", {}))
            logger.info(f"Server manifest accessible with {tool_count} tools")
        else:
            logger.error(f"Manifest endpoint returned status {manifest_response.status_code}")
            return False
        
        # Test tool invocation
        tool_response = requests.post(
            "http://localhost:8002/invoke/health_check",
            json={},
            timeout=5
        )
        if tool_response.status_code == 200:
            logger.info("Tool invocation successful")
        else:
            logger.error(f"Tool invocation returned status {tool_response.status_code}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error verifying server connection: {e}")
        return False

def main():
    """Main function to fix Claude MCP connection"""
    logger.info("Starting Claude MCP connection fix...")
    
    # Stop any running MCP servers
    stop_all_mcp_servers()
    time.sleep(1)  # Give processes time to terminate
    
    # Start enhanced MCP server
    if not start_enhanced_mcp_server():
        logger.error("Failed to start enhanced MCP server")
        return 1
    
    # Update Claude settings
    if not update_claude_settings():
        logger.error("Failed to update Claude settings")
        return 1
    
    # Verify server connection
    if not verify_server_connection():
        logger.error("Server verification failed")
        return 1
    
    logger.info("""
    ✅ Connection fix applied successfully!
    
    To complete the setup:
    1. Close VSCode completely
    2. Restart VSCode
    3. After VSCode restarts, open the Command Palette (Ctrl+Shift+P)
    4. Type and select "Developer: Reload Window"
    5. Start a new conversation with Claude
    
    Claude should now be able to see and use all IPFS Accelerate tools!
    """)
    return 0

if __name__ == "__main__":
    sys.exit(main())