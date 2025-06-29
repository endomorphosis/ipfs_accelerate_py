#!/usr/bin/env python3
"""
Script to clear the Claude extension cache and reset its connection to MCP servers.
"""

import os
import sys
import shutil
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("claude_cache_cleaner")

def clear_claude_cache():
    """Clear the Claude extension cache directories in VS Code"""
    # Paths to potential cache locations
    cache_paths = [
        Path.home() / ".config/Code/User/globalStorage/saoudrizwan.claude-dev/cache",
        Path.home() / ".config/Code - Insiders/User/globalStorage/saoudrizwan.claude-dev/cache",
    ]
    
    success = False
    for cache_path in cache_paths:
        if cache_path.exists():
            try:
                logger.info(f"Clearing cache at: {cache_path}")
                
                # Clear contents but keep directory
                for item in cache_path.iterdir():
                    if item.is_file():
                        item.unlink()
                        logger.info(f"  Deleted file: {item.name}")
                    elif item.is_dir():
                        shutil.rmtree(item)
                        logger.info(f"  Deleted directory: {item.name}")
                
                success = True
            except Exception as e:
                logger.error(f"Error clearing cache at {cache_path}: {e}")
    
    if not success:
        logger.warning("Could not find any Claude extension cache directories")
    
    return success

def reset_server_status():
    """Reset the server status cache file if it exists"""
    status_files = [
        Path.home() / ".config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/server_status.json",
        Path.home() / ".config/Code - Insiders/User/globalStorage/saoudrizwan.claude-dev/settings/server_status.json",
    ]
    
    for status_file in status_files:
        if status_file.exists():
            try:
                logger.info(f"Resetting server status file: {status_file}")
                os.remove(status_file)
            except Exception as e:
                logger.error(f"Error removing server status file: {e}")

def main():
    """Main function to clear Claude extension cache"""
    logger.info("Starting Claude extension cache cleanup...")
    
    # Clear the cache
    clear_claude_cache()
    
    # Reset server status
    reset_server_status()
    
    logger.info("""
    ✅ Claude extension cache cleared!
    
    To complete the reset:
    1. Close VS Code completely
    2. Restart VS Code
    3. After VS Code restarts, open the Command Palette (Ctrl+Shift+P)
    4. Type and select "Developer: Reload Window"
    5. Start a new conversation with Claude
    
    Claude should now properly connect to your IPFS Accelerate MCP server.
    """)
    return 0

if __name__ == "__main__":
    sys.exit(main())