#!/bin/bash

# Fix MCP Client Connection for Claude
# This script helps resolve connection issues between Claude and the MCP server

echo "=== Claude MCP Client Connection Fixer ==="
echo "This script will help resolve connection issues between Claude and the MCP server."

# Check if the MCP server is running
if ! pgrep -f "python.*start_mcp_with_ipfs.py" > /dev/null; then
    echo "❌ MCP server is not running. Starting it now..."
    bash restart_mcp_server.sh
    sleep 2
else
    echo "✅ MCP server is running."
fi

# Check if the SSE endpoint is accessible
echo -n "Testing SSE endpoint... "
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8002/sse | grep -q "200"; then
    echo "✅ SSE endpoint is accessible."
else
    echo "❌ SSE endpoint is not accessible. Please check the server logs."
    exit 1
fi

# Create a backup of the MCP settings file
SETTINGS_FILE="$HOME/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json"
BACKUP_FILE="$HOME/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json.bak"

echo "Creating backup of MCP settings file..."
cp "$SETTINGS_FILE" "$BACKUP_FILE"

# Update the MCP settings file
echo "Updating MCP settings file..."
cat > "$SETTINGS_FILE" << 'EOF'
{
  "mcpServers": {
    "ipfs-accelerate-mcp": {
      "disabled": false,
      "timeout": 60,
      "url": "http://localhost:8002/sse",
      "transportType": "sse"
    },
    "ipfs-accelerate-py": {
      "disabled": false,
      "timeout": 60,
      "url": "http://localhost:8000/sse",
      "transportType": "sse"
    },
    "default-mcp": {
      "disabled": false,
      "timeout": 60,
      "url": "http://localhost:8080/sse",
      "transportType": "sse"
    },
    "direct-ipfs-kit-mcp": {
      "disabled": false,
      "timeout": 60,
      "url": "http://localhost:3000/sse",
      "transportType": "sse"
    }
  }
}
EOF

echo "✅ MCP settings file updated."

# Instructions for restarting VSCode
echo ""
echo "=== Next Steps ==="
echo "To complete the fix, please:"
echo "1. Restart VSCode completely (File > Exit or Ctrl+Q, then reopen)"
echo "2. After VSCode restarts, reload the Claude extension:"
echo "   - Open the Command Palette (Ctrl+Shift+P)"
echo "   - Type and select 'Developer: Reload Window'"
echo ""
echo "3. Once VSCode and Claude are reloaded, test the connection with:"
echo "   use_mcp_tool with server_name='ipfs-accelerate-mcp', tool_name='ipfs_node_info', arguments={}"
echo ""
echo "If the issue persists, try these additional steps:"
echo "1. Check if the server is running on the correct port (8002)"
echo "2. Verify the SSE endpoint is working: curl http://localhost:8002/sse"
echo "3. Check the server logs for any errors"
echo "4. Ensure no firewall is blocking the connection"
echo ""
echo "=== Troubleshooting ==="
echo "If you need to restore the original settings, run:"
echo "cp \"$BACKUP_FILE\" \"$SETTINGS_FILE\""
echo ""
echo "Script completed."
