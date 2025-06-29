
import json
import sys
from typing import Dict, Any

def print_result(title, result):
    """Print a formatted tool result."""
    print(f"\n🔍 {title}")
    if isinstance(result, dict):
        print(json.dumps(result, indent=2))
    else:
        print(result)

def main():
    """Main test function."""
    
    # Server name is the host:port of the MCP server
    server_name = "localhost:8002"
    
    # Test ipfs_node_info
    print("\n=== Testing IPFS Node Info ===")
    print("Calling ipfs_node_info tool...")
    # This would be executed by Claude using use_mcp_tool
    # node_info = use_mcp_tool(server_name, "ipfs_node_info", {})
    # print_result("Node Info Result:", node_info)
    
    # Test ipfs_files_write
    print("\n=== Testing IPFS Files Write ===")
    print("Calling ipfs_files_write tool...")
    # This would be executed by Claude using use_mcp_tool
    # write_result = use_mcp_tool(server_name, "ipfs_files_write", {
    #     "path": "/client-test.txt",
    #     "content": "Test content from MCP client!"
    # })
    # print_result("Write Result:", write_result)
    
    # Test ipfs_files_read
    print("\n=== Testing IPFS Files Read ===")
    print("Calling ipfs_files_read tool...")
    # This would be executed by Claude using use_mcp_tool
    # read_result = use_mcp_tool(server_name, "ipfs_files_read", {
    #     "path": "/client-test.txt"
    # })
    # print_result("Read Result:", read_result)
    
    # Test ipfs_files_ls
    print("\n=== Testing IPFS Files List ===")
    print("Calling ipfs_files_ls tool...")
    # This would be executed by Claude using use_mcp_tool
    # ls_result = use_mcp_tool(server_name, "ipfs_files_ls", {
    #     "path": "/"
    # })
    # print_result("List Result:", ls_result)
    
    print("\n=== MCP Client Interface Test Template Complete ===")
    print("These commands are meant to be executed by Claude using use_mcp_tool")
    print("You can ask Claude to run each of these tools using the MCP interface")

if __name__ == "__main__":
    main()
