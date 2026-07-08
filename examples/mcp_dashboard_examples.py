"""
MCP Dashboard and SDK Examples

This file demonstrates how to use the enhanced MCP dashboard features
and JavaScript SDK to interact with MCP tools.
"""

# Example 1: Using the REST API to get all tools
import requests
import json

def get_all_tools():
    """Get all available MCP tools from the dashboard API."""
    response = requests.get('http://localhost:8899/api/mcp/tools')
    data = response.json()
    
    print(f"Total tools: {data['total']}")
    print(f"Categories: {data['category_count']}")
    
    for category, tools in data['categories'].items():
        print(f"\n{category} ({len(tools)} tools):")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")
    
    return data

# Example 2: Execute a tool via JSON-RPC
def call_tool_via_jsonrpc(tool_name, arguments):
    """Call an MCP tool using JSON-RPC."""
    request_body = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        },
        "id": 1
    }
    
    response = requests.post(
        'http://localhost:8899/jsonrpc',
        json=request_body,
        headers={'Content-Type': 'application/json'}
    )
    
    result = response.json()
    
    if 'error' in result:
        print(f"Error: {result['error']['message']}")
        return None
    
    return result['result']

# Example 3: GitHub operations
def list_github_repos(owner):
    """List GitHub repositories for an owner."""
    result = call_tool_via_jsonrpc('github_list_repos', {
        'owner': owner,
        'limit': 10
    })
    
    if result:
        print(f"\nRepositories for {owner}:")
        for repo in result.get('repos', []):
            print(f"  - {repo['name']}: {repo.get('description', 'No description')}")
    
    return result

# Example 4: Docker operations
def list_docker_containers():
    """List all Docker containers."""
    result = call_tool_via_jsonrpc('docker_list_containers', {
        'all': True
    })
    
    if result:
        print("\nDocker containers:")
        for container in result.get('containers', []):
            print(f"  - {container['id']}: {container['status']}")
    
    return result

# Example 5: Hardware information
def get_hardware_info():
    """Get hardware information."""
    result = call_tool_via_jsonrpc('hardware_get_info', {})
    
    if result:
        print("\nHardware Information:")
        print(f"  CPU: {result.get('cpu', 'Unknown')}")
        print(f"  Memory: {result.get('memory', 'Unknown')}")
        print(f"  GPU: {result.get('gpu', 'None')}")
    
    return result

# Example 6: IPFS operations
def list_ipfs_files(path='/'):
    """List files in IPFS."""
    result = call_tool_via_jsonrpc('ipfs_files_list', {
        'path': path
    })
    
    if result:
        print(f"\nIPFS files in {path}:")
        for file in result.get('files', []):
            print(f"  - {file['name']}: {file.get('size', 0)} bytes")
    
    return result

# Example 7: Network operations
def get_network_peers():
    """List network peers."""
    result = call_tool_via_jsonrpc('network_list_peers', {})
    
    if result:
        print("\nNetwork Peers:")
        for peer in result.get('peers', []):
            print(f"  - {peer['id']}: {peer.get('addr', 'Unknown address')}")
    
    return result

# Example 8: Batch execution
def batch_tool_calls():
    """Execute multiple tools in a batch."""
    batch_request = [
        {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "hardware_get_info",
                "arguments": {}
            },
            "id": 1
        },
        {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "network_list_peers",
                "arguments": {}
            },
            "id": 2
        },
        {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "docker_list_containers",
                "arguments": {"all": True}
            },
            "id": 3
        }
    ]
    
    response = requests.post(
        'http://localhost:8899/jsonrpc',
        json=batch_request,
        headers={'Content-Type': 'application/json'}
    )
    
    results = response.json()
    
    print("\nBatch execution results:")
    for i, result in enumerate(results):
        if 'error' in result:
            print(f"  Call {i+1}: Error - {result['error']['message']}")
        else:
            print(f"  Call {i+1}: Success")
            print(f"    {json.dumps(result['result'], indent=4)[:100]}...")
    
    return results

# JavaScript SDK Examples (to be used in browser or Node.js)
"""
// Initialize the SDK
const client = new MCPClient('/jsonrpc');

// Example 1: List GitHub repositories
async function listGitHubRepos() {
    try {
        const repos = await client.githubListRepos('octocat', 10);
        console.log('Repositories:', repos);
    } catch (error) {
        console.error('Error:', error);
    }
}

// Example 2: Get hardware info
async function getHardwareInfo() {
    try {
        const info = await client.hardwareGetInfo();
        console.log('Hardware:', info);
    } catch (error) {
        console.error('Error:', error);
    }
}

// Example 3: List Docker containers
async function listContainers() {
    try {
        const containers = await client.dockerListContainers(true);
        console.log('Containers:', containers);
    } catch (error) {
        console.error('Error:', error);
    }
}

// Example 4: Batch execution
async function batchCalls() {
    try {
        const results = await client.callToolsBatch([
            { name: 'hardware_get_info', arguments: {} },
            { name: 'network_list_peers', arguments: {} },
            { name: 'docker_list_containers', arguments: { all: true } }
        ]);
        
        results.forEach((result, index) => {
            if (result.error) {
                console.error(`Call ${index+1} failed:`, result.error);
            } else {
                console.log(`Call ${index+1} succeeded:`, result.result);
            }
        });
    } catch (error) {
        console.error('Batch error:', error);
    }
}

// Example 5: Using the generic callTool method
async function callAnyTool() {
    try {
        const result = await client.callTool('github_list_repos', {
            owner: 'octocat',
            limit: 5
        });
        console.log('Result:', result);
    } catch (error) {
        console.error('Error:', error);
    }
}
"""

if __name__ == '__main__':
    print("MCP Dashboard and SDK Examples")
    print("=" * 50)
    
    # Run examples (uncomment to test)
    # Note: Make sure the MCP dashboard is running on port 8899
    
    # get_all_tools()
    # list_github_repos('octocat')
    # list_docker_containers()
    # get_hardware_info()
    # list_ipfs_files()
    # get_network_peers()
    # batch_tool_calls()
    
    print("\nExamples are ready to run!")
    print("Uncomment the function calls in __main__ to test them.")
    print("\nMake sure the MCP dashboard is running:")
    print("  python -m ipfs_accelerate_py.mcp_dashboard")
