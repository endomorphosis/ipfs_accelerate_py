# MCP Dashboard User Guide

## Overview

The MCP (Model Control Plane) Dashboard provides a comprehensive web interface for managing and executing all MCP server tools. This guide covers the enhanced features added to improve tool coverage and align with the MCP JavaScript SDK.

## Features

### 1. Tool Discovery & Display

The dashboard now displays all 50+ available MCP tools organized by category:

- **GitHub Tools** - Repository, PR, and issue management
- **Docker Tools** - Container management and orchestration
- **Hardware Tools** - Hardware information and recommendations
- **Runner Tools** - GitHub Actions autoscaler management
- **IPFS Files Tools** - IPFS file operations
- **Network Tools** - P2P network management
- **Models** - AI model search and recommendations
- **Inference** - AI model inference operations
- **Workflows** - Workflow management
- **Dashboard** - Dashboard data and metrics
- **Other** - Additional utilities

### 2. Tool Search & Filter

The dashboard includes a powerful search feature that allows you to:

- Search tools by name (e.g., "github", "docker")
- Search by category (e.g., "Network", "Hardware")
- Search by description keywords
- Clear filters to show all tools

**How to use:**
1. Navigate to the "MCP Tools" tab
2. Type in the search box at the top
3. Results update in real-time
4. Click "Clear Search" to reset

### 3. Interactive Tool Execution

Each tool is clickable and opens an interactive execution modal with:

- Tool name and description
- Dynamic parameter form based on the tool's input schema
- Parameter validation (required fields marked with *)
- Execute button to run the tool
- Result display with syntax highlighting
- Error handling with detailed messages

**How to execute a tool:**
1. Click on any tool tag
2. Fill in the required parameters in the modal
3. Click "Execute Tool"
4. View the result in the result panel
5. Click "Cancel" or close to dismiss the modal

### 4. API Endpoints

The dashboard exposes several REST API endpoints:

#### GET `/api/mcp/tools`

Returns all available MCP tools with categorization and input schemas.

**Response:**
```json
{
  "tools": [
    {
      "name": "github_list_repos",
      "description": "List GitHub repositories",
      "category": "GitHub",
      "status": "active",
      "input_schema": {
        "properties": {
          "owner": {"type": "string"},
          "limit": {"type": "integer"}
        }
      }
    }
  ],
  "categories": {
    "GitHub": [...],
    "Docker": [...]
  },
  "total": 50,
  "category_count": 10
}
```

#### POST `/jsonrpc`

Execute MCP tools via JSON-RPC 2.0 protocol.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "github_list_repos",
    "arguments": {
      "owner": "octocat",
      "limit": 10
    }
  },
  "id": 1
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "repos": [...],
    "total": 10
  },
  "id": 1
}
```

## JavaScript SDK Integration

The enhanced MCP SDK (`mcp-sdk.js`) provides convenient methods for all tool categories.

### Basic Usage

```javascript
// Initialize the client
const client = new MCPClient('/jsonrpc');

// Call any tool by name
const result = await client.callTool('github_list_repos', { 
  owner: 'octocat', 
  limit: 10 
});
```

### Category-Specific Convenience Methods

#### GitHub Tools

```javascript
// List repositories
const repos = await client.githubListRepos('octocat', 30);

// Get specific repository
const repo = await client.githubGetRepo('octocat', 'hello-world');

// List pull requests
const prs = await client.githubListPrs('octocat', 'hello-world', 'open');

// Get specific PR
const pr = await client.githubGetPr('octocat', 'hello-world', 123);

// List issues
const issues = await client.githubListIssues('octocat', 'hello-world', 'open');

// Get specific issue
const issue = await client.githubGetIssue('octocat', 'hello-world', 456);
```

#### Docker Tools

```javascript
// Run container
const container = await client.dockerRunContainer('nginx:latest', null, { PORT: '8080' });

// List containers
const containers = await client.dockerListContainers(true);

// Stop container
await client.dockerStopContainer('container_id');

// Pull image
await client.dockerPullImage('redis:alpine');
```

#### Hardware Tools

```javascript
// Get hardware information
const hwInfo = await client.hardwareGetInfo();

// Test hardware
const testResults = await client.hardwareTest();

// Get hardware recommendations
const recommendations = await client.hardwareRecommend('inference');
```

#### Runner Tools

```javascript
// Start autoscaler
await client.runnerStartAutoscaler('myorg', 60);

// Stop autoscaler
await client.runnerStopAutoscaler();

// Get status
const status = await client.runnerGetStatus();

// List workflows
const workflows = await client.runnerListWorkflows('myorg', 'myrepo');
```

#### IPFS Tools

```javascript
// Add file
const result = await client.ipfsFilesAdd('/path/to/file', 'file content');

// Get file
await client.ipfsFilesGet('QmHash123...', '/output/path');

// Cat file
const content = await client.ipfsFilesCat('QmHash123...');

// Pin file
await client.ipfsFilesPin('QmHash123...');

// List files
const files = await client.ipfsFilesList('/');
```

#### Network Tools

```javascript
// List peers
const peers = await client.networkListPeers();

// Connect to peer
await client.networkConnectPeer('/ip4/1.2.3.4/tcp/4001/p2p/QmPeerId');

// Get swarm info
const swarmInfo = await client.networkGetSwarmInfo();

// Get bandwidth stats
const bandwidth = await client.networkGetBandwidth();

// Ping peer
const latency = await client.networkPingPeer('QmPeerId');
```

### Batch Execution

Execute multiple tools in parallel for better performance:

```javascript
const results = await client.callToolsBatch([
  { 
    name: 'github_list_repos', 
    arguments: { owner: 'octocat' } 
  },
  { 
    name: 'docker_list_containers', 
    arguments: { all: true } 
  },
  {
    name: 'hardware_get_info',
    arguments: {}
  }
]);

// Results array contains {result: ...} or {error: ...} for each call
results.forEach((res, index) => {
  if (res.error) {
    console.error(`Call ${index} failed:`, res.error);
  } else {
    console.log(`Call ${index} succeeded:`, res.result);
  }
});
```

## Tool Categories Reference

### GitHub Tools (6 tools)
- `github_list_repos` - List repositories
- `github_get_repo` - Get repository details
- `github_list_prs` - List pull requests
- `github_get_pr` - Get PR details
- `github_list_issues` - List issues
- `github_get_issue` - Get issue details

### Docker Tools (4 tools)
- `docker_run_container` - Run a container
- `docker_list_containers` - List containers
- `docker_stop_container` - Stop a container
- `docker_pull_image` - Pull an image

### Hardware Tools (3 tools)
- `hardware_get_info` - Get hardware information
- `hardware_test` - Run hardware tests
- `hardware_recommend` - Get hardware recommendations

### Runner Tools (7 tools)
- `runner_start_autoscaler` - Start GitHub Actions autoscaler
- `runner_stop_autoscaler` - Stop autoscaler
- `runner_get_status` - Get autoscaler status
- `runner_list_workflows` - List workflows
- `runner_provision_for_workflow` - Provision runner for workflow
- `runner_list_containers` - List runner containers
- `runner_stop_container` - Stop runner container

### IPFS Files Tools (7 tools)
- `ipfs_files_add` - Add file to IPFS
- `ipfs_files_get` - Get file from IPFS
- `ipfs_files_cat` - Cat file content
- `ipfs_files_pin` - Pin file
- `ipfs_files_unpin` - Unpin file
- `ipfs_files_list` - List files
- `ipfs_files_validate_cid` - Validate CID

### Network Tools (8 tools)
- `network_list_peers` - List connected peers
- `network_connect_peer` - Connect to peer
- `network_disconnect_peer` - Disconnect from peer
- `network_dht_put` - Put value in DHT
- `network_dht_get` - Get value from DHT
- `network_get_swarm_info` - Get swarm information
- `network_get_bandwidth` - Get bandwidth statistics
- `network_ping_peer` - Ping a peer

## Error Handling

All tool executions include comprehensive error handling:

```javascript
try {
  const result = await client.callTool('github_list_repos', { owner: 'invalid' });
  console.log('Success:', result);
} catch (error) {
  if (error instanceof MCPError) {
    console.error('MCP Error:', error.code, error.message);
    console.error('Additional data:', error.data);
  } else {
    console.error('Network or other error:', error);
  }
}
```

Error codes follow JSON-RPC 2.0 specification:
- `-32700` - Parse error
- `-32600` - Invalid Request
- `-32601` - Method not found / Tool not found
- `-32602` - Invalid params
- `-32603` - Internal error

## Best Practices

1. **Use category-specific methods** when available for better type checking and IDE support
2. **Batch multiple tool calls** when possible to reduce latency
3. **Handle errors gracefully** with try-catch blocks
4. **Cache tool metadata** to avoid repeated API calls
5. **Use search/filter** to quickly find the tool you need
6. **Test tools** with the interactive UI before integrating in code

## Troubleshooting

### Tools not loading
- Check that the MCP server is running
- Verify network connectivity
- Check browser console for errors
- Try refreshing the tools list

### Tool execution fails
- Verify all required parameters are provided
- Check parameter types match the schema
- Review error message for details
- Ensure tool is available and active

### Search not working
- Make sure tools have been loaded first
- Check that search input is properly focused
- Try clearing and re-entering search terms

## Future Enhancements

Planned features for future releases:
- Tool usage statistics and metrics
- Tool favorites/pinning
- Tool execution history
- Advanced filtering options
- Tool documentation viewer
- Example workflow templates

## Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Check the documentation
- Review existing issues for solutions
