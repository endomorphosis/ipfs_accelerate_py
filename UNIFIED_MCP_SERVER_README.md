# Unified IPFS Accelerate MCP Server

This project implements a unified Model Context Protocol (MCP) server that consolidates multiple IPFS Accelerate functionalities into a single server interface. The unified approach simplifies client configuration, reduces system resource usage, and provides a consistent interface for all IPFS Accelerate tools.

## Features

The unified MCP server combines four key functional areas:

1. **IPFS Core Functionality**
   - File operations (add, retrieve)
   - Mutable File System (MFS) operations

2. **Model Server Capabilities**
   - Model registry and management
   - Inference endpoint creation
   - Efficient batch inference

3. **API Endpoint Demultiplexer**
   - API key management
   - Load balancing across providers
   - Usage statistics

4. **Task Management**
   - Background task processing
   - Status tracking
   - Task prioritization

## Project Structure

- `unified_mcp_server.py` - The main server implementation
- `unified_mcp_server_test.py` - Comprehensive test script
- `run_unified_mcp.sh` - Setup and run script with dependency installation

## Getting Started

### Prerequisites

- Python 3.7+
- IPFS daemon (optional, but required for IPFS functionality)

### Installation and Setup

The easiest way to get started is to use the provided setup script:

```bash
./run_unified_mcp.sh
```

This script will:
1. Create a Python virtual environment
2. Install all required dependencies
3. Check for a running IPFS daemon
4. Start the unified MCP server
5. Run the test suite to verify functionality

### Manual Setup

If you prefer to set up manually:

1. Install dependencies:
   ```bash
   pip install flask flask_cors ipfshttpclient requests numpy psutil torch
   ```

2. Run the server:
   ```bash
   python unified_mcp_server.py --port 8001
   ```

3. In a separate terminal, run the test script:
   ```bash
   python unified_mcp_server_test.py --url http://localhost:8001
   ```

## Available Tools

### IPFS Core Tools

- `ipfs_add_file`: Add a file to IPFS
  - Parameters: `path` (file path)
  - Returns: CID, size, and name of the added file

- `ipfs_cat`: Retrieve content from IPFS
  - Parameters: `cid` (content identifier)
  - Returns: Content of the file

- `ipfs_files_write`: Write to IPFS MFS
  - Parameters: `path` (MFS path), `content` (file content)
  - Returns: Success status

- `ipfs_files_read`: Read from IPFS MFS
  - Parameters: `path` (MFS path)
  - Returns: Content of the MFS file

### Model Server Tools

- `list_models`: List available models
  - Parameters: None
  - Returns: List of available models and their capabilities

- `create_endpoint`: Create a model inference endpoint
  - Parameters: `model_name`, `device` (optional), `max_batch_size` (optional)
  - Returns: Endpoint ID and status

- `run_inference`: Run inference using a model endpoint
  - Parameters: `endpoint_id`, `inputs` (list of inputs)
  - Returns: Model outputs or embeddings

### API Multiplexer Tools

- `register_api_key`: Register a new API key
  - Parameters: `provider`, `api_key`
  - Returns: Success status

- `get_api_keys`: Get information about registered API keys
  - Parameters: None
  - Returns: List of providers and key counts

- `get_multiplexer_stats`: Get API usage statistics
  - Parameters: None
  - Returns: Request counts, success rates, etc.

- `simulate_api_request`: Simulate an API request
  - Parameters: `provider`, `prompt`
  - Returns: Simulated API response

### Task Management Tools

- `start_task`: Start a background processing task
  - Parameters: `task_type`, `priority` (optional), `params` (optional)
  - Returns: Task ID and status

- `get_task_status`: Get status of a running task
  - Parameters: `task_id`
  - Returns: Task status and results

- `list_tasks`: List all tasks
  - Parameters: `status_filter` (optional), `limit` (optional)
  - Returns: List of tasks and their statuses

### System Tools

- `health_check`: Check server health
  - Parameters: None
  - Returns: Server status information

- `get_hardware_info`: Get information about hardware
  - Parameters: None
  - Returns: CPU, memory, disk, GPU information

- `get_hardware_capabilities`: Get hardware capabilities for inference
  - Parameters: None
  - Returns: Detailed hardware capabilities

- `throughput_benchmark`: Run inference throughput benchmarks
  - Parameters: `model_type` (optional), `batch_sizes` (optional), `devices` (optional)
  - Returns: Benchmark results

- `quantize_model`: Quantize a model to reduce memory usage
  - Parameters: `model_name`, `quantization_type` (optional)
  - Returns: Size reduction information

## Testing

The `unified_mcp_server_test.py` script provides comprehensive testing of all server components. You can test specific categories:

```bash
python unified_mcp_server_test.py --url http://localhost:8001 --test-category connection
python unified_mcp_server_test.py --url http://localhost:8001 --test-category ipfs
python unified_mcp_server_test.py --url http://localhost:8001 --test-category models
python unified_mcp_server_test.py --url http://localhost:8001 --test-category api
python unified_mcp_server_test.py --url http://localhost:8001 --test-category tasks
```

## Extending the Server

You can add new tools to the server by using the `@register_tool` decorator:

```python
@register_tool("my_custom_tool")
def my_custom_tool(param1: str, param2: int = 42) -> Dict[str, Any]:
    """My custom tool description."""
    # Tool implementation
    return {
        "success": True,
        "result": "Custom tool result",
        "param1": param1,
        "param2": param2
    }
```

## API Endpoints

- `/` - Server information
- `/sse` - Server-Sent Events endpoint for MCP
- `/tools` - List all available tools
- `/mcp/manifest` - Get MCP manifest
- `/call_tool` - Call a registered tool

## Requirements

The server handles optional dependencies gracefully. For full functionality, the following packages are recommended:

- `flask` and `flask_cors` - For the HTTP server
- `ipfshttpclient` - For IPFS functionality
- `requests` - For HTTP requests
- `numpy` - For numerical operations
- `psutil` - For hardware information
- `torch` - For model inference

## Integration with ipfs_accelerate_py

This server is designed to integrate with the `ipfs_accelerate_py` package, exposing all of its functions as MCP tools. The error handling and optional dependency management ensure that the server can run with partial functionality even if some components are unavailable.

## Troubleshooting

- **Server fails to start**: Check unified_mcp_server.log for errors
- **IPFS functionality unavailable**: Ensure IPFS daemon is running (`ipfs daemon`)
- **Model server errors**: Check if PyTorch is installed correctly
- **Missing dependencies**: Run `./run_unified_mcp.sh` to install dependencies

## License

This project is licensed under the same license as the ipfs_accelerate_py package.
