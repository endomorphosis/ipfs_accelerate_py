# IPFS Accelerate CLI Tool

The `ipfs-accelerate` CLI tool provides a unified interface for all IPFS Accelerate functionality, including MCP server management, AI inference operations, file operations, and network management.

## Installation

```bash
pip install ipfs_accelerate_py
```

## Usage

### General Help

```bash
ipfs-accelerate --help
```

### MCP Server Management

#### Start MCP Server
```bash
# Basic server start
ipfs-accelerate mcp start

# Start server on custom port
ipfs-accelerate mcp start --port 8080

# Start server with dashboard
ipfs-accelerate mcp start --dashboard --open-browser
```

#### Start Dashboard Only
```bash
# Start dashboard on default port (8001)
ipfs-accelerate mcp dashboard

# Start dashboard on custom port
ipfs-accelerate mcp dashboard --dashboard-port 8080

# Open dashboard in browser automatically
ipfs-accelerate mcp dashboard --open-browser
```

#### Check Server Status
```bash
ipfs-accelerate mcp status
```

### AI Inference Operations

#### Text Generation
```bash
# Basic text generation
ipfs-accelerate inference generate --prompt "Hello, world!"

# Use specific model
ipfs-accelerate inference generate --model "gpt2" --prompt "Write a story about"

# Control generation parameters
ipfs-accelerate inference generate \
  --model "gpt2" \
  --prompt "Hello world" \
  --max-length 200 \
  --temperature 0.8

# Output as JSON
ipfs-accelerate inference generate \
  --prompt "Hello world" \
  --output-json
```

### File Operations

#### Add Files to IPFS
```bash
# Add a single file
ipfs-accelerate files add /path/to/file.txt

# Add file with JSON output
ipfs-accelerate files add /path/to/file.txt --output-json
```

### Model Management

#### List Available Models
```bash
# List models in human-readable format
ipfs-accelerate models list

# List models as JSON
ipfs-accelerate models list --output-json
```

### Network Operations

#### Check Network Status
```bash
# Check network status
ipfs-accelerate network status

# Get status as JSON
ipfs-accelerate network status --output-json
```

## Examples

### Complete Workflow Example

```bash
# 1. Start the MCP server with dashboard
ipfs-accelerate mcp start --dashboard --open-browser &

# 2. Wait for server to start
sleep 3

# 3. Check server status
ipfs-accelerate mcp status

# 4. List available models
ipfs-accelerate models list

# 5. Generate some text
ipfs-accelerate inference generate \
  --prompt "The future of AI is" \
  --max-length 100

# 6. Check network status
ipfs-accelerate network status

# 7. Add a file to IPFS
echo "Hello IPFS!" > hello.txt
ipfs-accelerate files add hello.txt

# 8. Clean up
rm hello.txt
```

### JSON Output Example

```bash
# All commands support JSON output for programmatic use
ipfs-accelerate models list --output-json | jq '.models[].id'
ipfs-accelerate network status --output-json | jq '.status'
ipfs-accelerate inference generate \
  --prompt "Hello" \
  --output-json | jq '.operation'
```

## Configuration

The CLI tool uses the same shared operations as the MCP server, ensuring consistency between interfaces. It provides fallback implementations when the full IPFS Accelerate core is not available.

### Environment Variables

- `MCP_DISABLE_IPFS`: Set to "1" to disable IPFS functionality
- `DEBUG`: Set to "1" to enable debug logging

### Command-line Options

- `--debug`: Enable debug logging
- `--output-json`: Output results as JSON (available on most commands)
- `--version`: Show version information

## Dashboard Features

The MCP dashboard provides:

- Real-time server status monitoring
- System information display
- Available commands reference
- Auto-refreshing data every 5 seconds
- REST API endpoint (`/api/status`) for programmatic access

![Dashboard Screenshot](https://github.com/user-attachments/assets/872b7ca3-90d6-496f-b538-255247735526)

## Architecture

The CLI tool is built with a shared operations architecture:

```
┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │    │  MCP Server     │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────┬─────────────┬─┘
                 │             │
        ┌────────▼─────────────▼────────┐
        │     Shared Operations         │
        │  ┌─────────────────────────┐  │
        │  │ InferenceOperations     │  │
        │  │ FileOperations          │  │
        │  │ ModelOperations         │  │
        │  │ NetworkOperations       │  │
        │  └─────────────────────────┘  │
        └──────────────┬─────────────────┘
                       │
        ┌──────────────▼─────────────────┐
        │    IPFS Accelerate Core        │
        └────────────────────────────────┘
```

This ensures that both the CLI and MCP server use the same underlying logic and provide consistent results.

## Error Handling

The CLI tool provides graceful error handling:

- Fallback implementations when dependencies are missing
- Clear error messages for common issues
- JSON error responses when `--output-json` is used
- Debug logging for troubleshooting

## Integration

The CLI tool can be easily integrated into scripts and workflows:

```bash
#!/bin/bash
# Example integration script

# Start MCP server
ipfs-accelerate mcp start --port 8080 &
MCP_PID=$!

# Wait for server to be ready
sleep 5

# Run some operations
MODELS=$(ipfs-accelerate models list --output-json)
echo "Available models: $(echo $MODELS | jq -r '.models[].id' | tr '\n' ' ')"

# Generate text
RESULT=$(ipfs-accelerate inference generate \
  --prompt "Hello world" \
  --output-json)
echo "Generated: $(echo $RESULT | jq -r '.result // .output // "No output"')"

# Cleanup
kill $MCP_PID
```

## Troubleshooting

### Common Issues

1. **Import errors**: The CLI provides fallback implementations when dependencies are missing
2. **Port conflicts**: Use custom ports with `--port` and `--dashboard-port` options
3. **Permission errors**: Ensure proper file permissions for file operations

### Debug Mode

Enable debug logging to see detailed operation information:

```bash
ipfs-accelerate --debug models list
```

### Checking Status

Always check the MCP server status if experiencing issues:

```bash
ipfs-accelerate mcp status
```