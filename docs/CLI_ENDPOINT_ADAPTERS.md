# CLI Endpoint Adapters for IPFS Accelerate

## Overview

The CLI Endpoint Adapters provide seamless integration of command-line AI tools into the IPFS Accelerate queue and multiplexing system. This allows you to use CLI tools like Claude Code, OpenAI Codex, and Google Gemini CLI in the same way you use API endpoints and local models.

## Supported CLI Tools

### 1. Claude Code (Anthropic)
- **CLI Command**: `claude`
- **Models**: claude-3-sonnet, claude-3-opus, claude-3-haiku
- **Use Cases**: Text generation, code generation, analysis
- **Installation**: Follow Anthropic's CLI installation guide

### 2. OpenAI Codex/ChatGPT CLI
- **CLI Command**: `openai`
- **Models**: gpt-3.5-turbo, gpt-4, codex
- **Use Cases**: Text generation, code generation, embeddings
- **Installation**: `pip install openai` (includes CLI)

### 3. Google Gemini CLI
- **CLI Command**: `gemini` or `gcloud`
- **Models**: gemini-pro, gemini-ultra
- **Use Cases**: Text generation, multimodal processing
- **Installation**: Google Cloud SDK with AI platform

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    IPFS Accelerate MCP                       │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         Multiplexed Inference System               │    │
│  │                                                     │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐     │    │
│  │  │  Local   │  │   API    │  │  CLI Tools   │     │    │
│  │  │  Models  │  │ Providers│  │  (NEW!)      │     │    │
│  │  └──────────┘  └──────────┘  └──────────────┘     │    │
│  │       │             │               │              │    │
│  └───────┼─────────────┼───────────────┼──────────────┘    │
│          │             │               │                    │
│          └─────────────┴───────────────┘                    │
│                        │                                    │
│              ┌─────────▼──────────┐                        │
│              │   Queue System     │                        │
│              │  & Load Balancer   │                        │
│              └────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **CLIEndpointAdapter (Base Class)**
   - Abstract base class for all CLI adapters
   - Handles subprocess execution
   - Manages statistics and error handling
   - Auto-detects CLI tool paths

2. **Specific Adapters**
   - `ClaudeCodeAdapter`: For Anthropic's Claude CLI
   - `OpenAICodexAdapter`: For OpenAI CLI tools
   - `GeminiCLIAdapter`: For Google Gemini CLI

3. **Integration Layer**
   - Registers CLI endpoints in the endpoint registry
   - Integrates with queue monitoring
   - Enables multiplexed inference with fallbacks

## Installation

### Prerequisites

Install the CLI tools you want to use:

```bash
# Claude Code (Anthropic)
# Follow installation from: https://claude.ai/cli

# OpenAI CLI
pip install openai

# Google Gemini CLI (via Google Cloud SDK)
# Install from: https://cloud.google.com/sdk/docs/install
gcloud components install ai-platform
```

### Verify Installation

```bash
# Check if CLI tools are available
which claude
which openai
which gcloud
```

## Usage

### Basic Usage

#### 1. Register CLI Endpoints

```python
from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import (
    ClaudeCodeAdapter,
    OpenAICodexAdapter,
    GeminiCLIAdapter,
    register_cli_endpoint
)

# Register Claude Code
claude = ClaudeCodeAdapter(
    endpoint_id="claude_primary",
    config={
        "model": "claude-3-sonnet",
        "max_tokens": 4096,
        "temperature": 0.7
    }
)
register_cli_endpoint(claude)

# Register OpenAI
openai = OpenAICodexAdapter(
    endpoint_id="openai_primary",
    config={
        "model": "gpt-4",
        "max_tokens": 2048
    }
)
register_cli_endpoint(openai)

# Register Gemini
gemini = GeminiCLIAdapter(
    endpoint_id="gemini_primary",
    config={"model": "gemini-pro"}
)
register_cli_endpoint(gemini)
```

#### 2. List Registered Endpoints

```python
from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import list_cli_endpoints

endpoints = list_cli_endpoints()
for endpoint in endpoints:
    print(f"{endpoint['endpoint_id']}: available={endpoint['available']}")
```

#### 3. Execute Inference

```python
from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import execute_cli_inference

result = execute_cli_inference(
    endpoint_id="claude_primary",
    prompt="Explain quantum computing in simple terms",
    task_type="text_generation",
    timeout=30
)

if result["status"] == "success":
    print(result["result"])
```

### Advanced Usage

#### Multiplexed Inference with Fallbacks

```python
from ipfs_accelerate_py.mcp.tools.enhanced_inference import multiplex_inference

result = multiplex_inference(
    prompt="Generate a Python function to sort a list",
    task_type="code_generation",
    model_preferences=[
        "claude_cli/claude-3-sonnet",    # Try CLI first
        "openai_cli/gpt-4",              # Fallback to OpenAI CLI
        "openai/gpt-3.5-turbo",          # Fallback to API
        "local/gpt2"                     # Final fallback
    ],
    max_retries=3
)
```

#### Queue Monitoring

CLI endpoints are automatically integrated into the queue monitoring system:

```python
from ipfs_accelerate_py.mcp.tools.enhanced_inference import get_queue_status

status = get_queue_status()

# View CLI endpoint statistics
for endpoint_id, endpoint in status["endpoint_queues"].items():
    if endpoint["endpoint_type"] == "cli_tool":
        print(f"{endpoint_id}:")
        print(f"  Provider: {endpoint['provider']}")
        print(f"  Status: {endpoint['status']}")
        print(f"  Avg Processing Time: {endpoint['avg_processing_time']}s")
```

### MCP Tools

When integrated with the MCP server, the following tools are available:

#### `register_cli_endpoint_tool`

Register a new CLI endpoint:

```python
# Via MCP
result = mcp.call_tool(
    "register_cli_endpoint_tool",
    cli_type="claude_cli",
    endpoint_id="my_claude_endpoint",
    model="claude-3-sonnet",
    temperature=0.7
)
```

#### `list_cli_endpoints_tool`

List all registered CLI endpoints:

```python
result = mcp.call_tool("list_cli_endpoints_tool")
# Returns: {"endpoints": [...], "count": 3, "status": "success"}
```

#### `cli_inference`

Execute inference via CLI:

```python
result = mcp.call_tool(
    "cli_inference",
    endpoint_id="my_claude_endpoint",
    prompt="Your prompt here",
    task_type="text_generation",
    timeout=30
)
```

#### `get_cli_providers`

Get information about available CLI providers:

```python
result = mcp.call_tool("get_cli_providers")
# Returns provider configurations and supported models
```

## Configuration

### Adapter Configuration Options

All adapters support the following configuration parameters:

```python
config = {
    "model": "model-name",           # Model to use (required)
    "max_tokens": 4096,              # Maximum output tokens
    "temperature": 0.7,              # Sampling temperature (0.0-1.0)
    "env_vars": {                    # Custom environment variables
        "API_KEY": "your-key",
        "CUSTOM_VAR": "value"
    }
}
```

### Custom CLI Path

If the CLI tool is not in PATH, specify the path explicitly:

```python
adapter = ClaudeCodeAdapter(
    endpoint_id="claude_custom",
    cli_path="/usr/local/bin/claude",
    config={"model": "claude-3-sonnet"}
)
```

## Error Handling

The adapters provide comprehensive error handling:

```python
result = execute_cli_inference(
    endpoint_id="my_endpoint",
    prompt="Test prompt",
    task_type="text_generation"
)

if result["status"] == "success":
    # Handle success
    output = result["result"]
elif result["status"] == "timeout":
    # Handle timeout
    print(f"Request timed out after {result['elapsed_time']}s")
elif result["status"] == "error":
    # Handle error
    print(f"Error: {result['error']}")
```

## Statistics and Monitoring

Each adapter tracks the following statistics:

- **requests**: Total number of requests
- **successes**: Number of successful requests
- **failures**: Number of failed requests
- **total_time**: Total execution time
- **avg_time**: Average execution time per request

Access statistics:

```python
stats = adapter.get_stats()
print(f"Success rate: {stats['stats']['successes']}/{stats['stats']['requests']}")
print(f"Average time: {stats['stats']['avg_time']:.2f}s")
```

## Integration with Queue System

CLI endpoints are fully integrated with the existing queue system:

1. **Automatic Registration**: CLI endpoints are added to `ENDPOINT_REGISTRY`
2. **Queue Monitoring**: Visible in `get_queue_status()` results
3. **Load Balancing**: Participate in endpoint selection and load balancing
4. **Failover**: Can be used as fallbacks in multiplexed inference

## Best Practices

### 1. Check Availability

Always check if a CLI tool is available before using it:

```python
if adapter.is_available():
    result = adapter.execute(prompt, task_type)
else:
    # Use fallback method
    pass
```

### 2. Set Appropriate Timeouts

CLI tools may take longer than API calls:

```python
result = execute_cli_inference(
    endpoint_id="my_endpoint",
    prompt=prompt,
    timeout=60  # Longer timeout for complex tasks
)
```

### 3. Use in Multiplexed Mode

Combine CLI tools with API endpoints for resilience:

```python
model_preferences = [
    "claude_cli/claude-3-sonnet",    # Fast, local
    "openai/gpt-4",                  # Reliable API fallback
    "local/gpt2"                     # Always available
]
```

### 4. Monitor Statistics

Regularly check endpoint statistics to identify issues:

```python
endpoints = list_cli_endpoints()
for endpoint in endpoints:
    if endpoint['stats']['failures'] > 10:
        print(f"Warning: {endpoint['endpoint_id']} has high failure rate")
```

## Troubleshooting

### CLI Tool Not Found

**Problem**: Adapter reports CLI tool not available

**Solutions**:
1. Verify CLI tool is installed: `which claude`
2. Add to PATH or specify full path
3. Check execute permissions: `chmod +x /path/to/cli`

### Authentication Errors

**Problem**: CLI returns authentication errors

**Solutions**:
1. Configure API keys: `export ANTHROPIC_API_KEY=...`
2. Run CLI login: `claude login` or `openai auth`
3. Pass environment variables in config

### Timeout Issues

**Problem**: Requests timeout frequently

**Solutions**:
1. Increase timeout parameter
2. Use smaller models or prompts
3. Check system resources

### Parse Errors

**Problem**: Unable to parse CLI output

**Solutions**:
1. Check CLI version compatibility
2. Verify output format (JSON vs text)
3. Review stderr for error messages

## Examples

See `examples/cli_endpoint_usage.py` for comprehensive examples covering:

1. Registering CLI endpoints
2. Listing and monitoring endpoints
3. Running inference
4. Multiplexed inference with fallbacks
5. Queue monitoring integration

## API Reference

### CLIEndpointAdapter

Base class for CLI adapters.

**Methods**:
- `__init__(endpoint_id, cli_path, config)`: Initialize adapter
- `is_available()`: Check if CLI tool is available
- `execute(prompt, task_type, timeout, **kwargs)`: Execute inference
- `get_stats()`: Get endpoint statistics

### ClaudeCodeAdapter

Adapter for Claude Code CLI.

**Inherits**: CLIEndpointAdapter

**Supported Models**: claude-3-sonnet, claude-3-opus, claude-3-haiku

### OpenAICodexAdapter

Adapter for OpenAI CLI tools.

**Inherits**: CLIEndpointAdapter

**Supported Models**: gpt-3.5-turbo, gpt-4, codex

### GeminiCLIAdapter

Adapter for Google Gemini CLI.

**Inherits**: CLIEndpointAdapter

**Supported Models**: gemini-pro, gemini-ultra

## Contributing

To add support for a new CLI tool:

1. Create a new adapter class inheriting from `CLIEndpointAdapter`
2. Implement `_detect_cli_path()`, `_format_prompt()`, and `_parse_response()`
3. Add to `CLI_PROVIDERS` in `enhanced_inference.py`
4. Add tests in `test_cli_endpoint_adapters.py`

## License

Same as IPFS Accelerate Python project.
