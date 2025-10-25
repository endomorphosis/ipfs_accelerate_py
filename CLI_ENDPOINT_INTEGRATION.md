# CLI Endpoint Integration - Quick Reference

## Overview

The IPFS Accelerate Python library now supports CLI-based AI tools (Claude Code, OpenAI Codex, Google Gemini CLI) as first-class endpoints in the queue and multiplexing system.

## Quick Start

### 1. Install CLI Tools

```bash
# Claude Code (Anthropic)
# Visit: https://claude.ai/cli

# OpenAI CLI
pip install openai

# Google Gemini (via gcloud)
# Visit: https://cloud.google.com/sdk/docs/install
```

### 2. Register CLI Endpoints

```python
from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import (
    ClaudeCodeAdapter,
    register_cli_endpoint
)

# Register Claude Code
claude = ClaudeCodeAdapter(
    endpoint_id="claude_primary",
    config={"model": "claude-3-sonnet"}
)
register_cli_endpoint(claude)
```

### 3. Use in Multiplexed Inference

```python
from ipfs_accelerate_py.mcp.tools.enhanced_inference import multiplex_inference

result = multiplex_inference(
    prompt="Your prompt here",
    model_preferences=[
        "claude_cli/claude-3-sonnet",   # Try CLI first
        "openai_cli/gpt-4",             # Fallback to OpenAI CLI
        "openai/gpt-3.5-turbo"          # API fallback
    ]
)
```

## Features

✅ **Seamless Integration** - Works with existing queue/multiplexing system
✅ **Auto-Detection** - Automatically finds CLI tools in PATH
✅ **Queue Monitoring** - Full visibility in queue status
✅ **Statistics Tracking** - Per-endpoint metrics and success rates
✅ **Flexible Fallbacks** - Mix CLI and API endpoints
✅ **MCP Tools** - Complete MCP server integration

## Supported CLI Tools

| Tool | Endpoint Type | Models | Status |
|------|--------------|--------|--------|
| Claude Code | `claude_cli` | claude-3-sonnet, claude-3-opus, claude-3-haiku | ✅ Ready |
| OpenAI CLI | `openai_cli` | gpt-3.5-turbo, gpt-4, codex | ✅ Ready |
| Google Gemini | `gemini_cli` | gemini-pro, gemini-ultra | ✅ Ready |

## Documentation

- **Complete Guide**: [`docs/CLI_ENDPOINT_ADAPTERS.md`](docs/CLI_ENDPOINT_ADAPTERS.md)
- **Examples**: [`examples/cli_endpoint_usage.py`](examples/cli_endpoint_usage.py)
- **Tests**: [`test_cli_endpoint_adapters.py`](test_cli_endpoint_adapters.py)

## MCP Server Integration

The CLI adapters expose the following MCP tools:

- `register_cli_endpoint_tool` - Register new CLI endpoints
- `list_cli_endpoints_tool` - List all registered CLI endpoints
- `cli_inference` - Execute inference via CLI
- `get_cli_providers` - Get provider information

## Architecture

```
User Request
     ↓
Multiplexed Inference System
     ↓
┌────┴────┬─────────┬──────────┐
│  Local  │   API   │   CLI    │
│ Models  │ Providers│  Tools   │
└─────────┴─────────┴──────────┘
     ↓         ↓         ↓
  Queue System & Load Balancer
           ↓
      Response
```

## Example: End-to-End Usage

```python
# 1. Register endpoints
from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import *

claude = ClaudeCodeAdapter("claude1", config={"model": "claude-3-sonnet"})
register_cli_endpoint(claude)

# 2. List endpoints
endpoints = list_cli_endpoints()
print(f"Registered: {len(endpoints)} endpoints")

# 3. Run inference
result = execute_cli_inference(
    endpoint_id="claude1",
    prompt="What is machine learning?",
    task_type="text_generation"
)

# 4. Monitor queue
from ipfs_accelerate_py.mcp.tools.enhanced_inference import get_queue_status
status = get_queue_status()
print(f"CLI endpoints: {status['summary']['cli_endpoints']}")
```

## Benefits

1. **Unified Interface** - Use CLI tools just like API endpoints
2. **Cost Optimization** - Mix free CLI tools with paid APIs
3. **Redundancy** - Multiple fallback options
4. **Local Control** - Run models locally via CLI
5. **Queue Integration** - Full monitoring and statistics

## Testing

Run the test suite:

```bash
python test_cli_endpoint_adapters.py
```

Run the examples:

```bash
python examples/cli_endpoint_usage.py
```

## Next Steps

1. ✅ Install desired CLI tools
2. ✅ Configure authentication/API keys
3. ✅ Register endpoints
4. ✅ Test with real requests
5. ✅ Monitor via queue status

---

For detailed information, see [docs/CLI_ENDPOINT_ADAPTERS.md](docs/CLI_ENDPOINT_ADAPTERS.md)
