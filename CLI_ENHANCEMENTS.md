# CLI Endpoint Adapters - Enhanced Features

## Summary of Enhancements

This document describes the enhancements made to the CLI endpoint adapters based on user feedback.

## 1. VSCode CLI Adapter Added ✅

A new adapter for Visual Studio Code CLI with GitHub Copilot integration.

### Features:
- Code generation and completion
- Code explanation
- GitHub Copilot Chat integration
- Platform-specific installation instructions

### Usage:
```python
from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import VSCodeCLIAdapter, register_cli_endpoint

# Register VSCode CLI
vscode = VSCodeCLIAdapter("vscode_copilot", config={"model": "copilot-chat"})
register_cli_endpoint(vscode)

# Use in multiplexed inference
result = multiplex_inference(
    prompt="Write a Python function to sort a list",
    model_preferences=["vscode_cli/copilot-code", "claude_cli/claude-3-sonnet"]
)
```

## 2. Configuration & Installation Methods ✅

All adapters now include `_config()` and `_install()` methods.

### _config() Method
Returns configuration instructions including:
- Configuration steps
- Environment variables required
- Configuration file locations
- Documentation links

Example:
```python
adapter = ClaudeCodeAdapter("claude1")
config_info = adapter._config()

# Returns:
# {
#     "tool_name": "Claude Code CLI",
#     "config_steps": [...],
#     "env_vars": {"ANTHROPIC_API_KEY": "..."},
#     "config_files": ["~/.config/claude/config.json"],
#     "documentation": "https://..."
# }
```

### _install() Method
Returns platform-specific installation instructions:
- Detects current platform (macOS, Linux, Windows)
- Provides multiple installation methods
- Includes verification commands

Example:
```python
adapter = OpenAICodexAdapter("openai1")
install_info = adapter._install()

# Returns:
# {
#     "tool_name": "OpenAI CLI",
#     "platform": "darwin",
#     "install_methods": [
#         {
#             "method": "pip",
#             "commands": ["pip install openai"]
#         }
#     ],
#     "verify_command": "openai --version"
# }
```

## 3. Input Sanitization & Security Hardening ✅

### sanitize_input()
Validates and sanitizes all string inputs:
- Maximum length enforcement
- Null byte detection
- Pattern matching validation
- Type checking

```python
# Automatically applied to all prompts and inputs
sanitized = sanitize_input(user_input, max_length=10000)
```

### validate_cli_args()
Prevents command injection attacks:
- Detects dangerous patterns (`;`, `|`, `&&`, etc.)
- Logs suspicious activity
- Validates all command arguments

```python
# Automatically applied to all CLI commands
args = validate_cli_args(cmd_args)
```

### Security Features:
- ✅ `shell=False` enforced on all subprocess calls
- ✅ Endpoint IDs validated with alphanumeric pattern
- ✅ Model names sanitized
- ✅ Numeric parameters validated
- ✅ Temperature ranges checked (0.0-1.0)

## 4. Enhanced MCP Server Tools ✅

### New MCP Tools

#### get_cli_config
Get configuration instructions for any CLI tool.

```python
# Via MCP server
result = mcp.call_tool("get_cli_config", cli_type="vscode_cli")
```

#### get_cli_install
Get installation instructions for any CLI tool.

```python
# Via MCP server
result = mcp.call_tool("get_cli_install", cli_type="claude_cli")
```

#### validate_cli_config
Validate configuration of a registered endpoint.

```python
# Via MCP server
result = mcp.call_tool("validate_cli_config", endpoint_id="claude1")
```

#### check_cli_version
Check version of installed CLI tool.

```python
# Via MCP server
result = mcp.call_tool("check_cli_version", endpoint_id="openai1")
```

#### get_cli_capabilities
Get capabilities and features of an endpoint.

```python
# Via MCP server
result = mcp.call_tool("get_cli_capabilities", endpoint_id="gemini1")
```

### Complete MCP Tool List:
1. `register_cli_endpoint_tool` - Register new CLI endpoints
2. `list_cli_endpoints_tool` - List all registered endpoints
3. `cli_inference` - Execute inference via CLI
4. `get_cli_providers` - Get provider information
5. `get_cli_config` ⭐ NEW - Configuration instructions
6. `get_cli_install` ⭐ NEW - Installation instructions
7. `validate_cli_config` ⭐ NEW - Validate configuration
8. `check_cli_version` ⭐ NEW - Check CLI version
9. `get_cli_capabilities` ⭐ NEW - Get capabilities

## 5. Additional Base Adapter Methods ✅

### check_version()
Check installed version of CLI tool.

```python
adapter = GeminiCLIAdapter("gemini1")
version_info = adapter.check_version()
# Returns: {"available": True, "version": "...", "returncode": 0}
```

### validate_config()
Validate current configuration.

```python
adapter.config = {"model": "claude-3-sonnet"}
validation = adapter.validate_config()
# Returns: {"valid": True, "issues": [], "config": {...}}
```

### get_capabilities()
Get adapter capabilities and features.

```python
capabilities = adapter.get_capabilities()
# Returns: {
#     "endpoint_id": "...",
#     "supported_tasks": ["text_generation", "code_generation"],
#     "config_fields": {...},
#     "version_info": {...}
# }
```

## 6. Direct CLI Usage ✅

All features are accessible from the ipfs-accelerate CLI:

```python
# Direct import and usage
from ipfs_accelerate_py.mcp.tools.cli_endpoint_adapters import (
    ClaudeCodeAdapter,
    OpenAICodexAdapter,
    GeminiCLIAdapter,
    VSCodeCLIAdapter
)

# Create and use directly
adapter = ClaudeCodeAdapter("my_claude", config={"model": "claude-3-sonnet"})

# Get configuration
config = adapter._config()
print(config["config_steps"])

# Get installation
install = adapter._install()
print(install["install_methods"])

# Execute inference
result = adapter.execute("Your prompt here", task_type="text_generation")
```

## Supported CLI Tools

| Tool | Provider | Models | Status |
|------|----------|--------|--------|
| Claude Code | Anthropic | claude-3-sonnet, claude-3-opus, claude-3-haiku | ✅ Ready |
| OpenAI CLI | OpenAI | gpt-3.5-turbo, gpt-4, codex | ✅ Ready |
| Google Gemini | Google | gemini-pro, gemini-ultra | ✅ Ready |
| VSCode CLI | GitHub | copilot-chat, copilot-code | ✅ Ready |

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                   MCP Server Layer                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │  9 MCP Tools (config, install, validate, etc.)    │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│              CLI Endpoint Adapters Layer                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐          │
│  │  Security  │  │   Base     │  │  Platform  │          │
│  │ Validation │  │  Adapter   │  │  Detection │          │
│  └────────────┘  └────────────┘  └────────────┘          │
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │  Claude  │ │  OpenAI  │ │  Gemini  │ │  VSCode  │    │
│  │ Adapter  │ │ Adapter  │ │ Adapter  │ │ Adapter  │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│              CLI Tools (Subprocess Execution)               │
│     claude     │     openai     │     gcloud     │ code    │
└────────────────────────────────────────────────────────────┘
```

## Example: Complete Workflow

```python
from ipfs_accelerate_py.mcp.tools.enhanced_inference import multiplex_inference

# 1. Get installation instructions
install_info = mcp.call_tool("get_cli_install", cli_type="vscode_cli")
print("Install VSCode:", install_info["install_methods"])

# 2. Register endpoint
result = mcp.call_tool(
    "register_cli_endpoint_tool",
    cli_type="vscode_cli",
    endpoint_id="vscode_primary",
    model="copilot-chat"
)

# 3. Validate configuration
validation = mcp.call_tool("validate_cli_config", endpoint_id="vscode_primary")

# 4. Check version
version = mcp.call_tool("check_cli_version", endpoint_id="vscode_primary")

# 5. Use in inference with automatic fallback
result = multiplex_inference(
    prompt="Write a function to calculate fibonacci",
    model_preferences=[
        "vscode_cli/copilot-code",
        "claude_cli/claude-3-sonnet",
        "openai/gpt-4"
    ]
)
```

## Security Best Practices

1. ✅ All inputs are sanitized before execution
2. ✅ Command injection patterns are detected and logged
3. ✅ Shell execution is disabled (`shell=False`)
4. ✅ Endpoint IDs are validated with strict patterns
5. ✅ Subprocess timeouts prevent hanging
6. ✅ Environment variables are isolated per adapter
7. ✅ Working directories are configurable and validated

## Testing

All features have been tested:
- ✅ VSCode adapter creation and registration
- ✅ _config() and _install() methods for all adapters
- ✅ Input sanitization with various attack vectors
- ✅ MCP tools integration
- ✅ Direct import and usage
- ✅ Platform detection (macOS, Linux, Windows)

---

For more information, see:
- [CLI Endpoint Integration Guide](CLI_ENDPOINT_INTEGRATION.md)
- [Complete Documentation](docs/CLI_ENDPOINT_ADAPTERS.md)
- [Usage Examples](examples/cli_endpoint_usage.py)
