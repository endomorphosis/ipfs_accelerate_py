# LLM Router for IPFS Accelerate

The LLM Router provides a unified interface for text generation across multiple LLM providers, with built-in caching, retry logic, and integration with the existing IPFS Accelerate endpoint multiplexing infrastructure.

## Features

- **Unified API**: Single `generate_text()` function works with all providers
- **Multiple Providers**: OpenRouter, Codex CLI, Copilot CLI/SDK, Gemini, Claude, Backend Manager, local HuggingFace
- **Automatic Fallback**: Tries multiple providers in order of availability
- **Response Caching**: CID-based or SHA256-based caching for deterministic results
- **Dependency Injection**: Share resources (caches, managers) across calls
- **Integration**: Works seamlessly with existing CLI wrappers and backend manager
- **No Duplication**: Reuses existing infrastructure (DualModeWrapper, BaseCLIWrapper)

## Quick Start

### Basic Usage

```python
from ipfs_accelerate_py import generate_text

# Auto-select best available provider
response = generate_text("Explain what IPFS is in one sentence")
print(response)
```

### Using a Specific Provider

```python
from ipfs_accelerate_py import generate_text

# Use OpenRouter
response = generate_text(
    "Write a Python function to calculate fibonacci",
    provider="openrouter",
    model_name="openai/gpt-4o-mini",
    max_tokens=256,
    temperature=0.7
)
```

### With Caching

```python
import os
from ipfs_accelerate_py import generate_text

# Enable response cache (enabled by default)
os.environ["IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE"] = "1"

# First call - cache miss
response1 = generate_text("What is 2+2?")

# Second call - cache hit (much faster)
response2 = generate_text("What is 2+2?")
```

### Custom Provider Registration

```python
from ipfs_accelerate_py import register_llm_provider, generate_text

# Define a custom provider
class MyProvider:
    def generate(self, prompt, *, model_name=None, **kwargs):
        return f"Custom response to: {prompt}"

# Register it
register_llm_provider("my_provider", lambda: MyProvider())

# Use it
response = generate_text("test", provider="my_provider")
```

### Dependency Injection

```python
from ipfs_accelerate_py import RouterDeps, generate_text

# Create shared deps container
deps = RouterDeps()

# You can inject pre-configured components
# deps.backend_manager = my_backend_manager
# deps.remote_cache = my_remote_cache

# All calls will share the same resources
response1 = generate_text("First request", deps=deps)
response2 = generate_text("Second request", deps=deps)

print(f"Shared cache has {len(deps.router_cache)} items")
```

## Available Providers

### Built-in Providers

#### 1. OpenRouter (`openrouter`)
API-based access to multiple LLM providers.

**Configuration:**
```bash
export OPENROUTER_API_KEY="your-api-key"
export IPFS_ACCELERATE_PY_OPENROUTER_MODEL="openai/gpt-4o-mini"  # optional
```

**Usage:**
```python
response = generate_text(
    "Your prompt",
    provider="openrouter",
    model_name="openai/gpt-4o-mini",
    max_tokens=256
)
```

#### 2. Codex CLI (`codex_cli`)
Uses existing OpenAI Codex CLI integration.

**Configuration:**
```bash
export IPFS_ACCELERATE_PY_CODEX_MODEL="gpt-3.5-turbo"  # optional
```

**Usage:**
```python
response = generate_text(
    "Write a sorting algorithm",
    provider="codex_cli"
)
```

#### 3. Copilot CLI (`copilot_cli`)
Uses existing GitHub Copilot CLI wrapper.

**Usage:**
```python
response = generate_text(
    "Suggest a git command to...",
    provider="copilot_cli"
)
```

#### 4. Copilot SDK (`copilot_sdk`)
Uses existing GitHub Copilot SDK wrapper.

**Configuration:**
```bash
export IPFS_ACCELERATE_PY_COPILOT_SDK_MODEL="gpt-4o"  # optional
```

**Usage:**
```python
response = generate_text(
    "Your prompt",
    provider="copilot_sdk",
    model_name="gpt-4o"
)
```

#### 5. Gemini (`gemini_cli`)
Uses existing Gemini CLI integration (SDK-based).

**Configuration:**
```bash
export GOOGLE_API_KEY="your-api-key"
export IPFS_ACCELERATE_PY_GEMINI_MODEL="gemini-pro"  # optional
```

**Usage:**
```python
response = generate_text(
    "Explain quantum computing",
    provider="gemini_cli",
    temperature=0.7
)
```

#### 6. Claude (`claude_code`)
Uses existing Claude Code CLI integration (SDK-based).

**Configuration:**
```bash
export ANTHROPIC_API_KEY="your-api-key"
export IPFS_ACCELERATE_PY_CLAUDE_MODEL="claude-3-5-sonnet-20241022"  # optional
```

**Usage:**
```python
response = generate_text(
    "Write a unit test for...",
    provider="claude_code"
)
```

#### 7. Backend Manager (`backend_manager`)
Uses InferenceBackendManager for distributed/multiplexed inference.

**Configuration:**
```bash
export IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER="1"
export IPFS_ACCELERATE_PY_LLM_LOAD_BALANCING="round_robin"  # or least_loaded, best_performance
```

**Usage:**
```python
response = generate_text(
    "Your prompt",
    provider="backend_manager"
)
```

#### 8. Local HuggingFace (`local_hf`)
Fallback to local transformers models.

**Configuration:**
```bash
export IPFS_ACCELERATE_PY_LLM_MODEL="gpt2"  # or any HF model
```

**Usage:**
```python
response = generate_text(
    "Your prompt",
    provider="local_hf",
    model_name="gpt2"
)
```

## Environment Variables

### Provider Selection
- `IPFS_ACCELERATE_PY_LLM_PROVIDER`: Force a specific provider (bypasses auto-detection)

### Caching
- `IPFS_ACCELERATE_PY_ROUTER_CACHE`: Enable/disable provider caching (default: "1")
- `IPFS_ACCELERATE_PY_ROUTER_RESPONSE_CACHE`: Enable/disable response caching (default: "1")
- `IPFS_ACCELERATE_PY_ROUTER_CACHE_KEY`: Cache key strategy ("sha256" or "cid", default: "sha256")
- `IPFS_ACCELERATE_PY_ROUTER_CACHE_CID_BASE`: CID encoding base (default: "base32")

### Backend Manager
- `IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER`: Enable backend manager provider (default: "0")
- `IPFS_ACCELERATE_PY_LLM_LOAD_BALANCING`: Load balancing strategy (default: "round_robin")

### Default Model
- `IPFS_ACCELERATE_PY_LLM_MODEL`: Default model for providers that support it

## Provider Resolution Order

When no provider is specified, the router tries providers in this order:

1. Backend Manager (if enabled via env var)
2. OpenRouter (if API key configured)
3. Codex CLI (if available)
4. Copilot CLI (if available)
5. Copilot SDK (if available)
6. Gemini (if API key configured)
7. Claude (if API key configured)
8. Local HuggingFace (if transformers installed)

## Integration with Existing Infrastructure

### CLI Wrappers
The router integrates with existing CLI wrappers without duplication:

```python
# These are already available and used by the router
from ipfs_accelerate_py.cli_integrations import (
    OpenAICodexCLIIntegration,
    GeminiCLIIntegration,
    ClaudeCodeCLIIntegration
)

from ipfs_accelerate_py.copilot_cli.wrapper import CopilotCLI
from ipfs_accelerate_py.copilot_sdk.wrapper import CopilotSDK
```

### Backend Manager
The router can use the InferenceBackendManager for distributed inference:

```python
from ipfs_accelerate_py import generate_text
import os

# Enable backend manager
os.environ["IPFS_ACCELERATE_PY_ENABLE_BACKEND_MANAGER"] = "1"

# This will route through the backend manager
response = generate_text("Your prompt", provider="backend_manager")
```

### Caching
Response caching integrates with the existing CID-based caching:

```python
from ipfs_accelerate_py import generate_text
import os

# Use CID-based caching
os.environ["IPFS_ACCELERATE_PY_ROUTER_CACHE_KEY"] = "cid"

# Responses are cached by content-addressed CID
response = generate_text("Your prompt")
```

## Advanced Usage

### Provider Instance Management

```python
from ipfs_accelerate_py import get_llm_provider, generate_text

# Get a provider instance
provider = get_llm_provider("openrouter")

# Reuse it for multiple requests
response1 = generate_text("First", provider_instance=provider)
response2 = generate_text("Second", provider_instance=provider)
```

### Custom Dependency Container

```python
from ipfs_accelerate_py import RouterDeps, generate_text
from ipfs_accelerate_py.inference_backend_manager import get_backend_manager

# Create custom deps with pre-configured components
deps = RouterDeps()
deps.backend_manager = get_backend_manager()

# All calls use the same backend manager
response1 = generate_text("First", deps=deps)
response2 = generate_text("Second", deps=deps)
```

### Clear Caches

```python
from ipfs_accelerate_py import clear_llm_router_caches

# Clear internal provider caches
clear_llm_router_caches()
```

## Examples

See `examples/llm_router_example.py` for comprehensive usage examples.

Run the example:
```bash
python examples/llm_router_example.py
```

## Testing

Run the integration tests:
```bash
python test/test_llm_router_integration.py
```

## Architecture

The LLM Router follows these design principles:

1. **No Import-Time Side Effects**: All heavy imports are lazy
2. **Reuse Existing Infrastructure**: No duplication of CLI wrappers or backend managers
3. **Dependency Injection**: Optional `RouterDeps` for sharing resources
4. **Provider Registry**: Extensible via `register_llm_provider()`
5. **Automatic Fallback**: Tries multiple providers in order
6. **CID-Based Caching**: Content-addressed caching for determinism
7. **Integration Ready**: Works with existing endpoint multiplexing

## Benefits Over ipfs_datasets_py Implementation

1. **Full Integration**: Seamlessly works with existing InferenceBackendManager
2. **No Duplication**: Reuses all existing CLI/SDK wrappers
3. **Distributed Ready**: Supports distributed/P2P inference via backend manager
4. **CID Caching**: Built-in CID-based caching support
5. **Existing Patterns**: Follows DualModeWrapper and BaseCLIWrapper patterns
6. **Endpoint Multiplexing**: Can multiplex across peers via backend manager

## Future Enhancements

- [ ] Add streaming support for long responses
- [ ] Add token counting and rate limiting
- [ ] Add more provider-specific optimizations
- [ ] Add metrics and monitoring integration
- [ ] Add distributed caching via libp2p
- [ ] Add provider health checks

## License

See the main project LICENSE file.
