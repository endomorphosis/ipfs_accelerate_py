# Phases 3-4 Implementation: Dual-Mode CLI/API and Secrets Manager

This document describes the implementation of Phases 3 and 4 from Pull Request #72.

## Overview

**Phase 3: CLI/API Dual Mode Support**
- Unified interface that tries CLI execution first, then falls back to Python SDK
- Enables flexibility to work with or without CLI tools installed
- Seamless integration with existing cache infrastructure

**Phase 4: Secrets Manager**
- Encrypted credential storage for secure API key management
- Environment variable fallback support
- Integration with all CLI/API providers
- Uses Fernet symmetric encryption from the cryptography library

## Phase 4: Secrets Manager

### Features

- **Encrypted Storage**: Uses Fernet (symmetric encryption) to protect credentials at rest
- **Environment Variable Fallback**: Automatically checks environment variables if credential not found in storage
- **Secure File Permissions**: Sets restrictive permissions (0o600) on secrets files
- **Persistence**: Credentials are automatically saved to disk
- **Global Instance**: Singleton pattern for consistent access across application

### Usage

#### Basic Usage

```python
from ipfs_accelerate_py.common.secrets_manager import get_global_secrets_manager

# Get global secrets manager instance
secrets = get_global_secrets_manager()

# Store a credential
secrets.set_credential("openai_api_key", "sk-...")

# Retrieve a credential
api_key = secrets.get_credential("openai_api_key")

# List all credential keys (not values)
keys = secrets.list_credential_keys()

# Delete a credential
secrets.delete_credential("old_api_key")
```

#### Environment Variable Fallback

The secrets manager automatically checks environment variables with multiple naming conventions:

```python
# All of these will find the same environment variable:
secrets.get_credential("openai_api_key")  # Checks OPENAI_API_KEY
secrets.get_credential("openai-api-key")  # Checks OPENAI_API_KEY
secrets.get_credential("openai.api.key")  # Checks OPENAI_API_KEY
```

Priority order:
1. In-memory cache (previously set credentials)
2. Environment variables (various naming formats)
3. Default value (if provided)

#### Storage Location

Default storage location: `~/.ipfs_accelerate/secrets.enc`

The encryption key is stored at: `~/.ipfs_accelerate/secrets.key`

You can also specify custom locations:

```python
from ipfs_accelerate_py.common.secrets_manager import SecretsManager

secrets = SecretsManager(
    secrets_file="/custom/path/secrets.enc",
    use_encryption=True
)
```

#### Disabling Encryption

For development or testing, you can disable encryption:

```python
secrets = SecretsManager(use_encryption=False)
```

**Warning**: Only use unencrypted storage in secure, isolated environments.

### Integration with CLI Tools

All CLI integrations now automatically retrieve API keys from the secrets manager:

```python
from ipfs_accelerate_py.cli_integrations.claude_code_cli_integration import ClaudeCodeCLIIntegration

# Automatically retrieves 'anthropic_api_key' from secrets manager
claude = ClaudeCodeCLIIntegration()

# Or provide explicitly
claude = ClaudeCodeCLIIntegration(api_key="sk-...")
```

### Supported Credential Keys

The following credential keys are automatically recognized:

- `openai_api_key` - OpenAI API
- `anthropic_api_key` - Anthropic/Claude API
- `google_api_key` - Google/Gemini API
- `groq_api_key` - Groq API
- `huggingface_token` - HuggingFace Hub

## Phase 3: CLI/API Dual Mode Support

### Features

- **Automatic CLI Detection**: Detects if CLI tools are installed in system PATH
- **Seamless Fallback**: Falls back to Python SDK if CLI not available or fails
- **Configurable Preference**: Choose to prefer CLI or SDK mode
- **Unified Caching**: Both modes use the same cache infrastructure
- **Error Recovery**: Automatically tries alternative mode on failure

### Architecture

The `DualModeWrapper` base class provides:

1. **CLI Detection**: Auto-detects CLI tools using `detect_cli_tool()`
2. **SDK Client Management**: Lazy-loads SDK clients when needed
3. **Fallback Logic**: Executes operations with automatic fallback
4. **Secrets Integration**: Retrieves API keys from secrets manager

### Usage

#### Claude Integration

```python
from ipfs_accelerate_py.cli_integrations.claude_code_cli_integration import ClaudeCodeCLIIntegration

# Initialize with SDK mode (default, since no official CLI)
claude = ClaudeCodeCLIIntegration()

# Send a chat message
response = claude.chat(
    message="Explain async/await in Python",
    model="claude-3-sonnet-20240229",
    temperature=0.0
)

print(response["response"])
print(f"Mode used: {response.get('mode', 'SDK')}")
print(f"Cached: {response.get('cached', False)}")
```

#### Gemini Integration

```python
from ipfs_accelerate_py.cli_integrations.gemini_cli_integration import GeminiCLIIntegration

# Initialize
gemini = GeminiCLIIntegration()

# Generate text
response = gemini.generate_text(
    prompt="Write a Python function to sort a list",
    model="gemini-pro",
    temperature=0.0
)

print(response["response"])
```

#### Groq Integration

```python
from ipfs_accelerate_py.cli_integrations.groq_cli_integration import GroqCLIIntegration

# Initialize
groq = GroqCLIIntegration()

# Chat
response = groq.chat(
    message="What is machine learning?",
    model="llama3-70b-8192",
    temperature=0.0
)

print(response["response"])
```

### CLI Detection

The `detect_cli_tool()` utility function automatically searches for CLI tools:

```python
from ipfs_accelerate_py.cli_integrations.dual_mode_wrapper import detect_cli_tool

# Try multiple possible names
cli_path = detect_cli_tool(["claude", "claude-cli"])

if cli_path:
    print(f"Found CLI at: {cli_path}")
else:
    print("CLI not found, will use SDK")
```

### Prefer CLI vs SDK

You can configure whether to prefer CLI or SDK mode:

```python
# Prefer SDK (default for Claude/Gemini/Groq)
claude = ClaudeCodeCLIIntegration(prefer_cli=False)

# Prefer CLI (will try CLI first, fall back to SDK)
claude = ClaudeCodeCLIIntegration(prefer_cli=True)
```

### Response Format

All dual-mode operations return a dictionary with:

```python
{
    "response": "...",        # The actual response content
    "cached": False,          # Whether response came from cache
    "mode": "SDK",            # Which mode was used: "CLI" or "SDK"
    "fallback": False         # Whether fallback was used
}
```

## Testing

### Phase 4 Tests (Secrets Manager)

Run the secrets manager tests:

```bash
python3 test_phase4_secrets_manager.py
```

Tests include:
- Basic credential storage and retrieval
- Persistence (save and reload)
- Environment variable fallback
- Encryption verification
- Unencrypted mode
- Global instance management

### Phase 3 Tests (Dual Mode)

Run the dual-mode tests:

```bash
python3 test_phase3_dual_mode.py
```

Tests include:
- CLI tool detection
- Integration imports
- Claude dual-mode initialization
- Gemini dual-mode initialization
- Groq dual-mode initialization
- Secrets manager integration

## Security Considerations

### Secrets Manager

1. **Encryption**: Uses Fernet (AES-128 in CBC mode with HMAC for authentication)
2. **Key Storage**: Encryption key stored separately from secrets file
3. **File Permissions**: Restricts file access to owner only (0o600)
4. **Environment Variables**: Checks environment for keys as fallback
5. **No Plaintext**: Credentials never stored in plaintext (when encryption enabled)

### Best Practices

1. **Use Encryption**: Always enable encryption in production
2. **Secure Key Storage**: Protect the encryption key file
3. **Environment Variables**: Use environment variables in CI/CD environments
4. **Rotation**: Regularly rotate API keys
5. **Access Control**: Limit file system access to secrets directory

## Migration from Previous Implementation

### For Existing Code Using Claude

```python
# Old code
from ipfs_accelerate_py.cli_integrations.claude_code_cli_integration import ClaudeCodeCLIIntegration

claude = ClaudeCodeCLIIntegration(api_key="sk-...")
response = claude.chat("Hello")

# New code (compatible, adds dual-mode and secrets support)
from ipfs_accelerate_py.cli_integrations.claude_code_cli_integration import ClaudeCodeCLIIntegration

# API key now retrieved from secrets manager if not provided
claude = ClaudeCodeCLIIntegration()
response = claude.chat("Hello")
# Response now includes mode info
print(f"Used {response.get('mode', 'SDK')} mode")
```

### Setting Up Credentials

```python
from ipfs_accelerate_py.common.secrets_manager import get_global_secrets_manager

# One-time setup
secrets = get_global_secrets_manager()
secrets.set_credential("anthropic_api_key", "sk-ant-...")
secrets.set_credential("google_api_key", "AIza...")
secrets.set_credential("groq_api_key", "gsk_...")

# Credentials are now available to all integrations
```

## Dependencies

Phase 4 (Secrets Manager) requires:
- `cryptography` - For encryption (install with: `pip install cryptography`)

Phase 3 (Dual Mode) requires:
- Existing SDK dependencies (anthropic, google-generativeai, groq)
- No additional dependencies

## Files Added/Modified

### New Files
- `ipfs_accelerate_py/common/secrets_manager.py` - Secrets manager implementation
- `ipfs_accelerate_py/cli_integrations/dual_mode_wrapper.py` - Dual-mode base class
- `test_phase4_secrets_manager.py` - Phase 4 tests
- `test_phase3_dual_mode.py` - Phase 3 tests
- `PHASES_3_4_IMPLEMENTATION.md` - This documentation

### Modified Files
- `ipfs_accelerate_py/cli_integrations/claude_code_cli_integration.py` - Added dual-mode support
- `ipfs_accelerate_py/cli_integrations/gemini_cli_integration.py` - Added dual-mode support
- `ipfs_accelerate_py/cli_integrations/groq_cli_integration.py` - Added dual-mode support

## Future Enhancements

1. **CLI Wrapper Scripts**: Create actual CLI wrapper scripts for providers that don't have official CLIs
2. **Secrets Rotation**: Automatic API key rotation functionality
3. **Multi-Backend Support**: Support for additional secret storage backends (e.g., AWS Secrets Manager, HashiCorp Vault)
4. **Audit Logging**: Track credential access and usage
5. **Role-Based Access**: Different credential sets for different use cases

## Support

For issues or questions:
- Open an issue on GitHub
- Reference Pull Request #72 for context
- Include test results when reporting problems
