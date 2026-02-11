"""
CLI Integrations with Common Cache Infrastructure

This module provides unified CLI wrappers for various AI/development tools,
all using the common cache infrastructure with CID-based lookups.

All CLI integrations:
- Use content-addressed caching (CID-based keys)
- Support automatic retry with exponential backoff
- Share common cache infrastructure
- Provide consistent API across different tools
- **NEW**: Support dual-mode CLI/SDK with automatic fallback
- **NEW**: Integrate with secrets manager for secure credential storage

Available CLI Wrappers:
- GitHubCLIIntegration: GitHub CLI (gh) with caching
- CopilotCLIIntegration: GitHub Copilot CLI with caching
- VSCodeCLIIntegration: VSCode CLI (code) with caching
- OpenAICodexCLIIntegration: OpenAI Codex CLI with caching
- ClaudeCodeCLIIntegration: Claude Code with dual-mode support
- GeminiCLIIntegration: Gemini with dual-mode support
- HuggingFaceCLIIntegration: HuggingFace CLI with caching
- VastAICLIIntegration: Vast AI CLI with caching
- GroqCLIIntegration: Groq with dual-mode support

Usage Example:
    from ipfs_accelerate_py.cli_integrations import GitHubCLIIntegration
    
    # Create GitHub CLI integration with caching
    gh = GitHubCLIIntegration(enable_cache=True)
    
    # List repositories (automatically cached)
    repos = gh.list_repos(owner="endomorphosis", limit=10)
    
    # Second call uses cache (instant response)
    repos = gh.list_repos(owner="endomorphosis", limit=10)

Dual-Mode Example (Phase 3):
    from ipfs_accelerate_py.cli_integrations import ClaudeCodeCLIIntegration
    
    # Initialize with automatic credential retrieval from secrets manager
    claude = ClaudeCodeCLIIntegration()
    
    # Automatically tries CLI first, falls back to SDK
    response = claude.chat("Explain Python decorators")
    print(f"Mode used: {response.get('mode', 'SDK')}")

Cache Benefits:
- 100-500x faster for cached responses
- Automatic CID-based deduplication
- P2P-ready architecture
- Thread-safe operations
"""

from .base_cli_wrapper import BaseCLIWrapper
from .dual_mode_wrapper import DualModeWrapper, detect_cli_tool
from .github_cli_integration import GitHubCLIIntegration, get_github_cli_integration
from .copilot_cli_integration import CopilotCLIIntegration, get_copilot_cli_integration
from .vscode_cli_integration import VSCodeCLIIntegration, get_vscode_cli_integration
from .openai_codex_cli_integration import OpenAICodexCLIIntegration, get_openai_codex_cli_integration
from .claude_code_cli_integration import ClaudeCodeCLIIntegration, get_claude_code_cli_integration
from .gemini_cli_integration import GeminiCLIIntegration, get_gemini_cli_integration
from .huggingface_cli_integration import HuggingFaceCLIIntegration, get_huggingface_cli_integration
from .vastai_cli_integration import VastAICLIIntegration, get_vastai_cli_integration
from .groq_cli_integration import GroqCLIIntegration, get_groq_cli_integration

__all__ = [
    # Base classes
    'BaseCLIWrapper',
    'DualModeWrapper',
    
    # Utilities
    'detect_cli_tool',
    
    # CLI Integration classes
    'GitHubCLIIntegration',
    'CopilotCLIIntegration',
    'VSCodeCLIIntegration',
    'OpenAICodexCLIIntegration',
    'ClaudeCodeCLIIntegration',
    'GeminiCLIIntegration',
    'HuggingFaceCLIIntegration',
    'VastAICLIIntegration',
    'GroqCLIIntegration',
    
    # Global instance getters
    'get_github_cli_integration',
    'get_copilot_cli_integration',
    'get_vscode_cli_integration',
    'get_openai_codex_cli_integration',
    'get_claude_code_cli_integration',
    'get_gemini_cli_integration',
    'get_huggingface_cli_integration',
    'get_vastai_cli_integration',
    'get_groq_cli_integration',
]


def get_all_cli_integrations():
    """
    Get all available CLI integrations.
    
    Returns:
        Dict mapping CLI names to their integration instances
    """
    return {
        'github': get_github_cli_integration(),
        'copilot': get_copilot_cli_integration(),
        'vscode': get_vscode_cli_integration(),
        'openai_codex': get_openai_codex_cli_integration(),
        'claude_code': get_claude_code_cli_integration(),
        'gemini': get_gemini_cli_integration(),
        'huggingface': get_huggingface_cli_integration(),
        'vastai': get_vastai_cli_integration(),
        'groq': get_groq_cli_integration(),
    }
