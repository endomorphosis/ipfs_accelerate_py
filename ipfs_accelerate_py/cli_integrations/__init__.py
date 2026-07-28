"""
CLI Integrations with Common Cache Infrastructure

This module provides unified CLI wrappers for various AI/development tools,
all using the common cache infrastructure with CID-based lookups.

All CLI integrations:
- Use content-addressed caching (CID-based keys) where safe
- Support automatic retry with exponential backoff for non-side-effecting ops
- Share common cache infrastructure
- Provide consistent API across different tools
- Support dual-mode CLI/SDK with automatic fallback (where applicable)
- Integrate with secrets manager for secure credential storage
- Expose lazy, metadata-only discovery that never starts CLI processes

Available CLI Wrappers:
- GitHubCLIIntegration: GitHub CLI (gh) with caching
- CopilotCLIIntegration: GitHub Copilot CLI (production command contract)
- VSCodeCLIIntegration: VSCode CLI (code) with caching
- OpenAICodexCLIIntegration: OpenAI Codex CLI via ``codex exec``
- ClaudeCodeCLIIntegration: Claude Code with dual-mode support
- GeminiCLIIntegration: Gemini with dual-mode support
- HuggingFaceCLIIntegration: HuggingFace CLI with caching
- VastAICLIIntegration: Vast AI CLI with caching
- GroqCLIIntegration: Groq with dual-mode support
- XAIGrokCLIIntegration: xAI Grok Build with Plan Mode, Subagents, and live Web/X Search
- MetaAICLIIntegration: Meta AI / Spark with Creative Mode and Vision Chat support
- GooseCLIIntegration: Block/AAIF Goose CLI facade over the canonical adapter

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

Lazy discovery (no process probes):
    from ipfs_accelerate_py.cli_integrations import list_cli_integrations

    for entry in list_cli_integrations():
        print(entry["name"], entry["getter"])

Cache Benefits:
- 100-500x faster for cached responses
- Automatic CID-based deduplication
- P2P-ready architecture
- Thread-safe operations
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any, Callable, Dict, List, Optional, TypedDict

from .base_cli_wrapper import (
    BaseCLIWrapper,
    parse_argv_override,
    resolve_command_override_from_env,
)
from .dual_mode_wrapper import DualModeWrapper, detect_cli_tool
from .api_key_pool import ApiKeyPool
from .github_cli_integration import GitHubCLIIntegration, get_github_cli_integration
from .copilot_cli_integration import CopilotCLIIntegration, get_copilot_cli_integration
from .vscode_cli_integration import VSCodeCLIIntegration, get_vscode_cli_integration
from .openai_codex_cli_integration import (
    OpenAICodexCLIIntegration,
    get_openai_codex_cli_integration,
)
from .claude_code_cli_integration import (
    ClaudeCodeCLIIntegration,
    get_claude_code_cli_integration,
)
from .gemini_cli_integration import GeminiCLIIntegration, get_gemini_cli_integration
from .huggingface_cli_integration import (
    HuggingFaceCLIIntegration,
    get_huggingface_cli_integration,
)
from .vastai_cli_integration import VastAICLIIntegration, get_vastai_cli_integration
from .groq_cli_integration import GroqCLIIntegration, get_groq_cli_integration
from .xai_grok_cli_integration import XAIGrokCLIIntegration, get_xai_grok_cli_integration
from .meta_ai_cli_integration import MetaAICLIIntegration, get_meta_ai_cli_integration
from .goose_cli_integration import GooseCLIIntegration, get_goose_cli_integration

__all__ = [
    # Base classes
    "BaseCLIWrapper",
    "DualModeWrapper",
    "ApiKeyPool",
    # Utilities
    "detect_cli_tool",
    "parse_argv_override",
    "resolve_command_override_from_env",
    "list_cli_integrations",
    "get_cli_integration_factory",
    # CLI Integration classes
    "GitHubCLIIntegration",
    "CopilotCLIIntegration",
    "VSCodeCLIIntegration",
    "OpenAICodexCLIIntegration",
    "ClaudeCodeCLIIntegration",
    "GeminiCLIIntegration",
    "HuggingFaceCLIIntegration",
    "VastAICLIIntegration",
    "GroqCLIIntegration",
    "XAIGrokCLIIntegration",
    "MetaAICLIIntegration",
    "GooseCLIIntegration",
    # Global instance getters
    "get_github_cli_integration",
    "get_copilot_cli_integration",
    "get_vscode_cli_integration",
    "get_openai_codex_cli_integration",
    "get_claude_code_cli_integration",
    "get_gemini_cli_integration",
    "get_huggingface_cli_integration",
    "get_vastai_cli_integration",
    "get_groq_cli_integration",
    "get_xai_grok_cli_integration",
    "get_meta_ai_cli_integration",
    "get_goose_cli_integration",
    "get_all_cli_integrations",
]


class CLIIntegrationMeta(TypedDict, total=False):
    """Metadata describing a registered CLI integration (no instance)."""

    name: str
    class_name: str
    getter: str
    module: str
    description: str
    side_effecting_default: bool
    command_contract: str


# Registry is pure metadata + import paths. Listing never imports heavy dual-mode
# stacks beyond this package and never constructs wrappers or probes tools.
_CLI_INTEGRATION_REGISTRY: tuple[CLIIntegrationMeta, ...] = (
    {
        "name": "github",
        "class_name": "GitHubCLIIntegration",
        "getter": "get_github_cli_integration",
        "module": "ipfs_accelerate_py.cli_integrations.github_cli_integration",
        "description": "GitHub CLI (gh)",
        "side_effecting_default": False,
        "command_contract": "gh",
    },
    {
        "name": "copilot",
        "class_name": "CopilotCLIIntegration",
        "getter": "get_copilot_cli_integration",
        "module": "ipfs_accelerate_py.cli_integrations.copilot_cli_integration",
        "description": "GitHub Copilot CLI (production @github/copilot contract)",
        "side_effecting_default": True,
        "command_contract": "copilot_production",
    },
    {
        "name": "vscode",
        "class_name": "VSCodeCLIIntegration",
        "getter": "get_vscode_cli_integration",
        "module": "ipfs_accelerate_py.cli_integrations.vscode_cli_integration",
        "description": "VSCode CLI (code)",
        "side_effecting_default": False,
        "command_contract": "code",
    },
    {
        "name": "openai_codex",
        "class_name": "OpenAICodexCLIIntegration",
        "getter": "get_openai_codex_cli_integration",
        "module": "ipfs_accelerate_py.cli_integrations.openai_codex_cli_integration",
        "description": "OpenAI Codex CLI via codex exec",
        "side_effecting_default": True,
        "command_contract": "codex exec",
    },
    {
        "name": "claude_code",
        "class_name": "ClaudeCodeCLIIntegration",
        "getter": "get_claude_code_cli_integration",
        "module": "ipfs_accelerate_py.cli_integrations.claude_code_cli_integration",
        "description": "Claude Code dual-mode CLI/SDK",
        "side_effecting_default": False,
        "command_contract": "claude",
    },
    {
        "name": "gemini",
        "class_name": "GeminiCLIIntegration",
        "getter": "get_gemini_cli_integration",
        "module": "ipfs_accelerate_py.cli_integrations.gemini_cli_integration",
        "description": "Gemini dual-mode CLI/SDK",
        "side_effecting_default": False,
        "command_contract": "gemini",
    },
    {
        "name": "huggingface",
        "class_name": "HuggingFaceCLIIntegration",
        "getter": "get_huggingface_cli_integration",
        "module": "ipfs_accelerate_py.cli_integrations.huggingface_cli_integration",
        "description": "HuggingFace CLI (huggingface-cli)",
        "side_effecting_default": False,
        "command_contract": "huggingface-cli",
    },
    {
        "name": "vastai",
        "class_name": "VastAICLIIntegration",
        "getter": "get_vastai_cli_integration",
        "module": "ipfs_accelerate_py.cli_integrations.vastai_cli_integration",
        "description": "Vast.ai CLI",
        "side_effecting_default": False,
        "command_contract": "vastai",
    },
    {
        "name": "groq",
        "class_name": "GroqCLIIntegration",
        "getter": "get_groq_cli_integration",
        "module": "ipfs_accelerate_py.cli_integrations.groq_cli_integration",
        "description": "Groq dual-mode CLI/SDK",
        "side_effecting_default": False,
        "command_contract": "groq",
    },
    {
        "name": "xai_grok",
        "class_name": "XAIGrokCLIIntegration",
        "getter": "get_xai_grok_cli_integration",
        "module": "ipfs_accelerate_py.cli_integrations.xai_grok_cli_integration",
        "description": "xAI Grok Build dual-mode",
        "side_effecting_default": False,
        "command_contract": "grok",
    },
    {
        "name": "meta_ai",
        "class_name": "MetaAICLIIntegration",
        "getter": "get_meta_ai_cli_integration",
        "module": "ipfs_accelerate_py.cli_integrations.meta_ai_cli_integration",
        "description": "Meta AI / Spark dual-mode",
        "side_effecting_default": False,
        "command_contract": "meta",
    },
    {
        "name": "goose",
        "class_name": "GooseCLIIntegration",
        "getter": "get_goose_cli_integration",
        "module": "ipfs_accelerate_py.cli_integrations.goose_cli_integration",
        "description": "Block/AAIF Goose CLI (canonical adapter facade)",
        "side_effecting_default": False,
        "command_contract": "goose_canonical",
    },
)


def list_cli_integrations() -> List[Dict[str, Any]]:
    """Return metadata for all registered CLI integrations.

    This API never instantiates wrappers, never runs CLI processes, and never
    installs software. Use getters (or :func:`get_cli_integration_factory`) to
    construct a specific integration on demand.
    """
    return [dict(entry) for entry in _CLI_INTEGRATION_REGISTRY]


def get_cli_integration_factory(name: str) -> Callable[[], Any]:
    """Return a zero-arg factory for the named integration (lazy import)."""
    key = str(name or "").strip().lower()
    for entry in _CLI_INTEGRATION_REGISTRY:
        if entry["name"] == key:
            module_name = entry["module"]
            getter_name = entry["getter"]

            def _factory(
                _module: str = module_name,
                _getter: str = getter_name,
            ) -> Any:
                mod = importlib.import_module(_module)
                getter = getattr(mod, _getter)
                return getter()

            return _factory
    raise KeyError(f"Unknown CLI integration: {name!r}")


def get_all_cli_integrations(
    *,
    instantiate: bool = False,
    eager: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Discover CLI integrations.

    By default (``instantiate=False``) returns a mapping of integration name to
    a zero-argument factory. Factories are not invoked, so no CLI process is
    started and dual-mode wrappers are not constructed.

    Pass ``instantiate=True`` (or the deprecated alias ``eager=True``) to build
    every integration immediately. That path is retained for older callers and
    emits a :class:`DeprecationWarning` because it can probe tools that still
    verify on init.

    Returns:
        Dict mapping CLI names to factories (default) or instances (eager).
    """
    if eager is not None:
        instantiate = bool(eager)

    if not instantiate:
        return {
            entry["name"]: get_cli_integration_factory(entry["name"])
            for entry in _CLI_INTEGRATION_REGISTRY
        }

    warnings.warn(
        "get_all_cli_integrations(instantiate=True) eagerly constructs every "
        "CLI wrapper and may probe installed tools. Prefer "
        "list_cli_integrations() for metadata or get_*_cli_integration() for a "
        "single tool.",
        DeprecationWarning,
        stacklevel=2,
    )
    return {
        entry["name"]: get_cli_integration_factory(entry["name"])()
        for entry in _CLI_INTEGRATION_REGISTRY
    }
