"""CLI vs API routing paths and static provider metadata.

Metadata is catalog-only (no secrets, no live probes on import).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Optional

from .observations import CallProtocol, protocol_for_provider


class RoutingPath(str, Enum):
    CLI = "cli"
    API = "api"


CLI_PROVIDERS: frozenset[str] = frozenset(
    {
        "muse_code",
        "goose_cli",
        "codex_cli",
        "copilot_cli",
        "grok_cli",
        "claude_code",
        "gemini_cli",
        "mistral_vibe",
    }
)

API_PROVIDERS: frozenset[str] = frozenset(
    {
        "meta_ai",
        "openai",
        "openrouter",
        "xai",
        "hf_inference_api",
        "claude_py",
        "gemini_py",
        "copilot_sdk",
    }
)

# Bounded operator metadata for fast session routing. Never includes keys.
PROVIDER_PATH_METADATA: dict[str, dict[str, Any]] = {
    "muse_code": {
        "path": RoutingPath.CLI.value,
        "display_name": "Muse Code CLI",
        "command": "muse",
        "side_effecting": True,
        "auth_env": ("META_API_KEY", "MODEL_API_KEY"),
    },
    "goose_cli": {
        "path": RoutingPath.CLI.value,
        "display_name": "Goose CLI",
        "command": "goose",
        "side_effecting": False,
        "auth_env": ("OPENAI_API_KEY", "GOOSE_PROVIDER"),
    },
    "codex_cli": {
        "path": RoutingPath.CLI.value,
        "display_name": "OpenAI Codex CLI",
        "command": "codex",
        "side_effecting": True,
        "auth_env": ("OPENAI_API_KEY",),
    },
    "copilot_cli": {
        "path": RoutingPath.CLI.value,
        "display_name": "GitHub Copilot CLI",
        "command": "copilot",
        "side_effecting": True,
        "auth_env": (),
    },
    "grok_cli": {
        "path": RoutingPath.CLI.value,
        "display_name": "xAI Grok CLI",
        "command": "grok",
        "side_effecting": False,
        "auth_env": ("XAI_API_KEY",),
    },
    "claude_code": {
        "path": RoutingPath.CLI.value,
        "display_name": "Claude Code CLI",
        "command": "claude",
        "side_effecting": False,
        "auth_env": ("ANTHROPIC_API_KEY",),
    },
    "gemini_cli": {
        "path": RoutingPath.CLI.value,
        "display_name": "Gemini CLI",
        "command": "gemini",
        "side_effecting": False,
        "auth_env": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    },
    "mistral_vibe": {
        "path": RoutingPath.CLI.value,
        "display_name": "Mistral Vibe CLI",
        "command": "vibe",
        "side_effecting": False,
        "auth_env": ("MISTRAL_API_KEY",),
    },
    "meta_ai": {
        "path": RoutingPath.API.value,
        "display_name": "Meta Model API",
        "command": "",
        "side_effecting": False,
        "auth_env": ("MODEL_API_KEY", "META_API_KEY"),
    },
    "openai": {
        "path": RoutingPath.API.value,
        "display_name": "OpenAI API",
        "command": "",
        "side_effecting": False,
        "auth_env": ("OPENAI_API_KEY",),
    },
    "openrouter": {
        "path": RoutingPath.API.value,
        "display_name": "OpenRouter API",
        "command": "",
        "side_effecting": False,
        "auth_env": ("OPENROUTER_API_KEY",),
    },
    "xai": {
        "path": RoutingPath.API.value,
        "display_name": "xAI API",
        "command": "",
        "side_effecting": False,
        "auth_env": ("XAI_API_KEY",),
    },
    "hf_inference_api": {
        "path": RoutingPath.API.value,
        "display_name": "Hugging Face Inference API",
        "command": "",
        "side_effecting": False,
        "auth_env": ("HF_TOKEN", "HUGGINGFACE_API_KEY"),
    },
    "claude_py": {
        "path": RoutingPath.API.value,
        "display_name": "Claude Python wrapper",
        "command": "",
        "side_effecting": False,
        "auth_env": ("ANTHROPIC_API_KEY",),
    },
    "gemini_py": {
        "path": RoutingPath.API.value,
        "display_name": "Gemini Python wrapper",
        "command": "",
        "side_effecting": False,
        "auth_env": ("GEMINI_API_KEY",),
    },
    "copilot_sdk": {
        "path": RoutingPath.API.value,
        "display_name": "GitHub Copilot SDK",
        "command": "",
        "side_effecting": False,
        "auth_env": (),
    },
}


def normalize_routing_path(value: Optional[str]) -> Optional[str]:
    raw = str(value or "").strip().lower().replace("-", "_")
    if not raw:
        return None
    if raw in {RoutingPath.CLI.value, "cli_tools", "cli_tool"}:
        return RoutingPath.CLI.value
    if raw in {RoutingPath.API.value, "http", "sdk", "api_providers", "api_provider"}:
        return RoutingPath.API.value
    return None


def path_for_provider(provider: str) -> str:
    key = str(provider or "").strip().lower().replace("-", "_")
    meta = PROVIDER_PATH_METADATA.get(key)
    if meta:
        return str(meta["path"])
    protocol = protocol_for_provider(key)
    if protocol is CallProtocol.CLI:
        return RoutingPath.CLI.value
    return RoutingPath.API.value


def providers_for_path(path: Optional[str]) -> frozenset[str]:
    normalized = normalize_routing_path(path)
    if normalized == RoutingPath.CLI.value:
        return CLI_PROVIDERS
    if normalized == RoutingPath.API.value:
        return API_PROVIDERS
    return CLI_PROVIDERS | API_PROVIDERS


def filter_names_for_path(names: list[str], path: Optional[str]) -> list[str]:
    allowed = providers_for_path(path)
    if allowed == (CLI_PROVIDERS | API_PROVIDERS):
        return list(names)
    return [name for name in names if name in allowed]


def provider_path_metadata(provider: str) -> Mapping[str, Any]:
    key = str(provider or "").strip().lower().replace("-", "_")
    return dict(PROVIDER_PATH_METADATA.get(key) or {"path": path_for_provider(key)})


class CliResumeStyle(str, Enum):
    """How a CLI provider resumes persisted work."""

    SESSION_ID = "session_id"
    RESUME = "resume"
    RESUME_AND_SESSION_ID = "resume_session_id"
    CODEX_RESUME = "codex_resume"


# Native session flags. Never includes prompts or secrets.
CLI_SESSION_CONTRACT: dict[str, dict[str, Any]] = {
    "muse_code": {
        "style": CliResumeStyle.SESSION_ID.value,
        "inject_kwarg": "session_id",
        "create_on_first": False,
    },
    "goose_cli": {
        "style": CliResumeStyle.RESUME_AND_SESSION_ID.value,
        "inject_kwarg": "session_id",
        "resume_kwarg": "resume_session",
        "create_on_first": True,
        "agent_only": True,
    },
    "copilot_cli": {
        "style": CliResumeStyle.RESUME.value,
        "inject_kwarg": "resume_session_id",
        "create_on_first": False,
    },
    "grok_cli": {
        "style": CliResumeStyle.RESUME.value,
        "inject_kwarg": "resume_session_id",
        "alt_kwarg": "chat_session_id",
        "create_on_first": True,
    },
    "claude_code": {
        "style": CliResumeStyle.RESUME.value,
        "inject_kwarg": "resume_session_id",
        "create_on_first": False,
    },
    "codex_cli": {
        "style": CliResumeStyle.CODEX_RESUME.value,
        "inject_kwarg": "resume_session_id",
        "create_on_first": False,
    },
    "gemini_cli": {
        "style": CliResumeStyle.RESUME.value,
        "inject_kwarg": "resume_session_id",
        "create_on_first": False,
    },
    "mistral_vibe": {
        "style": CliResumeStyle.SESSION_ID.value,
        "inject_kwarg": "session_id",
        "create_on_first": False,
    },
}

_NATIVE_SESSION_KWARGS = (
    "session_id",
    "resume_session_id",
    "chat_session_id",
    "resume_session",
    "continue_session",
)


def cli_session_contract(provider: str) -> dict[str, Any]:
    key = str(provider or "").strip().lower().replace("-", "_")
    return dict(CLI_SESSION_CONTRACT.get(key) or {})


def _kwargs_have_native_session(kwargs: Mapping[str, Any]) -> bool:
    for key in _NATIVE_SESSION_KWARGS:
        value = kwargs.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if key in {"resume_session", "continue_session"} and value:
            return True
    return False


def inject_cli_session_kwargs(
    provider: str,
    kwargs: dict[str, Any],
    *,
    native_session_id: str = "",
    allocation_session_id: str = "",
) -> dict[str, Any]:
    """Bind a stored native CLI session onto generate kwargs. Never overwrites."""
    contract = cli_session_contract(provider)
    if not contract:
        return kwargs
    if _kwargs_have_native_session(kwargs):
        return kwargs
    native = str(native_session_id or "").strip()[:256]
    alloc = str(allocation_session_id or "").strip()[:128]
    existing = bool(native)
    if not native and contract.get("create_on_first") and alloc:
        native = alloc
        existing = False
    if not native:
        return kwargs
    inject_key = str(contract.get("inject_kwarg") or "session_id")
    kwargs[inject_key] = native
    alt_key = str(contract.get("alt_kwarg") or "")
    if alt_key and inject_key == "resume_session_id" and not existing:
        kwargs[alt_key] = native
        kwargs.pop(inject_key, None)
    resume_key = str(contract.get("resume_kwarg") or "")
    if resume_key and existing:
        kwargs[resume_key] = True
    return kwargs


def native_session_from_kwargs(kwargs: Mapping[str, Any]) -> str:
    for key in ("session_id", "resume_session_id", "chat_session_id"):
        value = kwargs.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:256]
    return ""
