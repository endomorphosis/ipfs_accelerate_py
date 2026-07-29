"""Goose CLI compatibility facade over the canonical cli_runtime adapter.

Public import names and getters stay stable. Construction and listing never
install software, start a Goose process, or probe ``--version``. Chat and
authorized agent methods delegate to :class:`GooseCLIProvider`.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)


def _discover_goose_executable(
    explicit: Optional[str] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Locate goose without starting a process or installing."""
    env = os.environ if environ is None else environ
    if explicit and str(explicit).strip():
        path = os.path.expanduser(str(explicit).strip())
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        which = shutil.which(str(explicit).strip())
        if which:
            return which

    for key in (
        "IPFS_ACCELERATE_GOOSE_PATH",
        "IPFS_ACCELERATE_PY_GOOSE_PATH",
        "ipfs_accelerate_py_GOOSE_BIN",
        "IPFS_ACCELERATE_PY_GOOSE_BIN",
        "IPFS_ACCELERATE_AGENT_GOOSE_BIN",
        "GOOSE_BIN",
    ):
        raw = env.get(key)
        if not raw or not str(raw).strip():
            continue
        path = os.path.expanduser(str(raw).strip())
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        which = shutil.which(str(raw).strip())
        if which:
            return which

    try:
        from ..cli_runtime.installers.goose import discover_goose

        found = discover_goose(probe_version=False, environ=env)
        if found is not None and getattr(found, "available", False) and found.executable:
            return str(found.executable)
    except Exception:
        pass

    return shutil.which("goose")


class GooseCLIIntegration:
    """Compatibility wrapper that delegates to the canonical Goose adapter.

    This is intentionally *not* a :class:`DualModeWrapper` subclass: dual-mode
    construction probes ``--version`` and pulls secrets eagerly. The facade
    stays detect-only until an explicit chat/agent call.
    """

    def __init__(
        self,
        goose_path: Optional[str] = None,
        enable_cache: bool = False,
        cache: Any = None,
        *,
        adapter: Any = None,
        default_model: Optional[str] = None,
        default_goose_provider: Optional[str] = None,
        allow_install: bool = False,
        **kwargs: Any,
    ) -> None:
        # Cache is ignored for side-effecting agent work; chat may still go
        # through llm_router caches at higher layers. Compatibility callers
        # historically defaulted to True — we accept the flag but do not probe.
        _ = cache
        _ = kwargs
        self.enable_cache = bool(enable_cache)
        self.allow_install = bool(allow_install)
        self.default_model = default_model
        self.default_goose_provider = default_goose_provider
        self._explicit_path = goose_path
        self.cli_path = goose_path or _discover_goose_executable() or "goose"
        self._adapter = adapter
        self._adapter_resolved = adapter is not None

    def get_tool_name(self) -> str:
        return "Goose CLI"

    def is_available(self, *, probe: bool = False) -> bool:
        """Detect-only availability (PATH / configured path). Never installs."""
        _ = probe  # version probes are intentionally not performed here
        path = self.cli_path
        if not path:
            return False
        if os.path.sep in path or (os.path.altsep and os.path.altsep in path):
            return os.path.isfile(path) and os.access(path, os.X_OK)
        if path != "goose":
            found = shutil.which(path)
            return found is not None
        return _discover_goose_executable(self._explicit_path) is not None

    def _get_adapter(self) -> Any:
        if self._adapter is not None and self._adapter_resolved:
            return self._adapter
        from ..cli_runtime.providers.goose import create_goose_provider

        executable = _discover_goose_executable(self._explicit_path)
        self.cli_path = executable or self.cli_path or "goose"
        self._adapter = create_goose_provider(
            executable=executable,
            allow_install=self.allow_install,
            default_model=self.default_model,
            default_goose_provider=self.default_goose_provider,
        )
        self._adapter_resolved = True
        return self._adapter

    def get_adapter(self) -> Any:
        """Return the canonical :class:`GooseCLIProvider` (lazy)."""
        return self._get_adapter()

    def chat(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        timeout: float = 180.0,
        goose_provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Safe chat-only generation via the canonical adapter."""
        adapter = self._get_adapter()
        call_kwargs: Dict[str, Any] = {
            "timeout": timeout,
            "agent": False,
            **kwargs,
        }
        if goose_provider is not None:
            call_kwargs["goose_provider"] = goose_provider
        text = adapter.generate(prompt, model_name=model, **call_kwargs)
        return {
            "mode": "cli",
            "provider": "goose_cli",
            "text": text,
            "success": True,
            "side_effecting": False,
        }

    def agent(
        self,
        prompt: str,
        *,
        workspace: str,
        model: Optional[str] = None,
        timeout: float = 600.0,
        max_turns: int = 40,
        path_root: Optional[str] = None,
        goose_provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Authorized agent execution (explicit side-effect policy required)."""
        adapter = self._get_adapter()
        resolved_root = path_root or kwargs.pop("GOOSE_PATH_ROOT", None) or workspace
        call_kwargs: Dict[str, Any] = {
            "timeout": timeout,
            "agent": True,
            "workspace": workspace,
            "path_root": resolved_root,
            "max_turns": max_turns,
            "allow_side_effects": True,
            **kwargs,
        }
        if goose_provider is not None:
            call_kwargs["goose_provider"] = goose_provider
        # Agent is side-effecting: never rely on generic integration caching.
        text = adapter.generate(prompt, model_name=model, **call_kwargs)
        return {
            "mode": "agent",
            "provider": "goose_cli",
            "workspace": workspace,
            "text": text,
            "success": True,
            "side_effecting": True,
        }


_global_goose_cli: Optional[GooseCLIIntegration] = None


def get_goose_cli_integration() -> GooseCLIIntegration:
    """Get or create the global Goose CLI integration instance (lazy, no probe)."""
    global _global_goose_cli
    if _global_goose_cli is None:
        _global_goose_cli = GooseCLIIntegration()
    return _global_goose_cli


def reset_goose_cli_integration() -> None:
    """Test helper: drop the module-level singleton."""
    global _global_goose_cli
    _global_goose_cli = None
