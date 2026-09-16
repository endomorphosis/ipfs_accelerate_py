"""Muse Code CLI compatibility facade over the canonical cli_runtime adapter.

Public import names and getters stay stable. Construction and listing never
install software, start a Muse process, or probe ``--version``. Chat and
authorized agent methods delegate to :class:`MuseCLIProvider`.

Command contract: ``muse exec --prompt-file …`` (headless). Interactive TUI
(``muse``) is not used from this wrapper. Default automation keeps the OS
sandbox (``--disable-approval``); ``--yolo`` requires an explicit flag.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

_PATH_ENV = (
    "IPFS_ACCELERATE_MUSE_PATH",
    "IPFS_ACCELERATE_PY_MUSE_PATH",
    "ipfs_accelerate_py_MUSE_BIN",
    "IPFS_ACCELERATE_PY_MUSE_BIN",
    "IPFS_ACCELERATE_AGENT_MUSE_BIN",
    "MUSE_BIN",
    "MUSE_CLI_PATH",
)


def _discover_muse_executable(
    explicit: Optional[str] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Locate muse without starting a process or installing."""
    env = os.environ if environ is None else environ
    if explicit and str(explicit).strip():
        path = os.path.expanduser(str(explicit).strip())
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        which = shutil.which(str(explicit).strip())
        if which:
            return which

    for key in _PATH_ENV:
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
        from ..cli_runtime.installers.muse import discover_muse

        found = discover_muse(probe_version=False, environ=env)
        if found is not None and getattr(found, "available", False) and found.executable:
            return str(found.executable)
    except Exception:
        pass

    return shutil.which("muse")


class MuseCodeCLIIntegration:
    """Compatibility wrapper that delegates to the canonical Muse Code adapter.

    This is intentionally *not* a :class:`DualModeWrapper` subclass: dual-mode
    construction probes ``--version`` and pulls secrets eagerly. The facade
    stays detect-only until an explicit chat/agent call.
    """

    def __init__(
        self,
        muse_path: Optional[str] = None,
        enable_cache: bool = False,
        cache: Any = None,
        *,
        adapter: Any = None,
        default_model: Optional[str] = None,
        allow_install: bool = False,
        **kwargs: Any,
    ) -> None:
        _ = cache
        _ = kwargs
        self.enable_cache = bool(enable_cache)
        self.allow_install = bool(allow_install)
        self.default_model = default_model
        self._explicit_path = muse_path
        self.cli_path = muse_path or _discover_muse_executable() or "muse"
        self._adapter = adapter
        self._adapter_resolved = adapter is not None

    def get_tool_name(self) -> str:
        return "Muse Code CLI"

    def is_available(self, *, probe: bool = False) -> bool:
        """Detect-only availability (PATH / configured path). Never installs."""
        _ = probe
        path = self.cli_path
        if not path:
            return False
        if os.path.sep in path or (os.path.altsep and os.path.altsep in path):
            return os.path.isfile(path) and os.access(path, os.X_OK)
        if path != "muse":
            found = shutil.which(path)
            return found is not None
        return _discover_muse_executable(self._explicit_path) is not None

    def _get_adapter(self) -> Any:
        if self._adapter is not None and self._adapter_resolved:
            return self._adapter
        from ..cli_runtime.providers.muse import create_muse_provider

        executable = _discover_muse_executable(self._explicit_path)
        self.cli_path = executable or self.cli_path or "muse"
        self._adapter = create_muse_provider(
            executable=executable,
            allow_install=self.allow_install,
            default_model=self.default_model,
        )
        self._adapter_resolved = True
        return self._adapter

    def get_adapter(self) -> Any:
        """Return the canonical :class:`MuseCLIProvider` (lazy)."""
        return self._get_adapter()

    def chat(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        timeout: float = 180.0,
        max_model_steps: int = 8,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Bounded ``muse exec`` generation (sandbox on, no --yolo).

        Muse Code is still a coding agent: this path may edit files. The
        router treats it as side-effecting. ``max_model_steps`` is kept low
        so ordinary generate_text cannot run away.
        """
        adapter = self._get_adapter()
        call_kwargs: Dict[str, Any] = {
            "timeout": timeout,
            "agent": False,
            "max_model_steps": max_model_steps,
            "disable_approval": True,
            **kwargs,
        }
        text = adapter.generate(prompt, model_name=model, **call_kwargs)
        return {
            "mode": "cli",
            "provider": "muse_code",
            "text": text,
            "success": True,
            "side_effecting": True,
            "command_contract": "muse exec",
        }

    def agent(
        self,
        prompt: str,
        *,
        workspace: str,
        model: Optional[str] = None,
        timeout: float = 600.0,
        max_model_steps: int = 40,
        yolo: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Authorized agent execution (explicit side-effect policy required)."""
        adapter = self._get_adapter()
        call_kwargs: Dict[str, Any] = {
            "timeout": timeout,
            "agent": True,
            "workspace": workspace,
            "max_model_steps": max_model_steps,
            "allow_side_effects": True,
            "yolo": bool(yolo),
            "disable_approval": True,
            **kwargs,
        }
        text = adapter.generate(prompt, model_name=model, **call_kwargs)
        return {
            "mode": "agent",
            "provider": "muse_code",
            "workspace": workspace,
            "text": text,
            "success": True,
            "side_effecting": True,
            "command_contract": "muse exec",
        }

    def generate_code(
        self,
        prompt: str,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compatibility alias for :meth:`chat` / bounded ``muse exec``."""
        return self.chat(prompt, model=model, **kwargs)


_global_muse_code_cli: Optional[MuseCodeCLIIntegration] = None


def get_muse_code_cli_integration() -> MuseCodeCLIIntegration:
    """Get or create the global Muse Code CLI integration instance (lazy, no probe)."""
    global _global_muse_code_cli
    if _global_muse_code_cli is None:
        _global_muse_code_cli = MuseCodeCLIIntegration()
    return _global_muse_code_cli


def reset_muse_code_cli_integration() -> None:
    """Test helper: drop the module-level singleton."""
    global _global_muse_code_cli
    _global_muse_code_cli = None
