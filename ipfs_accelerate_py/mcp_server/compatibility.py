"""Shared compatibility helpers for shim and canonical MCP surfaces."""

from __future__ import annotations

from importlib import import_module
import os
import socket
from typing import Callable, Optional
import urllib.request


class _MissingDependencyStub:
    """Compatibility stub for optional symbols unavailable at import time."""

    def __init__(self, symbol_name: str):
        self._symbol_name = str(symbol_name)

    def __repr__(self) -> str:
        return f"<Unavailable {self._symbol_name}>"

    def __bool__(self) -> bool:
        return False

    def __call__(self, *args, **kwargs):
        _ = args, kwargs
        raise RuntimeError(f"{self._symbol_name} is unavailable in this environment")

    def __getattr__(self, _name: str):
        raise RuntimeError(f"{self._symbol_name} is unavailable in this environment")


def _missing_dependency_stub(symbol_name: str):
    """Create a consistent compatibility stub for an optional symbol."""
    return _MissingDependencyStub(symbol_name)


def _resolve_storage_wrapper_factory() -> Optional[Callable[..., object]]:
    """Resolve a storage-wrapper factory across historical import locations."""
    module_candidates = (
        "ipfs_accelerate_py.common.storage_wrapper",
        "ipfs_accelerate_py.mcplusplus_module.common.storage_wrapper",
        "test.common.storage_wrapper",
    )
    for module_name in module_candidates:
        try:
            module = import_module(module_name)
        except Exception:
            continue
        have_wrapper = bool(getattr(module, "HAVE_STORAGE_WRAPPER", False))
        factory = getattr(module, "get_storage_wrapper", None)
        if have_wrapper and callable(factory):
            return factory
    return None


def _create_storage_wrapper(**kwargs) -> Optional[object]:
    """Create a storage wrapper instance using the canonical resolver contract."""
    factory = _resolve_storage_wrapper_factory()
    if not callable(factory):
        return None
    try:
        return factory(**kwargs)
    except Exception:
        return None


def _resolve_p2p_registrars():
    """Resolve MCP++ P2P registrars with canonical shared fallback behavior."""
    try:
        taskqueue_module = import_module(
            "ipfs_accelerate_py.mcplusplus_module.tools.taskqueue_tools"
        )
        workflow_module = import_module(
            "ipfs_accelerate_py.mcplusplus_module.tools.workflow_tools"
        )
        return (
            taskqueue_module.register_p2p_taskqueue_tools,
            workflow_module.register_p2p_workflow_tools,
        )
    except (ImportError, AttributeError):
        from ipfs_accelerate_py.mcplusplus_module.tools import (
            register_p2p_taskqueue_tools,
            register_p2p_workflow_tools,
        )

        return register_p2p_taskqueue_tools, register_p2p_workflow_tools


def _detect_runner_name() -> str:
    """Detect runner identity from environment with hostname fallback."""
    runner_name = str(os.environ.get("RUNNER_NAME") or "").strip()
    if runner_name:
        return runner_name
    try:
        return socket.gethostname()
    except Exception:
        return "unknown-runner"


def _detect_public_ip() -> Optional[str]:
    """Best-effort public IP detection using multiple redundant services."""
    services = (
        "https://api.ipify.org",
        "https://ifconfig.me/ip",
        "https://icanhazip.com",
    )
    for service in services:
        try:
            with urllib.request.urlopen(service, timeout=5) as response:
                value = response.read().decode("utf-8").strip()
                if value:
                    return value
        except Exception:
            continue
    return None


__all__ = [
    "_MissingDependencyStub",
    "_missing_dependency_stub",
    "_resolve_storage_wrapper_factory",
    "_create_storage_wrapper",
    "_resolve_p2p_registrars",
    "_detect_runner_name",
    "_detect_public_ip",
]