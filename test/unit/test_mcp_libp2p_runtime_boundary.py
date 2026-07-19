"""Regression tests for the MCP++ libp2p runtime boundary."""

from __future__ import annotations

import importlib
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_RUNTIME = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "mcplusplus_module"
    / "p2p"
    / "libp2p_runtime.py"
).resolve()


def _production_python_files() -> list[Path]:
    roots = [
        REPO_ROOT / "ipfs_accelerate_py",
        REPO_ROOT.parent / "ipfs_datasets_py" / "ipfs_datasets_py",
    ]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if "__pycache__" in resolved.parts:
                continue
            files.append(resolved)
    return files


def test_direct_libp2p_imports_stay_inside_mcp_runtime_boundary() -> None:
    direct_import = re.compile(r"^\s*(?:from\s+libp2p\b|import\s+libp2p\b)", re.MULTILINE)
    dynamic_import = re.compile(r"__import__\(\s*[\"']libp2p(?:\.|[\"'])")

    offenders: list[str] = []
    for path in _production_python_files():
        if path == CANONICAL_RUNTIME:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if direct_import.search(text) or dynamic_import.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT.parent)))

    assert offenders == []


def test_legacy_libp2p_compat_wrappers_delegate_to_mcp_runtime() -> None:
    runtime = importlib.import_module(
        "ipfs_accelerate_py.mcplusplus_module.p2p.libp2p_runtime"
    )
    legacy_runtime = importlib.import_module("ipfs_accelerate_py.p2p_tasks.libp2p_runtime")
    legacy_compat = importlib.import_module("ipfs_accelerate_py.github_cli.libp2p_compat")

    assert legacy_runtime.new_libp2p_host is runtime.new_libp2p_host
    assert legacy_runtime.make_kad_dht is runtime.make_kad_dht
    assert legacy_compat.ensure_libp2p_compatible is runtime.ensure_libp2p_compatible
    assert legacy_compat.patch_libp2p_compatibility is runtime.patch_libp2p_compatibility


def test_legacy_github_connectivity_wrapper_delegates_to_mcp_connectivity() -> None:
    canonical = importlib.import_module("ipfs_accelerate_py.mcplusplus_module.p2p.connectivity")
    legacy = importlib.import_module("ipfs_accelerate_py.github_cli.p2p_connectivity")

    assert legacy.ConnectivityConfig is canonical.ConnectivityConfig
    assert legacy.UniversalConnectivity is canonical.UniversalConnectivity
    assert legacy.get_universal_connectivity is canonical.get_universal_connectivity
