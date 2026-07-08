"""Shared core + operations.

Historically, the richer shared implementation lived under `scripts/shared/`
and was imported as a top-level `shared` package by tweaking `PYTHONPATH`.

For service/runtime usage we expose a stable import path:
`ipfs_accelerate_py.shared`.

Installed environments should prefer the packaged `scripts.shared` module when
available, while repository checkouts keep a best-effort fallback that can load
the legacy top-level `shared` import path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def _extract_exports(scripts_shared: Any) -> Dict[str, Any]:
    """Collect the public shared exports from the canonical implementation."""
    exported = {
        name: getattr(scripts_shared, name)
        for name in getattr(scripts_shared, "__all__", [])
        if hasattr(scripts_shared, name)
    }

    # Be defensive: if __all__ is missing, fall back to known public names.
    if not exported:
        for name in (
            "SharedCore",
            "InferenceOperations",
            "FileOperations",
            "ModelOperations",
            "NetworkOperations",
            "QueueOperations",
            "TestOperations",
            "GitHubOperations",
            "CopilotOperations",
            "CopilotSDKOperations",
        ):
            if hasattr(scripts_shared, name):
                exported[name] = getattr(scripts_shared, name)

    return exported


def _try_import_scripts_shared() -> Dict[str, Any]:
    try:
        from scripts import shared as scripts_shared  # type: ignore

        return _extract_exports(scripts_shared)
    except Exception:
        repo_root = Path(__file__).resolve().parents[2]
        scripts_dir = repo_root / "scripts"
        if scripts_dir.exists():
            import sys

            scripts_path = str(scripts_dir)
            if scripts_path not in sys.path:
                sys.path.insert(0, scripts_path)

        # If `scripts/` is on sys.path, then `scripts/shared/` becomes
        # importable as a top-level package named `shared`.
        import shared as scripts_shared  # type: ignore

        return _extract_exports(scripts_shared)


try:
    _exported = _try_import_scripts_shared()
except Exception:  # pragma: no cover
    _exported = {}


if _exported:
    globals().update(_exported)
else:

    class SharedCore:
        def __init__(self):
            pass

        def get_status(self) -> Dict[str, Any]:
            return {"core_available": False, "fallback": True}

    class _UnavailableOps:
        def __init__(self, *_: Any, **__: Any):
            pass

        def __getattr__(self, name: str) -> Any:
            raise RuntimeError(f"Operation '{name}' unavailable (shared fallback)")

    InferenceOperations = _UnavailableOps
    FileOperations = _UnavailableOps
    ModelOperations = _UnavailableOps
    NetworkOperations = _UnavailableOps
    QueueOperations = _UnavailableOps
    TestOperations = _UnavailableOps
    GitHubOperations = _UnavailableOps
    CopilotOperations = _UnavailableOps
    CopilotSDKOperations = _UnavailableOps


__all__ = [
    "SharedCore",
    "InferenceOperations",
    "FileOperations",
    "ModelOperations",
    "NetworkOperations",
    "QueueOperations",
    "TestOperations",
    "GitHubOperations",
    "CopilotOperations",
    "CopilotSDKOperations",
]
