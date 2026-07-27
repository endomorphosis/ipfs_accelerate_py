"""Self-improvement package for agent_supervisor (ASREF).

Owns epoch contracts, successor refill, rollout/benchmark surfaces, and
supervisor v2 efficiency/state models. Higher packages may depend on
``self_improvement``; ``self_improvement`` must not form cycles with
``todo_daemon`` or ``integrations``.

Modules owned by bundle ``asref/self-improvement`` (see
``docs/architecture/asref/move_map.json``) move into this package via
``git mv`` without long-lived re-export stubs at the former flat paths.

**Temporary shadowing note:** Creating this package directory shadows the
flat module ``self_improvement.py`` for
``import ipfs_accelerate_py.agent_supervisor.self_improvement``. Until
ASREF-011 completes the move, this ``__init__`` re-exports the flat
module public API so existing callers keep working. That re-export is
compatibility only—not a permanent stub policy for other packages.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Final

__all__: Final[tuple[str, ...]] = (
    "SELF_IMPROVEMENT_PACKAGE_NAME",
    "SELF_IMPROVEMENT_OWNED_MODULES",
    "SELF_IMPROVEMENT_FORBIDDEN_DEPENDENTS",
)

SELF_IMPROVEMENT_PACKAGE_NAME: Final[str] = (
    "ipfs_accelerate_py.agent_supervisor.self_improvement"
)

# Stems owned by asref/self-improvement in docs/architecture/asref/move_map.json.
SELF_IMPROVEMENT_OWNED_MODULES: Final[tuple[str, ...]] = (
    "self_improvement",
    "self_improvement_completion",
    "self_improvement_rollout",
    "self_improvement_v2",
    "self_improvement_v2_rollout",
    "supervisor_efficiency_metrics",
    "supervisor_state_model",
    "supervisor_token_ledger",
    "supervisor_v2_benchmark",
    "supervisor_v2_contracts",
)

# Packages that must not be imported by self_improvement (DAG / cycle guard).
SELF_IMPROVEMENT_FORBIDDEN_DEPENDENTS: Final[tuple[str, ...]] = (
    "todo_daemon",
    "integrations",
)


def _load_flat_self_improvement() -> Any:
    """Load sibling flat ``self_improvement.py`` under a non-shadowing name.

    Relative imports inside the flat module expect package context
    ``ipfs_accelerate_py.agent_supervisor``.
    """
    legacy_name = "ipfs_accelerate_py.agent_supervisor._self_improvement_flat"
    existing = sys.modules.get(legacy_name)
    if existing is not None:
        return existing

    flat_path = Path(__file__).resolve().parent.parent / "self_improvement.py"
    if not flat_path.is_file():
        raise ImportError(
            "flat self_improvement.py missing; package re-export requires the "
            "pre-move module until ASREF-011 completes git mv ownership"
        )

    spec = importlib.util.spec_from_file_location(legacy_name, flat_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load flat self_improvement from {flat_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = "ipfs_accelerate_py.agent_supervisor"
    sys.modules[legacy_name] = module
    spec.loader.exec_module(module)
    return module


_flat = _load_flat_self_improvement()

# Re-export flat public API for temporary import compatibility.
_flat_all = tuple(getattr(_flat, "__all__", ()))
for _export_name in _flat_all:
    globals()[_export_name] = getattr(_flat, _export_name)
__all__ = tuple(dict.fromkeys((*__all__, *_flat_all)))  # type: ignore[assignment]
del _export_name, _flat_all
