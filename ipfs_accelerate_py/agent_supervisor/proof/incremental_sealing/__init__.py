"""Public IncrementalProofSealer facade (IPS-043).

Imports are lazy and have no process, network, key, or state side effects.
This module only re-exports the seven plan-document APIs.
"""

from __future__ import annotations

from typing import Any

PUBLIC_API_EVIDENCE = "ips/public-api@1"
CLI_EVIDENCE = "ips/cli@1"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "compare_full_and_incremental": (
        "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.explanations",
        "compare_full_and_incremental",
    ),
    "create_full_checkpoint": (
        "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint",
        "create_full_checkpoint",
    ),
    "create_incremental_plan": (
        "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner",
        "create_incremental_plan",
    ),
    "execute_incremental_plan": (
        "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.executor",
        "execute_incremental_plan",
    ),
    "explain_invalidation": (
        "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.explanations",
        "explain_invalidation",
    ),
    "explain_reuse": (
        "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.explanations",
        "explain_reuse",
    ),
    "verify_seal": (
        "ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.verification",
        "verify_seal",
    ),
}

__all__ = (
    "CLI_EVIDENCE",
    "PUBLIC_API_EVIDENCE",
    *sorted(_LAZY_EXPORTS),
)


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    module = __import__(module_name, fromlist=[attr])
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
