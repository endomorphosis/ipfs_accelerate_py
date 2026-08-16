"""Accelerator semantic-state consumer package.

Public names are unique, lazily imported, and side-effect free. This module
does not construct a second world snapshot or world view facade.
"""

from __future__ import annotations

from typing import Any

__all__ = (
    "SupervisorWorldView",
    "WorldSnapshotAdmissionError",
    "WorldSnapshotContractError",
    "WorldViewError",
    "build_world_snapshot",
    "parse_world_snapshot",
    "persist_semantic_baseline",
)


def __getattr__(name: str) -> Any:
    if name == "persist_semantic_baseline":
        from ipfs_accelerate_py.agent_supervisor.semantic_state.baseline import (
            persist_semantic_baseline,
        )

        return persist_semantic_baseline
    if name in {"parse_world_snapshot", "WorldSnapshotContractError"}:
        from ipfs_accelerate_py.agent_supervisor.semantic_state.world_snapshot_contracts import (
            WorldSnapshotContractError,
            parse_world_snapshot,
        )

        return {
            "parse_world_snapshot": parse_world_snapshot,
            "WorldSnapshotContractError": WorldSnapshotContractError,
        }[name]
    if name in {"build_world_snapshot", "WorldSnapshotAdmissionError"}:
        from ipfs_accelerate_py.agent_supervisor.semantic_state.world_snapshot_builder import (
            WorldSnapshotAdmissionError,
            build_world_snapshot,
        )

        return {
            "build_world_snapshot": build_world_snapshot,
            "WorldSnapshotAdmissionError": WorldSnapshotAdmissionError,
        }[name]
    if name in {"SupervisorWorldView", "WorldViewError"}:
        from ipfs_accelerate_py.agent_supervisor.semantic_state.world_view import (
            SupervisorWorldView,
            WorldViewError,
        )

        return {
            "SupervisorWorldView": SupervisorWorldView,
            "WorldViewError": WorldViewError,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
