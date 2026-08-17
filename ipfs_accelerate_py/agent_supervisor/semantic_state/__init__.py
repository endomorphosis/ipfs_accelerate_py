"""Semantic-state package: SCH harness plus LGSWF world snapshot/view.

Importing this package performs no I/O, starts no threads or processes, and
does not open a network connection. LGSWF world-snapshot names stay lazy.
"""

from __future__ import annotations

from typing import Any

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    AcceptanceDisposition,
    Availability,
    ContextPack,
    HarnessDisposition,
    HarnessError,
    HarnessMode,
    HarnessResult,
    ModelRoute,
    PatchProposal,
    RootRef,
    SemanticCapsuleRef,
    SemanticStateRootManifest,
    TestSelectionRef,
    UnavailableResult,
    VerificationReceipt,
    WorkKind,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.harness import (
    HarnessLoopOutcome,
    HarnessPolicy,
    HarnessRequest,
    SemanticCompressionHarness,
    harness_loop_descriptor,
    run_semantic_patch_loop,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import (
    SemanticStateWireCodec,
    semantic_state_interface_descriptor,
)

__all__ = [
    "AcceptanceDisposition",
    "Availability",
    "ContextPack",
    "HarnessDisposition",
    "HarnessError",
    "HarnessLoopOutcome",
    "HarnessMode",
    "HarnessPolicy",
    "HarnessRequest",
    "HarnessResult",
    "ModelRoute",
    "PatchProposal",
    "RootRef",
    "SemanticCapsuleRef",
    "SemanticCompressionHarness",
    "SemanticStateRootManifest",
    "SemanticStateWireCodec",
    "SupervisorWorldView",
    "TestSelectionRef",
    "UnavailableResult",
    "VerificationReceipt",
    "WorkKind",
    "WorldSnapshotAdmissionError",
    "WorldSnapshotContractError",
    "WorldViewError",
    "build_world_snapshot",
    "harness_loop_descriptor",
    "parse_world_snapshot",
    "persist_semantic_baseline",
    "run_semantic_patch_loop",
    "semantic_state_interface_descriptor",
]


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
