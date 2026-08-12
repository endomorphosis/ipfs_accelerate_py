"""Focused semantic-compression harness package.

Importing this package performs no I/O, starts no threads or processes, and
does not open a network connection.
"""

from __future__ import annotations

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
    "HarnessMode",
    "HarnessResult",
    "ModelRoute",
    "PatchProposal",
    "RootRef",
    "SemanticCapsuleRef",
    "SemanticStateRootManifest",
    "SemanticStateWireCodec",
    "TestSelectionRef",
    "UnavailableResult",
    "VerificationReceipt",
    "WorkKind",
    "semantic_state_interface_descriptor",
]
