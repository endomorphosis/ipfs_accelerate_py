"""PCCE-014: sealer persistence bridge."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.incremental_proof_sealer import (
    INTERFACE,
    V01_PERSISTENCE,
)
from ipfs_accelerate_py.proof_context.dependencies import DependencyUnavailable
from ipfs_accelerate_py.proof_context.sealing_bridge import (
    SimulatedSealError,
    persist_seal,
)


def test_public_capability_does_not_probe_on_import() -> None:
    assert INTERFACE == "IncrementalProofSealer@1"
    assert V01_PERSISTENCE.endswith("incremental_seal_store")


def test_persist_or_unavailable(tmp_path: Path) -> None:
    try:
        ref = persist_seal(tmp_path, b'{"kind":"checkpoint_seal"}')
    except DependencyUnavailable:
        return
    assert ref.cid.startswith("b")
    with pytest.raises(SimulatedSealError):
        persist_seal(tmp_path, b"{}", provenance="simulated")
