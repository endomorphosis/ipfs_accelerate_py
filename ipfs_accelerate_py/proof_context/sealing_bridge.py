"""Accelerator/kit bridge for IncrementalProofSealer persistence (PCCE-014)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ipfs_accelerate_py.proof_context.dependencies import DependencyUnavailable

PERSISTENCE = "ipfs_kit_py.proof_context.incremental_seal_store"


class SimulatedSealError(RuntimeError):
    reason = "simulated"


def open_seal_store(root: str | Path) -> Any:
    try:
        from ipfs_kit_py.proof_context.incremental_seal_store import (
            open_incremental_seal_store,
        )
    except ImportError as exc:
        raise DependencyUnavailable(
            "kit incremental seal store is unavailable; seals are not success"
        ) from exc
    return open_incremental_seal_store(root)


def persist_seal(
    root: str | Path,
    payload: bytes,
    *,
    provenance: str = "live",
    parent_cid: str | None = None,
    expected_parent_cid: str | None = None,
) -> Any:
    if provenance == "simulated":
        raise SimulatedSealError("simulated seals cannot be published")
    store = open_seal_store(root)
    return store.put_seal(
        payload,
        provenance=provenance,
        parent_cid=parent_cid,
        expected_parent_cid=expected_parent_cid,
    )
