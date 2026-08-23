"""Fixed-point completion and typed termination (EAAEF-104)."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final


FIXED_POINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/external-fixed-point@1"
)


class FixedPointError(ValueError):
    """Termination condition is incomplete."""


def terminate(
    *,
    goals_complete: bool,
    tests_current: bool,
    proofs_current: bool,
    invalidations_empty: bool,
    merge_queue_empty: bool,
    claims_empty: bool,
    source_root: str,
    semantic_root: str,
) -> Mapping[str, Any]:
    if not source_root or not semantic_root:
        raise FixedPointError("source and semantic roots are required")
    ok = all(
        (
            goals_complete,
            tests_current,
            proofs_current,
            invalidations_empty,
            merge_queue_empty,
            claims_empty,
        )
    )
    return MappingProxyType(
        {
            "schema": FIXED_POINT_SCHEMA,
            "terminal": "completed" if ok else "not_complete",
            "source_root": source_root,
            "semantic_root": semantic_root,
        }
    )
