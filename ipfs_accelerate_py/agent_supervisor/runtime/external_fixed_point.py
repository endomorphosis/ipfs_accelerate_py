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
    recovery_plan_delta_outstanding: bool = False,
    similarity_not_resolution: bool = False,
    cold_execution_required: bool = False,
    qualification_incomplete: bool = False,
    observations_not_preserved: bool = False,
    negative_memory_blocks: bool = False,
    world_root_cas_not_completion: bool = False,
    boundary_contract_required: bool = False,
) -> Mapping[str, Any]:
    if not source_root or not semantic_root:
        raise FixedPointError("source and semantic roots are required")
    blocked = any(
        (
            recovery_plan_delta_outstanding,
            similarity_not_resolution,
            cold_execution_required,
            qualification_incomplete,
            observations_not_preserved,
            negative_memory_blocks,
            world_root_cas_not_completion,
            boundary_contract_required,
        )
    )
    ok = all(
        (
            goals_complete,
            tests_current,
            proofs_current,
            invalidations_empty,
            merge_queue_empty,
            claims_empty,
            not blocked,
        )
    )
    return MappingProxyType(
        {
            "schema": FIXED_POINT_SCHEMA,
            "terminal": "completed" if ok else "not_complete",
            "source_root": source_root,
            "semantic_root": semantic_root,
            "recovery_plan_delta_outstanding": bool(
                recovery_plan_delta_outstanding
            ),
            "similarity_not_resolution": bool(similarity_not_resolution),
            "cold_execution_required": bool(cold_execution_required),
            "qualification_incomplete": bool(qualification_incomplete),
            "observations_not_preserved": bool(observations_not_preserved),
            "negative_memory_blocks": bool(negative_memory_blocks),
            "world_root_cas_not_completion": bool(world_root_cas_not_completion),
            "boundary_contract_required": bool(boundary_contract_required),
            "completion_authority": False,
        }
    )
