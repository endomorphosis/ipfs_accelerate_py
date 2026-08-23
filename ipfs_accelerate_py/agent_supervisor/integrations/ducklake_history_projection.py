"""Optional non-authoritative DuckLake history projection adapter.

This integration never opens DuckDB, never opens DuckLake, and never admits
scheduling, lease, or completion authority. Federation-owned projection
workers remain the typed source of range/cursor/checksum receipts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from ..federation.contracts import FederationBinding
from ..federation.ducklake_projection import (
    DuckLakeCapability,
    DuckLakeProjectionAuthorityError,
    DuckLakeProjectionWorker,
    ProjectionCursor,
    ProjectionPartition,
    ProjectionReceipt,
    SourceRange,
    project_event_range,
    projection_establishes_authority,
    projection_establishes_completion,
)

__all__ = (
    "DuckLakeCapability",
    "DuckLakeProjectionAuthorityError",
    "DuckLakeProjectionWorker",
    "ProjectionCursor",
    "ProjectionPartition",
    "ProjectionReceipt",
    "SourceRange",
    "project_event_range",
    "project_history",
    "projection_establishes_authority",
    "projection_establishes_completion",
)


def project_history(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Observational adapter. A receipt may be present; it is never authority."""

    if record.get("authoritative") is True:
        raise DuckLakeProjectionAuthorityError(
            "DuckLake cannot admit scheduling, lease, or completion authority"
        )
    return MappingProxyType(
        {
            "authoritative": False,
            "observed": bool(record.get("receipt")),
        }
    )


def project_range(
    source_range: SourceRange,
    partitions: Sequence[ProjectionPartition],
    *,
    binding: FederationBinding,
    capability: DuckLakeCapability,
    expected_fence: int,
    previous_cursor: ProjectionCursor | None = None,
    previous_receipt: ProjectionReceipt | None = None,
    fencing_epoch: int = 1,
) -> ProjectionReceipt:
    """Delegate an event-range projection to the federation worker."""

    return DuckLakeProjectionWorker().project(
        source_range,
        partitions,
        binding=binding,
        capability=capability,
        expected_fence=expected_fence,
        previous_cursor=previous_cursor,
        previous_receipt=previous_receipt,
        fencing_epoch=fencing_epoch,
    )
