"""Non-authoritative DuckLake history projection over exact event ranges.

DuckLake is optional, append-only, rebuildable, and eventually consistent. It
never schedules, leases, completes, or admits federation work. A projection
binds an event range, bounded immutable partitions, checksums, and a cursor.
Identical source-range checksums replay idempotently. Missing or lagging
DuckLake is a typed observation and does not block the DuckDB/Quack plane.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ..task_sources.control_plane_contracts import content_identity
from ..task_sources.quack_state_client import QuackStateClient, StatementKind
from .causal_graph import CausalGraphCommit, CausalGraphError
from .contracts import (
    FederationAuthorityError,
    FederationBinding,
    FederationContractError,
    _identifier,
    _integer,
)
from .fixed_point import FixedPointStore
from .registry import _template
from .retrieval_projection import retrieval_establishes_authority

PROJECTION_STATUSES = frozenset({"current", "lagging", "unavailable", "quarantined"})
MAX_PARTITION_EVENTS = 1_024
MAX_PARTITION_BYTES = 8 * 1024 * 1024
MAX_PARTITIONS = 256


class DuckLakeProjectionError(CausalGraphError):
    """Base typed DuckLake projection failure."""


class DuckLakeProjectionAuthorityError(FederationAuthorityError, DuckLakeProjectionError):
    """An attempt to mint scheduling, completion, or lease authority from DuckLake."""


def projection_establishes_authority() -> bool:
    return False


def projection_establishes_completion() -> bool:
    return False


def _reject_filesystem_path(value: str, name: str) -> None:
    if value.startswith(("/", "~")) or ".." in value.split("/"):
        raise DuckLakeProjectionAuthorityError(
            f"{name} accepts partition identities, not filesystem paths"
        )


@dataclass(frozen=True)
class DuckLakeCapability:
    """Typed DuckLake availability. Absence does not block the control plane."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/ducklake-capability@1"
    )

    available: bool
    lagging: bool = False
    interface: str = "DuckLakeHistoryProjection@1"

    def __post_init__(self) -> None:
        if type(self.available) is not bool or type(self.lagging) is not bool:
            raise FederationContractError("available and lagging must be boolean")
        _identifier(self.interface, "interface")


@dataclass(frozen=True)
class SourceRange:
    """Inclusive event-watermark range bound to one tree and source root."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/ducklake-source-range@1"
    )

    from_watermark: int
    to_watermark: int
    source_root: str
    tree_id: str
    event_count: int
    source_checksum: str = ""

    def __post_init__(self) -> None:
        _integer(self.from_watermark, "from_watermark")
        _integer(self.to_watermark, "to_watermark")
        if self.to_watermark < self.from_watermark:
            raise FederationContractError("source range to_watermark precedes from_watermark")
        _identifier(self.source_root, "source_root")
        _identifier(self.tree_id, "tree_id")
        _integer(self.event_count, "event_count", minimum=1)
        expected = self.to_watermark - self.from_watermark + 1
        if self.event_count != expected:
            raise FederationContractError("source range event_count does not cover the watermarks")
        _identifier(self.source_checksum, "source_checksum", required=False)


@dataclass(frozen=True)
class ProjectionPartition:
    """One bounded, immutable, checksummed history partition."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/ducklake-partition@1"
    )

    partition_id: str
    from_watermark: int
    to_watermark: int
    file_ref: str
    event_count: int
    byte_size: int
    checksum: str = ""

    def __post_init__(self) -> None:
        _identifier(self.partition_id, "partition_id")
        _integer(self.from_watermark, "from_watermark")
        _integer(self.to_watermark, "to_watermark")
        if self.to_watermark < self.from_watermark:
            raise FederationContractError("partition to_watermark precedes from_watermark")
        _reject_filesystem_path(str(self.file_ref), "file_ref")
        _identifier(self.file_ref, "file_ref")
        _integer(self.event_count, "event_count", minimum=1, maximum=MAX_PARTITION_EVENTS)
        expected = self.to_watermark - self.from_watermark + 1
        if self.event_count != expected:
            raise FederationContractError("partition event_count does not cover the watermarks")
        _integer(self.byte_size, "byte_size", minimum=1, maximum=MAX_PARTITION_BYTES)
        digest = partition_checksum(self)
        if self.checksum:
            if self.checksum != digest:
                raise DuckLakeProjectionError("partition checksum does not match the source range")
        else:
            object.__setattr__(self, "checksum", digest)


def partition_checksum(partition: ProjectionPartition) -> str:
    return content_identity(
        {
            "partition_id": partition.partition_id,
            "from_watermark": partition.from_watermark,
            "to_watermark": partition.to_watermark,
            "file_ref": partition.file_ref,
            "event_count": partition.event_count,
            "byte_size": partition.byte_size,
        }
    )


def source_range_checksum(
    source_range: SourceRange,
    partitions: Sequence[ProjectionPartition],
) -> str:
    return content_identity(
        {
            "from_watermark": source_range.from_watermark,
            "to_watermark": source_range.to_watermark,
            "source_root": source_range.source_root,
            "tree_id": source_range.tree_id,
            "event_count": source_range.event_count,
            "partitions": [item.checksum for item in partitions],
        }
    )


@dataclass(frozen=True)
class ProjectionCursor:
    """Durable projection cursor. Replay of a sealed range is idempotent."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/ducklake-projection-cursor@1"
    )

    source_root: str
    watermark: int
    partition_ordinal: int

    def __post_init__(self) -> None:
        _identifier(self.source_root, "source_root")
        _integer(self.watermark, "watermark")
        _integer(self.partition_ordinal, "partition_ordinal")


@dataclass(frozen=True)
class ProjectionReceipt:
    """Observational DuckLake projection evidence. Never authoritative."""

    SCHEMA: ClassVar[str] = (
        "ipfs_accelerate_py/agent-supervisor/causal-federation/ducklake-projection-receipt@1"
    )

    status: str
    source_root: str
    tree_id: str
    from_watermark: int
    to_watermark: int
    source_checksum: str
    cursor_watermark: int
    partition_ids: tuple[str, ...]
    authoritative: bool = False

    def __post_init__(self) -> None:
        status = _identifier(self.status, "status")
        if status not in PROJECTION_STATUSES:
            raise FederationContractError("projection status is not closed")
        object.__setattr__(self, "status", status)
        _identifier(self.source_root, "source_root")
        _identifier(self.tree_id, "tree_id")
        _integer(self.from_watermark, "from_watermark")
        _integer(self.to_watermark, "to_watermark")
        _identifier(self.source_checksum, "source_checksum", required=False)
        _integer(self.cursor_watermark, "cursor_watermark")
        ids = tuple(_identifier(item, "partition_ids") for item in self.partition_ids)
        if len(ids) != len(set(ids)):
            raise FederationContractError("partition_ids contains duplicates")
        object.__setattr__(self, "partition_ids", ids)
        if type(self.authoritative) is not bool:
            raise FederationContractError("authoritative must be boolean")
        if self.authoritative is not False:
            raise DuckLakeProjectionAuthorityError(
                "DuckLake cannot admit scheduling, lease, or completion authority"
            )

    @property
    def cid(self) -> str:
        return content_identity(
            {
                "status": self.status,
                "source_root": self.source_root,
                "tree_id": self.tree_id,
                "from_watermark": self.from_watermark,
                "to_watermark": self.to_watermark,
                "source_checksum": self.source_checksum,
                "partition_ids": list(self.partition_ids),
                "authoritative": False,
            }
        )


def _cover_range(
    source_range: SourceRange,
    partitions: Sequence[ProjectionPartition],
) -> tuple[ProjectionPartition, ...]:
    if not partitions:
        raise DuckLakeProjectionError("current projection requires bounded partitions")
    if len(partitions) > MAX_PARTITIONS:
        raise DuckLakeProjectionError("projection exceeds the bounded partition ceiling")
    ordered = tuple(sorted(partitions, key=lambda item: item.from_watermark))
    if ordered[0].from_watermark != source_range.from_watermark:
        raise DuckLakeProjectionError("partitions do not start at the source-range watermark")
    cursor = source_range.from_watermark
    covered = 0
    for item in ordered:
        if item.from_watermark != cursor:
            raise DuckLakeProjectionError("partitions leave a gap or overlap in the source range")
        covered += item.event_count
        cursor = item.to_watermark + 1
    if cursor - 1 != source_range.to_watermark or covered != source_range.event_count:
        raise DuckLakeProjectionError("partitions do not cover the source range")
    return ordered


def _unavailable_receipt(
    source_range: SourceRange,
    *,
    status: str,
) -> ProjectionReceipt:
    return ProjectionReceipt(
        status=status,
        source_root=source_range.source_root,
        tree_id=source_range.tree_id,
        from_watermark=source_range.from_watermark,
        to_watermark=source_range.to_watermark,
        source_checksum=source_range.source_checksum,
        cursor_watermark=source_range.from_watermark,
        partition_ids=(),
        authoritative=False,
    )


def project_event_range(
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
    """Project one event range into bounded checksummed partitions."""

    if not isinstance(source_range, SourceRange):
        raise FederationContractError("source range is required")
    if not isinstance(binding, FederationBinding):
        raise FederationContractError("binding must be a FederationBinding")
    if not isinstance(capability, DuckLakeCapability):
        raise FederationContractError("DuckLake capability is required")
    if retrieval_establishes_authority() is not False:
        raise DuckLakeProjectionAuthorityError("retrieval cannot mint a DuckLake projection")
    if projection_establishes_authority() is not False:
        raise DuckLakeProjectionAuthorityError(
            "DuckLake cannot admit scheduling, lease, or completion authority"
        )
    _integer(expected_fence, "expected_fence", minimum=1)
    _integer(fencing_epoch, "fencing_epoch", minimum=1)
    if expected_fence != fencing_epoch:
        raise DuckLakeProjectionAuthorityError("source fencing epoch is stale")
    if source_range.tree_id not in binding.repository_tree_ids:
        raise DuckLakeProjectionAuthorityError("projection tree is not bound to the federation")
    if capability.available is not True:
        status = "lagging" if capability.lagging else "unavailable"
        return _unavailable_receipt(source_range, status=status)
    ordered = _cover_range(source_range, partitions)
    checksum = source_range_checksum(source_range, ordered)
    if source_range.source_checksum and source_range.source_checksum != checksum:
        raise DuckLakeProjectionError("source-range checksum does not match the partitions")
    if (
        previous_receipt is not None
        and previous_receipt.from_watermark == source_range.from_watermark
        and previous_receipt.to_watermark == source_range.to_watermark
        and previous_receipt.source_checksum == checksum
        and previous_receipt.status == "current"
    ):
        return previous_receipt
    if previous_cursor is not None:
        if previous_cursor.source_root != source_range.source_root:
            raise DuckLakeProjectionError("projection cursor source root differs")
        if (
            previous_cursor.watermark + 1 != source_range.from_watermark
            and previous_cursor.watermark != source_range.to_watermark
        ):
            raise DuckLakeProjectionError("source range is not contiguous with the projection cursor")
    status = "lagging" if capability.lagging else "current"
    return ProjectionReceipt(
        status=status,
        source_root=source_range.source_root,
        tree_id=source_range.tree_id,
        from_watermark=source_range.from_watermark,
        to_watermark=source_range.to_watermark,
        source_checksum=checksum,
        cursor_watermark=source_range.to_watermark,
        partition_ids=tuple(item.partition_id for item in ordered),
        authoritative=False,
    )


class DuckLakeProjectionWorker:
    """Compile idempotent, range-bound DuckLake history partitions."""

    def project(
        self,
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
        return project_event_range(
            source_range,
            partitions,
            binding=binding,
            capability=capability,
            expected_fence=expected_fence,
            previous_cursor=previous_cursor,
            previous_receipt=previous_receipt,
            fencing_epoch=fencing_epoch,
        )


def _projection_templates() -> tuple[Any, ...]:
    return (
        _template(
            "casf_insert_ducklake_projection_receipt",
            """
            INSERT INTO federation_receipts (
                federation_receipt_id, tenant_id, federation_id, receipt_kind,
                federation_revision, control_plane_generation, event_watermark,
                issuer_id, content_ref, recorded_at
            ) VALUES (?, ?, ?, 'ducklake_projection', ?, ?, ?, ?, ?, ?)
            """,
            (
                "federation_receipt_id",
                "tenant_id",
                "federation_id",
                "federation_revision",
                "control_plane_generation",
                "event_watermark",
                "issuer_id",
                "content_ref",
                "recorded_at",
            ),
        ),
        _template(
            "casf_select_ducklake_projection_receipt",
            """
            SELECT federation_receipt_id, receipt_kind, event_watermark, content_ref
            FROM federation_receipts
            WHERE federation_receipt_id = ? AND tenant_id = ? AND federation_id = ?
            LIMIT 1
            """,
            ("federation_receipt_id", "tenant_id", "federation_id"),
            kind=StatementKind.QUERY,
        ),
    )


class DuckLakeProjectionStore(FixedPointStore):
    """Persist observational DuckLake projection receipts through Quack."""

    INTERFACE = "DuckLakeProjectionStore@1"

    def __init__(
        self,
        client: QuackStateClient,
        *,
        event_notifier: Callable[[int], None] | None = None,
        outbox_notifier: Callable[[int], None] | None = None,
        test_failure_hook: Callable[[str], None] | None = None,
        require_quack_authority: bool = False,
    ) -> None:
        if isinstance(client, (str, bytes, Path)):
            raise DuckLakeProjectionError("DuckLake projection store never accepts a database path")
        if not isinstance(client, QuackStateClient) or not client.attached:
            raise DuckLakeProjectionError(
                "DuckLake projection store requires an already-attached typed state client"
            )
        registered = set(client.list_templates())
        missing = [
            template.name
            for template in _projection_templates()
            if template.name not in registered
        ]
        if client.templates_sealed:
            if missing:
                raise DuckLakeProjectionError(
                    "DuckLake projection templates are absent from the sealed catalog"
                )
        else:
            for template in _projection_templates():
                client.register_template(template)
        super().__init__(
            client,
            event_notifier=event_notifier,
            outbox_notifier=outbox_notifier,
            test_failure_hook=test_failure_hook,
            require_quack_authority=require_quack_authority,
        )

    def record_projection(
        self,
        receipt: ProjectionReceipt,
        *,
        federation_id: str,
        binding: FederationBinding,
        expected_graph_revision: int,
        idempotency_key: str,
    ) -> CausalGraphCommit:
        if not isinstance(receipt, ProjectionReceipt):
            raise FederationContractError("projection receipt is required")
        if receipt.authoritative is not False:
            raise DuckLakeProjectionAuthorityError(
                "DuckLake cannot admit scheduling, lease, or completion authority"
            )
        receipt_id = "federation-receipt:" + receipt.cid
        return self._commit_fact(
            operation="federation.ducklake.projection.record",
            fact_id=receipt_id,
            federation_id=federation_id,
            binding=binding,
            expected_graph_revision=expected_graph_revision,
            idempotency_key=idempotency_key,
            changed_fact_refs=tuple(dict.fromkeys((receipt_id, *receipt.partition_ids))),
            payload_ref=receipt.cid,
            prepare_fact=lambda: None,
            apply_fact=lambda revision, recorded_at: self._insert_projection(
                receipt,
                receipt_id=receipt_id,
                federation_id=federation_id,
                tenant_id=binding.tenant_id,
                generation=binding.control_plane_generation,
                graph_revision=revision,
                recorded_at=recorded_at,
            ),
        )

    def load_projection(
        self,
        *,
        receipt_id: str,
        tenant_id: str,
        federation_id: str,
    ) -> Mapping[str, Any]:
        rows = self._client.execute(
            "casf_select_ducklake_projection_receipt",
            {
                "federation_receipt_id": _identifier(receipt_id, "receipt_id"),
                "tenant_id": _identifier(tenant_id, "tenant_id"),
                "federation_id": _identifier(federation_id, "federation_id"),
            },
        )
        if len(rows) != 1:
            raise DuckLakeProjectionError("DuckLake projection receipt is absent")
        return dict(rows[0])

    def _insert_projection(
        self,
        receipt: ProjectionReceipt,
        *,
        receipt_id: str,
        federation_id: str,
        tenant_id: str,
        generation: int,
        graph_revision: int,
        recorded_at: str,
    ) -> None:
        self._client.execute(
            "casf_insert_ducklake_projection_receipt",
            {
                "federation_receipt_id": receipt_id,
                "tenant_id": tenant_id,
                "federation_id": federation_id,
                "federation_revision": graph_revision,
                "control_plane_generation": generation,
                "event_watermark": receipt.to_watermark,
                "issuer_id": "ducklake-projection",
                "content_ref": receipt.cid,
                "recorded_at": recorded_at,
            },
        )


__all__ = (
    "DuckLakeCapability",
    "DuckLakeProjectionAuthorityError",
    "DuckLakeProjectionError",
    "DuckLakeProjectionStore",
    "DuckLakeProjectionWorker",
    "MAX_PARTITION_BYTES",
    "MAX_PARTITION_EVENTS",
    "ProjectionCursor",
    "ProjectionPartition",
    "ProjectionReceipt",
    "SourceRange",
    "partition_checksum",
    "project_event_range",
    "projection_establishes_authority",
    "projection_establishes_completion",
    "source_range_checksum",
)
