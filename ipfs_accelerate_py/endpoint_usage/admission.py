"""Versioned stored-state pins, separate from timestamped usage observations.

A pin identifies a whole ledger generation and the coordinator's configured
partition. Reads and clock advances do not change it; every stored write may.
It is a consistency precondition, never authority to spend capacity or a promise
that time-dependent headroom remains unchanged. Reservation admission must still
evaluate all limits using its current clock inside the owner's CAS transaction.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Optional

from .identity import require_catalog_id, stable_id
from .schema import UsageSnapshot
from .store import CapacityPartition, document_cid


USAGE_ADMISSION_SNAPSHOT_SCHEMA = "ipfs_accelerate_py.endpoint_usage.admission_snapshot@1"
USAGE_ADMISSION_SNAPSHOT_VERSION = 1
_ADMISSION_REVISION = re.compile(r"uadm_[0-9a-f]{64}\Z")
_PARTITION_ID = re.compile(r"upart_[0-9a-f]{64}\Z")
_DOCUMENT_CID = re.compile(r"b[a-z2-7]{58}\Z")


def validate_admission_revision(value: Any) -> str:
    if not isinstance(value, str) or not _ADMISSION_REVISION.fullmatch(value):
        raise ValueError("admission revision must be a uadm_ identity with 64 lowercase hexadecimal digits")
    return value


def _generation(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _cid(value: Any) -> str:
    if not isinstance(value, str) or not _DOCUMENT_CID.fullmatch(value):
        raise ValueError("document_cid must be a canonical CIDv1/raw/sha2-256")
    encoded = value[1:]
    binary = base64.b32decode(encoded.upper() + "=" * (-len(encoded) % 8))
    if (len(binary) != 36 or binary[:4] != b"\x01\x55\x12\x20"
            or base64.b32encode(binary).decode("ascii").lower().rstrip("=") != encoded):
        raise ValueError("document_cid must be a canonical CIDv1/raw/sha2-256")
    return value


def coordinator_partition_identity(partition: Optional[CapacityPartition]) -> str:
    """Bind configured authority even before its next materializing write."""
    if partition is None:
        return ""
    if not isinstance(partition, CapacityPartition):
        raise TypeError("coordinator_partition must be a CapacityPartition")
    return stable_id("upart", partition.to_dict())


def _revision(
    scope_id: str, ledger_revision: int, fence: int, state_cid: str, partition_id: str
) -> str:
    return stable_id("uadm", {
        "schema": USAGE_ADMISSION_SNAPSHOT_SCHEMA,
        "schema_version": USAGE_ADMISSION_SNAPSHOT_VERSION,
        "scope_id": scope_id,
        "ledger_revision": ledger_revision,
        "fence": fence,
        "document_cid": state_cid,
        "coordinator_partition_id": partition_id,
    })


def admission_revision_for_document(
    document: Mapping[str, Any], scope_id: str, *,
    coordinator_partition: Optional[CapacityPartition] = None,
) -> str:
    """Compute a pin without a clock read, snapshot rebuild, or store write."""
    if not isinstance(document, Mapping):
        raise TypeError("document must be a mapping")
    return _revision(
        require_catalog_id(scope_id, "scope_id"),
        _generation(document.get("revision"), "ledger_revision"),
        _generation(document.get("fence"), "fence"),
        document_cid(document),
        coordinator_partition_identity(coordinator_partition),
    )


@dataclass(frozen=True)
class UsageAdmissionSnapshot:
    """One immutable ledger read paired with its ordinary usage observation.

    Whole-ledger pins intentionally invalidate on unrelated scope writes,
    compaction, and the legacy coordinator's no-op replay transactions. This
    conservative first version needs no stored document migration.
    """

    SCHEMA: ClassVar[str] = USAGE_ADMISSION_SNAPSHOT_SCHEMA
    scope_id: str
    ledger_revision: int
    fence: int
    document_cid: str
    usage_snapshot: UsageSnapshot
    coordinator_partition_id: str = ""

    def __post_init__(self) -> None:
        require_catalog_id(self.scope_id, "scope_id")
        _generation(self.ledger_revision, "ledger_revision")
        _generation(self.fence, "fence")
        _cid(self.document_cid)
        if (not isinstance(self.coordinator_partition_id, str)
                or (self.coordinator_partition_id and not _PARTITION_ID.fullmatch(self.coordinator_partition_id))):
            raise ValueError("coordinator_partition_id must be empty or a stable partition identity")
        if not isinstance(self.usage_snapshot, UsageSnapshot):
            raise TypeError("usage_snapshot must be a UsageSnapshot")
        if self.usage_snapshot.scope_id != self.scope_id:
            raise ValueError("usage observation belongs to another scope")

    @property
    def schema(self) -> str:
        return self.SCHEMA

    @property
    def schema_version(self) -> int:
        return USAGE_ADMISSION_SNAPSHOT_VERSION

    @property
    def admission_revision(self) -> str:
        return _revision(self.scope_id, self.ledger_revision, self.fence,
                         self.document_cid, self.coordinator_partition_id)

    @property
    def snapshot(self) -> UsageSnapshot:
        return self.usage_snapshot

    @property
    def ledger_fence(self) -> int:
        return self.fence

    @property
    def ledger_state_id(self) -> str:
        return self.document_cid

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "scope_id": self.scope_id,
            "ledger_revision": self.ledger_revision,
            "fence": self.fence,
            "document_cid": self.document_cid,
            "coordinator_partition_id": self.coordinator_partition_id,
            "admission_revision": self.admission_revision,
            "usage_snapshot": self.usage_snapshot.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UsageAdmissionSnapshot":
        fields = {"schema", "schema_version", "scope_id", "ledger_revision", "fence",
                  "document_cid", "coordinator_partition_id", "admission_revision", "usage_snapshot"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("admission snapshot fields are missing or unknown")
        if value["schema"] != cls.SCHEMA or type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("unsupported admission snapshot schema")
        claimed = validate_admission_revision(value["admission_revision"])
        result = cls(
            scope_id=value["scope_id"], ledger_revision=value["ledger_revision"],
            fence=value["fence"], document_cid=value["document_cid"],
            usage_snapshot=UsageSnapshot.from_dict(value["usage_snapshot"]),
            coordinator_partition_id=value["coordinator_partition_id"],
        )
        if result.admission_revision != claimed:
            raise ValueError("admission_revision does not match its stored-state fields")
        return result


__all__ = [
    "USAGE_ADMISSION_SNAPSHOT_SCHEMA", "USAGE_ADMISSION_SNAPSHOT_VERSION",
    "UsageAdmissionSnapshot", "admission_revision_for_document",
    "coordinator_partition_identity", "validate_admission_revision",
]
