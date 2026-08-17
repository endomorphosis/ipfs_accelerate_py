"""Bounded manifest aggregation and capability-gated recursion (IPS-036).

Default path is Merkleized manifest aggregation labeled
``manifest_aggregation``.  It binds exact child identities, count, order,
duplicate rejection, root, terminal status, repository, and environment.  It
does **not** recursively verify child proofs or claim underlying test
execution.

Recursive aggregation is selectable only after a successful backend
capability probe.  Receipt aggregation records signer trust only.

Interfaces: ``ProofAggregator``, ``ManifestAggregationResult``,
``RecursiveAggregationResult``, ``aggregate_verified_units``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.backends import (
    ProofBackendCapability,
)

MANIFEST_EVIDENCE: Final[str] = "ips/manifest-aggregation@1"
RECURSIVE_EVIDENCE: Final[str] = "ips/recursive-aggregation@1"
MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "manifest-aggregation-result@1"
)
RECURSIVE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "recursive-aggregation-result@1"
)

AGGREGATION_LABEL_MANIFEST: Final[str] = "manifest_aggregation"
AGGREGATION_LABEL_RECURSIVE: Final[str] = "recursive_verification"

FAN_IN_LEVELS: Final[tuple[str, ...]] = (
    "leaf",
    "batch",
    "category",
    "repository",
)
DEFAULT_FAN_IN: Final[int] = 8

_FORBIDDEN_EXECUTION_CLAIMS: Final[frozenset[str]] = frozenset(
    {
        "tests executed",
        "tests ran",
        "test execution",
        "pytest executed",
        "underlying tests ran",
        "direct execution",
        "children recursively verified",
    }
)


class AggregationError(ValueError):
    """Fail-closed aggregation contract violation."""


class AggregationMode(str, Enum):
    MANIFEST = "manifest_aggregation"
    RECURSIVE = "recursive_verification"


class AggregationReason(str, Enum):
    AGGREGATED = "aggregated"
    MISSING_CHILD = "missing_child"
    DUPLICATE_CHILD = "duplicate_child"
    REORDERED_CHILDREN = "reordered_children"
    FAILED_CHILD = "failed_child"
    CHANGED_MANIFEST = "changed_manifest"
    STALE_AGGREGATE = "stale_aggregate"
    RECURSION_NOT_ADMITTED = "recursion_not_admitted"
    EXECUTION_OVERCLAIM = "execution_overclaim"


@dataclass(frozen=True, slots=True)
class VerifiedUnit:
    """One already-verified leaf presented for aggregation."""

    unit_id: str
    proof_object_cid: str
    category: str = "unit_test"
    terminal_status: str = "integrity_verified"
    repository_state_cid: str = ""
    environment_cid: str = ""
    failed: bool = False


@dataclass(frozen=True, slots=True)
class ManifestAggregationResult:
    """Merkle integrity/completeness aggregation.  Not recursive verification."""

    schema: str
    evidence_subset: str
    label: str
    mode: AggregationMode
    recursively_verifies_children: bool
    claims_test_execution: bool
    child_unit_ids: tuple[str, ...]
    child_count: int
    child_root: str
    repository_state_cid: str
    environment_cid: str
    terminal_status: str
    affected_levels: tuple[str, ...]
    signer_trust: str
    reason: AggregationReason
    accepted: bool

    def __post_init__(self) -> None:
        if self.recursively_verifies_children:
            raise AggregationError(
                "manifest aggregation must not recursively verify children"
            )
        if self.claims_test_execution:
            raise AggregationError(
                "manifest aggregation must not claim underlying test execution"
            )
        if self.label != AGGREGATION_LABEL_MANIFEST:
            raise AggregationError("manifest label must be manifest_aggregation")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "label": self.label,
            "mode": self.mode.value,
            "recursively_verifies_children": False,
            "claims_test_execution": False,
            "child_unit_ids": list(self.child_unit_ids),
            "child_count": self.child_count,
            "child_root": self.child_root,
            "repository_state_cid": self.repository_state_cid,
            "environment_cid": self.environment_cid,
            "terminal_status": self.terminal_status,
            "affected_levels": list(self.affected_levels),
            "signer_trust": self.signer_trust,
            "reason": self.reason.value,
            "accepted": self.accepted,
        }


@dataclass(frozen=True, slots=True)
class RecursiveAggregationResult:
    """Recursive child-proof aggregation admitted only by a live capability probe."""

    schema: str
    evidence_subset: str
    label: str
    mode: AggregationMode
    recursively_verifies_children: bool
    backend_id: str
    child_unit_ids: tuple[str, ...]
    child_count: int
    child_root: str
    accepted: bool
    reason: AggregationReason

    def __post_init__(self) -> None:
        if self.accepted and not self.recursively_verifies_children:
            raise AggregationError(
                "accepted recursive aggregation must verify children"
            )
        if self.label != AGGREGATION_LABEL_RECURSIVE and self.accepted:
            raise AggregationError("accepted recursive label mismatch")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "label": self.label,
            "mode": self.mode.value,
            "recursively_verifies_children": self.recursively_verifies_children,
            "backend_id": self.backend_id,
            "child_unit_ids": list(self.child_unit_ids),
            "child_count": self.child_count,
            "child_root": self.child_root,
            "accepted": self.accepted,
            "reason": self.reason.value,
        }


def _cid(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _leaf_cid(unit: VerifiedUnit) -> str:
    return _cid(
        {
            "domain": "ips.aggregation.leaf.v1",
            "unit_id": unit.unit_id,
            "proof_object_cid": unit.proof_object_cid,
            "category": unit.category,
            "terminal_status": unit.terminal_status,
        }
    )


def _fold_cids(cids: Sequence[str], *, fan_in: int) -> str:
    if not cids:
        return _cid({"domain": "ips.aggregation.empty.v1"})
    level = list(cids)
    while len(level) > 1:
        nxt: list[str] = []
        index = 0
        while index < len(level):
            batch = level[index : index + fan_in]
            nxt.append(
                _cid(
                    {
                        "domain": "ips.aggregation.batch.v1",
                        "children": batch,
                    }
                )
            )
            index += fan_in
        level = nxt
    return level[0]


class ProofAggregator:
    """Build bounded fan-in aggregates over already-verified units."""

    def __init__(self, *, fan_in: int = DEFAULT_FAN_IN) -> None:
        if fan_in < 2:
            raise AggregationError("fan_in must be >= 2")
        self.fan_in = fan_in

    def aggregate_verified_units(
        self,
        units: Sequence[VerifiedUnit],
        *,
        expected_unit_ids: Sequence[str] | None = None,
        previous_root: str | None = None,
        expected_root: str | None = None,
        capability: ProofBackendCapability | None = None,
        receipt_claim: str = "",
        prefer_recursion: bool = False,
    ) -> ManifestAggregationResult | RecursiveAggregationResult:
        reason = _reject_children(units, expected_unit_ids)
        ordered_ids = tuple(item.unit_id for item in units)
        root = _fold_cids(tuple(_leaf_cid(item) for item in units), fan_in=self.fan_in)
        if expected_root is not None and expected_root != root:
            reason = reason or AggregationReason.CHANGED_MANIFEST
        if previous_root is not None and previous_root == root and reason is None:
            # Recomputing the same root is idempotent; a different previous
            # root with an expected_root mismatch is already caught.
            pass
        if previous_root is not None and expected_root is not None:
            if previous_root != expected_root and expected_root != root:
                reason = AggregationReason.STALE_AGGREGATE
        if _claims_execution(receipt_claim):
            reason = AggregationReason.EXECUTION_OVERCLAIM

        repo = units[0].repository_state_cid if units else ""
        env = units[0].environment_cid if units else ""
        status = "aggregated" if reason is None else reason.value

        if prefer_recursion:
            return self._recursive(
                units,
                ordered_ids,
                root,
                reason,
                capability,
            )

        accepted = reason is None
        return ManifestAggregationResult(
            schema=MANIFEST_SCHEMA,
            evidence_subset=MANIFEST_EVIDENCE,
            label=AGGREGATION_LABEL_MANIFEST,
            mode=AggregationMode.MANIFEST,
            recursively_verifies_children=False,
            claims_test_execution=False,
            child_unit_ids=ordered_ids,
            child_count=len(ordered_ids),
            child_root=root,
            repository_state_cid=repo,
            environment_cid=env,
            terminal_status=status,
            affected_levels=FAN_IN_LEVELS if accepted else (),
            signer_trust="signer_allowlist_only",
            reason=reason or AggregationReason.AGGREGATED,
            accepted=accepted,
        )

    def _recursive(
        self,
        units: Sequence[VerifiedUnit],
        ordered_ids: tuple[str, ...],
        root: str,
        reason: AggregationReason | None,
        capability: ProofBackendCapability | None,
    ) -> RecursiveAggregationResult:
        admitted = bool(
            capability is not None and capability.recursive_verification is True
        )
        if not admitted:
            return RecursiveAggregationResult(
                schema=RECURSIVE_SCHEMA,
                evidence_subset=RECURSIVE_EVIDENCE,
                label=AGGREGATION_LABEL_MANIFEST,
                mode=AggregationMode.MANIFEST,
                recursively_verifies_children=False,
                backend_id=getattr(capability, "backend_id", "unavailable"),
                child_unit_ids=ordered_ids,
                child_count=len(ordered_ids),
                child_root=root,
                accepted=False,
                reason=AggregationReason.RECURSION_NOT_ADMITTED,
            )
        if reason is not None:
            return RecursiveAggregationResult(
                schema=RECURSIVE_SCHEMA,
                evidence_subset=RECURSIVE_EVIDENCE,
                label=AGGREGATION_LABEL_RECURSIVE,
                mode=AggregationMode.RECURSIVE,
                recursively_verifies_children=True,
                backend_id=capability.backend_id,
                child_unit_ids=ordered_ids,
                child_count=len(ordered_ids),
                child_root=root,
                accepted=False,
                reason=reason,
            )
        return RecursiveAggregationResult(
            schema=RECURSIVE_SCHEMA,
            evidence_subset=RECURSIVE_EVIDENCE,
            label=AGGREGATION_LABEL_RECURSIVE,
            mode=AggregationMode.RECURSIVE,
            recursively_verifies_children=True,
            backend_id=capability.backend_id,
            child_unit_ids=ordered_ids,
            child_count=len(ordered_ids),
            child_root=root,
            accepted=True,
            reason=AggregationReason.AGGREGATED,
        )


def aggregate_verified_units(
    units: Sequence[VerifiedUnit],
    *,
    expected_unit_ids: Sequence[str] | None = None,
    previous_root: str | None = None,
    expected_root: str | None = None,
    capability: ProofBackendCapability | None = None,
    receipt_claim: str = "",
    prefer_recursion: bool = False,
    fan_in: int = DEFAULT_FAN_IN,
) -> ManifestAggregationResult | RecursiveAggregationResult:
    """Public facade for bounded aggregation of verified units."""

    return ProofAggregator(fan_in=fan_in).aggregate_verified_units(
        units,
        expected_unit_ids=expected_unit_ids,
        previous_root=previous_root,
        expected_root=expected_root,
        capability=capability,
        receipt_claim=receipt_claim,
        prefer_recursion=prefer_recursion,
    )


def _reject_children(
    units: Sequence[VerifiedUnit],
    expected_unit_ids: Sequence[str] | None,
) -> AggregationReason | None:
    ids = [item.unit_id for item in units]
    if any(item.failed for item in units):
        return AggregationReason.FAILED_CHILD
    if len(ids) != len(set(ids)):
        return AggregationReason.DUPLICATE_CHILD
    if expected_unit_ids is not None:
        expected = list(expected_unit_ids)
        if set(ids) != set(expected):
            missing = [item for item in expected if item not in ids]
            if missing:
                return AggregationReason.MISSING_CHILD
            return AggregationReason.REORDERED_CHILDREN
        if ids != expected:
            return AggregationReason.REORDERED_CHILDREN
    return None


def _claims_execution(claim: str) -> bool:
    lowered = claim.lower()
    return any(token in lowered for token in _FORBIDDEN_EXECUTION_CLAIMS)


__all__ = (
    "AGGREGATION_LABEL_MANIFEST",
    "AGGREGATION_LABEL_RECURSIVE",
    "DEFAULT_FAN_IN",
    "FAN_IN_LEVELS",
    "MANIFEST_EVIDENCE",
    "RECURSIVE_EVIDENCE",
    "AggregationError",
    "AggregationMode",
    "AggregationReason",
    "ManifestAggregationResult",
    "ProofAggregator",
    "RecursiveAggregationResult",
    "VerifiedUnit",
    "aggregate_verified_units",
)
