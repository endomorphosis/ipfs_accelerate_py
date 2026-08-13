"""Full checkpoint seal construction (IPS-038).

``create_full_checkpoint`` discovers and freshly verifies or proves every
required unit, builds category roots and the complete manifest, verifies the
repository proof root, and returns a :class:`FullCheckpointSeal`.

A full checkpoint is mandatory for genesis/first-state and every mandated
fallback context.  Cache presence alone is never a fast path: every required
unit must be freshly verified under the current policy.  Simulated, unknown,
unavailable, failed, or unverified required units prevent ``sealed_full``.

Interfaces: ``FullCheckpointSeal``, ``FullCheckpointBuilder``,
``create_full_checkpoint``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    ProofMode,
    ProofTerminalStatus,
    SealStatus,
    parse_proof_mode,
    parse_terminal_status,
)

EVIDENCE_SUBSET: Final[str] = "ips/full-seal@1"
SEAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "full-checkpoint-seal@1"
)

# Canonical genesis parent when no accepted parent seal exists.
GENESIS_PARENT_SEAL: Final[str] = "ips.forest.genesis@1"

# Closed ordered forest categories (plan §7 RepositoryProofForest).
FOREST_CATEGORIES: Final[tuple[str, ...]] = (
    "source_integrity",
    "static_analysis",
    "type_check",
    "unit_test",
    "integration_test",
    "property_test",
    "formal_obligation",
    "direct_zk",
    "receipt_aggregation",
    "release_invariant",
)

_KIND_TO_CATEGORY: Final[dict[str, str]] = {
    "static_analysis": "static_analysis",
    "type_check": "type_check",
    "unit_test": "unit_test",
    "integration_test": "integration_test",
    "property_test": "property_test",
    "formal_obligation": "formal_obligation",
    "direct_zk_computation": "direct_zk",
    "direct_zk": "direct_zk",
    "receipt_aggregation": "receipt_aggregation",
    "release_invariant": "release_invariant",
    "source_integrity": "source_integrity",
}

_PASSING_STATUSES: Final[frozenset[ProofTerminalStatus]] = frozenset(
    {
        ProofTerminalStatus.PROVED,
        ProofTerminalStatus.INTEGRITY_VERIFIED,
        ProofTerminalStatus.SIGNED_ASSERTION_VERIFIED,
    }
)

_FAILED_STATUSES: Final[frozenset[ProofTerminalStatus]] = frozenset(
    {
        ProofTerminalStatus.FAILED,
        ProofTerminalStatus.PROOF_FAILED,
        ProofTerminalStatus.INVALID,
        ProofTerminalStatus.STALE,
        ProofTerminalStatus.DISPROVED,
    }
)

_MANDATED_FALLBACK_REASONS: Final[frozenset[str]] = frozenset(
    {
        "first_state",
        "missing_parent",
        "trust_policy_change",
        "schema_change",
        "canonicalization_change",
        "environment_change",
        "circuit_or_key_change",
        "incomplete_cache_key",
        "full_fallback_required",
        "uncertain_cache_integrity",
        "release_qualification",
        "dependency_lock_change",
        "low_reuse_ratio",
        "excessive_delta_chain_depth",
    }
)


class FullCheckpointError(ValueError):
    """Fail-closed full-checkpoint contract violation."""


class FullCheckpointReason(str, Enum):
    """Stable reason codes for full-checkpoint construction outcomes."""

    SEALED = "sealed"
    SIMULATED_REQUIRED_UNIT = "simulated_required_unit"
    UNKNOWN_REQUIRED_UNIT = "unknown_required_unit"
    UNAVAILABLE_REQUIRED_UNIT = "unavailable_required_unit"
    TIMEOUT_REQUIRED_UNIT = "timeout_required_unit"
    CANCELLED_REQUIRED_UNIT = "cancelled_required_unit"
    FAILED_REQUIRED_UNIT = "failed_required_unit"
    UNVERIFIED_REQUIRED_UNIT = "unverified_required_unit"
    CACHE_REUSE_WITHOUT_VERIFICATION = "cache_reuse_without_verification"
    INCOMPLETE_MANIFEST = "incomplete_manifest"
    ROOT_VERIFICATION_FAILED = "root_verification_failed"
    EMPTY_REQUIRED_SET = "empty_required_set"


class CheckpointContext(str, Enum):
    """Why a full checkpoint is being constructed."""

    FIRST_STATE = "first_state"
    MANDATED_FALLBACK = "mandated_fallback"
    HISTORICAL_PARENT = "historical_parent"
    EXPLICIT = "explicit"


@dataclass(frozen=True, slots=True)
class RepositoryStateView:
    """Accelerate-facing repository binding for full-checkpoint construction."""

    repository_id: str
    revision: str
    source_root_cid: str
    repository_state_cid: str
    environment_cid: str
    parent_revision_ids: tuple[str, ...] = ()

    def to_canonical(self) -> dict[str, Any]:
        return {
            "repository_id": self.repository_id,
            "revision": self.revision,
            "source_root_cid": self.source_root_cid,
            "repository_state_cid": self.repository_state_cid,
            "environment_cid": self.environment_cid,
            "parent_revision_ids": list(self.parent_revision_ids),
        }


@dataclass(frozen=True, slots=True)
class VerificationPolicyView:
    """Policy and schema bindings committed into the full seal."""

    policy_cid: str
    proof_schema_version: str = "1"
    canonicalization_version: str = "1"
    dependency_graph_schema_version: str = "graph@1"
    circuit_id: str = "n/a"
    verification_key_id: str = "n/a"

    def to_canonical(self) -> dict[str, Any]:
        return {
            "policy_cid": self.policy_cid,
            "proof_schema_version": self.proof_schema_version,
            "canonicalization_version": self.canonicalization_version,
            "dependency_graph_schema_version": self.dependency_graph_schema_version,
            "circuit_id": self.circuit_id,
            "verification_key_id": self.verification_key_id,
        }


@dataclass(frozen=True, slots=True)
class RequiredUnitEvidence:
    """One required unit presented for full-checkpoint verification.

    ``freshly_verified`` must be true for production sealing.  A cache hit that
    skips re-verification is reported via
    ``cache_reused_without_fresh_verification`` and always blocks
    ``sealed_full``.
    """

    unit_id: str
    proof_object_cid: str
    category: str = "unit_test"
    terminal_status: str = ProofTerminalStatus.INTEGRITY_VERIFIED.value
    proof_mode: str = ProofMode.INTEGRITY_ONLY.value
    required_for_seal: bool = True
    freshly_verified: bool = True
    cache_reused_without_fresh_verification: bool = False
    circuit_id: str = "n/a"
    verification_key_id: str = "n/a"

    def to_canonical(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "proof_object_cid": self.proof_object_cid,
            "category": self.category,
            "terminal_status": self.terminal_status,
            "proof_mode": self.proof_mode,
            "required_for_seal": self.required_for_seal,
            "freshly_verified": self.freshly_verified,
            "cache_reused_without_fresh_verification": (
                self.cache_reused_without_fresh_verification
            ),
            "circuit_id": self.circuit_id,
            "verification_key_id": self.verification_key_id,
        }


@dataclass(frozen=True, slots=True)
class FullCheckpointSeal:
    """Immutable full-checkpoint seal bound to repository, policy, and roots."""

    schema: str
    evidence_subset: str
    seal_status: SealStatus
    reason: FullCheckpointReason
    context: CheckpointContext
    repository_id: str
    revision: str
    source_root_cid: str
    repository_state_cid: str
    environment_cid: str
    policy_cid: str
    proof_schema_version: str
    canonicalization_version: str
    dependency_graph_schema_version: str
    circuit_id: str
    verification_key_id: str
    parent_seal_cid: str
    parent_revision_ids: tuple[str, ...]
    required_unit_ids: tuple[str, ...]
    verified_unit_ids: tuple[str, ...]
    rejected_unit_ids: tuple[str, ...]
    manifest_root_cid: str
    category_roots: Mapping[str, str]
    repository_proof_root: str
    aggregation_root: str
    fallback_reasons: tuple[str, ...]
    every_unit_freshly_verified: bool
    cache_reuse_hidden: bool
    sealed: bool

    def __post_init__(self) -> None:
        if self.sealed and self.seal_status is not SealStatus.SEALED_FULL:
            raise FullCheckpointError(
                "sealed=True requires seal_status sealed_full"
            )
        if self.seal_status is SealStatus.SEALED_FULL and not self.sealed:
            raise FullCheckpointError(
                "seal_status sealed_full requires sealed=True"
            )
        if self.sealed and self.cache_reuse_hidden:
            raise FullCheckpointError(
                "sealed_full cannot hide cache reuse without fresh verification"
            )
        if self.sealed and not self.every_unit_freshly_verified:
            raise FullCheckpointError(
                "sealed_full requires every required unit to be freshly verified"
            )
        if self.evidence_subset != EVIDENCE_SUBSET:
            raise FullCheckpointError(
                f"evidence_subset must be {EVIDENCE_SUBSET}"
            )
        if self.schema != SEAL_SCHEMA:
            raise FullCheckpointError(f"schema must be {SEAL_SCHEMA}")

    @property
    def is_genesis(self) -> bool:
        return self.parent_seal_cid == GENESIS_PARENT_SEAL

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "seal_status": self.seal_status.value,
            "reason": self.reason.value,
            "context": self.context.value,
            "repository_id": self.repository_id,
            "revision": self.revision,
            "source_root_cid": self.source_root_cid,
            "repository_state_cid": self.repository_state_cid,
            "environment_cid": self.environment_cid,
            "policy_cid": self.policy_cid,
            "proof_schema_version": self.proof_schema_version,
            "canonicalization_version": self.canonicalization_version,
            "dependency_graph_schema_version": self.dependency_graph_schema_version,
            "circuit_id": self.circuit_id,
            "verification_key_id": self.verification_key_id,
            "parent_seal_cid": self.parent_seal_cid,
            "parent_revision_ids": list(self.parent_revision_ids),
            "required_unit_ids": list(self.required_unit_ids),
            "verified_unit_ids": list(self.verified_unit_ids),
            "rejected_unit_ids": list(self.rejected_unit_ids),
            "manifest_root_cid": self.manifest_root_cid,
            "category_roots": {
                cat: self.category_roots[cat] for cat in FOREST_CATEGORIES
            },
            "repository_proof_root": self.repository_proof_root,
            "aggregation_root": self.aggregation_root,
            "fallback_reasons": list(self.fallback_reasons),
            "every_unit_freshly_verified": self.every_unit_freshly_verified,
            "cache_reuse_hidden": self.cache_reuse_hidden,
            "sealed": self.sealed,
            "genesis_parent_seal": GENESIS_PARENT_SEAL,
            "is_genesis": self.is_genesis,
        }

    def seal_cid(self) -> str:
        return _cid(
            {
                "domain": "ips.full_checkpoint.seal.v1",
                "payload": self.to_canonical(),
            }
        )


def _cid(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _parse_category(value: str) -> str:
    text = value.strip()
    if text in FOREST_CATEGORIES:
        return text
    mapped = _KIND_TO_CATEGORY.get(text)
    if mapped is not None:
        return mapped
    raise FullCheckpointError(
        f"unknown forest category {value!r}; closed set is {list(FOREST_CATEGORIES)}"
    )


def _leaf_cid(unit: RequiredUnitEvidence, *, position: int) -> str:
    return _cid(
        {
            "domain": "ips.full_checkpoint.leaf.v1",
            "unit_id": unit.unit_id,
            "proof_object_cid": unit.proof_object_cid,
            "category": _parse_category(unit.category),
            "terminal_status": unit.terminal_status,
            "position": position,
        }
    )


def _category_root(
    category: str, units: Sequence[RequiredUnitEvidence]
) -> str:
    ordered = sorted(units, key=lambda item: item.unit_id.encode("utf-8"))
    leaf_cids = [
        _leaf_cid(unit, position=index) for index, unit in enumerate(ordered)
    ]
    if not leaf_cids:
        return _cid(
            {
                "domain": "ips.full_checkpoint.empty.v1",
                "category": category,
            }
        )
    level = list(leaf_cids)
    while len(level) > 1:
        nxt: list[str] = []
        index = 0
        while index < len(level):
            if index + 1 < len(level):
                nxt.append(
                    _cid(
                        {
                            "domain": "ips.full_checkpoint.binary.v1",
                            "left": level[index],
                            "right": level[index + 1],
                        }
                    )
                )
                index += 2
            else:
                nxt.append(
                    _cid(
                        {
                            "domain": "ips.full_checkpoint.unary.v1",
                            "child": level[index],
                        }
                    )
                )
                index += 1
        level = nxt
    return _cid(
        {
            "domain": "ips.full_checkpoint.category.v1",
            "category": category,
            "leaf_count": len(ordered),
            "leaf_ids": [unit.unit_id for unit in ordered],
            "merkle_root": level[0],
        }
    )


def _manifest_root(unit_ids: Sequence[str], *, policy_cid: str) -> str:
    return _cid(
        {
            "domain": "ips.full_checkpoint.manifest.v1",
            "required_unit_ids": list(unit_ids),
            "policy_cid": policy_cid,
        }
    )


def _repository_root(
    *,
    state: RepositoryStateView,
    policy: VerificationPolicyView,
    parent_seal_cid: str,
    manifest_root_cid: str,
    category_roots: Mapping[str, str],
) -> str:
    return _cid(
        {
            "domain": "ips.full_checkpoint.repository.v1",
            "repository_id": state.repository_id,
            "revision": state.revision,
            "source_root_cid": state.source_root_cid,
            "repository_state_cid": state.repository_state_cid,
            "manifest_root_cid": manifest_root_cid,
            "environment_cid": state.environment_cid,
            "policy_cid": policy.policy_cid,
            "proof_schema_version": policy.proof_schema_version,
            "canonicalization_version": policy.canonicalization_version,
            "dependency_graph_schema_version": policy.dependency_graph_schema_version,
            "circuit_id": policy.circuit_id,
            "verification_key_id": policy.verification_key_id,
            "parent_seal_cid": parent_seal_cid,
            "parent_revision_ids": list(state.parent_revision_ids),
            "category_roots": {
                cat: category_roots[cat] for cat in FOREST_CATEGORIES
            },
        }
    )


def _aggregation_root(unit_ids: Sequence[str], category_roots: Mapping[str, str]) -> str:
    return _cid(
        {
            "domain": "ips.full_checkpoint.aggregation.v1",
            "label": "manifest_aggregation",
            "required_unit_ids": list(unit_ids),
            "category_roots": {
                cat: category_roots[cat] for cat in FOREST_CATEGORIES
            },
        }
    )


def _coerce_state(
    repository_state: RepositoryStateView | Mapping[str, Any],
) -> RepositoryStateView:
    if isinstance(repository_state, RepositoryStateView):
        return repository_state
    if not isinstance(repository_state, Mapping):
        raise FullCheckpointError(
            "repository_state must be RepositoryStateView or mapping"
        )
    parents = repository_state.get("parent_revision_ids", ())
    if parents in (None, "n/a", ""):
        parent_ids: tuple[str, ...] = ()
    elif isinstance(parents, Sequence) and not isinstance(parents, (str, bytes)):
        parent_ids = tuple(str(item) for item in parents)
    else:
        raise FullCheckpointError("parent_revision_ids must be a sequence")
    repo_id = str(
        repository_state.get("repository_id")
        or repository_state.get("repository")
        or ""
    )
    revision = str(repository_state.get("revision") or "")
    source = str(
        repository_state.get("source_root_cid")
        or repository_state.get("tree_cid")
        or ""
    )
    state_cid = str(
        repository_state.get("repository_state_cid")
        or repository_state.get("identity_cid")
        or repository_state.get("cid")
        or ""
    )
    environment = str(repository_state.get("environment_cid") or "")
    if not repo_id or not revision or not source or not state_cid or not environment:
        raise FullCheckpointError(
            "repository_state requires repository_id, revision, "
            "source_root_cid, repository_state_cid, and environment_cid"
        )
    if list(parent_ids) != sorted(parent_ids):
        raise FullCheckpointError("parent_revision_ids must be canonically sorted")
    if len(set(parent_ids)) != len(parent_ids):
        raise FullCheckpointError("parent_revision_ids must be unique")
    return RepositoryStateView(
        repository_id=repo_id,
        revision=revision,
        source_root_cid=source,
        repository_state_cid=state_cid,
        environment_cid=environment,
        parent_revision_ids=parent_ids,
    )


def _coerce_policy(
    verification_policy: VerificationPolicyView | Mapping[str, Any] | None,
) -> VerificationPolicyView:
    if verification_policy is None:
        raise FullCheckpointError("verification_policy is required")
    if isinstance(verification_policy, VerificationPolicyView):
        return verification_policy
    if not isinstance(verification_policy, Mapping):
        raise FullCheckpointError(
            "verification_policy must be VerificationPolicyView or mapping"
        )
    policy_cid = str(
        verification_policy.get("policy_cid")
        or verification_policy.get("cid")
        or ""
    )
    if not policy_cid:
        raise FullCheckpointError("verification_policy requires policy_cid")
    return VerificationPolicyView(
        policy_cid=policy_cid,
        proof_schema_version=str(
            verification_policy.get("proof_schema_version") or "1"
        ),
        canonicalization_version=str(
            verification_policy.get("canonicalization_version") or "1"
        ),
        dependency_graph_schema_version=str(
            verification_policy.get("dependency_graph_schema_version") or "graph@1"
        ),
        circuit_id=str(verification_policy.get("circuit_id") or "n/a"),
        verification_key_id=str(
            verification_policy.get("verification_key_id") or "n/a"
        ),
    )


def _coerce_unit(raw: RequiredUnitEvidence | Mapping[str, Any]) -> RequiredUnitEvidence:
    if isinstance(raw, RequiredUnitEvidence):
        return raw
    if not isinstance(raw, Mapping):
        raise FullCheckpointError(
            "units entries must be RequiredUnitEvidence or mapping"
        )
    unit_id = str(raw.get("unit_id") or raw.get("proof_unit_id") or "")
    proof_object_cid = str(
        raw.get("proof_object_cid") or raw.get("proof_cid") or ""
    )
    if not unit_id or not proof_object_cid:
        raise FullCheckpointError(
            "unit requires unit_id and proof_object_cid"
        )
    return RequiredUnitEvidence(
        unit_id=unit_id,
        proof_object_cid=proof_object_cid,
        category=str(raw.get("category") or raw.get("proof_unit_kind") or "unit_test"),
        terminal_status=str(
            raw.get("terminal_status")
            or ProofTerminalStatus.INTEGRITY_VERIFIED.value
        ),
        proof_mode=str(raw.get("proof_mode") or ProofMode.INTEGRITY_ONLY.value),
        required_for_seal=bool(raw.get("required_for_seal", True)),
        freshly_verified=bool(raw.get("freshly_verified", True)),
        cache_reused_without_fresh_verification=bool(
            raw.get("cache_reused_without_fresh_verification", False)
        ),
        circuit_id=str(raw.get("circuit_id") or "n/a"),
        verification_key_id=str(raw.get("verification_key_id") or "n/a"),
    )


def _resolve_parent_seal_cid(parent_seal_cid: str | None) -> str:
    if parent_seal_cid is None:
        return GENESIS_PARENT_SEAL
    text = str(parent_seal_cid).strip()
    if not text or text in {"n/a", "none", "null", "genesis"}:
        return GENESIS_PARENT_SEAL
    return text


def _resolve_context(
    parent_seal_cid: str,
    fallback_reasons: Sequence[str],
) -> CheckpointContext:
    reasons = tuple(fallback_reasons)
    if parent_seal_cid == GENESIS_PARENT_SEAL or "first_state" in reasons:
        return CheckpointContext.FIRST_STATE
    if reasons and any(reason in _MANDATED_FALLBACK_REASONS for reason in reasons):
        return CheckpointContext.MANDATED_FALLBACK
    if parent_seal_cid != GENESIS_PARENT_SEAL:
        return CheckpointContext.HISTORICAL_PARENT
    return CheckpointContext.EXPLICIT


def _unit_failure(
    unit: RequiredUnitEvidence,
) -> tuple[SealStatus, FullCheckpointReason] | None:
    """Return a blocking seal outcome for one required unit, or None if passable."""

    if not unit.required_for_seal:
        return None

    if unit.cache_reused_without_fresh_verification:
        return (
            SealStatus.INVALID_CACHE,
            FullCheckpointReason.CACHE_REUSE_WITHOUT_VERIFICATION,
        )
    if not unit.freshly_verified:
        return (
            SealStatus.VERIFICATION_FAILED,
            FullCheckpointReason.UNVERIFIED_REQUIRED_UNIT,
        )

    try:
        mode = parse_proof_mode(unit.proof_mode)
    except Exception:
        return SealStatus.UNKNOWN, FullCheckpointReason.UNKNOWN_REQUIRED_UNIT
    try:
        status = parse_terminal_status(unit.terminal_status)
    except Exception:
        return SealStatus.UNKNOWN, FullCheckpointReason.UNKNOWN_REQUIRED_UNIT

    if mode is ProofMode.SIMULATED or status is ProofTerminalStatus.SIMULATED:
        return SealStatus.SIMULATED_ONLY, FullCheckpointReason.SIMULATED_REQUIRED_UNIT
    if status is ProofTerminalStatus.UNKNOWN:
        return SealStatus.UNKNOWN, FullCheckpointReason.UNKNOWN_REQUIRED_UNIT
    if status is ProofTerminalStatus.UNAVAILABLE:
        return SealStatus.UNAVAILABLE, FullCheckpointReason.UNAVAILABLE_REQUIRED_UNIT
    if status is ProofTerminalStatus.TIMEOUT:
        return SealStatus.TIMEOUT, FullCheckpointReason.TIMEOUT_REQUIRED_UNIT
    if status is ProofTerminalStatus.CANCELLED:
        return SealStatus.CANCELLED, FullCheckpointReason.CANCELLED_REQUIRED_UNIT
    if status in _FAILED_STATUSES:
        return SealStatus.PROOF_FAILED, FullCheckpointReason.FAILED_REQUIRED_UNIT
    if status is ProofTerminalStatus.NOT_MODELED:
        return SealStatus.UNKNOWN, FullCheckpointReason.UNKNOWN_REQUIRED_UNIT
    if status not in _PASSING_STATUSES:
        return (
            SealStatus.VERIFICATION_FAILED,
            FullCheckpointReason.UNVERIFIED_REQUIRED_UNIT,
        )
    return None


# Stable priority: first match among required units (sorted by unit_id) wins,
# but simulated / nonterminal / failed kinds are reported by their own codes.
_REASON_PRIORITY: Final[tuple[FullCheckpointReason, ...]] = (
    FullCheckpointReason.INCOMPLETE_MANIFEST,
    FullCheckpointReason.EMPTY_REQUIRED_SET,
    FullCheckpointReason.CACHE_REUSE_WITHOUT_VERIFICATION,
    FullCheckpointReason.SIMULATED_REQUIRED_UNIT,
    FullCheckpointReason.UNKNOWN_REQUIRED_UNIT,
    FullCheckpointReason.UNAVAILABLE_REQUIRED_UNIT,
    FullCheckpointReason.TIMEOUT_REQUIRED_UNIT,
    FullCheckpointReason.CANCELLED_REQUIRED_UNIT,
    FullCheckpointReason.FAILED_REQUIRED_UNIT,
    FullCheckpointReason.UNVERIFIED_REQUIRED_UNIT,
    FullCheckpointReason.ROOT_VERIFICATION_FAILED,
    FullCheckpointReason.SEALED,
)


def _pick_failure(
    failures: Sequence[tuple[SealStatus, FullCheckpointReason]],
) -> tuple[SealStatus, FullCheckpointReason] | None:
    if not failures:
        return None
    by_reason = {reason: status for status, reason in failures}
    for reason in _REASON_PRIORITY:
        if reason in by_reason:
            return by_reason[reason], reason
    return failures[0]


class FullCheckpointBuilder:
    """Construct a full checkpoint seal after verifying every required unit."""

    def create(
        self,
        repository_state: RepositoryStateView | Mapping[str, Any],
        verification_policy: VerificationPolicyView | Mapping[str, Any] | None,
        *,
        units: Sequence[RequiredUnitEvidence | Mapping[str, Any]] = (),
        expected_unit_ids: Sequence[str] | None = None,
        parent_seal_cid: str | None = None,
        fallback_reasons: Sequence[str] = (),
        expected_repository_proof_root: str | None = None,
    ) -> FullCheckpointSeal:
        state = _coerce_state(repository_state)
        policy = _coerce_policy(verification_policy)
        resolved_parent = _resolve_parent_seal_cid(parent_seal_cid)
        reasons = tuple(
            sorted({str(item) for item in fallback_reasons if str(item).strip()})
        )
        context = _resolve_context(resolved_parent, reasons)

        parsed = [_coerce_unit(item) for item in units]
        # Full checkpoints only consider required units for seal completeness.
        required = [item for item in parsed if item.required_for_seal]
        required.sort(key=lambda item: item.unit_id.encode("utf-8"))

        unit_ids = tuple(item.unit_id for item in required)
        if len(unit_ids) != len(set(unit_ids)):
            raise FullCheckpointError("duplicate required unit_id")

        failures: list[tuple[SealStatus, FullCheckpointReason]] = []
        verified: list[str] = []
        rejected: list[str] = []

        if expected_unit_ids is not None:
            expected = tuple(sorted(set(str(item) for item in expected_unit_ids)))
            if set(unit_ids) != set(expected):
                failures.append(
                    (
                        SealStatus.INCOMPLETE_MANIFEST,
                        FullCheckpointReason.INCOMPLETE_MANIFEST,
                    )
                )
                missing = [item for item in expected if item not in set(unit_ids)]
                rejected.extend(missing)
        if not unit_ids:
            # An empty required set cannot produce a production full seal.
            failures.append(
                (
                    SealStatus.INCOMPLETE_MANIFEST,
                    FullCheckpointReason.EMPTY_REQUIRED_SET,
                )
            )

        for unit in required:
            block = _unit_failure(unit)
            if block is not None:
                failures.append(block)
                rejected.append(unit.unit_id)
            else:
                verified.append(unit.unit_id)

        # Build roots even on failure so the seal records the attempted forest.
        by_category: dict[str, list[RequiredUnitEvidence]] = {
            cat: [] for cat in FOREST_CATEGORIES
        }
        for unit in required:
            if unit.unit_id in verified:
                cat = _parse_category(unit.category)
                by_category[cat].append(unit)

        category_roots = {
            cat: _category_root(cat, by_category[cat]) for cat in FOREST_CATEGORIES
        }
        manifest_root = _manifest_root(unit_ids, policy_cid=policy.policy_cid)
        repository_root = _repository_root(
            state=state,
            policy=policy,
            parent_seal_cid=resolved_parent,
            manifest_root_cid=manifest_root,
            category_roots=category_roots,
        )
        aggregation_root = _aggregation_root(unit_ids, category_roots)

        if expected_repository_proof_root is not None:
            if expected_repository_proof_root != repository_root:
                failures.append(
                    (
                        SealStatus.VERIFICATION_FAILED,
                        FullCheckpointReason.ROOT_VERIFICATION_FAILED,
                    )
                )

        every_fresh = all(
            unit.freshly_verified and not unit.cache_reused_without_fresh_verification
            for unit in required
        )
        cache_hidden = any(
            unit.cache_reused_without_fresh_verification for unit in required
        )

        picked = _pick_failure(failures)
        if picked is None and unit_ids and every_fresh and not cache_hidden:
            seal_status = SealStatus.SEALED_FULL
            reason = FullCheckpointReason.SEALED
            sealed = True
        elif picked is None:
            seal_status = SealStatus.VERIFICATION_FAILED
            reason = FullCheckpointReason.UNVERIFIED_REQUIRED_UNIT
            sealed = False
        else:
            seal_status, reason = picked
            sealed = False

        return FullCheckpointSeal(
            schema=SEAL_SCHEMA,
            evidence_subset=EVIDENCE_SUBSET,
            seal_status=seal_status,
            reason=reason,
            context=context,
            repository_id=state.repository_id,
            revision=state.revision,
            source_root_cid=state.source_root_cid,
            repository_state_cid=state.repository_state_cid,
            environment_cid=state.environment_cid,
            policy_cid=policy.policy_cid,
            proof_schema_version=policy.proof_schema_version,
            canonicalization_version=policy.canonicalization_version,
            dependency_graph_schema_version=policy.dependency_graph_schema_version,
            circuit_id=policy.circuit_id,
            verification_key_id=policy.verification_key_id,
            parent_seal_cid=resolved_parent,
            parent_revision_ids=state.parent_revision_ids,
            required_unit_ids=unit_ids,
            verified_unit_ids=tuple(sorted(set(verified))),
            rejected_unit_ids=tuple(sorted(set(rejected))),
            manifest_root_cid=manifest_root,
            category_roots=category_roots,
            repository_proof_root=repository_root,
            aggregation_root=aggregation_root,
            fallback_reasons=reasons,
            every_unit_freshly_verified=every_fresh,
            cache_reuse_hidden=cache_hidden,
            sealed=sealed,
        )


def create_full_checkpoint(
    repository_state: RepositoryStateView | Mapping[str, Any],
    verification_policy: VerificationPolicyView | Mapping[str, Any] | None,
    *,
    units: Sequence[RequiredUnitEvidence | Mapping[str, Any]] = (),
    expected_unit_ids: Sequence[str] | None = None,
    parent_seal_cid: str | None = None,
    fallback_reasons: Sequence[str] = (),
    expected_repository_proof_root: str | None = None,
) -> FullCheckpointSeal:
    """Public facade matching the plan document's ``create_full_checkpoint``."""

    return FullCheckpointBuilder().create(
        repository_state,
        verification_policy,
        units=units,
        expected_unit_ids=expected_unit_ids,
        parent_seal_cid=parent_seal_cid,
        fallback_reasons=fallback_reasons,
        expected_repository_proof_root=expected_repository_proof_root,
    )


__all__ = (
    "EVIDENCE_SUBSET",
    "FOREST_CATEGORIES",
    "GENESIS_PARENT_SEAL",
    "SEAL_SCHEMA",
    "CheckpointContext",
    "FullCheckpointBuilder",
    "FullCheckpointError",
    "FullCheckpointReason",
    "FullCheckpointSeal",
    "RepositoryStateView",
    "RequiredUnitEvidence",
    "VerificationPolicyView",
    "create_full_checkpoint",
)
