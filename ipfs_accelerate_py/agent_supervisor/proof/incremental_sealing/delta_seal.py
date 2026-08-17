"""Parent-bound delta seals with all fourteen transition invariants (IPS-039).

``build_delta_seal`` verifies a parent-bound incremental transition: one exact
accepted parent seal, complete repository diff, invalidated/added/reused/
removed unit sets, removal authorizations, complete new manifest, rebuilt
forest/aggregate roots, and anti-replay branch/parent/revision binding.

All fourteen normative invariants from plan §8.2 are evaluated independently.
Any violation rejects ``sealed_incremental`` under production policy.

Interfaces: ``DeltaSeal``, ``DeltaTransitionStatement``, ``DeltaSealBuilder``,
``build_delta_seal``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
    FOREST_CATEGORIES,
    RepositoryStateView,
    VerificationPolicyView,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    ProofMode,
    ProofTerminalStatus,
    SealStatus,
    parse_proof_mode,
    parse_terminal_status,
)

EVIDENCE_SUBSET: Final[str] = "ips/delta-seal@1"
FOURTEEN_INVARIANTS_EVIDENCE: Final[str] = "ips/delta-fourteen-invariants@1"
SEAL_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "delta-seal@1"
)
TRANSITION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "delta-transition-statement@1"
)

# Closed ordered normative invariants (plan §8.2).
NORMATIVE_INVARIANTS: Final[tuple[str, ...]] = (
    "parent_accepted",
    "old_root_matches_parent",
    "new_root_matches_source",
    "complete_diff",
    "invalidated_have_new_proofs",
    "reuse_complete_cache_key",
    "deletions_authorized",
    "additions_present_and_proven",
    "manifest_complete",
    "forest_commits_exact_units",
    "no_stale_reuse",
    "no_blocking_units",
    "exact_parent_bound",
    "anti_replay_binding",
)

assert len(NORMATIVE_INVARIANTS) == 14

UNIT_DISPOSITIONS: Final[tuple[str, ...]] = (
    "reuse",
    "replace",
    "add",
    "remove",
)

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

_ACCEPTED_PARENT_STATUSES: Final[frozenset[str]] = frozenset(
    {
        SealStatus.SEALED_FULL.value,
        SealStatus.SEALED_INCREMENTAL.value,
    }
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


class DeltaSealError(ValueError):
    """Fail-closed delta-seal contract violation."""


class DeltaSealReason(str, Enum):
    """Stable reason codes for delta-seal construction outcomes."""

    SEALED = "sealed"
    PARENT_NOT_ACCEPTED = "parent_not_accepted"
    OLD_ROOT_MISMATCH = "old_root_mismatch"
    NEW_ROOT_MISMATCH = "new_root_mismatch"
    INCOMPLETE_DIFF = "incomplete_diff"
    MISSING_REPLACEMENT = "missing_replacement"
    STALE_REUSE = "stale_reuse"
    INCOMPLETE_CACHE_KEY = "incomplete_cache_key"
    UNAUTHORIZED_DELETION = "unauthorized_deletion"
    MISSING_ADDITION = "missing_addition"
    INCOMPLETE_MANIFEST = "incomplete_manifest"
    FOREST_MISMATCH = "forest_mismatch"
    OLD_AGGREGATE = "old_aggregate"
    LOST_LEAF = "lost_leaf"
    SIMULATED_REQUIRED_UNIT = "simulated_required_unit"
    UNKNOWN_REQUIRED_UNIT = "unknown_required_unit"
    UNAVAILABLE_REQUIRED_UNIT = "unavailable_required_unit"
    TIMEOUT_REQUIRED_UNIT = "timeout_required_unit"
    CANCELLED_REQUIRED_UNIT = "cancelled_required_unit"
    FAILED_REQUIRED_UNIT = "failed_required_unit"
    UNVERIFIED_REQUIRED_UNIT = "unverified_required_unit"
    WRONG_PARENT = "wrong_parent"
    WRONG_BRANCH = "wrong_branch"
    REPLAY_REJECTED = "replay_rejected"
    EMPTY_REQUIRED_SET = "empty_required_set"
    INVALID_DISPOSITION = "invalid_disposition"


class UnitDisposition(str, Enum):
    REUSE = "reuse"
    REPLACE = "replace"
    ADD = "add"
    REMOVE = "remove"


@dataclass(frozen=True, slots=True)
class ParentSealView:
    """Accepted parent seal bindings consumed by a delta transition."""

    seal_cid: str
    accepted: bool
    seal_status: str
    repository_id: str
    branch_id: str
    revision: str
    source_root_cid: str
    repository_state_cid: str
    environment_cid: str
    policy_cid: str
    manifest_root_cid: str
    forest_root_cid: str
    aggregation_root: str
    required_unit_ids: tuple[str, ...]
    unit_proof_cids: Mapping[str, str]
    parent_revision_ids: tuple[str, ...] = ()
    proof_schema_version: str = "1"
    canonicalization_version: str = "1"
    dependency_graph_schema_version: str = "graph@1"
    circuit_id: str = "n/a"
    verification_key_id: str = "n/a"
    logical_epoch: int = 0

    def to_canonical(self) -> dict[str, Any]:
        return {
            "seal_cid": self.seal_cid,
            "accepted": self.accepted,
            "seal_status": self.seal_status,
            "repository_id": self.repository_id,
            "branch_id": self.branch_id,
            "revision": self.revision,
            "source_root_cid": self.source_root_cid,
            "repository_state_cid": self.repository_state_cid,
            "environment_cid": self.environment_cid,
            "policy_cid": self.policy_cid,
            "manifest_root_cid": self.manifest_root_cid,
            "forest_root_cid": self.forest_root_cid,
            "aggregation_root": self.aggregation_root,
            "required_unit_ids": list(self.required_unit_ids),
            "unit_proof_cids": {
                key: self.unit_proof_cids[key]
                for key in sorted(self.unit_proof_cids)
            },
            "parent_revision_ids": list(self.parent_revision_ids),
            "proof_schema_version": self.proof_schema_version,
            "canonicalization_version": self.canonicalization_version,
            "dependency_graph_schema_version": self.dependency_graph_schema_version,
            "circuit_id": self.circuit_id,
            "verification_key_id": self.verification_key_id,
            "logical_epoch": self.logical_epoch,
        }


@dataclass(frozen=True, slots=True)
class DiffCommitmentView:
    """Bound complete repository diff for the transition."""

    diff_algorithm: str
    changed_artifact_commitment: str
    complete: bool
    changed_paths: tuple[str, ...] = ()

    def to_canonical(self) -> dict[str, Any]:
        return {
            "diff_algorithm": self.diff_algorithm,
            "changed_artifact_commitment": self.changed_artifact_commitment,
            "complete": self.complete,
            "changed_paths": list(self.changed_paths),
        }


@dataclass(frozen=True, slots=True)
class DeltaUnitEvidence:
    """One unit presented for a parent-bound delta transition."""

    unit_id: str
    disposition: str
    proof_object_cid: str = ""
    category: str = "unit_test"
    terminal_status: str = ProofTerminalStatus.INTEGRITY_VERIFIED.value
    proof_mode: str = ProofMode.INTEGRITY_ONLY.value
    required_for_seal: bool = True
    cache_key_complete: bool = True
    cache_key_unchanged: bool = True
    freshly_verified: bool = True
    newly_admitted: bool = False
    removal_authorized: bool = False
    parent_proof_object_cid: str = ""
    stale: bool = False

    def to_canonical(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "disposition": self.disposition,
            "proof_object_cid": self.proof_object_cid,
            "category": self.category,
            "terminal_status": self.terminal_status,
            "proof_mode": self.proof_mode,
            "required_for_seal": self.required_for_seal,
            "cache_key_complete": self.cache_key_complete,
            "cache_key_unchanged": self.cache_key_unchanged,
            "freshly_verified": self.freshly_verified,
            "newly_admitted": self.newly_admitted,
            "removal_authorized": self.removal_authorized,
            "parent_proof_object_cid": self.parent_proof_object_cid,
            "stale": self.stale,
        }


@dataclass(frozen=True, slots=True)
class DeltaTransitionStatement:
    """Explicit state transition bound into a delta seal."""

    schema: str
    parent_seal_cid: str
    branch_id: str
    old_source_root_cid: str
    old_repository_state_cid: str
    old_manifest_root_cid: str
    old_forest_root_cid: str
    old_aggregation_root: str
    new_source_root_cid: str
    new_repository_state_cid: str
    new_revision: str
    parent_revision_ids: tuple[str, ...]
    diff: DiffCommitmentView
    expected_manifest_unit_ids: tuple[str, ...]
    expected_surviving_leaf_ids: tuple[str, ...]
    forest_rebuilt: bool
    aggregation_rebuilt: bool
    logical_epoch: int = 1
    transition_id: str = ""

    def __post_init__(self) -> None:
        if self.schema != TRANSITION_SCHEMA:
            raise DeltaSealError(f"transition schema must be {TRANSITION_SCHEMA}")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "parent_seal_cid": self.parent_seal_cid,
            "branch_id": self.branch_id,
            "old_source_root_cid": self.old_source_root_cid,
            "old_repository_state_cid": self.old_repository_state_cid,
            "old_manifest_root_cid": self.old_manifest_root_cid,
            "old_forest_root_cid": self.old_forest_root_cid,
            "old_aggregation_root": self.old_aggregation_root,
            "new_source_root_cid": self.new_source_root_cid,
            "new_repository_state_cid": self.new_repository_state_cid,
            "new_revision": self.new_revision,
            "parent_revision_ids": list(self.parent_revision_ids),
            "diff": self.diff.to_canonical(),
            "expected_manifest_unit_ids": list(self.expected_manifest_unit_ids),
            "expected_surviving_leaf_ids": list(self.expected_surviving_leaf_ids),
            "forest_rebuilt": self.forest_rebuilt,
            "aggregation_rebuilt": self.aggregation_rebuilt,
            "logical_epoch": self.logical_epoch,
            "transition_id": self.transition_id,
        }


@dataclass(frozen=True, slots=True)
class DeltaSeal:
    """Immutable parent-bound incremental seal."""

    schema: str
    evidence_subset: str
    fourteen_invariants_evidence: str
    seal_status: SealStatus
    reason: DeltaSealReason
    repository_id: str
    branch_id: str
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
    logical_epoch: int
    transition_id: str
    diff_algorithm: str
    changed_artifact_commitment: str
    old_source_root_cid: str
    old_repository_state_cid: str
    old_manifest_root_cid: str
    old_forest_root_cid: str
    old_aggregation_root: str
    new_manifest_root_cid: str
    new_forest_root_cid: str
    new_aggregation_root: str
    category_roots: Mapping[str, str]
    reused_unit_ids: tuple[str, ...]
    replaced_unit_ids: tuple[str, ...]
    added_unit_ids: tuple[str, ...]
    removed_unit_ids: tuple[str, ...]
    required_unit_ids: tuple[str, ...]
    verified_unit_ids: tuple[str, ...]
    rejected_unit_ids: tuple[str, ...]
    invariants_passed: tuple[str, ...]
    invariants_failed: tuple[str, ...]
    sealed: bool

    def __post_init__(self) -> None:
        if self.sealed and self.seal_status is not SealStatus.SEALED_INCREMENTAL:
            raise DeltaSealError(
                "sealed=True requires seal_status sealed_incremental"
            )
        if self.seal_status is SealStatus.SEALED_INCREMENTAL and not self.sealed:
            raise DeltaSealError(
                "seal_status sealed_incremental requires sealed=True"
            )
        if self.sealed and self.invariants_failed:
            raise DeltaSealError(
                "sealed_incremental requires all fourteen invariants to pass"
            )
        if self.sealed and set(self.invariants_passed) != set(NORMATIVE_INVARIANTS):
            raise DeltaSealError(
                "sealed_incremental requires all fourteen normative invariants"
            )
        if self.evidence_subset != EVIDENCE_SUBSET:
            raise DeltaSealError(f"evidence_subset must be {EVIDENCE_SUBSET}")
        if self.fourteen_invariants_evidence != FOURTEEN_INVARIANTS_EVIDENCE:
            raise DeltaSealError(
                f"fourteen_invariants_evidence must be {FOURTEEN_INVARIANTS_EVIDENCE}"
            )
        if self.schema != SEAL_SCHEMA:
            raise DeltaSealError(f"schema must be {SEAL_SCHEMA}")

    def to_canonical(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "fourteen_invariants_evidence": self.fourteen_invariants_evidence,
            "seal_status": self.seal_status.value,
            "reason": self.reason.value,
            "repository_id": self.repository_id,
            "branch_id": self.branch_id,
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
            "logical_epoch": self.logical_epoch,
            "transition_id": self.transition_id,
            "diff_algorithm": self.diff_algorithm,
            "changed_artifact_commitment": self.changed_artifact_commitment,
            "old_source_root_cid": self.old_source_root_cid,
            "old_repository_state_cid": self.old_repository_state_cid,
            "old_manifest_root_cid": self.old_manifest_root_cid,
            "old_forest_root_cid": self.old_forest_root_cid,
            "old_aggregation_root": self.old_aggregation_root,
            "new_manifest_root_cid": self.new_manifest_root_cid,
            "new_forest_root_cid": self.new_forest_root_cid,
            "new_aggregation_root": self.new_aggregation_root,
            "category_roots": {
                cat: self.category_roots[cat] for cat in FOREST_CATEGORIES
            },
            "reused_unit_ids": list(self.reused_unit_ids),
            "replaced_unit_ids": list(self.replaced_unit_ids),
            "added_unit_ids": list(self.added_unit_ids),
            "removed_unit_ids": list(self.removed_unit_ids),
            "required_unit_ids": list(self.required_unit_ids),
            "verified_unit_ids": list(self.verified_unit_ids),
            "rejected_unit_ids": list(self.rejected_unit_ids),
            "invariants_passed": list(self.invariants_passed),
            "invariants_failed": list(self.invariants_failed),
            "normative_invariants": list(NORMATIVE_INVARIANTS),
            "sealed": self.sealed,
        }

    def seal_cid(self) -> str:
        return _cid(
            {
                "domain": "ips.delta_seal.seal.v1",
                "payload": self.to_canonical(),
            }
        )

    def all_invariants_passed(self) -> bool:
        return (
            not self.invariants_failed
            and set(self.invariants_passed) == set(NORMATIVE_INVARIANTS)
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
    raise DeltaSealError(
        f"unknown forest category {value!r}; closed set is {list(FOREST_CATEGORIES)}"
    )


def _parse_disposition(value: str) -> UnitDisposition:
    text = str(value).strip()
    try:
        return UnitDisposition(text)
    except ValueError as exc:
        raise DeltaSealError(
            f"unknown unit disposition {value!r}; closed set is {list(UNIT_DISPOSITIONS)}"
        ) from exc


def _leaf_cid(unit: DeltaUnitEvidence, *, position: int) -> str:
    return _cid(
        {
            "domain": "ips.delta_seal.leaf.v1",
            "unit_id": unit.unit_id,
            "proof_object_cid": unit.proof_object_cid,
            "category": _parse_category(unit.category),
            "terminal_status": unit.terminal_status,
            "position": position,
        }
    )


def _category_root(category: str, units: Sequence[DeltaUnitEvidence]) -> str:
    ordered = sorted(units, key=lambda item: item.unit_id.encode("utf-8"))
    leaf_cids = [
        _leaf_cid(unit, position=index) for index, unit in enumerate(ordered)
    ]
    if not leaf_cids:
        return _cid(
            {
                "domain": "ips.delta_seal.empty.v1",
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
                            "domain": "ips.delta_seal.binary.v1",
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
                            "domain": "ips.delta_seal.unary.v1",
                            "child": level[index],
                        }
                    )
                )
                index += 1
        level = nxt
    return _cid(
        {
            "domain": "ips.delta_seal.category.v1",
            "category": category,
            "leaf_count": len(ordered),
            "leaf_ids": [unit.unit_id for unit in ordered],
            "merkle_root": level[0],
        }
    )


def _manifest_root(unit_ids: Sequence[str], *, policy_cid: str) -> str:
    return _cid(
        {
            "domain": "ips.delta_seal.manifest.v1",
            "required_unit_ids": list(unit_ids),
            "policy_cid": policy_cid,
        }
    )


def _forest_root(
    *,
    state: RepositoryStateView,
    policy: VerificationPolicyView,
    parent_seal_cid: str,
    branch_id: str,
    logical_epoch: int,
    manifest_root_cid: str,
    category_roots: Mapping[str, str],
) -> str:
    return _cid(
        {
            "domain": "ips.delta_seal.forest.v1",
            "repository_id": state.repository_id,
            "branch_id": branch_id,
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
            "logical_epoch": logical_epoch,
            "category_roots": {
                cat: category_roots[cat] for cat in FOREST_CATEGORIES
            },
        }
    )


def _aggregation_root(
    unit_ids: Sequence[str], category_roots: Mapping[str, str]
) -> str:
    return _cid(
        {
            "domain": "ips.delta_seal.aggregation.v1",
            "label": "manifest_aggregation",
            "required_unit_ids": list(unit_ids),
            "category_roots": {
                cat: category_roots[cat] for cat in FOREST_CATEGORIES
            },
        }
    )


def _transition_id(
    *,
    parent_seal_cid: str,
    branch_id: str,
    revision: str,
    new_source_root_cid: str,
    new_repository_state_cid: str,
    logical_epoch: int,
) -> str:
    return _cid(
        {
            "domain": "ips.delta_seal.transition.v1",
            "parent_seal_cid": parent_seal_cid,
            "branch_id": branch_id,
            "revision": revision,
            "new_source_root_cid": new_source_root_cid,
            "new_repository_state_cid": new_repository_state_cid,
            "logical_epoch": logical_epoch,
        }
    )


def _coerce_state(
    repository_state: RepositoryStateView | Mapping[str, Any],
) -> RepositoryStateView:
    if isinstance(repository_state, RepositoryStateView):
        return repository_state
    if not isinstance(repository_state, Mapping):
        raise DeltaSealError(
            "repository_state must be RepositoryStateView or mapping"
        )
    parents = repository_state.get("parent_revision_ids", ())
    if parents in (None, "n/a", ""):
        parent_ids: tuple[str, ...] = ()
    elif isinstance(parents, Sequence) and not isinstance(parents, (str, bytes)):
        parent_ids = tuple(str(item) for item in parents)
    else:
        raise DeltaSealError("parent_revision_ids must be a sequence")
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
        raise DeltaSealError(
            "repository_state requires repository_id, revision, "
            "source_root_cid, repository_state_cid, and environment_cid"
        )
    if list(parent_ids) != sorted(parent_ids):
        raise DeltaSealError("parent_revision_ids must be canonically sorted")
    if len(set(parent_ids)) != len(parent_ids):
        raise DeltaSealError("parent_revision_ids must be unique")
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
        raise DeltaSealError("verification_policy is required")
    if isinstance(verification_policy, VerificationPolicyView):
        return verification_policy
    if not isinstance(verification_policy, Mapping):
        raise DeltaSealError(
            "verification_policy must be VerificationPolicyView or mapping"
        )
    policy_cid = str(
        verification_policy.get("policy_cid")
        or verification_policy.get("cid")
        or ""
    )
    if not policy_cid:
        raise DeltaSealError("verification_policy requires policy_cid")
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


def _coerce_diff(raw: DiffCommitmentView | Mapping[str, Any]) -> DiffCommitmentView:
    if isinstance(raw, DiffCommitmentView):
        return raw
    if not isinstance(raw, Mapping):
        raise DeltaSealError("diff must be DiffCommitmentView or mapping")
    paths = raw.get("changed_paths", ())
    if paths in (None, ""):
        path_tuple: tuple[str, ...] = ()
    elif isinstance(paths, Sequence) and not isinstance(paths, (str, bytes)):
        path_tuple = tuple(str(item) for item in paths)
    else:
        raise DeltaSealError("changed_paths must be a sequence")
    return DiffCommitmentView(
        diff_algorithm=str(raw.get("diff_algorithm") or ""),
        changed_artifact_commitment=str(
            raw.get("changed_artifact_commitment") or raw.get("commitment") or ""
        ),
        complete=bool(raw.get("complete", False)),
        changed_paths=path_tuple,
    )


def _coerce_parent(
    parent: ParentSealView | Mapping[str, Any],
) -> ParentSealView:
    if isinstance(parent, ParentSealView):
        return parent
    if not isinstance(parent, Mapping):
        raise DeltaSealError("parent must be ParentSealView or mapping")
    unit_ids_raw = parent.get("required_unit_ids", ())
    if not isinstance(unit_ids_raw, Sequence) or isinstance(unit_ids_raw, (str, bytes)):
        raise DeltaSealError("required_unit_ids must be a sequence")
    unit_ids = tuple(str(item) for item in unit_ids_raw)
    proofs_raw = parent.get("unit_proof_cids", {})
    if not isinstance(proofs_raw, Mapping):
        raise DeltaSealError("unit_proof_cids must be a mapping")
    proofs = {str(key): str(value) for key, value in proofs_raw.items()}
    parents = parent.get("parent_revision_ids", ())
    if parents in (None, "n/a", ""):
        parent_ids: tuple[str, ...] = ()
    elif isinstance(parents, Sequence) and not isinstance(parents, (str, bytes)):
        parent_ids = tuple(str(item) for item in parents)
    else:
        raise DeltaSealError("parent_revision_ids must be a sequence")
    seal_cid = str(parent.get("seal_cid") or parent.get("parent_seal_cid") or "")
    repo_id = str(parent.get("repository_id") or "")
    branch_id = str(parent.get("branch_id") or "main")
    revision = str(parent.get("revision") or "")
    source = str(parent.get("source_root_cid") or "")
    state_cid = str(parent.get("repository_state_cid") or "")
    environment = str(parent.get("environment_cid") or "")
    policy_cid = str(parent.get("policy_cid") or "")
    manifest = str(parent.get("manifest_root_cid") or "")
    forest = str(parent.get("forest_root_cid") or parent.get("repository_proof_root") or "")
    aggregation = str(parent.get("aggregation_root") or "")
    if not all(
        (
            seal_cid,
            repo_id,
            revision,
            source,
            state_cid,
            environment,
            policy_cid,
            manifest,
            forest,
            aggregation,
        )
    ):
        raise DeltaSealError(
            "parent requires seal_cid, repository_id, revision, source_root_cid, "
            "repository_state_cid, environment_cid, policy_cid, manifest_root_cid, "
            "forest_root_cid, and aggregation_root"
        )
    return ParentSealView(
        seal_cid=seal_cid,
        accepted=bool(parent.get("accepted", False)),
        seal_status=str(parent.get("seal_status") or ""),
        repository_id=repo_id,
        branch_id=branch_id,
        revision=revision,
        source_root_cid=source,
        repository_state_cid=state_cid,
        environment_cid=environment,
        policy_cid=policy_cid,
        manifest_root_cid=manifest,
        forest_root_cid=forest,
        aggregation_root=aggregation,
        required_unit_ids=unit_ids,
        unit_proof_cids=proofs,
        parent_revision_ids=parent_ids,
        proof_schema_version=str(parent.get("proof_schema_version") or "1"),
        canonicalization_version=str(parent.get("canonicalization_version") or "1"),
        dependency_graph_schema_version=str(
            parent.get("dependency_graph_schema_version") or "graph@1"
        ),
        circuit_id=str(parent.get("circuit_id") or "n/a"),
        verification_key_id=str(parent.get("verification_key_id") or "n/a"),
        logical_epoch=int(parent.get("logical_epoch") or 0),
    )


def _coerce_unit(raw: DeltaUnitEvidence | Mapping[str, Any]) -> DeltaUnitEvidence:
    if isinstance(raw, DeltaUnitEvidence):
        return raw
    if not isinstance(raw, Mapping):
        raise DeltaSealError("units entries must be DeltaUnitEvidence or mapping")
    unit_id = str(raw.get("unit_id") or raw.get("proof_unit_id") or "")
    if not unit_id:
        raise DeltaSealError("unit requires unit_id")
    disposition = str(raw.get("disposition") or "")
    if not disposition:
        raise DeltaSealError("unit requires disposition")
    return DeltaUnitEvidence(
        unit_id=unit_id,
        disposition=disposition,
        proof_object_cid=str(
            raw.get("proof_object_cid") or raw.get("proof_cid") or ""
        ),
        category=str(raw.get("category") or raw.get("proof_unit_kind") or "unit_test"),
        terminal_status=str(
            raw.get("terminal_status")
            or ProofTerminalStatus.INTEGRITY_VERIFIED.value
        ),
        proof_mode=str(raw.get("proof_mode") or ProofMode.INTEGRITY_ONLY.value),
        required_for_seal=bool(raw.get("required_for_seal", True)),
        cache_key_complete=bool(raw.get("cache_key_complete", True)),
        cache_key_unchanged=bool(raw.get("cache_key_unchanged", True)),
        freshly_verified=bool(raw.get("freshly_verified", True)),
        newly_admitted=bool(raw.get("newly_admitted", False)),
        removal_authorized=bool(raw.get("removal_authorized", False)),
        parent_proof_object_cid=str(raw.get("parent_proof_object_cid") or ""),
        stale=bool(raw.get("stale", False)),
    )


def _coerce_transition(
    transition: DeltaTransitionStatement | Mapping[str, Any],
) -> DeltaTransitionStatement:
    if isinstance(transition, DeltaTransitionStatement):
        return transition
    if not isinstance(transition, Mapping):
        raise DeltaSealError(
            "transition must be DeltaTransitionStatement or mapping"
        )
    diff_raw = transition.get("diff")
    if diff_raw is None:
        raise DeltaSealError("transition requires diff")
    diff = _coerce_diff(diff_raw)  # type: ignore[arg-type]

    def _seq(field: str) -> tuple[str, ...]:
        raw = transition.get(field, ())
        if raw in (None, ""):
            return ()
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise DeltaSealError(f"{field} must be a sequence")
        return tuple(str(item) for item in raw)

    schema = str(transition.get("schema") or TRANSITION_SCHEMA)
    return DeltaTransitionStatement(
        schema=schema,
        parent_seal_cid=str(transition.get("parent_seal_cid") or ""),
        branch_id=str(transition.get("branch_id") or "main"),
        old_source_root_cid=str(transition.get("old_source_root_cid") or ""),
        old_repository_state_cid=str(
            transition.get("old_repository_state_cid") or ""
        ),
        old_manifest_root_cid=str(transition.get("old_manifest_root_cid") or ""),
        old_forest_root_cid=str(transition.get("old_forest_root_cid") or ""),
        old_aggregation_root=str(transition.get("old_aggregation_root") or ""),
        new_source_root_cid=str(transition.get("new_source_root_cid") or ""),
        new_repository_state_cid=str(
            transition.get("new_repository_state_cid") or ""
        ),
        new_revision=str(transition.get("new_revision") or ""),
        parent_revision_ids=_seq("parent_revision_ids"),
        diff=diff,
        expected_manifest_unit_ids=_seq("expected_manifest_unit_ids"),
        expected_surviving_leaf_ids=_seq("expected_surviving_leaf_ids"),
        forest_rebuilt=bool(transition.get("forest_rebuilt", True)),
        aggregation_rebuilt=bool(transition.get("aggregation_rebuilt", True)),
        logical_epoch=int(transition.get("logical_epoch") or 1),
        transition_id=str(transition.get("transition_id") or ""),
    )


def _unit_blocking_status(
    unit: DeltaUnitEvidence,
) -> tuple[SealStatus, DeltaSealReason] | None:
    """Return a blocking seal outcome for one required non-removed unit."""

    if not unit.required_for_seal:
        return None
    disposition = _parse_disposition(unit.disposition)
    if disposition is UnitDisposition.REMOVE:
        return None

    try:
        mode = parse_proof_mode(unit.proof_mode)
    except Exception:
        return SealStatus.UNKNOWN, DeltaSealReason.UNKNOWN_REQUIRED_UNIT
    try:
        status = parse_terminal_status(unit.terminal_status)
    except Exception:
        return SealStatus.UNKNOWN, DeltaSealReason.UNKNOWN_REQUIRED_UNIT

    if mode is ProofMode.SIMULATED or status is ProofTerminalStatus.SIMULATED:
        return SealStatus.SIMULATED_ONLY, DeltaSealReason.SIMULATED_REQUIRED_UNIT
    if status is ProofTerminalStatus.UNKNOWN:
        return SealStatus.UNKNOWN, DeltaSealReason.UNKNOWN_REQUIRED_UNIT
    if status is ProofTerminalStatus.UNAVAILABLE:
        return SealStatus.UNAVAILABLE, DeltaSealReason.UNAVAILABLE_REQUIRED_UNIT
    if status is ProofTerminalStatus.TIMEOUT:
        return SealStatus.TIMEOUT, DeltaSealReason.TIMEOUT_REQUIRED_UNIT
    if status is ProofTerminalStatus.CANCELLED:
        return SealStatus.CANCELLED, DeltaSealReason.CANCELLED_REQUIRED_UNIT
    if status in _FAILED_STATUSES:
        return SealStatus.PROOF_FAILED, DeltaSealReason.FAILED_REQUIRED_UNIT
    if status is ProofTerminalStatus.NOT_MODELED:
        return SealStatus.UNKNOWN, DeltaSealReason.UNKNOWN_REQUIRED_UNIT
    if status not in _PASSING_STATUSES:
        return (
            SealStatus.VERIFICATION_FAILED,
            DeltaSealReason.UNVERIFIED_REQUIRED_UNIT,
        )
    if not unit.freshly_verified:
        return (
            SealStatus.VERIFICATION_FAILED,
            DeltaSealReason.UNVERIFIED_REQUIRED_UNIT,
        )
    return None


_REASON_PRIORITY: Final[tuple[DeltaSealReason, ...]] = (
    DeltaSealReason.WRONG_PARENT,
    DeltaSealReason.WRONG_BRANCH,
    DeltaSealReason.PARENT_NOT_ACCEPTED,
    DeltaSealReason.OLD_ROOT_MISMATCH,
    DeltaSealReason.NEW_ROOT_MISMATCH,
    DeltaSealReason.INCOMPLETE_DIFF,
    DeltaSealReason.INCOMPLETE_MANIFEST,
    DeltaSealReason.EMPTY_REQUIRED_SET,
    DeltaSealReason.MISSING_REPLACEMENT,
    DeltaSealReason.MISSING_ADDITION,
    DeltaSealReason.UNAUTHORIZED_DELETION,
    DeltaSealReason.STALE_REUSE,
    DeltaSealReason.INCOMPLETE_CACHE_KEY,
    DeltaSealReason.OLD_AGGREGATE,
    DeltaSealReason.LOST_LEAF,
    DeltaSealReason.FOREST_MISMATCH,
    DeltaSealReason.REPLAY_REJECTED,
    DeltaSealReason.SIMULATED_REQUIRED_UNIT,
    DeltaSealReason.UNKNOWN_REQUIRED_UNIT,
    DeltaSealReason.UNAVAILABLE_REQUIRED_UNIT,
    DeltaSealReason.TIMEOUT_REQUIRED_UNIT,
    DeltaSealReason.CANCELLED_REQUIRED_UNIT,
    DeltaSealReason.FAILED_REQUIRED_UNIT,
    DeltaSealReason.UNVERIFIED_REQUIRED_UNIT,
    DeltaSealReason.INVALID_DISPOSITION,
    DeltaSealReason.SEALED,
)


def _pick_failure(
    failures: Sequence[tuple[SealStatus, DeltaSealReason]],
) -> tuple[SealStatus, DeltaSealReason] | None:
    if not failures:
        return None
    by_reason = {reason: status for status, reason in failures}
    for reason in _REASON_PRIORITY:
        if reason in by_reason:
            return by_reason[reason], reason
    return failures[0]


class DeltaSealBuilder:
    """Construct a parent-bound delta seal after verifying all fourteen invariants."""

    def build(
        self,
        parent: ParentSealView | Mapping[str, Any],
        new_repository_state: RepositoryStateView | Mapping[str, Any],
        verification_policy: VerificationPolicyView | Mapping[str, Any] | None,
        transition: DeltaTransitionStatement | Mapping[str, Any],
        *,
        units: Sequence[DeltaUnitEvidence | Mapping[str, Any]] = (),
    ) -> DeltaSeal:
        parent_view = _coerce_parent(parent)
        state = _coerce_state(new_repository_state)
        policy = _coerce_policy(verification_policy)
        statement = _coerce_transition(transition)
        parsed = [_coerce_unit(item) for item in units]

        required = [item for item in parsed if item.required_for_seal]
        required.sort(key=lambda item: item.unit_id.encode("utf-8"))

        unit_ids = [item.unit_id for item in required]
        if len(unit_ids) != len(set(unit_ids)):
            raise DeltaSealError("duplicate required unit_id")

        by_disposition: dict[UnitDisposition, list[DeltaUnitEvidence]] = {
            kind: [] for kind in UnitDisposition
        }
        for unit in required:
            disposition = _parse_disposition(unit.disposition)
            by_disposition[disposition].append(unit)

        reused = tuple(
            item.unit_id for item in by_disposition[UnitDisposition.REUSE]
        )
        replaced = tuple(
            item.unit_id for item in by_disposition[UnitDisposition.REPLACE]
        )
        added = tuple(item.unit_id for item in by_disposition[UnitDisposition.ADD])
        removed = tuple(
            item.unit_id for item in by_disposition[UnitDisposition.REMOVE]
        )

        # New required set excludes authorized removals.
        present_units = [
            item
            for item in required
            if _parse_disposition(item.disposition) is not UnitDisposition.REMOVE
        ]
        present_ids = tuple(item.unit_id for item in present_units)

        failures: list[tuple[SealStatus, DeltaSealReason]] = []
        rejected: list[str] = []
        verified: list[str] = []
        invariant_ok: dict[str, bool] = {name: True for name in NORMATIVE_INVARIANTS}

        # --- Invariant 1: parent seal accepted under current policy ---
        parent_status_ok = parent_view.seal_status in _ACCEPTED_PARENT_STATUSES
        if not parent_view.accepted or not parent_status_ok:
            invariant_ok["parent_accepted"] = False
            failures.append(
                (SealStatus.STALE_PARENT, DeltaSealReason.PARENT_NOT_ACCEPTED)
            )

        # --- Invariant 13: exact parent seal binding ---
        if statement.parent_seal_cid != parent_view.seal_cid:
            invariant_ok["exact_parent_bound"] = False
            failures.append((SealStatus.STALE_PARENT, DeltaSealReason.WRONG_PARENT))

        # --- Invariant 14: branch/parent/revision anti-replay binding ---
        if statement.branch_id != parent_view.branch_id:
            invariant_ok["anti_replay_binding"] = False
            failures.append((SealStatus.STALE_PARENT, DeltaSealReason.WRONG_BRANCH))
        if state.repository_id != parent_view.repository_id:
            invariant_ok["anti_replay_binding"] = False
            failures.append((SealStatus.STALE_PARENT, DeltaSealReason.REPLAY_REJECTED))
        if statement.new_revision != state.revision:
            invariant_ok["anti_replay_binding"] = False
            failures.append((SealStatus.STALE_PARENT, DeltaSealReason.REPLAY_REJECTED))
        if tuple(statement.parent_revision_ids) != tuple(state.parent_revision_ids):
            invariant_ok["anti_replay_binding"] = False
            failures.append((SealStatus.STALE_PARENT, DeltaSealReason.REPLAY_REJECTED))
        # Source-root-only replay against a different parent history is rejected:
        # logical epoch must advance strictly past the accepted parent.
        if statement.logical_epoch <= parent_view.logical_epoch:
            invariant_ok["anti_replay_binding"] = False
            failures.append((SealStatus.STALE_PARENT, DeltaSealReason.REPLAY_REJECTED))

        # --- Invariant 2: old root matches declared parent state ---
        if (
            statement.old_source_root_cid != parent_view.source_root_cid
            or statement.old_repository_state_cid != parent_view.repository_state_cid
            or statement.old_manifest_root_cid != parent_view.manifest_root_cid
            or statement.old_forest_root_cid != parent_view.forest_root_cid
            or statement.old_aggregation_root != parent_view.aggregation_root
        ):
            invariant_ok["old_root_matches_parent"] = False
            failures.append(
                (SealStatus.VERIFICATION_FAILED, DeltaSealReason.OLD_ROOT_MISMATCH)
            )

        # --- Invariant 3: new root matches current source state ---
        if (
            statement.new_source_root_cid != state.source_root_cid
            or statement.new_repository_state_cid != state.repository_state_cid
        ):
            invariant_ok["new_root_matches_source"] = False
            failures.append(
                (SealStatus.VERIFICATION_FAILED, DeltaSealReason.NEW_ROOT_MISMATCH)
            )

        # --- Invariant 4: complete changed-artifact set for bound diff ---
        diff = statement.diff
        if (
            not diff.complete
            or not diff.diff_algorithm
            or not diff.changed_artifact_commitment
        ):
            invariant_ok["complete_diff"] = False
            failures.append(
                (SealStatus.INCOMPLETE_MANIFEST, DeltaSealReason.INCOMPLETE_DIFF)
            )

        # --- Invariant 5: every invalidated unit has a newly admitted proof ---
        for unit in by_disposition[UnitDisposition.REPLACE]:
            parent_proof = parent_view.unit_proof_cids.get(unit.unit_id, "")
            if (
                not unit.newly_admitted
                or not unit.freshly_verified
                or not unit.proof_object_cid
                or (
                    parent_proof
                    and unit.proof_object_cid == parent_proof
                    and not unit.newly_admitted
                )
            ):
                invariant_ok["invalidated_have_new_proofs"] = False
                failures.append(
                    (
                        SealStatus.VERIFICATION_FAILED,
                        DeltaSealReason.MISSING_REPLACEMENT,
                    )
                )
                rejected.append(unit.unit_id)
                continue
            # Replacement must not silently keep the stale parent proof object.
            if parent_proof and unit.proof_object_cid == parent_proof:
                invariant_ok["invalidated_have_new_proofs"] = False
                failures.append(
                    (
                        SealStatus.INVALID_CACHE,
                        DeltaSealReason.MISSING_REPLACEMENT,
                    )
                )
                rejected.append(unit.unit_id)

        # --- Invariant 6: reuse has complete unchanged cache key + fresh verify ---
        for unit in by_disposition[UnitDisposition.REUSE]:
            if not unit.cache_key_complete:
                invariant_ok["reuse_complete_cache_key"] = False
                failures.append(
                    (SealStatus.INVALID_CACHE, DeltaSealReason.INCOMPLETE_CACHE_KEY)
                )
                rejected.append(unit.unit_id)
            elif not unit.cache_key_unchanged or not unit.freshly_verified:
                invariant_ok["reuse_complete_cache_key"] = False
                failures.append(
                    (SealStatus.INVALID_CACHE, DeltaSealReason.STALE_REUSE)
                )
                rejected.append(unit.unit_id)

        # --- Invariant 7: deletions are explicit and authorized ---
        for unit in by_disposition[UnitDisposition.REMOVE]:
            if not unit.removal_authorized:
                invariant_ok["deletions_authorized"] = False
                failures.append(
                    (
                        SealStatus.VERIFICATION_FAILED,
                        DeltaSealReason.UNAUTHORIZED_DELETION,
                    )
                )
                rejected.append(unit.unit_id)

        # --- Invariant 8: every added required unit is present and proven ---
        for unit in by_disposition[UnitDisposition.ADD]:
            if (
                not unit.proof_object_cid
                or not unit.freshly_verified
                or not unit.newly_admitted
            ):
                invariant_ok["additions_present_and_proven"] = False
                failures.append(
                    (
                        SealStatus.VERIFICATION_FAILED,
                        DeltaSealReason.MISSING_ADDITION,
                    )
                )
                rejected.append(unit.unit_id)

        # --- Invariant 9: new required-unit manifest is complete ---
        expected_manifest = tuple(
            sorted(set(statement.expected_manifest_unit_ids))
        )
        actual_manifest = tuple(sorted(set(present_ids)))
        if expected_manifest and actual_manifest != expected_manifest:
            invariant_ok["manifest_complete"] = False
            failures.append(
                (
                    SealStatus.INCOMPLETE_MANIFEST,
                    DeltaSealReason.INCOMPLETE_MANIFEST,
                )
            )
            missing = [item for item in expected_manifest if item not in set(present_ids)]
            rejected.extend(missing)
        if not present_ids:
            invariant_ok["manifest_complete"] = False
            failures.append(
                (
                    SealStatus.INCOMPLETE_MANIFEST,
                    DeltaSealReason.EMPTY_REQUIRED_SET,
                )
            )

        # Parent units that disappeared without an explicit remove disposition.
        parent_required = set(parent_view.required_unit_ids)
        declared_ids = set(unit_ids)
        silent_loss = parent_required - declared_ids
        if silent_loss:
            invariant_ok["manifest_complete"] = False
            failures.append(
                (
                    SealStatus.INCOMPLETE_MANIFEST,
                    DeltaSealReason.INCOMPLETE_MANIFEST,
                )
            )
            rejected.extend(sorted(silent_loss))

        # --- Invariant 11: no stale or mismatched proof reused ---
        for unit in by_disposition[UnitDisposition.REUSE]:
            parent_proof = parent_view.unit_proof_cids.get(unit.unit_id, "")
            expected_parent_proof = unit.parent_proof_object_cid or parent_proof
            if unit.stale:
                invariant_ok["no_stale_reuse"] = False
                failures.append(
                    (SealStatus.INVALID_CACHE, DeltaSealReason.STALE_REUSE)
                )
                rejected.append(unit.unit_id)
            elif not unit.proof_object_cid:
                invariant_ok["no_stale_reuse"] = False
                failures.append(
                    (SealStatus.INVALID_CACHE, DeltaSealReason.STALE_REUSE)
                )
                rejected.append(unit.unit_id)
            elif expected_parent_proof and unit.proof_object_cid != expected_parent_proof:
                invariant_ok["no_stale_reuse"] = False
                failures.append(
                    (SealStatus.INVALID_CACHE, DeltaSealReason.STALE_REUSE)
                )
                rejected.append(unit.unit_id)
            elif parent_proof and unit.proof_object_cid != parent_proof:
                invariant_ok["no_stale_reuse"] = False
                failures.append(
                    (SealStatus.INVALID_CACHE, DeltaSealReason.STALE_REUSE)
                )
                rejected.append(unit.unit_id)

        # --- Invariant 12: no blocking / simulated / non-pass unit ---
        for unit in present_units:
            block = _unit_blocking_status(unit)
            if block is not None:
                invariant_ok["no_blocking_units"] = False
                failures.append(block)
                rejected.append(unit.unit_id)
            else:
                verified.append(unit.unit_id)

        # --- Build forest / aggregation roots from present units ---
        by_category: dict[str, list[DeltaUnitEvidence]] = {
            cat: [] for cat in FOREST_CATEGORIES
        }
        for unit in present_units:
            if unit.unit_id in verified:
                cat = _parse_category(unit.category)
                by_category[cat].append(unit)

        category_roots = {
            cat: _category_root(cat, by_category[cat]) for cat in FOREST_CATEGORIES
        }
        new_manifest_root = _manifest_root(
            present_ids, policy_cid=policy.policy_cid
        )
        new_forest_root = _forest_root(
            state=state,
            policy=policy,
            parent_seal_cid=parent_view.seal_cid,
            branch_id=statement.branch_id,
            logical_epoch=statement.logical_epoch,
            manifest_root_cid=new_manifest_root,
            category_roots=category_roots,
        )
        new_aggregation_root = _aggregation_root(present_ids, category_roots)

        transition_changed = bool(replaced or added or removed)
        # --- Invariant 10: forest commits to exact unit set ---
        if not statement.forest_rebuilt and transition_changed:
            invariant_ok["forest_commits_exact_units"] = False
            failures.append(
                (SealStatus.VERIFICATION_FAILED, DeltaSealReason.FOREST_MISMATCH)
            )
        if transition_changed and new_forest_root == parent_view.forest_root_cid:
            # A changed unit set must not reuse the parent forest root.
            invariant_ok["forest_commits_exact_units"] = False
            failures.append(
                (SealStatus.VERIFICATION_FAILED, DeltaSealReason.FOREST_MISMATCH)
            )

        # Old aggregate with a changed unit set is always rejected.
        if not statement.aggregation_rebuilt and transition_changed:
            invariant_ok["forest_commits_exact_units"] = False
            failures.append(
                (SealStatus.VERIFICATION_FAILED, DeltaSealReason.OLD_AGGREGATE)
            )
        if (
            transition_changed
            and new_aggregation_root == parent_view.aggregation_root
            and statement.aggregation_rebuilt
        ):
            # Rebuilt flag claimed but aggregate still equals the parent.
            invariant_ok["forest_commits_exact_units"] = False
            failures.append(
                (SealStatus.VERIFICATION_FAILED, DeltaSealReason.OLD_AGGREGATE)
            )

        # Lost leaf: every parent unit not removed must survive in the new set.
        expected_survivors = set(statement.expected_surviving_leaf_ids)
        if not expected_survivors:
            expected_survivors = parent_required - set(removed)
        actual_present = set(present_ids)
        lost = expected_survivors - actual_present
        if lost:
            invariant_ok["forest_commits_exact_units"] = False
            failures.append(
                (SealStatus.VERIFICATION_FAILED, DeltaSealReason.LOST_LEAF)
            )
            rejected.extend(sorted(lost))

        # Policy / environment continuity for incremental seals (non-fallback).
        if state.environment_cid != parent_view.environment_cid:
            invariant_ok["anti_replay_binding"] = False
            failures.append((SealStatus.STALE_PARENT, DeltaSealReason.REPLAY_REJECTED))
        if policy.policy_cid != parent_view.policy_cid:
            # Policy change requires full checkpoint, not a delta.
            invariant_ok["parent_accepted"] = False
            failures.append(
                (SealStatus.FULL_REPROOF_REQUIRED, DeltaSealReason.PARENT_NOT_ACCEPTED)
            )

        tid = statement.transition_id or _transition_id(
            parent_seal_cid=parent_view.seal_cid,
            branch_id=statement.branch_id,
            revision=state.revision,
            new_source_root_cid=state.source_root_cid,
            new_repository_state_cid=state.repository_state_cid,
            logical_epoch=statement.logical_epoch,
        )

        invariants_passed = tuple(
            name for name in NORMATIVE_INVARIANTS if invariant_ok[name]
        )
        invariants_failed = tuple(
            name for name in NORMATIVE_INVARIANTS if not invariant_ok[name]
        )

        picked = _pick_failure(failures)
        if picked is None and not invariants_failed and present_ids:
            seal_status = SealStatus.SEALED_INCREMENTAL
            reason = DeltaSealReason.SEALED
            sealed = True
        elif picked is None:
            seal_status = SealStatus.VERIFICATION_FAILED
            reason = DeltaSealReason.UNVERIFIED_REQUIRED_UNIT
            sealed = False
        else:
            seal_status, reason = picked
            sealed = False

        return DeltaSeal(
            schema=SEAL_SCHEMA,
            evidence_subset=EVIDENCE_SUBSET,
            fourteen_invariants_evidence=FOURTEEN_INVARIANTS_EVIDENCE,
            seal_status=seal_status,
            reason=reason,
            repository_id=state.repository_id,
            branch_id=statement.branch_id,
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
            parent_seal_cid=parent_view.seal_cid,
            parent_revision_ids=state.parent_revision_ids,
            logical_epoch=statement.logical_epoch,
            transition_id=tid,
            diff_algorithm=diff.diff_algorithm,
            changed_artifact_commitment=diff.changed_artifact_commitment,
            old_source_root_cid=statement.old_source_root_cid,
            old_repository_state_cid=statement.old_repository_state_cid,
            old_manifest_root_cid=statement.old_manifest_root_cid,
            old_forest_root_cid=statement.old_forest_root_cid,
            old_aggregation_root=statement.old_aggregation_root,
            new_manifest_root_cid=new_manifest_root,
            new_forest_root_cid=new_forest_root,
            new_aggregation_root=new_aggregation_root,
            category_roots=category_roots,
            reused_unit_ids=reused,
            replaced_unit_ids=replaced,
            added_unit_ids=added,
            removed_unit_ids=removed,
            required_unit_ids=present_ids,
            verified_unit_ids=tuple(sorted(set(verified))),
            rejected_unit_ids=tuple(sorted(set(rejected))),
            invariants_passed=invariants_passed,
            invariants_failed=invariants_failed,
            sealed=sealed,
        )


def build_delta_seal(
    parent: ParentSealView | Mapping[str, Any],
    new_repository_state: RepositoryStateView | Mapping[str, Any],
    verification_policy: VerificationPolicyView | Mapping[str, Any] | None,
    transition: DeltaTransitionStatement | Mapping[str, Any],
    *,
    units: Sequence[DeltaUnitEvidence | Mapping[str, Any]] = (),
) -> DeltaSeal:
    """Public facade matching the plan document's delta-seal construction."""

    return DeltaSealBuilder().build(
        parent,
        new_repository_state,
        verification_policy,
        transition,
        units=units,
    )


__all__ = (
    "EVIDENCE_SUBSET",
    "FOURTEEN_INVARIANTS_EVIDENCE",
    "FOREST_CATEGORIES",
    "NORMATIVE_INVARIANTS",
    "SEAL_SCHEMA",
    "TRANSITION_SCHEMA",
    "UNIT_DISPOSITIONS",
    "DeltaSeal",
    "DeltaSealBuilder",
    "DeltaSealError",
    "DeltaSealReason",
    "DeltaTransitionStatement",
    "DeltaUnitEvidence",
    "DiffCommitmentView",
    "ParentSealView",
    "UnitDisposition",
    "build_delta_seal",
)
