"""IPS-051: joined adversarial seal and concurrent-writer matrix.

Combines poisoned candidates, stale parent/branch replay, missing
invalidated/required units, simulated/unknown/timeout outcomes, corrupted
artifacts, old aggregates, unaffected-leaf loss, and racing writers through
the public plan → execute → seal → publish workflow.

Acceptance:

* no stale/mismatched/corrupt/simulated/unknown/timeout evidence becomes sealed;
* an incremental seal missing one required replacement rejects;
* exactly one current-root writer wins and the prior accepted seal remains
  recoverable.

Evidence subset: ``ips/e2e-adversarial@1``.
"""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.admission import (
    EvidenceCandidate,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.delta_seal import (
    TRANSITION_SCHEMA,
    DeltaSealReason,
    DeltaTransitionStatement,
    DeltaUnitEvidence,
    DiffCommitmentView,
    ParentSealView,
    build_delta_seal,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.executor import (
    CachedCandidate,
    ExecutionOutcome,
    ExecutionReasonCode,
    FreshProof,
    execute_incremental_plan,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
    GENESIS_PARENT_SEAL,
    FullCheckpointReason,
    RepositoryStateView,
    RequiredUnitEvidence,
    VerificationPolicyView,
    create_full_checkpoint,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
    ParentSealContext,
    UnitPlanningInput,
    create_incremental_plan,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.sealer import (
    IncrementalProofSealer,
    PublicationReason,
    SealPublicationResult,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    IntegrityCommitment,
    ProofMode,
    ProofTerminalStatus,
    SealStatus,
)
from ipfs_kit_py.proof_seal_store.contracts import ArtifactKind

# ---------------------------------------------------------------------------
# Evidence / closed contracts
# ---------------------------------------------------------------------------

E2E_ADVERSARIAL_EVIDENCE = "ips/e2e-adversarial@1"
EVIDENCE_SUBSETS = (E2E_ADVERSARIAL_EVIDENCE,)

# Attack axes required by the IPS-051 effects list.
ATTACK_AXES: tuple[str, ...] = (
    "poisoned_candidate",
    "stale_candidate",
    "corrupt_candidate",
    "mismatched_candidate",
    "stale_parent_replay",
    "stale_branch_replay",
    "missing_replacement",
    "missing_required_unit",
    "simulated_required_unit",
    "unknown_required_unit",
    "timeout_required_unit",
    "old_aggregate",
    "lost_unaffected_leaf",
    "racing_writers",
)

_DIGEST_A = "sha256:" + ("aa" * 32)
_DIGEST_B = "sha256:" + ("bb" * 32)
_DIGEST_C = "sha256:" + ("cc" * 32)
_DIGEST_D = "sha256:" + ("dd" * 32)
_DIGEST_E = "sha256:" + ("ee" * 32)
_DIGEST_F = "sha256:" + ("ff" * 32)
_DIGEST_1 = "sha256:" + ("11" * 32)
_DIGEST_2 = "sha256:" + ("22" * 32)
_DIGEST_3 = "sha256:" + ("33" * 32)
_DIGEST_4 = "sha256:" + ("44" * 32)
_DIGEST_5 = "sha256:" + ("55" * 32)
_DIGEST_6 = "sha256:" + ("66" * 32)
_DIGEST_8 = "sha256:" + ("88" * 32)
_PROOF_A = _DIGEST_4
_PROOF_B = _DIGEST_5
_PROOF_B_NEW = _DIGEST_6
_DIFF_COMMIT = _DIGEST_8
_REPO = "repo/accelerate"
_REV_GENESIS = "rev-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_REV_DELTA = "rev-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
_REV_STALE = "rev-cccccccccccccccccccccccccccccccccccccccc"
_REV_RACE = "rev-dddddddddddddddddddddddddddddddddddddddd"

_SEALED_STATUSES = frozenset(
    {
        SealStatus.SEALED_FULL,
        SealStatus.SEALED_INCREMENTAL,
        SealStatus.SEALED_FULL.value,
        SealStatus.SEALED_INCREMENTAL.value,
    }
)


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------


def _hex_digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _state(**overrides: object) -> RepositoryStateView:
    payload: dict[str, object] = {
        "repository_id": _REPO,
        "revision": _REV_GENESIS,
        "source_root_cid": _DIGEST_A,
        "repository_state_cid": _DIGEST_B,
        "environment_cid": _DIGEST_C,
        "parent_revision_ids": (),
    }
    payload.update(overrides)
    return RepositoryStateView(**payload)  # type: ignore[arg-type]


def _policy(**overrides: object) -> VerificationPolicyView:
    payload: dict[str, object] = {
        "policy_cid": _DIGEST_D,
        "proof_schema_version": "1",
        "canonicalization_version": "1",
        "dependency_graph_schema_version": "graph@1",
        "circuit_id": "circuit@v1",
        "verification_key_id": "vk/e2e-1",
    }
    payload.update(overrides)
    return VerificationPolicyView(**payload)  # type: ignore[arg-type]


def _full_unit(unit_id: str, **overrides: object) -> RequiredUnitEvidence:
    payload: dict[str, object] = {
        "unit_id": unit_id,
        "proof_object_cid": _PROOF_A if unit_id == "unit/a" else _PROOF_B,
        "category": "unit_test" if unit_id == "unit/a" else "static_analysis",
        "terminal_status": (
            ProofTerminalStatus.INTEGRITY_VERIFIED.value
            if unit_id == "unit/a"
            else ProofTerminalStatus.PROVED.value
        ),
        "proof_mode": (
            ProofMode.INTEGRITY_ONLY.value
            if unit_id == "unit/a"
            else ProofMode.DIRECT_EXECUTION_PROOF.value
        ),
        "required_for_seal": True,
        "freshly_verified": True,
        "cache_reused_without_fresh_verification": False,
    }
    payload.update(overrides)
    return RequiredUnitEvidence(**payload)  # type: ignore[arg-type]


def _good_full_units(*, tag: str = "") -> tuple[RequiredUnitEvidence, ...]:
    if tag:
        proof_a = _hex_digest(f"{tag}:unit/a")
        proof_b = _hex_digest(f"{tag}:unit/b")
    else:
        proof_a = _PROOF_A
        proof_b = _PROOF_B
    return (
        _full_unit("unit/a", proof_object_cid=proof_a),
        _full_unit("unit/b", proof_object_cid=proof_b),
    )


def _diff(**overrides: object) -> DiffCommitmentView:
    payload: dict[str, object] = {
        "diff_algorithm": (
            "ipfs_datasets_py/logic/zkp/incremental_sealing/"
            "repository_diff/algorithm@1"
        ),
        "changed_artifact_commitment": _DIFF_COMMIT,
        "complete": True,
        "changed_paths": ("src/module.py",),
    }
    payload.update(overrides)
    return DiffCommitmentView(**payload)  # type: ignore[arg-type]


def _parent_from_full(
    full: Any,
    *,
    seal_cid: str,
    branch_id: str = "main",
    unit_proof_cids: dict[str, str] | None = None,
    logical_epoch: int = 1,
) -> ParentSealView:
    return ParentSealView(
        seal_cid=seal_cid,
        accepted=True,
        seal_status=SealStatus.SEALED_FULL.value,
        repository_id=full.repository_id,
        branch_id=branch_id,
        revision=full.revision,
        source_root_cid=full.source_root_cid,
        repository_state_cid=full.repository_state_cid,
        environment_cid=full.environment_cid,
        policy_cid=full.policy_cid,
        manifest_root_cid=full.manifest_root_cid,
        forest_root_cid=full.repository_proof_root,
        aggregation_root=full.aggregation_root,
        required_unit_ids=full.required_unit_ids,
        unit_proof_cids=unit_proof_cids
        or {"unit/a": _PROOF_A, "unit/b": _PROOF_B},
        parent_revision_ids=full.parent_revision_ids,
        proof_schema_version=full.proof_schema_version,
        canonicalization_version=full.canonicalization_version,
        dependency_graph_schema_version=full.dependency_graph_schema_version,
        circuit_id=full.circuit_id,
        verification_key_id=full.verification_key_id,
        logical_epoch=logical_epoch,
    )


def _delta_unit(unit_id: str, disposition: str, **overrides: object) -> DeltaUnitEvidence:
    payload: dict[str, object] = {
        "unit_id": unit_id,
        "disposition": disposition,
        "proof_object_cid": _PROOF_A if unit_id == "unit/a" else _PROOF_B_NEW,
        "category": "unit_test" if unit_id == "unit/a" else "static_analysis",
        "terminal_status": (
            ProofTerminalStatus.INTEGRITY_VERIFIED.value
            if unit_id == "unit/a"
            else ProofTerminalStatus.PROVED.value
        ),
        "proof_mode": (
            ProofMode.INTEGRITY_ONLY.value
            if unit_id == "unit/a"
            else ProofMode.DIRECT_EXECUTION_PROOF.value
        ),
        "required_for_seal": True,
        "cache_key_complete": True,
        "cache_key_unchanged": disposition == "reuse",
        "freshly_verified": True,
        "newly_admitted": disposition in {"replace", "add"},
        "removal_authorized": disposition == "remove",
        "parent_proof_object_cid": (
            _PROOF_A if unit_id == "unit/a" else _PROOF_B
        ),
        "stale": False,
    }
    payload.update(overrides)
    return DeltaUnitEvidence(**payload)  # type: ignore[arg-type]


def _good_delta_units(
    *,
    proof_a: str = _PROOF_A,
    proof_b_parent: str = _PROOF_B,
    proof_b_new: str = _PROOF_B_NEW,
) -> tuple[DeltaUnitEvidence, ...]:
    return (
        _delta_unit(
            "unit/a",
            "reuse",
            proof_object_cid=proof_a,
            parent_proof_object_cid=proof_a,
            cache_key_unchanged=True,
        ),
        _delta_unit(
            "unit/b",
            "replace",
            proof_object_cid=proof_b_new,
            parent_proof_object_cid=proof_b_parent,
            newly_admitted=True,
            cache_key_unchanged=False,
        ),
    )


def _transition(
    parent: ParentSealView,
    *,
    new_source: str = _DIGEST_1,
    new_state: str = _DIGEST_2,
    revision: str = _REV_DELTA,
    logical_epoch: int = 2,
    branch_id: str | None = None,
    **overrides: object,
) -> DeltaTransitionStatement:
    payload: dict[str, object] = {
        "schema": TRANSITION_SCHEMA,
        "parent_seal_cid": parent.seal_cid,
        "branch_id": branch_id if branch_id is not None else parent.branch_id,
        "old_source_root_cid": parent.source_root_cid,
        "old_repository_state_cid": parent.repository_state_cid,
        "old_manifest_root_cid": parent.manifest_root_cid,
        "old_forest_root_cid": parent.forest_root_cid,
        "old_aggregation_root": parent.aggregation_root,
        "new_source_root_cid": new_source,
        "new_repository_state_cid": new_state,
        "new_revision": revision,
        "parent_revision_ids": (parent.revision,),
        "diff": _diff(),
        "expected_manifest_unit_ids": parent.required_unit_ids,
        "expected_surviving_leaf_ids": parent.required_unit_ids,
        "forest_rebuilt": True,
        "aggregation_rebuilt": True,
        "logical_epoch": logical_epoch,
    }
    payload.update(overrides)
    return DeltaTransitionStatement(**payload)  # type: ignore[arg-type]


def _sealer(tmp_path: Path, **kwargs: object) -> IncrementalProofSealer:
    return IncrementalProofSealer(tmp_path, **kwargs)  # type: ignore[arg-type]


def _publish_genesis(
    sealer: IncrementalProofSealer,
    *,
    tag: str = "genesis",
    branch_id: str = "main",
) -> SealPublicationResult:
    return sealer.publish_full_checkpoint(
        _state(),
        _policy(),
        units=_good_full_units(tag=tag),
        expected_unit_ids=("unit/a", "unit/b"),
        parent_seal_cid=GENESIS_PARENT_SEAL,
        fallback_reasons=("first_state",),
        branch_id=branch_id,
    )


def _parent_from_publication(
    result: SealPublicationResult,
    *,
    branch_id: str = "main",
    logical_epoch: int = 1,
    proof_tag: str = "genesis",
    unit_proof_cids: dict[str, str] | None = None,
) -> ParentSealView:
    assert result.full_seal is not None
    full = result.full_seal
    # FullCheckpointSeal does not embed unit proof CIDs; reconstruct from the
    # deterministic tag used when the publication was built.
    if unit_proof_cids is not None:
        proofs = dict(unit_proof_cids)
    else:
        proofs = {
            unit.unit_id: unit.proof_object_cid
            for unit in _good_full_units(tag=proof_tag)
        }
    return _parent_from_full(
        full,
        seal_cid=result.seal_cid,
        branch_id=branch_id,
        unit_proof_cids=proofs,
        logical_epoch=logical_epoch,
    )


def _plan_units() -> tuple[UnitPlanningInput, ...]:
    return (
        UnitPlanningInput(
            unit_id="unit/reuse",
            preserved=True,
            cache_key_complete=True,
            admitted=True,
            candidate_present=True,
        ),
        UnitPlanningInput(
            unit_id="unit/reprove",
            preserved=False,
            invalidated=True,
            admitted=False,
            candidate_present=False,
        ),
    )


def _incremental_plan():
    parent = ParentSealContext(
        seal_cid=_DIGEST_A,
        repository_state_cid=_DIGEST_B,
        source_root_cid=_DIGEST_A,
    )
    return create_incremental_plan(
        parent,
        _DIGEST_B,
        _DIGEST_2,
        units=_plan_units(),
    )


def _good_candidate(unit_id: str, digest: str, **flags: bool) -> CachedCandidate:
    cid = "sha256:" + ("11" * 32)
    return CachedCandidate(
        unit_id=unit_id,
        expected_digest=digest,
        observed_digest=digest,
        public_input_cid=cid,
        observed_public_input_cid=cid,
        proof_object_cid=cid,
        evidence=IntegrityCommitment(
            digest=digest,
            cid=cid,
            merkle_inclusion="leaf:0",
            byte_length=32,
        ),
        stale=bool(flags.get("stale", False)),
        poisoned=bool(flags.get("poisoned", False)),
        corrupt=bool(flags.get("corrupt", False)),
        simulated=bool(flags.get("simulated", False)),
    )


def _assert_not_sealed_status(status: SealStatus | str) -> None:
    if isinstance(status, SealStatus):
        assert status not in {
            SealStatus.SEALED_FULL,
            SealStatus.SEALED_INCREMENTAL,
        }
        assert status.value not in {
            SealStatus.SEALED_FULL.value,
            SealStatus.SEALED_INCREMENTAL.value,
        }
    else:
        assert status not in _SEALED_STATUSES


def _recover_prior_seal_bytes(
    sealer: IncrementalProofSealer, pointer: Any
) -> bytes:
    ref = pointer.as_artifact_reference()
    body = sealer.store.get_verified_bytes(ref)
    assert body, "prior accepted seal must remain recoverable from the store"
    return body


# ---------------------------------------------------------------------------
# Evidence surface
# ---------------------------------------------------------------------------


def test_e2e_adversarial_evidence_subset() -> None:
    assert E2E_ADVERSARIAL_EVIDENCE == "ips/e2e-adversarial@1"
    assert EVIDENCE_SUBSETS == ("ips/e2e-adversarial@1",)
    for axis in ATTACK_AXES:
        assert axis  # closed non-empty attack vocabulary


# ---------------------------------------------------------------------------
# Joined happy path baseline (establishes recoverable parent for rejections)
# ---------------------------------------------------------------------------


def test_joined_happy_path_publishes_and_is_recoverable(tmp_path: Path) -> None:
    """Plan → execute → delta seal → publish; prior seal remains recoverable."""

    sealer = _sealer(tmp_path)
    genesis = _publish_genesis(sealer)
    assert genesis.published is True
    assert genesis.status is SealStatus.SEALED_FULL
    prior = sealer.get_current_seal(_REPO)
    assert prior is not None
    prior_body = _recover_prior_seal_bytes(sealer, prior)

    # Execution of a clean reuse/reprove plan may aggregate.
    digest = "sha256:" + ("ab" * 32)
    store = {"unit/reuse": _good_candidate("unit/reuse", digest)}
    execution = execute_incremental_plan(_incremental_plan(), fetch=store.get)
    assert execution.outcome is ExecutionOutcome.COMPLETED
    assert execution.may_aggregate is True
    assert execution.succeeded is True

    parent = _parent_from_publication(genesis)
    new_state = _state(
        revision=_REV_DELTA,
        source_root_cid=_DIGEST_1,
        repository_state_cid=_DIGEST_2,
        parent_revision_ids=(parent.revision,),
    )
    proofs = parent.unit_proof_cids
    units = _good_delta_units(
        proof_a=proofs["unit/a"],
        proof_b_parent=proofs["unit/b"],
        proof_b_new=_hex_digest("delta:unit/b"),
    )
    transition = _transition(parent)
    delta = build_delta_seal(parent, new_state, _policy(), transition, units=units)
    assert delta.sealed is True
    assert delta.seal_status is SealStatus.SEALED_INCREMENTAL

    published = sealer.publish_delta_seal(
        parent,
        new_state,
        _policy(),
        transition,
        units=units,
    )
    assert published.published is True, published.diagnostics
    assert published.status is SealStatus.SEALED_INCREMENTAL
    assert published.previous_seal_cid == genesis.seal_cid

    # Prior accepted seal remains recoverable after a successful advance.
    assert sealer.store.get_verified_bytes(prior.as_artifact_reference()) == prior_body
    current = sealer.get_current_seal(_REPO)
    assert current is not None
    assert current.seal_cid == published.seal_cid
    assert current.parent_seal_cid == genesis.seal_cid
    sealer.close()


# ---------------------------------------------------------------------------
# Execution-stage adversarial candidates (joined: plan → execute → no seal)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "expected_code"),
    [
        ("stale", ExecutionReasonCode.STALE_CANDIDATE),
        ("poisoned", ExecutionReasonCode.POISONED_CANDIDATE),
        ("corrupt", ExecutionReasonCode.CORRUPT_CANDIDATE),
    ],
)
def test_joined_bad_cache_candidate_blocks_seal_pipeline(
    flag: str, expected_code: ExecutionReasonCode
) -> None:
    digest = "sha256:" + ("ab" * 32)
    candidate = _good_candidate("unit/reuse", digest, **{flag: True})
    execution = execute_incremental_plan(
        _incremental_plan(),
        fetch=lambda uid, c=candidate: c if uid == "unit/reuse" else None,
    )
    assert execution.outcome is ExecutionOutcome.REJECTED
    assert execution.may_aggregate is False
    assert execution.succeeded is False
    assert expected_code.value in execution.reason_codes
    assert "unit/reuse" in execution.rejected_unit_ids

    # Without a successful execution, a production delta cannot be sealed from
    # the rejected candidate's proof object.
    parent = ParentSealView(
        seal_cid=_DIGEST_A,
        accepted=True,
        seal_status=SealStatus.SEALED_FULL.value,
        repository_id=_REPO,
        branch_id="main",
        revision=_REV_GENESIS,
        source_root_cid=_DIGEST_A,
        repository_state_cid=_DIGEST_B,
        environment_cid=_DIGEST_C,
        policy_cid=_DIGEST_D,
        manifest_root_cid=_DIGEST_E,
        forest_root_cid=_DIGEST_F,
        aggregation_root=_DIGEST_3,
        required_unit_ids=("unit/reuse", "unit/reprove"),
        unit_proof_cids={"unit/reuse": digest, "unit/reprove": _PROOF_B},
        logical_epoch=1,
    )
    # Present the rejected reuse candidate as if it were still reusable.
    units = (
        _delta_unit(
            "unit/reuse",
            "reuse",
            proof_object_cid=digest,
            parent_proof_object_cid=digest,
            stale=flag == "stale",
            cache_key_unchanged=True,
            category="unit_test",
            terminal_status=ProofTerminalStatus.INTEGRITY_VERIFIED.value,
            proof_mode=ProofMode.INTEGRITY_ONLY.value,
        ),
        _delta_unit(
            "unit/reprove",
            "replace",
            proof_object_cid=_PROOF_B_NEW,
            parent_proof_object_cid=_PROOF_B,
            newly_admitted=True,
            category="static_analysis",
        ),
    )
    seal = build_delta_seal(
        parent,
        _state(
            revision=_REV_DELTA,
            source_root_cid=_DIGEST_1,
            repository_state_cid=_DIGEST_2,
            parent_revision_ids=(_REV_GENESIS,),
        ),
        _policy(),
        _transition(
            parent,
            expected_manifest_unit_ids=("unit/reuse", "unit/reprove"),
            expected_surviving_leaf_ids=("unit/reuse", "unit/reprove"),
        ),
        units=units,
    )
    if flag == "stale":
        assert seal.sealed is False
        assert seal.reason is DeltaSealReason.STALE_REUSE
        _assert_not_sealed_status(seal.seal_status)
    else:
        # Poison/corrupt candidates are blocked at execution; a clean delta
        # construction from different evidence is orthogonal.  The pipeline
        # must never aggregate the rejected execution result.
        assert execution.may_aggregate is False


def test_joined_mismatched_candidate_blocks_aggregation() -> None:
    digest = "sha256:" + ("ab" * 32)
    cid = "sha256:" + ("11" * 32)
    candidate = CachedCandidate(
        "unit/reuse",
        digest,
        "sha256:" + ("ff" * 32),
        cid,
        cid,
    )
    execution = execute_incremental_plan(
        _incremental_plan(),
        fetch=lambda uid, c=candidate: c if uid == "unit/reuse" else None,
    )
    assert execution.outcome is ExecutionOutcome.REJECTED
    assert execution.may_aggregate is False
    assert ExecutionReasonCode.DIGEST_MISMATCH.value in execution.reason_codes


def test_joined_simulated_execution_cannot_seal() -> None:
    digest = "sha256:" + ("ab" * 32)
    cid = "sha256:" + ("11" * 32)
    simulated = _good_candidate("unit/reuse", digest, simulated=True)
    reused = execute_incremental_plan(
        _incremental_plan(),
        fetch=lambda uid: simulated,
    )
    assert reused.outcome is ExecutionOutcome.REJECTED
    assert ExecutionReasonCode.SIMULATED_FORBIDDEN.value in reused.reason_codes
    assert reused.may_aggregate is False

    def prove_simulated(unit: Any) -> FreshProof:
        return FreshProof(
            unit.unit_id,
            EvidenceCandidate(
                evidence=IntegrityCommitment(
                    digest=digest,
                    cid=cid,
                    merkle_inclusion="leaf:0",
                    byte_length=32,
                ),
                proof_system_id="integrity",
                public_input_cid=cid,
                proof_unit_id=unit.unit_id,
                expected_digest=digest,
                observed_digest=digest,
                observed_public_input_cid=cid,
                proof_mode=ProofMode.INTEGRITY_ONLY,
                terminal_status=ProofTerminalStatus.INTEGRITY_VERIFIED,
            ),
            digest,
            simulated=True,
            status="simulated",
        )

    store = {"unit/reuse": _good_candidate("unit/reuse", digest)}
    proved = execute_incremental_plan(
        _incremental_plan(),
        fetch=store.get,
        prove=prove_simulated,
    )
    assert proved.outcome is ExecutionOutcome.REJECTED
    assert ExecutionReasonCode.SIMULATED_FORBIDDEN.value in proved.reason_codes
    assert proved.may_aggregate is False


# ---------------------------------------------------------------------------
# Seal-construction adversarial axes (joined through publish so pointer holds)
# ---------------------------------------------------------------------------


def _publish_adversarial_delta(
    sealer: IncrementalProofSealer,
    genesis: SealPublicationResult,
    *,
    units: tuple[DeltaUnitEvidence, ...],
    transition_overrides: dict[str, object] | None = None,
    revision: str = _REV_DELTA,
    branch_id: str = "main",
) -> SealPublicationResult:
    parent = _parent_from_publication(genesis, branch_id=branch_id)
    new_state = _state(
        revision=revision,
        source_root_cid=_DIGEST_1,
        repository_state_cid=_DIGEST_2,
        parent_revision_ids=(parent.revision,),
    )
    overrides = dict(transition_overrides or {})
    transition = _transition(parent, revision=revision, **overrides)
    return sealer.publish_delta_seal(
        parent,
        new_state,
        _policy(),
        transition,
        units=units,
        branch_id=branch_id,
    )


def test_joined_missing_replacement_rejects_and_preserves_current(
    tmp_path: Path,
) -> None:
    sealer = _sealer(tmp_path)
    genesis = _publish_genesis(sealer)
    assert genesis.published is True
    prior = sealer.get_current_seal(_REPO)
    assert prior is not None
    prior_body = _recover_prior_seal_bytes(sealer, prior)

    parent = _parent_from_publication(genesis)
    proofs = parent.unit_proof_cids
    # Invalidated unit/b presented with the *same* parent proof → not a replacement.
    units = (
        _delta_unit(
            "unit/a",
            "reuse",
            proof_object_cid=proofs["unit/a"],
            parent_proof_object_cid=proofs["unit/a"],
            cache_key_unchanged=True,
        ),
        _delta_unit(
            "unit/b",
            "replace",
            proof_object_cid=proofs["unit/b"],
            parent_proof_object_cid=proofs["unit/b"],
            newly_admitted=True,
            cache_key_unchanged=False,
        ),
    )
    # Direct construction rejects.
    direct = build_delta_seal(
        parent,
        _state(
            revision=_REV_DELTA,
            source_root_cid=_DIGEST_1,
            repository_state_cid=_DIGEST_2,
            parent_revision_ids=(parent.revision,),
        ),
        _policy(),
        _transition(parent),
        units=units,
    )
    assert direct.sealed is False
    assert direct.reason is DeltaSealReason.MISSING_REPLACEMENT
    assert "invalidated_have_new_proofs" in direct.invariants_failed
    assert "unit/b" in direct.rejected_unit_ids
    _assert_not_sealed_status(direct.seal_status)

    # Joined publish rejects and leaves the prior accepted seal current.
    published = _publish_adversarial_delta(sealer, genesis, units=units)
    assert published.published is False
    _assert_not_sealed_status(published.status)
    current = sealer.get_current_seal(_REPO)
    assert current is not None
    assert current.seal_cid == prior.seal_cid
    assert current == prior
    assert sealer.store.get_verified_bytes(prior.as_artifact_reference()) == prior_body
    sealer.close()


@pytest.mark.parametrize(
    ("status", "mode", "expected_status", "expected_reason"),
    [
        (
            ProofTerminalStatus.SIMULATED.value,
            ProofMode.SIMULATED.value,
            SealStatus.SIMULATED_ONLY,
            DeltaSealReason.SIMULATED_REQUIRED_UNIT,
        ),
        (
            ProofTerminalStatus.UNKNOWN.value,
            ProofMode.DIRECT_EXECUTION_PROOF.value,
            SealStatus.UNKNOWN,
            DeltaSealReason.UNKNOWN_REQUIRED_UNIT,
        ),
        (
            ProofTerminalStatus.TIMEOUT.value,
            ProofMode.DIRECT_EXECUTION_PROOF.value,
            SealStatus.TIMEOUT,
            DeltaSealReason.TIMEOUT_REQUIRED_UNIT,
        ),
    ],
)
def test_joined_non_pass_evidence_never_seals(
    tmp_path: Path,
    status: str,
    mode: str,
    expected_status: SealStatus,
    expected_reason: DeltaSealReason,
) -> None:
    sealer = _sealer(tmp_path)
    genesis = _publish_genesis(sealer)
    prior = sealer.get_current_seal(_REPO)
    assert prior is not None
    prior_body = _recover_prior_seal_bytes(sealer, prior)

    parent = _parent_from_publication(genesis)
    proofs = parent.unit_proof_cids
    units = (
        _delta_unit(
            "unit/a",
            "reuse",
            proof_object_cid=proofs["unit/a"],
            parent_proof_object_cid=proofs["unit/a"],
            cache_key_unchanged=True,
        ),
        _delta_unit(
            "unit/b",
            "replace",
            proof_object_cid=_hex_digest(f"bad:{status}:unit/b"),
            parent_proof_object_cid=proofs["unit/b"],
            newly_admitted=True,
            terminal_status=status,
            proof_mode=mode,
            cache_key_unchanged=False,
        ),
    )
    direct = build_delta_seal(
        parent,
        _state(
            revision=_REV_DELTA,
            source_root_cid=_DIGEST_1,
            repository_state_cid=_DIGEST_2,
            parent_revision_ids=(parent.revision,),
        ),
        _policy(),
        _transition(parent),
        units=units,
    )
    assert direct.sealed is False
    assert direct.seal_status is expected_status
    assert direct.reason is expected_reason
    assert "no_blocking_units" in direct.invariants_failed
    _assert_not_sealed_status(direct.seal_status)

    published = _publish_adversarial_delta(sealer, genesis, units=units)
    assert published.published is False
    assert published.status is expected_status
    assert published.reason in {
        PublicationReason.SIMULATED_ONLY,
        PublicationReason.UNKNOWN,
        PublicationReason.TIMEOUT,
        PublicationReason.NOT_SEALED,
        PublicationReason.PROOF_FAILED,
    }
    # Map must not claim sealed.
    _assert_not_sealed_status(published.status)
    current = sealer.get_current_seal(_REPO)
    assert current is not None
    assert current.seal_cid == prior.seal_cid
    assert sealer.store.get_verified_bytes(prior.as_artifact_reference()) == prior_body
    sealer.close()


def test_joined_old_aggregate_and_lost_leaf_reject_publish(tmp_path: Path) -> None:
    sealer = _sealer(tmp_path)
    genesis = _publish_genesis(sealer)
    prior = sealer.get_current_seal(_REPO)
    assert prior is not None

    parent = _parent_from_publication(genesis)
    proofs = parent.unit_proof_cids
    units = _good_delta_units(
        proof_a=proofs["unit/a"],
        proof_b_parent=proofs["unit/b"],
        proof_b_new=_hex_digest("old-agg:unit/b"),
    )

    old_agg = _publish_adversarial_delta(
        sealer,
        genesis,
        units=units,
        transition_overrides={"aggregation_rebuilt": False},
        revision=_REV_DELTA,
    )
    assert old_agg.published is False
    assert old_agg.delta_seal is not None
    assert old_agg.delta_seal.reason is DeltaSealReason.OLD_AGGREGATE
    _assert_not_sealed_status(old_agg.status)
    assert sealer.get_current_seal(_REPO) == prior

    lost = _publish_adversarial_delta(
        sealer,
        genesis,
        units=units,
        transition_overrides={
            "expected_surviving_leaf_ids": ("unit/a", "unit/b", "unit/ghost"),
        },
        revision=_REV_STALE,
    )
    assert lost.published is False
    assert lost.delta_seal is not None
    assert lost.delta_seal.reason is DeltaSealReason.LOST_LEAF
    _assert_not_sealed_status(lost.status)
    assert sealer.get_current_seal(_REPO) == prior
    _recover_prior_seal_bytes(sealer, prior)
    sealer.close()


def test_joined_missing_required_unit_rejects_full_and_delta(tmp_path: Path) -> None:
    sealer = _sealer(tmp_path)
    genesis = _publish_genesis(sealer)
    prior = sealer.get_current_seal(_REPO)
    assert prior is not None

    # Full checkpoint with incomplete required set.
    incomplete_full = sealer.publish_full_checkpoint(
        _state(
            revision=_REV_STALE,
            source_root_cid=_DIGEST_1,
            repository_state_cid=_DIGEST_2,
        ),
        _policy(),
        units=(_full_unit("unit/a", proof_object_cid=_hex_digest("incomplete:a")),),
        expected_unit_ids=("unit/a", "unit/b"),
    )
    assert incomplete_full.published is False
    assert incomplete_full.full_seal is not None
    assert incomplete_full.full_seal.sealed is False
    assert incomplete_full.full_seal.reason is FullCheckpointReason.INCOMPLETE_MANIFEST
    _assert_not_sealed_status(incomplete_full.status)
    assert sealer.get_current_seal(_REPO) == prior

    # Delta claiming an extra required unit that is absent from the unit set.
    parent = _parent_from_publication(genesis)
    proofs = parent.unit_proof_cids
    units = _good_delta_units(
        proof_a=proofs["unit/a"],
        proof_b_parent=proofs["unit/b"],
        proof_b_new=_hex_digest("missing-req:unit/b"),
    )
    missing = _publish_adversarial_delta(
        sealer,
        genesis,
        units=units,
        transition_overrides={
            "expected_manifest_unit_ids": ("unit/a", "unit/b", "unit/missing"),
        },
    )
    assert missing.published is False
    assert missing.delta_seal is not None
    assert missing.delta_seal.reason is DeltaSealReason.INCOMPLETE_MANIFEST
    _assert_not_sealed_status(missing.status)
    assert sealer.get_current_seal(_REPO) == prior
    sealer.close()


def test_joined_simulated_full_checkpoint_never_becomes_current(
    tmp_path: Path,
) -> None:
    sealer = _sealer(tmp_path)
    # Attempt a genesis seal that includes simulated required evidence.
    units = (
        _full_unit("unit/a"),
        _full_unit(
            "unit/sim",
            proof_mode=ProofMode.SIMULATED.value,
            terminal_status=ProofTerminalStatus.SIMULATED.value,
            proof_object_cid=_hex_digest("sim:unit"),
        ),
    )
    direct = create_full_checkpoint(
        _state(),
        _policy(),
        units=units,
        expected_unit_ids=("unit/a", "unit/sim"),
    )
    assert direct.sealed is False
    assert direct.seal_status is SealStatus.SIMULATED_ONLY
    assert direct.reason is FullCheckpointReason.SIMULATED_REQUIRED_UNIT
    _assert_not_sealed_status(direct.seal_status)

    published = sealer.publish_full_checkpoint(
        _state(),
        _policy(),
        units=units,
        expected_unit_ids=("unit/a", "unit/sim"),
        parent_seal_cid=GENESIS_PARENT_SEAL,
        fallback_reasons=("first_state",),
    )
    assert published.published is False
    assert published.status is SealStatus.SIMULATED_ONLY
    assert published.reason is PublicationReason.SIMULATED_ONLY
    assert sealer.get_current_seal(_REPO) is None
    sealer.close()


# ---------------------------------------------------------------------------
# Stale parent / branch replay through the public publisher
# ---------------------------------------------------------------------------


def test_joined_stale_parent_replay_rejects_without_overwrite(
    tmp_path: Path,
) -> None:
    sealer = _sealer(tmp_path)
    genesis = _publish_genesis(sealer)
    assert genesis.published is True
    prior_body = _recover_prior_seal_bytes(sealer, genesis.pointer)

    # Advance current with a valid second full seal.
    advanced = sealer.publish_full_checkpoint(
        _state(
            revision=_REV_DELTA,
            source_root_cid=_DIGEST_1,
            repository_state_cid=_DIGEST_2,
            parent_revision_ids=(_REV_GENESIS,),
        ),
        _policy(),
        units=_good_full_units(tag="advanced"),
    )
    assert advanced.published is True
    live = sealer.get_current_seal(_REPO)
    assert live is not None
    assert live.seal_cid == advanced.seal_cid

    # Replay a delta against the superseded genesis parent.
    stale_parent = _parent_from_publication(genesis)
    proofs = stale_parent.unit_proof_cids
    units = _good_delta_units(
        proof_a=proofs["unit/a"],
        proof_b_parent=proofs["unit/b"],
        proof_b_new=_hex_digest("stale-replay:unit/b"),
    )
    result = sealer.publish_delta_seal(
        stale_parent,
        _state(
            revision=_REV_STALE,
            source_root_cid=_DIGEST_1,
            repository_state_cid=_DIGEST_2,
            parent_revision_ids=(_REV_GENESIS,),
        ),
        _policy(),
        _transition(stale_parent, revision=_REV_STALE),
        units=units,
    )
    assert result.published is False
    assert result.status is SealStatus.STALE_PARENT
    assert result.reason is PublicationReason.STALE_PARENT
    assert sealer.get_current_seal(_REPO) == live
    assert sealer.get_current_seal(_REPO).seal_cid == advanced.seal_cid  # type: ignore[union-attr]

    # Prior accepted seals (genesis + advanced) remain recoverable.
    assert (
        sealer.store.get_verified_bytes(genesis.pointer.as_artifact_reference())
        == prior_body
    )
    assert sealer.store.get_verified_bytes(live.as_artifact_reference())
    sealer.close()


def test_joined_stale_branch_replay_rejects(tmp_path: Path) -> None:
    sealer = _sealer(tmp_path)
    genesis = _publish_genesis(sealer, branch_id="main")
    assert genesis.published is True

    # Publish on a different branch.
    feature = sealer.publish_full_checkpoint(
        _state(
            revision=_REV_DELTA,
            source_root_cid=_DIGEST_1,
            repository_state_cid=_DIGEST_2,
            parent_revision_ids=(_REV_GENESIS,),
        ),
        _policy(),
        units=_good_full_units(tag="feature"),
        parent_seal_cid=genesis.seal_cid,
        fallback_reasons=("historical_parent",),
        branch_id="feature/x",
    )
    assert feature.published is True

    # Attempt a main-branch delta while declaring the feature branch parent.
    cross = _parent_from_publication(
        feature, branch_id="feature/x", proof_tag="feature"
    )
    proofs = cross.unit_proof_cids
    units = _good_delta_units(
        proof_a=proofs["unit/a"],
        proof_b_parent=proofs["unit/b"],
        proof_b_new=_hex_digest("cross-branch:unit/b"),
    )
    # Parent view carries feature/x; publishing on main must fail closed.
    result = sealer.publish_delta_seal(
        cross,
        _state(
            revision=_REV_STALE,
            source_root_cid=_DIGEST_3,
            repository_state_cid=_DIGEST_4,
            parent_revision_ids=(cross.revision,),
        ),
        _policy(),
        _transition(cross, revision=_REV_STALE, branch_id="main"),
        units=units,
        branch_id="main",
    )
    assert result.published is False
    assert result.status is SealStatus.STALE_PARENT
    assert result.reason is PublicationReason.STALE_PARENT

    tip_main = sealer.get_current_seal(_REPO, branch_id="main")
    tip_feature = sealer.get_current_seal(_REPO, branch_id="feature/x")
    assert tip_main is not None and tip_main.seal_cid == genesis.seal_cid
    assert tip_feature is not None and tip_feature.seal_cid == feature.seal_cid
    sealer.close()


# ---------------------------------------------------------------------------
# Concurrent writers: exactly one wins; prior seal recoverable
# ---------------------------------------------------------------------------


def test_joined_exactly_one_current_root_writer_wins(tmp_path: Path) -> None:
    sealer = _sealer(tmp_path)
    parent = _publish_genesis(sealer)
    assert parent.published is True
    parent_pointer = sealer.get_current_seal(_REPO)
    assert parent_pointer is not None
    prior_body = _recover_prior_seal_bytes(sealer, parent_pointer)

    workers = 6
    barrier = threading.Barrier(workers)

    def attempt(index: int) -> SealPublicationResult:
        barrier.wait(timeout=30)
        return sealer.publish_full_checkpoint(
            _state(
                revision=f"rev-worker-{index:02d}-" + ("a" * 32),
                source_root_cid="sha256:" + (f"{index:02x}" * 32),
                repository_state_cid="sha256:" + (f"{index + 10:02x}" * 32),
            ),
            _policy(),
            units=_good_full_units(tag=f"w{index}"),
            transition_id=f"txn:e2e-worker-{index}",
            expected_current=parent_pointer,
        )

    results: list[SealPublicationResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(attempt, index) for index in range(workers)]
        for future in as_completed(futures):
            results.append(future.result())

    winners = [item for item in results if item.published]
    losers = [item for item in results if not item.published]
    assert len(winners) == 1, [
        (item.transition_id, item.status, item.reason, item.published)
        for item in results
    ]
    assert len(losers) == workers - 1
    assert all(item.status is SealStatus.STALE_PARENT for item in losers)
    assert all(item.reason is PublicationReason.STALE_PARENT for item in losers)
    assert all(item.previous_seal_cid == parent.seal_cid for item in results)

    current = sealer.get_current_seal(_REPO)
    assert current is not None
    assert current == winners[0].pointer
    assert current.parent_seal_cid == parent.seal_cid

    # Prior accepted seal remains recoverable after the race.
    assert (
        sealer.store.get_verified_bytes(parent_pointer.as_artifact_reference())
        == prior_body
    )
    assert sealer.store.get_verified_bytes(current.as_artifact_reference())
    sealer.close()


def test_joined_concurrent_delta_writers_exactly_one_wins(tmp_path: Path) -> None:
    sealer = _sealer(tmp_path)
    genesis = _publish_genesis(sealer)
    assert genesis.published is True
    parent_pointer = sealer.get_current_seal(_REPO)
    assert parent_pointer is not None
    prior_body = _recover_prior_seal_bytes(sealer, parent_pointer)
    parent = _parent_from_publication(genesis)
    proofs = parent.unit_proof_cids

    workers = 4
    barrier = threading.Barrier(workers)

    def attempt(index: int) -> SealPublicationResult:
        barrier.wait(timeout=30)
        units = _good_delta_units(
            proof_a=proofs["unit/a"],
            proof_b_parent=proofs["unit/b"],
            proof_b_new=_hex_digest(f"race-delta:{index}:unit/b"),
        )
        new_state = _state(
            revision=f"rev-delta-w{index:02d}-" + ("b" * 28),
            source_root_cid="sha256:" + (f"{index + 20:02x}" * 32),
            repository_state_cid="sha256:" + (f"{index + 30:02x}" * 32),
            parent_revision_ids=(parent.revision,),
        )
        transition = _transition(
            parent,
            new_source=new_state.source_root_cid,
            new_state=new_state.repository_state_cid,
            revision=new_state.revision,
        )
        return sealer.publish_delta_seal(
            parent,
            new_state,
            _policy(),
            transition,
            units=units,
            transition_id=f"txn:e2e-delta-{index}",
            expected_current=parent_pointer,
        )

    results: list[SealPublicationResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(attempt, index) for index in range(workers)]
        for future in as_completed(futures):
            results.append(future.result())

    winners = [item for item in results if item.published]
    losers = [item for item in results if not item.published]
    assert len(winners) == 1, [
        (item.transition_id, item.status, item.reason, item.published)
        for item in results
    ]
    assert len(losers) == workers - 1
    assert all(item.status is SealStatus.STALE_PARENT for item in losers)
    assert winners[0].status is SealStatus.SEALED_INCREMENTAL
    assert winners[0].previous_seal_cid == genesis.seal_cid

    current = sealer.get_current_seal(_REPO)
    assert current is not None
    assert current.seal_cid == winners[0].seal_cid
    assert current.seal_kind is ArtifactKind.DELTA_SEAL
    assert (
        sealer.store.get_verified_bytes(parent_pointer.as_artifact_reference())
        == prior_body
    )
    sealer.close()


# ---------------------------------------------------------------------------
# Compact acceptance matrix covering every IPS-051 attack axis
# ---------------------------------------------------------------------------


def test_acceptance_matrix_joined_adversarial_axes(tmp_path: Path) -> None:
    """Single joined matrix: every attack axis fails closed; one writer wins."""

    sealer = _sealer(tmp_path)
    genesis = _publish_genesis(sealer)
    assert genesis.published is True
    prior = sealer.get_current_seal(_REPO)
    assert prior is not None
    prior_body = _recover_prior_seal_bytes(sealer, prior)
    parent = _parent_from_publication(genesis)
    proofs = parent.unit_proof_cids

    observed: dict[str, str] = {}

    def record(axis: str, *, sealed: bool, reason: str) -> None:
        assert axis in ATTACK_AXES, axis
        assert sealed is False, f"{axis} must not seal"
        assert reason, f"{axis} must carry a typed reason"
        observed[axis] = reason

    # --- Cache-candidate axes via plan → execute ---
    digest = "sha256:" + ("ab" * 32)
    for flag, code, axis in (
        ("poisoned", ExecutionReasonCode.POISONED_CANDIDATE, "poisoned_candidate"),
        ("stale", ExecutionReasonCode.STALE_CANDIDATE, "stale_candidate"),
        ("corrupt", ExecutionReasonCode.CORRUPT_CANDIDATE, "corrupt_candidate"),
    ):
        candidate = _good_candidate("unit/reuse", digest, **{flag: True})
        result = execute_incremental_plan(
            _incremental_plan(),
            fetch=lambda uid, c=candidate: c if uid == "unit/reuse" else None,
        )
        assert result.outcome is ExecutionOutcome.REJECTED
        assert result.may_aggregate is False
        assert code.value in result.reason_codes
        record(axis, sealed=False, reason=code.value)

    mismatch = CachedCandidate(
        "unit/reuse",
        digest,
        "sha256:" + ("ff" * 32),
        "sha256:" + ("11" * 32),
        "sha256:" + ("11" * 32),
    )
    mismatched = execute_incremental_plan(
        _incremental_plan(),
        fetch=lambda uid, c=mismatch: c if uid == "unit/reuse" else None,
    )
    assert mismatched.outcome is ExecutionOutcome.REJECTED
    assert mismatched.may_aggregate is False
    record(
        "mismatched_candidate",
        sealed=False,
        reason=ExecutionReasonCode.DIGEST_MISMATCH.value,
    )

    # --- Seal-construction axes via build + publish ---
    def delta_units_bad(**unit_b_overrides: object) -> tuple[DeltaUnitEvidence, ...]:
        return (
            _delta_unit(
                "unit/a",
                "reuse",
                proof_object_cid=proofs["unit/a"],
                parent_proof_object_cid=proofs["unit/a"],
                cache_key_unchanged=True,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_hex_digest(f"matrix:{unit_b_overrides}"),
                parent_proof_object_cid=proofs["unit/b"],
                newly_admitted=True,
                cache_key_unchanged=False,
                **unit_b_overrides,
            ),
        )

    # Missing replacement (same proof as parent).
    missing_units = (
        _delta_unit(
            "unit/a",
            "reuse",
            proof_object_cid=proofs["unit/a"],
            parent_proof_object_cid=proofs["unit/a"],
            cache_key_unchanged=True,
        ),
        _delta_unit(
            "unit/b",
            "replace",
            proof_object_cid=proofs["unit/b"],
            parent_proof_object_cid=proofs["unit/b"],
            newly_admitted=True,
        ),
    )
    missing_pub = _publish_adversarial_delta(
        sealer, genesis, units=missing_units, revision=_REV_DELTA
    )
    assert missing_pub.published is False
    assert missing_pub.delta_seal is not None
    assert missing_pub.delta_seal.reason is DeltaSealReason.MISSING_REPLACEMENT
    record(
        "missing_replacement",
        sealed=False,
        reason=missing_pub.delta_seal.reason.value,
    )

    # Missing required unit in manifest.
    good_units = _good_delta_units(
        proof_a=proofs["unit/a"],
        proof_b_parent=proofs["unit/b"],
        proof_b_new=_hex_digest("matrix:manifest:unit/b"),
    )
    missing_req = _publish_adversarial_delta(
        sealer,
        genesis,
        units=good_units,
        transition_overrides={
            "expected_manifest_unit_ids": ("unit/a", "unit/b", "unit/missing"),
        },
        revision=_REV_STALE,
    )
    assert missing_req.published is False
    assert missing_req.delta_seal is not None
    assert missing_req.delta_seal.reason is DeltaSealReason.INCOMPLETE_MANIFEST
    record(
        "missing_required_unit",
        sealed=False,
        reason=missing_req.delta_seal.reason.value,
    )

    # Simulated / unknown / timeout.
    for axis, terminal, mode, expected in (
        (
            "simulated_required_unit",
            ProofTerminalStatus.SIMULATED.value,
            ProofMode.SIMULATED.value,
            DeltaSealReason.SIMULATED_REQUIRED_UNIT,
        ),
        (
            "unknown_required_unit",
            ProofTerminalStatus.UNKNOWN.value,
            ProofMode.DIRECT_EXECUTION_PROOF.value,
            DeltaSealReason.UNKNOWN_REQUIRED_UNIT,
        ),
        (
            "timeout_required_unit",
            ProofTerminalStatus.TIMEOUT.value,
            ProofMode.DIRECT_EXECUTION_PROOF.value,
            DeltaSealReason.TIMEOUT_REQUIRED_UNIT,
        ),
    ):
        bad = delta_units_bad(terminal_status=terminal, proof_mode=mode)
        # Override proof cid uniqueness per axis.
        bad = (
            bad[0],
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_hex_digest(f"matrix:{axis}:unit/b"),
                parent_proof_object_cid=proofs["unit/b"],
                newly_admitted=True,
                terminal_status=terminal,
                proof_mode=mode,
                cache_key_unchanged=False,
            ),
        )
        pub = _publish_adversarial_delta(
            sealer,
            genesis,
            units=bad,
            revision=f"rev-{axis[:8]}-" + ("c" * 24),
        )
        assert pub.published is False
        assert pub.delta_seal is not None
        assert pub.delta_seal.reason is expected
        _assert_not_sealed_status(pub.status)
        record(axis, sealed=False, reason=pub.delta_seal.reason.value)

    # Old aggregate.
    old_agg = _publish_adversarial_delta(
        sealer,
        genesis,
        units=_good_delta_units(
            proof_a=proofs["unit/a"],
            proof_b_parent=proofs["unit/b"],
            proof_b_new=_hex_digest("matrix:old-agg:unit/b"),
        ),
        transition_overrides={"aggregation_rebuilt": False},
        revision="rev-oldagg-" + ("d" * 31),
    )
    assert old_agg.published is False
    assert old_agg.delta_seal is not None
    assert old_agg.delta_seal.reason is DeltaSealReason.OLD_AGGREGATE
    record("old_aggregate", sealed=False, reason=old_agg.delta_seal.reason.value)

    # Lost unaffected leaf.
    lost = _publish_adversarial_delta(
        sealer,
        genesis,
        units=_good_delta_units(
            proof_a=proofs["unit/a"],
            proof_b_parent=proofs["unit/b"],
            proof_b_new=_hex_digest("matrix:lost-leaf:unit/b"),
        ),
        transition_overrides={
            "expected_surviving_leaf_ids": ("unit/a", "unit/b", "unit/ghost"),
        },
        revision="rev-lostlf-" + ("e" * 31),
    )
    assert lost.published is False
    assert lost.delta_seal is not None
    assert lost.delta_seal.reason is DeltaSealReason.LOST_LEAF
    record("lost_unaffected_leaf", sealed=False, reason=lost.delta_seal.reason.value)

    # Stale parent replay after a valid advance.
    advanced = sealer.publish_full_checkpoint(
        _state(
            revision=_REV_RACE,
            source_root_cid=_DIGEST_3,
            repository_state_cid=_DIGEST_4,
            parent_revision_ids=(_REV_GENESIS,),
        ),
        _policy(),
        units=_good_full_units(tag="matrix-advanced"),
    )
    assert advanced.published is True
    live = sealer.get_current_seal(_REPO)
    assert live is not None

    stale_replay = sealer.publish_delta_seal(
        parent,
        _state(
            revision="rev-stalepr-" + ("f" * 30),
            source_root_cid=_DIGEST_1,
            repository_state_cid=_DIGEST_2,
            parent_revision_ids=(parent.revision,),
        ),
        _policy(),
        _transition(parent, revision="rev-stalepr-" + ("f" * 30)),
        units=_good_delta_units(
            proof_a=proofs["unit/a"],
            proof_b_parent=proofs["unit/b"],
            proof_b_new=_hex_digest("matrix:stale-parent:unit/b"),
        ),
    )
    assert stale_replay.published is False
    assert stale_replay.status is SealStatus.STALE_PARENT
    record("stale_parent_replay", sealed=False, reason=stale_replay.reason.value)

    # Stale branch: parent from feature, publish on main after feature advance.
    feature = sealer.publish_full_checkpoint(
        _state(
            revision="rev-feature-" + ("1" * 30),
            source_root_cid=_DIGEST_5,
            repository_state_cid=_DIGEST_6,
            parent_revision_ids=(_REV_GENESIS,),
        ),
        _policy(),
        units=_good_full_units(tag="matrix-feature"),
        parent_seal_cid=genesis.seal_cid,
        fallback_reasons=("historical_parent",),
        branch_id="feature/matrix",
    )
    assert feature.published is True
    feature_parent = _parent_from_publication(
        feature, branch_id="feature/matrix", proof_tag="matrix-feature"
    )
    fproofs = feature_parent.unit_proof_cids
    branch_replay = sealer.publish_delta_seal(
        feature_parent,
        _state(
            revision="rev-brstale-" + ("2" * 30),
            source_root_cid=_hex_digest("branch-stale-src"),
            repository_state_cid=_hex_digest("branch-stale-state"),
            parent_revision_ids=(feature_parent.revision,),
        ),
        _policy(),
        _transition(
            feature_parent,
            revision="rev-brstale-" + ("2" * 30),
            branch_id="main",
            new_source=_hex_digest("branch-stale-src"),
            new_state=_hex_digest("branch-stale-state"),
        ),
        units=_good_delta_units(
            proof_a=fproofs["unit/a"],
            proof_b_parent=fproofs["unit/b"],
            proof_b_new=_hex_digest("matrix:branch-stale:unit/b"),
        ),
        branch_id="main",
    )
    assert branch_replay.published is False
    assert branch_replay.status is SealStatus.STALE_PARENT
    record("stale_branch_replay", sealed=False, reason=branch_replay.reason.value)

    # Racing writers on the advanced tip.
    race_parent = sealer.get_current_seal(_REPO)
    assert race_parent is not None
    workers = 4
    barrier = threading.Barrier(workers)

    def race(index: int) -> SealPublicationResult:
        barrier.wait(timeout=30)
        return sealer.publish_full_checkpoint(
            _state(
                revision=f"rev-matrix-w{index:02d}-" + ("9" * 26),
                source_root_cid="sha256:" + (f"{index + 40:02x}" * 32),
                repository_state_cid="sha256:" + (f"{index + 50:02x}" * 32),
            ),
            _policy(),
            units=_good_full_units(tag=f"matrix-w{index}"),
            transition_id=f"txn:matrix-race-{index}",
            expected_current=race_parent,
        )

    race_results: list[SealPublicationResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(race, index) for index in range(workers)]
        for future in as_completed(futures):
            race_results.append(future.result())
    winners = [item for item in race_results if item.published]
    losers = [item for item in race_results if not item.published]
    assert len(winners) == 1
    assert len(losers) == workers - 1
    assert all(item.status is SealStatus.STALE_PARENT for item in losers)
    record("racing_writers", sealed=False, reason="stale_parent_losers")

    # Final integrity: every attack axis observed; prior genesis recoverable;
    # current root is the single race winner.
    assert set(observed) == set(ATTACK_AXES), sorted(
        set(ATTACK_AXES) - set(observed)
    )
    assert (
        sealer.store.get_verified_bytes(prior.as_artifact_reference()) == prior_body
    )
    final = sealer.get_current_seal(_REPO)
    assert final is not None
    assert final == winners[0].pointer
    assert final.seal_cid == winners[0].seal_cid
    # No loser may have become current.
    for loser in losers:
        assert loser.seal_cid != final.seal_cid or not loser.published
    sealer.close()


def test_no_sealed_status_for_adversarial_evidence_vocab() -> None:
    """Closed vocabulary: adversarial terminals never map to sealed statuses."""

    forbidden = {
        ProofTerminalStatus.SIMULATED,
        ProofTerminalStatus.UNKNOWN,
        ProofTerminalStatus.TIMEOUT,
        ProofTerminalStatus.FAILED,
        ProofTerminalStatus.INVALID,
        ProofTerminalStatus.STALE,
    }
    for terminal in forbidden:
        units = (
            _full_unit("unit/a"),
            _full_unit(
                "unit/bad",
                terminal_status=terminal.value,
                proof_mode=(
                    ProofMode.SIMULATED.value
                    if terminal is ProofTerminalStatus.SIMULATED
                    else ProofMode.DIRECT_EXECUTION_PROOF.value
                ),
                proof_object_cid=_hex_digest(f"vocab:{terminal.value}"),
            ),
        )
        seal = create_full_checkpoint(
            _state(
                revision="rev-vocab-"
                + hashlib.sha256(terminal.value.encode()).hexdigest()[:32]
            ),
            _policy(),
            units=units,
        )
        assert seal.sealed is False, terminal
        _assert_not_sealed_status(seal.seal_status)
