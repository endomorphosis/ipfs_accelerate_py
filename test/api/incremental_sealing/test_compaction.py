"""IPS-042: periodic checkpoints and delta-chain compaction."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.checkpoint_policy import (
    CHECKPOINT_TRIGGERS,
    DECISION_SCHEMA,
    DEFAULT_FULL_CHECKPOINT_EVERY_N_SEALS,
    DEFAULT_MAX_DELTA_CHAIN_DEPTH,
    DEFAULT_MIN_REUSE_RATIO_BASIS_POINTS,
    EVIDENCE_SUBSET as POLICY_EVIDENCE,
    POLICY_SCHEMA,
    CheckpointDecision,
    CheckpointEvaluationState,
    CheckpointMode,
    CheckpointPolicy,
    CheckpointPolicyError,
    CheckpointTrigger,
    decide_checkpoint,
    evaluate_checkpoint_policy,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.compaction import (
    EVIDENCE_SUBSET as COMPACTION_EVIDENCE,
    GOAL_EVIDENCE_SUBSET,
    OUTCOME_SCHEMA,
    RETENTION_SCHEMA,
    CompactionError,
    CompactionOutcome,
    CompactionReason,
    RetentionPolicy,
    SealChainEntry,
    compact_seal_chain,
    entry_from_seal,
    verify_seal_chain,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.delta_seal import (
    TRANSITION_SCHEMA,
    DeltaTransitionStatement,
    DeltaUnitEvidence,
    DiffCommitmentView,
    ParentSealView,
    build_delta_seal,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
    FOREST_CATEGORIES,
    GENESIS_PARENT_SEAL,
    FullCheckpointSeal,
    RepositoryStateView,
    RequiredUnitEvidence,
    VerificationPolicyView,
    create_full_checkpoint,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.evidence import (
    ProofMode,
    ProofTerminalStatus,
    SealStatus,
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
_DIGEST_7 = "sha256:" + ("77" * 32)
_VK = "vk/prod-1"
_POLICY = _DIGEST_D
_TRUSTED = (_VK, "n/a")


def _state(**overrides: object) -> RepositoryStateView:
    payload: dict[str, object] = {
        "repository_id": "repo/accelerate",
        "revision": "rev-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "source_root_cid": _DIGEST_A,
        "repository_state_cid": _DIGEST_B,
        "environment_cid": _DIGEST_C,
        "parent_revision_ids": (),
    }
    payload.update(overrides)
    return RepositoryStateView(**payload)  # type: ignore[arg-type]


def _policy(**overrides: object) -> VerificationPolicyView:
    payload: dict[str, object] = {
        "policy_cid": _POLICY,
        "proof_schema_version": "1",
        "canonicalization_version": "1",
        "dependency_graph_schema_version": "graph@1",
        "circuit_id": "circuit@v1",
        "verification_key_id": _VK,
    }
    payload.update(overrides)
    return VerificationPolicyView(**payload)  # type: ignore[arg-type]


def _unit(unit_id: str, **overrides: object) -> RequiredUnitEvidence:
    payload: dict[str, object] = {
        "unit_id": unit_id,
        "proof_object_cid": _DIGEST_E,
        "category": "unit_test",
        "terminal_status": ProofTerminalStatus.INTEGRITY_VERIFIED.value,
        "proof_mode": ProofMode.INTEGRITY_ONLY.value,
        "required_for_seal": True,
        "freshly_verified": True,
        "cache_reused_without_fresh_verification": False,
        "circuit_id": "circuit@v1",
        "verification_key_id": _VK,
    }
    payload.update(overrides)
    return RequiredUnitEvidence(**payload)  # type: ignore[arg-type]


def _good_units() -> tuple[RequiredUnitEvidence, ...]:
    return (
        _unit("unit/a"),
        _unit(
            "unit/b",
            category="static_analysis",
            proof_object_cid=_DIGEST_F,
            terminal_status=ProofTerminalStatus.PROVED.value,
            proof_mode=ProofMode.DIRECT_EXECUTION_PROOF.value,
        ),
    )


def _full_seal(**overrides: object) -> FullCheckpointSeal:
    return create_full_checkpoint(
        _state(),
        _policy(),
        units=_good_units(),
        expected_unit_ids=("unit/a", "unit/b"),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
        **overrides,
    )


def _parent_view_from_full(seal: FullCheckpointSeal) -> ParentSealView:
    return ParentSealView(
        seal_cid=seal.seal_cid(),
        accepted=True,
        seal_status=SealStatus.SEALED_FULL.value,
        repository_id=seal.repository_id,
        branch_id="main",
        revision=seal.revision,
        source_root_cid=seal.source_root_cid,
        repository_state_cid=seal.repository_state_cid,
        environment_cid=seal.environment_cid,
        policy_cid=seal.policy_cid,
        manifest_root_cid=seal.manifest_root_cid,
        forest_root_cid=seal.repository_proof_root,
        aggregation_root=seal.aggregation_root,
        required_unit_ids=seal.required_unit_ids,
        unit_proof_cids={
            "unit/a": _DIGEST_E,
            "unit/b": _DIGEST_F,
        },
        proof_schema_version=seal.proof_schema_version,
        canonicalization_version=seal.canonicalization_version,
        dependency_graph_schema_version=seal.dependency_graph_schema_version,
        circuit_id=seal.circuit_id,
        verification_key_id=seal.verification_key_id,
        logical_epoch=1,
    )


def _delta_unit(unit_id: str, disposition: str, **overrides: object) -> DeltaUnitEvidence:
    payload: dict[str, object] = {
        "unit_id": unit_id,
        "disposition": disposition,
        "proof_object_cid": _DIGEST_4 if disposition != "remove" else "",
        "category": "unit_test",
        "terminal_status": ProofTerminalStatus.INTEGRITY_VERIFIED.value,
        "proof_mode": ProofMode.INTEGRITY_ONLY.value,
        "required_for_seal": True,
        "cache_key_complete": True,
        "cache_key_unchanged": disposition == "reuse",
        "freshly_verified": True,
        "newly_admitted": disposition in {"replace", "add"},
        "removal_authorized": disposition == "remove",
        "parent_proof_object_cid": _DIGEST_4 if disposition == "reuse" else "",
        "stale": False,
    }
    payload.update(overrides)
    return DeltaUnitEvidence(**payload)  # type: ignore[arg-type]


def _delta_from_parent(parent_seal: FullCheckpointSeal):
    parent = _parent_view_from_full(parent_seal)
    new_state = _state(
        revision="rev-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        source_root_cid=_DIGEST_1,
        repository_state_cid=_DIGEST_2,
        parent_revision_ids=(parent_seal.revision,),
    )
    transition = DeltaTransitionStatement(
        schema=TRANSITION_SCHEMA,
        parent_seal_cid=parent.seal_cid,
        branch_id="main",
        old_source_root_cid=parent.source_root_cid,
        old_repository_state_cid=parent.repository_state_cid,
        old_manifest_root_cid=parent.manifest_root_cid,
        old_forest_root_cid=parent.forest_root_cid,
        old_aggregation_root=parent.aggregation_root,
        new_source_root_cid=new_state.source_root_cid,
        new_repository_state_cid=new_state.repository_state_cid,
        new_revision=new_state.revision,
        parent_revision_ids=new_state.parent_revision_ids,
        diff=DiffCommitmentView(
            diff_algorithm="exact_artifact_set@1",
            changed_artifact_commitment=_DIGEST_6,
            complete=True,
            changed_paths=("src/mod.py",),
        ),
        expected_manifest_unit_ids=("unit/a", "unit/b"),
        expected_surviving_leaf_ids=("unit/a", "unit/b"),
        forest_rebuilt=True,
        aggregation_rebuilt=True,
        logical_epoch=2,
        transition_id="transition/1",
    )
    return build_delta_seal(
        parent,
        new_state,
        _policy(),
        transition,
        units=(
            _delta_unit(
                "unit/a",
                "reuse",
                proof_object_cid=_DIGEST_E,
                parent_proof_object_cid=_DIGEST_E,
                cache_key_unchanged=True,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_DIGEST_7,
                parent_proof_object_cid=_DIGEST_F,
                newly_admitted=True,
                cache_key_unchanged=False,
                category="static_analysis",
                terminal_status=ProofTerminalStatus.PROVED.value,
                proof_mode=ProofMode.DIRECT_EXECUTION_PROOF.value,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Checkpoint policy contracts
# ---------------------------------------------------------------------------


def test_policy_evidence_and_defaults() -> None:
    assert POLICY_EVIDENCE == "ips/checkpoint-policy@1"
    assert COMPACTION_EVIDENCE == "ips/chain-compaction@1"
    assert GOAL_EVIDENCE_SUBSET == "ips/compaction@1"
    assert POLICY_SCHEMA.endswith("checkpoint-policy@1")
    assert DECISION_SCHEMA.endswith("checkpoint-decision@1")
    assert OUTCOME_SCHEMA.endswith("compaction-outcome@1")
    assert RETENTION_SCHEMA.endswith("retention-policy@1")
    assert DEFAULT_FULL_CHECKPOINT_EVERY_N_SEALS == 50
    assert DEFAULT_MAX_DELTA_CHAIN_DEPTH == 32
    assert DEFAULT_MIN_REUSE_RATIO_BASIS_POINTS == 2500
    assert "periodic_cadence" in CHECKPOINT_TRIGGERS
    assert "release_tag" in CHECKPOINT_TRIGGERS
    assert "cache_corruption" in CHECKPOINT_TRIGGERS
    assert "excessive_delta_chain_depth" in CHECKPOINT_TRIGGERS
    assert "low_reuse_ratio" in CHECKPOINT_TRIGGERS


def test_default_policy_allows_incremental_when_no_trigger() -> None:
    decision = decide_checkpoint(
        seals_since_last_full_checkpoint=3,
        delta_chain_depth=3,
        estimated_reuse_ratio_basis_points=8000,
        has_accepted_parent=True,
    )
    assert isinstance(decision, CheckpointDecision)
    assert decision.mode is CheckpointMode.INCREMENTAL
    assert decision.require_full_checkpoint is False
    assert decision.allow_incremental is True
    assert decision.reasons == ()
    assert decision.incremental_override_honored is False
    assert decision.evidence_subset == POLICY_EVIDENCE


def test_periodic_cadence_forces_full_checkpoint() -> None:
    decision = decide_checkpoint(
        seals_since_last_full_checkpoint=50,
        delta_chain_depth=1,
        estimated_reuse_ratio_basis_points=9000,
    )
    assert decision.require_full_checkpoint is True
    assert decision.mode is CheckpointMode.FULL_CHECKPOINT
    assert CheckpointTrigger.PERIODIC_CADENCE.value in decision.reasons
    assert decision.allow_incremental is False


def test_release_tag_forces_full_checkpoint() -> None:
    decision = decide_checkpoint(
        seals_since_last_full_checkpoint=1,
        is_release_tag=True,
        estimated_reuse_ratio_basis_points=9000,
    )
    assert decision.require_full_checkpoint is True
    assert CheckpointTrigger.RELEASE_TAG.value in decision.reasons


@pytest.mark.parametrize(
    ("flag", "trigger"),
    [
        ("circuit_or_key_changed", CheckpointTrigger.CIRCUIT_OR_KEY_CHANGE),
        ("dependency_lock_changed", CheckpointTrigger.DEPENDENCY_LOCK_CHANGE),
        ("trust_policy_changed", CheckpointTrigger.TRUST_POLICY_CHANGE),
        ("schema_changed", CheckpointTrigger.SCHEMA_CHANGE),
        ("canonicalization_changed", CheckpointTrigger.CANONICALIZATION_CHANGE),
        ("environment_changed", CheckpointTrigger.ENVIRONMENT_CHANGE),
        ("cache_corruption_detected", CheckpointTrigger.CACHE_CORRUPTION),
        ("uncertain_cache_integrity", CheckpointTrigger.UNCERTAIN_CACHE_INTEGRITY),
        ("is_first_state", CheckpointTrigger.FIRST_STATE),
        ("force_full_checkpoint", CheckpointTrigger.EXPLICIT_FORCE),
        ("full_fallback_required", CheckpointTrigger.FULL_FALLBACK_REQUIRED),
    ],
)
def test_mandated_triggers_force_full(
    flag: str, trigger: CheckpointTrigger
) -> None:
    decision = decide_checkpoint(
        seals_since_last_full_checkpoint=0,
        delta_chain_depth=0,
        estimated_reuse_ratio_basis_points=9000,
        **{flag: True},
    )
    assert decision.require_full_checkpoint is True
    assert trigger.value in decision.reasons


def test_low_reuse_ratio_forces_full() -> None:
    decision = decide_checkpoint(
        seals_since_last_full_checkpoint=1,
        delta_chain_depth=1,
        estimated_reuse_ratio_basis_points=1000,
    )
    assert decision.require_full_checkpoint is True
    assert CheckpointTrigger.LOW_REUSE_RATIO.value in decision.reasons


def test_excessive_delta_chain_depth_forces_full() -> None:
    decision = decide_checkpoint(
        seals_since_last_full_checkpoint=1,
        delta_chain_depth=32,
        estimated_reuse_ratio_basis_points=9000,
    )
    assert decision.require_full_checkpoint is True
    assert CheckpointTrigger.EXCESSIVE_DELTA_CHAIN_DEPTH.value in decision.reasons


def test_missing_parent_forces_full() -> None:
    decision = decide_checkpoint(
        has_accepted_parent=False,
        seals_since_last_full_checkpoint=1,
        estimated_reuse_ratio_basis_points=9000,
    )
    assert decision.require_full_checkpoint is True
    assert CheckpointTrigger.MISSING_PARENT.value in decision.reasons


def test_incremental_caller_cannot_override_trigger() -> None:
    decision = decide_checkpoint(
        seals_since_last_full_checkpoint=50,
        prefer_incremental=True,
        estimated_reuse_ratio_basis_points=9000,
    )
    assert decision.require_full_checkpoint is True
    assert decision.allow_incremental is False
    assert decision.incremental_override_attempted is True
    assert decision.incremental_override_honored is False


def test_policy_rejects_allow_incremental_override() -> None:
    with pytest.raises(CheckpointPolicyError, match="allow_incremental_override"):
        CheckpointPolicy(allow_incremental_override=True)


def test_custom_thresholds_are_respected() -> None:
    policy = CheckpointPolicy(
        full_checkpoint_every_n_seals=5,
        max_delta_chain_depth=4,
        min_reuse_ratio_basis_points=5000,
    )
    cadence = evaluate_checkpoint_policy(
        policy,
        CheckpointEvaluationState(
            seals_since_last_full_checkpoint=5,
            estimated_reuse_ratio_basis_points=9000,
        ),
    )
    assert CheckpointTrigger.PERIODIC_CADENCE.value in cadence.reasons

    depth = evaluate_checkpoint_policy(
        policy,
        CheckpointEvaluationState(
            delta_chain_depth=4,
            estimated_reuse_ratio_basis_points=9000,
        ),
    )
    assert CheckpointTrigger.EXCESSIVE_DELTA_CHAIN_DEPTH.value in depth.reasons

    reuse = evaluate_checkpoint_policy(
        policy,
        CheckpointEvaluationState(
            estimated_reuse_ratio_basis_points=4999,
        ),
    )
    assert CheckpointTrigger.LOW_REUSE_RATIO.value in reuse.reasons

    ok = evaluate_checkpoint_policy(
        policy,
        CheckpointEvaluationState(
            seals_since_last_full_checkpoint=4,
            delta_chain_depth=3,
            estimated_reuse_ratio_basis_points=5000,
        ),
    )
    assert ok.allow_incremental is True


def test_decision_and_policy_are_content_addressed() -> None:
    policy = CheckpointPolicy.default()
    decision = decide_checkpoint(seals_since_last_full_checkpoint=1)
    assert policy.policy_cid().startswith("sha256:")
    assert decision.decision_cid().startswith("sha256:")
    assert decision.policy_cid == policy.policy_cid()
    again = CheckpointPolicy.from_canonical(policy.to_canonical())
    assert again.policy_cid() == policy.policy_cid()


# ---------------------------------------------------------------------------
# Chain compaction positive path
# ---------------------------------------------------------------------------


def test_compact_verified_chain_builds_full_checkpoint_and_retains_history() -> None:
    full = _full_seal()
    assert full.sealed is True
    delta = _delta_from_parent(full)
    assert delta.sealed is True

    units = (
        _unit("unit/a", proof_object_cid=_DIGEST_E),
        _unit(
            "unit/b",
            category="static_analysis",
            proof_object_cid=_DIGEST_7,
            terminal_status=ProofTerminalStatus.PROVED.value,
            proof_mode=ProofMode.DIRECT_EXECUTION_PROOF.value,
        ),
    )
    retention = RetentionPolicy(
        required_historical_seal_cids=(full.seal_cid(), delta.seal_cid()),
        required_evidence_cids=(_DIGEST_E, _DIGEST_7),
    )
    outcome = compact_seal_chain(
        delta,
        retention,
        _policy(),
        seal_chain=(full, delta),
        units=units,
        expected_unit_ids=("unit/a", "unit/b"),
        repository_state=_state(
            revision=delta.revision,
            source_root_cid=delta.source_root_cid,
            repository_state_cid=delta.repository_state_cid,
            parent_revision_ids=delta.parent_revision_ids,
        ),
        trusted_keys=_TRUSTED,
        verify_current_cryptographically=True,
    )
    assert isinstance(outcome, CompactionOutcome)
    assert outcome.sealed is True
    assert outcome.reason is CompactionReason.COMPACTED
    assert outcome.chain_verified is True
    assert outcome.manifest_verified is True
    assert outcome.units_verified is True
    assert outcome.forest_verified is True
    assert outcome.retention_satisfied is True
    assert outcome.seal is not None
    assert isinstance(outcome.seal, FullCheckpointSeal)
    assert outcome.seal.sealed is True
    assert outcome.seal.seal_status is SealStatus.SEALED_FULL
    assert outcome.seal.parent_seal_cid == delta.seal_cid()
    assert set(outcome.seal.required_unit_ids) == {"unit/a", "unit/b"}
    assert set(outcome.seal.category_roots) == set(FOREST_CATEGORIES)
    assert outcome.seal.repository_proof_root.startswith("sha256:")
    assert outcome.seal.manifest_root_cid.startswith("sha256:")
    assert full.seal_cid() in outcome.retained_historical_seal_cids
    assert delta.seal_cid() in outcome.retained_historical_seal_cids
    assert outcome.compacted_seal_cid in outcome.retained_historical_seal_cids
    assert _DIGEST_E in outcome.retained_evidence_cids
    assert _DIGEST_7 in outcome.retained_evidence_cids
    assert outcome.to_canonical()["history_rewritten"] is False
    assert outcome.to_canonical()["evidence_silently_deleted"] is False
    assert outcome.evidence_subset == COMPACTION_EVIDENCE
    assert outcome.goal_evidence_subset == GOAL_EVIDENCE_SUBSET


def test_compact_full_only_chain() -> None:
    full = _full_seal()
    outcome = compact_seal_chain(
        full,
        RetentionPolicy.default(),
        _policy(),
        seal_chain=(full,),
        units=_good_units(),
        expected_unit_ids=("unit/a", "unit/b"),
        trusted_keys=_TRUSTED,
    )
    assert outcome.sealed is True
    assert outcome.seal is not None
    assert outcome.seal.parent_seal_cid == full.seal_cid()
    assert full.seal_cid() in outcome.verified_chain_seal_cids


def test_entry_from_seal_and_chain_verify_helpers() -> None:
    full = _full_seal()
    entry = entry_from_seal(full)
    assert isinstance(entry, SealChainEntry)
    assert entry.seal_cid == full.seal_cid()
    assert entry.parent_seal_cid == GENESIS_PARENT_SEAL
    assert entry.accepted is True
    ok, cids, message, _details = verify_seal_chain(
        (entry,), current_seal_cid=full.seal_cid()
    )
    assert ok is True
    assert cids == (full.seal_cid(),)
    assert "verified" in message


# ---------------------------------------------------------------------------
# Rejection paths (acceptance criteria)
# ---------------------------------------------------------------------------


def test_broken_chain_rejects_rather_than_compacts() -> None:
    full = _full_seal()
    delta = _delta_from_parent(full)
    # Break the parent link on the tip entry.
    broken = entry_from_seal(delta)
    broken_payload = broken.to_canonical()
    broken_payload["parent_seal_cid"] = "sha256:" + ("00" * 32)
    broken_entry = SealChainEntry(
        seal_cid=str(broken_payload["seal_cid"]),
        parent_seal_cid=str(broken_payload["parent_seal_cid"]),
        seal_status=str(broken_payload["seal_status"]),
        seal_kind=str(broken_payload["seal_kind"]),
        accepted=True,
    )
    outcome = compact_seal_chain(
        delta,
        RetentionPolicy.default(),
        _policy(),
        seal_chain=(entry_from_seal(full), broken_entry),
        units=_good_units(),
        trusted_keys=_TRUSTED,
        verify_current_cryptographically=False,
    )
    assert outcome.sealed is False
    assert outcome.reason is CompactionReason.BROKEN_CHAIN
    assert outcome.seal is None or outcome.seal.sealed is False


def test_incomplete_history_when_current_not_tip_rejects() -> None:
    full = _full_seal()
    delta = _delta_from_parent(full)
    # Provide only the full seal as the chain while claiming delta is current.
    outcome = compact_seal_chain(
        delta,
        RetentionPolicy.default(),
        _policy(),
        seal_chain=(full,),  # will append delta; this should still link
        units=_good_units(),
        trusted_keys=_TRUSTED,
        verify_current_cryptographically=False,
    )
    # Appending current should succeed when parent matches.
    assert outcome.sealed is True

    # Explicit wrong tip: chain ends at full but current is a different seal cid.
    orphan = SealChainEntry(
        seal_cid="sha256:" + ("ab" * 32),
        parent_seal_cid=full.seal_cid(),
        seal_status=SealStatus.SEALED_INCREMENTAL.value,
        seal_kind="delta_seal",
        accepted=True,
    )
    outcome2 = compact_seal_chain(
        delta,
        RetentionPolicy.default(),
        _policy(),
        seal_chain=(entry_from_seal(full), orphan),
        units=_good_units(),
        trusted_keys=_TRUSTED,
        verify_current_cryptographically=False,
    )
    assert outcome2.sealed is False
    assert outcome2.reason in {
        CompactionReason.INCOMPLETE_HISTORY,
        CompactionReason.BROKEN_CHAIN,
    }


def test_required_evidence_loss_rejects() -> None:
    full = _full_seal()
    missing_evidence = "sha256:" + ("de" * 32)
    retention = RetentionPolicy(
        required_historical_seal_cids=(full.seal_cid(),),
        required_evidence_cids=(missing_evidence,),
        retain_entire_verified_chain=True,
    )
    outcome = compact_seal_chain(
        full,
        retention,
        _policy(),
        seal_chain=(full,),
        units=_good_units(),
        trusted_keys=_TRUSTED,
        available_evidence_cids=(),  # missing required evidence not supplied
        verify_current_cryptographically=False,
    )
    assert outcome.sealed is False
    assert outcome.reason is CompactionReason.REQUIRED_EVIDENCE_LOST
    assert missing_evidence in outcome.missing_required_references


def test_required_historical_seal_missing_rejects() -> None:
    full = _full_seal()
    ghost = "sha256:" + ("cd" * 32)
    retention = RetentionPolicy(
        required_historical_seal_cids=(ghost,),
        retain_entire_verified_chain=True,
    )
    outcome = compact_seal_chain(
        full,
        retention,
        _policy(),
        seal_chain=(full,),
        units=_good_units(),
        trusted_keys=_TRUSTED,
        available_evidence_cids=(),
        verify_current_cryptographically=False,
    )
    assert outcome.sealed is False
    assert outcome.reason is CompactionReason.RETENTION_REFERENCE_MISSING
    assert ghost in outcome.missing_required_references


def test_unverified_unit_rejects_compaction() -> None:
    full = _full_seal()
    units = (
        _unit("unit/a", freshly_verified=False),
        _unit(
            "unit/b",
            category="static_analysis",
            proof_object_cid=_DIGEST_F,
            terminal_status=ProofTerminalStatus.PROVED.value,
            proof_mode=ProofMode.DIRECT_EXECUTION_PROOF.value,
        ),
    )
    outcome = compact_seal_chain(
        full,
        RetentionPolicy.default(),
        _policy(),
        seal_chain=(full,),
        units=units,
        expected_unit_ids=("unit/a", "unit/b"),
        trusted_keys=_TRUSTED,
        verify_current_cryptographically=False,
    )
    assert outcome.sealed is False
    assert outcome.reason is CompactionReason.UNIT_VERIFICATION_FAILED


def test_incomplete_manifest_rejects() -> None:
    full = _full_seal()
    outcome = compact_seal_chain(
        full,
        RetentionPolicy.default(),
        _policy(),
        seal_chain=(full,),
        units=(_unit("unit/a"),),
        expected_unit_ids=("unit/a", "unit/b"),
        trusted_keys=_TRUSTED,
        verify_current_cryptographically=False,
    )
    assert outcome.sealed is False
    assert outcome.reason is CompactionReason.MANIFEST_INCOMPLETE


def test_non_accepted_current_seal_rejects() -> None:
    full = create_full_checkpoint(
        _state(),
        _policy(),
        units=(
            _unit(
                "unit/sim",
                proof_mode=ProofMode.SIMULATED.value,
                terminal_status=ProofTerminalStatus.SIMULATED.value,
            ),
        ),
    )
    assert full.sealed is False
    outcome = compact_seal_chain(
        full,
        RetentionPolicy.default(),
        _policy(),
        seal_chain=(full,),
        units=_good_units(),
        trusted_keys=_TRUSTED,
        verify_current_cryptographically=False,
    )
    assert outcome.sealed is False
    assert outcome.reason is CompactionReason.CURRENT_SEAL_NOT_ACCEPTED


def test_empty_units_rejects() -> None:
    full = _full_seal()
    outcome = compact_seal_chain(
        full,
        RetentionPolicy.default(),
        _policy(),
        seal_chain=(full,),
        units=(),
        trusted_keys=_TRUSTED,
        verify_current_cryptographically=False,
    )
    assert outcome.sealed is False
    assert outcome.reason is CompactionReason.MANIFEST_INCOMPLETE


def test_unaccepted_mid_chain_rejects() -> None:
    full = _full_seal()
    bad = SealChainEntry(
        seal_cid="sha256:" + ("ef" * 32),
        parent_seal_cid=full.seal_cid(),
        seal_status=SealStatus.VERIFICATION_FAILED.value,
        seal_kind="delta_seal",
        accepted=False,
    )
    tip = SealChainEntry(
        seal_cid="sha256:" + ("fa" * 32),
        parent_seal_cid=bad.seal_cid,
        seal_status=SealStatus.SEALED_INCREMENTAL.value,
        seal_kind="delta_seal",
        accepted=True,
    )
    # Current must match tip for tip check; use tip as current mapping.
    current = {
        "seal_cid": tip.seal_cid,
        "parent_seal_cid": tip.parent_seal_cid,
        "seal_status": tip.seal_status,
        "sealed": True,
        "repository_id": "repo/accelerate",
        "revision": "rev-tip",
        "source_root_cid": _DIGEST_1,
        "repository_state_cid": _DIGEST_2,
        "environment_cid": _DIGEST_C,
        "policy_cid": _POLICY,
        "required_unit_ids": ("unit/a", "unit/b"),
        "verification_key_id": _VK,
        "proof_schema_version": "1",
        "canonicalization_version": "1",
        "dependency_graph_schema_version": "graph@1",
        "circuit_id": "circuit@v1",
        "parent_revision_ids": (),
    }
    outcome = compact_seal_chain(
        current,
        RetentionPolicy.default(),
        _policy(),
        seal_chain=(entry_from_seal(full), bad, tip),
        units=_good_units(),
        trusted_keys=_TRUSTED,
        verify_current_cryptographically=False,
    )
    assert outcome.sealed is False
    assert outcome.reason is CompactionReason.BROKEN_CHAIN


def test_compaction_outcome_rejects_inconsistent_sealed_state() -> None:
    with pytest.raises(CompactionError):
        CompactionOutcome(
            schema=OUTCOME_SCHEMA,
            evidence_subset=COMPACTION_EVIDENCE,
            goal_evidence_subset=GOAL_EVIDENCE_SUBSET,
            sealed=True,
            reason=CompactionReason.COMPACTED,
            seal=None,
            current_seal_cid="",
            compacted_seal_cid="",
            parent_of_compacted="",
            chain_verified=True,
            manifest_verified=True,
            units_verified=True,
            forest_verified=True,
            retention_satisfied=True,
            verified_chain_seal_cids=(),
            retained_historical_seal_cids=(),
            retained_evidence_cids=(),
            missing_required_references=(),
            message="bad",
        )
