"""IPS-039: parent-bound delta seals with all fourteen transition invariants."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.delta_seal import (
    EVIDENCE_SUBSET,
    FOURTEEN_INVARIANTS_EVIDENCE,
    NORMATIVE_INVARIANTS,
    SEAL_SCHEMA,
    TRANSITION_SCHEMA,
    UNIT_DISPOSITIONS,
    DeltaSeal,
    DeltaSealBuilder,
    DeltaSealError,
    DeltaSealReason,
    DeltaTransitionStatement,
    DeltaUnitEvidence,
    DiffCommitmentView,
    ParentSealView,
    UnitDisposition,
    build_delta_seal,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
    FOREST_CATEGORIES,
    RepositoryStateView,
    VerificationPolicyView,
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
_PARENT_SEAL = "sha256:" + ("99" * 32)
_OLD_SOURCE = _DIGEST_A
_OLD_STATE = _DIGEST_B
_NEW_SOURCE = _DIGEST_1
_NEW_STATE = _DIGEST_2
_ENV = _DIGEST_C
_POLICY = _DIGEST_D
_OLD_MANIFEST = _DIGEST_E
_OLD_FOREST = _DIGEST_F
_OLD_AGG = _DIGEST_3
_PROOF_A = _DIGEST_4
_PROOF_B = _DIGEST_5
_PROOF_B_NEW = _DIGEST_6
_PROOF_C = "sha256:" + ("77" * 32)
_DIFF_COMMIT = "sha256:" + ("88" * 32)


def _parent(**overrides: object) -> ParentSealView:
    payload: dict[str, object] = {
        "seal_cid": _PARENT_SEAL,
        "accepted": True,
        "seal_status": SealStatus.SEALED_FULL.value,
        "repository_id": "repo/accelerate",
        "branch_id": "main",
        "revision": "rev-old",
        "source_root_cid": _OLD_SOURCE,
        "repository_state_cid": _OLD_STATE,
        "environment_cid": _ENV,
        "policy_cid": _POLICY,
        "manifest_root_cid": _OLD_MANIFEST,
        "forest_root_cid": _OLD_FOREST,
        "aggregation_root": _OLD_AGG,
        "required_unit_ids": ("unit/a", "unit/b"),
        "unit_proof_cids": {"unit/a": _PROOF_A, "unit/b": _PROOF_B},
        "parent_revision_ids": (),
        "logical_epoch": 1,
    }
    payload.update(overrides)
    return ParentSealView(**payload)  # type: ignore[arg-type]


def _state(**overrides: object) -> RepositoryStateView:
    payload: dict[str, object] = {
        "repository_id": "repo/accelerate",
        "revision": "rev-new",
        "source_root_cid": _NEW_SOURCE,
        "repository_state_cid": _NEW_STATE,
        "environment_cid": _ENV,
        "parent_revision_ids": ("rev-old",),
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
        "verification_key_id": "vk/1",
    }
    payload.update(overrides)
    return VerificationPolicyView(**payload)  # type: ignore[arg-type]


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


def _transition(**overrides: object) -> DeltaTransitionStatement:
    payload: dict[str, object] = {
        "schema": TRANSITION_SCHEMA,
        "parent_seal_cid": _PARENT_SEAL,
        "branch_id": "main",
        "old_source_root_cid": _OLD_SOURCE,
        "old_repository_state_cid": _OLD_STATE,
        "old_manifest_root_cid": _OLD_MANIFEST,
        "old_forest_root_cid": _OLD_FOREST,
        "old_aggregation_root": _OLD_AGG,
        "new_source_root_cid": _NEW_SOURCE,
        "new_repository_state_cid": _NEW_STATE,
        "new_revision": "rev-new",
        "parent_revision_ids": ("rev-old",),
        "diff": _diff(),
        "expected_manifest_unit_ids": ("unit/a", "unit/b"),
        "expected_surviving_leaf_ids": ("unit/a", "unit/b"),
        "forest_rebuilt": True,
        "aggregation_rebuilt": True,
        "logical_epoch": 2,
    }
    payload.update(overrides)
    return DeltaTransitionStatement(**payload)  # type: ignore[arg-type]


def _unit(unit_id: str, disposition: str, **overrides: object) -> DeltaUnitEvidence:
    payload: dict[str, object] = {
        "unit_id": unit_id,
        "disposition": disposition,
        "proof_object_cid": _PROOF_A if unit_id == "unit/a" else _PROOF_B_NEW,
        "category": "unit_test",
        "terminal_status": ProofTerminalStatus.INTEGRITY_VERIFIED.value,
        "proof_mode": ProofMode.INTEGRITY_ONLY.value,
        "required_for_seal": True,
        "cache_key_complete": True,
        "cache_key_unchanged": True,
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


def _good_units() -> tuple[DeltaUnitEvidence, ...]:
    """Reuse unit/a; replace unit/b with a newly admitted proof."""

    return (
        _unit(
            "unit/a",
            "reuse",
            proof_object_cid=_PROOF_A,
            parent_proof_object_cid=_PROOF_A,
        ),
        _unit(
            "unit/b",
            "replace",
            proof_object_cid=_PROOF_B_NEW,
            parent_proof_object_cid=_PROOF_B,
            newly_admitted=True,
            terminal_status=ProofTerminalStatus.PROVED.value,
            proof_mode=ProofMode.DIRECT_EXECUTION_PROOF.value,
        ),
    )


def _seal(**kwargs: object) -> DeltaSeal:
    return build_delta_seal(
        kwargs.pop("parent", _parent()),  # type: ignore[arg-type]
        kwargs.pop("state", _state()),  # type: ignore[arg-type]
        kwargs.pop("policy", _policy()),  # type: ignore[arg-type]
        kwargs.pop("transition", _transition()),  # type: ignore[arg-type]
        units=kwargs.pop("units", _good_units()),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Schema / closed vocabulary
# ---------------------------------------------------------------------------


def test_evidence_subset_schema_and_fourteen_invariants() -> None:
    assert EVIDENCE_SUBSET == "ips/delta-seal@1"
    assert FOURTEEN_INVARIANTS_EVIDENCE == "ips/delta-fourteen-invariants@1"
    assert SEAL_SCHEMA.endswith("delta-seal@1")
    assert TRANSITION_SCHEMA.endswith("delta-transition-statement@1")
    assert len(NORMATIVE_INVARIANTS) == 14
    assert NORMATIVE_INVARIANTS[0] == "parent_accepted"
    assert NORMATIVE_INVARIANTS[13] == "anti_replay_binding"
    assert set(UNIT_DISPOSITIONS) == {"reuse", "replace", "add", "remove"}
    assert "unit_test" in FOREST_CATEGORIES


def test_happy_path_seals_incremental_and_passes_all_fourteen() -> None:
    seal = _seal()
    assert isinstance(seal, DeltaSeal)
    assert seal.sealed is True
    assert seal.seal_status is SealStatus.SEALED_INCREMENTAL
    assert seal.reason is DeltaSealReason.SEALED
    assert seal.all_invariants_passed() is True
    assert seal.invariants_failed == ()
    assert set(seal.invariants_passed) == set(NORMATIVE_INVARIANTS)
    assert seal.parent_seal_cid == _PARENT_SEAL
    assert seal.branch_id == "main"
    assert seal.revision == "rev-new"
    assert seal.reused_unit_ids == ("unit/a",)
    assert seal.replaced_unit_ids == ("unit/b",)
    assert seal.added_unit_ids == ()
    assert seal.removed_unit_ids == ()
    assert seal.required_unit_ids == ("unit/a", "unit/b")
    assert seal.verified_unit_ids == ("unit/a", "unit/b")
    assert seal.rejected_unit_ids == ()
    assert seal.new_manifest_root_cid.startswith("sha256:")
    assert seal.new_forest_root_cid.startswith("sha256:")
    assert seal.new_aggregation_root.startswith("sha256:")
    assert seal.new_forest_root_cid != _OLD_FOREST
    assert seal.new_aggregation_root != _OLD_AGG
    assert set(seal.category_roots) == set(FOREST_CATEGORIES)
    assert seal.diff_algorithm
    assert seal.changed_artifact_commitment == _DIFF_COMMIT
    assert seal.logical_epoch == 2
    assert seal.transition_id.startswith("sha256:")
    assert seal.evidence_subset == EVIDENCE_SUBSET
    assert seal.fourteen_invariants_evidence == FOURTEEN_INVARIANTS_EVIDENCE


def test_builder_and_mapping_inputs_are_deterministic() -> None:
    first = _seal()
    second = DeltaSealBuilder().build(
        _parent().to_canonical(),
        {
            "repository_id": "repo/accelerate",
            "revision": "rev-new",
            "source_root_cid": _NEW_SOURCE,
            "repository_state_cid": _NEW_STATE,
            "environment_cid": _ENV,
            "parent_revision_ids": ("rev-old",),
        },
        {
            "policy_cid": _POLICY,
            "circuit_id": "circuit@v1",
            "verification_key_id": "vk/1",
        },
        {
            "schema": TRANSITION_SCHEMA,
            "parent_seal_cid": _PARENT_SEAL,
            "branch_id": "main",
            "old_source_root_cid": _OLD_SOURCE,
            "old_repository_state_cid": _OLD_STATE,
            "old_manifest_root_cid": _OLD_MANIFEST,
            "old_forest_root_cid": _OLD_FOREST,
            "old_aggregation_root": _OLD_AGG,
            "new_source_root_cid": _NEW_SOURCE,
            "new_repository_state_cid": _NEW_STATE,
            "new_revision": "rev-new",
            "parent_revision_ids": ("rev-old",),
            "diff": _diff().to_canonical(),
            "expected_manifest_unit_ids": ("unit/a", "unit/b"),
            "expected_surviving_leaf_ids": ("unit/a", "unit/b"),
            "forest_rebuilt": True,
            "aggregation_rebuilt": True,
            "logical_epoch": 2,
        },
        units=[unit.to_canonical() for unit in _good_units()],
    )
    assert first.to_canonical() == second.to_canonical()
    assert first.seal_cid() == second.seal_cid()
    assert first.seal_cid().startswith("sha256:")


# ---------------------------------------------------------------------------
# Independent invariant tests (1–14)
# ---------------------------------------------------------------------------


def test_invariant_1_parent_not_accepted_rejects() -> None:
    seal = _seal(parent=_parent(accepted=False))
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.PARENT_NOT_ACCEPTED
    assert "parent_accepted" in seal.invariants_failed
    assert seal.seal_status is SealStatus.STALE_PARENT


def test_invariant_1_parent_non_sealed_status_rejects() -> None:
    seal = _seal(parent=_parent(seal_status=SealStatus.VERIFICATION_FAILED.value))
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.PARENT_NOT_ACCEPTED
    assert "parent_accepted" in seal.invariants_failed


def test_invariant_2_old_root_mismatch_rejects() -> None:
    seal = _seal(
        transition=_transition(old_source_root_cid=_DIGEST_6),
    )
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.OLD_ROOT_MISMATCH
    assert "old_root_matches_parent" in seal.invariants_failed


def test_invariant_3_new_root_mismatch_rejects() -> None:
    seal = _seal(
        transition=_transition(new_source_root_cid=_DIGEST_6),
    )
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.NEW_ROOT_MISMATCH
    assert "new_root_matches_source" in seal.invariants_failed


def test_invariant_4_incomplete_diff_rejects() -> None:
    seal = _seal(
        transition=_transition(diff=_diff(complete=False)),
    )
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.INCOMPLETE_DIFF
    assert "complete_diff" in seal.invariants_failed

    empty_algo = _seal(
        transition=_transition(diff=_diff(diff_algorithm="", complete=True)),
    )
    assert empty_algo.sealed is False
    assert empty_algo.reason is DeltaSealReason.INCOMPLETE_DIFF


def test_invariant_5_missing_replacement_rejects() -> None:
    units = (
        _unit("unit/a", "reuse", proof_object_cid=_PROOF_A),
        _unit(
            "unit/b",
            "replace",
            proof_object_cid=_PROOF_B,  # same as parent → not a replacement
            newly_admitted=True,
            parent_proof_object_cid=_PROOF_B,
        ),
    )
    seal = _seal(units=units)
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.MISSING_REPLACEMENT
    assert "invalidated_have_new_proofs" in seal.invariants_failed
    assert "unit/b" in seal.rejected_unit_ids


def test_invariant_5_unadmitted_replacement_rejects() -> None:
    units = (
        _unit("unit/a", "reuse", proof_object_cid=_PROOF_A),
        _unit(
            "unit/b",
            "replace",
            proof_object_cid=_PROOF_B_NEW,
            newly_admitted=False,
        ),
    )
    seal = _seal(units=units)
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.MISSING_REPLACEMENT
    assert "invalidated_have_new_proofs" in seal.invariants_failed


def test_invariant_6_incomplete_cache_key_on_reuse_rejects() -> None:
    units = (
        _unit(
            "unit/a",
            "reuse",
            proof_object_cid=_PROOF_A,
            cache_key_complete=False,
        ),
        _unit("unit/b", "replace", proof_object_cid=_PROOF_B_NEW, newly_admitted=True),
    )
    seal = _seal(units=units)
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.INCOMPLETE_CACHE_KEY
    assert "reuse_complete_cache_key" in seal.invariants_failed


def test_invariant_6_and_11_stale_reuse_rejects() -> None:
    units = (
        _unit(
            "unit/a",
            "reuse",
            proof_object_cid=_PROOF_A,
            stale=True,
        ),
        _unit("unit/b", "replace", proof_object_cid=_PROOF_B_NEW, newly_admitted=True),
    )
    seal = _seal(units=units)
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.STALE_REUSE
    assert "no_stale_reuse" in seal.invariants_failed


def test_invariant_6_changed_cache_key_on_reuse_rejects() -> None:
    units = (
        _unit(
            "unit/a",
            "reuse",
            proof_object_cid=_PROOF_A,
            cache_key_unchanged=False,
        ),
        _unit("unit/b", "replace", proof_object_cid=_PROOF_B_NEW, newly_admitted=True),
    )
    seal = _seal(units=units)
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.STALE_REUSE
    assert "reuse_complete_cache_key" in seal.invariants_failed


def test_invariant_7_unauthorized_deletion_rejects() -> None:
    units = (
        _unit("unit/a", "reuse", proof_object_cid=_PROOF_A),
        _unit(
            "unit/b",
            "remove",
            removal_authorized=False,
            proof_object_cid=_PROOF_B,
        ),
    )
    seal = _seal(
        units=units,
        transition=_transition(
            expected_manifest_unit_ids=("unit/a",),
            expected_surviving_leaf_ids=("unit/a",),
        ),
    )
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.UNAUTHORIZED_DELETION
    assert "deletions_authorized" in seal.invariants_failed
    assert "unit/b" in seal.rejected_unit_ids


def test_invariant_7_authorized_deletion_seals() -> None:
    units = (
        _unit("unit/a", "reuse", proof_object_cid=_PROOF_A),
        _unit(
            "unit/b",
            "remove",
            removal_authorized=True,
            proof_object_cid=_PROOF_B,
        ),
    )
    seal = _seal(
        units=units,
        transition=_transition(
            expected_manifest_unit_ids=("unit/a",),
            expected_surviving_leaf_ids=("unit/a",),
        ),
    )
    assert seal.sealed is True
    assert seal.removed_unit_ids == ("unit/b",)
    assert seal.required_unit_ids == ("unit/a",)
    assert "deletions_authorized" in seal.invariants_passed


def test_invariant_8_missing_addition_rejects() -> None:
    units = (
        _unit("unit/a", "reuse", proof_object_cid=_PROOF_A),
        _unit("unit/b", "replace", proof_object_cid=_PROOF_B_NEW, newly_admitted=True),
        _unit(
            "unit/c",
            "add",
            proof_object_cid=_PROOF_C,
            newly_admitted=False,
            parent_proof_object_cid="",
        ),
    )
    seal = _seal(
        units=units,
        transition=_transition(
            expected_manifest_unit_ids=("unit/a", "unit/b", "unit/c"),
            expected_surviving_leaf_ids=("unit/a", "unit/b"),
        ),
    )
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.MISSING_ADDITION
    assert "additions_present_and_proven" in seal.invariants_failed


def test_invariant_8_proven_addition_seals() -> None:
    units = (
        _unit("unit/a", "reuse", proof_object_cid=_PROOF_A),
        _unit("unit/b", "replace", proof_object_cid=_PROOF_B_NEW, newly_admitted=True),
        _unit(
            "unit/c",
            "add",
            proof_object_cid=_PROOF_C,
            newly_admitted=True,
            parent_proof_object_cid="",
            category="static_analysis",
        ),
    )
    seal = _seal(
        units=units,
        transition=_transition(
            expected_manifest_unit_ids=("unit/a", "unit/b", "unit/c"),
            expected_surviving_leaf_ids=("unit/a", "unit/b"),
        ),
    )
    assert seal.sealed is True
    assert seal.added_unit_ids == ("unit/c",)
    assert "additions_present_and_proven" in seal.invariants_passed


def test_invariant_9_incomplete_manifest_rejects() -> None:
    seal = _seal(
        transition=_transition(
            expected_manifest_unit_ids=("unit/a", "unit/b", "unit/missing"),
        ),
    )
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.INCOMPLETE_MANIFEST
    assert "manifest_complete" in seal.invariants_failed
    assert "unit/missing" in seal.rejected_unit_ids


def test_invariant_9_silent_parent_unit_loss_rejects() -> None:
    # unit/b from parent is neither replaced, reused, nor removed.
    units = (_unit("unit/a", "reuse", proof_object_cid=_PROOF_A),)
    seal = _seal(
        units=units,
        transition=_transition(
            expected_manifest_unit_ids=("unit/a",),
            expected_surviving_leaf_ids=("unit/a", "unit/b"),
        ),
    )
    assert seal.sealed is False
    assert seal.reason in {
        DeltaSealReason.INCOMPLETE_MANIFEST,
        DeltaSealReason.LOST_LEAF,
    }
    assert "manifest_complete" in seal.invariants_failed or "forest_commits_exact_units" in seal.invariants_failed


def test_invariant_10_old_aggregate_rejects() -> None:
    seal = _seal(transition=_transition(aggregation_rebuilt=False))
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.OLD_AGGREGATE
    assert "forest_commits_exact_units" in seal.invariants_failed


def test_invariant_10_forest_not_rebuilt_rejects() -> None:
    seal = _seal(transition=_transition(forest_rebuilt=False))
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.FOREST_MISMATCH
    assert "forest_commits_exact_units" in seal.invariants_failed


def test_invariant_10_lost_leaf_rejects() -> None:
    units = (
        _unit("unit/a", "reuse", proof_object_cid=_PROOF_A),
        _unit("unit/b", "replace", proof_object_cid=_PROOF_B_NEW, newly_admitted=True),
    )
    seal = _seal(
        units=units,
        transition=_transition(
            expected_manifest_unit_ids=("unit/a", "unit/b"),
            # Claims unit/ghost survived but it is absent from the unit set.
            expected_surviving_leaf_ids=("unit/a", "unit/b", "unit/ghost"),
        ),
    )
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.LOST_LEAF
    assert "forest_commits_exact_units" in seal.invariants_failed
    assert "unit/ghost" in seal.rejected_unit_ids


def test_invariant_11_mismatched_reuse_proof_rejects() -> None:
    units = (
        _unit(
            "unit/a",
            "reuse",
            proof_object_cid=_PROOF_C,  # not the parent proof
            parent_proof_object_cid=_PROOF_A,
        ),
        _unit("unit/b", "replace", proof_object_cid=_PROOF_B_NEW, newly_admitted=True),
    )
    seal = _seal(units=units)
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.STALE_REUSE
    assert "no_stale_reuse" in seal.invariants_failed


def test_invariant_12_simulated_required_unit_rejects() -> None:
    units = (
        _unit("unit/a", "reuse", proof_object_cid=_PROOF_A),
        _unit(
            "unit/b",
            "replace",
            proof_object_cid=_PROOF_B_NEW,
            newly_admitted=True,
            proof_mode=ProofMode.SIMULATED.value,
            terminal_status=ProofTerminalStatus.SIMULATED.value,
        ),
    )
    seal = _seal(units=units)
    assert seal.sealed is False
    assert seal.seal_status is SealStatus.SIMULATED_ONLY
    assert seal.reason is DeltaSealReason.SIMULATED_REQUIRED_UNIT
    assert "no_blocking_units" in seal.invariants_failed
    assert seal.seal_status is not SealStatus.SEALED_INCREMENTAL


@pytest.mark.parametrize(
    ("status", "expected_seal", "expected_reason"),
    [
        (
            ProofTerminalStatus.UNKNOWN.value,
            SealStatus.UNKNOWN,
            DeltaSealReason.UNKNOWN_REQUIRED_UNIT,
        ),
        (
            ProofTerminalStatus.UNAVAILABLE.value,
            SealStatus.UNAVAILABLE,
            DeltaSealReason.UNAVAILABLE_REQUIRED_UNIT,
        ),
        (
            ProofTerminalStatus.FAILED.value,
            SealStatus.PROOF_FAILED,
            DeltaSealReason.FAILED_REQUIRED_UNIT,
        ),
        (
            ProofTerminalStatus.TIMEOUT.value,
            SealStatus.TIMEOUT,
            DeltaSealReason.TIMEOUT_REQUIRED_UNIT,
        ),
        (
            ProofTerminalStatus.CANCELLED.value,
            SealStatus.CANCELLED,
            DeltaSealReason.CANCELLED_REQUIRED_UNIT,
        ),
        (
            ProofTerminalStatus.INVALID.value,
            SealStatus.PROOF_FAILED,
            DeltaSealReason.FAILED_REQUIRED_UNIT,
        ),
    ],
)
def test_invariant_12_non_pass_evidence_rejects(
    status: str,
    expected_seal: SealStatus,
    expected_reason: DeltaSealReason,
) -> None:
    units = (
        _unit("unit/a", "reuse", proof_object_cid=_PROOF_A),
        _unit(
            "unit/b",
            "replace",
            proof_object_cid=_PROOF_B_NEW,
            newly_admitted=True,
            terminal_status=status,
        ),
    )
    seal = _seal(units=units)
    assert seal.sealed is False
    assert seal.seal_status is expected_seal
    assert seal.reason is expected_reason
    assert "no_blocking_units" in seal.invariants_failed


def test_invariant_13_wrong_parent_rejects() -> None:
    seal = _seal(
        transition=_transition(parent_seal_cid="sha256:" + ("00" * 32)),
    )
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.WRONG_PARENT
    assert "exact_parent_bound" in seal.invariants_failed


def test_invariant_14_wrong_branch_rejects() -> None:
    seal = _seal(transition=_transition(branch_id="feature/other"))
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.WRONG_BRANCH
    assert "anti_replay_binding" in seal.invariants_failed


def test_invariant_14_replay_with_stale_epoch_rejects() -> None:
    # Source root may repeat after a rollback, but the transition must advance
    # the logical epoch relative to the accepted parent — no seal replay.
    seal = _seal(transition=_transition(logical_epoch=1))  # parent epoch is 1
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.REPLAY_REJECTED
    assert "anti_replay_binding" in seal.invariants_failed


def test_invariant_14_environment_change_rejects_delta() -> None:
    seal = _seal(state=_state(environment_cid=_DIGEST_6))
    assert seal.sealed is False
    assert seal.reason is DeltaSealReason.REPLAY_REJECTED
    assert "anti_replay_binding" in seal.invariants_failed


# ---------------------------------------------------------------------------
# Composite acceptance scenarios from the task statement
# ---------------------------------------------------------------------------


def test_wrong_parent_and_branch_both_reject_independently() -> None:
    wrong_parent = _seal(
        transition=_transition(parent_seal_cid="sha256:" + ("ab" * 32)),
    )
    wrong_branch = _seal(transition=_transition(branch_id="release/1"))
    assert wrong_parent.reason is DeltaSealReason.WRONG_PARENT
    assert wrong_branch.reason is DeltaSealReason.WRONG_BRANCH
    assert wrong_parent.sealed is False
    assert wrong_branch.sealed is False


def test_each_normative_invariant_is_independently_observable() -> None:
    """Every invariant appears in invariants_passed on the happy path and can fail alone."""

    good = _seal()
    assert set(good.invariants_passed) == set(NORMATIVE_INVARIANTS)

    single_failures: dict[str, DeltaSeal] = {
        "parent_accepted": _seal(parent=_parent(accepted=False)),
        "old_root_matches_parent": _seal(
            transition=_transition(old_forest_root_cid=_DIGEST_6)
        ),
        "new_root_matches_source": _seal(
            transition=_transition(new_repository_state_cid=_DIGEST_6)
        ),
        "complete_diff": _seal(transition=_transition(diff=_diff(complete=False))),
        "invalidated_have_new_proofs": _seal(
            units=(
                _unit("unit/a", "reuse", proof_object_cid=_PROOF_A),
                _unit(
                    "unit/b",
                    "replace",
                    proof_object_cid=_PROOF_B,
                    newly_admitted=True,
                ),
            )
        ),
        "reuse_complete_cache_key": _seal(
            units=(
                _unit(
                    "unit/a",
                    "reuse",
                    proof_object_cid=_PROOF_A,
                    cache_key_complete=False,
                ),
                _unit(
                    "unit/b",
                    "replace",
                    proof_object_cid=_PROOF_B_NEW,
                    newly_admitted=True,
                ),
            )
        ),
        "deletions_authorized": _seal(
            units=(
                _unit("unit/a", "reuse", proof_object_cid=_PROOF_A),
                _unit("unit/b", "remove", removal_authorized=False),
            ),
            transition=_transition(
                expected_manifest_unit_ids=("unit/a",),
                expected_surviving_leaf_ids=("unit/a",),
            ),
        ),
        "additions_present_and_proven": _seal(
            units=(
                _unit("unit/a", "reuse", proof_object_cid=_PROOF_A),
                _unit(
                    "unit/b",
                    "replace",
                    proof_object_cid=_PROOF_B_NEW,
                    newly_admitted=True,
                ),
                _unit(
                    "unit/c",
                    "add",
                    proof_object_cid="",
                    newly_admitted=False,
                    parent_proof_object_cid="",
                ),
            ),
            transition=_transition(
                expected_manifest_unit_ids=("unit/a", "unit/b", "unit/c"),
                expected_surviving_leaf_ids=("unit/a", "unit/b"),
            ),
        ),
        "manifest_complete": _seal(
            transition=_transition(
                expected_manifest_unit_ids=("unit/a", "unit/b", "unit/x"),
            )
        ),
        "forest_commits_exact_units": _seal(
            transition=_transition(aggregation_rebuilt=False)
        ),
        "no_stale_reuse": _seal(
            units=(
                _unit("unit/a", "reuse", proof_object_cid=_PROOF_A, stale=True),
                _unit(
                    "unit/b",
                    "replace",
                    proof_object_cid=_PROOF_B_NEW,
                    newly_admitted=True,
                ),
            )
        ),
        "no_blocking_units": _seal(
            units=(
                _unit("unit/a", "reuse", proof_object_cid=_PROOF_A),
                _unit(
                    "unit/b",
                    "replace",
                    proof_object_cid=_PROOF_B_NEW,
                    newly_admitted=True,
                    terminal_status=ProofTerminalStatus.FAILED.value,
                ),
            )
        ),
        "exact_parent_bound": _seal(
            transition=_transition(parent_seal_cid="sha256:" + ("cd" * 32))
        ),
        "anti_replay_binding": _seal(transition=_transition(branch_id="other")),
    }

    assert set(single_failures) == set(NORMATIVE_INVARIANTS)
    for name, seal in single_failures.items():
        assert seal.sealed is False, name
        assert name in seal.invariants_failed, (
            f"{name} should fail; failed={seal.invariants_failed} reason={seal.reason}"
        )


def test_sealed_incremental_invariant_rejects_partial_invariants() -> None:
    with pytest.raises(DeltaSealError, match="fourteen"):
        DeltaSeal(
            schema=SEAL_SCHEMA,
            evidence_subset=EVIDENCE_SUBSET,
            fourteen_invariants_evidence=FOURTEEN_INVARIANTS_EVIDENCE,
            seal_status=SealStatus.SEALED_INCREMENTAL,
            reason=DeltaSealReason.SEALED,
            repository_id="repo/x",
            branch_id="main",
            revision="rev",
            source_root_cid=_DIGEST_A,
            repository_state_cid=_DIGEST_B,
            environment_cid=_DIGEST_C,
            policy_cid=_DIGEST_D,
            proof_schema_version="1",
            canonicalization_version="1",
            dependency_graph_schema_version="graph@1",
            circuit_id="c",
            verification_key_id="vk",
            parent_seal_cid=_PARENT_SEAL,
            parent_revision_ids=(),
            logical_epoch=2,
            transition_id="tid",
            diff_algorithm="algo",
            changed_artifact_commitment=_DIFF_COMMIT,
            old_source_root_cid=_DIGEST_A,
            old_repository_state_cid=_DIGEST_B,
            old_manifest_root_cid=_DIGEST_E,
            old_forest_root_cid=_DIGEST_F,
            old_aggregation_root=_DIGEST_3,
            new_manifest_root_cid=_DIGEST_E,
            new_forest_root_cid=_DIGEST_F,
            new_aggregation_root=_DIGEST_3,
            category_roots={cat: _DIGEST_F for cat in FOREST_CATEGORIES},
            reused_unit_ids=("unit/a",),
            replaced_unit_ids=(),
            added_unit_ids=(),
            removed_unit_ids=(),
            required_unit_ids=("unit/a",),
            verified_unit_ids=("unit/a",),
            rejected_unit_ids=(),
            invariants_passed=NORMATIVE_INVARIANTS[:10],
            invariants_failed=(),
            sealed=True,
        )


def test_missing_policy_or_parent_fields_fail_closed() -> None:
    with pytest.raises(DeltaSealError, match="verification_policy"):
        build_delta_seal(_parent(), _state(), None, _transition(), units=_good_units())
    with pytest.raises(DeltaSealError, match="parent requires"):
        build_delta_seal(
            {"seal_cid": _PARENT_SEAL},
            _state(),
            _policy(),
            _transition(),
            units=_good_units(),
        )


def test_unit_disposition_enum_matches_closed_set() -> None:
    assert {item.value for item in UnitDisposition} == set(UNIT_DISPOSITIONS)
