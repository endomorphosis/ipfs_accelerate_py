"""IPS-041: seal verification, explanations, and cost comparison APIs."""

from __future__ import annotations

import hashlib

from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.delta_seal import (
    TRANSITION_SCHEMA,
    DeltaSeal,
    DeltaTransitionStatement,
    DeltaUnitEvidence,
    DiffCommitmentView,
    ParentSealView,
    build_delta_seal,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.explanations import (
    BOUND_CACHE_KEY_FIELDS,
    COMPARISON_EVIDENCE,
    EXPLANATION_EVIDENCE,
    FullIncrementalComparison,
    InvalidationDisposition,
    ProofInvalidationExplanation,
    ProofReuseExplanation,
    ReuseDisposition,
    compare_full_and_incremental,
    explain_invalidation,
    explain_reuse,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
    FullCheckpointSeal,
    RepositoryStateView,
    RequiredUnitEvidence,
    VerificationPolicyView,
    create_full_checkpoint,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.metrics import (
    CostProvenance,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
    ParentSealContext,
    PlanMode,
    UnitPlanningInput,
    create_incremental_plan,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.verification import (
    EVIDENCE_SUBSET,
    VERIFICATION_STAGES,
    SealKind,
    SealVerificationReason,
    SealVerificationResult,
    UnitProofView,
    verify_seal,
)
from ipfs_datasets_py.logic.zkp.incremental_sealing.cache_key import REQUIRED_FIELDS
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
_PARENT_SEAL = "sha256:" + ("99" * 32)
_VK = "vk/prod-1"
_POLICY = _DIGEST_D
_TRUSTED = (_VK, "n/a")


def _state(**overrides: object) -> RepositoryStateView:
    payload = {
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
    payload = {
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
    payload = {
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


def _full_seal(**overrides: object) -> FullCheckpointSeal:
    return create_full_checkpoint(
        _state(),
        _policy(),
        units=(
            _unit("unit/a"),
            _unit(
                "unit/b",
                category="static_analysis",
                proof_object_cid=_DIGEST_F,
                terminal_status=ProofTerminalStatus.PROVED.value,
                proof_mode=ProofMode.DIRECT_EXECUTION_PROOF.value,
            ),
        ),
        expected_unit_ids=("unit/a", "unit/b"),
        parent_seal_cid=None,
        fallback_reasons=("first_state", "missing_parent"),
        **overrides,
    )


def _parent_view(**overrides: object) -> ParentSealView:
    payload: dict[str, object] = {
        "seal_cid": _PARENT_SEAL,
        "accepted": True,
        "seal_status": SealStatus.SEALED_FULL.value,
        "repository_id": "repo/accelerate",
        "branch_id": "main",
        "revision": "rev-old",
        "source_root_cid": _DIGEST_A,
        "repository_state_cid": _DIGEST_B,
        "environment_cid": _DIGEST_C,
        "policy_cid": _POLICY,
        "manifest_root_cid": _DIGEST_E,
        "forest_root_cid": _DIGEST_F,
        "aggregation_root": _DIGEST_3,
        "required_unit_ids": ("unit/a", "unit/b"),
        "unit_proof_cids": {"unit/a": _DIGEST_4, "unit/b": _DIGEST_5},
        "proof_schema_version": "1",
        "canonicalization_version": "1",
        "dependency_graph_schema_version": "graph@1",
        "circuit_id": "circuit@v1",
        "verification_key_id": _VK,
    }
    payload.update(overrides)
    return ParentSealView(**payload)  # type: ignore[arg-type]


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


def _delta_seal() -> DeltaSeal:
    parent = _parent_view()
    new_state = _state(
        revision="rev-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        source_root_cid=_DIGEST_1,
        repository_state_cid=_DIGEST_2,
        parent_revision_ids=("rev-old",),
    )
    transition = DeltaTransitionStatement(
        schema=TRANSITION_SCHEMA,
        parent_seal_cid=_PARENT_SEAL,
        branch_id="main",
        old_source_root_cid=_DIGEST_A,
        old_repository_state_cid=_DIGEST_B,
        old_manifest_root_cid=_DIGEST_E,
        old_forest_root_cid=_DIGEST_F,
        old_aggregation_root=_DIGEST_3,
        new_source_root_cid=_DIGEST_1,
        new_repository_state_cid=_DIGEST_2,
        new_revision=new_state.revision,
        parent_revision_ids=("rev-old",),
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
                proof_object_cid=_DIGEST_4,
                parent_proof_object_cid=_DIGEST_4,
                cache_key_unchanged=True,
            ),
            _delta_unit(
                "unit/b",
                "replace",
                proof_object_cid=_DIGEST_7,
                parent_proof_object_cid=_DIGEST_5,
                newly_admitted=True,
                cache_key_unchanged=False,
                terminal_status=ProofTerminalStatus.PROVED.value,
                proof_mode=ProofMode.DIRECT_EXECUTION_PROOF.value,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Evidence / schema contracts
# ---------------------------------------------------------------------------


def test_evidence_subsets_and_bound_cache_key_fields() -> None:
    assert EVIDENCE_SUBSET == "ips/seal-verification@1"
    assert EXPLANATION_EVIDENCE == "ips/reuse-invalidation-explanation@1"
    assert COMPARISON_EVIDENCE == "ips/full-incremental-comparison@1"
    assert tuple(BOUND_CACHE_KEY_FIELDS) == tuple(REQUIRED_FIELDS)
    assert "statement_cid" in BOUND_CACHE_KEY_FIELDS
    assert "verification_key_id" in BOUND_CACHE_KEY_FIELDS
    assert "policy_cid" in BOUND_CACHE_KEY_FIELDS
    assert "type" in VERIFICATION_STAGES
    assert "cryptography" in VERIFICATION_STAGES


# ---------------------------------------------------------------------------
# verify_seal positive path
# ---------------------------------------------------------------------------


def test_verify_full_seal_accepts_under_trusted_keys_and_policy() -> None:
    seal = _full_seal()
    result = verify_seal(seal, _TRUSTED, _policy())
    assert isinstance(result, SealVerificationResult)
    assert result.accepted is True
    assert result.reason is SealVerificationReason.ACCEPTED
    assert result.seal_kind is SealKind.FULL_CHECKPOINT
    assert result.seal_status == SealStatus.SEALED_FULL.value
    assert result.failed_stage is None
    assert "status" in result.stages_passed
    assert "key" in result.stages_passed
    assert "policy" in result.stages_passed
    assert result.to_canonical()["proving_key_exported"] is False
    assert result.to_canonical()["evidence_subset"] == EVIDENCE_SUBSET


def test_verify_delta_seal_accepts_with_parent_history() -> None:
    seal = _delta_seal()
    assert seal.sealed is True
    result = verify_seal(
        seal,
        _TRUSTED,
        _policy(),
        parent_seal=_parent_view(),
        parent_chain=(_PARENT_SEAL,),
    )
    assert result.accepted is True
    assert result.seal_kind is SealKind.DELTA_SEAL
    assert result.seal_status == SealStatus.SEALED_INCREMENTAL.value
    assert "parent" in result.stages_passed
    assert "history" in result.stages_passed


# ---------------------------------------------------------------------------
# Rejection paths (acceptance criteria)
# ---------------------------------------------------------------------------


def test_unknown_status_rejects() -> None:
    seal = _full_seal().to_canonical()
    seal["seal_status"] = "mystery_status"
    seal["sealed"] = True
    result = verify_seal(seal, _TRUSTED, _policy())
    assert result.accepted is False
    assert result.reason is SealVerificationReason.UNKNOWN_STATUS
    assert result.failed_stage == "status"


def test_unknown_proof_system_rejects() -> None:
    seal = _full_seal()
    result = verify_seal(
        seal,
        _TRUSTED,
        _policy(),
        unit_proofs=(
            UnitProofView(
                unit_id="unit/a",
                proof_object_cid=_DIGEST_E,
                proof_system_id="invented_system",
            ),
        ),
    )
    assert result.accepted is False
    assert result.reason is SealVerificationReason.UNKNOWN_PROOF_SYSTEM
    assert result.failed_stage == "proof_system"


def test_wrong_verification_key_rejects() -> None:
    seal = _full_seal()
    result = verify_seal(seal, ("vk/other",), _policy())
    assert result.accepted is False
    assert result.reason is SealVerificationReason.UNALLOWLISTED_VERIFICATION_KEY
    assert result.failed_stage == "key"


def test_wrong_policy_rejects() -> None:
    seal = _full_seal()
    other = _policy(policy_cid=_DIGEST_1)
    result = verify_seal(seal, _TRUSTED, other)
    assert result.accepted is False
    assert result.reason is SealVerificationReason.WRONG_POLICY
    assert result.failed_stage == "policy"


def test_wrong_parent_rejects_delta() -> None:
    seal = _delta_seal()
    wrong_parent = _parent_view(seal_cid=_DIGEST_1)
    result = verify_seal(
        seal,
        _TRUSTED,
        _policy(),
        parent_seal=wrong_parent,
        parent_chain=(_DIGEST_1,),
    )
    assert result.accepted is False
    assert result.reason is SealVerificationReason.WRONG_PARENT
    assert result.failed_stage == "parent"


def test_wrong_root_rejects() -> None:
    seal = _full_seal()
    result = verify_seal(
        seal,
        _TRUSTED,
        _policy(),
        expected_source_root_cid=_DIGEST_1,
    )
    assert result.accepted is False
    assert result.reason is SealVerificationReason.WRONG_ROOT
    assert result.failed_stage == "root"


def test_modified_inputs_reject() -> None:
    seal = _full_seal()
    result = verify_seal(
        seal,
        _TRUSTED,
        _policy(),
        unit_proofs=(
            UnitProofView(
                unit_id="unit/a",
                proof_object_cid=_DIGEST_E,
                public_input_cid=_DIGEST_A,
                observed_public_input_cid=_DIGEST_1,
                proof_system_id="integrity",
            ),
        ),
    )
    assert result.accepted is False
    assert result.reason is SealVerificationReason.MODIFIED_INPUTS
    assert result.failed_stage == "inputs"


def test_incomplete_history_rejects_delta_without_parent() -> None:
    seal = _delta_seal()
    result = verify_seal(seal, _TRUSTED, _policy())
    assert result.accepted is False
    assert result.reason is SealVerificationReason.INCOMPLETE_HISTORY
    assert result.failed_stage == "history"


def test_cryptographic_failure_rejects() -> None:
    seal = _full_seal()
    proof_bytes = b"tampered-proof-material"
    wrong_digest = "sha256:" + ("00" * 32)
    result = verify_seal(
        seal,
        _TRUSTED,
        _policy(),
        unit_proofs=(
            UnitProofView(
                unit_id="unit/a",
                proof_object_cid=wrong_digest,
                proof_bytes=proof_bytes,
                expected_proof_digest=wrong_digest,
                proof_system_id="integrity",
            ),
        ),
    )
    assert result.accepted is False
    assert result.reason is SealVerificationReason.CRYPTOGRAPHIC_FAILURE
    assert result.failed_stage == "cryptography"


def test_signature_failure_rejects_signed_system() -> None:
    seal = _full_seal()
    result = verify_seal(
        seal,
        _TRUSTED,
        _policy(),
        unit_proofs=(
            UnitProofView(
                unit_id="unit/a",
                proof_object_cid=_DIGEST_E,
                proof_system_id="signed_receipt",
                signature="unsigned",
            ),
        ),
    )
    assert result.accepted is False
    assert result.reason is SealVerificationReason.SIGNATURE_FAILURE
    assert result.failed_stage == "signature"


def test_non_accepted_status_rejects() -> None:
    seal = _full_seal().to_canonical()
    seal["seal_status"] = SealStatus.PROOF_FAILED.value
    seal["sealed"] = False
    result = verify_seal(seal, _TRUSTED, _policy())
    assert result.accepted is False
    assert result.reason is SealVerificationReason.NON_ACCEPTED_STATUS


def test_cryptographic_success_with_matching_digest() -> None:
    seal = _full_seal()
    proof_bytes = b"valid-proof-material"
    digest = "sha256:" + hashlib.sha256(proof_bytes).hexdigest()
    result = verify_seal(
        seal,
        _TRUSTED,
        _policy(),
        unit_proofs=(
            UnitProofView(
                unit_id="unit/a",
                proof_object_cid=digest,
                proof_bytes=proof_bytes,
                expected_proof_digest=digest,
                proof_system_id="integrity",
                freshly_verified=True,
            ),
        ),
    )
    assert result.accepted is True


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------


def test_explain_reuse_covers_every_bound_cache_key_field() -> None:
    seal = _delta_seal()
    explanation = explain_reuse(seal, "unit/a")
    assert isinstance(explanation, ProofReuseExplanation)
    assert explanation.evidence_subset == EXPLANATION_EVIDENCE
    assert explanation.reused is True
    assert explanation.disposition is ReuseDisposition.REUSED
    assert explanation.bound_cache_key_fields == BOUND_CACHE_KEY_FIELDS
    covered = {
        item.field_name for item in explanation.equal_cache_key_fields
    } | {item.field_name for item in explanation.unequal_cache_key_fields}
    assert covered == set(BOUND_CACHE_KEY_FIELDS)
    assert explanation.to_canonical()["file_unchanged_is_not_reuse_authority"] is True
    assert explanation.to_canonical()["substitutes_for_verification"] is False
    assert all(item.equal for item in explanation.equal_cache_key_fields)


def test_explain_reuse_reports_changed_fields_as_rejection() -> None:
    seal = _delta_seal()
    explanation = explain_reuse(
        seal,
        "unit/a",
        changed_fields=("source_root_cid", "environment_cid"),
    )
    assert explanation.reused is False
    assert explanation.disposition is ReuseDisposition.REJECTED
    unequal = {item.field_name for item in explanation.unequal_cache_key_fields}
    assert "source_root_cid" in unequal
    assert "environment_cid" in unequal
    assert explanation.reason == "cache_key_field_changed"


def test_explain_invalidation_covers_fields_and_paths() -> None:
    parent = ParentSealContext(
        seal_cid=_PARENT_SEAL,
        repository_state_cid=_DIGEST_B,
        source_root_cid=_DIGEST_A,
        environment_cid=_DIGEST_C,
        policy_cid=_POLICY,
    )
    plan = create_incremental_plan(
        parent,
        _DIGEST_B,
        _DIGEST_2,
        units=(
            UnitPlanningInput(
                unit_id="unit/a",
                preserved=True,
                cache_key_complete=True,
                admitted=True,
                candidate_present=True,
            ),
            UnitPlanningInput(
                unit_id="unit/b",
                preserved=False,
                invalidated=True,
                cache_key_complete=True,
                admitted=False,
                candidate_present=True,
            ),
        ),
    )
    assert plan.mode is PlanMode.INCREMENTAL
    explanation = explain_invalidation(
        plan,
        "unit/b",
        changed_fields=("source_root_cid", "source_artifact_cids"),
        seed_node_ids=("src/mod.py",),
        direct_triggers=("source_implementation_change",),
        invalidation_paths=(
            {
                "seed_node_id": "src/mod.py",
                "target_node_id": "unit/b",
                "edge_types": ("proof_depends_on",),
                "node_ids": ("src/mod.py", "unit/b"),
            },
        ),
    )
    assert isinstance(explanation, ProofInvalidationExplanation)
    assert explanation.invalidated is True
    assert explanation.disposition is InvalidationDisposition.INVALIDATE
    assert explanation.bound_cache_key_fields == BOUND_CACHE_KEY_FIELDS
    changed = {item.field_name for item in explanation.changed_cache_key_fields}
    unchanged = {item.field_name for item in explanation.unchanged_cache_key_fields}
    assert changed | unchanged == set(BOUND_CACHE_KEY_FIELDS)
    assert "source_root_cid" in changed
    assert explanation.invalidation_paths
    assert explanation.invalidation_paths[0].seed_node_id == "src/mod.py"
    assert explanation.to_canonical()["substitutes_for_verification"] is False


def test_explain_invalidation_preserved_unit_lists_all_fields() -> None:
    parent = ParentSealContext(
        seal_cid=_PARENT_SEAL,
        repository_state_cid=_DIGEST_B,
    )
    plan = create_incremental_plan(
        parent,
        _DIGEST_B,
        _DIGEST_2,
        units=(
            UnitPlanningInput(
                unit_id="unit/kept",
                preserved=True,
                cache_key_complete=True,
                admitted=True,
                candidate_present=True,
            ),
        ),
    )
    explanation = explain_invalidation(plan, "unit/kept")
    assert explanation.disposition is InvalidationDisposition.PRESERVE
    assert explanation.invalidated is False
    assert not explanation.changed_cache_key_fields
    assert len(explanation.unchanged_cache_key_fields) == len(BOUND_CACHE_KEY_FIELDS)


# ---------------------------------------------------------------------------
# Cost comparison
# ---------------------------------------------------------------------------


def test_compare_full_and_incremental_estimated() -> None:
    parent = ParentSealContext(
        seal_cid=_PARENT_SEAL,
        repository_state_cid=_DIGEST_B,
        source_root_cid=_DIGEST_A,
    )
    comparison = compare_full_and_incremental(
        _DIGEST_2,
        parent,
        _policy(),
        units=(
            UnitPlanningInput(
                unit_id="unit/a",
                preserved=True,
                cache_key_complete=True,
                admitted=True,
                candidate_present=True,
            ),
            UnitPlanningInput(
                unit_id="unit/b",
                preserved=False,
                invalidated=True,
                cache_key_complete=True,
                admitted=False,
                candidate_present=False,
            ),
        ),
        estimated=True,
    )
    assert isinstance(comparison, FullIncrementalComparison)
    assert comparison.evidence_subset == COMPARISON_EVIDENCE
    assert comparison.mode_selected == PlanMode.INCREMENTAL.value
    assert comparison.incremental_reuse_units >= 1
    assert comparison.incremental_prove_units >= 1
    assert comparison.full_required_units >= 2
    assert comparison.estimated is True
    # Estimates never become measured savings.
    assert comparison.cost_comparison.savings_provenance is CostProvenance.UNKNOWN
    assert comparison.cost_comparison.compute_saved_cpu_ms is None
    assert comparison.to_canonical()["estimated_as_measured"] is False
    assert comparison.to_canonical()["substitutes_for_verification"] is False


def test_compare_with_measured_costs_reports_savings() -> None:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.metrics import (
        ProofMetricsCollector,
    )

    full_c = ProofMetricsCollector()
    full_c.record_units(required=2, reused=0, invalidated=2, proved=2, cache_hits=0)
    full_c.observe_cpu_ms(1000)
    full_c.observe_wall_ms(1200)
    full_c.observe_storage_growth_bytes(8000)
    full_c.observe_leaf_ms(20)
    full_c.observe_aggregate_ms(8)
    full_c.observe_verify_ms(6)

    inc_c = ProofMetricsCollector()
    inc_c.record_units(required=2, reused=1, invalidated=1, proved=1, cache_hits=1)
    inc_c.observe_cpu_ms(400)
    inc_c.observe_wall_ms(500)
    inc_c.observe_storage_growth_bytes(3000)
    inc_c.observe_leaf_ms(10)
    inc_c.observe_aggregate_ms(4)
    inc_c.observe_verify_ms(3)

    parent = ParentSealContext(
        seal_cid=_PARENT_SEAL,
        repository_state_cid=_DIGEST_B,
    )
    plan = create_incremental_plan(
        parent,
        _DIGEST_B,
        _DIGEST_2,
        units=(
            UnitPlanningInput(
                unit_id="unit/a",
                preserved=True,
                cache_key_complete=True,
                admitted=True,
                candidate_present=True,
            ),
            UnitPlanningInput(
                unit_id="unit/b",
                preserved=False,
                invalidated=True,
                cache_key_complete=True,
                admitted=False,
            ),
        ),
    )
    comparison = compare_full_and_incremental(
        _DIGEST_2,
        parent,
        _policy(),
        plan=plan,
        full_cost=full_c.snapshot(),
        incremental_cost=inc_c.snapshot(),
        estimated=False,
    )
    assert comparison.estimated is False
    assert comparison.cost_comparison.compute_saved_cpu_ms == 600
    assert comparison.cost_comparison.savings_provenance is CostProvenance.MEASURED
    assert comparison.visible_failure is False


def test_explanations_never_export_secrets() -> None:
    seal = _delta_seal()
    explanation = explain_reuse(seal, "unit/a")
    canonical = explanation.to_canonical()
    blob = str(canonical)
    assert "witness" not in blob
    assert "proving_key_bytes" not in blob
    assert canonical["verification_evidence"].get("proof_bytes") is None
