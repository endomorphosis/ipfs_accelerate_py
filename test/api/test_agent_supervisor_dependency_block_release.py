"""Dependency-satisfied blocks complete only from current evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.byte_target_eligibility_probe import (
    measure_byte_target_eligibility_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.graph_roundtrip_probe import (
    measure_graph_roundtrip_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.snapshot_custody_probe import (
    measure_snapshot_custody_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.shared_module_probe import (
    measure_shared_module_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.v2_owner_envelope_probe import (
    measure_v2_owner_envelope_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.typed_decoder_probe import (
    measure_typed_decoder_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.shared_lake_build_probe import (
    measure_shared_lake_build_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.consequence_audit_probe import (
    measure_consequence_audit_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.invalidation_keys_probe import (
    measure_invalidation_keys_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.incremental_lake_agreement_probe import (
    measure_incremental_lake_agreement_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.release_coordinator_probe import (
    measure_release_coordinator_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.modality_causal_comparison_probe import (
    measure_modality_causal_comparison_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.modality_error_report_probe import (
    measure_modality_error_report_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.training_metric_report_probe import (
    measure_training_metric_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.width_control_specification_probe import (
    measure_width_control_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.context_budget_audit_probe import (
    measure_context_budget_audit_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.occurrence_visibility_audit_probe import (
    measure_occurrence_visibility_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.label_mask_audit_probe import (
    measure_label_mask_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.teacher_identity_audit_probe import (
    measure_teacher_identity_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.population_coverage_audit_probe import (
    measure_population_coverage_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.storage_concurrency_audit_probe import (
    measure_storage_concurrency_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.width_comparison_cell_probe import (
    measure_native384_comparison_probe,
    measure_native768_comparison_probe,
    measure_sparse8_comparison_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.unreviewed_representation_probe import (
    measure_unreviewed_representation_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.codec_intact_probe import (
    measure_codec_intact_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.definition_correspondence_probe import (
    measure_definition_correspondence_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.dependency_block_resolution import (
    EXPECTED_BYTE_TARGET_PROBE,
    EXPECTED_CODEC_PROBE,
    EXPECTED_DECLARED_CLOSURE_PROBE,
    EXPECTED_DECLARED_MEANING_PROBE,
    EXPECTED_DEFINITION_CORRESPONDENCE_PROBE,
    EXPECTED_FAMILY_LOWERING_PROBE,
    EXPECTED_FAMILY_ROUTE_PROBE,
    EXPECTED_LOOKUP_PARTITION_PROBE,
    EXPECTED_SCOPED_GRAPH_PROBE,
    EXPECTED_GRAPH_ROUNDTRIP_PROBE,
    EXPECTED_SNAPSHOT_CUSTODY_PROBE,
    EXPECTED_SHARED_MODULE_PROBE,
    EXPECTED_V2_OWNER_ENVELOPE_PROBE,
    EXPECTED_TYPED_DECODER_PROBE,
    EXPECTED_SHARED_LAKE_BUILD_PROBE,
    EXPECTED_CONSEQUENCE_AUDIT_PROBE,
    EXPECTED_INVALIDATION_KEYS_PROBE,
    EXPECTED_INCREMENTAL_LAKE_AGREEMENT_PROBE,
    EXPECTED_RELEASE_COORDINATOR_PROBE,
    EXPECTED_MODALITY_CAUSAL_COMPARISON_PROBE,
    EXPECTED_MODALITY_ERROR_REPORT_PROBE,
    EXPECTED_TRAINING_METRIC_PROBE,
    EXPECTED_WIDTH_CONTROL_PROBE,
    EXPECTED_SPARSE8_PROBE,
    EXPECTED_NATIVE384_PROBE,
    EXPECTED_NATIVE768_PROBE,
    EXPECTED_CONTEXT_BUDGET_PROBE,
    EXPECTED_OCCURRENCE_VISIBILITY_PROBE,
    EXPECTED_LABEL_MASK_PROBE,
    EXPECTED_TEACHER_IDENTITY_PROBE,
    EXPECTED_POPULATION_COVERAGE_PROBE,
    EXPECTED_STORAGE_CONCURRENCY_PROBE,
    EXPECTED_UNREVIEWED_REPRESENTATION_PROBE,
    byte_target_eligibility_evidence,
    graph_roundtrip_evidence,
    snapshot_custody_evidence,
    shared_module_evidence,
    v2_owner_envelope_evidence,
    typed_decoder_evidence,
    shared_lake_build_evidence,
    consequence_audit_evidence,
    invalidation_keys_evidence,
    incremental_lake_agreement_evidence,
    release_coordinator_evidence,
    modality_causal_comparison_evidence,
    modality_error_report_evidence,
    training_metric_evidence,
    width_control_evidence,
    sparse8_comparison_evidence,
    native384_comparison_evidence,
    native768_comparison_evidence,
    context_budget_evidence,
    occurrence_visibility_evidence,
    label_mask_evidence,
    teacher_identity_evidence,
    population_coverage_evidence,
    storage_concurrency_evidence,
    unreviewed_representation_evidence,
    catalogue_resolution_decision,
    codec_intact_evidence,
    declared_closure_evidence,
    declared_meaning_evidence,
    definition_correspondence_evidence,
    dependency_release_decision,
    dependency_resolution_receipt,
    family_lowering_evidence,
    family_route_evidence,
    lookup_partition_evidence,
    markdown_blocked_task_can_reopen,
    profile_limit_evidence,
    scoped_graph_evidence,
    recorded_fact_evidence,
    resolution_obligation_evidence,
    restatement_retirement,
    substantive_resolution_evidence,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.dependency_closure_probe import (
    measure_dependency_closure_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.meaning_material_probe import (
    measure_declared_meaning_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.lookup_partition_probe import (
    measure_lookup_partition_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.family_lowering_probe import (
    measure_family_lowering_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.family_route_probe import (
    measure_family_route_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.scoped_graph_probe import (
    measure_scoped_graph_probe,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import (
    open_intent_repository,
)

pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for dependency-block release tests",
)

_BOILERPLATE = (
    "Requires a reviewed current job packet, exact eligible inputs "
    "and fresh purpose-specific native admission."
)


def _manual_body(reason: str, *, stage: str = "validation") -> dict[str, object]:
    return {
        "completion": "manual",
        "execution_packet_required": True,
        "blocked_reason": reason,
        "resource_stage": stage,
        "required_stage_packets": [stage],
        "gold_or_teacher_admission_granted": False,
        "admitted": False,
        "formalized": False,
        "constitution_roundtrip_ok": False,
    }


def test_classifier_keeps_review_publication_and_unknown_reasons() -> None:
    boilerplate = _manual_body(_BOILERPLATE, stage="corpus")
    assert (
        catalogue_resolution_decision(
            status="blocked",
            body=boilerplate,
            dependency_statuses=("completed",),
            active_block=False,
        )
        == "release"
    )
    review = _manual_body(
        "Requires actual independent legal/formal reviewer capabilities; "
        "agents may prepare packets but cannot invent reviews.",
        stage="semantic_review",
    )
    assert (
        catalogue_resolution_decision(
            status="blocked",
            body=review,
            dependency_statuses=("completed",),
            active_block=False,
        )
        == "substantive_review_or_publication_hold"
    )
    publication = _manual_body(_BOILERPLATE, stage="publication")
    publication["required_stage_packets"] = ["publication"]
    assert (
        dependency_release_decision(
            status="blocked",
            body=publication,
            dependency_statuses=("completed",),
            active_block=False,
            criteria=(
                "No automatic runtime/default/gold/teacher promotion; "
                "preserve prior coherent releases and failure evidence.",
            ),
        )
        == "substantive_review_or_publication_hold"
    )
    assert (
        dependency_release_decision(
            status="blocked",
            body={"blocked_reason": "operator_reconciliation_required"},
            dependency_statuses=("completed",),
            active_block=False,
        )
        == "unclassified_blocked_reason"
    )
    assert markdown_blocked_task_can_reopen(
        {"blocked reason": ""}, depends_on_count=1
    )
    assert markdown_blocked_task_can_reopen(
        {"blocked reason": _BOILERPLATE}, depends_on_count=1
    )
    assert not markdown_blocked_task_can_reopen(
        {"blocked reason": "operator_reconciliation_required"},
        depends_on_count=1,
    )
    assert not markdown_blocked_task_can_reopen({}, depends_on_count=0)
    receipt = dependency_resolution_receipt()
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False


def _seed(tmp_path: Path):
    repo = open_intent_repository(
        tmp_path / "control.duckdb",
        owner_id="owner:dependency-release",
        evidence_freshness_seconds=0,
    )
    repo.upsert_objective(
        objective_id="objective:scientific",
        objective_alias="scientific",
        title="Scientific",
    )
    repo.upsert_goal(
        goal_cid="goal:scientific",
        goal_alias="scientific-goal",
        title="Scientific goal",
        objective_id="objective:scientific",
    )
    repo.upsert_plan(
        plan_cid="plan:scientific",
        goal_cid="goal:scientific",
        plan_alias="scientific-plan",
        status="active",
    )
    return repo


def _task(
    repo,
    cid: str,
    alias: str,
    *,
    status: str,
    body: dict,
    dependencies=(),
    acceptance=None,
):
    repo.upsert_task(
        task_cid=cid,
        task_alias=alias,
        goal_cid="goal:scientific",
        plan_cid="plan:scientific",
        objective_id="objective:scientific",
        status=status,
        body=body,
        dependencies=list(dependencies),
        acceptance=acceptance
        or [
            {
                "criterion": "current validation evidence is already recorded",
                "evidence_kind": "validation",
            }
        ],
    )


def test_release_completes_only_recorded_evidence_and_leaves_holds(
    tmp_path: Path,
) -> None:
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:predecessor",
        "LIR-DONE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    _task(
        repo,
        "task:ready-evidence",
        "LIR-EVIDENCE",
        status="blocked",
        body=_manual_body(""),
        dependencies=["task:predecessor"],
    )
    _task(
        repo,
        "task:pending",
        "LIR-PENDING",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="corpus"),
        dependencies=["task:predecessor"],
    )
    _task(
        repo,
        "task:review",
        "LIR-REVIEW",
        status="blocked",
        body=_manual_body(
            "agents may prepare packets but cannot invent reviews.",
            stage="semantic_review",
        ),
        dependencies=["task:predecessor"],
    )
    _task(
        repo,
        "task:open-dep",
        "LIR-WAITING",
        status="blocked",
        body=_manual_body(""),
        dependencies=["task:review"],
    )
    repo.record_evidence(
        task_cid="task:ready-evidence",
        evidence_kind="validation",
        digest="sha256:" + ("ab" * 32),
        body={"admitted": False, "formalized": False},
    )

    result = repo.release_dependency_satisfied_blocks()

    completed = {item["task_alias"] for item in result["completed"]}
    pending = {item["task_alias"] for item in result["pending"]}
    held = {item["task_alias"]: item["decision"] for item in result["held"]}
    assert completed == {"LIR-EVIDENCE"}
    assert pending == {"LIR-PENDING"}
    assert result["spawned"] == []
    assert held["LIR-REVIEW"] == "substantive_review_or_publication_hold"
    assert "LIR-WAITING" not in held
    assert "LIR-WAITING" not in pending
    evidence = repo.get_task("task:ready-evidence")
    assert evidence is not None
    assert evidence["status"] == "completed"
    receipt = evidence["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_from_current_evidence"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    assert evidence["body"]["gold_or_teacher_admission_granted"] is False
    assert repo.get_task("task:pending")["status"] == "blocked"
    assert repo.get_task("task:review")["status"] == "blocked"
    assert repo.get_task("task:open-dep")["status"] == "blocked"

    again = repo.release_dependency_satisfied_blocks()
    assert again["completed"] == []
    assert {item["task_alias"] for item in again["pending"]} == {"LIR-PENDING"}
    assert again["spawned"] == []
    assert repo.get_task("LIR-PENDING--missing-record") is None
    assert repo.get_task("task:pending")["dependencies"] == ("task:predecessor",)


_EVALUATION_ACCEPTANCE = (
    (
        "Selected N and comparable K denominators, component effects, "
        "seed uncertainty and named losses remain separate."
    ),
    (
        "Useful source-conditioned performance can qualify its scope "
        "without proving a latent bottleneck benefit."
    ),
)
_DENOMINATOR_OWNER = """
full_selected_structure_diagnostics
comparable_pair_structure_diagnostics
Full selected and comparable-pair rates are separate descriptive structural diagnostics; neither is reviewed accuracy or training qualification.
Generation dimension/checkpoint/encoder/settings fields and no-target flags are declarations only; no model, numerical execution, target-withholding or latent-only inference is authenticated.
neither is reviewed accuracy or training qualification.
"""
_METRIC_OWNER = (
    _DENOMINATOR_OWNER
    + "component_effects_remain_separate: true\n"
    + "seed_uncertainty_remains_separate: true\n"
    + "named_losses_remain_separate: true\n"
)


def _call_evidence(repo) -> None:
    repo.record_evidence(
        task_cid="task:predecessor",
        evidence_kind="validation",
        digest="sha256:" + ("11" * 32),
        body={
            "model_executed": True,
            "producer_called": True,
            "legal_acceptance_claimed": False,
            "mean_coordinate_mse_is_semantic_superiority": False,
            "latent_bottleneck_benefit_proven": False,
            "training_executed": False,
            "admitted": False,
            "formalized": False,
            "gold_or_teacher_admission_granted": False,
            "constitution_roundtrip_ok": False,
        },
    )


def test_release_splits_a_blocked_goal_into_recorded_obligations(
    tmp_path: Path,
) -> None:
    repo = _seed(tmp_path)
    (tmp_path / "legal_relational_batch_evaluation_v1.py").write_text(
        _DENOMINATOR_OWNER
    )
    _task(
        repo,
        "task:predecessor",
        "LIR-DONE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    _call_evidence(repo)
    _task(
        repo,
        "task:pending--missing-record",
        "LIR-PENDING--missing-record",
        status="blocked",
        body={
            "completion": "manual",
            "execution_packet_required": True,
            "blocked_reason": (
                "Recorded result is absent. This subgoal stays blocked until "
                "that result is stored."
            ),
            "resource_stage": "analysis",
            "missing_record_subgoal": True,
            "parent_task_cid": "task:pending",
            "parent_task_alias": "LIR-PENDING",
            "admitted": False,
            "formalized": False,
            "gold_or_teacher_admission_granted": False,
            "constitution_roundtrip_ok": False,
            "completion_authority": False,
        },
        dependencies=["task:predecessor"],
        acceptance=tuple(
            f"Produce a recorded result that satisfies: {item}"
            for item in _EVALUATION_ACCEPTANCE
        ),
    )
    _task(
        repo,
        "task:pending",
        "LIR-PENDING",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="evaluation"),
            "existing_owners": ["legal_relational_batch_evaluation_v1.py"],
        },
        dependencies=["task:predecessor", "task:pending--missing-record"],
        acceptance=_EVALUATION_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    completed = {item["task_alias"] for item in result["completed"]}
    assert completed == {
        "LIR-PENDING--00-n-comparable-k-denominators",
        "LIR-PENDING--04-useful-source-conditioned-performance",
    }
    assert "LIR-PENDING" not in completed
    parent = repo.get_task("task:pending")
    assert parent["status"] == "blocked"
    assert "task:pending--missing-record" not in parent["dependencies"]
    assert "task:pending--01-component-effects" in parent["dependencies"]
    assert "task:pending--02-seed-uncertainty" in parent["dependencies"]
    assert "task:pending--03-named-losses" in parent["dependencies"]
    for alias in (
        "LIR-PENDING--01-component-effects",
        "LIR-PENDING--02-seed-uncertainty",
        "LIR-PENDING--03-named-losses",
    ):
        child = repo.get_task(alias)
        assert child["status"] == "blocked"
        assert child["body"]["resolution_subgoal"] is True
        assert child["body"]["admitted"] is False
        assert child["body"]["formalized"] is False
        assert child["body"]["gold_or_teacher_admission_granted"] is False
        assert child["body"]["constitution_roundtrip_ok"] is False
        assert child["body"]["completion_authority"] is False
    denominators = repo.get_task("LIR-PENDING--00-n-comparable-k-denominators")
    assert denominators["status"] == "completed"
    receipt = denominators["body"]["completion_receipt"]
    assert receipt["operation"] == "dependency_block_resolved_resolution_obligation"
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    evidence = repo._dependency_evidence_records(denominators["task_cid"])
    assert evidence[0]["body"]["selected_n_and_comparable_k_separate"] is True
    assert evidence[0]["body"]["component_effects_recorded"] is False
    assert evidence[0]["body"]["seed_uncertainty_recorded"] is False
    assert evidence[0]["body"]["named_losses_recorded"] is False
    scope = repo.get_task("LIR-PENDING--04-useful-source-conditioned-performance")
    scope_evidence = repo._dependency_evidence_records(scope["task_cid"])
    assert scope_evidence[0]["body"]["usefulness_measured"] is False
    assert scope_evidence[0]["body"]["latent_bottleneck_benefit_proven"] is False
    restatement = repo.get_task("task:pending--missing-record")
    assert restatement["status"] == "blocked"
    assert restatement["body"]["resolution_decomposed"] is True
    assert "task:pending--01-component-effects" in restatement["dependencies"]

    for alias in (
        "LIR-PENDING--01-component-effects--00-cited-comparison-cells-record",
        "LIR-PENDING--01-component-effects--01-evaluation-owner-records-component",
        "LIR-PENDING--02-seed-uncertainty--00-cited-comparison-cells-record",
        "LIR-PENDING--02-seed-uncertainty--01-evaluation-owner-records-seed",
        "LIR-PENDING--03-named-losses--00-cited-comparison-cells-record",
        "LIR-PENDING--03-named-losses--01-evaluation-owner-records-named",
    ):
        work = repo.get_task(alias)
        assert work["status"] == "blocked"
        assert work["body"]["resolution_work"] is True
        assert work["body"]["admitted"] is False
        assert work["body"]["completion_authority"] is False
    assert repo.get_task("task:pending--01-component-effects")["body"][
        "resolution_decomposed"
    ] is True

    again = repo.release_dependency_satisfied_blocks()
    assert again["completed"] == []
    assert again["spawned"] == []
    assert repo.get_task("task:pending")["status"] == "blocked"


def test_release_completes_a_goal_when_every_obligation_is_recorded(
    tmp_path: Path,
) -> None:
    repo = _seed(tmp_path)
    (tmp_path / "legal_relational_batch_evaluation_v1.py").write_text(_METRIC_OWNER)
    _task(
        repo,
        "task:predecessor",
        "LIR-DONE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    _call_evidence(repo)
    _task(
        repo,
        "task:pending",
        "LIR-PENDING",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="evaluation"),
            "existing_owners": ["legal_relational_batch_evaluation_v1.py"],
        },
        dependencies=["task:predecessor"],
        acceptance=_EVALUATION_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert repo.get_task("task:pending")["status"] == "completed"
    assert "LIR-PENDING" in {item["task_alias"] for item in result["completed"]}
    parent = repo.get_task("task:pending")
    receipt = parent["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_resolution_obligations"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    evidence = repo._dependency_evidence_records("task:pending")
    assert evidence[0]["body"]["training_executed"] is False
    assert evidence[0]["body"]["legal_acceptance_claimed"] is False
    assert evidence[0]["body"]["satisfied"] is True
    assert len(evidence[0]["body"]["obligations_satisfied"]) == 5


def test_release_keeps_a_remote_body_hold_until_every_body_matches(
    tmp_path: Path,
) -> None:
    repo = _seed(tmp_path)
    (tmp_path / "selected-assets-public-join.json").write_text(
        """
        {
          "asset_record_count": 16,
          "records_with_current_advertised_lfs_match": 2,
          "byte_exact_payload_verification_executed": false,
          "payload_downloaded": false
        }
        """
    )
    _task(
        repo,
        "task:predecessor",
        "LIR-DONE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    _task(
        repo,
        "task:remote--missing-record",
        "LIR-REMOTE--missing-record",
        status="blocked",
        body={
            "completion": "manual",
            "execution_packet_required": True,
            "blocked_reason": (
                "Recorded result is absent. This subgoal stays blocked until "
                "that result is stored."
            ),
            "resource_stage": "analysis",
            "missing_record_subgoal": True,
            "parent_task_cid": "task:remote",
            "parent_task_alias": "LIR-REMOTE",
            "admitted": False,
            "formalized": False,
            "gold_or_teacher_admission_granted": False,
            "constitution_roundtrip_ok": False,
            "completion_authority": False,
        },
        dependencies=["task:predecessor"],
        acceptance=(
            "Produce a recorded result that satisfies: All remote required bodies match exact immutable commit/path/hash expectations.",
            "Produce a recorded result that satisfies: Partial/uncertain upload keeps the previous coherent release selected; preserve old evidence and failed models.",
        ),
    )
    _task(
        repo,
        "task:remote",
        "LIR-REMOTE",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="publication"),
            "existing_owners": ["selected-assets-public-join.json"],
        },
        dependencies=["task:predecessor", "task:remote--missing-record"],
        acceptance=(
            "All remote required bodies match exact immutable commit/path/hash expectations.",
            "Partial/uncertain upload keeps the previous coherent release selected; preserve old evidence and failed models.",
        ),
    )

    result = repo.release_dependency_satisfied_blocks()

    assert result["completed"] == []
    assert repo.get_task("task:remote")["status"] == "blocked"
    remote = repo.get_task("LIR-REMOTE--00-all-remote-required-bodies")
    assert remote is not None
    assert remote["status"] == "blocked"
    assert "task:remote--missing-record" not in repo.get_task("task:remote")["dependencies"]

    matched_dir = tmp_path / "matched"
    matched_dir.mkdir()
    matched = _seed(matched_dir)
    (matched_dir / "selected-assets-public-join.json").write_text(
        """
        {
          "asset_record_count": 2,
          "records_with_current_advertised_lfs_match": 2,
          "byte_exact_payload_verification_executed": true,
          "payload_downloaded": true
        }
        """
    )
    _task(
        matched,
        "task:predecessor",
        "LIR-DONE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    _task(
        matched,
        "task:remote--missing-record",
        "LIR-REMOTE--missing-record",
        status="blocked",
        body={
            "completion": "manual",
            "execution_packet_required": True,
            "blocked_reason": (
                "Recorded result is absent. This subgoal stays blocked until "
                "that result is stored."
            ),
            "resource_stage": "analysis",
            "missing_record_subgoal": True,
            "parent_task_cid": "task:remote",
            "parent_task_alias": "LIR-REMOTE",
            "admitted": False,
            "formalized": False,
            "gold_or_teacher_admission_granted": False,
            "constitution_roundtrip_ok": False,
            "completion_authority": False,
        },
        dependencies=["task:predecessor"],
        acceptance=(
            "Produce a recorded result that satisfies: All remote required bodies match exact immutable commit/path/hash expectations.",
            "Produce a recorded result that satisfies: Partial/uncertain upload keeps the previous coherent release selected; preserve old evidence and failed models.",
        ),
    )
    _task(
        matched,
        "task:remote",
        "LIR-REMOTE",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="publication"),
            "existing_owners": ["selected-assets-public-join.json"],
        },
        dependencies=["task:predecessor", "task:remote--missing-record"],
        acceptance=(
            "All remote required bodies match exact immutable commit/path/hash expectations.",
            "Partial/uncertain upload keeps the previous coherent release selected; preserve old evidence and failed models.",
        ),
    )

    matched_result = matched.release_dependency_satisfied_blocks()

    assert {
        item["task_alias"] for item in matched_result["completed"]
    } == {"LIR-REMOTE--00-all-remote-required-bodies"}
    assert matched.get_task("task:remote")["status"] == "blocked"
    preserved = matched.get_task("LIR-REMOTE--02-preserve-old-evidence-failed")
    assert preserved is not None
    assert preserved["status"] == "blocked"


def _authority_body(**extra: object) -> dict[str, object]:
    body: dict[str, object] = {
        "admitted": False,
        "formalized": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
        "legal_acceptance_claimed": False,
        "federal_corpus_acceptance_claimed": False,
        "training_executed": False,
        "reviewed_accuracy": False,
    }
    body.update(extra)
    return body


def test_distinct_cells_do_not_satisfy_a_metric_obligation() -> None:
    evidence = (
        {
            "task_cid": "task:cells",
            "digest": "sha256:" + ("ab" * 32),
            "body": _authority_body(
                operation="dependency_block_resolved_resolution_work",
                cells_recorded_separately=True,
                component_effects_recorded=False,
                seed_uncertainty_recorded=False,
                named_losses_recorded=False,
                pooled_score_recorded=False,
            ),
        },
    )
    assert (
        resolution_obligation_evidence(
            ("component effects remain separate.",),
            evidence,
            None,
        )
        is None
    )
    assert (
        recorded_fact_evidence(
            (
                "Selected N and comparable K denominators, component effects, "
                "seed uncertainty and named losses remain separate.",
            ),
            evidence,
        )
        is None
    )


def test_release_resolves_a_high_level_goal_from_recorded_facts(
    tmp_path: Path,
) -> None:
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:predecessor",
        "LIR-DONE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:predecessor",
        evidence_kind="validation",
        digest="sha256:" + ("11" * 32),
        body=_authority_body(
            legal_meaning_established=True,
            semantic_definition_binding_verified=True,
            lake_build_executed=True,
        ),
    )
    _task(
        repo,
        "task:goal",
        "LIR-GOAL",
        status="blocked",
        body=_manual_body(_BOILERPLATE),
        dependencies=["task:predecessor"],
        acceptance=(
            "Accepted components record complete material meaning.",
            "Per-use bindings are recorded on the accepted components.",
            "Real build evidence is recorded for the accepted components.",
        ),
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-GOAL"}
    assert result["spawned"] == []
    goal = repo.get_task("task:goal")
    assert goal["status"] == "completed"
    receipt = goal["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_resolution_obligations"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    evidence = repo._dependency_evidence_records("task:goal")
    assert evidence[0]["body"]["legal_acceptance_claimed"] is False
    assert evidence[0]["body"]["satisfied"] is True
    assert set(evidence[0]["body"]["recorded_facts"]) == {
        "legal_meaning_established",
        "semantic_definition_binding_verified",
        "lake_build_executed",
    }
    again = repo.release_dependency_satisfied_blocks()
    assert again["completed"] == []
    assert again["spawned"] == []


def test_release_completes_a_metric_obligation_from_a_separated_measurement(
    tmp_path: Path,
) -> None:
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:predecessor",
        "LIR-DONE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:predecessor",
        evidence_kind="validation",
        digest="sha256:" + ("22" * 32),
        body=_authority_body(
            metric="component_effects",
            metric_remains_separate=True,
            component_effects_recorded=True,
            pooled_score_recorded=False,
        ),
    )
    _task(
        repo,
        "task:owner-line",
        "LIR-OWNER",
        status="blocked",
        body={
            **_manual_body(
                "This fact stays blocked until that recorded result is stored."
            ),
            "resolution_work": True,
            "obligation_kind": "owner_metric_component_effects",
        },
        dependencies=["task:predecessor"],
        acceptance=(
            "The evaluation owner records component_effects_remain_separate: "
            "true beside the structure diagnostics.",
        ),
    )
    _task(
        repo,
        "task:metric",
        "LIR-METRIC",
        status="blocked",
        body={
            **_manual_body(
                "This obligation stays blocked until its recorded result is stored."
            ),
            "resolution_subgoal": True,
            "obligation_kind": "component_effects",
        },
        dependencies=["task:predecessor", "task:owner-line"],
        acceptance=("component effects remain separate.",),
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-METRIC"}
    metric = repo.get_task("task:metric")
    assert metric["status"] == "completed"
    assert "task:owner-line" not in metric["dependencies"]
    receipt = metric["body"]["completion_receipt"]
    assert receipt["operation"] == "dependency_block_resolved_resolution_obligation"
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    stored = repo._dependency_evidence_records("task:metric")
    assert stored[0]["body"]["metric_remains_separate"] is True
    assert stored[0]["body"]["component_effects_recorded"] is True
    assert stored[0]["body"]["reviewed_accuracy"] is False
    assert stored[0]["body"]["legal_acceptance_claimed"] is False
    owner = repo.get_task("task:owner-line")
    assert owner["status"] == "blocked"
    again = repo.release_dependency_satisfied_blocks()
    assert again["completed"] == []
    assert again["spawned"] == []
    assert repo.get_task("task:owner-line")["status"] == "blocked"


def test_release_splits_a_multi_fact_clause_without_completing_it(
    tmp_path: Path,
) -> None:
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:predecessor",
        "LIR-DONE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:predecessor",
        evidence_kind="validation",
        digest="sha256:" + ("33" * 32),
        body=_authority_body(
            legal_meaning_established=False,
            semantic_definition_binding_verified=False,
            lake_build_executed=False,
        ),
    )
    _task(
        repo,
        "task:goal",
        "LIR-GOAL",
        status="blocked",
        body=_manual_body(
            "The high-level goal stays blocked until each clause is recorded."
        ),
        dependencies=["task:predecessor"],
        acceptance=(
            "Every accepted component retains complete material meaning, "
            "per-use bindings and real build evidence.",
            "Input context and output capacities stay separate.",
        ),
    )

    result = repo.release_dependency_satisfied_blocks()

    assert result["completed"] == []
    spawned = {item["task_alias"] for item in result["spawned"]}
    assert "LIR-GOAL--00-every-accepted-component-retains" in spawned
    assert "LIR-GOAL--01-input-context-output-capacities" in spawned
    assert {
        "LIR-GOAL--00-every-accepted-component-retains--00-accepted-components-record-complete",
        "LIR-GOAL--00-every-accepted-component-retains--01-per-use-bindings-are",
        "LIR-GOAL--00-every-accepted-component-retains--02-real-build-evidence-is",
    } <= spawned
    for alias in spawned:
        child = repo.get_task(alias)
        assert child["status"] == "blocked"
        assert child["body"]["admitted"] is False
        assert child["body"]["formalized"] is False
        assert child["body"]["gold_or_teacher_admission_granted"] is False
        assert child["body"]["constitution_roundtrip_ok"] is False
        assert child["body"]["completion_authority"] is False
    assert repo.get_task("task:goal")["status"] == "blocked"
    again = repo.release_dependency_satisfied_blocks()
    assert again["completed"] == []
    assert again["spawned"] == []


def test_restatement_retirement_refuses_a_present_marker_and_other_work() -> None:
    owners = {
        "evaluation": {
            "text": "component_effects_remain_separate: true\n",
            "sha256": "abc",
        }
    }
    assert restatement_retirement(
        {
            "resolution_work": True,
            "obligation_kind": "owner_metric_component_effects",
        },
        owners,
    ) is None
    assert restatement_retirement(
        {
            "resolution_work": True,
            "obligation_kind": "partial_upload_policy_recorded",
        },
        owners,
    ) is None
    assert restatement_retirement(
        {"missing_record_subgoal": True, "resolution_decomposed": True},
        None,
    ) is None


def test_release_skips_a_superseded_missing_record_without_completing_the_parent(
    tmp_path: Path,
) -> None:
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:predecessor",
        "LIR-DONE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    _task(
        repo,
        "task:goal",
        "LIR-GOAL",
        status="blocked",
        body={
            **_manual_body("The parent stays blocked until the recorded clauses exist."),
            "resolution_decomposed": True,
        },
        dependencies=["task:predecessor"],
        acceptance=(
            "component effects remain separate.",
            "seed uncertainty remain separate.",
        ),
    )
    _task(
        repo,
        "task:goal--missing-record",
        "LIR-GOAL--missing-record",
        status="blocked",
        body={
            **_manual_body("Recorded result is absent."),
            "missing_record_subgoal": True,
            "resolution_decomposed": True,
            "superseded_by": ["task:goal--00", "task:goal--01"],
        },
        dependencies=["task:predecessor"],
        acceptance=(
            "Produce a recorded result that satisfies: component effects remain separate.",
            "Produce a recorded result that satisfies: seed uncertainty remain separate.",
        ),
    )
    result = repo.release_dependency_satisfied_blocks()
    assert result["completed"] == []
    assert {item["task_alias"] for item in result["retired"]} == {
        "LIR-GOAL--missing-record"
    }
    retired = repo.get_task("LIR-GOAL--missing-record")
    assert retired["status"] == "skipped"
    assert retired["body"]["admitted"] is False
    assert retired["body"]["formalized"] is False
    assert retired["body"]["gold_or_teacher_admission_granted"] is False
    assert retired["body"]["constitution_roundtrip_ok"] is False
    assert retired["body"]["completion_receipt"]["operation"] == (
        "dependency_block_restatement_retired"
    )
    evidence = repo._dependency_evidence_records("task:goal--missing-record")
    assert evidence[0]["body"]["satisfied"] is False
    assert evidence[0]["body"]["legal_acceptance_claimed"] is False
    assert repo.get_task("LIR-GOAL")["status"] == "blocked"


def test_release_keeps_a_missing_record_when_a_real_task_depends_on_it(
    tmp_path: Path,
) -> None:
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:predecessor",
        "LIR-DONE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    _task(
        repo,
        "task:missing",
        "LIR-MISSING",
        status="blocked",
        body={
            **_manual_body("Recorded result is absent."),
            "missing_record_subgoal": True,
            "resolution_decomposed": True,
            "superseded_by": ["task:other"],
        },
        dependencies=["task:predecessor"],
    )
    _task(
        repo,
        "task:other",
        "LIR-OTHER",
        status="blocked",
        body=_manual_body("This task still needs the missing record."),
        dependencies=["task:missing"],
    )
    result = repo.release_dependency_satisfied_blocks()
    assert result["retired"] == []
    assert repo.get_task("LIR-MISSING")["status"] == "blocked"


def test_release_skips_an_absent_owner_marker_and_leaves_the_metric_blocked(
    tmp_path: Path,
) -> None:
    repo = _seed(tmp_path)
    relative = "ipfs_datasets_py/logic/autoformal/legal_relational_batch_evaluation_v1.py"
    owner = tmp_path / relative
    owner.parent.mkdir(parents=True)
    owner.write_text(
        "full_selected_structure_diagnostics\n"
        "comparable_pair_structure_diagnostics\n"
        "Full selected and comparable-pair rates are separate descriptive "
        "structural diagnostics; neither is reviewed accuracy or training "
        "qualification.\n",
        encoding="utf-8",
    )
    _task(
        repo,
        "task:cells",
        "LIR-CELLS",
        status="completed",
        body={
            "completion": "manual",
            "execution_packet_required": True,
            "cells_recorded_separately": True,
            "component_effects_recorded": False,
            "metric_remains_separate": False,
        },
    )
    _task(
        repo,
        "task:owner-line",
        "LIR-OWNER",
        status="blocked",
        body={
            **_manual_body("This fact stays blocked until that recorded result is stored."),
            "resolution_work": True,
            "obligation_kind": "owner_metric_component_effects",
            "existing_owners": [relative],
        },
        dependencies=["task:cells"],
        acceptance=(
            "The evaluation owner records component_effects_remain_separate: true "
            "beside the structure diagnostics.",
        ),
    )
    _task(
        repo,
        "task:metric",
        "LIR-METRIC",
        status="blocked",
        body={
            **_manual_body("This obligation stays blocked until its recorded result is stored."),
            "resolution_subgoal": True,
            "resolution_decomposed": True,
        },
        dependencies=["task:cells", "task:owner-line"],
        acceptance=("component effects remain separate.",),
    )
    result = repo.release_dependency_satisfied_blocks()
    assert result["completed"] == []
    assert {item["task_alias"] for item in result["retired"]} == {"LIR-OWNER"}
    assert repo.get_task("LIR-OWNER")["status"] == "skipped"
    metric = repo.get_task("LIR-METRIC")
    assert metric["status"] == "blocked"
    assert metric["body"].get("admitted") is not True
    assert metric["body"].get("legal_acceptance_claimed") is not True
    evidence = repo._dependency_evidence_records("task:owner-line")
    assert evidence[0]["body"]["satisfied"] is False
    assert evidence[0]["body"]["retirement_reason"] == "pinned_owner_marker_absent"


def test_release_creates_subgoals_for_an_unresolved_high_level_goal(
    tmp_path: Path,
) -> None:
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:predecessor",
        "LIR-DONE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    _task(
        repo,
        "task:goal",
        "LIR-GOAL",
        status="blocked",
        body=_manual_body(
            "The high-level goal stays blocked until each clause is recorded."
        ),
        dependencies=["task:predecessor"],
        acceptance=(
            "Every accepted component retains complete material meaning.",
            "Input context and output capacities stay separate.",
        ),
    )

    result = repo.release_dependency_satisfied_blocks()

    assert result["completed"] == []
    assert {item["task_alias"] for item in result["spawned"]} == {
        "LIR-GOAL--00-every-accepted-component-retains",
        "LIR-GOAL--01-input-context-output-capacities",
    }
    parent = repo.get_task("task:goal")
    assert parent["status"] == "blocked"
    assert parent["body"]["resolution_decomposed"] is True
    for alias in (
        "LIR-GOAL--00-every-accepted-component-retains",
        "LIR-GOAL--01-input-context-output-capacities",
    ):
        child = repo.get_task(alias)
        assert child["status"] == "blocked"
        assert child["body"]["resolution_subgoal"] is True
        assert child["body"]["admitted"] is False
        assert child["body"]["formalized"] is False
        assert child["body"]["gold_or_teacher_admission_granted"] is False
        assert child["body"]["constitution_roundtrip_ok"] is False
        assert child["body"]["completion_authority"] is False
        assert child["task_cid"] in parent["dependencies"]
    again = repo.release_dependency_satisfied_blocks()
    assert again["completed"] == []
    assert again["spawned"] == []
    assert repo.get_task("task:goal")["status"] == "blocked"


def test_release_clears_an_evidence_backed_chain_in_one_call(
    tmp_path: Path,
) -> None:
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:predecessor",
        "LIR-DONE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    _task(
        repo,
        "task:tail",
        "LIR-TAIL",
        status="blocked",
        body=_manual_body(""),
        dependencies=["task:mid"],
    )
    _task(
        repo,
        "task:mid",
        "LIR-MID",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="corpus"),
        dependencies=["task:predecessor"],
    )
    _task(
        repo,
        "task:review",
        "LIR-REVIEW",
        status="blocked",
        body=_manual_body(
            "Requires actual independent review of the assembled interpretation.",
            stage="semantic_review",
        ),
        dependencies=["task:tail"],
    )
    for cid in ("task:tail", "task:mid"):
        repo.record_evidence(
            task_cid=cid,
            evidence_kind="validation",
            digest="sha256:" + ("cd" if cid.endswith("tail") else "ef") * 32,
            body={"admitted": False, "formalized": False},
        )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {
        "LIR-MID",
        "LIR-TAIL",
    }
    assert result["pending"] == []
    assert {item["task_alias"] for item in result["held"]} == {"LIR-REVIEW"}
    assert repo.get_task("task:mid")["status"] == "completed"
    assert repo.get_task("task:tail")["status"] == "completed"
    assert repo.get_task("task:review")["status"] == "blocked"
    for cid in ("task:mid", "task:tail"):
        receipt = repo.get_task(cid)["body"]["completion_receipt"]
        assert receipt["admitted"] is False
        assert receipt["formalized"] is False
        assert receipt["gold_or_teacher_admission_granted"] is False
        assert receipt["constitution_roundtrip_ok"] is False


_PUBLICATION_ACCEPTANCE = [
    {
        "criterion": (
            "Distinguish implementation qualification, matched engineering "
            "gain and statutory generalization."
        ),
    },
    {
        "criterion": (
            "No automatic runtime/default/gold/teacher promotion; preserve "
            "prior coherent releases and failure evidence."
        ),
    },
]


def _evaluation_facts() -> dict[str, object]:
    return {
        "criteria_verified": 3,
        "model_forwards": 0,
        "training_executed": False,
        "full_ir_gained_or_lost": 0,
        "winner_selected": False,
        "heldout_uscode_fidelity": False,
        "admitted": False,
        "formalized": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }


def test_substantive_resolution_refuses_reviewer_identity() -> None:
    evidence = (
        {
            "task_cid": "task:prep",
            "digest": "sha256:" + ("11" * 32),
            "body": _evaluation_facts(),
        },
    )
    assert (
        substantive_resolution_evidence(
            (
                "Actual reviewer identity/trust/competence, independence, time and exposure evidence are retained.",
                "Confirm the complete norm, or preserve a blocker.",
            ),
            evidence,
        )
        is None
    )
    explicit = substantive_resolution_evidence(
        ("Confirm the complete norm, or preserve a blocker.",),
        evidence,
    )
    assert explicit is not None
    assert explicit["reviewer_claimed"] is False
    assert explicit["decision_fabricated"] is False
    assert explicit["operation"] == "dependency_block_resolved_explicit_blocker"


def test_release_records_publication_non_promotion_and_leaves_reviewer_hold(
    tmp_path: Path,
) -> None:
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:eval",
        "LIR-EVAL",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:eval",
        evidence_kind="validation",
        digest="sha256:" + ("12" * 32),
        body=_evaluation_facts(),
    )
    _task(
        repo,
        "task:publication",
        "LIR-PUB",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="publication"),
        dependencies=["task:eval"],
        acceptance=_PUBLICATION_ACCEPTANCE,
    )
    _task(
        repo,
        "task:review",
        "LIR-REVIEW",
        status="blocked",
        body=_manual_body(
            "Requires actual independent legal/formal reviewer capabilities.",
            stage="semantic_review",
        ),
        dependencies=["task:eval"],
        acceptance=[
            {
                "criterion": (
                    "Actual reviewer identity/trust/competence, independence, "
                    "time and exposure evidence are retained."
                ),
            },
            {
                "criterion": (
                    "Every material argument has a supported interpretation "
                    "or explicit blocker; no reviewer or decision is fabricated."
                ),
            },
        ],
    )
    _task(
        repo,
        "task:thin",
        "LIR-THIN",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="publication"),
        dependencies=["task:review"],
        acceptance=_PUBLICATION_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare-eval",
        "LIR-BARE-EVAL",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:bare-eval",
        evidence_kind="validation",
        digest="sha256:" + ("13" * 32),
        body={"admitted": False, "formalized": False},
    )
    _task(
        repo,
        "task:bare-pub",
        "LIR-BARE",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="publication"),
        dependencies=["task:bare-eval"],
        acceptance=_PUBLICATION_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-PUB"}
    assert {item["task_alias"] for item in result["held"]} == {
        "LIR-REVIEW",
        "LIR-BARE",
    }
    published = repo.get_task("task:publication")
    assert published["status"] == "completed"
    receipt = published["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_publication_non_promotion"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["reviewer_claimed"] is False
    assert repo.get_task("task:review")["status"] == "blocked"
    assert repo.get_task("task:thin")["status"] == "blocked"
    assert repo.get_task("task:bare-pub")["status"] == "blocked"


_REVIEW_ACCEPTANCE = [
    {
        "criterion": (
            "Actual reviewer identity/trust/competence, independence, time "
            "and exposure evidence are retained."
        ),
    },
    {
        "criterion": (
            "Every material argument, qualifier and alternative has a "
            "supported interpretation or explicit blocker; no reviewer or "
            "decision is fabricated."
        ),
    },
    {
        "criterion": (
            "Confirm the complete definition, complete norm and second "
            "physical use under actual governing scope, or preserve a blocker."
        ),
    },
]


def test_release_records_blank_review_as_explicit_blocker_without_naming_one(
    tmp_path: Path,
) -> None:
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:packet",
        "LIR-PACKET",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:packet",
        evidence_kind="validation",
        digest="sha256:" + ("21" * 32),
        body={
            "blank_answer_slots": 60,
            "admitted": False,
            "formalized": False,
            "gold_or_teacher_admission_granted": False,
        },
    )
    _task(
        repo,
        "task:review",
        "LIR-REVIEW",
        status="blocked",
        body=_manual_body(
            "Requires actual independent legal/formal reviewer capabilities.",
            stage="semantic_review",
        ),
        dependencies=["task:packet"],
        acceptance=_REVIEW_ACCEPTANCE,
    )
    _task(
        repo,
        "task:next",
        "LIR-NEXT",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="corpus"),
        dependencies=["task:review"],
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-REVIEW"}
    assert {item["task_alias"] for item in result["pending"]} == {"LIR-NEXT"}
    review = repo.get_task("task:review")
    assert review["status"] == "completed"
    receipt = review["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_review_explicit_blocker"
    )
    assert receipt["reviewer_claimed"] is False
    assert receipt["decision_fabricated"] is False
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert repo.get_task("task:next")["status"] == "blocked"


_PROFILE_ACCEPTANCE = [
    {
        "criterion": (
            "Preserve arity, ordered roles, binders, implication direction, "
            "modality, counting, exceptions and time."
        ),
    },
    {
        "criterion": (
            "Current relational equality/constant/function/temporal limits "
            "and separate counted-frame semantics are explicit."
        ),
    },
]

_PROFILE_MARKERS = {
    "legal_relational_norm_v1.py": (
        "1 <= symbol.arity <= 8\n"
        "ordered distinct bounded argument roles required\n"
        "only explicitly bound Object variables supported\n"
        "exactly binary connective required\n"
        "constants/functions/other terms unsupported\n"
        '"O": "obligation"\n'
    ),
    "legal_counted_frame_v1.py": (
        '"semantics": "arbitrary_index_without_order"\n'
        "No nonempty membership, temporal aggregation, deontic operator or source meaning is asserted.\n"
        "TimeFrame is an arbitrary index with no order, duration, simultaneity or calendar interpretation.\n"
        "Each object is counted by built-in identity within the same event/time pair, without occurrence multiplicity.\n"
        '"maximum": 3\n'
    ),
    "legal_family_routes.py": (
        '"supported_routes": ["deontic"]\n'
        '"cross_family_equivalence_verified": False\n'
        '"all_logic_families_supported": False\n'
        "no temporal operators\n"
        "no deontic operators\n"
    ),
}


def test_release_records_empty_profile_comparison_from_owner_limits(
    tmp_path: Path,
) -> None:
    owners = []
    for name, text in _PROFILE_MARKERS.items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        owners.append(f"owners/{name}")
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:review",
        "LIR-REVIEW",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:review",
        evidence_kind="validation",
        digest="sha256:" + ("31" * 32),
        body={
            "operation": "dependency_block_resolved_review_explicit_blocker",
            "reviewer_identity": {
                "status": "not_supplied",
                "identity_asserted": False,
                "blank_answer_slots": 60,
            },
            "admitted": False,
            "formalized": False,
            "gold_or_teacher_admission_granted": False,
        },
    )
    _task(
        repo,
        "task:profile",
        "LIR-PROFILE",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="corpus"),
            "existing_owners": owners,
        },
        dependencies=["task:review"],
        acceptance=_PROFILE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:next",
        "LIR-NEXT",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="validation"),
        dependencies=["task:profile"],
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-PROFILE"}
    assert {item["task_alias"] for item in result["pending"]} == {"LIR-NEXT"}
    profile = repo.get_task("task:profile")
    assert profile["status"] == "completed"
    receipt = profile["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_profile_limits_from_owners"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["reviewer_claimed"] is False
    assert repo.get_task("task:next")["status"] == "blocked"
    owners_by_role = {
        "relational": {"path": "owners/legal_relational_norm_v1.py", "sha256": "abc", "text": _PROFILE_MARKERS["legal_relational_norm_v1.py"]},
        "counted_frame": {"path": "owners/legal_counted_frame_v1.py", "sha256": "abc", "text": _PROFILE_MARKERS["legal_counted_frame_v1.py"].replace('"maximum": 3', '"maximum": 9')},
        "family": {"path": "owners/legal_family_routes.py", "sha256": "abc", "text": _PROFILE_MARKERS["legal_family_routes.py"]},
    }
    assert profile_limit_evidence(
        [item["criterion"] for item in _PROFILE_ACCEPTANCE],
        (
            {
                "task_cid": "task:review",
                "digest": "sha256:" + ("31" * 32),
                "body": {
                    "operation": "dependency_block_resolved_review_explicit_blocker",
                    "reviewer_identity": {
                        "identity_asserted": False,
                        "blank_answer_slots": 60,
                    },
                },
            },
        ),
        owners_by_role,
    ) is None


_CODEC_ACCEPTANCE = [
    {
        "criterion": (
            "Lossless AST/syntax correspondence, independent positive/negative "
            "cases and exact unsupported refusals pass."
        ),
    },
    {
        "criterion": (
            "Existing codecs/profiles remain intact; no opaque atoms or "
            "omitted text disguise unsupported material."
        ),
    },
]

_CODEC_MARKERS = {
    "ast.py": (
        "opaque string/bytes payloads are rejected\n"
        "opaque unversioned payloads are rejected\n"
    ),
    "legal_relational_norm_v1.py": (
        "TypedExpression must survive exact serialized reconstruction\n"
        "constants/functions/other terms unsupported\n"
        "unsupported relational node: \n"
    ),
    "legal_source_typed_norm_join_v1.py": (
        "unsupported derived typed provenance kind\n"
    ),
}


def _codec_dependency():
    return (
        {
            "task_cid": "task:profile",
            "digest": "sha256:" + ("32" * 32),
            "body": {
                "operation": "dependency_block_resolved_profile_limits_from_owners",
                "selected_meaning_count": 0,
                "meanings_approved": False,
                "review_disposition": "explicit_blocker",
                "admitted": False,
                "formalized": False,
                "gold_or_teacher_admission_granted": False,
                "constitution_roundtrip_ok": False,
            },
        },
    )


def test_codec_probe_matches_the_live_owners() -> None:
    assert measure_codec_intact_probe() == EXPECTED_CODEC_PROBE


def test_codec_evidence_refuses_a_selected_meaning_or_a_changed_codec() -> None:
    owners = {
        "ast": {"path": "owners/ast.py", "sha256": "abc", "text": _CODEC_MARKERS["ast.py"]},
        "relational": {
            "path": "owners/legal_relational_norm_v1.py",
            "sha256": "abc",
            "text": _CODEC_MARKERS["legal_relational_norm_v1.py"],
        },
        "join": {
            "path": "owners/legal_source_typed_norm_join_v1.py",
            "sha256": "abc",
            "text": _CODEC_MARKERS["legal_source_typed_norm_join_v1.py"],
        },
    }
    criteria = [item["criterion"] for item in _CODEC_ACCEPTANCE]
    changed = dict(EXPECTED_CODEC_PROBE)
    changed["vocabulary_size"] = 260
    assert codec_intact_evidence(criteria, _codec_dependency(), owners, changed) is None
    selected = dict(_codec_dependency()[0])
    selected["body"] = dict(selected["body"])
    selected["body"]["selected_meaning_count"] = 1
    assert codec_intact_evidence(criteria, (selected,), owners, EXPECTED_CODEC_PROBE) is None


def test_release_records_no_extension_when_the_codec_stays_intact(
    tmp_path: Path,
) -> None:
    owners = []
    for name, text in _CODEC_MARKERS.items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        owners.append(f"owners/{name}")
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:profile",
        "LIR-PROFILE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:profile",
        evidence_kind="validation",
        digest="sha256:" + ("32" * 32),
        body=_codec_dependency()[0]["body"],
    )
    _task(
        repo,
        "task:codec",
        "LIR-CODEC",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": owners,
        },
        dependencies=["task:profile"],
        acceptance=_CODEC_ACCEPTANCE,
    )
    _task(
        repo,
        "task:next",
        "LIR-NEXT",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="corpus"),
        dependencies=["task:codec"],
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-CODEC"}
    assert {item["task_alias"] for item in result["pending"]} == {"LIR-NEXT"}
    codec = repo.get_task("task:codec")
    assert codec["status"] == "completed"
    receipt = codec["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_codec_intact_no_extension"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["reviewer_claimed"] is False
    assert repo.get_task("task:next")["status"] == "blocked"


_GRAPH_ACCEPTANCE = [
    {
        "criterion": (
            "Exact source/version/interpretation companions and ordered "
            "arguments are retained."
        ),
    },
    {
        "criterion": (
            "Equal spelling does not merge senses, categories do not imply "
            "ground persons and normative action types do not establish "
            "actual events."
        ),
    },
]
_FAMILY_ACCEPTANCE = [
    {
        "criterion": (
            "Obligations are not actual actions; permissions are not duties; "
            "legal premises remain explicit hypotheses/content."
        ),
    },
    {
        "criterion": (
            "Pin semantic library/import closure and axiom policy; no "
            "automatic broad cross-family equivalence."
        ),
    },
]
_SENSE_NEXT_ACCEPTANCE = [
    {
        "criterion": (
            "Do not count the existing two-unary-definition alias component "
            "as a complete operative norm."
        ),
    },
    {
        "criterion": (
            "Every argument and qualifier survives source/AST/graph "
            "correspondence; unresolved alternatives retain separate branches."
        ),
    },
]
_FAMILY_NEXT_ACCEPTANCE = [
    {
        "criterion": (
            "Lossless structure, signature and annotation preservation under "
            "exact profiles."
        ),
    },
    {
        "criterion": (
            "Counted-frame specialized text and DCEC ground fragments retain "
            "their separate support boundaries."
        ),
    },
]
_MODALITY_ACCEPTANCE = [
    {
        "criterion": (
            "Same initialization, source cases, trainable mask, optimizer, "
            "candidates and budget within each causal comparison."
        ),
    },
    {
        "criterion": (
            "Preserve named permission and complete-rule loss-ID vetoes; "
            "E1 engineering ranking work is separately tracked."
        ),
    },
]
_TRAINING_ACCEPTANCE = [
    {
        "criterion": (
            "Presence versus owner errors and critical modality reversals "
            "are separately reported."
        ),
    },
    {
        "criterion": (
            "Failed acceptance remains failed despite aggregate gains; no "
            "automatic default/teacher promotion."
        ),
    },
    {
        "criterion": (
            "Materialize separate producer and post-freeze evaluation "
            "packets/owners; primary training/inference authority grants no "
            "evaluation reference access."
        ),
    },
]
_GRAPH_OWNER_TEXT = {
    "corpus_graph.py": (
        "nor merges equivalent entities nor qualifies legal meaning\n"
        "role edges require an explicit ordinal\n"
        "duplicate role argument position\n"
    ),
    "legal_source_typed_norm_join_v1.py": (
        "The existing source and relational owners retain their separate contracts.\n"
        "Membership object/event/time roles, equality endpoints and lexical "
        "binding are syntax facts, not reviewed legal identities or temporal scopes.\n"
        "does not encode the complete statutory norm, its modality, participants, "
        "duties or applicability.\n"
    ),
}
_ROUTE_OWNER_TEXT = {
    "legal_family_routes.py": (
        "does not make it equivalent to the legal rule.\n"
        "broader family equivalence is unverified\n"
        "Quantifiers/cognition/events are deliberately outside this route.\n"
        '"supported_routes": ["deontic"]\n'
    ),
    "legal_corpus_shared_lake.py": (
        "Kernel checking does not establish law.\n"
        "preparation confers no build authority.\n"
        "Every child's explicit Interpretation, carrier, time, definition and "
        "alias premises remain unchanged.\n"
        "The conclusion is the parameterized modal action; it does not "
        "establish actual action truth or legal truth.\n"
    ),
}


def _intact_codec_dependency():
    return (
        {
            "task_cid": "task:codec",
            "digest": "sha256:" + ("41" * 32),
            "body": {
                "operation": "dependency_block_resolved_codec_intact_no_extension",
                "extensions_added": 0,
                "meanings_requiring_extension": 0,
                "meanings_approved": False,
                "vocabulary_size": 259,
                "admitted": False,
                "formalized": False,
                "gold_or_teacher_admission_granted": False,
                "constitution_roundtrip_ok": False,
            },
        },
    )


def _marker_owners(names: dict[str, str]) -> dict[str, dict[str, str]]:
    roles = {
        "corpus_graph.py": "graph",
        "legal_source_typed_norm_join_v1.py": "join",
        "legal_family_routes.py": "routes",
        "legal_corpus_shared_lake.py": "lake",
    }
    return {
        roles[name]: {"path": "owners/" + name, "sha256": "abc", "text": text}
        for name, text in names.items()
    }


def test_graph_and_family_probes_match_the_live_owners() -> None:
    assert measure_scoped_graph_probe() == EXPECTED_SCOPED_GRAPH_PROBE
    assert measure_family_route_probe() == EXPECTED_FAMILY_ROUTE_PROBE


def test_graph_and_family_evidence_refuse_other_tasks_and_changed_probes() -> None:
    graph_criteria = [item["criterion"] for item in _GRAPH_ACCEPTANCE]
    family_criteria = [item["criterion"] for item in _FAMILY_ACCEPTANCE]
    graph_owners = _marker_owners(_GRAPH_OWNER_TEXT)
    route_owners = _marker_owners(_ROUTE_OWNER_TEXT)
    dependency = _intact_codec_dependency()
    graph = scoped_graph_evidence(
        graph_criteria, dependency, graph_owners, EXPECTED_SCOPED_GRAPH_PROBE
    )
    family = family_route_evidence(
        family_criteria, dependency, route_owners, EXPECTED_FAMILY_ROUTE_PROBE
    )
    assert graph is not None and graph["senses_merged"] is False
    assert graph["category_implies_person"] is False
    assert graph["normative_action_establishes_event"] is False
    assert graph["admitted"] is False
    assert family is not None and family["permission_rewritten_as_obligation"] is False
    assert family["modal_action_establishes_actual_action_truth"] is False
    assert family["lake_build_executed"] is False
    assert family["cross_family_equivalence_verified"] is False
    assert family["admitted"] is False
    changed_graph = dict(EXPECTED_SCOPED_GRAPH_PROBE)
    changed_graph["senses_merged"] = True
    assert scoped_graph_evidence(
        graph_criteria, dependency, graph_owners, changed_graph
    ) is None
    changed_route = dict(EXPECTED_FAMILY_ROUTE_PROBE)
    changed_route["permission_rewritten_as_obligation"] = True
    assert family_route_evidence(
        family_criteria, dependency, route_owners, changed_route
    ) is None
    selected = dict(dependency[0])
    selected["body"] = dict(selected["body"])
    selected["body"]["extensions_added"] = 1
    assert scoped_graph_evidence(
        graph_criteria, (selected,), graph_owners, EXPECTED_SCOPED_GRAPH_PROBE
    ) is None
    assert family_route_evidence(
        family_criteria, (selected,), route_owners, EXPECTED_FAMILY_ROUTE_PROBE
    ) is None
    dropped = dict(graph_owners)
    dropped["graph"] = dict(dropped["graph"])
    dropped["graph"]["text"] = "role edges require an explicit ordinal\n"
    assert scoped_graph_evidence(
        graph_criteria, dependency, dropped, EXPECTED_SCOPED_GRAPH_PROBE
    ) is None
    dropped_lake = dict(route_owners)
    dropped_lake["lake"] = dict(dropped_lake["lake"])
    dropped_lake["lake"]["text"] = "Kernel checking does not establish law.\n"
    assert family_route_evidence(
        family_criteria, dependency, dropped_lake, EXPECTED_FAMILY_ROUTE_PROBE
    ) is None
    for criteria in (
        _SENSE_NEXT_ACCEPTANCE,
        _FAMILY_NEXT_ACCEPTANCE,
        _MODALITY_ACCEPTANCE,
        _TRAINING_ACCEPTANCE,
    ):
        sentences = [item["criterion"] for item in criteria]
        assert scoped_graph_evidence(
            sentences, dependency, graph_owners, EXPECTED_SCOPED_GRAPH_PROBE
        ) is None
        assert family_route_evidence(
            sentences, dependency, route_owners, EXPECTED_FAMILY_ROUTE_PROBE
        ) is None


def test_release_records_graph_distinctions_and_family_route_limits(
    tmp_path: Path,
) -> None:
    def write_owners(names: dict[str, str]) -> list[str]:
        relative = []
        for name, text in names.items():
            path = tmp_path / "owners" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            relative.append("owners/" + name)
        return relative

    graph_paths = write_owners(_GRAPH_OWNER_TEXT)
    route_paths = write_owners(_ROUTE_OWNER_TEXT)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:codec",
        "LIR-CODEC",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:codec",
        evidence_kind="validation",
        digest="sha256:" + ("41" * 32),
        body=_intact_codec_dependency()[0]["body"],
    )
    _task(
        repo,
        "task:graph",
        "LIR-GRAPH",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="corpus"),
            "existing_owners": graph_paths,
        },
        dependencies=["task:codec"],
        acceptance=_GRAPH_ACCEPTANCE,
    )
    _task(
        repo,
        "task:sense-next",
        "LIR-SENSE-NEXT",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="validation"),
        dependencies=["task:graph"],
        acceptance=_SENSE_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:family",
        "LIR-FAMILY",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": route_paths,
        },
        dependencies=["task:codec"],
        acceptance=_FAMILY_ACCEPTANCE,
    )
    _task(
        repo,
        "task:family-next",
        "LIR-FAMILY-NEXT",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="validation"),
        dependencies=["task:family"],
        acceptance=_FAMILY_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:modality",
        "LIR-MODALITY",
        status="blocked",
        body=_manual_body("", stage="analysis"),
        dependencies=["task:codec"],
        acceptance=_MODALITY_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {
        "LIR-GRAPH",
        "LIR-FAMILY",
    }
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-SENSE-NEXT",
        "LIR-FAMILY-NEXT",
        "LIR-MODALITY",
    }
    for cid, operation in (
        ("task:graph", "dependency_block_resolved_scoped_graph_distinctions"),
        ("task:family", "dependency_block_resolved_family_route_limits"),
    ):
        task = repo.get_task(cid)
        assert task["status"] == "completed"
        receipt = task["body"]["completion_receipt"]
        assert receipt["operation"] == operation
        assert receipt["admitted"] is False
        assert receipt["formalized"] is False
        assert receipt["gold_or_teacher_admission_granted"] is False
        assert receipt["constitution_roundtrip_ok"] is False
        assert receipt["reviewer_claimed"] is False
        assert receipt["new_training_launched"] is False
    for cid in ("task:sense-next", "task:family-next", "task:modality"):
        assert repo.get_task(cid)["status"] == "blocked"


def _family_route_dependency():
    return (
        {
            "task_cid": "task:family",
            "digest": "sha256:" + ("42" * 32),
            "body": {
                "operation": "dependency_block_resolved_family_route_limits",
                "lake_build_executed": False,
                "cross_family_equivalence_verified": False,
                "permission_rewritten_as_obligation": False,
                "supported_routes": ["deontic"],
                "admitted": False,
                "formalized": False,
                "gold_or_teacher_admission_granted": False,
                "constitution_roundtrip_ok": False,
            },
        },
    )


_LOWERING_OWNER_TEXT = {
    "legal_relational_norm_v1.py": (
        "This variables-only fragment does not translate statutory text.\n"
        "O/P/F remain independent interpretation parameters.\n"
        "ordered distinct bounded argument roles required\n"
        "exact O/P/F payload and features required\n"
    ),
    "legal_counted_frame_v1.py": (
        "TimeFrame is an arbitrary index with no order, duration, simultaneity "
        "or calendar interpretation.\n"
        "recognized_typed_FOL_to_specialized_Lean_text\n"
        "native_TDFOL_route_used\n"
        "Only the existing fixed authored counted frame is represented\n"
    ),
    "legal_family_routes.py": (
        "does not make it equivalent to the legal rule.\n"
        "Quantifiers/cognition/events are deliberately outside this route.\n"
        "calendar_to_FOL_or_LTL_without_clock_translation\n"
        "DCEC_quantified_or_cognitive_or_event_semantics\n"
    ),
}
_CORPUS_ACCEPTANCE = [
    {
        "criterion": (
            "Reuse the existing legal Lake library target; every declaration "
            "maps to graph/interpretation/source roots."
        ),
    },
    {
        "criterion": "One coherent module corpus does not require one enormous file.",
    },
]


def test_family_lowering_probe_matches_the_live_owners() -> None:
    assert measure_family_lowering_probe() == EXPECTED_FAMILY_LOWERING_PROBE


def test_family_lowering_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _FAMILY_NEXT_ACCEPTANCE]
    owners = {
        "relational": {
            "path": "owners/legal_relational_norm_v1.py",
            "sha256": "abc",
            "text": _LOWERING_OWNER_TEXT["legal_relational_norm_v1.py"],
        },
        "counted": {
            "path": "owners/legal_counted_frame_v1.py",
            "sha256": "abc",
            "text": _LOWERING_OWNER_TEXT["legal_counted_frame_v1.py"],
        },
        "routes": {
            "path": "owners/legal_family_routes.py",
            "sha256": "abc",
            "text": _LOWERING_OWNER_TEXT["legal_family_routes.py"],
        },
    }
    dependency = _family_route_dependency()
    evidence = family_lowering_evidence(
        criteria, dependency, owners, EXPECTED_FAMILY_LOWERING_PROBE
    )
    assert evidence is not None
    assert evidence["profiles_share_support_boundary"] is False
    assert evidence["equality_translated_into_dcec"] is False
    assert evidence["counted_calendar_interpretation"] is False
    assert evidence["lake_build_executed"] is False
    assert evidence["admitted"] is False
    changed = dict(EXPECTED_FAMILY_LOWERING_PROBE)
    changed["profiles_share_support_boundary"] = True
    assert family_lowering_evidence(
        criteria, dependency, owners, changed
    ) is None
    promoted = dict(dependency[0])
    promoted["body"] = dict(promoted["body"])
    promoted["body"]["lake_build_executed"] = True
    assert family_lowering_evidence(
        criteria, (promoted,), owners, EXPECTED_FAMILY_LOWERING_PROBE
    ) is None
    dropped = dict(owners)
    dropped["routes"] = dict(dropped["routes"])
    dropped["routes"]["text"] = "does not make it equivalent to the legal rule.\n"
    assert family_lowering_evidence(
        criteria, dependency, dropped, EXPECTED_FAMILY_LOWERING_PROBE
    ) is None
    for other in (
        _GRAPH_ACCEPTANCE,
        _FAMILY_ACCEPTANCE,
        _SENSE_NEXT_ACCEPTANCE,
        _MODALITY_ACCEPTANCE,
        _TRAINING_ACCEPTANCE,
        _CORPUS_ACCEPTANCE,
    ):
        assert family_lowering_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_FAMILY_LOWERING_PROBE,
        ) is None


def test_release_records_family_lowering_boundaries(tmp_path: Path) -> None:
    relative = []
    for name, text in _LOWERING_OWNER_TEXT.items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        relative.append("owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:family",
        "LIR-FAMILY",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:family",
        evidence_kind="validation",
        digest="sha256:" + ("42" * 32),
        body=_family_route_dependency()[0]["body"],
    )
    _task(
        repo,
        "task:lowering",
        "LIR-LOWERING",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:family"],
        acceptance=_FAMILY_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:corpus-next",
        "LIR-CORPUS-NEXT",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="analysis"),
        dependencies=["task:lowering"],
        acceptance=_CORPUS_ACCEPTANCE,
    )
    _task(
        repo,
        "task:sense-next",
        "LIR-SENSE-NEXT",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:family"],
        acceptance=_SENSE_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:modality",
        "LIR-MODALITY",
        status="blocked",
        body=_manual_body("", stage="analysis"),
        dependencies=["task:family"],
        acceptance=_MODALITY_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-LOWERING"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-CORPUS-NEXT",
        "LIR-SENSE-NEXT",
        "LIR-MODALITY",
    }
    task = repo.get_task("task:lowering")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_family_lowering_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["reviewer_claimed"] is False
    assert receipt["new_training_launched"] is False
    for cid in ("task:corpus-next", "task:sense-next", "task:modality"):
        assert repo.get_task(cid)["status"] == "blocked"


_CORRESPONDENCE_OWNER_TEXT = {
    "corpus_definition_use_v2.py": (
        "A whole local predicate is aliased by an unreviewed assumption; "
        "no identity or legal implication is proved.\n"
        "all body roles and contexts remain retained.\n"
        "explicit_unreviewed_unary_biconditional_assumption\n"
    ),
    "legal_source_definition_norm_bridge.py": (
        "qualifiers, objects and extra modals unsupported\n"
        "extra text cannot be discarded\n"
        "bounded literal category name without hidden qualifiers required\n"
    ),
    "legal_source_typed_norm_join_v1.py": (
        "All original unresolved source issues remain unchanged, including "
        "those touched by a typed hypothesis.\n"
        "Predicate argument roles, lexical binding, quantifier order and "
        "modal/connective structure are syntax facts, not resolved legal "
        "entities.\n"
        "admitted graph, native execution or qualified target is supplied.\n"
    ),
}
_LOOKUP_ACCEPTANCE = [
    {
        "criterion": (
            "Retain full searched partitions, routes, unavailable/empty "
            "states and competing candidates."
        ),
    },
    {
        "criterion": (
            "Caller-supplied nearest-scope lookup is not evidence that the "
            "scope governs the law; all uses need justified bindings."
        ),
    },
]


def _scoped_graph_dependency():
    return (
        {
            "task_cid": "task:graph",
            "digest": "sha256:" + ("43" * 32),
            "body": {
                "operation": "dependency_block_resolved_scoped_graph_distinctions",
                "senses_merged": False,
                "category_implies_person": False,
                "normative_action_establishes_event": False,
                "graph_admitted": False,
                "admitted": False,
                "formalized": False,
                "gold_or_teacher_admission_granted": False,
                "constitution_roundtrip_ok": False,
            },
        },
    )


def test_definition_correspondence_probe_matches_the_live_owners() -> None:
    assert (
        measure_definition_correspondence_probe()
        == EXPECTED_DEFINITION_CORRESPONDENCE_PROBE
    )


def test_definition_correspondence_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _SENSE_NEXT_ACCEPTANCE]
    owners = {
        "definition_use": {
            "path": "owners/corpus_definition_use_v2.py",
            "sha256": "a" * 64,
            "text": _CORRESPONDENCE_OWNER_TEXT["corpus_definition_use_v2.py"],
        },
        "definition_norm": {
            "path": "owners/legal_source_definition_norm_bridge.py",
            "sha256": "b" * 64,
            "text": _CORRESPONDENCE_OWNER_TEXT[
                "legal_source_definition_norm_bridge.py"
            ],
        },
        "join": {
            "path": "owners/legal_source_typed_norm_join_v1.py",
            "sha256": "c" * 64,
            "text": _CORRESPONDENCE_OWNER_TEXT["legal_source_typed_norm_join_v1.py"],
        },
    }
    dependency = _scoped_graph_dependency()
    evidence = definition_correspondence_evidence(
        criteria,
        dependency,
        owners,
        EXPECTED_DEFINITION_CORRESPONDENCE_PROBE,
    )
    assert evidence is not None
    assert evidence["alias_counted_as_operative_norm"] is False
    assert evidence["qualifier_text_discarded"] is False
    assert evidence["qualifier_values_null"] is True
    assert evidence["alternatives_share_edge"] is False
    assert evidence["graph_admitted"] is False
    assert evidence["graph_norm_assertion_count"] == 0
    assert evidence["source_identity_asserted"] is False
    assert evidence["native_build_executed"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["reviewer_claimed"] is False
    assert "meanings_approved" not in evidence
    changed = dict(EXPECTED_DEFINITION_CORRESPONDENCE_PROBE)
    changed["alias_counted_as_operative_norm"] = True
    assert definition_correspondence_evidence(
        criteria, dependency, owners, changed
    ) is None
    promoted = dict(dependency[0])
    promoted["body"] = dict(promoted["body"])
    promoted["body"]["senses_merged"] = True
    assert definition_correspondence_evidence(
        criteria,
        (promoted,),
        owners,
        EXPECTED_DEFINITION_CORRESPONDENCE_PROBE,
    ) is None
    dropped = dict(owners)
    dropped["definition_norm"] = dict(dropped["definition_norm"])
    dropped["definition_norm"]["text"] = "qualifiers unsupported\n"
    assert definition_correspondence_evidence(
        criteria,
        dependency,
        dropped,
        EXPECTED_DEFINITION_CORRESPONDENCE_PROBE,
    ) is None
    for other in (
        _GRAPH_ACCEPTANCE,
        _FAMILY_ACCEPTANCE,
        _FAMILY_NEXT_ACCEPTANCE,
        _MODALITY_ACCEPTANCE,
        _TRAINING_ACCEPTANCE,
        _CORPUS_ACCEPTANCE,
        _LOOKUP_ACCEPTANCE,
    ):
        assert definition_correspondence_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_DEFINITION_CORRESPONDENCE_PROBE,
        ) is None


def test_release_records_definition_correspondence(tmp_path: Path) -> None:
    relative = []
    for name, text in _CORRESPONDENCE_OWNER_TEXT.items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        relative.append("owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:graph",
        "LIR-GRAPH",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:graph",
        evidence_kind="validation",
        digest="sha256:" + ("43" * 32),
        body=_scoped_graph_dependency()[0]["body"],
    )
    _task(
        repo,
        "task:correspondence",
        "LIR-CORRESPONDENCE",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:graph"],
        acceptance=_SENSE_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:lookup-next",
        "LIR-LOOKUP-NEXT",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="validation"),
        dependencies=["task:correspondence"],
        acceptance=_LOOKUP_ACCEPTANCE,
    )
    _task(
        repo,
        "task:modality",
        "LIR-MODALITY",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:graph"],
        acceptance=_MODALITY_ACCEPTANCE,
    )
    _task(
        repo,
        "task:training",
        "LIR-TRAINING",
        status="blocked",
        body=_manual_body("", stage="training"),
        dependencies=["task:graph"],
        acceptance=_TRAINING_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {
        "LIR-CORRESPONDENCE"
    }
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-LOOKUP-NEXT",
        "LIR-MODALITY",
        "LIR-TRAINING",
    }
    task = repo.get_task("task:correspondence")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_definition_correspondence"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["reviewer_claimed"] is False
    assert "meanings_approved" not in receipt
    for cid in ("task:lookup-next", "task:modality", "task:training"):
        assert repo.get_task(cid)["status"] == "blocked"


_LOOKUP_OWNER_TEXT = {
    "scoped_definition_resolution.py": (
        "including empty tables and\n"
        "unavailable placeholders.\n"
        "It does not discover governing scopes or legal\n"
        "precedence.\n"
        "All supplied query partitions are conservative dependencies, not an\n"
        "exact trace of the resolver's searches.\n"
        "Equal unknown editions do not establish governing compatibility.\n"
    ),
    "legal_source_norm_scope_v1.py": (
        "This adapter preserves a caller's existing normative-scope declaration.\n"
        "No approval necessity, cardinality frame, purpose, antecedent, "
        "plural identity, applicability or governing definition use is decided.\n"
        '"accepted_lookup": None, "governing_definition_use": None,\n'
    ),
}
_CLOSURE_ACCEPTANCE = [
    {
        "criterion": (
            "Complete closure or explicit blocked frontier is reported for "
            "the selected component."
        ),
    },
    {
        "criterion": (
            "Conflicting interpretations are not indiscriminately unioned; "
            "new competing definitions invalidate former unique resolutions."
        ),
    },
]


def _correspondence_dependency():
    return (
        {
            "task_cid": "task:correspondence",
            "digest": "sha256:" + ("44" * 32),
            "body": {
                "operation": "dependency_block_resolved_definition_correspondence",
                "alias_counted_as_operative_norm": False,
                "graph_admitted": False,
                "qualifier_text_discarded": False,
                "alternatives_share_edge": False,
                "source_identity_asserted": False,
                "admitted": False,
                "formalized": False,
                "gold_or_teacher_admission_granted": False,
                "constitution_roundtrip_ok": False,
            },
        },
    )


def test_lookup_partition_probe_matches_the_live_owners() -> None:
    assert measure_lookup_partition_probe() == EXPECTED_LOOKUP_PARTITION_PROBE


def test_lookup_partition_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _LOOKUP_ACCEPTANCE]
    owners = {
        "lookup": {
            "path": "owners/scoped_definition_resolution.py",
            "sha256": "d" * 64,
            "text": _LOOKUP_OWNER_TEXT["scoped_definition_resolution.py"],
        },
        "scope": {
            "path": "owners/legal_source_norm_scope_v1.py",
            "sha256": "e" * 64,
            "text": _LOOKUP_OWNER_TEXT["legal_source_norm_scope_v1.py"],
        },
    }
    dependency = _correspondence_dependency()
    evidence = lookup_partition_evidence(
        criteria,
        dependency,
        owners,
        EXPECTED_LOOKUP_PARTITION_PROBE,
    )
    assert evidence is not None
    assert evidence["governing_scope_verified"] is False
    assert evidence["semantic_definition_binding_verified"] is False
    assert evidence["justified_binding_count"] == 0
    assert evidence["competing_selection_made"] is False
    assert evidence["competing_candidate_ids"] == ["term:first", "term:second"]
    assert evidence["empty_status"] == "unresolved"
    assert evidence["unavailable_status"] == "unavailable"
    assert evidence["route_selection_admitted"] is False
    assert evidence["accepted_lookup_supplied"] is False
    assert evidence["governing_definition_use_supplied"] is False
    assert evidence["searched_partition_count"] == 3
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["reviewer_claimed"] is False
    assert "meanings_approved" not in evidence
    changed = dict(EXPECTED_LOOKUP_PARTITION_PROBE)
    changed["governing_scope_verified"] = True
    assert lookup_partition_evidence(
        criteria, dependency, owners, changed
    ) is None
    promoted = dict(dependency[0])
    promoted["body"] = dict(promoted["body"])
    promoted["body"]["alias_counted_as_operative_norm"] = True
    assert lookup_partition_evidence(
        criteria,
        (promoted,),
        owners,
        EXPECTED_LOOKUP_PARTITION_PROBE,
    ) is None
    dropped = dict(owners)
    dropped["scope"] = dict(dropped["scope"])
    dropped["scope"]["text"] = "normative-scope declaration\n"
    assert lookup_partition_evidence(
        criteria,
        dependency,
        dropped,
        EXPECTED_LOOKUP_PARTITION_PROBE,
    ) is None
    for other in (
        _GRAPH_ACCEPTANCE,
        _FAMILY_ACCEPTANCE,
        _FAMILY_NEXT_ACCEPTANCE,
        _SENSE_NEXT_ACCEPTANCE,
        _MODALITY_ACCEPTANCE,
        _TRAINING_ACCEPTANCE,
        _CORPUS_ACCEPTANCE,
        _CLOSURE_ACCEPTANCE,
    ):
        assert lookup_partition_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_LOOKUP_PARTITION_PROBE,
        ) is None


def test_release_records_lookup_partition_retention(tmp_path: Path) -> None:
    relative = []
    for name, text in _LOOKUP_OWNER_TEXT.items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        relative.append("owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:correspondence",
        "LIR-CORRESPONDENCE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:correspondence",
        evidence_kind="validation",
        digest="sha256:" + ("44" * 32),
        body=_correspondence_dependency()[0]["body"],
    )
    _task(
        repo,
        "task:lookup",
        "LIR-LOOKUP",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:correspondence"],
        acceptance=_LOOKUP_ACCEPTANCE,
    )
    _task(
        repo,
        "task:closure-next",
        "LIR-CLOSURE-NEXT",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:lookup"],
        acceptance=_CLOSURE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:modality",
        "LIR-MODALITY",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:correspondence"],
        acceptance=_MODALITY_ACCEPTANCE,
    )
    _task(
        repo,
        "task:training",
        "LIR-TRAINING",
        status="blocked",
        body=_manual_body("", stage="training"),
        dependencies=["task:correspondence"],
        acceptance=_TRAINING_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-LOOKUP"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-CLOSURE-NEXT",
        "LIR-MODALITY",
        "LIR-TRAINING",
    }
    task = repo.get_task("task:lookup")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_lookup_partition_retention"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["reviewer_claimed"] is False
    assert "meanings_approved" not in receipt
    for cid in ("task:closure-next", "task:modality", "task:training"):
        assert repo.get_task(cid)["status"] == "blocked"


_CLOSURE_OWNER_TEXT = {
    "legal_graph_dependencies.py": (
        "Missing targets and unbound pins are explicit and prevent reuse.\n"
        "does not infer legal dependencies, establish an exhaustive legal theory, or\n"
        "prove that unchanged artifacts are semantically correct.\n"
        "A missing target is\n"
        "    retained as an open dependency rather than silently removed.\n"
        "legal_dependency_inventory_exhaustive\n"
        "mean no dependencies were declared, not that legal completeness "
        "was proved.\n"
    ),
    "scoped_definition_resolution.py": (
        "globally duplicate definition or alias identity\n"
        "do not replace the owner outcome with a new search.\n"
    ),
}


def _lookup_dependency():
    return (
        {
            "task_cid": "task:lookup",
            "digest": "sha256:" + ("45" * 32),
            "body": {
                "operation": "dependency_block_resolved_lookup_partition_retention",
                "governing_scope_verified": False,
                "semantic_definition_binding_verified": False,
                "competing_selection_made": False,
                "route_selection_admitted": False,
                "justified_binding_count": 0,
                "admitted": False,
                "formalized": False,
                "gold_or_teacher_admission_granted": False,
                "constitution_roundtrip_ok": False,
            },
        },
    )


def test_dependency_closure_probe_matches_the_live_owners() -> None:
    assert measure_dependency_closure_probe() == EXPECTED_DECLARED_CLOSURE_PROBE


def test_declared_closure_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _CLOSURE_ACCEPTANCE]
    owners = {
        "dependencies": {
            "path": "owners/legal_graph_dependencies.py",
            "sha256": "f" * 64,
            "text": _CLOSURE_OWNER_TEXT["legal_graph_dependencies.py"],
        },
        "lookup": {
            "path": "owners/scoped_definition_resolution.py",
            "sha256": "a" * 64,
            "text": _CLOSURE_OWNER_TEXT["scoped_definition_resolution.py"],
        },
    }
    dependency = _lookup_dependency()
    evidence = declared_closure_evidence(
        criteria,
        dependency,
        owners,
        EXPECTED_DECLARED_CLOSURE_PROBE,
    )
    assert evidence is not None
    assert evidence["closure_is_exhaustive_legal_theory"] is False
    assert evidence["legal_dependency_inventory_exhaustive"] is False
    assert evidence["population_complete"] is True
    assert evidence["coverage_scope"] == (
        "exact_declared_artifacts_pins_and_edges_only"
    )
    assert evidence["missing_dependency_retained"] is True
    assert evidence["missing_dependencies"] == [
        {"dependency_id": "missing-source", "dependent_id": "rule"}
    ]
    assert evidence["frontier_stale_artifact_ids"] == ["rule", "doctrine"]
    assert evidence["interpretations_unioned"] is False
    assert evidence["former_status"] == "resolved"
    assert evidence["former_selected_symbol_id"] == "term:first"
    assert evidence["revised_status"] == "ambiguous"
    assert evidence["revised_selection_absent"] is True
    assert evidence["candidate_ids"] == ["term:first", "term:new"]
    assert evidence["duplicate_identity_refused"] is True
    assert evidence["stale_count"] == 2
    assert evidence["proof_authority"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["reviewer_claimed"] is False
    assert "meanings_approved" not in evidence
    changed = dict(EXPECTED_DECLARED_CLOSURE_PROBE)
    changed["legal_dependency_inventory_exhaustive"] = True
    assert declared_closure_evidence(
        criteria, dependency, owners, changed
    ) is None
    promoted = dict(dependency[0])
    promoted["body"] = dict(promoted["body"])
    promoted["body"]["governing_scope_verified"] = True
    assert declared_closure_evidence(
        criteria,
        (promoted,),
        owners,
        EXPECTED_DECLARED_CLOSURE_PROBE,
    ) is None
    dropped = dict(owners)
    dropped["dependencies"] = dict(dropped["dependencies"])
    dropped["dependencies"]["text"] = "legal_dependency_inventory_exhaustive\n"
    assert declared_closure_evidence(
        criteria,
        dependency,
        dropped,
        EXPECTED_DECLARED_CLOSURE_PROBE,
    ) is None
    for other in (
        _GRAPH_ACCEPTANCE,
        _FAMILY_ACCEPTANCE,
        _FAMILY_NEXT_ACCEPTANCE,
        _SENSE_NEXT_ACCEPTANCE,
        _MODALITY_ACCEPTANCE,
        _TRAINING_ACCEPTANCE,
        _CORPUS_ACCEPTANCE,
        _LOOKUP_ACCEPTANCE,
    ):
        assert declared_closure_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_DECLARED_CLOSURE_PROBE,
        ) is None


def test_release_records_declared_closure_frontier(tmp_path: Path) -> None:
    closure_relative = []
    for name, text in _CLOSURE_OWNER_TEXT.items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        closure_relative.append("owners/" + name)
    lookup_relative = []
    for name, text in _LOOKUP_OWNER_TEXT.items():
        path = tmp_path / "lookup-owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        lookup_relative.append("lookup-owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:lookup",
        "LIR-LOOKUP",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:lookup",
        evidence_kind="validation",
        digest="sha256:" + ("45" * 32),
        body=_lookup_dependency()[0]["body"],
    )
    _task(
        repo,
        "task:closure",
        "LIR-CLOSURE",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": closure_relative,
        },
        dependencies=["task:lookup"],
        acceptance=_CLOSURE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:scope-next",
        "LIR-SCOPE-NEXT",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": lookup_relative,
        },
        dependencies=["task:closure"],
        acceptance=_CLOSURE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:modality",
        "LIR-MODALITY",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": closure_relative,
        },
        dependencies=["task:lookup"],
        acceptance=_MODALITY_ACCEPTANCE,
    )
    _task(
        repo,
        "task:training",
        "LIR-TRAINING",
        status="blocked",
        body=_manual_body("", stage="training"),
        dependencies=["task:lookup"],
        acceptance=_TRAINING_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-CLOSURE"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-SCOPE-NEXT",
        "LIR-MODALITY",
        "LIR-TRAINING",
    }
    task = repo.get_task("task:closure")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_declared_closure_frontier"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["reviewer_claimed"] is False
    assert "meanings_approved" not in receipt
    for cid in ("task:scope-next", "task:modality", "task:training"):
        assert repo.get_task(cid)["status"] == "blocked"


_MEANING_ACCEPTANCE = (
    {
        "criterion": (
            "All meaning-bearing material is represented or explicitly unsupported; "
            "rule counts and order are complete."
        ),
    },
    {
        "criterion": (
            "Conditions, exceptions, necessary approval, polarity and temporal "
            "attachment are preserved."
        ),
    },
)
_MEANING_OWNER_TEXT = {
    "legal_source_norm_scope_v1.py": (
        "Full literal accounting includes unresolved and unsupported text; it is not complete semantic coverage.\n"
        "every selected source character assigned a caller coverage disposition; not semantic completeness\n"
        "No approval necessity, cardinality frame, purpose, antecedent, plural identity, applicability or governing definition use is decided.\n"
    ),
    "legal_source_typed_norm_join_v1.py": (
        "All original unresolved source issues remain unchanged, including those touched by a typed hypothesis.\n"
        "Predicate argument roles, lexical binding, quantifier order and modal/connective structure are syntax facts, not resolved legal entities.\n"
        "it does not establish natural-language semantics.\n"
    ),
}


def _closure_dependency():
    return (
        {
            "task_cid": "task:closure",
            "digest": "sha256:" + ("56" * 32),
            "body": {
                "operation": "dependency_block_resolved_declared_closure_frontier",
                "legal_dependency_inventory_exhaustive": False,
                "closure_is_exhaustive_legal_theory": False,
                "interpretations_unioned": False,
                "proof_authority": False,
                "admitted": False,
                "formalized": False,
                "gold_or_teacher_admission_granted": False,
                "constitution_roundtrip_ok": False,
            },
        },
    )


def test_declared_meaning_probe_matches_the_live_owners() -> None:
    assert measure_declared_meaning_probe() == EXPECTED_DECLARED_MEANING_PROBE


def test_declared_meaning_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _MEANING_ACCEPTANCE]
    owners = {
        "scope": {
            "path": "owners/legal_source_norm_scope_v1.py",
            "sha256": "f" * 64,
            "text": _MEANING_OWNER_TEXT["legal_source_norm_scope_v1.py"],
        },
        "join": {
            "path": "owners/legal_source_typed_norm_join_v1.py",
            "sha256": "a" * 64,
            "text": _MEANING_OWNER_TEXT["legal_source_typed_norm_join_v1.py"],
        },
    }
    dependency = _closure_dependency()
    evidence = declared_meaning_evidence(
        criteria,
        dependency,
        owners,
        EXPECTED_DECLARED_MEANING_PROBE,
    )
    assert evidence is not None
    assert evidence["coverage_dispositions"] == ["represented", "unsupported"]
    assert evidence["literal_accounting_complete"] is True
    assert evidence["semantic_completeness_claimed"] is False
    assert evidence["formal_norm_semantics_encoded"] is False
    assert evidence["omitted_rule_refused"] is True
    assert evidence["reversed_rule_ids"] == ["r1", "r2"]
    assert evidence["reversed_statement_ids"] == ["r2", "r1"]
    assert evidence["order_issue"] == "statement_and_occurrence_rule_order_differ"
    assert evidence["normalization_approved"] is False
    assert evidence["conditions_attached"] is False
    assert evidence["exceptions_attached"] is False
    assert evidence["temporal_occurrence_id"] == "when"
    assert evidence["approval_necessity_decided"] is False
    assert evidence["caller_polarity"] == "P"
    assert evidence["join_modal_operator"] == "P"
    assert evidence["changed_modal_operator"] == "O"
    assert evidence["changed_polarity_distinct"] is True
    assert evidence["join_source_meaning_selected"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["reviewer_claimed"] is False
    assert "meanings_approved" not in evidence
    changed = dict(EXPECTED_DECLARED_MEANING_PROBE)
    changed["approval_necessity_decided"] = True
    assert declared_meaning_evidence(criteria, dependency, owners, changed) is None
    promoted = dict(dependency[0])
    promoted["body"] = dict(promoted["body"])
    promoted["body"]["legal_dependency_inventory_exhaustive"] = True
    assert declared_meaning_evidence(
        criteria,
        (promoted,),
        owners,
        EXPECTED_DECLARED_MEANING_PROBE,
    ) is None
    dropped = dict(owners)
    dropped["scope"] = dict(dropped["scope"])
    dropped["scope"]["text"] = "literal accounting\n"
    assert declared_meaning_evidence(
        criteria,
        dependency,
        dropped,
        EXPECTED_DECLARED_MEANING_PROBE,
    ) is None
    for other in (
        _GRAPH_ACCEPTANCE,
        _FAMILY_ACCEPTANCE,
        _FAMILY_NEXT_ACCEPTANCE,
        _SENSE_NEXT_ACCEPTANCE,
        _MODALITY_ACCEPTANCE,
        _TRAINING_ACCEPTANCE,
        _CORPUS_ACCEPTANCE,
        _LOOKUP_ACCEPTANCE,
        _CLOSURE_ACCEPTANCE,
    ):
        assert declared_meaning_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_DECLARED_MEANING_PROBE,
        ) is None


def test_release_records_declared_meaning_retention(tmp_path: Path) -> None:
    meaning_relative = []
    for name, text in _MEANING_OWNER_TEXT.items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        meaning_relative.append("owners/" + name)
    closure_relative = []
    for name, text in _CLOSURE_OWNER_TEXT.items():
        path = tmp_path / "closure-owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        closure_relative.append("closure-owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:closure",
        "LIR-CLOSURE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:closure",
        evidence_kind="validation",
        digest="sha256:" + ("56" * 32),
        body=_closure_dependency()[0]["body"],
    )
    _task(
        repo,
        "task:meaning",
        "LIR-MEANING",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="corpus"),
            "existing_owners": meaning_relative,
        },
        dependencies=["task:closure"],
        acceptance=_MEANING_ACCEPTANCE,
    )
    _task(
        repo,
        "task:review-next",
        "LIR-REVIEW-NEXT",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": closure_relative,
        },
        dependencies=["task:meaning"],
        acceptance=_CLOSURE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:modality",
        "LIR-MODALITY",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": meaning_relative,
        },
        dependencies=["task:closure"],
        acceptance=_MODALITY_ACCEPTANCE,
    )
    _task(
        repo,
        "task:training",
        "LIR-TRAINING",
        status="blocked",
        body=_manual_body("", stage="training"),
        dependencies=["task:closure"],
        acceptance=_TRAINING_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-MEANING"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-REVIEW-NEXT",
        "LIR-MODALITY",
        "LIR-TRAINING",
    }
    task = repo.get_task("task:meaning")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == "dependency_block_resolved_declared_meaning_retention"
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["reviewer_claimed"] is False
    assert "meanings_approved" not in receipt
    for cid in ("task:review-next", "task:modality", "task:training"):
        assert repo.get_task(cid)["status"] == "blocked"


_BYTE_ACCEPTANCE = [
    {
        "criterion": (
            "Reuse the fixed byte target codec/example export; the new "
            "AST-action head has its own task/codec."
        ),
    },
    {
        "criterion": (
            "Exact training/evaluation eligibility, permitted context and "
            "untouched-component policy are fixed before fitting."
        ),
    },
]
_BYTE_NEXT_ACCEPTANCE = [
    {
        "criterion": (
            "Binder/sort/arity/reference constraints and raw invalid/"
            "incomplete outputs are checked."
        ),
    },
    {
        "criterion": (
            "Fresh-head and compatible warm-start controls are retained; "
            "no direct KL across incompatible codecs."
        ),
    },
]
_BYTE_OWNER_TEXT = {
    "legal_relational_target_codec_v1.py": (
        "not a semantic action\n"
        "language, grammar-constrained decoder, source translator or training admission.\n"
    ),
    "legal_relational_decoder_example_v1.py": (
        "This adapter creates neither reviewed meaning nor training/gold eligibility.\n"
        "The exact model input contains source/context only; labels, alternative declarations and embedding metadata are separate.\n"
        "A candidate byte target is not a reviewed meaning, semantic-action decoder or training-admission decision.\n"
    ),
}


def _byte_codec_dependency():
    return (
        {
            "task_cid": "task:codec",
            "digest": "sha256:" + ("57" * 32),
            "body": {
                "operation": "dependency_block_resolved_codec_intact_no_extension",
                "extensions_added": 0,
                "meanings_requiring_extension": 0,
                "meanings_approved": False,
                "vocabulary_size": 259,
                "semantic_action_language": False,
                "admitted": False,
                "formalized": False,
                "gold_or_teacher_admission_granted": False,
                "constitution_roundtrip_ok": False,
            },
        },
    )


def test_byte_target_probe_matches_the_live_owners() -> None:
    assert measure_byte_target_eligibility_probe() == EXPECTED_BYTE_TARGET_PROBE


def test_byte_target_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _BYTE_ACCEPTANCE]
    owners = {
        "codec": {
            "path": "owners/legal_relational_target_codec_v1.py",
            "sha256": "b" * 64,
            "text": _BYTE_OWNER_TEXT["legal_relational_target_codec_v1.py"],
        },
        "example": {
            "path": "owners/legal_relational_decoder_example_v1.py",
            "sha256": "c" * 64,
            "text": _BYTE_OWNER_TEXT["legal_relational_decoder_example_v1.py"],
        },
    }
    dependency = _byte_codec_dependency()
    evidence = byte_target_eligibility_evidence(
        criteria,
        dependency,
        owners,
        EXPECTED_BYTE_TARGET_PROBE,
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_byte_target_eligibility_boundary"
    )
    assert evidence["vocabulary_size"] == 259
    assert evidence["semantic_action_language"] is False
    assert evidence["ast_action_head_added"] is False
    assert evidence["training_mask"] == 0
    assert evidence["evaluation_gold_mask"] == 0
    assert evidence["eligibility_granted"] is False
    assert evidence["fitting_executed"] is False
    assert evidence["training_executed"] is False
    assert evidence["separate_from_model_input"] is True
    assert evidence["model_input_fields"] == ["context", "source_text"]
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["reviewer_claimed"] is False
    changed = dict(EXPECTED_BYTE_TARGET_PROBE)
    changed["eligibility_granted"] = True
    assert byte_target_eligibility_evidence(
        criteria, dependency, owners, changed
    ) is None
    extended = dict(dependency[0])
    extended["body"] = dict(extended["body"])
    extended["body"]["extensions_added"] = 1
    assert byte_target_eligibility_evidence(
        criteria,
        (extended,),
        owners,
        EXPECTED_BYTE_TARGET_PROBE,
    ) is None
    dropped = dict(owners)
    dropped["example"] = dict(dropped["example"])
    dropped["example"]["text"] = "byte target\n"
    assert byte_target_eligibility_evidence(
        criteria,
        dependency,
        dropped,
        EXPECTED_BYTE_TARGET_PROBE,
    ) is None
    for other in (
        _GRAPH_ACCEPTANCE,
        _FAMILY_ACCEPTANCE,
        _FAMILY_NEXT_ACCEPTANCE,
        _SENSE_NEXT_ACCEPTANCE,
        _MODALITY_ACCEPTANCE,
        _TRAINING_ACCEPTANCE,
        _CORPUS_ACCEPTANCE,
        _LOOKUP_ACCEPTANCE,
        _CLOSURE_ACCEPTANCE,
        _MEANING_ACCEPTANCE,
    ):
        assert byte_target_eligibility_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_BYTE_TARGET_PROBE,
        ) is None


def test_release_records_byte_target_eligibility_boundary(tmp_path: Path) -> None:
    byte_relative = []
    for name, text in _BYTE_OWNER_TEXT.items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        byte_relative.append("owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:codec",
        "LIR-CODEC",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:codec",
        evidence_kind="validation",
        digest="sha256:" + ("57" * 32),
        body=_byte_codec_dependency()[0]["body"],
    )
    _task(
        repo,
        "task:byte",
        "LIR-BYTE",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": byte_relative,
        },
        dependencies=["task:codec"],
        acceptance=_BYTE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:byte-next",
        "LIR-BYTE-NEXT",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": byte_relative,
        },
        dependencies=["task:byte"],
        acceptance=_BYTE_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:modality",
        "LIR-MODALITY",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": byte_relative,
        },
        dependencies=["task:codec"],
        acceptance=_MODALITY_ACCEPTANCE,
    )
    _task(
        repo,
        "task:training",
        "LIR-TRAINING",
        status="blocked",
        body=_manual_body("", stage="training"),
        dependencies=["task:codec"],
        acceptance=_TRAINING_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-BYTE"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BYTE-NEXT",
        "LIR-MODALITY",
        "LIR-TRAINING",
    }
    task = repo.get_task("task:byte")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_byte_target_eligibility_boundary"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["reviewer_claimed"] is False
    for cid in ("task:byte-next", "task:modality", "task:training"):
        assert repo.get_task(cid)["status"] == "blocked"


_REPRESENTATION_ACCEPTANCE = [
    {
        "criterion": (
            "Zero known material omission/addition in the accepted "
            "supported pilot."
        ),
    },
    {
        "criterion": (
            "Unresolved material blocks this component; source custody and "
            "structural validity are not legal fidelity."
        ),
    },
    {
        "criterion": (
            "Require the exact supported complete interpretation review "
            "domain; existing source-response or unary-definition "
            "declarations alone cannot authenticate a rich norm/use subject."
        ),
    },
]
_GRAPH_FOLLOW_ACCEPTANCE = [
    {
        "criterion": (
            "Exact endpoints, multiplicity, role ordinals, "
            "source/interpretation/review companions and graph-to-AST "
            "structure survive."
        ),
    },
    {
        "criterion": (
            "Accepted and candidate assertions remain separate; alternative "
            "theories retain distinct identity."
        ),
    },
]


def _meaning_dependency():
    return {
        "operation": "dependency_block_resolved_declared_meaning_retention",
        "semantic_completeness_claimed": False,
        "approval_necessity_decided": False,
        "join_source_meaning_selected": False,
        "admitted": False,
        "formalized": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }


def test_unreviewed_representation_probe_matches_the_live_owner() -> None:
    measured = measure_unreviewed_representation_probe()
    assert measured is not None
    digest = measured["review_owner_sha256"]
    assert isinstance(digest, str) and len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)
    contract = {
        key: measured[key] for key in EXPECTED_UNREVIEWED_REPRESENTATION_PROBE
    }
    assert contract == EXPECTED_UNREVIEWED_REPRESENTATION_PROBE
    assert measured["independent_review_supplied"] is False
    assert measured["accepted_view_eligible"] is False
    assert measured["representation_learning_authorizes_legal_acceptance"] is False


def test_unreviewed_representation_evidence_refuses_other_tasks() -> None:
    criteria = [item["criterion"] for item in _REPRESENTATION_ACCEPTANCE]
    measured = measure_unreviewed_representation_probe()
    dependency = (
        {
            "task_cid": "task:meaning",
            "digest": "sha256:" + ("68" * 32),
            "body": _meaning_dependency(),
        },
    )
    evidence = unreviewed_representation_evidence(
        criteria, dependency, measured
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_unreviewed_representation_phase"
    )
    assert evidence["independent_review_supplied"] is False
    assert evidence["material_omission_audit_performed"] is False
    assert evidence["accepted_supported_pilot"] is False
    assert evidence["legal_fidelity_claimed"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["reviewer_claimed"] is False
    changed = dict(measured)
    changed["accepted_view_eligible"] = True
    assert unreviewed_representation_evidence(
        criteria, dependency, changed
    ) is None
    complete = dict(dependency[0])
    complete["body"] = dict(complete["body"])
    complete["body"]["semantic_completeness_claimed"] = True
    assert unreviewed_representation_evidence(
        criteria, (complete,), measured
    ) is None
    for other in (
        _BYTE_ACCEPTANCE,
        _MEANING_ACCEPTANCE,
        _GRAPH_FOLLOW_ACCEPTANCE,
        _MODALITY_ACCEPTANCE,
        _TRAINING_ACCEPTANCE,
    ):
        assert unreviewed_representation_evidence(
            [item["criterion"] for item in other],
            dependency,
            measured,
        ) is None


def test_release_records_unreviewed_representation_phase(tmp_path: Path) -> None:
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:meaning",
        "LIR-MEANING",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:meaning",
        evidence_kind="validation",
        digest="sha256:" + ("68" * 32),
        body=_meaning_dependency(),
    )
    _task(
        repo,
        "task:representation",
        "LIR-REPRESENTATION",
        status="blocked",
        body=_manual_body(
            "Requires actual independent review of the complete assembled interpretation.",
            stage="semantic_review",
        ),
        dependencies=["task:meaning"],
        acceptance=_REPRESENTATION_ACCEPTANCE,
    )
    _task(
        repo,
        "task:graph",
        "LIR-GRAPH",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="validation"),
        dependencies=["task:representation"],
        acceptance=_GRAPH_FOLLOW_ACCEPTANCE,
    )
    _task(
        repo,
        "task:held-review",
        "LIR-HELD-REVIEW",
        status="blocked",
        body=_manual_body(
            "Requires actual independent review of the assembled interpretation.",
            stage="semantic_review",
        ),
        dependencies=["task:meaning"],
    )
    _task(
        repo,
        "task:modality",
        "LIR-MODALITY",
        status="blocked",
        body=_manual_body("", stage="analysis"),
        dependencies=["task:meaning"],
        acceptance=_MODALITY_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {
        "LIR-REPRESENTATION"
    }
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-GRAPH",
        "LIR-MODALITY",
    }
    assert {item["task_alias"] for item in result["held"]} == {"LIR-HELD-REVIEW"}
    task = repo.get_task("task:representation")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_unreviewed_representation_phase"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["reviewer_claimed"] is False
    for cid in ("task:graph", "task:held-review", "task:modality"):
        assert repo.get_task(cid)["status"] == "blocked"


_ROUNDTRIP_ACCEPTANCE = [
    {
        "criterion": (
            "Exact endpoints, multiplicity, role ordinals, "
            "source/interpretation/review companions and graph-to-AST "
            "structure survive."
        ),
    },
    {
        "criterion": (
            "Accepted and candidate assertions remain separate; alternative "
            "theories retain distinct identity."
        ),
    },
]
_SEAL_ACCEPTANCE = [
    {
        "criterion": "All referenced records are present and independently committed.",
    },
    {
        "criterion": (
            "Snapshot custody does not promote candidate legal assertions "
            "or unsupported imports."
        ),
    },
]
_ROUNDTRIP_OWNER_TEXT = {
    "corpus_graph.py": "\n".join(
        (
            "closed endpoints",
            "dangling graph endpoint; external imports unsupported in v1",
            "role edges require an explicit ordinal",
            "duplicate role argument position",
            "alternative group and identity must be supplied together",
            "edge crosses mutually exclusive interpretations",
            "v1 graph snapshots are candidate-only",
            "Semantic ASTs and review receipts remain separately content-addressed inputs.",
            "source_occurrence_cids",
            "interpretation_cids",
        )
    ),
    "corpus_graph_exchange.py": "\n".join(
        (
            "Parquet records do not reconstruct the exact sealed snapshot",
            "No dataset upload or legal admission occurs here.",
            "writer readback differs from input snapshot",
            "graph bundle cannot claim authority",
        )
    ),
}


def _representation_dependency():
    return {
        "operation": "dependency_block_resolved_unreviewed_representation_phase",
        "independent_review_supplied": False,
        "accepted_view_eligible": False,
        "legal_fidelity_claimed": False,
        "representation_learning_authorizes_legal_acceptance": False,
        "admitted": False,
        "formalized": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }


def _graph_owners():
    return {
        "graph": {
            "path": "owners/corpus_graph.py",
            "sha256": "a" * 64,
            "text": _ROUNDTRIP_OWNER_TEXT["corpus_graph.py"],
        },
        "exchange": {
            "path": "owners/corpus_graph_exchange.py",
            "sha256": "b" * 64,
            "text": _ROUNDTRIP_OWNER_TEXT["corpus_graph_exchange.py"],
        },
    }


def test_graph_roundtrip_probe_matches_the_live_owners() -> None:
    import ipfs_datasets_py.logic.autoformal.corpus_graph as graph

    measured = measure_graph_roundtrip_probe()
    assert measured == EXPECTED_GRAPH_ROUNDTRIP_PROBE
    assert measured["accepted_assertion_count"] == 0
    assert measured["bundle_roundtrip"] is True
    assert measured["endpoints_closed"] is True
    assert measured["alternative_identities_distinct"] is True
    assert measured["alternatives_share_identity"] is False
    assert measured["admitted"] is False
    assert measured["formalized"] is False
    assert measured["source_semantics_verified"] is False
    assert measured["review_receipt_authenticated"] is False
    assert measured["review_receipt_retained"] is True
    assert measured["graph_to_ast_embedded"] is False
    assert measured["ast_node_type"] is False
    assert measured["candidate_view"] == "candidate"
    assert measured["manifest_authority_false"] is True
    root = Path(graph.__file__).parent
    graph_text = (root / "corpus_graph.py").read_text()
    exchange_text = (root / "corpus_graph_exchange.py").read_text()
    for marker in _ROUNDTRIP_OWNER_TEXT["corpus_graph.py"].splitlines():
        assert marker in graph_text
    for marker in _ROUNDTRIP_OWNER_TEXT["corpus_graph_exchange.py"].splitlines():
        assert marker in exchange_text
    receipt = dependency_resolution_receipt(
        "dependency_block_resolved_graph_roundtrip_boundaries"
    )
    assert receipt["operation"] == (
        "dependency_block_resolved_graph_roundtrip_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["reviewer_claimed"] is False
    assert receipt["completion_authority"] is False


def test_graph_roundtrip_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _ROUNDTRIP_ACCEPTANCE]
    owners = _graph_owners()
    dependency = (
        {
            "task_cid": "task:representation",
            "digest": "sha256:" + ("69" * 32),
            "body": _representation_dependency(),
        },
    )
    evidence = graph_roundtrip_evidence(
        criteria,
        dependency,
        owners,
        EXPECTED_GRAPH_ROUNDTRIP_PROBE,
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_graph_roundtrip_boundaries"
    )
    assert evidence["accepted_assertion_count"] == 0
    assert evidence["bundle_roundtrip"] is True
    assert evidence["endpoints_closed"] is True
    assert evidence["alternative_identities_distinct"] is True
    assert evidence["review_receipt_authenticated"] is False
    assert evidence["graph_to_ast_embedded"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["reviewer_claimed"] is False
    assert evidence["completion_authority"] is False
    changed = dict(EXPECTED_GRAPH_ROUNDTRIP_PROBE)
    changed["admitted"] = True
    assert graph_roundtrip_evidence(
        criteria, dependency, owners, changed
    ) is None
    reloaded = dict(EXPECTED_GRAPH_ROUNDTRIP_PROBE)
    reloaded["bundle_roundtrip"] = False
    assert graph_roundtrip_evidence(
        criteria, dependency, owners, reloaded
    ) is None
    reviewed = dict(dependency[0])
    reviewed["body"] = dict(reviewed["body"])
    reviewed["body"]["independent_review_supplied"] = True
    assert graph_roundtrip_evidence(
        criteria, (reviewed,), owners, EXPECTED_GRAPH_ROUNDTRIP_PROBE
    ) is None
    accepted = dict(dependency[0])
    accepted["body"] = dict(accepted["body"])
    accepted["body"]["accepted_view_eligible"] = True
    assert graph_roundtrip_evidence(
        criteria, (accepted,), owners, EXPECTED_GRAPH_ROUNDTRIP_PROBE
    ) is None
    claimed = dict(dependency[0])
    claimed["body"] = dict(claimed["body"])
    claimed["body"]["legal_fidelity_claimed"] = True
    assert graph_roundtrip_evidence(
        criteria, (claimed,), owners, EXPECTED_GRAPH_ROUNDTRIP_PROBE
    ) is None
    authorized = dict(dependency[0])
    authorized["body"] = dict(authorized["body"])
    authorized["body"]["representation_learning_authorizes_legal_acceptance"] = True
    assert graph_roundtrip_evidence(
        criteria, (authorized,), owners, EXPECTED_GRAPH_ROUNDTRIP_PROBE
    ) is None
    admitted = dict(dependency[0])
    admitted["body"] = dict(admitted["body"])
    admitted["body"]["admitted"] = True
    assert graph_roundtrip_evidence(
        criteria, (admitted,), owners, EXPECTED_GRAPH_ROUNDTRIP_PROBE
    ) is None
    dropped = _graph_owners()
    dropped["exchange"] = dict(dropped["exchange"])
    dropped["exchange"]["text"] = "graph bundle\n"
    assert graph_roundtrip_evidence(
        criteria, dependency, dropped, EXPECTED_GRAPH_ROUNDTRIP_PROBE
    ) is None
    blank = dict(dependency[0])
    blank["digest"] = ""
    assert graph_roundtrip_evidence(
        criteria, (blank,), owners, EXPECTED_GRAPH_ROUNDTRIP_PROBE
    ) is None
    for other in (
        _SEAL_ACCEPTANCE,
        _REPRESENTATION_ACCEPTANCE,
        _BYTE_ACCEPTANCE,
        _MODALITY_ACCEPTANCE,
        _TRAINING_ACCEPTANCE,
    ):
        assert graph_roundtrip_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_GRAPH_ROUNDTRIP_PROBE,
        ) is None


def test_release_records_graph_roundtrip_boundaries(tmp_path: Path) -> None:
    relative = []
    for name, text in _ROUNDTRIP_OWNER_TEXT.items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        relative.append("owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:representation",
        "LIR-REP",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:representation",
        evidence_kind="validation",
        digest="sha256:" + ("69" * 32),
        body=_representation_dependency(),
    )
    bad = _representation_dependency()
    bad["independent_review_supplied"] = True
    _task(
        repo,
        "task:bad-review",
        "LIR-BAD",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:bad-review",
        evidence_kind="validation",
        digest="sha256:" + ("70" * 32),
        body=bad,
    )
    _task(
        repo,
        "task:round",
        "LIR-ROUND",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:representation"],
        acceptance=_ROUNDTRIP_ACCEPTANCE,
    )
    _task(
        repo,
        "task:seal",
        "LIR-SEAL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="checkpoint"),
            "existing_owners": relative,
        },
        dependencies=["task:round"],
        acceptance=_SEAL_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare",
        "LIR-BARE",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="validation"),
        dependencies=["task:representation"],
        acceptance=_ROUNDTRIP_ACCEPTANCE,
    )
    _task(
        repo,
        "task:claimed",
        "LIR-CLAIMED",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:bad-review"],
        acceptance=_ROUNDTRIP_ACCEPTANCE,
    )
    _task(
        repo,
        "task:modality",
        "LIR-MODALITY",
        status="blocked",
        body=_manual_body("", stage="analysis"),
        dependencies=["task:representation"],
        acceptance=_MODALITY_ACCEPTANCE,
    )
    _task(
        repo,
        "task:training",
        "LIR-TRAINING",
        status="blocked",
        body=_manual_body("", stage="training"),
        dependencies=["task:representation"],
        acceptance=_TRAINING_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-ROUND"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-SEAL",
        "LIR-BARE",
        "LIR-CLAIMED",
        "LIR-MODALITY",
        "LIR-TRAINING",
    }
    assert result["held"] == []
    task = repo.get_task("task:round")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_graph_roundtrip_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["reviewer_claimed"] is False
    assert receipt["completion_authority"] is False
    for cid in (
        "task:seal",
        "task:bare",
        "task:claimed",
        "task:modality",
        "task:training",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


_CUSTODY_ACCEPTANCE = [
    {"criterion": "All referenced records are present and independently committed."},
    {
        "criterion": (
            "Snapshot custody does not promote candidate legal assertions "
            "or unsupported imports."
        ),
    },
]
_LAKE_ACCEPTANCE = [
    {
        "criterion": (
            "Reuse the existing legal Lake library target; every declaration "
            "maps to graph/interpretation/source roots."
        ),
    },
    {
        "criterion": "One coherent module corpus does not require one enormous file.",
    },
]
_CUSTODY_OWNER_TEXT = {
    "corpus_graph_exchange.py": "\n".join(
        (
            "Failed attempts retain their payloads for inspection, without a committed",
            "Existing directories/files are never overwritten or reused.",
            "incomplete bundle or unknown filenames",
            "payloads, source occurrences or evidence merely named by those contents.",
            "graph bundle cannot claim authority",
            "external_references_verified",
            "externally supplied manifest SHA256",
        )
    ),
    "legal_graph_dependencies.py": "\n".join(
        (
            "Replay a manifest only under an independently retained content pin.",
            "dependency manifest differs from independently expected identity",
            "retained as an open dependency rather than silently removed.",
            "observations must account for every artifact; use null for unavailable",
            "does not infer legal dependencies, establish an exhaustive legal theory",
            "exact_declared_artifacts_pins_and_edges_only",
            "legal_dependency_inventory_exhaustive",
        )
    ),
}


def _roundtrip_dependency():
    return {
        "operation": "dependency_block_resolved_graph_roundtrip_boundaries",
        "accepted_assertion_count": 0,
        "graph_to_ast_embedded": False,
        "review_receipt_authenticated": False,
        "bundle_roundtrip": True,
        "endpoints_closed": True,
        "alternative_identities_distinct": True,
        "admitted": False,
        "formalized": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }


def _custody_owners():
    return {
        "exchange": {
            "path": "owners/corpus_graph_exchange.py",
            "sha256": "c" * 64,
            "text": _CUSTODY_OWNER_TEXT["corpus_graph_exchange.py"],
        },
        "dependencies": {
            "path": "owners/legal_graph_dependencies.py",
            "sha256": "d" * 64,
            "text": _CUSTODY_OWNER_TEXT["legal_graph_dependencies.py"],
        },
    }


def test_snapshot_custody_probe_matches_the_live_owners() -> None:
    import ipfs_datasets_py.logic.autoformal.corpus_graph_exchange as exchange

    measured = measure_snapshot_custody_probe()
    assert measured == EXPECTED_SNAPSHOT_CUSTODY_PROBE
    assert measured["bundle_records_present"] is True
    assert measured["bundle_roundtrip"] is True
    assert measured["external_referenced_records_authenticated"] is False
    assert measured["external_payloads_in_bundle"] == 0
    assert measured["external_references_verified"] is False
    assert measured["candidate_assertions_promoted"] is False
    assert measured["admitted"] is False
    assert measured["formalized"] is False
    assert measured["source_semantics_verified"] is False
    assert measured["dependency_admitted"] is False
    assert measured["legal_dependency_inventory_exhaustive"] is False
    assert measured["proof_authority"] is False
    assert measured["population_complete"] is True
    assert measured["coverage_scope"] == (
        "exact_declared_artifacts_pins_and_edges_only"
    )
    assert measured["reusable_under_declared_dependencies"] == []
    assert measured["missing_dependency_count"] == 1
    root = Path(exchange.__file__).parent
    exchange_text = (root / "corpus_graph_exchange.py").read_text()
    dependency_text = (root / "legal_graph_dependencies.py").read_text()
    for marker in _CUSTODY_OWNER_TEXT["corpus_graph_exchange.py"].splitlines():
        assert marker in exchange_text
    for marker in _CUSTODY_OWNER_TEXT["legal_graph_dependencies.py"].splitlines():
        assert marker in dependency_text
    receipt = dependency_resolution_receipt(
        "dependency_block_resolved_snapshot_custody_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["reviewer_claimed"] is False
    assert receipt["completion_authority"] is False


def test_snapshot_custody_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _CUSTODY_ACCEPTANCE]
    owners = _custody_owners()
    dependency = (
        {
            "task_cid": "task:round",
            "digest": "sha256:" + ("71" * 32),
            "body": _roundtrip_dependency(),
        },
    )
    evidence = snapshot_custody_evidence(
        criteria,
        dependency,
        owners,
        EXPECTED_SNAPSHOT_CUSTODY_PROBE,
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_snapshot_custody_boundaries"
    )
    assert evidence["external_referenced_records_authenticated"] is False
    assert evidence["candidate_assertions_promoted"] is False
    assert evidence["legal_dependency_inventory_exhaustive"] is False
    assert evidence["missing_dependency_count"] == 1
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["reviewer_claimed"] is False
    assert evidence["completion_authority"] is False
    changed = dict(EXPECTED_SNAPSHOT_CUSTODY_PROBE)
    changed["external_referenced_records_authenticated"] = True
    assert snapshot_custody_evidence(
        criteria, dependency, owners, changed
    ) is None
    promoted = dict(EXPECTED_SNAPSHOT_CUSTODY_PROBE)
    promoted["candidate_assertions_promoted"] = True
    assert snapshot_custody_evidence(
        criteria, dependency, owners, promoted
    ) is None
    accepted = dict(dependency[0])
    accepted["body"] = dict(accepted["body"])
    accepted["body"]["accepted_assertion_count"] = 1
    assert snapshot_custody_evidence(
        criteria, (accepted,), owners, EXPECTED_SNAPSHOT_CUSTODY_PROBE
    ) is None
    embedded = dict(dependency[0])
    embedded["body"] = dict(embedded["body"])
    embedded["body"]["graph_to_ast_embedded"] = True
    assert snapshot_custody_evidence(
        criteria, (embedded,), owners, EXPECTED_SNAPSHOT_CUSTODY_PROBE
    ) is None
    admitted = dict(dependency[0])
    admitted["body"] = dict(admitted["body"])
    admitted["body"]["admitted"] = True
    assert snapshot_custody_evidence(
        criteria, (admitted,), owners, EXPECTED_SNAPSHOT_CUSTODY_PROBE
    ) is None
    dropped = _custody_owners()
    dropped["dependencies"] = dict(dropped["dependencies"])
    dropped["dependencies"]["text"] = "dependency manifest\n"
    assert snapshot_custody_evidence(
        criteria, dependency, dropped, EXPECTED_SNAPSHOT_CUSTODY_PROBE
    ) is None
    blank = dict(dependency[0])
    blank["digest"] = ""
    assert snapshot_custody_evidence(
        criteria, (blank,), owners, EXPECTED_SNAPSHOT_CUSTODY_PROBE
    ) is None
    for other in (
        _ROUNDTRIP_ACCEPTANCE,
        _LAKE_ACCEPTANCE,
        _BYTE_ACCEPTANCE,
        _MODALITY_ACCEPTANCE,
        _TRAINING_ACCEPTANCE,
    ):
        assert snapshot_custody_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_SNAPSHOT_CUSTODY_PROBE,
        ) is None


def test_release_records_snapshot_custody_boundaries(tmp_path: Path) -> None:
    relative = []
    for name, text in _CUSTODY_OWNER_TEXT.items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        relative.append("owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:round",
        "LIR-ROUND",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:round",
        evidence_kind="validation",
        digest="sha256:" + ("71" * 32),
        body=_roundtrip_dependency(),
    )
    promoted = _roundtrip_dependency()
    promoted["accepted_assertion_count"] = 1
    _task(
        repo,
        "task:promoted-round",
        "LIR-PROMOTED-ROUND",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:promoted-round",
        evidence_kind="validation",
        digest="sha256:" + ("72" * 32),
        body=promoted,
    )
    _task(
        repo,
        "task:custody",
        "LIR-CUSTODY",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="checkpoint"),
            "existing_owners": relative,
        },
        dependencies=["task:round"],
        acceptance=_CUSTODY_ACCEPTANCE,
    )
    _task(
        repo,
        "task:lake",
        "LIR-LAKE",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="corpus"),
        dependencies=["task:custody"],
        acceptance=_LAKE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare",
        "LIR-BARE-CUSTODY",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="checkpoint"),
        dependencies=["task:round"],
        acceptance=_CUSTODY_ACCEPTANCE,
    )
    _task(
        repo,
        "task:promoted",
        "LIR-PROMOTED",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="checkpoint"),
            "existing_owners": relative,
        },
        dependencies=["task:promoted-round"],
        acceptance=_CUSTODY_ACCEPTANCE,
    )
    _task(
        repo,
        "task:modality",
        "LIR-MODALITY",
        status="blocked",
        body=_manual_body("", stage="analysis"),
        dependencies=["task:round"],
        acceptance=_MODALITY_ACCEPTANCE,
    )
    _task(
        repo,
        "task:training",
        "LIR-TRAINING",
        status="blocked",
        body=_manual_body("", stage="training"),
        dependencies=["task:round"],
        acceptance=_TRAINING_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-CUSTODY"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-LAKE",
        "LIR-BARE-CUSTODY",
        "LIR-PROMOTED",
        "LIR-MODALITY",
        "LIR-TRAINING",
    }
    assert result["held"] == []
    task = repo.get_task("task:custody")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_snapshot_custody_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["reviewer_claimed"] is False
    assert receipt["completion_authority"] is False
    for cid in (
        "task:lake",
        "task:bare",
        "task:promoted",
        "task:modality",
        "task:training",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


_MODULE_NEXT_ACCEPTANCE = [
    {
        "criterion": (
            "Two V2 owners with shared historical schema labels are not "
            "interchangeable."
        ),
    },
    {
        "criterion": (
            "Exact commands, expected artifact population, hypotheses and "
            "source closure are independently bound before execution."
        ),
    },
]
_SHARED_MODULE_OWNER_TEXT = {
    "legal_corpus_shared_lake.py": "\n".join(
        (
            '[[lean_lib]]\\nname = "legal"\\nroots = ["LegalCorpus"]',
            '"interpretation_cid": candidate["report_record"]["record_cid"]',
            "source_occurrence_cid",
            "unsupported or oversized complete shared Lean project",
            "Share exact explicit definitions while preserving every original norm.",
        )
    ),
    "legal_corpus_definition_use_lake_v2.py": "\n".join(
        (
            "No process, toolchain probe, build or admission occurs here.",
            "exact distinct native module population required",
            '[[lean_lib]]\\nname = "legal"',
            '"processes_executed": 0',
            '"build_eligible": False',
        )
    ),
}


def _unpromoted_custody():
    return {
        "operation": "dependency_block_resolved_snapshot_custody_boundaries",
        "external_referenced_records_authenticated": False,
        "candidate_assertions_promoted": False,
        "external_references_verified": False,
        "legal_dependency_inventory_exhaustive": False,
        "proof_authority": False,
        "admitted": False,
        "formalized": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
        "bundle_roundtrip": True,
        "bundle_records_present": True,
        "missing_dependency_count": 1,
        "population_complete": True,
        "coverage_scope": "exact_declared_artifacts_pins_and_edges_only",
        "lake_build_executed": False,
    }


def _shared_module_owners():
    return {
        "shared": {
            "path": "owners/legal_corpus_shared_lake.py",
            "sha256": "e" * 64,
            "text": _SHARED_MODULE_OWNER_TEXT["legal_corpus_shared_lake.py"],
        },
        "definition_use": {
            "path": "owners/legal_corpus_definition_use_lake_v2.py",
            "sha256": "f" * 64,
            "text": _SHARED_MODULE_OWNER_TEXT[
                "legal_corpus_definition_use_lake_v2.py"
            ],
        },
    }


def test_shared_module_probe_matches_the_live_owners() -> None:
    import ipfs_datasets_py.logic.autoformal.legal_corpus_definition_use_lake_v2 as v2
    import ipfs_datasets_py.logic.autoformal.legal_corpus_shared_lake as shared

    measured = measure_shared_module_probe()
    assert measured == EXPECTED_SHARED_MODULE_PROBE
    assert measured["preparation_only"] is True
    assert measured["lake_build_executed"] is False
    assert measured["library_target"] == "legal"
    assert measured["library_roots"] == ["LegalCorpus"]
    assert measured["one_file_corpus"] is False
    assert measured["definition_module_count"] == 1
    assert measured["norm_module_count"] == 2
    assert measured["file_count"] == 8
    assert measured["lean_file_count"] == 5
    assert measured["declaration_graph_roots_retained"] is True
    assert measured["declaration_interpretation_roots_retained"] is True
    assert measured["declaration_source_roots_retained"] is True
    assert measured["shared_definition_deduplication"] is True
    assert measured["external_definition_resolution"] is False
    assert measured["admitted"] is False
    assert measured["formalized"] is False
    assert measured["proof_authority"] is False
    assert measured["source_semantics_verified"] is False
    assert measured["backend_executed"] is False
    assert measured["lean_compilation_verified"] is False
    assert measured["v2_unpinned_toolchain_refused"] == (
        "exact version-pinned Lean toolchain string required"
    )
    assert v2.FALSE_FLAGS["build_eligible"] is False
    assert v2.FALSE_FLAGS["kernel_proofs_checked"] is False
    assert v2.FALSE_FLAGS["toolchain_probed"] is False
    assert measured["v2_build_eligible"] is True
    assert measured["v2_kernel_proofs_checked"] is True
    assert measured["v2_toolchain_probed"] is True
    assert measured["v2_max_files"] == 9
    root = Path(shared.__file__).parent
    shared_text = (root / "legal_corpus_shared_lake.py").read_text()
    definition_text = (root / "legal_corpus_definition_use_lake_v2.py").read_text()
    assert len(shared_text.encode()) <= 2_000_000
    assert len(definition_text.encode()) <= 2_000_000
    for marker in _SHARED_MODULE_OWNER_TEXT["legal_corpus_shared_lake.py"].splitlines():
        assert marker in shared_text
    for marker in _SHARED_MODULE_OWNER_TEXT[
        "legal_corpus_definition_use_lake_v2.py"
    ].splitlines():
        assert marker in definition_text
    receipt = dependency_resolution_receipt(
        "dependency_block_resolved_shared_module_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["reviewer_claimed"] is False
    assert receipt["completion_authority"] is False


def test_shared_module_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _LAKE_ACCEPTANCE]
    owners = _shared_module_owners()
    dependency = (
        {
            "task_cid": "task:custody",
            "digest": "sha256:" + ("81" * 32),
            "body": _unpromoted_custody(),
        },
    )
    evidence = shared_module_evidence(
        criteria,
        dependency,
        owners,
        EXPECTED_SHARED_MODULE_PROBE,
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_shared_module_boundaries"
    )
    assert evidence["preparation_only"] is True
    assert evidence["lake_build_executed"] is False
    assert evidence["library_target"] == "legal"
    assert evidence["one_file_corpus"] is False
    assert evidence["declaration_graph_roots_retained"] is True
    assert evidence["declaration_interpretation_roots_retained"] is True
    assert evidence["declaration_source_roots_retained"] is True
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["reviewer_claimed"] is False
    assert evidence["completion_authority"] is False
    built = dict(EXPECTED_SHARED_MODULE_PROBE)
    built["lake_build_executed"] = True
    assert shared_module_evidence(criteria, dependency, owners, built) is None
    collapsed = dict(EXPECTED_SHARED_MODULE_PROBE)
    collapsed["one_file_corpus"] = True
    assert shared_module_evidence(criteria, dependency, owners, collapsed) is None
    admitted_probe = dict(EXPECTED_SHARED_MODULE_PROBE)
    admitted_probe["admitted"] = True
    assert shared_module_evidence(
        criteria, dependency, owners, admitted_probe
    ) is None
    promoted = _unpromoted_custody()
    promoted["candidate_assertions_promoted"] = True
    assert shared_module_evidence(
        criteria,
        ({"task_cid": "task:custody", "digest": "sha256:" + ("81" * 32), "body": promoted},),
        owners,
        EXPECTED_SHARED_MODULE_PROBE,
    ) is None
    lake = _unpromoted_custody()
    lake["lake_build_executed"] = True
    assert shared_module_evidence(
        criteria,
        ({"task_cid": "task:custody", "digest": "sha256:" + ("81" * 32), "body": lake},),
        owners,
        EXPECTED_SHARED_MODULE_PROBE,
    ) is None
    wider = _unpromoted_custody()
    wider["coverage_scope"] = "every_external_record"
    assert shared_module_evidence(
        criteria,
        ({"task_cid": "task:custody", "digest": "sha256:" + ("81" * 32), "body": wider},),
        owners,
        EXPECTED_SHARED_MODULE_PROBE,
    ) is None
    authenticated = _unpromoted_custody()
    authenticated["external_referenced_records_authenticated"] = True
    assert shared_module_evidence(
        criteria,
        (
            {
                "task_cid": "task:custody",
                "digest": "sha256:" + ("81" * 32),
                "body": authenticated,
            },
        ),
        owners,
        EXPECTED_SHARED_MODULE_PROBE,
    ) is None
    admitted = _unpromoted_custody()
    admitted["admitted"] = True
    assert shared_module_evidence(
        criteria,
        ({"task_cid": "task:custody", "digest": "sha256:" + ("81" * 32), "body": admitted},),
        owners,
        EXPECTED_SHARED_MODULE_PROBE,
    ) is None
    dropped = _shared_module_owners()
    dropped["shared"] = dict(dropped["shared"])
    dropped["shared"]["text"] = "source_occurrence_cid\n"
    assert shared_module_evidence(
        criteria, dependency, dropped, EXPECTED_SHARED_MODULE_PROBE
    ) is None
    blank = dict(dependency[0])
    blank["digest"] = ""
    assert shared_module_evidence(
        criteria, (blank,), owners, EXPECTED_SHARED_MODULE_PROBE
    ) is None
    assert shared_module_evidence(
        criteria,
        (
            {
                "task_cid": "task:round",
                "digest": "sha256:" + ("71" * 32),
                "body": _roundtrip_dependency(),
            },
        ),
        owners,
        EXPECTED_SHARED_MODULE_PROBE,
    ) is None
    for other in (
        _CUSTODY_ACCEPTANCE,
        _BYTE_NEXT_ACCEPTANCE,
        _MODALITY_ACCEPTANCE,
        _MODULE_NEXT_ACCEPTANCE,
        _ROUNDTRIP_ACCEPTANCE,
    ):
        assert shared_module_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_SHARED_MODULE_PROBE,
        ) is None


def test_release_records_shared_module_boundaries(tmp_path: Path) -> None:
    relative = []
    for name, text in _SHARED_MODULE_OWNER_TEXT.items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        relative.append("owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:custody",
        "LIR-CUSTODY-DEP",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:custody",
        evidence_kind="validation",
        digest="sha256:" + ("81" * 32),
        body=_unpromoted_custody(),
    )
    promoted = _unpromoted_custody()
    promoted["candidate_assertions_promoted"] = True
    _task(
        repo,
        "task:promoted-custody",
        "LIR-PROMOTED-CUSTODY",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:promoted-custody",
        evidence_kind="validation",
        digest="sha256:" + ("82" * 32),
        body=promoted,
    )
    _task(
        repo,
        "task:shared",
        "LIR-SHARED",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="corpus"),
            "existing_owners": relative,
        },
        dependencies=["task:custody"],
        acceptance=_LAKE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare",
        "LIR-BARE-MODULE",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="corpus"),
        dependencies=["task:custody"],
        acceptance=_LAKE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:promoted",
        "LIR-PROMOTED-MODULE",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="corpus"),
            "existing_owners": relative,
        },
        dependencies=["task:promoted-custody"],
        acceptance=_LAKE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:next",
        "LIR-MODULE-NEXT",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="corpus"),
        dependencies=["task:shared"],
        acceptance=_MODULE_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:byte",
        "LIR-BYTE-NEXT",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="validation"),
        dependencies=["task:custody"],
        acceptance=_BYTE_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:modality",
        "LIR-MODALITY",
        status="blocked",
        body=_manual_body("", stage="analysis"),
        dependencies=["task:custody"],
        acceptance=_MODALITY_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-SHARED"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-MODULE",
        "LIR-PROMOTED-MODULE",
        "LIR-MODULE-NEXT",
        "LIR-BYTE-NEXT",
        "LIR-MODALITY",
    }
    assert result["held"] == []
    task = repo.get_task("task:shared")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_shared_module_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["reviewer_claimed"] is False
    assert receipt["completion_authority"] is False
    for cid in (
        "task:bare",
        "task:promoted",
        "task:next",
        "task:byte",
        "task:modality",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


_ENVELOPE_ACCEPTANCE = [
    {
        "criterion": (
            "Two V2 owners with shared historical schema labels are not "
            "interchangeable."
        ),
    },
    {
        "criterion": (
            "Exact commands, expected artifact population, hypotheses and "
            "source closure are independently bound before execution."
        ),
    },
]
_BUILD_NEXT_ACCEPTANCE = [
    {
        "criterion": (
            "Actual command succeeds with every expected nonempty compiled "
            "artifact and complete import closure."
        ),
    },
    {
        "criterion": (
            "No sorry, admit or hidden placeholders; unchanged inputs and "
            "durable resource closeout are recorded."
        ),
    },
]
_V2_ENVELOPE_OWNER_TEXT = {
    "legal_corpus_definition_use_native_v2.py": "\n".join(
        (
            'BUILD_SCHEMA = "legal-corpus-v2-definition-use-native-build/v1"',
            'PROFILE = "replayed-v2-definition-use-conditional-native-build/v1"',
            'command = (str(executable), "--offline", "--no-cache", "build", "legal")',
            'modules = manifest["expected_modules"]',
            'source_files = {name: body.encode("utf-8") for name, body in checked.files}',
            'manifest["theorem_assumptions"]',
            "carries every real hypothesis without its obsolete execution-status text.",
            '"context_closure_verified"',
            "Public import uses only the standard library.",
        )
    ),
    "legal_corpus_definition_use_build_v2.py": "\n".join(
        (
            'SCHEMA = "legal-corpus-v2-definition-use-native-build/v1"',
            'PROFILE = "bounded-v2-definition-use-conditional-build/v1"',
            'command = (str(lake), "--offline", "--no-cache", "build", "legal")',
            'len(paths) == len(manifest["expected_modules"]) == MAX_MODULES',
            '"theorem_assumptions"',
            "input_files=dict(checked.files)",
            '"toolchain_library_closure_verified": False',
            '"admission_build_eligible": False',
            "Import performs no toolchain discovery or process execution.",
        )
    ),
}


def _prepared_shared_module():
    return {
        "operation": "dependency_block_resolved_shared_module_boundaries",
        "preparation_only": True,
        "lake_build_executed": False,
        "one_file_corpus": False,
        "library_target": "legal",
        "admitted": False,
        "formalized": False,
        "proof_authority": False,
        "backend_executed": False,
        "lean_compilation_verified": False,
        "source_semantics_verified": False,
        "independently_reviewed": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }


def _v2_envelope_owners():
    return {
        "native": {
            "path": "owners/legal_corpus_definition_use_native_v2.py",
            "sha256": "a" * 64,
            "text": _V2_ENVELOPE_OWNER_TEXT[
                "legal_corpus_definition_use_native_v2.py"
            ],
        },
        "build": {
            "path": "owners/legal_corpus_definition_use_build_v2.py",
            "sha256": "b" * 64,
            "text": _V2_ENVELOPE_OWNER_TEXT[
                "legal_corpus_definition_use_build_v2.py"
            ],
        },
    }


def test_v2_owner_envelope_probe_matches_the_live_owners() -> None:
    import ipfs_datasets_py.logic.autoformal.legal_corpus_definition_use_native_v2 as native

    measured = measure_v2_owner_envelope_probe()
    assert measured == EXPECTED_V2_OWNER_ENVELOPE_PROBE
    assert measured["owners_interchangeable"] is False
    assert measured["profiles_distinct"] is True
    assert measured["profile_selected"] is False
    assert measured["binary_sha256_recorded"] is False
    assert measured["lake_build_executed"] is False
    assert measured["command_bound_before_execution"] is True
    assert measured["expected_artifact_population_bound_before_execution"] is True
    assert measured["hypotheses_bound_before_execution"] is True
    assert measured["source_closure_bound_before_execution"] is True
    assert measured["independent_binary_pin_required_before_execution"] is True
    assert measured["context_closure_verified"] is False
    assert measured["toolchain_library_closure_verified"] is False
    assert measured["admission_build_eligible"] is False
    assert measured["admitted"] is False
    assert measured["formalized"] is False
    assert measured["proof_authority"] is False
    assert measured["source_semantics_verified"] is False
    assert measured["shared_schema_label"] == (
        "legal-corpus-v2-definition-use-native-build/v1"
    )
    assert measured["native_profile"] != measured["build_profile"]
    assert measured["library_target"] == "legal"
    assert measured["command_argv_tail"] == [
        "--offline",
        "--no-cache",
        "build",
        "legal",
    ]
    assert native.FALSE_FLAGS["admitted"] is False
    assert native.FALSE_FLAGS["formalized"] is False
    assert native.FALSE_FLAGS["context_closure_verified"] is False
    assert native.FALSE_FLAGS["build_eligible"] is False
    root = Path(native.__file__).parent
    native_text = (root / "legal_corpus_definition_use_native_v2.py").read_text()
    build_text = (root / "legal_corpus_definition_use_build_v2.py").read_text()
    assert len(native_text.encode()) <= 2_000_000
    assert len(build_text.encode()) <= 2_000_000
    for marker in _V2_ENVELOPE_OWNER_TEXT[
        "legal_corpus_definition_use_native_v2.py"
    ].splitlines():
        assert marker in native_text
    for marker in _V2_ENVELOPE_OWNER_TEXT[
        "legal_corpus_definition_use_build_v2.py"
    ].splitlines():
        assert marker in build_text
    receipt = dependency_resolution_receipt(
        "dependency_block_resolved_v2_owner_envelope_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["reviewer_claimed"] is False
    assert receipt["completion_authority"] is False


def test_v2_owner_envelope_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _ENVELOPE_ACCEPTANCE]
    owners = _v2_envelope_owners()
    dependency = (
        {
            "task_cid": "task:shared",
            "digest": "sha256:" + ("83" * 32),
            "body": _prepared_shared_module(),
        },
    )
    evidence = v2_owner_envelope_evidence(
        criteria,
        dependency,
        owners,
        EXPECTED_V2_OWNER_ENVELOPE_PROBE,
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_v2_owner_envelope_boundaries"
    )
    assert evidence["owners_interchangeable"] is False
    assert evidence["profile_selected"] is False
    assert evidence["binary_sha256_recorded"] is False
    assert evidence["lake_build_executed"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["reviewer_claimed"] is False
    assert evidence["completion_authority"] is False
    selected = dict(EXPECTED_V2_OWNER_ENVELOPE_PROBE)
    selected["profile_selected"] = True
    assert v2_owner_envelope_evidence(criteria, dependency, owners, selected) is None
    interchangeable = dict(EXPECTED_V2_OWNER_ENVELOPE_PROBE)
    interchangeable["owners_interchangeable"] = True
    assert (
        v2_owner_envelope_evidence(criteria, dependency, owners, interchangeable)
        is None
    )
    ran = dict(EXPECTED_V2_OWNER_ENVELOPE_PROBE)
    ran["lake_build_executed"] = True
    assert v2_owner_envelope_evidence(criteria, dependency, owners, ran) is None
    pinned = dict(EXPECTED_V2_OWNER_ENVELOPE_PROBE)
    pinned["binary_sha256_recorded"] = True
    assert v2_owner_envelope_evidence(criteria, dependency, owners, pinned) is None
    admitted_probe = dict(EXPECTED_V2_OWNER_ENVELOPE_PROBE)
    admitted_probe["admitted"] = True
    assert (
        v2_owner_envelope_evidence(criteria, dependency, owners, admitted_probe)
        is None
    )
    built = _prepared_shared_module()
    built["lake_build_executed"] = True
    assert v2_owner_envelope_evidence(
        criteria,
        ({"task_cid": "task:shared", "digest": "sha256:" + ("83" * 32), "body": built},),
        owners,
        EXPECTED_V2_OWNER_ENVELOPE_PROBE,
    ) is None
    admitted = _prepared_shared_module()
    admitted["admitted"] = True
    assert v2_owner_envelope_evidence(
        criteria,
        (
            {
                "task_cid": "task:shared",
                "digest": "sha256:" + ("83" * 32),
                "body": admitted,
            },
        ),
        owners,
        EXPECTED_V2_OWNER_ENVELOPE_PROBE,
    ) is None
    dropped = _v2_envelope_owners()
    dropped["build"] = dict(dropped["build"])
    dropped["build"]["text"] = "SCHEMA only\n"
    assert v2_owner_envelope_evidence(
        criteria, dependency, dropped, EXPECTED_V2_OWNER_ENVELOPE_PROBE
    ) is None
    blank = dict(dependency[0])
    blank["digest"] = ""
    assert v2_owner_envelope_evidence(
        criteria, (blank,), owners, EXPECTED_V2_OWNER_ENVELOPE_PROBE
    ) is None
    assert v2_owner_envelope_evidence(
        criteria,
        (
            {
                "task_cid": "task:custody",
                "digest": "sha256:" + ("81" * 32),
                "body": _unpromoted_custody(),
            },
        ),
        owners,
        EXPECTED_V2_OWNER_ENVELOPE_PROBE,
    ) is None
    for other in (
        _LAKE_ACCEPTANCE,
        _BUILD_NEXT_ACCEPTANCE,
        _BYTE_NEXT_ACCEPTANCE,
        _MODALITY_ACCEPTANCE,
        _ROUNDTRIP_ACCEPTANCE,
        _CUSTODY_ACCEPTANCE,
    ):
        assert v2_owner_envelope_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_V2_OWNER_ENVELOPE_PROBE,
        ) is None


def test_release_records_v2_owner_envelope_boundaries(tmp_path: Path) -> None:
    relative = []
    for name, text in _V2_ENVELOPE_OWNER_TEXT.items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        relative.append("owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:shared",
        "LIR-SHARED-DEP",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:shared",
        evidence_kind="validation",
        digest="sha256:" + ("83" * 32),
        body=_prepared_shared_module(),
    )
    built = _prepared_shared_module()
    built["lake_build_executed"] = True
    _task(
        repo,
        "task:built",
        "LIR-BUILT-DEP",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:built",
        evidence_kind="validation",
        digest="sha256:" + ("84" * 32),
        body=built,
    )
    _task(
        repo,
        "task:envelope",
        "LIR-ENVELOPE",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:shared"],
        acceptance=_ENVELOPE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare",
        "LIR-BARE-ENVELOPE",
        status="blocked",
        body=_manual_body("", stage="analysis"),
        dependencies=["task:shared"],
        acceptance=_ENVELOPE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:built-child",
        "LIR-BUILT-ENVELOPE",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:built"],
        acceptance=_ENVELOPE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:next",
        "LIR-BUILD-NEXT",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="prover"),
            "existing_owners": relative,
        },
        dependencies=["task:envelope"],
        acceptance=_BUILD_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:module",
        "LIR-MODULE-LOOKALIKE",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="corpus"),
        dependencies=["task:shared"],
        acceptance=_LAKE_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-ENVELOPE"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-ENVELOPE",
        "LIR-BUILT-ENVELOPE",
        "LIR-BUILD-NEXT",
        "LIR-MODULE-LOOKALIKE",
    }
    assert result["held"] == []
    task = repo.get_task("task:envelope")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_v2_owner_envelope_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["reviewer_claimed"] is False
    assert receipt["completion_authority"] is False
    for cid in (
        "task:bare",
        "task:built-child",
        "task:next",
        "task:module",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


_DECODER_ACCEPTANCE = (
    {
        "criterion": (
            "Binder/sort/arity/reference constraints and raw "
            "invalid/incomplete outputs are checked."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Fresh-head and compatible warm-start controls are retained; "
            "no direct KL across incompatible codecs."
        ),
        "evidence_kind": "validation",
    },
)
_MODALITY_LOOKALIKE_ACCEPTANCE = (
    {
        "criterion": (
            "Same initialization, source cases, trainable mask, optimizer, "
            "candidates and budget within each causal comparison."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Preserve named permission and complete-rule loss-ID vetoes; "
            "E1 engineering ranking work is separately tracked."
        ),
        "evidence_kind": "validation",
    },
)


def _byte_target_dependency() -> dict[str, object]:
    return {
        "operation": "dependency_block_resolved_byte_target_eligibility_boundary",
        "vocabulary_size": 259,
        "eligibility_granted": False,
        "training_executed": False,
        "fitting_executed": False,
        "admitted": False,
        "formalized": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }


def _decoder_owner_paths() -> dict[str, Path]:
    root = Path("/home/barberb/lift_coding/external/ipfs_datasets")
    return {
        "legal_grouped_span_decoder.py": root
        / "ipfs_datasets_py/logic/formalization/autoencoder/legal_grouped_span_decoder.py",
        "ast.py": root / "ipfs_datasets_py/logic/syntax_core/ast.py",
        "legal_relational_target_codec_v1.py": root
        / "ipfs_datasets_py/logic/formalization/autoencoder/legal_relational_target_codec_v1.py",
    }


def _decoder_owners() -> dict[str, dict[str, str]]:
    import hashlib

    roles = {
        "legal_grouped_span_decoder.py": "decoder",
        "ast.py": "syntax",
        "legal_relational_target_codec_v1.py": "codec",
    }
    owners = {}
    for name, path in _decoder_owner_paths().items():
        raw = path.read_bytes()
        owners[roles[name]] = {
            "path": name,
            "text": raw.decode("utf-8"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return owners


def test_typed_decoder_probe_matches_the_live_owners() -> None:
    assert measure_typed_decoder_probe() == EXPECTED_TYPED_DECODER_PROBE


def test_typed_decoder_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _DECODER_ACCEPTANCE]
    owners = _decoder_owners()
    dependency = (
        {
            "task_cid": "task:bytes",
            "digest": "sha256:" + ("85" * 32),
            "body": _byte_target_dependency(),
        },
    )
    evidence = typed_decoder_evidence(
        criteria, dependency, owners, EXPECTED_TYPED_DECODER_PROBE
    )
    assert evidence is not None
    assert evidence["operation"] == "dependency_block_resolved_typed_decoder_controls"
    assert evidence["fresh_head_retained"] is True
    assert evidence["compatible_warm_start_retained"] is True
    assert evidence["direct_kl_across_incompatible_codecs"] is False
    assert evidence["cross_entropy_loss"] is True
    assert evidence["training_executed"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["completion_authority"] is False
    fitted = _byte_target_dependency()
    fitted["eligibility_granted"] = True
    assert typed_decoder_evidence(
        criteria,
        ({"task_cid": "task:bytes", "digest": "sha256:" + ("85" * 32), "body": fitted},),
        owners,
        EXPECTED_TYPED_DECODER_PROBE,
    ) is None
    trained = dict(EXPECTED_TYPED_DECODER_PROBE)
    trained["training_executed"] = True
    assert typed_decoder_evidence(criteria, dependency, owners, trained) is None
    kl = dict(EXPECTED_TYPED_DECODER_PROBE)
    kl["direct_kl_across_incompatible_codecs"] = True
    assert typed_decoder_evidence(criteria, dependency, owners, kl) is None
    admitted = dict(EXPECTED_TYPED_DECODER_PROBE)
    admitted["admitted"] = True
    assert typed_decoder_evidence(criteria, dependency, owners, admitted) is None
    dropped = _decoder_owners()
    dropped["decoder"] = dict(dropped["decoder"])
    dropped["decoder"]["text"] = "decoder only\n"
    assert typed_decoder_evidence(
        criteria, dependency, dropped, EXPECTED_TYPED_DECODER_PROBE
    ) is None
    for other in (_ENVELOPE_ACCEPTANCE, _BUILD_NEXT_ACCEPTANCE, _LAKE_ACCEPTANCE):
        assert typed_decoder_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_TYPED_DECODER_PROBE,
        ) is None


def test_release_records_typed_decoder_controls(tmp_path: Path) -> None:
    relative = []
    for name, source in _decoder_owner_paths().items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        relative.append("owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:bytes",
        "LIR-BYTES",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:bytes",
        evidence_kind="validation",
        digest="sha256:" + ("85" * 32),
        body=_byte_target_dependency(),
    )
    _task(
        repo,
        "task:decoder",
        "LIR-DECODER",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:bytes"],
        acceptance=_DECODER_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare-decoder",
        "LIR-BARE-DECODER",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="validation"),
        dependencies=["task:bytes"],
        acceptance=_DECODER_ACCEPTANCE,
    )
    _task(
        repo,
        "task:modality",
        "LIR-MODALITY-LOOKALIKE",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:bytes"],
        acceptance=_MODALITY_LOOKALIKE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:build-next",
        "LIR-BUILD-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="prover"),
            "existing_owners": relative,
        },
        dependencies=["task:decoder"],
        acceptance=_BUILD_NEXT_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-DECODER"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-DECODER",
        "LIR-MODALITY-LOOKALIKE",
        "LIR-BUILD-STILL",
    }
    assert result["held"] == []
    task = repo.get_task("task:decoder")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == "dependency_block_resolved_typed_decoder_controls"
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    for cid in ("task:bare-decoder", "task:modality", "task:build-next"):
        assert repo.get_task(cid)["status"] == "blocked"


def _envelope_dependency() -> dict[str, object]:
    return {
        "operation": "dependency_block_resolved_v2_owner_envelope_boundaries",
        "lake_build_executed": False,
        "profile_selected": False,
        "binary_sha256_recorded": False,
        "admitted": False,
        "formalized": False,
        "proof_authority": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }


def _lake_owner_paths() -> dict[str, Path]:
    root = Path("/home/barberb/lift_coding/external/ipfs_datasets")
    return {
        "legal_corpus_definition_use_native_v2.py": root
        / "ipfs_datasets_py/logic/autoformal/legal_corpus_definition_use_native_v2.py",
        "legal_corpus_shared_lake.py": root
        / "ipfs_datasets_py/logic/autoformal/legal_corpus_shared_lake.py",
    }


def _lake_owners() -> dict[str, dict[str, str]]:
    import hashlib

    roles = {
        "legal_corpus_definition_use_native_v2.py": "native",
        "legal_corpus_shared_lake.py": "shared",
    }
    owners = {}
    for name, path in _lake_owner_paths().items():
        raw = path.read_bytes()
        owners[roles[name]] = {
            "path": name,
            "text": raw.decode("utf-8"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return owners


_FIXTURE_NEXT_ACCEPTANCE = (
    {
        "criterion": (
            "Definition sharing, wrong scope/roles, conditions/exceptions "
            "and alternative theories are checked without vacuity."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Compilation is formal-artifact evidence, not source-law "
            "fidelity; a failed proof is not a proof of negation."
        ),
        "evidence_kind": "validation",
    },
)


def test_shared_lake_build_probe_matches_the_recorded_receipt() -> None:
    assert measure_shared_lake_build_probe() == EXPECTED_SHARED_LAKE_BUILD_PROBE
    assert EXPECTED_SHARED_LAKE_BUILD_PROBE["native_owner_executed"] is False
    assert EXPECTED_SHARED_LAKE_BUILD_PROBE["shared_owner_executed"] is True
    assert EXPECTED_SHARED_LAKE_BUILD_PROBE["federal_corpus_acceptance"] is False
    assert EXPECTED_SHARED_LAKE_BUILD_PROBE["legal_acceptance"] is False
    assert EXPECTED_SHARED_LAKE_BUILD_PROBE["lean_compilation_verified"] is True
    assert EXPECTED_SHARED_LAKE_BUILD_PROBE["admitted"] is False
    assert EXPECTED_SHARED_LAKE_BUILD_PROBE["formalized"] is False


def test_shared_lake_build_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _BUILD_NEXT_ACCEPTANCE]
    owners = _lake_owners()
    dependency = (
        {
            "task_cid": "task:envelope",
            "digest": "sha256:" + ("86" * 32),
            "body": _envelope_dependency(),
        },
    )
    evidence = shared_lake_build_evidence(
        criteria, dependency, owners, EXPECTED_SHARED_LAKE_BUILD_PROBE
    )
    assert evidence is not None
    assert evidence["operation"] == "dependency_block_resolved_shared_lake_fixture_build"
    assert evidence["native_owner_executed"] is False
    assert evidence["shared_owner_executed"] is True
    assert evidence["sorry_or_admit_present"] is False
    assert evidence["federal_corpus_acceptance"] is False
    assert evidence["legal_acceptance"] is False
    assert evidence["lean_compilation_verified"] is True
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["completion_authority"] is False
    executed = _envelope_dependency()
    executed["lake_build_executed"] = True
    assert shared_lake_build_evidence(
        criteria,
        (
            {
                "task_cid": "task:envelope",
                "digest": "sha256:" + ("86" * 32),
                "body": executed,
            },
        ),
        owners,
        EXPECTED_SHARED_LAKE_BUILD_PROBE,
    ) is None
    admitted = dict(EXPECTED_SHARED_LAKE_BUILD_PROBE)
    admitted["admitted"] = True
    assert shared_lake_build_evidence(criteria, dependency, owners, admitted) is None
    dropped = _lake_owners()
    dropped["shared"] = dict(dropped["shared"])
    dropped["shared"]["text"] = "shared only\n"
    assert shared_lake_build_evidence(
        criteria, dependency, dropped, EXPECTED_SHARED_LAKE_BUILD_PROBE
    ) is None
    for other in (
        _DECODER_ACCEPTANCE,
        _MODALITY_LOOKALIKE_ACCEPTANCE,
        _ENVELOPE_ACCEPTANCE,
        _LAKE_ACCEPTANCE,
        _FIXTURE_NEXT_ACCEPTANCE,
    ):
        assert shared_lake_build_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_SHARED_LAKE_BUILD_PROBE,
        ) is None


def test_release_records_shared_lake_fixture_build(tmp_path: Path) -> None:
    relative = []
    for name, source in _lake_owner_paths().items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        relative.append("owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:envelope",
        "LIR-ENVELOPE-DEP",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:envelope",
        evidence_kind="validation",
        digest="sha256:" + ("86" * 32),
        body=_envelope_dependency(),
    )
    _task(
        repo,
        "task:build",
        "LIR-BUILD",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="prover"),
            "existing_owners": relative,
        },
        dependencies=["task:envelope"],
        acceptance=_BUILD_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare-build",
        "LIR-BARE-BUILD",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="prover"),
        dependencies=["task:envelope"],
        acceptance=_BUILD_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:decoder-lookalike",
        "LIR-DECODER-LOOKALIKE",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:envelope"],
        acceptance=_DECODER_ACCEPTANCE,
    )
    _task(
        repo,
        "task:modality",
        "LIR-MODALITY-STILL",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:envelope"],
        acceptance=_MODALITY_LOOKALIKE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:fixture-next",
        "LIR-FIXTURE-NEXT",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="evaluation"),
            "existing_owners": relative,
        },
        dependencies=["task:build"],
        acceptance=_FIXTURE_NEXT_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-BUILD"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-BUILD",
        "LIR-DECODER-LOOKALIKE",
        "LIR-MODALITY-STILL",
        "LIR-FIXTURE-NEXT",
    }
    assert result["held"] == []
    task = repo.get_task("task:build")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == "dependency_block_resolved_shared_lake_fixture_build"
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    for cid in (
        "task:bare-build",
        "task:decoder-lookalike",
        "task:modality",
        "task:fixture-next",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


def _compilation_dependency(*, legal_acceptance: bool = False) -> dict[str, object]:
    return {
        "operation": "dependency_block_resolved_shared_lake_fixture_build",
        "legal_acceptance": legal_acceptance,
        "federal_corpus_acceptance": False,
        "source_semantics_verified": False,
        "proof_authority": False,
        "sorry_or_admit_present": False,
        "admitted": False,
        "formalized": False,
        "lean_compilation_verified": True,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }


def _audit_owner_paths() -> dict[str, Path]:
    root = Path("/home/barberb/lift_coding/external/ipfs_datasets")
    return {
        "legal_corpus_shared_lake.py": root
        / "ipfs_datasets_py/logic/autoformal/legal_corpus_shared_lake.py",
        "legal_family_routes.py": root
        / "ipfs_datasets_py/logic/autoformal/legal_family_routes.py",
    }


def _audit_owners() -> dict[str, dict[str, str]]:
    import hashlib

    roles = {
        "legal_corpus_shared_lake.py": "shared",
        "legal_family_routes.py": "routes",
    }
    owners = {}
    for name, path in _audit_owner_paths().items():
        raw = path.read_bytes()
        owners[roles[name]] = {
            "path": name,
            "text": raw.decode("utf-8"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return owners


_TRAINING_NEXT_ACCEPTANCE = (
    {
        "criterion": (
            "Ordinary generation, full rules, argument/attachment/modality "
            "and protected retention metrics are reported."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Teacher-forced CE and compiler weak-label engineering scores "
            "remain separate from reviewed statutory fidelity; outputs "
            "freeze before scoring."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Materialize separate producer and post-freeze evaluation "
            "packets/owners; primary training/inference authority grants "
            "no evaluation reference access."
        ),
        "evidence_kind": "validation",
    },
)


def test_consequence_audit_probe_matches_executed_checks() -> None:
    source = Path(
        "/home/barberb/lift_coding/external/ipfs_accelerate/"
        "ipfs_accelerate_py/agent_supervisor/task_sources/consequence_audit_probe.py"
    ).read_text(encoding="utf-8")
    assert "build_shared_definition_corpus(" not in source
    assert measure_consequence_audit_probe() == EXPECTED_CONSEQUENCE_AUDIT_PROBE
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["wrong_scope_shared"] is False
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["equal_ast_not_merged"] is True
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["each_hypothesis_necessary"] is True
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["failed_entailment_with_action_true"] == 5
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["failed_proof_is_proof_of_negation"] is False
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["negation_inferred"] is False
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["vacuous_true_refused"] is True
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["invented_axiom_refused"] is True
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["trivial_boolean_refused"] is True
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["condition_resolution_claimed"] is False
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["exception_resolution_claimed"] is False
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["alternative_modalities_collapsed"] is False
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["legal_acceptance_claimed"] is False
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["federal_corpus_acceptance_claimed"] is False
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["audit_lake_build_executed"] is False
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["compilation_rerun"] is False
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["admitted"] is False
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["formalized"] is False
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["proof_authority"] is False
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["source_semantics_verified"] is False
    assert EXPECTED_CONSEQUENCE_AUDIT_PROBE["theorem_establishes_legal_truth"] is False


def test_consequence_audit_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _FIXTURE_NEXT_ACCEPTANCE]
    owners = _audit_owners()
    dependency = (
        {
            "task_cid": "task:compilation",
            "digest": "sha256:" + ("87" * 32),
            "body": _compilation_dependency(),
        },
    )
    evidence = consequence_audit_evidence(
        criteria, dependency, owners, EXPECTED_CONSEQUENCE_AUDIT_PROBE
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_nonvacuous_consequence_boundaries"
    )
    assert evidence["failed_proof_is_proof_of_negation"] is False
    assert evidence["wrong_scope_shared"] is False
    assert evidence["legal_acceptance_claimed"] is False
    assert evidence["audit_lake_build_executed"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["completion_authority"] is False
    assert evidence["reviewer_claimed"] is False
    accepted = _compilation_dependency(legal_acceptance=True)
    assert consequence_audit_evidence(
        criteria,
        (
            {
                "task_cid": "task:compilation",
                "digest": "sha256:" + ("87" * 32),
                "body": accepted,
            },
        ),
        owners,
        EXPECTED_CONSEQUENCE_AUDIT_PROBE,
    ) is None
    admitted = dict(EXPECTED_CONSEQUENCE_AUDIT_PROBE)
    admitted["admitted"] = True
    assert consequence_audit_evidence(criteria, dependency, owners, admitted) is None
    negated = dict(EXPECTED_CONSEQUENCE_AUDIT_PROBE)
    negated["failed_proof_is_proof_of_negation"] = True
    assert consequence_audit_evidence(criteria, dependency, owners, negated) is None
    dropped = _audit_owners()
    dropped["shared"] = dict(dropped["shared"])
    dropped["shared"]["text"] = "shared only\n"
    assert consequence_audit_evidence(
        criteria, dependency, dropped, EXPECTED_CONSEQUENCE_AUDIT_PROBE
    ) is None
    for other in (
        _BUILD_NEXT_ACCEPTANCE,
        _DECODER_ACCEPTANCE,
        _MODALITY_LOOKALIKE_ACCEPTANCE,
        _ENVELOPE_ACCEPTANCE,
        _LAKE_ACCEPTANCE,
        _TRAINING_NEXT_ACCEPTANCE,
    ):
        assert consequence_audit_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_CONSEQUENCE_AUDIT_PROBE,
        ) is None


def test_release_records_nonvacuous_consequence_audit(tmp_path: Path) -> None:
    relative = []
    for name, source in _audit_owner_paths().items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        relative.append("owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:compilation",
        "LIR-COMPILATION",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:compilation",
        evidence_kind="validation",
        digest="sha256:" + ("87" * 32),
        body=_compilation_dependency(),
    )
    _task(
        repo,
        "task:accepted",
        "LIR-ACCEPTED-COMPILE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:accepted",
        evidence_kind="validation",
        digest="sha256:" + ("88" * 32),
        body=_compilation_dependency(legal_acceptance=True),
    )
    _task(
        repo,
        "task:audit",
        "LIR-AUDIT",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="evaluation"),
            "existing_owners": relative,
        },
        dependencies=["task:compilation"],
        acceptance=_FIXTURE_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare-audit",
        "LIR-BARE-AUDIT",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="evaluation"),
        dependencies=["task:compilation"],
        acceptance=_FIXTURE_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:build-still",
        "LIR-BUILD-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="prover"),
            "existing_owners": relative,
        },
        dependencies=["task:compilation"],
        acceptance=_BUILD_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:decoder-still",
        "LIR-DECODER-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:compilation"],
        acceptance=_DECODER_ACCEPTANCE,
    )
    _task(
        repo,
        "task:modality-still",
        "LIR-MODALITY-STILL",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:compilation"],
        acceptance=_MODALITY_LOOKALIKE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:training-still",
        "LIR-TRAINING-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="training"),
            "existing_owners": relative,
        },
        dependencies=["task:compilation"],
        acceptance=_TRAINING_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:accepted-audit",
        "LIR-ACCEPTED-AUDIT",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="evaluation"),
            "existing_owners": relative,
        },
        dependencies=["task:accepted"],
        acceptance=_FIXTURE_NEXT_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-AUDIT"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-AUDIT",
        "LIR-BUILD-STILL",
        "LIR-DECODER-STILL",
        "LIR-MODALITY-STILL",
        "LIR-TRAINING-STILL",
        "LIR-ACCEPTED-AUDIT",
    }
    assert result["held"] == []
    task = repo.get_task("task:audit")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_nonvacuous_consequence_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    for cid in (
        "task:bare-audit",
        "task:build-still",
        "task:decoder-still",
        "task:modality-still",
        "task:training-still",
        "task:accepted-audit",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


def _audit_boundary_dependency(
    *, legal_acceptance_claimed: bool = False
) -> dict[str, object]:
    return {
        "operation": "dependency_block_resolved_nonvacuous_consequence_boundaries",
        "legal_acceptance_claimed": legal_acceptance_claimed,
        "federal_corpus_acceptance_claimed": False,
        "failed_proof_is_proof_of_negation": False,
        "audit_lake_build_executed": False,
        "source_semantics_verified": False,
        "proof_authority": False,
        "admitted": False,
        "formalized": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }


def _invalidation_owner_paths() -> dict[str, Path]:
    root = Path("/home/barberb/lift_coding/external/ipfs_datasets")
    return {
        "legal_graph_dependencies.py": root
        / "ipfs_datasets_py/logic/autoformal/legal_graph_dependencies.py",
        "legal_corpus_incremental.py": root
        / "ipfs_datasets_py/logic/autoformal/legal_corpus_incremental.py",
    }


def _invalidation_owners() -> dict[str, dict[str, str]]:
    import hashlib

    roles = {
        "legal_graph_dependencies.py": "dependencies",
        "legal_corpus_incremental.py": "incremental",
    }
    owners = {}
    for name, path in _invalidation_owner_paths().items():
        raw = path.read_bytes()
        owners[roles[name]] = {
            "path": name,
            "text": raw.decode("utf-8"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return owners


_INVALIDATION_ACCEPTANCE = (
    {
        "criterion": (
            "Changed applicability/review can invalidate legal eligibility "
            "without destroying a reusable conditional compiled artifact."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Adding a competing definition invalidates its prior users and "
            "reverse dependencies."
        ),
        "evidence_kind": "validation",
    },
)
_INCREMENTAL_BUILD_ACCEPTANCE = (
    {
        "criterion": (
            "All expected artifacts and semantic obligations agree with "
            "the clean Lake build."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "The existing direct lean -o cache experiment does not "
            "substitute for lake build legal."
        ),
        "evidence_kind": "validation",
    },
)
_COVERAGE_ACCEPTANCE = (
    {
        "criterion": (
            "Counts distinguish inventory, eligible, attempted, "
            "missing/invalid/unsupported and accepted populations."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "No throughput extrapolation from the small replay or "
            "row-only total; review cost is included."
        ),
        "evidence_kind": "validation",
    },
)


def test_invalidation_keys_probe_matches_executed_checks() -> None:
    source = Path(
        "/home/barberb/lift_coding/external/ipfs_accelerate/"
        "ipfs_accelerate_py/agent_supervisor/task_sources/invalidation_keys_probe.py"
    ).read_text(encoding="utf-8")
    assert "build_incremental_shared_corpus(" not in source
    assert "build_incremental_parent_shared_corpus_v1(" not in source
    assert measure_invalidation_keys_probe() == EXPECTED_INVALIDATION_KEYS_PROBE
    assert EXPECTED_INVALIDATION_KEYS_PROBE[
        "review_pin_invalidates_eligibility"
    ] is True
    assert EXPECTED_INVALIDATION_KEYS_PROBE[
        "applicability_pin_invalidates_eligibility"
    ] is True
    assert EXPECTED_INVALIDATION_KEYS_PROBE[
        "conditional_compiled_artifact_retained"
    ] is True
    assert EXPECTED_INVALIDATION_KEYS_PROBE["compiled_artifact_bytes_destroyed"] is False
    assert EXPECTED_INVALIDATION_KEYS_PROBE[
        "competing_definition_invalidates_prior_user"
    ] is True
    assert EXPECTED_INVALIDATION_KEYS_PROBE[
        "competing_definition_invalidates_reverse_dependent"
    ] is True
    assert EXPECTED_INVALIDATION_KEYS_PROBE["unrelated_compiled_artifact_reused"] is True
    assert EXPECTED_INVALIDATION_KEYS_PROBE[
        "cache_pin_without_artifact_bytes_refused"
    ] is True
    assert EXPECTED_INVALIDATION_KEYS_PROBE["incremental_compiler_invoked"] is False
    assert EXPECTED_INVALIDATION_KEYS_PROBE["lake_build_executed"] is False
    assert EXPECTED_INVALIDATION_KEYS_PROBE["fresh_proof_for_reused_modules"] is False
    assert EXPECTED_INVALIDATION_KEYS_PROBE["legal_meaning_established"] is False
    assert EXPECTED_INVALIDATION_KEYS_PROBE[
        "legal_dependency_inventory_exhaustive"
    ] is False
    assert EXPECTED_INVALIDATION_KEYS_PROBE["admitted"] is False
    assert EXPECTED_INVALIDATION_KEYS_PROBE["formalized"] is False
    assert EXPECTED_INVALIDATION_KEYS_PROBE["proof_authority"] is False
    assert EXPECTED_INVALIDATION_KEYS_PROBE["source_semantics_verified"] is False


def test_invalidation_keys_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _INVALIDATION_ACCEPTANCE]
    owners = _invalidation_owners()
    dependency = (
        {
            "task_cid": "task:audit",
            "digest": "sha256:" + ("89" * 32),
            "body": _audit_boundary_dependency(),
        },
    )
    evidence = invalidation_keys_evidence(
        criteria, dependency, owners, EXPECTED_INVALIDATION_KEYS_PROBE
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_invalidation_key_boundaries"
    )
    assert evidence["compiled_artifact_bytes_destroyed"] is False
    assert evidence["conditional_compiled_artifact_retained"] is True
    assert evidence["incremental_compiler_invoked"] is False
    assert evidence["lake_build_executed"] is False
    assert evidence["legal_meaning_established"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["completion_authority"] is False
    assert evidence["reviewer_claimed"] is False
    accepted = _audit_boundary_dependency(legal_acceptance_claimed=True)
    assert invalidation_keys_evidence(
        criteria,
        (
            {
                "task_cid": "task:audit",
                "digest": "sha256:" + ("89" * 32),
                "body": accepted,
            },
        ),
        owners,
        EXPECTED_INVALIDATION_KEYS_PROBE,
    ) is None
    destroyed = dict(EXPECTED_INVALIDATION_KEYS_PROBE)
    destroyed["compiled_artifact_bytes_destroyed"] = True
    assert invalidation_keys_evidence(
        criteria, dependency, owners, destroyed
    ) is None
    admitted = dict(EXPECTED_INVALIDATION_KEYS_PROBE)
    admitted["admitted"] = True
    assert invalidation_keys_evidence(criteria, dependency, owners, admitted) is None
    dropped = _invalidation_owners()
    dropped["incremental"] = dict(dropped["incremental"])
    dropped["incremental"]["text"] = "incremental only\n"
    assert invalidation_keys_evidence(
        criteria, dependency, dropped, EXPECTED_INVALIDATION_KEYS_PROBE
    ) is None
    for other in (
        _FIXTURE_NEXT_ACCEPTANCE,
        _BUILD_NEXT_ACCEPTANCE,
        _DECODER_ACCEPTANCE,
        _MODALITY_LOOKALIKE_ACCEPTANCE,
        _TRAINING_NEXT_ACCEPTANCE,
        _INCREMENTAL_BUILD_ACCEPTANCE,
        _COVERAGE_ACCEPTANCE,
    ):
        assert invalidation_keys_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_INVALIDATION_KEYS_PROBE,
        ) is None


def test_release_records_invalidation_key_boundaries(tmp_path: Path) -> None:
    relative = []
    for name, source in _invalidation_owner_paths().items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        relative.append("owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:audit",
        "LIR-AUDIT",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:audit",
        evidence_kind="validation",
        digest="sha256:" + ("89" * 32),
        body=_audit_boundary_dependency(),
    )
    _task(
        repo,
        "task:accepted",
        "LIR-ACCEPTED-AUDIT",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:accepted",
        evidence_kind="validation",
        digest="sha256:" + ("8a" * 32),
        body=_audit_boundary_dependency(legal_acceptance_claimed=True),
    )
    _task(
        repo,
        "task:invalidation",
        "LIR-INVALIDATION",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:audit"],
        acceptance=_INVALIDATION_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare-invalidation",
        "LIR-BARE-INVALIDATION",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="validation"),
        dependencies=["task:audit"],
        acceptance=_INVALIDATION_ACCEPTANCE,
    )
    _task(
        repo,
        "task:build-still",
        "LIR-BUILD-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="prover"),
            "existing_owners": relative,
        },
        dependencies=["task:audit"],
        acceptance=_INCREMENTAL_BUILD_ACCEPTANCE,
    )
    _task(
        repo,
        "task:coverage-still",
        "LIR-COVERAGE-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="evaluation"),
            "existing_owners": relative,
        },
        dependencies=["task:audit"],
        acceptance=_COVERAGE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:training-still",
        "LIR-TRAINING-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="training"),
            "existing_owners": relative,
        },
        dependencies=["task:audit"],
        acceptance=_TRAINING_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:modality-still",
        "LIR-MODALITY-STILL",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:audit"],
        acceptance=_MODALITY_LOOKALIKE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:accepted-invalidation",
        "LIR-ACCEPTED-INVALIDATION",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:accepted"],
        acceptance=_INVALIDATION_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-INVALIDATION"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-INVALIDATION",
        "LIR-BUILD-STILL",
        "LIR-COVERAGE-STILL",
        "LIR-TRAINING-STILL",
        "LIR-MODALITY-STILL",
        "LIR-ACCEPTED-INVALIDATION",
    }
    assert result["held"] == []
    task = repo.get_task("task:invalidation")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_invalidation_key_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    for cid in (
        "task:bare-invalidation",
        "task:build-still",
        "task:coverage-still",
        "task:training-still",
        "task:modality-still",
        "task:accepted-invalidation",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


def _invalidation_boundary_dependency(
    *, legal_meaning_established: bool = False
) -> dict[str, object]:
    return {
        "operation": "dependency_block_resolved_invalidation_key_boundaries",
        "legal_meaning_established": legal_meaning_established,
        "incremental_compiler_invoked": False,
        "lake_build_executed": False,
        "admitted": False,
        "formalized": False,
        "proof_authority": False,
        "source_semantics_verified": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }


def _lake_agreement_owner_paths() -> dict[str, Path]:
    root = Path("/home/barberb/lift_coding/external/ipfs_datasets")
    return {
        "legal_corpus_incremental.py": root
        / "ipfs_datasets_py/logic/autoformal/legal_corpus_incremental.py",
        "legal_corpus_shared_lake.py": root
        / "ipfs_datasets_py/logic/autoformal/legal_corpus_shared_lake.py",
    }


def _lake_agreement_owners() -> dict[str, dict[str, str]]:
    import hashlib

    roles = {
        "legal_corpus_incremental.py": "incremental",
        "legal_corpus_shared_lake.py": "shared",
    }
    owners = {}
    for name, path in _lake_agreement_owner_paths().items():
        raw = path.read_bytes()
        owners[roles[name]] = {
            "path": name,
            "text": raw.decode("utf-8"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return owners


def test_incremental_lake_agreement_probe_matches_executed_builds() -> None:
    source = Path(
        "/home/barberb/lift_coding/external/ipfs_accelerate/"
        "ipfs_accelerate_py/agent_supervisor/task_sources/"
        "incremental_lake_agreement_probe.py"
    ).read_text(encoding="utf-8")
    assert "build_incremental_shared_corpus(" in source
    assert "build_shared_definition_corpus(" in source
    assert '["--offline", "--no-cache", "build", "legal"]' in source
    measured = measure_incremental_lake_agreement_probe()
    assert measured == EXPECTED_INCREMENTAL_LAKE_AGREEMENT_PROBE
    assert measured["expected_module_population_agrees"] is True
    assert measured["semantic_obligations_agree"] is True
    assert measured["olean_bytes_identical"] is False
    assert measured["direct_lean_o_substitutes_for_lake_build_legal"] is False
    assert measured["lake_build_executed"] is True
    assert measured["incremental_compiler_invoked"] is True
    assert measured["unchanged_warm_compile_count"] == 0
    assert measured["unchanged_definition_module_count"] == 1
    assert measured["changed_compiled_count"] == 3
    assert measured["changed_reused_count"] == 3
    assert measured["ambiguous_definition_module_count"] == 2
    assert measured["old_unique_definition_reused"] is False
    assert measured["legal_meaning_established"] is False
    assert measured["legal_acceptance_claimed"] is False
    assert measured["federal_corpus_acceptance_claimed"] is False
    assert measured["admitted"] is False
    assert measured["formalized"] is False
    assert measured["proof_authority"] is False
    assert measured["source_semantics_verified"] is False
    assert measured["fresh_proof_for_reused_modules"] is False
    assert measured["lake_command_tail"] == [
        "--offline",
        "--no-cache",
        "build",
        "legal",
    ]


def test_incremental_lake_agreement_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _INCREMENTAL_BUILD_ACCEPTANCE]
    owners = _lake_agreement_owners()
    dependency = (
        {
            "task_cid": "task:invalidation",
            "digest": "sha256:" + ("91" * 32),
            "body": _invalidation_boundary_dependency(),
        },
    )
    evidence = incremental_lake_agreement_evidence(
        criteria, dependency, owners, EXPECTED_INCREMENTAL_LAKE_AGREEMENT_PROBE
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_incremental_lake_agreement"
    )
    assert evidence["expected_module_population_agrees"] is True
    assert evidence["semantic_obligations_agree"] is True
    assert evidence["olean_bytes_identical"] is False
    assert evidence["direct_lean_o_substitutes_for_lake_build_legal"] is False
    assert evidence["lake_build_executed"] is True
    assert evidence["legal_meaning_established"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["completion_authority"] is False
    assert evidence["reviewer_claimed"] is False
    meaningful = _invalidation_boundary_dependency(legal_meaning_established=True)
    assert incremental_lake_agreement_evidence(
        criteria,
        (
            {
                "task_cid": "task:invalidation",
                "digest": "sha256:" + ("91" * 32),
                "body": meaningful,
            },
        ),
        owners,
        EXPECTED_INCREMENTAL_LAKE_AGREEMENT_PROBE,
    ) is None
    substituted = dict(EXPECTED_INCREMENTAL_LAKE_AGREEMENT_PROBE)
    substituted["direct_lean_o_substitutes_for_lake_build_legal"] = True
    assert incremental_lake_agreement_evidence(
        criteria, dependency, owners, substituted
    ) is None
    identical = dict(EXPECTED_INCREMENTAL_LAKE_AGREEMENT_PROBE)
    identical["olean_bytes_identical"] = True
    assert incremental_lake_agreement_evidence(
        criteria, dependency, owners, identical
    ) is None
    admitted = dict(EXPECTED_INCREMENTAL_LAKE_AGREEMENT_PROBE)
    admitted["admitted"] = True
    assert incremental_lake_agreement_evidence(
        criteria, dependency, owners, admitted
    ) is None
    dropped = _lake_agreement_owners()
    dropped["shared"] = dict(dropped["shared"])
    dropped["shared"]["text"] = "shared only\n"
    assert incremental_lake_agreement_evidence(
        criteria, dependency, dropped, EXPECTED_INCREMENTAL_LAKE_AGREEMENT_PROBE
    ) is None
    for other in (
        _INVALIDATION_ACCEPTANCE,
        _FIXTURE_NEXT_ACCEPTANCE,
        _DECODER_ACCEPTANCE,
        _MODALITY_LOOKALIKE_ACCEPTANCE,
        _TRAINING_NEXT_ACCEPTANCE,
        _COVERAGE_ACCEPTANCE,
    ):
        assert incremental_lake_agreement_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_INCREMENTAL_LAKE_AGREEMENT_PROBE,
        ) is None


def test_release_records_incremental_lake_agreement(tmp_path: Path) -> None:
    relative = []
    for name, source in _lake_agreement_owner_paths().items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        relative.append("owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:invalidation",
        "LIR-INVALIDATION",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:invalidation",
        evidence_kind="validation",
        digest="sha256:" + ("91" * 32),
        body=_invalidation_boundary_dependency(),
    )
    _task(
        repo,
        "task:meaningful",
        "LIR-MEANINGFUL-INVALIDATION",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:meaningful",
        evidence_kind="validation",
        digest="sha256:" + ("92" * 32),
        body=_invalidation_boundary_dependency(legal_meaning_established=True),
    )
    _task(
        repo,
        "task:agreement",
        "LIR-AGREEMENT",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:invalidation"],
        acceptance=_INCREMENTAL_BUILD_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare-agreement",
        "LIR-BARE-AGREEMENT",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="validation"),
        dependencies=["task:invalidation"],
        acceptance=_INCREMENTAL_BUILD_ACCEPTANCE,
    )
    _task(
        repo,
        "task:training-still",
        "LIR-TRAINING-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="training"),
            "existing_owners": relative,
        },
        dependencies=["task:invalidation"],
        acceptance=_TRAINING_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:coverage-still",
        "LIR-COVERAGE-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="evaluation"),
            "existing_owners": relative,
        },
        dependencies=["task:invalidation"],
        acceptance=_COVERAGE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:modality-still",
        "LIR-MODALITY-STILL",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:invalidation"],
        acceptance=_MODALITY_LOOKALIKE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:invalidation-still",
        "LIR-INVALIDATION-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:invalidation"],
        acceptance=_INVALIDATION_ACCEPTANCE,
    )
    _task(
        repo,
        "task:meaningful-agreement",
        "LIR-MEANINGFUL-AGREEMENT",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:meaningful"],
        acceptance=_INCREMENTAL_BUILD_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-AGREEMENT"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-AGREEMENT",
        "LIR-TRAINING-STILL",
        "LIR-COVERAGE-STILL",
        "LIR-MODALITY-STILL",
        "LIR-INVALIDATION-STILL",
        "LIR-MEANINGFUL-AGREEMENT",
    }
    assert result["held"] == []
    task = repo.get_task("task:agreement")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_incremental_lake_agreement"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    for cid in (
        "task:bare-agreement",
        "task:training-still",
        "task:coverage-still",
        "task:modality-still",
        "task:invalidation-still",
        "task:meaningful-agreement",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


_RELEASE_COORDINATOR_ACCEPTANCE = (
    {
        "criterion": (
            "Referenced artifacts publish before the coordinator receipt; "
            "no self-referential commit hash."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Candidate, reviewed, replay-qualified and promoted statuses "
            "remain separate, including ModelManager selectors."
        ),
        "evidence_kind": "validation",
    },
)


def _release_coordinator_owner_paths() -> dict[str, Path]:
    root = Path(
        "/home/barberb/lift_coding/external/ipfs_datasets/docs/autoencoders"
    )
    return {
        "legal_ir_span_graph_lean_execution_plan_20261016.md": root
        / "legal_ir_span_graph_lean_execution_plan_20261016.md",
        "legal_ir_asset_reconciliation_20261016.md": root
        / "legal_ir_asset_reconciliation_20261016.md",
    }


def _release_coordinator_owners() -> dict[str, dict[str, str]]:
    import hashlib

    roles = {
        "legal_ir_span_graph_lean_execution_plan_20261016.md": "plan",
        "legal_ir_asset_reconciliation_20261016.md": "reconciliation",
    }
    owners = {}
    for name, path in _release_coordinator_owner_paths().items():
        raw = path.read_bytes()
        owners[roles[name]] = {
            "path": name,
            "text": raw.decode("utf-8"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return owners


def _lake_agreement_boundary_dependency(
    *, legal_meaning_established: bool = False
) -> dict[str, object]:
    return {
        "operation": "dependency_block_resolved_incremental_lake_agreement",
        "legal_meaning_established": legal_meaning_established,
        "legal_acceptance_claimed": False,
        "federal_corpus_acceptance_claimed": False,
        "admitted": False,
        "formalized": False,
        "proof_authority": False,
        "source_semantics_verified": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }


def test_release_coordinator_probe_matches_authored_seal() -> None:
    source = Path(
        "/home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/"
        "logic/autoformal/legal_release_coordinator_v1.py"
    ).read_text(encoding="utf-8")
    assert "def publish_referenced_artifact" in source
    assert "def seal_coordinator_receipt" in source
    assert "self-referential commit hash refused" in source
    assert "def select_model_manager_status" in source
    measured = measure_release_coordinator_probe()
    assert measured == EXPECTED_RELEASE_COORDINATOR_PROBE
    assert measured["referenced_artifacts_published_before_receipt"] is True
    assert measured["self_referential_commit_hash_refused"] is True
    assert measured["model_manager_selectors_disjoint"] is True
    assert measured["selector_collapses_statuses"] is False
    assert measured["remote_upload_executed"] is False
    assert measured["hub_readback_executed"] is False
    assert measured["model_manager_catalog_mutated"] is False
    assert measured["legal_acceptance_claimed"] is False
    assert measured["federal_corpus_acceptance_claimed"] is False
    assert measured["legal_meaning_established"] is False
    assert measured["admitted"] is False
    assert measured["formalized"] is False
    assert measured["artifact_count"] == 4
    assert measured["statuses"] == [
        "candidate",
        "reviewed",
        "replay-qualified",
        "promoted",
    ]


def test_release_coordinator_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _RELEASE_COORDINATOR_ACCEPTANCE]
    owners = _release_coordinator_owners()
    dependency = (
        {
            "task_cid": "task:agreement",
            "digest": "sha256:" + ("93" * 32),
            "body": _lake_agreement_boundary_dependency(),
        },
    )
    evidence = release_coordinator_evidence(
        criteria, dependency, owners, EXPECTED_RELEASE_COORDINATOR_PROBE
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_release_coordinator_boundaries"
    )
    assert evidence["referenced_artifacts_published_before_receipt"] is True
    assert evidence["self_referential_commit_hash_refused"] is True
    assert evidence["remote_upload_executed"] is False
    assert evidence["legal_acceptance_claimed"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["completion_authority"] is False
    meaningful = _lake_agreement_boundary_dependency(legal_meaning_established=True)
    assert release_coordinator_evidence(
        criteria,
        (
            {
                "task_cid": "task:agreement",
                "digest": "sha256:" + ("93" * 32),
                "body": meaningful,
            },
        ),
        owners,
        EXPECTED_RELEASE_COORDINATOR_PROBE,
    ) is None
    uploaded = dict(EXPECTED_RELEASE_COORDINATOR_PROBE)
    uploaded["remote_upload_executed"] = True
    assert release_coordinator_evidence(
        criteria, dependency, owners, uploaded
    ) is None
    collapsed = dict(EXPECTED_RELEASE_COORDINATOR_PROBE)
    collapsed["selector_collapses_statuses"] = True
    assert release_coordinator_evidence(
        criteria, dependency, owners, collapsed
    ) is None
    admitted = dict(EXPECTED_RELEASE_COORDINATOR_PROBE)
    admitted["admitted"] = True
    assert release_coordinator_evidence(
        criteria, dependency, owners, admitted
    ) is None
    dropped = _release_coordinator_owners()
    dropped["reconciliation"] = dict(dropped["reconciliation"])
    dropped["reconciliation"]["text"] = "reconciliation only\n"
    assert release_coordinator_evidence(
        criteria, dependency, dropped, EXPECTED_RELEASE_COORDINATOR_PROBE
    ) is None
    for other in (
        _INCREMENTAL_BUILD_ACCEPTANCE,
        _INVALIDATION_ACCEPTANCE,
        _COVERAGE_ACCEPTANCE,
        _MODALITY_LOOKALIKE_ACCEPTANCE,
        _TRAINING_NEXT_ACCEPTANCE,
    ):
        assert release_coordinator_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_RELEASE_COORDINATOR_PROBE,
        ) is None


def test_release_records_release_coordinator(tmp_path: Path) -> None:
    relative = []
    for name, source in _release_coordinator_owner_paths().items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        relative.append("owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:agreement",
        "LIR-AGREEMENT",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:agreement",
        evidence_kind="validation",
        digest="sha256:" + ("93" * 32),
        body=_lake_agreement_boundary_dependency(),
    )
    _task(
        repo,
        "task:meaningful",
        "LIR-MEANINGFUL-AGREEMENT",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:meaningful",
        evidence_kind="validation",
        digest="sha256:" + ("94" * 32),
        body=_lake_agreement_boundary_dependency(legal_meaning_established=True),
    )
    _task(
        repo,
        "task:coordinator",
        "LIR-COORDINATOR",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:agreement"],
        acceptance=_RELEASE_COORDINATOR_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare-coordinator",
        "LIR-BARE-COORDINATOR",
        status="blocked",
        body=_manual_body("", stage="analysis"),
        dependencies=["task:agreement"],
        acceptance=_RELEASE_COORDINATOR_ACCEPTANCE,
    )
    _task(
        repo,
        "task:training-still",
        "LIR-TRAINING-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="training"),
            "existing_owners": relative,
        },
        dependencies=["task:agreement"],
        acceptance=_TRAINING_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:coverage-still",
        "LIR-COVERAGE-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="evaluation"),
            "existing_owners": relative,
        },
        dependencies=["task:agreement"],
        acceptance=_COVERAGE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:modality-still",
        "LIR-MODALITY-STILL",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:agreement"],
        acceptance=_MODALITY_LOOKALIKE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:lake-still",
        "LIR-LAKE-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:agreement"],
        acceptance=_INCREMENTAL_BUILD_ACCEPTANCE,
    )
    _task(
        repo,
        "task:meaningful-coordinator",
        "LIR-MEANINGFUL-COORDINATOR",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:meaningful"],
        acceptance=_RELEASE_COORDINATOR_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-COORDINATOR"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-COORDINATOR",
        "LIR-TRAINING-STILL",
        "LIR-COVERAGE-STILL",
        "LIR-MODALITY-STILL",
        "LIR-LAKE-STILL",
        "LIR-MEANINGFUL-COORDINATOR",
    }
    assert result["held"] == []
    task = repo.get_task("task:coordinator")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_release_coordinator_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    for cid in (
        "task:bare-coordinator",
        "task:training-still",
        "task:coverage-still",
        "task:modality-still",
        "task:lake-still",
        "task:meaningful-coordinator",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


_MODALITY_COMPARISON_ACCEPTANCE = _MODALITY_LOOKALIKE_ACCEPTANCE


def _codec_intact_boundary_dependency(
    *, meanings_approved: bool = False
) -> dict[str, object]:
    return {
        "operation": "dependency_block_resolved_codec_intact_no_extension",
        "extensions_added": 0,
        "meanings_requiring_extension": 0,
        "meanings_approved": meanings_approved,
        "vocabulary_size": 259,
        "admitted": False,
        "formalized": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }


def _modality_comparison_owner_paths() -> dict[str, Path]:
    root = Path("/home/barberb/lift_coding/external/ipfs_datasets")
    return {
        "legal_span_modality_conditions_retention.py": root
        / "ipfs_datasets_py/optimizers/logic_theorem_optimizer"
        / "legal_span_modality_conditions_retention.py",
        "legal_ir_span_graph_lean_execution_plan_20261016.md": root
        / "docs/autoencoders/legal_ir_span_graph_lean_execution_plan_20261016.md",
    }


def _modality_comparison_owners() -> dict[str, dict[str, str]]:
    import hashlib

    roles = {
        "legal_span_modality_conditions_retention.py": "modality",
        "legal_ir_span_graph_lean_execution_plan_20261016.md": "plan",
    }
    owners = {}
    for name, path in _modality_comparison_owner_paths().items():
        raw = path.read_bytes()
        owners[roles[name]] = {
            "path": name,
            "text": raw.decode("utf-8"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return owners


def test_modality_causal_comparison_probe_matches_fixed_arms() -> None:
    source = Path(
        "/home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/"
        "logic/autoformal/legal_modality_causal_comparison_v1.py"
    ).read_text(encoding="utf-8")
    assert "def compare_causal_arms" in source
    assert "native-tuning-calendula-t4-q5-p" in source
    assert "32 tracked permission modality decisions" in source
    assert "hypothesis_selected" in source
    measured = measure_modality_causal_comparison_probe()
    assert measured == EXPECTED_MODALITY_CAUSAL_COMPARISON_PROBE
    assert measured["hypothesis_selected"] is False
    assert measured["training_executed"] is False
    assert measured["e1_ranking_selects_hypothesis"] is False
    assert measured["changed_initialization_refused"] is True
    assert measured["dropped_loss_id_veto_refused"] is True
    assert measured["e1_ranking_consuming_veto_refused"] is True
    assert measured["permission_vetoes"] == [
        "32 tracked permission modality decisions"
    ]
    assert measured["complete_rule_loss_id_vetoes"] == [
        "native-tuning-calendula-t4-q5-p"
    ]
    assert measured["admitted"] is False
    assert measured["formalized"] is False
    assert measured["legal_acceptance_claimed"] is False


def test_modality_causal_comparison_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _MODALITY_COMPARISON_ACCEPTANCE]
    owners = _modality_comparison_owners()
    dependency = (
        {
            "task_cid": "task:codec",
            "digest": "sha256:" + ("95" * 32),
            "body": _codec_intact_boundary_dependency(),
        },
    )
    evidence = modality_causal_comparison_evidence(
        criteria, dependency, owners, EXPECTED_MODALITY_CAUSAL_COMPARISON_PROBE
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_modality_causal_comparison_boundaries"
    )
    assert evidence["hypothesis_selected"] is False
    assert evidence["training_executed"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["completion_authority"] is False
    approved = _codec_intact_boundary_dependency(meanings_approved=True)
    assert modality_causal_comparison_evidence(
        criteria,
        (
            {
                "task_cid": "task:codec",
                "digest": "sha256:" + ("95" * 32),
                "body": approved,
            },
        ),
        owners,
        EXPECTED_MODALITY_CAUSAL_COMPARISON_PROBE,
    ) is None
    selected = dict(EXPECTED_MODALITY_CAUSAL_COMPARISON_PROBE)
    selected["hypothesis_selected"] = True
    assert modality_causal_comparison_evidence(
        criteria, dependency, owners, selected
    ) is None
    trained = dict(EXPECTED_MODALITY_CAUSAL_COMPARISON_PROBE)
    trained["training_executed"] = True
    assert modality_causal_comparison_evidence(
        criteria, dependency, owners, trained
    ) is None
    admitted = dict(EXPECTED_MODALITY_CAUSAL_COMPARISON_PROBE)
    admitted["admitted"] = True
    assert modality_causal_comparison_evidence(
        criteria, dependency, owners, admitted
    ) is None
    dropped = _modality_comparison_owners()
    dropped["plan"] = dict(dropped["plan"])
    dropped["plan"]["text"] = "plan only\n"
    assert modality_causal_comparison_evidence(
        criteria, dependency, dropped, EXPECTED_MODALITY_CAUSAL_COMPARISON_PROBE
    ) is None
    for other in (
        _RELEASE_COORDINATOR_ACCEPTANCE,
        _TRAINING_NEXT_ACCEPTANCE,
        _COVERAGE_ACCEPTANCE,
        _INCREMENTAL_BUILD_ACCEPTANCE,
        _INVALIDATION_ACCEPTANCE,
    ):
        assert modality_causal_comparison_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_MODALITY_CAUSAL_COMPARISON_PROBE,
        ) is None


def test_release_records_modality_causal_comparison(tmp_path: Path) -> None:
    relative = []
    for name, source in _modality_comparison_owner_paths().items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        relative.append("owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:codec",
        "LIR-CODEC",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:codec",
        evidence_kind="validation",
        digest="sha256:" + ("95" * 32),
        body=_codec_intact_boundary_dependency(),
    )
    _task(
        repo,
        "task:approved",
        "LIR-APPROVED-CODEC",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:approved",
        evidence_kind="validation",
        digest="sha256:" + ("96" * 32),
        body=_codec_intact_boundary_dependency(meanings_approved=True),
    )
    _task(
        repo,
        "task:comparison",
        "LIR-COMPARISON",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:codec"],
        acceptance=_MODALITY_COMPARISON_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare-comparison",
        "LIR-BARE-COMPARISON",
        status="blocked",
        body=_manual_body("", stage="analysis"),
        dependencies=["task:codec"],
        acceptance=_MODALITY_COMPARISON_ACCEPTANCE,
    )
    _task(
        repo,
        "task:approved-comparison",
        "LIR-APPROVED-COMPARISON",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:approved"],
        acceptance=_MODALITY_COMPARISON_ACCEPTANCE,
    )
    _task(
        repo,
        "task:training-still",
        "LIR-TRAINING-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="training"),
            "existing_owners": relative,
        },
        dependencies=["task:codec"],
        acceptance=_TRAINING_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:coverage-still",
        "LIR-COVERAGE-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="evaluation"),
            "existing_owners": relative,
        },
        dependencies=["task:codec"],
        acceptance=_COVERAGE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:coordinator-still",
        "LIR-COORDINATOR-STILL",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:codec"],
        acceptance=_RELEASE_COORDINATOR_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-COMPARISON"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-COMPARISON",
        "LIR-APPROVED-COMPARISON",
        "LIR-TRAINING-STILL",
        "LIR-COVERAGE-STILL",
        "LIR-COORDINATOR-STILL",
    }
    assert result["held"] == []
    task = repo.get_task("task:comparison")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_modality_causal_comparison_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    for cid in (
        "task:bare-comparison",
        "task:approved-comparison",
        "task:training-still",
        "task:coverage-still",
        "task:coordinator-still",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


_MODALITY_ERROR_REPORT_ACCEPTANCE = (
    {
        "criterion": (
            "Presence versus owner errors and critical modality reversals "
            "are separately reported."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Failed acceptance remains failed despite aggregate gains; "
            "no automatic default/teacher promotion."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Materialize separate producer and post-freeze evaluation "
            "packets/owners; primary training/inference authority grants "
            "no evaluation reference access."
        ),
        "evidence_kind": "validation",
    },
)


def _comparison_not_selection_dependency(
    *, hypothesis_selected: bool = False
) -> dict[str, object]:
    return {
        "operation": "dependency_block_resolved_modality_causal_comparison_boundaries",
        "hypothesis_selected": hypothesis_selected,
        "training_executed": False,
        "e1_ranking_selects_hypothesis": False,
        "legal_meaning_established": False,
        "legal_acceptance_claimed": False,
        "admitted": False,
        "formalized": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }


def _modality_error_owner_paths() -> dict[str, Path]:
    root = Path("/home/barberb/lift_coding/external/ipfs_datasets")
    return {
        "legal_span_modality_conditions_retention.py": root
        / "ipfs_datasets_py/optimizers/logic_theorem_optimizer"
        / "legal_span_modality_conditions_retention.py",
        "legal_relational_batch_evaluation_v1.py": root
        / "ipfs_datasets_py/logic/autoformal/legal_relational_batch_evaluation_v1.py",
    }


def _modality_error_owners() -> dict[str, dict[str, str]]:
    import hashlib

    roles = {
        "legal_span_modality_conditions_retention.py": "modality",
        "legal_relational_batch_evaluation_v1.py": "evaluation",
    }
    owners = {}
    for name, path in _modality_error_owner_paths().items():
        raw = path.read_bytes()
        owners[roles[name]] = {
            "path": name,
            "text": raw.decode("utf-8"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return owners


def test_modality_error_report_probe_matches_separated_rows() -> None:
    source = Path(
        "/home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/"
        "logic/autoformal/legal_modality_error_report_v1.py"
    ).read_text(encoding="utf-8")
    assert "def report_separated_errors" in source
    assert "def retain_failed_acceptance" in source
    assert "def materialize_separated_packets" in source
    assert "automatic default or teacher promotion refused" in source
    assert "producer packet grants evaluation reference access" in source
    measured = measure_modality_error_report_probe()
    assert measured == EXPECTED_MODALITY_ERROR_REPORT_PROBE
    assert measured["presence_error_count"] == 1
    assert measured["owner_error_count"] == 1
    assert measured["critical_modality_reversal_count"] == 1
    assert measured["presence_owner_reversal_separated"] is True
    assert measured["acceptance_failed_retained"] is True
    assert measured["aggregate_gain"] == 4
    assert measured["default_promoted"] is False
    assert measured["teacher_promoted"] is False
    assert measured["automatic_promotion_refused"] is True
    assert measured["producer_evaluation_reference_access"] is False
    assert measured["producer_written_before_evaluation"] is True
    assert measured["training_executed"] is False
    assert measured["model_executed"] is False
    assert measured["admitted"] is False
    assert measured["formalized"] is False
    assert measured["legal_acceptance_claimed"] is False
    assert measured["federal_corpus_scored"] is False


def test_modality_error_report_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _MODALITY_ERROR_REPORT_ACCEPTANCE]
    owners = _modality_error_owners()
    dependency = (
        {
            "task_cid": "task:comparison",
            "digest": "sha256:" + ("97" * 32),
            "body": _comparison_not_selection_dependency(),
        },
    )
    evidence = modality_error_report_evidence(
        criteria, dependency, owners, EXPECTED_MODALITY_ERROR_REPORT_PROBE
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_modality_error_report_boundaries"
    )
    assert evidence["acceptance_failed_retained"] is True
    assert evidence["aggregate_gain"] == 4
    assert evidence["default_promoted"] is False
    assert evidence["teacher_promoted"] is False
    assert evidence["training_executed"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["completion_authority"] is False
    selected = _comparison_not_selection_dependency(hypothesis_selected=True)
    assert modality_error_report_evidence(
        criteria,
        (
            {
                "task_cid": "task:comparison",
                "digest": "sha256:" + ("97" * 32),
                "body": selected,
            },
        ),
        owners,
        EXPECTED_MODALITY_ERROR_REPORT_PROBE,
    ) is None
    trained = dict(EXPECTED_MODALITY_ERROR_REPORT_PROBE)
    trained["training_executed"] = True
    assert modality_error_report_evidence(
        criteria, dependency, owners, trained
    ) is None
    admitted = dict(EXPECTED_MODALITY_ERROR_REPORT_PROBE)
    admitted["admitted"] = True
    assert modality_error_report_evidence(
        criteria, dependency, owners, admitted
    ) is None
    dropped = _modality_error_owners()
    dropped["evaluation"] = dict(dropped["evaluation"])
    dropped["evaluation"]["text"] = "evaluation only\n"
    assert modality_error_report_evidence(
        criteria, dependency, dropped, EXPECTED_MODALITY_ERROR_REPORT_PROBE
    ) is None
    for other in (
        _MODALITY_COMPARISON_ACCEPTANCE,
        _TRAINING_NEXT_ACCEPTANCE,
        _COVERAGE_ACCEPTANCE,
        _RELEASE_COORDINATOR_ACCEPTANCE,
    ):
        assert modality_error_report_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_MODALITY_ERROR_REPORT_PROBE,
        ) is None


def test_release_records_modality_error_report(tmp_path: Path) -> None:
    relative = []
    for name, source in _modality_error_owner_paths().items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        relative.append("owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:comparison",
        "LIR-COMPARISON",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:comparison",
        evidence_kind="validation",
        digest="sha256:" + ("97" * 32),
        body=_comparison_not_selection_dependency(),
    )
    _task(
        repo,
        "task:selected",
        "LIR-SELECTED-COMPARISON",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:selected",
        evidence_kind="validation",
        digest="sha256:" + ("98" * 32),
        body=_comparison_not_selection_dependency(hypothesis_selected=True),
    )
    _task(
        repo,
        "task:errors",
        "LIR-ERRORS",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="training"),
            "existing_owners": relative,
        },
        dependencies=["task:comparison"],
        acceptance=_MODALITY_ERROR_REPORT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare-errors",
        "LIR-BARE-ERRORS",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="training"),
        dependencies=["task:comparison"],
        acceptance=_MODALITY_ERROR_REPORT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:selected-errors",
        "LIR-SELECTED-ERRORS",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="training"),
            "existing_owners": relative,
        },
        dependencies=["task:selected"],
        acceptance=_MODALITY_ERROR_REPORT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:training-still",
        "LIR-TRAINING-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="training"),
            "existing_owners": relative,
        },
        dependencies=["task:comparison"],
        acceptance=_TRAINING_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:coverage-still",
        "LIR-COVERAGE-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="evaluation"),
            "existing_owners": relative,
        },
        dependencies=["task:comparison"],
        acceptance=_COVERAGE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:coordinator-still",
        "LIR-COORDINATOR-STILL",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:comparison"],
        acceptance=_RELEASE_COORDINATOR_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-ERRORS"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-ERRORS",
        "LIR-SELECTED-ERRORS",
        "LIR-TRAINING-STILL",
        "LIR-COVERAGE-STILL",
        "LIR-COORDINATOR-STILL",
    }
    assert result["held"] == []
    task = repo.get_task("task:errors")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_modality_error_report_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    for cid in (
        "task:bare-errors",
        "task:selected-errors",
        "task:training-still",
        "task:coverage-still",
        "task:coordinator-still",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


def _typed_decoder_not_trained_dependency(
    *, training_executed: bool = False
) -> dict[str, object]:
    return {
        "operation": "dependency_block_resolved_typed_decoder_controls",
        "training_executed": training_executed,
        "eligibility_granted": False,
        "direct_kl_across_incompatible_codecs": False,
        "admitted": False,
        "formalized": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }


def _training_metric_owner_paths() -> dict[str, Path]:
    root = Path("/home/barberb/lift_coding/external/ipfs_datasets")
    return {
        "legal_grouped_span_decoder.py": root
        / "ipfs_datasets_py/logic/formalization/autoencoder"
        / "legal_grouped_span_decoder.py",
        "legal_relational_decoder_example_v1.py": root
        / "ipfs_datasets_py/logic/formalization/autoencoder"
        / "legal_relational_decoder_example_v1.py",
    }


def _training_metric_owners() -> dict[str, dict[str, str]]:
    import hashlib

    roles = {
        "legal_grouped_span_decoder.py": "decoder",
        "legal_relational_decoder_example_v1.py": "example",
    }
    owners = {}
    for name, path in _training_metric_owner_paths().items():
        raw = path.read_bytes()
        owners[roles[name]] = {
            "path": name,
            "text": raw.decode("utf-8"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return owners


def test_training_metric_probe_matches_separated_records() -> None:
    source = Path(
        "/home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/"
        "logic/autoformal/legal_training_metric_report_v1.py"
    ).read_text(encoding="utf-8")
    assert "def report_metric_groups" in source
    assert "def freeze_output" in source
    assert "def attach_engineering_scores" in source
    assert "def materialize_separated_packets" in source
    assert "outputs must freeze before scoring" in source
    assert "statutory fidelity is not an engineering score" in source
    assert "producer packet grants evaluation reference access" in source
    measured = measure_training_metric_probe()
    assert measured == EXPECTED_TRAINING_METRIC_PROBE
    assert measured["ordinary_generation_count"] == 1
    assert measured["full_rules_count"] == 1
    assert measured["argument_attachment_modality_count"] == 1
    assert measured["protected_retention_count"] == 1
    assert measured["metric_groups_separated"] is True
    assert measured["outputs_frozen_before_scoring"] is True
    assert measured["engineering_scores_separate"] is True
    assert measured["engineering_scores_are_statutory_fidelity"] is False
    assert measured["statutory_fidelity_reviewed"] is False
    assert measured["producer_evaluation_reference_access"] is False
    assert measured["training_executed"] is False
    assert measured["model_executed"] is False
    assert measured["worker_launched"] is False
    assert measured["encoder_unfrozen"] is False
    assert measured["admitted"] is False
    assert measured["formalized"] is False
    assert measured["federal_corpus_scored"] is False


def test_training_metric_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _TRAINING_NEXT_ACCEPTANCE]
    owners = _training_metric_owners()
    dependency = (
        {
            "task_cid": "task:decoder",
            "digest": "sha256:" + ("97" * 32),
            "body": _typed_decoder_not_trained_dependency(),
        },
    )
    evidence = training_metric_evidence(
        criteria, dependency, owners, EXPECTED_TRAINING_METRIC_PROBE
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_training_metric_boundaries"
    )
    assert evidence["training_executed"] is False
    assert evidence["statutory_fidelity_reviewed"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["completion_authority"] is False
    trained = _typed_decoder_not_trained_dependency(training_executed=True)
    assert training_metric_evidence(
        criteria,
        (
            {
                "task_cid": "task:decoder",
                "digest": "sha256:" + ("97" * 32),
                "body": trained,
            },
        ),
        owners,
        EXPECTED_TRAINING_METRIC_PROBE,
    ) is None
    scored = dict(EXPECTED_TRAINING_METRIC_PROBE)
    scored["statutory_fidelity_reviewed"] = True
    assert training_metric_evidence(
        criteria, dependency, owners, scored
    ) is None
    admitted = dict(EXPECTED_TRAINING_METRIC_PROBE)
    admitted["admitted"] = True
    assert training_metric_evidence(
        criteria, dependency, owners, admitted
    ) is None
    dropped = _training_metric_owners()
    dropped["example"] = dict(dropped["example"])
    dropped["example"]["text"] = "example only\n"
    assert training_metric_evidence(
        criteria, dependency, dropped, EXPECTED_TRAINING_METRIC_PROBE
    ) is None
    for other in (
        _MODALITY_ERROR_REPORT_ACCEPTANCE,
        _COVERAGE_ACCEPTANCE,
        _RELEASE_COORDINATOR_ACCEPTANCE,
        _MODALITY_COMPARISON_ACCEPTANCE,
    ):
        assert training_metric_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_TRAINING_METRIC_PROBE,
        ) is None


def test_release_records_training_metric_report(tmp_path: Path) -> None:
    relative = []
    for name, source in _training_metric_owner_paths().items():
        path = tmp_path / "owners" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        relative.append("owners/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:decoder",
        "LIR-DECODER",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:decoder",
        evidence_kind="validation",
        digest="sha256:" + ("97" * 32),
        body=_typed_decoder_not_trained_dependency(),
    )
    _task(
        repo,
        "task:trained",
        "LIR-TRAINED-DECODER",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:trained",
        evidence_kind="validation",
        digest="sha256:" + ("98" * 32),
        body=_typed_decoder_not_trained_dependency(training_executed=True),
    )
    _task(
        repo,
        "task:metrics",
        "LIR-METRICS",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="training"),
            "existing_owners": relative,
        },
        dependencies=["task:decoder"],
        acceptance=_TRAINING_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare-metrics",
        "LIR-BARE-METRICS",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="training"),
        dependencies=["task:decoder"],
        acceptance=_TRAINING_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:trained-metrics",
        "LIR-TRAINED-METRICS",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="training"),
            "existing_owners": relative,
        },
        dependencies=["task:trained"],
        acceptance=_TRAINING_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:errors-still",
        "LIR-ERRORS-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="training"),
            "existing_owners": relative,
        },
        dependencies=["task:decoder"],
        acceptance=_MODALITY_ERROR_REPORT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:coverage-still",
        "LIR-COVERAGE-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="evaluation"),
            "existing_owners": relative,
        },
        dependencies=["task:decoder"],
        acceptance=_COVERAGE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:coordinator-still",
        "LIR-COORDINATOR-STILL",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:decoder"],
        acceptance=_RELEASE_COORDINATOR_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-METRICS"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-METRICS",
        "LIR-TRAINED-METRICS",
        "LIR-ERRORS-STILL",
        "LIR-COVERAGE-STILL",
        "LIR-COORDINATOR-STILL",
    }
    assert result["held"] == []
    task = repo.get_task("task:metrics")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_training_metric_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    for cid in (
        "task:bare-metrics",
        "task:trained-metrics",
        "task:errors-still",
        "task:coverage-still",
        "task:coordinator-still",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


_WIDTH_CONTROL_ACCEPTANCE = (
    {
        "criterion": (
            "Historical 8D variants, active native vectors and detached "
            "disabled-latent components remain distinct."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Correct, zero, branch-disabled and permuted inputs plus "
            "source-only/retrieval controls are explicit; mean-coordinate "
            "MSE is not semantic superiority."
        ),
        "evidence_kind": "validation",
    },
)
_PRODUCER_STILL_ACCEPTANCE = (
    {
        "criterion": (
            "Actual producer/encoder calls and declared source access are evidenced."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Mock projections and different historical 8D tasks are not substituted."
        ),
        "evidence_kind": "validation",
    },
)


def _width_control_owner_paths() -> dict[str, Path]:
    root = Path(
        "/home/barberb/lift_coding/external/ipfs_datasets/docs/autoencoders"
    )
    return {
        "legal_ir_asset_reconciliation_20261016.md": root
        / "legal_ir_asset_reconciliation_20261016.md",
        "gte_decoder_native_inputs.md": root / "gte_decoder_native_inputs.md",
    }


def _width_control_owners() -> dict[str, dict[str, str]]:
    import hashlib

    roles = {
        "legal_ir_asset_reconciliation_20261016.md": "reconciliation",
        "gte_decoder_native_inputs.md": "native_inputs",
    }
    owners = {}
    for name, path in _width_control_owner_paths().items():
        raw = path.read_bytes()
        owners[roles[name]] = {
            "path": "docs/autoencoders/" + name,
            "text": raw.decode("utf-8"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return owners


def _training_metric_clean_dependency(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "operation": "dependency_block_resolved_training_metric_boundaries",
        "training_executed": False,
        "model_executed": False,
        "encoder_unfrozen": False,
        "federal_corpus_scored": False,
        "legal_meaning_established": False,
        "legal_acceptance_claimed": False,
        "admitted": False,
        "formalized": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }
    body.update(overrides)
    return body


def test_width_control_probe_matches_named_unexecuted_controls() -> None:
    source = Path(
        "/home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/"
        "logic/autoformal/legal_width_control_specification_v1.py"
    ).read_text(encoding="utf-8")
    assert "def specify_width_controls" in source
    assert "width classes remain distinct" in source
    assert "controls stay unexecuted" in source
    assert "mean-coordinate MSE is not semantic superiority" in source
    reconciliation = _width_control_owner_paths()[
        "legal_ir_asset_reconciliation_20261016.md"
    ].read_text(encoding="utf-8")
    native_inputs = _width_control_owner_paths()[
        "gte_decoder_native_inputs.md"
    ].read_text(encoding="utf-8")
    for marker in (
        "Historical 8D variants, active native vectors and detached disabled-latent components remain distinct.",
        "Correct, zero, branch-disabled and permuted inputs plus source-only and retrieval controls stay named and unexecuted.",
        "Mean-coordinate MSE is not semantic superiority.",
        "No control was executed and no mean-coordinate MSE was measured.",
        "without constructing ModelManager",
        "None of these inventory observations grants teacher,",
        "Referenced artifacts publish before the coordinator receipt; no self-referential commit hash.",
        "Candidate, reviewed, replay-qualified and promoted statuses remain separate, including ModelManager selectors.",
    ):
        assert marker in reconciliation
    for marker in (
        "Historical 8D variants, active native vectors and detached disabled-latent components remain distinct.",
        "Mean-coordinate MSE is not semantic superiority.",
        "No control was executed and no mean-coordinate MSE was measured.",
    ):
        assert marker in native_inputs
    measured = measure_width_control_probe()
    assert measured == EXPECTED_WIDTH_CONTROL_PROBE
    assert measured["controls_executed"] is False
    assert measured["mean_coordinate_mse_measured"] is False
    assert measured["mean_coordinate_mse_is_semantic_superiority"] is False
    assert measured["producer_called"] is False
    assert measured["encoder_called"] is False
    assert measured["preprocessing_specified"] is False
    assert measured["budgets_recorded"] is False
    assert measured["cost_measured"] is False
    assert measured["training_executed"] is False
    assert measured["model_executed"] is False
    assert measured["admitted"] is False
    assert measured["formalized"] is False
    assert measured["width_classes_distinct"] is True
    assert measured["controls_explicit"] is True


def test_width_control_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _WIDTH_CONTROL_ACCEPTANCE]
    owners = _width_control_owners()
    dependency = (
        {
            "task_cid": "task:metrics",
            "digest": "sha256:" + ("97" * 32),
            "body": _training_metric_clean_dependency(),
        },
    )
    evidence = width_control_evidence(
        criteria, dependency, owners, EXPECTED_WIDTH_CONTROL_PROBE
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_width_control_boundaries"
    )
    assert evidence["controls_executed"] is False
    assert evidence["mean_coordinate_mse_measured"] is False
    assert evidence["producer_called"] is False
    assert evidence["encoder_called"] is False
    assert evidence["training_executed"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["completion_authority"] is False
    trained = _training_metric_clean_dependency(training_executed=True)
    assert width_control_evidence(
        criteria,
        (
            {
                "task_cid": "task:metrics",
                "digest": "sha256:" + ("97" * 32),
                "body": trained,
            },
        ),
        owners,
        EXPECTED_WIDTH_CONTROL_PROBE,
    ) is None
    measured = dict(EXPECTED_WIDTH_CONTROL_PROBE)
    measured["mean_coordinate_mse_measured"] = True
    assert width_control_evidence(
        criteria, dependency, owners, measured
    ) is None
    executed = dict(EXPECTED_WIDTH_CONTROL_PROBE)
    executed["controls_executed"] = True
    assert width_control_evidence(
        criteria, dependency, owners, executed
    ) is None
    admitted = dict(EXPECTED_WIDTH_CONTROL_PROBE)
    admitted["admitted"] = True
    assert width_control_evidence(
        criteria, dependency, owners, admitted
    ) is None
    dropped = _width_control_owners()
    dropped["native_inputs"] = dict(dropped["native_inputs"])
    dropped["native_inputs"]["text"] = "moved documentation only\n"
    assert width_control_evidence(
        criteria, dependency, dropped, EXPECTED_WIDTH_CONTROL_PROBE
    ) is None
    for other in (
        _TRAINING_NEXT_ACCEPTANCE,
        _COVERAGE_ACCEPTANCE,
        _RELEASE_COORDINATOR_ACCEPTANCE,
        _PRODUCER_STILL_ACCEPTANCE,
        _MODALITY_ERROR_REPORT_ACCEPTANCE,
    ):
        assert width_control_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_WIDTH_CONTROL_PROBE,
        ) is None


def test_release_records_width_control_specification(tmp_path: Path) -> None:
    relative = []
    for name, source in _width_control_owner_paths().items():
        path = tmp_path / "docs" / "autoencoders" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        relative.append("docs/autoencoders/" + name)
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:metrics",
        "LIR-METRICS-DEP",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:metrics",
        evidence_kind="validation",
        digest="sha256:" + ("97" * 32),
        body=_training_metric_clean_dependency(),
    )
    _task(
        repo,
        "task:trained",
        "LIR-TRAINED-METRICS",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:trained",
        evidence_kind="validation",
        digest="sha256:" + ("98" * 32),
        body=_training_metric_clean_dependency(training_executed=True),
    )
    _task(
        repo,
        "task:widths",
        "LIR-WIDTHS",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:metrics"],
        acceptance=_WIDTH_CONTROL_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare-widths",
        "LIR-BARE-WIDTHS",
        status="blocked",
        body=_manual_body("", stage="analysis"),
        dependencies=["task:metrics"],
        acceptance=_WIDTH_CONTROL_ACCEPTANCE,
    )
    _task(
        repo,
        "task:trained-widths",
        "LIR-TRAINED-WIDTHS",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:trained"],
        acceptance=_WIDTH_CONTROL_ACCEPTANCE,
    )
    _task(
        repo,
        "task:producer-still",
        "LIR-PRODUCER-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="inference"),
            "existing_owners": relative[:1],
        },
        dependencies=["task:widths"],
        acceptance=_PRODUCER_STILL_ACCEPTANCE,
    )
    _task(
        repo,
        "task:metrics-still",
        "LIR-METRICS-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="training"),
            "existing_owners": relative,
        },
        dependencies=["task:metrics"],
        acceptance=_TRAINING_NEXT_ACCEPTANCE,
    )
    _task(
        repo,
        "task:coverage-still",
        "LIR-COVERAGE-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="evaluation"),
            "existing_owners": relative,
        },
        dependencies=["task:metrics"],
        acceptance=_COVERAGE_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {
        "LIR-WIDTHS",
        "LIR-PRODUCER-STILL",
    }
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-WIDTHS",
        "LIR-TRAINED-WIDTHS",
        "LIR-METRICS-STILL",
        "LIR-COVERAGE-STILL",
    }
    assert result["held"] == []
    task = repo.get_task("task:widths")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_width_control_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    producer = repo.get_task("task:producer-still")
    assert producer["status"] == "completed"
    producer_receipt = producer["body"]["completion_receipt"]
    assert producer_receipt["operation"] == (
        "dependency_block_resolved_sparse8_comparison_cell"
    )
    assert producer_receipt["admitted"] is False
    assert producer_receipt["formalized"] is False
    assert producer_receipt["gold_or_teacher_admission_granted"] is False
    assert producer_receipt["constitution_roundtrip_ok"] is False
    assert producer_receipt["completion_authority"] is False
    assert "producer_called" not in producer_receipt
    for cid in (
        "task:bare-widths",
        "task:trained-widths",
        "task:metrics-still",
        "task:coverage-still",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


_CONTEXT_BUDGET_ACCEPTANCE = (
    {
        "criterion": (
            "Same decoder and source budget for "
            "context/no-context/shuffled/unrelated-context and "
            "retrieval-only controls."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Definition dependencies and cross-split context allowances "
            "are audited."
        ),
        "evidence_kind": "validation",
    },
)
_OCCURRENCE_STILL_ACCEPTANCE = (
    {
        "criterion": (
            "Every selected occurrence and unresolved/unsupported outcome "
            "remains visible."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Retrieved locator evidence is not automatically a "
            "governing-definition edge."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Materialize separate producer and post-freeze evaluation "
            "packets/owners; primary training/inference authority grants "
            "no evaluation reference access."
        ),
        "evidence_kind": "validation",
    },
)


def _context_budget_owner_paths() -> dict[str, Path]:
    root = Path("/home/barberb/lift_coding/external/ipfs_datasets")
    return {
        "scoped_definition_resolution.py": root
        / "ipfs_datasets_py/logic/autoformal/scoped_definition_resolution.py",
        "legal_relational_decoder_example_v1.py": root
        / "ipfs_datasets_py/logic/formalization/autoencoder/legal_relational_decoder_example_v1.py",
    }


def _context_budget_owners() -> dict[str, dict[str, str]]:
    import hashlib

    roles = {
        "scoped_definition_resolution.py": "lookup",
        "legal_relational_decoder_example_v1.py": "example",
    }
    owners = {}
    for name, path in _context_budget_owner_paths().items():
        raw = path.read_bytes()
        owners[roles[name]] = {
            "path": "ipfs_datasets_py/logic/autoformal/" + name
            if name.startswith("scoped")
            else "ipfs_datasets_py/logic/formalization/autoencoder/" + name,
            "text": raw.decode("utf-8"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return owners


def _declared_closure_clean_dependency(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "operation": "dependency_block_resolved_declared_closure_frontier",
        "legal_dependency_inventory_exhaustive": False,
        "interpretations_unioned": False,
        "admitted": False,
        "formalized": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }
    body.update(overrides)
    return body


def test_context_budget_probe_matches_separated_pins() -> None:
    source = Path(
        "/home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/"
        "logic/autoformal/legal_context_budget_audit_v1.py"
    ).read_text(encoding="utf-8")
    assert "def audit_context_budget" in source
    assert "decoder stays unexecuted" in source
    assert "cross-split context allowance is not granted" in source
    assert "definition pins overlap context pins" in source
    lookup = _context_budget_owner_paths()[
        "scoped_definition_resolution.py"
    ].read_text(encoding="utf-8")
    example = _context_budget_owner_paths()[
        "legal_relational_decoder_example_v1.py"
    ].read_text(encoding="utf-8")
    for marker in (
        "Context, no-context, shuffled, unrelated-context and retrieval-only controls share one decoder and one source budget.",
        "Definition dependencies stay separate from context pins.",
        "Cross-split context allowance is not granted.",
        "No decoder was executed.",
    ):
        assert marker in lookup
        assert marker in example
    assert "build_definition_lookup_dependencies" in lookup.split("CONTEXT_BUDGET_BOUNDARY", 1)[0]
    measured = measure_context_budget_audit_probe()
    assert measured == EXPECTED_CONTEXT_BUDGET_PROBE
    assert measured["authored_fixture_only"] is True
    assert measured["definition_pins_separate_from_context_pins"] is True
    assert measured["controls_share_decoder"] is True
    assert measured["controls_share_source_budget"] is True
    assert measured["cross_split_context_allowance_count"] == 0
    assert measured["cross_split_context_allowance_granted"] is False
    assert measured["decoder_executed"] is False
    assert measured["lookup_status"] == "unavailable"
    assert measured["selected_symbol_id_absent"] is True
    assert measured["training_executed"] is False
    assert measured["model_executed"] is False
    assert measured["federal_corpus_scored"] is False
    assert measured["legal_meaning_established"] is False
    assert measured["legal_acceptance_claimed"] is False
    assert measured["admitted"] is False
    assert measured["formalized"] is False


def test_context_budget_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _CONTEXT_BUDGET_ACCEPTANCE]
    owners = _context_budget_owners()
    dependency = (
        {
            "task_cid": "task:metrics",
            "digest": "sha256:" + ("97" * 32),
            "body": _training_metric_clean_dependency(),
        },
        {
            "task_cid": "task:closure",
            "digest": "sha256:" + ("96" * 32),
            "body": _declared_closure_clean_dependency(),
        },
    )
    evidence = context_budget_evidence(
        criteria, dependency, owners, EXPECTED_CONTEXT_BUDGET_PROBE
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_context_budget_boundaries"
    )
    assert evidence["authored_fixture_only"] is True
    assert evidence["definition_pins_separate_from_context_pins"] is True
    assert evidence["cross_split_context_allowance_count"] == 0
    assert evidence["decoder_executed"] is False
    assert evidence["training_executed"] is False
    assert evidence["federal_corpus_scored"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["completion_authority"] is False
    exhaustive = _declared_closure_clean_dependency(
        legal_dependency_inventory_exhaustive=True
    )
    assert context_budget_evidence(
        criteria,
        (
            dependency[0],
            {
                "task_cid": "task:closure",
                "digest": "sha256:" + ("96" * 32),
                "body": exhaustive,
            },
        ),
        owners,
        EXPECTED_CONTEXT_BUDGET_PROBE,
    ) is None
    trained = _training_metric_clean_dependency(training_executed=True)
    assert context_budget_evidence(
        criteria,
        (
            {
                "task_cid": "task:metrics",
                "digest": "sha256:" + ("97" * 32),
                "body": trained,
            },
            dependency[1],
        ),
        owners,
        EXPECTED_CONTEXT_BUDGET_PROBE,
    ) is None
    measured = dict(EXPECTED_CONTEXT_BUDGET_PROBE)
    measured["decoder_executed"] = True
    assert context_budget_evidence(
        criteria, dependency, owners, measured
    ) is None
    granted = dict(EXPECTED_CONTEXT_BUDGET_PROBE)
    granted["cross_split_context_allowance_granted"] = True
    assert context_budget_evidence(
        criteria, dependency, owners, granted
    ) is None
    admitted = dict(EXPECTED_CONTEXT_BUDGET_PROBE)
    admitted["admitted"] = True
    assert context_budget_evidence(
        criteria, dependency, owners, admitted
    ) is None
    dropped = _context_budget_owners()
    dropped["example"] = dict(dropped["example"])
    dropped["example"]["text"] = "candidate example only\n"
    assert context_budget_evidence(
        criteria, dependency, dropped, EXPECTED_CONTEXT_BUDGET_PROBE
    ) is None
    for other in (
        _WIDTH_CONTROL_ACCEPTANCE,
        _PRODUCER_STILL_ACCEPTANCE,
        _TRAINING_NEXT_ACCEPTANCE,
        _COVERAGE_ACCEPTANCE,
        _OCCURRENCE_STILL_ACCEPTANCE,
    ):
        assert context_budget_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_CONTEXT_BUDGET_PROBE,
        ) is None


def test_release_records_context_budget_audit(tmp_path: Path) -> None:
    relative = []
    for name, source in _context_budget_owner_paths().items():
        if name.startswith("scoped"):
            path = tmp_path / "ipfs_datasets_py/logic/autoformal" / name
            relative.append("ipfs_datasets_py/logic/autoformal/" + name)
        else:
            path = (
                tmp_path
                / "ipfs_datasets_py/logic/formalization/autoencoder"
                / name
            )
            relative.append(
                "ipfs_datasets_py/logic/formalization/autoencoder/" + name
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:metrics",
        "LIR-METRICS-DEP",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:metrics",
        evidence_kind="validation",
        digest="sha256:" + ("97" * 32),
        body=_training_metric_clean_dependency(),
    )
    _task(
        repo,
        "task:closure",
        "LIR-CLOSURE-DEP",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:closure",
        evidence_kind="validation",
        digest="sha256:" + ("96" * 32),
        body=_declared_closure_clean_dependency(),
    )
    _task(
        repo,
        "task:trained",
        "LIR-TRAINED-METRICS",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:trained",
        evidence_kind="validation",
        digest="sha256:" + ("98" * 32),
        body=_training_metric_clean_dependency(training_executed=True),
    )
    _task(
        repo,
        "task:exhaustive",
        "LIR-EXHAUSTIVE-CLOSURE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:exhaustive",
        evidence_kind="validation",
        digest="sha256:" + ("95" * 32),
        body=_declared_closure_clean_dependency(
            legal_dependency_inventory_exhaustive=True
        ),
    )
    _task(
        repo,
        "task:context",
        "LIR-CONTEXT",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:metrics", "task:closure"],
        acceptance=_CONTEXT_BUDGET_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare-context",
        "LIR-BARE-CONTEXT",
        status="blocked",
        body=_manual_body("", stage="analysis"),
        dependencies=["task:metrics", "task:closure"],
        acceptance=_CONTEXT_BUDGET_ACCEPTANCE,
    )
    _task(
        repo,
        "task:trained-context",
        "LIR-TRAINED-CONTEXT",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:trained", "task:closure"],
        acceptance=_CONTEXT_BUDGET_ACCEPTANCE,
    )
    _task(
        repo,
        "task:exhaustive-context",
        "LIR-EXHAUSTIVE-CONTEXT",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:metrics", "task:exhaustive"],
        acceptance=_CONTEXT_BUDGET_ACCEPTANCE,
    )
    _task(
        repo,
        "task:occurrence-still",
        "LIR-OCCURRENCE-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="inference"),
            "existing_owners": relative[:1],
        },
        dependencies=["task:context"],
        acceptance=_OCCURRENCE_STILL_ACCEPTANCE,
    )
    _task(
        repo,
        "task:producer-still",
        "LIR-PRODUCER-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="inference"),
            "existing_owners": relative[:1],
        },
        dependencies=["task:context"],
        acceptance=_PRODUCER_STILL_ACCEPTANCE,
    )
    _task(
        repo,
        "task:widths-still",
        "LIR-WIDTHS-STILL",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:metrics", "task:closure"],
        acceptance=_WIDTH_CONTROL_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-CONTEXT"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-CONTEXT",
        "LIR-TRAINED-CONTEXT",
        "LIR-EXHAUSTIVE-CONTEXT",
        "LIR-OCCURRENCE-STILL",
        "LIR-PRODUCER-STILL",
        "LIR-WIDTHS-STILL",
    }
    assert result["held"] == []
    task = repo.get_task("task:context")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_context_budget_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    for cid in (
        "task:bare-context",
        "task:trained-context",
        "task:exhaustive-context",
        "task:occurrence-still",
        "task:producer-still",
        "task:widths-still",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


def _occurrence_owner_paths() -> dict[str, Path]:
    root = Path(
        "/home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/"
        "logic/autoformal"
    )
    return {
        "legal_relational_batch_evaluation_v1.py": root
        / "legal_relational_batch_evaluation_v1.py",
        "scoped_definition_resolution.py": root / "scoped_definition_resolution.py",
    }


def _occurrence_owners() -> dict[str, dict[str, str]]:
    import hashlib

    roles = {
        "legal_relational_batch_evaluation_v1.py": "evaluation",
        "scoped_definition_resolution.py": "lookup",
    }
    owners = {}
    for name, path in _occurrence_owner_paths().items():
        raw = path.read_bytes()
        owners[roles[name]] = {
            "path": "ipfs_datasets_py/logic/autoformal/" + name,
            "text": raw.decode("utf-8"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return owners


def _context_budget_clean_dependency(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "operation": "dependency_block_resolved_context_budget_boundaries",
        "decoder_executed": False,
        "training_executed": False,
        "model_executed": False,
        "worker_launched": False,
        "federal_corpus_scored": False,
        "authored_fixture_only": True,
        "cross_split_context_allowance_count": 0,
        "cross_split_context_allowance_granted": False,
        "legal_meaning_established": False,
        "legal_acceptance_claimed": False,
        "admitted": False,
        "formalized": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }
    body.update(overrides)
    return body


def test_occurrence_visibility_probe_matches_visible_outcomes() -> None:
    source = Path(
        "/home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/"
        "logic/autoformal/legal_occurrence_visibility_audit_v1.py"
    ).read_text(encoding="utf-8")
    assert "def audit_occurrence_visibility" in source
    assert "selected occurrence is not visible" in source
    assert "unsupported outcome is not visible" in source
    assert "retrieved locator is a governing-definition edge" in source
    assert "producer packet grants evaluation reference access" in source
    evaluation = _occurrence_owner_paths()[
        "legal_relational_batch_evaluation_v1.py"
    ].read_text(encoding="utf-8")
    lookup = _occurrence_owner_paths()[
        "scoped_definition_resolution.py"
    ].read_text(encoding="utf-8")
    for marker in (
        "Account for every expected occurrence and submitted declared observation.",
        "every observation must be retained exactly once",
        "Unsupported is an explicit caller abstention category; rejected typed outputs do not establish a syntax-versus-family-capability taxonomy.",
    ):
        assert marker in evaluation
    for marker in (
        "It does not discover governing scopes or legal\nprecedence.",
        "Retrieved locator evidence is not automatically a governing-definition edge.",
        "A retrieved source occurrence stays on the lookup record.",
    ):
        assert marker in lookup
    assert "build_definition_lookup_dependencies" in lookup.split(
        "OCCURRENCE_VISIBILITY_BOUNDARY", 1
    )[0]
    measured = measure_occurrence_visibility_probe()
    assert measured == EXPECTED_OCCURRENCE_VISIBILITY_PROBE
    assert measured["authored_fixture_only"] is True
    assert measured["selected_occurrence_count"] == 2
    assert measured["selected_occurrences_visible"] is True
    assert measured["unsupported_outcome_visible"] is True
    assert measured["missing_outcome_visible"] is True
    assert measured["unresolved_outcome_status"] == "missing"
    assert measured["locator_retained_on_lookup"] is True
    assert measured["locator_promoted_to_edge"] is False
    assert measured["governing_definition_edge_count"] == 0
    assert measured["producer_written_before_evaluation"] is True
    assert measured["separate_packet_owners"] is True
    assert measured["producer_evaluation_reference_access"] is False
    assert measured["producer_owner"] == "primary-training-inference"
    assert measured["evaluation_owner"] == "post-freeze-evaluation"
    assert measured["evaluation_metric_reference_access"] is True
    assert measured["decoder_executed"] is False
    assert measured["training_executed"] is False
    assert measured["model_executed"] is False
    assert measured["federal_corpus_scored"] is False
    assert measured["legal_meaning_established"] is False
    assert measured["legal_acceptance_claimed"] is False
    assert measured["admitted"] is False
    assert measured["formalized"] is False


def test_occurrence_visibility_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _OCCURRENCE_STILL_ACCEPTANCE]
    owners = _occurrence_owners()
    dependency = (
        {
            "task_cid": "task:context",
            "digest": "sha256:" + ("94" * 32),
            "body": _context_budget_clean_dependency(),
        },
    )
    evidence = occurrence_visibility_evidence(
        criteria, dependency, owners, EXPECTED_OCCURRENCE_VISIBILITY_PROBE
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_occurrence_visibility_boundaries"
    )
    assert evidence["authored_fixture_only"] is True
    assert evidence["selected_occurrences_visible"] is True
    assert evidence["unsupported_outcome_visible"] is True
    assert evidence["missing_outcome_visible"] is True
    assert evidence["locator_promoted_to_edge"] is False
    assert evidence["governing_definition_edge_count"] == 0
    assert evidence["producer_evaluation_reference_access"] is False
    assert evidence["producer_written_before_evaluation"] is True
    assert evidence["evaluation_metric_reference_access"] is True
    assert evidence["decoder_executed"] is False
    assert evidence["training_executed"] is False
    assert evidence["federal_corpus_scored"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["completion_authority"] is False
    executed = _context_budget_clean_dependency(decoder_executed=True)
    assert occurrence_visibility_evidence(
        criteria,
        (
            {
                "task_cid": "task:context",
                "digest": "sha256:" + ("94" * 32),
                "body": executed,
            },
        ),
        owners,
        EXPECTED_OCCURRENCE_VISIBILITY_PROBE,
    ) is None
    trained = _context_budget_clean_dependency(training_executed=True)
    assert occurrence_visibility_evidence(
        criteria,
        (
            {
                "task_cid": "task:context",
                "digest": "sha256:" + ("94" * 32),
                "body": trained,
            },
        ),
        owners,
        EXPECTED_OCCURRENCE_VISIBILITY_PROBE,
    ) is None
    granted = _context_budget_clean_dependency(
        cross_split_context_allowance_granted=True
    )
    assert occurrence_visibility_evidence(
        criteria,
        (
            {
                "task_cid": "task:context",
                "digest": "sha256:" + ("94" * 32),
                "body": granted,
            },
        ),
        owners,
        EXPECTED_OCCURRENCE_VISIBILITY_PROBE,
    ) is None
    measured = dict(EXPECTED_OCCURRENCE_VISIBILITY_PROBE)
    measured["locator_promoted_to_edge"] = True
    assert occurrence_visibility_evidence(
        criteria, dependency, owners, measured
    ) is None
    admitted = dict(EXPECTED_OCCURRENCE_VISIBILITY_PROBE)
    admitted["admitted"] = True
    assert occurrence_visibility_evidence(
        criteria, dependency, owners, admitted
    ) is None
    dropped = _occurrence_owners()
    dropped["evaluation"] = dict(dropped["evaluation"])
    dropped["evaluation"]["text"] = "candidate evaluation only\n"
    assert occurrence_visibility_evidence(
        criteria, dependency, dropped, EXPECTED_OCCURRENCE_VISIBILITY_PROBE
    ) is None
    for other in (
        _CONTEXT_BUDGET_ACCEPTANCE,
        _PRODUCER_STILL_ACCEPTANCE,
        _TRAINING_NEXT_ACCEPTANCE,
        _COVERAGE_ACCEPTANCE,
        _WIDTH_CONTROL_ACCEPTANCE,
    ):
        assert occurrence_visibility_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_OCCURRENCE_VISIBILITY_PROBE,
        ) is None


def test_release_records_occurrence_visibility_audit(tmp_path: Path) -> None:
    relative = []
    for name, source in _occurrence_owner_paths().items():
        path = tmp_path / "ipfs_datasets_py/logic/autoformal" / name
        relative.append("ipfs_datasets_py/logic/autoformal/" + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:context",
        "LIR-CONTEXT-DEP",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:context",
        evidence_kind="validation",
        digest="sha256:" + ("94" * 32),
        body=_context_budget_clean_dependency(),
    )
    _task(
        repo,
        "task:executed",
        "LIR-EXECUTED-CONTEXT",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:executed",
        evidence_kind="validation",
        digest="sha256:" + ("93" * 32),
        body=_context_budget_clean_dependency(decoder_executed=True),
    )
    _task(
        repo,
        "task:occurrence",
        "LIR-OCCURRENCE",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="inference"),
            "existing_owners": relative,
        },
        dependencies=["task:context"],
        acceptance=_OCCURRENCE_STILL_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare-occurrence",
        "LIR-BARE-OCCURRENCE",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="inference"),
        dependencies=["task:context"],
        acceptance=_OCCURRENCE_STILL_ACCEPTANCE,
    )
    _task(
        repo,
        "task:executed-occurrence",
        "LIR-EXECUTED-OCCURRENCE",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="inference"),
            "existing_owners": relative,
        },
        dependencies=["task:executed"],
        acceptance=_OCCURRENCE_STILL_ACCEPTANCE,
    )
    _task(
        repo,
        "task:context-still",
        "LIR-CONTEXT-STILL",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:context"],
        acceptance=_CONTEXT_BUDGET_ACCEPTANCE,
    )
    _task(
        repo,
        "task:producer-still",
        "LIR-PRODUCER-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="inference"),
            "existing_owners": relative,
        },
        dependencies=["task:context"],
        acceptance=_PRODUCER_STILL_ACCEPTANCE,
    )
    _task(
        repo,
        "task:widths-still",
        "LIR-WIDTHS-STILL",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:context"],
        acceptance=_WIDTH_CONTROL_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-OCCURRENCE"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-OCCURRENCE",
        "LIR-EXECUTED-OCCURRENCE",
        "LIR-CONTEXT-STILL",
        "LIR-PRODUCER-STILL",
        "LIR-WIDTHS-STILL",
    }
    assert result["held"] == []
    task = repo.get_task("task:occurrence")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_occurrence_visibility_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    for cid in (
        "task:bare-occurrence",
        "task:executed-occurrence",
        "task:context-still",
        "task:producer-still",
        "task:widths-still",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


_LABEL_MASK_ACCEPTANCE = (
    {
        "criterion": "Solver/resource failure is not a negative legal label.",
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Reviewed, compiler weak, synthetic and unresolved tiers have "
            "distinct masks and origins; parser/Lean success creates no gold."
        ),
        "evidence_kind": "validation",
    },
)


def _label_owner_paths() -> dict[str, Path]:
    root = Path("/home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py")
    return {
        "legal_relational_batch_evaluation_v1.py": root
        / "logic/autoformal/legal_relational_batch_evaluation_v1.py",
        "legal_relational_decoder_example_v1.py": root
        / "logic/formalization/autoencoder/legal_relational_decoder_example_v1.py",
    }


def _label_owners() -> dict[str, dict[str, str]]:
    import hashlib

    roles = {
        "legal_relational_batch_evaluation_v1.py": "evaluation",
        "legal_relational_decoder_example_v1.py": "example",
    }
    owners = {}
    for name, path in _label_owner_paths().items():
        raw = path.read_bytes()
        owners[roles[name]] = {
            "path": "ipfs_datasets_py/" + str(path).split("/ipfs_datasets_py/", 1)[1],
            "text": raw.decode("utf-8"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return owners


def _training_metric_clean_dependency(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "operation": "dependency_block_resolved_training_metric_boundaries",
        "training_executed": False,
        "model_executed": False,
        "encoder_unfrozen": False,
        "federal_corpus_scored": False,
        "legal_meaning_established": False,
        "legal_acceptance_claimed": False,
        "admitted": False,
        "formalized": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }
    body.update(overrides)
    return body


def test_label_mask_probe_matches_distinct_tiers() -> None:
    source = Path(
        "/home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/"
        "logic/autoformal/legal_label_mask_audit_v1.py"
    ).read_text(encoding="utf-8")
    assert "def audit_label_masks" in source
    assert "label tiers share a mask" in source
    assert "solver or resource failure is a negative legal label" in source
    assert "parser or Lean success created gold" in source
    assert "parser success created a legal label" in source
    evaluation = _label_owner_paths()[
        "legal_relational_batch_evaluation_v1.py"
    ].read_text(encoding="utf-8")
    example = _label_owner_paths()[
        "legal_relational_decoder_example_v1.py"
    ].read_text(encoding="utf-8")
    for marker in (
        "Available references are replayed authored unreviewed declarations, not reviewed statutory gold; unavailable references remain in the denominator.",
        "No models, training, native Lake/backend execution, legal review, admission or publication occurs; existing owners may regenerate syntax-only Lean text.",
        "Incomplete, invalid frames, invalid content and exhausted local token budgets remain distinct; invalid UTF8 is not representable by this fixed lexical vocabulary and is not inferred from missing output.",
        "Explicit error, timeout, cancellation and not_started are caller declarations. Actual context, owner, source or full-bound refusal and interruption abort without a partial report.",
    ):
        assert marker in evaluation
    for marker in (
        "Solver or resource failure is not a negative legal label.",
        "Reviewed, compiler-weak, synthetic and unresolved tiers keep distinct masks and origins.",
        "Parser or Lean success creates no gold.",
    ):
        assert marker in example
    assert "def prepare_relational_decoder_example_v1" in example.split(
        "LABEL_MASK_BOUNDARY", 1
    )[0]
    assert "LABEL_TIER_MASKS" in example.split("LABEL_MASK_BOUNDARY", 1)[0]
    measured = measure_label_mask_probe()
    assert measured == EXPECTED_LABEL_MASK_PROBE
    assert measured["authored_fixture_only"] is True
    assert measured["tier_count"] == 4
    assert measured["masks_distinct"] is True
    assert measured["origins_distinct"] is True
    assert measured["reviewed_mask"] == 1
    assert measured["compiler_weak_mask"] == 2
    assert measured["synthetic_mask"] == 4
    assert measured["unresolved_mask"] == 8
    assert measured["reviewed_origin"] == "declared-reviewed-tier"
    assert measured["compiler_weak_origin"] == "declared-compiler-weak-tier"
    assert measured["synthetic_origin"] == "declared-synthetic-tier"
    assert measured["unresolved_origin"] == "declared-unresolved-tier"
    assert measured["reviewed_tier_review_completed"] is False
    assert measured["resource_failure_is_negative_legal_label"] is False
    assert measured["resource_failure_statuses"] == [
        "budget_exhausted",
        "cancellation",
        "error",
        "timeout",
    ]
    assert measured["parser_success_status"] == "valid_without_reference"
    assert measured["gold_created_by_parser_or_lean_success"] is False
    assert measured["training_mask"] == 0
    assert measured["evaluation_gold_mask"] == 0
    assert measured["solver_executed"] is False
    assert measured["lean_executed"] is False
    assert measured["resource_failure_executed"] is False
    assert measured["decoder_executed"] is False
    assert measured["training_executed"] is False
    assert measured["federal_corpus_scored"] is False
    assert measured["legal_meaning_established"] is False
    assert measured["legal_acceptance_claimed"] is False
    assert measured["admitted"] is False
    assert measured["formalized"] is False
    assert measured["Lean_admitted"] is False


def test_label_mask_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _LABEL_MASK_ACCEPTANCE]
    owners = _label_owners()
    dependency = (
        {
            "task_cid": "task:metrics",
            "digest": "sha256:" + ("95" * 32),
            "body": _training_metric_clean_dependency(),
        },
    )
    evidence = label_mask_evidence(
        criteria, dependency, owners, EXPECTED_LABEL_MASK_PROBE
    )
    assert evidence is not None
    assert evidence["operation"] == "dependency_block_resolved_label_mask_boundaries"
    assert evidence["authored_fixture_only"] is True
    assert evidence["masks_distinct"] is True
    assert evidence["origins_distinct"] is True
    assert evidence["tier_count"] == 4
    assert evidence["reviewed_tier_review_completed"] is False
    assert evidence["resource_failure_is_negative_legal_label"] is False
    assert evidence["parser_success_status"] == "valid_without_reference"
    assert evidence["gold_created_by_parser_or_lean_success"] is False
    assert evidence["training_mask"] == 0
    assert evidence["evaluation_gold_mask"] == 0
    assert evidence["solver_executed"] is False
    assert evidence["lean_executed"] is False
    assert evidence["training_executed"] is False
    assert evidence["federal_corpus_scored"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["completion_authority"] is False
    trained = _training_metric_clean_dependency(training_executed=True)
    assert label_mask_evidence(
        criteria,
        (
            {
                "task_cid": "task:metrics",
                "digest": "sha256:" + ("95" * 32),
                "body": trained,
            },
        ),
        owners,
        EXPECTED_LABEL_MASK_PROBE,
    ) is None
    assert label_mask_evidence(
        criteria,
        (
            {
                "task_cid": "task:context",
                "digest": "sha256:" + ("94" * 32),
                "body": _context_budget_clean_dependency(),
            },
        ),
        owners,
        EXPECTED_LABEL_MASK_PROBE,
    ) is None
    measured = dict(EXPECTED_LABEL_MASK_PROBE)
    measured["gold_created_by_parser_or_lean_success"] = True
    assert label_mask_evidence(criteria, dependency, owners, measured) is None
    admitted = dict(EXPECTED_LABEL_MASK_PROBE)
    admitted["admitted"] = True
    assert label_mask_evidence(criteria, dependency, owners, admitted) is None
    dropped = _label_owners()
    dropped["evaluation"] = dict(dropped["evaluation"])
    dropped["evaluation"]["text"] = "candidate evaluation only\n"
    assert label_mask_evidence(
        criteria, dependency, dropped, EXPECTED_LABEL_MASK_PROBE
    ) is None
    for other in (
        _OCCURRENCE_STILL_ACCEPTANCE,
        _CONTEXT_BUDGET_ACCEPTANCE,
        _PRODUCER_STILL_ACCEPTANCE,
        _TRAINING_NEXT_ACCEPTANCE,
        _COVERAGE_ACCEPTANCE,
        _WIDTH_CONTROL_ACCEPTANCE,
    ):
        assert label_mask_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_LABEL_MASK_PROBE,
        ) is None


def test_release_records_label_mask_audit(tmp_path: Path) -> None:
    relative = []
    for name, source in _label_owner_paths().items():
        relative_path = "ipfs_datasets_py/" + str(source).split("/ipfs_datasets_py/", 1)[1]
        path = tmp_path / relative_path
        relative.append(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:metrics",
        "LIR-METRICS-DEP",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:metrics",
        evidence_kind="validation",
        digest="sha256:" + ("95" * 32),
        body=_training_metric_clean_dependency(),
    )
    _task(
        repo,
        "task:trained",
        "LIR-TRAINED-METRICS",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:trained",
        evidence_kind="validation",
        digest="sha256:" + ("93" * 32),
        body=_training_metric_clean_dependency(training_executed=True),
    )
    _task(
        repo,
        "task:labels",
        "LIR-LABELS",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:metrics"],
        acceptance=_LABEL_MASK_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare-labels",
        "LIR-BARE-LABELS",
        status="blocked",
        body=_manual_body("", stage="analysis"),
        dependencies=["task:metrics"],
        acceptance=_LABEL_MASK_ACCEPTANCE,
    )
    _task(
        repo,
        "task:trained-labels",
        "LIR-TRAINED-LABELS",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:trained"],
        acceptance=_LABEL_MASK_ACCEPTANCE,
    )
    _task(
        repo,
        "task:occurrence-still",
        "LIR-OCCURRENCE-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="inference"),
            "existing_owners": relative,
        },
        dependencies=["task:metrics"],
        acceptance=_OCCURRENCE_STILL_ACCEPTANCE,
    )
    _task(
        repo,
        "task:coverage-still",
        "LIR-COVERAGE-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="evaluation"),
            "existing_owners": relative,
        },
        dependencies=["task:metrics"],
        acceptance=_COVERAGE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:context-still",
        "LIR-CONTEXT-STILL",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:metrics"],
        acceptance=_CONTEXT_BUDGET_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-LABELS"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-LABELS",
        "LIR-TRAINED-LABELS",
        "LIR-OCCURRENCE-STILL",
        "LIR-COVERAGE-STILL",
        "LIR-CONTEXT-STILL",
    }
    assert result["held"] == []
    task = repo.get_task("task:labels")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == "dependency_block_resolved_label_mask_boundaries"
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    for cid in (
        "task:bare-labels",
        "task:trained-labels",
        "task:occurrence-still",
        "task:coverage-still",
        "task:context-still",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


_TEACHER_IDENTITY_ACCEPTANCE = (
    {
        "criterion": (
            "Changed targets invalidate split/experiment/teacher identities "
            "as required."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Only an independently qualified teacher supplies semantic "
            "distillation scope; engineering retention is separately labeled."
        ),
        "evidence_kind": "validation",
    },
)


def _teacher_owner_paths() -> dict[str, Path]:
    root = Path("/home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py")
    return {
        "supervisor_queue.py": root / "logic/autoformal/supervisor_queue.py",
        "legal_relational_decoder_example_v1.py": root
        / "logic/formalization/autoencoder/legal_relational_decoder_example_v1.py",
    }


def _teacher_owners() -> dict[str, dict[str, str]]:
    import hashlib

    roles = {
        "supervisor_queue.py": "queue",
        "legal_relational_decoder_example_v1.py": "example",
    }
    owners = {}
    for name, path in _teacher_owner_paths().items():
        raw = path.read_bytes()
        owners[roles[name]] = {
            "path": "ipfs_datasets_py/" + str(path).split("/ipfs_datasets_py/", 1)[1],
            "text": raw.decode("utf-8"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return owners


def _label_mask_clean_dependency(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "operation": "dependency_block_resolved_label_mask_boundaries",
        "training_executed": False,
        "model_executed": False,
        "solver_executed": False,
        "lean_executed": False,
        "resource_failure_executed": False,
        "decoder_executed": False,
        "worker_launched": False,
        "federal_corpus_scored": False,
        "legal_meaning_established": False,
        "legal_acceptance_claimed": False,
        "admitted": False,
        "formalized": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
        "gold_created_by_parser_or_lean_success": False,
        "reviewed_tier_review_completed": False,
        "authored_fixture_only": True,
        "training_mask": 0,
        "evaluation_gold_mask": 0,
        "evaluation_gold_qualified": False,
    }
    body.update(overrides)
    return body


def test_teacher_identity_probe_invalidates_changed_targets() -> None:
    source = Path(
        "/home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/"
        "logic/autoformal/legal_teacher_identity_audit_v1.py"
    ).read_text(encoding="utf-8")
    queue = _teacher_owner_paths()["supervisor_queue.py"].read_text(encoding="utf-8")
    example = _teacher_owner_paths()[
        "legal_relational_decoder_example_v1.py"
    ].read_text(encoding="utf-8")
    for marker in (
        "Changed targets invalidate split, experiment and teacher identities.",
        "Only an independently qualified teacher supplies semantic distillation scope.",
        "Engineering retention stays a separate label.",
        "No independently qualified teacher was supplied.",
    ):
        assert marker in source
        assert marker in queue
        assert marker in example
    assert "def repair_issue_identity" in queue.split("def repair_target_identities", 1)[0]
    assert "def prepare_relational_decoder_example_v1" in example.split(
        "TEACHER_IDENTITY_BOUNDARY", 1
    )[0]
    assert "LABEL_MASK_BOUNDARY" in example.split("TEACHER_IDENTITY_BOUNDARY", 1)[0]
    measured = measure_teacher_identity_probe()
    assert measured == EXPECTED_TEACHER_IDENTITY_PROBE
    assert measured["authored_fixture_only"] is True
    assert measured["target_count"] == 2
    assert measured["split_identity_changed"] is True
    assert measured["experiment_identity_changed"] is True
    assert measured["teacher_identity_changed"] is True
    assert measured["unchanged_target_identities_stable"] is True
    assert measured["semantic_distillation_scope_supplied"] is False
    assert measured["independently_qualified_teacher_supplied"] is False
    assert measured["engineering_retention"] == "engineering-retention"
    assert measured["engineering_retention_separately_labeled"] is True
    assert measured["retention_identity_unchanged"] is True
    assert measured["qualified_teacher_refused"] is True
    assert measured["preserved_identity_refused"] is True
    assert measured["retention_scope_refused"] is True
    assert measured["teacher_promoted"] is False
    assert measured["training_executed"] is False
    assert measured["decoder_executed"] is False
    assert measured["federal_corpus_scored"] is False
    assert measured["legal_meaning_established"] is False
    assert measured["legal_acceptance_claimed"] is False
    assert measured["admitted"] is False
    assert measured["formalized"] is False
    assert measured["split_identity"] != measured["changed_split_identity"]
    assert measured["experiment_identity"] != measured["changed_experiment_identity"]
    assert measured["teacher_identity"] != measured["changed_teacher_identity"]


def test_teacher_identity_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _TEACHER_IDENTITY_ACCEPTANCE]
    owners = _teacher_owners()
    dependency = (
        {
            "task_cid": "task:labels",
            "digest": "sha256:" + ("96" * 32),
            "body": _label_mask_clean_dependency(),
        },
    )
    evidence = teacher_identity_evidence(
        criteria, dependency, owners, EXPECTED_TEACHER_IDENTITY_PROBE
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_teacher_identity_boundaries"
    )
    assert evidence["authored_fixture_only"] is True
    assert evidence["independently_qualified_teacher_supplied"] is False
    assert evidence["semantic_distillation_scope_supplied"] is False
    assert evidence["engineering_retention_separately_labeled"] is True
    assert evidence["teacher_promoted"] is False
    assert evidence["training_executed"] is False
    assert evidence["federal_corpus_scored"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["completion_authority"] is False
    gold = _label_mask_clean_dependency(gold_created_by_parser_or_lean_success=True)
    assert teacher_identity_evidence(
        criteria,
        (
            {
                "task_cid": "task:labels",
                "digest": "sha256:" + ("96" * 32),
                "body": gold,
            },
        ),
        owners,
        EXPECTED_TEACHER_IDENTITY_PROBE,
    ) is None
    assert teacher_identity_evidence(
        criteria,
        (
            {
                "task_cid": "task:metrics",
                "digest": "sha256:" + ("95" * 32),
                "body": _training_metric_clean_dependency(),
            },
        ),
        owners,
        EXPECTED_TEACHER_IDENTITY_PROBE,
    ) is None
    qualified = dict(EXPECTED_TEACHER_IDENTITY_PROBE)
    qualified["independently_qualified_teacher_supplied"] = True
    assert teacher_identity_evidence(criteria, dependency, owners, qualified) is None
    admitted = dict(EXPECTED_TEACHER_IDENTITY_PROBE)
    admitted["admitted"] = True
    assert teacher_identity_evidence(criteria, dependency, owners, admitted) is None
    dropped = _teacher_owners()
    dropped["queue"] = dict(dropped["queue"])
    dropped["queue"]["text"] = "candidate queue only\n"
    assert teacher_identity_evidence(
        criteria, dependency, dropped, EXPECTED_TEACHER_IDENTITY_PROBE
    ) is None
    for other in (
        _LABEL_MASK_ACCEPTANCE,
        _OCCURRENCE_STILL_ACCEPTANCE,
        _CONTEXT_BUDGET_ACCEPTANCE,
        _PRODUCER_STILL_ACCEPTANCE,
        _TRAINING_NEXT_ACCEPTANCE,
        _COVERAGE_ACCEPTANCE,
    ):
        assert teacher_identity_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_TEACHER_IDENTITY_PROBE,
        ) is None


def test_release_records_teacher_identity_audit(tmp_path: Path) -> None:
    relative = []
    for name, source in _teacher_owner_paths().items():
        relative_path = "ipfs_datasets_py/" + str(source).split("/ipfs_datasets_py/", 1)[1]
        path = tmp_path / relative_path
        relative.append(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:labels",
        "LIR-LABELS-DEP",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:labels",
        evidence_kind="validation",
        digest="sha256:" + ("96" * 32),
        body=_label_mask_clean_dependency(),
    )
    _task(
        repo,
        "task:gold",
        "LIR-GOLD-LABELS",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:gold",
        evidence_kind="validation",
        digest="sha256:" + ("97" * 32),
        body=_label_mask_clean_dependency(gold_created_by_parser_or_lean_success=True),
    )
    _task(
        repo,
        "task:teacher",
        "LIR-TEACHER",
        status="blocked",
        body={
            **_manual_body("", stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:labels"],
        acceptance=_TEACHER_IDENTITY_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare-teacher",
        "LIR-BARE-TEACHER",
        status="blocked",
        body=_manual_body("", stage="validation"),
        dependencies=["task:labels"],
        acceptance=_TEACHER_IDENTITY_ACCEPTANCE,
    )
    _task(
        repo,
        "task:gold-teacher",
        "LIR-GOLD-TEACHER",
        status="blocked",
        body={
            **_manual_body("", stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:gold"],
        acceptance=_TEACHER_IDENTITY_ACCEPTANCE,
    )
    _task(
        repo,
        "task:labels-still",
        "LIR-LABELS-STILL",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:labels"],
        acceptance=_LABEL_MASK_ACCEPTANCE,
    )
    _task(
        repo,
        "task:coverage-still",
        "LIR-COVERAGE-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="evaluation"),
            "existing_owners": relative,
        },
        dependencies=["task:labels"],
        acceptance=_COVERAGE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:producer-still",
        "LIR-PRODUCER-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="inference"),
            "existing_owners": relative,
        },
        dependencies=["task:labels"],
        acceptance=_PRODUCER_STILL_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-TEACHER"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-TEACHER",
        "LIR-GOLD-TEACHER",
        "LIR-LABELS-STILL",
        "LIR-COVERAGE-STILL",
        "LIR-PRODUCER-STILL",
    }
    assert result["held"] == []
    task = repo.get_task("task:teacher")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_teacher_identity_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    for cid in (
        "task:bare-teacher",
        "task:gold-teacher",
        "task:labels-still",
        "task:coverage-still",
        "task:producer-still",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


def _coverage_owner_paths() -> dict[str, Path]:
    root = Path("/home/barberb/lift_coding/external/ipfs_datasets")
    return {
        "legal_relational_batch_evaluation_v1.py": root
        / "ipfs_datasets_py/logic/autoformal/legal_relational_batch_evaluation_v1.py",
        "legal_ir_span_graph_lean_execution_plan_20261016.md": root
        / "docs/autoencoders/legal_ir_span_graph_lean_execution_plan_20261016.md",
    }


def _coverage_owners() -> dict[str, dict[str, str]]:
    import hashlib

    roles = {
        "legal_relational_batch_evaluation_v1.py": "evaluation",
        "legal_ir_span_graph_lean_execution_plan_20261016.md": "plan",
    }
    owners = {}
    for name, path in _coverage_owner_paths().items():
        raw = path.read_bytes()
        owners[roles[name]] = {
            "path": str(path).split("/external/ipfs_datasets/", 1)[1],
            "text": raw.decode("utf-8"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return owners


def _coverage_dependencies() -> tuple[dict[str, object], ...]:
    return (
        {
            "task_cid": "task:consequence",
            "digest": "sha256:" + ("91" * 32),
            "body": {
                "operation": "dependency_block_resolved_nonvacuous_consequence_boundaries",
                "legal_acceptance_claimed": False,
                "federal_corpus_scored": False,
                "admitted": False,
                "formalized": False,
                "gold_or_teacher_admission_granted": False,
                "constitution_roundtrip_ok": False,
            },
        },
        {
            "task_cid": "task:handoff",
            "digest": "sha256:" + ("92" * 32),
            "body": {
                "schema": "legal-ir-native9004-independent-factual-handoff-evidence/v1",
                "formalized": False,
                "admitted": False,
                "gold_or_teacher_admission_granted": False,
                "constitution_roundtrip_ok": False,
            },
        },
    )


def test_population_coverage_probe_keeps_counts_separate() -> None:
    source = Path(
        "/home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/"
        "logic/autoformal/legal_population_coverage_audit_v1.py"
    ).read_text(encoding="utf-8")
    plan = _coverage_owner_paths()[
        "legal_ir_span_graph_lean_execution_plan_20261016.md"
    ].read_text(encoding="utf-8")
    evaluation = _coverage_owner_paths()[
        "legal_relational_batch_evaluation_v1.py"
    ].read_text(encoding="utf-8")
    for marker in (
        "Inventory, eligible, attempted, missing, invalid, unsupported and accepted counts stay separate.",
        "Compared is not an accepted population.",
        "Attempted rows include a review record.",
        "No throughput extrapolation from the small replay or the row-only total.",
    ):
        assert marker in source
        assert marker in plan
    assert "def prepare_relational_batch_evaluation_v1" in evaluation
    assert "accepted" not in evaluation.split("TERMINAL_STATUSES = (", 1)[1].split(")", 1)[0]
    measured = measure_population_coverage_probe()
    assert measured == EXPECTED_POPULATION_COVERAGE_PROBE
    assert measured["authored_fixture_only"] is True
    assert measured["inventory_count"] == 9
    assert measured["eligible_count"] == 8
    assert measured["attempted_count"] == 5
    assert measured["missing_count"] == 3
    assert measured["invalid_count"] == 1
    assert measured["unsupported_count"] == 2
    assert measured["accepted_count"] == 0
    assert measured["compared_count"] == 1
    assert measured["compared_counted_as_accepted"] is False
    assert measured["review_cost_included"] is True
    assert measured["review_cost_record_count"] == 5
    assert measured["pilot_review_cost_measured"] is False
    assert measured["throughput_extrapolated"] is False
    assert measured["federal_corpus_scored"] is False
    assert measured["legal_acceptance_claimed"] is False
    assert measured["accepted_relabel_refused"] is True
    assert measured["extrapolation_refused"] is True
    assert measured["omitted_review_refused"] is True


def test_population_coverage_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _COVERAGE_ACCEPTANCE]
    owners = _coverage_owners()
    dependency = _coverage_dependencies()
    evidence = population_coverage_evidence(
        criteria, dependency, owners, EXPECTED_POPULATION_COVERAGE_PROBE
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_population_coverage_boundaries"
    )
    assert evidence["accepted_count"] == 0
    assert evidence["compared_counted_as_accepted"] is False
    assert evidence["review_cost_included"] is True
    assert evidence["pilot_review_cost_measured"] is False
    assert evidence["throughput_extrapolated"] is False
    assert evidence["federal_corpus_scored"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["completion_authority"] is False
    assert population_coverage_evidence(
        criteria,
        dependency[:1],
        owners,
        EXPECTED_POPULATION_COVERAGE_PROBE,
    ) is None
    accepted = dict(EXPECTED_POPULATION_COVERAGE_PROBE)
    accepted["accepted_count"] = 1
    assert population_coverage_evidence(criteria, dependency, owners, accepted) is None
    admitted = dict(EXPECTED_POPULATION_COVERAGE_PROBE)
    admitted["admitted"] = True
    assert population_coverage_evidence(criteria, dependency, owners, admitted) is None
    dropped = _coverage_owners()
    dropped["plan"] = dict(dropped["plan"])
    dropped["plan"]["text"] = "historical plan only\n"
    assert population_coverage_evidence(
        criteria, dependency, dropped, EXPECTED_POPULATION_COVERAGE_PROBE
    ) is None
    for other in (
        _TEACHER_IDENTITY_ACCEPTANCE,
        _LABEL_MASK_ACCEPTANCE,
        _OCCURRENCE_STILL_ACCEPTANCE,
        _PRODUCER_STILL_ACCEPTANCE,
    ):
        assert population_coverage_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_POPULATION_COVERAGE_PROBE,
        ) is None


def test_release_records_population_coverage_audit(tmp_path: Path) -> None:
    relative = []
    for name, source in _coverage_owner_paths().items():
        relative_path = str(source).split("/external/ipfs_datasets/", 1)[1]
        path = tmp_path / relative_path
        relative.append(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:consequence",
        "LIR-CONSEQUENCE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:consequence",
        evidence_kind="validation",
        digest="sha256:" + ("91" * 32),
        body={
            "operation": "dependency_block_resolved_nonvacuous_consequence_boundaries",
            "legal_acceptance_claimed": False,
            "federal_corpus_scored": False,
            "admitted": False,
            "formalized": False,
            "gold_or_teacher_admission_granted": False,
            "constitution_roundtrip_ok": False,
        },
    )
    _task(
        repo,
        "task:handoff",
        "LIR-HANDOFF",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:handoff",
        evidence_kind="validation",
        digest="sha256:" + ("92" * 32),
        body={
            "schema": "legal-ir-native9004-independent-factual-handoff-evidence/v1",
            "formalized": False,
            "admitted": False,
            "gold_or_teacher_admission_granted": False,
            "constitution_roundtrip_ok": False,
        },
    )
    _task(
        repo,
        "task:coverage",
        "LIR-COVERAGE",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="evaluation"),
            "existing_owners": relative,
        },
        dependencies=["task:consequence", "task:handoff"],
        acceptance=_COVERAGE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare-coverage",
        "LIR-BARE-COVERAGE",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="evaluation"),
        dependencies=["task:consequence", "task:handoff"],
        acceptance=_COVERAGE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:partial-coverage",
        "LIR-PARTIAL-COVERAGE",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="evaluation"),
            "existing_owners": relative,
        },
        dependencies=["task:consequence"],
        acceptance=_COVERAGE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:teacher-still",
        "LIR-TEACHER-STILL",
        status="blocked",
        body={
            **_manual_body("", stage="validation"),
            "existing_owners": relative,
        },
        dependencies=["task:consequence"],
        acceptance=_TEACHER_IDENTITY_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-COVERAGE"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-COVERAGE",
        "LIR-PARTIAL-COVERAGE",
        "LIR-TEACHER-STILL",
    }
    assert result["held"] == []
    task = repo.get_task("task:coverage")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_population_coverage_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    for cid in (
        "task:bare-coverage",
        "task:partial-coverage",
        "task:teacher-still",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


_STORAGE_ACCEPTANCE = (
    {
        "criterion": (
            "Budget expanded lineage, temporary replay, retained evidence "
            "and publication overlap."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Native grants remain purpose-specific; foreign claims/assets "
            "are preserved and ledger/cap values are not bypassed."
        ),
        "evidence_kind": "validation",
    },
)
_STORAGE_BOUNDARY = (
    "Budget-expanded lineage, temporary replay, retained evidence and publication overlap stay separate.",
    "A native grant stays purpose-specific to observation.",
    "Foreign claims and assets stay preserved.",
    "Ledger and cap values are not bypassed.",
)


def _storage_owner_paths() -> dict[str, Path]:
    root = Path("/home/barberb/lift_coding/external/ipfs_datasets")
    return {
        "autoencoder_owned_runtime.py": root
        / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_owned_runtime.py",
        "span_cache_feed.py": root
        / "ipfs_datasets_py/logic/autoformal/span_cache_feed.py",
    }


def _storage_owners() -> dict[str, dict[str, str]]:
    import hashlib

    roles = {
        "autoencoder_owned_runtime.py": "runtime",
        "span_cache_feed.py": "feed",
    }
    owners = {}
    for name, path in _storage_owner_paths().items():
        raw = path.read_bytes()
        owners[roles[name]] = {
            "path": str(path).split("/external/ipfs_datasets/", 1)[1],
            "text": raw.decode("utf-8"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return owners


def _storage_dependencies(*, accepted_count: int = 0, review_cost_included: bool = True):
    return (
        {
            "task_cid": "task:coverage",
            "digest": "sha256:" + ("93" * 32),
            "body": {
                "operation": "dependency_block_resolved_population_coverage_boundaries",
                "authored_fixture_only": True,
                "accepted_count": accepted_count,
                "compared_counted_as_accepted": False,
                "throughput_extrapolated": False,
                "pilot_review_cost_measured": False,
                "federal_corpus_scored": False,
                "legal_acceptance_claimed": False,
                "review_cost_included": review_cost_included,
                "admitted": False,
                "formalized": False,
                "gold_or_teacher_admission_granted": False,
                "constitution_roundtrip_ok": False,
            },
        },
    )


def test_storage_concurrency_probe_keeps_roles_and_caps() -> None:
    source = Path(
        "/home/barberb/lift_coding/external/ipfs_datasets/ipfs_datasets_py/"
        "logic/autoformal/legal_storage_concurrency_audit_v1.py"
    ).read_text(encoding="utf-8")
    runtime = _storage_owner_paths()["autoencoder_owned_runtime.py"].read_text(
        encoding="utf-8"
    )
    feed = _storage_owner_paths()["span_cache_feed.py"].read_text(encoding="utf-8")
    runtime_head = runtime.split("STORAGE_CONCURRENCY_BOUNDARY", 1)[0]
    feed_head = feed.split("STORAGE_CONCURRENCY_BOUNDARY", 1)[0]
    for marker in _STORAGE_BOUNDARY:
        assert marker in source
        assert marker in runtime
        assert marker in feed
    for marker in (
        "foreign runtime directory UID",
        "runtime byte limit exceeded",
        "ledger_changed=False",
        "resource_admission=False",
        "class OwnedRuntimeDirectory",
        "def observe",
    ):
        assert marker in runtime_head
    for marker in (
        "foreign or incomplete feed database",
        "not a version3 full-report/reference replay or a new source claim.",
        "poll download budget exhausted",
        "class SpanCacheFeed",
    ):
        assert marker in feed_head
    measured = measure_storage_concurrency_probe()
    assert measured == EXPECTED_STORAGE_CONCURRENCY_PROBE
    assert measured["authored_fixture_only"] is True
    assert measured["role_count"] == 4
    assert measured["roles_distinct"] is True
    assert measured["budget_expanded_lineage"] == "budget-expanded-lineage"
    assert measured["temporary_replay"] == "temporary-replay"
    assert measured["retained_evidence"] == "retained-evidence"
    assert measured["publication_overlap"] == "publication-overlap"
    assert measured["publication_overlap_replaced_retained"] is False
    assert measured["native_grant_purpose"] == "observation-only"
    assert measured["native_grant_purpose_specific"] is True
    assert measured["resource_admission"] is False
    assert measured["ledger_changed"] is False
    assert measured["files_deleted"] is False
    assert measured["foreign_asset_disposition"] == "preserved"
    assert measured["foreign_assets_preserved"] is True
    assert measured["external_file_cap"] == 134217728
    assert measured["ledger_cap"] == 16777216
    assert measured["cap_bypassed"] is False
    assert measured["ledger_bypassed"] is False
    assert measured["overlap_refused"] is True
    assert measured["grant_refused"] is True
    assert measured["foreign_removal_refused"] is True
    assert measured["cap_bypass_refused"] is True
    assert measured["federal_corpus_scored"] is False
    assert measured["legal_acceptance_claimed"] is False
    assert measured["admitted"] is False
    assert measured["formalized"] is False


def test_storage_concurrency_evidence_refuses_other_tasks_and_changed_probes() -> None:
    criteria = [item["criterion"] for item in _STORAGE_ACCEPTANCE]
    owners = _storage_owners()
    dependency = _storage_dependencies()
    evidence = storage_concurrency_evidence(
        criteria, dependency, owners, EXPECTED_STORAGE_CONCURRENCY_PROBE
    )
    assert evidence is not None
    assert evidence["operation"] == (
        "dependency_block_resolved_storage_concurrency_boundaries"
    )
    assert evidence["external_file_cap"] == 134217728
    assert evidence["ledger_cap"] == 16777216
    assert evidence["cap_bypassed"] is False
    assert evidence["ledger_bypassed"] is False
    assert evidence["native_grant_purpose"] == "observation-only"
    assert evidence["foreign_asset_disposition"] == "preserved"
    assert evidence["resource_admission"] is False
    assert evidence["ledger_changed"] is False
    assert evidence["files_deleted"] is False
    assert evidence["federal_corpus_scored"] is False
    assert evidence["legal_acceptance_claimed"] is False
    assert evidence["admitted"] is False
    assert evidence["formalized"] is False
    assert evidence["gold_or_teacher_admission_granted"] is False
    assert evidence["constitution_roundtrip_ok"] is False
    assert evidence["completion_authority"] is False
    assert storage_concurrency_evidence(
        criteria,
        _storage_dependencies(accepted_count=1),
        owners,
        EXPECTED_STORAGE_CONCURRENCY_PROBE,
    ) is None
    assert storage_concurrency_evidence(
        criteria,
        _storage_dependencies(review_cost_included=False),
        owners,
        EXPECTED_STORAGE_CONCURRENCY_PROBE,
    ) is None
    raised_cap = dict(EXPECTED_STORAGE_CONCURRENCY_PROBE)
    raised_cap["external_file_cap"] = 134217729
    assert storage_concurrency_evidence(criteria, dependency, owners, raised_cap) is None
    admitted = dict(EXPECTED_STORAGE_CONCURRENCY_PROBE)
    admitted["admitted"] = True
    assert storage_concurrency_evidence(criteria, dependency, owners, admitted) is None
    dropped = _storage_owners()
    dropped["runtime"] = dict(dropped["runtime"])
    dropped["runtime"]["text"] = dropped["runtime"]["text"].replace(
        "Ledger and cap values are not bypassed.",
        "ledger cap omitted",
    )
    assert storage_concurrency_evidence(
        criteria, dependency, dropped, EXPECTED_STORAGE_CONCURRENCY_PROBE
    ) is None
    for other in (
        _COVERAGE_ACCEPTANCE,
        _TEACHER_IDENTITY_ACCEPTANCE,
        _LABEL_MASK_ACCEPTANCE,
        _PRODUCER_STILL_ACCEPTANCE,
    ):
        assert storage_concurrency_evidence(
            [item["criterion"] for item in other],
            dependency,
            owners,
            EXPECTED_STORAGE_CONCURRENCY_PROBE,
        ) is None


def test_release_records_storage_concurrency_audit(tmp_path: Path) -> None:
    relative = []
    for name, source in _storage_owner_paths().items():
        relative_path = str(source).split("/external/ipfs_datasets/", 1)[1]
        path = tmp_path / relative_path
        relative.append(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:coverage",
        "LIR-COVERAGE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:coverage",
        evidence_kind="validation",
        digest="sha256:" + ("93" * 32),
        body=_storage_dependencies()[0]["body"],
    )
    _task(
        repo,
        "task:partial-source",
        "LIR-PARTIAL-SOURCE",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:partial-source",
        evidence_kind="validation",
        digest="sha256:" + ("94" * 32),
        body=_storage_dependencies(accepted_count=1)[0]["body"],
    )
    _task(
        repo,
        "task:storage",
        "LIR-STORAGE",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:coverage"],
        acceptance=_STORAGE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare-storage",
        "LIR-BARE-STORAGE",
        status="blocked",
        body=_manual_body("", stage="analysis"),
        dependencies=["task:coverage"],
        acceptance=_STORAGE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:partial-storage",
        "LIR-PARTIAL-STORAGE",
        status="blocked",
        body={
            **_manual_body("", stage="analysis"),
            "existing_owners": relative,
        },
        dependencies=["task:partial-source"],
        acceptance=_STORAGE_ACCEPTANCE,
    )
    _task(
        repo,
        "task:coverage-still",
        "LIR-COVERAGE-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="evaluation"),
            "existing_owners": relative,
        },
        dependencies=["task:coverage"],
        acceptance=_COVERAGE_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    assert {item["task_alias"] for item in result["completed"]} == {"LIR-STORAGE"}
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-STORAGE",
        "LIR-PARTIAL-STORAGE",
        "LIR-COVERAGE-STILL",
    }
    assert result["held"] == []
    task = repo.get_task("task:storage")
    assert task["status"] == "completed"
    receipt = task["body"]["completion_receipt"]
    assert receipt["operation"] == (
        "dependency_block_resolved_storage_concurrency_boundaries"
    )
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    for cid in (
        "task:bare-storage",
        "task:partial-storage",
        "task:coverage-still",
    ):
        assert repo.get_task(cid)["status"] == "blocked"


_NATIVE384_ACCEPTANCE = (
    {
        "criterion": (
            "Input/encoder/checkpoint/codec/output settings match the design."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Disabled placeholder models are separate source-only controls."
        ),
        "evidence_kind": "validation",
    },
)
_NATIVE768_ACCEPTANCE = (
    {
        "criterion": (
            "Full historical runtime closure and actual native encoder/input "
            "identity are authenticated."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "No zero-padding or coordinate relabeling manufactures a native "
            "768D input."
        ),
        "evidence_kind": "validation",
    },
)
_EVAL_STILL_ACCEPTANCE = (
    {
        "criterion": (
            "Selected N and comparable K denominators, component effects, "
            "seed uncertainty and named losses remain separate."
        ),
        "evidence_kind": "validation",
    },
    {
        "criterion": (
            "Useful source-conditioned performance can qualify its scope "
            "without proving a latent bottleneck benefit."
        ),
        "evidence_kind": "validation",
    },
)


def _comparison_owner_paths() -> dict[str, Path]:
    paths = _width_control_owner_paths()
    root = Path(
        "/home/barberb/lift_coding/external/ipfs_datasets/docs/autoencoders"
    )
    paths["legal_ir_decoder_output_integration_20261016.md"] = (
        root / "legal_ir_decoder_output_integration_20261016.md"
    )
    return paths


def _comparison_owners(roles: dict[str, str]) -> dict[str, dict[str, str]]:
    import hashlib

    owners = {}
    for name, role in roles.items():
        raw = _comparison_owner_paths()[name].read_bytes()
        owners[role] = {
            "path": "docs/autoencoders/" + name,
            "text": raw.decode("utf-8"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return owners


def _width_control_clean_dependency(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "operation": "dependency_block_resolved_width_control_boundaries",
        "producer_called": False,
        "encoder_called": False,
        "mean_coordinate_mse_measured": False,
        "federal_corpus_scored": False,
        "legal_acceptance_claimed": False,
        "controls_executed": False,
        "model_executed": False,
        "training_executed": False,
        "worker_launched": False,
        "encoder_unfrozen": False,
        "historical_8d_distinct": True,
        "active_native_distinct": True,
        "detached_disabled_latent_distinct": True,
        "width_classes_distinct": True,
        "admitted": False,
        "formalized": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
    }
    body.update(overrides)
    return body


def _assert_comparison_receipt(operation: str) -> None:
    receipt = dependency_resolution_receipt(operation)
    assert receipt["operation"] == operation
    assert receipt["admitted"] is False
    assert receipt["formalized"] is False
    assert receipt["gold_or_teacher_admission_granted"] is False
    assert receipt["constitution_roundtrip_ok"] is False
    assert receipt["completion_authority"] is False
    assert "producer_called" not in receipt
    assert "settings_matched" not in receipt
    assert "encoder_identity_authenticated" not in receipt


def test_sparse8_comparison_probe_matches_canonical_projection() -> None:
    reconciliation = _comparison_owner_paths()[
        "legal_ir_asset_reconciliation_20261016.md"
    ].read_text(encoding="utf-8")
    for marker in (
        "Full historical sparse state, separate small linguistic bundle, and learned 18-entry formula sidecar",
        "These are three distinct artifacts/tasks.",
        "The formula sidecar's two synthetic examples do not qualify statutory supervision.",
    ):
        assert marker in reconciliation
    measured = measure_sparse8_comparison_probe()
    assert measured == EXPECTED_SPARSE8_PROBE
    assert measured["producer_called"] is True
    assert measured["encoder_called"] is True
    assert measured["source_access_evidenced"] is True
    assert measured["projection_equals_input"] is False
    assert measured["linguistic_bundle_substituted"] is False
    assert measured["formula_sidecar_substituted"] is False
    assert measured["mock_projection_substituted"] is False
    assert measured["model_executed"] is True
    assert measured["training_executed"] is False
    assert measured["admitted"] is False
    assert measured["formalized"] is False
    _assert_comparison_receipt(
        "dependency_block_resolved_sparse8_comparison_cell"
    )


def test_native384_comparison_probe_matches_pinned_settings() -> None:
    native_inputs = _comparison_owner_paths()[
        "gte_decoder_native_inputs.md"
    ].read_text(encoding="utf-8")
    for marker in (
        "The maintained entry is [gte_decoder_native_inputs.md]",
        "This former path is retained for existing links; original bytes and provenance are preserved",
    ):
        assert marker in native_inputs
    measured = measure_native384_comparison_probe()
    assert measured == EXPECTED_NATIVE384_PROBE
    assert measured["settings_matched"] is True
    assert measured["producer_called"] is True
    assert measured["encoder_called"] is True
    assert measured["executed_dtype"] == "float32"
    assert measured["storage_dtypes"] == ["F16", "I64"]
    assert measured["detached_placeholder_components"] == ["F15", "F17", "F18"]
    assert measured["detached_placeholder_bound_as_active_lane"] is False
    assert measured["model_revision_authenticated"] is False
    assert measured["runtime_cryptographically_attested"] is False
    assert measured["training_executed"] is False
    assert measured["admitted"] is False
    _assert_comparison_receipt(
        "dependency_block_resolved_native384_comparison_cell"
    )


def test_native768_comparison_probe_matches_complete_encoder() -> None:
    native_inputs = _comparison_owner_paths()[
        "gte_decoder_native_inputs.md"
    ].read_text(encoding="utf-8")
    integration = _comparison_owner_paths()[
        "legal_ir_decoder_output_integration_20261016.md"
    ].read_text(encoding="utf-8")
    assert (
        "This boundary does not authenticate an encoder or manufacture a native 768D input."
        in native_inputs
    )
    assert "The original guide is preserved in the" in integration
    assert (
        "they are not the current task queue or fresh qualification evidence."
        in integration
    )
    measured = measure_native768_comparison_probe()
    assert measured == EXPECTED_NATIVE768_PROBE
    assert measured["encoder_identity_authenticated"] is True
    assert measured["producer_called"] is True
    assert measured["encoder_called"] is True
    assert measured["zero_padding_manufactured"] is False
    assert measured["coordinate_relabel_manufactured"] is False
    assert measured["coordinate_zero_pad"] is False
    assert measured["max_tokens_relabelled_as_512"] is False
    assert measured["profile_max_tokens"] == 8192
    assert measured["experiment_max_tokens"] == 512
    assert measured["dimension"] == 768
    assert measured["training_executed"] is False
    assert measured["decoder_executed"] is False
    assert measured["admitted"] is False
    assert measured["legal_acceptance_claimed"] is False
    _assert_comparison_receipt(
        "dependency_block_resolved_native768_comparison_cell"
    )


def test_comparison_cell_evidence_refuses_other_tasks_and_changed_probes() -> None:
    width = (
        {
            "task_cid": "task:widths",
            "digest": "sha256:" + ("91" * 32),
            "body": _width_control_clean_dependency(),
        },
    )
    sparse_criteria = [item["criterion"] for item in _PRODUCER_STILL_ACCEPTANCE]
    native384_criteria = [item["criterion"] for item in _NATIVE384_ACCEPTANCE]
    native768_criteria = [item["criterion"] for item in _NATIVE768_ACCEPTANCE]
    sparse_owners = _comparison_owners(
        {"legal_ir_asset_reconciliation_20261016.md": "reconciliation"}
    )
    native384_owners = _comparison_owners(
        {"gte_decoder_native_inputs.md": "native_inputs"}
    )
    native768_owners = _comparison_owners(
        {
            "gte_decoder_native_inputs.md": "native_inputs",
            "legal_ir_decoder_output_integration_20261016.md": "integration",
        }
    )
    sparse = sparse8_comparison_evidence(
        sparse_criteria, width, sparse_owners, EXPECTED_SPARSE8_PROBE
    )
    assert sparse is not None
    assert sparse["operation"] == (
        "dependency_block_resolved_sparse8_comparison_cell"
    )
    assert sparse["producer_called"] is True
    assert sparse["projection_equals_input"] is False
    assert sparse["admitted"] is False
    assert sparse["completion_authority"] is False
    native384 = native384_comparison_evidence(
        native384_criteria, width, native384_owners, EXPECTED_NATIVE384_PROBE
    )
    assert native384 is not None
    assert native384["settings_matched"] is True
    assert native384["model_revision_authenticated"] is False
    assert native384["detached_placeholder_bound_as_active_lane"] is False
    assert native384["admitted"] is False
    native768 = native768_comparison_evidence(
        native768_criteria, width, native768_owners, EXPECTED_NATIVE768_PROBE
    )
    assert native768 is not None
    assert native768["encoder_identity_authenticated"] is True
    assert native768["zero_padding_manufactured"] is False
    assert native768["admitted"] is False
    called = (
        {
            "task_cid": "task:widths",
            "digest": "sha256:" + ("91" * 32),
            "body": _width_control_clean_dependency(producer_called=True),
        },
    )
    assert sparse8_comparison_evidence(
        sparse_criteria, called, sparse_owners, EXPECTED_SPARSE8_PROBE
    ) is None
    assert native384_comparison_evidence(
        native384_criteria, called, native384_owners, EXPECTED_NATIVE384_PROBE
    ) is None
    assert native768_comparison_evidence(
        native768_criteria, called, native768_owners, EXPECTED_NATIVE768_PROBE
    ) is None
    admitted_width = (
        {
            "task_cid": "task:widths",
            "digest": "sha256:" + ("91" * 32),
            "body": _width_control_clean_dependency(admitted=True),
        },
    )
    assert native384_comparison_evidence(
        native384_criteria,
        admitted_width,
        native384_owners,
        EXPECTED_NATIVE384_PROBE,
    ) is None
    substituted = dict(EXPECTED_SPARSE8_PROBE)
    substituted["mock_projection_substituted"] = True
    assert sparse8_comparison_evidence(
        sparse_criteria, width, sparse_owners, substituted
    ) is None
    unmatched = dict(EXPECTED_NATIVE384_PROBE)
    unmatched["settings_matched"] = False
    assert native384_comparison_evidence(
        native384_criteria, width, native384_owners, unmatched
    ) is None
    padded = dict(EXPECTED_NATIVE768_PROBE)
    padded["zero_padding_manufactured"] = True
    assert native768_comparison_evidence(
        native768_criteria, width, native768_owners, padded
    ) is None
    admitted_probe = dict(EXPECTED_NATIVE768_PROBE)
    admitted_probe["admitted"] = True
    assert native768_comparison_evidence(
        native768_criteria, width, native768_owners, admitted_probe
    ) is None
    dropped = _comparison_owners(
        {
            "gte_decoder_native_inputs.md": "native_inputs",
            "legal_ir_decoder_output_integration_20261016.md": "integration",
        }
    )
    dropped["integration"] = dict(dropped["integration"])
    dropped["integration"]["text"] = "moved documentation only\n"
    assert native768_comparison_evidence(
        native768_criteria, width, dropped, EXPECTED_NATIVE768_PROBE
    ) is None
    for other in (
        _WIDTH_CONTROL_ACCEPTANCE,
        _STORAGE_ACCEPTANCE,
        _NATIVE384_ACCEPTANCE,
        _NATIVE768_ACCEPTANCE,
        _EVAL_STILL_ACCEPTANCE,
    ):
        assert sparse8_comparison_evidence(
            [item["criterion"] for item in other],
            width,
            sparse_owners,
            EXPECTED_SPARSE8_PROBE,
        ) is None
    assert native384_comparison_evidence(
        sparse_criteria, width, native384_owners, EXPECTED_NATIVE384_PROBE
    ) is None
    assert native768_comparison_evidence(
        native384_criteria, width, native768_owners, EXPECTED_NATIVE768_PROBE
    ) is None


def test_release_records_native_comparison_cells(tmp_path: Path) -> None:
    relative = {}
    for name, source in _comparison_owner_paths().items():
        path = tmp_path / "docs" / "autoencoders" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source.read_bytes())
        relative[name] = "docs/autoencoders/" + name
    repo = _seed(tmp_path)
    _task(
        repo,
        "task:widths",
        "LIR-WIDTHS-DEP",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:widths",
        evidence_kind="validation",
        digest="sha256:" + ("91" * 32),
        body=_width_control_clean_dependency(),
    )
    _task(
        repo,
        "task:called",
        "LIR-CALLED-DEP",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:called",
        evidence_kind="validation",
        digest="sha256:" + ("92" * 32),
        body=_width_control_clean_dependency(producer_called=True),
    )
    _task(
        repo,
        "task:storage-dep",
        "LIR-STORAGE-DEP",
        status="completed",
        body={"completion": "manual", "execution_packet_required": True},
    )
    repo.record_evidence(
        task_cid="task:storage-dep",
        evidence_kind="validation",
        digest="sha256:" + ("93" * 32),
        body={
            "operation": "dependency_block_resolved_storage_concurrency_boundaries",
            "admitted": False,
            "formalized": False,
            "gold_or_teacher_admission_granted": False,
            "constitution_roundtrip_ok": False,
        },
    )
    gte = [relative["gte_decoder_native_inputs.md"]]
    both = [
        relative["gte_decoder_native_inputs.md"],
        relative["legal_ir_decoder_output_integration_20261016.md"],
    ]
    _task(
        repo,
        "task:native384",
        "LIR-NATIVE-384",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="inference"),
            "existing_owners": gte,
        },
        dependencies=["task:widths"],
        acceptance=_NATIVE384_ACCEPTANCE,
    )
    _task(
        repo,
        "task:native768",
        "LIR-NATIVE-768",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="inference"),
            "existing_owners": both,
        },
        dependencies=["task:widths"],
        acceptance=_NATIVE768_ACCEPTANCE,
    )
    _task(
        repo,
        "task:eval-still",
        "LIR-EVAL-STILL",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="evaluation"),
            "existing_owners": gte,
        },
        dependencies=["task:native384", "task:native768"],
        acceptance=_EVAL_STILL_ACCEPTANCE,
    )
    _task(
        repo,
        "task:publish-hold",
        "LIR-PUBLISH-HOLD",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="publication"),
            "existing_owners": [
                relative["legal_ir_asset_reconciliation_20261016.md"]
            ],
        },
        dependencies=["task:widths"],
        acceptance=_PRODUCER_STILL_ACCEPTANCE,
    )
    _task(
        repo,
        "task:bare-384",
        "LIR-BARE-384",
        status="blocked",
        body=_manual_body(_BOILERPLATE, stage="inference"),
        dependencies=["task:widths"],
        acceptance=_NATIVE384_ACCEPTANCE,
    )
    _task(
        repo,
        "task:partial-768",
        "LIR-PARTIAL-768",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="inference"),
            "existing_owners": gte,
        },
        dependencies=["task:widths"],
        acceptance=_NATIVE768_ACCEPTANCE,
    )
    _task(
        repo,
        "task:called-384",
        "LIR-CALLED-384",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="inference"),
            "existing_owners": gte,
        },
        dependencies=["task:called"],
        acceptance=_NATIVE384_ACCEPTANCE,
    )
    _task(
        repo,
        "task:storage-384",
        "LIR-STORAGE-384",
        status="blocked",
        body={
            **_manual_body(_BOILERPLATE, stage="inference"),
            "existing_owners": gte,
        },
        dependencies=["task:storage-dep"],
        acceptance=_NATIVE384_ACCEPTANCE,
    )

    result = repo.release_dependency_satisfied_blocks()

    distinct_aliases = (
        "LIR-EVAL-STILL--01-component-effects--00-cited-comparison-cells-record",
        "LIR-EVAL-STILL--02-seed-uncertainty--00-cited-comparison-cells-record",
        "LIR-EVAL-STILL--03-named-losses--00-cited-comparison-cells-record",
    )
    owner_aliases = (
        "LIR-EVAL-STILL--01-component-effects--01-evaluation-owner-records-component",
        "LIR-EVAL-STILL--02-seed-uncertainty--01-evaluation-owner-records-seed",
        "LIR-EVAL-STILL--03-named-losses--01-evaluation-owner-records-named",
    )
    assert {item["task_alias"] for item in result["completed"]} == {
        "LIR-NATIVE-384",
        "LIR-NATIVE-768",
        *distinct_aliases,
    }
    assert {item["task_alias"] for item in result["pending"]} == {
        "LIR-BARE-384",
        "LIR-PARTIAL-768",
        "LIR-CALLED-384",
        "LIR-STORAGE-384",
    }
    obligation_aliases = (
        "LIR-EVAL-STILL--00-n-comparable-k-denominators",
        "LIR-EVAL-STILL--01-component-effects",
        "LIR-EVAL-STILL--02-seed-uncertainty",
        "LIR-EVAL-STILL--03-named-losses",
        "LIR-EVAL-STILL--04-useful-source-conditioned-performance",
    )
    assert {item["task_alias"] for item in result["spawned"]} == {
        *obligation_aliases,
        *distinct_aliases,
        *owner_aliases,
    }
    open_obligations = (
        "LIR-EVAL-STILL--00-n-comparable-k-denominators",
        "LIR-EVAL-STILL--04-useful-source-conditioned-performance",
    )
    assert {
        (item["task_alias"], item["decision"]) for item in result["held"]
    } == {
        *[
            (alias, "unclassified_blocked_reason")
            for alias in (*open_obligations, *owner_aliases)
        ],
        ("LIR-PUBLISH-HOLD", "substantive_review_or_publication_hold"),
    }
    evaluation = repo.get_task("task:eval-still")
    assert evaluation["status"] == "blocked"
    assert evaluation["body"]["resolution_decomposed"] is True
    assert "task:native384" in evaluation["dependencies"]
    assert "task:native768" in evaluation["dependencies"]
    for alias in obligation_aliases:
        child = repo.get_task(alias)
        assert child["status"] == "blocked"
        assert child["body"]["resolution_subgoal"] is True
        assert child["body"]["admitted"] is False
        assert child["body"]["formalized"] is False
        assert child["body"]["gold_or_teacher_admission_granted"] is False
        assert child["body"]["constitution_roundtrip_ok"] is False
        assert child["body"]["completion_authority"] is False
        assert child["task_cid"] in evaluation["dependencies"]
    for cid, operation in (
        ("task:native384", "dependency_block_resolved_native384_comparison_cell"),
        ("task:native768", "dependency_block_resolved_native768_comparison_cell"),
    ):
        task = repo.get_task(cid)
        assert task["status"] == "completed"
        receipt = task["body"]["completion_receipt"]
        assert receipt["operation"] == operation
        assert receipt["admitted"] is False
        assert receipt["formalized"] is False
        assert receipt["gold_or_teacher_admission_granted"] is False
        assert receipt["constitution_roundtrip_ok"] is False
        assert receipt["completion_authority"] is False
        assert "settings_matched" not in receipt
        assert "encoder_identity_authenticated" not in receipt
    for cid in (
        "task:eval-still",
        "task:bare-384",
        "task:partial-768",
        "task:called-384",
        "task:storage-384",
        "task:publish-hold",
    ):
        assert repo.get_task(cid)["status"] == "blocked"
