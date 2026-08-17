"""Tests for SCG-039 privacy-filtered report and dashboard-data projections.

Acceptance criteria enforced here:

* Required final-report fields are representable.
* Raw private source, secrets, arbitrary paths, and human/model free-form
  authority are absent from public projections.
* Unavailable / simulated / heuristic / proof-scope fields are explicit.
* Machine-readable summary (dashboard-data) and detail (report) views.
* No network imports; pure projection surface.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.semantic_governor.audit_contracts import (
    RouteTier,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.calibration_contracts import (
    EvidencePartition,
)

from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    AcceptanceDisposition,
    ComparativeOutcome,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.metrics import (
    MetricsCohort,
    MetricsObservation,
    collect_metrics,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.report import (
    BUILD_DASHBOARD_DATA_INTERFACE,
    BUILD_GOVERNOR_REPORT_INTERFACE,
    DASHBOARD_DATA_INTERFACE,
    GOVERNOR_REPORT_INTERFACE,
    REQUIRED_FINAL_REPORT_FIELDS,
    SCG_DASHBOARD_DATA_EVIDENCE,
    SCG_FINAL_REPORT_EVIDENCE,
    DashboardData,
    EvidenceMode,
    FreeFormAuthorityError,
    GovernorReport,
    HeuristicClass,
    ProofScopeKind,
    ReportError,
    SealScopeStatus,
    build_dashboard_data,
    build_dashboard_data_interface_id,
    build_governor_report,
    build_governor_report_interface_id,
    dashboard_data_evidence_id,
    dashboard_data_interface_id,
    evidence_modes,
    final_report_evidence_id,
    governor_report_interface_id,
    heuristic_classes,
    project_governor_public,
    proof_scope_kinds,
    reject_free_form_authority,
    required_final_report_fields,
    seal_scope_statuses,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_governor/report.py"
)

# Proposal-gate-safe sentinels only (never embed concrete credential material).
# Field-name rejection covers auth keys; text canaries use admitted synthetic forms.
CANARY_API_KEY = "sk_live_not_a_real_key"
CANARY_PASSWORD = "canary-password"
CANARY_BEARER = "Bearer canary-secret-value"
CANARY_AUTH_TOKEN = "canary-auth-token"
COMMIT_SHA = "a" * 40


# ---------------------------------------------------------------------------
# Fixtures / recipes
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _obs(observation_id: str = "obs_0001", *, cohort: str = MetricsCohort.LIVE.value, **overrides: Any) -> MetricsObservation:
    fields: dict[str, Any] = {
        "observation_id": observation_id,
        "receipt_cid": _cid(f"receipt-{observation_id}"),
        "cohort": cohort,
        "route_tier": RouteTier.MEDIUM.value,
        "comparative_outcome": ComparativeOutcome.EQUIVALENT_SUCCESS.value,
        "acceptance_disposition": AcceptanceDisposition.ACCEPTED.value,
        "raw_tokens": 1000,
        "retrieval_tokens": 800,
        "compressed_tokens": 400,
        "expanded_tokens": 400,
        "accepted_patch": True,
        "regression": False,
        "selected_test_false_negative": False,
        "proof_failure": False,
        "review_disagreement": False,
        "intentional_omission_present": False,
        "omission_detected_before_execution": False,
        "omission_detected_after_execution": False,
        "critical_omission": False,
        "critical_omission_accepted": False,
        "expansion_used": False,
        "expansion_true_positive": False,
        "expansion_false_positive": False,
        "expansion_false_negative": False,
        "escalated": False,
        "retried": False,
        "input_tokens": 400,
        "output_tokens": 100,
        "baseline_model_spend_micros": 10_000,
        "model_spend_micros": 4_000,
        "verification_compute_micros": 500,
        "shadow_compute_micros": 300,
        "audit_overhead_micros": 200,
        "calibration_use": False,
        "calibration_revision": None,
        "omission_failure": False,
        "task_class": "local_bug",
        "partition": EvidencePartition.DEVELOPMENT.value,
        "metadata": {},
    }
    fields.update(overrides)
    return MetricsObservation(**fields)


def _sample_metrics():
    live = _obs(
        "live_1",
        raw_tokens=2000,
        compressed_tokens=500,
        intentional_omission_present=True,
        omission_detected_before_execution=True,
        critical_omission=True,
        expansion_used=True,
        expansion_true_positive=True,
        escalated=True,
    )
    sim = _obs(
        "sim_1",
        cohort=MetricsCohort.SIMULATED.value,
        acceptance_disposition=AcceptanceDisposition.NOT_ACCEPTED.value,
        accepted_patch=False,
        regression=True,
        raw_tokens=100,
        compressed_tokens=10,
    )
    return collect_metrics([live, sim])


# ---------------------------------------------------------------------------
# Module hygiene
# ---------------------------------------------------------------------------


def test_module_exports_required_interfaces() -> None:
    assert SCG_DASHBOARD_DATA_EVIDENCE == "scg/dashboard-data@1"
    assert SCG_FINAL_REPORT_EVIDENCE == "scg/final-report@1"
    assert BUILD_GOVERNOR_REPORT_INTERFACE == "build_governor_report@1"
    assert BUILD_DASHBOARD_DATA_INTERFACE == "build_dashboard_data@1"
    assert GOVERNOR_REPORT_INTERFACE == "GovernorReport@1"
    assert DASHBOARD_DATA_INTERFACE == "DashboardData@1"
    assert dashboard_data_evidence_id() == SCG_DASHBOARD_DATA_EVIDENCE
    assert final_report_evidence_id() == SCG_FINAL_REPORT_EVIDENCE
    assert build_governor_report_interface_id() == BUILD_GOVERNOR_REPORT_INTERFACE
    assert build_dashboard_data_interface_id() == BUILD_DASHBOARD_DATA_INTERFACE
    assert governor_report_interface_id() == GOVERNOR_REPORT_INTERFACE
    assert dashboard_data_interface_id() == DASHBOARD_DATA_INTERFACE
    assert MODULE_PATH.is_file()
    assert "live" in evidence_modes()
    assert SealScopeStatus.UNAVAILABLE.value in seal_scope_statuses()
    assert ProofScopeKind.UNAVAILABLE.value in proof_scope_kinds()
    assert HeuristicClass.NONE.value in heuristic_classes()


def test_module_is_pure_no_network_imports() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"socket", "http", "urllib", "requests", "aiohttp", "httpx"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                assert root not in forbidden
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            assert root not in forbidden


# ---------------------------------------------------------------------------
# Required final-report fields are representable
# ---------------------------------------------------------------------------


def test_required_final_report_fields_always_present() -> None:
    report = build_governor_report()
    payload = report.to_dict()
    required = required_final_report_fields()
    assert required == REQUIRED_FINAL_REPORT_FIELDS
    for field_name in required:
        assert field_name in payload, field_name
    # Empty inputs → explicit unavailable, not fabricated success.
    assert payload["seal_scope"]["status"] == SealScopeStatus.UNAVAILABLE.value
    assert payload["seal_scope"]["unavailable"] is True
    assert payload["proof_scope"]["unavailable"] is True
    assert payload["heuristics"]["unavailable"] is True
    assert payload["quality"]["unavailable"] is True
    assert "metric_report_cid" in payload["unavailable_fields"]
    assert payload["evidence_mode"] == EvidenceMode.UNAVAILABLE.value
    assert payload["report_cid"] == report.report_cid


def test_report_round_trip_and_cid_determinism() -> None:
    report = build_governor_report(
        inspected_commits=[COMMIT_SHA],
        implemented_commits=[COMMIT_SHA],
        consumed_interfaces=["GovernorMetricsCollector@1", "build_governor_report@1"],
        remaining_production_risks=["sealer_unavailable"],
    )
    payload = report.to_dict()
    restored = GovernorReport.from_dict(payload)
    assert restored.report_cid == report.report_cid
    assert restored.to_dict() == payload
    # Deterministic identity for identical inputs.
    again = build_governor_report(
        inspected_commits=[COMMIT_SHA],
        implemented_commits=[COMMIT_SHA],
        consumed_interfaces=["GovernorMetricsCollector@1", "build_governor_report@1"],
        remaining_production_risks=["sealer_unavailable"],
    )
    assert again.report_cid == report.report_cid


def test_report_binds_commits_interfaces_histories_and_scopes() -> None:
    history = _cid("history-1")
    candidate = _cid("candidate-1")
    eval_report = _cid("eval-1")
    rollback_cid = _cid("rollback-1")
    report = build_governor_report(
        histories=[history],
        inspected_commits=[COMMIT_SHA, _cid("inspected-tree")],
        implemented_commits=[COMMIT_SHA],
        consumed_interfaces=[
            "SemanticCompressionGovernor@1",
            "GovernorMetricsCollector@1",
        ],
        rules={
            "proposed_count": 2,
            "rejected_count": 1,
            "promoted_count": 0,
            "candidate_cids": [candidate],
            "evaluation_report_cids": [eval_report],
            "unavailable": False,
        },
        rollback={
            "rollback_count": 1,
            "rollback_decision_cids": [rollback_cid],
            "last_rollback_decision_cid": rollback_cid,
            "unavailable": False,
        },
        seal_scope={
            "status": SealScopeStatus.UNAVAILABLE.value,
            "unavailable": True,
        },
        proof_scope={
            "kind": ProofScopeKind.STRUCTURAL_NON_ZK.value,
            "claim_kinds": ["exact_artifacts_evaluated"],
            "claims_semantic_sufficiency": False,
            "is_zero_knowledge": False,
            "unavailable": False,
        },
        heuristics={
            "classification": HeuristicClass.PRESENT.value,
            "heuristic_labels": ["coverage_heuristic"],
            "treated_as_exact": False,
            "unavailable": False,
        },
        remaining_production_risks=[
            "sealer_unavailable",
            "limited_held_out_coverage",
        ],
    )
    payload = report.to_dict()
    assert history in payload["audit_population"]["history_cids"]
    assert COMMIT_SHA in payload["inspected_commits"]
    assert "GovernorMetricsCollector@1" in payload["consumed_interfaces"]
    assert payload["rules"]["proposed_count"] == 2
    assert payload["rollback"]["rollback_count"] == 1
    assert payload["proof_scope"]["kind"] == ProofScopeKind.STRUCTURAL_NON_ZK.value
    assert payload["proof_scope"]["claims_semantic_sufficiency"] is False
    assert payload["proof_scope"]["is_zero_knowledge"] is False
    assert payload["heuristics"]["classification"] == HeuristicClass.PRESENT.value
    assert payload["heuristics"]["treated_as_exact"] is False
    assert "sealer_unavailable" in payload["remaining_production_risks"]


# ---------------------------------------------------------------------------
# Metrics → report projection
# ---------------------------------------------------------------------------


def test_build_governor_report_from_metrics_separates_cohorts() -> None:
    metrics = _sample_metrics()
    report = build_governor_report(metrics=metrics, inspected_commits=[COMMIT_SHA])
    payload = report.to_dict()
    assert payload["evidence_mode"] == EvidenceMode.MIXED.value
    assert payload["simulated_metrics_present"] is True
    assert payload["metric_report_cid"] == metrics.report_cid
    assert payload["audit_population"]["live_audits"] == 1
    assert payload["audit_population"]["simulated_audits"] == 1
    # Live quality only: simulated regression must not appear as live.
    assert payload["quality"]["accepted_patch_count"] == 1
    assert payload["quality"]["regression_count"] == 0
    assert payload["quality"]["unavailable"] is False
    assert payload["final_context_reduction"]["median_context_reduction_bp"] is not None
    assert payload["omission_detection"]["detected_before_execution_count"] == 1
    assert payload["omission_detection"]["critical_omission_count"] == 1
    assert payload["expansion"]["expansion_count"] == 1
    assert payload["route_distribution"]["escalation_count"] == 1
    assert payload["overhead_and_cost"]["net_savings_micros"] is not None


def test_unavailable_measurements_are_not_zero_success() -> None:
    report = build_governor_report()
    payload = report.to_dict()
    assert payload["quality"]["accepted_rate_bp"] is None
    assert payload["final_context_reduction"]["median_context_reduction_bp"] is None
    assert payload["overhead_and_cost"]["cost_per_accepted_patch_micros"] is None
    assert payload["quality"]["unavailable"] is True
    assert "quality" in payload["unavailable_fields"]


# ---------------------------------------------------------------------------
# Privacy: private source, secrets, paths, free-form authority
# ---------------------------------------------------------------------------


def test_rejects_raw_private_source() -> None:
    with pytest.raises(ReportError) as excinfo:
        build_governor_report(metadata={"raw_private_source": "class Secret: pass"})
    assert excinfo.value.reason_code in {
        "private_or_model_authority",
        "private_source_rejected",
    }
    with pytest.raises(ReportError):
        project_governor_public({"summary": "ok", "source_text": "leak"})


def test_rejects_secrets() -> None:
    with pytest.raises(ReportError):
        build_governor_report(metadata={"api_key": CANARY_API_KEY})
    with pytest.raises(ReportError):
        build_governor_report(metadata={"password": CANARY_PASSWORD})
    with pytest.raises(ReportError):
        build_governor_report(metadata={"note": CANARY_BEARER})
    with pytest.raises(ReportError):
        project_governor_public({"auth_token": CANARY_AUTH_TOKEN})


def test_rejects_arbitrary_host_paths() -> None:
    with pytest.raises(ReportError) as excinfo:
        build_governor_report(metadata={"note": "/tmp/secret.bin"})
    assert excinfo.value.reason_code == "host_path_rejected"
    with pytest.raises(ReportError):
        build_governor_report(metadata={"workspace_path": "/home/alice/repo"})
    with pytest.raises(ReportError):
        project_governor_public({"workdir": "C:\\Users\\alice\\repo"})


def test_rejects_human_and_model_free_form_authority() -> None:
    with pytest.raises((ReportError, FreeFormAuthorityError)):
        build_governor_report(metadata={"model_authority": True})
    with pytest.raises((ReportError, FreeFormAuthorityError)):
        build_governor_report(metadata={"human_free_form_authority": "ship_it"})
    with pytest.raises((ReportError, FreeFormAuthorityError)):
        build_governor_report(metadata={"free_form_authority": "yes"})
    with pytest.raises((ReportError, FreeFormAuthorityError)):
        build_governor_report(metadata={"promotion_authority": "model"})
    with pytest.raises(FreeFormAuthorityError):
        reject_free_form_authority({"human_override_authority": True})


def test_public_projection_admits_cids_commits_and_token_counts() -> None:
    portable = {
        "report_cid": _cid("r"),
        "metric_report_cid": _cid("m"),
        "inspected_commits": [COMMIT_SHA],
        "expanded_tokens_total": 400,
        "raw_tokens_total": 1000,
        "summary": "portable facts only",
    }
    projected = project_governor_public(portable)
    assert projected["expanded_tokens_total"] == 400
    assert projected["inspected_commits"] == [COMMIT_SHA]


def test_commit_refs_reject_host_paths() -> None:
    with pytest.raises(ReportError):
        build_governor_report(inspected_commits=["/home/alice/repo"])
    with pytest.raises(ReportError):
        build_governor_report(implemented_commits=["~/src"])


# ---------------------------------------------------------------------------
# Proof scope / heuristics fail-closed
# ---------------------------------------------------------------------------


def test_proof_scope_rejects_semantic_sufficiency_and_zk_overclaim() -> None:
    with pytest.raises(ReportError) as excinfo:
        build_governor_report(
            proof_scope={
                "kind": ProofScopeKind.BOUNDED_ARTIFACT_EVALUATION.value,
                "claims_semantic_sufficiency": True,
                "is_zero_knowledge": False,
                "unavailable": False,
            }
        )
    assert excinfo.value.reason_code == "proof_scope_overclaim"
    with pytest.raises(ReportError) as excinfo:
        build_governor_report(
            proof_scope={
                "kind": ProofScopeKind.BOUNDED_ARTIFACT_EVALUATION.value,
                "claims_semantic_sufficiency": False,
                "is_zero_knowledge": True,
                "unavailable": False,
            }
        )
    assert excinfo.value.reason_code == "proof_scope_overclaim"


def test_heuristics_never_treated_as_exact() -> None:
    with pytest.raises(ReportError) as excinfo:
        build_governor_report(
            heuristics={
                "classification": HeuristicClass.PRESENT.value,
                "heuristic_labels": ["foo"],
                "treated_as_exact": True,
                "unavailable": False,
            }
        )
    assert excinfo.value.reason_code == "heuristic_as_exact_rejected"


# ---------------------------------------------------------------------------
# Dashboard-data summary projection
# ---------------------------------------------------------------------------


def test_build_dashboard_data_summary_from_report() -> None:
    metrics = _sample_metrics()
    report = build_governor_report(
        metrics=metrics,
        inspected_commits=[COMMIT_SHA],
        remaining_production_risks=["sealer_unavailable"],
        seal_scope={"status": SealScopeStatus.UNAVAILABLE.value, "unavailable": True},
        proof_scope={
            "kind": ProofScopeKind.UNAVAILABLE.value,
            "unavailable": True,
        },
        heuristics={
            "classification": HeuristicClass.EXCLUDED_FROM_EXACT.value,
            "heuristic_labels": ["capsule_heuristic"],
            "treated_as_exact": False,
            "unavailable": False,
        },
    )
    dashboard = build_dashboard_data(report, metrics=metrics)
    payload = dashboard.to_dict()
    assert payload["evidence"] == SCG_DASHBOARD_DATA_EVIDENCE
    assert payload["report_cid"] == report.report_cid
    assert payload["dashboard_cid"] == dashboard.dashboard_cid
    assert payload["live_observation_count"] == 1
    assert payload["simulated_observation_count"] == 1
    assert payload["simulated_metrics_present"] is True
    assert payload["seal_status"] == SealScopeStatus.UNAVAILABLE.value
    assert payload["proof_scope_kind"] == ProofScopeKind.UNAVAILABLE.value
    assert (
        payload["heuristic_classification"]
        == HeuristicClass.EXCLUDED_FROM_EXACT.value
    )
    assert "sealer_unavailable" in payload["remaining_production_risks"]
    assert payload["median_context_reduction_bp"] is not None
    # Round-trip
    restored = DashboardData.from_dict(payload)
    assert restored.dashboard_cid == dashboard.dashboard_cid


def test_build_dashboard_data_from_kwargs_without_prior_report() -> None:
    metrics = _sample_metrics()
    dashboard = build_dashboard_data(
        metrics=metrics,
        inspected_commits=[COMMIT_SHA],
    )
    assert dashboard.evidence == SCG_DASHBOARD_DATA_EVIDENCE
    assert dashboard.live_observation_count == 1
    assert dashboard.report_cid
    assert dashboard.metric_report_cid == metrics.report_cid


def test_dashboard_empty_inputs_are_explicitly_unavailable() -> None:
    dashboard = build_dashboard_data()
    payload = dashboard.to_dict()
    assert payload["evidence_mode"] == EvidenceMode.UNAVAILABLE.value
    assert payload["seal_status"] == SealScopeStatus.UNAVAILABLE.value
    assert payload["proof_scope_kind"] == ProofScopeKind.UNAVAILABLE.value
    assert payload["live_observation_count"] is None
    assert payload["median_context_reduction_bp"] is None


def test_dashboard_rejects_private_metadata() -> None:
    with pytest.raises(ReportError):
        build_dashboard_data(metadata={"raw_private_source": "leak"})
    with pytest.raises(ReportError):
        build_dashboard_data(metadata={"secret": "x"})


def test_forged_report_cid_rejected() -> None:
    report = build_governor_report(inspected_commits=[COMMIT_SHA])
    payload = report.to_dict()
    payload["report_cid"] = _cid("forged")
    with pytest.raises(ReportError):
        GovernorReport.from_dict(payload)


def test_forged_dashboard_cid_rejected() -> None:
    dashboard = build_dashboard_data()
    payload = dashboard.to_dict()
    payload["dashboard_cid"] = _cid("forged")
    with pytest.raises(ReportError):
        DashboardData.from_dict(payload)
