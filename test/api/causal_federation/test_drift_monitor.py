"""CASF-033 exact architecture/event drift monitor acceptance tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.contracts import (
    FederationBinding,
    FederationBoundsError,
    FederationContractError,
    UnknownNormativeFieldError,
)
from ipfs_accelerate_py.agent_supervisor.federation.drift_monitor import (
    DRIFT_REPORT_SCHEMA,
    FEDERATION_DRIFT_MONITOR_INTERFACE,
    MAX_DRIFT_FINDINGS,
    DriftKind,
    DriftMonitorError,
    FederationDriftMonitor,
    FederationDriftRoots,
    StaleDriftReportError,
    closed_event_catalog_root,
    produce_drift_report,
    validate_current_drift_report,
)
from ipfs_accelerate_py.agent_supervisor.federation.events import (
    DomainEvent,
    EventClass,
    EventEffectClass,
)

NOW = "2026-08-24T00:00:00Z"


def _roots(**overrides: object) -> FederationDriftRoots:
    values: dict[str, object] = {
        "tenant_id": "tenant:test",
        "federation_id": "federation:test",
        "repository_id": "repository:accelerate",
        "repository_tree_id": "tree:current",
        "control_plane_generation": 7,
        "schema_root": "schema:current",
        "operation_catalog_root": "operations:current",
        "event_catalog_root": closed_event_catalog_root(),
        "causal_graph_root": "causal:current",
        "causal_graph_revision": 19,
        "event_watermark": 10,
    }
    values.update(overrides)
    return FederationDriftRoots(**values)  # type: ignore[arg-type]


def _event(
    sequence: int,
    *,
    event_id: str | None = None,
    parents: tuple[str, ...] = (),
    tenant_id: str = "tenant:test",
    federation_id: str = "federation:test",
    repository_id: str = "repository:accelerate",
    tree_id: str = "tree:current",
) -> DomainEvent:
    identity = event_id or f"event:{sequence}"
    return DomainEvent(
        event_id=identity,
        event_cid=f"cid:event-{sequence}",
        event_type=EventClass.TASK_READY,
        stream_id="stream:test",
        stream_sequence=sequence,
        global_sequence=sequence,
        causal_parent_ids=parents,
        correlation_id="correlation:test",
        causation_id="causation:test",
        tenant_id=tenant_id,
        federation_id=federation_id,
        supervisor_id="supervisor:test",
        task_id="CASF-033",
        repository_id=repository_id,
        tree_id=tree_id,
        goal_id="CASF-G041",
        subgoal_id="CASF-G041",
        symbol_id="",
        contract_id="",
        proof_obligation_id="",
        resource_class="cpu-standard-local-proof",
        payload_ref="payload:test",
        changed_fact_refs=(f"fact:{sequence}",),
        effect_class=EventEffectClass.READ_ONLY,
        recorded_at=NOW,
        expires_at="",
        deduplication_key=f"dedup:{sequence}",
    )


def _binding() -> FederationBinding:
    return FederationBinding(
        tenant_id="tenant:test",
        repository_ids=("repository:accelerate",),
        repository_tree_ids=("tree:current",),
        program_id="agent-supervisor-causal-event-federation-v1",
        objective_ref="CASF-G000",
        objective_revision=1,
        policy_ref="policy:test",
        policy_revision=1,
        operation_catalog_ref="operations:current",
        control_plane_generation=7,
        causal_graph_revision=19,
        semantic_state_roots=("semantic:current",),
        supervisor_population=1,
        budget_ref="budget:test",
        expires_at="2026-08-25T00:00:00Z",
        issuer="issuer:test",
        authorization_evidence_ref="evidence:test",
    )


def test_exact_unchanged_roots_are_current_and_non_authoritative() -> None:
    roots = _roots()
    report = FederationDriftMonitor(roots).observe(roots, observed_at=NOW)

    assert FederationDriftMonitor.INTERFACE == FEDERATION_DRIFT_MONITOR_INTERFACE
    assert report.SCHEMA == DRIFT_REPORT_SCHEMA
    assert report.current is True
    assert report.findings == ()
    assert report.to_dict()["authority"] is False
    assert report.to_dict()["production_state_changed"] is False
    assert report.to_dict()["ducklake_authoritative"] is False
    assert report.to_dict()["model_calls"] == 0
    assert report.to_dict()["provider_calls"] == 0


def test_binding_projects_only_the_exact_repository_tree() -> None:
    roots = FederationDriftRoots.from_binding(
        _binding(),
        federation_id="federation:test",
        schema_root="schema:current",
        event_catalog_root=closed_event_catalog_root(),
        causal_graph_root="causal:current",
        event_watermark=10,
    )
    assert roots == _roots()

    with pytest.raises(DriftMonitorError, match="disagrees"):
        FederationDriftRoots.from_binding(
            _binding(),
            federation_id="federation:test",
            schema_root="schema:current",
            event_catalog_root=closed_event_catalog_root(),
            causal_graph_root="causal:current",
            event_watermark=10,
            repository_tree_id="tree:stale",
        )

    with pytest.raises(DriftMonitorError, match="canonical CASF program"):
        FederationDriftRoots.from_binding(
            replace(_binding(), program_id="foreign-federation-program"),
            federation_id="federation:test",
            schema_root="schema:current",
            event_catalog_root=closed_event_catalog_root(),
            causal_graph_root="causal:current",
            event_watermark=10,
        )


def test_schema_operation_event_and_causal_drift_are_precise() -> None:
    baseline = _roots()
    observed = replace(
        baseline,
        schema_root="schema:changed",
        operation_catalog_root="operations:changed",
        event_catalog_root="events:changed",
        causal_graph_root="causal:changed",
        causal_graph_revision=20,
    )
    report = produce_drift_report(baseline, observed, observed_at=NOW)
    codes = {item.code for item in report.findings}

    assert report.drifted is True
    assert {
        "schema_root_changed",
        "operation_catalog_root_changed",
        "event_catalog_root_changed",
        "causal_graph_root_changed",
        "causal_graph_revision_changed",
    } <= codes
    assert {item.kind for item in report.findings} >= {
        DriftKind.SCHEMA,
        DriftKind.OPERATION,
        DriftKind.EVENT,
        DriftKind.CAUSAL,
    }
    assert all(item.blocks_promotion for item in report.findings)


def test_exact_contiguous_event_window_advances_without_drift() -> None:
    baseline = _roots()
    observed = replace(baseline, event_watermark=12)
    report = FederationDriftMonitor(baseline).observe(
        observed,
        events=(
            _event(11),
            _event(12, parents=("event:11",)),
        ),
        observed_at=NOW,
    )

    assert report.current is True
    assert report.event_range_start == 11
    assert report.event_range_end == 12
    assert report.observed_event_count == 2


def test_gaps_duplicates_binding_drift_and_missing_parents_are_reported() -> None:
    baseline = _roots()
    observed = replace(baseline, event_watermark=13)
    report = FederationDriftMonitor(baseline).observe(
        observed,
        events=(
            _event(11, parents=("event:unseen",)),
            _event(13, event_id="event:duplicate", tree_id="tree:stale"),
            _event(13, event_id="event:duplicate"),
        ),
        observed_at=NOW,
    )
    codes = {item.code for item in report.findings}

    assert "event_range_not_exact" in codes
    assert "duplicate_event_id" in codes
    assert "duplicate_global_sequence" in codes
    assert "event_tree_id_changed" in codes
    assert "event_causal_parent_missing" in codes


def test_watermark_regression_and_unobserved_advance_fail_closed() -> None:
    baseline = _roots()
    regressed = FederationDriftMonitor(baseline).observe(
        replace(baseline, event_watermark=9), observed_at=NOW
    )
    advanced = FederationDriftMonitor(baseline).observe(
        replace(baseline, event_watermark=11), observed_at=NOW
    )

    assert {item.code for item in regressed.findings} == {
        "event_watermark_regressed"
    }
    assert {item.code for item in advanced.findings} == {
        "event_watermark_advance_unobserved"
    }


def test_oversized_event_window_is_reported_without_materializing_it() -> None:
    """A bounded observer must not allocate an arbitrary watermark span."""

    baseline = _roots()
    observed = replace(baseline, event_watermark=2**63 - 1)
    report = FederationDriftMonitor(baseline).observe(
        observed,
        events=(_event(2**63 - 1),),
        observed_at=NOW,
    )

    finding = next(
        item
        for item in report.findings
        if item.code == "event_window_exceeds_observation_bound"
    )
    assert finding.expected == "at_most:4096"
    assert finding.observed == str(2**63 - 1 - baseline.event_watermark)


def test_finding_volume_limit_fails_closed_before_creating_a_report() -> None:
    """Adversarial causal-parent vectors cannot create an oversized receipt."""

    baseline = _roots()
    event_count = 64
    events = tuple(
        _event(
            baseline.event_watermark + offset,
            tenant_id="tenant:other",
            parents=tuple(f"parent:{offset}:{parent}" for parent in range(256)),
        )
        for offset in range(1, event_count + 1)
    )
    observed = replace(
        baseline,
        event_watermark=baseline.event_watermark + event_count,
    )

    with pytest.raises(FederationBoundsError, match="drift finding bound exceeded"):
        FederationDriftMonitor(baseline).observe(
            observed,
            events=events,
            observed_at=NOW,
        )


def test_report_identity_is_deterministic_and_current_tree_validated() -> None:
    roots = _roots()
    first = produce_drift_report(roots, roots, observed_at=NOW)
    second = produce_drift_report(roots, roots, observed_at=NOW)
    assert first.report_id == second.report_id

    validation = validate_current_drift_report(
        first,
        current_repository_tree_id="tree:current",
        current_control_plane_generation=7,
        require_drift_free=True,
    )
    assert validation["current_tree_bound"] is True
    assert validation["authority"] is False

    with pytest.raises(StaleDriftReportError, match="stale tree"):
        validate_current_drift_report(
            first,
            current_repository_tree_id="tree:new",
            current_control_plane_generation=7,
        )
    with pytest.raises(StaleDriftReportError, match="stale generation"):
        validate_current_drift_report(
            first,
            current_repository_tree_id="tree:current",
            current_control_plane_generation=8,
        )


def test_wire_contracts_are_closed_and_reject_forged_evidence() -> None:
    roots = _roots()
    report = produce_drift_report(roots, roots, observed_at=NOW)

    assert FederationDriftRoots.from_dict(roots.to_dict()) == roots
    assert type(report).from_dict(report.to_dict()) == report

    with pytest.raises(UnknownNormativeFieldError, match="unknown fields"):
        FederationDriftRoots.from_dict({**roots.to_dict(), "extension": "no"})
    with pytest.raises(DriftMonitorError, match="status disagrees"):
        type(report).from_dict({**report.to_dict(), "status": "drifted"})
    with pytest.raises(DriftMonitorError, match="authority must be false"):
        type(report).from_dict({**report.to_dict(), "authority": True})

    drifted = produce_drift_report(
        roots,
        replace(roots, schema_root="schema:changed"),
        observed_at=NOW,
    )
    oversized = drifted.to_dict()
    oversized["findings"] = [drifted.findings[0].to_dict()] * (
        MAX_DRIFT_FINDINGS + 1
    )
    with pytest.raises(FederationBoundsError, match="drift finding bound exceeded"):
        type(report).from_dict(oversized)


def test_monitor_rejects_non_sequence_event_inputs() -> None:
    roots = _roots()
    with pytest.raises(DriftMonitorError, match="events must be an array"):
        FederationDriftMonitor(roots).observe(
            roots,
            events={"event": _event(11)},  # type: ignore[arg-type]
            observed_at=NOW,
        )


def test_monitor_rejects_paths_and_missing_capability_roots() -> None:
    with pytest.raises(DriftMonitorError, match="never a database path"):
        FederationDriftMonitor(Path("control.duckdb"))  # type: ignore[arg-type]
    with pytest.raises(FederationContractError):
        _roots(schema_root="")
