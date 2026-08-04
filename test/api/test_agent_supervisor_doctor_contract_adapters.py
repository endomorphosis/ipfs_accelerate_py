"""Tests for DiagnosisObligationBridge@1 doctor contract adapters (PDR-010)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.doctor_contract_adapters import (
    DIAGNOSIS_OBLIGATION_BRIDGE_INTERFACE,
    AuthorityRootBridge,
    DiagnosisObligationBridge,
    DoctorContractAdapterError,
    DoctorContractAdapterReplayError,
    DoctorContractAdapterTamperError,
    FindingBridge,
    SnapshotBridge,
    adapt_deterministic_finding_to_diagnostic,
    adapt_diagnostic_finding_to_deterministic,
    adapt_diagnostic_roots_to_deterministic,
    adapt_diagnostic_snapshot_to_deterministic,
    assert_same_repository,
    portable_diagnostic_snapshot_projection,
    round_trip_deterministic_finding,
    round_trip_deterministic_snapshot,
    round_trip_diagnostic_finding,
    round_trip_diagnostic_snapshot,
)
from ipfs_accelerate_py.agent_supervisor.analysis.doctor_repository_diagnostics import (
    DoctorAuthorityRoots as DiagRoots,
    DoctorDiagnosticFinding,
    DoctorDiagnosticInput,
    DoctorSourceUnit,
    ExpectationSourceKind,
    FindingDisposition,
    FindingKind,
    diagnose_repository,
)
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DeterministicDoctorFinding,
    DoctorAuthorityRoots as DetRoots,
    DoctorEvidenceRole,
    DoctorEvidenceSnapshot as DetSnapshot,
    DoctorRepairDisposition,
)


def _diag_roots(**overrides: str) -> DiagRoots:
    base = {
        "repository_id": "repository:doctor-test",
        "forest_id": "forest:one",
        "tree_id": "tree:one",
        "overlay_id": "overlay:one",
        "config_id": "config:one",
        "toolchain_id": "toolchain:deterministic-doctor@1",
        "policy_id": "policy:doctor-test",
        "parser_id": "parser:program-ast-adapters@1",
        "file_root_id": "file-root:one",
        "ast_index_id": "ast:one",
        "dependency_graph_id": "graph:one",
    }
    base.update(overrides)
    return DiagRoots(**base)


def _det_roots(**overrides: str) -> DetRoots:
    base = {
        "repository_id": "repository:doctor-test",
        "forest_id": "forest:one",
        "tree_id": "tree:one",
        "overlay_id": "overlay:one",
        "file_root_id": "file-root:one",
        "ast_root_id": "ast:one",
        "graph_id": "graph:one",
        "corpus_id": "corpus:one",
        "index_id": "index:one",
        "model_id": "model:one",
        "cache_id": "cache:one",
        "operator_registry_id": "operators:one",
        "translator_id": "translator:one",
        "solver_id": "solver:one",
        "kernel_id": "kernel:one",
        "toolchain_id": "toolchain:one",
        "policy_id": "policy:one",
        "sandbox_id": "sandbox:one",
        "environment_id": "environment:one",
        "lease_id": "",
    }
    base.update(overrides)
    return DetRoots(**base)


def _diag_finding(
    *,
    kind: FindingKind = FindingKind.CONTRACT,
    disposition: FindingDisposition = FindingDisposition.ABSTAIN,
    path: str = "src/service.py",
    symbol: str = "dispatch",
    expectation_ref: str = "expectation:contract:dispatch",
) -> DoctorDiagnosticFinding:
    return DoctorDiagnosticFinding(
        kind=kind,
        disposition=disposition,
        path=path,
        symbol=symbol,
        message="observed signature disagrees with expectation",
        observation_refs=("fact:call:dispatch",),
        expectation_source=ExpectationSourceKind.REVIEWED_CONTRACT,
        expectation_ref=expectation_ref,
        expectation_precedence=10,
        open_frontier_refs=("frontier:optional:cfg",),
        evidence_refs=("fact:call:dispatch", "span:src/service.py:1"),
        details={"rule": "arity_mismatch"},
    )


def test_interface_constant() -> None:
    assert DIAGNOSIS_OBLIGATION_BRIDGE_INTERFACE == "DiagnosisObligationBridge@1"


def test_authority_root_bridge_round_trip() -> None:
    diag = _diag_roots()
    det = adapt_diagnostic_roots_to_deterministic(diag)
    assert det.repository_id == diag.repository_id
    assert det.tree_id == diag.tree_id
    assert det.forest_id == diag.forest_id
    # All required deterministic fields are populated.
    for name in (
        "file_root_id",
        "ast_root_id",
        "graph_id",
        "corpus_id",
        "index_id",
        "model_id",
        "cache_id",
        "operator_registry_id",
        "translator_id",
        "solver_id",
        "kernel_id",
        "toolchain_id",
        "policy_id",
        "sandbox_id",
        "environment_id",
    ):
        assert getattr(det, name), name

    bridge = AuthorityRootBridge.bridge(diag)
    restored = AuthorityRootBridge.from_dict(bridge.to_dict())
    assert restored.content_id == bridge.content_id
    assert restored.repository_id == diag.repository_id


def test_cross_repository_replay_fails_closed_on_roots() -> None:
    with pytest.raises(DoctorContractAdapterReplayError):
        adapt_diagnostic_roots_to_deterministic(
            _diag_roots(), require_repository_id="repository:other"
        )
    with pytest.raises(DoctorContractAdapterReplayError):
        assert_same_repository("repository:a", "repository:b")


def test_finding_round_trip_preserves_issue_cid() -> None:
    finding = _diag_finding()
    roots = _det_roots()
    restored = round_trip_diagnostic_finding(
        finding, roots=roots, snapshot_id="snapshot:fixture"
    )
    assert restored.finding_cid == finding.finding_cid
    assert restored.path == finding.path
    assert restored.symbol == finding.symbol
    assert restored.kind is finding.kind
    assert restored.expectation_ref == finding.expectation_ref
    assert list(restored.observation_refs) == list(finding.observation_refs)

    bridge = FindingBridge.bridge(
        finding, roots=roots, snapshot_id="snapshot:fixture"
    )
    det_finding = bridge.materialize_deterministic()
    assert det_finding.diagnostic_ref == finding.finding_cid
    assert det_finding.roots.repository_id == roots.repository_id
    assert det_finding.snapshot_id == "snapshot:fixture"

    # Deterministic → diagnostic via overlay remains lossless.
    back = adapt_deterministic_finding_to_diagnostic(
        det_finding, diagnostic_overlay=dict(bridge.diagnostic_payload)
    )
    assert back.finding_cid == finding.finding_cid


def test_finding_bridge_from_dict_round_trip() -> None:
    finding = _diag_finding()
    bridge = FindingBridge.bridge(
        finding, roots=_det_roots(), snapshot_id="snapshot:fixture"
    )
    restored = FindingBridge.from_dict(bridge.to_dict())
    assert restored.content_id == bridge.content_id
    assert restored.issue_cid == finding.finding_cid
    assert restored.materialize_diagnostic().finding_cid == finding.finding_cid


def test_supported_finding_without_expectation_downgrades() -> None:
    finding = DoctorDiagnosticFinding(
        kind=FindingKind.CONTRACT,
        disposition=FindingDisposition.SUPPORTED,
        path="src/a.py",
        observation_refs=("fact:1",),
        expectation_source=ExpectationSourceKind.NONE,
        expectation_ref="",
    )
    det_finding = adapt_diagnostic_finding_to_deterministic(
        finding, roots=_det_roots(), snapshot_id="snapshot:fixture"
    )
    # Deterministic schema forbids supported without expected-behavior refs.
    assert det_finding.disposition is DoctorRepairDisposition.ABSTAIN


def test_issue_cid_mismatch_fails_closed() -> None:
    finding = _diag_finding()
    bridge = FindingBridge.bridge(
        finding, roots=_det_roots(), snapshot_id="snapshot:fixture"
    )
    payload = bridge.to_dict()
    payload["issue_cid"] = "b" + "0" * 58
    with pytest.raises(
        (DoctorContractAdapterError, DoctorContractAdapterTamperError)
    ):
        FindingBridge.from_dict(payload)


def test_finding_tampering_fails_closed() -> None:
    finding = _diag_finding()
    bridge = FindingBridge.bridge(
        finding, roots=_det_roots(), snapshot_id="snapshot:fixture"
    )
    payload = bridge.to_dict()
    payload["content_id"] = "b" + "1" * 58
    with pytest.raises(DoctorContractAdapterTamperError):
        FindingBridge.from_dict(payload)

    payload = bridge.to_dict()
    payload["unexpected"] = True
    with pytest.raises(DoctorContractAdapterError):
        FindingBridge.from_dict(payload)


def test_body_and_secret_material_rejected_on_bridge() -> None:
    finding = _diag_finding()
    bridge = FindingBridge.bridge(
        finding, roots=_det_roots(), snapshot_id="snapshot:fixture"
    )
    payload = bridge.to_dict()
    payload["source"] = "def x():\n    return 1\n"
    with pytest.raises(DoctorContractAdapterError):
        FindingBridge.from_dict(payload)

    payload = bridge.to_dict()
    payload["api_key"] = "sk-secret"
    with pytest.raises(DoctorContractAdapterError):
        FindingBridge.from_dict(payload)

    # Body inside diagnostic payload details is rejected at finding construction.
    with pytest.raises(Exception):
        DoctorDiagnosticFinding(
            kind=FindingKind.CONTRACT,
            disposition=FindingDisposition.ABSTAIN,
            details={"source": "secret body"},
        )


def test_duplicate_issue_cids_fail_closed() -> None:
    finding = _diag_finding()
    bridge = FindingBridge.bridge(
        finding, roots=_det_roots(), snapshot_id="snapshot:fixture"
    )
    with pytest.raises(DoctorContractAdapterError):
        DiagnosisObligationBridge(
            repository_id="repository:doctor-test",
            finding_bridges=(bridge, bridge),
        )


def test_cross_repository_finding_bridge_fails() -> None:
    finding = _diag_finding()
    with pytest.raises(DoctorContractAdapterReplayError):
        FindingBridge.bridge(
            finding,
            roots=_det_roots(),
            snapshot_id="snapshot:fixture",
            require_repository_id="repository:other",
        )


def test_live_diagnostic_snapshot_round_trip() -> None:
    sources = [
        DoctorSourceUnit(
            path="src/service.py",
            source_bytes=b"def dispatch(request: str) -> str:\n    return request\n",
            language="python",
        ),
        DoctorSourceUnit(
            path="src/consumer.py",
            source_bytes=(
                b"from src.service import dispatch\n\n"
                b"def consume(payload: str) -> str:\n    return dispatch(payload)\n"
            ),
            language="python",
        ),
    ]
    snapshot = diagnose_repository(
        sources,
        authority_roots=_diag_roots(),
    )
    assert snapshot.provider_call_count == 0

    det_snap = adapt_diagnostic_snapshot_to_deterministic(snapshot)
    assert det_snap.roots.repository_id == "repository:doctor-test"
    assert det_snap.file_blob_cids
    assert det_snap.invalidation_refs

    bridge = round_trip_diagnostic_snapshot(snapshot)
    assert bridge.repository_id == "repository:doctor-test"
    assert bridge.diagnostic_snapshot_cid == snapshot.snapshot_cid
    restored_det = bridge.materialize_deterministic()
    assert restored_det.content_id == det_snap.content_id

    # Finding family round-trips through the snapshot bridge.
    for item in bridge.finding_bridges:
        assert item.materialize_diagnostic().finding_cid == item.issue_cid
        assert (
            item.materialize_deterministic().roots.repository_id
            == bridge.repository_id
        )

    portable = portable_diagnostic_snapshot_projection(snapshot)
    assert portable["snapshot_cid"] == snapshot.snapshot_cid
    assert "ast_index" not in portable  # no full AST body
    # No source body keys.
    def _no_body(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key).lower().replace("-", "_")
                assert key_text not in {
                    "body",
                    "source",
                    "source_body",
                    "source_text",
                    "source_bytes",
                    "snippet",
                    "file_text",
                    "raw_ast",
                    "ast_body",
                }
                _no_body(item)
        elif isinstance(value, list):
            for item in value:
                _no_body(item)

    _no_body(portable)
    _no_body(bridge.to_dict())


def test_diagnosis_obligation_bridge_from_diagnostic_snapshot() -> None:
    snapshot = diagnose_repository(
        [
            (
                "src/mod.py",
                "def run(value):\n    return value\n",
            )
        ],
        authority_roots=_diag_roots(),
    )
    obligation = DiagnosisObligationBridge.from_diagnostic_snapshot(
        snapshot,
        causal_slice_refs=("slice:mod:run",),
        notes=("fixture",),
    )
    assert obligation.repository_id == "repository:doctor-test"
    assert obligation.snapshot_bridge is not None
    assert obligation.root_bridge is not None
    restored = DiagnosisObligationBridge.from_dict(obligation.to_dict())
    assert restored.content_id == obligation.content_id
    assert restored.causal_slice_refs == ("slice:mod:run",)

    # Tamper fails closed.
    payload = obligation.to_dict()
    payload["content_id"] = "b" + "2" * 58
    with pytest.raises(DoctorContractAdapterTamperError):
        DiagnosisObligationBridge.from_dict(payload)


def test_deterministic_snapshot_self_round_trip() -> None:
    roots = _det_roots()
    snap = DetSnapshot(
        roots=roots,
        snapshot_id="snapshot:det-fixture",
        file_blob_cids=("blob:a", "blob:b"),
        completeness="complete",
        invalidation_refs=(roots.tree_id,),
    )
    restored = round_trip_deterministic_snapshot(snap)
    assert restored.content_id == snap.content_id


def test_deterministic_finding_structural_round_trip() -> None:
    roots = _det_roots()
    finding = DeterministicDoctorFinding(
        roots=roots,
        finding_id="finding:one",
        snapshot_id="snapshot:fixture",
        disposition=DoctorRepairDisposition.ABSTAIN,
        observed_fact_refs=("fact:a",),
        expected_behavior_refs=("expectation:a",),
        evidence_role=DoctorEvidenceRole.OBSERVED_FACT,
        finding_kind="contract",
        open_frontier_refs=("frontier:optional:cfg",),
        invalidation_refs=(roots.tree_id,),
    )
    restored = round_trip_deterministic_finding(finding)
    assert restored.roots.repository_id == roots.repository_id
    assert restored.snapshot_id == finding.snapshot_id
    assert set(restored.observed_fact_refs) == set(finding.observed_fact_refs)


def test_snapshot_bridge_rejects_cross_repository_finding() -> None:
    finding = _diag_finding()
    foreign = FindingBridge.bridge(
        finding,
        roots=_det_roots(repository_id="repository:foreign", forest_id="forest:f", tree_id="tree:f"),
        snapshot_id="snapshot:fixture",
    )
    local_roots = _det_roots()
    det_snap = DetSnapshot(
        roots=local_roots,
        snapshot_id="snapshot:local",
        file_blob_cids=("blob:a",),
        completeness="complete",
        invalidation_refs=(local_roots.tree_id,),
    )
    with pytest.raises(DoctorContractAdapterReplayError):
        SnapshotBridge(
            repository_id=local_roots.repository_id,
            diagnostic_snapshot_cid="cid:diag",
            diagnostic_snapshot_id="snap:diag",
            deterministic_snapshot_id=det_snap.snapshot_id,
            deterministic_content_id=det_snap.content_id,
            portable_diagnostic={
                "schema": "ipfs_accelerate_py/agent-supervisor/doctor-evidence-snapshot@1",
                "snapshot_cid": "cid:diag",
                "snapshot_id": "snap:diag",
            },
            deterministic_payload=det_snap.to_dict(),
            finding_bridges=(foreign,),
        )


def test_schemas_are_not_silently_aliased() -> None:
    """Diagnostic and deterministic schemas remain distinct after bridging."""
    finding = _diag_finding()
    det_finding = adapt_diagnostic_finding_to_deterministic(
        finding, roots=_det_roots(), snapshot_id="snapshot:fixture"
    )
    assert finding.to_dict()["schema"].endswith("doctor-diagnostic-finding@1")
    assert det_finding.to_dict()["schema"].endswith(
        "deterministic-doctor/finding@1"
    )
    assert finding.to_dict()["schema"] != det_finding.to_dict()["schema"]
