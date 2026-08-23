"""Hermetic PCAR-007 duplicate-authority detection tests."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.architecture_ir import (
    ArchitectureEdge,
    ArchitectureIR,
    ArchitectureNode,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.authority_graph import (
    ArbitrationRationale,
    ConcernKind,
    FormalArbitration,
    LoserClassification,
    OwnerDisposition,
    resolve_authority_ownership,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.contracts import (
    Confidence,
    EdgeKind,
    NodeKind,
    SourceFactIdentity,
    SourceSpan,
)
from ipfs_accelerate_py.agent_supervisor.architecture_refactorer.duplicate_authority import (
    BYPASS_SCHEMA,
    CLOSED_COLLISION_DISPOSITIONS,
    CLOSED_COLLISION_KINDS,
    CLOSED_DUPLICATE_AUTHORITY_BLOCKERS,
    CLOSED_SURFACES,
    COLLISION_SCHEMA,
    CONTENT_IDENTITY_IS_NOT_AUTHORITY,
    DEFAULT_FRESHNESS,
    DETECTOR_CAN_AUTHORIZE_CHANGES,
    DETECTOR_CAN_REMEDIATE,
    DETECTOR_CAN_SELECT_CANONICAL,
    DUPLICATE_AUTHORITY_EVIDENCE,
    DUPLICATE_AUTHORITY_SCHEMA,
    DUPLICATE_AUTHORITY_VERSION,
    EFFECT_CLASS,
    EXTRACTOR_IDENTITY,
    HEURISTIC_CRITICAL_PROMOTION_PROHIBITED,
    ONE_OWNER_INVARIANT,
    REQUIRED_COLLISION_KINDS,
    REQUIRED_SURFACES,
    REEXPORT_IS_NOT_AUTHORITY,
    SILENT_ARBITRATION_PROHIBITED,
    SURFACE_SCHEMA,
    TASK_ID,
    AuthorityCollision,
    BypassFinding,
    CollisionDisposition,
    CollisionKind,
    DuplicateAuthorityAuthorityError,
    DuplicateAuthorityBlockerKind,
    DuplicateAuthorityDetector,
    DuplicateAuthorityError,
    DuplicateAuthorityReport,
    SurfaceDivergenceFinding,
    SurfaceKind,
    build_duplicate_authority_report,
    detect_duplicate_authorities,
    lookup_owner_by_content_identity,
    refuse_authorization,
    refuse_canonical_selection,
    refuse_remediation,
    silently_select_canonical,
)
from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
_FRESHNESS = "pcar-007-fixture"
_EXTRACTOR = "pcar-007-fixture"
_CAP_PATH = "ipfs_accelerate_py/agent_supervisor/control/capability_resolver.py"
_RECEIPT_PATH = "ipfs_accelerate_py/agent_supervisor/contracts/execution.py"
_STATE_PATH = "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py"
_AUTH_PATH = "ipfs_accelerate_py/agent_supervisor/control/authorization_logic.py"
_POLICY_PATH = "ipfs_accelerate_py/agent_supervisor/control/profile_authority.py"
_CLI_PATH = "ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py"
_MCP_PATH = "ipfs_accelerate_py/agent_supervisor/entrypoints/mcp_server.py"
_PY_PATH = "ipfs_accelerate_py/agent_supervisor/entrypoints/python_api.py"
_SIM_PATH = "ipfs_accelerate_py/agent_supervisor/runtime/provider_usage.py"
_COMPAT_PATH = "ipfs_accelerate_py/agent_supervisor/todo_daemon/legacy_landed_review.py"
_TEST_PATH = "test/api/architecture_refactorer/test_duplicate_authority.py"
_OP_PATH = "ipfs_accelerate_py/agent_supervisor/control/control_contracts.py"


def _span(path: str, start: int, end: int | None = None) -> SourceSpan:
    return SourceSpan(path, start, start if end is None else end)


def _fact(
    path: str,
    start: int,
    *,
    confidence: Confidence = Confidence.EXACT,
    end: int | None = None,
) -> SourceFactIdentity:
    return SourceFactIdentity(
        extractor_identity=_EXTRACTOR,
        span=_span(path, start, end),
        confidence=confidence,
        freshness=_FRESHNESS,
        repository_tree=_TREE,
    )


def _node(
    node_id: str,
    kind: NodeKind,
    path: str,
    start: int,
    *,
    confidence: Confidence = Confidence.EXACT,
) -> ArchitectureNode:
    return ArchitectureNode(
        node_id=node_id,
        kind=kind,
        provenance=_fact(path, start, confidence=confidence),
    )


def _edge(
    edge_id: str,
    kind: EdgeKind,
    source: str,
    target: str,
    path: str,
    start: int,
    *,
    confidence: Confidence = Confidence.EXACT,
) -> ArchitectureEdge:
    return ArchitectureEdge(
        edge_id=edge_id,
        kind=kind,
        source=source,
        target=target,
        provenance=_fact(path, start, confidence=confidence),
    )


def _graph(
    nodes: tuple[ArchitectureNode, ...],
    edges: tuple[ArchitectureEdge, ...] = (),
) -> ArchitectureIR:
    return ArchitectureIR.from_parts(
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        nodes=nodes,
        edges=edges,
    )


def _kinds(report: DuplicateAuthorityReport) -> set[CollisionKind]:
    return set(report.detected_kinds)


def _provider_collision_graph(*, adapt: bool = False) -> ArchitectureIR:
    nodes = (
        _node("n-cap-a", NodeKind.AUTHORITY, _CAP_PATH, 10),
        _node("n-cap-b", NodeKind.AUTHORITY, _CAP_PATH, 40),
        _node("n-provider", NodeKind.PROVIDER, _CAP_PATH, 80),
    )
    edges = [
        _edge("e-a", EdgeKind.AUTHORIZES, "n-cap-a", "n-provider", _CAP_PATH, 10),
        _edge("e-b", EdgeKind.AUTHORIZES, "n-cap-b", "n-provider", _CAP_PATH, 40),
    ]
    if adapt:
        edges.append(
            _edge("e-adapt", EdgeKind.ADAPTS, "n-cap-b", "n-cap-a", _CAP_PATH, 40)
        )
    return _graph(nodes, tuple(edges))


def _receipt_collision_graph() -> ArchitectureIR:
    return _graph(
        (
            _node("n-rec-a", NodeKind.AUTHORITY, _RECEIPT_PATH, 12),
            _node("n-rec-b", NodeKind.AUTHORITY, _RECEIPT_PATH, 44),
            _node("n-receipt", NodeKind.RECEIPT, _RECEIPT_PATH, 90),
        ),
        (
            _edge(
                "e-rec-a",
                EdgeKind.CONFIRMS,
                "n-rec-a",
                "n-receipt",
                _RECEIPT_PATH,
                12,
            ),
            _edge(
                "e-rec-b",
                EdgeKind.CONFIRMS,
                "n-rec-b",
                "n-receipt",
                _RECEIPT_PATH,
                44,
            ),
        ),
    )


def _state_collision_graph() -> ArchitectureIR:
    return _graph(
        (
            _node("n-st-a", NodeKind.AUTHORITY, _STATE_PATH, 20),
            _node("n-st-b", NodeKind.AUTHORITY, _STATE_PATH, 60),
            _node("n-state", NodeKind.STATE, _STATE_PATH, 100),
        ),
        (
            _edge("e-st-a", EdgeKind.PERSISTS, "n-st-a", "n-state", _STATE_PATH, 20),
            _edge("e-st-b", EdgeKind.PERSISTS, "n-st-b", "n-state", _STATE_PATH, 60),
        ),
    )


def _compatibility_bypass_graph(*, adapt: bool = False) -> ArchitectureIR:
    nodes = (
        _node("n-compat", NodeKind.COMPATIBILITY, _COMPAT_PATH, 8),
        _node("n-prod", NodeKind.AUTHORITY, _AUTH_PATH, 20),
    )
    edges = [
        _edge("e-bypass", EdgeKind.AUTHORIZES, "n-compat", "n-prod", _COMPAT_PATH, 8),
    ]
    if adapt:
        edges.append(
            _edge("e-adapt", EdgeKind.ADAPTS, "n-compat", "n-prod", _COMPAT_PATH, 8)
        )
    return _graph(nodes, tuple(edges))


def _control_bypass_graph() -> ArchitectureIR:
    return _graph(
        (
            _node("n-policy", NodeKind.POLICY, _POLICY_PATH, 12),
            _node("n-subject", NodeKind.OPERATION, _OP_PATH, 40),
            _node("n-tool", NodeKind.SYMBOL, _CLI_PATH, 18),
        ),
        (
            _edge(
                "e-policy",
                EdgeKind.EVALUATES_POLICY,
                "n-policy",
                "n-subject",
                _POLICY_PATH,
                12,
            ),
            _edge("e-tool", EdgeKind.EXECUTES, "n-tool", "n-subject", _CLI_PATH, 18),
        ),
    )


def _simulation_flow_graph(*, quarantined: bool = False) -> ArchitectureIR:
    nodes = (
        _node("n-sim", NodeKind.SIMULATION, _SIM_PATH, 14),
        _node("n-live", NodeKind.AUTHORITY, _AUTH_PATH, 30),
    )
    if quarantined:
        edges = (
            _edge(
                "e-fallback",
                EdgeKind.FALLBACKS_TO,
                "n-live",
                "n-sim",
                _SIM_PATH,
                14,
            ),
        )
    else:
        edges = (
            _edge("e-flow", EdgeKind.CONFIRMS, "n-sim", "n-live", _SIM_PATH, 14),
        )
    return _graph(nodes, edges)


def _surface_divergence_graph(*, complete: bool = False) -> ArchitectureIR:
    nodes = [
        _node("n-op", NodeKind.OPERATION, _OP_PATH, 10),
        _node("n-py", NodeKind.ENTRYPOINT, _PY_PATH, 12),
        _node("n-cli", NodeKind.ENTRYPOINT, _CLI_PATH, 14),
    ]
    edges = [
        _edge("e-py", EdgeKind.IMPLEMENTS, "n-py", "n-op", _PY_PATH, 12),
        _edge("e-cli", EdgeKind.IMPLEMENTS, "n-cli", "n-op", _CLI_PATH, 14),
    ]
    if complete:
        nodes.append(_node("n-mcp", NodeKind.ENTRYPOINT, _MCP_PATH, 16))
        edges.append(
            _edge("e-mcp", EdgeKind.IMPLEMENTS, "n-mcp", "n-op", _MCP_PATH, 16)
        )
    return _graph(tuple(nodes), tuple(edges))


def _reexport_graph(*, claims_authority: bool = True) -> ArchitectureIR:
    nodes = (
        _node("n-reexport", NodeKind.AUTHORITY, _OP_PATH, 8),
        _node("n-real", NodeKind.AUTHORITY, _OP_PATH, 40),
        _node("n-subject", NodeKind.SYMBOL, _OP_PATH, 60),
    )
    edges = [
        _edge("e-re", EdgeKind.REEXPORTS, "n-reexport", "n-real", _OP_PATH, 8),
    ]
    if claims_authority:
        edges.append(
            _edge(
                "e-auth",
                EdgeKind.AUTHORIZES,
                "n-reexport",
                "n-subject",
                _OP_PATH,
                8,
            )
        )
    return _graph(nodes, tuple(edges))


def _obsolete_test_graph(*, tests_canonical: bool = False) -> ArchitectureIR:
    nodes = (
        _node("n-test", NodeKind.TEST, _TEST_PATH, 20),
        _node("n-legacy", NodeKind.COMPATIBILITY, _COMPAT_PATH, 40),
        _node("n-canon", NodeKind.AUTHORITY, _AUTH_PATH, 12),
    )
    edges = [
        _edge("e-super", EdgeKind.SUPERSEDES, "n-canon", "n-legacy", _AUTH_PATH, 12),
        _edge("e-test-legacy", EdgeKind.TESTS, "n-test", "n-legacy", _TEST_PATH, 20),
    ]
    if tests_canonical:
        edges.append(
            _edge("e-test-canon", EdgeKind.TESTS, "n-test", "n-canon", _TEST_PATH, 20)
        )
    return _graph(nodes, tuple(edges))


def _required_detection_graph() -> ArchitectureIR:
    return _graph(
        (
            _node("n-cap-a", NodeKind.AUTHORITY, _CAP_PATH, 10),
            _node("n-cap-b", NodeKind.AUTHORITY, _CAP_PATH, 40),
            _node("n-provider", NodeKind.PROVIDER, _CAP_PATH, 80),
            _node("n-rec-a", NodeKind.AUTHORITY, _RECEIPT_PATH, 12),
            _node("n-rec-b", NodeKind.AUTHORITY, _RECEIPT_PATH, 44),
            _node("n-receipt", NodeKind.RECEIPT, _RECEIPT_PATH, 90),
            _node("n-st-a", NodeKind.AUTHORITY, _STATE_PATH, 20),
            _node("n-st-b", NodeKind.AUTHORITY, _STATE_PATH, 60),
            _node("n-state", NodeKind.STATE, _STATE_PATH, 100),
            _node("n-compat", NodeKind.COMPATIBILITY, _COMPAT_PATH, 8),
            _node("n-prod", NodeKind.AUTHORITY, _AUTH_PATH, 20),
            _node("n-policy", NodeKind.POLICY, _POLICY_PATH, 12),
            _node("n-subject", NodeKind.OPERATION, _OP_PATH, 200),
            _node("n-tool", NodeKind.SYMBOL, _CLI_PATH, 18),
            _node("n-sim", NodeKind.SIMULATION, _SIM_PATH, 14),
            _node("n-live", NodeKind.AUTHORITY, _AUTH_PATH, 300),
            _node("n-op", NodeKind.OPERATION, _OP_PATH, 10),
            _node("n-py", NodeKind.ENTRYPOINT, _PY_PATH, 12),
            _node("n-cli", NodeKind.ENTRYPOINT, _CLI_PATH, 14),
            _node("n-reexport", NodeKind.AUTHORITY, _OP_PATH, 8),
            _node("n-real", NodeKind.AUTHORITY, _OP_PATH, 40),
            _node("n-re-subject", NodeKind.SYMBOL, _OP_PATH, 60),
            _node("n-test", NodeKind.TEST, _TEST_PATH, 20),
            _node("n-legacy", NodeKind.COMPATIBILITY, _COMPAT_PATH, 40),
            _node("n-canon", NodeKind.AUTHORITY, _AUTH_PATH, 12),
        ),
        (
            _edge("e-a", EdgeKind.AUTHORIZES, "n-cap-a", "n-provider", _CAP_PATH, 10),
            _edge("e-b", EdgeKind.AUTHORIZES, "n-cap-b", "n-provider", _CAP_PATH, 40),
            _edge(
                "e-rec-a",
                EdgeKind.CONFIRMS,
                "n-rec-a",
                "n-receipt",
                _RECEIPT_PATH,
                12,
            ),
            _edge(
                "e-rec-b",
                EdgeKind.CONFIRMS,
                "n-rec-b",
                "n-receipt",
                _RECEIPT_PATH,
                44,
            ),
            _edge("e-st-a", EdgeKind.PERSISTS, "n-st-a", "n-state", _STATE_PATH, 20),
            _edge("e-st-b", EdgeKind.PERSISTS, "n-st-b", "n-state", _STATE_PATH, 60),
            _edge(
                "e-bypass", EdgeKind.AUTHORIZES, "n-compat", "n-prod", _COMPAT_PATH, 8
            ),
            _edge(
                "e-policy",
                EdgeKind.EVALUATES_POLICY,
                "n-policy",
                "n-subject",
                _POLICY_PATH,
                12,
            ),
            _edge("e-tool", EdgeKind.EXECUTES, "n-tool", "n-subject", _CLI_PATH, 18),
            _edge("e-flow", EdgeKind.CONFIRMS, "n-sim", "n-live", _SIM_PATH, 14),
            _edge("e-py", EdgeKind.IMPLEMENTS, "n-py", "n-op", _PY_PATH, 12),
            _edge("e-cli", EdgeKind.IMPLEMENTS, "n-cli", "n-op", _CLI_PATH, 14),
            _edge("e-re", EdgeKind.REEXPORTS, "n-reexport", "n-real", _OP_PATH, 8),
            _edge(
                "e-auth",
                EdgeKind.AUTHORIZES,
                "n-reexport",
                "n-re-subject",
                _OP_PATH,
                8,
            ),
            _edge(
                "e-super", EdgeKind.SUPERSEDES, "n-canon", "n-legacy", _AUTH_PATH, 12
            ),
            _edge(
                "e-test-legacy", EdgeKind.TESTS, "n-test", "n-legacy", _TEST_PATH, 20
            ),
        ),
    )


def test_closed_vocabulary_and_authority_invariants() -> None:
    assert DUPLICATE_AUTHORITY_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/duplicate-authority-finding@1"
    )
    assert DUPLICATE_AUTHORITY_VERSION == 1
    assert DUPLICATE_AUTHORITY_EVIDENCE == "pcar/duplicate-authority-finding@1"
    assert EXTRACTOR_IDENTITY == "pcar-007-duplicate-authority-detector"
    assert TASK_ID == "PCAR-007"
    assert DEFAULT_FRESHNESS == "pcar-007-duplicate-authority"
    assert EFFECT_CLASS == "read_only_analysis"
    assert DETECTOR_CAN_AUTHORIZE_CHANGES is False
    assert DETECTOR_CAN_REMEDIATE is False
    assert DETECTOR_CAN_SELECT_CANONICAL is False
    assert CONTENT_IDENTITY_IS_NOT_AUTHORITY is True
    assert REEXPORT_IS_NOT_AUTHORITY is True
    assert SILENT_ARBITRATION_PROHIBITED is True
    assert ONE_OWNER_INVARIANT is True
    assert HEURISTIC_CRITICAL_PROMOTION_PROHIBITED is True
    assert tuple(item.value for item in REQUIRED_COLLISION_KINDS) == (
        "independent_provider_capability",
        "independent_receipt_decision",
        "competing_state_owner",
        "compatibility_bypass",
        "control_bypass",
        "simulation_to_production_flow",
        "python_cli_mcp_divergence",
        "reexport_authority",
        "obsolete_authority_test",
    )
    assert CLOSED_COLLISION_KINDS == {item.value for item in CollisionKind}
    assert CLOSED_COLLISION_DISPOSITIONS == {
        "collision",
        "formally_arbitrated",
        "false_positive",
        "unknown",
    }
    assert CLOSED_DUPLICATE_AUTHORITY_BLOCKERS == {
        item.value for item in DuplicateAuthorityBlockerKind
    }
    assert CLOSED_SURFACES == {"python", "cli", "mcp"}
    assert tuple(item.value for item in REQUIRED_SURFACES) == ("python", "cli", "mcp")
    assert COLLISION_SCHEMA.endswith("authority-collision@1")
    assert BYPASS_SCHEMA.endswith("authority-bypass-finding@1")
    assert SURFACE_SCHEMA.endswith("surface-divergence-finding@1")
    with pytest.raises(ValueError):
        CollisionKind("aesthetic score")
    with pytest.raises(ValueError):
        CollisionDisposition("ignore")
    with pytest.raises(ValueError):
        DuplicateAuthorityBlockerKind("warn")
    with pytest.raises(ValueError):
        SurfaceKind("http")


def test_required_detections_are_all_emitted() -> None:
    report = detect_duplicate_authorities(_required_detection_graph())
    assert set(REQUIRED_COLLISION_KINDS) <= _kinds(report)
    assert report.fails_closed is True
    assert report.one_owner_invariant is False
    for kind in REQUIRED_COLLISION_KINDS:
        if kind in {
            CollisionKind.COMPATIBILITY_BYPASS,
            CollisionKind.CONTROL_BYPASS,
        }:
            assert any(item.kind is kind for item in report.bypasses)
        elif kind is CollisionKind.PYTHON_CLI_MCP_DIVERGENCE:
            assert report.surface_divergences
            assert all(
                item.kind is CollisionKind.PYTHON_CLI_MCP_DIVERGENCE
                for item in report.surface_divergences
            )
        else:
            assert report.collisions_of(kind)
        for finding in report.collisions_of(kind):
            assert finding.provenance.span.path
            assert finding.reachability
            assert finding.disposition is CollisionDisposition.COLLISION


def test_independent_provider_capability_is_detected() -> None:
    report = detect_duplicate_authorities(_provider_collision_graph())
    findings = report.collisions_of(CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY)
    assert findings
    finding = findings[0]
    assert set(finding.node_ids) >= {"n-cap-a", "n-cap-b", "n-provider"}
    assert finding.concern is ConcernKind.PROVIDER_CAPABILITY
    assert "e-a" in finding.edge_ids and "e-b" in finding.edge_ids
    assert finding.provenance.repository_tree == _TREE
    assert any(
        item.kind is DuplicateAuthorityBlockerKind.MULTIPLE_PRODUCTION_AUTHORITIES
        for item in report.blockers
    )


def test_independent_receipt_decision_is_detected() -> None:
    report = detect_duplicate_authorities(_receipt_collision_graph())
    findings = report.collisions_of(CollisionKind.INDEPENDENT_RECEIPT_DECISION)
    assert findings
    assert set(findings[0].node_ids) >= {"n-rec-a", "n-rec-b", "n-receipt"}
    assert findings[0].concern is ConcernKind.COMPLETION_EVIDENCE


def test_competing_state_owners_are_detected() -> None:
    report = detect_duplicate_authorities(_state_collision_graph())
    findings = report.collisions_of(CollisionKind.COMPETING_STATE_OWNER)
    assert findings
    assert set(findings[0].node_ids) >= {"n-st-a", "n-st-b", "n-state"}
    assert findings[0].concern is ConcernKind.STATE_PERSISTENCE


def test_compatibility_and_control_bypasses_are_detected() -> None:
    compat = detect_duplicate_authorities(_compatibility_bypass_graph())
    assert any(item.kind is CollisionKind.COMPATIBILITY_BYPASS for item in compat.bypasses)
    bypass = compat.bypasses[0]
    assert bypass.bypass_node_id == "n-compat"
    assert bypass.production_node_id == "n-prod"
    assert bypass.reachability
    control = detect_duplicate_authorities(_control_bypass_graph())
    assert any(item.kind is CollisionKind.CONTROL_BYPASS for item in control.bypasses)
    assert control.bypasses[0].bypass_node_id == "n-tool"


def test_simulation_to_production_flow_is_detected_with_reachability() -> None:
    report = detect_duplicate_authorities(_simulation_flow_graph())
    findings = report.collisions_of(CollisionKind.SIMULATION_TO_PRODUCTION_FLOW)
    assert findings
    finding = findings[0]
    assert "n-sim" in finding.node_ids
    assert "n-live" in finding.node_ids
    assert "n-sim" in finding.reachability
    assert "n-live" in finding.reachability
    assert finding.edge_ids


def test_python_cli_mcp_divergence_is_detected() -> None:
    report = detect_duplicate_authorities(_surface_divergence_graph())
    assert report.surface_divergences
    finding = report.surface_divergences[0]
    assert finding.operation_node_id == "n-op"
    assert SurfaceKind.PYTHON in finding.present_surfaces
    assert SurfaceKind.CLI in finding.present_surfaces
    assert SurfaceKind.MCP in finding.missing_surfaces
    assert finding.kind is CollisionKind.PYTHON_CLI_MCP_DIVERGENCE
    complete = detect_duplicate_authorities(_surface_divergence_graph(complete=True))
    assert complete.surface_divergences == ()
    assert complete.collisions_of(CollisionKind.PYTHON_CLI_MCP_DIVERGENCE) == ()


def test_reexport_authority_and_obsolete_tests_are_detected() -> None:
    reexport = detect_duplicate_authorities(_reexport_graph())
    findings = reexport.collisions_of(CollisionKind.REEXPORT_AUTHORITY)
    assert findings
    assert "n-reexport" in findings[0].node_ids
    obsolete = detect_duplicate_authorities(_obsolete_test_graph())
    tests = obsolete.collisions_of(CollisionKind.OBSOLETE_AUTHORITY_TEST)
    assert tests
    assert "n-test" in tests[0].node_ids
    assert "n-legacy" in tests[0].node_ids


def test_formal_arbitration_is_not_a_collision() -> None:
    architecture = _provider_collision_graph()
    colliding = detect_duplicate_authorities(architecture)
    assert colliding.collisions_of(CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY)
    arbitration = FormalArbitration(
        concern=ConcernKind.PROVIDER_CAPABILITY,
        canonical_owner_node_id="n-cap-a",
        loser_classifications=(
            LoserClassification("n-cap-b", OwnerDisposition.ADAPTER),
        ),
        arbitrator_identity="pcar-007-provider-capability-arbitration",
        rationale=ArbitrationRationale.EXPLICIT_REVIEWED_CONTRACT,
        provenance=_fact(_CAP_PATH, 10),
        evidence_node_ids=("n-cap-a", "n-cap-b"),
        evidence_edge_ids=("e-a", "e-b"),
    )
    report = detect_duplicate_authorities(architecture, arbitrations=(arbitration,))
    assert report.collisions_of(CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY) == ()
    assert any(
        item.disposition is CollisionDisposition.FORMALLY_ARBITRATED
        and item.kind is CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY
        for item in report.rejected
    )
    assert all(
        item.kind is not CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY
        for item in report.collisions
    )


def test_adapter_projection_legacy_and_simulation_are_false_positives() -> None:
    adapted = detect_duplicate_authorities(_provider_collision_graph(adapt=True))
    assert adapted.collisions_of(CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY) == ()
    assert any(
        item.disposition is CollisionDisposition.FALSE_POSITIVE
        for item in adapted.rejected
    )
    compat_adapter = detect_duplicate_authorities(
        _compatibility_bypass_graph(adapt=True)
    )
    assert compat_adapter.bypasses == ()
    assert all(
        item.kind is not CollisionKind.COMPATIBILITY_BYPASS
        for item in compat_adapter.collisions
    )
    quarantined = detect_duplicate_authorities(
        _simulation_flow_graph(quarantined=True)
    )
    assert quarantined.collisions_of(CollisionKind.SIMULATION_TO_PRODUCTION_FLOW) == ()
    projection = detect_duplicate_authorities(
        _graph(
            (
                _node("n-auth", NodeKind.AUTHORITY, _OP_PATH, 10),
                _node("n-proj", NodeKind.GENERATED, _CLI_PATH, 12),
                _node("n-provider", NodeKind.PROVIDER, _CAP_PATH, 20),
            ),
            (
                _edge(
                    "e-auth",
                    EdgeKind.AUTHORIZES,
                    "n-auth",
                    "n-provider",
                    _OP_PATH,
                    10,
                ),
                _edge(
                    "e-proj",
                    EdgeKind.AUTHORIZES,
                    "n-proj",
                    "n-provider",
                    _CLI_PATH,
                    12,
                ),
                _edge(
                    "e-gen", EdgeKind.GENERATES, "n-auth", "n-proj", _CLI_PATH, 12
                ),
            ),
        )
    )
    assert projection.collisions_of(CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY) == ()
    reexport_only = detect_duplicate_authorities(_reexport_graph(claims_authority=False))
    assert reexport_only.collisions_of(CollisionKind.REEXPORT_AUTHORITY) == ()
    current_test = detect_duplicate_authorities(
        _obsolete_test_graph(tests_canonical=True)
    )
    assert current_test.collisions_of(CollisionKind.OBSOLETE_AUTHORITY_TEST) == ()


def test_unknown_production_ownership_emits_a_blocker() -> None:
    architecture = _graph(
        (
            _node("n-unknown", NodeKind.AUTHORITY, _AUTH_PATH, 15),
            _node("n-receipt", NodeKind.RECEIPT, _RECEIPT_PATH, 40),
        ),
        (
            _edge(
                "e-unknown",
                EdgeKind.AUTHORIZES,
                "n-unknown",
                "n-receipt",
                _AUTH_PATH,
                15,
            ),
        ),
    )
    ownership = resolve_authority_ownership(
        architecture,
        (
            {
                "concern": ConcernKind.AUTHORIZATION.value,
                "owner_node_id": "n-unknown",
                "disposition": OwnerDisposition.UNKNOWN.value,
                "evidence_edge_ids": ["e-unknown"],
            },
        ),
    )
    report = detect_duplicate_authorities(architecture, ownership)
    assert report.fails_closed is True
    assert any(
        item.kind
        in {
            DuplicateAuthorityBlockerKind.UNKNOWN_PRODUCTION_OWNER,
            DuplicateAuthorityBlockerKind.UNKNOWN_OWNER,
        }
        and "n-unknown" in item.node_ids
        for item in report.blockers
    )


def test_heuristic_evidence_is_not_promoted_to_a_collision() -> None:
    architecture = _graph(
        (
            _node(
                "n-cap-a",
                NodeKind.AUTHORITY,
                _CAP_PATH,
                10,
                confidence=Confidence.HEURISTIC,
            ),
            _node(
                "n-cap-b",
                NodeKind.AUTHORITY,
                _CAP_PATH,
                40,
                confidence=Confidence.HEURISTIC,
            ),
            _node(
                "n-provider",
                NodeKind.PROVIDER,
                _CAP_PATH,
                80,
                confidence=Confidence.HEURISTIC,
            ),
        ),
        (
            _edge(
                "e-a",
                EdgeKind.AUTHORIZES,
                "n-cap-a",
                "n-provider",
                _CAP_PATH,
                10,
                confidence=Confidence.HEURISTIC,
            ),
            _edge(
                "e-b",
                EdgeKind.AUTHORIZES,
                "n-cap-b",
                "n-provider",
                _CAP_PATH,
                40,
                confidence=Confidence.HEURISTIC,
            ),
        ),
    )
    report = detect_duplicate_authorities(architecture)
    assert report.collisions_of(CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY) == ()
    assert any(
        item.disposition is CollisionDisposition.UNKNOWN
        and item.kind is CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY
        for item in report.unknown
    )


def test_empty_graph_has_no_collisions_or_false_positives() -> None:
    report = detect_duplicate_authorities(_graph((), ()))
    assert report.collisions == ()
    assert report.bypasses == ()
    assert report.surface_divergences == ()
    assert report.blockers == ()
    assert report.rejected == ()
    assert report.unknown == ()
    assert report.fails_closed is False
    assert report.one_owner_invariant is True
    assert report.detected_kinds == frozenset()


def test_detector_cannot_remediate_authorize_or_select() -> None:
    detector = DuplicateAuthorityDetector()
    architecture = _provider_collision_graph()
    report = detector.detect(architecture)
    assert report == detect_duplicate_authorities(architecture)
    assert build_duplicate_authority_report(architecture) == report
    with pytest.raises(DuplicateAuthorityAuthorityError, match="cannot authorize"):
        detector.authorize_change("refactor")
    with pytest.raises(DuplicateAuthorityAuthorityError, match="cannot execute"):
        detector.remediate("consolidate")
    with pytest.raises(DuplicateAuthorityAuthorityError, match="cannot select"):
        detector.select_canonical()
    with pytest.raises(DuplicateAuthorityAuthorityError, match="cannot authorize"):
        report.authorize_change("promotion")
    with pytest.raises(DuplicateAuthorityAuthorityError, match="cannot execute"):
        report.remediate("delete")
    with pytest.raises(DuplicateAuthorityAuthorityError, match="cannot select"):
        report.select_canonical("owner")
    with pytest.raises(DuplicateAuthorityAuthorityError, match="cannot authorize"):
        refuse_authorization("change")
    with pytest.raises(DuplicateAuthorityAuthorityError, match="cannot execute"):
        refuse_remediation("remediation")
    with pytest.raises(DuplicateAuthorityAuthorityError, match="cannot select"):
        refuse_canonical_selection("canonical owner")
    with pytest.raises(DuplicateAuthorityError, match="content identity"):
        lookup_owner_by_content_identity("baguqeera" + ("a" * 50))
    with pytest.raises(DuplicateAuthorityError, match="silent arbitration"):
        silently_select_canonical(("n-a", "n-b"))
    with pytest.raises(DuplicateAuthorityError, match="cannot authorize"):
        DuplicateAuthorityDetector(can_authorize_changes=True)
    with pytest.raises(DuplicateAuthorityError, match="cannot remediate"):
        DuplicateAuthorityDetector(can_remediate=True)
    with pytest.raises(DuplicateAuthorityError, match="cannot select"):
        DuplicateAuthorityDetector(can_select_canonical=True)


def test_round_trip_and_canonical_identity() -> None:
    report = detect_duplicate_authorities(_required_detection_graph())
    payload = report.to_dict()
    restored = DuplicateAuthorityReport.from_mapping(payload)
    assert restored == report
    assert restored.to_dict() == payload
    assert restored.to_json() == json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert DuplicateAuthorityReport.from_json(restored.to_json()) == report
    claimed = payload.pop("content_identity")
    validate_cid(claimed, codecs=("dag-json",))
    assert claimed == cid_for_dag_json(payload)
    assert claimed == report.content_identity
    assert not claimed.startswith("sha256:")
    reversed_graph = ArchitectureIR.from_parts(
        repository_tree=_TREE,
        freshness=_FRESHNESS,
        nodes=tuple(reversed(_required_detection_graph().nodes)),
        edges=tuple(reversed(_required_detection_graph().edges)),
    )
    reversed_report = detect_duplicate_authorities(reversed_graph)
    assert reversed_report.content_identity == report.content_identity
    assert reversed_report.to_dict() == report.to_dict()


def test_unknown_fields_and_identity_mismatch_are_rejected() -> None:
    payload = detect_duplicate_authorities(_provider_collision_graph()).to_dict()
    unknown = dict(payload)
    unknown["hidden"] = True
    identity_payload = {
        key: value for key, value in unknown.items() if key != "content_identity"
    }
    unknown["content_identity"] = cid_for_dag_json(identity_payload)
    with pytest.raises(DuplicateAuthorityError, match="unknown duplicate-authority field"):
        DuplicateAuthorityReport.from_mapping(unknown)
    missing = {key: value for key, value in payload.items() if key != "freshness"}
    with pytest.raises(DuplicateAuthorityError, match="missing duplicate-authority field"):
        DuplicateAuthorityReport.from_mapping(missing)
    forged = dict(payload)
    forged["content_identity"] = "sha256:" + ("00" * 32)
    with pytest.raises(DuplicateAuthorityError, match="content identity mismatch"):
        DuplicateAuthorityReport.from_mapping(forged)
    schema = dict(payload)
    schema["schema"] = schema["schema"] + "-extra"
    identity_payload = {
        key: value for key, value in schema.items() if key != "content_identity"
    }
    schema["content_identity"] = cid_for_dag_json(identity_payload)
    with pytest.raises(DuplicateAuthorityError, match="unexpected duplicate-authority schema"):
        DuplicateAuthorityReport.from_mapping(schema)
    authorize = dict(payload)
    authorize["can_authorize_changes"] = True
    identity_payload = {
        key: value for key, value in authorize.items() if key != "content_identity"
    }
    authorize["content_identity"] = cid_for_dag_json(identity_payload)
    with pytest.raises(DuplicateAuthorityError, match="cannot authorize"):
        DuplicateAuthorityReport.from_mapping(authorize)


def test_collision_records_reject_content_identity_owners_and_non_probative_critical() -> None:
    provenance = _fact(_CAP_PATH, 10)
    with pytest.raises(DuplicateAuthorityError, match="content identity"):
        AuthorityCollision(
            kind=CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY,
            concern=ConcernKind.PROVIDER_CAPABILITY,
            disposition=CollisionDisposition.COLLISION,
            message="duplicate",
            node_ids=(
                "baguqeeraaf2trsznudx7wxyyocgpkaoqf5smketqgebile3aobqxbcvdddbq",
            ),
            provenance=provenance,
        )
    with pytest.raises(DuplicateAuthorityError, match="cannot prove a critical"):
        AuthorityCollision(
            kind=CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY,
            concern=ConcernKind.PROVIDER_CAPABILITY,
            disposition=CollisionDisposition.COLLISION,
            message="duplicate",
            node_ids=("n-a", "n-b"),
            provenance=_fact(_CAP_PATH, 10, confidence=Confidence.HEURISTIC),
            confidence=Confidence.HEURISTIC,
        )
    with pytest.raises(DuplicateAuthorityError, match="bypass kind"):
        BypassFinding(
            kind=CollisionKind.COMPETING_STATE_OWNER,
            bypass_node_id="n-a",
            production_node_id="n-b",
            disposition=CollisionDisposition.COLLISION,
            message="nope",
            provenance=provenance,
        )
    with pytest.raises(DuplicateAuthorityError, match="python_cli_mcp_divergence"):
        SurfaceDivergenceFinding(
            operation_node_id="n-op",
            present_surfaces=(SurfaceKind.PYTHON,),
            missing_surfaces=(SurfaceKind.CLI, SurfaceKind.MCP),
            disposition=CollisionDisposition.COLLISION,
            message="nope",
            provenance=provenance,
            kind=CollisionKind.CONTROL_BYPASS,
        )
