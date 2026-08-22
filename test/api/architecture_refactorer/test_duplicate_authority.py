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
    INITIAL_CONCERNS,
    INITIAL_CONCERN_SOURCE_BINDINGS,
    ArbitrationRationale,
    ConcernClaim,
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
    BLOCKER_KINDS,
    BYPASS_KINDS,
    BypassFinding,
    CLOSED_COLLISION_KINDS,
    CLOSED_FINDING_DISPOSITIONS,
    CLOSED_SURFACES,
    COLLISION_SCHEMA,
    COLLISION_VERSION,
    CONTENT_IDENTITY_IS_NOT_AUTHORITY,
    DEFAULT_FRESHNESS,
    DETECTOR_CAN_AUTHORIZE_CHANGES,
    DETECTOR_CAN_REMEDIATE,
    DETECTOR_CAN_SELECT_OWNER,
    DUPLICATE_AUTHORITY_EVIDENCE,
    DUPLICATE_AUTHORITY_SCHEMA,
    DUPLICATE_AUTHORITY_VERSION,
    EFFECT_CLASS,
    EXTRACTOR_IDENTITY,
    HEURISTIC_CRITICAL_PROMOTION_PROHIBITED,
    REEXPORT_IS_NOT_AUTHORITY,
    REQUIRED_DETECTIONS,
    REQUIRED_SURFACES,
    SILENT_ARBITRATION_PROHIBITED,
    SurfaceDivergenceFinding,
    TASK_ID,
    UNKNOWN_PRODUCTION_OWNER_BLOCKS,
    AuthorityCollision,
    CollisionKind,
    DuplicateAuthorityAuthorityError,
    DuplicateAuthorityDetector,
    DuplicateAuthorityError,
    DuplicateAuthorityReport,
    FindingDisposition,
    SurfaceKind,
    build_duplicate_authority_report,
    detect_duplicate_authorities,
    lookup_owner_by_content_identity,
    recognize_formal_arbitration,
    refuse_heuristic_promotion,
    refuse_owner_selection,
    refuse_remediation,
)
from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json, validate_cid

_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
_FRESHNESS = "pcar-007-fixture"
_EXTRACTOR = "pcar-007-fixture"
_PY_PATH = "ipfs_accelerate_py/agent_supervisor/runtime/operations.py"
_CLI_PATH = "ipfs_accelerate_py/agent_supervisor/entrypoints/cli.py"
_MCP_PATH = "ipfs_accelerate_py/mcp_server/tools.py"
_COMPAT_PATH = "ipfs_accelerate_py/agent_supervisor/compat/legacy_dispatch.py"
_SIM_PATH = "ipfs_accelerate_py/agent_supervisor/runtime/provider_usage.py"
_STATE_PATH = "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py"
_TEST_PATH = "test/api/architecture_refactorer/test_legacy_authority.py"
_CTRL_PATH = "ipfs_accelerate_py/agent_supervisor/control/control_plane.py"


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


def _detect(
    architecture: ArchitectureIR,
    ownership=None,
    *,
    claims=None,
    arbitrations=None,
) -> DuplicateAuthorityReport:
    return detect_duplicate_authorities(
        architecture,
        ownership,
        claims=claims,
        arbitrations=arbitrations,
    )


def test_closed_collision_vocabulary() -> None:
    assert DUPLICATE_AUTHORITY_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/duplicate-authority-report@1"
    )
    assert DUPLICATE_AUTHORITY_VERSION == 1
    assert DUPLICATE_AUTHORITY_EVIDENCE == "pcar/duplicate-authority-finding@1"
    assert COLLISION_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/duplicate-authority-finding@1"
    )
    assert COLLISION_VERSION == 1
    assert EXTRACTOR_IDENTITY == "pcar-007-duplicate-authority"
    assert TASK_ID == "PCAR-007"
    assert DEFAULT_FRESHNESS == "pcar-007-duplicate-authority"
    assert EFFECT_CLASS == "read_only_analysis"
    assert DETECTOR_CAN_AUTHORIZE_CHANGES is False
    assert DETECTOR_CAN_SELECT_OWNER is False
    assert DETECTOR_CAN_REMEDIATE is False
    assert HEURISTIC_CRITICAL_PROMOTION_PROHIBITED is True
    assert CONTENT_IDENTITY_IS_NOT_AUTHORITY is True
    assert REEXPORT_IS_NOT_AUTHORITY is True
    assert SILENT_ARBITRATION_PROHIBITED is True
    assert UNKNOWN_PRODUCTION_OWNER_BLOCKS is True
    assert tuple(item.value for item in REQUIRED_DETECTIONS) == (
        "independent_provider_capability",
        "independent_receipt_producer",
        "competing_state_owner",
        "compatibility_bypass",
        "control_bypass",
        "simulation_to_production_flow",
        "python_cli_mcp_divergence",
        "reexport_authority",
        "obsolete_authority_test",
    )
    assert CLOSED_COLLISION_KINDS == {item.value for item in CollisionKind}
    assert set(REQUIRED_DETECTIONS) <= set(CollisionKind)
    assert CLOSED_FINDING_DISPOSITIONS == {
        "collision",
        "false_positive",
        "unknown",
        "blocker",
    }
    assert CLOSED_SURFACES == {"python", "cli", "mcp", "unknown"}
    assert REQUIRED_SURFACES == (SurfaceKind.PYTHON, SurfaceKind.CLI, SurfaceKind.MCP)
    assert CollisionKind.COMPATIBILITY_BYPASS in BYPASS_KINDS
    assert CollisionKind.CONTROL_BYPASS in BYPASS_KINDS
    assert CollisionKind.UNKNOWN_PRODUCTION_OWNER in BLOCKER_KINDS
    assert BypassFinding is AuthorityCollision
    assert SurfaceDivergenceFinding is AuthorityCollision
    with pytest.raises(ValueError):
        CollisionKind("aesthetic score")
    with pytest.raises(ValueError):
        FindingDisposition("ignore")
    with pytest.raises(ValueError):
        SurfaceKind("graphql")


def test_independent_provider_capability_is_detected() -> None:
    architecture = _graph(
        (
            _node("n-cap-a", NodeKind.PROVIDER, _PY_PATH, 10),
            _node("n-cap-b", NodeKind.PROVIDER, _PY_PATH, 40),
            _node("n-subject", NodeKind.OPERATION, _PY_PATH, 80),
        ),
        (
            _edge("e-a", EdgeKind.AUTHORIZES, "n-cap-a", "n-subject", _PY_PATH, 10),
            _edge("e-b", EdgeKind.AUTHORIZES, "n-cap-b", "n-subject", _PY_PATH, 40),
        ),
    )
    report = _detect(architecture)
    findings = report.findings_of(CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY)
    assert findings
    assert all(
        item.disposition is FindingDisposition.COLLISION for item in findings
    )
    assert {"n-cap-a", "n-cap-b", "n-subject"} <= set(findings[0].node_ids)
    assert report.one_owner_invariant_holds is False


def test_independent_receipt_producers_are_detected() -> None:
    architecture = _graph(
        (
            _node("n-r1", NodeKind.RECEIPT, _PY_PATH, 12),
            _node("n-r2", NodeKind.RECEIPT, _PY_PATH, 48),
            _node("n-op", NodeKind.OPERATION, _PY_PATH, 90),
        ),
        (
            _edge("e-r1", EdgeKind.CONFIRMS, "n-r1", "n-op", _PY_PATH, 12),
            _edge("e-r2", EdgeKind.CONFIRMS, "n-r2", "n-op", _PY_PATH, 48),
        ),
    )
    report = _detect(architecture)
    findings = report.findings_of(CollisionKind.INDEPENDENT_RECEIPT_PRODUCER)
    assert findings
    assert findings[0].disposition is FindingDisposition.COLLISION
    assert {"n-r1", "n-r2"} <= set(findings[0].node_ids)


def test_competing_state_owners_are_detected() -> None:
    architecture = _graph(
        (
            _node("n-store-a", NodeKind.STATE, _STATE_PATH, 20),
            _node("n-store-b", NodeKind.STATE, _STATE_PATH, 80),
            _node("n-fact", NodeKind.STATE, _STATE_PATH, 120),
        ),
        (
            _edge("e-a", EdgeKind.PERSISTS, "n-store-a", "n-fact", _STATE_PATH, 20),
            _edge("e-b", EdgeKind.WRITES, "n-store-b", "n-fact", _STATE_PATH, 80),
        ),
    )
    report = _detect(architecture)
    findings = report.findings_of(CollisionKind.COMPETING_STATE_OWNER)
    assert findings
    assert findings[0].disposition is FindingDisposition.COLLISION
    assert {"n-store-a", "n-store-b", "n-fact"} <= set(findings[0].node_ids)


def test_compatibility_bypass_is_detected() -> None:
    architecture = _graph(
        (
            _node("n-compat", NodeKind.COMPATIBILITY, _COMPAT_PATH, 8),
            _node("n-auth", NodeKind.AUTHORITY, _CTRL_PATH, 40),
            _node("n-op", NodeKind.OPERATION, _PY_PATH, 12),
        ),
        (
            _edge("e-auth", EdgeKind.AUTHORIZES, "n-auth", "n-op", _CTRL_PATH, 40),
            _edge("e-bypass", EdgeKind.EXECUTES, "n-compat", "n-op", _COMPAT_PATH, 8),
        ),
    )
    report = _detect(architecture)
    findings = report.findings_of(CollisionKind.COMPATIBILITY_BYPASS)
    collisions = [
        item for item in findings if item.disposition is FindingDisposition.BLOCKER
    ]
    assert collisions
    assert "n-compat" in collisions[0].node_ids
    assert collisions[0].is_bypass is True
    assert collisions[0] in report.bypass_findings


def test_control_bypass_is_detected() -> None:
    architecture = _graph(
        (
            _node("n-auth", NodeKind.AUTHORITY, _CTRL_PATH, 20),
            _node("n-tool", NodeKind.ENTRYPOINT, _PY_PATH, 88),
            _node("n-policy", NodeKind.POLICY, _CTRL_PATH, 60),
        ),
        (
            _edge(
                "e-canon", EdgeKind.AUTHORIZES, "n-auth", "n-policy", _CTRL_PATH, 20
            ),
            _edge(
                "e-bypass", EdgeKind.EVALUATES_POLICY, "n-tool", "n-policy", _PY_PATH, 88
            ),
        ),
    )
    report = _detect(architecture)
    findings = [
        item
        for item in report.findings_of(CollisionKind.CONTROL_BYPASS)
        if item.disposition is FindingDisposition.BLOCKER
    ]
    assert findings
    assert {"n-tool", "n-policy"} <= set(findings[0].node_ids)


def test_simulation_to_production_flow_is_detected() -> None:
    architecture = _graph(
        (
            _node("n-sim", NodeKind.SIMULATION, _SIM_PATH, 14),
            _node("n-op", NodeKind.OPERATION, _PY_PATH, 30),
            _node("n-state", NodeKind.STATE, _STATE_PATH, 90),
        ),
        (
            _edge("e-call", EdgeKind.CALLS, "n-sim", "n-op", _SIM_PATH, 14),
            _edge("e-write", EdgeKind.WRITES, "n-op", "n-state", _PY_PATH, 30),
        ),
    )
    report = _detect(architecture)
    findings = [
        item
        for item in report.findings_of(CollisionKind.SIMULATION_TO_PRODUCTION_FLOW)
        if item.disposition is FindingDisposition.BLOCKER
    ]
    assert findings
    flow = findings[0]
    assert flow.reachability_path
    assert set(flow.reachability_path) <= set(flow.edge_ids)
    assert "n-sim" in flow.node_ids
    assert "n-state" in flow.node_ids or "n-op" in flow.node_ids
    assert flow.provenance.span.path in {_SIM_PATH, _PY_PATH, _STATE_PATH}


def test_python_cli_mcp_divergence_is_detected() -> None:
    architecture = _graph(
        (
            _node("n-py", NodeKind.ENTRYPOINT, _PY_PATH, 10),
            _node("n-cli", NodeKind.ENTRYPOINT, _CLI_PATH, 16),
            _node("n-mcp", NodeKind.ENTRYPOINT, _MCP_PATH, 22),
            _node("n-op", NodeKind.OPERATION, _PY_PATH, 80),
            _node("n-auth-py", NodeKind.AUTHORITY, _PY_PATH, 40),
            _node("n-auth-cli", NodeKind.AUTHORITY, _CLI_PATH, 40),
        ),
        (
            _edge("e-py-op", EdgeKind.IMPLEMENTS, "n-py", "n-op", _PY_PATH, 10),
            _edge("e-cli-op", EdgeKind.IMPLEMENTS, "n-cli", "n-op", _CLI_PATH, 16),
            _edge("e-mcp-op", EdgeKind.IMPLEMENTS, "n-mcp", "n-op", _MCP_PATH, 22),
            _edge("e-py-auth", EdgeKind.AUTHORIZES, "n-py", "n-auth-py", _PY_PATH, 10),
            _edge(
                "e-cli-auth", EdgeKind.AUTHORIZES, "n-cli", "n-auth-cli", _CLI_PATH, 16
            ),
            _edge("e-mcp-auth", EdgeKind.AUTHORIZES, "n-mcp", "n-auth-cli", _MCP_PATH, 22),
        ),
    )
    report = _detect(architecture)
    findings = [
        item
        for item in report.findings_of(CollisionKind.PYTHON_CLI_MCP_DIVERGENCE)
        if item.disposition is FindingDisposition.COLLISION
    ]
    assert findings
    assert findings[0].is_surface_divergence is True
    assert findings[0] in report.surface_divergence_findings
    assert {SurfaceKind.PYTHON, SurfaceKind.CLI, SurfaceKind.MCP} <= set(
        findings[0].surfaces
    )


def test_reexport_authority_is_detected() -> None:
    architecture = _graph(
        (
            _node("n-reexport", NodeKind.AUTHORITY, _PY_PATH, 4),
            _node("n-real", NodeKind.AUTHORITY, _PY_PATH, 40),
        ),
        (
            _edge(
                "e-reexport",
                EdgeKind.REEXPORTS,
                "n-reexport",
                "n-real",
                _PY_PATH,
                4,
            ),
        ),
    )
    report = _detect(architecture)
    findings = report.findings_of(CollisionKind.REEXPORT_AUTHORITY)
    assert findings
    assert findings[0].disposition is FindingDisposition.COLLISION
    assert {"n-reexport", "n-real"} <= set(findings[0].node_ids)


def test_obsolete_authority_test_is_detected() -> None:
    architecture = _graph(
        (
            _node("n-test", NodeKind.TEST, _TEST_PATH, 12),
            _node("n-legacy", NodeKind.COMPATIBILITY, _COMPAT_PATH, 30),
            _node("n-auth", NodeKind.AUTHORITY, _CTRL_PATH, 50),
        ),
        (
            _edge(
                "e-super", EdgeKind.SUPERSEDES, "n-auth", "n-legacy", _CTRL_PATH, 50
            ),
            _edge("e-test", EdgeKind.TESTS, "n-test", "n-legacy", _TEST_PATH, 12),
        ),
    )
    report = _detect(architecture)
    findings = [
        item
        for item in report.findings_of(CollisionKind.OBSOLETE_AUTHORITY_TEST)
        if item.disposition is FindingDisposition.COLLISION
    ]
    assert findings
    assert "n-test" in findings[0].node_ids
    assert "n-legacy" in findings[0].node_ids


def test_formal_arbitration_is_not_a_collision() -> None:
    binding = INITIAL_CONCERN_SOURCE_BINDINGS[0]
    extra = [
        item
        for item in INITIAL_CONCERN_SOURCE_BINDINGS
        if item.concern is ConcernKind.CONTENT_IDENTITY
        and item.recommended_disposition is OwnerDisposition.ADAPTER
    ][0]
    architecture = _graph(
        (
            _node("n-a", NodeKind.AUTHORITY, binding.path, binding.start_line),
            _node("n-b", NodeKind.AUTHORITY, extra.path, extra.start_line),
            _node("n-subject", NodeKind.SYMBOL, binding.path, binding.start_line + 1),
        ),
        (
            _edge(
                "e-a",
                EdgeKind.AUTHORIZES,
                "n-a",
                "n-subject",
                binding.path,
                binding.start_line,
            ),
            _edge(
                "e-b",
                EdgeKind.AUTHORIZES,
                "n-b",
                "n-subject",
                extra.path,
                extra.start_line,
            ),
            _edge(
                "e-adapt",
                EdgeKind.ADAPTS,
                "n-b",
                "n-a",
                extra.path,
                extra.start_line,
            ),
        ),
    )
    claims = (
        ConcernClaim(
            ConcernKind.CONTENT_IDENTITY, "n-a", OwnerDisposition.CANONICAL, ("e-a",)
        ),
        ConcernClaim(
            ConcernKind.CONTENT_IDENTITY, "n-b", OwnerDisposition.CANONICAL, ("e-b",)
        ),
    )
    arbitration = FormalArbitration(
        concern=ConcernKind.CONTENT_IDENTITY,
        canonical_owner_node_id="n-a",
        loser_classifications=(
            LoserClassification("n-b", OwnerDisposition.ADAPTER),
        ),
        arbitrator_identity="pcar-007-content-identity-arbitration",
        rationale=ArbitrationRationale.ADAPTS_EDGE,
        provenance=_fact(extra.path, extra.start_line, confidence=Confidence.EXACT),
        evidence_node_ids=("n-a", "n-b"),
        evidence_edge_ids=("e-adapt",),
    )
    ownership = resolve_authority_ownership(
        architecture, claims, arbitrations=(arbitration,)
    )
    report = _detect(architecture, ownership)
    assert recognize_formal_arbitration(("n-a", "n-b"), arbitration) is True
    assert recognize_formal_arbitration(("n-a", "n-b"), ownership=ownership) is True
    assert arbitration.content_identity in report.recognized_arbitrations
    competing = [
        item
        for item in report.findings
        if {"n-a", "n-b"} <= set(item.node_ids)
    ]
    assert competing
    assert all(item.disposition is not FindingDisposition.COLLISION for item in competing)
    assert any(item.formally_arbitrated is True for item in competing)
    assert all(
        item.disposition is FindingDisposition.FALSE_POSITIVE
        for item in competing
        if item.formally_arbitrated
    )


def test_adapters_projections_legacy_and_quarantined_simulation_are_false_positives() -> None:
    architecture = _graph(
        (
            _node("n-auth", NodeKind.AUTHORITY, _CTRL_PATH, 10),
            _node("n-adapter", NodeKind.COMPATIBILITY, _COMPAT_PATH, 12),
            _node("n-proj", NodeKind.GENERATED, _CLI_PATH, 8),
            _node("n-legacy", NodeKind.COMPATIBILITY, _COMPAT_PATH, 80),
            _node("n-sim", NodeKind.SIMULATION, _SIM_PATH, 20),
            _node("n-op", NodeKind.OPERATION, _PY_PATH, 40),
            _node("n-policy", NodeKind.POLICY, _CTRL_PATH, 60),
            _node("n-state", NodeKind.STATE, _STATE_PATH, 30),
        ),
        (
            _edge("e-auth-op", EdgeKind.AUTHORIZES, "n-auth", "n-op", _CTRL_PATH, 10),
            _edge(
                "e-auth-pol", EdgeKind.AUTHORIZES, "n-auth", "n-policy", _CTRL_PATH, 10
            ),
            _edge("e-adapt", EdgeKind.ADAPTS, "n-adapter", "n-auth", _COMPAT_PATH, 12),
            _edge("e-adapt-exec", EdgeKind.EXECUTES, "n-adapter", "n-op", _COMPAT_PATH, 12),
            _edge("e-proj", EdgeKind.GENERATES, "n-auth", "n-proj", _CLI_PATH, 8),
            _edge(
                "e-legacy", EdgeKind.SUPERSEDES, "n-auth", "n-legacy", _COMPAT_PATH, 80
            ),
            _edge("e-sim", EdgeKind.FALLBACKS_TO, "n-auth", "n-sim", _SIM_PATH, 20),
        ),
    )
    report = _detect(architecture)
    assert not [
        item
        for item in report.findings_of(CollisionKind.COMPATIBILITY_BYPASS)
        if item.disposition in {FindingDisposition.COLLISION, FindingDisposition.BLOCKER}
    ]
    assert not [
        item
        for item in report.findings_of(CollisionKind.CONTROL_BYPASS)
        if item.disposition in {FindingDisposition.COLLISION, FindingDisposition.BLOCKER}
        and "n-adapter" in item.node_ids
    ]
    sim = report.findings_of(CollisionKind.SIMULATION_TO_PRODUCTION_FLOW)
    assert sim
    assert all(item.disposition is FindingDisposition.FALSE_POSITIVE for item in sim)
    assert report.one_owner_invariant_holds is True


def test_shared_python_cli_mcp_authority_is_a_false_positive() -> None:
    architecture = _graph(
        (
            _node("n-py", NodeKind.ENTRYPOINT, _PY_PATH, 10),
            _node("n-cli", NodeKind.ENTRYPOINT, _CLI_PATH, 16),
            _node("n-mcp", NodeKind.ENTRYPOINT, _MCP_PATH, 22),
            _node("n-op", NodeKind.OPERATION, _PY_PATH, 80),
            _node("n-auth", NodeKind.AUTHORITY, _PY_PATH, 40),
        ),
        (
            _edge("e-py-op", EdgeKind.IMPLEMENTS, "n-py", "n-op", _PY_PATH, 10),
            _edge("e-cli-op", EdgeKind.IMPLEMENTS, "n-cli", "n-op", _CLI_PATH, 16),
            _edge("e-mcp-op", EdgeKind.IMPLEMENTS, "n-mcp", "n-op", _MCP_PATH, 22),
            _edge("e-py-auth", EdgeKind.AUTHORIZES, "n-py", "n-auth", _PY_PATH, 10),
            _edge("e-cli-auth", EdgeKind.AUTHORIZES, "n-cli", "n-auth", _CLI_PATH, 16),
            _edge("e-mcp-auth", EdgeKind.AUTHORIZES, "n-mcp", "n-auth", _MCP_PATH, 22),
        ),
    )
    report = _detect(architecture)
    findings = report.findings_of(CollisionKind.PYTHON_CLI_MCP_DIVERGENCE)
    assert findings
    assert all(item.disposition is FindingDisposition.FALSE_POSITIVE for item in findings)
    assert CollisionKind.PYTHON_CLI_MCP_DIVERGENCE not in {
        item.kind for item in report.collisions
    }


def test_migration_test_covering_canonical_and_legacy_is_a_false_positive() -> None:
    architecture = _graph(
        (
            _node("n-test", NodeKind.TEST, _TEST_PATH, 12),
            _node("n-legacy", NodeKind.COMPATIBILITY, _COMPAT_PATH, 30),
            _node("n-auth", NodeKind.AUTHORITY, _CTRL_PATH, 50),
        ),
        (
            _edge(
                "e-super", EdgeKind.SUPERSEDES, "n-auth", "n-legacy", _CTRL_PATH, 50
            ),
            _edge("e-test-legacy", EdgeKind.TESTS, "n-test", "n-legacy", _TEST_PATH, 12),
            _edge("e-test-canon", EdgeKind.TESTS, "n-test", "n-auth", _TEST_PATH, 20),
        ),
    )
    report = _detect(architecture)
    findings = report.findings_of(CollisionKind.OBSOLETE_AUTHORITY_TEST)
    assert findings
    assert all(item.disposition is FindingDisposition.FALSE_POSITIVE for item in findings)


def test_import_only_edges_are_not_authority_collisions() -> None:
    architecture = _graph(
        (
            _node("n-a", NodeKind.MODULE, _PY_PATH, 1),
            _node("n-b", NodeKind.MODULE, _CLI_PATH, 1),
            _node("n-shared", NodeKind.MODULE, _PY_PATH, 8),
        ),
        (
            _edge("e-a", EdgeKind.IMPORTS, "n-a", "n-shared", _PY_PATH, 1),
            _edge("e-b", EdgeKind.IMPORTS, "n-b", "n-shared", _CLI_PATH, 1),
        ),
    )
    report = _detect(architecture)
    assert report.collisions == ()
    assert report.blockers == ()
    assert report.one_owner_invariant_holds is True


def test_heuristic_signals_are_not_promoted() -> None:
    architecture = _graph(
        (
            _node(
                "n-cap-a",
                NodeKind.PROVIDER,
                _PY_PATH,
                10,
                confidence=Confidence.HEURISTIC,
            ),
            _node(
                "n-cap-b",
                NodeKind.PROVIDER,
                _PY_PATH,
                40,
                confidence=Confidence.OPAQUE,
            ),
            _node("n-subject", NodeKind.OPERATION, _PY_PATH, 80),
        ),
        (
            _edge(
                "e-a",
                EdgeKind.AUTHORIZES,
                "n-cap-a",
                "n-subject",
                _PY_PATH,
                10,
                confidence=Confidence.HEURISTIC,
            ),
            _edge(
                "e-b",
                EdgeKind.AUTHORIZES,
                "n-cap-b",
                "n-subject",
                _PY_PATH,
                40,
                confidence=Confidence.OPAQUE,
            ),
        ),
    )
    report = _detect(architecture)
    findings = report.findings_of(CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY)
    assert findings
    assert all(item.disposition is FindingDisposition.UNKNOWN for item in findings)
    assert findings == report.unknowns or set(findings) <= set(report.unknowns)
    assert not [
        item
        for item in findings
        if item.disposition in {FindingDisposition.COLLISION, FindingDisposition.BLOCKER}
    ]
    with pytest.raises(DuplicateAuthorityError, match="heuristic-only"):
        refuse_heuristic_promotion("promotion")
    detector = DuplicateAuthorityDetector()
    with pytest.raises(DuplicateAuthorityError, match="heuristic-only"):
        detector.promote_heuristic()


def test_unknown_production_ownership_emits_blocker() -> None:
    architecture = _graph(
        (_node("n-unknown", NodeKind.AUTHORITY, _CTRL_PATH, 12),)
    )
    ownership = resolve_authority_ownership(
        architecture,
        (
            ConcernClaim(
                ConcernKind.TASK_IDENTITY,
                "n-unknown",
                OwnerDisposition.UNKNOWN,
            ),
        ),
    )
    report = _detect(architecture, ownership)
    blockers = report.findings_of(CollisionKind.UNKNOWN_PRODUCTION_OWNER)
    assert blockers
    assert all(item.disposition is FindingDisposition.BLOCKER for item in blockers)
    assert report.fails_closed is True
    assert report.one_owner_invariant_holds is False
    assert "n-unknown" in blockers[0].node_ids


def test_multiple_production_authorities_without_arbitration_block() -> None:
    binding = [
        item
        for item in INITIAL_CONCERN_SOURCE_BINDINGS
        if item.concern is ConcernKind.CONTENT_IDENTITY
        and item.recommended_disposition is OwnerDisposition.CANONICAL
    ][0]
    extra = [
        item
        for item in INITIAL_CONCERN_SOURCE_BINDINGS
        if item.concern is ConcernKind.CONTENT_IDENTITY
        and item.recommended_disposition is OwnerDisposition.ADAPTER
    ][0]
    architecture = _graph(
        (
            _node("n-a", NodeKind.AUTHORITY, binding.path, binding.start_line),
            _node("n-b", NodeKind.AUTHORITY, extra.path, extra.start_line),
            _node("n-subject", NodeKind.SYMBOL, binding.path, binding.start_line + 1),
        ),
        (
            _edge(
                "e-a",
                EdgeKind.AUTHORIZES,
                "n-a",
                "n-subject",
                binding.path,
                binding.start_line,
            ),
            _edge(
                "e-b",
                EdgeKind.AUTHORIZES,
                "n-b",
                "n-subject",
                extra.path,
                extra.start_line,
            ),
        ),
    )
    claims = (
        ConcernClaim(
            ConcernKind.CONTENT_IDENTITY, "n-a", OwnerDisposition.CANONICAL, ("e-a",)
        ),
        ConcernClaim(
            ConcernKind.CONTENT_IDENTITY, "n-b", OwnerDisposition.CANONICAL, ("e-b",)
        ),
    )
    ownership = resolve_authority_ownership(architecture, claims)
    report = _detect(architecture, ownership)
    blockers = [
        item
        for item in report.findings_of(CollisionKind.MULTIPLE_PRODUCTION_AUTHORITIES)
        if item.disposition is FindingDisposition.BLOCKER
        and item.concern is ConcernKind.CONTENT_IDENTITY
    ]
    assert blockers
    assert {"n-a", "n-b"} <= set(blockers[0].node_ids)
    assert report.fails_closed is True


def test_one_owner_invariant_holds_for_a_single_canonical_authority() -> None:
    architecture = _graph(
        (
            _node("n-auth", NodeKind.AUTHORITY, _CTRL_PATH, 10),
            _node("n-op", NodeKind.OPERATION, _PY_PATH, 40),
        ),
        (
            _edge("e-auth", EdgeKind.AUTHORIZES, "n-auth", "n-op", _CTRL_PATH, 10),
        ),
    )
    report = _detect(architecture)
    assert report.collisions == ()
    assert report.blockers == ()
    assert report.one_owner_invariant_holds is True
    assert report.fails_closed is False
    detector = DuplicateAuthorityDetector()
    assert detector.detect(architecture) == report
    assert build_duplicate_authority_report(architecture) == report


def test_detector_cannot_remediate_or_select_owner() -> None:
    architecture = _graph(
        (_node("n-auth", NodeKind.AUTHORITY, _CTRL_PATH, 10),)
    )
    report = _detect(architecture)
    detector = DuplicateAuthorityDetector()
    assert detector.can_authorize_changes is False
    assert detector.can_select_owner is False
    assert detector.can_remediate is False
    assert detector.task_id == "PCAR-007"
    assert detector.effect_class == "read_only_analysis"
    with pytest.raises(DuplicateAuthorityAuthorityError, match="cannot"):
        detector.authorize_change("refactor")
    with pytest.raises(DuplicateAuthorityAuthorityError, match="canonical owner"):
        detector.select_owner("provider capability")
    with pytest.raises(DuplicateAuthorityAuthorityError, match="cannot"):
        detector.remediate()
    with pytest.raises(DuplicateAuthorityAuthorityError, match="cannot"):
        detector.consolidate()
    with pytest.raises(DuplicateAuthorityAuthorityError, match="cannot"):
        report.authorize_change("merge")
    with pytest.raises(DuplicateAuthorityAuthorityError, match="canonical owner"):
        report.select_owner()
    with pytest.raises(DuplicateAuthorityAuthorityError, match="cannot"):
        report.remediate()
    with pytest.raises(DuplicateAuthorityAuthorityError, match="cannot"):
        refuse_remediation("delete")
    with pytest.raises(DuplicateAuthorityAuthorityError, match="canonical owner"):
        refuse_owner_selection("pick")
    with pytest.raises(DuplicateAuthorityError, match="content identity"):
        lookup_owner_by_content_identity("baguqeera" + ("a" * 50))


def test_content_identity_cannot_be_a_finding_node() -> None:
    architecture = _graph(
        (
            _node("n-cap-a", NodeKind.PROVIDER, _PY_PATH, 10),
            _node("n-cap-b", NodeKind.PROVIDER, _PY_PATH, 40),
            _node("n-subject", NodeKind.OPERATION, _PY_PATH, 80),
        ),
        (
            _edge("e-a", EdgeKind.AUTHORIZES, "n-cap-a", "n-subject", _PY_PATH, 10),
            _edge("e-b", EdgeKind.AUTHORIZES, "n-cap-b", "n-subject", _PY_PATH, 40),
        ),
    )
    report = _detect(architecture)
    finding = report.findings_of(CollisionKind.INDEPENDENT_PROVIDER_CAPABILITY)[0]
    with pytest.raises(DuplicateAuthorityError, match="content identity"):
        AuthorityCollision(
            kind=finding.kind,
            disposition=finding.disposition,
            concern=finding.concern,
            message=finding.message,
            node_ids=(finding.content_identity,),
            edge_ids=finding.edge_ids,
            provenance=finding.provenance,
        )


def test_round_trip_and_canonical_identity() -> None:
    architecture = _graph(
        (
            _node("n-cap-a", NodeKind.PROVIDER, _PY_PATH, 10),
            _node("n-cap-b", NodeKind.PROVIDER, _PY_PATH, 40),
            _node("n-subject", NodeKind.OPERATION, _PY_PATH, 80),
        ),
        (
            _edge("e-a", EdgeKind.AUTHORIZES, "n-cap-a", "n-subject", _PY_PATH, 10),
            _edge("e-b", EdgeKind.AUTHORIZES, "n-cap-b", "n-subject", _PY_PATH, 40),
        ),
    )
    report = _detect(architecture)
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
    reversed_report = _detect(
        ArchitectureIR.from_parts(
            repository_tree=architecture.repository_tree,
            freshness=architecture.freshness,
            nodes=tuple(reversed(architecture.nodes)),
            edges=tuple(reversed(architecture.edges)),
        )
    )
    assert reversed_report.content_identity == report.content_identity
    assert reversed_report.to_dict() == report.to_dict()


def test_unknown_fields_and_identity_mismatch_are_rejected() -> None:
    architecture = _graph(
        (
            _node("n-cap-a", NodeKind.PROVIDER, _PY_PATH, 10),
            _node("n-cap-b", NodeKind.PROVIDER, _PY_PATH, 40),
            _node("n-subject", NodeKind.OPERATION, _PY_PATH, 80),
        ),
        (
            _edge("e-a", EdgeKind.AUTHORIZES, "n-cap-a", "n-subject", _PY_PATH, 10),
            _edge("e-b", EdgeKind.AUTHORIZES, "n-cap-b", "n-subject", _PY_PATH, 40),
        ),
    )
    payload = _detect(architecture).to_dict()
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
    with pytest.raises(DuplicateAuthorityError, match="unexpected duplicate-authority-report schema"):
        DuplicateAuthorityReport.from_mapping(schema)


def test_required_detections_cover_the_plan_vocabulary() -> None:
    architecture = _graph(
        (
            _node("n-cap-a", NodeKind.PROVIDER, _PY_PATH, 10),
            _node("n-cap-b", NodeKind.PROVIDER, _PY_PATH, 40),
            _node("n-subject", NodeKind.OPERATION, _PY_PATH, 80),
            _node("n-r1", NodeKind.RECEIPT, _PY_PATH, 100),
            _node("n-r2", NodeKind.RECEIPT, _PY_PATH, 120),
            _node("n-store-a", NodeKind.STATE, _STATE_PATH, 20),
            _node("n-store-b", NodeKind.STATE, _STATE_PATH, 80),
            _node("n-fact", NodeKind.STATE, _STATE_PATH, 140),
            _node("n-compat", NodeKind.COMPATIBILITY, _COMPAT_PATH, 8),
            _node("n-auth", NodeKind.AUTHORITY, _CTRL_PATH, 40),
            _node("n-tool", NodeKind.ENTRYPOINT, _PY_PATH, 200),
            _node("n-policy", NodeKind.POLICY, _CTRL_PATH, 60),
            _node("n-sim", NodeKind.SIMULATION, _SIM_PATH, 14),
            _node("n-py", NodeKind.ENTRYPOINT, _PY_PATH, 300),
            _node("n-cli", NodeKind.ENTRYPOINT, _CLI_PATH, 16),
            _node("n-op", NodeKind.OPERATION, _PY_PATH, 400),
            _node("n-auth-py", NodeKind.AUTHORITY, _PY_PATH, 420),
            _node("n-auth-cli", NodeKind.AUTHORITY, _CLI_PATH, 40),
            _node("n-reexport", NodeKind.AUTHORITY, _PY_PATH, 500),
            _node("n-real", NodeKind.AUTHORITY, _PY_PATH, 540),
            _node("n-test", NodeKind.TEST, _TEST_PATH, 12),
            _node("n-legacy", NodeKind.COMPATIBILITY, _COMPAT_PATH, 30),
        ),
        (
            _edge("e-cap-a", EdgeKind.AUTHORIZES, "n-cap-a", "n-subject", _PY_PATH, 10),
            _edge("e-cap-b", EdgeKind.AUTHORIZES, "n-cap-b", "n-subject", _PY_PATH, 40),
            _edge("e-r1", EdgeKind.CONFIRMS, "n-r1", "n-op", _PY_PATH, 100),
            _edge("e-r2", EdgeKind.CONFIRMS, "n-r2", "n-op", _PY_PATH, 120),
            _edge("e-st-a", EdgeKind.PERSISTS, "n-store-a", "n-fact", _STATE_PATH, 20),
            _edge("e-st-b", EdgeKind.WRITES, "n-store-b", "n-fact", _STATE_PATH, 80),
            _edge("e-auth-op", EdgeKind.AUTHORIZES, "n-auth", "n-subject", _CTRL_PATH, 40),
            _edge("e-bypass", EdgeKind.EXECUTES, "n-compat", "n-subject", _COMPAT_PATH, 8),
            _edge(
                "e-canon", EdgeKind.AUTHORIZES, "n-auth", "n-policy", _CTRL_PATH, 40
            ),
            _edge(
                "e-tool",
                EdgeKind.EVALUATES_POLICY,
                "n-tool",
                "n-policy",
                _PY_PATH,
                200,
            ),
            _edge("e-sim", EdgeKind.WRITES, "n-sim", "n-fact", _SIM_PATH, 14),
            _edge("e-py-op", EdgeKind.IMPLEMENTS, "n-py", "n-op", _PY_PATH, 300),
            _edge("e-cli-op", EdgeKind.IMPLEMENTS, "n-cli", "n-op", _CLI_PATH, 16),
            _edge("e-py-auth", EdgeKind.AUTHORIZES, "n-py", "n-auth-py", _PY_PATH, 300),
            _edge(
                "e-cli-auth", EdgeKind.AUTHORIZES, "n-cli", "n-auth-cli", _CLI_PATH, 16
            ),
            _edge(
                "e-reexport",
                EdgeKind.REEXPORTS,
                "n-reexport",
                "n-real",
                _PY_PATH,
                500,
            ),
            _edge(
                "e-super", EdgeKind.SUPERSEDES, "n-auth", "n-legacy", _CTRL_PATH, 40
            ),
            _edge("e-test", EdgeKind.TESTS, "n-test", "n-legacy", _TEST_PATH, 12),
        ),
    )
    report = _detect(architecture)
    detected = {
        item.kind
        for item in report.findings
        if item.disposition
        in {FindingDisposition.COLLISION, FindingDisposition.BLOCKER}
    }
    assert set(REQUIRED_DETECTIONS) <= detected
    assert INITIAL_CONCERNS


def test_claims_without_prebuilt_ownership_still_emit_blockers() -> None:
    architecture = _graph(
        (
            _node("n-a", NodeKind.AUTHORITY, _CTRL_PATH, 10),
            _node("n-b", NodeKind.AUTHORITY, _CTRL_PATH, 40),
            _node("n-subject", NodeKind.SYMBOL, _CTRL_PATH, 80),
        ),
        (
            _edge("e-a", EdgeKind.AUTHORIZES, "n-a", "n-subject", _CTRL_PATH, 10),
            _edge("e-b", EdgeKind.AUTHORIZES, "n-b", "n-subject", _CTRL_PATH, 40),
        ),
    )
    claims = (
        ConcernClaim(
            ConcernKind.POLICY_DECISION, "n-a", OwnerDisposition.CANONICAL, ("e-a",)
        ),
        ConcernClaim(
            ConcernKind.POLICY_DECISION, "n-b", OwnerDisposition.CANONICAL, ("e-b",)
        ),
    )
    report = _detect(architecture, claims=claims)
    assert report.ownership_graph_identity
    assert report.findings_of(CollisionKind.MULTIPLE_PRODUCTION_AUTHORITIES)
    assert report.findings_of(CollisionKind.UNKNOWN_PRODUCTION_OWNER)
