"""Tests for bounded interprocedural security-property analysis (VFS-030)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.security_contract_analysis import (
    ANALYZER_VERSION,
    GOAL_ID,
    MAX_PATH_HOPS,
    SECURITY_ANALYSIS_AUTHORIZES_REPAIR,
    SECURITY_ANALYSIS_IS_COMPLETION_EVIDENCE,
    SECURITY_CONTRACT_ANALYSIS_VERSION,
    AnalysisVerdict,
    EdgeResolution,
    FindingClassification,
    FlowRole,
    ForbiddenBodyError,
    ForgedSecurityIdentityError,
    SecurityAnalysisConfig,
    SecurityContractAnalysisError,
    SecurityEvidence,
    SecurityFinding,
    SecurityRuleFamily,
    ThreatPath,
    ThreatPathOrigin,
    analyze_security_contracts,
    build_security_finding,
    classify_security_finding,
    make_evidence,
    make_flow_edge,
    make_flow_node,
    make_security_property,
    security_rule_families,
    security_rule_spec,
    security_rule_specs,
    vulnerability_requirements_met,
)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_closed_rule_catalog_has_ten_families() -> None:
    families = security_rule_families()
    assert len(families) == 10
    assert len(security_rule_specs()) == 10
    assert set(families) == set(SecurityRuleFamily)
    # Deterministic order matches enum catalog order.
    assert families[0] is SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS
    assert families[-1] is SecurityRuleFamily.MCP_SCHEMA_DISPATCH_CONFUSION


def test_rule_spec_lookup_and_defaults() -> None:
    spec = security_rule_spec(SecurityRuleFamily.SECRET_FLOW)
    assert spec.rule_id == "sec/secret-flow"
    assert "secret_material" in spec.source_tags
    assert "log_emit" in spec.sink_tags
    assert "secret_redact" in spec.sanitizer_tags


def test_authority_flags_are_fail_closed() -> None:
    assert SECURITY_ANALYSIS_IS_COMPLETION_EVIDENCE is False
    assert SECURITY_ANALYSIS_AUTHORIZES_REPAIR is False
    assert GOAL_ID == "VFS-030"
    assert SECURITY_CONTRACT_ANALYSIS_VERSION == 1
    assert ANALYZER_VERSION.startswith("security-contract-analysis")


# ---------------------------------------------------------------------------
# Vulnerability gate
# ---------------------------------------------------------------------------


def test_vulnerability_requires_four_ingredients() -> None:
    prop = make_security_property(
        "prop:path",
        SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS,
        resource="vfs.root",
        statement="Paths must stay under the declared root.",
    )
    path = ThreatPath(
        path_id="path:1",
        node_ids=("src", "sink"),
        origin=ThreatPathOrigin.REACHABLE,
        edge_ids=("e1",),
        hop_count=1,
    )
    evidence = make_evidence("artifact:cex:1")

    ok, missing = vulnerability_requirements_met(
        security_property=None,
        threat_path=path,
        impact="escape",
        evidence=evidence,
    )
    assert ok is False
    assert "security_property" in missing

    ok, missing = vulnerability_requirements_met(
        security_property=prop,
        threat_path=None,
        impact="escape",
        evidence=evidence,
    )
    assert ok is False
    assert "threat_path" in missing

    ok, missing = vulnerability_requirements_met(
        security_property=prop,
        threat_path=path,
        impact="",
        evidence=evidence,
    )
    assert ok is False
    assert "impact" in missing

    ok, missing = vulnerability_requirements_met(
        security_property=prop,
        threat_path=path,
        impact="escape",
        evidence=SecurityEvidence(),
    )
    assert ok is False
    assert "evidence" in missing

    ok, missing = vulnerability_requirements_met(
        security_property=prop,
        threat_path=path,
        impact="escape",
        evidence=evidence,
    )
    assert ok is True
    assert missing == ()


def test_classify_vulnerability_vs_drift_vs_suspicion_vs_dynamic() -> None:
    prop = make_security_property(
        "prop:auth",
        SecurityRuleFamily.AUTHORIZATION_CAPABILITY_BYPASS,
        resource="admin",
        statement="Privileged actions require capability checks.",
    )
    closed = ThreatPath(
        path_id="p:closed",
        node_ids=("a", "b"),
        origin=ThreatPathOrigin.REACHABLE,
        hop_count=1,
    )
    dynamic = ThreatPath(
        path_id="p:dyn",
        node_ids=("a", "b"),
        origin=ThreatPathOrigin.REACHABLE,
        has_unknown_dynamic=True,
        hop_count=1,
    )
    evidence = make_evidence("cid:ev")

    assert (
        classify_security_finding(
            security_property=prop,
            threat_path=closed,
            impact="bypass",
            evidence=evidence,
        )
        is FindingClassification.VULNERABILITY
    )
    assert (
        classify_security_finding(
            security_property=None,
            threat_path=closed,
            impact="",
            evidence=evidence,
        )
        is FindingClassification.CORRECTNESS_DRIFT
    )
    assert (
        classify_security_finding(
            security_property=prop,
            threat_path=closed,
            impact="bypass",
            evidence=SecurityEvidence(),
        )
        is FindingClassification.SUSPICION
    )
    assert (
        classify_security_finding(
            security_property=None,
            threat_path=dynamic,
            impact="",
            evidence=evidence,
        )
        is FindingClassification.UNKNOWN_DYNAMIC
    )
    assert (
        classify_security_finding(
            security_property=prop,
            threat_path=dynamic,
            impact="bypass",
            evidence=evidence,
        )
        is FindingClassification.SUSPICION
    )


def test_vulnerability_finding_rejects_incomplete_gate() -> None:
    with pytest.raises(SecurityContractAnalysisError, match="vulnerability"):
        build_security_finding(
            family=SecurityRuleFamily.SECRET_FLOW,
            classification=FindingClassification.VULNERABILITY,
            summary="leaked",
            impact="disclosure",
            # missing property, path, evidence
        )


# ---------------------------------------------------------------------------
# Seeded true / false positives and unknown dynamic
# ---------------------------------------------------------------------------


def _path_traversal_graph(*, sanitized: bool = False, dynamic: bool = False):
    nodes = [
        make_flow_node(
            "n:user",
            "pkg.api.user_path",
            role=FlowRole.SOURCE,
            tags=("untrusted_path",),
            path="src/api.py",
            repository_id="repository:alpha",
            interface="mcp://vfs/read",
            line=10,
        ),
        make_flow_node(
            "n:mid",
            "pkg.vfs.join",
            role=FlowRole.PASSTHROUGH,
            tags=(),
            path="src/vfs.py",
            repository_id="repository:alpha",
        ),
        make_flow_node(
            "n:open",
            "pkg.vfs.open",
            role=FlowRole.SINK,
            tags=("fs_open",),
            path="src/vfs.py",
            repository_id="repository:alpha",
            interface="mcp://vfs/open",
            line=40,
        ),
    ]
    if sanitized:
        nodes.insert(
            2,
            make_flow_node(
                "n:jail",
                "pkg.vfs.scope_confine",
                role=FlowRole.SANITIZER,
                tags=("scope_confine",),
                path="src/vfs.py",
                repository_id="repository:alpha",
            ),
        )
        edges = [
            make_flow_edge("e1", "n:user", "n:mid"),
            make_flow_edge("e2", "n:mid", "n:jail"),
            make_flow_edge("e3", "n:jail", "n:open"),
        ]
    elif dynamic:
        edges = [
            make_flow_edge("e1", "n:user", "n:mid"),
            make_flow_edge(
                "e2",
                "n:mid",
                "n:open",
                resolution=EdgeResolution.DYNAMIC,
            ),
        ]
    else:
        edges = [
            make_flow_edge("e1", "n:user", "n:mid"),
            make_flow_edge("e2", "n:mid", "n:open"),
        ]
    return nodes, edges


def test_seeded_true_positive_path_traversal_vulnerability() -> None:
    nodes, edges = _path_traversal_graph()
    prop = make_security_property(
        "prop:vfs-root",
        SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS,
        resource="vfs.root",
        statement="Untrusted paths must not escape the declared VFS root.",
        interface="mcp://vfs/open",
    )
    report = analyze_security_contracts(
        nodes=nodes,
        edges=edges,
        properties=[prop],
        default_evidence=make_evidence("artifact:cex:path"),
        config=SecurityAnalysisConfig(
            tree_id="tree:1", policy_revision="policy:v1"
        ),
        families=[SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS],
    )
    assert report.verdict is AnalysisVerdict.FINDINGS
    vulns = report.vulnerabilities
    assert len(vulns) == 1
    finding = vulns[0]
    assert finding.classification is FindingClassification.VULNERABILITY
    assert finding.seed_label == "true_positive"
    assert finding.family is SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS
    assert finding.security_property is not None
    assert finding.threat_path is not None
    assert finding.impact
    assert finding.evidence.has_evidence
    assert "pkg.api.user_path" in finding.symbols
    assert finding.tree_id == "tree:1"


def test_seeded_false_positive_sanitized_path() -> None:
    nodes, edges = _path_traversal_graph(sanitized=True)
    prop = make_security_property(
        "prop:vfs-root",
        SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS,
        resource="vfs.root",
        statement="Untrusted paths must not escape the declared VFS root.",
    )
    report = analyze_security_contracts(
        nodes=nodes,
        edges=edges,
        properties=[prop],
        default_evidence=make_evidence("artifact:cex:path"),
        families=[SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS],
    )
    assert report.findings
    assert report.vulnerabilities == ()
    fp = [f for f in report.findings if f.seed_label == "false_positive"]
    assert fp
    assert all(
        f.classification is FindingClassification.CORRECTNESS_DRIFT
        for f in fp
    )
    assert "sanitized_path" in fp[0].missing_requirements


def test_seeded_false_positive_without_declared_property() -> None:
    nodes, edges = _path_traversal_graph()
    report = analyze_security_contracts(
        nodes=nodes,
        edges=edges,
        properties=[],
        default_evidence=make_evidence("artifact:cex:path"),
        families=[SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS],
    )
    assert report.vulnerabilities == ()
    assert report.findings
    assert all(
        f.classification is FindingClassification.CORRECTNESS_DRIFT
        for f in report.findings
    )
    assert all(f.seed_label == "false_positive" for f in report.findings)


def test_seeded_false_positive_missing_evidence_is_suspicion() -> None:
    nodes, edges = _path_traversal_graph()
    prop = make_security_property(
        "prop:vfs-root",
        SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS,
        resource="vfs.root",
        statement="Untrusted paths must not escape the declared VFS root.",
    )
    report = analyze_security_contracts(
        nodes=nodes,
        edges=edges,
        properties=[prop],
        default_evidence=SecurityEvidence(),
        families=[SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS],
    )
    assert report.vulnerabilities == ()
    assert any(
        f.classification is FindingClassification.SUSPICION
        for f in report.findings
    )


def test_seeded_unknown_dynamic_path() -> None:
    nodes, edges = _path_traversal_graph(dynamic=True)
    report = analyze_security_contracts(
        nodes=nodes,
        edges=edges,
        properties=[],
        default_evidence=make_evidence("artifact:dyn"),
        families=[SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS],
    )
    assert report.vulnerabilities == ()
    dyn = [
        f
        for f in report.findings
        if f.classification is FindingClassification.UNKNOWN_DYNAMIC
    ]
    assert dyn
    assert dyn[0].seed_label == "unknown_dynamic"
    assert dyn[0].threat_path is not None
    assert dyn[0].threat_path.has_unknown_dynamic is True


# ---------------------------------------------------------------------------
# All ten rule families (compact seeded recipes)
# ---------------------------------------------------------------------------


_FAMILY_FIXTURES: dict[SecurityRuleFamily, tuple[tuple[str, ...], tuple[str, ...]]] = {
    SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS: (
        ("untrusted_path",),
        ("fs_open",),
    ),
    SecurityRuleFamily.AUTHORIZATION_CAPABILITY_BYPASS: (
        ("unauthenticated",),
        ("privileged_action",),
    ),
    SecurityRuleFamily.UNSAFE_DESERIALIZATION_COMMAND: (
        ("untrusted_bytes",),
        ("pickle_loads",),
    ),
    SecurityRuleFamily.SECRET_FLOW: (
        ("secret_material",),
        ("log_emit",),
    ),
    SecurityRuleFamily.CID_INTEGRITY_BYPASS: (
        ("unverified_bytes",),
        ("cid_accept",),
    ),
    SecurityRuleFamily.CACHE_POISONING_STALENESS: (
        ("stale_cache",),
        ("cache_serve",),
    ),
    SecurityRuleFamily.SYMLINK_ESCAPE: (
        ("symlink_path",),
        ("fs_follow",),
    ),
    SecurityRuleFamily.SILENT_FALLBACK_MOCK_SUCCESS: (
        ("backend_error",),
        ("mock_success",),
    ),
    SecurityRuleFamily.JOURNAL_ATOMICITY_VIOLATION: (
        ("partial_write",),
        ("commit_visible",),
    ),
    SecurityRuleFamily.MCP_SCHEMA_DISPATCH_CONFUSION: (
        ("schema_drift",),
        ("mcp_invoke",),
    ),
}


@pytest.mark.parametrize("family", list(SecurityRuleFamily))
def test_each_rule_family_emits_true_positive(family: SecurityRuleFamily) -> None:
    source_tags, sink_tags = _FAMILY_FIXTURES[family]
    nodes = [
        make_flow_node(
            "s",
            f"src.{family.value}",
            role=FlowRole.SOURCE,
            tags=source_tags,
            repository_id="repository:seed",
        ),
        make_flow_node(
            "k",
            f"sink.{family.value}",
            role=FlowRole.SINK,
            tags=sink_tags,
            repository_id="repository:seed",
        ),
    ]
    edges = [make_flow_edge("e", "s", "k")]
    prop = make_security_property(
        f"prop:{family.value}",
        family,
        resource=f"resource:{family.value}",
        statement=f"Property for {family.value}",
    )
    report = analyze_security_contracts(
        nodes=nodes,
        edges=edges,
        properties=[prop],
        default_evidence=make_evidence(f"artifact:{family.value}"),
        families=[family],
    )
    assert len(report.vulnerabilities) == 1
    assert report.vulnerabilities[0].family is family
    assert report.vulnerabilities[0].seed_label == "true_positive"


# ---------------------------------------------------------------------------
# Bounds, identities, serialization
# ---------------------------------------------------------------------------


def test_report_is_deterministic_and_round_trips() -> None:
    nodes, edges = _path_traversal_graph()
    prop = make_security_property(
        "prop:vfs-root",
        SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS,
        resource="vfs.root",
        statement="root confinement",
    )
    cfg = SecurityAnalysisConfig(tree_id="t", policy_revision="p")
    a = analyze_security_contracts(
        nodes=nodes,
        edges=edges,
        properties=[prop],
        default_evidence=make_evidence("a:1"),
        config=cfg,
        families=[SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS],
    )
    b = analyze_security_contracts(
        nodes=nodes,
        edges=edges,
        properties=[prop],
        default_evidence=make_evidence("a:1"),
        config=cfg,
        families=[SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS],
    )
    assert a.report_id == b.report_id
    assert a.to_dict() == b.to_dict()

    restored = type(a).from_dict(a.to_record())
    assert restored.report_id == a.report_id
    assert len(restored.findings) == len(a.findings)
    assert restored.findings[0].finding_id == a.findings[0].finding_id


def test_forged_identity_rejected() -> None:
    node = make_flow_node("n1", "sym")
    payload = node.to_dict()
    payload["content_id"] = "forged:not-real"
    with pytest.raises(ForgedSecurityIdentityError):
        type(node).from_dict(payload)


def test_evidence_rejects_secret_notes() -> None:
    with pytest.raises(ForbiddenBodyError):
        SecurityEvidence(notes=("password=hunter2",))


def test_evidence_from_dict_rejects_body_keys() -> None:
    with pytest.raises(ForbiddenBodyError):
        SecurityEvidence.from_dict(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/security-evidence@1",
                "schema_version": 1,
                "contract_version": 1,
                "artifact_cids": (),
                "counterexample_cids": (),
                "proof_cids": (),
                "runtime_cids": (),
                "graph_slice_cids": (),
                "notes": (),
                "source_body": "def evil(): pass",
            }
        )


def test_declared_threat_path_can_justify_vulnerability() -> None:
    nodes = [
        make_flow_node(
            "s",
            "src",
            role=FlowRole.SOURCE,
            tags=("secret_material",),
        ),
        make_flow_node(
            "k",
            "sink",
            role=FlowRole.SINK,
            tags=("log_emit",),
        ),
    ]
    # No edges — only a declared path.
    prop = make_security_property(
        "prop:secret",
        SecurityRuleFamily.SECRET_FLOW,
        resource="credentials",
        statement="Secrets must not reach logs.",
    )
    declared = ThreatPath(
        path_id="declared:1",
        node_ids=("s", "k"),
        origin=ThreatPathOrigin.DECLARED,
        hop_count=1,
    )
    report = analyze_security_contracts(
        nodes=nodes,
        edges=[],
        properties=[prop],
        declared_paths=[declared],
        default_evidence=make_evidence("artifact:secret"),
        families=[SecurityRuleFamily.SECRET_FLOW],
    )
    assert len(report.vulnerabilities) == 1
    assert (
        report.vulnerabilities[0].threat_path.origin
        is ThreatPathOrigin.DECLARED
    )


def test_max_findings_truncation() -> None:
    nodes = [
        make_flow_node(
            "s",
            "src",
            role=FlowRole.SOURCE,
            tags=("untrusted_path",),
        ),
    ]
    edges = []
    # Many sinks from one source.
    for i in range(5):
        nodes.append(
            make_flow_node(
                f"k{i}",
                f"sink{i}",
                role=FlowRole.SINK,
                tags=("fs_open",),
            )
        )
        edges.append(make_flow_edge(f"e{i}", "s", f"k{i}"))
    prop = make_security_property(
        "prop",
        SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS,
        resource="r",
        statement="s",
    )
    report = analyze_security_contracts(
        nodes=nodes,
        edges=edges,
        properties=[prop],
        default_evidence=make_evidence("a"),
        config=SecurityAnalysisConfig(max_findings=2),
        families=[SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS],
    )
    assert report.truncated is True
    assert len(report.findings) <= 2
    assert report.verdict is AnalysisVerdict.BOUNDED


def test_empty_graph_is_empty_verdict() -> None:
    report = analyze_security_contracts(nodes=[], edges=[])
    assert report.verdict is AnalysisVerdict.EMPTY
    assert report.findings == ()


def test_clean_graph_without_source_sink() -> None:
    nodes = [
        make_flow_node("a", "only", role=FlowRole.PASSTHROUGH, tags=()),
    ]
    report = analyze_security_contracts(nodes=nodes, edges=[])
    assert report.verdict is AnalysisVerdict.CLEAN
    assert report.findings == ()


def test_config_bounds_enforced() -> None:
    with pytest.raises(Exception):
        SecurityAnalysisConfig(max_hops=0)
    with pytest.raises(Exception):
        SecurityAnalysisConfig(max_hops=MAX_PATH_HOPS + 1)


def test_finding_content_id_stable() -> None:
    path = ThreatPath(
        path_id="p",
        node_ids=("a", "b"),
        hop_count=1,
    )
    prop = make_security_property(
        "prop",
        SecurityRuleFamily.CID_INTEGRITY_BYPASS,
        resource="cid",
        statement="CID must bind content.",
    )
    finding = build_security_finding(
        family=SecurityRuleFamily.CID_INTEGRITY_BYPASS,
        classification=FindingClassification.VULNERABILITY,
        summary="bypass",
        impact="tamper",
        security_property=prop,
        threat_path=path,
        evidence=make_evidence("c:1"),
    )
    assert finding.finding_id == finding.content_id
    assert SecurityFinding.from_dict(finding.to_record()).finding_id == (
        finding.finding_id
    )


def test_duplicate_node_id_rejected() -> None:
    nodes = [
        make_flow_node("n", "a"),
        make_flow_node("n", "b"),
    ]
    with pytest.raises(SecurityContractAnalysisError, match="duplicate"):
        analyze_security_contracts(nodes=nodes, edges=[])


def test_interprocedural_multi_hop_path() -> None:
    nodes = [
        make_flow_node(
            "s", "entry", role=FlowRole.SOURCE, tags=("untrusted_bytes",)
        ),
        make_flow_node("m1", "decode", role=FlowRole.PASSTHROUGH),
        make_flow_node("m2", "handoff", role=FlowRole.PASSTHROUGH),
        make_flow_node(
            "k", "exec", role=FlowRole.SINK, tags=("subprocess_shell",)
        ),
    ]
    edges = [
        make_flow_edge("e1", "s", "m1"),
        make_flow_edge("e2", "m1", "m2"),
        make_flow_edge("e3", "m2", "k"),
    ]
    prop = make_security_property(
        "prop:cmd",
        SecurityRuleFamily.UNSAFE_DESERIALIZATION_COMMAND,
        resource="shell",
        statement="No untrusted shell.",
    )
    report = analyze_security_contracts(
        nodes=nodes,
        edges=edges,
        properties=[prop],
        default_evidence=make_evidence("a"),
        families=[SecurityRuleFamily.UNSAFE_DESERIALIZATION_COMMAND],
    )
    assert len(report.vulnerabilities) == 1
    assert report.vulnerabilities[0].threat_path.hop_count == 3
    assert report.vulnerabilities[0].threat_path.node_ids == (
        "s",
        "m1",
        "m2",
        "k",
    )
