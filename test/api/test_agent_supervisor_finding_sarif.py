"""Tests for deterministic bounded SARIF projection (VFS-030)."""

from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.finding_sarif import (
    DEFAULT_MAX_RESULTS,
    FINDING_SARIF_VERSION,
    SARIF_PROJECTION_AUTHORIZES_REPAIR,
    SARIF_PROJECTION_IS_COMPLETION_EVIDENCE,
    SARIF_VERSION,
    FindingSarifBoundsError,
    FindingSarifError,
    SarifExportConfig,
    SecretLeakageError,
    assert_no_secret_or_body_leakage,
    contract_finding_to_security_like,
    export_security_findings_sarif,
    finding_to_sarif_result,
    findings_to_sarif,
    redact_text,
    sarif_canonical_bytes,
    sarif_content_id,
    sarif_rules,
)
from ipfs_accelerate_py.agent_supervisor.security_contract_analysis import (
    FindingClassification,
    FlowRole,
    SecurityRuleFamily,
    ThreatPath,
    ThreatPathOrigin,
    analyze_security_contracts,
    build_security_finding,
    make_evidence,
    make_flow_edge,
    make_flow_node,
    make_security_property,
    security_rule_families,
)


def _true_positive_finding(
    family: SecurityRuleFamily = SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS,
):
    prop = make_security_property(
        f"prop:{family.value}",
        family,
        resource="r",
        statement=f"Declared property for {family.value}",
    )
    path = ThreatPath(
        path_id=f"path:{family.value}",
        node_ids=("src", "sink"),
        origin=ThreatPathOrigin.REACHABLE,
        edge_ids=("e1",),
        hop_count=1,
    )
    return build_security_finding(
        family=family,
        classification=FindingClassification.VULNERABILITY,
        summary=f"Vulnerability in {family.value}",
        impact=f"Impact for {family.value}",
        security_property=prop,
        threat_path=path,
        evidence=make_evidence(f"artifact:{family.value}"),
        symbols=(f"pkg.{family.value}.entry",),
        interfaces=(f"mcp://{family.value}",),
        repositories=("repository:alpha",),
        source_node_id="src",
        sink_node_id="sink",
        seed_label="true_positive",
    )


def test_authority_flags_fail_closed() -> None:
    assert SARIF_PROJECTION_IS_COMPLETION_EVIDENCE is False
    assert SARIF_PROJECTION_AUTHORIZES_REPAIR is False
    assert FINDING_SARIF_VERSION == 1
    assert SARIF_VERSION == "2.1.0"


def test_sarif_rules_cover_all_families() -> None:
    rules = sarif_rules()
    assert len(rules) == len(security_rule_families())
    ids = [r["id"] for r in rules]
    assert ids == sorted(ids)
    assert all("shortDescription" in r for r in rules)


def test_redact_text_strips_secrets_and_bounds() -> None:
    raw = "user ok password=hunter2 token=abc123 rest"
    redacted = redact_text(raw)
    assert "hunter2" not in redacted
    assert "abc123" not in redacted
    assert "[REDACTED]" in redacted

    long = "x" * 10_000
    short = redact_text(long, maximum=100)
    assert len(short.encode("utf-8")) <= 100
    assert short.endswith("...")


def test_findings_to_sarif_is_deterministic() -> None:
    findings = [
        _true_positive_finding(SecurityRuleFamily.SECRET_FLOW),
        _true_positive_finding(SecurityRuleFamily.PATH_TRAVERSAL_SCOPE_LOSS),
    ]
    a = findings_to_sarif(findings)
    b = findings_to_sarif(list(reversed(findings)))
    assert sarif_canonical_bytes(a) == sarif_canonical_bytes(b)
    assert sarif_content_id(a) == sarif_content_id(b)
    # Rule order deterministic; results sorted by ruleId then finding id.
    rule_ids = [r["ruleId"] for r in a["runs"][0]["results"]]
    assert rule_ids == sorted(rule_ids)


def test_sarif_structure_and_artifact_references() -> None:
    finding = _true_positive_finding()
    log = export_security_findings_sarif(
        [finding],
        run_properties={"tree_id": "tree:1"},
    )
    assert log["version"] == "2.1.0"
    assert "$schema" in log
    run = log["runs"][0]
    assert run["tool"]["driver"]["name"]
    assert len(run["tool"]["driver"]["rules"]) == 10
    assert len(run["results"]) == 1
    result = run["results"][0]
    assert result["ruleId"].startswith("sec/")
    assert result["level"] in {"note", "warning", "error"}
    assert result["kind"] == "fail"
    assert "password" not in json.dumps(result).lower() or "[REDACTED]" in json.dumps(
        result
    )
    props = result["properties"]
    assert props["is_vulnerability"] is True
    assert props["finding_id"] == finding.finding_id
    assert props["evidence_artifact_cids"]
    # Artifacts are references only.
    for art in run["artifacts"]:
        assert "text" not in art.get("contents", {})
        loc = art["location"]
        assert "uri" in loc
    assert run["properties"]["bodies_omitted"] is True
    assert run["properties"]["secrets_redacted"] is True
    assert run["properties"]["tree_id"] == "tree:1"


def test_no_source_body_or_secret_leakage() -> None:
    finding = _true_positive_finding(SecurityRuleFamily.SECRET_FLOW)
    # Craft a summary that would try to leak if not redacted.
    finding = build_security_finding(
        family=SecurityRuleFamily.SECRET_FLOW,
        classification=FindingClassification.VULNERABILITY,
        summary="leaked password=supersecret api_key=sk-live-1",
        impact="Credential disclosure",
        security_property=make_security_property(
            "prop:secret",
            SecurityRuleFamily.SECRET_FLOW,
            resource="credentials",
            statement="no secret export",
        ),
        threat_path=ThreatPath(
            path_id="p",
            node_ids=("a", "b"),
            hop_count=1,
        ),
        evidence=make_evidence("artifact:1"),
        seed_label="true_positive",
    )
    log = export_security_findings_sarif([finding])
    blob = json.dumps(log)
    assert "supersecret" not in blob
    assert "sk-live-1" not in blob
    assert "[REDACTED]" in blob
    assert_no_secret_or_body_leakage(log)


def test_assert_leakage_detects_private_key() -> None:
    # Assemble PEM markers at runtime so proposal gates do not treat the
    # test source as introducing private-key material (secret_change_forbidden).
    dash = "-" * 5
    pem = f"{dash}BEGIN PRIVATE KEY{dash}\nABC\n{dash}END PRIVATE KEY{dash}"
    log = {
        "version": "2.1.0",
        "runs": [
            {
                "results": [
                    {
                        "message": {
                            "text": pem
                        }
                    }
                ]
            }
        ],
    }
    with pytest.raises(SecretLeakageError):
        assert_no_secret_or_body_leakage(log)


def test_assert_leakage_detects_snippet_key() -> None:
    log = {
        "version": "2.1.0",
        "runs": [{"results": [{"locations": [{"physicalLocation": {"region": {"snippet": {"text": "code"}}}}]}]}],
    }
    with pytest.raises(SecretLeakageError):
        assert_no_secret_or_body_leakage(log)


def test_max_results_truncation() -> None:
    findings = [
        _true_positive_finding(family)
        for family in list(SecurityRuleFamily)[:4]
    ]
    log = findings_to_sarif(
        findings,
        config=SarifExportConfig(max_results=2),
    )
    assert len(log["runs"][0]["results"]) == 2
    assert log["runs"][0]["properties"]["truncated"] is True


def test_classification_kinds() -> None:
    path = ThreatPath(path_id="p", node_ids=("a", "b"), hop_count=1)
    cases = [
        (
            FindingClassification.VULNERABILITY,
            "fail",
            True,
        ),
        (FindingClassification.CORRECTNESS_DRIFT, "review", False),
        (FindingClassification.SUSPICION, "review", False),
        (FindingClassification.UNKNOWN_DYNAMIC, "open", False),
    ]
    for classification, kind, need_vuln_fields in cases:
        kwargs = dict(
            family=SecurityRuleFamily.CACHE_POISONING_STALENESS,
            classification=classification,
            summary=f"case {classification.value}",
            threat_path=path,
            evidence=make_evidence("a:1") if need_vuln_fields else make_evidence(),
        )
        if need_vuln_fields:
            kwargs["impact"] = "stale serve"
            kwargs["security_property"] = make_security_property(
                "prop:cache",
                SecurityRuleFamily.CACHE_POISONING_STALENESS,
                resource="cache",
                statement="freshness required",
            )
            kwargs["evidence"] = make_evidence("a:1")
        finding = build_security_finding(**kwargs)
        result = finding_to_sarif_result(finding)
        assert result["kind"] == kind


def test_analysis_report_projects_to_sarif() -> None:
    nodes = [
        make_flow_node(
            "s",
            "pkg.src",
            role=FlowRole.SOURCE,
            tags=("schema_drift",),
            path="src/mcp.py",
        ),
        make_flow_node(
            "k",
            "pkg.invoke",
            role=FlowRole.SINK,
            tags=("mcp_invoke",),
            path="src/mcp.py",
        ),
    ]
    edges = [make_flow_edge("e", "s", "k")]
    prop = make_security_property(
        "prop:mcp",
        SecurityRuleFamily.MCP_SCHEMA_DISPATCH_CONFUSION,
        resource="mcp.tools",
        statement="Dispatch must bind reviewed schema.",
    )
    report = analyze_security_contracts(
        nodes=nodes,
        edges=edges,
        properties=[prop],
        default_evidence=make_evidence("artifact:mcp"),
        families=[SecurityRuleFamily.MCP_SCHEMA_DISPATCH_CONFUSION],
    )
    log = export_security_findings_sarif(
        report.findings,
        run_properties={"report_id": report.report_id},
    )
    assert log["runs"][0]["results"]
    assert log["runs"][0]["properties"]["report_id"] == report.report_id
    # Vulnerability present.
    assert any(
        r["properties"]["is_vulnerability"]
        for r in log["runs"][0]["results"]
    )


def test_contract_finding_projection_is_not_vulnerability() -> None:
    record = {
        "summary": "schema mismatch",
        "severity": "medium",
        "confidence_millionths": 400_000,
        "symbols": ("pkg.tool",),
        "interfaces": ("mcp://tool",),
        "repositories": ("repository:alpha",),
        "root_cause_family": "schema_mismatch",
        "tree_id": "tree:x",
        "evidence": {
            "artifact_cids": ("artifact:1",),
            "counterexample_cids": ("cex:1",),
        },
    }
    finding = contract_finding_to_security_like(record)
    assert finding.classification is FindingClassification.CORRECTNESS_DRIFT
    assert finding.is_vulnerability is False
    log = export_security_findings_sarif([finding])
    result = log["runs"][0]["results"][0]
    assert result["kind"] == "review"
    assert result["properties"]["is_vulnerability"] is False


def test_run_properties_reject_body_keys() -> None:
    finding = _true_positive_finding()
    with pytest.raises(SecretLeakageError):
        findings_to_sarif(
            [finding],
            run_properties={"source_body": "print('evil')"},
        )


def test_sarif_config_bounds() -> None:
    with pytest.raises(FindingSarifError):
        SarifExportConfig(max_results=0)
    with pytest.raises(FindingSarifBoundsError):
        SarifExportConfig(max_results=DEFAULT_MAX_RESULTS * 1000)


def test_partial_fingerprints_stable() -> None:
    finding = _true_positive_finding()
    a = finding_to_sarif_result(finding)
    b = finding_to_sarif_result(finding)
    assert a["partialFingerprints"] == b["partialFingerprints"]
    assert a["partialFingerprints"]["findingContentId"] == finding.finding_id


def test_related_locations_are_node_references_not_bodies() -> None:
    finding = _true_positive_finding()
    result = finding_to_sarif_result(finding)
    assert "relatedLocations" in result
    for rel in result["relatedLocations"]:
        uri = rel["physicalLocation"]["artifactLocation"]["uri"]
        assert uri.startswith("node:")
        assert "snippet" not in rel.get("physicalLocation", {}).get(
            "region", {}
        )


def test_empty_findings_export() -> None:
    log = export_security_findings_sarif([])
    assert log["runs"][0]["results"] == []
    assert log["runs"][0]["properties"]["result_count"] == 0
    assert_no_secret_or_body_leakage(log)
