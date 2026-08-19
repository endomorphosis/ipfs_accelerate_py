"""FACP-019: repository-wide ambiguous-claim scanner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.formal_claim_scanner import (
    FORBIDDEN_GENERIC_FIELDS,
    TYPED_COMPATIBILITY_ALIASES,
    AllowlistEntry,
    AmbiguousClaimAllowlist,
    ClaimKind,
    FindingDisposition,
    FormalClaimScannerError,
    SCHEMA,
    TASK_ID,
    apply_allowlist,
    classify_field_name,
    describe_compatibility_alias,
    findings_for_corpus_entry,
    load_defect_corpus,
    repair_family_for_field,
    scan_python_source,
    scan_seeded_corpus,
    scan_tree,
)


def _repo_root() -> Path:
    # test/api -> test -> external/ipfs_accelerate -> workspace root
    return Path(__file__).resolve().parents[4]


def _default_corpus_path() -> Path:
    return (
        _repo_root()
        / "implementation_plan"
        / "formal_assurance_control_plane"
        / "baseline"
        / "defect_corpus.jsonl"
    )


SEED_SOURCE = '''\
"""Seeded ambiguous-claim fixture for FACP-019."""


def register_endpoint(name: str) -> dict:
    api_available = True
    payload = {
        "success": True,
        "available": True,
        "supported": True,
        "verified": True,
        "proven": True,
        # Typed FCA predicates must remain non-ambiguous.
        "production_supported": True,
        "effect_successful": False,
        "proof_reusable": False,
    }
    return report_status(payload, api_available=api_available)


def report_status(payload: dict, *, api_available: bool = False) -> dict:
    return payload


def naming_only_demo(success, available):
    """Parameter names alone must not become defects."""
    helper = success
    return helper


def mock_capability_probe() -> dict:
    return {"capability": True, "mock": True, "cid": "deadbeef"}
'''


def test_classify_distinguishes_forbidden_fields_from_typed_aliases() -> None:
    for name in sorted(FORBIDDEN_GENERIC_FIELDS):
        assert classify_field_name(name) is ClaimKind.FORBIDDEN_GENERIC
    for name in (
        "production_supported",
        "effect_successful",
        "proof_reusable",
        "receipt_authoritative",
        "release_admissible",
        "proof.verified",
        "origin.live_observed",
        "policy.allowed",
    ):
        assert name in TYPED_COMPATIBILITY_ALIASES or classify_field_name(name) is (
            ClaimKind.TYPED_COMPATIBILITY_ALIAS
        )
        assert classify_field_name(name) is ClaimKind.TYPED_COMPATIBILITY_ALIAS
        meta = describe_compatibility_alias(name)
        assert meta is not None
        assert meta["ambiguous"] is False


def test_naming_alone_is_not_a_defect() -> None:
    source = (
        "def demo(success, available, verified):\n"
        "    helper = success\n"
        "    return available\n"
    )
    findings = scan_python_source(source, path="naming_only.py")
    assert findings == ()


def test_scanner_finds_seeded_claim_bindings_with_trace_and_repair_family(
    tmp_path: Path,
) -> None:
    module = tmp_path / "pkg" / "seeded_api.py"
    module.parent.mkdir(parents=True)
    module.write_text(SEED_SOURCE, encoding="utf-8")

    report = scan_tree(tmp_path, relative_paths=["pkg/seeded_api.py"])
    assert report.schema == SCHEMA
    assert report.scanned_paths == ("pkg/seeded_api.py",)

    reject = report.reject_findings
    fields = {item.field_name for item in reject}
    assert "success" in fields
    assert "available" in fields
    assert "supported" in fields
    assert "verified" in fields
    assert "proven" in fields
    assert "api_available" in fields
    assert "capability" in fields
    assert "cid" in fields
    # Typed aliases must not appear as reject findings.
    assert "production_supported" not in fields
    assert "effect_successful" not in fields
    assert "proof_reusable" not in fields

    for item in reject:
        assert item.source_span.path == "pkg/seeded_api.py"
        assert item.source_span.start_line >= 1
        assert item.abstract_trace.steps
        assert item.abstract_trace.summary
        assert item.repair_family
        assert item.disposition in {
            FindingDisposition.REJECT,
            FindingDisposition.CORPUS_BOUND,
        }

    success = next(item for item in reject if item.field_name == "success")
    assert success.repair_family == "false_success"
    assert any(step.kind == "claim_site" for step in success.abstract_trace.steps)


def test_typed_compatibility_aliases_are_preserved_in_seed_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "aliases.py"
    path.write_text(
        "\n".join(
            [
                "def gate(envelope):",
                "    return {",
                '        "production_supported": True,',
                '        "effect_successful": True,',
                '        "proof.verified": True,',
                '        "success": True,',
                "    }",
                "",
            ]
        ),
        encoding="utf-8",
    )
    findings = scan_python_source(path.read_text(encoding="utf-8"), path="aliases.py")
    assert {item.field_name for item in findings} == {"success"}
    assert all(item.field_name != "production_supported" for item in findings)


def test_allowlist_suppresses_noise_but_cannot_suppress_corpus_defects(
    tmp_path: Path,
) -> None:
    noisy = tmp_path / "noise.py"
    noisy.write_text(
        'def demo():\n    return {"success": True, "available": True}\n',
        encoding="utf-8",
    )
    report = scan_tree(tmp_path, relative_paths=["noise.py"])
    assert len(report.reject_findings) >= 2

    allowlist = AmbiguousClaimAllowlist(
        entries=(
            AllowlistEntry(
                entry_id="allow:noise-success",
                reason="fixture documentation example",
                path_suffix="noise.py",
                field_name="success",
            ),
        )
    )
    applied = apply_allowlist(report.findings, allowlist)
    by_field = {item.field_name: item for item in applied}
    assert by_field["success"].disposition is FindingDisposition.ALLOWLISTED
    assert by_field["success"].allowlist_entry_id == "allow:noise-success"
    assert by_field["available"].disposition is FindingDisposition.REJECT

    corpus_entry = {
        "seed_id": "seed:test-false-success",
        "defect_id": "defect:test-false-success",
        "family": "false_success",
        "title": "seeded success true",
        "roadmap_seed": True,
        "expected_illegal_promotion": "success:true -> live success",
        "source_spans": [
            {
                "path": "noise.py",
                "start_line": 2,
                "end_line": 2,
                "symbol": "demo",
                "excerpt": '"success": True',
            }
        ],
        "call_flow_path": ["demo", "public_api"],
    }
    corpus_findings = findings_for_corpus_entry(corpus_entry)
    assert len(corpus_findings) == 1
    corpus_finding = corpus_findings[0]
    assert corpus_finding.is_corpus_defect
    assert corpus_finding.repair_family == "false_success"
    assert corpus_finding.abstract_trace.steps
    assert any(step.kind == "repair_family" for step in corpus_finding.abstract_trace.steps)

    aggressive = AmbiguousClaimAllowlist(
        entries=(
            AllowlistEntry(
                entry_id="allow:suppress-everything",
                reason="illegal attempt to hide corpus defects",
                path_suffix="noise.py",
                field_name="success",
            ),
        )
    )
    assert aggressive.may_suppress(corpus_finding) is False
    after = apply_allowlist(corpus_findings, aggressive)
    assert len(after) == 1
    assert after[0].disposition is FindingDisposition.CORPUS_BOUND
    assert after[0].allowlist_entry_id == ""


def test_scan_seeded_corpus_emits_spans_traces_and_repair_families() -> None:
    corpus_path = _default_corpus_path()
    if not corpus_path.is_file():
        pytest.skip(f"defect corpus unavailable: {corpus_path}")

    seed_ids = [
        "seed:api-available-default-true",
        "seed:hardcoded-hwtest-true",
        "seed:mock-worker-cuda-true",
        "seed:raw-sha256-as-cid",
    ]
    # Attempt to allowlist corpus paths; scanner must still retain them.
    allowlist = AmbiguousClaimAllowlist(
        entries=(
            AllowlistEntry(
                entry_id="allow:corpus-should-not-apply",
                reason="corpus defects are not suppressible",
                path_suffix="ipfs_accelerate.py",
                field_name="success",
            ),
            AllowlistEntry(
                entry_id="allow:corpus-available",
                reason="corpus defects are not suppressible",
                path_suffix="ipfs_accelerate.py",
                field_name="available",
            ),
        )
    )
    report = scan_seeded_corpus(
        corpus_path=corpus_path,
        repo_root=_repo_root(),
        seed_ids=seed_ids,
        allowlist=allowlist,
    )

    assert report.to_dict()["task_id"] == TASK_ID
    bound = set(report.corpus_seed_ids_bound)
    assert set(seed_ids) <= bound

    by_seed: dict[str, list] = {}
    for finding in report.findings:
        assert finding.disposition is not FindingDisposition.ALLOWLISTED
        assert finding.is_corpus_defect
        assert finding.source_span.path
        assert finding.source_span.start_line >= 1
        assert finding.abstract_trace.steps
        assert finding.repair_family
        by_seed.setdefault(finding.corpus_seed_id, []).append(finding)

    assert by_seed["seed:api-available-default-true"][0].repair_family == "false_success"
    assert by_seed["seed:mock-worker-cuda-true"][0].repair_family == "mock_capability"
    assert by_seed["seed:raw-sha256-as-cid"][0].repair_family == "pseudo_cid"
    assert any(
        step.kind in {"seed_span", "claim_site", "call_flow", "repair_family"}
        for finding in report.findings
        for step in finding.abstract_trace.steps
    )


def test_load_defect_corpus_round_trip_schema() -> None:
    corpus_path = _default_corpus_path()
    if not corpus_path.is_file():
        pytest.skip(f"defect corpus unavailable: {corpus_path}")
    entries = load_defect_corpus(corpus_path)
    assert len(entries) >= 10
    assert all("seed_id" in entry and "family" in entry for entry in entries[:5])


def test_repair_family_mapping_covers_forbidden_fields() -> None:
    assert repair_family_for_field("success") == "false_success"
    assert repair_family_for_field("cid") == "pseudo_cid"
    assert (
        repair_family_for_field("available", context="MockWorker.test_hardware")
        == "mock_capability"
    )
    with pytest.raises(FormalClaimScannerError):
        repair_family_for_field("not_a_claim_field")


def test_scan_tree_allowlist_keeps_corpus_bound_findings(tmp_path: Path) -> None:
    module = tmp_path / "service.py"
    module.write_text(
        "def run():\n    return {'success': True}\n",
        encoding="utf-8",
    )
    corpus_entries = [
        {
            "seed_id": "seed:tmp-success",
            "defect_id": "defect:tmp-success",
            "family": "false_success",
            "title": "tmp success",
            "roadmap_seed": True,
            "source_spans": [
                {
                    "path": "service.py",
                    "start_line": 2,
                    "end_line": 2,
                    "symbol": "run",
                    "excerpt": "success",
                }
            ],
            "call_flow_path": ["run"],
        }
    ]
    allowlist = AmbiguousClaimAllowlist(
        entries=(
            AllowlistEntry(
                entry_id="allow:service-success",
                reason="should not hide corpus",
                path_suffix="service.py",
                field_name="success",
            ),
        )
    )
    report = scan_tree(
        tmp_path,
        relative_paths=["service.py"],
        allowlist=allowlist,
        corpus_entries=corpus_entries,
    )
    corpus_bound = [item for item in report.findings if item.is_corpus_defect]
    assert corpus_bound
    assert all(
        item.disposition is FindingDisposition.CORPUS_BOUND for item in corpus_bound
    )
    assert "seed:tmp-success" in report.corpus_seed_ids_bound
    payload = report.to_dict()
    assert payload["schema"] == SCHEMA
    assert payload["finding_count"] == len(report.findings)
    json.dumps(payload)  # report must be JSON-serializable
