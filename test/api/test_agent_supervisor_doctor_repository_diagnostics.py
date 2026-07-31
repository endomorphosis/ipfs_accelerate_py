"""Tests for deterministic doctor repository diagnostics (LPR-030)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (
    AuthorityRoots,
    BrokenContractTrace,
    EvidenceReference,
    SourceSpan,
    TraceDisposition,
)
from ipfs_accelerate_py.agent_supervisor.analysis.doctor_repository_diagnostics import (
    SUPPORTED_ADAPTER_LANGUAGES,
    DoctorAuthorityRoots,
    DoctorDiagnosticFinding,
    DoctorDiagnosticInput,
    DoctorDiagnosticsAuthorityError,
    DoctorDiagnosticsBoundsError,
    DoctorDiagnosticsError,
    DoctorDiagnosticsMixedRootError,
    DoctorDiagnosticsStaleError,
    DoctorDiagnosticsSymlinkError,
    DoctorEvidenceCompiler,
    DoctorSnapshotPolicy,
    DoctorSourceUnit,
    ExpectationSourceKind,
    FindingDisposition,
    FindingKind,
    QuerySurface,
    StructuredValidationFailure,
    compile_doctor_evidence_snapshot,
    diagnose_repository,
)


def _roots(**overrides: str) -> DoctorAuthorityRoots:
    base = {
        "repository_id": "repository:doctor-test",
        "forest_id": "forest:one",
        "tree_id": "tree:one",
        "config_id": "config:one",
        "toolchain_id": "toolchain:deterministic-doctor@1",
        "policy_id": "policy:doctor-test",
        "parser_id": "parser:program-ast-adapters@1",
    }
    base.update(overrides)
    return DoctorAuthorityRoots(**base)


def _python_pair() -> list[tuple[str, str]]:
    return [
        (
            "src/service.py",
            """\
from typing import Protocol
import transport.client as client

class ServiceContract(Protocol):
    def dispatch(self, request: str) -> str: ...

def dispatch(request: str) -> str:
    wire = client.Client()
    return wire.send(request)
""",
        ),
        (
            "src/consumer.py",
            """\
from src.service import dispatch

def consume(payload: str) -> str:
    return dispatch(payload)
""",
        ),
    ]


def _mixed_sources() -> list[dict[str, object]]:
    return [
        {
            "path": "src/service.py",
            "source": "def run(value):\n    return value\n",
            "language": "python",
        },
        {
            "path": "src/adapter.ts",
            "source": "export function run(value: string): string { return value; }\n",
            "language": "typescript",
        },
        {
            "path": "config/tool.json",
            "source": '{"name": "tool", "type": "object"}\n',
            "language": "json",
        },
        {
            "path": "docs/note.md",
            "source": "# Note\n\nMust validate callers.\n",
            "language": "markdown",
        },
    ]


def test_compile_parses_python_and_supported_adapters_as_inert_bytes() -> None:
    snapshot = diagnose_repository(
        _mixed_sources(),
        authority_roots=_roots(),
    )

    assert snapshot.provider_call_count == 0
    assert snapshot.source_write_count == 0
    assert snapshot.completeness["path_count"] == 4
    assert set(snapshot.completeness["languages"]) >= {
        "python",
        "typescript",
        "json",
        "markdown",
    }
    assert set(SUPPORTED_ADAPTER_LANGUAGES) >= set(snapshot.completeness["languages"])
    # Durable identity/finding payloads never embed raw source body fields.
    identity = snapshot._identity_payload()

    def _assert_no_body_fields(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key).lower().replace("-", "_")
                assert key_text not in {
                    "body",
                    "source",
                    "source_body",
                    "source_text",
                    "source_bytes",
                    "contents",
                    "snippet",
                    "file_text",
                    "raw_ast",
                    "ast_body",
                }
                _assert_no_body_fields(item)
        elif isinstance(value, list):
            for item in value:
                _assert_no_body_fields(item)

    _assert_no_body_fields(identity)
    for finding in snapshot.findings:
        _assert_no_body_fields(finding.to_dict())
    assert snapshot.authority_roots.parser_id
    assert snapshot.authority_roots.toolchain_id
    assert snapshot.authority_roots.ast_index_id == snapshot.ast_index.index_id
    assert snapshot.authority_roots.config_id
    assert snapshot.snapshot_cid.startswith("b")


def test_query_imports_exports_aliases_wrappers_entry_points_and_call_sites() -> None:
    snapshot = diagnose_repository(_python_pair(), authority_roots=_roots())

    imports = snapshot.query(QuerySurface.IMPORTS)
    assert any(hit.target.startswith("transport.client") for hit in imports.hits)
    assert any(hit.name == "dispatch" for hit in imports.hits)

    aliases = snapshot.query(QuerySurface.ALIASES)
    assert any(hit.name == "client" for hit in aliases.hits)

    exports = snapshot.query(QuerySurface.EXPORTS)
    assert any(hit.name == "dispatch" for hit in exports.hits)
    assert any(hit.name == "consume" for hit in exports.hits)

    entry_points = snapshot.query(QuerySurface.ENTRY_POINTS)
    assert {hit.name for hit in entry_points.hits} >= {"dispatch", "consume", "ServiceContract"}

    call_sites = snapshot.query(QuerySurface.CALL_SITES)
    assert any("send" in hit.target or "send" in hit.name for hit in call_sites.hits)

    wrappers = snapshot.query(QuerySurface.WRAPPERS)
    assert wrappers.surface is QuerySurface.WRAPPERS

    # Path-filtered query
    only_service = snapshot.query(QuerySurface.EXPORTS, path="src/service.py")
    assert all(hit.path == "src/service.py" for hit in only_service.hits)
    assert any(hit.name == "dispatch" for hit in only_service.hits)


def test_expectation_source_and_precedence_separate_from_observations() -> None:
    roots = AuthorityRoots(
        repository_id="repository:doctor-test",
        forest_id="forest:one",
        tree_id="tree:one",
        graph_id="graph:one",
        index_id="index:one",
        model_id="model:none",
        config_id="config:one",
        translator_id="translator:one",
        toolchain_id="toolchain:one",
        policy_id="policy:one",
    )
    evidence = EvidenceReference(
        "resolver_receipt", "evidence:resolver", "call:send", "test"
    )
    trace = BrokenContractTrace(
        roots=roots,
        caller_span=SourceSpan("src/service.py", 10, 40, "blob:service"),
        caller_symbol_id="symbol:dispatch",
        receiver_reference="client.Client",
        disposition=TraceDisposition.RESOLVED_MISMATCH,
        target_span=SourceSpan("transport/client.py", 1, 20, "blob:client"),
        evidence_refs=(evidence,),
    )
    failure = StructuredValidationFailure(
        failure_id="val:arity-1",
        kind="call_arity",
        path="src/service.py",
        symbol="dispatch",
        message="expected two arguments",
        expectation_source=ExpectationSourceKind.REVIEWED_CONTRACT,
        expectation_ref="contract:dispatch@2",
        observed_ref="fact:observed-dispatch",
    )

    snapshot = diagnose_repository(
        _python_pair(),
        authority_roots=_roots(),
        broken_traces=(trace,),
        validation_failures=(failure,),
        expectation_refs=("contract:dispatch@2",),
    )

    trace_findings = [
        item for item in snapshot.findings if item.kind is FindingKind.TRACE_JOIN
    ]
    assert len(trace_findings) == 1
    joined = trace_findings[0]
    assert joined.expectation_source is ExpectationSourceKind.BROKEN_TRACE
    assert joined.expectation_precedence == 100
    assert joined.expectation_ref == trace.content_id
    assert joined.observation_refs  # joined to current AST/evidence facts
    assert "client.Client" in joined.message

    validation_findings = [
        item for item in snapshot.findings if item.kind is FindingKind.VALIDATION_JOIN
    ]
    assert len(validation_findings) == 1
    val = validation_findings[0]
    assert val.expectation_source is ExpectationSourceKind.REVIEWED_CONTRACT
    assert val.expectation_precedence == 200
    assert val.expectation_ref == "contract:dispatch@2"
    assert "fact:observed-dispatch" in val.observation_refs
    # Observations never rewrite expectation source.
    assert val.expectation_source is not ExpectationSourceKind.NONE


def test_open_frontiers_for_python_only_and_unsupported_analyses() -> None:
    source = """\
import ctypes

def patch(target, name, value):
    setattr(target, name, value)

def run():
    try:
        return ctypes.CDLL("libx.so")
    except OSError:
        return None
"""
    snapshot = diagnose_repository(
        [("src/native.py", source)],
        authority_roots=_roots(),
    )
    frontiers = set(snapshot.open_frontiers)
    assert "frontier:reflection" in frontiers
    assert "frontier:exception_propagation" in frontiers
    assert "frontier:cfg_control_flow" in frontiers
    assert "frontier:native_ffi" in frontiers
    assert "frontier:concurrency" in frontiers
    assert "frontier:interprocedural_dataflow" in frontiers
    frontier_query = snapshot.query(QuerySurface.FRONTIERS)
    assert {hit.name for hit in frontier_query.hits} == frontiers


def test_findings_issue_canonical_cids_and_are_stable() -> None:
    first = diagnose_repository(_python_pair(), authority_roots=_roots())
    second = diagnose_repository(list(reversed(_python_pair())), authority_roots=_roots())

    assert first.snapshot_cid == second.snapshot_cid
    assert first.snapshot_id == second.snapshot_id
    assert first.finding_cids == second.finding_cids
    assert all(cid.startswith("b") for cid in first.finding_cids)
    for finding in first.findings:
        rebuilt = DoctorDiagnosticFinding(
            kind=finding.kind,
            disposition=finding.disposition,
            path=finding.path,
            symbol=finding.symbol,
            message=finding.message,
            observation_refs=finding.observation_refs,
            expectation_source=finding.expectation_source,
            expectation_ref=finding.expectation_ref,
            expectation_precedence=finding.expectation_precedence,
            open_frontier_refs=finding.open_frontier_refs,
            evidence_refs=finding.evidence_refs,
            details=dict(finding.details),
        )
        assert rebuilt._payload() == finding._payload()
        assert rebuilt.finding_cid == finding.finding_cid
        assert first.finding_for_cid(finding.finding_cid) == finding


def test_incremental_invalidation_identity_equivalent_to_clean_rebuild() -> None:
    sources = _python_pair()
    clean = compile_doctor_evidence_snapshot(
        DoctorDiagnosticInput(sources=tuple(sources), authority_roots=_roots())
    )
    incremental = compile_doctor_evidence_snapshot(
        DoctorDiagnosticInput(sources=tuple(sources), authority_roots=_roots()),
        previous=clean,
    )
    assert incremental.rebuild_mode == "incremental"
    assert clean.rebuild_mode == "clean"
    assert incremental.snapshot_cid == clean.snapshot_cid
    assert incremental.snapshot_id == clean.snapshot_id
    assert incremental.ast_index.index_id == clean.ast_index.index_id
    assert incremental.finding_cids == clean.finding_cids

    # Change one blob; identity must move, and warm reuse still fails closed to new facts.
    changed = [
        sources[0],
        (
            "src/consumer.py",
            "from src.service import dispatch\n\ndef consume(payload: str, limit: int) -> str:\n    return dispatch(payload)\n",
        ),
    ]
    rebuilt = compile_doctor_evidence_snapshot(
        DoctorDiagnosticInput(sources=tuple(changed), authority_roots=_roots()),
        previous=clean,
    )
    assert rebuilt.snapshot_cid != clean.snapshot_cid
    assert rebuilt.ast_index.index_id != clean.ast_index.index_id


def test_malformed_oversized_symlink_stale_mixed_root_fail_closed(
    tmp_path: Path,
) -> None:
    # Oversized source
    policy = DoctorSnapshotPolicy(max_source_bytes=32, max_total_bytes=64)
    with pytest.raises(DoctorDiagnosticsBoundsError):
        diagnose_repository(
            [("src/big.py", "x" * 100)],
            authority_roots=_roots(),
            policy=policy,
        )

    # Malformed path escape
    with pytest.raises(DoctorDiagnosticsAuthorityError):
        DoctorSourceUnit(path="../escape.py", source_bytes=b"print(1)\n")

    # Mixed roots without permission
    with pytest.raises(DoctorDiagnosticsMixedRootError):
        diagnose_repository(
            [
                DoctorSourceUnit(
                    path="a.py", source_bytes=b"def a():\n    return 1\n", root_id="root:a"
                ),
                DoctorSourceUnit(
                    path="b.py", source_bytes=b"def b():\n    return 2\n", root_id="root:b"
                ),
            ],
            authority_roots=_roots(),
            policy=DoctorSnapshotPolicy(allow_mixed_roots=False),
        )

    # Stale tree binding
    with pytest.raises(DoctorDiagnosticsStaleError):
        diagnose_repository(
            _python_pair(),
            authority_roots=_roots(tree_id="tree:current"),
            claimed_tree_id="tree:stale",
        )

    # Symlink escape under repository_root
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("def leaked():\n    return 1\n", encoding="utf-8")
    link = root / "leaked.py"
    link.symlink_to(outside)
    with pytest.raises(DoctorDiagnosticsSymlinkError):
        diagnose_repository(
            [DoctorSourceUnit(path="leaked.py", source_bytes=b"")],
            repository_root=str(root),
            authority_roots=_roots(),
        )

    # Provider calls / source writes rejected
    with pytest.raises(DoctorDiagnosticsError):
        DoctorDiagnosticInput(
            sources=(("a.py", "x=1\n"),),
            authority_roots=_roots(),
            provider_call_count=1,
        )
    with pytest.raises(DoctorDiagnosticsError):
        DoctorDiagnosticInput(
            sources=(("a.py", "x=1\n"),),
            authority_roots=_roots(),
            source_write_count=1,
        )

    # Successful path under repository root reads inert bytes only
    good = root / "ok.py"
    good.write_text("def ok():\n    return 1\n", encoding="utf-8")
    before = good.read_text(encoding="utf-8")
    snapshot = diagnose_repository(
        [DoctorSourceUnit(path="ok.py", source_bytes=b"")],
        repository_root=str(root),
        authority_roots=_roots(),
    )
    assert snapshot.completeness["path_count"] == 1
    assert good.read_text(encoding="utf-8") == before
    assert snapshot.provider_call_count == 0
    assert snapshot.source_write_count == 0


def test_compiler_class_and_mapping_input_round_trip() -> None:
    compiler = DoctorEvidenceCompiler(DoctorSnapshotPolicy(policy_id="policy:map"))
    snapshot = compiler.diagnose(
        {
            "sources": _python_pair(),
            "authority_roots": _roots().to_dict(),
            "expectation_refs": ["contract:x"],
        }
    )
    assert snapshot.policy.policy_id == "policy:map"
    assert snapshot.authority_roots.repository_id == "repository:doctor-test"
    assert "contract:x"  # expectation refs accepted without inventing behavior
    assert snapshot.completeness["expectation_ref_count"] == 1
    # Query findings surface
    findings = snapshot.query("findings")
    assert findings.surface is QuerySurface.FINDINGS
    assert findings.count == len(findings.hits)


def test_syntax_malformed_python_emits_finding_without_writes() -> None:
    snapshot = diagnose_repository(
        [("src/broken.py", "def broken(\n")],
        authority_roots=_roots(),
    )
    assert any(item.kind is FindingKind.SYNTAX for item in snapshot.findings)
    assert snapshot.completeness["adapter_malformed_count"] >= 1
    assert snapshot.source_write_count == 0


def test_authority_roots_bind_from_contract_repair_roots() -> None:
    repair_roots = AuthorityRoots(
        repository_id="repository:doctor-test",
        forest_id="forest:one",
        tree_id="tree:one",
        graph_id="graph:one",
        index_id="index:one",
        model_id="model:none",
        config_id="config:one",
        translator_id="translator:one",
        toolchain_id="toolchain:one",
        policy_id="policy:one",
    )
    bound = DoctorAuthorityRoots.from_mapping(repair_roots)
    assert bound.repository_id == "repository:doctor-test"
    assert bound.ast_index_id == "index:one"
    assert bound.dependency_graph_id == "graph:one"
    assert bound.translator_id == "translator:one"
