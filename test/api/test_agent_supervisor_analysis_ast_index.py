from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis_ast_index import (
    ASTEvidenceIndex,
    ASTEvidenceKind,
    AnalysisASTIndex,
    AnalysisASTIndexError,
    build_analysis_ast_index,
)
from ipfs_accelerate_py.agent_supervisor.conflict_graph import (
    ASTBlobRecord,
    build_python_ast_blob_record,
)


def _record(
    source: str,
    blob: str,
) -> ASTBlobRecord:
    # Parsing belongs to the canonical conflict-graph boundary.  The index
    # receives only the resulting ASTBlobRecord.
    return build_python_ast_blob_record(source, blob_identity=blob)


def _snapshot() -> list[tuple[str, ASTBlobRecord]]:
    return [
        (
            "src/consumer.py",
            _record(
                """from src.service import Service

def consume(request):
    service = Service()
    return service.dispatch(request)
""",
                "blob:consumer-v1",
            ),
        ),
        (
            "src/service.py",
            _record(
                """from typing import Protocol

class ServiceContract(Protocol):
    def dispatch(self, request): ...

class Service:
    def dispatch(self, request):
        self.status = "running"
        return request
""",
                "blob:service-v1",
            ),
        ),
    ]


def test_index_is_deterministic_and_round_trips_canonical_records() -> None:
    forward = build_analysis_ast_index(_snapshot())
    reverse = build_analysis_ast_index(reversed(_snapshot()))

    assert forward.index_id == reverse.index_id
    assert forward.paths == ("src/consumer.py", "src/service.py")
    assert [item.to_dict() for item in forward.path_records] == [
        item.to_dict() for item in reverse.path_records
    ]
    assert AnalysisASTIndex.from_json(forward.to_json()) == forward
    assert ASTEvidenceIndex is AnalysisASTIndex

    serialized = forward.to_dict()
    assert all(
        "ast_record" in item and "source" not in item["ast_record"]
        for item in serialized["path_records"]
    )


def test_incremental_build_reuses_the_exact_unchanged_blob_record() -> None:
    original = build_analysis_ast_index(_snapshot())
    # Deserialization creates equivalent but distinct candidate objects.  A
    # warm build must select the object already held in its canonical cache.
    candidates = [
        (item.path, ASTBlobRecord.from_dict(item.ast_record.to_dict()))
        for item in original.path_records
    ]
    assert candidates[0][1] is not original.path_records[0].ast_record

    warm = build_analysis_ast_index(candidates, previous=original)

    assert warm.index_id == original.index_id
    assert warm.stats.reused_blob_count == 2
    assert warm.stats.new_blob_count == 0
    assert warm.stats.cache_hit_ratio == 1.0
    assert all(
        current.ast_record is previous.ast_record
        for current, previous in zip(warm.path_records, original.path_records)
    )
    assert warm.invalidations == ()


def test_changed_and_deleted_paths_remove_stale_evidence_and_emit_receipts() -> None:
    original = build_analysis_ast_index(_snapshot())
    changed_service = _record(
        """class Service:
    def execute(self, request):
        return request
""",
        "blob:service-v2",
    )

    rebuilt = build_analysis_ast_index(
        [("src/service.py", changed_service)],
        previous=original,
    )

    assert rebuilt.stats.changed_path_count == 1
    assert rebuilt.stats.deleted_path_count == 1
    assert rebuilt.stats.invalidated_blob_count == 2
    assert {item.reason for item in rebuilt.invalidations} == {
        "blob_changed",
        "path_deleted",
    }
    assert {
        "blob:consumer-v1",
        "blob:service-v1",
    }.issubset(rebuilt.invalidated_blob_ids)
    assert rebuilt.query_definitions("dispatch").evidence == ()
    assert [item.symbol for item in rebuilt.query_definitions("execute").evidence] == [
        "Service.execute"
    ]


def test_rename_reuses_blob_without_invalidating_immutable_record() -> None:
    record = _snapshot()[1][1]
    original = build_analysis_ast_index([("src/service.py", record)])

    renamed = build_analysis_ast_index(
        [("src/runtime/service.py", ASTBlobRecord.from_dict(record.to_dict()))],
        previous=original,
    )

    assert renamed.stats.reused_blob_count == 1
    assert renamed.stats.renamed_path_count == 1
    assert renamed.stats.invalidated_blob_count == 0
    assert renamed.invalidations == ()
    assert renamed.path_records[0].ast_record is original.path_records[0].ast_record
    assert renamed.query_paths("runtime").evidence[0].path == "src/runtime/service.py"


def test_symbol_definition_import_call_and_reference_relationship_lookup() -> None:
    index = build_analysis_ast_index(_snapshot())

    symbols = index.query_symbols("dispatch")
    assert [item.symbol for item in symbols.evidence] == [
        "Service.dispatch",
        "ServiceContract.dispatch",
    ]
    assert all(item.kind is ASTEvidenceKind.SYMBOL for item in symbols.evidence)
    assert all(item.relationship == "defines" for item in symbols.evidence)
    assert all(item.line_start > 0 for item in symbols.evidence)
    assert all(item.symbol_hash.startswith("sha256:") for item in symbols.evidence)

    definitions = index.query_definitions("Service.dispatch")
    assert definitions.evidence[0].target == "src.service.Service.dispatch"
    assert definitions.evidence[0].blob_identity == "blob:service-v1"
    assert definitions.evidence[0].source_sha256.startswith("sha256:")
    assert definitions.evidence[0].record_id.startswith("ast-sha256:")

    imports = index.query_imports("src.service")
    assert imports.evidence[0].relationship == "imports"
    assert imports.evidence[0].value == "from src.service import Service"

    calls = index.query_calls("service.dispatch")
    assert calls.evidence[0].relationship == "calls"
    assert calls.evidence[0].symbol == "consume"
    assert calls.evidence[0].target == "service.dispatch"

    references = index.query_references("Service")
    assert {item.relationship for item in references.evidence} == {
        "calls",
        "imports",
    }
    assert all(item.kind is ASTEvidenceKind.REFERENCE for item in references.evidence)


def test_objective_term_ranking_is_stable_explainable_and_compact() -> None:
    index = build_analysis_ast_index(_snapshot())

    first = index.query_objective_terms(
        ["service", "dispatch"], max_results=10, max_bytes=16_000
    )
    second = build_analysis_ast_index(reversed(_snapshot())).query_objective_terms(
        ["service", "dispatch"], max_results=10, max_bytes=16_000
    )

    assert first.to_dict() == second.to_dict()
    assert first.evidence
    assert [item.score for item in first.evidence] == sorted(
        (item.score for item in first.evidence), reverse=True
    )
    assert all(item.ranking_explanations for item in first.evidence)
    assert any(
        reason.startswith("term_overlap:")
        for item in first.evidence
        for reason in item.ranking_explanations
    )
    result_json = first.to_json()
    assert "source_text" not in result_json
    assert '"source"' not in result_json
    assert "ast_text" not in result_json


def test_queries_enforce_strict_result_and_utf8_byte_bounds() -> None:
    records = [
        (
            f"src/service_{number}.py",
            _record(
                f"class Service{number}:\n"
                "    def dispatch(self, request):\n"
                "        return request\n",
                f"blob:service-{number}",
            ),
        )
        for number in range(8)
    ]
    index = build_analysis_ast_index(records)

    count_limited = index.query_definitions(
        "dispatch", max_results=2, max_bytes=32_000
    )
    assert len(count_limited.evidence) == 2
    assert count_limited.total_matches == 8
    assert count_limited.truncation.result_limit_reached is True
    assert count_limited.truncation.byte_limit_reached is False
    assert count_limited.truncation.omitted_results == 6

    byte_limited = index.query_objective_terms(
        "service dispatch", max_results=1000, max_bytes=900
    )
    encoded = byte_limited.to_json().encode("utf-8")
    assert 0 < len(byte_limited.evidence) < byte_limited.total_matches
    assert byte_limited.truncation.byte_limit_reached is True
    assert byte_limited.truncation.max_results == 100  # hard query cap
    assert byte_limited.byte_count == len(encoded)
    assert len(encoded) <= 900
    assert json.loads(encoded)["truncation"]["truncated"] is True

    with pytest.raises(AnalysisASTIndexError, match="at least 256"):
        index.query_paths(max_bytes=255)


def test_index_rejects_unsafe_or_unassociated_paths() -> None:
    record = _snapshot()[0][1]
    with pytest.raises(AnalysisASTIndexError, match="escapes"):
        build_analysis_ast_index([("../outside.py", record)])
    with pytest.raises(AnalysisASTIndexError, match="repository path"):
        build_analysis_ast_index([record])


def test_symbol_queries_reject_prefix_and_partial_substring_matches() -> None:
    record = _record(
        "def foobar():\n    return True\n",
        "blob:foobar-v1",
    )
    index = build_analysis_ast_index([("src/foobar.py", record)])

    assert index.query_symbols("foobar").evidence
    assert index.query_symbols("foo").evidence == ()
    assert index.query_symbols("oob").evidence == ()
    assert index.query_definitions("bar").evidence == ()
