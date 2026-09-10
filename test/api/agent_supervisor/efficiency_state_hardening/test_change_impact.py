"""ASEH-050 deterministic AST/symbol change-impact acceptance tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.schema_protocol_change_impact import (
    SchemaProtocolChangeImpactAuthorityError,
    require_ast_before_schema_protocol_analysis,
)
from ipfs_accelerate_py.agent_supervisor.analysis.semantic_dependency_graph import (
    ModelAnalysisBeforeASTError,
    SourceChangeKind,
    analyze_ast_symbol_change,
    require_ast_before_model,
)


def test_exact_file_symbol_provenance_changed_contract_and_reverse_cone() -> None:
    before = {
        "pkg/service.py": """
def service(value: int) -> int:
    return value


def caller() -> int:
    return service(1)
""",
    }
    after = {
        "pkg/service.py": """
def service(value: int, extra: int) -> int:
    return value + extra


def caller() -> int:
    return service(1)
""",
    }

    report = analyze_ast_symbol_change(before, after)

    service = "symbol:pkg/service.py:service"
    caller = "symbol:pkg/service.py:caller"
    assert report.classification is SourceChangeKind.SEMANTIC
    assert report.changed_symbol_ids == (service,)
    assert caller in report.reverse_dependency_cone.symbol_ids
    assert report.impact_complete
    assert len(report.before_files) == len(report.after_files) == 1
    assert report.before_files[0].path == "pkg/service.py"
    assert report.before_files[0].content_sha256 != report.after_files[0].content_sha256
    assert report.changed_contracts[0].symbol_id == service
    assert "signature_or_interface" in report.changed_contracts[0].change_kinds
    assert all(ref.startswith("source:pkg/service.py:sha256:") for ref in report.changed_contracts[0].provenance_refs)


def test_dynamic_dispatch_broadens_uncertainty_and_cannot_claim_complete_cone() -> None:
    report = analyze_ast_symbol_change(
        {"pkg/hooks.py": "def run(target):\n    return getattr(target, name)()\n"},
        {"pkg/hooks.py": "def run(target):\n    return getattr(target, name)()\n\ndef changed():\n    return 1\n"},
    )

    assert report.classification is SourceChangeKind.UNKNOWN
    assert report.impact_complete is False
    assert report.dependency_cone.complete is False
    assert report.reverse_dependency_cone.complete is False
    assert any(ref.startswith("dynamic_") for ref in report.uncertainty_refs)


def test_documentation_and_formatting_changes_are_not_reported_as_semantic() -> None:
    docs = analyze_ast_symbol_change(
        {"docs/guide.md": "# old\n"}, {"docs/guide.md": "# new\n"}
    )
    formatting = analyze_ast_symbol_change(
        {"pkg/value.py": "VALUE = 1\n"},
        {"pkg/value.py": "\n# explain the constant\nVALUE = 1\n"},
    )
    docstring = analyze_ast_symbol_change(
        {"pkg/value.py": 'def value():\n    """Old explanation."""\n    return 1\n'},
        {"pkg/value.py": 'def value():\n    """New explanation."""\n    return 1\n'},
    )

    assert docs.classification is SourceChangeKind.DOCUMENTATION_ONLY
    assert docs.changed_symbol_ids == ()
    assert formatting.classification is SourceChangeKind.FORMATTING_ONLY
    assert formatting.changed_symbol_ids == ()
    assert docstring.classification is SourceChangeKind.DOCUMENTATION_ONLY
    assert docstring.changed_symbol_ids == ()


def test_no_model_can_precede_or_contaminate_ast_analysis() -> None:
    report = analyze_ast_symbol_change({"pkg/a.py": "x = 1\n"}, {"pkg/a.py": "x = 2\n"})

    require_ast_before_model(report)
    with pytest.raises(ModelAnalysisBeforeASTError, match="zero model invocations"):
        require_ast_before_model(replace(report, model_invocation_count=1))
    with pytest.raises(SchemaProtocolChangeImpactAuthorityError, match="zero model invocations"):
        require_ast_before_schema_protocol_analysis(replace(report, model_invocation_count=1))
