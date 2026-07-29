from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.code_security_facts import (
    ChangedCodeDiff,
    CodeSecurityDiagnosticCode,
    CodeSecurityExtractionStatus,
    CodeSecurityFact,
    CodeSecurityFactError,
    CodeSecurityFactKind,
    CodeSecurityFactSet,
    extract_code_security_facts,
)


def _diff(
    *,
    before: str | None,
    after: str | None,
    old_path: str = "src/service.py",
    new_path: str = "src/service.py",
    language: str = "python",
) -> ChangedCodeDiff:
    return ChangedCodeDiff(
        tree_id="git-tree:candidate",
        diff_id="git-diff:reviewed",
        old_path=old_path,
        new_path=new_path,
        before_source=before,
        after_source=after,
        before_blob_id="git-blob:before" if before is not None else "",
        after_blob_id="git-blob:after" if after is not None else "",
        language=language,
    )


def _values(result: CodeSecurityFactSet, kind: CodeSecurityFactKind) -> set[str]:
    return {item.value for item in result.by_kind(kind)}


def test_extracts_every_dimension_with_exact_tree_blob_diff_ast_and_scope() -> None:
    before = """\
def stable():
    return 1

def update(client, payload):
    return payload
"""
    after = """\
import requests as http

def stable():
    return 1

def update(client, payload):
    if client.enabled:
        client.state = "running"
        return http.post(payload)
    return payload
"""
    result = extract_code_security_facts(_diff(before=before, after=after))

    assert result.status is CodeSecurityExtractionStatus.EXTRACTED
    assert set(CodeSecurityFactKind) == {item.kind for item in result.facts}
    assert {"import", "write", "return", "invoke"}.issubset(
        _values(result, CodeSecurityFactKind.ACTION)
    )
    assert {"requests", "client.state", "requests.post", "caller"}.issubset(
        _values(result, CodeSecurityFactKind.TARGET)
    )
    assert {"module_import", "state_mutation", "network", "call"}.issubset(
        _values(result, CodeSecurityFactKind.CAPABILITY)
    )
    assert {"module_load", "state_update", "function_return", "call"}.issubset(
        _values(result, CodeSecurityFactKind.EFFECT)
    )
    assert _values(result, CodeSecurityFactKind.GUARD)
    assert _values(result, CodeSecurityFactKind.LANGUAGE) == {"python"}
    assert "<module>" in _values(result, CodeSecurityFactKind.SOURCE_SCOPE)
    assert "update" in _values(result, CodeSecurityFactKind.SOURCE_SCOPE)

    for fact in result.facts:
        assert fact.binding.tree_id == "git-tree:candidate"
        assert fact.binding.diff_id == "git-diff:reviewed"
        assert fact.binding.blob_id in {"git-blob:before", "git-blob:after"}
        assert fact.binding.source_sha256.startswith("sha256:")
        assert fact.binding.ast_id.startswith("ast:sha256:")
        assert fact.source_scope.path == "src/service.py"
        assert fact.source_scope.symbol in {"<module>", "update"}
        assert fact.source_scope.line_start >= 1
        assert fact.source_scope.line_end >= fact.source_scope.line_start


def test_only_changed_ast_scope_is_attributed_and_renames_bind_each_side() -> None:
    before = """\
def stable():
    return trusted()

def changed():
    return old_call()
"""
    after = before.replace("old_call()", "new_call()")
    result = extract_code_security_facts(
        _diff(
            before=before,
            after=after,
            old_path="old/service.py",
            new_path="new/service.py",
        )
    )

    assert {item.source_scope.symbol for item in result.facts} == {"changed"}
    assert "trusted" not in _values(result, CodeSecurityFactKind.TARGET)
    assert {"old_call", "new_call"}.issubset(
        _values(result, CodeSecurityFactKind.TARGET)
    )
    old_facts = [
        item for item in result.facts if item.source_scope.delta.value == "removed"
    ]
    new_facts = [
        item for item in result.facts if item.source_scope.delta.value == "added"
    ]
    assert old_facts and all(item.source_scope.path == "old/service.py" for item in old_facts)
    assert new_facts and all(item.source_scope.path == "new/service.py" for item in new_facts)


def test_comments_docstrings_and_string_literals_cannot_inject_facts() -> None:
    hostile = """\
def harmless():
    # os.remove('/important')
    payload = "requests.get('https://attacker') and exec(secret)"
    return payload
"""
    result = extract_code_security_facts(
        _diff(before=None, after=hostile, old_path="", new_path="src/harmless.py")
    )
    serialized_values = {item.value for item in result.facts}

    assert "os.remove" not in serialized_values
    assert "requests.get" not in serialized_values
    assert "exec" not in serialized_values
    assert "filesystem" not in serialized_values
    assert "network" not in serialized_values
    assert "code_execution" not in serialized_values
    # Literal bodies are not retained in canonical outputs.
    assert "attacker" not in result.to_json()
    assert "important" not in result.to_json()


def test_unsupported_parse_failures_missing_sources_and_dynamic_calls_are_explicit() -> None:
    unsupported = extract_code_security_facts(
        _diff(before="x", after="y", language="rust")
    )
    assert unsupported.status is CodeSecurityExtractionStatus.UNSUPPORTED
    assert unsupported.diagnostics[0].code is CodeSecurityDiagnosticCode.UNSUPPORTED_LANGUAGE

    invalid = extract_code_security_facts(
        _diff(before=None, after="def broken(:\n", old_path="", new_path="broken.py")
    )
    assert invalid.status is CodeSecurityExtractionStatus.INVALID
    assert invalid.diagnostics[0].code is CodeSecurityDiagnosticCode.PARSE_ERROR

    missing = extract_code_security_facts(
        _diff(before=None, after=None)
    )
    assert missing.status is CodeSecurityExtractionStatus.UNSUPPORTED
    assert {item.code for item in missing.diagnostics} == {
        CodeSecurityDiagnosticCode.MISSING_SOURCE
    }

    dynamic = extract_code_security_facts(
        _diff(
            before=None,
            after="def run(registry):\n    return registry['handler']()\n",
            old_path="",
            new_path="src/dynamic.py",
        )
    )
    assert dynamic.status is CodeSecurityExtractionStatus.PARTIAL
    assert CodeSecurityDiagnosticCode.DYNAMIC_CALL_TARGET in {
        item.code for item in dynamic.diagnostics
    }
    assert "registry" not in _values(dynamic, CodeSecurityFactKind.TARGET)


def test_canonical_round_trip_is_stable_tamper_evident_and_non_authoritative() -> None:
    result = extract_code_security_facts(
        _diff(
            before=None,
            after="import os\n\ndef remove(path):\n    os.remove(path)\n",
            old_path="",
            new_path="src/remove.py",
        )
    )
    restored = CodeSecurityFactSet.from_json(result.to_json())

    assert restored.fact_set_id == result.fact_set_id
    assert [item.fact_id for item in restored.facts] == [
        item.fact_id for item in result.facts
    ]
    assert not result.grants_execution_authority
    assert not result.authorizes_completion
    assert not result.establishes_generated_code_correctness
    assert all(not item.grants_execution_authority for item in result.facts)
    assert all(not item.authorizes_completion for item in result.facts)

    forged = json.loads(result.to_json())
    forged["grants_execution_authority"] = True
    with pytest.raises(CodeSecurityFactError, match="cannot set"):
        CodeSecurityFactSet.from_dict(forged)

    forged = result.facts[0].to_dict()
    forged["value"] = "forged"
    with pytest.raises(CodeSecurityFactError, match="fact_id"):
        CodeSecurityFact.from_dict(forged)


def test_multi_file_diff_is_deterministic_and_rejects_cross_revision_aggregation() -> None:
    first = _diff(
        before=None,
        after="def first():\n    return one()\n",
        old_path="",
        new_path="src/first.py",
    )
    second = _diff(
        before=None,
        after="def second():\n    return two()\n",
        old_path="",
        new_path="src/second.py",
    )

    result = extract_code_security_facts([second, first])
    repeated = extract_code_security_facts([first, second])

    assert result.fact_set_id == repeated.fact_set_id
    assert {item.source_scope.path for item in result.facts} == {
        "src/first.py",
        "src/second.py",
    }

    mismatched = ChangedCodeDiff(
        tree_id="git-tree:other",
        diff_id=first.diff_id,
        new_path="src/other.py",
        after_source="value = 1\n",
    )
    with pytest.raises(CodeSecurityFactError, match="same tree_id"):
        extract_code_security_facts([first, mismatched])


@pytest.mark.parametrize(
    "payload",
    [
        {"tree_id": "", "diff_id": "diff", "path": "a.py"},
        {"tree_id": "tree", "diff_id": "", "path": "a.py"},
        {"tree_id": "tree", "diff_id": "diff", "path": "../escape.py"},
        {
            "tree_id": "tree",
            "diff_id": "diff",
            "path": "a.py",
            "unexpected_authority": "allow",
        },
    ],
)
def test_changed_diff_rejects_unbound_escaped_and_unknown_inputs(
    payload: dict[str, object],
) -> None:
    with pytest.raises(CodeSecurityFactError):
        ChangedCodeDiff.from_dict(payload)
