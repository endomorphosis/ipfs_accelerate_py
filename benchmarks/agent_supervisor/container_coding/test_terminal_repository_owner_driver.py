"""Authored finalization fragments; no worker, provider or native resource run.

Execute the actual repository-owner branch and durable-write tail from run's
finally block. Separate structural checks locate that branch after the real
cleanup, receipt collection and context accounting calls. These checks do not
claim that the isolated repository-source route is enabled.
"""
from __future__ import annotations

import ast
import builtins
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_repository_owner_observation as observation


DIRECTORY = Path(__file__).parent
KEY = "repository_source_owner_observation"
OWNER_SCHEMA = "supervisor-repository-source-owner-binding@1"
SELECTED = ("repository-source-raw@1", "repository-source-compact@1")
_MISSING = object()


def _contains(node, value):
    return any(isinstance(item, ast.Constant) and item.value == value
               for item in ast.walk(node))


def _finalization():
    tree = ast.parse((DIRECTORY / "terminal_container_supervisor.py").read_text())
    run = next(node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == "run")
    candidates = [node.finalbody for node in ast.walk(run)
                  if isinstance(node, ast.Try) and node.finalbody
                  and any(_contains(item, KEY) for item in node.finalbody)]
    assert len(candidates) == 1
    body = candidates[0]
    selected = [index for index, item in enumerate(body)
                if isinstance(item, ast.If) and _contains(item, KEY)]
    assert len(selected) == 1
    return body, selected[0]


def _report(policy=_MISSING, *, rows=_MISSING, completed=False):
    report = dict(task_completed=completed, worker_cleanup_returncode=0,
                  provider_invocations=[] if rows is _MISSING else rows,
                  context_input_audit={"status": "unknown"},
                  native_usage={"input_tokens": 23, "output_tokens": None},
                  error={"type": "AuthoredFailure", "message": "retained original failure"})
    if policy is not _MISSING:
        report["repository_source_policy"] = policy
    return report


def _execute(monkeypatch, report, *, collect=None, scope=_MISSING, import_error=None):
    del monkeypatch  # No global import mutation: only this fragment's builtins change.
    if scope is _MISSING:
        scope = {"task_cid": "captured-cid", "task_alias": "captured-task",
                 "baseline_commit": "a" * 40, "outputs": []}
    calls, writes, imports = [], [], []
    observed = observation.unavailable()
    observed.update(status="observed", reason=None)
    module = ModuleType("benchmarks.agent_supervisor.container_coding.terminal_repository_owner_observation")

    def capture(**kwargs):
        calls.append(kwargs)
        return collect(**kwargs) if collect is not None else deepcopy(observed)

    module.collect = capture
    module.unavailable = observation.unavailable
    actual_import = builtins.__import__

    def controlled_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"terminal_repository_owner_observation", module.__name__}:
            imports.append(name)
            if import_error is not None:
                raise import_error("private collector import detail")
            return module
        return actual_import(name, globals, locals, fromlist, level)

    body, selected = _finalization()
    namespace = dict(__package__="benchmarks.agent_supervisor.container_coding",
                     __builtins__={**vars(builtins), "__import__": controlled_import},
                     ROOT=Path("/authored/owner"), report=report,
                     output_observation_scope=scope,
                     time=SimpleNamespace(monotonic=lambda: 31.0), started=10.0,
                     report_path=Path("/authored/result.json"),
                     _write=lambda path, value: writes.append((path, deepcopy(value))))
    exec(compile(ast.Module(body=body[selected:], type_ignores=[]),
                 "<actual-owner-observation-finally-tail>", "exec"), namespace)
    return dict(calls=calls, writes=writes, imports=imports, scope=scope,
                expected=observed)


def _assert_preserved(report, original, result):
    assert {key: value for key, value in report.items() if key not in {KEY, "seconds"}} == original
    assert result["writes"] == [(Path("/authored/result.json"), report)]
    assert report["seconds"] == 21.0


def test_collector_is_after_cleanup_receipts_and_accounting_before_durable_write():
    body, selected = _finalization()
    def position(name):
        positions = [index for index, item in enumerate(body)
                     if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id == name for node in ast.walk(item))]
        assert len(positions) == 1
        return positions[0]
    cleanup = [index for index, item in enumerate(body) if _contains(item, "--cleanup")]
    assert len(cleanup) == 1
    assert cleanup[0] < position("_collect_provider_usage_rows") < position("_final_context_and_usage") < selected
    assert selected < position("_write")
    assert len(body[selected:]) == 3  # The actual branch, elapsed time and durable write.


@pytest.mark.parametrize("policy", SELECTED)
@pytest.mark.parametrize("completed", [False, True])
def test_selected_report_gets_separate_observation_without_outcome_or_usage_changes(monkeypatch, policy, completed):
    report = _report(policy, completed=completed)
    original = deepcopy(report)
    result = _execute(monkeypatch, report)
    assert report[KEY] == result["expected"]
    assert len(result["calls"]) == len(result["imports"]) == 1
    call = result["calls"][0]
    assert call == {"parent": Path("/authored/owner/worker-observations"),
                    "scope": result["scope"], "report": report}
    assert call["scope"] is result["scope"] and call["report"] is report
    _assert_preserved(report, original, result)


@pytest.mark.parametrize("policy", [_MISSING, "legacy"])
@pytest.mark.parametrize("rows", [_MISSING, [], None, 7, {},
    [None, "malformed", {"repository_source_policy": "legacy"}],
    [{"phase": "coding", "repository_source_policy": {"schema": "supervisor-repository-source-policy@1"}}]])
def test_unselected_legacy_shape_and_dependencies_remain_unchanged(monkeypatch, policy, rows):
    report = _report(policy, rows=rows)
    original = deepcopy(report)
    result = _execute(monkeypatch, report, import_error=AssertionError)
    assert KEY not in report
    assert result["calls"] == result["imports"] == []
    _assert_preserved(report, original, result)


@pytest.mark.parametrize("policy", [_MISSING, "legacy"])
def test_observed_owner_binding_is_not_hidden_by_legacy_report_label(monkeypatch, policy):
    binding = dict(schema=OWNER_SCHEMA, usage={"status": "unavailable", "reason": "owner_audit_required",
        "successful_reads": None, "store_cleanup_complete": None}, token_savings_measured=False)
    row = dict(phase="coding", repository_source_policy=binding,
               native_rollout_usage={"input_tokens": 17, "output_tokens": 4})
    report = _report(policy, rows=[row])
    original = deepcopy(report)
    result = _execute(monkeypatch, report)
    assert len(result["calls"]) == 1 and KEY in report
    assert row["repository_source_policy"] is binding
    _assert_preserved(report, original, result)


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError, OSError])
def test_collector_error_preserves_original_failure_and_durable_report(monkeypatch, error_type):
    def fail(**kwargs):
        raise error_type("private collector failure detail")
    report = _report(SELECTED[0])
    original = deepcopy(report)
    result = _execute(monkeypatch, report, collect=fail)
    assert report[KEY] == observation.unavailable("collector_failed")
    assert "private collector" not in repr(report)
    _assert_preserved(report, original, result)


@pytest.mark.parametrize("error_type", [ImportError, RuntimeError])
def test_unavailable_import_uses_the_same_closed_failure_shape(monkeypatch, error_type):
    report = _report(SELECTED[1])
    original = deepcopy(report)
    result = _execute(monkeypatch, report, import_error=error_type)
    assert len(result["imports"]) == 1 and result["calls"] == []
    assert report[KEY] == observation.unavailable("collector_failed")
    assert "private collector" not in repr(report)
    _assert_preserved(report, original, result)


@pytest.mark.parametrize("cleanup", [0, 1, None])
def test_cleanup_evidence_is_forwarded_to_collector_without_inventing_success(monkeypatch, cleanup):
    report = _report(SELECTED[0])
    report["worker_cleanup_returncode"] = cleanup
    original = deepcopy(report)
    result = _execute(monkeypatch, report, collect=lambda **kwargs: observation.unavailable())
    assert result["calls"][0]["report"]["worker_cleanup_returncode"] is cleanup
    assert report[KEY] == observation.unavailable()
    _assert_preserved(report, original, result)


def test_missing_captured_scope_remains_missing(monkeypatch):
    report = _report(SELECTED[0])
    result = _execute(monkeypatch, report, scope=None,
                      collect=lambda **kwargs: observation.unavailable())
    assert result["calls"][0]["scope"] is None
    assert report[KEY] == observation.unavailable()


def test_archive_source_contract_includes_collector_and_pure_audit_once():
    tree = ast.parse((DIRECTORY / "terminal_repository_source_policy.py").read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "SOURCE_PATHS"
                           for target in node.targets)]
    assert len(definitions) == 1
    paths = ast.literal_eval(definitions[0].value)
    for name in ("terminal_repository_owner_observation.py", "terminal_repository_owner_audit.py"):
        assert paths.count("benchmarks/agent_supervisor/container_coding/" + name) == 1
    assert paths.count("ipfs_accelerate_py/agent_supervisor/runtime/repository_source_mcp.py") == 1
