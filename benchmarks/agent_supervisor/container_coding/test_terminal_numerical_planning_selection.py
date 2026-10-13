"""Balanced unresolved hints and exact source-replayed obligation selection."""
from copy import deepcopy
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as native
from benchmarks.agent_supervisor.container_coding.test_terminal_numerical_control_context import (
    control_context, _report, SOURCE, HISTORICAL_SOURCE,
)


LOOPS = '''def accumulate(values):
    result = 0
    for value in values:
        result = result + value
    for value in values:
        result = result + value
    return result
'''
PLAIN = "import numpy as np\n" + "\n".join(
    f"def plain_{i}(value):\n    return np.sqrt(value)\n" for i in range(3))


def _formal(source, path="sample.py"):
    return native._modules(native.CONTROL_PROFILE)[1].formalize_numerical_source(source, path=path)


def _summary(reports):
    return native._summary(reports, "authored-tree", profile=native.CONTROL_PROFILE)


def _counts(summary, reports):
    expected = dict(pending_obligations=sum(len(r["pending_obligations"]) for r in reports.values()),
        per_function=sum(r["summary"]["functions"] for r in reports.values()),
        assertion_requirements=sum(len(r["assertion_requirements"]) for r in reports.values()),
        compound_contracts=sum(len(r["compound_contracts"]) for r in reports.values()),
        control_references=len(reports))
    omissions = dict(pending_obligations="omitted_obligation_summaries",
        per_function="omitted_function_summaries", assertion_requirements="omitted_assertion_requirement_summaries",
        compound_contracts="omitted_compound_contract_summaries", control_references="omitted_control_reference_summaries")
    assert len(native._bytes(summary)) <= 8192
    for key, total in expected.items():
        assert len(summary[key]) + summary[omissions[key]] == total


def test_balanced_hints_keep_paths_functions_loops_and_all_proof_phases():
    reports = {"a.py": _formal(PLAIN, "a.py"), "z.py": _formal(LOOPS, "z.py")}
    summary = _summary(reports)
    _counts(summary, reports)
    hints = summary["pending_obligations"]
    assert {row["path"] for row in hints} == set(reports)
    assert {(row["path"], f) for row in hints for f in row["function_ids"]} == {
        (path, f["function_id"]) for path, report in reports.items()
        for f in report["summary"]["per_function"]}
    proofs = [row for row in hints if row["scope"] == "proof_applicability"]
    assert {row["proof_phase"] for row in proofs} == {"activation_entry",
        "normal_or_continue_backedge", "exit_cause_and_post_state", "whole_activation_progress"}
    assert len({row["proof_scope_control_region_id"] for row in proofs}) == 2
    assert _summary(dict(reversed(list(reports.items())))) == summary
    contract = summary["planning_hint_contract"]
    assert contract["full_context_required"] is True
    assert contract["relevance_ranked"] is contract["reference_lists_truncated"] is False
    assert "Source anchor only" in contract["scope_meanings"]["proof_applicability"]
    assert all(contract[k] is False for k in ("evaluation_guard_verified", "runtime_truthiness_verified",
        "runtime_effects_verified", "standalone_program_evaluation_complete", "proof_phase_established",
        "proof_applicability_verified"))
    for hint in hints:
        full = next(row for row in reports[hint["path"]]["pending_obligations"]
                    if row["obligation_id"] == hint["obligation_id"])
        for key in ("kind", "function_ids", "expression_id"):
            assert hint[key] == full[key]
        for key in ("guarded_region_ids", "control_region_ids", "loop_region_ids"):
            assert hint.get(key, []) == full[key]
        assert hint["cfg_ids"] == sorted({row["cfg_id"] for row in full["cfg_occurrences"]})
        if "proof_scope" in full:
            assert hint["proof_phase"] == full["proof_scope"]["proof_phase"]
            assert hint["proof_scope_control_region_id"] == full["proof_scope"]["control_region_id"]
            assert hint["proof_scope_cfg_id"] == full["proof_scope"]["cfg_id"]


@pytest.mark.parametrize("category", ["functions", "assertions", "compounds"])
def test_packing_preserves_omissions_already_applied_by_native_summary(category):
    if category == "functions":
        source = "\n".join(f"def item_{i}(value):\n    return value + 1\n" for i in range(20))
        key, full_key = "per_function", "functions"
    elif category == "assertions":
        source = "def inspect(value):\n" + "".join(f"    assert value > {i}\n" for i in range(6)) + "    return value\n"
        key, full_key = "assertion_requirements", "assertion_requirements"
    else:
        body = "    values, vectors = np.linalg.eig(matrix)\n    chosen = np.argmax(np.abs(values))\n    return values[chosen], vectors[:, chosen]\n"
        source = "import numpy as np\n" + "\n".join(f"def pair_{i}(matrix):\n" + body for i in range(9))
        key, full_key = "compound_contracts", "compound_contracts"
    report = _formal(source)
    full_count = report["summary"][full_key] if category == "functions" else len(report[full_key])
    assert len(report["summary"][key]) < full_count
    _counts(_summary({"sample.py": report}), {"sample.py": report})


def test_oversized_rows_do_not_evict_all_short_actionable_hints():
    report = _formal(LOOPS)
    reports = {"a" * 20000 + ".py": report, "z.py": report}
    summary = _summary(reports)
    _counts(summary, reports)
    assert summary["pending_obligations"]
    assert {row["path"] for row in summary["pending_obligations"]} == {"z.py"}
    assert summary["obligation_retrieval"]["source_paths"] == ["z.py"]
    assert summary["obligation_retrieval"]["omitted_source_paths"] == 1


def test_no_pending_work_keeps_empty_hints_and_explicit_scope():
    reports = {"empty.py": _formal("# authored empty module\n", "empty.py")}
    summary = _summary(reports)
    _counts(summary, reports)
    assert summary["pending_obligations"] == []
    assert summary["planning_hint_contract"]["full_context_required"] is True


def test_exact_hint_id_returns_complete_owner_row_and_no_continuation(control_context):
    root, receipt = control_context
    full = _report(receipt)
    expected = {row["obligation_id"]: row for row in full["pending_obligations"]}
    before = native._bytes(full)
    assert "--obligation-id" in receipt["summary"]["obligation_retrieval"]["isolated_worker_usage"]
    for hint in receipt["summary"]["pending_obligations"]:
        page = native.retrieve_numerical_obligations(repository=root, expected_receipt=receipt,
            path="sample.py", obligation_id=hint["obligation_id"], kind=hint["kind"],
            function_id=hint["function_ids"][0])
        assert page["rows"] == [expected[hint["obligation_id"]]]
        assert page["next_cursor"] is None
        assert len(native._page_bytes(page)) <= 8192
    assert native._bytes(_report(receipt)) == before


def test_omitted_and_none_id_preserve_native_page_bytes_and_pager_call_shape(control_context, monkeypatch):
    from ipfs_datasets_py.logic.software_verification import numerical_obligation_retrieval as pager
    root, receipt = control_context
    original = pager.page_numerical_obligations
    calls = []
    def spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(pager, "page_numerical_obligations", spy)
    options = dict(repository=root, expected_receipt=receipt, path="sample.py", limit=1)
    before = native.retrieve_numerical_obligations(**options)
    after = native.retrieve_numerical_obligations(**options, obligation_id=None)
    assert native._page_bytes(before) == native._page_bytes(after)
    assert all("obligation_id" not in call for call in calls)
    hint = receipt["summary"]["pending_obligations"][0]
    native.retrieve_numerical_obligations(**options, obligation_id=hint["obligation_id"])
    assert calls[-1]["obligation_id"] == hint["obligation_id"]


def test_exact_id_conjunctive_filters_and_foreign_scan_cursor(control_context):
    root, receipt = control_context
    options = dict(repository=root, expected_receipt=receipt, path="sample.py")
    hint = receipt["summary"]["pending_obligations"][0]
    page = native.retrieve_numerical_obligations(**options, limit=1)
    assert page["next_cursor"] is not None
    with pytest.raises(ValueError):
        native.retrieve_numerical_obligations(**options, obligation_id=hint["obligation_id"], cursor=page["next_cursor"])
    miss = native.retrieve_numerical_obligations(**options, obligation_id=hint["obligation_id"], kind="unknown")
    assert miss["rows"] == [] and miss["next_cursor"] is None
    miss = native.retrieve_numerical_obligations(**options, obligation_id="unselected-id")
    assert miss["rows"] == [] and miss["next_cursor"] is None


@pytest.mark.parametrize("value", ["", False, [], "x" * 513, "\ud800"])
def test_invalid_exact_id_refuses(control_context, value):
    root, receipt = control_context
    with pytest.raises(ValueError):
        native.retrieve_numerical_obligations(repository=root, expected_receipt=receipt,
            path="sample.py", obligation_id=value)


def test_exact_id_does_not_bypass_stale_source_validation(control_context):
    root, receipt = control_context
    hint = receipt["summary"]["pending_obligations"][0]
    (root / "sample.py").write_text(SOURCE + "\n# authored drift\n")
    with pytest.raises(ValueError, match="source"):
        native.retrieve_numerical_obligations(repository=root, expected_receipt=receipt,
            path="sample.py", obligation_id=hint["obligation_id"])
