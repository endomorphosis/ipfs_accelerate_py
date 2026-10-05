"""Closed bootstrap diagnostics preserve all admission checks."""
import json
import threading
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import (
    AdmittedBenchmarkRuntime, BOOTSTRAP_FAILURE_PHASES, BOOTSTRAP_FAILURE_REASONS,
    _bootstrap_tree_rejection,
)


def diagnostic_runtime():
    runtime = object.__new__(AdmittedBenchmarkRuntime)
    runtime._startup_trace_lock = threading.Lock()
    runtime._startup_trace = []
    runtime._startup_trace_truncated = False
    runtime._bootstrap_failure_counts = {}
    runtime.bootstrap_receipts = []
    runtime.bootstrap_errors = [{"type": "ValueError", "message": "PRIVATE_BODY"}]
    runtime.start_timeout_ms = None
    runtime.timeout_ms = 20_000
    return runtime


def test_diagnostics_count_failure_buckets_without_exporting_exception_bodies():
    runtime = diagnostic_runtime()
    runtime._record_bootstrap_failure("process_tree", "process_root_missing")
    runtime._record_bootstrap_failure("process_tree", "process_root_missing")
    runtime._record_bootstrap_failure("request", "request_mismatch")
    result = runtime.startup_diagnostics()
    assert result["bootstrap_failure_counts"] == [
        {"phase": "process_tree", "reason": "process_root_missing", "count": 2},
        {"phase": "request", "reason": "request_mismatch", "count": 1},
    ]
    assert "PRIVATE_BODY" not in json.dumps(result)
    assert result["bootstrap_error_count"] == 1
    runtime._bootstrap_failure_counts[("request", "request_mismatch")] = 65535
    runtime._record_bootstrap_failure("request", "request_mismatch")
    assert runtime._bootstrap_failure_counts[("request", "request_mismatch")] == 65535
    with pytest.raises(ValueError, match="unknown bootstrap failure"):
        runtime._record_bootstrap_failure("PRIVATE_BODY", "unknown")


@pytest.mark.parametrize("roots,matches,pid,scope,expected", [
    ([], [], 2, True, "process_root_missing"),
    ([1, 3], [2], 2, True, "process_root_ambiguous"),
    ([1], [], 2, True, "process_child_missing"),
    ([1], [2, 2], 2, True, "process_child_ambiguous"),
    ([1], [1], 1, True, "process_child_is_root"),
    ([3], [2], 2, True, "process_parent_mismatch"),
    ([1], [2], 2, False, "process_scope_mismatch"),
    ([1], [2], 2, True, None),
])
def test_native_child_classification_preserves_exact_checks(roots, matches, pid, scope, expected):
    tree = SimpleNamespace(roots=[SimpleNamespace(pid=item) for item in roots])
    selected = [SimpleNamespace(pid=item, parent_pid=1) for item in matches]
    process = SimpleNamespace(child_scope_matches=lambda *_: scope)
    assert _bootstrap_tree_rejection(tree, selected, pid=pid,
        process=process, profile=object()) == expected


def test_failure_projection_uses_the_same_closed_taxonomy():
    from benchmarks.agent_supervisor.container_coding.terminal_container_supervisor import _project_native_startup
    runtime = diagnostic_runtime()
    for phase in sorted(BOOTSTRAP_FAILURE_PHASES):
        runtime._bootstrap_failure_counts = {(phase, "unknown"): 1}
        assert _project_native_startup(runtime.startup_diagnostics())["bootstrap_failure_counts"]
    for reason in sorted(BOOTSTRAP_FAILURE_REASONS):
        runtime._bootstrap_failure_counts = {("process_tree", reason): 1}
        assert _project_native_startup(runtime.startup_diagnostics())["bootstrap_failure_counts"]


def test_failure_buckets_are_complete_and_bounded_by_the_closed_product():
    runtime = diagnostic_runtime()
    for phase in BOOTSTRAP_FAILURE_PHASES:
        for reason in BOOTSTRAP_FAILURE_REASONS:
            runtime._record_bootstrap_failure(phase, reason)
    rows = runtime.startup_diagnostics()["bootstrap_failure_counts"]
    assert len(rows) == len(BOOTSTRAP_FAILURE_PHASES) * len(BOOTSTRAP_FAILURE_REASONS)
    assert len(rows) > 16
    assert all(row["count"] == 1 for row in rows)
    assert [(row["phase"], row["reason"]) for row in rows] == sorted(
        (phase, reason) for phase in BOOTSTRAP_FAILURE_PHASES
        for reason in BOOTSTRAP_FAILURE_REASONS)
