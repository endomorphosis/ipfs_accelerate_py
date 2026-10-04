"""Retain genuine native test outcomes inside the bounded offline test image."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET


TARGETS = (
    "test/api/test_finite_repository_execution.py",
    "test/api/test_finite_repository_candidate_runner.py",
    "test/api/test_admitted_runtime_launch_bounds.py",
    "test/integration/test_admitted_benchmark_runtime.py",
)
MODULES = (
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_admission",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_execution",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_candidate_runner",
    "ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime",
    "ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge",
    "ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository",
    "ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state",
    "benchmarks.agent_supervisor.container_coding.finite_repository_native_tests",
)


def main():
    output = Path(sys.argv[1]).absolute()
    targets = tuple(sys.argv[2:]) or TARGETS
    if any(target.split("::", 1)[0] not in TARGETS for target in targets):
        raise ValueError("native test selection must use the closed qualification files")
    if output.exists() or output.resolve() != output:
        raise ValueError("fresh exact native test output required")
    output.mkdir(mode=0o700)
    import duckdb
    import _duckdb
    import pytest
    sources = [Path(importlib.import_module(name).__file__) for name in MODULES]
    sources += [Path(target).absolute() for target in TARGETS]

    def pins():
        return [{"path": str(path), "bytes": path.stat().st_size,
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources]

    before = pins()
    started = time.monotonic()
    status = pytest.main(["-q", "-o", "addopts=", "-o", "pythonpath=",
        "-o", "cache_dir=" + str(output / "pytest-cache"), "--import-mode=importlib",
        "--basetemp=" + str(output / "native"), "--junitxml=" + str(output / "junit.xml"), *targets])
    after = pins()
    cases = list(ET.parse(output / "junit.xml").iter("testcase"))
    ledgers = []
    ledger_paths = set((output / "native").rglob("resource-admission*.json"))
    ledger_paths.update((output / "native").rglob("admission.json"))
    for path in sorted(ledger_paths):
        value = json.loads(path.read_bytes())
        if "leases" in value and "waiters" in value:
            ledgers.append({"path": str(path), "active_leases": len(value["leases"]),
                            "waiting_requests": len(value["waiters"])})
    summary = {"schema": "finite-repository-offline-native-tests@1",
        "pytest_exit": int(status), "tests": len(cases),
        "failures": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
        "elapsed_seconds": time.monotonic() - started,
        "python": sys.executable, "python_version": sys.version,
        "duckdb": duckdb.__version__, "pytest": pytest.__version__, "owner_uid": os.geteuid(),
        "native_duckdb_sha256": hashlib.sha256(Path(_duckdb.__file__).read_bytes()).hexdigest(),
        "selected_source_before": before, "selected_source_after": after,
        "selected_sources_unchanged": before == after, "resource_ledgers": ledgers,
        "selected_test_targets": list(targets),
        "scope": "native test cases in offline bounded image; worker UID isolation is separately qualified"}
    summary["passed"] = summary["tests"] - summary["failures"] - summary["errors"] - summary["skipped"]
    (output / "current-summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("pytest_exit", "tests", "passed", "failures", "errors", "skipped")}), flush=True)
    if before != after or any(row["active_leases"] or row["waiting_requests"] for row in ledgers):
        return 1
    return int(status)


if __name__ == "__main__":
    raise SystemExit(main())
