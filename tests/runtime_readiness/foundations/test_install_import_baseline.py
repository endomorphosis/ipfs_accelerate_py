"""Root-level implied validation mirror for KITA-004 baseline artifacts.

The authoritative harness and suite live under the nested ``ipfs_kit_py``
package tree. This module asserts the declared outputs exist and that the
nested pytest target remains loadable from a superproject checkout.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[3]
NESTED_ROOT = WORKSPACE / "ipfs_kit_py"
BASELINE_PY = NESTED_ROOT / "benchmarks" / "runtime_readiness" / "baseline.py"
WORKLOADS_JSON = NESTED_ROOT / "benchmarks" / "runtime_readiness" / "workloads.json"
FLOORS_JSON = NESTED_ROOT / "benchmarks" / "runtime_readiness" / "reference_floors.json"
NESTED_TEST = (
    NESTED_ROOT
    / "tests"
    / "runtime_readiness"
    / "foundations"
    / "test_install_import_baseline.py"
)


def test_declared_outputs_present_from_superproject():
    assert BASELINE_PY.is_file()
    assert WORKLOADS_JSON.is_file()
    assert FLOORS_JSON.is_file()
    assert NESTED_TEST.is_file()


def test_floors_provisional_and_committed_primary_metric():
    floors = json.loads(FLOORS_JSON.read_text(encoding="utf-8"))
    workloads = json.loads(WORKLOADS_JSON.read_text(encoding="utf-8"))
    assert floors["status"] == "provisional"
    assert floors["reviewed"] is False
    assert floors["comparison_rules"]["immutable"] is True
    assert workloads["comparison_binding"]["primary_throughput_metric"] == "committed_tps"
    assert floors["observation_anchors"]["transaction_specific_slo"]["slo_present"] is False
    for path_class in ("cold", "warm", "cache"):
        assert path_class in workloads["path_classes"]


def test_nested_check_schema_cli():
    proc = subprocess.run(
        [
            sys.executable,
            str(BASELINE_PY),
            "--profile",
            "ci-reference",
            "--check-schema",
        ],
        cwd=str(NESTED_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    report = json.loads(proc.stdout)
    assert report["ok"] is True
    assert report["primary_throughput_metric"] == "committed_tps"
    assert report["transaction_specific_slo_present"] is False
