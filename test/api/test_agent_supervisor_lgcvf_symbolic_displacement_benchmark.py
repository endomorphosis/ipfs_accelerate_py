"""Focused contract tests for the standalone LGCVF benchmark command."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "benchmark_lgcvf_symbolic_displacement.py"


def _load_script() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "benchmark_lgcvf_symbolic_displacement_tested", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _vertical_result(*, context_reduction_bps: int = 0) -> dict[str, Any]:
    paired = {
        "schema": "lgcvf-paired-benchmark@1",
        "cohort": "hermetic_local_execution",
        "production_authoritative": False,
        "baseline": {
            "context_tokens": 400,
            "model_calls": 0,
            "raw_source_bytes": 1_600,
            "tests_selected": 3,
            "verification_strategy": "raw-source-full-fixture",
        },
        "challenger": {
            "abstract_state_reused": 1,
            "capsule_reuse_count": 3,
            "context_tokens": 500,
            "deterministic_closures": 1,
            "model_calls": 0,
            "proof_test_reuse_bps": 10_000,
            "selected_tests": 2,
            "solver_session_replay_manifests": 1,
            "verification_strategy": "contracts-and-incremental-verification",
        },
        "comparison": {
            "accepted_patch_quality_equal": True,
            "context_reduction_bps": context_reduction_bps,
            "critical_omissions_accepted": 0,
            "model_call_reduction_bps": 0,
            "safety_floor_violations": 0,
        },
        "limitations": ["fixture only"],
    }
    return {
        "benchmark": paired,
        "result_cid": "cid:vertical",
        "release_qualified": False,
        "production_authorized": False,
        "model_invocation_count": 0,
        "fixture": {"base_commit": "a" * 40, "base_tree": "b" * 40},
        "proof_carrying_artifact": {
            "artifact_cid": "cid:artifact",
            "payload": {"policy_root": "cid:policy"},
        },
        "artifact_verification": {
            "replay_receipt_cid": "cid:replay",
            "valid": True,
        },
    }


def _qualification_result(benchmark: ModuleType) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "schema": "lgcvf-independent-pytest-observation@1",
        "suite_id": "fixed-independent-suite",
        "manifest": {"manifest_cid": "cid:manifest"},
        "collected": 1,
        "passed_count": 1,
        "failed_count": 0,
        "skipped_count": 0,
        "xfailed_count": 0,
        "xpassed_count": 0,
        "error_count": 0,
        "nodeids_cid": "cid:nodeids",
        "exit_code": 0,
        "passed": True,
        "isolation": {"network_denied": True, "write_root": "temporary"},
        "duration_ms": 1,
        "transcript_sha256": "sha256:" + "a" * 64,
        "failure_tail": "",
    }
    observation["observation_cid"] = benchmark.content_identity(observation)
    value: dict[str, Any] = {
        "schema": benchmark.QUALIFICATION_SCHEMA,
        "plan_cid": benchmark.QUALIFICATION_PLAN_CID,
        "predecessor_plan_cid": "cid:predecessor",
        "cohort": "hermetic_local_execution",
        "candidate_suites_are_self_authority": False,
        "independent_fixed_manifest_executed": True,
        "checkout_fingerprint_cid": "bafkcheckout",
        "checkout_unchanged": True,
        "passed": True,
        "totals": {
            "collected": 1,
            "passed_count": 1,
            "failed_count": 0,
            "skipped_count": 0,
            "xfailed_count": 0,
            "xpassed_count": 0,
            "error_count": 0,
        },
        "suites": [observation],
        "task_implementation_complete": False,
        "test_qualification_complete": True,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
        "production_authoritative": False,
        "limitations": ["hermetic only"],
    }
    value["result_cid"] = benchmark.content_identity(value)
    return value


def _install_qualification_replay(
    benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    replayed: dict[str, Any],
) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], **kwargs: Any) -> Any:
        calls.append(command)
        assert kwargs == {
            "cwd": benchmark._REPOSITORY_ROOT,
            "check": False,
            "capture_output": True,
            "text": True,
            "timeout": benchmark.QUALIFICATION_REPLAY_TIMEOUT_SECONDS,
        }
        return benchmark.subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(replayed),
            stderr="",
        )

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    return calls


def test_threshold_miss_is_truthful_valid_partial_result() -> None:
    benchmark = _load_script()
    report = benchmark.build_report(_vertical_result(context_reduction_bps=0))

    assert report["overall_disposition"] == "partial"
    thresholds = {item["threshold_id"]: item for item in report["thresholds"]}
    assert thresholds["median_context_reduction_bps"]["disposition"] == "missed"
    assert thresholds["warm_cache_model_call_reduction_bps"]["disposition"] == "not_evaluated"
    assert report["task_class_coverage"]["missing"]
    assert report["production_authoritative"] is False
    assert benchmark.validate_report(report) == ()


def test_command_writes_and_checks_identical_machine_result(tmp_path: Path) -> None:
    benchmark = _load_script()
    output = tmp_path / "benchmark.json"

    def runner(**_kwargs: Any) -> dict[str, Any]:
        return _vertical_result(context_reduction_bps=0)

    assert benchmark.main(("--output", str(output)), runner=runner) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["overall_disposition"] == "partial"
    assert (
        benchmark.main(
            ("--check", "--output", str(output)),
            runner=runner,
            qualification_gate=lambda: "cid:qualification",
        )
        == 0
    )


def test_check_returns_nonzero_for_schema_or_reconstruction_drift(
    tmp_path: Path,
) -> None:
    benchmark = _load_script()
    output = tmp_path / "benchmark.json"

    def initial_runner(**_kwargs: Any) -> dict[str, Any]:
        return _vertical_result(context_reduction_bps=0)

    def changed_runner(**_kwargs: Any) -> dict[str, Any]:
        return _vertical_result(context_reduction_bps=5_000)

    assert benchmark.main(("--output", str(output)), runner=initial_runner) == 0
    assert (
        benchmark.main(
            ("--check", "--output", str(output)),
            runner=changed_runner,
            qualification_gate=lambda: "cid:qualification",
        )
        == 1
    )

    output.write_text('{"schema":"wrong"}\n', encoding="utf-8")
    assert (
        benchmark.main(
            ("--check", "--output", str(output)),
            runner=initial_runner,
            qualification_gate=lambda: "cid:qualification",
        )
        == 1
    )


def test_check_requires_independent_qualification_gate(tmp_path: Path) -> None:
    benchmark = _load_script()
    output = tmp_path / "benchmark.json"

    def runner(**_kwargs: Any) -> dict[str, Any]:
        return _vertical_result(context_reduction_bps=0)

    assert benchmark.main(("--output", str(output)), runner=runner) == 0
    assert benchmark.main(("--check", "--output", str(output)), runner=runner) == 1


def test_qualification_gate_runs_exact_protected_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = _load_script()
    stored = _qualification_result(benchmark)
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(stored), encoding="utf-8")
    replayed = json.loads(json.dumps(stored))
    replayed["suites"][0]["duration_ms"] = 2
    replayed["suites"][0]["observation_cid"] = benchmark.content_identity(
        {
            key: item
            for key, item in replayed["suites"][0].items()
            if key != "observation_cid"
        }
    )
    replayed["result_cid"] = benchmark.content_identity(
        {key: item for key, item in replayed.items() if key != "result_cid"}
    )
    calls = _install_qualification_replay(benchmark, monkeypatch, replayed)

    assert benchmark._validate_independent_qualification_gate(path) == stored["result_cid"]
    assert calls == [
        (
            sys.executable,
            str(benchmark.QUALIFICATION_VALIDATOR),
            "--check",
        )
    ]


def test_self_hashed_minimal_qualification_cannot_open_benchmark_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = _load_script()
    forged: dict[str, Any] = {
        "schema": benchmark.QUALIFICATION_SCHEMA,
        "plan_cid": benchmark.QUALIFICATION_PLAN_CID,
        "passed": True,
        "test_qualification_complete": True,
        "objective_complete": False,
        "release_qualified": False,
        "production_authorized": False,
    }
    forged["result_cid"] = benchmark.content_identity(forged)
    path = tmp_path / "forged-qualification.json"
    path.write_text(json.dumps(forged), encoding="utf-8")
    calls = _install_qualification_replay(
        benchmark,
        monkeypatch,
        _qualification_result(benchmark),
    )

    with pytest.raises(
        benchmark.BenchmarkSchemaError,
        match="qualification suite population is absent",
    ):
        benchmark._validate_independent_qualification_gate(path)
    assert calls == [
        (
            sys.executable,
            str(benchmark.QUALIFICATION_VALIDATOR),
            "--check",
        )
    ]
