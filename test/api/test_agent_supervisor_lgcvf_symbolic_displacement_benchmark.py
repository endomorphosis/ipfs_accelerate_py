"""Focused contract tests for the standalone LGCVF benchmark command."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

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
    assert benchmark.main(("--check", "--output", str(output)), runner=runner) == 0


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
    assert benchmark.main(("--check", "--output", str(output)), runner=changed_runner) == 1

    output.write_text('{"schema":"wrong"}\n', encoding="utf-8")
    assert benchmark.main(("--check", "--output", str(output)), runner=initial_runner) == 1
