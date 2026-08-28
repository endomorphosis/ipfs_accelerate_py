#!/usr/bin/env python3
"""Run and validate the hermetic LGCVF symbolic-displacement benchmark.

The command is a transport wrapper around the public compositional-verification
vertical-slice API. It does not implement a second benchmark or admit its own
evidence. Threshold misses are valid results; non-zero exit means execution,
schema, or checked-result reconstruction failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (  # noqa: E402
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.validation.compositional_verification_vertical import (  # noqa: E402
    VerticalSliceError,
    run_compositional_verification_vertical_slice,
)
from ipfs_accelerate_py.agent_supervisor.validation.lgcvf_task_class_coverage import (  # noqa: E402
    run_task_class_coverage_extension,
)

REPORT_SCHEMA: Final[str] = "lgcvf-symbolic-displacement-benchmark@1"
REPORT_INTERFACE: Final[str] = "LgcvfSymbolicDisplacementBenchmark@1"
PAIRED_RESULT_SCHEMA: Final[str] = "lgcvf-paired-benchmark@1"
DEFAULT_OUTPUT: Final[Path] = (
    _REPOSITORY_ROOT
    / "data"
    / "agent_supervisor"
    / "logic_governed_compositional_verification_fabric"
    / "benchmark_result.json"
)
TASK_CLASS_COVERAGE_OUTPUT: Final[Path] = (
    _REPOSITORY_ROOT
    / "data"
    / "agent_supervisor"
    / "logic_governed_compositional_verification_fabric"
    / "task_class_coverage_extension.json"
)
MODEL_CALL_DISPLACEMENT_OUTPUT: Final[Path] = (
    _REPOSITORY_ROOT
    / "data"
    / "agent_supervisor"
    / "logic_governed_compositional_verification_fabric"
    / "model_call_displacement.json"
)
CONTEXT_REDUCTION_OUTPUT: Final[Path] = (
    _REPOSITORY_ROOT
    / "data"
    / "agent_supervisor"
    / "logic_governed_compositional_verification_fabric"
    / "context_reduction_measurement.json"
)
QUALIFICATION_OUTPUT: Final[Path] = (
    _REPOSITORY_ROOT
    / "data"
    / "agent_supervisor"
    / "logic_governed_compositional_verification_fabric"
    / "independent_qualification_result.json"
)
QUALIFICATION_SCHEMA: Final[str] = "lgcvf-independent-hermetic-qualification@1"
QUALIFICATION_PLAN_CID: Final[str] = (
    "baguqeerabxn5kkewz44v4chz6vbt3kcozfj4rvhh4gpdp54645blhemhvloq"
)
QUALIFICATION_VALIDATOR: Final[Path] = (
    _REPOSITORY_ROOT
    / "scripts"
    / "qualify_logic_governed_compositional_verification_fabric.py"
)
# Ordinary qualification permits one protected suite to consume up to 3,600
# seconds.  Allow bounded setup and the preceding short suites as well.
QUALIFICATION_REPLAY_TIMEOUT_SECONDS: Final[int] = 4_000

REQUIRED_TASK_CLASSES: Final[tuple[str, ...]] = (
    "local_bug_repair",
    "cross_module_contract_change",
    "exception_behavior_change",
    "schema_serializer_change",
    "configuration_change",
    "dependency_api_migration",
    "security_policy_change",
    "concurrency_interference_change",
    "proof_repair",
    "behavior_preserving_refactor",
    "dynamic_opaque_python_escalation",
    "repeated_maintenance_warm_cache",
)

# Containing an exception or configuration edge does not count as mutating that
# behavior. Only the classes actually exercised by this fixture are listed.
OBSERVED_TASK_CLASSES: Final[tuple[str, ...]] = (
    "local_bug_repair",
    "cross_module_contract_change",
    "repeated_maintenance_warm_cache",
)


class BenchmarkSchemaError(RuntimeError):
    """The benchmark report cannot be admitted as the declared schema."""


VerticalRunner = Callable[..., dict[str, Any]]
QualificationGate = Callable[[], str]


def _threshold(
    *,
    threshold_id: str,
    target: int | bool,
    observed: int | bool | None,
    comparison: str,
    not_evaluated_reason: str = "",
) -> dict[str, Any]:
    """Evaluate one threshold without converting unavailable data to success."""

    if observed is None:
        return {
            "comparison": comparison,
            "disposition": "not_evaluated",
            "observed": None,
            "reason": not_evaluated_reason,
            "target": target,
            "threshold_id": threshold_id,
        }
    if comparison == "equal":
        met = observed == target
    elif comparison == "at_least":
        if isinstance(observed, bool) or isinstance(target, bool):
            raise BenchmarkSchemaError(f"{threshold_id}: at_least requires integer operands")
        met = observed >= target
    else:
        raise BenchmarkSchemaError(f"{threshold_id}: unsupported comparison")
    return {
        "comparison": comparison,
        "disposition": "met" if met else "missed",
        "observed": observed,
        "reason": "",
        "target": target,
        "threshold_id": threshold_id,
    }


def _build_thresholds(
    paired: Mapping[str, Any],
    *,
    observed_classes: Sequence[str] = OBSERVED_TASK_CLASSES,
) -> list[dict[str, Any]]:
    comparison = paired["comparison"]
    challenger = paired["challenger"]
    return [
        _threshold(
            threshold_id="zero_safety_floor_violations",
            target=0,
            observed=comparison["safety_floor_violations"],
            comparison="equal",
        ),
        _threshold(
            threshold_id="zero_critical_omissions_accepted",
            target=0,
            observed=comparison["critical_omissions_accepted"],
            comparison="equal",
        ),
        _threshold(
            threshold_id="median_context_reduction_bps",
            target=5_000,
            observed=comparison["context_reduction_bps"],
            comparison="at_least",
        ),
        _threshold(
            threshold_id="warm_cache_model_call_reduction_bps",
            target=5_000,
            observed=None,
            comparison="at_least",
            not_evaluated_reason=(
                "both fixture routes made zero model calls; a repeated task with "
                "a nonzero baseline is required to measure displacement"
            ),
        ),
        _threshold(
            threshold_id="symbolically_closable_deterministic_route_share_bps",
            target=2_500,
            observed=10_000 if challenger["deterministic_closures"] == 1 else 0,
            comparison="at_least",
        ),
        _threshold(
            threshold_id="unaffected_proof_test_reuse_bps",
            target=8_000,
            observed=challenger["proof_test_reuse_bps"],
            comparison="at_least",
        ),
        _threshold(
            threshold_id="accepted_patch_quality_not_lower",
            target=True,
            observed=comparison["accepted_patch_quality_equal"],
            comparison="equal",
        ),
        _threshold(
            threshold_id="representative_task_class_coverage",
            target=len(REQUIRED_TASK_CLASSES),
            observed=len(tuple(observed_classes)),
            comparison="at_least",
        ),
    ]


def _overall_disposition(thresholds: Sequence[Mapping[str, Any]]) -> str:
    safety_ids = {
        "zero_safety_floor_violations",
        "zero_critical_omissions_accepted",
        "accepted_patch_quality_not_lower",
    }
    if any(
        item["threshold_id"] in safety_ids and item["disposition"] != "met" for item in thresholds
    ):
        return "no_go"
    if any(item["disposition"] != "met" for item in thresholds):
        return "partial"
    return "development_targets_met"


def _reproducible_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return evidence expected to reproduce despite fresh test-run receipts."""

    return {
        key: report.get(key)
        for key in (
            "schema",
            "interface",
            "cohort",
            "production_authoritative",
            "release_qualified",
            "production_authorized",
            "overall_disposition",
            "pairing",
            "task_class_coverage",
            "paired_result",
            "thresholds",
            "excluded_cohorts",
            "limitations",
        )
    }


def build_report(
    vertical_result: Mapping[str, Any],
    *,
    observed_classes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Project independently checked public-route output into a paired report."""

    paired = vertical_result.get("benchmark")
    if not isinstance(paired, Mapping):
        raise BenchmarkSchemaError("vertical result has no paired benchmark mapping")
    observed = (
        tuple(observed_classes)
        if observed_classes is not None
        else OBSERVED_TASK_CLASSES
    )
    thresholds = _build_thresholds(paired, observed_classes=observed)
    missing = sorted(set(REQUIRED_TASK_CLASSES) - set(observed))
    artifact = vertical_result.get("proof_carrying_artifact")
    artifact_verification = vertical_result.get("artifact_verification")
    fixture = vertical_result.get("fixture")
    if not all(isinstance(item, Mapping) for item in (artifact, artifact_verification, fixture)):
        raise BenchmarkSchemaError("vertical evidence bindings are absent")
    artifact_payload = artifact.get("payload")
    if not isinstance(artifact_payload, Mapping):
        raise BenchmarkSchemaError("proof-carrying artifact payload is absent")
    if artifact_verification.get("valid") is not True:
        raise BenchmarkSchemaError("proof-carrying artifact was not independently validated")

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "interface": REPORT_INTERFACE,
        "cohort": paired.get("cohort"),
        "production_authoritative": paired.get("production_authoritative"),
        "release_qualified": bool(vertical_result.get("release_qualified", False)),
        "production_authorized": bool(vertical_result.get("production_authorized", False)),
        "overall_disposition": _overall_disposition(thresholds),
        "execution_evidence": {
            # Test execution receipts intentionally bind per-run observations
            # such as durations. Their identities need not reproduce; the
            # semantic benchmark projection below must.
            "vertical_result_cid": vertical_result.get("result_cid"),
            "artifact_cid": artifact.get("artifact_cid"),
            "artifact_verification_receipt_cid": artifact_verification.get("replay_receipt_cid"),
            "fresh_execution_receipts_reproducible": False,
        },
        "pairing": {
            "scope": "single_vertical_run_same_fixture_and_acceptance_oracle",
            "repository_base_commit": fixture.get("base_commit"),
            "repository_base_tree": fixture.get("base_tree"),
            "policy_root": artifact_payload.get("policy_root"),
            "model_invocation_count": vertical_result.get("model_invocation_count"),
            "note": (
                "the raw-source baseline is a context/test-selection projection; "
                "separate baseline wall-time and resource execution is not measured"
            ),
        },
        "task_class_coverage": {
            "required": list(REQUIRED_TASK_CLASSES),
            "observed": list(observed),
            "missing": missing,
        },
        "paired_result": dict(paired),
        "thresholds": thresholds,
        "excluded_cohorts": [
            "simulated",
            "live_local_model_execution",
            "live_remote_model_execution",
            "production_authoritative_evidence",
        ],
        "limitations": [
            "this is a hermetic local fixture, not a representative maintenance suite",
            "threshold misses and unavailable measurements remain visible",
            "no live-model, remote-model, external-verifier, release, or "
            "production evidence is aggregated",
        ],
    }
    report["reproducible_projection_cid"] = content_identity(_reproducible_projection(report))
    report["report_cid"] = content_identity(report)
    issues = validate_report(report)
    if issues:
        raise BenchmarkSchemaError("; ".join(issues))
    return report


def _is_nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_report(report: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate schema and identity only; a missed threshold remains valid data."""

    issues: list[str] = []
    if report.get("schema") != REPORT_SCHEMA:
        issues.append("schema_mismatch")
    if report.get("interface") != REPORT_INTERFACE:
        issues.append("interface_mismatch")
    if report.get("cohort") != "hermetic_local_execution":
        issues.append("cohort_not_hermetic_local_execution")
    if report.get("production_authoritative") is not False:
        issues.append("hermetic_cohort_claims_production_authority")
    if report.get("release_qualified") is not False:
        issues.append("fixture_claims_release_qualification")
    if report.get("production_authorized") is not False:
        issues.append("fixture_claims_production_authorization")
    if report.get("overall_disposition") not in {
        "development_targets_met",
        "partial",
        "no_go",
    }:
        issues.append("invalid_overall_disposition")

    execution_evidence = report.get("execution_evidence")
    if not isinstance(execution_evidence, Mapping):
        issues.append("execution_evidence_missing")
    elif execution_evidence.get("fresh_execution_receipts_reproducible") is not False:
        issues.append("fresh_execution_receipts_misclassified")
    pairing = report.get("pairing")
    if not isinstance(pairing, Mapping):
        issues.append("pairing_missing")
    elif not isinstance(pairing.get("policy_root"), str) or not pairing.get("policy_root"):
        issues.append("pairing_policy_root_missing")

    paired = report.get("paired_result")
    if not isinstance(paired, Mapping):
        issues.append("paired_result_missing")
    else:
        if paired.get("schema") != PAIRED_RESULT_SCHEMA:
            issues.append("paired_result_schema_mismatch")
        for section in ("baseline", "challenger", "comparison"):
            if not isinstance(paired.get(section), Mapping):
                issues.append(f"paired_result_{section}_missing")
        if paired.get("cohort") != report.get("cohort"):
            issues.append("paired_result_cohort_mismatch")
        if paired.get("production_authoritative") is not False:
            issues.append("paired_result_claims_production_authority")

    thresholds = report.get("thresholds")
    if not isinstance(thresholds, list) or not thresholds:
        issues.append("thresholds_missing")
    else:
        seen: set[str] = set()
        for index, item in enumerate(thresholds):
            if not isinstance(item, Mapping):
                issues.append(f"threshold_{index}_not_mapping")
                continue
            threshold_id = item.get("threshold_id")
            if not isinstance(threshold_id, str) or not threshold_id:
                issues.append(f"threshold_{index}_identity_missing")
            elif threshold_id in seen:
                issues.append(f"threshold_{threshold_id}_duplicate")
            else:
                seen.add(threshold_id)
            if item.get("disposition") not in {"met", "missed", "not_evaluated"}:
                issues.append(f"threshold_{index}_invalid_disposition")
            observed = item.get("observed")
            if observed is not None and not (
                isinstance(observed, bool) or _is_nonnegative_integer(observed)
            ):
                issues.append(f"threshold_{index}_invalid_observed")

    coverage = report.get("task_class_coverage")
    if not isinstance(coverage, Mapping):
        issues.append("task_class_coverage_missing")
    else:
        required = coverage.get("required")
        observed = coverage.get("observed")
        missing = coverage.get("missing")
        if required != list(REQUIRED_TASK_CLASSES):
            issues.append("required_task_classes_mismatch")
        if not isinstance(observed, list) or not isinstance(missing, list):
            issues.append("task_class_coverage_invalid")
        elif sorted(set(required) - set(observed)) != missing:
            issues.append("missing_task_classes_not_reconstructed")

    claimed_cid = report.get("report_cid")
    if not isinstance(claimed_cid, str) or not claimed_cid:
        issues.append("report_cid_missing")
    else:
        payload = {key: value for key, value in report.items() if key != "report_cid"}
        if content_identity(payload) != claimed_cid:
            issues.append("report_cid_mismatch")
    projection_cid = report.get("reproducible_projection_cid")
    if not isinstance(projection_cid, str) or not projection_cid:
        issues.append("reproducible_projection_cid_missing")
    elif content_identity(_reproducible_projection(report)) != projection_cid:
        issues.append("reproducible_projection_cid_mismatch")
    return tuple(sorted(set(issues)))


def run_benchmark(
    *,
    fixture_root: Path | None = None,
    runner: VerticalRunner = run_compositional_verification_vertical_slice,
    persist_successors: bool = False,
) -> dict[str, Any]:
    """Execute the existing public route and build its benchmark projection."""

    result = runner(fixture_root=fixture_root)
    if runner is run_compositional_verification_vertical_slice:
        coverage = run_task_class_coverage_extension(fixture_root=fixture_root)
        observed = tuple(coverage["observed"])
    else:
        coverage = {
            "schema": "lgcvf-task-class-coverage-extension@1",
            "observed": list(OBSERVED_TASK_CLASSES),
            "missing": sorted(set(REQUIRED_TASK_CLASSES) - set(OBSERVED_TASK_CLASSES)),
        }
        observed = OBSERVED_TASK_CLASSES
    report = build_report(result, observed_classes=observed)
    if persist_successors:
        _write_successor_measurements(report, coverage)
    return report


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_successor_measurements(
    report: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> None:
    """Persist S003-S005 measurements without claiming 121/123 authority."""

    _write_json(TASK_CLASS_COVERAGE_OUTPUT, coverage)
    paired = report.get("paired_result")
    comparison = paired.get("comparison") if isinstance(paired, Mapping) else {}
    challenger = paired.get("challenger") if isinstance(paired, Mapping) else {}
    baseline = paired.get("baseline") if isinstance(paired, Mapping) else {}
    thresholds = {
        item.get("threshold_id"): item
        for item in report.get("thresholds") or ()
        if isinstance(item, Mapping)
    }
    model_calls = int(challenger.get("model_calls") or 0) + int(
        baseline.get("model_calls") or 0
    )
    model_payload = {
        "schema": "lgcvf-model-call-displacement@1",
        "cohort": "hermetic_local_execution",
        "production_authoritative": False,
        "release_qualified": False,
        "production_authorized": False,
        "baseline_model_calls": baseline.get("model_calls"),
        "challenger_model_calls": challenger.get("model_calls"),
        "threshold": thresholds.get("warm_cache_model_call_reduction_bps"),
        "evaluated": model_calls > 0,
        "limitations": [
            "both paired routes remain zero-model in this hermetic fixture",
            "live local/remote model displacement is outside this measurement",
        ],
    }
    model_payload["report_cid"] = content_identity(model_payload)
    _write_json(MODEL_CALL_DISPLACEMENT_OUTPUT, model_payload)
    context_payload = {
        "schema": "lgcvf-context-reduction-measurement@1",
        "cohort": "hermetic_local_execution",
        "production_authoritative": False,
        "release_qualified": False,
        "production_authorized": False,
        "baseline_context_tokens": baseline.get("context_tokens"),
        "challenger_context_tokens": challenger.get("context_tokens"),
        "context_reduction_bps": comparison.get("context_reduction_bps"),
        "threshold": thresholds.get("median_context_reduction_bps"),
        "limitations": [
            "measurement is the single hermetic fixture paired projection",
            "a typed miss remains visible and is valid successor output",
        ],
    }
    context_payload["report_cid"] = content_identity(context_payload)
    _write_json(CONTEXT_REDUCTION_OUTPUT, context_payload)


def write_report_atomic(report: Mapping[str, Any], destination: Path) -> Path:
    """Write a validated report without exposing a partial output file."""

    issues = validate_report(report)
    if issues:
        raise BenchmarkSchemaError("; ".join(issues))
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".lgcvf-benchmark-", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return destination


def _load_checked_report(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkSchemaError(f"unable to read checked report: {error}") from error
    if not isinstance(loaded, dict):
        raise BenchmarkSchemaError("checked report is not a JSON object")
    issues = validate_report(loaded)
    if issues:
        raise BenchmarkSchemaError("; ".join(issues))
    return loaded


def _qualification_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Match the protected qualifier's reproducible evidence projection."""

    observations = value.get("suites")
    if not isinstance(observations, list):
        raise BenchmarkSchemaError("independent qualification suite population is absent")
    suites: list[dict[str, Any]] = []
    suite_fields = (
        "schema",
        "suite_id",
        "manifest",
        "collected",
        "passed_count",
        "failed_count",
        "skipped_count",
        "xfailed_count",
        "xpassed_count",
        "error_count",
        "nodeids_cid",
        "exit_code",
        "passed",
        "isolation",
    )
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping):
            raise BenchmarkSchemaError(
                f"independent qualification suite {index} is not an object"
            )
        suites.append({field: observation.get(field) for field in suite_fields})
    stable_fields = (
        "schema",
        "plan_cid",
        "predecessor_plan_cid",
        "cohort",
        "candidate_suites_are_self_authority",
        "independent_fixed_manifest_executed",
        "checkout_fingerprint_cid",
        "checkout_unchanged",
        "passed",
        "totals",
        "task_implementation_complete",
        "test_qualification_complete",
        "objective_complete",
        "release_qualified",
        "production_authorized",
        "production_authoritative",
        "limitations",
    )
    return {field: value.get(field) for field in stable_fields} | {"suites": suites}


def _run_protected_qualification_validator() -> dict[str, Any]:
    """Reconstruct qualification in the repository-owned isolated judge."""

    try:
        validator = QUALIFICATION_VALIDATOR.resolve(strict=True)
    except OSError as error:
        raise BenchmarkSchemaError(
            f"protected qualification validator is unavailable: {error}"
        ) from error
    if validator != QUALIFICATION_VALIDATOR.absolute() or not validator.is_file():
        raise BenchmarkSchemaError(
            "protected qualification validator path is not an exact regular file"
        )
    try:
        completed = subprocess.run(
            (sys.executable, str(validator), "--check"),
            cwd=_REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=QUALIFICATION_REPLAY_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BenchmarkSchemaError(
            f"protected qualification reconstruction failed: {error}"
        ) from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2_000:]
        raise BenchmarkSchemaError(
            "protected qualification reconstruction returned "
            f"{completed.returncode}: {detail}"
        )
    try:
        replayed = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise BenchmarkSchemaError(
            "protected qualification reconstruction did not emit one JSON object"
        ) from error
    if not isinstance(replayed, dict):
        raise BenchmarkSchemaError(
            "protected qualification reconstruction root is not an object"
        )
    return replayed


def _validate_independent_qualification_gate(
    path: Path = QUALIFICATION_OUTPUT,
) -> str:
    """Require the protected LGCVF-113 result before benchmark admission."""

    try:
        encoded = path.read_bytes()
        value = json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkSchemaError(
            f"independent qualification result is unavailable: {error}"
        ) from error
    if not isinstance(value, dict):
        raise BenchmarkSchemaError("independent qualification result is not an object")
    claimed = value.get("result_cid")
    body = {key: item for key, item in value.items() if key != "result_cid"}
    if not isinstance(claimed, str) or content_identity(body) != claimed:
        raise BenchmarkSchemaError("independent qualification result identity differs")
    if (
        value.get("schema") != QUALIFICATION_SCHEMA
        or value.get("plan_cid") != QUALIFICATION_PLAN_CID
        or value.get("passed") is not True
        or value.get("test_qualification_complete") is not True
    ):
        raise BenchmarkSchemaError("independent qualification is stale or unsuccessful")
    if any(
        value.get(field) is not False
        for field in ("objective_complete", "release_qualified", "production_authorized")
    ):
        raise BenchmarkSchemaError("independent qualification raises unsupported authority")
    replayed = _run_protected_qualification_validator()
    try:
        if path.read_bytes() != encoded:
            raise BenchmarkSchemaError(
                "independent qualification changed during protected reconstruction"
            )
    except OSError as error:
        raise BenchmarkSchemaError(
            f"independent qualification disappeared during reconstruction: {error}"
        ) from error
    replayed_claimed = replayed.get("result_cid")
    replayed_body = {
        key: item for key, item in replayed.items() if key != "result_cid"
    }
    if (
        not isinstance(replayed_claimed, str)
        or content_identity(replayed_body) != replayed_claimed
    ):
        raise BenchmarkSchemaError(
            "protected qualification reconstruction identity differs"
        )
    if _qualification_projection(value) != _qualification_projection(replayed):
        raise BenchmarkSchemaError(
            "protected qualification reconstruction differs from stored evidence"
        )
    return claimed


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: VerticalRunner = run_compositional_verification_vertical_slice,
    qualification_gate: QualificationGate = _validate_independent_qualification_gate,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="reconstruct and compare the checked machine result without writing",
    )
    parser.add_argument("--json", action="store_true", help="print the full report")
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if arguments.check:
            qualification_gate()
        existing = _load_checked_report(arguments.output) if arguments.check else None
        report = run_benchmark(
            fixture_root=arguments.fixture_root,
            runner=runner,
            persist_successors=not arguments.check
            and runner is run_compositional_verification_vertical_slice,
        )
        if existing is not None and _reproducible_projection(existing) != _reproducible_projection(
            report
        ):
            raise BenchmarkSchemaError(
                "checked_result_drift: fresh reproducible projection differs from output"
            )
        if not arguments.check:
            write_report_atomic(report, arguments.output)
    except (BenchmarkSchemaError, VerticalSliceError, OSError) as error:
        print(
            json.dumps(
                {"error": str(error), "schema": REPORT_SCHEMA, "status": "failed"},
                sort_keys=True,
            )
        )
        return 1

    summary = {
        "cohort": report["cohort"],
        "overall_disposition": report["overall_disposition"],
        "output": str(arguments.output),
        "report_cid": report["report_cid"],
        "reproducible_projection_cid": report["reproducible_projection_cid"],
        "status": "checked" if arguments.check else "completed",
    }
    print(json.dumps(report if arguments.json else summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
