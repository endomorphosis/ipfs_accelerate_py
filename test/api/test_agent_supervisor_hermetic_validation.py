from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.code_evidence_graph import (
    ChangedASTSymbol,
    CodeImpactIndex,
)
from ipfs_accelerate_py.agent_supervisor.validation_runtime import (
    HermeticValidationRuntime,
    ValidationCancellationToken,
    ValidationResourceBounds,
    ValidationRuntimeError,
    build_hermetic_validation_runtime,
    build_validation_environment,
)
from ipfs_accelerate_py.agent_supervisor.validation_scheduler import (
    HermeticValidationPolicy,
    ImpactValidationCheck,
    ImpactValidationKind,
    RepositoryValidationPolicy,
    SeededValidationDefect,
    ValidationOutcome,
    ValidationScheduler,
    ValidationTechnique,
    build_impact_selected_validation_dag,
    classify_validation_attempts,
    hermetic_validation_runner,
    validation_benchmark,
)


TREE_ID = "tree:hermetic-validation"


def _hermetic_result(
    runtime_context: HermeticValidationRuntime,
    **values: object,
) -> dict[str, object]:
    return {
        **values,
        "runtime_id": runtime_context.runtime_id,
        "cancellation_id": runtime_context.cancellation_id,
    }


def _index() -> CodeImpactIndex:
    return CodeImpactIndex(
        repository_tree_id=TREE_ID,
        symbol_paths={
            "pkg.provider.value": "pkg/provider.py",
            "pkg.consumer.read": "pkg/consumer.py",
            "tests.consumer.test_read": "test/test_consumer.py",
        },
        symbol_dependencies={
            "pkg.consumer.read": ("pkg.provider.value",),
            "tests.consumer.test_read": ("pkg.consumer.read",),
        },
        path_dependencies={
            "pkg/consumer.py": ("pkg/provider.py",),
            "test/test_consumer.py": ("pkg/consumer.py",),
        },
        validation_targets={
            "integration": ("tests.consumer.test_read",),
        },
    )


def _changed_symbol() -> ChangedASTSymbol:
    return ChangedASTSymbol(
        symbol="pkg.provider.value",
        path="pkg/provider.py",
        interface_changed=True,
    )


def _checks(*, integration_cacheable: bool = False):
    return (
        ImpactValidationCheck(
            "syntax",
            ImpactValidationKind.SYNTAX,
            "python -m compileall -q pkg",
            cacheable=False,
        ),
        ImpactValidationCheck(
            "type",
            ImpactValidationKind.TYPE,
            "python -m mypy pkg",
            cacheable=False,
        ),
        ImpactValidationCheck(
            "interface",
            ImpactValidationKind.INTERFACE,
            "python tools/interface.py",
            cacheable=False,
        ),
        ImpactValidationCheck(
            "unit",
            ImpactValidationKind.UNIT,
            "pytest -q test/test_provider.py",
            cacheable=False,
        ),
        ImpactValidationCheck(
            "integration",
            ImpactValidationKind.INTEGRATION,
            "pytest -q test/test_consumer.py",
            targets=("tests.consumer.test_read",),
            cacheable=integration_cacheable,
        ),
        ImpactValidationCheck(
            "contract",
            ImpactValidationKind.CONTRACT,
            "pytest -q test/test_contract.py",
            cacheable=False,
        ),
        ImpactValidationCheck(
            "runtime",
            ImpactValidationKind.RUNTIME,
            "python tools/smoke.py",
            cacheable=False,
        ),
        ImpactValidationCheck(
            "differential",
            ImpactValidationKind.INTEGRATION,
            "python tools/differential.py",
            technique=ValidationTechnique.DIFFERENTIAL,
            cacheable=False,
        ),
        ImpactValidationCheck(
            "metamorphic",
            ImpactValidationKind.UNIT,
            "python tools/metamorphic.py",
            technique=ValidationTechnique.METAMORPHIC,
            cacheable=False,
        ),
        ImpactValidationCheck(
            "mutation",
            ImpactValidationKind.UNIT,
            "python tools/mutation.py",
            technique=ValidationTechnique.MUTATION,
            cacheable=False,
        ),
    )


def test_runtime_identity_pins_every_execution_boundary(tmp_path: Path) -> None:
    runtime = build_hermetic_validation_runtime(
        command="python -c 'print(1)'",
        workspace_path=tmp_path,
        repository_tree_id=TREE_ID,
        environment={
            "AWS_SECRET_ACCESS_KEY": "must-not-cross",
            "CI": "1",
        },
        timeout_seconds=12,
        cancellation_id="cancel:fixture",
        resource_bounds=ValidationResourceBounds(
            cpu_seconds=4,
            memory_bytes=512 * 1024 * 1024,
            output_file_bytes=1024 * 1024,
            open_files=64,
            processes=16,
        ),
    )
    restored = HermeticValidationRuntime.from_dict(runtime.to_dict())

    assert restored.runtime_id == runtime.runtime_id
    assert runtime.command_argv[:4] == (
        "/bin/bash",
        "--noprofile",
        "--norc",
        "-c",
    )
    assert runtime.network_mode.value == "none"
    assert runtime.filesystem_mode.value == "read_only_root_workspace"
    assert runtime.timeout_seconds == 12
    assert runtime.cancellation_id == "cancel:fixture"
    assert runtime.resource_bounds.processes == 16
    assert set(dict(runtime.toolchain)).issuperset(
        {
            "bash_path",
            "bash_sha256",
            "python_path",
            "python_sha256",
            "path_identity",
            "isolation_sha256",
        }
    )
    assert "AWS_SECRET_ACCESS_KEY" not in dict(runtime.environment)
    assert dict(runtime.environment) == build_validation_environment({"CI": "1"})


def test_cancellation_is_fenced_by_exact_identity() -> None:
    token = ValidationCancellationToken("validation:one")

    assert token.cancel(cancellation_id="validation:other") is False
    assert token.cancelled is False
    assert token.cancel(
        cancellation_id="validation:one", reason="newer tree"
    )
    assert token.cancelled is True
    assert token.reason == "newer tree"

    with pytest.raises(ValidationRuntimeError, match="identity"):
        ValidationCancellationToken("")


def test_technique_coverage_is_orthogonal_to_existing_check_kinds() -> None:
    policy = RepositoryValidationPolicy(
        required_techniques=(
            ValidationTechnique.CONTRACT,
            ValidationTechnique.DIFFERENTIAL,
            ValidationTechnique.METAMORPHIC,
            ValidationTechnique.MUTATION,
        )
    )
    plan = build_impact_selected_validation_dag(
        impact_index=_index(),
        checks=_checks(),
        changed_symbols=(_changed_symbol(),),
        repository_policy=policy,
    )

    assert plan.coverage_complete
    selected = {node.check_id: node for node in plan.selected_nodes}
    assert {
        selected[name].check.technique
        for name in ("contract", "differential", "metamorphic", "mutation")
    } == {
        ValidationTechnique.CONTRACT,
        ValidationTechnique.DIFFERENTIAL,
        ValidationTechnique.METAMORPHIC,
        ValidationTechnique.MUTATION,
    }

    incomplete = build_impact_selected_validation_dag(
        impact_index=_index(),
        checks=tuple(
            item for item in _checks() if item.check_id != "mutation"
        ),
        changed_symbols=(_changed_symbol(),),
        repository_policy=policy,
    )
    assert "missing_mandatory_mutation_technique" in incomplete.uncovered_impact


@pytest.mark.parametrize(
    ("attempts", "expected"),
    (
        ([{"returncode": 7}, {"returncode": 7}], "deterministic_failure"),
        ([{"returncode": 0}, {"returncode": 7}], "flaky"),
        ([{"returncode": 124, "timed_out": True}], "timeout"),
        (
            [{"returncode": 75, "infrastructure_failure": True}],
            "infrastructure_failure",
        ),
        ([{"returncode": 79, "inconclusive": True}], "inconclusive"),
        ([{"returncode": 130, "cancelled": True}], "cancelled"),
        ([{"returncode": 0}, {"returncode": 0}], "passed"),
    ),
)
def test_terminal_outcomes_are_disjoint(attempts, expected: str) -> None:
    assert classify_validation_attempts(attempts).value == expected


def test_complete_dag_stabilizes_flakes_and_detects_transitive_seed(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, int]] = []
    runtime_ids: dict[str, str] = {}

    @hermetic_validation_runner
    def runner(*, spec, runtime_context, attempt_number, **_kwargs):
        assert isinstance(runtime_context, HermeticValidationRuntime)
        assert runtime_context.repository_tree_id == TREE_ID
        calls.append((spec.validation_id, attempt_number))
        runtime_ids.setdefault(spec.validation_id, runtime_context.runtime_id)
        assert runtime_ids[spec.validation_id] == runtime_context.runtime_id
        if spec.validation_id == "syntax":
            return _hermetic_result(
                runtime_context,
                returncode=2,
                output="syntax failed",
            )
        if spec.validation_id == "integration":
            return _hermetic_result(
                runtime_context,
                returncode=7,
                output="consumer exposed seeded defect",
                seeded_defect_id="seed:transitive-provider",
            )
        if spec.validation_id == "differential":
            return _hermetic_result(
                runtime_context,
                returncode=0 if attempt_number == 1 else 9,
                output=f"differential attempt {attempt_number}",
            )
        if spec.validation_id == "metamorphic":
            raise subprocess.TimeoutExpired(spec.command, 1)
        if spec.validation_id == "mutation":
            raise RuntimeError("sandbox service unavailable")
        if spec.validation_id == "runtime":
            return _hermetic_result(
                runtime_context,
                returncode=0,
                inconclusive=True,
            )
        return _hermetic_result(
            runtime_context,
            returncode=0,
            output="passed",
        )

    report = ValidationScheduler(
        max_workers=4,
        resource_budget=4,
    ).run_hermetic_impact_selected(
        _checks(),
        workspace_path=tmp_path,
        impact_index=_index(),
        changed_symbols=(_changed_symbol(),),
        dependency_state="fixture",
        runner=runner,
        seeded_defects=(
            SeededValidationDefect(
                "seed:transitive-provider",
                "pkg/provider.py",
                ("integration",),
            ),
        ),
        baseline_time_to_first_failure_seconds=(10.0, 11.0, 12.0),
        optimized_time_to_first_failure_seconds=(6.0, 6.5, 7.0),
    )

    assert report["passed"] is False
    assert {check_id for check_id, _attempt in calls} == {
        node["check_id"]
        for node in report["impact_validation_receipt"]["dag"]["nodes"]
        if node["selected"]
    }
    results = {
        result["validation_id"]: result for result in report["results"]
    }
    assert results["syntax"]["outcome"] == "deterministic_failure"
    assert results["differential"]["outcome"] == "flaky"
    assert results["differential"]["intermittent_pass"] is True
    assert results["differential"]["authoritative"] is False
    assert results["metamorphic"]["outcome"] == "timeout"
    assert results["mutation"]["outcome"] == "infrastructure_failure"
    assert results["runtime"]["outcome"] == "inconclusive"
    assert report["seeded_defect_summary"] == {
        "seeded_count": 1,
        "detected_count": 1,
        "escaped_count": 0,
        "zero_escaped": True,
    }
    assert report["seeded_defects"][0]["transitive"] is True
    assert report["seeded_defects"][0]["transitive_impact_chains"][
        "integration"
    ] == [
        "pkg.provider.value",
        "pkg.consumer.read",
        "tests.consumer.test_read",
    ]
    assert report["time_to_first_failure_benchmark"]["reduction"] >= 0.30
    assert report["time_to_first_failure_benchmark"]["passed"] is True
    assert report["completion_authoritative"] is False
    assert report["authoritative"] is False


def test_exact_deterministic_diagnostic_is_reused_without_authority(
    tmp_path: Path,
) -> None:
    integration_calls = 0

    @hermetic_validation_runner
    def runner(*, spec, runtime_context, **_kwargs):
        nonlocal integration_calls
        if spec.validation_id == "integration":
            integration_calls += 1
            return _hermetic_result(
                runtime_context,
                returncode=7,
                output="exact stable diagnostic",
            )
        return _hermetic_result(
            runtime_context,
            returncode=0,
            output="passed",
        )

    scheduler = ValidationScheduler(
        cache_dir=tmp_path / "cache",
        max_workers=4,
        resource_budget=4,
    )
    kwargs = {
        "workspace_path": tmp_path,
        "impact_index": _index(),
        "changed_symbols": (_changed_symbol(),),
        "dependency_state": "fixture",
        "runner": runner,
    }
    first = scheduler.run_hermetic_impact_selected(
        _checks(integration_cacheable=True), **kwargs
    )
    second = scheduler.run_hermetic_impact_selected(
        _checks(integration_cacheable=True), **kwargs
    )
    first_result = next(
        item
        for item in first["results"]
        if item["command"] == "pytest -q test/test_consumer.py"
    )
    second_result = next(
        item
        for item in second["results"]
        if item["command"] == "pytest -q test/test_consumer.py"
    )

    assert integration_calls == 2
    assert first_result["attempt_count"] == 2
    assert second_result["diagnostic_cache_hit"] is True
    assert second_result["output"] == "exact stable diagnostic"
    assert (
        second_result["validation_result_digest"]
        == first_result["validation_result_digest"]
    )
    assert second_result["diagnostic_id"] == first_result["diagnostic_id"]
    assert second_result["authoritative"] is False
    assert second["completion_authoritative"] is False


def test_benchmark_requires_thirty_percent_median_improvement() -> None:
    passing = validation_benchmark(
        baseline_seconds=(10, 11, 12),
        optimized_seconds=(6, 7, 7.5),
    )
    failing = validation_benchmark(
        baseline_seconds=(10, 11, 12),
        optimized_seconds=(8, 8.5, 9),
    )

    assert passing["passed"] is True
    assert passing["reduction"] >= 0.30
    assert failing["passed"] is False
