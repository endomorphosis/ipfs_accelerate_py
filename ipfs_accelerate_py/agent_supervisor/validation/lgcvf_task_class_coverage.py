"""Paired hermetic checks for representative LGCVF task classes.

Each case copies the vertical fixture, applies one fault, requires tests to
fail, restores the repair, and requires tests to pass.  Classes are recorded
only when both sides are observed.  This is not live-model, release, or
production evidence.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.validation.compositional_verification_vertical import (
    _copy_fixture,
    _fixture_default,
)

VERTICAL_CLASSES: Final[tuple[str, ...]] = (
    "local_bug_repair",
    "cross_module_contract_change",
    "repeated_maintenance_warm_cache",
)

_CASES: Final[tuple[dict[str, str], ...]] = (
    {
        "task_class": "exception_behavior_change",
        "path": "pkg/module_a.py",
        "before": 'raise ValueError("limit must be non-negative")',
        "after": "raise TypeError(limit)",
    },
    {
        "task_class": "configuration_change",
        "path": "pkg/schema.py",
        "before": "MAX_PRODUCED_VALUE = 20",
        "after": "MAX_PRODUCED_VALUE = 1",
    },
    {
        "task_class": "schema_serializer_change",
        "path": "pkg/codec.py",
        "before": 'return f"produced:{value}"',
        "after": 'return f"raw:{value}"',
    },
    {
        "task_class": "dependency_api_migration",
        "path": "pkg/compat.py",
        "before": "from .module_a import produce as produce_v2",
        "after": "from .module_a import missing_produce as produce_v2",
    },
    {
        "task_class": "security_policy_change",
        "path": "pkg/policy.py",
        "before": "MAXIMUM_CALLER_LIMIT = 1_000",
        "after": "MAXIMUM_CALLER_LIMIT = -1",
    },
    {
        "task_class": "concurrency_interference_change",
        "path": "pkg/lock.py",
        "before": "return CONSUMER_LOCK, PRESENTER_LOCK",
        "after": "return PRESENTER_LOCK, CONSUMER_LOCK",
    },
    {
        "task_class": "proof_repair",
        "path": "pkg/proof.py",
        "before": 'OBLIGATION = "producer-upper-bound <= schema.MAX_PRODUCED_VALUE"',
        "after": 'OBLIGATION = "producer-upper-bound is unconstrained"',
    },
    {
        "task_class": "behavior_preserving_refactor",
        "path": "pkg/unaffected.py",
        "before": 'label = "unaffected"\n    return label',
        "after": 'token = "unaffected"\n    return token',
        "preserve_tests": "true",
    },
    {
        "task_class": "dynamic_opaque_python_escalation",
        "path": "pkg/plugin.py",
        "before": 'getter = getattr(module, "stable_label")',
        "after": 'getter = getattr(module, "_missing_label")',
    },
)


def _run_tests(root: Path) -> int:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(root), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests",
        ),
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
        check=False,
    )
    return int(completed.returncode)


def _mutate(root: Path, relative: str, before: str, after: str) -> None:
    path = root / relative
    text = path.read_text(encoding="utf-8")
    if before not in text:
        raise RuntimeError(f"{relative} does not contain the expected baseline text")
    path.write_text(text.replace(before, after, 1), encoding="utf-8")


def run_task_class_coverage_extension(
    *,
    fixture_root: Path | str | None = None,
) -> dict[str, Any]:
    """Observe extra task classes with paired fail/pass pytest oracles."""

    fixture = Path(fixture_root) if fixture_root is not None else _fixture_default()
    observed: list[str] = list(VERTICAL_CLASSES)
    cases: list[dict[str, Any]] = []
    temp_root = Path(tempfile.mkdtemp(prefix="lgcvf-task-class-"))
    try:
        for spec in _CASES:
            work = temp_root / spec["task_class"]
            if work.exists():
                shutil.rmtree(work)
            _copy_fixture(fixture.resolve(), work)
            baseline = _run_tests(work)
            _mutate(work, spec["path"], spec["before"], spec["after"])
            after_fault = _run_tests(work)
            _mutate(work, spec["path"], spec["after"], spec["before"])
            after_repair = _run_tests(work)
            preserve = spec.get("preserve_tests") == "true"
            if preserve:
                exercised = baseline == 0 and after_fault == 0 and after_repair == 0
            else:
                exercised = baseline == 0 and after_fault != 0 and after_repair == 0
            record = {
                "task_class": spec["task_class"],
                "path": spec["path"],
                "baseline_returncode": baseline,
                "fault_returncode": after_fault,
                "repair_returncode": after_repair,
                "preserve_tests": preserve,
                "exercised": exercised,
            }
            cases.append(record)
            if exercised:
                observed.append(spec["task_class"])
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    required = (
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
    unique = []
    for item in observed:
        if item not in unique:
            unique.append(item)
    report: dict[str, Any] = {
        "schema": "lgcvf-task-class-coverage-extension@1",
        "cohort": "hermetic_local_execution",
        "production_authoritative": False,
        "release_qualified": False,
        "production_authorized": False,
        "required": list(required),
        "observed": unique,
        "missing": [item for item in required if item not in unique],
        "cases": cases,
        "limitations": [
            "paired pytest oracles on a hermetic fixture, not a live maintenance suite",
            "vertical-slice classes are imported from the existing doctor route",
            "no live-model, remote-model, or production evidence is aggregated",
        ],
    }
    report["report_cid"] = content_identity(report)
    return report
