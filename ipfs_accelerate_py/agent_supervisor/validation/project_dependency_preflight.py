"""Detect static project dependency drift before implementation dispatch.

The preflight is deliberately detection-only.  Importing this module performs
no subprocess, package-manager, network, or filesystem mutation.  At explicit
call time it reads only static PEP-621 dependency declarations and evaluates
them in the same approved, sealed Python environment used by authoritative
validation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

_CHILD_PROBE_MODE = __name__ == "__main__" and "--probe" in sys.argv[1:]
if not _CHILD_PROBE_MODE:
    from .validation_commands import validation_command_repository_root
    from .validation_runtime import (
        build_validation_environment,
        sealed_validation_python_runner,
        validation_environment_for_runner,
        validation_python_launcher_environment,
    )

PROJECT_DEPENDENCY_PREFLIGHT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/validation-project-dependency-preflight@1"
)
PROJECT_DEPENDENCY_PROBE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/validation-project-dependency-probe@1"
)
PROJECT_DEPENDENCY_PREFLIGHT_BACKOFF_SECONDS = 300
MAX_PYPROJECT_BYTES = 2 * 1024 * 1024
MAX_STATIC_REQUIREMENTS = 512
MAX_REQUIREMENT_BYTES = 2048
MAX_PROBE_OUTPUT_BYTES = 2 * 1024 * 1024
DEPENDENCY_PROBE_TIMEOUT_SECONDS = 30.0


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _load_pyproject(payload: bytes) -> Mapping[str, Any]:
    """Parse TOML lazily so importing the supervisor has no parser side effect."""

    text = payload.decode("utf-8")
    try:
        import tomllib

        parsed = tomllib.loads(text)
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        try:
            import tomli

            parsed = tomli.loads(text)
        except ModuleNotFoundError:
            import toml

            parsed = toml.loads(text)
    if not isinstance(parsed, Mapping):
        raise ValueError("pyproject.toml must contain a TOML table")
    return parsed


def _bounded_static_project(
    workspace_path: Path,
    relative_root: str,
) -> dict[str, Any]:
    """Read one safe project root and return its static PEP-621 contract."""

    workspace = workspace_path.resolve(strict=True)
    candidate = workspace / relative_root
    try:
        project_root = candidate.resolve(strict=True)
    except FileNotFoundError:
        return {
            "root": relative_root,
            "applicable": False,
            "passed": True,
            "reason": "project_root_not_present_before_implementation",
        }
    try:
        project_root.relative_to(workspace)
    except ValueError:
        return {
            "root": relative_root,
            "applicable": True,
            "passed": False,
            "reason": "project_root_escapes_workspace",
        }
    if not project_root.is_dir():
        return {
            "root": relative_root,
            "applicable": True,
            "passed": False,
            "reason": "project_root_is_not_directory",
        }

    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.is_file():
        return {
            "root": relative_root,
            "applicable": False,
            "passed": True,
            "reason": "pep621_metadata_not_present",
        }
    try:
        payload = pyproject_path.read_bytes()
    except OSError as exc:
        return {
            "root": relative_root,
            "applicable": True,
            "passed": False,
            "reason": "pyproject_read_failed",
            "error_type": type(exc).__name__,
        }
    if len(payload) > MAX_PYPROJECT_BYTES:
        return {
            "root": relative_root,
            "applicable": True,
            "passed": False,
            "reason": "pyproject_exceeds_preflight_bound",
            "pyproject_bytes": len(payload),
            "maximum_pyproject_bytes": MAX_PYPROJECT_BYTES,
        }

    pyproject_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        parsed = _load_pyproject(payload)
    except (UnicodeError, ValueError) as exc:
        return {
            "root": relative_root,
            "applicable": True,
            "passed": False,
            "reason": "pyproject_parse_failed",
            "pyproject_sha256": pyproject_sha256,
            "error_type": type(exc).__name__,
            "error": str(exc)[-1000:],
        }
    project = parsed.get("project")
    if not isinstance(project, Mapping):
        return {
            "root": relative_root,
            "applicable": False,
            "passed": True,
            "reason": "pep621_project_table_not_present",
            "pyproject_sha256": pyproject_sha256,
        }
    dynamic = project.get("dynamic", [])
    if not isinstance(dynamic, list) or not all(isinstance(item, str) for item in dynamic):
        return {
            "root": relative_root,
            "applicable": True,
            "passed": False,
            "reason": "pep621_dynamic_field_invalid",
            "pyproject_sha256": pyproject_sha256,
        }
    dependencies = project.get("dependencies")
    if dependencies is None and "dependencies" in dynamic:
        return {
            "root": relative_root,
            "project_name": str(project.get("name") or ""),
            "applicable": False,
            "passed": True,
            "reason": "project_dependencies_are_dynamic",
            "pyproject_sha256": pyproject_sha256,
        }
    if dependencies is None:
        dependencies = []
    if not isinstance(dependencies, list) or not all(
        isinstance(item, str) for item in dependencies
    ):
        return {
            "root": relative_root,
            "project_name": str(project.get("name") or ""),
            "applicable": True,
            "passed": False,
            "reason": "pep621_dependencies_must_be_static_strings",
            "pyproject_sha256": pyproject_sha256,
        }
    if len(dependencies) > MAX_STATIC_REQUIREMENTS:
        return {
            "root": relative_root,
            "project_name": str(project.get("name") or ""),
            "applicable": True,
            "passed": False,
            "reason": "pep621_dependencies_exceed_preflight_bound",
            "pyproject_sha256": pyproject_sha256,
            "requirement_count": len(dependencies),
            "maximum_requirement_count": MAX_STATIC_REQUIREMENTS,
        }
    oversized = [
        index
        for index, requirement in enumerate(dependencies)
        if len(requirement.encode("utf-8")) > MAX_REQUIREMENT_BYTES
    ]
    if oversized:
        return {
            "root": relative_root,
            "project_name": str(project.get("name") or ""),
            "applicable": True,
            "passed": False,
            "reason": "pep621_requirement_exceeds_preflight_bound",
            "pyproject_sha256": pyproject_sha256,
            "oversized_requirement_indexes": oversized[:20],
            "maximum_requirement_bytes": MAX_REQUIREMENT_BYTES,
        }
    requires_python = project.get("requires-python") or ""
    if not isinstance(requires_python, str):
        return {
            "root": relative_root,
            "project_name": str(project.get("name") or ""),
            "applicable": True,
            "passed": False,
            "reason": "pep621_requires_python_must_be_string",
            "pyproject_sha256": pyproject_sha256,
        }
    return {
        "root": relative_root,
        "project_name": str(project.get("name") or ""),
        "applicable": bool(dependencies or requires_python),
        "passed": True,
        "reason": (
            "static_project_dependencies_collected"
            if dependencies or requires_python
            else "static_project_has_no_runtime_requirements"
        ),
        "pyproject_sha256": pyproject_sha256,
        "requirements": list(dependencies),
        "requires_python": requires_python,
    }


def _evaluate_dependency_payload(
    payload: Mapping[str, Any],
    *,
    version_getter: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, Any]:
    """Evaluate requirements without importing any requested distribution."""

    result: dict[str, Any] = {
        "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
        "passed": False,
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "projects": [],
    }
    try:
        from packaging.markers import default_environment
        from packaging.requirements import InvalidRequirement, Requirement
        from packaging.specifiers import InvalidSpecifier, SpecifierSet
        from packaging.version import InvalidVersion, Version
    except (ImportError, ModuleNotFoundError) as exc:
        result.update(
            {
                "reason": "dependency_probe_packaging_unavailable",
                "error_type": type(exc).__name__,
            }
        )
        return result

    projects = payload.get("projects")
    if not isinstance(projects, list):
        result["reason"] = "dependency_probe_payload_invalid"
        return result
    environment = default_environment()
    environment["extra"] = ""
    all_passed = True
    for source_project in projects:
        if not isinstance(source_project, Mapping):
            all_passed = False
            result["projects"].append(
                {
                    "passed": False,
                    "reason": "dependency_probe_project_invalid",
                }
            )
            continue
        project: dict[str, Any] = {
            "root": str(source_project.get("root") or ""),
            "project_name": str(source_project.get("project_name") or ""),
            "pyproject_sha256": str(source_project.get("pyproject_sha256") or ""),
            "passed": True,
            "reason": "project_dependencies_satisfied",
            "missing": [],
            "incompatible": [],
            "invalid": [],
            "marker_skipped": [],
            "observed": [],
        }
        requires_python = str(source_project.get("requires_python") or "")
        if requires_python:
            try:
                python_specifier = SpecifierSet(requires_python)
                if Version(platform.python_version()) not in python_specifier:
                    project["passed"] = False
                    project["incompatible"].append(
                        {
                            "kind": "python",
                            "requirement": requires_python,
                            "installed_version": platform.python_version(),
                        }
                    )
            except (InvalidSpecifier, InvalidVersion) as exc:
                project["passed"] = False
                project["invalid"].append(
                    {
                        "kind": "requires-python",
                        "requirement": requires_python,
                        "error_type": type(exc).__name__,
                    }
                )
        requirements = source_project.get("requirements")
        if not isinstance(requirements, list):
            project["passed"] = False
            project["invalid"].append(
                {
                    "kind": "dependencies",
                    "error_type": "InvalidProbePayload",
                }
            )
            requirements = []
        for raw_requirement in requirements:
            requirement_text = str(raw_requirement)
            try:
                requirement = Requirement(requirement_text)
            except InvalidRequirement as exc:
                project["passed"] = False
                project["invalid"].append(
                    {
                        "kind": "dependency",
                        "requirement": requirement_text,
                        "error_type": type(exc).__name__,
                    }
                )
                continue
            try:
                applies = requirement.marker is None or requirement.marker.evaluate(
                    environment=environment
                )
            except Exception as exc:
                project["passed"] = False
                project["invalid"].append(
                    {
                        "kind": "marker",
                        "requirement": requirement_text,
                        "error_type": type(exc).__name__,
                    }
                )
                continue
            if not applies:
                project["marker_skipped"].append(requirement_text)
                continue
            try:
                installed_version = version_getter(requirement.name)
            except importlib.metadata.PackageNotFoundError:
                project["passed"] = False
                project["missing"].append(
                    {
                        "name": requirement.name,
                        "requirement": requirement_text,
                    }
                )
                continue
            except Exception as exc:
                project["passed"] = False
                project["invalid"].append(
                    {
                        "kind": "distribution_metadata",
                        "name": requirement.name,
                        "requirement": requirement_text,
                        "error_type": type(exc).__name__,
                    }
                )
                continue
            observed = {
                "name": requirement.name,
                "requirement": requirement_text,
                "installed_version": installed_version,
            }
            project["observed"].append(observed)
            try:
                compatible = (
                    not requirement.specifier or Version(installed_version) in requirement.specifier
                )
            except InvalidVersion as exc:
                compatible = False
                observed["error_type"] = type(exc).__name__
            if not compatible:
                project["passed"] = False
                project["incompatible"].append(observed)
        if project["passed"] is not True:
            project["reason"] = "project_dependency_drift_detected"
            all_passed = False
        result["projects"].append(project)
    result.update(
        {
            "passed": all_passed,
            "reason": (
                "project_dependencies_satisfied"
                if all_passed
                else "project_dependency_drift_detected"
            ),
        }
    )
    return result


def _probe_main() -> int:
    """Private child entry point; emits exactly one bounded JSON object."""

    try:
        payload = json.load(sys.stdin)
        result = _evaluate_dependency_payload(payload)
    except Exception as exc:
        result = {
            "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
            "passed": False,
            "reason": "dependency_probe_unhandled_error",
            "error_type": type(exc).__name__,
            "error": str(exc)[-1000:],
        }
    sys.stdout.write(_canonical_json(result))
    sys.stdout.write("\n")
    return 0


def _run_dependency_probe(
    payload: Mapping[str, Any],
    *,
    environment: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Run the metadata-only probe through the approved sealed interpreter."""

    validation_environment = build_validation_environment(environment)
    validation_environment = validation_environment_for_runner(
        validation_environment,
        _run_dependency_probe,
    )
    module_path = Path(__file__).resolve(strict=True)
    source_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    with validation_python_launcher_environment(validation_environment) as (
        launcher_environment,
        launcher_receipt,
    ):
        try:
            completed = subprocess.run(
                [
                    launcher_environment["PYTHON"],
                    str(module_path),
                    "--probe",
                ],
                input=_canonical_json(payload),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=DEPENDENCY_PROBE_TIMEOUT_SECONDS,
                check=False,
                env=launcher_environment,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
                "passed": False,
                "reason": "dependency_probe_process_unavailable",
                "error_type": type(exc).__name__,
                "preflight_source_sha256": source_sha256,
            }
        output = str(completed.stdout or "")
        launcher = {
            "content_sha256": launcher_receipt.content_sha256,
            "interpreter_sha256": launcher_receipt.interpreter_sha256,
            "interpreter_stat": launcher_receipt.interpreter_stat,
            "mode": launcher_receipt.mode,
            "policy_sha256": launcher_receipt.policy_sha256,
            "sealed": launcher_receipt.sealed,
        }
        if completed.returncode != 0:
            return {
                "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
                "passed": False,
                "reason": "dependency_probe_process_failed",
                "returncode": int(completed.returncode),
                "output_tail": output[-4000:],
                "preflight_source_sha256": source_sha256,
                "validation_python_launcher": launcher,
            }
        if len(output.encode("utf-8")) > MAX_PROBE_OUTPUT_BYTES:
            return {
                "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
                "passed": False,
                "reason": "dependency_probe_output_exceeded_bound",
                "output_bytes": len(output.encode("utf-8")),
                "maximum_output_bytes": MAX_PROBE_OUTPUT_BYTES,
                "preflight_source_sha256": source_sha256,
                "validation_python_launcher": launcher,
            }
        try:
            result = json.loads(output)
        except (TypeError, ValueError) as exc:
            return {
                "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
                "passed": False,
                "reason": "dependency_probe_output_invalid",
                "error_type": type(exc).__name__,
                "output_tail": output[-4000:],
                "preflight_source_sha256": source_sha256,
                "validation_python_launcher": launcher,
            }
        if (
            not isinstance(result, dict)
            or result.get("schema") != PROJECT_DEPENDENCY_PROBE_SCHEMA
            or not isinstance(result.get("passed"), bool)
        ):
            return {
                "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
                "passed": False,
                "reason": "dependency_probe_receipt_invalid",
                "preflight_source_sha256": source_sha256,
                "validation_python_launcher": launcher,
            }
        result["preflight_source_sha256"] = source_sha256
        result["validation_python_launcher"] = launcher
        return result


if not _CHILD_PROBE_MODE:
    _run_dependency_probe = sealed_validation_python_runner(_run_dependency_probe)


def preflight_validation_project_dependencies(
    workspace_path: Path | str,
    validation_commands: Sequence[str],
    *,
    environment: Mapping[str, object] | None = None,
    probe_runner: Callable[..., dict[str, Any]] = _run_dependency_probe,
) -> dict[str, Any]:
    """Compare static PEP-621 requirements with the approved interpreter."""

    workspace = Path(workspace_path)
    roots: list[str] = []
    invalid_commands: list[dict[str, Any]] = []
    for index, command in enumerate(validation_commands):
        root = validation_command_repository_root(str(command))
        if root is None:
            invalid_commands.append(
                {
                    "command_index": index,
                    "command_sha256": hashlib.sha256(str(command).encode("utf-8")).hexdigest(),
                    "reason": "validation_repository_root_is_unsafe",
                }
            )
            continue
        if root not in roots:
            roots.append(root)

    projects = [_bounded_static_project(workspace, relative_root) for relative_root in roots]
    static_projects = [
        project
        for project in projects
        if project.get("applicable") is True
        and project.get("passed") is True
        and (project.get("requirements") or project.get("requires_python"))
    ]
    collection_failures = [project for project in projects if project.get("passed") is not True]
    base_receipt: dict[str, Any] = {
        "schema": PROJECT_DEPENDENCY_PREFLIGHT_SCHEMA,
        "workspace": str(workspace.resolve()),
        "applicable": bool(static_projects),
        "install_attempted": False,
        "network_accessed": False,
        "validation_command_count": len(validation_commands),
        "project_roots": roots,
        "projects": projects,
        "invalid_commands": invalid_commands,
    }
    if invalid_commands or collection_failures:
        receipt = {
            **base_receipt,
            "passed": False,
            "reason": "project_dependency_contract_collection_failed",
            "remediation": {
                "kind": "repair_static_project_dependency_contract",
                "automatic_provisioning": False,
            },
        }
    elif not static_projects:
        receipt = {
            **base_receipt,
            "passed": True,
            "reason": (
                "no_declared_validation_commands"
                if not validation_commands
                else "no_static_pep621_dependencies"
            ),
        }
    else:
        probe_payload = {
            "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
            "projects": [
                {
                    key: project.get(key)
                    for key in (
                        "root",
                        "project_name",
                        "pyproject_sha256",
                        "requirements",
                        "requires_python",
                    )
                }
                for project in static_projects
            ],
        }
        try:
            raw_probe = probe_runner(
                probe_payload,
                environment=environment,
            )
        except Exception as exc:
            raw_probe = {
                "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
                "passed": False,
                "reason": "dependency_probe_infrastructure_error",
                "error_type": type(exc).__name__,
                "error": str(exc)[-1000:],
            }
        probe = (
            dict(raw_probe)
            if isinstance(raw_probe, Mapping)
            else {
                "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
                "passed": False,
                "reason": "dependency_probe_receipt_invalid",
                "error_type": type(raw_probe).__name__,
            }
        )
        missing = [
            item
            for project in probe.get("projects", [])
            if isinstance(project, Mapping)
            for item in project.get("missing", [])
            if isinstance(item, Mapping)
        ]
        incompatible = [
            item
            for project in probe.get("projects", [])
            if isinstance(project, Mapping)
            for item in project.get("incompatible", [])
            if isinstance(item, Mapping)
        ]
        invalid = [
            item
            for project in probe.get("projects", [])
            if isinstance(project, Mapping)
            for item in project.get("invalid", [])
            if isinstance(item, Mapping)
        ]
        passed = probe.get("passed") is True
        probe_reason = str(probe.get("reason") or "")
        drift_detected = probe_reason == "project_dependency_drift_detected"
        receipt = {
            **base_receipt,
            "passed": passed,
            "reason": (
                "approved_validation_environment_satisfies_project_dependencies"
                if passed
                else (
                    "approved_validation_environment_dependency_drift"
                    if drift_detected
                    else ("approved_validation_environment_dependency_probe_failed")
                )
            ),
            "probe": probe,
            "missing_requirements": missing,
            "incompatible_requirements": incompatible,
            "invalid_requirements": invalid,
        }
        if not passed:
            receipt["remediation"] = {
                "kind": (
                    "provision_approved_validation_environment"
                    if drift_detected
                    else "repair_approved_validation_dependency_probe"
                ),
                "python_executable": str(probe.get("python_executable") or ""),
                "requirements": [
                    str(item.get("requirement") or "")
                    for item in (*missing, *incompatible)
                    if str(item.get("requirement") or "")
                ][:MAX_STATIC_REQUIREMENTS],
                "automatic_provisioning": False,
                "rerun_required": True,
            }
    receipt["receipt_id"] = _content_sha256(receipt)
    return receipt


if __name__ == "__main__":
    raise SystemExit(_probe_main() if "--probe" in sys.argv[1:] else 64)
