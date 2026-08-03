"""Detect static project dependency drift before implementation dispatch.

The preflight is deliberately detection-only.  Importing this module performs
no subprocess, package-manager, network, or filesystem mutation.  At explicit
call time it reads only static PEP-621 dependency declarations and evaluates
them in the same approved, sealed Python environment used by authoritative
validation.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import stat as stat_module
import subprocess
import sys
import tempfile
import threading
import zlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

_CHILD_PROBE_MODE = __name__ == "__main__" and "--probe" in sys.argv[1:]
if not _CHILD_PROBE_MODE:
    from .validation_commands import (
        ValidationDependencyScope,
        validation_command_dependency_scope,
        validation_command_repository_root,
    )
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
PROJECT_DEPENDENCY_PREFLIGHT_MAX_BACKOFF_SECONDS = 1800
MAX_PYPROJECT_BYTES = 2 * 1024 * 1024
MAX_DEPENDENCY_MANIFEST_FILES = 16
MAX_DEPENDENCY_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_STATIC_REQUIREMENTS = 512
MAX_REQUIREMENT_BYTES = 2048
MAX_INSTALLED_VERSION_BYTES = 512
MAX_PROBE_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_PROBE_SOURCE_BYTES = 512 * 1024
BOUNDED_FILE_READ_CHUNK_BYTES = 64 * 1024
MAX_DEPENDENCY_CLOSURE_NODES = 256
MAX_DEPENDENCY_CLOSURE_EDGES = 2048
MAX_DEPENDENCY_CLOSURE_DEPTH = 32
MAX_DEPENDENCY_CLOSURE_REQUIREMENTS = 4096
MAX_DEPENDENCY_CLOSURE_CONTEXTS = 4096
MAX_DEPENDENCY_CLOSURE_REQUIREMENT_BYTES = MAX_REQUIREMENT_BYTES
MAX_DEPENDENCY_CLOSURE_METADATA_TEXT_BYTES = 2 * 1024 * 1024
MAX_DEPENDENCY_CLOSURE_INSTALLED_VERSION_BYTES = MAX_INSTALLED_VERSION_BYTES
DEPENDENCY_PROBE_TIMEOUT_SECONDS = 30.0
PYTEST_OPTIONAL_DEPENDENCY_EXTRA_PRIORITY = (
    "test",
    "testing",
    "dev",
)
PYTEST_COMMAND_PATTERN = re.compile(r"(?<![A-Za-z0-9_.-])pytest(?=$|[\s;&|])")
_DEFAULT_VERSION_GETTER = importlib.metadata.version
_DEFAULT_REQUIRES_GETTER = object()
_PROBE_ARGV_BOOTSTRAP = (
    "import base64,hashlib,sys,zlib;"
    "payload=sys.argv.pop(1);"
    "source=zlib.decompress(base64.b85decode(payload));"
    "scope={'__name__':'__main__','__file__':'<dependency-probe>',"
    "'__package__':None,'__probe_source_sha256__':"
    "hashlib.sha256(source).hexdigest()};"
    "exec(compile(source,'<dependency-probe>','exec'),scope)"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _content_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class _BoundedFileTooLarge(ValueError):
    """A regular metadata file exceeded its declared snapshot bound."""

    def __init__(self, observed_bytes: int, maximum_bytes: int) -> None:
        super().__init__("metadata file exceeds bounded snapshot size")
        self.observed_bytes = int(observed_bytes)
        self.maximum_bytes = int(maximum_bytes)


class _BoundedFileSnapshotRace(ValueError):
    """A metadata path or opened file changed during snapshot collection."""


def _stat_snapshot(stat_result: os.stat_result) -> tuple[int, ...]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_module.S_IFMT(stat_result.st_mode)),
        int(stat_result.st_nlink),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
    )


def _require_directory_snapshot(
    path: Path,
    expected_snapshot: tuple[int, ...],
) -> os.stat_result:
    """Require one already-resolved directory path to retain its identity."""

    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise _BoundedFileSnapshotRace(
            "metadata containment root disappeared"
        ) from exc
    if (
        not stat_module.S_ISDIR(current.st_mode)
        or _stat_snapshot(current) != expected_snapshot
    ):
        raise _BoundedFileSnapshotRace(
            "metadata containment root identity changed"
        )
    return current


def _read_bounded_contained_regular_file(
    containment_root: Path,
    candidate: Path,
    *,
    maximum_bytes: int,
    expected_containment_root_snapshot: tuple[int, ...],
) -> tuple[Path, bytes]:
    """Read one stable regular-file snapshot without following an escape.

    The path is resolved and checked before opening, the final component is
    opened no-follow where the platform supports it, and both the descriptor
    and path identities are revalidated after a chunk-bounded read.
    """

    if maximum_bytes < 0:
        raise ValueError("metadata file byte bound is invalid")
    root = containment_root
    if not root.is_absolute() or not candidate.is_absolute():
        raise ValueError("metadata containment paths must be absolute")
    _require_directory_snapshot(root, expected_containment_root_snapshot)
    initial_path_stat = candidate.lstat()
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise _BoundedFileSnapshotRace("metadata path disappeared during snapshot") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("metadata file escapes project root") from exc

    target_stat = os.stat(resolved, follow_symlinks=False)
    if not stat_module.S_ISREG(target_stat.st_mode):
        raise ValueError("metadata source is not a regular file")
    if target_stat.st_size > maximum_bytes:
        raise _BoundedFileTooLarge(target_stat.st_size, maximum_bytes)

    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(resolved, flags)
    try:
        opened_stat = os.fstat(descriptor)
        if not stat_module.S_ISREG(opened_stat.st_mode) or _stat_snapshot(
            opened_stat
        ) != _stat_snapshot(target_stat):
            raise _BoundedFileSnapshotRace("metadata file changed before bounded read")
        if opened_stat.st_size > maximum_bytes:
            raise _BoundedFileTooLarge(opened_stat.st_size, maximum_bytes)

        chunks: list[bytes] = []
        observed_bytes = 0
        while True:
            remaining = maximum_bytes + 1 - observed_bytes
            if remaining <= 0:
                raise _BoundedFileTooLarge(observed_bytes, maximum_bytes)
            chunk = os.read(
                descriptor,
                min(BOUNDED_FILE_READ_CHUNK_BYTES, remaining),
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed_bytes += len(chunk)
            if observed_bytes > maximum_bytes:
                raise _BoundedFileTooLarge(observed_bytes, maximum_bytes)
        final_descriptor_stat = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    payload = b"".join(chunks)
    if len(payload) != opened_stat.st_size or _stat_snapshot(
        final_descriptor_stat
    ) != _stat_snapshot(opened_stat):
        raise _BoundedFileSnapshotRace("metadata file changed during bounded read")

    try:
        _require_directory_snapshot(root, expected_containment_root_snapshot)
        final_path_stat = candidate.lstat()
        final_resolved = candidate.resolve(strict=True)
        final_resolved.relative_to(root)
        final_target_stat = os.stat(final_resolved, follow_symlinks=False)
    except (FileNotFoundError, ValueError) as exc:
        raise _BoundedFileSnapshotRace("metadata path changed after bounded read") from exc
    if (
        final_resolved != resolved
        or _stat_snapshot(final_path_stat) != _stat_snapshot(initial_path_stat)
        or _stat_snapshot(final_target_stat) != _stat_snapshot(opened_stat)
    ):
        raise _BoundedFileSnapshotRace("metadata path identity changed during bounded read")
    return resolved, payload


def _retry_fingerprint(receipt: Mapping[str, Any]) -> str:
    """Bind retry cadence to the failure contract, not an ephemeral worktree."""

    def stable(value: object) -> object:
        if isinstance(value, Mapping):
            return {
                str(key): stable(item)
                for key, item in value.items()
                if str(key)
                not in {
                    "error_sha256",
                    "interpreter_stat",
                    "output_sha256",
                    "receipt_id",
                    "retry_fingerprint",
                    "workspace",
                }
            }
        if isinstance(value, list):
            return [stable(item) for item in value]
        return value

    return _content_sha256(stable(receipt))


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


def _validation_command_invokes_pytest(command: str) -> bool:
    try:
        tokens = shlex.split(str(command), posix=True)
    except ValueError:
        tokens = []
    for index, token in enumerate(tokens):
        executable = PurePosixPath(token.replace("\\", "/")).name
        if executable in {"py.test", "pytest"}:
            return True
        if token == "-m" and index + 1 < len(tokens) and tokens[index + 1] == "pytest":
            return True
    return bool(PYTEST_COMMAND_PATTERN.search(str(command)))


def _safe_project_dependency_file(
    project_root: Path,
    raw_path: str,
) -> Path:
    normalized = str(raw_path or "").strip()
    candidate = PurePosixPath(normalized)
    if (
        not normalized
        or "\\" in normalized
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.as_posix() == "."
    ):
        raise ValueError("dynamic dependency file path is unsafe")
    return project_root / candidate.as_posix()


def _setuptools_file_backed_requirement_source(
    dependency_source: object,
    project_root: Path,
    *,
    maximum_total_bytes: int,
    expected_project_root_snapshot: tuple[int, ...],
    maximum_files: int = MAX_DEPENDENCY_MANIFEST_FILES,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Resolve one reviewed setuptools file source without guessing."""

    raw_files = dependency_source.get("file") if isinstance(dependency_source, Mapping) else None
    if isinstance(raw_files, str):
        files = [raw_files]
    elif isinstance(raw_files, list) and all(isinstance(item, str) for item in raw_files):
        files = list(raw_files)
    else:
        raise ValueError("dynamic dependencies are not setuptools file-backed")
    if maximum_files < 1 or not files or len(files) > maximum_files:
        raise ValueError("dynamic dependency file count is invalid")

    requirements: list[str] = []
    manifests: list[dict[str, Any]] = []
    total_bytes = 0
    for raw_file in files:
        candidate = _safe_project_dependency_file(project_root, raw_file)
        path, payload = _read_bounded_contained_regular_file(
            project_root,
            candidate,
            maximum_bytes=maximum_total_bytes - total_bytes,
            expected_containment_root_snapshot=(
                expected_project_root_snapshot
            ),
        )
        total_bytes += len(payload)
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("dynamic dependency file is not UTF-8") from exc
        logical_line = ""
        for source_line in text.splitlines():
            stripped = source_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            continued = stripped.endswith("\\")
            fragment = stripped[:-1].rstrip() if continued else stripped
            logical_line = f"{logical_line}{fragment}".strip()
            if continued:
                continue
            if logical_line.startswith("-"):
                raise ValueError("dynamic dependency file contains package-manager options")
            requirements.append(logical_line)
            logical_line = ""
        if logical_line:
            raise ValueError("dynamic dependency file has unterminated continuation")
        manifests.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    return requirements, manifests


def _setuptools_dynamic_configuration(
    parsed: Mapping[str, Any],
) -> Mapping[str, Any]:
    tool = parsed.get("tool")
    setuptools = tool.get("setuptools") if isinstance(tool, Mapping) else None
    dynamic = setuptools.get("dynamic") if isinstance(setuptools, Mapping) else None
    if not isinstance(dynamic, Mapping):
        raise ValueError("setuptools dynamic configuration is unavailable")
    return dynamic


def _setuptools_file_backed_dependencies(
    parsed: Mapping[str, Any],
    project_root: Path,
    expected_project_root_snapshot: tuple[int, ...],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Resolve the reviewed dynamic project dependency source."""

    dynamic = _setuptools_dynamic_configuration(parsed)
    return _setuptools_file_backed_requirement_source(
        dynamic.get("dependencies"),
        project_root,
        maximum_total_bytes=MAX_DEPENDENCY_MANIFEST_BYTES,
        expected_project_root_snapshot=expected_project_root_snapshot,
    )


def _pytest_validation_dependencies(
    parsed: Mapping[str, Any],
    project: Mapping[str, Any],
    project_root: Path,
    expected_project_root_snapshot: tuple[int, ...],
    dynamic_fields: Sequence[str],
    *,
    pytest_invoked: bool,
    maximum_manifest_bytes: int,
    maximum_manifest_files: int,
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[dict[str, Any]],
    str,
]:
    """Select ``test``, then ``testing``, then ``dev`` for pytest commands.

    The runner distribution itself is always required.  At most one declared
    extra is selected so similarly named extras cannot silently combine into a
    larger, environment-dependent contract.
    """

    requirements = ["pytest"]
    marker_extras = [""]
    if not pytest_invoked:
        return [], [], [], [], "not_applicable"

    optional_is_dynamic = "optional-dependencies" in dynamic_fields
    optional = project.get("optional-dependencies")
    if optional_is_dynamic:
        if optional is not None:
            raise ValueError("PEP-621 optional-dependencies cannot be static and dynamic")
        dynamic = _setuptools_dynamic_configuration(parsed)
        optional = dynamic.get("optional-dependencies")
        if not isinstance(optional, Mapping):
            raise ValueError("dynamic optional-dependencies are not setuptools file-backed")
        selected = next(
            (name for name in PYTEST_OPTIONAL_DEPENDENCY_EXTRA_PRIORITY if name in optional),
            "",
        )
        if not selected:
            return (
                requirements,
                marker_extras,
                [],
                [],
                "setuptools_dynamic_file",
            )
        declared, manifests = _setuptools_file_backed_requirement_source(
            optional.get(selected),
            project_root,
            maximum_total_bytes=maximum_manifest_bytes,
            expected_project_root_snapshot=expected_project_root_snapshot,
            maximum_files=maximum_manifest_files,
        )
        requirements.extend(declared)
        marker_extras.extend([selected] * len(declared))
        return (
            requirements,
            marker_extras,
            [selected],
            manifests,
            "setuptools_dynamic_file",
        )

    if optional is None:
        optional = {}
    if not isinstance(optional, Mapping):
        raise ValueError("PEP-621 optional-dependencies must be a table")
    selected = next(
        (name for name in PYTEST_OPTIONAL_DEPENDENCY_EXTRA_PRIORITY if name in optional),
        "",
    )
    if not selected:
        return requirements, marker_extras, [], [], "pep621_static"
    declared = optional.get(selected)
    if not isinstance(declared, list) or not all(isinstance(item, str) for item in declared):
        raise ValueError(f"validation dependency extra {selected!r} is invalid")
    requirements.extend(declared)
    marker_extras.extend([selected] * len(declared))
    return (
        requirements,
        marker_extras,
        [selected],
        [],
        "pep621_static",
    )


def _public_project_contract(
    project: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove raw PEP-508 text before durable receipts or events."""

    result = {
        str(key): value
        for key, value in project.items()
        if key
        not in {
            "dependency_manifests",
            "requirement_marker_extras",
            "requirements",
            "requires_python",
        }
    }
    requirements = project.get("requirements")
    if isinstance(requirements, list):
        result["requirement_count"] = len(requirements)
        result["requirement_sha256"] = [
            hashlib.sha256(str(item).encode("utf-8")).hexdigest() for item in requirements
        ]
    requires_python = project.get("requires_python")
    if isinstance(requires_python, str) and requires_python:
        result["requires_python_declared"] = True
        result["requires_python_sha256"] = hashlib.sha256(
            requires_python.encode("utf-8")
        ).hexdigest()
    else:
        result["requires_python_declared"] = False
    manifests = project.get("dependency_manifests")
    if isinstance(manifests, list):
        result["dependency_manifests"] = [
            {
                "path_sha256": hashlib.sha256(
                    str(manifest.get("path") or "").encode("utf-8")
                ).hexdigest(),
                "content_sha256": str(manifest.get("sha256") or ""),
                "bytes": int(manifest.get("bytes") or 0),
            }
            for manifest in manifests
            if isinstance(manifest, Mapping)
        ]
    return result


def _project_name_sha256(project: Mapping[str, Any]) -> str:
    value = project.get("name")
    if value is None:
        return ""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _bounded_static_project(
    workspace_path: Path,
    relative_root: str,
    *,
    pytest_invoked: bool = False,
) -> dict[str, Any]:
    """Read one safe project root and return its static PEP-621 contract."""

    workspace = workspace_path.resolve(strict=True)
    candidate = workspace / relative_root
    try:
        project_root = candidate.resolve(strict=True)
    except FileNotFoundError:
        if pytest_invoked:
            return {
                "root": relative_root,
                "project_name_sha256": "",
                "applicable": True,
                "passed": True,
                "reason": "validation_runner_requirement_collected",
                "requirements": ["pytest"],
                "requirement_marker_extras": [""],
                "requires_python": "",
                "dependency_source": "validation_command_runner",
                "validation_dependency_source": "validation_command_runner",
                "dependency_manifests": [],
                "pytest_invoked": True,
                "selected_validation_extras": [],
            }
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
    try:
        project_root_stat = os.stat(project_root, follow_symlinks=False)
    except OSError as exc:
        return {
            "root": relative_root,
            "applicable": True,
            "passed": False,
            "reason": "project_root_is_not_directory",
            "error_type": type(exc).__name__,
        }
    if not stat_module.S_ISDIR(project_root_stat.st_mode):
        return {
            "root": relative_root,
            "applicable": True,
            "passed": False,
            "reason": "project_root_is_not_directory",
        }
    project_root_snapshot = _stat_snapshot(project_root_stat)

    pyproject_path = project_root / "pyproject.toml"
    try:
        _require_directory_snapshot(project_root, project_root_snapshot)
        pyproject_path.lstat()
    except FileNotFoundError:
        try:
            _require_directory_snapshot(project_root, project_root_snapshot)
        except ValueError as exc:
            return {
                "root": relative_root,
                "applicable": True,
                "passed": False,
                "reason": "pyproject_path_or_snapshot_invalid",
                "error_type": type(exc).__name__,
            }
        if pytest_invoked:
            return {
                "root": relative_root,
                "project_name_sha256": "",
                "applicable": True,
                "passed": True,
                "reason": "validation_runner_requirement_collected",
                "requirements": ["pytest"],
                "requirement_marker_extras": [""],
                "requires_python": "",
                "dependency_source": "validation_command_runner",
                "validation_dependency_source": "validation_command_runner",
                "dependency_manifests": [],
                "pytest_invoked": True,
                "selected_validation_extras": [],
            }
        return {
            "root": relative_root,
            "applicable": False,
            "passed": True,
            "reason": "pep621_metadata_not_present",
        }
    except ValueError as exc:
        return {
            "root": relative_root,
            "applicable": True,
            "passed": False,
            "reason": "pyproject_path_or_snapshot_invalid",
            "error_type": type(exc).__name__,
        }
    except OSError as exc:
        return {
            "root": relative_root,
            "applicable": True,
            "passed": False,
            "reason": "pyproject_read_failed",
            "error_type": type(exc).__name__,
        }
    try:
        _, payload = _read_bounded_contained_regular_file(
            project_root,
            pyproject_path,
            maximum_bytes=MAX_PYPROJECT_BYTES,
            expected_containment_root_snapshot=project_root_snapshot,
        )
    except _BoundedFileTooLarge as exc:
        return {
            "root": relative_root,
            "applicable": True,
            "passed": False,
            "reason": "pyproject_exceeds_preflight_bound",
            "pyproject_bytes": exc.observed_bytes,
            "maximum_pyproject_bytes": exc.maximum_bytes,
        }
    except OSError as exc:
        return {
            "root": relative_root,
            "applicable": True,
            "passed": False,
            "reason": "pyproject_read_failed",
            "error_type": type(exc).__name__,
        }
    except ValueError as exc:
        return {
            "root": relative_root,
            "applicable": True,
            "passed": False,
            "reason": "pyproject_path_or_snapshot_invalid",
            "error_type": type(exc).__name__,
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
        }
    project = parsed.get("project")
    if not isinstance(project, Mapping):
        if pytest_invoked:
            return {
                "root": relative_root,
                "project_name_sha256": "",
                "applicable": True,
                "passed": True,
                "reason": "validation_runner_requirement_collected",
                "pyproject_sha256": pyproject_sha256,
                "requirements": ["pytest"],
                "requirement_marker_extras": [""],
                "requires_python": "",
                "dependency_source": "validation_command_runner",
                "validation_dependency_source": "validation_command_runner",
                "dependency_manifests": [],
                "pytest_invoked": True,
                "selected_validation_extras": [],
            }
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
    if "requires-python" in dynamic:
        return {
            "root": relative_root,
            "project_name_sha256": _project_name_sha256(project),
            "applicable": True,
            "passed": False,
            "reason": "pep621_requires_python_dynamic_unresolved",
            "pyproject_sha256": pyproject_sha256,
        }
    dependencies = project.get("dependencies")
    dependency_source = "pep621_static"
    dependency_manifests: list[dict[str, Any]] = []
    if "dependencies" in dynamic:
        if dependencies is not None:
            return {
                "root": relative_root,
                "project_name_sha256": _project_name_sha256(project),
                "applicable": True,
                "passed": False,
                "reason": "pep621_dependencies_static_and_dynamic",
                "pyproject_sha256": pyproject_sha256,
            }
        try:
            dependencies, dependency_manifests = _setuptools_file_backed_dependencies(
                parsed,
                project_root,
                project_root_snapshot,
            )
        except (OSError, UnicodeError, ValueError) as exc:
            return {
                "root": relative_root,
                "project_name_sha256": _project_name_sha256(project),
                "applicable": True,
                "passed": False,
                "reason": "dynamic_dependencies_unresolved",
                "pyproject_sha256": pyproject_sha256,
                "error_type": type(exc).__name__,
            }
        dependency_source = "setuptools_dynamic_file"
    elif dependencies is None:
        dependencies = []
    if not isinstance(dependencies, list) or not all(
        isinstance(item, str) for item in dependencies
    ):
        return {
            "root": relative_root,
            "project_name_sha256": _project_name_sha256(project),
            "applicable": True,
            "passed": False,
            "reason": "pep621_dependencies_must_be_static_strings",
            "pyproject_sha256": pyproject_sha256,
        }
    requirement_marker_extras = [""] * len(dependencies)
    try:
        (
            validation_dependencies,
            validation_marker_extras,
            selected_extras,
            validation_manifests,
            validation_dependency_source,
        ) = _pytest_validation_dependencies(
            parsed,
            project,
            project_root,
            project_root_snapshot,
            dynamic,
            pytest_invoked=pytest_invoked,
            maximum_manifest_bytes=(
                MAX_DEPENDENCY_MANIFEST_BYTES
                - sum(int(manifest.get("bytes") or 0) for manifest in dependency_manifests)
            ),
            maximum_manifest_files=(
                MAX_DEPENDENCY_MANIFEST_FILES
                - len(dependency_manifests)
            ),
        )
    except (OSError, ValueError) as exc:
        return {
            "root": relative_root,
            "project_name_sha256": _project_name_sha256(project),
            "applicable": True,
            "passed": False,
            "reason": "validation_dependencies_unresolved",
            "pyproject_sha256": pyproject_sha256,
            "error_type": type(exc).__name__,
        }
    dependencies = [*dependencies, *validation_dependencies]
    requirement_marker_extras.extend(validation_marker_extras)
    dependency_manifests.extend(validation_manifests)
    if len(dependencies) > MAX_STATIC_REQUIREMENTS:
        return {
            "root": relative_root,
            "project_name_sha256": _project_name_sha256(project),
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
            "project_name_sha256": _project_name_sha256(project),
            "applicable": True,
            "passed": False,
            "reason": "pep621_requirement_exceeds_preflight_bound",
            "pyproject_sha256": pyproject_sha256,
            "oversized_requirement_indexes": oversized[:20],
            "maximum_requirement_bytes": MAX_REQUIREMENT_BYTES,
        }
    requires_python = project.get("requires-python", "")
    if not isinstance(requires_python, str):
        return {
            "root": relative_root,
            "project_name_sha256": _project_name_sha256(project),
            "applicable": True,
            "passed": False,
            "reason": "pep621_requires_python_must_be_string",
            "pyproject_sha256": pyproject_sha256,
        }
    return {
        "root": relative_root,
        "project_name_sha256": _project_name_sha256(project),
        "applicable": bool(dependencies or requires_python),
        "passed": True,
        "reason": (
            "static_project_dependencies_collected"
            if dependencies or requires_python
            else "static_project_has_no_runtime_requirements"
        ),
        "pyproject_sha256": pyproject_sha256,
        "requirements": list(dependencies),
        "requirement_marker_extras": requirement_marker_extras,
        "requires_python": requires_python,
        "dependency_source": dependency_source,
        "validation_dependency_source": validation_dependency_source,
        "dependency_manifests": dependency_manifests,
        "pytest_invoked": pytest_invoked,
        "selected_validation_extras": selected_extras,
    }


class _DependencyClosureEvaluator:
    """Verify a bounded installed-distribution closure using metadata only."""

    def __init__(
        self,
        project: dict[str, Any],
        *,
        marker_environment: Mapping[str, str],
        requirement_factory: Callable[[str], Any],
        invalid_requirement_type: type[Exception],
        version_factory: Callable[[str], Any],
        invalid_version_type: type[Exception],
        canonicalize_name: Callable[[str], str],
        version_getter: Callable[[str], str],
        requires_getter: Callable[[str], Sequence[str] | None] | None,
    ) -> None:
        self.project = project
        self.marker_environment = dict(marker_environment)
        self.requirement_factory = requirement_factory
        self.invalid_requirement_type = invalid_requirement_type
        self.version_factory = version_factory
        self.invalid_version_type = invalid_version_type
        self.canonicalize_name = canonicalize_name
        self.version_getter = version_getter
        self.requires_getter = requires_getter

        self.node_states: dict[str, dict[str, Any]] = {}
        self.metadata_cache: dict[str, tuple[str, ...] | None] = {}
        self.expanded_contexts: set[tuple[str, str]] = set()
        self.processed_metadata_edges: set[tuple[str, str]] = set()
        self.cycles: dict[tuple[str, ...], dict[str, Any]] = {}
        self.edge_count = 0
        self.requirement_evaluation_count = 0
        self.metadata_requirement_count = 0
        self.metadata_text_bytes = 0
        self.stopped_on_bound = False

    def _append_invalid(self, record: Mapping[str, Any]) -> None:
        self.project["passed"] = False
        self.project["invalid"].append(dict(record))

    def _bound_failure(
        self,
        bound: str,
        *,
        maximum: int,
        observed: int,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if self.stopped_on_bound:
            return
        record: dict[str, Any] = {
            "kind": "dependency_closure_bound",
            "bound": str(bound),
            "maximum": int(maximum),
            "observed": int(observed),
        }
        if context:
            record.update(
                {
                    str(key): value
                    for key, value in context.items()
                    if value not in (None, "")
                }
            )
        self._append_invalid(record)
        self.stopped_on_bound = True

    @staticmethod
    def _deduplicate(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            encoded = _canonical_json(record)
            if encoded in seen:
                continue
            seen.add(encoded)
            result.append(dict(record))
        return result

    def _safe_requirement_record(
        self,
        requirement: Any,
        *,
        requirement_sha256: str,
        source: str,
        parent_name: str,
        depth: int,
    ) -> dict[str, Any]:
        name = str(self.canonicalize_name(str(requirement.name)))
        extras = sorted(
            {
                str(self.canonicalize_name(str(extra)))
                for extra in requirement.extras
            }
        )
        specifier = str(requirement.specifier)
        record: dict[str, Any] = {
            "name": name,
            "requirement": (
                name
                + (f"[{','.join(extras)}]" if extras else "")
                + specifier
            ),
            "requirement_sha256": requirement_sha256,
            "specifier": specifier,
            "extras": extras,
            "source": source,
            "depth": int(depth),
        }
        if parent_name:
            record["parent_name"] = parent_name
        if requirement.marker is not None:
            record["marker_sha256"] = hashlib.sha256(
                str(requirement.marker).encode("utf-8", errors="surrogatepass")
            ).hexdigest()
        return record

    def _node_state(self, name: str) -> dict[str, Any] | None:
        cached = self.node_states.get(name)
        if cached is not None:
            return cached
        observed_nodes = len(self.node_states) + 1
        if observed_nodes > MAX_DEPENDENCY_CLOSURE_NODES:
            self._bound_failure(
                "nodes",
                maximum=MAX_DEPENDENCY_CLOSURE_NODES,
                observed=observed_nodes,
                context={"name": name},
            )
            return None

        try:
            raw_version = self.version_getter(name)
        except importlib.metadata.PackageNotFoundError:
            state = {"status": "missing"}
            self.node_states[name] = state
            return state
        except Exception as exc:
            state = {
                "status": "invalid",
                "kind": "distribution_version_metadata",
                "error_type": type(exc).__name__,
            }
            self.node_states[name] = state
            return state
        if not isinstance(raw_version, str):
            state = {
                "status": "invalid",
                "kind": "installed_version",
                "error_type": "InstalledVersionIsNotString",
                "value_type": type(raw_version).__name__,
            }
            self.node_states[name] = state
            return state

        encoded_version = raw_version.encode("utf-8", errors="surrogatepass")
        version_sha256 = hashlib.sha256(encoded_version).hexdigest()
        if len(encoded_version) > MAX_INSTALLED_VERSION_BYTES:
            state = {
                "status": "invalid",
                "kind": "installed_version",
                "error_type": "InstalledVersionExceedsBound",
                "installed_version_sha256": version_sha256,
                "installed_version_bytes": len(encoded_version),
            }
            self.node_states[name] = state
            return state
        if len(encoded_version) > MAX_DEPENDENCY_CLOSURE_INSTALLED_VERSION_BYTES:
            self._bound_failure(
                "installed_version_bytes",
                maximum=MAX_DEPENDENCY_CLOSURE_INSTALLED_VERSION_BYTES,
                observed=len(encoded_version),
                context={
                    "name": name,
                    "installed_version_sha256": version_sha256,
                },
            )
            state = {"status": "invalid"}
            self.node_states[name] = state
            return None
        try:
            parsed_version = self.version_factory(raw_version)
        except self.invalid_version_type as exc:
            state = {
                "status": "invalid",
                "kind": "installed_version",
                "error_type": type(exc).__name__,
                "installed_version_sha256": version_sha256,
                "installed_version_bytes": len(encoded_version),
            }
            self.node_states[name] = state
            return state
        state = {
            "status": "present",
            "version": str(parsed_version),
            "parsed_version": parsed_version,
            "version_sha256": version_sha256,
        }
        self.node_states[name] = state
        return state

    def _metadata_requirements(self, name: str) -> tuple[str, ...] | None:
        if name in self.metadata_cache:
            return self.metadata_cache[name]
        if self.requires_getter is None:
            self.metadata_cache[name] = ()
            return ()
        try:
            raw_requirements = self.requires_getter(name)
        except Exception as exc:
            self._append_invalid(
                {
                    "kind": "distribution_requirements_metadata",
                    "name": name,
                    "error_type": type(exc).__name__,
                }
            )
            self.metadata_cache[name] = None
            return None
        if raw_requirements is None:
            self.metadata_cache[name] = ()
            return ()
        if not isinstance(raw_requirements, (list, tuple)):
            self._append_invalid(
                {
                    "kind": "distribution_requirements_metadata",
                    "name": name,
                    "error_type": "InvalidMetadataRequirementsType",
                    "value_type": type(raw_requirements).__name__,
                }
            )
            self.metadata_cache[name] = None
            return None
        projected_count = (
            self.requirement_evaluation_count + len(raw_requirements)
        )
        if projected_count > MAX_DEPENDENCY_CLOSURE_REQUIREMENTS:
            self._bound_failure(
                "requirements",
                maximum=MAX_DEPENDENCY_CLOSURE_REQUIREMENTS,
                observed=projected_count,
                context={"name": name},
            )
            self.metadata_cache[name] = None
            return None

        normalized: list[str] = []
        for index, raw_requirement in enumerate(raw_requirements):
            if not isinstance(raw_requirement, str):
                self._append_invalid(
                    {
                        "kind": "distribution_requirements_metadata",
                        "name": name,
                        "metadata_requirement_index": index,
                        "error_type": "InvalidMetadataRequirementType",
                        "value_type": type(raw_requirement).__name__,
                    }
                )
                self.metadata_cache[name] = None
                return None
            encoded = raw_requirement.encode("utf-8", errors="surrogatepass")
            requirement_sha256 = hashlib.sha256(encoded).hexdigest()
            if len(encoded) > MAX_DEPENDENCY_CLOSURE_REQUIREMENT_BYTES:
                self._bound_failure(
                    "requirement_bytes",
                    maximum=MAX_DEPENDENCY_CLOSURE_REQUIREMENT_BYTES,
                    observed=len(encoded),
                    context={
                        "name": name,
                        "requirement_sha256": requirement_sha256,
                    },
                )
                self.metadata_cache[name] = None
                return None
            self.metadata_text_bytes += len(encoded)
            if (
                self.metadata_text_bytes
                > MAX_DEPENDENCY_CLOSURE_METADATA_TEXT_BYTES
            ):
                self._bound_failure(
                    "metadata_text_bytes",
                    maximum=MAX_DEPENDENCY_CLOSURE_METADATA_TEXT_BYTES,
                    observed=self.metadata_text_bytes,
                    context={"name": name},
                )
                self.metadata_cache[name] = None
                return None
            normalized.append(raw_requirement)
        normalized.sort(
            key=lambda value: (
                hashlib.sha256(
                    value.encode("utf-8", errors="surrogatepass")
                ).hexdigest(),
                value,
            )
        )
        self.metadata_requirement_count += len(normalized)
        result = tuple(normalized)
        self.metadata_cache[name] = result
        return result

    def _record_cycle(
        self,
        ancestry: tuple[str, ...],
        target_name: str,
    ) -> None:
        start = ancestry.index(target_name)
        body = ancestry[start:]
        rotations = tuple(body[index:] + body[:index] for index in range(len(body)))
        canonical = min(rotations)
        path = canonical + (canonical[0],)
        self.cycles[path] = {
            "path": list(path),
            "cycle_sha256": hashlib.sha256(
                "\0".join(path).encode("utf-8")
            ).hexdigest(),
        }

    def _expand_distribution(
        self,
        name: str,
        *,
        active_extra: str,
        node_depth: int,
        ancestry: tuple[str, ...],
    ) -> None:
        context_key = (name, active_extra)
        if self.stopped_on_bound or context_key in self.expanded_contexts:
            return
        observed_contexts = len(self.expanded_contexts) + 1
        if observed_contexts > MAX_DEPENDENCY_CLOSURE_CONTEXTS:
            self._bound_failure(
                "contexts",
                maximum=MAX_DEPENDENCY_CLOSURE_CONTEXTS,
                observed=observed_contexts,
                context={"name": name, "depth": node_depth},
            )
            return
        self.expanded_contexts.add(context_key)
        requirements = self._metadata_requirements(name)
        if requirements is None or self.stopped_on_bound:
            return
        for raw_requirement in requirements:
            self._visit_requirement(
                raw_requirement,
                source="distribution_metadata",
                parent_name=name,
                depth=node_depth + 1,
                ancestry=ancestry,
                active_extra=active_extra,
            )
            if self.stopped_on_bound:
                return

    def _visit_requirement(
        self,
        raw_requirement: object,
        *,
        source: str,
        parent_name: str,
        depth: int,
        ancestry: tuple[str, ...],
        active_extra: str,
    ) -> None:
        if self.stopped_on_bound:
            return
        self.requirement_evaluation_count += 1
        if (
            self.requirement_evaluation_count
            > MAX_DEPENDENCY_CLOSURE_REQUIREMENTS
        ):
            self._bound_failure(
                "requirements",
                maximum=MAX_DEPENDENCY_CLOSURE_REQUIREMENTS,
                observed=self.requirement_evaluation_count,
                context={"parent_name": parent_name, "depth": depth},
            )
            return
        if not isinstance(raw_requirement, str):
            self._append_invalid(
                {
                    "kind": "dependency",
                    "source": source,
                    "parent_name": parent_name,
                    "depth": depth,
                    "error_type": "InvalidRequirementType",
                    "value_type": type(raw_requirement).__name__,
                }
            )
            return
        encoded = raw_requirement.encode("utf-8", errors="surrogatepass")
        requirement_sha256 = hashlib.sha256(encoded).hexdigest()
        if len(encoded) > MAX_DEPENDENCY_CLOSURE_REQUIREMENT_BYTES:
            self._bound_failure(
                "requirement_bytes",
                maximum=MAX_DEPENDENCY_CLOSURE_REQUIREMENT_BYTES,
                observed=len(encoded),
                context={
                    "parent_name": parent_name,
                    "depth": depth,
                    "requirement_sha256": requirement_sha256,
                },
            )
            return
        try:
            requirement = self.requirement_factory(raw_requirement)
        except self.invalid_requirement_type as exc:
            self._append_invalid(
                {
                    "kind": "dependency",
                    "source": source,
                    "parent_name": parent_name,
                    "depth": depth,
                    "requirement_sha256": requirement_sha256,
                    "error_type": type(exc).__name__,
                }
            )
            return
        safe_record = self._safe_requirement_record(
            requirement,
            requirement_sha256=requirement_sha256,
            source=source,
            parent_name=parent_name,
            depth=depth,
        )
        if source == "project" and active_extra:
            safe_record["selected_extra"] = active_extra
        elif source == "distribution_metadata" and requirement.marker is not None:
            safe_record["marker_extra"] = active_extra
        marker_environment = dict(self.marker_environment)
        marker_environment["extra"] = active_extra
        try:
            applies = requirement.marker is None or requirement.marker.evaluate(
                environment=marker_environment
            )
        except Exception as exc:
            self._append_invalid(
                {
                    **safe_record,
                    "kind": "marker",
                    "marker_extra": active_extra,
                    "error_type": type(exc).__name__,
                }
            )
            return
        if not applies:
            self.project["marker_skipped"].append(
                {
                    **safe_record,
                    "marker_extra": active_extra,
                }
            )
            return

        if source == "distribution_metadata":
            edge_identity = (parent_name, requirement_sha256)
            if edge_identity in self.processed_metadata_edges:
                return
            self.processed_metadata_edges.add(edge_identity)
        self.edge_count += 1
        if self.edge_count > MAX_DEPENDENCY_CLOSURE_EDGES:
            self._bound_failure(
                "edges",
                maximum=MAX_DEPENDENCY_CLOSURE_EDGES,
                observed=self.edge_count,
                context={
                    "name": safe_record["name"],
                    "parent_name": parent_name,
                    "depth": depth,
                },
            )
            return
        if requirement.url:
            self._append_invalid(
                {
                    **safe_record,
                    "kind": "direct_reference_unverifiable",
                    "direct_reference_sha256": hashlib.sha256(
                        requirement.url.encode(
                            "utf-8",
                            errors="surrogatepass",
                        )
                    ).hexdigest(),
                }
            )
            return

        name = str(safe_record["name"])
        is_cycle = name in ancestry
        if depth > MAX_DEPENDENCY_CLOSURE_DEPTH:
            self._bound_failure(
                "depth",
                maximum=MAX_DEPENDENCY_CLOSURE_DEPTH,
                observed=depth,
                context={"name": name, "parent_name": parent_name},
            )
            return
        state = self._node_state(name)
        if state is None or self.stopped_on_bound:
            return
        if state.get("status") == "missing":
            self.project["passed"] = False
            self.project["missing"].append(safe_record)
            return
        if state.get("status") != "present":
            self._append_invalid(
                {
                    **safe_record,
                    **{
                        str(key): value
                        for key, value in state.items()
                        if key != "status"
                    },
                }
            )
            return

        observed = {
            **safe_record,
            "installed_version": str(state["version"]),
            "installed_version_sha256": str(state["version_sha256"]),
        }
        self.project["observed"].append(observed)
        if (
            requirement.specifier
            and state["parsed_version"] not in requirement.specifier
        ):
            self.project["passed"] = False
            self.project["incompatible"].append(observed)
        requested_extras = tuple(safe_record["extras"])
        if is_cycle:
            self._record_cycle(ancestry, name)
            for marker_extra in ("", *requested_extras):
                self._expand_distribution(
                    name,
                    active_extra=str(marker_extra),
                    node_depth=depth,
                    ancestry=ancestry,
                )
                if self.stopped_on_bound:
                    return
            return

        child_ancestry = ancestry + (name,)
        for marker_extra in ("", *requested_extras):
            self._expand_distribution(
                name,
                active_extra=str(marker_extra),
                node_depth=depth,
                ancestry=child_ancestry,
            )
            if self.stopped_on_bound:
                return

    def evaluate(
        self,
        requirements: Sequence[object],
        marker_extras: Sequence[str],
    ) -> None:
        for requirement_index, raw_requirement in enumerate(requirements):
            self._visit_requirement(
                raw_requirement,
                source="project",
                parent_name="",
                depth=0,
                ancestry=(),
                active_extra=marker_extras[requirement_index],
            )
            if self.stopped_on_bound:
                break

        for field_name in (
            "missing",
            "incompatible",
            "invalid",
            "marker_skipped",
            "observed",
        ):
            self.project[field_name] = self._deduplicate(
                self.project[field_name]
            )
        cycles = [self.cycles[key] for key in sorted(self.cycles)]
        self.project["dependency_closure"] = {
            "mode": (
                "recursive_installed_metadata"
                if self.requires_getter is not None
                else "direct_only_injected_inventory"
            ),
            "node_count": len(self.node_states),
            "edge_count": self.edge_count,
            "requirement_evaluation_count": self.requirement_evaluation_count,
            "metadata_distribution_count": len(self.metadata_cache),
            "metadata_requirement_count": self.metadata_requirement_count,
            "metadata_text_bytes": self.metadata_text_bytes,
            "expanded_context_count": len(self.expanded_contexts),
            "cycle_count": len(cycles),
            "cycles": cycles,
            "stopped_on_bound": self.stopped_on_bound,
            "bounds": {
                "nodes": MAX_DEPENDENCY_CLOSURE_NODES,
                "edges": MAX_DEPENDENCY_CLOSURE_EDGES,
                "depth": MAX_DEPENDENCY_CLOSURE_DEPTH,
                "requirements": MAX_DEPENDENCY_CLOSURE_REQUIREMENTS,
                "contexts": MAX_DEPENDENCY_CLOSURE_CONTEXTS,
                "requirement_bytes": (
                    MAX_DEPENDENCY_CLOSURE_REQUIREMENT_BYTES
                ),
                "metadata_text_bytes": (
                    MAX_DEPENDENCY_CLOSURE_METADATA_TEXT_BYTES
                ),
                "installed_version_bytes": (
                    MAX_DEPENDENCY_CLOSURE_INSTALLED_VERSION_BYTES
                ),
            },
        }


def _evaluate_dependency_payload(
    payload: Mapping[str, Any],
    *,
    version_getter: Callable[[str], str] = _DEFAULT_VERSION_GETTER,
    requires_getter: (
        Callable[[str], Sequence[str] | None] | None | object
    ) = _DEFAULT_REQUIRES_GETTER,
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
        from packaging.utils import canonicalize_name
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
    if requires_getter is _DEFAULT_REQUIRES_GETTER:
        effective_requires_getter = (
            importlib.metadata.requires
            if version_getter is _DEFAULT_VERSION_GETTER
            else None
        )
    elif requires_getter is None:
        effective_requires_getter = None
    elif callable(requires_getter):
        effective_requires_getter = requires_getter
    else:
        result.update(
            {
                "reason": "dependency_probe_metadata_getter_invalid",
                "error_type": type(requires_getter).__name__,
            }
        )
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
            "project_name_sha256": str(source_project.get("project_name_sha256") or ""),
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
            requires_python_sha256 = hashlib.sha256(requires_python.encode("utf-8")).hexdigest()
            try:
                python_specifier = SpecifierSet(requires_python)
                if Version(platform.python_version()) not in python_specifier:
                    project["passed"] = False
                    project["incompatible"].append(
                        {
                            "kind": "python",
                            "requirement_sha256": requires_python_sha256,
                            "installed_version": platform.python_version(),
                        }
                    )
            except (InvalidSpecifier, InvalidVersion) as exc:
                project["passed"] = False
                project["invalid"].append(
                    {
                        "kind": "requires-python",
                        "requirement_sha256": requires_python_sha256,
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
        marker_extras = source_project.get("requirement_marker_extras")
        if (
            not isinstance(marker_extras, list)
            or len(marker_extras) != len(requirements)
            or not all(
                isinstance(item, str)
                and item
                in {
                    "",
                    *PYTEST_OPTIONAL_DEPENDENCY_EXTRA_PRIORITY,
                }
                for item in marker_extras
            )
        ):
            project["passed"] = False
            project["invalid"].append(
                {
                    "kind": "requirement_marker_extras",
                    "error_type": "InvalidProbePayload",
                }
            )
            marker_extras = []
            requirements = []
        closure = _DependencyClosureEvaluator(
            project,
            marker_environment=environment,
            requirement_factory=Requirement,
            invalid_requirement_type=InvalidRequirement,
            version_factory=Version,
            invalid_version_type=InvalidVersion,
            canonicalize_name=canonicalize_name,
            version_getter=version_getter,
            requires_getter=effective_requires_getter,
        )
        closure.evaluate(requirements, marker_extras)
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
            "error_sha256": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
        }
    source_attestation = str(globals().get("__probe_source_sha256__") or "")
    if source_attestation:
        result["probe_source_sha256"] = source_attestation
    sys.stdout.write(_canonical_json(result))
    sys.stdout.write("\n")
    return 0


def _run_bounded_probe_process(
    command: Sequence[str],
    *,
    input_payload: bytes,
    environment: Mapping[str, str],
) -> tuple[int | None, bytes, dict[str, Any]]:
    """Capture child output incrementally and kill it at the byte/time bound."""

    with tempfile.TemporaryFile() as input_stream:
        input_stream.write(input_payload)
        input_stream.seek(0)
        process = subprocess.Popen(
            list(command),
            stdin=input_stream,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=dict(environment),
        )
        if process.stdout is None:  # pragma: no cover - Popen contract guard
            process.kill()
            process.wait()
            return None, b"", {"reason": "dependency_probe_pipe_missing"}
        output = bytearray()
        output_exceeded = threading.Event()
        reader_error: list[str] = []

        def read_output() -> None:
            try:
                while True:
                    chunk = process.stdout.read(64 * 1024)
                    if not chunk:
                        return
                    remaining = MAX_PROBE_OUTPUT_BYTES + 1 - len(output)
                    if remaining > 0:
                        output.extend(chunk[:remaining])
                    if len(output) > MAX_PROBE_OUTPUT_BYTES:
                        output_exceeded.set()
                        process.kill()
                        return
            except (OSError, ValueError) as exc:
                reader_error.append(type(exc).__name__)
                try:
                    process.kill()
                except OSError:
                    pass

        reader = threading.Thread(
            target=read_output,
            name="validation-dependency-probe-output",
            daemon=True,
        )
        reader.start()
        timed_out = False
        try:
            returncode = process.wait(timeout=DEPENDENCY_PROBE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            returncode = process.wait()
        reader.join(timeout=2.0)
        try:
            process.stdout.close()
        except OSError:
            pass
        if timed_out:
            return (
                returncode,
                bytes(output),
                {"reason": "dependency_probe_process_timed_out"},
            )
        if output_exceeded.is_set():
            return (
                returncode,
                bytes(output[:MAX_PROBE_OUTPUT_BYTES]),
                {
                    "reason": "dependency_probe_output_exceeded_bound",
                    "maximum_output_bytes": MAX_PROBE_OUTPUT_BYTES,
                },
            )
        if reader.is_alive() or reader_error:
            return (
                returncode,
                bytes(output),
                {
                    "reason": "dependency_probe_output_read_failed",
                    "error_type": (reader_error[0] if reader_error else "ReaderThreadDidNotStop"),
                },
            )
        return returncode, bytes(output), {}


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
    source = module_path.read_bytes()
    source_sha256 = hashlib.sha256(source).hexdigest()
    if len(source) > MAX_PROBE_SOURCE_BYTES:
        return {
            "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
            "passed": False,
            "reason": "dependency_probe_source_exceeded_bound",
            "source_bytes": len(source),
            "maximum_source_bytes": MAX_PROBE_SOURCE_BYTES,
            "preflight_source_sha256": source_sha256,
        }
    encoded_source = base64.b85encode(zlib.compress(source, level=9))
    with validation_python_launcher_environment(validation_environment) as (
        launcher_environment,
        launcher_receipt,
    ):
        try:
            returncode, output_bytes, process_error = _run_bounded_probe_process(
                [
                    launcher_environment["PYTHON"],
                    "-c",
                    _PROBE_ARGV_BOOTSTRAP,
                    encoded_source.decode("ascii"),
                    "--probe",
                ],
                input_payload=_canonical_json(payload).encode("utf-8"),
                environment=launcher_environment,
            )
        except OSError as exc:
            return {
                "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
                "passed": False,
                "reason": "dependency_probe_process_unavailable",
                "error_type": type(exc).__name__,
                "preflight_source_sha256": source_sha256,
            }
        output = output_bytes.decode("utf-8", errors="replace")
        launcher = {
            "content_sha256": launcher_receipt.content_sha256,
            "interpreter_sha256": launcher_receipt.interpreter_sha256,
            "interpreter_stat": launcher_receipt.interpreter_stat,
            "mode": launcher_receipt.mode,
            "policy_sha256": launcher_receipt.policy_sha256,
            "sealed": launcher_receipt.sealed,
        }
        source_delivery = {
            "mode": "compressed_argv_copy",
            "sha256": source_sha256,
            "bytes": len(source),
        }
        if process_error:
            return {
                "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
                "passed": False,
                **process_error,
                "preflight_source_delivery": source_delivery,
                "validation_python_launcher": launcher,
            }
        if returncode != 0:
            return {
                "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
                "passed": False,
                "reason": "dependency_probe_process_failed",
                "returncode": int(returncode or 0),
                "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
                "output_bytes": len(output_bytes),
                "preflight_source_delivery": source_delivery,
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
                "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
                "output_bytes": len(output_bytes),
                "preflight_source_delivery": source_delivery,
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
                "preflight_source_delivery": source_delivery,
                "validation_python_launcher": launcher,
            }
        if result.get("probe_source_sha256") != source_sha256:
            return {
                "schema": PROJECT_DEPENDENCY_PROBE_SCHEMA,
                "passed": False,
                "reason": "dependency_probe_source_attestation_mismatch",
                "preflight_source_delivery": source_delivery,
                "validation_python_launcher": launcher,
            }
        result["preflight_source_delivery"] = source_delivery
        result["validation_python_launcher"] = launcher
        return result


if not _CHILD_PROBE_MODE:
    _run_dependency_probe = sealed_validation_python_runner(_run_dependency_probe)


def _preflight_validation_project_dependencies(
    workspace_path: Path | str,
    validation_commands: Sequence[str],
    *,
    environment: Mapping[str, object] | None = None,
    probe_runner: Callable[..., dict[str, Any]] = _run_dependency_probe,
) -> dict[str, Any]:
    """Compare static PEP-621 requirements with the approved interpreter."""

    workspace = Path(workspace_path)
    validation_roots: list[str] = []
    project_roots: list[str] = []
    pytest_roots: set[str] = set()
    dependency_neutral_commands: list[dict[str, Any]] = []
    invalid_commands: list[dict[str, Any]] = []
    for index, command in enumerate(validation_commands):
        command_text = str(command)
        root = validation_command_repository_root(command_text)
        if root is None:
            invalid_commands.append(
                {
                    "command_index": index,
                    "command_sha256": hashlib.sha256(command_text.encode("utf-8")).hexdigest(),
                    "reason": "validation_repository_root_is_unsafe",
                }
            )
            continue
        if root not in validation_roots:
            validation_roots.append(root)
        dependency_scope = validation_command_dependency_scope(command_text)
        if dependency_scope is ValidationDependencyScope.DEPENDENCY_NEUTRAL:
            dependency_neutral_commands.append(
                {
                    "command_index": index,
                    "command_sha256": hashlib.sha256(
                        command_text.encode("utf-8")
                    ).hexdigest(),
                    "root": root,
                    "scope": dependency_scope.value,
                }
            )
            continue
        if root not in project_roots:
            project_roots.append(root)
        if _validation_command_invokes_pytest(command_text):
            pytest_roots.add(root)

    projects = [
        _bounded_static_project(
            workspace,
            relative_root,
            pytest_invoked=relative_root in pytest_roots,
        )
        for relative_root in project_roots
    ]
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
        "automatic_install_attempted": False,
        "probe_scope": "installed_distribution_metadata",
        "validation_command_count": len(validation_commands),
        "validation_roots": validation_roots,
        "project_roots": project_roots,
        "projects": [_public_project_contract(project) for project in projects],
        "dependency_neutral_command_count": len(dependency_neutral_commands),
        "dependency_neutral_commands": dependency_neutral_commands,
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
                else (
                    "validation_commands_dependency_neutral"
                    if len(dependency_neutral_commands) == len(validation_commands)
                    else "no_static_pep621_dependencies"
                )
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
                        "project_name_sha256",
                        "pyproject_sha256",
                        "requirements",
                        "requirement_marker_extras",
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
                "error_sha256": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
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
        contract_invalid = bool(invalid)
        receipt = {
            **base_receipt,
            "passed": passed,
            "reason": (
                "approved_validation_environment_satisfies_project_dependencies"
                if passed
                else (
                    "project_dependency_contract_validation_failed"
                    if contract_invalid
                    else ("approved_validation_environment_dependency_probe_failed")
                    if not drift_detected
                    else "approved_validation_environment_dependency_drift"
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
                    "repair_static_project_dependency_contract"
                    if contract_invalid
                    else (
                        "provision_approved_validation_environment"
                        if drift_detected
                        else "repair_approved_validation_dependency_probe"
                    )
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
    receipt["retry_fingerprint"] = _retry_fingerprint(receipt)
    return receipt


def project_dependency_preflight_error_receipt(
    workspace_path: Path | str,
    validation_commands: Sequence[str],
    exc: BaseException,
) -> dict[str, Any]:
    """Return a secret-safe typed receipt for unexpected preflight failure."""

    receipt: dict[str, Any] = {
        "schema": PROJECT_DEPENDENCY_PREFLIGHT_SCHEMA,
        "workspace": str(Path(workspace_path)),
        "passed": False,
        "applicable": True,
        "reason": "project_dependency_preflight_infrastructure_error",
        "error_type": type(exc).__name__,
        "error_sha256": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
        "validation_command_sha256": [
            hashlib.sha256(str(command).encode("utf-8")).hexdigest()
            for command in validation_commands
        ],
        "automatic_install_attempted": False,
        "probe_scope": "installed_distribution_metadata",
        "remediation": {
            "kind": "repair_approved_validation_dependency_probe",
            "automatic_provisioning": False,
            "rerun_required": True,
        },
    }
    receipt["receipt_id"] = _content_sha256(receipt)
    receipt["retry_fingerprint"] = _retry_fingerprint(receipt)
    return receipt


def project_dependency_preflight_backoff_seconds(
    retry_fingerprint: str,
    prior_retry_fingerprints: Sequence[str],
) -> int:
    """Return bounded exponential cooldown for one unchanged drift receipt."""

    fingerprint = str(retry_fingerprint or "").strip()
    repeated = 0
    if fingerprint:
        for prior in prior_retry_fingerprints:
            if str(prior or "").strip() != fingerprint:
                break
            repeated += 1
    multiplier = 1 << min(repeated, 8)
    return min(
        PROJECT_DEPENDENCY_PREFLIGHT_MAX_BACKOFF_SECONDS,
        PROJECT_DEPENDENCY_PREFLIGHT_BACKOFF_SECONDS * multiplier,
    )


def preflight_validation_project_dependencies(
    workspace_path: Path | str,
    validation_commands: Sequence[str],
    *,
    environment: Mapping[str, object] | None = None,
    probe_runner: Callable[..., dict[str, Any]] = _run_dependency_probe,
) -> dict[str, Any]:
    """Return a receipt on every path; unexpected failures also fail closed."""

    try:
        return _preflight_validation_project_dependencies(
            workspace_path,
            validation_commands,
            environment=environment,
            probe_runner=probe_runner,
        )
    except Exception as exc:
        return project_dependency_preflight_error_receipt(
            workspace_path,
            validation_commands,
            exc,
        )


if __name__ == "__main__":
    raise SystemExit(_probe_main() if "--probe" in sys.argv[1:] else 64)
