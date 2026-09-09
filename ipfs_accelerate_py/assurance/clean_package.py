"""Fail-closed PCPR-052 Accelerate clean package.

A clean Accelerate install must not require sibling source trees, recursive
submodules, editable local dependencies, implicit sys.path injection,
import-time network, or mutable Git-branch release requires.

This module is not release authority: it does not write DuckDB or Quack
state and never emits a closed PCPR release outcome. Live claims require
live evidence. Simulated results are not live. Missing compute backends
stay typed unavailable. Built wheel and sdist artifacts that are not
present remain typed unavailable and are not recorded as a published
release.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import tomllib
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

INTERFACE: Final = "AccelerateCleanPackage@1"
SCHEMA: Final = "ipfs_accelerate_py/assurance/clean-package@1"
VERDICT_SCHEMA: Final = "ipfs_accelerate_py/assurance/clean-package-verdict@1"
PCPR_052_TASK_ID: Final = "PCPR-052"
PCPR_052_GOAL_ID: Final = "PCPR-G600"
PCPR_051_TASK_ID: Final = "PCPR-051"
PCPR_050_TASK_ID: Final = "PCPR-050"
PCPR_043_TASK_ID: Final = "PCPR-043"
PCPR_039_TASK_ID: Final = "PCPR-039"
PCPR_038_TASK_ID: Final = "PCPR-038"
PCPR_037_TASK_ID: Final = "PCPR-037"
PCPR_001_TASK_ID: Final = "PCPR-001"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = (
    "proof-carrying-platform-qualification-and-release-v1"
)
EVIDENCE_ID: Final = "pcpr/accelerate-clean-package@1"

PACKAGE_NAME: Final = "ipfs_accelerate_py"
PACKAGE_VERSION: Final = "0.0.45"
CANONICAL_PYTHON_REQUIRES: Final = ">=3.12"
CANONICAL_PYTHON_CLASSIFIER: Final = "Programming Language :: Python :: 3.12"
CANONICAL_LICENSE_CLASSIFIER: Final = (
    "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)"
)
CANONICAL_SPDX: Final = "AGPL-3.0-or-later"
RUNTIME_REQUIRES_AUTHORITY: Final = "requirements.txt"
PYPROJECT_CLEAN_PACKAGE_TABLE: Final = "tool.ipfs_accelerate_py.clean-package"
NAMED_GIT_EXTRAS: Final[tuple[str, ...]] = (
    "libp2p",
    "mcp-p2p",
    "ipfs-transformers",
    "ipfs-model-manager",
)
NAMED_GIT_EXTRAS_ROLE: Final = "source-checkout-development-and-integration"
INNER_DEVELOPMENT_REQUIREMENTS: Final = "ipfs_accelerate_py/requirements.txt"

SIBLING_TREES: Final[tuple[str, ...]] = (
    "ipfs_datasets_py",
    "ipfs_kit_py",
    "ipfs_model_manager_py",
    "ipfs_transformers_py",
    ".tools",
)
RECURSIVE_SUBMODULE_TREES: Final[tuple[str, ...]] = (
    "docs/fastmcp",
    "docs/mcp-python-sdk",
    "ipfs_accelerate_py/mcplusplus",
    "test/doc-builder",
    "test/huggingface_doc_builder",
    "test/huggingface_transformers",
)
MANIFEST_PRUNE_LINES: Final[tuple[str, ...]] = (
    "prune ipfs_datasets_py",
    "prune ipfs_kit_py",
    "prune ipfs_model_manager_py",
    "prune ipfs_transformers_py",
    "prune .tools",
    "prune docs/fastmcp",
    "prune docs/mcp-python-sdk",
    "prune ipfs_accelerate_py/mcplusplus",
    "prune test/doc-builder",
    "prune test/huggingface_doc_builder",
    "prune test/huggingface_transformers",
)
FIND_PACKAGE_EXCLUDES: Final[tuple[str, ...]] = (
    "ipfs_datasets_py*",
    "ipfs_kit_py*",
    "ipfs_model_manager_py*",
    "ipfs_transformers_py*",
)
FIND_PACKAGE_INCLUDES: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py",
    "ipfs_accelerate_py.*",
    "scripts",
    "scripts.*",
)

CLOSED_RELEASE_OUTCOMES: Final[frozenset[str]] = frozenset(
    {
        "release_candidate_qualified",
        "non_promoted_supervisor_unqualified",
        "non_promoted_import_or_false_success",
        "non_promoted_live_storage_gap",
        "non_promoted_live_compute_gap",
        "non_promoted_solver_gap",
        "non_promoted_packaging_gap",
        "non_promoted_dependency_reproducibility",
        "non_promoted_security_failure",
        "non_promoted_interoperability_gap",
        "non_promoted_reference_workflow_failure",
        "non_promoted_unmeasured",
        "non_promoted_operator_gate_required",
    }
)
PROMOTION_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "supervisor_promoted",
        "supervisor_non_promoted",
        "rnd_non_promoted",
        "typed_unavailable",
        "typed_blocked",
    }
)
EVIDENCE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "measured",
        "measured_live",
        "measured_hermetic",
        "estimated",
        "simulated",
        "unavailable",
    }
)

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_clean_package.py",
)

SEALED_PATH: Final = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
SEALED_PYTHON: Final = "/usr/bin/python3.12"

_VCS_REQUIREMENT_RE: Final = re.compile(
    r"""(?:^|[\s"'=,@])(?:git\+|hg\+|svn\+|bzr\+|file://)""",
    re.IGNORECASE,
)
_EDITABLE_REQUIREMENT_RE: Final = re.compile(
    r"""(?:^|[\s"'])(?:-e|--editable)\b""",
    re.IGNORECASE,
)
_SYS_PATH_RE: Final = re.compile(
    r"""sys\.path\.(?:insert|append|extend)\s*\(""",
)
_CID_RE: Final = re.compile(r"^b[a-z2-7]+$")


class CleanPackageError(Exception):
    """Fail-closed PCPR-052 contract error."""


class CleanPackageAdmissionError(CleanPackageError):
    """Raised when a packaging input is rejected."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def content_identity(value: Any) -> str:
    """CIDv1 DAG-JSON/sha2-256 identity (baguqeera…)."""

    digest = hashlib.sha256(canonical_json_bytes(value)).digest()
    raw = b"\x01\xa9\x02\x12\x20" + digest
    return "b" + base64.b32encode(raw).decode("ascii").rstrip("=").lower()


def discover_accelerate_root(start: Path | None = None) -> Path | None:
    here = Path(start or __file__).resolve()
    for candidate in (here, *here.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "setup.py").is_file()
            and (candidate / "ipfs_accelerate_py" / "__init__.py").is_file()
        ):
            return candidate
    return None


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CleanPackageError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise CleanPackageError(f"{name} is not an admitted evidence kind")
    return kind


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise CleanPackageError(
            f"{name} must not be a closed PCPR release outcome"
        )


def clean_package_manifest() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "interface": INTERFACE,
        "package_name": PACKAGE_NAME,
        "package_version": PACKAGE_VERSION,
        "python_requires": CANONICAL_PYTHON_REQUIRES,
        "sibling_trees": list(SIBLING_TREES),
        "recursive_submodule_trees": list(RECURSIVE_SUBMODULE_TREES),
        "requires_sibling_source_trees": False,
        "requires_recursive_submodules": False,
        "editable_local_dependencies": False,
        "sys_path_injection": False,
        "import_network": False,
        "mutable_git_release_requires": False,
        "runtime_requires_authority": RUNTIME_REQUIRES_AUTHORITY,
        "development_requirements_role": (
            "source-checkout-development-and-integration"
        ),
        "named_git_extras": list(NAMED_GIT_EXTRAS),
        "named_git_extras_role": NAMED_GIT_EXTRAS_ROLE,
        "import_side_effects": "none",
        "task_id": PCPR_052_TASK_ID,
        "goal_id": PCPR_052_GOAL_ID,
    }


def scan_requirement_text(text: str) -> dict[str, tuple[str, ...]]:
    """Return forbidden requirement kinds found in a dependency document."""

    body = text or ""
    vcs = tuple(
        sorted({match.group(0).strip() for match in _VCS_REQUIREMENT_RE.finditer(body)})
    )
    editable = tuple(
        sorted(
            {
                match.group(0).strip()
                for match in _EDITABLE_REQUIREMENT_RE.finditer(body)
            }
        )
    )
    return {
        "vcs": vcs,
        "editable": editable,
    }


def parse_setup_metadata(text: str) -> dict[str, Any]:
    body = _text(text, "setup.py")
    return {
        "python_requires_constant": (
            CANONICAL_PYTHON_REQUIRES
            if f'PYTHON_REQUIRES = "{CANONICAL_PYTHON_REQUIRES}"' in body
            else None
        ),
        "uses_python_requires_constant": "python_requires=PYTHON_REQUIRES" in body,
        "has_install_requires_assignment": bool(
            re.search(r"^install_requires\s*=", body, re.MULTILINE)
        ),
        "reads_requirements_txt": 'this_directory / "requirements.txt"' in body,
        "find_includes_constant": "SETUP_PACKAGE_INCLUDES" in body,
        "find_excludes_constant": "SETUP_PACKAGE_EXCLUDES" in body,
        "find_packages_uses_excludes": "exclude=SETUP_PACKAGE_EXCLUDES" in body,
    }


def parse_pyproject_clean_package(text: str) -> dict[str, Any]:
    payload = tomllib.loads(_text(text, "pyproject.toml"))
    if not isinstance(payload, dict):
        raise CleanPackageError("pyproject.toml must be a table")
    project = payload.get("project")
    if not isinstance(project, dict):
        raise CleanPackageError("pyproject.toml [project] is required")
    requires_python = project.get("requires-python")
    classifiers = project.get("classifiers")
    project_classifiers = (
        tuple(str(item) for item in classifiers)
        if isinstance(classifiers, list)
        else ()
    )
    dependencies = project.get("dependencies")
    dynamic = project.get("dynamic")
    optional = project.get("optional-dependencies")
    optional_map: dict[str, list[str]] = {}
    if isinstance(optional, dict):
        for name, items in optional.items():
            if isinstance(items, list):
                optional_map[str(name)] = [str(item) for item in items]
    tool = payload.get("tool")
    clean: dict[str, Any] = {}
    find_exclude: tuple[str, ...] = ()
    find_include: tuple[str, ...] = ()
    package_data_keys: tuple[str, ...] = ()
    dynamic_dependency_files: tuple[str, ...] = ()
    if isinstance(tool, dict):
        accelerate = tool.get("ipfs_accelerate_py")
        if isinstance(accelerate, dict):
            table = accelerate.get("clean-package")
            if isinstance(table, dict):
                clean = dict(table)
        setuptools = tool.get("setuptools")
        if isinstance(setuptools, dict):
            packages = setuptools.get("packages")
            if isinstance(packages, dict):
                find = packages.get("find")
                if isinstance(find, dict):
                    raw_exclude = find.get("exclude")
                    if isinstance(raw_exclude, list):
                        find_exclude = tuple(str(item) for item in raw_exclude)
                    raw_include = find.get("include")
                    if isinstance(raw_include, list):
                        find_include = tuple(str(item) for item in raw_include)
            package_data = setuptools.get("package-data")
            if isinstance(package_data, dict):
                package_data_keys = tuple(str(key) for key in package_data)
            dyn = setuptools.get("dynamic")
            if isinstance(dyn, dict):
                deps = dyn.get("dependencies")
                if isinstance(deps, dict):
                    file_val = deps.get("file")
                    if isinstance(file_val, list):
                        dynamic_dependency_files = tuple(
                            str(item) for item in file_val
                        )
                    elif isinstance(file_val, str):
                        dynamic_dependency_files = (file_val,)
    return {
        "requires_python": requires_python,
        "classifiers": list(project_classifiers),
        "dependencies": (
            list(dependencies) if isinstance(dependencies, list) else dependencies
        ),
        "dynamic": list(dynamic) if isinstance(dynamic, list) else dynamic,
        "dynamic_dependency_files": list(dynamic_dependency_files),
        "optional_dependencies": optional_map,
        "clean_package": clean,
        "find_exclude": list(find_exclude),
        "find_include": list(find_include),
        "package_data_keys": list(package_data_keys),
    }


def parse_manifest_prunes(text: str) -> tuple[str, ...]:
    lines = []
    for raw in _text(text, "MANIFEST.in").splitlines():
        stripped = raw.strip()
        if stripped.startswith("prune "):
            lines.append(stripped)
    return tuple(lines)


def package_init_injects_sys_path(text: str) -> bool:
    return bool(_SYS_PATH_RE.search(_text(text, "__init__.py")))


REQUIRED_GOOD_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "release_requires_have_no_vcs",
        "release_requires_have_no_editable",
        "pyproject_dynamic_dependencies_from_requirements",
        "python_requires_agrees",
        "python_classifier_agrees",
        "pyproject_python_classifier_agrees",
        "sibling_packages_excluded",
        "sibling_submodules_pruned",
        "recursive_submodules_pruned",
        "pyproject_clean_package_table",
        "package_init_has_no_sys_path_injection",
        "git_extras_are_not_release_profile",
        "full_extra_has_no_vcs",
        "all_extra_has_no_vcs",
        "inner_requirements_git_is_not_release_profile",
        "setup_find_packages_excludes_siblings",
        "manifest_advertises_clean_package",
        "isolated_import_without_siblings",
    }
)
FORBIDDEN_PRESENT_PROBE_IDS: Final[frozenset[str]] = frozenset(
    {
        "mutable_git_release_requires",
        "editable_local_release_requires",
        "sibling_source_imported",
        "sys_path_injection_observed",
        "simulated_results_represented_as_live",
        "runtime_unavailable_represented_as_live",
        "built_wheel_represented_as_live",
        "built_sdist_represented_as_live",
        "published_release_represented_as_live",
    }
)


@dataclass(frozen=True)
class OutcomeProbe:
    probe_id: str
    present: bool | None
    evidence_kind: str
    live: bool
    simulated_represented_as_live: bool
    reason: str
    details: Mapping[str, Any] = MappingProxyType({})

    def to_mapping(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "present": self.present,
            "evidence_kind": self.evidence_kind,
            "live": self.live,
            "simulated_represented_as_live": self.simulated_represented_as_live,
            "reason": self.reason,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class CleanPackageVerdict:
    schema: str
    interface: str
    promotion_status: str
    supervisor_disposition: str
    closed_release_outcome: str | None
    release_claim: bool
    completion_authoritative: bool
    contracts_frozen: bool
    duckdb_or_quack_state_written: bool
    sibling_source_required: bool
    mutable_git_release_requires: bool
    isolated_import_without_siblings: bool
    simulated_results_represented_as_live: bool
    live_compute_qualified: bool
    live_compute_evidence_kind: str
    built_wheel_evidence_kind: str
    built_sdist_evidence_kind: str
    this_task_created_competing_authority: bool
    probes: tuple[OutcomeProbe, ...]
    blockers: tuple[str, ...]
    verdict_cid: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "verdict_cid": self.verdict_cid,
            "promotion_status": self.promotion_status,
            "supervisor_disposition": self.supervisor_disposition,
            "closed_release_outcome": self.closed_release_outcome,
            "release_claim": self.release_claim,
            "completion_authoritative": self.completion_authoritative,
            "contracts_frozen": self.contracts_frozen,
            "duckdb_or_quack_state_written": self.duckdb_or_quack_state_written,
            "sibling_source_required": self.sibling_source_required,
            "mutable_git_release_requires": self.mutable_git_release_requires,
            "isolated_import_without_siblings": (
                self.isolated_import_without_siblings
            ),
            "simulated_results_represented_as_live": (
                self.simulated_results_represented_as_live
            ),
            "live_compute_qualified": self.live_compute_qualified,
            "live_compute_evidence_kind": self.live_compute_evidence_kind,
            "built_wheel_evidence_kind": self.built_wheel_evidence_kind,
            "built_sdist_evidence_kind": self.built_sdist_evidence_kind,
            "this_task_created_competing_authority": (
                self.this_task_created_competing_authority
            ),
            "blocker_count": len(self.blockers),
            "blockers": list(self.blockers),
            "evidence_kind": "measured",
        }


def _probe(
    probe_id: str,
    present: bool | None,
    *,
    reason: str,
    evidence_kind: str = "measured",
    live: bool = False,
    details: Mapping[str, Any] | None = None,
) -> OutcomeProbe:
    return OutcomeProbe(
        probe_id=probe_id,
        present=present,
        evidence_kind=evidence_kind,
        live=live,
        simulated_represented_as_live=False,
        reason=reason,
        details=MappingProxyType(dict(details or {})),
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _which_sealed(name: str) -> str:
    path = shutil.which(name, path=SEALED_PATH)
    if not path:
        return "unavailable"
    resolved = Path(path)
    if not resolved.is_file():
        return "unavailable"
    return str(resolved)


def _sealed_python_module_origin(name: str) -> str:
    python = Path(SEALED_PYTHON)
    if not python.is_file():
        return "unavailable"
    script = (
        "import importlib.util, sys; "
        f"spec = importlib.util.find_spec({name!r}); "
        "print(spec.origin if spec is not None and spec.origin else 'unavailable')"
    )
    try:
        completed = subprocess.run(
            [str(python), "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env={
                "PATH": SEALED_PATH,
                "PYTHONNOUSERSITE": "1",
                "HOME": "/tmp/ipfs-accelerate-validation-home-pcpr052",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    origin = (completed.stdout or "").strip() or "unavailable"
    if origin == "unavailable" or "/home/" in origin or ".local" in origin:
        return "unavailable"
    return origin


def observe_sealed_validation_environment() -> dict[str, Any]:
    """Measure the sealed PATH; missing tools stay typed unavailable."""

    python3_12 = Path(SEALED_PYTHON)
    python_version = "unavailable"
    if python3_12.is_file():
        try:
            completed = subprocess.run(
                [str(python3_12), "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
                env={"PATH": SEALED_PATH, "PYTHONNOUSERSITE": "1"},
            )
            python_version = (
                completed.stdout or completed.stderr or ""
            ).strip() or "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            python_version = "unavailable"
    pytest_origin = _sealed_python_module_origin("pytest")
    pytest_version = "unavailable"
    if pytest_origin != "unavailable":
        try:
            completed = subprocess.run(
                [
                    str(python3_12),
                    "-c",
                    "import pytest; print(pytest.__version__)",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
                env={"PATH": SEALED_PATH, "PYTHONNOUSERSITE": "1"},
            )
            pytest_version = (completed.stdout or "").strip() or "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            pytest_version = "unavailable"
    nvidia = _which_sealed("nvidia-smi")
    return {
        "PATH": SEALED_PATH,
        "python_unqualified": _which_sealed("python"),
        "python3": _which_sealed("python3"),
        "python3_12": str(python3_12) if python3_12.is_file() else "unavailable",
        "python_version": python_version,
        "git_path": _which_sealed("git"),
        "pip_path": _which_sealed("pip"),
        "usr_bin_user_writable": os.access("/usr/bin", os.W_OK),
        "usr_local_bin_user_writable": os.access("/usr/local/bin", os.W_OK),
        "pytest_cli": _which_sealed("pytest"),
        "pytest_module": pytest_origin,
        "pytest_module_version": pytest_version,
        "setuptools_module": _sealed_python_module_origin("setuptools"),
        "wheel_module": _sealed_python_module_origin("wheel"),
        "build_module": _sealed_python_module_origin("build"),
        "pip_module": _sealed_python_module_origin("pip"),
        "duckdb": _which_sealed("duckdb"),
        "z3": _which_sealed("z3"),
        "cvc5": _which_sealed("cvc5"),
        "lean": _which_sealed("lean"),
        "coqtop": _which_sealed("coqtop"),
        "ipfs": _which_sealed("ipfs"),
        "nvcc": _which_sealed("nvcc"),
        "tsc": _which_sealed("tsc"),
        "node_path": _which_sealed("node"),
        "nvidia_smi": nvidia,
        "nvidia_smi_is_not_contract_qualification": True,
        "evidence_kind": "measured",
    }


def probe_isolated_import_without_siblings(
    root: Path,
    *,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Import Accelerate from a sibling-free overlay. Not a live release."""

    python = python_executable or sys.executable
    if not python:
        return {
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "import_ok": False,
            "sibling_source_imported": False,
            "reason": "Python interpreter is not present.",
        }
    overlay = Path(tempfile.mkdtemp(prefix="pcpr052-overlay-"))
    script = textwrap.dedent(
        r"""
        import json
        import os
        import socket
        import sys

        effects = []

        def _deny_net(api, *a, **k):
            effects.append({"kind": "network", "api": api})
            raise OSError(f"PCPR-052 network denied: {api}")

        socket.create_connection = lambda *a, **k: _deny_net("create_connection")
        socket.getaddrinfo = lambda *a, **k: _deny_net("getaddrinfo")

        overlay = os.environ["PCPR052_OVERLAY"]
        accel_root = os.path.realpath(os.environ["PCPR052_ACCEL_ROOT"])
        sibling_names = (
            "ipfs_datasets_py",
            "ipfs_kit_py",
            "ipfs_model_manager_py",
            "ipfs_transformers_py",
        )

        def _is_source_checkout(path: str) -> bool:
            if not path:
                return False
            resolved = os.path.realpath(path)
            if resolved == accel_root or resolved.startswith(accel_root + os.sep):
                return True
            if "site-packages" in resolved or "dist-packages" in resolved:
                return False
            for name in sibling_names:
                marker = os.sep + name
                if resolved.endswith(marker) or (marker + os.sep) in resolved:
                    return True
            return False

        filtered = []
        for part in sys.path:
            if not part:
                continue
            if os.path.realpath(part) == os.path.realpath(overlay):
                filtered.append(part)
                continue
            if _is_source_checkout(part):
                continue
            filtered.append(part)
        sys.path = [overlay, *[p for p in filtered if os.path.realpath(p) != os.path.realpath(overlay)]]
        for name in list(sys.modules):
            if name == "ipfs_accelerate_py" or name.startswith("ipfs_accelerate_py."):
                del sys.modules[name]
            if name in sibling_names or name.startswith(
                (
                    "ipfs_datasets_py.",
                    "ipfs_kit_py.",
                    "ipfs_model_manager_py.",
                    "ipfs_transformers_py.",
                )
            ):
                del sys.modules[name]

        import ipfs_accelerate_py
        from ipfs_accelerate_py.assurance import clean_package as accel_clean

        sibling_hits = {}
        for name in sibling_names:
            try:
                imported = __import__(name)
                location = getattr(imported, "__file__", "") or ""
                sibling_hits[name] = location
            except Exception as exc:
                sibling_hits[name] = type(exc).__name__

        sibling_source_imported = any(
            _is_source_checkout(str(value)) for value in sibling_hits.values()
        )
        path_has_accel_root = os.path.realpath(accel_root) in {
            os.path.realpath(part) for part in sys.path if part
        }
        payload = {
            "import_ok": True,
            "version": getattr(ipfs_accelerate_py, "__version__", None),
            "clean_package_interface": getattr(accel_clean, "INTERFACE", None),
            "requires_sibling_repos": accel_clean.clean_package_manifest().get(
                "requires_sibling_source_trees"
            ),
            "sibling_hits": sibling_hits,
            "sibling_source_imported": sibling_source_imported,
            "path_has_accel_root": path_has_accel_root,
            "network_effects": [e for e in effects if e.get("kind") == "network"],
            "sys_path_injection": path_has_accel_root,
        }
        print("PCPR052::" + json.dumps(payload, sort_keys=True, default=str))
        """
    )
    try:
        os.symlink(
            root / "ipfs_accelerate_py",
            overlay / "ipfs_accelerate_py",
            target_is_directory=True,
        )
        env = {
            "PATH": SEALED_PATH,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(overlay),
            "PCPR052_OVERLAY": str(overlay),
            "PCPR052_ACCEL_ROOT": str(root),
            "IPFS_ACCEL_SKIP_CORE": "1",
            "HOME": os.environ.get(
                "HOME", "/tmp/ipfs-accelerate-validation-home-pcpr052"
            ),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        }
        completed = subprocess.run(
            [python, "-c", script],
            cwd=str(overlay),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        shutil.rmtree(overlay, ignore_errors=True)
        return {
            "status": "failed",
            "evidence_kind": "measured_hermetic",
            "import_ok": False,
            "sibling_source_imported": False,
            "reason": f"Isolated import subprocess failed: {exc}",
        }
    shutil.rmtree(overlay, ignore_errors=True)
    if completed.returncode != 0:
        return {
            "status": "failed",
            "evidence_kind": "measured_hermetic",
            "import_ok": False,
            "sibling_source_imported": False,
            "exit_code": completed.returncode,
            "reason": "Isolated import subprocess did not exit 0.",
            "stderr_tail": (completed.stderr or "")[-1000:],
        }
    line = next(
        (ln for ln in completed.stdout.splitlines() if ln.startswith("PCPR052::")),
        None,
    )
    if line is None:
        return {
            "status": "failed",
            "evidence_kind": "measured_hermetic",
            "import_ok": False,
            "sibling_source_imported": False,
            "reason": "Isolated import subprocess emitted no PCPR052 observation.",
        }
    observation = json.loads(line[len("PCPR052::") :])
    ok = (
        observation.get("import_ok") is True
        and observation.get("clean_package_interface") == INTERFACE
        and observation.get("requires_sibling_repos") is False
        and observation.get("sibling_source_imported") is False
        and observation.get("sys_path_injection") is False
        and not observation.get("network_effects")
    )
    sibling_hits = observation.get("sibling_hits") or {}
    return {
        "status": "observed",
        "evidence_kind": "measured_hermetic",
        "import_ok": bool(observation.get("import_ok")),
        "clean_package_interface": observation.get("clean_package_interface"),
        "requires_sibling_repos": observation.get("requires_sibling_repos"),
        "sibling_source_imported": bool(
            observation.get("sibling_source_imported")
        ),
        "sys_path_injection": bool(observation.get("sys_path_injection")),
        "network_effects": bool(observation.get("network_effects")),
        "version": observation.get("version"),
        "sibling_datasets": sibling_hits.get("ipfs_datasets_py"),
        "sibling_kit": sibling_hits.get("ipfs_kit_py"),
        "sibling_model_manager": sibling_hits.get("ipfs_model_manager_py"),
        "sibling_transformers": sibling_hits.get("ipfs_transformers_py"),
        "ok": ok,
        "reason": (
            "Isolated import succeeded without sibling source trees, sys.path "
            "injection, or network."
            if ok
            else "Isolated import still requires a sibling tree or mutates path/network."
        ),
    }


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sibling_member_hits(names: Sequence[str]) -> list[str]:
    markers = (
        "/ipfs_datasets_py/",
        "/ipfs_kit_py/",
        "/ipfs_model_manager_py/",
        "/ipfs_transformers_py/",
        "/docs/fastmcp/",
        "/docs/mcp-python-sdk/",
        "/test/doc-builder/",
        "/test/huggingface_doc_builder/",
        "/test/huggingface_transformers/",
    )
    prefix_markers = (
        "ipfs_datasets_py/",
        "ipfs_kit_py/",
        "ipfs_model_manager_py/",
        "ipfs_transformers_py/",
    )
    hits = []
    for name in names:
        normalized = name.replace("\\", "/")
        if any(normalized.startswith(marker) for marker in prefix_markers):
            hits.append(name)
            continue
        if any(marker in normalized for marker in markers):
            # Nested package path ipfs_accelerate_py/mcplusplus is pruned from
            # sdist; treat only sibling-root members as hits in wheels.
            if "ipfs_accelerate_py/mcplusplus" in normalized:
                continue
            hits.append(name)
    return hits


def inspect_wheel_members(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = sorted(archive.namelist())
    sibling_hits = _sibling_member_hits(names)
    package_prefix = f"{PACKAGE_NAME}/"
    return {
        "member_count": len(names),
        "has_package": any(
            name.startswith(package_prefix)
            or f".data/purelib/{package_prefix}" in name
            for name in names
        ),
        "has_clean_package_module": any(
            name.endswith("assurance/clean_package.py") for name in names
        ),
        "sibling_members": sibling_hits,
        "license_present": any(
            name.endswith("LICENSE") or name.endswith("LICENSE.txt")
            for name in names
        ),
        "platform_data_layout": any(".data/purelib/" in name for name in names),
    }


def inspect_sdist_members(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        names = sorted(member.name for member in archive.getmembers())
    sibling_hits = _sibling_member_hits(names)
    return {
        "member_count": len(names),
        "has_package": any(
            f"/{PACKAGE_NAME}/" in name or name.endswith(f"/{PACKAGE_NAME}")
            for name in names
        ),
        "has_license": any(name.endswith("LICENSE") for name in names),
        "has_clean_package_module": any(
            name.endswith("assurance/clean_package.py") for name in names
        ),
        "sibling_members": sibling_hits,
    }


def probe_hermetic_distribution_build(
    root: Path,
    *,
    python_executable: str | None = None,
    dist_dir: Path | None = None,
    build: bool = False,
) -> dict[str, Any]:
    """Optionally build wheel+sdist in a temp dir. Never a published release."""

    python = python_executable or (
        SEALED_PYTHON if Path(SEALED_PYTHON).is_file() else sys.executable
    )
    setuptools_origin = _sealed_python_module_origin("setuptools")
    wheel_origin = _sealed_python_module_origin("wheel")
    build_origin = _sealed_python_module_origin("build")
    tools = {
        "python": python if python else "unavailable",
        "setuptools": setuptools_origin,
        "wheel": wheel_origin,
        "build": build_origin,
    }
    if not build:
        return {
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "reason": (
                "Hermetic distribution build was not requested. Source-tree "
                "dist/ is not a live artifact."
            ),
            "tools": tools,
            "wheel": None,
            "sdist": None,
        }
    missing = [
        name
        for name, origin in (
            ("python", python),
            ("setuptools", setuptools_origin),
            ("wheel", wheel_origin),
            ("build", build_origin),
        )
        if not origin or origin == "unavailable"
    ]
    if missing:
        return {
            "status": "unavailable",
            "evidence_kind": "unavailable",
            "reason": (
                "Sealed-environment build tools are absent: "
                + ", ".join(missing)
            ),
            "tools": tools,
            "wheel": None,
            "sdist": None,
        }
    out_dir = dist_dir or Path(tempfile.mkdtemp(prefix="pcpr052-dist-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": SEALED_PATH,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "SOURCE_DATE_EPOCH": "0",
        "HOME": "/tmp/ipfs-accelerate-validation-home-pcpr052",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "IPFS_ACCELERATE_PY_SETUP_AUTO_TORCH": "0",
        "IPFS_ACCEL_SKIP_CORE": "1",
    }
    try:
        completed = subprocess.run(
            [
                python,
                "-m",
                "build",
                "--wheel",
                "--sdist",
                "--outdir",
                str(out_dir),
            ],
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "failed",
            "evidence_kind": "measured_hermetic",
            "reason": f"Hermetic build subprocess failed: {exc}",
            "tools": tools,
            "wheel": None,
            "sdist": None,
        }
    wheels = sorted(out_dir.glob("*.whl"))
    sdists = sorted(out_dir.glob("*.tar.gz"))
    if completed.returncode != 0 or not wheels or not sdists:
        return {
            "status": "failed",
            "evidence_kind": "measured_hermetic",
            "reason": "Hermetic build did not produce both wheel and sdist.",
            "exit_code": completed.returncode,
            "tools": tools,
            "wheel": None,
            "sdist": None,
            "stderr_tail": (completed.stderr or "")[-2000:],
        }
    wheel_path = wheels[0]
    sdist_path = sdists[0]
    members = inspect_wheel_members(wheel_path)
    sdist_members = inspect_sdist_members(sdist_path)
    return {
        "status": "observed",
        "evidence_kind": "measured_hermetic",
        "reason": (
            "Hermetic wheel and sdist were built in a private directory. "
            "They are not a published PCPR release."
        ),
        "tools": tools,
        "exit_code": 0,
        "wheel": {
            "name": wheel_path.name,
            "bytes": wheel_path.stat().st_size,
            "sha256": _sha256_file(wheel_path),
            "members": members,
            "path": str(wheel_path),
        },
        "sdist": {
            "name": sdist_path.name,
            "bytes": sdist_path.stat().st_size,
            "sha256": _sha256_file(sdist_path),
            "members": sdist_members,
            "path": str(sdist_path),
        },
        "sibling_members_in_wheel": members["sibling_members"],
        "sibling_members_in_sdist": sdist_members["sibling_members"],
        "published": False,
        "outdir": str(out_dir),
    }


def probe_hermetic_wheel_install(
    wheel_path: Path,
    *,
    python_executable: str | None = None,
    accelerate_root: Path | None = None,
) -> dict[str, Any]:
    """Install a private wheel with --no-deps --no-index. Not a release."""

    python = python_executable or (
        SEALED_PYTHON if Path(SEALED_PYTHON).is_file() else sys.executable
    )
    target = Path(tempfile.mkdtemp(prefix="pcpr052-install-"))
    env = {
        "PATH": SEALED_PATH,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/tmp/ipfs-accelerate-validation-home-pcpr052",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "IPFS_ACCEL_SKIP_CORE": "1",
    }
    try:
        completed = subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--no-compile",
                "--target",
                str(target),
                str(wheel_path),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        shutil.rmtree(target, ignore_errors=True)
        return {
            "status": "failed",
            "evidence_kind": "measured_hermetic",
            "import_ok": False,
            "reason": f"Hermetic pip install failed: {exc}",
        }
    if completed.returncode != 0:
        shutil.rmtree(target, ignore_errors=True)
        return {
            "status": "failed",
            "evidence_kind": "measured_hermetic",
            "import_ok": False,
            "exit_code": completed.returncode,
            "reason": "Hermetic pip install did not exit 0.",
            "stderr_tail": (completed.stderr or "")[-1000:],
        }
    script = textwrap.dedent(
        r"""
        import json, os, sys
        prefix = os.environ["PCPR052_PREFIX"]
        accel_root = os.environ["PCPR052_ACCEL_ROOT"]
        sys.path = [prefix, *[p for p in sys.path if os.path.realpath(p) != os.path.realpath(accel_root)]]
        import ipfs_accelerate_py
        from ipfs_accelerate_py.assurance.clean_package import INTERFACE, clean_package_manifest
        hits = {}
        for name in (
            "ipfs_datasets_py",
            "ipfs_kit_py",
            "ipfs_model_manager_py",
            "ipfs_transformers_py",
        ):
            try:
                imported = __import__(name)
                hits[name] = getattr(imported, "__file__", "") or ""
            except Exception as exc:
                hits[name] = type(exc).__name__
        imported_from = os.path.realpath(os.path.dirname(ipfs_accelerate_py.__file__))
        payload = {
            "import_ok": True,
            "version": getattr(ipfs_accelerate_py, "__version__", None),
            "imported_from_install_prefix": imported_from.startswith(os.path.realpath(prefix)),
            "imported_from_source_tree": imported_from.startswith(os.path.realpath(accel_root)),
            "clean_package_interface": INTERFACE,
            "requires_sibling_repos": clean_package_manifest()["requires_sibling_source_trees"],
            "ipfs_datasets_py": hits.get("ipfs_datasets_py"),
            "ipfs_kit_py": hits.get("ipfs_kit_py"),
            "ipfs_model_manager_py": hits.get("ipfs_model_manager_py"),
            "ipfs_transformers_py": hits.get("ipfs_transformers_py"),
        }
        print("PCPR052INSTALL::" + json.dumps(payload, sort_keys=True, default=str))
        """
    )
    accel_root = accelerate_root or discover_accelerate_root() or Path(".")
    try:
        imported = subprocess.run(
            [python, "-c", script],
            env={
                **env,
                "PCPR052_PREFIX": str(target),
                "PCPR052_ACCEL_ROOT": str(accel_root),
                "PYTHONPATH": str(target),
            },
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    finally:
        shutil.rmtree(target, ignore_errors=True)
    if imported.returncode != 0:
        return {
            "status": "failed",
            "evidence_kind": "measured_hermetic",
            "import_ok": False,
            "exit_code": imported.returncode,
            "reason": "Hermetic installed import did not exit 0.",
            "stderr_tail": (imported.stderr or "")[-1000:],
        }
    line = next(
        (
            ln
            for ln in imported.stdout.splitlines()
            if ln.startswith("PCPR052INSTALL::")
        ),
        None,
    )
    if line is None:
        return {
            "status": "failed",
            "evidence_kind": "measured_hermetic",
            "import_ok": False,
            "reason": "Hermetic installed import emitted no observation.",
        }
    observation = json.loads(line[len("PCPR052INSTALL::") :])
    return {
        "command": (
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin "
            "HOME=<ipfs-accelerate-validation-home-pcpr052-*> PYTHONNOUSERSITE=1 "
            f"{python} -m pip install --no-deps --no-index --no-compile "
            "--target <private-temp> <wheel>"
        ),
        "exit_code": 0,
        "import_ok": bool(observation.get("import_ok")),
        "imported_from_install_prefix": bool(
            observation.get("imported_from_install_prefix")
        ),
        "imported_from_source_tree": bool(
            observation.get("imported_from_source_tree")
        ),
        "requires_sibling_tests_tree": False,
        "ipfs_datasets_py": observation.get("ipfs_datasets_py"),
        "ipfs_kit_py": observation.get("ipfs_kit_py"),
        "ipfs_model_manager_py": observation.get("ipfs_model_manager_py"),
        "ipfs_transformers_py": observation.get("ipfs_transformers_py"),
        "clean_package_interface": observation.get("clean_package_interface"),
        "requires_sibling_repos": bool(observation.get("requires_sibling_repos")),
        "version": observation.get("version"),
        "evidence_kind": "measured_hermetic",
        "live": False,
        "published": False,
    }


def current_head_static_probes(
    start: Path | None = None,
    *,
    python_executable: str | None = None,
) -> tuple[OutcomeProbe, ...]:
    root = discover_accelerate_root(start)
    probes: list[OutcomeProbe] = []
    if root is None:
        missing = "accelerate source root was not found"
        for probe_id in sorted(REQUIRED_GOOD_PROBE_IDS):
            probes.append(_probe(probe_id, False, reason=missing))
        probes.extend(_terminal_negative_probes(root=None))
        return tuple(probes)

    try:
        setup_text = _read_text(root / "setup.py")
        pyproject_text = _read_text(root / "pyproject.toml")
        manifest_text = _read_text(root / "MANIFEST.in")
        init_text = _read_text(root / "ipfs_accelerate_py" / "__init__.py")
        requirements_text = _read_text(root / "requirements.txt")
        inner_req_path = root / "ipfs_accelerate_py" / "requirements.txt"
        inner_requirements_text = (
            _read_text(inner_req_path) if inner_req_path.is_file() else ""
        )
        setup_meta = parse_setup_metadata(setup_text)
        pyproject = parse_pyproject_clean_package(pyproject_text)
        prunes = parse_manifest_prunes(manifest_text)
        req_scan = scan_requirement_text(requirements_text)
        inner_scan = scan_requirement_text(inner_requirements_text)
        init_sys_path = package_init_injects_sys_path(init_text)
        optional = pyproject.get("optional_dependencies") or {}
        full_scan = scan_requirement_text("\n".join(optional.get("full") or []))
        all_scan = scan_requirement_text("\n".join(optional.get("all") or []))
        named_git = {
            name: scan_requirement_text("\n".join(optional.get(name) or []))
            for name in NAMED_GIT_EXTRAS
        }
    except (OSError, CleanPackageError, tomllib.TOMLDecodeError) as exc:
        for probe_id in sorted(REQUIRED_GOOD_PROBE_IDS):
            probes.append(_probe(probe_id, False, reason=str(exc)))
        probes.extend(_terminal_negative_probes(root=root))
        return tuple(probes)

    release_vcs = bool(req_scan["vcs"] or full_scan["vcs"] or all_scan["vcs"])
    release_editable = bool(
        req_scan["editable"] or full_scan["editable"] or all_scan["editable"]
    )
    probes.append(
        _probe(
            "release_requires_have_no_vcs",
            not req_scan["vcs"],
            reason=(
                "requirements.txt release requires contain no VCS markers."
                if not req_scan["vcs"]
                else "Release requires still contain VCS markers."
            ),
            details=req_scan,
        )
    )
    probes.append(
        _probe(
            "release_requires_have_no_editable",
            not req_scan["editable"],
            reason=(
                "requirements.txt release requires contain no editable markers."
                if not req_scan["editable"]
                else "Release requires still contain editable markers."
            ),
            details=req_scan,
        )
    )
    probes.append(
        _probe(
            "mutable_git_release_requires",
            release_vcs,
            reason=(
                "Mutable Git-branch markers are present in release requires."
                if release_vcs
                else "Release requires do not pin a mutable Git branch."
            ),
            details={"core": req_scan, "full": full_scan, "all": all_scan},
        )
    )
    probes.append(
        _probe(
            "editable_local_release_requires",
            release_editable,
            reason=(
                "Editable local markers are present in release requires."
                if release_editable
                else "Release requires do not use editable local paths."
            ),
            details={"core": req_scan, "full": full_scan, "all": all_scan},
        )
    )
    dynamic_ok = (
        pyproject.get("dynamic") == ["dependencies"]
        or (
            isinstance(pyproject.get("dynamic"), list)
            and "dependencies" in pyproject.get("dynamic")
        )
    ) and pyproject.get("dynamic_dependency_files") == ["requirements.txt"]
    probes.append(
        _probe(
            "pyproject_dynamic_dependencies_from_requirements",
            dynamic_ok and not req_scan["vcs"] and not req_scan["editable"],
            reason=(
                "pyproject.toml loads runtime dependencies dynamically from requirements.txt."
                if dynamic_ok and not req_scan["vcs"]
                else "pyproject.toml runtime dependency authority is missing or not PyPI-style."
            ),
            details={
                "dynamic": pyproject.get("dynamic"),
                "dynamic_dependency_files": pyproject.get(
                    "dynamic_dependency_files"
                ),
                "static_dependencies": pyproject.get("dependencies"),
            },
        )
    )
    python_requires_ok = (
        pyproject.get("requires_python") == CANONICAL_PYTHON_REQUIRES
        and setup_meta.get("python_requires_constant") == CANONICAL_PYTHON_REQUIRES
        and setup_meta.get("uses_python_requires_constant") is True
    )
    probes.append(
        _probe(
            "python_requires_agrees",
            python_requires_ok,
            reason=(
                "setup.py and pyproject.toml both require Python >=3.12."
                if python_requires_ok
                else "Python requires metadata does not agree on >=3.12."
            ),
            details={
                "setup": setup_meta.get("python_requires_constant"),
                "pyproject": pyproject.get("requires_python"),
            },
        )
    )
    pyproject_classifiers = tuple(pyproject.get("classifiers") or ())
    probes.append(
        _probe(
            "python_classifier_agrees",
            CANONICAL_PYTHON_CLASSIFIER in pyproject_classifiers
            and "Programming Language :: Python :: 3.10" not in pyproject_classifiers,
            reason=(
                "pyproject.toml advertises Python 3.12 and not the stale 3.10 classifier."
                if CANONICAL_PYTHON_CLASSIFIER in pyproject_classifiers
                else "pyproject.toml Python classifier does not match requires-python."
            ),
            details={"classifiers": list(pyproject_classifiers)},
        )
    )
    probes.append(
        _probe(
            "pyproject_python_classifier_agrees",
            CANONICAL_PYTHON_CLASSIFIER in pyproject_classifiers
            and CANONICAL_LICENSE_CLASSIFIER in pyproject_classifiers,
            reason=(
                "pyproject.toml classifiers agree on Python 3.12 and AGPL-3.0-or-later."
                if CANONICAL_LICENSE_CLASSIFIER in pyproject_classifiers
                else "pyproject.toml classifiers do not agree with the clean package."
            ),
            details={"classifiers": list(pyproject_classifiers)},
        )
    )
    find_exclude = tuple(pyproject.get("find_exclude") or ())
    find_include = tuple(pyproject.get("find_include") or ())
    package_data_keys = tuple(pyproject.get("package_data_keys") or ())
    sibling_excluded = (
        all(marker in find_exclude for marker in FIND_PACKAGE_EXCLUDES)
        and tuple(find_include) == FIND_PACKAGE_INCLUDES
        and not any(item.startswith("ipfs_datasets_py") for item in find_include)
        and not any(item.startswith("ipfs_kit_py") for item in find_include)
    )
    probes.append(
        _probe(
            "sibling_packages_excluded",
            sibling_excluded,
            reason=(
                "pyproject.toml package discovery includes only ipfs_accelerate_py "
                "and scripts, and excludes siblings."
                if sibling_excluded
                else "Sibling packages are still included in package discovery."
            ),
            details={
                "find_include": list(find_include),
                "find_exclude": list(find_exclude),
                "package_data_keys": list(package_data_keys),
            },
        )
    )
    sibling_prune_ok = all(line in prunes for line in MANIFEST_PRUNE_LINES[:4])
    probes.append(
        _probe(
            "sibling_submodules_pruned",
            sibling_prune_ok,
            reason=(
                "MANIFEST.in prunes sibling submodule trees from the sdist."
                if sibling_prune_ok
                else "MANIFEST.in does not prune sibling submodule trees."
            ),
            details={"prunes": list(prunes)},
        )
    )
    recursive_ok = all(
        f"prune {name}" in prunes for name in RECURSIVE_SUBMODULE_TREES
    )
    probes.append(
        _probe(
            "recursive_submodules_pruned",
            recursive_ok,
            reason=(
                "MANIFEST.in prunes recursive submodule trees from the sdist."
                if recursive_ok
                else "MANIFEST.in does not prune recursive submodule trees."
            ),
            details={"prunes": list(prunes)},
        )
    )
    table = pyproject.get("clean_package") or {}
    table_ok = (
        table.get("schema") == SCHEMA
        and table.get("interface") == INTERFACE
        and table.get("requires-sibling-source-trees") is False
        and table.get("mutable-git-release-requires") is False
        and table.get("sys-path-injection") is False
        and table.get("import-network") is False
        and table.get("requires-recursive-submodules") is False
    )
    probes.append(
        _probe(
            "pyproject_clean_package_table",
            table_ok,
            reason=(
                "pyproject.toml declares the AccelerateCleanPackage@1 constraints."
                if table_ok
                else "pyproject.toml clean-package table is missing or disagrees."
            ),
            details={"table": table},
        )
    )
    probes.append(
        _probe(
            "package_init_has_no_sys_path_injection",
            not init_sys_path,
            reason=(
                "ipfs_accelerate_py/__init__.py does not insert sibling paths."
                if not init_sys_path
                else "Package import still mutates sys.path."
            ),
        )
    )
    git_extras_ok = (
        not req_scan["vcs"]
        and not full_scan["vcs"]
        and not all_scan["vcs"]
        and all(named_git[name]["vcs"] for name in NAMED_GIT_EXTRAS)
    )
    probes.append(
        _probe(
            "git_extras_are_not_release_profile",
            git_extras_ok,
            reason=(
                "Named libp2p, mcp-p2p, ipfs-transformers, and ipfs-model-manager "
                "extras may carry source-checkout Git pins; they are not "
                "requirements.txt or the full/all extras."
                if git_extras_ok
                else "Git pins leaked into the release profile or named extras are missing."
            ),
            details={
                "named_git_extras": {
                    name: dict(scan) for name, scan in named_git.items()
                },
                "core_vcs": list(req_scan["vcs"]),
                "full_vcs": list(full_scan["vcs"]),
                "all_vcs": list(all_scan["vcs"]),
                "named_git_extras_role": NAMED_GIT_EXTRAS_ROLE,
            },
        )
    )
    probes.append(
        _probe(
            "full_extra_has_no_vcs",
            not full_scan["vcs"] and not full_scan["editable"],
            reason=(
                "The full extra is a convenience release extra without VCS or editable pins."
                if not full_scan["vcs"] and not full_scan["editable"]
                else "The full extra still contains VCS or editable pins."
            ),
            details=full_scan,
        )
    )
    probes.append(
        _probe(
            "all_extra_has_no_vcs",
            not all_scan["vcs"] and not all_scan["editable"],
            reason=(
                "The all extra is a convenience release extra without VCS or editable pins."
                if not all_scan["vcs"] and not all_scan["editable"]
                else "The all extra still contains VCS or editable pins."
            ),
            details=all_scan,
        )
    )
    inner_ok = bool(inner_scan["vcs"]) and not req_scan["vcs"]
    probes.append(
        _probe(
            "inner_requirements_git_is_not_release_profile",
            inner_ok,
            reason=(
                "ipfs_accelerate_py/requirements.txt may carry source-checkout Git pins; "
                "it is not install_requires or the pyproject dynamic release profile."
                if inner_ok
                else "Inner development requirements are missing Git pins or leaked into release requires."
            ),
            details={
                "inner_requirements": INNER_DEVELOPMENT_REQUIREMENTS,
                "inner_vcs": list(inner_scan["vcs"]),
                "release_vcs": list(req_scan["vcs"]),
            },
        )
    )
    setup_ok = (
        setup_meta.get("find_packages_uses_excludes") is True
        and setup_meta.get("find_excludes_constant") is True
        and setup_meta.get("reads_requirements_txt") is True
    )
    probes.append(
        _probe(
            "setup_find_packages_excludes_siblings",
            setup_ok,
            reason=(
                "setup.py find_packages excludes sibling trees and reads requirements.txt."
                if setup_ok
                else "setup.py still discovers sibling packages or does not read requirements.txt."
            ),
            details=setup_meta,
        )
    )

    manifest = clean_package_manifest()
    manifest_ok = (
        manifest["interface"] == INTERFACE
        and manifest["schema"] == SCHEMA
        and manifest["requires_sibling_source_trees"] is False
        and manifest["requires_recursive_submodules"] is False
        and manifest["mutable_git_release_requires"] is False
        and manifest["runtime_requires_authority"] == RUNTIME_REQUIRES_AUTHORITY
    )
    probes.append(
        _probe(
            "manifest_advertises_clean_package",
            manifest_ok,
            reason=(
                "AccelerateCleanPackage@1 manifest advertises sibling-free packaging "
                "and does not require sibling repositories."
                if manifest_ok
                else "AccelerateCleanPackage@1 manifest disagrees with the clean-package contract."
            ),
            details=manifest,
        )
    )

    isolated = probe_isolated_import_without_siblings(
        root,
        python_executable=python_executable,
    )
    probes.append(
        _probe(
            "isolated_import_without_siblings",
            isolated.get("ok") is True,
            evidence_kind=str(isolated.get("evidence_kind") or "measured_hermetic"),
            reason=str(isolated.get("reason") or "isolated import not observed"),
            details={
                "import_ok": isolated.get("import_ok"),
                "clean_package_interface": isolated.get("clean_package_interface"),
                "sibling_source_imported": isolated.get(
                    "sibling_source_imported"
                ),
                "sys_path_injection": isolated.get("sys_path_injection"),
                "network_effects": isolated.get("network_effects"),
                "version": isolated.get("version"),
                "sibling_datasets": isolated.get("sibling_datasets"),
                "sibling_kit": isolated.get("sibling_kit"),
                "sibling_model_manager": isolated.get("sibling_model_manager"),
                "sibling_transformers": isolated.get("sibling_transformers"),
            },
        )
    )
    probes.append(
        _probe(
            "sibling_source_imported",
            bool(isolated.get("sibling_source_imported")),
            evidence_kind=str(isolated.get("evidence_kind") or "measured_hermetic"),
            reason=(
                "Isolated import loaded a sibling source tree."
                if isolated.get("sibling_source_imported")
                else "Isolated import did not load a sibling source tree."
            ),
        )
    )
    probes.append(
        _probe(
            "sys_path_injection_observed",
            bool(isolated.get("sys_path_injection")),
            evidence_kind=str(isolated.get("evidence_kind") or "measured_hermetic"),
            reason=(
                "Isolated import injected the source checkout onto sys.path."
                if isolated.get("sys_path_injection")
                else "Isolated import did not inject the source checkout."
            ),
        )
    )

    probes.extend(_terminal_negative_probes(root=root))
    return tuple(probes)


def _terminal_negative_probes(*, root: Path | None) -> tuple[OutcomeProbe, ...]:
    dist_dir = (root / "dist") if root is not None else None
    wheels = ()
    sdists = ()
    if dist_dir is not None and dist_dir.is_dir():
        wheels = tuple(
            sorted(path.name for path in dist_dir.glob("*.whl") if path.is_file())
        )
        sdists = tuple(
            sorted(
                path.name
                for path in dist_dir.glob("*.tar.gz")
                if path.is_file()
            )
        )
    return (
        _probe(
            "simulated_results_represented_as_live",
            False,
            reason="Simulated results are not represented as live.",
        ),
        _probe(
            "runtime_unavailable_represented_as_live",
            False,
            reason="Clean-package probes are not represented as live.",
        ),
        _probe(
            "built_wheel_represented_as_live",
            False,
            reason="No built wheel is represented as a live published release.",
        ),
        _probe(
            "built_sdist_represented_as_live",
            False,
            reason="No built sdist is represented as a live published release.",
        ),
        _probe(
            "published_release_represented_as_live",
            False,
            reason="This task does not publish a PCPR release.",
        ),
        _probe(
            "built_wheel_metadata",
            None if not wheels else True,
            evidence_kind="unavailable" if not wheels else "measured",
            reason=(
                "No built wheel artifact is present in source-tree dist/; hermetic "
                "temp-dir builds are recorded separately and are not a release."
                if not wheels
                else "Source-tree dist/ contains a wheel name; it is not a published release."
            ),
            details={"artifacts": list(wheels)},
        ),
        _probe(
            "built_sdist_metadata",
            None if not sdists else True,
            evidence_kind="unavailable" if not sdists else "measured",
            reason=(
                "No built sdist artifact is present in source-tree dist/; hermetic "
                "temp-dir builds are recorded separately and are not a release."
                if not sdists
                else "Source-tree dist/ contains an sdist name; it is not a published release."
            ),
            details={"artifacts": list(sdists)},
        ),
        _probe(
            "live_compute_qualification",
            None,
            evidence_kind="unavailable",
            reason=(
                "This task does not qualify live Accelerate compute. Missing CPU/"
                "CUDA/model/provider evidence stays typed unavailable and is not "
                "recorded as False or passing."
            ),
        ),
    )


def qualify_clean_package(probes: Sequence[OutcomeProbe]) -> CleanPackageVerdict:
    if not probes:
        raise CleanPackageError("at least one probe is required")
    normalized: list[OutcomeProbe] = []
    blockers: list[str] = []
    for probe in probes:
        kind = _kind(probe.evidence_kind, "evidence_kind")
        if probe.live and kind != "measured_live":
            raise CleanPackageError("live claims require measured_live evidence")
        if probe.simulated_represented_as_live:
            raise CleanPackageError(
                "simulated results must not be represented as live"
            )
        normalized.append(probe)
        if probe.probe_id in {
            "live_compute_qualification",
            "built_wheel_metadata",
            "built_sdist_metadata",
        }:
            continue
        if probe.probe_id in FORBIDDEN_PRESENT_PROBE_IDS and probe.present is True:
            blockers.append(probe.probe_id)
        if probe.probe_id in REQUIRED_GOOD_PROBE_IDS and probe.present is not True:
            blockers.append(probe.probe_id)

    promotion_status = "rnd_non_promoted"
    _reject_closed_release_value(promotion_status, "promotion_status")
    wheel_kind = next(
        (
            item.evidence_kind
            for item in normalized
            if item.probe_id == "built_wheel_metadata"
        ),
        "unavailable",
    )
    sdist_kind = next(
        (
            item.evidence_kind
            for item in normalized
            if item.probe_id == "built_sdist_metadata"
        ),
        "unavailable",
    )
    isolated = next(
        (
            item.present is True
            for item in normalized
            if item.probe_id == "isolated_import_without_siblings"
        ),
        False,
    )
    sibling_required = not isolated
    git_requires = next(
        (
            item.present is True
            for item in normalized
            if item.probe_id == "mutable_git_release_requires"
        ),
        False,
    )
    payload = {
        "schema": VERDICT_SCHEMA,
        "interface": INTERFACE,
        "task_id": PCPR_052_TASK_ID,
        "goal_id": PCPR_052_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "sibling_source_required": sibling_required,
        "mutable_git_release_requires": git_requires,
        "isolated_import_without_siblings": isolated,
        "simulated_results_represented_as_live": False,
        "live_compute_qualified": False,
        "live_compute_evidence_kind": "unavailable",
        "built_wheel_evidence_kind": wheel_kind,
        "built_sdist_evidence_kind": sdist_kind,
        "this_task_created_competing_authority": False,
        "probes": [item.to_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
    }
    return CleanPackageVerdict(
        schema=VERDICT_SCHEMA,
        interface=INTERFACE,
        promotion_status=promotion_status,
        supervisor_disposition="supervisor_non_promoted",
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        contracts_frozen=False,
        duckdb_or_quack_state_written=False,
        sibling_source_required=sibling_required,
        mutable_git_release_requires=git_requires,
        isolated_import_without_siblings=isolated,
        simulated_results_represented_as_live=False,
        live_compute_qualified=False,
        live_compute_evidence_kind="unavailable",
        built_wheel_evidence_kind=wheel_kind,
        built_sdist_evidence_kind=sdist_kind,
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
    )


def qualify_current_head_clean_package(
    start: Path | None = None,
    *,
    python_executable: str | None = None,
) -> CleanPackageVerdict:
    return qualify_clean_package(
        current_head_static_probes(
            start, python_executable=python_executable
        )
    )


def pcpr_052_receipt_promotion(
    verdict: CleanPackageVerdict,
) -> dict[str, Any]:
    if verdict.closed_release_outcome is not None:
        raise CleanPackageError(
            "clean package must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise CleanPackageError("clean package must not claim a PCPR release")
    if verdict.completion_authoritative:
        raise CleanPackageError("clean package completion is not authoritative")
    if verdict.duckdb_or_quack_state_written:
        raise CleanPackageError(
            "clean package must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise CleanPackageError(
            "promotion_status must not be a closed release outcome"
        )
    return verdict.to_mapping()


# Pinned identity of the ordinary current-head static verdict. Drift means
# the default payload changed and the outer receipt must be regenerated.
CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeeraq44javupocw7ue4ih2gh32op2dv34uez2ijnnsqoutztklctr2da"
)


__all__ = [
    "CANONICAL_PYTHON_CLASSIFIER",
    "CANONICAL_PYTHON_REQUIRES",
    "CLOSED_RELEASE_OUTCOMES",
    "CURRENT_HEAD_NON_PROMOTION_VERDICT_CID",
    "CleanPackageAdmissionError",
    "CleanPackageError",
    "CleanPackageVerdict",
    "FIND_PACKAGE_EXCLUDES",
    "HERMETIC_CANDIDATE_SUITES",
    "INTERFACE",
    "MANIFEST_PRUNE_LINES",
    "NAMED_GIT_EXTRAS",
    "OutcomeProbe",
    "PCPR_052_GOAL_ID",
    "PCPR_052_TASK_ID",
    "RECURSIVE_SUBMODULE_TREES",
    "SCHEMA",
    "SEALED_PATH",
    "SEALED_PYTHON",
    "SIBLING_TREES",
    "clean_package_manifest",
    "content_identity",
    "current_head_static_probes",
    "discover_accelerate_root",
    "inspect_wheel_members",
    "observe_sealed_validation_environment",
    "parse_manifest_prunes",
    "parse_pyproject_clean_package",
    "parse_setup_metadata",
    "pcpr_052_receipt_promotion",
    "probe_hermetic_distribution_build",
    "probe_hermetic_wheel_install",
    "probe_isolated_import_without_siblings",
    "qualify_clean_package",
    "qualify_current_head_clean_package",
    "scan_requirement_text",
]
