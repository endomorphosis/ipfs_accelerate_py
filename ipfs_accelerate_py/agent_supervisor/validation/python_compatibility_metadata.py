"""Fail-closed PCPR-036 Python compatibility metadata alignment.

PCPR-036 aligns packaging metadata, classifiers, CI declarations, and
installer selection on the real Python 3.12 floor. Declared versions
must be versions this tree targets. This module is not release
authority: it does not write DuckDB or Quack state, does not run live
GitHub Actions, does not qualify CPU/CUDA, and never emits a closed
PCPR release outcome.

Live claims require live evidence. Simulated results stay Simulated.
Missing CI execution and older interpreters stay typed unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.assurance.python_compatibility import (
    CACHE_IMAGE_FROM_LINE,
    CI_PYTHON_VERSION_LINE,
    CI_WORKFLOW_RELPATHS,
    COMPATIBILITY_COMMAND,
    COMPATIBILITY_COMMAND_ENTRY,
    DECLARED_PYTHON_VERSIONS,
    FORBIDDEN_CLASSIFIERS,
    FORBIDDEN_PYTHON_VERSIONS,
    INTERFACE as FLOOR_INTERFACE,
    METADATA_RELPATHS,
    PYPROJECT_REQUIRES_PYTHON_LINE,
    PYTHON_FLOOR,
    REQUIRED_CLASSIFIER,
    REQUIRES_PYTHON,
    SCHEMA as FLOOR_SCHEMA,
    SETUP_PYTHON_REQUIRES,
    STABLE_COMMAND,
    STABLE_COMMAND_ENTRY,
    TASK_ID as FLOOR_TASK_ID,
    source_has_forbidden_classifiers,
    source_has_required_classifier,
)
from ..proof.formal_verification_contracts import content_identity
from .canonical_supervisor_contract_freeze import CLOSED_RELEASE_OUTCOMES
from .direct_objective_event_driven_qualification import (
    CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
    qualify_current_head_without_live_campaign,
)
from .legacy_mock_coordinator_quarantine import (
    discover_accelerate_root,
    discover_portfolio_root,
)
from .mutable_dependency_pinning import (
    CURRENT_HEAD_NON_PROMOTION_VERDICT_CID as PCPR_035_VERDICT_CID,
)
import re


COMPATIBILITY_EVALUATOR_INTERFACE: Final = "PythonCompatibilityMetadata@1"
COMPATIBILITY_EVALUATOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/python-compatibility-metadata@1"
)
COMPATIBILITY_VERDICT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/"
    "python-compatibility-metadata-verdict@1"
)

PCPR_036_TASK_ID: Final = "PCPR-036"
PCPR_036_GOAL_ID: Final = "PCPR-G420"
PCPR_035_TASK_ID: Final = "PCPR-035"
PCPR_001_TASK_ID: Final = "PCPR-001"
PCPR_PROGRAM_ID: Final = "proof-carrying-platform-qualification-and-release-v1"
PCPR_BOARD_NAMESPACE: Final = "proof-carrying-platform-qualification-and-release-v1"

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
PROMOTION_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "supervisor_promoted",
        "supervisor_non_promoted",
        "rnd_non_promoted",
        "typed_unavailable",
        "typed_blocked",
    }
)

CURRENT_HEAD_OUTER_COMMIT: Final = "1be2dcca07d57b59af5de920f031b2b4dc1fed82"
CURRENT_HEAD_OUTER_TREE: Final = "9118261961df95c7a2d019c73d96699a479b5213"
CURRENT_HEAD_OUTER_SUBJECT: Final = (
    "Merge commit '80217810f2b8915989080797d30325fd19b3685e' into "
    "agent/proof-carrying-platform-qualification-and-release-v1"
)
CURRENT_HEAD_ORIGIN_MAIN: Final = "bb8869ed72eb7002434345d9969efee729c4f7f6"
CURRENT_HEAD_ACCELERATOR_COMMIT: Final = (
    "9bdc03fb2aa10ee1f7a72f2f9c7efaa19bd08653"
)
CURRENT_HEAD_ACCELERATOR_TREE: Final = "10ccf93a671ba3b17cf08be8ea20285ddd4e8d54"
CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN: Final = (
    "f8c2f633fa6a781b822176fd63e1a229f96b581c"
)
CURRENT_HEAD_DATASETS_COMMIT: Final = "b913deb13341ff9fc68f586395d6beabf83d9f11"
CURRENT_HEAD_DATASETS_TREE: Final = "0c378d3e3dba73aaa2eb1848c88d50a87838c219"
CURRENT_HEAD_KIT_COMMIT: Final = "24510963ec43ffeaf3e088c4f215a6a9dbb5cbd5"
CURRENT_HEAD_KIT_TREE: Final = "c41ee7822def5c3c0518cc82e250a1d784a12fc1"

HERMETIC_CANDIDATE_SUITES: Final[tuple[str, ...]] = (
    "test/api/test_agent_supervisor_python_compatibility_metadata.py",
)

FLOOR_MODULE_RELPATH: Final = (
    "ipfs_accelerate_py/assurance/python_compatibility.py"
)
PYPROJECT_RELPATH: Final = "pyproject.toml"
SETUP_RELPATH: Final = "setup.py"
README_RELPATH: Final = "README.md"
INSTALL_DOC_RELPATH: Final = "docs/guides/getting-started/installation.md"
FAQ_RELPATH: Final = "docs/guides/troubleshooting/faq.md"
INSTALL_SH_RELPATH: Final = "install/install.sh"
INSTALL_PS1_RELPATH: Final = "install/install.ps1"
CACHE_IMAGE_RELPATH: Final = "install/Dockerfile.cache"

COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
STALE_FLOOR_NEEDLES: Final[tuple[str, ...]] = (
    'requires-python = ">=3.8"',
    'python_requires=">=3.8"',
    "Programming Language :: Python :: 3.8",
    "Python 3.8+",
    "python-3.8%2B",
)


class PythonCompatibilityMetadataError(ValueError):
    """Malformed compatibility evidence or a forbidden authority claim."""


@dataclass(frozen=True)
class CompatibilityProbe:
    """One measured or typed-unavailable compatibility observation."""

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
class CompatibilityVerdict:
    """Fail-closed PCPR-036 decision. Never a closed release."""

    schema: str
    interface: str
    promotion_status: str
    supervisor_disposition: str
    closed_release_outcome: None
    release_claim: bool
    completion_authoritative: bool
    contracts_frozen: bool
    duckdb_or_quack_state_written: bool
    declared_python_floor: str
    requires_python: str
    declared_python_versions: tuple[str, ...]
    metadata_agrees_on_floor: bool
    ci_declares_same_floor: bool
    commands_retain_explicit_migration: bool
    simulated_results_represented_as_live: bool
    live_ci_matrix_executed: bool
    live_ci_matrix_evidence_kind: str
    live_older_python_qualified: bool
    live_older_python_evidence_kind: str
    this_task_created_competing_authority: bool
    probes: tuple[CompatibilityProbe, ...]
    blockers: tuple[str, ...]
    verdict_cid: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "promotion_status": self.promotion_status,
            "supervisor_disposition": self.supervisor_disposition,
            "closed_release_outcome": self.closed_release_outcome,
            "release_claim": self.release_claim,
            "completion_authoritative": self.completion_authoritative,
            "contracts_frozen": self.contracts_frozen,
            "duckdb_or_quack_state_written": self.duckdb_or_quack_state_written,
            "declared_python_floor": self.declared_python_floor,
            "requires_python": self.requires_python,
            "declared_python_versions": list(self.declared_python_versions),
            "metadata_agrees_on_floor": self.metadata_agrees_on_floor,
            "ci_declares_same_floor": self.ci_declares_same_floor,
            "commands_retain_explicit_migration": (
                self.commands_retain_explicit_migration
            ),
            "simulated_results_represented_as_live": (
                self.simulated_results_represented_as_live
            ),
            "live_ci_matrix_executed": self.live_ci_matrix_executed,
            "live_ci_matrix_evidence_kind": self.live_ci_matrix_evidence_kind,
            "live_older_python_qualified": self.live_older_python_qualified,
            "live_older_python_evidence_kind": (
                self.live_older_python_evidence_kind
            ),
            "this_task_created_competing_authority": (
                self.this_task_created_competing_authority
            ),
            "probes": [item.to_mapping() for item in self.probes],
            "blockers": list(self.blockers),
            "verdict_cid": self.verdict_cid,
        }


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PythonCompatibilityMetadataError(f"{name} must be a non-empty string")
    return value.strip()


def _kind(value: Any, name: str) -> str:
    kind = _text(value, name)
    if kind not in EVIDENCE_KINDS:
        raise PythonCompatibilityMetadataError(
            f"{name} is not an admitted evidence kind"
        )
    return kind


def _git_object_id(value: Any, name: str) -> str:
    text = _text(value, name)
    if COMMIT_RE.fullmatch(text) is None:
        raise PythonCompatibilityMetadataError(
            f"{name} must be a 40-character lowercase git object id"
        )
    return text


def _reject_closed_release_value(value: Any, name: str) -> None:
    if isinstance(value, str) and value in CLOSED_RELEASE_OUTCOMES:
        raise PythonCompatibilityMetadataError(
            f"{name} must not be a closed PCPR release outcome"
        )


def _require_ancestor(flag: bool, name: str) -> None:
    if flag is not True:
        raise PythonCompatibilityMetadataError(f"{name} must be true")


def _read_source(root: Path, relpath: str) -> str | None:
    path = root / relpath
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _probe(
    probe_id: str,
    *,
    present: bool | None,
    evidence_kind: str,
    reason: str,
    details: Mapping[str, Any] | None = None,
) -> CompatibilityProbe:
    return CompatibilityProbe(
        probe_id=probe_id,
        present=present,
        evidence_kind=evidence_kind,
        live=False,
        simulated_represented_as_live=False,
        reason=reason,
        details=MappingProxyType(dict(details or {})),
    )


def _file_probe(
    probe_id: str,
    source: str | None,
    relpath: str,
    ok: bool,
    ok_reason: str,
    fail_reason: str,
) -> CompatibilityProbe:
    if source is None:
        return _probe(
            probe_id,
            present=None,
            evidence_kind="unavailable",
            reason=f"{relpath} is not present and is not recorded as empty.",
            details={"relpath": relpath},
        )
    return _probe(
        probe_id,
        present=ok,
        evidence_kind="measured",
        reason=ok_reason if ok else fail_reason,
        details={"relpath": relpath},
    )


def current_head_compatibility_probes(
    *,
    accelerate_root: Path | None = None,
    portfolio_root: Path | None = None,
) -> tuple[CompatibilityProbe, ...]:
    """Measured current-tree probes. Missing files stay typed unavailable."""

    del portfolio_root  # reserved for gitlink binding; unused by these probes
    root = accelerate_root or discover_accelerate_root()
    if root is None or not root.is_dir():
        return (
            _probe(
                "accelerate_source_tree",
                present=None,
                evidence_kind="unavailable",
                reason=(
                    "Accelerate source tree is not present and is not recorded as empty."
                ),
            ),
        )

    probes: list[CompatibilityProbe] = []
    floor_src = _read_source(root, FLOOR_MODULE_RELPATH)
    floor_ok = bool(floor_src) and (
        PYTHON_FLOOR == "3.12"
        and REQUIRES_PYTHON == ">=3.12"
        and DECLARED_PYTHON_VERSIONS == ("3.12",)
        and FORBIDDEN_PYTHON_VERSIONS == ("3.8", "3.9", "3.10", "3.11")
        and STABLE_COMMAND == "ipfs-accelerate"
        and COMPATIBILITY_COMMAND == "ipfs_accelerate"
    )
    probes.append(
        _file_probe(
            "canonical_floor_table",
            floor_src,
            FLOOR_MODULE_RELPATH,
            floor_ok,
            "Canonical floor table declares Python 3.12 only, with command migration.",
            "Canonical floor table is missing or still advertises an untested floor.",
        )
    )

    pyproject = _read_source(root, PYPROJECT_RELPATH)
    pyproject_req_ok = bool(pyproject) and (
        PYPROJECT_REQUIRES_PYTHON_LINE in (pyproject or "")
        and 'requires-python = ">=3.8"' not in (pyproject or "")
    )
    probes.append(
        _file_probe(
            "pyproject_requires_python",
            pyproject,
            PYPROJECT_RELPATH,
            pyproject_req_ok,
            "pyproject.toml requires-python is >=3.12 and no longer advertises >=3.8.",
            "pyproject.toml still advertises an untested Python floor.",
        )
    )
    pyproject_clf_ok = bool(pyproject) and source_has_required_classifier(
        pyproject or ""
    ) and not source_has_forbidden_classifiers(pyproject or "")
    probes.append(
        _file_probe(
            "pyproject_classifiers",
            pyproject,
            PYPROJECT_RELPATH,
            pyproject_clf_ok,
            "pyproject classifiers list Python 3.12 and omit 3.8-3.11.",
            "pyproject classifiers still declare an untested Python version.",
        )
    )
    preflight_ok = bool(pyproject) and (
        '[tool.ipfs-accelerate-agent-supervisor.project-dependency-preflight]'
        in (pyproject or "")
        and PYPROJECT_REQUIRES_PYTHON_LINE in (pyproject or "")
    )
    probes.append(
        _file_probe(
            "preflight_requires_python",
            pyproject,
            PYPROJECT_RELPATH,
            preflight_ok,
            "Project-dependency preflight requires-python agrees on >=3.12.",
            "Project-dependency preflight still disagrees with the Python floor.",
        )
    )
    scripts_ok = bool(pyproject) and (
        f'{STABLE_COMMAND} = "{STABLE_COMMAND_ENTRY}"' in (pyproject or "")
        and f'{COMPATIBILITY_COMMAND} = "{COMPATIBILITY_COMMAND_ENTRY}"'
        in (pyproject or "")
    )
    probes.append(
        _file_probe(
            "command_hierarchy_retains_compatibility_migration",
            pyproject,
            PYPROJECT_RELPATH,
            scripts_ok,
            "ipfs-accelerate remains the stable CLI; ipfs_accelerate is retained as a compatibility alias.",
            "Stable CLI or compatibility alias is missing from packaging scripts.",
        )
    )

    setup_src = _read_source(root, SETUP_RELPATH)
    setup_req_ok = bool(setup_src) and SETUP_PYTHON_REQUIRES in (setup_src or "")
    probes.append(
        _file_probe(
            "setup_python_requires",
            setup_src,
            SETUP_RELPATH,
            setup_req_ok,
            "setup.py python_requires is >=3.12.",
            "setup.py still advertises an untested Python floor.",
        )
    )
    setup_clf_ok = bool(setup_src) and source_has_required_classifier(
        setup_src or ""
    ) and not source_has_forbidden_classifiers(setup_src or "")
    probes.append(
        _file_probe(
            "setup_classifiers",
            setup_src,
            SETUP_RELPATH,
            setup_clf_ok,
            "setup.py classifiers list Python 3.12 and omit 3.8-3.11.",
            "setup.py classifiers still declare an untested Python version.",
        )
    )

    ci_missing: list[str] = []
    ci_ok_paths: list[str] = []
    ci_bad: list[str] = []
    for relpath in CI_WORKFLOW_RELPATHS:
        source = _read_source(root, relpath)
        if source is None:
            ci_missing.append(relpath)
            continue
        if CI_PYTHON_VERSION_LINE in source and "python-version: [3.8" not in source:
            ci_ok_paths.append(relpath)
        else:
            ci_bad.append(relpath)
    if ci_missing and not ci_ok_paths and not ci_bad:
        probes.append(
            _probe(
                "ci_workflows_declare_python_312",
                present=None,
                evidence_kind="unavailable",
                reason="CI workflow files are not present and are not recorded as empty.",
                details={"relpaths": list(CI_WORKFLOW_RELPATHS)},
            )
        )
    else:
        ci_ok = bool(ci_ok_paths) and not ci_missing and not ci_bad
        probes.append(
            _probe(
                "ci_workflows_declare_python_312",
                present=ci_ok,
                evidence_kind="measured",
                reason=(
                    "amd64, arm64, and multiarch workflows declare PYTHON_VERSION 3.12."
                    if ci_ok
                    else "A declared CI workflow still advertises an untested Python version."
                ),
                details={
                    "ok": ci_ok_paths,
                    "missing": ci_missing,
                    "disagree": ci_bad,
                },
            )
        )

    cache_src = _read_source(root, CACHE_IMAGE_RELPATH)
    cache_ok = bool(cache_src) and CACHE_IMAGE_FROM_LINE in (cache_src or "") and (
        "FROM python:3.11-slim" not in (cache_src or "")
    )
    probes.append(
        _file_probe(
            "cache_image_declares_python_312",
            cache_src,
            CACHE_IMAGE_RELPATH,
            cache_ok,
            "Cache image FROM line uses python:3.12-slim.",
            "Cache image still uses an undeclared Python base.",
        )
    )

    install_sh = _read_source(root, INSTALL_SH_RELPATH)
    install_sh_ok = bool(install_sh) and (
        "Python 3.12+" in (install_sh or "")
        and "python3.8" not in (install_sh or "")
        and "python3.9" not in (install_sh or "")
        and "python3.11 python3.10 python3.9" not in (install_sh or "")
    )
    probes.append(
        _file_probe(
            "installer_requires_python_312",
            install_sh,
            INSTALL_SH_RELPATH,
            install_sh_ok,
            "Unix installer selects Python 3.12+ and rejects the old 3.8 floor.",
            "Unix installer still accepts an untested Python interpreter.",
        )
    )
    install_ps1 = _read_source(root, INSTALL_PS1_RELPATH)
    install_ps1_ok = bool(install_ps1) and (
        "Python 3.12+" in (install_ps1 or "")
        and "python3.8" not in (install_ps1 or "")
        and "python3.11" not in (install_ps1 or "")
    )
    probes.append(
        _file_probe(
            "windows_installer_requires_python_312",
            install_ps1,
            INSTALL_PS1_RELPATH,
            install_ps1_ok,
            "Windows installer selects Python 3.12+ and rejects the old 3.8 floor.",
            "Windows installer still accepts an untested Python interpreter.",
        )
    )

    readme = _read_source(root, README_RELPATH)
    readme_ok = bool(readme) and "python-3.12%2B" in (readme or "") and (
        "python-3.8%2B" not in (readme or "")
    )
    probes.append(
        _file_probe(
            "readme_declares_python_312",
            readme,
            README_RELPATH,
            readme_ok,
            "README badge declares Python 3.12+.",
            "README still advertises Python 3.8+.",
        )
    )
    install_doc = _read_source(root, INSTALL_DOC_RELPATH)
    install_doc_ok = bool(install_doc) and (
        PYPROJECT_REQUIRES_PYTHON_LINE in (install_doc or "")
        and "Python 3.12 or newer" in (install_doc or "")
        and STABLE_COMMAND in (install_doc or "")
        and COMPATIBILITY_COMMAND in (install_doc or "")
        and "documented compatibility alias" in (install_doc or "")
    )
    probes.append(
        _file_probe(
            "installation_docs_match_floor",
            install_doc,
            INSTALL_DOC_RELPATH,
            install_doc_ok,
            "Installation guide matches requires-python >=3.12 and retains command migration.",
            "Installation guide still disagrees with the Python floor or drops command migration.",
        )
    )
    faq = _read_source(root, FAQ_RELPATH)
    faq_ok = bool(faq) and "Python 3.12 or" in (faq or "") and (
        "declares **Python 3.8 or" not in (faq or "")
    )
    probes.append(
        _file_probe(
            "faq_matches_floor",
            faq,
            FAQ_RELPATH,
            faq_ok,
            "FAQ no longer treats Python 3.8 as declared packaging metadata.",
            "FAQ still documents the 3.8 packaging floor as current.",
        )
    )

    stale_hits: list[dict[str, str]] = []
    for relpath in (*METADATA_RELPATHS, *CI_WORKFLOW_RELPATHS):
        source = _read_source(root, relpath)
        if source is None:
            continue
        for needle in STALE_FLOOR_NEEDLES:
            if needle in source:
                stale_hits.append({"relpath": relpath, "needle": needle})
    probes.append(
        _probe(
            "declared_versions_only_3_12",
            present=not stale_hits,
            evidence_kind="measured",
            reason=(
                "Qualified metadata, docs, installer, and CI declarations no longer advertise 3.8-3.11."
                if not stale_hits
                else "A qualified metadata surface still advertises an untested Python version."
            ),
            details={"stale_hits": stale_hits, "declared": list(DECLARED_PYTHON_VERSIONS)},
        )
    )

    probes.append(
        _probe(
            "live_ci_matrix_execution",
            present=None,
            evidence_kind="unavailable",
            reason=(
                "This task does not execute GitHub Actions. CI files declare "
                "Python 3.12; live matrix execution stays typed unavailable "
                "and is not recorded as False or passing."
            ),
        )
    )
    probes.append(
        _probe(
            "live_older_python_qualification",
            present=None,
            evidence_kind="unavailable",
            reason=(
                "Python 3.8-3.11 are not declared. Missing older-interpreter "
                "evidence stays typed unavailable and is not recorded as False "
                "or passing. This is not live CPU, CUDA, or provider qualification."
            ),
            details={"undeclared": list(FORBIDDEN_PYTHON_VERSIONS)},
        )
    )
    return tuple(probes)


def qualify_python_compatibility_metadata(
    *,
    probes: Sequence[CompatibilityProbe],
    duckdb_or_quack_state_written: bool = False,
    simulated_results_represented_as_live: bool = False,
    live_ci_matrix_executed: bool = False,
    live_older_python_qualified: bool = False,
    this_task_created_competing_authority: bool = False,
) -> CompatibilityVerdict:
    """Fail-closed compatibility verdict. Never a closed release."""

    if duckdb_or_quack_state_written:
        raise PythonCompatibilityMetadataError(
            "python compatibility metadata must not write DuckDB or Quack state"
        )
    if simulated_results_represented_as_live:
        raise PythonCompatibilityMetadataError(
            "simulated compatibility results cannot be represented as live"
        )
    if live_ci_matrix_executed or live_older_python_qualified:
        raise PythonCompatibilityMetadataError(
            "this task cannot claim live CI execution or older-Python qualification"
        )
    if this_task_created_competing_authority:
        raise PythonCompatibilityMetadataError(
            "python compatibility metadata cannot mint a competing authority"
        )

    normalized: list[CompatibilityProbe] = []
    for item in probes:
        if not isinstance(item, CompatibilityProbe):
            raise PythonCompatibilityMetadataError(
                "probes must be CompatibilityProbe values"
            )
        _kind(item.evidence_kind, f"{item.probe_id}.evidence_kind")
        if item.live:
            raise PythonCompatibilityMetadataError(
                f"{item.probe_id} cannot claim live evidence"
            )
        if item.simulated_represented_as_live:
            raise PythonCompatibilityMetadataError(
                f"{item.probe_id} cannot represent simulation as live"
            )
        if item.evidence_kind in {"estimated", "simulated", "measured_live"}:
            raise PythonCompatibilityMetadataError(
                f"{item.probe_id} evidence_kind {item.evidence_kind} is not admitted here"
            )
        normalized.append(item)

    by_id = {item.probe_id: item for item in normalized}
    blockers: list[str] = []

    def _require(probe_id: str, blocker: str) -> bool:
        item = by_id.get(probe_id)
        ok = bool(item and item.present is True)
        if not ok:
            blockers.append(blocker)
        return ok

    table_ok = _require("canonical_floor_table", "floor_table_incomplete")
    req_ok = _require("pyproject_requires_python", "pyproject_requires_python_drift")
    clf_ok = _require("pyproject_classifiers", "pyproject_classifier_drift")
    preflight_ok = _require("preflight_requires_python", "preflight_requires_python_drift")
    setup_req_ok = _require("setup_python_requires", "setup_python_requires_drift")
    setup_clf_ok = _require("setup_classifiers", "setup_classifier_drift")
    ci_ok = _require("ci_workflows_declare_python_312", "ci_python_version_drift")
    cache_ok = _require("cache_image_declares_python_312", "cache_image_python_drift")
    sh_ok = _require("installer_requires_python_312", "unix_installer_python_drift")
    ps1_ok = _require(
        "windows_installer_requires_python_312", "windows_installer_python_drift"
    )
    readme_ok = _require("readme_declares_python_312", "readme_python_badge_drift")
    docs_ok = _require("installation_docs_match_floor", "installation_doc_drift")
    faq_ok = _require("faq_matches_floor", "faq_python_floor_drift")
    only_ok = _require("declared_versions_only_3_12", "undeclared_python_version_advertised")
    commands_ok = _require(
        "command_hierarchy_retains_compatibility_migration",
        "command_migration_missing",
    )

    metadata_ok = (
        table_ok
        and req_ok
        and clf_ok
        and preflight_ok
        and setup_req_ok
        and setup_clf_ok
        and readme_ok
        and docs_ok
        and faq_ok
        and only_ok
        and sh_ok
        and ps1_ok
        and cache_ok
    )
    ci_declares = ci_ok

    unavailable_count = sum(
        1 for item in normalized if item.evidence_kind == "unavailable"
    )
    if blockers:
        promotion_status = "typed_blocked"
    elif unavailable_count == len(normalized):
        promotion_status = "typed_unavailable"
    else:
        promotion_status = "rnd_non_promoted"

    if promotion_status not in PROMOTION_STATUSES:
        raise PythonCompatibilityMetadataError(
            "internal promotion status is not admitted"
        )
    if promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise PythonCompatibilityMetadataError(
            "python compatibility metadata must not mint a closed release outcome"
        )

    payload = {
        "schema": COMPATIBILITY_VERDICT_SCHEMA,
        "interface": COMPATIBILITY_EVALUATOR_INTERFACE,
        "task_id": PCPR_036_TASK_ID,
        "goal_id": PCPR_036_GOAL_ID,
        "promotion_status": promotion_status,
        "supervisor_disposition": "supervisor_non_promoted",
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "declared_python_floor": PYTHON_FLOOR,
        "requires_python": REQUIRES_PYTHON,
        "declared_python_versions": list(DECLARED_PYTHON_VERSIONS),
        "metadata_agrees_on_floor": metadata_ok,
        "ci_declares_same_floor": ci_declares,
        "commands_retain_explicit_migration": commands_ok,
        "simulated_results_represented_as_live": False,
        "live_ci_matrix_executed": False,
        "live_ci_matrix_evidence_kind": "unavailable",
        "live_older_python_qualified": False,
        "live_older_python_evidence_kind": "unavailable",
        "this_task_created_competing_authority": False,
        "probes": [item.to_mapping() for item in normalized],
        "blockers": list(dict.fromkeys(blockers)),
    }
    return CompatibilityVerdict(
        schema=COMPATIBILITY_VERDICT_SCHEMA,
        interface=COMPATIBILITY_EVALUATOR_INTERFACE,
        promotion_status=promotion_status,
        supervisor_disposition="supervisor_non_promoted",
        closed_release_outcome=None,
        release_claim=False,
        completion_authoritative=False,
        contracts_frozen=False,
        duckdb_or_quack_state_written=False,
        declared_python_floor=PYTHON_FLOOR,
        requires_python=REQUIRES_PYTHON,
        declared_python_versions=DECLARED_PYTHON_VERSIONS,
        metadata_agrees_on_floor=metadata_ok,
        ci_declares_same_floor=ci_declares,
        commands_retain_explicit_migration=commands_ok,
        simulated_results_represented_as_live=False,
        live_ci_matrix_executed=False,
        live_ci_matrix_evidence_kind="unavailable",
        live_older_python_qualified=False,
        live_older_python_evidence_kind="unavailable",
        this_task_created_competing_authority=False,
        probes=tuple(normalized),
        blockers=tuple(dict.fromkeys(blockers)),
        verdict_cid=content_identity(payload),
    )


def qualify_current_head_compatibility() -> CompatibilityVerdict:
    """Ordinary current-head PCPR-036 evaluation: honest R&D metadata alignment."""

    return qualify_python_compatibility_metadata(
        probes=current_head_compatibility_probes()
    )


CURRENT_HEAD_NON_PROMOTION_VERDICT_CID: Final = (
    "baguqeera7faex76px5ynmgrqgg27m36x6lbepssz5gom3t45pfkd6h4bdsaa"
)


def pcpr_036_receipt_promotion(verdict: CompatibilityVerdict) -> dict[str, Any]:
    """Compact outer-receipt promotion fields. Never a closed release."""

    if verdict.closed_release_outcome is not None:
        raise PythonCompatibilityMetadataError(
            "python compatibility metadata must not mint a closed release outcome"
        )
    if verdict.release_claim:
        raise PythonCompatibilityMetadataError(
            "python compatibility metadata must not claim a PCPR release"
        )
    if verdict.completion_authoritative:
        raise PythonCompatibilityMetadataError(
            "python compatibility metadata completion is not authoritative"
        )
    if verdict.duckdb_or_quack_state_written:
        raise PythonCompatibilityMetadataError(
            "python compatibility metadata must not write DuckDB or Quack state"
        )
    if verdict.promotion_status in CLOSED_RELEASE_OUTCOMES:
        raise PythonCompatibilityMetadataError(
            "promotion_status must not be a closed release outcome"
        )
    if verdict.promotion_status not in PROMOTION_STATUSES:
        raise PythonCompatibilityMetadataError(
            "promotion_status is not an admitted PCPR-036 status"
        )
    if verdict.promotion_status == "supervisor_promoted":
        raise PythonCompatibilityMetadataError(
            "python compatibility metadata cannot promote the supervisor"
        )
    if verdict.live_ci_matrix_executed or verdict.live_older_python_qualified:
        raise PythonCompatibilityMetadataError(
            "this task cannot claim live CI execution or older-Python qualification"
        )
    if not verdict.metadata_agrees_on_floor:
        raise PythonCompatibilityMetadataError(
            "metadata must agree on the declared Python 3.12 floor"
        )
    if not verdict.commands_retain_explicit_migration:
        raise PythonCompatibilityMetadataError(
            "commands must retain explicit compatibility migration"
        )
    return {
        "schema": COMPATIBILITY_VERDICT_SCHEMA,
        "interface": COMPATIBILITY_EVALUATOR_INTERFACE,
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": verdict.promotion_status,
        "supervisor_disposition": verdict.supervisor_disposition,
        "closed_release_outcome": None,
        "release_claim": False,
        "completion_authoritative": False,
        "contracts_frozen": False,
        "duckdb_or_quack_state_written": False,
        "declared_python_floor": verdict.declared_python_floor,
        "requires_python": verdict.requires_python,
        "declared_python_versions": list(verdict.declared_python_versions),
        "metadata_agrees_on_floor": True,
        "ci_declares_same_floor": verdict.ci_declares_same_floor,
        "commands_retain_explicit_migration": True,
        "simulated_results_represented_as_live": False,
        "live_ci_matrix_executed": False,
        "live_ci_matrix_evidence_kind": "unavailable",
        "live_older_python_qualified": False,
        "live_older_python_evidence_kind": "unavailable",
        "this_task_created_competing_authority": (
            verdict.this_task_created_competing_authority
        ),
        "blocker_count": len(verdict.blockers),
        "blockers": list(verdict.blockers),
        "evidence_kind": "measured",
    }


def current_head_pcpr_036_receipt_promotion() -> dict[str, Any]:
    return pcpr_036_receipt_promotion(qualify_current_head_compatibility())


def pcpr_036_receipt_negative_results() -> dict[str, Any]:
    return {
        "simulated_presence_cannot_count_as_live": True,
        "simulated_results_not_represented_as_live": True,
        "estimated_values_rejected": True,
        "hermetic_pass_cannot_qualify_live": True,
        "closed_release_outcome_not_emitted": True,
        "direct_database_bypass_not_used": True,
        "live_ci_matrix_not_claimed": True,
        "live_older_python_not_claimed": True,
        "untested_python_version_not_declared": True,
        "compatibility_command_not_removed": True,
        "evidence_kind": "measured",
    }


def current_head_pcpr_036_receipt_sections() -> dict[str, Any]:
    verdict = qualify_current_head_compatibility()
    promotion = pcpr_036_receipt_promotion(verdict)
    qualification = qualify_current_head_without_live_campaign()
    return {
        "qualification_verdict": promotion,
        "qualification_prerequisite": {
            "task_id": PCPR_035_TASK_ID,
            "goal_id": "PCPR-G420",
            "promotion_status": "rnd_non_promoted",
            "pinning_verdict_cid": PCPR_035_VERDICT_CID,
            "supervisor_live_qualification_task_id": PCPR_001_TASK_ID,
            "supervisor_live_qualification_promotion_status": qualification.promotion_status,
            "live_qualification_evidence_kind": "unavailable",
            "verdict_cid": CURRENT_HEAD_UNAVAILABLE_VERDICT_CID,
            "closed_release_outcome": None,
            "release_claim": False,
            "completion_authoritative": False,
            "reason": (
                "PCPR-035 pinned mutable Git dependencies. This task aligns "
                "Python compatibility metadata on the 3.12 floor. PCPR-001 live "
                "qualification remains rnd_non_promoted and is not promoted by "
                "this receipt."
            ),
            "evidence_kind": "measured",
        },
        "compatibility": {
            "schema": COMPATIBILITY_EVALUATOR_SCHEMA,
            "interface": COMPATIBILITY_EVALUATOR_INTERFACE,
            "floor_schema": FLOOR_SCHEMA,
            "floor_interface": FLOOR_INTERFACE,
            "floor_task_id": FLOOR_TASK_ID,
            "declared_python_floor": PYTHON_FLOOR,
            "requires_python": REQUIRES_PYTHON,
            "declared_python_versions": list(DECLARED_PYTHON_VERSIONS),
            "forbidden_python_versions": list(FORBIDDEN_PYTHON_VERSIONS),
            "required_classifier": REQUIRED_CLASSIFIER,
            "forbidden_classifiers": list(FORBIDDEN_CLASSIFIERS),
            "stable_command": STABLE_COMMAND,
            "compatibility_command": COMPATIBILITY_COMMAND,
            "metadata_agrees_on_floor": verdict.metadata_agrees_on_floor,
            "ci_declares_same_floor": verdict.ci_declares_same_floor,
            "commands_retain_explicit_migration": (
                verdict.commands_retain_explicit_migration
            ),
            "probes": [item.to_mapping() for item in verdict.probes],
            "evidence_kind": "measured",
        },
        "negative_results": pcpr_036_receipt_negative_results(),
        "verdict_cid": verdict.verdict_cid,
        "promotion_status": promotion["promotion_status"],
        "closed_release_outcome": None,
        "release_claim": False,
        "contracts_frozen": False,
    }


def pcpr_036_current_tree_binding(
    *,
    outer_commit: str,
    outer_tree: str,
    outer_subject: str,
    origin_main: str,
    origin_main_is_ancestor: bool,
    accelerator_pre_change_commit: str,
    accelerator_pre_change_tree: str,
    accelerator_gitlink: str,
    accelerator_origin_main: str,
    accelerator_origin_main_is_ancestor: bool,
    datasets_commit: str,
    datasets_tree: str,
    datasets_gitlink: str,
    kit_commit: str,
    kit_tree: str,
    kit_gitlink: str,
) -> dict[str, Any]:
    outer = _git_object_id(outer_commit, "outer_commit")
    tree = _git_object_id(outer_tree, "outer_tree")
    subject = _text(outer_subject, "outer_subject")
    origin = _git_object_id(origin_main, "origin_main")
    _require_ancestor(origin_main_is_ancestor, "origin_main_is_ancestor")
    accel = _git_object_id(
        accelerator_pre_change_commit, "accelerator_pre_change_commit"
    )
    accel_tree = _git_object_id(
        accelerator_pre_change_tree, "accelerator_pre_change_tree"
    )
    accel_link = _git_object_id(accelerator_gitlink, "accelerator_gitlink")
    accel_origin = _git_object_id(
        accelerator_origin_main, "accelerator_origin_main"
    )
    _require_ancestor(
        accelerator_origin_main_is_ancestor, "accelerator_origin_main_is_ancestor"
    )
    if accel != accel_link:
        raise PythonCompatibilityMetadataError(
            "accelerator_pre_change_commit must equal accelerator_gitlink"
        )
    datasets = _git_object_id(datasets_commit, "datasets_commit")
    datasets_tree_id = _git_object_id(datasets_tree, "datasets_tree")
    datasets_link = _git_object_id(datasets_gitlink, "datasets_gitlink")
    if datasets != datasets_link:
        raise PythonCompatibilityMetadataError(
            "datasets_commit must equal datasets_gitlink"
        )
    kit = _git_object_id(kit_commit, "kit_commit")
    kit_tree_id = _git_object_id(kit_tree, "kit_tree")
    kit_link = _git_object_id(kit_gitlink, "kit_gitlink")
    if kit != kit_link:
        raise PythonCompatibilityMetadataError("kit_commit must equal kit_gitlink")
    _reject_closed_release_value(subject, "outer_subject")
    return {
        "outer_repository": "endomorphosis/lift_coding",
        "owning_repository_for_receipts": "ipfs_accelerate_py",
        "outer_commit": outer,
        "outer_tree": tree,
        "outer_subject": subject,
        "origin_main": origin,
        "origin_main_is_ancestor": True,
        "accelerator_pre_change_commit": accel,
        "accelerator_pre_change_tree": accel_tree,
        "accelerator_gitlink": accel_link,
        "accelerator_origin_main": accel_origin,
        "accelerator_origin_main_is_ancestor": True,
        "accelerator_post_change_commit": "pending nested commit after admission",
        "accelerator_post_change_tree": (
            "dirty-worktree; exact CID after accepted nested commit"
        ),
        "datasets_commit": datasets,
        "datasets_tree": datasets_tree_id,
        "datasets_gitlink": datasets_link,
        "kit_commit": kit,
        "kit_tree": kit_tree_id,
        "kit_gitlink": kit_link,
        "evidence_kind": "measured",
    }


def current_head_pcpr_036_current_tree_binding() -> dict[str, Any]:
    return pcpr_036_current_tree_binding(
        outer_commit=CURRENT_HEAD_OUTER_COMMIT,
        outer_tree=CURRENT_HEAD_OUTER_TREE,
        outer_subject=CURRENT_HEAD_OUTER_SUBJECT,
        origin_main=CURRENT_HEAD_ORIGIN_MAIN,
        origin_main_is_ancestor=True,
        accelerator_pre_change_commit=CURRENT_HEAD_ACCELERATOR_COMMIT,
        accelerator_pre_change_tree=CURRENT_HEAD_ACCELERATOR_TREE,
        accelerator_gitlink=CURRENT_HEAD_ACCELERATOR_COMMIT,
        accelerator_origin_main=CURRENT_HEAD_ACCELERATOR_ORIGIN_MAIN,
        accelerator_origin_main_is_ancestor=True,
        datasets_commit=CURRENT_HEAD_DATASETS_COMMIT,
        datasets_tree=CURRENT_HEAD_DATASETS_TREE,
        datasets_gitlink=CURRENT_HEAD_DATASETS_COMMIT,
        kit_commit=CURRENT_HEAD_KIT_COMMIT,
        kit_tree=CURRENT_HEAD_KIT_TREE,
        kit_gitlink=CURRENT_HEAD_KIT_COMMIT,
    )


def validate_pcpr_036_outer_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if an outer PCPR-036 receipt claims a closed release."""

    if not isinstance(payload, Mapping):
        raise PythonCompatibilityMetadataError("outer receipt must be a mapping")
    if payload.get("task_id") != PCPR_036_TASK_ID:
        raise PythonCompatibilityMetadataError("outer receipt task_id must be PCPR-036")
    _reject_closed_release_value(payload.get("status"), "status")
    _reject_closed_release_value(payload.get("promotion_status"), "promotion_status")
    if payload.get("release_claim") is True:
        raise PythonCompatibilityMetadataError("outer receipt must not claim a release")
    if payload.get("completion_authoritative") is True:
        raise PythonCompatibilityMetadataError(
            "outer receipt completion is not authoritative"
        )

    verdict_section = payload.get("qualification_verdict")
    if not isinstance(verdict_section, Mapping):
        raise PythonCompatibilityMetadataError("qualification_verdict must be a mapping")
    _reject_closed_release_value(
        verdict_section.get("promotion_status"),
        "qualification_verdict.promotion_status",
    )
    _reject_closed_release_value(
        verdict_section.get("closed_release_outcome"),
        "qualification_verdict.closed_release_outcome",
    )
    if verdict_section.get("closed_release_outcome") is not None:
        raise PythonCompatibilityMetadataError(
            "qualification_verdict.closed_release_outcome must not be a closed PCPR release outcome"
        )
    if verdict_section.get("release_claim") is True:
        raise PythonCompatibilityMetadataError(
            "qualification_verdict must not claim a release"
        )
    if verdict_section.get("duckdb_or_quack_state_written") is True:
        raise PythonCompatibilityMetadataError(
            "python compatibility metadata must not write DuckDB or Quack state"
        )
    if verdict_section.get("contracts_frozen") is True:
        raise PythonCompatibilityMetadataError(
            "python compatibility metadata cannot freeze contracts"
        )
    if verdict_section.get("live_ci_matrix_executed") is True:
        raise PythonCompatibilityMetadataError(
            "this task cannot claim live CI matrix execution"
        )
    if verdict_section.get("live_older_python_qualified") is True:
        raise PythonCompatibilityMetadataError(
            "this task cannot claim live older-Python qualification"
        )
    promotion_status = verdict_section.get("promotion_status")
    if promotion_status not in PROMOTION_STATUSES:
        raise PythonCompatibilityMetadataError(
            "qualification_verdict.promotion_status is not an admitted PCPR-036 status"
        )
    if promotion_status == "supervisor_promoted":
        raise PythonCompatibilityMetadataError(
            "python compatibility metadata cannot promote the supervisor"
        )

    acceptance = payload.get("acceptance")
    if isinstance(acceptance, Mapping):
        _reject_closed_release_value(
            acceptance.get("promotion_status"), "acceptance.promotion_status"
        )
        if acceptance.get("closed_release_outcome") is not None:
            raise PythonCompatibilityMetadataError(
                "acceptance.closed_release_outcome must be null"
            )
        if acceptance.get("release_claim") is True:
            raise PythonCompatibilityMetadataError("acceptance must not claim a release")

    expected = current_head_pcpr_036_receipt_promotion()
    if verdict_section.get("verdict_cid") != expected["verdict_cid"]:
        raise PythonCompatibilityMetadataError(
            "qualification_verdict.verdict_cid must match the evaluator"
        )
    if promotion_status != expected["promotion_status"]:
        raise PythonCompatibilityMetadataError(
            "qualification_verdict.promotion_status must match the evaluator"
        )
    if expected["verdict_cid"] != CURRENT_HEAD_NON_PROMOTION_VERDICT_CID:
        raise PythonCompatibilityMetadataError(
            "pinned current-head non-promotion CID drifted from the evaluator"
        )

    return {
        "valid": True,
        "task_id": PCPR_036_TASK_ID,
        "promotion_status": promotion_status,
        "closed_release_outcome": None,
        "release_claim": False,
        "verdict_cid": expected["verdict_cid"],
        "evidence_kind": "measured",
    }
