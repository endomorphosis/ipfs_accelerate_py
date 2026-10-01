"""Screened Doctor diagnostics for the independently admitted task input scope.

This is a separate, explicitly partial analysis, never a replacement for a
whole-checkout evidence bundle. The signed manifest continues to bind *all*
tracked source, including omitted consumers. A repair operator must establish
its own local contract and account for the open frontiers before using these
diagnostics; this adapter grants no proof, edit, or completion authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from ..analysis.doctor_contract_adapters import materialize_runtime_diagnostics
from ..analysis.doctor_repository_diagnostics import (
    DoctorAuthorityRoots, DoctorSourceUnit, diagnose_repository,
)
from ..analysis.planning_analysis_factory import (
    PlanningAnalysisSecretError, _contains_secret, _credential_path_reason,
)
from ..proof.formal_verification_contracts import canonical_json, content_identity
from . import local_planning_admission as local


class ScopedDoctorAnalysisError(ValueError):
    """A scoped diagnostic input lost its independently admitted binding."""


def _selection(repository: Path, admission: Mapping, task_cid: str) -> tuple[dict, dict, tuple[str, ...]]:
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    manifest = verified["manifest"]
    if (repository.resolve(strict=True) != repository
            or str(repository) != manifest["repository"]):
        raise ScopedDoctorAnalysisError("Doctor scope requires the exact admitted repository")
    tasks = [task for task in verified["graph"].tasks if task.task_cid == task_cid]
    if len(tasks) != 1:
        raise ScopedDoctorAnalysisError("Doctor scope requires one existing admitted task")
    specs = [spec for spec in manifest["tasks"] if spec["task_key"] == tasks[0].task_key]
    if len(specs) != 1 or set(specs[0]["scope_paths"]) != set(tasks[0].scope_paths):
        raise ScopedDoctorAnalysisError("Doctor scope differs from the independent task declaration")
    spec = specs[0]
    paths = tuple(sorted(set(spec["scope_paths"]) & set(manifest["sources"])))
    absent = set(spec["scope_paths"]) - set(paths)
    if not paths or not absent <= set(manifest.get("created_outputs", ())):
        raise ScopedDoctorAnalysisError("Doctor scope contains unavailable undeclared source")
    return verified, spec, paths


def _capture(repository: Path, manifest: Mapping, paths: tuple[str, ...]) -> dict[str, bytes]:
    sources = {}
    for name in paths:
        path = repository / name
        if path.is_symlink() or path.resolve(strict=True) != path or not path.is_file():
            raise ScopedDoctorAnalysisError("Doctor scoped source is missing or symlinked")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["sources"][name]["sha256"]:
            raise ScopedDoctorAnalysisError("Doctor scoped source differs from the signed manifest")
        # Use the same path and byte detectors as the whole-checkout factory.
        # Do not interpolate the refused name or bytes into diagnostics.
        if _credential_path_reason(name) or _contains_secret(raw):
            raise PlanningAnalysisSecretError("secret-like material refused in the admitted Doctor task scope")
        sources[name] = raw
    return sources


@dataclass(frozen=True)
class ScopedDoctorAnalysis:
    """Exact inert bytes and native diagnostics with an explicit partial scope."""

    repository: Path
    task_cid: str
    sources: Mapping[str, bytes] = field(repr=False)
    diagnostic_snapshot: Any
    snapshot: Any
    findings: tuple[Any, ...]
    _admission_json: str = field(repr=False)
    _report_json: str = field(repr=False)

    @property
    def report(self) -> dict:
        """Return a detached body-free observation; it carries no authority."""
        return json.loads(self._report_json)

    def assert_current(self) -> None:
        """Recheck the whole manifest as well as every captured source byte."""
        verified, spec, paths = _selection(
            self.repository, json.loads(self._admission_json), self.task_cid,
        )
        report = self.report
        if (verified["receipt"]["manifest_cid"] != report["manifest_cid"]
                or verified["current_source_tree_id"] != report["source_tree_id"]
                or content_identity(spec) != report["task_spec_cid"]
                or paths != tuple(self.sources)):
            raise ScopedDoctorAnalysisError("Doctor task scope or manifest binding changed")
        if _capture(self.repository, verified["manifest"], paths) != dict(self.sources):
            raise ScopedDoctorAnalysisError("Doctor scoped source changed after analysis")


def build_scoped_doctor_analysis(*, repository: Path, admission: Mapping, task_cid: str) -> ScopedDoctorAnalysis:
    """Analyze all existing signed task inputs without caller-directed narrowing.

    Undeclared files cannot enter the analysis. Absent declared-create outputs
    have no source evidence. Omitted tracked files remain bound by the original
    manifest and are explicitly outside this diagnostic claim. No target module
    is imported and no model or prover is invoked by this preparation stage.
    """
    repository = Path(repository).absolute()
    admission_json = canonical_json(admission)
    verified, spec, paths = _selection(repository, json.loads(admission_json), task_cid)
    manifest = verified["manifest"]
    sources = _capture(repository, manifest, paths)
    inventory = {name: manifest["sources"][name] for name in paths}
    omitted = {name: value for name, value in manifest["sources"].items() if name not in sources}
    scope = {
        "schema": "supervisor-doctor-signed-task-scope@1",
        "manifest_cid": verified["receipt"]["manifest_cid"], "task_cid": task_cid,
        "task_spec_cid": content_identity(spec),
        "source_tree_id": verified["current_source_tree_id"],
        "sources": inventory,
    }
    scope_id = content_identity(scope)
    roots = DoctorAuthorityRoots(
        repository_id=manifest["repository_cid"], forest_id=scope_id,
        tree_id=scope_id, overlay_id=scope_id, file_root_id=scope_id,
        blob_root_id=scope_id, config_id=content_identity({"paths": paths}),
        policy_id=content_identity({"mode": "signed-task-scoped-diagnostics-only@1"}),
    )
    diagnostics = diagnose_repository([
        DoctorSourceUnit(path=name, source_bytes=raw,
                         blob_identity="sha256:" + inventory[name]["sha256"])
        for name, raw in sources.items()
    ], authority_roots=roots)
    snapshot, findings, bridge_id = materialize_runtime_diagnostics(
        diagnostics, require_repository_id=manifest["repository_cid"],
    )
    report = {
        "schema": "supervisor-doctor-scoped-analysis@1", "status": "available",
        "manifest_cid": verified["receipt"]["manifest_cid"], "task_cid": task_cid,
        "task_spec_cid": content_identity(spec), "source_tree_id": scope["source_tree_id"],
        "scope_id": scope_id, "scope_selection": "all-existing-independent-task-scope-paths@1",
        "source_hashes": {name: value["sha256"] for name, value in inventory.items()},
        "source_count": len(sources), "source_bytes": sum(map(len, sources.values())),
        "omitted_source_count": len(omitted),
        "omitted_source_inventory_cid": content_identity({"sources": omitted}),
        "omission_reason": "tracked sources outside the independently declared task scope",
        "absent_declared_create_outputs": sorted(set(spec["scope_paths"]) - set(paths)),
        "secret_screen": "native-planning-path-and-byte-detectors@1",
        "diagnostic_snapshot_id": snapshot.snapshot_id, "diagnostic_bridge_id": bridge_id,
        "finding_count": len(findings), "open_frontiers": list(diagnostics.open_frontiers),
        "scoped_inventory_complete": True, "whole_repository_analysis": False,
        "full_static_analysis": False, "provider_calls": 0,
        "proof_created": False, "mutation_authority": False, "completion_authority": False,
    }
    report["analysis_cid"] = content_identity(report)
    result = ScopedDoctorAnalysis(repository, task_cid, MappingProxyType(sources), diagnostics,
        snapshot, tuple(findings), admission_json, canonical_json(report))
    result.assert_current()
    return result
