"""Current-tree reconciliation for deterministic contract repair.

This is deliberately an observation-only inventory.  Older WPD, SCA, and RPR
boards are useful pointers to code, but are not evidence that their claims
apply to the checkout being observed.  The report therefore binds every
reused component to current source and test bytes, assigns one closed
classification, and calls out synthetic Planner/Doctor identities separately.
It never imports or executes the components it inspects.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

CURRENT_IMPLEMENTATION_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-current-state@1"
)
CURRENT_IMPLEMENTATION_COMPONENT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-component@1"
)
SYNTHETIC_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-synthetic-evidence@1"
)
CURRENT_STATE_FILENAME = "current-state.json"


class ReuseClassification(str, Enum):
    """Closed disposition for a component named by a legacy repair program."""

    IMPLEMENTED_CURRENT = "implemented_current"
    STALE = "stale"
    INCOMPLETE = "incomplete"
    UNWIRED = "unwired"
    CONFLICTING = "conflicting"


@dataclass(frozen=True)
class CurrentImplementationComponent:
    """One byte-bound component classification in the current checkout."""

    component_id: str
    legacy_programs: tuple[str, ...]
    classification: ReuseClassification
    source_paths: tuple[str, ...]
    test_paths: tuple[str, ...]
    evidence: tuple[Mapping[str, Any], ...]
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CURRENT_IMPLEMENTATION_COMPONENT_SCHEMA,
            "component_id": self.component_id,
            "legacy_programs": list(self.legacy_programs),
            "classification": self.classification.value,
            "source_paths": list(self.source_paths),
            "test_paths": list(self.test_paths),
            "evidence": [dict(item) for item in self.evidence],
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class CurrentImplementationEvidence:
    """Portable projection of current WPD/SCA/RPR and Planner/Doctor evidence."""

    repository_commit: str
    components: tuple[CurrentImplementationComponent, ...]
    synthetic_evidence: tuple[Mapping[str, Any], ...]
    report_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        counts = {classification.value: 0 for classification in ReuseClassification}
        for component in self.components:
            counts[component.classification.value] += 1
        payload = {
            "schema": CURRENT_IMPLEMENTATION_EVIDENCE_SCHEMA,
            "repository_commit": self.repository_commit,
            "components": [component.to_dict() for component in self.components],
            "classification_counts": counts,
            "synthetic_evidence": [dict(item) for item in self.synthetic_evidence],
            "authoritative": False,
            "completion_authorized": False,
        }
        payload["report_id"] = self.report_id or _content_id(payload)
        return payload


@dataclass(frozen=True)
class CurrentImplementationEvidenceValidation:
    """Fail-closed replay result for a stored current-state projection.

    ``repository_commit`` and ``report_id`` record the observation that
    produced an artifact.  They intentionally do not need to equal a later
    checkout's values, but the historical commit must remain reachable from
    the checkout being validated and every non-provenance observation must
    still match exactly.
    """

    valid: bool = False
    observed_repository_commit: str = ""
    current_repository_commit: str = ""
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ComponentSpec:
    component_id: str
    programs: tuple[str, ...]
    source_paths: tuple[str, ...]
    test_paths: tuple[str, ...]
    present_classification: ReuseClassification
    present_reason: str
    required_tokens: tuple[tuple[str, str], ...] = ()


_ACCELERATE = "external/ipfs_accelerate/"
_SUPERVISOR = _ACCELERATE + "ipfs_accelerate_py/agent_supervisor/"
_TEST = _ACCELERATE + "test/api/"

# The catalog is intentionally small and explicit: these are the components
# DCR-010 reuses from the legacy programs, not a claim that every matching
# filename in the repository belongs to one of them.
_COMPONENTS: tuple[_ComponentSpec, ...] = (
    _ComponentSpec(
        "wpd.implementation_disposition",
        ("WPD",),
        (_SUPERVISOR + "todo_daemon/implementation_disposition.py",),
        (_TEST + "test_agent_supervisor_implementation_disposition.py",),
        ReuseClassification.IMPLEMENTED_CURRENT,
        "source_and_focused_test_present",
    ),
    _ComponentSpec(
        "sca.analyzer_health",
        ("SCA",),
        (_SUPERVISOR + "analysis/analyzer_health.py",),
        (_TEST + "test_agent_supervisor_analysis_escalation.py",),
        ReuseClassification.IMPLEMENTED_CURRENT,
        "source_and_focused_test_present",
    ),
    _ComponentSpec(
        "sca.stored_baseline",
        ("SCA",),
        ("data/agent_supervisor/swissknife_contract_assurance/baseline/summary.md",),
        (),
        ReuseClassification.STALE,
        "historical_baseline_is_not_bound_to_the_current_forest",
    ),
    _ComponentSpec(
        "rpr.change_propagation_pipeline",
        ("RPR",),
        (_SUPERVISOR + "analysis/change_propagation_pipeline.py",),
        (_TEST + "test_agent_supervisor_change_propagation_integration.py",),
        ReuseClassification.UNWIRED,
        "feature_gated_pipeline_has_no_current_live_wiring_receipt",
    ),
    _ComponentSpec(
        "wpd.pre_implementation_provider_gate",
        ("WPD", "Planner", "Doctor"),
        (_SUPERVISOR + "todo_daemon/pre_implementation_provider_gate.py",),
        (_TEST + "test_agent_supervisor_implementation_daemon_planner_doctor_hook.py",),
        ReuseClassification.CONFLICTING,
        "legacy_residual_authorization_conflicts_with_deterministic_only_authority",
        ((
            _SUPERVISOR + "todo_daemon/pre_implementation_provider_gate.py",
            "allow_legacy_residual: bool = True",
        ),),
    ),
    _ComponentSpec(
        "wpd.pre_implementation_kernel",
        ("WPD", "Planner", "Doctor"),
        (_SUPERVISOR + "todo_daemon/pre_implementation_kernel.py",),
        (_TEST + "test_agent_supervisor_implementation_daemon_planner_doctor_hook.py",),
        ReuseClassification.CONFLICTING,
        "derived_planner_and_doctor_view_ids_are_synthetic_not_service_evidence",
        ((
            _SUPERVISOR + "todo_daemon/pre_implementation_kernel.py",
            '"view": "planner"',
        ), (
            _SUPERVISOR + "todo_daemon/pre_implementation_kernel.py",
            '"view": "doctor"',
        )),
    ),
    _ComponentSpec(
        "doctor.default_factory",
        ("WPD", "Doctor"),
        (_SUPERVISOR + "control/default_doctor_factory.py",),
        (_TEST + "test_agent_supervisor_deterministic_doctor_end_to_end.py",),
        ReuseClassification.UNWIRED,
        "factory_exists_but_has_no_current_live_doctor_service_receipt",
    ),
    _ComponentSpec(
        "planner.default_factory",
        ("WPD", "Planner"),
        (_SUPERVISOR + "planning/default_planner_factory.py",),
        (_TEST + "test_agent_supervisor_formal_planning_contracts.py",),
        ReuseClassification.UNWIRED,
        "factory_exists_but_has_no_current_live_planner_service_receipt",
    ),
    _ComponentSpec(
        "planner_doctor.live_verification",
        ("Planner", "Doctor"),
        (_SUPERVISOR + "validation/planner_doctor_live_benchmark.py",),
        (_TEST + "test_agent_supervisor_deterministic_doctor_live_fixed_point.py",),
        ReuseClassification.INCOMPLETE,
        "required_live_verification_module_or_receipt_is_absent",
    ),
)


def _content_id(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _valid_git_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _repository_root(value: str | os.PathLike[str] | None) -> Path:
    if value is not None:
        return Path(value).resolve()
    # analysis -> agent_supervisor -> ipfs_accelerate_py -> ipfs_accelerate
    # -> external -> workspace root
    return Path(__file__).resolve().parents[5]


def _git_commit(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    commit = result.stdout.strip()
    return commit if _valid_git_commit(commit) else "unavailable"


def _is_ancestor_commit(root: Path, observed: str, current: str) -> bool:
    """Return whether a validated observed commit is reachable from ``HEAD``."""

    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                os.fspath(root),
                "merge-base",
                "--is-ancestor",
                observed,
                current,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _load_stored_current_evidence(
    source: (
        CurrentImplementationEvidence | Mapping[str, Any] | str | os.PathLike[str]
    ),
) -> Mapping[str, Any] | None:
    """Load a stored projection without accepting duplicate JSON keys."""

    if isinstance(source, CurrentImplementationEvidence):
        return source.to_dict()
    if isinstance(source, Mapping):
        return dict(source)

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"duplicate JSON key: {key}")
            payload[key] = value
        return payload

    try:
        payload = json.loads(
            Path(source).read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, TypeError, ValueError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _without_observation_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact evidence surface that must survive a later commit."""

    return {
        key: item
        for key, item in value.items()
        if key not in {"repository_commit", "report_id"}
    }


def _path_evidence(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    record: dict[str, Any] = {"path": relative_path, "exists": path.is_file()}
    if path.is_file():
        data = path.read_bytes()
        record.update({"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return record


def _all_tokens_present(root: Path, required_tokens: Iterable[tuple[str, str]]) -> bool:
    for relative_path, token in required_tokens:
        path = root / relative_path
        if not path.is_file() or token not in path.read_text(encoding="utf-8", errors="replace"):
            return False
    return True


def _component_from_spec(root: Path, spec: _ComponentSpec) -> CurrentImplementationComponent:
    source_evidence = tuple(_path_evidence(root, path) for path in spec.source_paths)
    test_evidence = tuple(_path_evidence(root, path) for path in spec.test_paths)
    missing = [item["path"] for item in (*source_evidence, *test_evidence) if not item["exists"]]
    if missing:
        classification = ReuseClassification.INCOMPLETE
        reasons = ("required_current_evidence_missing", *sorted(missing))
    elif spec.required_tokens and not _all_tokens_present(root, spec.required_tokens):
        # A legacy label without its observed condition is not enough to keep a
        # previous conflict classification.  The present implementation needs
        # a fresh review instead of inheriting a stale conclusion.
        classification = ReuseClassification.INCOMPLETE
        reasons = ("classification_probe_no_longer_matches_current_bytes",)
    else:
        classification = spec.present_classification
        reasons = (spec.present_reason,)
    return CurrentImplementationComponent(
        component_id=spec.component_id,
        legacy_programs=spec.programs,
        classification=classification,
        source_paths=spec.source_paths,
        test_paths=spec.test_paths,
        evidence=(*source_evidence, *test_evidence),
        reason_codes=reasons,
    )


def _synthetic_evidence(root: Path) -> tuple[dict[str, Any], ...]:
    gate = _SUPERVISOR + "todo_daemon/pre_implementation_provider_gate.py"
    kernel = _SUPERVISOR + "todo_daemon/pre_implementation_kernel.py"
    findings: list[dict[str, Any]] = []
    probes = (
        ("legacy_residual_packet", gate, "legacy_worker_prompt_residual", "provider_authorization"),
        ("synthetic_planner_view", kernel, '"view": "planner"', "planner_evidence"),
        ("synthetic_doctor_view", kernel, '"view": "doctor"', "doctor_evidence"),
    )
    for evidence_id, relative_path, token, affects in probes:
        path = root / relative_path
        present = path.is_file() and token in path.read_text(encoding="utf-8", errors="replace")
        findings.append(
            {
                "schema": SYNTHETIC_EVIDENCE_SCHEMA,
                "evidence_id": evidence_id,
                "path": relative_path,
                "token": token,
                "present": present,
                "classification": ReuseClassification.CONFLICTING.value if present else ReuseClassification.IMPLEMENTED_CURRENT.value,
                "affects": affects,
                "authoritative": False,
                "reason_code": "synthetic_identity_cannot_establish_live_wiring" if present else "synthetic_pattern_not_observed",
            }
        )
    return tuple(findings)


def reconcile_current_evidence(
    repository_root: str | os.PathLike[str] | None = None,
) -> CurrentImplementationEvidence:
    """Classify the DCR-010 reused components against one current checkout.

    Missing files never become a successful observation.  The returned report
    is a non-authoritative diagnostic projection; it cannot promote a legacy
    task or authorize a repair.
    """

    root = _repository_root(repository_root)
    if not root.is_dir():
        raise ValueError(f"repository root does not exist: {root}")
    components = tuple(_component_from_spec(root, spec) for spec in _COMPONENTS)
    synthetic = _synthetic_evidence(root)
    report = CurrentImplementationEvidence(
        repository_commit=_git_commit(root),
        components=components,
        synthetic_evidence=synthetic,
    )
    # Compute the ID from the complete stable projection, including the
    # classifications.  This keeps report consumers from confusing it with a
    # timestamped status document.
    payload = report.to_dict()
    return CurrentImplementationEvidence(
        repository_commit=report.repository_commit,
        components=report.components,
        synthetic_evidence=report.synthetic_evidence,
        report_id=payload["report_id"],
    )


def validate_current_evidence(
    stored_evidence: (
        CurrentImplementationEvidence | Mapping[str, Any] | str | os.PathLike[str]
    ),
    repository_root: str | os.PathLike[str] | None = None,
) -> CurrentImplementationEvidenceValidation:
    """Fail closed unless stored current-state evidence still describes ``HEAD``.

    A report ID is a self-CID over the complete historical projection.  The
    historical commit and that CID are observation provenance, rather than a
    claim that the report was created at the checkout currently being
    validated.  A descendant checkout is therefore accepted only when the
    observed commit is an ancestor and a fresh component/synthetic scan is an
    exact match after removing those two provenance fields.
    """

    stored = _load_stored_current_evidence(stored_evidence)
    if stored is None:
        return CurrentImplementationEvidenceValidation(
            reason_codes=("stored_current_evidence_unreadable",),
        )

    observed_commit = stored.get("repository_commit")
    if not _valid_git_commit(observed_commit):
        return CurrentImplementationEvidenceValidation(
            reason_codes=("observed_repository_commit_invalid",),
        )
    assert isinstance(observed_commit, str)

    claimed_report_id = stored.get("report_id")
    stored_without_report_id = {
        key: value for key, value in stored.items() if key != "report_id"
    }
    try:
        expected_report_id = _content_id(stored_without_report_id)
    except (TypeError, ValueError):
        return CurrentImplementationEvidenceValidation(
            observed_repository_commit=observed_commit,
            reason_codes=("stored_current_evidence_not_canonical",),
        )
    if (
        not isinstance(claimed_report_id, str)
        or claimed_report_id != expected_report_id
    ):
        return CurrentImplementationEvidenceValidation(
            observed_repository_commit=observed_commit,
            reason_codes=("stored_report_id_mismatch",),
        )

    root = _repository_root(repository_root)
    if not root.is_dir():
        return CurrentImplementationEvidenceValidation(
            observed_repository_commit=observed_commit,
            reason_codes=("repository_root_unavailable",),
        )
    current_commit = _git_commit(root)
    if not _valid_git_commit(current_commit):
        return CurrentImplementationEvidenceValidation(
            observed_repository_commit=observed_commit,
            current_repository_commit=current_commit,
            reason_codes=("current_repository_commit_unavailable",),
        )
    if not _is_ancestor_commit(root, observed_commit, current_commit):
        return CurrentImplementationEvidenceValidation(
            observed_repository_commit=observed_commit,
            current_repository_commit=current_commit,
            reason_codes=("observed_repository_commit_not_ancestor",),
        )

    try:
        current = reconcile_current_evidence(root).to_dict()
    except (OSError, ValueError):
        return CurrentImplementationEvidenceValidation(
            observed_repository_commit=observed_commit,
            current_repository_commit=current_commit,
            reason_codes=("current_evidence_reconciliation_failed",),
        )
    if current.get("repository_commit") != current_commit:
        return CurrentImplementationEvidenceValidation(
            observed_repository_commit=observed_commit,
            current_repository_commit=current_commit,
            reason_codes=("repository_head_changed_during_validation",),
        )
    if _without_observation_provenance(stored) != _without_observation_provenance(
        current
    ):
        return CurrentImplementationEvidenceValidation(
            observed_repository_commit=observed_commit,
            current_repository_commit=current_commit,
            reason_codes=("current_evidence_drift",),
        )
    return CurrentImplementationEvidenceValidation(
        valid=True,
        observed_repository_commit=observed_commit,
        current_repository_commit=current_commit,
    )


def write_current_evidence(
    output_path: str | os.PathLike[str],
    repository_root: str | os.PathLike[str] | None = None,
) -> CurrentImplementationEvidence:
    """Atomically persist a reconciliation projection and return it."""

    evidence = reconcile_current_evidence(repository_root)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(evidence.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary_name = handle.name
    os.replace(temporary_name, destination)
    return evidence


__all__ = [
    "CURRENT_IMPLEMENTATION_COMPONENT_SCHEMA",
    "CURRENT_IMPLEMENTATION_EVIDENCE_SCHEMA",
    "CURRENT_STATE_FILENAME",
    "SYNTHETIC_EVIDENCE_SCHEMA",
    "CurrentImplementationComponent",
    "CurrentImplementationEvidence",
    "CurrentImplementationEvidenceValidation",
    "ReuseClassification",
    "reconcile_current_evidence",
    "validate_current_evidence",
    "write_current_evidence",
]
