"""DCR-010 read-only reconciliation of current deterministic-repair evidence.

The scanner binds byte digests to a caller-supplied commit identity or to a
deterministic dirty-overlay identity.  It never imports candidate modules,
executes provider code, or treats file presence as evidence of wiring/readiness.
Its result is inventory evidence only and can never mark repair ready.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

CURRENT_IMPLEMENTATION_EVIDENCE_INTERFACE: Final[str] = "CurrentImplementationEvidence@1"
CURRENT_IMPLEMENTATION_EVIDENCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/deterministic-repair-current-state@1"
)


class CurrentComponentStatus(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Exactly one reconciliation outcome for every requested component."""

    IMPLEMENTED_CURRENT = "implemented_current"
    STALE = "stale"
    INCOMPLETE = "incomplete"
    UNWIRED = "unwired"
    CONFLICTING = "conflicting"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _cid(value: Any, *, prefix: str) -> str:
    return f"{prefix}:sha256:{hashlib.sha256(_canonical(value)).hexdigest()}"


def _file_digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _relative_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError("component paths must be non-empty relative paths")
    return candidate.as_posix()


@dataclass(frozen=True)
class CurrentEvidenceComponentSpec:
    """One re-used WPD/SCA/RPR/Doctor/Planner/live-wiring component contract."""

    component_id: str
    family: str
    paths: tuple[str, ...]
    required_markers: tuple[str, ...] = ()
    wiring_markers: tuple[str, ...] = ()
    expected_digests: Mapping[str, str] | None = None
    conflict_markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.component_id or not self.family or not self.paths:
            raise ValueError("component_id, family, and paths are required")
        object.__setattr__(self, "paths", tuple(_relative_path(path) for path in self.paths))
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("component paths must be unique")
        if not all(
            isinstance(item, str) and item
            for item in (*self.required_markers, *self.wiring_markers, *self.conflict_markers)
        ):
            raise ValueError("component markers must be non-empty text")
        expected = {
            _relative_path(key): str(value)
            for key, value in dict(self.expected_digests or {}).items()
        }
        if set(expected).difference(self.paths):
            raise ValueError("expected digest paths must belong to the component")
        object.__setattr__(self, "expected_digests", expected)


@dataclass(frozen=True)
class CurrentComponentEvidence:
    component_id: str
    family: str
    status: CurrentComponentStatus
    file_digests: Mapping[str, str]
    missing_paths: tuple[str, ...] = ()
    absent_markers: tuple[str, ...] = ()
    stale_paths: tuple[str, ...] = ()
    unwired_markers: tuple[str, ...] = ()
    conflicting_markers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "family": self.family,
            "status": self.status.value,
            "file_digests": dict(sorted(self.file_digests.items())),
            "missing_paths": list(self.missing_paths),
            "absent_markers": list(self.absent_markers),
            "stale_paths": list(self.stale_paths),
            "unwired_markers": list(self.unwired_markers),
            "conflicting_markers": list(self.conflicting_markers),
        }


# This is a scanner inventory, not an assertion that any component is ready.
# Required markers prevent a bare file from being classified current; callers
# may additionally bind expected digests for a specific reviewed revision.
DEFAULT_REUSED_COMPONENTS: Final[tuple[CurrentEvidenceComponentSpec, ...]] = (
    CurrentEvidenceComponentSpec(
        "wpd.pre_implementation_provider_gate",
        "WPD",
        ("ipfs_accelerate_py/agent_supervisor/todo_daemon/pre_implementation_provider_gate.py",),
        required_markers=("def evaluate_provider_gate", "def assert_provider_dispatch_allowed"),
    ),
    CurrentEvidenceComponentSpec(
        "sca.symbolic_repair",
        "SCA",
        ("ipfs_accelerate_py/agent_supervisor/sca_symbolic_repair.py",),
        required_markers=("class SymbolicRepairPolicy", "def run_symbolic_repair_stack"),
    ),
    CurrentEvidenceComponentSpec(
        "rpr.admission",
        "RPR",
        ("ipfs_accelerate_py/agent_supervisor/sca_rpr_admission.py",),
        required_markers=("def admit_implement_task",),
    ),
    CurrentEvidenceComponentSpec(
        "doctor.deterministic_runtime",
        "Doctor",
        ("ipfs_accelerate_py/agent_supervisor/runtime/deterministic_doctor_runtime.py",),
        required_markers=("class DeterministicDoctorRuntime",),
    ),
    CurrentEvidenceComponentSpec(
        "planner.symbolic_candidate",
        "Planner",
        ("ipfs_accelerate_py/agent_supervisor/planning/symbolic_candidate_planner.py",),
        required_markers=("class SymbolicCandidatePlanner",),
    ),
    CurrentEvidenceComponentSpec(
        "live_wiring.deterministic_repair_provider",
        "live-wiring",
        ("ipfs_accelerate_py/agent_supervisor/runtime/deterministic_repair_provider.py",),
        required_markers=("class DeterministicRepairProvider",),
        wiring_markers=("DeterministicRepairAuthorityPolicy",),
    ),
)


@dataclass(frozen=True)
class SyntheticPathFinding:
    """Static detection of legacy synthetic Planner/Doctor residual routing."""

    detected: bool
    status: CurrentComponentStatus
    provider_gate_path: str
    kernel_path: str
    implementation_daemon_path: str
    flags: tuple[str, ...]
    file_digests: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected": self.detected,
            "status": self.status.value,
            "provider_gate_path": self.provider_gate_path,
            "kernel_path": self.kernel_path,
            "implementation_daemon_path": self.implementation_daemon_path,
            "flags": list(self.flags),
            "file_digests": dict(sorted(self.file_digests.items())),
        }


@dataclass(frozen=True)
class CurrentImplementationEvidence:
    """Byte-stable, non-authoritative DCR-010 evidence receipt."""

    snapshot_kind: str
    snapshot_identity: str
    components: tuple[CurrentComponentEvidence, ...]
    synthetic_planner_doctor_path: SyntheticPathFinding

    INTERFACE: Final[str] = CURRENT_IMPLEMENTATION_EVIDENCE_INTERFACE

    def __post_init__(self) -> None:
        if self.snapshot_kind not in {"commit", "dirty_overlay"}:
            raise ValueError("snapshot_kind must be commit or dirty_overlay")
        if not self.snapshot_identity:
            raise ValueError("snapshot_identity is required")
        ids = [component.component_id for component in self.components]
        if len(ids) != len(set(ids)):
            raise ValueError("components must have unique component_id values")
        object.__setattr__(
            self, "components", tuple(sorted(self.components, key=lambda item: item.component_id))
        )

    @property
    def receipt_id(self) -> str:
        return _cid(self.to_dict(include_receipt=False), prefix="current-implementation-evidence")

    @property
    def repair_ready(self) -> bool:
        """Always false: reconciliation is never repair admission evidence."""

        return False

    def to_dict(self, *, include_receipt: bool = True) -> dict[str, Any]:
        payload = {
            "schema": CURRENT_IMPLEMENTATION_EVIDENCE_SCHEMA,
            "interface": self.INTERFACE,
            "snapshot_kind": self.snapshot_kind,
            "snapshot_identity": self.snapshot_identity,
            "components": [component.to_dict() for component in self.components],
            "synthetic_planner_doctor_path": self.synthetic_planner_doctor_path.to_dict(),
            "repair_ready": False,
            "provider_or_llm_invoked": False,
        }
        if include_receipt:
            payload["receipt_id"] = self.receipt_id
        return payload


def _read(root: Path, relative: str) -> bytes | None:
    path = root / relative
    try:
        if not path.is_file():
            return None
        return path.read_bytes()
    except OSError:
        return None


def _overlay_identity(files: Mapping[str, bytes]) -> str:
    return _cid(
        {path: _file_digest(data) for path, data in sorted(files.items())}, prefix="dirty-overlay"
    )


def _synthetic_finding(root: Path) -> SyntheticPathFinding:
    gate = "ipfs_accelerate_py/agent_supervisor/todo_daemon/pre_implementation_provider_gate.py"
    kernel = "ipfs_accelerate_py/agent_supervisor/todo_daemon/pre_implementation_kernel.py"
    daemon = "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py"
    payloads = {path: _read(root, path) for path in (gate, kernel, daemon)}
    text = {
        path: (data or b"").decode("utf-8", errors="replace") for path, data in payloads.items()
    }
    flags: list[str] = []
    if "allow_legacy_residual: bool = True" in text[gate]:
        flags.append("legacy_residual_allowance")
    if "allow_legacy_residual=True" in text[daemon]:
        flags.append("live_daemon_enables_legacy_residual")
    if "legacy_worker_prompt_residual" in text[gate]:
        flags.append("legacy_residual_packet_cid")
    if (
        "planner_available: bool = True" in text[gate]
        or "planner_available: bool = True" in text[kernel]
    ):
        flags.append("synthetic_planner_availability")
    if (
        "doctor_available: bool = True" in text[gate]
        or "doctor_available: bool = True" in text[kernel]
    ):
        flags.append("synthetic_doctor_availability")
    if (
        "implementation_disposition_cid(" in text[gate]
        or "implementation_disposition_cid(" in text[kernel]
    ):
        flags.append("synthetic_disposition_cid")
    digests = {path: _file_digest(data) for path, data in payloads.items() if data is not None}
    detected = bool(flags)
    return SyntheticPathFinding(
        detected=detected,
        status=CurrentComponentStatus.CONFLICTING
        if detected
        else CurrentComponentStatus.IMPLEMENTED_CURRENT,
        provider_gate_path=gate,
        kernel_path=kernel,
        implementation_daemon_path=daemon,
        flags=tuple(sorted(flags)),
        file_digests=digests,
    )


def reconcile_current_evidence(
    repo_root: str | Path,
    components: Sequence[CurrentEvidenceComponentSpec] = DEFAULT_REUSED_COMPONENTS,
    *,
    commit_id: str = "",
    dirty_overlay_identity: str = "",
) -> CurrentImplementationEvidence:
    """Read and classify evidence deterministically without running any code.

    Exactly one of ``commit_id`` or ``dirty_overlay_identity`` may be supplied.
    If neither is given the scanner creates an identity from the bytes it read;
    this is explicitly a dirty-overlay snapshot rather than a cleanliness claim.
    """

    if commit_id and dirty_overlay_identity:
        raise ValueError("commit and dirty-overlay identities are mutually exclusive")
    root = Path(repo_root).resolve()
    component_rows: list[CurrentComponentEvidence] = []
    all_files: dict[str, bytes] = {}
    for spec in components:
        bodies = {path: _read(root, path) for path in spec.paths}
        for path, body in bodies.items():
            if body is not None:
                all_files[path] = body
        missing = tuple(sorted(path for path, body in bodies.items() if body is None))
        joined = "\n".join(
            (body or b"").decode("utf-8", errors="replace") for body in bodies.values()
        )
        digests = {path: _file_digest(body) for path, body in bodies.items() if body is not None}
        absent = tuple(sorted(marker for marker in spec.required_markers if marker not in joined))
        stale = tuple(
            sorted(
                path
                for path, expected in spec.expected_digests.items()
                if digests.get(path) != expected
            )
        )
        unwired = tuple(sorted(marker for marker in spec.wiring_markers if marker not in joined))
        conflicting = tuple(sorted(marker for marker in spec.conflict_markers if marker in joined))
        if conflicting:
            status = CurrentComponentStatus.CONFLICTING
        elif missing or absent:
            status = CurrentComponentStatus.INCOMPLETE
        elif stale:
            status = CurrentComponentStatus.STALE
        elif unwired:
            status = CurrentComponentStatus.UNWIRED
        else:
            status = CurrentComponentStatus.IMPLEMENTED_CURRENT
        component_rows.append(
            CurrentComponentEvidence(
                component_id=spec.component_id,
                family=spec.family,
                status=status,
                file_digests=digests,
                missing_paths=missing,
                absent_markers=absent,
                stale_paths=stale,
                unwired_markers=unwired,
                conflicting_markers=conflicting,
            )
        )
    synthetic = _synthetic_finding(root)
    if commit_id:
        snapshot_kind, snapshot_identity = "commit", commit_id
    else:
        snapshot_kind = "dirty_overlay"
        snapshot_identity = dirty_overlay_identity or _overlay_identity(all_files)
    return CurrentImplementationEvidence(
        snapshot_kind=snapshot_kind,
        snapshot_identity=snapshot_identity,
        components=tuple(component_rows),
        synthetic_planner_doctor_path=synthetic,
    )


__all__ = [
    "CURRENT_IMPLEMENTATION_EVIDENCE_INTERFACE",
    "CURRENT_IMPLEMENTATION_EVIDENCE_SCHEMA",
    "DEFAULT_REUSED_COMPONENTS",
    "CurrentComponentEvidence",
    "CurrentComponentStatus",
    "CurrentEvidenceComponentSpec",
    "CurrentImplementationEvidence",
    "SyntheticPathFinding",
    "reconcile_current_evidence",
]
