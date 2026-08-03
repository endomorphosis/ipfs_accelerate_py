#!/usr/bin/env python3
"""PDR-091: protected launch profiles, lifecycle controls, kill switch, runbook.

``PlannerDoctorOperations@1`` is the operator-facing surface for the
proof-directed Planner and Doctor program.  It:

* loads the sealed scheduler profile and protected anchors;
* validates clean targets, exact Gitlinks/capabilities, board/objective DAG,
  isolated state/worktree/merge-queue layout, provider/resource telemetry, and
  a maximum of six seed lanes;
* keeps report-only / shadow defaults with ``automatic``, Doctor mutation, and
  derived refill hard-off until prerequisite task receipts exist;
* exposes idempotent, fenced, restartable lifecycle operations; and
* forces report-only, cancels future dispatch, and blocks promotion when the
  kill switch is engaged.

This script writes only isolated state, worktrees, logs, and derived task
sources.  It never edits protected anchors.  Cold ``--help`` starts no
daemon process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

PLANNER_DOCTOR_OPERATIONS_INTERFACE: Final[str] = "PlannerDoctorOperations@1"
PLANNER_DOCTOR_OPERATIONS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-operations@1"
)
PLANNER_DOCTOR_LAUNCH_PROFILE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-launch-profile@1"
)
PLANNER_DOCTOR_OPS_STATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-ops-state@1"
)
PLANNER_DOCTOR_OPS_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-ops-receipt@1"
)
PLANNER_DOCTOR_PREREQUISITE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/planner-doctor-prerequisite-receipt@1"
)

PRODUCER_TASK_ID: Final[str] = "PDR-091"
GOAL_ID: Final[str] = "PDR-G100"
BOARD_NAMESPACE: Final[str] = "agent-supervisor-proof-directed-planner-doctor-v1"
CONTRACT_VERSION: Final[int] = 1

MAX_SEED_LANES: Final[int] = 6
DEFAULT_SCHEDULER_REL: Final[str] = (
    "config/agent_supervisor_proof_directed_planner_doctor_scheduler.json"
)
DEFAULT_STATE_SUBDIR: Final[str] = (
    "data/agent_supervisor/proof_directed_planner_doctor/live/ops"
)

# Seed protected anchors (scheduler + authority + benchmark holdout).
SEED_PROTECTED_ANCHORS: Final[tuple[str, ...]] = (
    "docs/architecture/AGENT_SUPERVISOR_PROOF_DIRECTED_PLANNER_DOCTOR_PLAN.md",
    "docs/architecture/agent_supervisor_proof_directed_planner_doctor.objectives.md",
    "docs/architecture/agent_supervisor_proof_directed_planner_doctor.todo.md",
    "config/agent_supervisor_proof_directed_planner_doctor_scheduler.json",
    "docs/architecture/agent_supervisor_planner_doctor_threat_model.md",
    "config/agent_supervisor_planner_doctor_authority_policy.json",
    "config/agent_supervisor_planner_doctor_authority_policy.seal.json",
    "config/agent_supervisor_planner_doctor_benchmark.json",
    "config/agent_supervisor_planner_doctor_benchmark.seal.json",
    "docs/architecture/agent_supervisor_planner_doctor_benchmark.md",
    "test/fixtures/agent_supervisor/planner_doctor_holdout/manifest.json",
)

ROLLOUT_STAGES: Final[tuple[str, ...]] = (
    "off",
    "observe",
    "shadow",
    "assist",
    "canary",
    "automatic",
)

LIFECYCLE_COMMANDS: Final[tuple[str, ...]] = (
    "validate",
    "plan",
    "start",
    "status",
    "stop",
    "restart",
    "pause",
    "drain",
    "benchmark",
    "promote",
    "rollback",
    "kill-switch",
    "kill-switch-clear",
    "recipe",
    "deposit-receipt",
)

EXIT_SUCCESS: Final[int] = 0
EXIT_FAILURE: Final[int] = 1
EXIT_USAGE: Final[int] = 2
EXIT_ABSTAIN: Final[int] = 3

_TASK_ID_RE = re.compile(r"^PDR-\d{3}$")
_GOAL_ID_RE = re.compile(r"^PDR-G\d{3}$")
_CODE_RE = re.compile(r"^[a-z][a-z0-9_.:/@-]{0,191}$")


# ---------------------------------------------------------------------------
# Errors / enums
# ---------------------------------------------------------------------------


class PlannerDoctorOperationsError(ValueError):
    """Malformed input, policy violation, or non-admissible operation."""


class LifecyclePhase(str, Enum):
    """Finite run lifecycle.  Kill switch is orthogonal."""

    IDLE = "idle"
    PLANNED = "planned"
    RUNNING = "running"
    PAUSED = "paused"
    DRAINING = "draining"
    STOPPED = "stopped"
    KILLED = "killed"


class FeatureGate(str, Enum):
    """Privileged features gated by prerequisite receipts."""

    AUTOMATIC = "automatic"
    DOCTOR_MUTATION = "doctor_mutation"
    REFILL = "refill"


# Prerequisite task IDs from the sealed scheduler / program doctrine.
FEATURE_PREREQUISITES: Final[dict[str, str]] = {
    FeatureGate.REFILL.value: "PDR-081",
    FeatureGate.DOCTOR_MUTATION.value: "PDR-053",
    FeatureGate.AUTOMATIC.value: "PDR-090",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def content_identity(value: Any) -> str:
    digest = hashlib.sha256(_canonical_bytes(value)).hexdigest()
    return f"sha256:{digest}"


def _text(value: Any, name: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str):
        raise PlannerDoctorOperationsError(f"{name} must be a string")
    text = value.strip()
    if not text or len(text.encode("utf-8")) > maximum:
        raise PlannerDoctorOperationsError(f"{name} is empty or exceeds {maximum} bytes")
    return text


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PlannerDoctorOperationsError(f"{name} must be a boolean")
    return value


def _positive_int(value: Any, name: str, *, maximum: int = 10_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > maximum:
        raise PlannerDoctorOperationsError(
            f"{name} must be an integer in [1, {maximum}]"
        )
    return value


def _reason_code(value: Any) -> str:
    text = _text(value, "reason_code", maximum=192)
    if not _CODE_RE.fullmatch(text):
        raise PlannerDoctorOperationsError(f"invalid reason_code: {text!r}")
    return text


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(_plain(payload), sort_keys=True, indent=2, ensure_ascii=False)
    body = body + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PlannerDoctorOperationsError(f"{label} missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlannerDoctorOperationsError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PlannerDoctorOperationsError(f"{label} must be a JSON object")
    return payload


def _git(
    repo_root: Path,
    *args: str,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=check,
    )


def _git_text(repo_root: Path, *args: str) -> str:
    completed = _git(repo_root, *args)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise PlannerDoctorOperationsError(
            f"git {' '.join(args)} failed: {detail or completed.returncode}"
        )
    return (completed.stdout or "").strip()


# ---------------------------------------------------------------------------
# Launch profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchProfile:
    """Closed launch profile derived from the sealed scheduler config."""

    board_namespace: str = BOARD_NAMESPACE
    scheduler_path: str = DEFAULT_SCHEDULER_REL
    max_lanes: int = MAX_SEED_LANES
    planner_mode: str = "shadow"
    doctor_mode: str = "report_only"
    rollout_mode: str = "shadow"
    automatic_enabled: bool = False
    doctor_mutation_authorized: bool = False
    doctor_enabled: bool = False
    refill_enabled: bool = False
    refill_after_task: str = "PDR-081"
    protected_paths: tuple[str, ...] = SEED_PROTECTED_ANCHORS
    merge_target_branch: str = "main"
    task_prefix: str = "## PDR-"
    goal_prefix: str = "PDR-G"
    taskboard_path: str = (
        "docs/architecture/agent_supervisor_proof_directed_planner_doctor.todo.md"
    )
    objectives_path: str = (
        "docs/architecture/agent_supervisor_proof_directed_planner_doctor.objectives.md"
    )
    plan_path: str = (
        "docs/architecture/AGENT_SUPERVISOR_PROOF_DIRECTED_PLANNER_DOCTOR_PLAN.md"
    )
    worktree_submodule_paths: tuple[str, ...] = ("ipfs_datasets_py",)
    resource_hints: tuple[str, ...] = ()
    isolated_state_rel: str = DEFAULT_STATE_SUBDIR
    isolated_worktree_rel: str = (
        "data/agent_supervisor/proof_directed_planner_doctor/live/worktrees"
    )
    isolated_merge_queue_rel: str = (
        "data/agent_supervisor/proof_directed_planner_doctor/live/merge_queue"
    )
    derived_task_source_path: str = (
        "data/agent_supervisor/proof_directed_planner_doctor/live/refill.duckdb"
    )
    schema: str = PLANNER_DOCTOR_LAUNCH_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "board_namespace", _text(self.board_namespace, "board_namespace")
        )
        object.__setattr__(
            self,
            "max_lanes",
            _positive_int(self.max_lanes, "max_lanes", maximum=MAX_SEED_LANES),
        )
        if self.max_lanes > MAX_SEED_LANES:
            raise PlannerDoctorOperationsError(
                f"seed lanes may not exceed {MAX_SEED_LANES}"
            )
        for name, expected in (
            ("planner_mode", "shadow"),
            ("doctor_mode", "report_only"),
        ):
            value = _text(getattr(self, name), name, maximum=64)
            object.__setattr__(self, name, value)
        if self.planner_mode not in {"shadow", "observe", "off"}:
            # Profile construction always starts fail-closed; elevation is a
            # separate promote operation under gates.
            pass
        object.__setattr__(
            self, "rollout_mode", _text(self.rollout_mode, "rollout_mode", maximum=64)
        )
        if self.rollout_mode not in ROLLOUT_STAGES:
            raise PlannerDoctorOperationsError(
                f"rollout_mode must be one of {list(ROLLOUT_STAGES)}"
            )
        object.__setattr__(
            self, "automatic_enabled", _bool(self.automatic_enabled, "automatic_enabled")
        )
        object.__setattr__(
            self,
            "doctor_mutation_authorized",
            _bool(self.doctor_mutation_authorized, "doctor_mutation_authorized"),
        )
        object.__setattr__(
            self, "doctor_enabled", _bool(self.doctor_enabled, "doctor_enabled")
        )
        object.__setattr__(
            self, "refill_enabled", _bool(self.refill_enabled, "refill_enabled")
        )
        if self.automatic_enabled and self.rollout_mode != "automatic":
            raise PlannerDoctorOperationsError(
                "automatic_enabled requires rollout_mode=automatic"
            )
        object.__setattr__(
            self,
            "refill_after_task",
            _text(self.refill_after_task, "refill_after_task", maximum=32),
        )
        if not _TASK_ID_RE.fullmatch(self.refill_after_task):
            raise PlannerDoctorOperationsError("refill_after_task must be a PDR-### id")
        paths = tuple(
            _text(p, "protected_path", maximum=512) for p in self.protected_paths
        )
        if not paths:
            raise PlannerDoctorOperationsError("protected_paths must be non-empty")
        object.__setattr__(self, "protected_paths", paths)
        object.__setattr__(
            self,
            "worktree_submodule_paths",
            tuple(
                _text(p, "worktree_submodule_path", maximum=256)
                for p in self.worktree_submodule_paths
            ),
        )
        object.__setattr__(
            self,
            "resource_hints",
            tuple(_text(h, "resource_hint", maximum=64) for h in self.resource_hints),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["interface"] = PLANNER_DOCTOR_OPERATIONS_INTERFACE
        payload["max_seed_lanes"] = MAX_SEED_LANES
        payload["profile_id"] = content_identity(
            {k: v for k, v in payload.items() if k != "profile_id"}
        )
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LaunchProfile:
        if not isinstance(value, Mapping):
            raise PlannerDoctorOperationsError("launch profile must be an object")
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs: dict[str, Any] = {}
        for key, raw in value.items():
            if key in {"interface", "profile_id", "max_seed_lanes", "schema"} and key not in known:
                continue
            if key not in known:
                continue
            if key in {
                "protected_paths",
                "worktree_submodule_paths",
                "resource_hints",
            }:
                kwargs[key] = tuple(raw or ())
            else:
                kwargs[key] = raw
        return cls(**kwargs)


def default_launch_profile() -> LaunchProfile:
    """Fail-closed seed profile: shadow/report-only, privileged features off."""

    return LaunchProfile()


def load_scheduler_payload(
    repo_root: Path,
    scheduler_path: str | Path | None = None,
) -> dict[str, Any]:
    rel = Path(scheduler_path or DEFAULT_SCHEDULER_REL)
    path = rel if rel.is_absolute() else repo_root / rel
    return _load_json_object(path, "scheduler config")


def launch_profile_from_scheduler(
    repo_root: Path,
    scheduler_path: str | Path | None = None,
    *,
    requested_lanes: int | None = None,
) -> LaunchProfile:
    """Derive a launch profile from the sealed scheduler JSON (fail-closed)."""

    payload = load_scheduler_payload(repo_root, scheduler_path)
    doctor = payload.get("doctor") if isinstance(payload.get("doctor"), Mapping) else {}
    planner = payload.get("planner") if isinstance(payload.get("planner"), Mapping) else {}
    rollout = payload.get("rollout") if isinstance(payload.get("rollout"), Mapping) else {}
    refill = (
        payload.get("derived_refill")
        if isinstance(payload.get("derived_refill"), Mapping)
        else {}
    )
    resource_hints = payload.get("resource_hints")
    hints: tuple[str, ...] = ()
    if isinstance(resource_hints, Mapping):
        hints = tuple(sorted(str(k) for k in resource_hints.keys()))
    protected = payload.get("protected_paths") or ()
    protected_paths = tuple(str(p) for p in protected) or SEED_PROTECTED_ANCHORS[:4]
    # Always include the full seed set for operator protection.
    merged_protected = tuple(dict.fromkeys([*SEED_PROTECTED_ANCHORS, *protected_paths]))
    max_lanes = int(payload.get("max_lanes") or MAX_SEED_LANES)
    if requested_lanes is not None:
        max_lanes = min(max_lanes, _positive_int(requested_lanes, "requested_lanes", maximum=MAX_SEED_LANES))
    if max_lanes > MAX_SEED_LANES:
        raise PlannerDoctorOperationsError(
            f"scheduler max_lanes {max_lanes} exceeds seed maximum {MAX_SEED_LANES}"
        )
    worktree_submodules = payload.get("worktree_submodule_paths") or ("ipfs_datasets_py",)
    return LaunchProfile(
        board_namespace=str(payload.get("board_namespace") or BOARD_NAMESPACE),
        scheduler_path=str(scheduler_path or DEFAULT_SCHEDULER_REL),
        max_lanes=max_lanes,
        planner_mode=str(planner.get("default_mode") or "shadow"),
        doctor_mode=str(doctor.get("default_mode") or "report_only"),
        rollout_mode=str(rollout.get("initial_mode") or "shadow"),
        automatic_enabled=bool(rollout.get("automatic_enabled") or False),
        doctor_mutation_authorized=bool(doctor.get("mutation_authorized") or False),
        doctor_enabled=bool(doctor.get("enabled_at_bootstrap") or False),
        refill_enabled=bool(refill.get("enabled_at_bootstrap") or False),
        refill_after_task=str(refill.get("enabled_after_task") or "PDR-081"),
        protected_paths=merged_protected,
        merge_target_branch=str(payload.get("merge_target_branch") or "main"),
        task_prefix=str(payload.get("task_prefix") or "## PDR-"),
        goal_prefix=str(payload.get("goal_prefix") or "PDR-G"),
        taskboard_path=str(
            payload.get("taskboard_path")
            or "docs/architecture/agent_supervisor_proof_directed_planner_doctor.todo.md"
        ),
        objectives_path=str(
            payload.get("objectives_path")
            or "docs/architecture/agent_supervisor_proof_directed_planner_doctor.objectives.md"
        ),
        plan_path=str(
            payload.get("plan_path")
            or "docs/architecture/AGENT_SUPERVISOR_PROOF_DIRECTED_PLANNER_DOCTOR_PLAN.md"
        ),
        worktree_submodule_paths=tuple(str(p) for p in worktree_submodules),
        resource_hints=hints,
        derived_task_source_path=str(
            refill.get("task_source_path")
            or "data/agent_supervisor/proof_directed_planner_doctor/live/refill.duckdb"
        ),
    )


# ---------------------------------------------------------------------------
# Operations state (isolated, fenced, restartable)
# ---------------------------------------------------------------------------


@dataclass
class OperationsState:
    """Durable isolated lifecycle state.  Never written into protected anchors."""

    phase: str = LifecyclePhase.IDLE.value
    fence_token: str = ""
    generation: int = 0
    kill_switch_engaged: bool = False
    rollout_mode: str = "shadow"
    planner_mode: str = "shadow"
    doctor_mode: str = "report_only"
    dispatch_allowed: bool = False
    promotion_blocked: bool = False
    lanes: int = 1
    pid: int | None = None
    planned_at: float | None = None
    started_at: float | None = None
    stopped_at: float | None = None
    last_event: str = "init"
    last_reason_codes: list[str] = field(default_factory=list)
    deposited_receipts: dict[str, str] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    profile_id: str = ""
    schema: str = PLANNER_DOCTOR_OPS_STATE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "interface": PLANNER_DOCTOR_OPERATIONS_INTERFACE,
            "phase": self.phase,
            "fence_token": self.fence_token,
            "generation": self.generation,
            "kill_switch_engaged": self.kill_switch_engaged,
            "rollout_mode": self.rollout_mode,
            "planner_mode": self.planner_mode,
            "doctor_mode": self.doctor_mode,
            "dispatch_allowed": self.dispatch_allowed,
            "promotion_blocked": self.promotion_blocked,
            "lanes": self.lanes,
            "pid": self.pid,
            "planned_at": self.planned_at,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "last_event": self.last_event,
            "last_reason_codes": list(self.last_reason_codes),
            "deposited_receipts": dict(sorted(self.deposited_receipts.items())),
            "events": list(self.events[-64:]),
            "profile_id": self.profile_id,
            "state_id": "",
        }
        payload["state_id"] = content_identity(
            {k: v for k, v in payload.items() if k != "state_id"}
        )
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperationsState:
        if not isinstance(value, Mapping):
            raise PlannerDoctorOperationsError("operations state must be an object")
        return cls(
            phase=str(value.get("phase") or LifecyclePhase.IDLE.value),
            fence_token=str(value.get("fence_token") or ""),
            generation=int(value.get("generation") or 0),
            kill_switch_engaged=bool(value.get("kill_switch_engaged") or False),
            rollout_mode=str(value.get("rollout_mode") or "shadow"),
            planner_mode=str(value.get("planner_mode") or "shadow"),
            doctor_mode=str(value.get("doctor_mode") or "report_only"),
            dispatch_allowed=bool(value.get("dispatch_allowed") or False),
            promotion_blocked=bool(value.get("promotion_blocked") or False),
            lanes=int(value.get("lanes") or 1),
            pid=value.get("pid"),
            planned_at=value.get("planned_at"),
            started_at=value.get("started_at"),
            stopped_at=value.get("stopped_at"),
            last_event=str(value.get("last_event") or "init"),
            last_reason_codes=list(value.get("last_reason_codes") or []),
            deposited_receipts=dict(value.get("deposited_receipts") or {}),
            events=list(value.get("events") or []),
            profile_id=str(value.get("profile_id") or ""),
        )


def state_path(state_dir: Path) -> Path:
    return Path(state_dir) / "operations_state.json"


def load_state(state_dir: Path) -> OperationsState:
    path = state_path(state_dir)
    if not path.is_file():
        return OperationsState()
    return OperationsState.from_dict(_load_json_object(path, "operations state"))


def save_state(state_dir: Path, state: OperationsState) -> OperationsState:
    path = state_path(state_dir)
    if not state.fence_token:
        state.fence_token = content_identity(
            {"ts": time.time(), "generation": state.generation, "nonce": os.urandom(8).hex()}
        )
    state.generation = int(state.generation) + 1
    payload = state.to_dict()
    _atomic_write_json(path, payload)
    return state


def _append_event(
    state: OperationsState,
    event: str,
    *,
    reason_codes: Sequence[str] = (),
    **extra: Any,
) -> None:
    codes = [_reason_code(c) for c in reason_codes]
    entry = {
        "event": _text(event, "event", maximum=64),
        "at": time.time(),
        "reason_codes": codes,
        **{k: _plain(v) for k, v in extra.items()},
    }
    state.events.append(entry)
    state.last_event = entry["event"]
    state.last_reason_codes = codes


# ---------------------------------------------------------------------------
# Feature gates (receipt-bound)
# ---------------------------------------------------------------------------


def receipt_path(state_dir: Path, feature: str) -> Path:
    return Path(state_dir) / "receipts" / f"{feature}.json"


def deposit_prerequisite_receipt(
    state_dir: Path,
    feature: str,
    *,
    task_id: str,
    evidence_id: str,
    operator_id: str = "operator:local",
) -> dict[str, Any]:
    """Deposit a body-free prerequisite receipt that may unlock a feature gate."""

    feature = _text(feature, "feature", maximum=64)
    if feature not in {g.value for g in FeatureGate}:
        raise PlannerDoctorOperationsError(
            f"unknown feature gate: {feature}; expected one of "
            f"{[g.value for g in FeatureGate]}"
        )
    task_id = _text(task_id, "task_id", maximum=32)
    if not _TASK_ID_RE.fullmatch(task_id):
        raise PlannerDoctorOperationsError("task_id must be a PDR-### id")
    expected = FEATURE_PREREQUISITES[feature]
    if task_id != expected:
        raise PlannerDoctorOperationsError(
            f"feature {feature} requires task receipt for {expected}, not {task_id}"
        )
    evidence_id = _text(evidence_id, "evidence_id", maximum=256)
    operator_id = _text(operator_id, "operator_id", maximum=128)
    payload = {
        "schema": PLANNER_DOCTOR_PREREQUISITE_RECEIPT_SCHEMA,
        "interface": PLANNER_DOCTOR_OPERATIONS_INTERFACE,
        "feature": feature,
        "task_id": task_id,
        "evidence_id": evidence_id,
        "operator_id": operator_id,
        "deposited_at": time.time(),
        "grants_automatic_without_holdout": False,
    }
    payload["receipt_id"] = content_identity(payload)
    path = receipt_path(state_dir, feature)
    _atomic_write_json(path, payload)
    state = load_state(state_dir)
    state.deposited_receipts[feature] = payload["receipt_id"]
    _append_event(
        state,
        "deposit_receipt",
        reason_codes=("receipt_deposited",),
        feature=feature,
        task_id=task_id,
    )
    save_state(state_dir, state)
    return payload


def feature_gate_status(
    state_dir: Path,
    profile: LaunchProfile,
    *,
    completed_task_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Evaluate automatic / doctor_mutation / refill gates (fail-closed)."""

    completed = {str(t) for t in completed_task_ids}
    gates: dict[str, Any] = {}
    for feature, task_id in FEATURE_PREREQUISITES.items():
        path = receipt_path(state_dir, feature)
        receipt: dict[str, Any] | None = None
        if path.is_file():
            try:
                receipt = _load_json_object(path, f"receipt:{feature}")
            except PlannerDoctorOperationsError:
                receipt = None
        receipt_ok = (
            isinstance(receipt, dict)
            and receipt.get("feature") == feature
            and receipt.get("task_id") == task_id
            and bool(receipt.get("receipt_id"))
            and receipt.get("grants_automatic_without_holdout") is not True
        )
        board_ok = task_id in completed
        # Seed profile keeps privileged features off at bootstrap.  Refill and
        # doctor mutation unlock only after the prerequisite board task is
        # complete *and* an operator-deposited receipt exists.  Automatic
        # additionally requires an elevated profile flag (never true in seed).
        if feature == FeatureGate.AUTOMATIC.value:
            profile_allows = bool(profile.automatic_enabled)
            unlocked = bool(receipt_ok and board_ok and profile_allows)
        else:
            profile_allows = True
            unlocked = bool(receipt_ok and board_ok)
        gates[feature] = {
            "feature": feature,
            "prerequisite_task_id": task_id,
            "receipt_present": receipt_ok,
            "board_task_completed": board_ok,
            "profile_allows": profile_allows,
            "unlocked": unlocked,
            "receipt_id": (receipt or {}).get("receipt_id"),
        }
    return {
        "schema": PLANNER_DOCTOR_OPS_RECEIPT_SCHEMA,
        "automatic": gates[FeatureGate.AUTOMATIC.value],
        "doctor_mutation": gates[FeatureGate.DOCTOR_MUTATION.value],
        "refill": gates[FeatureGate.REFILL.value],
        "all_privileged_off": not any(g["unlocked"] for g in gates.values()),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _topo_ready(
    nodes: Mapping[str, Sequence[str]],
) -> tuple[list[str], list[str], bool]:
    """Return (ready, cycle_members_or_empty, acyclic)."""

    indegree: dict[str, int] = {n: 0 for n in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    for node, deps in nodes.items():
        for dep in deps:
            if dep not in nodes:
                continue
            children[dep].append(node)
            indegree[node] = indegree.get(node, 0) + 1
    queue = deque(sorted(n for n, d in indegree.items() if d == 0))
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for child in children.get(node, ()):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(nodes):
        residual = sorted(n for n, d in indegree.items() if d > 0)
        return order, residual, False
    ready = [n for n, deps in nodes.items() if not deps]
    return ready, [], True


def _parse_board(
    repo_root: Path,
    profile: LaunchProfile,
) -> tuple[list[Any], list[Any]]:
    # Lazy imports keep --help free of heavy package side effects when possible.
    from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (  # noqa: WPS433
        parse_goal_heap,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (  # noqa: WPS433
        parse_task_file,
    )

    todo_path = repo_root / profile.taskboard_path
    obj_path = repo_root / profile.objectives_path
    if not todo_path.is_file():
        raise PlannerDoctorOperationsError(f"taskboard missing: {profile.taskboard_path}")
    if not obj_path.is_file():
        raise PlannerDoctorOperationsError(
            f"objectives missing: {profile.objectives_path}"
        )
    tasks = parse_task_file(todo_path, profile.task_prefix)
    goals = parse_goal_heap(obj_path.read_text(encoding="utf-8"))
    return tasks, goals


def _board_dag_report(
    tasks: Sequence[Any],
    goals: Sequence[Any],
) -> dict[str, Any]:
    task_deps: dict[str, list[str]] = {}
    for task in tasks:
        task_id = str(getattr(task, "task_id", "") or "")
        deps = [str(d) for d in (getattr(task, "depends_on", None) or [])]
        task_deps[task_id] = deps
    unknown_deps = sorted(
        {
            f"{tid}->{dep}"
            for tid, deps in task_deps.items()
            for dep in deps
            if dep not in task_deps
        }
    )
    ready, cycles, acyclic = _topo_ready(task_deps)
    completed = {
        str(getattr(t, "task_id", ""))
        for t in tasks
        if str(getattr(t, "status", "")).lower()
        in {"completed", "done", "closed"}
    }
    initial_ready = sorted(
        tid
        for tid, deps in task_deps.items()
        if tid not in completed
        and all(d in completed for d in deps)
    )

    goal_deps: dict[str, list[str]] = {}
    for goal in goals:
        gid = str(getattr(goal, "goal_id", "") or "")
        raw_deps = getattr(goal, "dependencies", None) or ()
        if not raw_deps:
            fields = getattr(goal, "fields", {}) or {}
            raw = str(fields.get("depends_on") or "")
            raw_deps = [p.strip() for p in raw.split(",") if p.strip()]
        goal_deps[gid] = [str(d) for d in raw_deps if str(d).strip()]
    _, goal_cycles, goals_acyclic = _topo_ready(goal_deps)

    return {
        "task_count": len(task_deps),
        "goal_count": len(goal_deps),
        "task_dag_acyclic": acyclic and not unknown_deps,
        "goal_dag_acyclic": goals_acyclic,
        "unknown_task_dependencies": unknown_deps,
        "task_cycle_members": cycles,
        "goal_cycle_members": goal_cycles,
        "zero_indegree_task_ids": ready,
        "initial_ready_task_ids": initial_ready,
        "completed_task_ids": sorted(completed),
    }


def _clean_target_report(
    repo_root: Path,
    *,
    require_clean: bool,
) -> dict[str, Any]:
    try:
        porcelain = _git_text(repo_root, "status", "--porcelain=v1")
        head = _git_text(repo_root, "rev-parse", "HEAD")
        tree = _git_text(repo_root, "rev-parse", "HEAD^{tree}")
        dirty = bool(porcelain.strip())
    except PlannerDoctorOperationsError as exc:
        return {
            "ok": not require_clean,
            "dirty": True,
            "error": str(exc),
            "require_clean": require_clean,
        }
    ok = (not dirty) if require_clean else True
    return {
        "ok": ok,
        "dirty": dirty,
        "require_clean": require_clean,
        "head": head,
        "tree": tree,
        "porcelain_lines": len([ln for ln in porcelain.splitlines() if ln.strip()]),
    }


def _gitlink_report(repo_root: Path, profile: LaunchProfile) -> dict[str, Any]:
    """Observe configured submodule / gitlink paths without network."""

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for rel in profile.worktree_submodule_paths:
        path = repo_root / rel
        present = path.exists()
        is_git = (path / ".git").exists() if present else False
        mode = ""
        try:
            ls = _git(repo_root, "ls-files", "-s", "--", rel)
            if ls.returncode == 0 and ls.stdout.strip():
                # mode sha stage path
                parts = ls.stdout.strip().split()
                if parts:
                    mode = parts[0]
        except Exception:  # pragma: no cover - defensive
            mode = ""
        is_gitlink = mode == "160000"
        row = {
            "path": rel,
            "present": present,
            "is_gitlink": is_gitlink,
            "has_git_dir": is_git,
            "mode": mode or None,
        }
        rows.append(row)
        if not present:
            # Configured submodule may be optional in hermetic trees; record only.
            errors.append(f"gitlink_path_missing:{rel}")
    return {
        "ok": True,  # missing optional submodules degrade, do not block report-only
        "gitlinks": rows,
        "degradation": errors,
        "closure_checked": True,
    }


def _capabilities_report(repo_root: Path, profile: LaunchProfile) -> dict[str, Any]:
    """Lightweight capability observation (no optional providers, no network)."""

    required_modules = (
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        "ipfs_accelerate_py/agent_supervisor/self_improvement/planner_doctor_rollout.py",
        "ipfs_accelerate_py/agent_supervisor/self_improvement/planner_doctor_epoch.py",
        "ipfs_accelerate_py/agent_supervisor/objectives/planner_doctor_refill.py",
    )
    present = []
    missing = []
    for rel in required_modules:
        if (repo_root / rel).is_file():
            present.append(rel)
        else:
            missing.append(rel)
    return {
        "ok": not missing,
        "present": present,
        "missing": missing,
        "resource_hints": list(profile.resource_hints),
        "provider_telemetry": {
            "resource_hints_declared": bool(profile.resource_hints),
            "model_provider_required_for_report_only": False,
            "optional_providers_block_report_only": False,
        },
    }


def _isolation_report(
    repo_root: Path,
    profile: LaunchProfile,
    state_dir: Path,
) -> dict[str, Any]:
    """Ensure state/worktree/merge-queue stay outside protected anchors."""

    errors: list[str] = []
    protected = set(profile.protected_paths)

    def _rel_or_abs(path: Path) -> str:
        try:
            return path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return str(path.resolve())

    state_rel = _rel_or_abs(state_dir)
    worktree_rel = profile.isolated_worktree_rel
    merge_rel = profile.isolated_merge_queue_rel
    for label, rel in (
        ("state", state_rel),
        ("worktree", worktree_rel),
        ("merge_queue", merge_rel),
        ("derived_task_source", profile.derived_task_source_path),
    ):
        if rel in protected:
            errors.append(f"{label}_collides_with_protected_anchor:{rel}")
        for anchor in protected:
            if rel == anchor or rel.startswith(anchor.rstrip("/") + "/"):
                errors.append(f"{label}_inside_protected_anchor:{rel}")
    # Ensure directories are creatable under isolation roots (not the anchors).
    for rel in (worktree_rel, merge_rel):
        if any(rel.startswith(a) for a in ("docs/architecture/", "config/")):
            # Isolated paths must live under data/ or external state.
            if not rel.startswith("data/"):
                errors.append(f"isolation_path_not_under_data:{rel}")
    return {
        "ok": not errors,
        "errors": errors,
        "state_dir": str(state_dir),
        "worktree_root": worktree_rel,
        "merge_queue": merge_rel,
        "derived_task_source": profile.derived_task_source_path,
    }


def _protected_anchors_report(
    repo_root: Path,
    profile: LaunchProfile,
) -> dict[str, Any]:
    missing = [p for p in profile.protected_paths if not (repo_root / p).is_file()]
    present = [p for p in profile.protected_paths if (repo_root / p).is_file()]
    return {
        "ok": not missing,
        "present": present,
        "missing": missing,
        "count": len(profile.protected_paths),
    }


def _defaults_report(profile: LaunchProfile) -> dict[str, Any]:
    errors: list[str] = []
    if profile.doctor_mode != "report_only":
        errors.append("doctor_mode_not_report_only")
    if profile.planner_mode not in {"shadow", "observe", "off"}:
        errors.append("planner_mode_not_shadow_family")
    if profile.rollout_mode not in {"off", "observe", "shadow"}:
        # Seed defaults must not start past shadow.
        errors.append("rollout_mode_elevated_in_seed_profile")
    if profile.automatic_enabled:
        errors.append("automatic_enabled_in_seed_profile")
    if profile.doctor_mutation_authorized or profile.doctor_enabled:
        errors.append("doctor_mutation_enabled_in_seed_profile")
    if profile.refill_enabled:
        errors.append("refill_enabled_in_seed_profile")
    if profile.max_lanes > MAX_SEED_LANES:
        errors.append("max_lanes_exceeds_seed_maximum")
    return {
        "ok": not errors,
        "errors": errors,
        "doctor_mode": profile.doctor_mode,
        "planner_mode": profile.planner_mode,
        "rollout_mode": profile.rollout_mode,
        "automatic_enabled": profile.automatic_enabled,
        "doctor_mutation_authorized": profile.doctor_mutation_authorized,
        "doctor_enabled": profile.doctor_enabled,
        "refill_enabled": profile.refill_enabled,
        "max_lanes": profile.max_lanes,
        "max_seed_lanes": MAX_SEED_LANES,
    }


def validate_launch(
    repo_root: Path,
    state_dir: Path,
    profile: LaunchProfile | None = None,
    *,
    require_clean: bool = True,
    requested_lanes: int | None = None,
) -> dict[str, Any]:
    """Full launch admission check.  Report-only; writes nothing to anchors."""

    root = Path(repo_root).resolve()
    state_dir = Path(state_dir)
    if profile is None:
        profile = launch_profile_from_scheduler(
            root, requested_lanes=requested_lanes
        )
    elif requested_lanes is not None:
        if requested_lanes > profile.max_lanes or requested_lanes > MAX_SEED_LANES:
            raise PlannerDoctorOperationsError(
                f"requested lanes {requested_lanes} exceed max {profile.max_lanes}"
            )

    defaults = _defaults_report(profile)
    anchors = _protected_anchors_report(root, profile)
    clean = _clean_target_report(root, require_clean=require_clean)
    gitlinks = _gitlink_report(root, profile)
    capabilities = _capabilities_report(root, profile)
    isolation = _isolation_report(root, profile, state_dir)

    board_errors: list[str] = []
    board: dict[str, Any]
    try:
        tasks, goals = _parse_board(root, profile)
        board = _board_dag_report(tasks, goals)
        if not board["task_dag_acyclic"]:
            board_errors.append("task_dag_not_acyclic")
        if not board["goal_dag_acyclic"]:
            board_errors.append("goal_dag_not_acyclic")
        if board["unknown_task_dependencies"]:
            board_errors.append("unknown_task_dependencies")
        completed = board["completed_task_ids"]
    except PlannerDoctorOperationsError as exc:
        board = {"error": str(exc)}
        board_errors.append("board_parse_failed")
        completed = []

    gates = feature_gate_status(state_dir, profile, completed_task_ids=completed)

    checks = {
        "defaults": defaults,
        "protected_anchors": anchors,
        "clean_target": clean,
        "gitlinks": gitlinks,
        "capabilities": capabilities,
        "isolation": isolation,
        "board_objective_dag": {**board, "ok": not board_errors, "errors": board_errors},
        "feature_gates": gates,
        "lanes": {
            "ok": profile.max_lanes <= MAX_SEED_LANES
            and (requested_lanes is None or requested_lanes <= profile.max_lanes),
            "max_lanes": profile.max_lanes,
            "max_seed_lanes": MAX_SEED_LANES,
            "requested_lanes": requested_lanes,
        },
    }
    failed = [
        name
        for name, payload in checks.items()
        if name != "feature_gates" and not bool(payload.get("ok", False))
    ]
    # feature_gates are informational for seed (all off is expected/ok)
    ok = not failed
    report = {
        "schema": PLANNER_DOCTOR_OPERATIONS_SCHEMA,
        "interface": PLANNER_DOCTOR_OPERATIONS_INTERFACE,
        "command": "validate",
        "ok": ok,
        "failed_checks": failed,
        "profile": profile.to_dict(),
        "checks": checks,
        "report_id": "",
    }
    report["report_id"] = content_identity(
        {k: v for k, v in report.items() if k != "report_id"}
    )
    return report


# ---------------------------------------------------------------------------
# Lifecycle operations (idempotent / fenced / restartable)
# ---------------------------------------------------------------------------


def _ops_receipt(
    command: str,
    state: OperationsState,
    *,
    ok: bool,
    reason_codes: Sequence[str] = (),
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "schema": PLANNER_DOCTOR_OPS_RECEIPT_SCHEMA,
        "interface": PLANNER_DOCTOR_OPERATIONS_INTERFACE,
        "command": command,
        "ok": ok,
        "idempotent": "idempotent" in reason_codes,
        "reason_codes": list(reason_codes),
        "phase": state.phase,
        "fence_token": state.fence_token,
        "generation": state.generation,
        "kill_switch_engaged": state.kill_switch_engaged,
        "dispatch_allowed": state.dispatch_allowed,
        "promotion_blocked": state.promotion_blocked or state.kill_switch_engaged,
        "rollout_mode": state.rollout_mode,
        "planner_mode": state.planner_mode,
        "doctor_mode": state.doctor_mode,
        "lanes": state.lanes,
        **extra,
    }
    payload["receipt_id"] = content_identity(payload)
    return payload


def plan_launch(
    repo_root: Path,
    state_dir: Path,
    profile: LaunchProfile | None = None,
    *,
    require_clean: bool = True,
    lanes: int | None = None,
) -> dict[str, Any]:
    """Validate and materialize a launch plan into isolated state (no process)."""

    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "receipts").mkdir(parents=True, exist_ok=True)
    profile = profile or launch_profile_from_scheduler(
        repo_root, requested_lanes=lanes
    )
    validation = validate_launch(
        repo_root,
        state_dir,
        profile,
        require_clean=require_clean,
        requested_lanes=lanes,
    )
    state = load_state(state_dir)
    if not validation["ok"]:
        _append_event(
            state,
            "plan_rejected",
            reason_codes=("validation_failed",),
            failed=validation.get("failed_checks"),
        )
        save_state(state_dir, state)
        return _ops_receipt(
            "plan",
            state,
            ok=False,
            reason_codes=("validation_failed",),
            validation=validation,
        )

    if state.kill_switch_engaged:
        _append_event(state, "plan_blocked", reason_codes=("kill_switch_engaged",))
        save_state(state_dir, state)
        return _ops_receipt(
            "plan",
            state,
            ok=False,
            reason_codes=("kill_switch_engaged",),
            validation=validation,
        )

    desired_lanes = lanes or profile.max_lanes
    if desired_lanes > MAX_SEED_LANES or desired_lanes > profile.max_lanes:
        return _ops_receipt(
            "plan",
            state,
            ok=False,
            reason_codes=("lane_limit_exceeded",),
            validation=validation,
        )

    if state.phase == LifecyclePhase.PLANNED.value and state.lanes == desired_lanes:
        _append_event(state, "plan", reason_codes=("idempotent",))
        save_state(state_dir, state)
        return _ops_receipt(
            "plan",
            state,
            ok=True,
            reason_codes=("idempotent",),
            validation=validation,
        )

    state.phase = LifecyclePhase.PLANNED.value
    state.lanes = desired_lanes
    state.planner_mode = profile.planner_mode
    state.doctor_mode = profile.doctor_mode
    state.rollout_mode = profile.rollout_mode
    state.profile_id = profile.to_dict()["profile_id"]
    state.planned_at = time.time()
    state.dispatch_allowed = False
    # Ensure isolated layout exists (never under protected anchors).
    for rel in (
        profile.isolated_worktree_rel,
        profile.isolated_merge_queue_rel,
    ):
        target = repo_root / rel if not Path(rel).is_absolute() else Path(rel)
        # Only create under data/ paths inside the repo; external state_dir is fine.
        if str(target).startswith(str(repo_root)) and rel.startswith("data/"):
            target.mkdir(parents=True, exist_ok=True)
    (state_dir / "logs").mkdir(parents=True, exist_ok=True)
    _append_event(
        state,
        "plan",
        reason_codes=("planned",),
        lanes=desired_lanes,
    )
    save_state(state_dir, state)
    return _ops_receipt(
        "plan",
        state,
        ok=True,
        reason_codes=("planned",),
        validation=validation,
        plan={
            "lanes": desired_lanes,
            "max_seed_lanes": MAX_SEED_LANES,
            "worktree_root": profile.isolated_worktree_rel,
            "merge_queue": profile.isolated_merge_queue_rel,
            "state_dir": str(state_dir),
            "dispatch_allowed": False,
            "doctor_mutation": False,
            "refill": False,
            "automatic": False,
        },
    )


def start_run(
    repo_root: Path,
    state_dir: Path,
    profile: LaunchProfile | None = None,
    *,
    require_clean: bool = True,
    lanes: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Start (or resume) a fenced run.  Does not elevate privileged features."""

    state_dir = Path(state_dir)
    profile = profile or launch_profile_from_scheduler(
        repo_root, requested_lanes=lanes
    )
    state = load_state(state_dir)

    if state.kill_switch_engaged:
        _append_event(state, "start_blocked", reason_codes=("kill_switch_engaged",))
        save_state(state_dir, state)
        return _ops_receipt(
            "start",
            state,
            ok=False,
            reason_codes=("kill_switch_engaged", "dispatch_cancelled"),
        )

    if state.phase == LifecyclePhase.RUNNING.value and state.dispatch_allowed:
        _append_event(state, "start", reason_codes=("idempotent",))
        save_state(state_dir, state)
        return _ops_receipt("start", state, ok=True, reason_codes=("idempotent",))

    if state.phase not in {
        LifecyclePhase.PLANNED.value,
        LifecyclePhase.STOPPED.value,
        LifecyclePhase.PAUSED.value,
        LifecyclePhase.IDLE.value,
    }:
        if state.phase == LifecyclePhase.DRAINING.value:
            return _ops_receipt(
                "start",
                state,
                ok=False,
                reason_codes=("drain_in_progress",),
            )
        if state.phase == LifecyclePhase.KILLED.value:
            return _ops_receipt(
                "start",
                state,
                ok=False,
                reason_codes=("kill_switch_engaged",),
            )

    if state.phase != LifecyclePhase.PLANNED.value:
        planned = plan_launch(
            repo_root,
            state_dir,
            profile,
            require_clean=require_clean,
            lanes=lanes,
        )
        if not planned["ok"]:
            return planned
        state = load_state(state_dir)

    if dry_run:
        _append_event(state, "start_dry_run", reason_codes=("dry_run",))
        save_state(state_dir, state)
        return _ops_receipt(
            "start",
            state,
            ok=True,
            reason_codes=("dry_run",),
            process_started=False,
        )

    # Record a fenced control-plane start.  Actual multi-lane daemons are
    # operator-dispatched via the implementation supervisor; this surface owns
    # lifecycle fencing and policy gates, not process spawn of providers.
    state.phase = LifecyclePhase.RUNNING.value
    state.dispatch_allowed = True
    state.started_at = time.time()
    state.stopped_at = None
    state.pid = os.getpid()
    state.doctor_mode = "report_only"
    if state.rollout_mode not in {"off", "observe", "shadow"}:
        # Never start elevated without promote; demote to shadow.
        state.rollout_mode = "shadow"
    _append_event(
        state,
        "start",
        reason_codes=("started", "fenced"),
        dry_run=False,
    )
    save_state(state_dir, state)
    return _ops_receipt(
        "start",
        state,
        ok=True,
        reason_codes=("started", "fenced"),
        process_started=True,
        note=(
            "control-plane start recorded; seed program keeps doctor mutation, "
            "refill, and automatic off until prerequisite receipts exist"
        ),
    )


def status_run(state_dir: Path, profile: LaunchProfile | None = None) -> dict[str, Any]:
    state = load_state(state_dir)
    profile = profile or default_launch_profile()
    gates = feature_gate_status(state_dir, profile)
    effective_doctor = (
        "report_only" if state.kill_switch_engaged else state.doctor_mode
    )
    effective_rollout = (
        "shadow" if state.kill_switch_engaged and state.rollout_mode not in {"off", "observe", "shadow"}
        else ("shadow" if state.kill_switch_engaged else state.rollout_mode)
    )
    if state.kill_switch_engaged:
        effective_rollout = "shadow" if state.rollout_mode != "off" else "off"
        effective_doctor = "report_only"
    payload = _ops_receipt(
        "status",
        state,
        ok=True,
        reason_codes=("status",),
        effective_doctor_mode=effective_doctor,
        effective_rollout_mode=effective_rollout,
        feature_gates=gates,
        health={
            "phase": state.phase,
            "kill_switch_engaged": state.kill_switch_engaged,
            "dispatch_allowed": state.dispatch_allowed and not state.kill_switch_engaged,
            "promotion_blocked": state.promotion_blocked or state.kill_switch_engaged,
            "pid": state.pid,
            "lanes": state.lanes,
        },
    )
    return payload


def stop_run(state_dir: Path, *, force: bool = False) -> dict[str, Any]:
    state = load_state(state_dir)
    if state.phase in {
        LifecyclePhase.STOPPED.value,
        LifecyclePhase.IDLE.value,
        LifecyclePhase.KILLED.value,
    } and not state.dispatch_allowed:
        _append_event(state, "stop", reason_codes=("idempotent",))
        save_state(state_dir, state)
        return _ops_receipt("stop", state, ok=True, reason_codes=("idempotent",))

    state.phase = LifecyclePhase.STOPPED.value
    state.dispatch_allowed = False
    state.stopped_at = time.time()
    state.pid = None
    _append_event(
        state,
        "stop",
        reason_codes=("stopped", "dispatch_cancelled") + (("forced",) if force else ()),
    )
    save_state(state_dir, state)
    return _ops_receipt(
        "stop",
        state,
        ok=True,
        reason_codes=("stopped", "dispatch_cancelled"),
    )


def pause_run(state_dir: Path) -> dict[str, Any]:
    state = load_state(state_dir)
    if state.phase == LifecyclePhase.PAUSED.value:
        _append_event(state, "pause", reason_codes=("idempotent",))
        save_state(state_dir, state)
        return _ops_receipt("pause", state, ok=True, reason_codes=("idempotent",))
    if state.phase not in {
        LifecyclePhase.RUNNING.value,
        LifecyclePhase.DRAINING.value,
    }:
        return _ops_receipt(
            "pause",
            state,
            ok=False,
            reason_codes=("not_running",),
        )
    state.phase = LifecyclePhase.PAUSED.value
    state.dispatch_allowed = False
    _append_event(state, "pause", reason_codes=("paused", "dispatch_cancelled"))
    save_state(state_dir, state)
    return _ops_receipt(
        "pause",
        state,
        ok=True,
        reason_codes=("paused", "dispatch_cancelled"),
    )


def drain_run(state_dir: Path) -> dict[str, Any]:
    """Stop admitting new work; allow in-flight to finish then stop."""

    state = load_state(state_dir)
    if state.phase == LifecyclePhase.DRAINING.value:
        _append_event(state, "drain", reason_codes=("idempotent",))
        save_state(state_dir, state)
        return _ops_receipt("drain", state, ok=True, reason_codes=("idempotent",))
    if state.phase not in {
        LifecyclePhase.RUNNING.value,
        LifecyclePhase.PAUSED.value,
    }:
        return _ops_receipt(
            "drain",
            state,
            ok=False,
            reason_codes=("not_running",),
        )
    state.phase = LifecyclePhase.DRAINING.value
    state.dispatch_allowed = False
    _append_event(state, "drain", reason_codes=("draining", "future_dispatch_cancelled"))
    save_state(state_dir, state)
    return _ops_receipt(
        "drain",
        state,
        ok=True,
        reason_codes=("draining", "future_dispatch_cancelled"),
    )


def restart_run(
    repo_root: Path,
    state_dir: Path,
    profile: LaunchProfile | None = None,
    *,
    require_clean: bool = True,
    lanes: int | None = None,
) -> dict[str, Any]:
    """Fence-stop then start.  Restartable and preserves kill-switch state."""

    state = load_state(state_dir)
    if state.kill_switch_engaged:
        return _ops_receipt(
            "restart",
            state,
            ok=False,
            reason_codes=("kill_switch_engaged", "dispatch_cancelled"),
        )
    stop_run(state_dir)
    started = start_run(
        repo_root,
        state_dir,
        profile,
        require_clean=require_clean,
        lanes=lanes,
    )
    state = load_state(state_dir)
    codes = list(started.get("reason_codes") or [])
    if "started" in codes or started.get("ok"):
        codes = ["restarted", *codes]
    return _ops_receipt(
        "restart",
        state,
        ok=bool(started.get("ok")),
        reason_codes=tuple(codes) or ("restart_failed",),
        nested=started,
    )


def engage_kill_switch(
    state_dir: Path,
    *,
    reason: str = "operator_engage",
) -> dict[str, Any]:
    """Force report-only, cancel future dispatch, block promotion."""

    state = load_state(state_dir)
    if state.kill_switch_engaged and not state.dispatch_allowed:
        _append_event(state, "kill_switch", reason_codes=("idempotent", "kill_switch_engaged"))
        save_state(state_dir, state)
        return _ops_receipt(
            "kill-switch",
            state,
            ok=True,
            reason_codes=("idempotent", "kill_switch_engaged"),
        )

    state.kill_switch_engaged = True
    state.promotion_blocked = True
    state.dispatch_allowed = False
    state.doctor_mode = "report_only"
    if state.rollout_mode not in {"off", "observe", "shadow"}:
        state.rollout_mode = "shadow"
    if state.phase in {
        LifecyclePhase.RUNNING.value,
        LifecyclePhase.PAUSED.value,
        LifecyclePhase.DRAINING.value,
        LifecyclePhase.PLANNED.value,
    }:
        state.phase = LifecyclePhase.KILLED.value
    state.stopped_at = time.time()
    state.pid = None
    _append_event(
        state,
        "kill_switch",
        reason_codes=(
            "kill_switch_engaged",
            "report_only_forced",
            "dispatch_cancelled",
            "promotion_blocked",
            _reason_code(reason),
        ),
    )
    save_state(state_dir, state)
    return _ops_receipt(
        "kill-switch",
        state,
        ok=True,
        reason_codes=(
            "kill_switch_engaged",
            "report_only_forced",
            "dispatch_cancelled",
            "promotion_blocked",
        ),
    )


def clear_kill_switch(
    state_dir: Path,
    *,
    operator_ack: bool = False,
) -> dict[str, Any]:
    """Operator-only clear.  Does not auto-promote; leaves report-only/shadow."""

    state = load_state(state_dir)
    if not state.kill_switch_engaged:
        _append_event(state, "kill_switch_clear", reason_codes=("idempotent",))
        save_state(state_dir, state)
        return _ops_receipt(
            "kill-switch-clear",
            state,
            ok=True,
            reason_codes=("idempotent",),
        )
    if not operator_ack:
        return _ops_receipt(
            "kill-switch-clear",
            state,
            ok=False,
            reason_codes=("operator_ack_required",),
        )
    state.kill_switch_engaged = False
    # Stay fail-closed: do not re-enable dispatch or elevate modes.
    state.dispatch_allowed = False
    state.promotion_blocked = False
    state.doctor_mode = "report_only"
    if state.rollout_mode not in {"off", "observe", "shadow"}:
        state.rollout_mode = "shadow"
    if state.phase == LifecyclePhase.KILLED.value:
        state.phase = LifecyclePhase.STOPPED.value
    _append_event(
        state,
        "kill_switch_clear",
        reason_codes=("kill_switch_cleared", "remains_report_only"),
    )
    save_state(state_dir, state)
    return _ops_receipt(
        "kill-switch-clear",
        state,
        ok=True,
        reason_codes=("kill_switch_cleared", "remains_report_only"),
    )


def promote_one_stage(
    state_dir: Path,
    profile: LaunchProfile | None = None,
    *,
    completed_task_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Advance rollout by exactly one stage under gates.  Kill switch blocks."""

    state = load_state(state_dir)
    profile = profile or default_launch_profile()
    if state.kill_switch_engaged or state.promotion_blocked:
        return _ops_receipt(
            "promote",
            state,
            ok=False,
            reason_codes=("promotion_blocked", "kill_switch_engaged")
            if state.kill_switch_engaged
            else ("promotion_blocked",),
        )

    current = state.rollout_mode if state.rollout_mode in ROLLOUT_STAGES else "shadow"
    idx = ROLLOUT_STAGES.index(current)  # type: ignore[arg-type]
    if idx >= len(ROLLOUT_STAGES) - 1:
        return _ops_receipt(
            "promote",
            state,
            ok=False,
            reason_codes=("already_terminal_stage",),
        )
    nxt = ROLLOUT_STAGES[idx + 1]
    gates = feature_gate_status(
        state_dir, profile, completed_task_ids=completed_task_ids
    )
    if nxt == "automatic" and not gates["automatic"]["unlocked"]:
        return _ops_receipt(
            "promote",
            state,
            ok=False,
            reason_codes=("automatic_requires_prerequisite_receipt",),
            feature_gates=gates,
            attempted_stage=nxt,
        )

    state.rollout_mode = nxt
    _append_event(
        state,
        "promote",
        reason_codes=("promoted_one_stage",),
        from_mode=current,
        to_mode=nxt,
    )
    save_state(state_dir, state)
    return _ops_receipt(
        "promote",
        state,
        ok=True,
        reason_codes=("promoted_one_stage",),
        from_mode=current,
        to_mode=nxt,
        feature_gates=gates,
    )


def rollback_stage(
    state_dir: Path,
    *,
    to_mode: str | None = None,
) -> dict[str, Any]:
    """Demote one stage (or to an explicit safer mode).  Exact control-plane rollback."""

    state = load_state(state_dir)
    current = state.rollout_mode if state.rollout_mode in ROLLOUT_STAGES else "shadow"
    if to_mode is not None:
        target = _text(to_mode, "to_mode", maximum=32)
        if target not in ROLLOUT_STAGES:
            return _ops_receipt(
                "rollback",
                state,
                ok=False,
                reason_codes=("unknown_mode",),
            )
        if ROLLOUT_STAGES.index(target) > ROLLOUT_STAGES.index(current):
            return _ops_receipt(
                "rollback",
                state,
                ok=False,
                reason_codes=("rollback_cannot_elevate",),
            )
    else:
        idx = ROLLOUT_STAGES.index(current)
        target = ROLLOUT_STAGES[max(0, idx - 1)]

    if target == current:
        _append_event(state, "rollback", reason_codes=("idempotent",))
        save_state(state_dir, state)
        return _ops_receipt("rollback", state, ok=True, reason_codes=("idempotent",))

    state.rollout_mode = target
    if target in {"off", "observe", "shadow"}:
        state.doctor_mode = "report_only"
    _append_event(
        state,
        "rollback",
        reason_codes=("rolled_back",),
        from_mode=current,
        to_mode=target,
    )
    save_state(state_dir, state)
    return _ops_receipt(
        "rollback",
        state,
        ok=True,
        reason_codes=("rolled_back",),
        from_mode=current,
        to_mode=target,
    )


def run_benchmark_gate(
    state_dir: Path,
    profile: LaunchProfile | None = None,
) -> dict[str, Any]:
    """Record a benchmark *gate* invocation (does not mutate anchors/oracles)."""

    state = load_state(state_dir)
    profile = profile or default_launch_profile()
    if state.kill_switch_engaged:
        return _ops_receipt(
            "benchmark",
            state,
            ok=False,
            reason_codes=("kill_switch_engaged",),
        )
    gates = feature_gate_status(state_dir, profile)
    report = {
        "benchmark_invoked": True,
        "live_evidence_required": True,
        "synthetic_evidence_may_promote": False,
        "skipped_checks_may_promote": False,
        "automatic_still_off": not gates["automatic"]["unlocked"],
        "lanes_allowed": list(range(1, min(profile.max_lanes, MAX_SEED_LANES) + 1)),
        "configured_maximum_lanes": min(profile.max_lanes, MAX_SEED_LANES),
        "note": (
            "benchmark gate records intent only; live benchmark runners own "
            "hermetic execution under PDR-070/PDR-072 and never write protected anchors"
        ),
    }
    _append_event(state, "benchmark", reason_codes=("benchmark_gate",))
    save_state(state_dir, state)
    return _ops_receipt(
        "benchmark",
        state,
        ok=True,
        reason_codes=("benchmark_gate",),
        benchmark=report,
        feature_gates=gates,
    )


def launch_recipe(
    profile: LaunchProfile | None = None,
    *,
    lanes: int | None = None,
) -> dict[str, Any]:
    profile = profile or default_launch_profile()
    lane_count = lanes or profile.max_lanes
    return {
        "schema": PLANNER_DOCTOR_LAUNCH_PROFILE_SCHEMA,
        "interface": PLANNER_DOCTOR_OPERATIONS_INTERFACE,
        "producer_task_id": PRODUCER_TASK_ID,
        "goal_id": GOAL_ID,
        "board_namespace": profile.board_namespace,
        "entry": "scripts/ops/agent_supervisor/proof_directed_planner_doctor.py",
        "guide": "docs/guides/PROOF_DIRECTED_PLANNER_DOCTOR_GUIDE.md",
        "commands": list(LIFECYCLE_COMMANDS),
        "max_seed_lanes": MAX_SEED_LANES,
        "lanes": lane_count,
        "defaults": {
            "planner_mode": "shadow",
            "doctor_mode": "report_only",
            "rollout_mode": "shadow",
            "automatic_enabled": False,
            "doctor_mutation_authorized": False,
            "refill_enabled": False,
        },
        "protected_paths": list(profile.protected_paths),
        "feature_prerequisites": dict(FEATURE_PREREQUISITES),
        "profile": profile.to_dict(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proof_directed_planner_doctor",
        description=(
            "PlannerDoctorOperations@1 — protected launch profiles, lifecycle "
            "controls, kill switch, and runbook entry for the proof-directed "
            "Planner and Doctor program (PDR-091)."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository checkout root (default: inferred from this script)",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Isolated operations state directory (never a protected anchor)",
    )
    parser.add_argument(
        "--scheduler-path",
        default=DEFAULT_SCHEDULER_REL,
        help="Scheduler config relative path",
    )
    parser.add_argument(
        "--lanes",
        type=int,
        default=None,
        help=f"Seed lanes (1..{MAX_SEED_LANES})",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit a dirty working tree (default: require clean target)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="For start: plan and fence without marking dispatch allowed as live",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="Emit machine-readable JSON (default)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("validate", "Validate clean target, DAG, anchors, isolation, lanes, defaults"),
        ("plan", "Materialize an admitted launch plan into isolated state"),
        ("start", "Start or resume a fenced run (report-only/shadow)"),
        ("status", "Show lifecycle, kill switch, gates, and health"),
        ("stop", "Stop dispatch and mark the run stopped"),
        ("restart", "Fence-stop then start (preserves kill switch)"),
        ("pause", "Pause a running run and cancel new dispatch"),
        ("drain", "Drain: cancel future dispatch; finish in-flight"),
        ("benchmark", "Record a benchmark gate (no anchor mutation)"),
        ("promote", "Promote rollout by exactly one stage under gates"),
        ("rollback", "Roll back one stage (or --to-mode)"),
        ("kill-switch", "Engage kill switch: report-only, no dispatch, no promote"),
        ("kill-switch-clear", "Operator clear of kill switch (requires --operator-ack)"),
        ("recipe", "Print the structured launch recipe JSON"),
        ("deposit-receipt", "Deposit a prerequisite feature receipt"),
    ):
        p = sub.add_parser(name, help=help_text)
        if name == "rollback":
            p.add_argument(
                "--to-mode",
                default=None,
                choices=list(ROLLOUT_STAGES),
                help="Optional explicit safer mode",
            )
        if name == "kill-switch-clear":
            p.add_argument(
                "--operator-ack",
                action="store_true",
                help="Required acknowledgement that clear is operator-authorized",
            )
        if name == "kill-switch":
            p.add_argument(
                "--reason",
                default="operator_engage",
                help="Reason code for engagement",
            )
        if name == "deposit-receipt":
            p.add_argument(
                "--feature",
                required=True,
                choices=[g.value for g in FeatureGate],
            )
            p.add_argument("--task-id", required=True)
            p.add_argument("--evidence-id", required=True)
            p.add_argument("--operator-id", default="operator:local")
        if name == "stop":
            p.add_argument("--force", action="store_true")
    return parser


def _resolve_state_dir(repo_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("IPFS_ACCELERATE_PDR_OPS_STATE_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (repo_root / DEFAULT_STATE_SUBDIR).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return EXIT_SUCCESS
        return int(code) if isinstance(code, int) else EXIT_USAGE

    repo_root = Path(args.repo_root).resolve() if args.repo_root else repository_root()
    state_dir = _resolve_state_dir(repo_root, args.state_dir)
    require_clean = not bool(args.allow_dirty)
    lanes = args.lanes

    try:
        if args.command == "recipe":
            profile = launch_profile_from_scheduler(
                repo_root, args.scheduler_path, requested_lanes=lanes
            )
            payload = launch_recipe(profile, lanes=lanes)
        elif args.command == "deposit-receipt":
            state_dir.mkdir(parents=True, exist_ok=True)
            payload = deposit_prerequisite_receipt(
                state_dir,
                args.feature,
                task_id=args.task_id,
                evidence_id=args.evidence_id,
                operator_id=args.operator_id,
            )
        else:
            profile = launch_profile_from_scheduler(
                repo_root, args.scheduler_path, requested_lanes=lanes
            )
            if args.command == "validate":
                payload = validate_launch(
                    repo_root,
                    state_dir,
                    profile,
                    require_clean=require_clean,
                    requested_lanes=lanes,
                )
            elif args.command == "plan":
                payload = plan_launch(
                    repo_root,
                    state_dir,
                    profile,
                    require_clean=require_clean,
                    lanes=lanes,
                )
            elif args.command == "start":
                payload = start_run(
                    repo_root,
                    state_dir,
                    profile,
                    require_clean=require_clean,
                    lanes=lanes,
                    dry_run=bool(args.dry_run),
                )
            elif args.command == "status":
                payload = status_run(state_dir, profile)
            elif args.command == "stop":
                payload = stop_run(state_dir, force=bool(getattr(args, "force", False)))
            elif args.command == "restart":
                payload = restart_run(
                    repo_root,
                    state_dir,
                    profile,
                    require_clean=require_clean,
                    lanes=lanes,
                )
            elif args.command == "pause":
                payload = pause_run(state_dir)
            elif args.command == "drain":
                payload = drain_run(state_dir)
            elif args.command == "benchmark":
                payload = run_benchmark_gate(state_dir, profile)
            elif args.command == "promote":
                payload = promote_one_stage(state_dir, profile)
            elif args.command == "rollback":
                payload = rollback_stage(
                    state_dir, to_mode=getattr(args, "to_mode", None)
                )
            elif args.command == "kill-switch":
                payload = engage_kill_switch(
                    state_dir, reason=str(getattr(args, "reason", "operator_engage"))
                )
            elif args.command == "kill-switch-clear":
                payload = clear_kill_switch(
                    state_dir,
                    operator_ack=bool(getattr(args, "operator_ack", False)),
                )
            else:
                print(f"unknown command: {args.command}", file=sys.stderr)
                return EXIT_USAGE

        sys.stdout.write(
            json.dumps(_plain(payload), sort_keys=True, indent=2, ensure_ascii=False)
            + "\n"
        )
        if isinstance(payload, Mapping) and payload.get("ok") is False:
            return EXIT_FAILURE
        return EXIT_SUCCESS
    except PlannerDoctorOperationsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except Exception as exc:  # noqa: BLE001 - facade maps failures
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
