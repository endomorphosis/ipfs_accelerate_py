"""Classify campaign blockers and keep independent ready work moving.

A blocked task is one of eight kinds: dependency-running, software-defect,
source-outage, gated-input, credentials, permission, hold or exhausted-budget.
The kind is evidence, not a label the caller can swap. Independent ready tasks
stay in the continuation set. A blocked task is rechecked only when an admitted
probe is due or a prerequisite actually changed. Heartbeats, logs, pids,
attempts and the clock do not qualify.

This module does not start work, admit a repair, signal a process, refund a
budget or grant completion. Unknown effects stay unknown. A hold is not idle
work. Exhausted budget is visible and is not success.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .worldwide_legal_search_progress import (
    BUDGETS,
    DEFAULT_LIMITS,
    ProgressIdentityError,
    ProgressLedger,
    canonical_stage,
    digest,
    incident_identity,
)
from .worldwide_legal_search_recovery import HOLD_MARKERS

SCHEMA = "wwl.blocker.decision@1"
STATE_SCHEMA = "wwl.blocker-state@1"
PROBE_SCHEMA = "wwl.blocker.probe@1"
BLOCKER_KINDS = (
    "dependency-running",
    "software-defect",
    "source-outage",
    "gated-input",
    "credentials",
    "permission",
    "hold",
    "exhausted-budget",
)
EXTERNAL_KINDS = ("credentials", "permission", "gated-input", "source-outage")
EVIDENCE_PRECEDENCE = (
    "credentials",
    "permission",
    "gated-input",
    "source-outage",
    "exhausted-budget",
    "software-defect",
)
ADMITTED_RECHECK_REASONS = ("admitted_probe", "prerequisite_changed")
RECHECKABLE = frozenset({
    *BLOCKER_KINDS,
    "effects-unknown",
    "waiting-on-blocked-dependency",
    "waiting-on-dependency",
})
OUTAGE_STATUSES = (408, 429, 500, 502, 503, 504)
CREDENTIAL_STATUSES = (401,)
PERMISSION_STATUSES = (403,)
KIND_ALIASES = {
    "dependency-running": "dependency-running",
    "dependency_running": "dependency-running",
    "software-defect": "software-defect",
    "software_defect": "software-defect",
    "source-outage": "source-outage",
    "source_outage": "source-outage",
    "source-unavailable": "source-outage",
    "source_unavailable": "source-outage",
    "hf_outage": "source-outage",
    "gated-input": "gated-input",
    "gated_input": "gated-input",
    "gated": "gated-input",
    "gated_corpus": "gated-input",
    "credentials": "credentials",
    "credential": "credentials",
    "permission": "permission",
    "rights": "permission",
    "revoked_permission": "permission",
    "hold": "hold",
    "exhausted-budget": "exhausted-budget",
    "exhausted_budget": "exhausted-budget",
    "budget": "exhausted-budget",
    "budget_exhausted": "exhausted-budget",
    "none": "none",
    "ready": "none",
}
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REASON = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:/-]{0,160}$")
_SECRET_KEYS = frozenset({
    "secret", "token", "password", "api_key", "apikey", "authorization",
    "credential", "access_token", "private_key", "refresh_token",
})
_RESET_FIELDS = frozenset({
    "reset", "reset_budget", "reset_budgets", "reset_progress", "refund",
    "budgets_reset", "progress_reset",
})
_NOISE_FIELDS = frozenset({"pid", "heartbeat_at", "log_bytes", "attempt", "log", "observed_at"})
_BOARD_FIELDS = frozenset({
    "observed_at", "hold", "hold_markers", "probe_id", "tasks", "completion_authority",
    *_NOISE_FIELDS, *_RESET_FIELDS,
})
_TASK_FIELDS = frozenset({
    "task_id", "depends_on", "status", "blocked_kind", "kind", "hold", "hold_markers",
    "credential_ref", "credential_state", "permission_ref", "permission_state",
    "gate_ref", "gate_state", "source_ref", "source_state", "source_status",
    "failure_key", "reproducer_receipt", "reproducer", "budgets", "budget", "incident",
    "effects_unknown", "progress_token", "completion_authority",
    *_NOISE_FIELDS, *_RESET_FIELDS,
})
_STATUSES = frozenset({"ready", "running", "done", "blocked"})
_CREDENTIAL_STATES = frozenset({"", "missing", "invalid", "present"})
_PERMISSION_STATES = frozenset({"", "revoked", "denied", "restricted", "granted"})
_GATE_STATES = frozenset({"", "closed", "open"})
_SOURCE_STATES = frozenset({"", "outage", "available"})
_BLOCKING_CREDENTIALS = frozenset({"missing", "invalid"})
_BLOCKING_PERMISSIONS = frozenset({"revoked", "denied", "restricted"})
_MAX_TASKS = 256
_MAX_DEPENDENCIES = 32
_MAX_PROBES = 256
_MAX_REPRODUCER = 65536
_MAX_LOG = 4096
_MAX_CHECKPOINT = 1_048_576
_ACTIVE = frozenset({"running", "dependency-running"})
# dependency-running is active work, not an upstream failure. A waiter on that
# chain stays dependency-running and is not relabeled as a blocked dependency.
_BLOCKING_UPSTREAM = frozenset({
    *(kind for kind in BLOCKER_KINDS if kind != "dependency-running"),
    "effects-unknown",
    "waiting-on-blocked-dependency",
})


class BlockerContractError(ValueError):
    """Observation, probe or checkpoint violates the blocker contract."""


def canonical_blocker_kind(value: Any) -> str:
    """Map a watchdog alias onto one canonical kind. Unknown labels are refused."""
    if not isinstance(value, str) or not value.strip():
        raise BlockerContractError("blocker kind is not recognized")
    kind = KIND_ALIASES.get(value.strip().lower())
    if kind is None:
        raise BlockerContractError("blocker kind is not recognized")
    return kind


def reproducer_receipt(body: Any) -> str:
    """Digest the reproducer the classifier actually verified."""
    if isinstance(body, str):
        if len(body) > _MAX_REPRODUCER:
            raise BlockerContractError("reproducer exceeds the verification bound")
        return digest(body)
    if not isinstance(body, Mapping):
        raise BlockerContractError("reproducer must be an object or a string")
    _reject_secret_keys(body)
    try:
        encoded = json.dumps(dict(body), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise BlockerContractError("reproducer is not canonical JSON") from error
    if len(encoded) > _MAX_REPRODUCER:
        raise BlockerContractError("reproducer exceeds the verification bound")
    return digest(dict(body))


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise BlockerContractError(f"{name} must be a bounded durable identifier")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value.strip().lower()) is None:
        raise BlockerContractError(f"{name} must be a sha256 digest")
    return value.strip().lower()


def _timestamp(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BlockerContractError(f"{name} must be a finite timestamp")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")) or number < 0:
        raise BlockerContractError(f"{name} must be a finite timestamp")
    return number


def _whole(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise BlockerContractError(f"{name} must be an integer >= {minimum}")
    return value


def _reject_secret_keys(value: Mapping[str, Any]) -> None:
    found = {str(key).lower() for key in value} & _SECRET_KEYS
    if found:
        raise BlockerContractError("blocker evidence must not carry credential material")


def _reject_resets(value: Mapping[str, Any]) -> None:
    if value.get("completion_authority") is True:
        raise BlockerContractError("blocker classification cannot grant completion authority")
    present = _RESET_FIELDS.intersection(value)
    if any(value.get(field) not in (False, None) for field in present):
        raise BlockerContractError("blocker classification cannot reset a durable budget")


def _markers(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BlockerContractError("hold markers must be a list")
    markers = []
    for item in value:
        if not isinstance(item, str) or item not in HOLD_MARKERS:
            raise BlockerContractError("hold marker is not recognized")
        if item not in markers:
            markers.append(item)
    return markers


def _noise(value: Mapping[str, Any]) -> None:
    if "pid" in value and value["pid"] is not None:
        _whole(value["pid"], "pid", minimum=1)
    if "attempt" in value and value["attempt"] is not None:
        _whole(value["attempt"], "attempt")
    if "log_bytes" in value and value["log_bytes"] is not None:
        _whole(value["log_bytes"], "log_bytes")
    if "heartbeat_at" in value and value["heartbeat_at"] is not None:
        _timestamp(value["heartbeat_at"], "heartbeat_at")
    if "log" in value and value["log"] is not None:
        if not isinstance(value["log"], str) or len(value["log"]) > _MAX_LOG:
            raise BlockerContractError("log noise exceeds the observation bound")
    if "observed_at" in value and value["observed_at"] is not None:
        _timestamp(value["observed_at"], "observed_at")


class BlockerBoard:
    """Durable blocker book for one campaign run.

    The optional progress ledger is the budget authority. An observation cannot
    under-report it, and exhaustion that did not come from the ledger stays
    exhausted. Probe admission and classification do not spend a budget.
    """

    def __init__(self, run_id: str, *, ledger: ProgressLedger | None = None):
        self.run_id = _identifier(run_id, "run_id")
        if ledger is not None:
            if not isinstance(ledger, ProgressLedger):
                raise BlockerContractError("blocker ledger must be a progress ledger")
            if ledger.run_id != self.run_id:
                raise BlockerContractError("blocker run does not match the progress ledger")
            if any(ledger.limits.get(name) != DEFAULT_LIMITS[name] for name in BUDGETS):
                raise BlockerContractError("durable budget differs from the campaign ledger")
        self.ledger = ledger
        self.records: dict[str, dict[str, Any]] = {}
        self.probes: dict[str, dict[str, Any]] = {}
        self.completion_authority = False

    def admit_probe(self, *, task_id: str, probe_id: str, not_before: Any, reason: str) -> dict[str, Any]:
        """Admit one future recheck. Admission does not resume the task."""
        task = _identifier(task_id, "task_id")
        probe = _identifier(probe_id, "probe_id")
        when = _timestamp(not_before, "not_before")
        if not isinstance(reason, str) or _REASON.fullmatch(reason) is None:
            raise BlockerContractError("probe reason must be bounded text")
        record = self.records.get(task)
        if record is None or not record["recheckable"]:
            raise BlockerContractError("admitted probe requires a classified blocker")
        if when < record["classified_at"]:
            raise BlockerContractError("admitted probe predates the classification")
        if probe in self.probes:
            raise BlockerContractError("admitted probe already exists")
        if any(item["task_id"] == task and not item["consumed"] for item in self.probes.values()):
            raise BlockerContractError("task already has an admitted probe")
        if len(self.probes) >= _MAX_PROBES:
            raise BlockerContractError("admitted probe book is full")
        body = {
            "schema": PROBE_SCHEMA,
            "probe_id": probe,
            "task_id": task,
            "not_before": when,
            "reason": reason,
            "consumed": False,
            "consumed_at": None,
            "completion_authority": False,
        }
        self.probes[probe] = body
        return dict(body)

    def classify(self, board: Mapping[str, Any]) -> dict[str, Any]:
        """Classify every task. Independent ready work is continued immediately."""
        if not isinstance(board, Mapping):
            raise BlockerContractError("blocker board must be an object")
        normalized = self._normalize_board(board)
        tasks = normalized["tasks"]
        order = _topological(tasks)
        derived: dict[str, dict[str, Any]] = {}
        for task_id in order:
            task = tasks[task_id]
            # A dropped observation budget is a refund even when the task is still
            # blocked and would otherwise have no distinguished label.
            _guard_budget(
                self.records.get(task_id),
                {"exhausted_budgets": task["exhausted_budgets"], "incident_key": task["incident_key"]},
                self.ledger,
            )
            derived[task_id] = _derive(task, derived, campaign_hold=normalized["held"])
        probe = self._due_probe(normalized)
        records = {task_id: _copy_record(record) for task_id, record in self.records.items()}
        probes = {probe_id: dict(item) for probe_id, item in self.probes.items()}
        rows = []
        consumed = False
        for task_id in sorted(tasks):
            row, consumed_now = _apply_recheck(
                records, probes, tasks[task_id], derived[task_id], normalized, probe, self.ledger,
            )
            consumed = consumed or consumed_now
            rows.append(row)
        _invariants(tasks, rows, campaign_hold=normalized["held"])
        decision = _decision(self.run_id, normalized, rows, consumed)
        self.records = records
        self.probes = probes
        return decision

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema": STATE_SCHEMA,
            "run_id": self.run_id,
            "records": [_public_record(self.records[task_id]) for task_id in sorted(self.records)],
            "probes": [_public_probe(self.probes[probe_id]) for probe_id in sorted(self.probes)],
            "completion_authority": False,
        }
        return {**body, "digest": digest(body)}

    def save(self, path: str | Path) -> None:
        """Atomically replace a checkpoint. A crash cannot drop a blocker."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
        descriptor, temporary = tempfile.mkstemp(prefix=".wwl-blockers-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str | Path, *, ledger: ProgressLedger | None = None) -> BlockerBoard:
        raw = Path(path).read_bytes()
        if len(raw) > _MAX_CHECKPOINT:
            raise BlockerContractError("blocker checkpoint exceeds the admission bound")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise BlockerContractError("blocker checkpoint is not JSON") from error
        return cls.from_dict(value, ledger=ledger)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, ledger: ProgressLedger | None = None) -> BlockerBoard:
        if not isinstance(value, Mapping) or value.get("schema") != STATE_SCHEMA:
            raise BlockerContractError("blocker checkpoint schema is not recognized")
        if value.get("completion_authority") is not False:
            raise BlockerContractError("blocker checkpoint cannot grant completion authority")
        body = {key: value[key] for key in ("schema", "run_id", "records", "probes", "completion_authority")}
        if set(value) - set(body) - {"digest"}:
            raise BlockerContractError("blocker checkpoint has unknown fields")
        if value.get("digest") != digest(body):
            raise BlockerContractError("blocker checkpoint digest does not match")
        board = cls(str(value.get("run_id") or ""), ledger=ledger)
        records = value.get("records")
        probes = value.get("probes")
        if not isinstance(records, list) or not isinstance(probes, list):
            raise BlockerContractError("blocker checkpoint records are malformed")
        for item in records:
            record = _restore_record(item)
            if record["task_id"] in board.records:
                raise BlockerContractError("blocker checkpoint repeats a task")
            board.records[record["task_id"]] = record
        for item in probes:
            probe = _restore_probe(item, board.records)
            if probe["probe_id"] in board.probes:
                raise BlockerContractError("blocker checkpoint repeats a probe")
            board.probes[probe["probe_id"]] = probe
        open_for: dict[str, str] = {}
        for probe in board.probes.values():
            if probe["consumed"]:
                continue
            if probe["task_id"] in open_for:
                raise BlockerContractError("task already has an admitted probe")
            open_for[probe["task_id"]] = probe["probe_id"]
        return board

    def _normalize_board(self, board: Mapping[str, Any]) -> dict[str, Any]:
        _reject_secret_keys(board)
        unknown = set(board) - _BOARD_FIELDS
        if unknown:
            raise BlockerContractError("blocker board has unknown fields")
        _reject_resets(board)
        _noise(board)
        observed_at = _timestamp(board.get("observed_at"), "observed_at")
        if "hold" in board and type(board.get("hold")) is not bool:
            raise BlockerContractError("hold must be a boolean")
        markers = _markers(board.get("hold_markers"))
        held = board.get("hold") is True or bool(markers)
        if held and not markers:
            markers = ["HOLD"]
        raw_tasks = board.get("tasks")
        if not isinstance(raw_tasks, list) or len(raw_tasks) > _MAX_TASKS:
            raise BlockerContractError("blocker board tasks are malformed")
        tasks: dict[str, dict[str, Any]] = {}
        for raw in raw_tasks:
            task = self._normalize_task(raw, campaign_hold=held)
            if task["task_id"] in tasks:
                raise BlockerContractError("blocker board repeats a task")
            tasks[task["task_id"]] = task
        probe_id = board.get("probe_id") or ""
        if probe_id:
            probe_id = _identifier(probe_id, "probe_id")
            if probe_id not in self.probes:
                raise BlockerContractError("probe is not admitted")
        return {
            "observed_at": observed_at,
            "held": held,
            "hold_markers": markers,
            "probe_id": probe_id,
            "tasks": tasks,
        }

    def _normalize_task(self, raw: Any, *, campaign_hold: bool) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise BlockerContractError("task must be an object")
        _reject_secret_keys(raw)
        if set(raw) - _TASK_FIELDS:
            raise BlockerContractError("task has unknown fields")
        _reject_resets(raw)
        _noise(raw)
        task_id = _identifier(raw.get("task_id"), "task_id")
        status = raw.get("status")
        if status not in _STATUSES:
            raise BlockerContractError("task status is not recognized")
        depends_on = _dependencies(raw.get("depends_on"))
        if task_id in depends_on:
            raise BlockerContractError("dependency cycle")
        hold_markers = _markers(raw.get("hold_markers"))
        if "hold" in raw and type(raw.get("hold")) is not bool:
            raise BlockerContractError("hold must be a boolean")
        held = raw.get("hold") is True or bool(hold_markers)
        if held and not hold_markers:
            hold_markers = ["HOLD"]
        if raw.get("effects_unknown") not in (None, True, False):
            raise BlockerContractError("effects_unknown must be a boolean")
        effects_unknown = raw.get("effects_unknown") is True
        evidence, exhausted, exhaustion_source, incident_key = self._evidence(raw, task_id)
        failure_key, receipt = _reproducer(raw)
        if failure_key:
            evidence_kinds = list(evidence["kinds"])
            if "software-defect" not in evidence_kinds:
                evidence["kinds"].append("software-defect")
            evidence["failure_key"] = failure_key
            evidence["reproducer_receipt"] = receipt
        progress_token = raw.get("progress_token") or ""
        if progress_token:
            progress_token = _digest(progress_token, "progress_token")
        if status == "running" and not progress_token:
            raise BlockerContractError("running work requires a progress token")
        if status == "done" and (evidence["kinds"] or effects_unknown or held or exhausted):
            raise BlockerContractError("done work cannot carry a blocker")
        if status != "blocked" and not campaign_hold and (evidence["kinds"] or effects_unknown or held or exhausted):
            raise BlockerContractError("ready work cannot carry blocker evidence")
        claimed = _claimed_kind(raw)
        if "budget" in raw and raw.get("budget") is not None:
            if not isinstance(raw.get("budget"), str) or raw.get("budget") not in BUDGETS:
                raise BlockerContractError("budget claim is not a campaign budget")
            if raw["budget"] not in exhausted:
                raise BlockerContractError("budget claim does not match exhaustion")
        return {
            "task_id": task_id,
            "status": status,
            "depends_on": depends_on,
            "hold": held,
            "hold_markers": hold_markers,
            "effects_unknown": effects_unknown,
            "kinds": evidence["kinds"],
            "credential_ref": evidence["credential_ref"],
            "credential_state": evidence["credential_state"],
            "permission_ref": evidence["permission_ref"],
            "permission_state": evidence["permission_state"],
            "gate_ref": evidence["gate_ref"],
            "gate_state": evidence["gate_state"],
            "source_ref": evidence["source_ref"],
            "source_state": evidence["source_state"],
            "source_status": evidence["source_status"],
            "failure_key": evidence["failure_key"],
            "reproducer_receipt": evidence["reproducer_receipt"],
            "exhausted_budgets": exhausted,
            "exhaustion_source": exhaustion_source,
            "incident_key": incident_key,
            "progress_token": progress_token,
            "claimed_kind": claimed,
        }

    def _evidence(self, raw: Mapping[str, Any], task_id: str) -> tuple[dict[str, Any], list[str], str, str]:
        credential_state = _enum(raw.get("credential_state") or "", _CREDENTIAL_STATES, "credential_state")
        permission_state = _enum(raw.get("permission_state") or "", _PERMISSION_STATES, "permission_state")
        gate_state = _enum(raw.get("gate_state") or "", _GATE_STATES, "gate_state")
        source_state = _enum(raw.get("source_state") or "", _SOURCE_STATES, "source_state")
        source_status = _whole(raw.get("source_status") or 0, "source_status")
        credential_ref = _optional_id(raw.get("credential_ref") or "", "credential_ref")
        permission_ref = _optional_id(raw.get("permission_ref") or "", "permission_ref")
        gate_ref = _optional_id(raw.get("gate_ref") or "", "gate_ref")
        source_ref = _optional_id(raw.get("source_ref") or "", "source_ref")
        if source_status in CREDENTIAL_STATUSES:
            if credential_state == "present" or source_state == "available":
                raise BlockerContractError("source status conflicts with credential evidence")
            credential_state = credential_state or "missing"
            credential_ref = credential_ref or source_ref
        elif source_status in PERMISSION_STATUSES:
            if permission_state == "granted" or source_state == "available":
                raise BlockerContractError("source status conflicts with permission evidence")
            permission_state = permission_state or "revoked"
            permission_ref = permission_ref or source_ref
        elif source_status in OUTAGE_STATUSES:
            if source_state == "available":
                raise BlockerContractError("source status conflicts with source evidence")
            source_state = "outage"
        elif source_status and source_status not in range(200, 300):
            raise BlockerContractError("source status is not a distinguished blocker")
        elif 200 <= source_status < 300 and source_state == "outage":
            raise BlockerContractError("source status conflicts with source evidence")
        kinds: list[str] = []
        if credential_state in _BLOCKING_CREDENTIALS:
            if not credential_ref:
                raise BlockerContractError("credential blocker requires a credential ref")
            kinds.append("credentials")
        if permission_state in _BLOCKING_PERMISSIONS:
            if not permission_ref:
                raise BlockerContractError("permission blocker requires a permission ref")
            kinds.append("permission")
        if gate_state == "closed":
            if not gate_ref:
                raise BlockerContractError("gated input requires a gate ref")
            kinds.append("gated-input")
        if source_state == "outage":
            if not source_ref:
                raise BlockerContractError("source outage requires a source ref")
            kinds.append("source-outage")
        exhausted, source, incident_key = self._exhaustion(raw, task_id)
        if exhausted and "exhausted-budget" not in kinds:
            kinds.append("exhausted-budget")
        ordered = [kind for kind in EVIDENCE_PRECEDENCE if kind in kinds]
        return {
            "kinds": ordered,
            "credential_ref": credential_ref,
            "credential_state": credential_state,
            "permission_ref": permission_ref,
            "permission_state": permission_state,
            "gate_ref": gate_ref,
            "gate_state": gate_state,
            "source_ref": source_ref,
            "source_state": source_state,
            "source_status": source_status,
            "failure_key": "",
            "reproducer_receipt": "",
        }, exhausted, source, incident_key

    def _exhaustion(self, raw: Mapping[str, Any], task_id: str) -> tuple[list[str], str, str]:
        incident_key = _incident_key(self.run_id, task_id, raw.get("incident"))
        observed = _observed_budgets(raw.get("budgets"))
        if self.ledger is not None and incident_key and incident_key in self.ledger.incidents:
            stored = self.ledger.incidents[incident_key]
            for name, row in observed.items():
                if row["spent"] != stored["spent"][name] or row["limit"] != stored["limits"][name]:
                    raise BlockerContractError("budget restatement does not match the progress ledger")
            exhausted = [name for name in BUDGETS if stored["spent"][name] >= stored["limits"][name]]
            return exhausted, ("ledger" if exhausted else ""), incident_key
        if observed:
            for name, row in observed.items():
                if row["limit"] != DEFAULT_LIMITS[name]:
                    raise BlockerContractError("budget limit differs from the campaign ledger")
                if row["spent"] > row["limit"]:
                    raise BlockerContractError("budget spend exceeds the durable limit")
            exhausted = [name for name, row in observed.items() if row["spent"] >= row["limit"]]
            return exhausted, ("observation" if exhausted else ""), incident_key
        return [], "", incident_key

    def _due_probe(self, normalized: Mapping[str, Any]) -> dict[str, Any] | None:
        probe_id = normalized["probe_id"]
        if not probe_id:
            return None
        probe = self.probes[probe_id]
        if probe["task_id"] not in normalized["tasks"]:
            raise BlockerContractError("probe task is not on the board")
        if probe["consumed"]:
            raise BlockerContractError("admitted probe was already consumed")
        return probe


def _dependencies(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > _MAX_DEPENDENCIES:
        raise BlockerContractError("depends_on is malformed")
    names = [_identifier(item, "task_id") for item in value]
    if len(names) != len(set(names)):
        raise BlockerContractError("depends_on repeats a task")
    return names


def _enum(value: Any, allowed: frozenset[str], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise BlockerContractError(f"{name} is not recognized")
    return value


def _optional_id(value: Any, name: str) -> str:
    if value in ("", None):
        return ""
    return _identifier(value, name)


def _claimed_kind(raw: Mapping[str, Any]) -> str:
    claims = []
    for field in ("blocked_kind", "kind"):
        if field in raw and raw[field] not in (None, ""):
            claims.append(canonical_blocker_kind(raw[field]))
    if not claims:
        return ""
    if any(item != claims[0] for item in claims):
        raise BlockerContractError("blocker kind claims do not match")
    return claims[0]


def _reproducer(raw: Mapping[str, Any]) -> tuple[str, str]:
    failure = raw.get("failure_key") or ""
    receipt = raw.get("reproducer_receipt") or ""
    body = raw.get("reproducer")
    if failure in ("", None) and receipt in ("", None) and body is None:
        return "", ""
    if not failure or not receipt or body is None:
        raise BlockerContractError("software defect requires a verified reproducer")
    failure = _identifier(failure, "failure_key")
    expected = reproducer_receipt(body)
    if _digest(receipt, "reproducer_receipt") != expected:
        raise BlockerContractError("reproducer receipt does not match the reproducer")
    if isinstance(body, Mapping) and body.get("failure_key") not in (None, failure):
        raise BlockerContractError("reproducer failure key does not match")
    return failure, expected


def _observed_budgets(value: Any) -> dict[str, dict[str, int]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise BlockerContractError("budgets must be an object")
    unknown = set(value) - set(BUDGETS)
    if unknown:
        raise BlockerContractError("unknown repair budget")
    observed = {}
    for name, row in value.items():
        if not isinstance(row, Mapping) or set(row) - {"spent", "limit"}:
            raise BlockerContractError("budget row is malformed")
        observed[name] = {"spent": _whole(row.get("spent"), "spent"), "limit": _whole(row.get("limit"), "limit", minimum=1)}
    return observed


def _incident_key(run_id: str, task_id: str, incident: Any) -> str:
    if incident is None:
        return ""
    if not isinstance(incident, Mapping) or set(incident) - {"stage", "defect", "incident_id"}:
        raise BlockerContractError("incident binding is malformed")
    try:
        stage = canonical_stage(incident.get("stage"))
        key = incident_identity(run_id=run_id, task_id=task_id, stage=stage, defect=incident.get("defect", ""))
    except ProgressIdentityError as error:
        raise BlockerContractError(str(error)) from error
    claimed = incident.get("incident_id")
    if claimed not in (None, "") and claimed != key:
        raise BlockerContractError("incident id does not match the durable incident identity")
    return key


def _topological(tasks: Mapping[str, Mapping[str, Any]]) -> list[str]:
    visiting: set[str] = set()
    visited: set[str] = set()
    order: list[str] = []

    def walk(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            raise BlockerContractError("dependency cycle")
        visiting.add(task_id)
        for dependency in tasks[task_id]["depends_on"]:
            if dependency not in tasks:
                raise BlockerContractError("dependency names an unknown task")
            walk(dependency)
        visiting.remove(task_id)
        visited.add(task_id)
        order.append(task_id)

    for task_id in sorted(tasks):
        walk(task_id)
    return order


def _derive(task: Mapping[str, Any], done: Mapping[str, Mapping[str, Any]], *, campaign_hold: bool) -> dict[str, Any]:
    unmet = [dependency for dependency in task["depends_on"] if done[dependency]["classification"] != "done"]
    active = [dependency for dependency in unmet if done[dependency]["classification"] in _ACTIVE]
    blocked = [dependency for dependency in unmet if done[dependency]["classification"] in _BLOCKING_UPSTREAM]
    evidence = list(task["kinds"])
    if campaign_hold and task["status"] != "done":
        classification = "hold"
        matched = ["hold", *evidence]
    elif task["hold"]:
        classification = "hold"
        matched = ["hold", *evidence]
    elif task["effects_unknown"]:
        classification = "effects-unknown"
        matched = list(evidence)
    elif evidence:
        classification = evidence[0]
        matched = list(evidence)
    elif not unmet and task["status"] == "done":
        classification = "done"
        matched = []
    elif not unmet and task["status"] == "running":
        classification = "running"
        matched = []
    elif not unmet and task["status"] == "ready":
        classification = "ready"
        matched = []
    elif unmet and blocked:
        classification = "waiting-on-blocked-dependency"
        matched = []
    elif unmet and len(active) == len(unmet):
        classification = "dependency-running"
        matched = ["dependency-running"]
    elif unmet:
        classification = "waiting-on-dependency"
        matched = []
    else:
        raise BlockerContractError("blocked task has no distinguished blocker")
    _status_allows(task["status"], classification, campaign_hold=campaign_hold)
    claimed = task["claimed_kind"]
    if claimed == "none" and classification not in {"ready", "running", "done"}:
        raise BlockerContractError("blocker kind does not match the evidence")
    if claimed not in {"", "none"} and claimed != classification:
        raise BlockerContractError("blocker kind does not match the evidence")
    chain = _chain(task["task_id"], done, task["depends_on"])
    progress_identity = {
        "token": task["progress_token"],
        "upstream": [done[dependency]["progress_identity"] for dependency in task["depends_on"]],
    }
    facts = {
        "task_id": task["task_id"],
        "depends_on": list(task["depends_on"]),
        "status": task["status"],
        "campaign_hold": campaign_hold and task["status"] != "done",
        "hold": task["hold"],
        "hold_markers": list(task["hold_markers"]),
        "credential_ref": task["credential_ref"],
        "credential_state": task["credential_state"],
        "permission_ref": task["permission_ref"],
        "permission_state": task["permission_state"],
        "gate_ref": task["gate_ref"],
        "gate_state": task["gate_state"],
        "source_ref": task["source_ref"],
        "source_state": task["source_state"],
        "source_status": task["source_status"],
        "failure_key": task["failure_key"],
        "reproducer_receipt": task["reproducer_receipt"],
        "exhausted_budgets": list(task["exhausted_budgets"]),
        "effects_unknown": task["effects_unknown"],
        "incident_key": task["incident_key"],
        "dependencies": [
            {
                "task_id": dependency,
                "classification": done[dependency]["classification"],
                "progress": done[dependency]["progress_identity"],
            }
            for dependency in task["depends_on"]
        ],
    }
    return {
        "classification": classification,
        "matched_kinds": matched,
        "dependency_chain": chain,
        "blocked_dependencies": blocked,
        "facts": facts,
        "progress_token": task["progress_token"],
        "progress_identity": progress_identity,
        "exhausted_budgets": list(task["exhausted_budgets"]),
        "exhaustion_source": task["exhaustion_source"] if task["exhausted_budgets"] else "",
        "incident_key": task["incident_key"],
        "effects_unknown": task["effects_unknown"],
    }


def _status_allows(status: str, classification: str, *, campaign_hold: bool) -> None:
    if status == "done" and classification != "done":
        raise BlockerContractError("done work cannot carry a blocker")
    if status == "running" and classification not in {"running", "hold"}:
        raise BlockerContractError("running work cannot carry blocker evidence")
    if status == "blocked" and classification in {"ready", "running", "done", "waiting-on-dependency"}:
        raise BlockerContractError("blocked task has no distinguished blocker")
    if classification == "hold" and not campaign_hold and status != "blocked":
        raise BlockerContractError("hold marker requires a blocked task")
    if classification in EVIDENCE_PRECEDENCE and status != "blocked":
        raise BlockerContractError("ready work cannot carry blocker evidence")
    if classification == "effects-unknown" and status != "blocked":
        raise BlockerContractError("unknown effects require a blocked task")
    if classification == "ready" and status != "ready":
        raise BlockerContractError("status does not match the blocker classification")
    if classification == "running" and status != "running":
        raise BlockerContractError("status does not match the blocker classification")
    if classification == "waiting-on-dependency" and status != "ready":
        raise BlockerContractError("status does not match the blocker classification")


def _chain(task_id: str, done: Mapping[str, Mapping[str, Any]], depends_on: Sequence[str]) -> list[str]:
    seen: list[str] = []

    def walk(current: str) -> None:
        dependencies = depends_on if current == task_id else []
        if current != task_id:
            return
        for dependency in dependencies:
            if done[dependency]["classification"] not in _ACTIVE or dependency in seen:
                continue
            seen.append(dependency)
            _walk_active(dependency, done, seen)

    walk(task_id)
    return seen


def _walk_active(task_id: str, done: Mapping[str, Mapping[str, Any]], seen: list[str]) -> None:
    # The stored chain on a dependency is already the transitive active set.
    for dependency in done[task_id]["dependency_chain"]:
        if dependency not in seen:
            seen.append(dependency)


def _apply_recheck(
    records: dict[str, dict[str, Any]],
    probes: dict[str, dict[str, Any]],
    task: Mapping[str, Any],
    derived: Mapping[str, Any],
    board: Mapping[str, Any],
    probe: Mapping[str, Any] | None,
    ledger: ProgressLedger | None,
) -> tuple[dict[str, Any], bool]:
    task_id = task["task_id"]
    facts_digest = digest(derived["facts"])
    previous = records.get(task_id)
    classification = str(derived["classification"])
    recheckable = classification in RECHECKABLE
    if previous is not None and previous["prerequisite_digest"] == facts_digest and previous["classification"] != classification:
        raise BlockerContractError("classification drifted without a prerequisite change")
    _guard_budget(previous, derived, ledger)
    digest_changed = previous is not None and previous["prerequisite_digest"] != facts_digest
    due = bool(
        probe is not None
        and probe["task_id"] == task_id
        and board["observed_at"] >= probe["not_before"]
    )
    engaged = recheckable or (previous is not None and previous["recheckable"])
    if due and not engaged:
        raise BlockerContractError("admitted probe requires a classified blocker")
    if previous is None:
        recheck_admitted = False
        reason = "initial" if recheckable else "not_blocked"
    elif due or (digest_changed and engaged):
        recheck_admitted = True
        reason = "prerequisite_changed" if digest_changed else "admitted_probe"
    else:
        recheck_admitted = False
        reason = "prerequisite_unchanged" if engaged else "not_blocked"
    consumed = False
    if due and recheck_admitted:
        probes[str(probe["probe_id"])]["consumed"] = True
        probes[str(probe["probe_id"])]["consumed_at"] = board["observed_at"]
        consumed = True
    classified_at = board["observed_at"] if previous is None or recheck_admitted else previous["classified_at"]
    records[task_id] = {
        "task_id": task_id,
        "classification": classification,
        "kind": classification if classification in BLOCKER_KINDS else "",
        "prerequisite_digest": facts_digest,
        "exhausted_budgets": list(derived["exhausted_budgets"]),
        "exhaustion_source": derived["exhaustion_source"],
        "incident_key": derived["incident_key"],
        "classified_at": classified_at,
        "recheckable": recheckable,
    }
    kind = classification if classification in BLOCKER_KINDS else ""
    row = {
        "task_id": task_id,
        "classification": classification,
        "kind": kind,
        "blocker": kind in BLOCKER_KINDS and kind != "dependency-running",
        "external": kind in EXTERNAL_KINDS,
        "failure": kind == "software-defect",
        "held": kind == "hold",
        "dependency_running": kind == "dependency-running",
        "matched_kinds": list(derived["matched_kinds"]),
        "continues": classification == "ready",
        "recheck_admitted": recheck_admitted,
        "recheck_reason": reason,
        "prerequisite_digest": facts_digest,
        "dependency_chain": list(derived["dependency_chain"]),
        "blocked_dependencies": list(derived["blocked_dependencies"]),
        "exhausted_budgets": list(derived["exhausted_budgets"]),
        "effects_unknown": bool(derived["effects_unknown"]),
        "idle_failure": False,
        "recoverable_idleness": False,
        "success": False,
        "repair_admitted": False,
        "lifecycle_mutation": False,
        "completion_authority": False,
    }
    return row, consumed


def _guard_budget(
    previous: Mapping[str, Any] | None,
    derived: Mapping[str, Any],
    ledger: ProgressLedger | None,
) -> None:
    """Observation-sourced exhaustion is sticky.

    A ledger-sourced incident can stop being the task's blocker only when the
    ledger already has a different incident that does not carry those budgets.
    Omitting the spend, or renaming the defect without the ledger, is a refund.
    """
    if previous is None:
        return
    lost = [name for name in previous["exhausted_budgets"] if name not in derived["exhausted_budgets"]]
    if not lost:
        return
    incident_key = derived["incident_key"]
    if (
        previous["exhaustion_source"] != "ledger"
        or ledger is None
        or not incident_key
        or incident_key == previous["incident_key"]
        or incident_key not in ledger.incidents
    ):
        raise BlockerContractError("exhausted budget cannot be refunded")


def _invariants(tasks: Mapping[str, Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], *, campaign_hold: bool) -> None:
    by_id = {row["task_id"]: row for row in rows}
    continued = [row["task_id"] for row in rows if row["continues"]]
    blocked = [row["task_id"] for row in rows if row["blocker"]]
    if set(continued) & set(blocked):
        raise BlockerContractError("independent continuation overlaps a blocker")
    if campaign_hold and continued:
        raise BlockerContractError("a hold continued ready work")
    for task_id in continued:
        for dependency in tasks[task_id]["depends_on"]:
            if by_id[dependency]["classification"] != "done":
                raise BlockerContractError("continuation depends on unfinished work")
    for row in rows:
        if row["success"] or row["completion_authority"] or row["repair_admitted"] or row["lifecycle_mutation"]:
            raise BlockerContractError("blocker classification granted authority")
        if row["recheck_admitted"] and row["recheck_reason"] not in ADMITTED_RECHECK_REASONS:
            raise BlockerContractError("recheck was not admitted")
        if row["recheck_reason"] in ADMITTED_RECHECK_REASONS and not row["recheck_admitted"]:
            raise BlockerContractError("recheck was not admitted")
        if row["dependency_running"] and (row["idle_failure"] or row["continues"] or row["blocker"]):
            raise BlockerContractError("dependency-running was treated as idle failure")
        if row["held"] and (row["recoverable_idleness"] or row["continues"]):
            raise BlockerContractError("hold was treated as recoverable idleness")


def _decision(run_id: str, board: Mapping[str, Any], rows: list[dict[str, Any]], probe_consumed: bool) -> dict[str, Any]:
    body = {
        "schema": SCHEMA,
        "run_id": run_id,
        "observed_at": board["observed_at"],
        "held": board["held"],
        "hold_markers": list(board["hold_markers"]),
        "tasks": rows,
        "continue_task_ids": [row["task_id"] for row in rows if row["continues"]],
        "blocked_task_ids": [row["task_id"] for row in rows if row["blocker"]],
        "dependency_running_task_ids": [row["task_id"] for row in rows if row["dependency_running"]],
        "running_task_ids": [row["task_id"] for row in rows if row["classification"] == "running"],
        "rechecked_task_ids": [row["task_id"] for row in rows if row["recheck_admitted"]],
        "unresolved_task_ids": [row["task_id"] for row in rows if row["classification"] == "effects-unknown"],
        "probe_id": board["probe_id"],
        "probe_consumed": probe_consumed,
        "repair_admitted": False,
        "lifecycle_mutation": False,
        "execution_completed": False,
        "success": False,
        "completion_authority": False,
    }
    return {**body, "digest": digest(body)}


def _copy_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {**record, "exhausted_budgets": list(record["exhausted_budgets"])}


def _public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_id": record["task_id"],
        "classification": record["classification"],
        "kind": record["kind"],
        "prerequisite_digest": record["prerequisite_digest"],
        "exhausted_budgets": list(record["exhausted_budgets"]),
        "exhaustion_source": record["exhaustion_source"],
        "incident_key": record["incident_key"],
        "classified_at": record["classified_at"],
        "recheckable": record["recheckable"],
    }


def _public_probe(probe: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": PROBE_SCHEMA,
        "probe_id": probe["probe_id"],
        "task_id": probe["task_id"],
        "not_before": probe["not_before"],
        "reason": probe["reason"],
        "consumed": probe["consumed"],
        "consumed_at": probe["consumed_at"],
        "completion_authority": False,
    }


def _restore_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BlockerContractError("stored blocker record is malformed")
    expected = {
        "task_id", "classification", "kind", "prerequisite_digest", "exhausted_budgets",
        "exhaustion_source", "incident_key", "classified_at", "recheckable",
    }
    if set(value) != expected:
        raise BlockerContractError("stored blocker record is malformed")
    classification = value.get("classification")
    allowed = set(BLOCKER_KINDS) | {"ready", "running", "done", "effects-unknown", "waiting-on-blocked-dependency", "waiting-on-dependency"}
    if classification not in allowed:
        raise BlockerContractError("stored blocker classification is not recognized")
    kind = value.get("kind")
    if kind not in {"", *BLOCKER_KINDS} or (classification in BLOCKER_KINDS and kind != classification) or (classification not in BLOCKER_KINDS and kind != ""):
        raise BlockerContractError("stored blocker kind does not match its classification")
    budgets = value.get("exhausted_budgets")
    if not isinstance(budgets, list) or any(item not in BUDGETS for item in budgets) or len(budgets) != len(set(budgets)):
        raise BlockerContractError("stored exhausted budgets are malformed")
    source = value.get("exhaustion_source")
    if source not in {"", "ledger", "observation"}:
        raise BlockerContractError("stored exhaustion source is not recognized")
    if bool(budgets) != bool(source):
        raise BlockerContractError("stored exhaustion source does not match the budgets")
    incident_key = value.get("incident_key")
    if incident_key not in ("",) and (not isinstance(incident_key, str) or _DIGEST.fullmatch(incident_key) is None):
        raise BlockerContractError("stored incident key is malformed")
    if type(value.get("recheckable")) is not bool or value["recheckable"] != (classification in RECHECKABLE):
        raise BlockerContractError("stored recheck flag does not match the classification")
    return {
        "task_id": _identifier(value.get("task_id"), "task_id"),
        "classification": classification,
        "kind": kind,
        "prerequisite_digest": _digest(value.get("prerequisite_digest"), "prerequisite_digest"),
        "exhausted_budgets": list(budgets),
        "exhaustion_source": source,
        "incident_key": incident_key or "",
        "classified_at": _timestamp(value.get("classified_at"), "classified_at"),
        "recheckable": value["recheckable"],
    }


def _restore_probe(value: Any, records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("schema") != PROBE_SCHEMA or value.get("completion_authority") is not False:
        raise BlockerContractError("stored probe is malformed")
    expected = {"schema", "probe_id", "task_id", "not_before", "reason", "consumed", "consumed_at", "completion_authority"}
    if set(value) != expected or type(value.get("consumed")) is not bool:
        raise BlockerContractError("stored probe is malformed")
    if not isinstance(value.get("reason"), str) or _REASON.fullmatch(value["reason"]) is None:
        raise BlockerContractError("stored probe reason is malformed")
    task_id = _identifier(value.get("task_id"), "task_id")
    if task_id not in records:
        raise BlockerContractError("stored probe requires a classified blocker")
    consumed_at = value.get("consumed_at")
    if value["consumed"]:
        consumed_at = _timestamp(consumed_at, "consumed_at")
    elif consumed_at is not None:
        raise BlockerContractError("stored probe consumption does not match")
    return {
        "schema": PROBE_SCHEMA,
        "probe_id": _identifier(value.get("probe_id"), "probe_id"),
        "task_id": task_id,
        "not_before": _timestamp(value.get("not_before"), "not_before"),
        "reason": value["reason"],
        "consumed": value["consumed"],
        "consumed_at": consumed_at,
        "completion_authority": False,
    }
