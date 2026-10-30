"""Meaningful progress cursor and durable incident identity.

Progress is an accepted artifact change, a canonical task transition, a
validation-stage advance, or a verified merge. Log growth, heartbeat updates,
PID changes and repeated retries of the same stage do not move the cursor and
do not refund provider, task, validation, merge, recover or repair budgets.

An incident id binds the run, the task, the canonical stage and the normalized
defect. Process ids, attempt numbers, heartbeats and log text are not part of
that identity, so a restart cannot open a fresh budget for the same defect.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "wwl.progress.decision@1"
STATE_SCHEMA = "wwl.meaningful-progress-state@1"
CURSOR_SCHEMA = "wwl.progress-cursor@1"
INCIDENT_SCHEMA = "wwl.incident-identity@1"

CANONICAL_STAGES = ("task", "validation", "merge")
STAGE_ORDER = {"task": 0, "validation": 1, "merge": 2}
STAGE_ALIASES = {
    "task": "task",
    "implementation": "task",
    "implement": "task",
    "coding": "task",
    "validation": "validation",
    "validate": "validation",
    "test": "validation",
    "testing": "validation",
    "merge": "merge",
    "merged": "merge",
    "promotion": "merge",
    "promote": "merge",
}
MEANINGFUL_REASONS = (
    "verified_merge",
    "validation_stage_advance",
    "canonical_task_transition",
    "accepted_artifact_change",
)
BUDGETS = ("task", "validation", "merge", "provider", "repair", "recover")
# Campaign policy: 3 task attempts, separate validation and merge budgets,
# 4 provider calls, 2 deterministic recoveries and 2 code repairs.
DEFAULT_LIMITS = {
    "task": 3,
    "validation": 3,
    "merge": 3,
    "provider": 4,
    "repair": 2,
    "recover": 2,
}
_NOISE_KINDS = frozenset({"log", "heartbeat", "pid"})
_STAGE_KINDS = frozenset({"", "stage", "artifact", "retry", "same_stage_retry", *CANONICAL_STAGES})
_RESET_FIELDS = frozenset({"reset", "reset_budget", "reset_budgets", "reset_progress", "refund", "budgets_reset", "progress_reset"})
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_PID_TEXT = re.compile(r"\b(?:pid|ppid|owner_pid)\s*[=:]?\s*\d+\b", re.IGNORECASE)
_ATTEMPT_TEXT = re.compile(r"\b(?:attempt|retry|retries)\s*[=:#]?\s*\d+\b", re.IGNORECASE)
_TIME_TEXT = re.compile(r"\b\d{4}-\d{2}-\d{2}[t ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:z|[+-]\d{2}:?\d{2})?\b", re.IGNORECASE)
_ADDR_TEXT = re.compile(r"\b0x[0-9a-f]+\b", re.IGNORECASE)
_HEARTBEAT_TEXT = re.compile(r"\bheartbeat(?:_at|_at_ms|_seq)?\s*[=:]?\s*[\d.]+\b", re.IGNORECASE)
_LOG_TEXT = re.compile(r"\blog(?:_bytes|_lines|_seq)?\s*[=:]?\s*\d+\b", re.IGNORECASE)
_MAX_TEXT = 4096


class ProgressIdentityError(ValueError):
    """Observation or checkpoint does not match the progress contract."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def canonical_stage(value: Any) -> str:
    """Map an observed stage onto task, validation or merge."""
    if not isinstance(value, str) or not value.strip():
        raise ProgressIdentityError("canonical stage is required")
    stage = STAGE_ALIASES.get(value.strip().lower())
    if stage is None:
        raise ProgressIdentityError("stage is not a canonical task, validation or merge stage")
    return stage


def _normalize_text(value: str) -> str:
    if len(value) > _MAX_TEXT:
        raise ProgressIdentityError("defect text exceeds the durable identity bound")
    text = value.strip().lower()
    for pattern in (_PID_TEXT, _ATTEMPT_TEXT, _TIME_TEXT, _ADDR_TEXT, _HEARTBEAT_TEXT, _LOG_TEXT):
        text = pattern.sub(" ", text)
    return " ".join(text.split())


def normalize_defect(value: Any) -> str:
    """Collapse PID, attempt, heartbeat and log churn out of a defect signature.

    A stable ``failure_key`` wins over freeform log text. Two reports that
    differ only by process id, attempt count or timestamps are one defect.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return _normalize_text(value)
    if not isinstance(value, Mapping):
        raise ProgressIdentityError("defect must be a string or object")
    stable: dict[str, str] = {}
    failure_key = value.get("failure_key")
    if isinstance(failure_key, str) and failure_key.strip():
        stable["failure_key"] = _normalize_text(failure_key)
        for key in ("error_type", "exception_type", "signature"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                stable[key] = _normalize_text(item)
    else:
        for key in ("signature", "error_type", "exception_type"):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                stable[key] = _normalize_text(item)
        if "signature" not in stable and "error_type" not in stable and "exception_type" not in stable:
            message = value.get("message", value.get("reason"))
            if isinstance(message, str) and message.strip():
                stable["message"] = _normalize_text(message)
    if not stable:
        return ""
    return json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def incident_identity(*, run_id: str, task_id: str, stage: str, defect: Any = "") -> str:
    """Durable incident id. PID and attempt are intentionally absent."""
    body = {
        "schema": INCIDENT_SCHEMA,
        "run_id": _identifier(run_id, "run_id"),
        "task_id": _identifier(task_id, "task_id"),
        "stage": canonical_stage(stage),
        "defect": normalize_defect(defect),
    }
    return digest(body)


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ProgressIdentityError(f"{name} must be a bounded durable identifier")
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ProgressIdentityError(f"{name} must be a sha256 digest")
    text = value.strip().lower()
    if _DIGEST.fullmatch(text) is None:
        raise ProgressIdentityError(f"{name} must be a sha256 digest")
    return text


def _timestamp(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProgressIdentityError(f"{name} must be a finite timestamp")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")) or number < 0:
        raise ProgressIdentityError(f"{name} must be a finite timestamp")
    return number


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ProgressIdentityError(f"{name} must be a non-negative integer")
    return value


def _artifact_path(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ProgressIdentityError("artifact path must be a bounded relative path")
    if value.startswith("/") or "\\" in value or "\x00" in value:
        raise ProgressIdentityError("artifact path must be a bounded relative path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ProgressIdentityError("artifact path must be a bounded relative path")
    leaf = parts[-1].lower()
    if leaf.endswith(".log") or "heartbeat" in leaf:
        raise ProgressIdentityError("logs and heartbeats are not accepted artifacts")
    return value


def _limits(value: Mapping[str, Any] | None) -> dict[str, int]:
    limits = dict(DEFAULT_LIMITS)
    if value is None:
        return limits
    if not isinstance(value, Mapping):
        raise ProgressIdentityError("budget limits must be an object")
    unknown = set(value) - set(BUDGETS)
    if unknown:
        raise ProgressIdentityError("unknown repair budget")
    for name, limit in value.items():
        if type(limit) is not int or not 0 < limit <= 1000:
            raise ProgressIdentityError("budget limits must be small positive integers")
        limits[name] = limit
    return limits


def _empty_cursor(run_id: str) -> dict[str, Any]:
    return {
        "schema": CURSOR_SCHEMA,
        "run_id": run_id,
        "task_id": "",
        "stage": "",
        "accepted_artifacts": [],
        "validation_receipt": "",
        "merge_receipt": "",
    }


def _artifact_pairs(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        raise ProgressIdentityError("accepted artifacts must be a list")
    pairs: list[list[str]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ProgressIdentityError("accepted artifact entry is malformed")
        path, content = _artifact_path(item[0]), _digest(item[1], "artifact digest")
        if path in seen:
            raise ProgressIdentityError("accepted artifact path repeated")
        seen.add(path)
        pairs.append([path, content])
    pairs.sort()
    return pairs


class ProgressLedger:
    """In-memory cursor with an atomic checkpoint.

    Observations never refund spent budgets. A checkpoint reloads the same
    cursor, incident ids and spent counts; it does not open a new episode.
    """

    def __init__(self, run_id: str, *, limits: Mapping[str, Any] | None = None):
        self.run_id = _identifier(run_id, "run_id")
        self.limits = _limits(limits)
        self.cursor = _empty_cursor(self.run_id)
        self.progress_token = digest(self.cursor)
        self.last_progress_at: float | None = None
        self.last_progress_reason = ""
        self.observed_stage = ""
        self.observed_task_id = ""
        self.attempt: int | None = None
        self.pid: int | None = None
        self.heartbeat_at: float | None = None
        self.log_bytes: int | None = None
        self._defects: dict[tuple[str, str], str] = {}
        self.incidents: dict[str, dict[str, Any]] = {}
        self.noise = {"log": 0, "heartbeat": 0, "pid": 0, "same_stage_retry": 0}
        self.completion_authority = False

    def observe(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Record one observation. Noise does not move the cursor or the budgets."""
        if not isinstance(event, Mapping):
            raise ProgressIdentityError("observation must be an object")
        if any(field in event and event[field] not in (False, None) for field in _RESET_FIELDS):
            raise ProgressIdentityError("an observation cannot reset progress or repair budgets")
        if event.get("completion_authority") is True:
            raise ProgressIdentityError("an observation cannot grant completion authority")
        if _identifier(event.get("run_id"), "run_id") != self.run_id:
            raise ProgressIdentityError("observation run does not match this ledger")
        kind = event.get("kind", "")
        if not isinstance(kind, str) or kind not in _STAGE_KINDS | _NOISE_KINDS:
            raise ProgressIdentityError("observation kind is not recognized")
        previous_token = self.progress_token
        previous_progress_at = self.last_progress_at
        previous_budgets = self._budget_snapshot()
        pid_changed = self._note_pid(event)
        heartbeat_observed = self._note_heartbeat(event)
        log_observed = self._note_log(event)
        if kind in _NOISE_KINDS:
            # A log, heartbeat or PID sample is not stage evidence. It cannot
            # open an incident, switch tasks, or present a new defect.
            self.noise[kind] += 1
            incident_id = self._bind_incident({}, stage=self.observed_stage, task_id=self.observed_task_id)
            self._assert_stable(previous_token=previous_token, previous_progress_at=previous_progress_at, previous_budgets=previous_budgets, meaningful=False)
            return self._decision(
                meaningful=False,
                reason=kind if kind != "pid" else "pid_change",
                previous_token=previous_token,
                incident_id=incident_id,
                pid_changed=pid_changed,
                heartbeat_observed=heartbeat_observed,
                log_observed=log_observed,
                same_stage_retry=False,
                signals=[],
            )
        task_id = self._event_task(event, required=True)
        stage = self._event_stage(event)
        signals, proposed = self._propose(event, task_id=task_id, stage=stage)
        proposed_token = digest(proposed)
        if proposed_token == self.progress_token:
            signals = []
        attempt_advanced = self._attempt_advanced(event)
        stage_matches = stage == self.observed_stage or (stage == "" and kind in {"retry", "same_stage_retry"})
        same_stage = bool(self.observed_stage) and task_id == self.observed_task_id and stage_matches
        retry = (not signals) and same_stage and (attempt_advanced or kind in {"retry", "same_stage_retry"})
        meaningful = bool(signals)
        if meaningful:
            observed_at = _timestamp(event.get("observed_at"), "observed_at")
            self.cursor = proposed
            self.progress_token = proposed_token
            if self.last_progress_at is None or observed_at >= self.last_progress_at:
                self.last_progress_at = observed_at
            self.last_progress_reason = _primary(signals)
        if stage:
            self.observed_stage = stage
        self.observed_task_id = task_id
        if "attempt" in event and event["attempt"] is not None:
            self.attempt = _count(event["attempt"], "attempt")
        if retry:
            self.noise["same_stage_retry"] += 1
        reason = _primary(signals) if signals else self._noise_reason(
            retry=retry, pid_changed=pid_changed, heartbeat_observed=heartbeat_observed,
            log_observed=log_observed, stage=stage,
        )
        incident_id = self._bind_incident(event, stage=self.observed_stage, task_id=task_id)
        self._assert_stable(previous_token=previous_token, previous_progress_at=previous_progress_at, previous_budgets=previous_budgets, meaningful=meaningful)
        return self._decision(
            meaningful=meaningful,
            reason=reason,
            previous_token=previous_token,
            incident_id=incident_id,
            pid_changed=pid_changed,
            heartbeat_observed=heartbeat_observed,
            log_observed=log_observed,
            same_stage_retry=retry,
            signals=signals,
        )

    def charge(self, incident_id: str, budget: str) -> dict[str, Any]:
        """Spend one unit of a named budget. Exhaustion is a blocker, not success."""
        if not isinstance(incident_id, str) or incident_id not in self.incidents:
            raise ProgressIdentityError("unknown incident")
        if budget not in BUDGETS:
            raise ProgressIdentityError("unknown repair budget")
        incident = self.incidents[incident_id]
        limit = incident["limits"][budget]
        spent = incident["spent"][budget]
        charged = spent < limit
        if charged:
            incident["spent"][budget] = spent + 1
            spent += 1
        return {
            "schema": "wwl.progress.budget-charge@1",
            "incident_id": incident_id,
            "budget": budget,
            "spent": spent,
            "limit": limit,
            "remaining": limit - spent,
            "charged": charged,
            "exhausted": spent >= limit,
            "status": "charged" if charged else "blocked",
            "reason": f"{budget}_budget_exhausted" if not charged else f"{budget}_budget_charged",
            "budgets_reset": False,
            "progress_reset": False,
            "progress_token": self.progress_token,
            "last_progress_at": self.last_progress_at,
            "completion_authority": False,
        }

    def to_dict(self) -> dict[str, Any]:
        incidents = []
        for incident_id in sorted(self.incidents):
            incident = self.incidents[incident_id]
            incidents.append({
                "incident_id": incident_id,
                "run_id": incident["run_id"],
                "task_id": incident["task_id"],
                "stage": incident["stage"],
                "defect": incident["defect"],
                "limits": {name: incident["limits"][name] for name in BUDGETS},
                "spent": {name: incident["spent"][name] for name in BUDGETS},
            })
        return {
            "schema": STATE_SCHEMA,
            "run_id": self.run_id,
            "limits": {name: self.limits[name] for name in BUDGETS},
            "cursor": {
                **self.cursor,
                "accepted_artifacts": [list(pair) for pair in self.cursor["accepted_artifacts"]],
            },
            "progress_token": self.progress_token,
            "last_progress_at": self.last_progress_at,
            "last_progress_reason": self.last_progress_reason,
            "observed_stage": self.observed_stage,
            "observed_task_id": self.observed_task_id,
            "attempt": self.attempt,
            "pid": self.pid,
            "heartbeat_at": self.heartbeat_at,
            "log_bytes": self.log_bytes,
            "defects": [{"task_id": task, "stage": stage, "defect": defect} for (task, stage), defect in sorted(self._defects.items())],
            "incidents": incidents,
            "noise": dict(self.noise),
            "completion_authority": False,
        }

    def save(self, path: str | Path) -> None:
        """Atomically replace a checkpoint. A crash cannot leave a partial budget."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical_bytes(self.to_dict()) + b"\n"
        descriptor, temporary = tempfile.mkstemp(prefix=".wwl-progress-", dir=target.parent)
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
    def load(cls, path: str | Path) -> ProgressLedger:
        raw = Path(path).read_bytes()
        if len(raw) > 1_048_576:
            raise ProgressIdentityError("progress checkpoint exceeds the admission bound")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ProgressIdentityError("progress checkpoint is not JSON") from error
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProgressLedger:
        if not isinstance(value, Mapping) or value.get("schema") != STATE_SCHEMA:
            raise ProgressIdentityError("progress checkpoint schema is not recognized")
        if value.get("completion_authority") is not False:
            raise ProgressIdentityError("progress checkpoint cannot grant completion authority")
        ledger = cls(str(value.get("run_id") or ""), limits=value.get("limits") if isinstance(value.get("limits"), Mapping) else None)
        cursor = value.get("cursor")
        if not isinstance(cursor, Mapping):
            raise ProgressIdentityError("progress cursor is missing")
        restored = _empty_cursor(ledger.run_id)
        task_id = cursor.get("task_id") or ""
        if task_id:
            restored["task_id"] = _identifier(task_id, "task_id")
        stage = cursor.get("stage") or ""
        if stage:
            restored["stage"] = canonical_stage(stage)
            if task_id == "":
                raise ProgressIdentityError("a cursor stage requires a task")
        restored["accepted_artifacts"] = _artifact_pairs(list(cursor.get("accepted_artifacts") or []))
        for name in ("validation_receipt", "merge_receipt"):
            receipt = cursor.get(name) or ""
            restored[name] = _digest(receipt, name) if receipt else ""
        if restored["merge_receipt"] and restored["stage"] != "merge":
            raise ProgressIdentityError("a merge receipt requires the merge stage")
        if restored["validation_receipt"] and restored["stage"] not in {"validation", "merge"}:
            raise ProgressIdentityError("a validation receipt requires validation or merge")
        token = digest(restored)
        if value.get("progress_token") != token:
            raise ProgressIdentityError("progress token does not match the cursor")
        ledger.cursor = restored
        ledger.progress_token = token
        if value.get("last_progress_at") is not None:
            ledger.last_progress_at = _timestamp(value.get("last_progress_at"), "last_progress_at")
        elif restored["task_id"] or restored["accepted_artifacts"] or restored["merge_receipt"]:
            raise ProgressIdentityError("a moved cursor requires last_progress_at")
        reason = value.get("last_progress_reason") or ""
        if reason not in {"", *MEANINGFUL_REASONS}:
            raise ProgressIdentityError("progress reason is not canonical")
        ledger.last_progress_reason = reason
        ledger.observed_stage = canonical_stage(value["observed_stage"]) if value.get("observed_stage") else ""
        ledger.observed_task_id = _identifier(value["observed_task_id"], "task_id") if value.get("observed_task_id") else ""
        if value.get("attempt") is not None:
            ledger.attempt = _count(value.get("attempt"), "attempt")
        if value.get("pid") is not None:
            pid = value.get("pid")
            if type(pid) is not int or pid <= 0:
                raise ProgressIdentityError("pid must be a positive integer")
            ledger.pid = pid
        if value.get("heartbeat_at") is not None:
            ledger.heartbeat_at = _timestamp(value.get("heartbeat_at"), "heartbeat_at")
        if value.get("log_bytes") is not None:
            ledger.log_bytes = _count(value.get("log_bytes"), "log_bytes")
        ledger._defects = {}
        for item in value.get("defects") or []:
            if not isinstance(item, Mapping):
                raise ProgressIdentityError("stored defect is malformed")
            key = (_identifier(item.get("task_id"), "task_id"), canonical_stage(item.get("stage")))
            ledger._defects[key] = normalize_defect(item.get("defect", ""))
        ledger.incidents = {}
        for item in value.get("incidents") or []:
            ledger._restore_incident(item)
        noise = value.get("noise") or {}
        if not isinstance(noise, Mapping) or set(noise) - set(ledger.noise):
            raise ProgressIdentityError("noise counters are malformed")
        for name in ledger.noise:
            if name in noise:
                ledger.noise[name] = _count(noise[name], name)
        return ledger

    def _propose(self, event: Mapping[str, Any], *, task_id: str, stage: str) -> tuple[list[str], dict[str, Any]]:
        proposed = {
            **self.cursor,
            "accepted_artifacts": [list(pair) for pair in self.cursor["accepted_artifacts"]],
        }
        signals: list[str] = []
        if task_id != proposed["task_id"]:
            proposed["task_id"] = task_id
            proposed["stage"] = ""
            proposed["accepted_artifacts"] = []
            proposed["validation_receipt"] = ""
            proposed["merge_receipt"] = ""
            signals.append("canonical_task_transition")
        if stage == "task" and proposed["stage"] == "":
            proposed["stage"] = "task"
            if "canonical_task_transition" not in signals:
                signals.append("canonical_task_transition")
        artifacts = self._event_artifacts(event)
        if artifacts:
            current = dict(proposed["accepted_artifacts"])
            changed = False
            for path, content in artifacts:
                if current.get(path) != content:
                    current[path] = content
                    changed = True
            if changed:
                proposed["accepted_artifacts"] = sorted(current.items())
                signals.append("accepted_artifact_change")
        validation_receipt = self._optional_digest(event, "validation_receipt")
        if event.get("validation_accepted") is True:
            if not validation_receipt:
                raise ProgressIdentityError("an accepted validation result requires a receipt digest")
            if validation_receipt != proposed["validation_receipt"] or proposed["stage"] in {"", "task"}:
                proposed["validation_receipt"] = validation_receipt
                if proposed["stage"] in {"", "task"}:
                    proposed["stage"] = "validation"
                signals.append("validation_stage_advance")
        elif stage == "validation" and proposed["stage"] in {"", "task"}:
            proposed["stage"] = "validation"
            signals.append("validation_stage_advance")
        merge_receipt = self._optional_digest(event, "merge_receipt") or self._optional_digest(event, "accepted_tree")
        if event.get("merge_accepted") is True:
            if not merge_receipt:
                raise ProgressIdentityError("a verified merge requires a receipt digest")
            if merge_receipt != proposed["merge_receipt"] or proposed["stage"] != "merge":
                proposed["merge_receipt"] = merge_receipt
                proposed["stage"] = "merge"
                signals.append("verified_merge")
        if stage == "merge" and event.get("merge_accepted") is not True and proposed["stage"] == "merge":
            # A later unverified retry must not clear a merge that already landed.
            pass
        return list(dict.fromkeys(signals)), proposed

    def _event_artifacts(self, event: Mapping[str, Any]) -> list[tuple[str, str]]:
        value = event.get("accepted_artifacts")
        if value is None:
            return []
        if not isinstance(value, list):
            raise ProgressIdentityError("accepted artifacts must be a list")
        pairs: list[tuple[str, str]] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, Mapping):
                raise ProgressIdentityError("accepted artifact must be an object")
            if item.get("accepted") is not True:
                continue
            path = _artifact_path(item.get("path"))
            content = _digest(item.get("digest"), "artifact digest")
            if path in seen:
                raise ProgressIdentityError("accepted artifact path repeated")
            seen.add(path)
            pairs.append((path, content))
        return pairs

    def _event_task(self, event: Mapping[str, Any], *, required: bool) -> str:
        if "task_id" not in event or event.get("task_id") in (None, ""):
            if required:
                raise ProgressIdentityError("task_id is required")
            return ""
        return _identifier(event.get("task_id"), "task_id")

    def _event_stage(self, event: Mapping[str, Any]) -> str:
        kind = str(event.get("kind") or "")
        if "stage" in event and event.get("stage") not in (None, ""):
            return canonical_stage(event.get("stage"))
        if kind in CANONICAL_STAGES:
            return kind
        return ""

    def _optional_digest(self, event: Mapping[str, Any], name: str) -> str:
        if name not in event or event.get(name) in (None, ""):
            return ""
        return _digest(event.get(name), name)

    def _attempt_advanced(self, event: Mapping[str, Any]) -> bool:
        if "attempt" not in event or event.get("attempt") is None:
            return False
        attempt = _count(event.get("attempt"), "attempt")
        return self.attempt is not None and attempt != self.attempt

    def _note_pid(self, event: Mapping[str, Any]) -> bool:
        if "pid" not in event or event.get("pid") is None:
            return False
        pid = event.get("pid")
        if type(pid) is not int or pid <= 0:
            raise ProgressIdentityError("pid must be a positive integer")
        changed = self.pid is not None and pid != self.pid
        self.pid = pid
        return changed

    def _note_heartbeat(self, event: Mapping[str, Any]) -> bool:
        kind = str(event.get("kind") or "")
        if kind != "heartbeat" and "heartbeat_at" not in event and "heartbeat_at_ms" not in event:
            return False
        if "heartbeat_at" in event and event.get("heartbeat_at") is not None:
            self.heartbeat_at = _timestamp(event.get("heartbeat_at"), "heartbeat_at")
        elif "heartbeat_at_ms" in event and event.get("heartbeat_at_ms") is not None:
            millis = event.get("heartbeat_at_ms")
            if type(millis) is not int or millis < 0:
                raise ProgressIdentityError("heartbeat_at_ms must be a non-negative integer")
            self.heartbeat_at = millis / 1000
        return True

    def _note_log(self, event: Mapping[str, Any]) -> bool:
        kind = str(event.get("kind") or "")
        if kind != "log" and "log" not in event and "log_bytes" not in event and "log_lines" not in event:
            return False
        if "log_bytes" in event and event.get("log_bytes") is not None:
            self.log_bytes = _count(event.get("log_bytes"), "log_bytes")
        elif "log" in event and isinstance(event.get("log"), str):
            if len(event["log"]) > 1_048_576:
                raise ProgressIdentityError("log observation exceeds the admission bound")
            self.log_bytes = len(event["log"])
        return True

    def _noise_reason(self, *, retry: bool, pid_changed: bool, heartbeat_observed: bool, log_observed: bool, stage: str) -> str:
        if retry:
            return "same_stage_retry"
        if stage == "merge" and self.cursor.get("stage") != "merge":
            return "unverified_merge"
        if pid_changed:
            return "pid_change"
        if heartbeat_observed:
            return "heartbeat"
        if log_observed:
            return "log"
        return "unchanged"

    def _bind_incident(self, event: Mapping[str, Any], *, stage: str, task_id: str) -> str:
        if not task_id or not stage:
            return ""
        if "defect" in event:
            defect = normalize_defect(event.get("defect"))
            self._defects[(task_id, stage)] = defect
        else:
            defect = self._defects.get((task_id, stage), "")
        incident_id = incident_identity(run_id=self.run_id, task_id=task_id, stage=stage, defect=defect)
        if incident_id not in self.incidents:
            self.incidents[incident_id] = {
                "incident_id": incident_id,
                "run_id": self.run_id,
                "task_id": task_id,
                "stage": stage,
                "defect": defect,
                "limits": dict(self.limits),
                "spent": {name: 0 for name in BUDGETS},
            }
        return incident_id

    def _restore_incident(self, value: Any) -> None:
        if not isinstance(value, Mapping):
            raise ProgressIdentityError("stored incident is malformed")
        task_id = _identifier(value.get("task_id"), "task_id")
        stage = canonical_stage(value.get("stage"))
        defect = normalize_defect(value.get("defect", ""))
        expected = incident_identity(run_id=self.run_id, task_id=task_id, stage=stage, defect=defect)
        if value.get("incident_id") != expected or value.get("run_id") != self.run_id:
            raise ProgressIdentityError("stored incident identity does not match its binding")
        limits = value.get("limits")
        spent = value.get("spent")
        if not isinstance(limits, Mapping) or not isinstance(spent, Mapping):
            raise ProgressIdentityError("stored incident budget is malformed")
        if set(limits) != set(BUDGETS) or set(spent) != set(BUDGETS):
            raise ProgressIdentityError("stored incident budget is malformed")
        restored_limits: dict[str, int] = {}
        restored_spent: dict[str, int] = {}
        for name in BUDGETS:
            limit = limits[name]
            used = spent[name]
            if type(limit) is not int or limit != self.limits[name] or type(used) is not int or used < 0 or used > limit:
                raise ProgressIdentityError("stored incident budget was rewritten")
            restored_limits[name] = limit
            restored_spent[name] = used
        self.incidents[expected] = {
            "incident_id": expected,
            "run_id": self.run_id,
            "task_id": task_id,
            "stage": stage,
            "defect": defect,
            "limits": restored_limits,
            "spent": restored_spent,
        }

    def _assert_stable(self, *, previous_token: str, previous_progress_at: float | None, previous_budgets: tuple[tuple[str, tuple[tuple[str, int], ...]], ...], meaningful: bool) -> None:
        if self.progress_token != previous_token and not meaningful:
            raise ProgressIdentityError("progress token changed without a canonical advance")
        if not meaningful and self.last_progress_at != previous_progress_at:
            raise ProgressIdentityError("noise observation reset the progress clock")
        current = dict(self._budget_snapshot())
        for incident_id, spent in previous_budgets:
            if current.get(incident_id) != spent:
                raise ProgressIdentityError("observation reset a repair budget")

    def _budget_snapshot(self) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
        rows = []
        for incident_id in sorted(self.incidents):
            spent = self.incidents[incident_id]["spent"]
            rows.append((incident_id, tuple((name, spent[name]) for name in BUDGETS)))
        return tuple(rows)

    def _budget_view(self, incident_id: str) -> dict[str, dict[str, int]]:
        if not incident_id:
            return {}
        incident = self.incidents[incident_id]
        return {
            name: {
                "spent": incident["spent"][name],
                "limit": incident["limits"][name],
                "remaining": incident["limits"][name] - incident["spent"][name],
            }
            for name in BUDGETS
        }

    def _decision(
        self,
        *,
        meaningful: bool,
        reason: str,
        previous_token: str,
        incident_id: str,
        pid_changed: bool,
        heartbeat_observed: bool,
        log_observed: bool,
        same_stage_retry: bool,
        signals: list[str],
    ) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "meaningful": meaningful,
            "reason": reason,
            "signals": list(signals),
            "progress_token": self.progress_token,
            "previous_progress_token": previous_token,
            "progress_advanced": meaningful and self.progress_token != previous_token,
            "progress_reset": False,
            "budgets_reset": False,
            "last_progress_at": self.last_progress_at,
            "last_progress_reason": self.last_progress_reason,
            "incident_id": incident_id,
            "repair_budget": self._budget_view(incident_id),
            "cursor_stage": self.cursor["stage"],
            "cursor_task_id": self.cursor["task_id"],
            "observed_stage": self.observed_stage,
            "pid_changed": pid_changed,
            "heartbeat_observed": heartbeat_observed,
            "log_observed": log_observed,
            "same_stage_retry": same_stage_retry,
            "completion_authority": False,
        }


def _primary(signals: list[str]) -> str:
    for name in MEANINGFUL_REASONS:
        if name in signals:
            return name
    return ""
