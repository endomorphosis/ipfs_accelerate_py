"""One-campaign progress adapter for Native's finite deterministic recovery.

The observer and recovery callback are trusted application adapters. They keep
ownership, process fencing, task-source verification and implementation budgets
at their existing boundaries. This module calls no provider and grants no task
completion authority. Progress tokens must describe real work, never polls or
heartbeats; owner-token changes alone do not refresh the recovery budget.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import time
from typing import Any

from ..prompt.prompt_workflow import RecoveryAttemptOutcome, RescueOperation, _validate_cid
from .recovery_diagnostics import diagnose_supervisor_incident
from .supervisor_recovery import (
    ProgrammaticRecoveryController, ProgrammaticRecoveryPolicy, RecoveryActionObservation,
)

OBSERVATION_SCHEMA = "campaign-progress-observation@1"
STATE_SCHEMA = "campaign-progress-watchdog-state@1"
RESULT_SCHEMA = "campaign-progress-watchdog-result@1"
_SCOPE = {"repository_root_cid", "policy_root", "run_cid"}
_BOOL = {"healthy", "completion_verified", "manual_hold", "progress_verified",
         "owner_binding_verified", "recovery_permitted"}
_TEXT = {"status", "hold_reason", "active_task_id", "active_task_cid", "progress_token", "owner_token"}
_FIELDS = {"schema", "scope", "pending_task_ids", "attempt", "grace_until_ms"} | _BOOL | _TEXT
_HOLDS = {"manual_hold", "awaiting_user", "cancelled", "capacity_quarantine", "protected_incident",
          "attempt_budget_exhausted", "dependency_unavailable"}
_WAIT = {"waiting_for_user", "manual_hold", "paused", "cancelled", "capacity_quarantine",
         "protected_incident", "budget_exhausted", "blocked"}
_ACTIVE = {"active", "post_provider", "running", "maintenance", "selected", "starting"}
_FAILURE = {"stalled", "no_progress", "daemon_failed", "stopped"}
_STATUS = _WAIT | _ACTIVE | _FAILURE | {"complete", "identity_mismatch", "state_error", "board_error", "stop_incomplete", "unknown"}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _scope(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _SCOPE:
        raise ValueError("campaign_scope_invalid")
    if any(type(v) is not str or not v or len(v) > 512 or "\0" in v for v in value.values()):
        raise ValueError("campaign_scope_invalid")
    try:
        for key, member in value.items():
            _validate_cid(member, key)
    except ValueError as exc:
        raise ValueError("campaign_scope_invalid") from exc
    return dict(value)


@dataclass(frozen=True)
class CampaignProgressWatchdogPolicy:
    confirmation_count: int = 2
    confirmation_interval_ms: int = 30_000
    cooldown_ms: int = 60_000
    max_recovery_attempts: int = 2
    deadline_ms: int = 120_000
    startup_grace_ms: int = 180_000
    max_incidents: int = 128

    def __post_init__(self):
        for name, maximum in (("confirmation_count", 32), ("confirmation_interval_ms", 3_600_000), ("max_recovery_attempts", 8),
                              ("deadline_ms", 1_800_000), ("startup_grace_ms", 900_000), ("max_incidents", 1024)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("campaign_watchdog_policy_invalid:" + name)
        if type(self.cooldown_ms) is not int or not 0 <= self.cooldown_ms <= 86_400_000:
            raise ValueError("campaign_watchdog_policy_invalid:cooldown_ms")


class CampaignProgressWatchdog:
    """Observe once, or perform one bounded, revalidated Native recovery action.

    ``observe`` returns exactly OBSERVATION_SCHEMA's metadata fields. Complete
    requires no pending/active work, attempt zero, and no active progress.
    ``recover`` receives Native ProgrammaticRecoveryContext; it must respect
    its absolute deadline and return RecoveryActionObservation with the actual
    effects ``old_lane_fenced`` and ``lane_restarted``. A fresh observation,
    rather than callback success alone, establishes recovery. The application
    supplies its existing stop/recover/doctor/start operations inside that one
    callback. No handler for retry/rearm or provider rescue is registered.
    """

    def __init__(self, state_dir: Path | str, *, repository_root: Path | str, scope: Mapping[str, str],
                 observe: Callable[[], Mapping[str, Any]], recover: Callable[[Any], Any],
                 policy: CampaignProgressWatchdogPolicy | None = None,
                 clock_ms: Callable[[], int] | None = None):
        if not callable(observe) or not callable(recover):
            raise TypeError("campaign_watchdog_callbacks_required")
        self.state_dir = Path(state_dir).absolute()
        self.repository_root = Path(repository_root).absolute()
        if (self.repository_root.resolve(strict=True) != self.repository_root
                or not self.repository_root.is_dir()):
            raise ValueError("campaign_repository_root_invalid")
        self.scope = _scope(scope)
        self.observe, self.recover = observe, recover
        self.policy = policy or CampaignProgressWatchdogPolicy()
        self.clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self.state_path = self.state_dir / "campaign-progress-watchdog.json"
        self.policy_digest = _digest(vars(self.policy))

    def _now(self) -> int:
        value = self.clock_ms()
        if type(value) is not int or value < 0:
            raise ValueError("campaign_watchdog_clock_invalid")
        return value

    def _observe(self) -> dict[str, Any]:
        raw = self.observe()
        if not isinstance(raw, Mapping) or set(raw) != _FIELDS or raw.get("schema") != OBSERVATION_SCHEMA:
            raise ValueError("campaign_observation_schema_invalid")
        value = dict(raw)
        if _scope(value["scope"]) != self.scope:
            raise ValueError("campaign_observation_scope_changed")
        if any(type(value[k]) is not bool for k in _BOOL):
            raise ValueError("campaign_observation_boolean_invalid")
        if any(type(value[k]) is not str or len(value[k]) > 512 or "\0" in value[k] for k in _TEXT):
            raise ValueError("campaign_observation_text_invalid")
        if value["status"] not in _STATUS or value["hold_reason"] not in _HOLDS | {""}:
            raise ValueError("campaign_observation_status_invalid")
        if not value["progress_token"] or not value["owner_token"]:
            raise ValueError("campaign_observation_identity_missing")
        pending = value["pending_task_ids"]
        if (not isinstance(pending, list) or len(pending) > 4096 or len(set(pending)) != len(pending)
                or any(type(v) is not str or not v or len(v) > 256 for v in pending)):
            raise ValueError("campaign_observation_pending_invalid")
        if any(type(value[k]) is not int or value[k] < 0 for k in ("attempt", "grace_until_ms")):
            raise ValueError("campaign_observation_integer_invalid")
        if bool(value["active_task_id"]) != bool(value["active_task_cid"]):
            raise ValueError("campaign_observation_task_identity_invalid")
        if value["completion_verified"] and (value["status"] != "complete" or not value["healthy"]
                or pending or value["active_task_id"] or value["attempt"] or value["progress_verified"]):
            raise ValueError("campaign_observation_completion_conflict")
        return json.loads(json.dumps(value, sort_keys=True))

    def _result(self, status: str, reason: str, **extra) -> dict[str, Any]:
        return {"schema": RESULT_SCHEMA, "status": status, "reason": reason,
                "scope": dict(self.scope), "completion_authority": False,
                "provider_calls_by_watchdog": 0, "attempt_refunded": False, **extra}

    def _passive(self, value, now, *, grace_limit=None):
        if value["completion_verified"]:
            return self._result("complete", "verified_campaign_complete")
        if value["status"] == "complete":
            return self._result("refused", "campaign_completion_unverified")
        if value["manual_hold"] or value["hold_reason"] or value["status"] in _WAIT:
            return self._result("held", value["hold_reason"] or "intentional_wait_or_block")
        if value["progress_verified"] and value["healthy"] and value["status"] in _ACTIVE:
            return self._result("healthy", "verified_work_progress")
        if value["status"] in _ACTIVE:
            until = min(value["grace_until_ms"], grace_limit if grace_limit is not None else now + self.policy.startup_grace_ms)
            if now < until:
                return self._result("recovering" if value["status"] == "starting" else "healthy", "bounded_startup_grace")
        return None

    def _directory(self):
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.state_dir.lstat()
        if (self.state_dir.resolve(strict=True) != self.state_dir or not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700):
            raise ValueError("campaign_watchdog_directory_unowned")

    def _load(self):
        try:
            fd = os.open(self.state_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return {"schema": STATE_SCHEMA, "scope": self.scope, "repository_root": str(self.repository_root), "policy_digest": self.policy_digest,
                    "incidents": {}, "grace": {}}
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > 1_048_576):
                raise ValueError("campaign_watchdog_state_unowned")
            raw = stream.read(1_048_577)
            after = os.fstat(stream.fileno())
            if (len(raw) != info.st_size or any(getattr(info, key) != getattr(after, key)
                    for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))):
                raise ValueError("campaign_watchdog_state_changed")
        value = json.loads(raw)
        if (not isinstance(value, dict) or set(value) != {"schema", "scope", "repository_root", "policy_digest", "incidents", "grace"}
                or value["schema"] != STATE_SCHEMA or value["scope"] != self.scope
                or value["repository_root"] != str(self.repository_root)
                or value["policy_digest"] != self.policy_digest or not isinstance(value["incidents"], dict)
                or not isinstance(value["grace"], dict) or len(value["incidents"]) > self.policy.max_incidents):
            raise ValueError("campaign_watchdog_state_invalid")
        return value

    def _store(self, value):
        fd, path = tempfile.mkstemp(prefix=".campaign-watchdog-", dir=self.state_dir)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(value, stream, sort_keys=True, separators=(",", ":")); stream.write("\n")
                stream.flush(); os.fsync(stream.fileno())
            os.replace(path, self.state_path)
            directory = os.open(self.state_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def run_once(self) -> dict[str, Any]:
        try:
            before, now = self._observe(), self._now()
            passive = self._passive(before, now)
            if passive and passive["reason"] != "bounded_startup_grace":
                return passive  # Complete and holds create no incident or recovery artifacts.
            self._directory()
            lock = self.state_dir / ".campaign-progress-watchdog.lock"
            fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
            with os.fdopen(fd, "r+b") as stream:
                info = os.fstat(stream.fileno())
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                        or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600):
                    raise ValueError("campaign_watchdog_lock_unowned")
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return self._result("refused", "watchdog_owner_busy")
                return self._run_locked(before, now)
        except Exception as exc:
            return self._result("error", str(exc) if isinstance(exc, ValueError) and str(exc).startswith("campaign_")
                                else type(exc).__name__)

    def reconcile_health_pending_restart(self, *, incident_key, expected_terminal_cid,
                                         action_observation, verify_effects):
        """Resolve only proved complete restart effects, without replay or refund.

        ``verify_effects`` is a trusted read-only application observer, like
        ``observe``. It binds the historical restarted owner and later actual
        work to the exact action receipt. A fresh owned active observation is
        independently required. The original Native terminal stays immutable.
        """
        from .campaign_reconciliation import reconcile_health_pending_restart
        return reconcile_health_pending_restart(self, incident_key=incident_key,
            expected_terminal_cid=expected_terminal_cid,
            action_observation=action_observation, verify_effects=verify_effects)

    def _run_locked(self, before, now):
        state = self._load()
        grace_key = _digest([before["progress_token"], before["owner_token"]])
        grace = state["grace"]
        if grace.get("key") != grace_key:
            grace = state["grace"] = {"key": grace_key, "until_ms": min(before["grace_until_ms"], now + self.policy.startup_grace_ms)}
        passive = self._passive(before, now, grace_limit=grace.get("until_ms", 0))
        if passive:
            self._store(state)
            return passive
        if (not before["pending_task_ids"] or not before["owner_binding_verified"]
                or not before["recovery_permitted"] or before["status"] not in _ACTIVE | _FAILURE):
            return self._result("refused", "recovery_not_authorized_or_no_pending_work")
        # PID/owner and polling times deliberately do not create a new budget.
        key = _digest([before["active_task_id"], before["active_task_cid"], before["attempt"], before["progress_token"]])
        incidents = state["incidents"]
        if key not in incidents:
            if len(incidents) >= self.policy.max_incidents:
                return self._result("exhausted", "incident_storage_bound")
            incidents[key] = {"confirmations": 0, "confirmation_owner": before["owner_token"], "last_confirmation_ms": 0,
                              "attempts": 0, "last_attempt_ms": 0, "phase": "observing"}
        entry = incidents[key]
        if (not isinstance(entry, dict) or set(entry) - {"confirmations", "confirmation_owner", "last_confirmation_ms", "attempts", "last_attempt_ms", "phase", "terminal_cid", "reconciliation_cid"}
                or any(type(entry.get(k)) is not int or entry[k] < 0 for k in ("confirmations", "last_confirmation_ms", "attempts", "last_attempt_ms"))
                or type(entry.get("confirmation_owner")) is not str
                or entry.get("phase") not in {"observing", "in_flight", "finished", "uncertain"}):
            raise ValueError("campaign_watchdog_incident_invalid")
        if entry["confirmation_owner"] != before["owner_token"]:
            entry.update(confirmations=0, last_confirmation_ms=0, confirmation_owner=before["owner_token"])
            self._store(state)  # A new owner gets fresh confirmation, never a fresh action budget.
        if entry["phase"] in {"in_flight", "uncertain"}:
            return self._result("refused", "recovery_effects_uncertain", recovery_attempts=entry["attempts"])
        if entry["attempts"] >= self.policy.max_recovery_attempts:
            return self._result("exhausted", "incident_recovery_budget_exhausted", recovery_attempts=entry["attempts"])
        if entry["attempts"] and now < entry["last_attempt_ms"] + self.policy.cooldown_ms:
            return self._result("cooldown", "incident_recovery_cooldown", recovery_attempts=entry["attempts"],
                                retry_at_ms=entry["last_attempt_ms"] + self.policy.cooldown_ms)
        if not entry["confirmations"] or now >= entry["last_confirmation_ms"] + self.policy.confirmation_interval_ms:
            entry["confirmations"] = min(entry["confirmations"] + 1, self.policy.confirmation_count)
            entry["last_confirmation_ms"] = now
        if entry["confirmations"] < self.policy.confirmation_count:
            self._store(state)
            return self._result("confirming", "failure_confirmation_pending", confirmations=entry["confirmations"])
        entry.update(attempts=entry["attempts"] + 1, last_attempt_ms=now, phase="in_flight")
        self._store(state)  # Reserve the finite attempt before any effect.
        invoked, callback_error, effects_uncertain = False, False, False

        def handler(context):
            nonlocal invoked, callback_error, effects_uncertain
            current = self._observe()
            if current != before or self._now() >= context.deadline_at_ms:
                return RecoveryActionObservation(False, reason="observation_changed_before_effect")
            invoked = True
            try:
                raw = self.recover(context)
                raw = ProgrammaticRecoveryController._observation(raw, context.action)
            except Exception:
                callback_error = True
                raise
            effects_uncertain = raw.partial or (raw.succeeded and (
                set(raw.observed_effects) != set(context.action.expected_effects)
                or len(raw.observed_effects) != len(context.action.expected_effects)))
            return RecoveryActionObservation(raw.succeeded, raw.observed_effects,
                {"healthy": raw.healthy}, reason="application_recovery_observed", partial=raw.partial)

        def health(_context, _observation):
            nonlocal effects_uncertain
            try:
                current = self._observe()
            except Exception:
                effects_uncertain = True
                raise
            return {"healthy": bool(current["completion_verified"] or (
                current["healthy"] and current["progress_verified"] and current["status"] in _ACTIVE))}

        diagnosis = diagnose_supervisor_incident(repository_root=str(self.repository_root), state_root=str(self.state_dir),
            **self.scope, observed_at_ms=now, process={"target_id": self.scope["run_cid"], "failed": True,
                "lane_identity_current": True, "retry_permitted": False, "progress_token": before["progress_token"],
                "campaign_incident": key, "recovery_ordinal": entry["attempts"]})
        controller = ProgrammaticRecoveryController(self.state_dir, handlers={RescueOperation.RESTART_LANE: handler},
            health_check=health, policy=ProgrammaticRecoveryPolicy(max_attempts_per_action=1, max_total_attempts=1,
                max_actions=2, cooldown_ms=self.policy.cooldown_ms, deadline_ms=self.policy.deadline_ms), clock_ms=self.clock_ms)
        result = controller.recover(diagnosis)
        timed_out = any(attempt.outcome is RecoveryAttemptOutcome.TIMED_OUT for attempt in result.attempts)
        entry.update(phase="uncertain" if callback_error or effects_uncertain or timed_out else "finished",
                     terminal_cid=result.terminal_cid, confirmations=0, last_confirmation_ms=0)
        self._store(state)
        if not invoked:
            return self._result("refused", "observation_changed_before_effect", recovery_attempts=entry["attempts"])
        if callback_error:
            return self._result("error", "application_recovery_effects_uncertain", recovery_attempts=entry["attempts"])
        if timed_out:
            return self._result("error", "recovery_deadline_exhausted", recovery_attempts=entry["attempts"])
        current = self._observe()
        if current["completion_verified"]:
            # Completion can win the race with restart. Report the independent
            # terminal observation, not an invented lane_restarted effect;
            # retain the conservative ledger if the action was incomplete.
            return self._result("complete", "verified_campaign_complete", recovery_attempts=entry["attempts"],
                terminal_cid=result.terminal_cid, recovery_effects_uncertain=effects_uncertain)
        if effects_uncertain:
            return self._result("error", "application_recovery_effects_unproven", recovery_attempts=entry["attempts"])
        if result.recovered:
            return self._result("recovered", "fresh_semantic_health_restored", recovery_attempts=entry["attempts"], terminal_cid=result.terminal_cid)
        passive = self._passive(current, self._now())
        if passive:
            return {**passive, "recovery_attempts": entry["attempts"], "terminal_cid": result.terminal_cid}
        return self._result("exhausted" if entry["attempts"] >= self.policy.max_recovery_attempts else "cooldown",
            "recovery_not_yet_healthy", recovery_attempts=entry["attempts"], terminal_cid=result.terminal_cid)
