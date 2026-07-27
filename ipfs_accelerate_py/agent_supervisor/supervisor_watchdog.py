"""Outer watchdog process that monitors and restarts the bundle supervisor.

This provides a second level of fault tolerance: even if the bundle supervisor
itself crashes or hangs, this watchdog will detect the failure and restart it.

Designed for systemd-style long-running operation (hours/days/weeks).

Usage:
    python -m ipfs_accelerate_py.agent_supervisor.supervisor_watchdog \
        --manifest-path data/agent_supervisor/bundle_lanes/bundle_lanes.json \
        --repo-root .

Environment variables:
    WATCHDOG_CHECK_INTERVAL_SECONDS: How often to check lane health (default: 120)
    WATCHDOG_LANE_TIMEOUT_SECONDS: Consider a lane dead if no heartbeat for this long (default: 600)
    WATCHDOG_MAX_CONSECUTIVE_RESTARTS: After this many rapid restarts, back off (default: 5)
    WATCHDOG_LOG_AGGREGATION_DIR: Directory for unified structured logs (default: state_root/logs/aggregated)
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import signal
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence

from .control.control_plane import (
    LIFECYCLE_STATUS_SCHEMA,
    SupervisorLifecycleState,
)
from .prompt_workflow import RecordStatus, RescueOperation, prompt_workflow_cid
from .recovery_diagnostics import (
    RecoveryDiagnosticError,
    RecoveryDiagnosis,
    diagnose_supervisor_incident,
)
from .rescue_planner import (
    RescuePlanner,
    RescuePlanningRequest,
)
from .supervisor_recovery import (
    ProgrammaticRecoveryController,
    ProgrammaticRecoveryPolicy,
    ProgrammaticRecoveryResult,
    RecoveryActionObservation,
)
from .scheduler_metrics import scheduler_snapshot, scheduler_state_events

logger = logging.getLogger(__name__)

SUPERVISOR_LIFECYCLE_STATES: Final[tuple[str, ...]] = (
    tuple(state.value for state in SupervisorLifecycleState)
)
_LIFECYCLE_STATE_SET: Final[frozenset[str]] = frozenset(
    SUPERVISOR_LIFECYCLE_STATES
)
_TRANSITIONAL_STATES: Final[frozenset[str]] = frozenset(
    {"starting", "draining", "stopping"}
)
_INTENTIONAL_NON_RUNNING_STATES: Final[frozenset[str]] = frozenset(
    {"paused", "draining", "blocked", "stopping", "stopped", "failed"}
)
_STATE_ALIASES: Final[dict[str, str]] = {
    "alive": "healthy",
    "ok": "healthy",
    "running": "healthy",
    "ready": "healthy",
    "unhealthy": "degraded",
    "hung": "degraded",
    "error": "failed",
    "crashed": "failed",
    "shutdown": "stopped",
    "shutdown_complete": "stopped",
}
AUTONOMOUS_UNSTALL_STATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomous-unstall-state@1"
)
AUTONOMOUS_UNSTALL_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomous-unstall-result@1"
)
_AUTONOMOUS_UNSTALL_EVIDENCE_SLOTS: Final[frozenset[str]] = frozenset(
    {
        "status",
        "health",
        "process",
        "heartbeat",
        "event",
        "lease",
        "lock",
        "task",
        "attempt",
        "task_source",
        "worktree",
        "merge",
        "provider",
        "validation",
        "disk",
    }
)
_AUTONOMOUS_UNSTALL_PHASES: Final[frozenset[str]] = frozenset(
    {
        "deterministic_recovery",
        "quarantined",
        "recovered",
        "rescue_executing",
        "rescue_previewed",
        "rescue_previewing",
    }
)
_AUTONOMOUS_UNSTALL_MAX_STATE_BYTES: Final[int] = 4 * 1024 * 1024


@dataclass(frozen=True)
class AutonomousUnstallPolicy:
    """Explicit operating policy for one bounded autonomous unstall lane.

    Deterministic recovery is enabled independently from model rescue.  A
    provider cannot be called unless both rescue preview and provider access
    are explicitly enabled under a non-empty policy identity.  Execution has
    a separate opt-in and still passes through ``RescueOrchestrator``.
    """

    enabled: bool = True
    rescue_preview_enabled: bool = False
    rescue_execution_enabled: bool = False
    allow_provider_calls: bool = False
    operating_policy_id: str = ""
    max_incidents: int = 128
    max_provider_calls: int = 4
    max_rescue_executions: int = 4
    circuit_breaker_failures: int = 2
    deterministic_max_attempts_per_action: int = 2
    deterministic_max_total_attempts: int = 8
    deterministic_max_actions: int = 8
    cooldown_ms: int = 30_000
    deadline_ms: int = 120_000

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "rescue_preview_enabled",
            "rescue_execution_enabled",
            "allow_provider_calls",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in (
            "max_incidents",
            "max_provider_calls",
            "max_rescue_executions",
            "circuit_breaker_failures",
            "deterministic_max_attempts_per_action",
            "deterministic_max_total_attempts",
            "deterministic_max_actions",
            "deadline_ms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.cooldown_ms, bool)
            or not isinstance(self.cooldown_ms, int)
            or self.cooldown_ms < 0
        ):
            raise ValueError("cooldown_ms must be a nonnegative integer")
        if self.rescue_execution_enabled and not self.rescue_preview_enabled:
            raise ValueError("rescue execution requires rescue preview")
        if self.allow_provider_calls and not self.rescue_preview_enabled:
            raise ValueError("provider access requires rescue preview")
        if (
            self.rescue_preview_enabled
            or self.rescue_execution_enabled
            or self.allow_provider_calls
        ) and not self.operating_policy_id.strip():
            raise ValueError(
                "rescue requires an explicit operating_policy_id"
            )

    @property
    def deterministic_policy(self) -> ProgrammaticRecoveryPolicy:
        return ProgrammaticRecoveryPolicy(
            max_attempts_per_action=self.deterministic_max_attempts_per_action,
            max_total_attempts=self.deterministic_max_total_attempts,
            max_actions=self.deterministic_max_actions,
            cooldown_ms=self.cooldown_ms,
            deadline_ms=self.deadline_ms,
        )


class AutonomousUnstallCoordinator:
    """Compose semantic diagnosis, deterministic recovery, and optional rescue.

    The coordinator owns only the affected incident scope.  Its durable state
    is written before every effectful phase so a restart during recovery is
    visible and fails closed.  The supplied action handlers and orchestrator
    remain the only effect authorities.
    """

    def __init__(
        self,
        *,
        state_dir: Path | str,
        repository_root: Path | str,
        repository_root_cid: str,
        policy_root: str,
        run_cid: str,
        policy: AutonomousUnstallPolicy | None = None,
        recovery_handlers: Mapping[
            RescueOperation | str, Callable[[Any], Any]
        ]
        | None = None,
        health_probe: Callable[[], Mapping[str, Any]] | None = None,
        root_probe: Callable[[], Mapping[str, str]] | None = None,
        quarantine_scope: Callable[
            [Sequence[str], str, str], Mapping[str, Any] | None
        ]
        | None = None,
        event_publisher: Callable[[str, Mapping[str, Any]], Any] | None = None,
        rescue_planner: RescuePlanner | None = None,
        rescue_request_factory: Callable[
            [RecoveryDiagnosis, Any, Mapping[str, str]], RescuePlanningRequest
        ]
        | None = None,
        rescue_orchestrator: Any = None,
        rescue_execution_request_factory: Callable[
            [RecoveryDiagnosis, Any, Any, Mapping[str, str]], Any
        ]
        | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.repository_root = Path(repository_root).resolve()
        self.repository_root_cid = str(repository_root_cid).strip()
        self.policy_root = str(policy_root).strip()
        self.run_cid = str(run_cid).strip()
        if not all(
            (self.repository_root_cid, self.policy_root, self.run_cid)
        ):
            raise ValueError("current repository, policy, and run roots are required")
        self.policy = policy or AutonomousUnstallPolicy()
        self.recovery_handlers = dict(recovery_handlers or {})
        self.health_probe = health_probe
        self.root_probe = root_probe
        self.quarantine_scope = quarantine_scope
        self.event_publisher = event_publisher
        self.rescue_planner = rescue_planner
        self.rescue_request_factory = rescue_request_factory
        self.rescue_orchestrator = rescue_orchestrator
        self.rescue_execution_request_factory = (
            rescue_execution_request_factory
        )
        self.clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self.state_path = self.state_dir / "autonomous-unstall-state.json"
        self.recovery_state_dir = self.state_dir / "autonomous-unstall-recovery"
        self._lock = threading.RLock()

    def _roots(self) -> dict[str, str]:
        roots = {
            "repository_root_cid": self.repository_root_cid,
            "policy_root": self.policy_root,
            "run_cid": self.run_cid,
        }
        if self.root_probe is not None:
            observed = self.root_probe()
            if not isinstance(observed, Mapping):
                raise ValueError("root_probe must return a mapping")
            for key in tuple(roots):
                value = str(observed.get(key) or "").strip()
                if not value:
                    raise ValueError(f"root_probe omitted {key}")
                roots[key] = value
        return roots

    def _load_state(self) -> dict[str, Any]:
        try:
            raw = self.state_path.read_bytes()
            if len(raw) > _AUTONOMOUS_UNSTALL_MAX_STATE_BYTES:
                raise ValueError("autonomous unstall state exceeds byte bound")
            payload = json.loads(raw)
        except FileNotFoundError:
            return {
                "schema": AUTONOMOUS_UNSTALL_STATE_SCHEMA,
                "incidents": {},
                "rescue_runtime": {},
                "updated_at_ms": 0,
            }
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            return self._corrupt_state_fallback(exc)
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != AUTONOMOUS_UNSTALL_STATE_SCHEMA
            or not isinstance(payload.get("incidents"), dict)
        ):
            return self._corrupt_state_fallback(
                ValueError("unsupported autonomous unstall state")
            )
        if "rescue_runtime" not in payload:
            payload["rescue_runtime"] = {}
        elif not isinstance(payload.get("rescue_runtime"), dict):
            return self._corrupt_state_fallback(
                ValueError("invalid autonomous unstall rescue runtime")
            )
        try:
            for incident_cid, entry in payload["incidents"].items():
                target_ids = entry.get("target_ids") if isinstance(
                    entry, Mapping
                ) else None
                if (
                    not isinstance(incident_cid, str)
                    or not incident_cid.strip()
                    or not isinstance(entry, Mapping)
                    or (
                        entry.get("incident_cid")
                        and entry.get("incident_cid") != incident_cid
                    )
                    or str(entry.get("phase") or "")
                    not in _AUTONOMOUS_UNSTALL_PHASES
                    or not isinstance(target_ids, list)
                    or any(
                        not isinstance(item, str) or not item.strip()
                        for item in target_ids
                    )
                ):
                    raise ValueError(
                        "invalid autonomous unstall incident entry"
                    )
                for key in ("created_at_ms", "updated_at_ms"):
                    if key not in entry:
                        continue
                    value = entry[key]
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                    ):
                        raise ValueError(
                            "invalid autonomous unstall incident timestamp"
                        )
            runtime = payload["rescue_runtime"]
            for key in (
                "provider_calls",
                "executions",
                "consecutive_failures",
                "last_provider_call_ms",
            ):
                if key not in runtime:
                    continue
                value = runtime[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                ):
                    raise ValueError(
                        f"invalid autonomous unstall runtime {key}"
                    )
            if (
                "circuit_open" in runtime
                and not isinstance(runtime["circuit_open"], bool)
            ):
                raise ValueError(
                    "invalid autonomous unstall runtime circuit_open"
                )
            if (
                "reason" in runtime
                and not isinstance(runtime["reason"], str)
            ):
                raise ValueError(
                    "invalid autonomous unstall runtime reason"
                )
        except (TypeError, ValueError) as exc:
            return self._corrupt_state_fallback(exc)
        return payload

    def _corrupt_state_fallback(self, exc: BaseException) -> dict[str, Any]:
        """Preserve corrupt bytes and fail closed for uncertain rescue history."""

        backup = self.state_path.with_name(
            f"{self.state_path.name}.corrupt-{self.clock_ms()}"
        )
        try:
            os.replace(self.state_path, backup)
        except OSError:
            pass
        return {
            "schema": AUTONOMOUS_UNSTALL_STATE_SCHEMA,
            "incidents": {},
            "rescue_runtime": {
                "circuit_open": True,
                "reason": "corrupt_coordination_state",
            },
            "updated_at_ms": self.clock_ms(),
            "state_repair": {
                "reason": "corrupt_coordination_state_quarantined",
                "backup_path": str(backup),
                "error_type": type(exc).__name__,
            },
        }

    def _store_state(self, state: Mapping[str, Any]) -> None:
        payload = dict(state)
        payload["schema"] = AUTONOMOUS_UNSTALL_STATE_SCHEMA
        payload["updated_at_ms"] = self.clock_ms()
        incidents = payload.get("incidents")
        if not isinstance(incidents, dict):
            raise ValueError("autonomous unstall incidents must be a mapping")
        if len(incidents) > self.policy.max_incidents:
            ordered = sorted(
                incidents.items(),
                key=lambda item: int(
                    item[1].get("updated_at_ms", 0)
                    if isinstance(item[1], Mapping)
                    else 0
                ),
                reverse=True,
            )
            payload["incidents"] = dict(ordered[: self.policy.max_incidents])
        encoded = (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.state_path.name}.", dir=self.state_path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.state_path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _publish(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self.event_publisher is None:
            return
        try:
            self.event_publisher(event_type, dict(payload))
        except Exception as exc:
            # Durable state is the authoritative control projection.  A
            # transient publisher outage must not replay an already-started
            # effect or break the finite recovery pass.
            logger.warning(
                "Autonomous unstall event publisher failed for %s: %s",
                event_type,
                type(exc).__name__,
            )

    @staticmethod
    def _healthy(value: Mapping[str, Any]) -> bool:
        """Require consistent semantic health, never mere process liveness."""

        if value.get("work_complete") is True:
            return False
        if "healthy" in value and value.get("healthy") is not True:
            return False
        if any(
            value.get(key) is True
            for key in (
                "failed",
                "heartbeat_stale",
                "stale",
                "state_inconsistent",
                "unexpected_effect",
            )
        ):
            return False
        status = str(value.get("status") or value.get("state") or "").lower()
        if status in {
            "blocked",
            "degraded",
            "failed",
            "stale",
            "stopped",
            "unhealthy",
        }:
            return False
        # pid_alive/alive without one of these semantic signals is not health.
        return bool(
            value.get("healthy") is True
            or status in {"healthy", "ok", "recovered"}
        )

    def _health(self) -> dict[str, Any]:
        if self.health_probe is None:
            return {"healthy": False, "reason": "health_probe_unavailable"}
        raw = self.health_probe()
        if not isinstance(raw, Mapping):
            return {"healthy": False, "reason": "invalid_health_probe"}
        result = dict(raw)
        result["healthy"] = self._healthy(result)
        result["completion_authority"] = False
        return result

    def _record(
        self,
        state: dict[str, Any],
        incident_cid: str,
        *,
        phase: str,
        reason: str,
        targets: Sequence[str],
        **values: Any,
    ) -> dict[str, Any]:
        now_ms = self.clock_ms()
        previous = state["incidents"].get(incident_cid)
        previous_created_at = (
            previous.get("created_at_ms")
            if isinstance(previous, Mapping)
            else None
        )
        entry = {
            "incident_cid": incident_cid,
            "phase": phase,
            "reason": reason,
            "target_ids": list(targets),
            "created_at_ms": (
                previous_created_at
                if isinstance(previous_created_at, int)
                and not isinstance(previous_created_at, bool)
                and previous_created_at >= 0
                else now_ms
            ),
            "updated_at_ms": now_ms,
            "independent_work_preserved": True,
            "completion_authority": False,
            "work_complete": False,
            **values,
        }
        if isinstance(previous, Mapping):
            for key in (
                "incident_kind",
                "operating_policy",
                "reason_codes",
            ):
                if key not in entry and key in previous:
                    entry[key] = previous[key]
        state_repair = state.get("state_repair")
        if isinstance(state_repair, Mapping):
            entry.setdefault("state_repair", dict(state_repair))
        state["incidents"][incident_cid] = entry
        self._store_state(state)
        self._publish("autonomous_unstall_" + phase, entry)
        return entry

    def _quarantine(
        self,
        state: dict[str, Any],
        diagnosis: RecoveryDiagnosis,
        reason: str,
        *,
        deterministic: Mapping[str, Any] | None = None,
        rescue: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        effect: Mapping[str, Any] = {}
        if self.quarantine_scope is not None:
            raw = self.quarantine_scope(
                diagnosis.target_ids, diagnosis.incident_cid, reason
            )
            if isinstance(raw, Mapping):
                effect = dict(raw)
        entry = self._record(
            state,
            diagnosis.incident_cid,
            phase="quarantined",
            reason=reason,
            targets=diagnosis.target_ids,
            quarantined=True,
            quarantine_effect=effect,
            deterministic=dict(deterministic or {}),
            rescue=dict(rescue or {}),
        )
        return self._result(diagnosis, entry, deduplicated=False)

    @staticmethod
    def _result(
        diagnosis: RecoveryDiagnosis,
        entry: Mapping[str, Any],
        *,
        deduplicated: bool,
    ) -> dict[str, Any]:
        result = {
            "schema": AUTONOMOUS_UNSTALL_RESULT_SCHEMA,
            "incident_cid": diagnosis.incident_cid,
            "incident_kind": diagnosis.kind.value,
            "target_ids": list(diagnosis.target_ids),
            "status": str(entry.get("phase") or ""),
            "reason": str(entry.get("reason") or ""),
            "recovered": entry.get("phase") == "recovered",
            "quarantined": bool(entry.get("quarantined")),
            "deduplicated": deduplicated,
            "independent_work_preserved": True,
            "completion_authority": False,
            "work_complete": False,
            "deterministic": dict(entry.get("deterministic") or {}),
            "rescue": dict(entry.get("rescue") or {}),
        }
        for key in ("health", "operating_policy", "state_repair"):
            value = entry.get(key)
            if isinstance(value, Mapping):
                result[key] = dict(value)
        return result

    def _wrapped_handlers(
        self,
        initial_roots: Mapping[str, str],
        handlers: Mapping[RescueOperation | str, Callable[[Any], Any]],
    ) -> dict[RescueOperation | str, Callable[[Any], Any]]:
        wrapped: dict[RescueOperation | str, Callable[[Any], Any]] = {}

        for operation, handler in handlers.items():
            def invoke(
                context: Any,
                *,
                selected: Callable[[Any], Any] = handler,
            ) -> Any:
                if self._roots() != dict(initial_roots):
                    return RecoveryActionObservation(
                        succeeded=False,
                        post_action_health={
                            "healthy": False,
                            "reason": "semantic_root_drift_before_action",
                        },
                        reason="semantic_root_drift_before_action",
                    )
                raw = selected(context)
                if self._roots() != dict(initial_roots):
                    return RecoveryActionObservation(
                        succeeded=False,
                        post_action_health={
                            "healthy": False,
                            "reason": "semantic_root_drift_after_action",
                        },
                        reason="semantic_root_drift_after_action",
                        partial=True,
                    )
                # The controller verifies exact observed effects separately.
                if isinstance(raw, Mapping):
                    checked = dict(raw)
                    checked["post_action_health"] = self._health()
                    return checked
                return raw

            wrapped[operation] = invoke
        return wrapped

    def _unstall_locked(
        self,
        *,
        evidence: Mapping[str, Any],
        prior_actions: Sequence[Mapping[str, Any]] = (),
        recovery_handlers: Mapping[
            RescueOperation | str, Callable[[Any], Any]
        ]
        | None = None,
    ) -> dict[str, Any]:
        """Run one finite incident-bound recovery pass."""

        if not self.policy.enabled:
            return {
                "schema": AUTONOMOUS_UNSTALL_RESULT_SCHEMA,
                "status": "disabled",
                "reason": "autonomous_unstall_disabled",
                "recovered": False,
                "quarantined": False,
                "deduplicated": False,
                "independent_work_preserved": True,
                "completion_authority": False,
                "work_complete": False,
            }
        unknown = set(evidence).difference(_AUTONOMOUS_UNSTALL_EVIDENCE_SLOTS)
        if unknown:
            raise RecoveryDiagnosticError(
                "unknown autonomous unstall evidence slots: "
                + ", ".join(sorted(unknown))
            )
        with self._lock:
            roots = self._roots()
            diagnosis = diagnose_supervisor_incident(
                repository_root=str(self.repository_root),
                state_root=str(self.state_dir.resolve()),
                repository_root_cid=roots["repository_root_cid"],
                policy_root=roots["policy_root"],
                run_cid=roots["run_cid"],
                prior_actions=prior_actions,
                observed_at_ms=self.clock_ms(),
                **dict(evidence),
            )
            state = self._load_state()
            existing = state["incidents"].get(diagnosis.incident_cid)
            if isinstance(existing, Mapping):
                phase = str(existing.get("phase") or "")
                if phase == "rescue_executing":
                    return self._quarantine(
                        state,
                        diagnosis,
                        "restart_during_rescue_uncertain_effects",
                        deterministic=existing.get("deterministic"),
                        rescue=existing.get("rescue"),
                    )
                if phase == "rescue_previewing":
                    return self._quarantine(
                        state,
                        diagnosis,
                        "restart_during_rescue_preview_provider_call_suppressed",
                        deterministic=existing.get("deterministic"),
                        rescue=existing.get("rescue"),
                    )
                if phase in {
                    "recovered",
                    "quarantined",
                    "rescue_previewed",
                }:
                    return self._result(
                        diagnosis, existing, deduplicated=True
                    )

            self._record(
                state,
                diagnosis.incident_cid,
                phase="deterministic_recovery",
                reason="semantic_incident_detected",
                targets=diagnosis.target_ids,
                incident_kind=diagnosis.kind.value,
                reason_codes=list(diagnosis.reason_codes),
                operating_policy={
                    "operating_policy_id": self.policy.operating_policy_id,
                    "rescue_preview_enabled": (
                        self.policy.rescue_preview_enabled
                    ),
                    "rescue_execution_enabled": (
                        self.policy.rescue_execution_enabled
                    ),
                    "allow_provider_calls": self.policy.allow_provider_calls,
                    "cooldown_ms": self.policy.cooldown_ms,
                    "deadline_ms": self.policy.deadline_ms,
                    "max_total_attempts": (
                        self.policy.deterministic_max_total_attempts
                    ),
                    "max_actions": self.policy.deterministic_max_actions,
                    "max_provider_calls": self.policy.max_provider_calls,
                    "max_rescue_executions": (
                        self.policy.max_rescue_executions
                    ),
                    "circuit_breaker_failures": (
                        self.policy.circuit_breaker_failures
                    ),
                },
            )
            handlers = {
                **self.recovery_handlers,
                **dict(recovery_handlers or {}),
            }
            controller = ProgrammaticRecoveryController(
                self.recovery_state_dir,
                handlers=self._wrapped_handlers(roots, handlers),
                health_check=lambda _context, _observation: self._health(),
                policy=self.policy.deterministic_policy,
                clock_ms=self.clock_ms,
            )
            deterministic_result: ProgrammaticRecoveryResult = (
                controller.recover(diagnosis)
            )
            deterministic = {
                "terminal_cid": deterministic_result.terminal_cid,
                "recovered": deterministic_result.recovered,
                "quarantined": deterministic_result.quarantined,
                "deduplicated": deterministic_result.deduplicated,
                "attempt_count": len(deterministic_result.attempts),
                "attempts": [
                    item.to_record() for item in deterministic_result.attempts
                ],
                "exhaustion_receipt_cid": (
                    ""
                    if deterministic_result.exhaustion_receipt is None
                    else deterministic_result.exhaustion_receipt.receipt_cid
                ),
            }
            if self._roots() != roots:
                return self._quarantine(
                    state,
                    diagnosis,
                    "semantic_root_drift_after_deterministic_recovery",
                    deterministic=deterministic,
                )
            health = self._health()
            if deterministic_result.recovered and self._healthy(health):
                entry = self._record(
                    state,
                    diagnosis.incident_cid,
                    phase="recovered",
                    reason="deterministic_health_restored",
                    targets=diagnosis.target_ids,
                    deterministic=deterministic,
                    health=health,
                    quarantined=False,
                )
                return self._result(
                    diagnosis, entry, deduplicated=deterministic_result.deduplicated
                )
            if (
                deterministic_result.receipt is not None
                and deterministic_result.receipt.quarantined
            ):
                return self._quarantine(
                    state,
                    diagnosis,
                    "deterministic_scope_quarantined",
                    deterministic=deterministic,
                )
            exhaustion = deterministic_result.exhaustion_receipt
            if exhaustion is None:
                return self._quarantine(
                    state,
                    diagnosis,
                    "deterministic_terminal_receipt_missing",
                    deterministic=deterministic,
                )

            rescue_allowed = bool(
                self.policy.rescue_preview_enabled
                and self.policy.allow_provider_calls
                and self.policy.operating_policy_id
                and exhaustion.status is RecordStatus.QUARANTINED
                and not exhaustion.circuit_open
            )
            if not rescue_allowed:
                return self._quarantine(
                    state,
                    diagnosis,
                    "rescue_not_permitted_after_deterministic_exhaustion",
                    deterministic=deterministic,
                )
            if (
                self.rescue_planner is None
                or self.rescue_request_factory is None
            ):
                return self._quarantine(
                    state,
                    diagnosis,
                    "rescue_preview_adapter_unavailable",
                    deterministic=deterministic,
                )

            runtime = state.setdefault("rescue_runtime", {})
            provider_calls = int(runtime.get("provider_calls") or 0)
            executions = int(runtime.get("executions") or 0)
            failures = int(runtime.get("consecutive_failures") or 0)
            last_provider_call_ms = int(
                runtime.get("last_provider_call_ms") or 0
            )
            if runtime.get("circuit_open") is True:
                return self._quarantine(
                    state,
                    diagnosis,
                    "persistent_rescue_circuit_open",
                    deterministic=deterministic,
                )
            if provider_calls >= self.policy.max_provider_calls:
                runtime["circuit_open"] = True
                runtime["reason"] = "provider_call_budget_exhausted"
                return self._quarantine(
                    state,
                    diagnosis,
                    "persistent_provider_call_budget_exhausted",
                    deterministic=deterministic,
                )
            now_ms = self.clock_ms()
            if (
                last_provider_call_ms
                and now_ms
                < last_provider_call_ms + self.policy.cooldown_ms
            ):
                return self._quarantine(
                    state,
                    diagnosis,
                    "persistent_rescue_cooldown_active",
                    deterministic=deterministic,
                )
            # Reserve the non-renewable provider-call slot before invocation.
            # A crash cannot make an uncertain call replayable.
            runtime.update(
                {
                    "provider_calls": provider_calls + 1,
                    "executions": executions,
                    "consecutive_failures": failures,
                    "last_provider_call_ms": now_ms,
                    "circuit_open": False,
                }
            )
            self._record(
                state,
                diagnosis.incident_cid,
                phase="rescue_previewing",
                reason="current_deterministic_exhaustion_qualified",
                targets=diagnosis.target_ids,
                deterministic=deterministic,
                exhaustion_receipt_cid=exhaustion.receipt_cid,
            )
            try:
                planning_request = self.rescue_request_factory(
                    diagnosis, exhaustion, roots
                )
            except Exception as exc:
                failures += 1
                runtime["consecutive_failures"] = failures
                if failures >= self.policy.circuit_breaker_failures:
                    runtime["circuit_open"] = True
                    runtime["reason"] = "preview_adapter_failure_threshold"
                return self._quarantine(
                    state,
                    diagnosis,
                    "rescue_preview_adapter_failed",
                    deterministic=deterministic,
                    rescue={
                        "provider_invoked": False,
                        "executed": False,
                        "error_type": type(exc).__name__,
                    },
                )
            try:
                planning = self.rescue_planner.plan(planning_request)
            except Exception as exc:
                # The provider may already have received the request.  Its
                # pre-reserved slot is deliberately not refunded or replayed.
                failures += 1
                runtime["consecutive_failures"] = failures
                if failures >= self.policy.circuit_breaker_failures:
                    runtime["circuit_open"] = True
                    runtime["reason"] = "planner_exception_threshold"
                return self._quarantine(
                    state,
                    diagnosis,
                    "rescue_planner_failed",
                    deterministic=deterministic,
                    rescue={
                        "provider_invoked": True,
                        "provider_effect_uncertain": True,
                        "executed": False,
                        "error_type": type(exc).__name__,
                    },
                )
            if planning.proposed:
                runtime["consecutive_failures"] = 0
            else:
                failures += 1
                runtime["consecutive_failures"] = failures
                if failures >= self.policy.circuit_breaker_failures:
                    runtime["circuit_open"] = True
                    runtime["reason"] = "planner_failure_threshold"
            rescue = {
                "planning": planning.to_dict(),
                "provider_invoked": planning.provider_invoked,
                "executed": False,
            }
            if self._roots() != roots:
                return self._quarantine(
                    state,
                    diagnosis,
                    "semantic_root_drift_after_rescue_preview",
                    deterministic=deterministic,
                    rescue=rescue,
                )
            if not planning.proposed:
                return self._quarantine(
                    state,
                    diagnosis,
                    str(planning.reason_code or "rescue_no_plan"),
                    deterministic=deterministic,
                    rescue=rescue,
                )
            if not self.policy.rescue_execution_enabled:
                entry = self._record(
                    state,
                    diagnosis.incident_cid,
                    phase="rescue_previewed",
                    reason="rescue_execution_not_permitted",
                    targets=diagnosis.target_ids,
                    deterministic=deterministic,
                    rescue=rescue,
                    quarantined=True,
                )
                return self._result(diagnosis, entry, deduplicated=False)
            if (
                self.rescue_orchestrator is None
                or self.rescue_execution_request_factory is None
            ):
                return self._quarantine(
                    state,
                    diagnosis,
                    "rescue_execution_adapter_unavailable",
                    deterministic=deterministic,
                    rescue=rescue,
                )
            if executions >= self.policy.max_rescue_executions:
                runtime["circuit_open"] = True
                runtime["reason"] = "rescue_execution_budget_exhausted"
                return self._quarantine(
                    state,
                    diagnosis,
                    "persistent_rescue_execution_budget_exhausted",
                    deterministic=deterministic,
                    rescue=rescue,
                )

            runtime["executions"] = executions + 1
            self._record(
                state,
                diagnosis.incident_cid,
                phase="rescue_executing",
                reason="explicit_operating_policy_permits_execution",
                targets=diagnosis.target_ids,
                deterministic=deterministic,
                rescue=rescue,
            )
            try:
                execution_request = (
                    self.rescue_execution_request_factory(
                        diagnosis, exhaustion, planning.plan, roots
                    )
                )
            except Exception as exc:
                runtime["consecutive_failures"] = failures + 1
                rescue["execution_request_created"] = False
                rescue["error_type"] = type(exc).__name__
                return self._quarantine(
                    state,
                    diagnosis,
                    "rescue_execution_adapter_failed",
                    deterministic=deterministic,
                    rescue=rescue,
                )
            try:
                execution = self.rescue_orchestrator.execute(
                    execution_request
                )
            except Exception as exc:
                # The orchestrator may have crossed an effect boundary before
                # failing.  Persist uncertainty and quarantine; never retry it.
                runtime["consecutive_failures"] = failures + 1
                runtime["circuit_open"] = True
                runtime["reason"] = "rescue_execution_effect_uncertain"
                rescue["execution_started"] = True
                rescue["execution_effect_uncertain"] = True
                rescue["error_type"] = type(exc).__name__
                return self._quarantine(
                    state,
                    diagnosis,
                    "rescue_execution_failed_with_uncertain_effects",
                    deterministic=deterministic,
                    rescue=rescue,
                )
            rescue["executed"] = True
            rescue["execution"] = dict(execution.to_dict())
            if self._roots() != roots:
                return self._quarantine(
                    state,
                    diagnosis,
                    "semantic_root_drift_after_rescue_execution",
                    deterministic=deterministic,
                    rescue=rescue,
                )
            health = self._health()
            if bool(execution.recovered) and self._healthy(health):
                entry = self._record(
                    state,
                    diagnosis.incident_cid,
                    phase="recovered",
                    reason="rescue_health_restored",
                    targets=diagnosis.target_ids,
                    deterministic=deterministic,
                    rescue=rescue,
                    health=health,
                    quarantined=False,
                )
                return self._result(diagnosis, entry, deduplicated=False)
            return self._quarantine(
                state,
                diagnosis,
                "rescue_stopped_without_verified_health",
                deterministic=deterministic,
                rescue=rescue,
            )

    def unstall(
        self,
        *,
        evidence: Mapping[str, Any],
        prior_actions: Sequence[Mapping[str, Any]] = (),
        recovery_handlers: Mapping[
            RescueOperation | str, Callable[[Any], Any]
        ]
        | None = None,
    ) -> dict[str, Any]:
        """Serialize durable incident/provider state across processes."""

        self.state_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.state_dir / ".autonomous-unstall.lock"
        with lock_path.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                return self._unstall_locked(
                    evidence=evidence,
                    prior_actions=prior_actions,
                    recovery_handlers=recovery_handlers,
                )
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    recover = unstall


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pid_alive(pid: int) -> bool:
    """Check if a process with given PID is alive."""
    if isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Permission denial still proves that a process owns the PID.
        return True
    except OSError:
        return False


def _lifecycle_state(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    text = _STATE_ALIASES.get(text, text)
    return text if text in _LIFECYCLE_STATE_SET else ""


def _timestamp_seconds(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        # LifecycleStatus exposes millisecond timestamps.
        return number / 1000.0 if number > 10_000_000_000 else number
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _nonnegative_int(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, result)


def _timestamp_text_from_ms(value: int) -> str:
    if value <= 0:
        return ""
    return (
        datetime.fromtimestamp(value / 1000.0, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _status_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _active_leases(payload: Mapping[str, Any]) -> list[str]:
    value = payload.get("active_leases")
    if isinstance(value, (list, tuple)):
        return sorted(
            {str(item).strip() for item in value if str(item).strip()}
        )[:256]
    if isinstance(value, Mapping):
        return sorted(
            {str(item).strip() for item in value if str(item).strip()}
        )[:256]
    lease_id = str(payload.get("lease_id") or "").strip()
    return [lease_id] if lease_id else []


def lifecycle_status_projection(
    *,
    pid_check: Mapping[str, Any],
    heartbeat_check: Mapping[str, Any],
    state: str | None = None,
    phase: str | None = None,
    target_id: str = "supervisor:lane",
) -> dict[str, Any]:
    """Project lane observations into the shared lifecycle status schema.

    The control plane owns lifecycle mutation policy.  The watchdog only
    reconciles process/heartbeat observations and emits the same public shape,
    so a caller never has to interpret a second set of health fields.
    """

    observed_state = _lifecycle_state(state or heartbeat_check.get("state"))
    alive = bool(pid_check.get("alive"))
    stale = bool(heartbeat_check.get("stale"))
    if not observed_state:
        observed_state = "healthy" if alive and not stale else (
            "degraded" if alive else "stopped"
        )
    if alive and stale and observed_state not in _INTENTIONAL_NON_RUNNING_STATES:
        observed_state = "degraded"
    elif not alive and observed_state == "healthy":
        observed_state = "degraded"

    heartbeat_at = str(heartbeat_check.get("heartbeat_at") or "")
    heartbeat_at_ms = heartbeat_check.get("heartbeat_at_ms")
    if heartbeat_at_ms is None:
        timestamp = _timestamp_seconds(heartbeat_at)
        heartbeat_at_ms = None if timestamp is None else int(timestamp * 1000)
    heartbeat_at_ms = _nonnegative_int(heartbeat_at_ms)
    heartbeat_at = _timestamp_text_from_ms(heartbeat_at_ms)

    updated_at = str(heartbeat_check.get("updated_at") or heartbeat_at or "")
    updated_at_ms = heartbeat_check.get("updated_at_ms")
    if updated_at_ms is None:
        timestamp = _timestamp_seconds(updated_at)
        updated_at_ms = None if timestamp is None else int(timestamp * 1000)
    updated_at_ms = _nonnegative_int(updated_at_ms)
    updated_at = _timestamp_text_from_ms(updated_at_ms)

    reasons = heartbeat_check.get("backpressure_reasons")
    if not isinstance(reasons, (list, tuple)):
        reasons = []
    leases = sorted(
        {
            str(item).strip()
            for item in heartbeat_check.get("active_leases", ())
            if str(item).strip()
        }
    )[:256]
    result: dict[str, Any] = {
        "schema": LIFECYCLE_STATUS_SCHEMA,
        "target_id": str(target_id or "supervisor:lane"),
        "state": observed_state,
        "phase": str(phase or heartbeat_check.get("phase") or observed_state),
        "heartbeat_at_ms": heartbeat_at_ms,
        "heartbeat_at": heartbeat_at,
        "pid": (
            pid_check.get("pid")
            if isinstance(pid_check.get("pid"), int)
            and not isinstance(pid_check.get("pid"), bool)
            and pid_check.get("pid") > 0
            else None
        ),
        "active_leases": leases,
        "active_lease_count": len(leases),
        "refill_state": str(heartbeat_check.get("refill_state") or "idle"),
        "backpressure": bool(heartbeat_check.get("backpressure", False)),
        "backpressure_reasons": sorted(
            {str(item).strip() for item in reasons if str(item).strip()}
        )[:256],
        "terminal_reason": str(heartbeat_check.get("terminal_reason") or ""),
        "transition_id": str(heartbeat_check.get("transition_id") or ""),
        "generation": _nonnegative_int(heartbeat_check.get("generation")),
        "fencing_epoch": (
            None
            if heartbeat_check.get("fencing_epoch") in (None, "")
            else _nonnegative_int(heartbeat_check.get("fencing_epoch"))
        ),
        "updated_at_ms": updated_at_ms,
        "updated_at": updated_at,
    }
    return result


def watchdog_process_status(
    state: str,
    *,
    phase: str,
    terminal_reason: str = "",
    backpressure_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a canonical status for watchdog-level outcomes."""

    now_ms = int(time.time() * 1000)
    running = state != "stopped"
    return lifecycle_status_projection(
        pid_check={
            "pid": os.getpid() if running else None,
            "alive": running,
        },
        heartbeat_check={
            "state": state,
            "phase": phase,
            "heartbeat_at_ms": now_ms,
            "updated_at_ms": now_ms,
            "active_leases": (),
            "refill_state": "idle",
            "backpressure": bool(backpressure_reasons),
            "backpressure_reasons": tuple(backpressure_reasons),
            "terminal_reason": terminal_reason,
            "generation": 0,
            "fencing_epoch": None,
        },
        target_id="supervisor:watchdog",
    )


def read_lane_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read the bundle lane manifest JSON."""
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read manifest %s: %s", manifest_path, exc)
        return {}


def read_scheduler_snapshot(manifest_path: Path) -> dict[str, Any]:
    """Return the exact operator snapshot embedded by the live scheduler."""

    manifest = read_lane_manifest(manifest_path)
    snapshot = manifest.get("scheduler_snapshot")
    return dict(snapshot) if isinstance(snapshot, dict) else {}


def check_lane_pid(state_dir: Path, state_prefix: str) -> dict[str, Any]:
    """Check if a lane's supervisor process is alive by its PID file."""
    pid_path = state_dir / f"{state_prefix}_bundle_supervisor.pid"
    result: dict[str, Any] = {"pid_path": str(pid_path), "alive": False}

    if not pid_path.exists():
        result["reason"] = "no_pid_file"
        return result

    try:
        pid = int(pid_path.read_text().strip())
        result["pid"] = pid
        if pid <= 0:
            result["reason"] = "invalid_pid"
        elif pid_alive(pid):
            result["alive"] = True
        else:
            result["reason"] = "process_dead"
    except (ValueError, OSError) as exc:
        result["reason"] = f"pid_read_error: {exc}"

    return result


def check_lane_heartbeat(state_dir: Path, state_prefix: str, *, timeout_seconds: float) -> dict[str, Any]:
    """Check if a lane has updated its status file recently."""
    status_path = state_dir / f"{state_prefix}_status.json"
    result: dict[str, Any] = {"status_path": str(status_path), "stale": False}

    if not status_path.exists():
        result["stale"] = True
        result["reason"] = "no_status_file"
        return result

    try:
        stat = status_path.stat()
        status = _status_payload(status_path)
        if status.get("heartbeat_at_ms") not in (None, ""):
            try:
                heartbeat_timestamp = float(status["heartbeat_at_ms"]) / 1000.0
            except (TypeError, ValueError):
                heartbeat_timestamp = None
        elif status.get("heartbeat_at") not in (None, ""):
            heartbeat_timestamp = _timestamp_seconds(status["heartbeat_at"])
        elif status.get("updated_at_ms") not in (None, ""):
            try:
                heartbeat_timestamp = float(status["updated_at_ms"]) / 1000.0
            except (TypeError, ValueError):
                heartbeat_timestamp = None
        else:
            heartbeat_timestamp = _timestamp_seconds(
                status.get("updated_at") or status.get("timestamp")
            )
        age_seconds = time.time() - (
            heartbeat_timestamp if heartbeat_timestamp is not None else stat.st_mtime
        )
        # Future timestamps can occur under small host clock skews.
        age_seconds = max(0.0, age_seconds)
        result["age_seconds"] = age_seconds
        if age_seconds > timeout_seconds:
            result["stale"] = True
            result["reason"] = "heartbeat_timeout"
        result["phase"] = str(status.get("active_phase") or status.get("phase") or "")
        result["schema"] = str(status.get("schema") or "")
        result["active_task_id"] = str(status.get("active_task_id") or "")
        result["heartbeat_at"] = str(status.get("heartbeat_at") or "")
        result["heartbeat_at_ms"] = status.get("heartbeat_at_ms")
        result["updated_at"] = str(status.get("updated_at") or "")
        result["updated_at_ms"] = status.get("updated_at_ms")
        result["state"] = _lifecycle_state(
            status.get("state")
            or status.get("lifecycle_state")
            or status.get("status")
        )
        result["active_leases"] = _active_leases(status)
        result["refill_state"] = status.get("refill_state")
        result["backpressure"] = bool(status.get("backpressure", False))
        raw_reasons = status.get("backpressure_reasons")
        result["backpressure_reasons"] = (
            list(raw_reasons[:256])
            if isinstance(raw_reasons, (list, tuple))
            else []
        )
        result["terminal_reason"] = str(
            status.get("terminal_reason") or status.get("reason") or ""
        )
        result["transition_id"] = str(status.get("transition_id") or "")
        result["generation"] = status.get("generation") or 0
        result["fencing_epoch"] = status.get("fencing_epoch")
        status_pid = status.get("pid")
        if status_pid not in (None, ""):
            try:
                result["status_pid"] = int(status_pid)
            except (TypeError, ValueError):
                result["state_inconsistent"] = True
                result["state_reason"] = "invalid_status_pid"
    except OSError as exc:
        result["stale"] = True
        result["reason"] = f"stat_error: {exc}"

    return result


def _replace_pid_file(pid_path: Path, pid: int) -> None:
    """Atomically repair a PID file from a live canonical status record."""

    temporary = pid_path.with_name(
        f".{pid_path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    try:
        temporary.write_text(f"{pid}\n", encoding="utf-8")
        os.replace(temporary, pid_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def restart_lane(
    lane_info: dict[str, Any],
    *,
    repo_root: Path,
    lifecycle_transition: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    | None = None,
) -> dict[str, Any]:
    """Delegate restart to an authorized fenced lifecycle transition.

    A watchdog observation and a manifest command are not signal or launch
    authority.  In particular, a dead PID file cannot prove that detached
    descendants are absent.  The embedding control service supplies a callback
    which resolves the bound ``OperationRequest`` and invokes
    :class:`LifecycleOrchestrator`; without it the watchdog reports the missing
    authority instead of performing a raw ``Popen``.
    """

    del repo_root
    if lifecycle_transition is None:
        return {
            "restarted": False,
            "reason": "lifecycle_orchestrator_required",
            "control_recovery_required": True,
        }
    try:
        raw = lifecycle_transition(dict(lane_info))
    except Exception as exc:
        return {
            "restarted": False,
            "reason": f"lifecycle_transition_failed: {exc}",
            "control_recovery_required": True,
        }
    result = dict(raw) if isinstance(raw, Mapping) else {}
    transition = result.get("transition")
    if isinstance(transition, Mapping):
        result.setdefault("receipt_id", transition.get("receipt_id"))
        new_tree = transition.get("new_tree")
        if isinstance(new_tree, Mapping):
            members = new_tree.get("members")
            if isinstance(members, Sequence) and members:
                first = members[0]
                if isinstance(first, Mapping):
                    result.setdefault("new_pid", first.get("pid"))
    restarted = bool(
        result.get("restarted")
        or result.get("succeeded")
        or result.get("status") == "succeeded"
        or (
            isinstance(transition, Mapping)
            and transition.get("phase") == "committed"
        )
    )
    result["restarted"] = restarted
    result.setdefault("pid_persisted", False)
    if restarted and not isinstance(result.get("new_pid"), int):
        return {
            **result,
            "restarted": False,
            "reason": "lifecycle_receipt_missing_new_process_identity",
        }
    return result


def aggregate_logs(
    lanes: Sequence[dict[str, Any]],
    *,
    repo_root: Path,
    output_dir: Path,
    max_lines_per_lane: int = 100,
) -> dict[str, Any]:
    """Aggregate recent log lines from all lanes into a unified structured log.

    This provides a single view of what's happening across all parallel lanes
    without needing to tail multiple log files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregated: list[dict[str, Any]] = []

    for lane in lanes:
        log_path_str = lane.get("log_path", "")
        if not log_path_str:
            continue
        log_path = Path(log_path_str) if Path(log_path_str).is_absolute() else repo_root / log_path_str
        if not log_path.exists():
            continue

        bundle_key = lane.get("bundle_key", "unknown")
        try:
            # Read last N lines efficiently
            with log_path.open("rb") as f:
                # Seek to end, read backward to find last N lines
                f.seek(0, 2)
                file_size = f.tell()
                # Read last 64KB or whole file
                read_size = min(file_size, 65536)
                f.seek(max(0, file_size - read_size))
                tail_bytes = f.read()

            lines = tail_bytes.decode("utf-8", errors="replace").splitlines()[-max_lines_per_lane:]
            for line in lines:
                aggregated.append({
                    "lane": bundle_key,
                    "line": line,
                })
        except OSError:
            continue

    # Write aggregated log
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"aggregated_{timestamp}.jsonl"
    try:
        with output_path.open("w", encoding="utf-8") as f:
            for entry in aggregated:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        return {"aggregated": False, "error": str(exc)}

    # Prune old aggregated logs (keep last 24)
    existing = sorted(output_dir.glob("aggregated_*.jsonl"))
    for old_file in existing[:-24]:
        try:
            old_file.unlink()
        except OSError:
            pass

    return {
        "aggregated": True,
        "output_path": str(output_path),
        "total_lines": len(aggregated),
        "lane_count": len(lanes),
    }


class SupervisorWatchdog:
    """Outer watchdog that monitors bundle supervisor lanes."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        repo_root: Path,
        check_interval: float = 120.0,
        lane_timeout: float = 600.0,
        max_consecutive_restarts: int = 5,
        log_aggregation_dir: Path | None = None,
        lifecycle_restart: Callable[
            [Mapping[str, Any]], Mapping[str, Any]
        ]
        | None = None,
        autonomous_unstall_policy: AutonomousUnstallPolicy | None = None,
        autonomous_unstall_factory: Callable[..., AutonomousUnstallCoordinator]
        | None = None,
        control_event_publisher: Callable[
            [str, Mapping[str, Any]], Any
        ]
        | None = None,
        rescue_planner: RescuePlanner | None = None,
        rescue_request_factory: Callable[
            [RecoveryDiagnosis, Any, Mapping[str, str]], RescuePlanningRequest
        ]
        | None = None,
        rescue_orchestrator: Any = None,
        rescue_execution_request_factory: Callable[
            [RecoveryDiagnosis, Any, Any, Mapping[str, str]], Any
        ]
        | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.repo_root = repo_root
        self.check_interval = check_interval
        self.lane_timeout = lane_timeout
        self.max_consecutive_restarts = max_consecutive_restarts
        self.log_aggregation_dir = log_aggregation_dir or (
            manifest_path.parent / "logs" / "aggregated"
        )
        self.lifecycle_restart = lifecycle_restart
        self.autonomous_unstall_policy = autonomous_unstall_policy
        self.autonomous_unstall_factory = (
            autonomous_unstall_factory or AutonomousUnstallCoordinator
        )
        self.control_event_publisher = control_event_publisher
        self.rescue_planner = rescue_planner
        self.rescue_request_factory = rescue_request_factory
        self.rescue_orchestrator = rescue_orchestrator
        self.rescue_execution_request_factory = (
            rescue_execution_request_factory
        )
        self._consecutive_restart_counts: dict[str, int] = {}
        self._recent_restarts: dict[str, tuple[int, float]] = {}
        self._generation = 0
        self._running = True

    @staticmethod
    def _manifest_unstall_policy(
        manifest: Mapping[str, Any],
    ) -> AutonomousUnstallPolicy | None:
        raw = manifest.get("autonomous_unstall_policy")
        if not isinstance(raw, Mapping):
            return None
        allowed = {
            item.name
            for item in AutonomousUnstallPolicy.__dataclass_fields__.values()
        }
        values = {key: value for key, value in raw.items() if key in allowed}
        return AutonomousUnstallPolicy(**values)

    def _watchdog_unstall(
        self,
        *,
        manifest: Mapping[str, Any],
        lane: Mapping[str, Any],
        lane_started: Mapping[str, Any],
        bundle_key: str,
        state_dir: Path,
        state_prefix: str,
        pid_check: Mapping[str, Any],
        heartbeat_check: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        policy = (
            self.autonomous_unstall_policy
            or self._manifest_unstall_policy(manifest)
        )
        if policy is None or not policy.enabled:
            return None
        restart_info = dict(lane_started or lane)
        restart_info.setdefault("pid_path", str(pid_check.get("pid_path") or ""))

        def current_health() -> Mapping[str, Any]:
            current_pid = check_lane_pid(state_dir, state_prefix)
            current_heartbeat = check_lane_heartbeat(
                state_dir,
                state_prefix,
                timeout_seconds=self.lane_timeout,
            )
            healthy = bool(
                current_pid.get("alive")
                and not current_heartbeat.get("stale")
                and not current_heartbeat.get("state_inconsistent")
            )
            return {
                "healthy": healthy,
                "state": "healthy" if healthy else "degraded",
                "pid_alive": bool(current_pid.get("alive")),
                "heartbeat_stale": bool(current_heartbeat.get("stale")),
                "work_complete": False,
            }

        handlers: dict[RescueOperation, Callable[[Any], Any]] = {}
        if self.lifecycle_restart is not None:
            def restart(context: Any) -> Mapping[str, Any]:
                result = restart_lane(
                    restart_info,
                    repo_root=self.repo_root,
                    lifecycle_transition=self.lifecycle_restart,
                )
                if not result.get("restarted"):
                    return {
                        "succeeded": False,
                        "observed_effects": (),
                        "reason": str(result.get("reason") or "restart_failed"),
                    }
                new_pid = result.get("new_pid")
                if isinstance(new_pid, int):
                    self._recent_restarts[bundle_key] = (
                        new_pid,
                        time.monotonic(),
                    )
                return {
                    "succeeded": True,
                    "observed_effects": context.action.expected_effects,
                    "reason": "fenced_lane_restart_committed",
                }

            handlers[RescueOperation.RESTART_LANE] = restart

        def current_roots(
            current_manifest: Mapping[str, Any],
        ) -> dict[str, str]:
            if not current_manifest:
                return {}
            current_lanes = current_manifest.get("lanes")
            if not isinstance(current_lanes, Sequence):
                return {}
            current_lane = next(
                (
                    item
                    for item in current_lanes
                    if isinstance(item, Mapping)
                    and str(item.get("bundle_key") or "") == bundle_key
                ),
                None,
            )
            if current_lane is None:
                return {}
            identity = {
                "repository": str(self.repo_root.resolve()),
                "manifest": str(self.manifest_path.resolve()),
                "tree": str(current_manifest.get("tree_id") or ""),
                "bundle_key": bundle_key,
                "state_dir": str(current_lane.get("state_dir") or ""),
                "state_prefix": str(current_lane.get("state_prefix") or ""),
            }
            return {
                "repository_root_cid": str(
                    current_manifest.get("repository_root_cid")
                    or prompt_workflow_cid(
                        {"watchdog-repository": identity}
                    )
                ),
                "policy_root": str(
                    current_manifest.get("policy_root")
                    or prompt_workflow_cid(
                        {
                            "watchdog-policy": (
                                policy.operating_policy_id
                                or "deterministic-only"
                            )
                        }
                    )
                ),
                "run_cid": str(
                    current_manifest.get("run_cid")
                    or prompt_workflow_cid(
                        {"watchdog-lane-run": identity}
                    )
                ),
            }

        roots = current_roots(manifest)
        coordinator = self.autonomous_unstall_factory(
            state_dir=state_dir / "autonomous-unstall",
            repository_root=self.repo_root,
            repository_root_cid=roots["repository_root_cid"],
            policy_root=roots["policy_root"],
            run_cid=roots["run_cid"],
            policy=policy,
            recovery_handlers=handlers,
            health_probe=current_health,
            root_probe=lambda: current_roots(
                read_lane_manifest(self.manifest_path)
            ),
            quarantine_scope=lambda targets, incident, reason: {
                "target_ids": list(targets),
                "incident_cid": incident,
                "reason": reason,
                "scope": "lane",
            },
            event_publisher=self.control_event_publisher,
            rescue_planner=self.rescue_planner,
            rescue_request_factory=self.rescue_request_factory,
            rescue_orchestrator=self.rescue_orchestrator,
            rescue_execution_request_factory=(
                self.rescue_execution_request_factory
            ),
        )
        return coordinator.unstall(
            evidence={
                "status": {
                    "lane_id": bundle_key,
                    "state": heartbeat_check.get("state") or "degraded",
                    "state_inconsistent": bool(
                        heartbeat_check.get("state_inconsistent")
                    ),
                },
                "health": {
                    "lane_id": bundle_key,
                    "healthy": False,
                    "reason": (
                        heartbeat_check.get("state_reason")
                        or heartbeat_check.get("reason")
                        or pid_check.get("reason")
                        or "lane_unhealthy"
                    ),
                },
                "process": {
                    "lane_id": bundle_key,
                    "alive": bool(pid_check.get("alive")),
                    "failed": not bool(pid_check.get("alive")),
                    "pid": pid_check.get("pid"),
                },
                "heartbeat": {
                    "lane_id": bundle_key,
                    "stale": bool(heartbeat_check.get("stale")),
                    "age_ms": int(
                        float(heartbeat_check.get("age_seconds") or 0) * 1000
                    ),
                },
            }
        )

    def run(self) -> dict[str, Any]:
        """Run the watchdog loop indefinitely."""
        logger.info(
            "Watchdog started: manifest=%s, check_interval=%.0fs, lane_timeout=%.0fs",
            self.manifest_path,
            self.check_interval,
            self.lane_timeout,
        )

        # Handle SIGTERM gracefully
        def handle_signal(signum, frame):
            logger.info("Watchdog received signal %d, shutting down", signum)
            self._running = False

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        total_checks = 0
        total_restarts = 0

        while self._running:
            try:
                report = self._check_cycle()
                total_checks += 1
                total_restarts += report.get("restarts", 0)

                if report.get("restarts", 0) > 0:
                    logger.warning(
                        "Watchdog cycle %d: restarted %d lanes",
                        total_checks,
                        report["restarts"],
                    )
                else:
                    logger.debug("Watchdog cycle %d: all lanes healthy", total_checks)

            except Exception as exc:
                logger.error("Watchdog cycle error: %s", exc, exc_info=True)

            # Sleep in small increments so we can respond to signals
            sleep_remaining = self.check_interval
            while sleep_remaining > 0 and self._running:
                chunk = min(sleep_remaining, 5.0)
                time.sleep(chunk)
                sleep_remaining -= chunk

        return {
            "status": watchdog_process_status(
                "stopped",
                phase="watchdog_stopped",
                terminal_reason="watchdog_shutdown",
            ),
            "legacy_status": "shutdown",
            "total_checks": total_checks,
            "total_restarts": total_restarts,
        }

    def _check_cycle(self) -> dict[str, Any]:
        """Run one health-check cycle across all lanes."""
        manifest = read_lane_manifest(self.manifest_path)
        if not manifest:
            return {
                "timestamp": utc_now(),
                "error": "manifest_empty",
                "restarts": 0,
                "status": watchdog_process_status(
                    "blocked",
                    phase="manifest_unavailable",
                    backpressure_reasons=("manifest_empty",),
                ),
            }

        lanes = manifest.get("lanes", [])
        started = manifest.get("started", [])
        dynamic_authority = bool(manifest.get("authoritative")) and str(
            manifest.get("schema") or ""
        ).startswith("ipfs_accelerate_py.agent_supervisor.dynamic_bundle_scheduler")

        restarts = 0
        reports: list[dict[str, Any]] = []

        for i, lane in enumerate(lanes):
            bundle_key = lane.get("bundle_key", f"lane_{i}")
            state_dir = Path(lane.get("state_dir", ""))
            state_prefix = lane.get("state_prefix", "")

            if not state_dir.is_absolute():
                state_dir = self.repo_root / state_dir

            # Find matching started info
            lane_started = next(
                (s for s in started if s.get("bundle_key") == bundle_key),
                {},
            )

            # Check PID
            pid_check = check_lane_pid(state_dir, state_prefix)

            # Check heartbeat
            heartbeat_check = check_lane_heartbeat(
                state_dir, state_prefix, timeout_seconds=self.lane_timeout
            )
            canonical_status = (
                heartbeat_check.get("schema") == LIFECYCLE_STATUS_SCHEMA
            )
            observed_state = _lifecycle_state(heartbeat_check.get("state"))
            recovery: dict[str, Any] = {}

            # The canonical status binds the process it describes.  Repair a
            # stale PID file from a live status PID instead of launching an
            # unfenced duplicate.
            status_pid = heartbeat_check.get("status_pid")
            pid_file_pid = pid_check.get("pid")
            untrusted_live_status_pid = False
            if (
                canonical_status
                and isinstance(status_pid, int)
                and status_pid > 0
                and status_pid != pid_file_pid
            ):
                status_pid_alive = pid_alive(status_pid)
                if status_pid_alive and not heartbeat_check.get("stale", False):
                    pid_path = Path(str(pid_check["pid_path"]))
                    try:
                        pid_path.parent.mkdir(parents=True, exist_ok=True)
                        _replace_pid_file(pid_path, status_pid)
                    except OSError as exc:
                        heartbeat_check["state_inconsistent"] = True
                        heartbeat_check["state_reason"] = (
                            f"pid_file_repair_error: {exc}"
                        )
                    else:
                        recovery = {
                            "kind": "stale_pid_file",
                            "previous_pid": pid_file_pid,
                            "recovered_pid": status_pid,
                        }
                        self._generation += 1
                        pid_check = {
                            **pid_check,
                            "pid": status_pid,
                            "alive": True,
                        }
                        pid_check.pop("reason", None)
                elif status_pid_alive:
                    untrusted_live_status_pid = True
                    heartbeat_check["state_inconsistent"] = True
                    heartbeat_check["state_reason"] = "stale_status_pid_alive"
                else:
                    heartbeat_check["state_inconsistent"] = True
                    heartbeat_check["state_reason"] = "status_pid_dead"

            report = {
                "bundle_key": bundle_key,
                "pid_check": pid_check,
                "heartbeat_check": heartbeat_check,
                "action": "none",
            }

            alive = bool(pid_check["alive"])
            heartbeat_stale = bool(heartbeat_check.get("stale", False))
            inconsistent = bool(heartbeat_check.get("state_inconsistent"))
            recent = self._recent_restarts.get(bundle_key)
            in_startup_grace = bool(
                recent
                and alive
                and pid_check.get("pid") == recent[0]
                and time.monotonic() - recent[1] <= self.lane_timeout
            )
            if in_startup_grace and heartbeat_stale:
                observed_state = "starting"
                recovery = {
                    "kind": "restart_heartbeat_pending",
                    "pid": pid_check.get("pid"),
                }
            elif recent and (
                not heartbeat_stale or pid_check.get("pid") != recent[0]
            ):
                self._recent_restarts.pop(bundle_key, None)

            terminal_process_conflict = (
                canonical_status
                and observed_state in {"stopped", "failed"}
                and alive
            )
            if terminal_process_conflict:
                inconsistent = True
                heartbeat_check["state_inconsistent"] = True
                heartbeat_check["state_reason"] = "terminal_state_pid_alive"
            suppressed_state = (
                canonical_status
                and observed_state in _INTENTIONAL_NON_RUNNING_STATES
                and not terminal_process_conflict
                and not (alive and heartbeat_stale)
            )
            needs_restart = (
                not in_startup_grace
                and not suppressed_state
                and (not alive or heartbeat_stale or inconsistent)
            )

            if needs_restart:
                unstall_result = self._watchdog_unstall(
                    manifest=manifest,
                    lane=lane,
                    lane_started=lane_started,
                    bundle_key=str(bundle_key),
                    state_dir=state_dir,
                    state_prefix=str(state_prefix),
                    pid_check=pid_check,
                    heartbeat_check=heartbeat_check,
                )
                if unstall_result is not None:
                    report["autonomous_unstall"] = unstall_result
                    if unstall_result.get("recovered"):
                        report["action"] = "autonomous_unstall_recovered"
                        report["reason"] = str(
                            unstall_result.get("reason") or ""
                        )
                        refreshed_pid = check_lane_pid(state_dir, state_prefix)
                        refreshed_heartbeat = check_lane_heartbeat(
                            state_dir,
                            state_prefix,
                            timeout_seconds=self.lane_timeout,
                        )
                        report["pid_check"] = refreshed_pid
                        report["heartbeat_check"] = refreshed_heartbeat
                        report["status"] = lifecycle_status_projection(
                            pid_check=refreshed_pid,
                            heartbeat_check=refreshed_heartbeat,
                            target_id=str(bundle_key),
                        )
                        reports.append(report)
                        restarts += int(
                            any(
                                item.get("operation") == "restart_lane"
                                and item.get("outcome") == "succeeded"
                                for item in unstall_result.get(
                                    "deterministic", {}
                                ).get("attempts", ())
                                if isinstance(item, Mapping)
                            )
                        )
                        continue
                    if unstall_result.get("quarantined"):
                        report["action"] = "autonomous_unstall_quarantined"
                        report["reason"] = str(
                            unstall_result.get("reason") or ""
                        )
                        report["status"] = lifecycle_status_projection(
                            pid_check=pid_check,
                            heartbeat_check=heartbeat_check,
                            state="blocked",
                            target_id=str(bundle_key),
                        )
                        reports.append(report)
                        continue
                if dynamic_authority:
                    # The persistent scheduler owns the fenced lease and is
                    # the only process allowed to replace its leased wrapper.
                    # Starting the raw lane command here would create an
                    # unfenced duplicate.
                    report["action"] = "scheduler_recovery_required"
                    report["reason"] = "dynamic_scheduler_owns_restart"
                    report["status"] = lifecycle_status_projection(
                        pid_check=pid_check,
                        heartbeat_check=heartbeat_check,
                        state="blocked",
                        target_id=str(bundle_key),
                    )
                    if recovery:
                        report["recovery"] = recovery
                    reports.append(report)
                    continue
                if canonical_status and (alive or untrusted_live_status_pid):
                    # Replacing a live canonical process requires the control
                    # plane's lease/fencing check.  The watchdog reports the
                    # requirement and never starts a second raw lane.
                    report["action"] = "fenced_stop_required"
                    report["reason"] = str(
                        heartbeat_check.get("state_reason")
                        or heartbeat_check.get("reason")
                        or "live_process_unhealthy"
                    )
                    observed_state = "blocked"
                # Check consecutive restart count for backoff
                count = self._consecutive_restart_counts.get(bundle_key, 0)
                if report["action"] == "fenced_stop_required":
                    pass
                elif count >= self.max_consecutive_restarts:
                    # Exponential backoff: skip this cycle
                    backoff_cycles = 2 ** min(count - self.max_consecutive_restarts, 5)
                    report["action"] = "backoff"
                    report["backoff_cycles"] = backoff_cycles
                    # Decrement so we'll try again eventually
                    self._consecutive_restart_counts[bundle_key] = count + 1
                else:
                    restart_info = dict(lane_started or lane)
                    restart_info.setdefault("pid_path", str(pid_check["pid_path"]))
                    if self.lifecycle_restart is None:
                        # Keep the call shape compatible with injected test and
                        # deployment shims.  The built-in helper fails closed.
                        restart_result = restart_lane(
                            restart_info, repo_root=self.repo_root
                        )
                    else:
                        restart_result = restart_lane(
                            restart_info,
                            repo_root=self.repo_root,
                            lifecycle_transition=self.lifecycle_restart,
                        )
                    report["action"] = "restarted" if restart_result.get("restarted") else "restart_failed"
                    report["restart_result"] = restart_result
                    if restart_result.get("restarted"):
                        restarts += 1
                        self._consecutive_restart_counts[bundle_key] = count + 1
                        new_pid = int(restart_result["new_pid"])
                        self._recent_restarts[bundle_key] = (
                            new_pid,
                            time.monotonic(),
                        )
                        self._generation += 1
                        if observed_state in _TRANSITIONAL_STATES:
                            recovery = {
                                "kind": "interrupted_transition",
                                "previous_state": observed_state,
                                "restarted_pid": new_pid,
                            }
                        observed_state = "starting"
                        pid_check = {
                            **pid_check,
                            "pid": new_pid,
                            "alive": True,
                        }
                        pid_check.pop("reason", None)
                    else:
                        self._consecutive_restart_counts[bundle_key] = count + 1
                        observed_state = "failed"
            else:
                # Healthy - reset consecutive restart counter
                self._consecutive_restart_counts[bundle_key] = 0
                if suppressed_state:
                    if (
                        not alive
                        and observed_state
                        in {"paused", "draining", "stopping"}
                    ):
                        interrupted_state = observed_state
                        if interrupted_state == "stopping":
                            observed_state = "stopped"
                            terminal_reason = (
                                "stop_completed_after_process_exit"
                            )
                        else:
                            observed_state = "failed"
                            terminal_reason = (
                                f"{interrupted_state}_process_exited"
                            )
                        heartbeat_check["terminal_reason"] = terminal_reason
                        report["action"] = "control_recovery_required"
                        report["reason"] = terminal_reason
                        recovery = {
                            "kind": "interrupted_transition",
                            "previous_state": interrupted_state,
                            "recovered_state": observed_state,
                        }
                    else:
                        report["action"] = "state_preserved"
                        report["reason"] = f"canonical_{observed_state}"

            report["status"] = lifecycle_status_projection(
                pid_check=pid_check,
                heartbeat_check=heartbeat_check,
                state=observed_state,
                target_id=str(bundle_key),
            )
            if recovery:
                report["recovery"] = recovery
            reports.append(report)

        # Aggregate logs periodically
        if started:
            try:
                aggregate_logs(
                    started,
                    repo_root=self.repo_root,
                    output_dir=self.log_aggregation_dir,
                )
            except Exception as exc:
                logger.debug("Log aggregation failed: %s", exc)

        embedded_snapshot = manifest.get("scheduler_snapshot")
        if isinstance(embedded_snapshot, dict):
            operator_snapshot = dict(embedded_snapshot)
        else:
            events = scheduler_state_events(
                [dict(lane) for lane in lanes if isinstance(lane, dict)],
                timestamp=utc_now(),
            )
            operator_snapshot = scheduler_snapshot(events).to_dict()

        timestamp = utc_now()
        lane_states = [
            str(item.get("status", {}).get("state") or "") for item in reports
        ]
        state_priority = (
            "failed",
            "blocked",
            "degraded",
            "starting",
            "stopping",
            "draining",
        )
        aggregate_state = next(
            (state for state in state_priority if state in lane_states),
            (
                "stopped"
                if not lane_states or all(state == "stopped" for state in lane_states)
                else (
                    "paused"
                    if all(state == "paused" for state in lane_states)
                    else "healthy"
                )
            ),
        )
        timestamp_seconds = _timestamp_seconds(timestamp)
        timestamp_ms = (
            None if timestamp_seconds is None else int(timestamp_seconds * 1000)
        )
        status_timestamp = _timestamp_text_from_ms(timestamp_ms or 0)
        aggregate_leases = sorted(
            {
                str(lease)
                for item in reports
                for lease in item.get("status", {}).get("active_leases", ())
                if str(lease)
            }
        )[:256]
        aggregate_status = {
            "schema": LIFECYCLE_STATUS_SCHEMA,
            "target_id": "supervisor:watchdog",
            "state": aggregate_state,
            "phase": "watchdog_check",
            "heartbeat_at_ms": timestamp_ms,
            "heartbeat_at": status_timestamp,
            "pid": os.getpid(),
            "active_leases": aggregate_leases,
            "active_lease_count": len(aggregate_leases),
            "refill_state": "idle",
            "backpressure": any(
                bool(item.get("status", {}).get("backpressure"))
                for item in reports
            ),
            "backpressure_reasons": sorted(
                {
                    str(reason)
                    for item in reports
                    for reason in item.get("status", {}).get(
                        "backpressure_reasons", ()
                    )
                }
            )[:256],
            "terminal_reason": "",
            "transition_id": "",
            "generation": self._generation,
            "fencing_epoch": 0,
            "updated_at_ms": timestamp_ms,
            "updated_at": status_timestamp,
        }
        return {
            "timestamp": timestamp,
            "lane_count": len(lanes),
            "restarts": restarts,
            "reports": reports,
            "status": aggregate_status,
            "scheduler_snapshot": operator_snapshot,
            "scheduler_snapshot_id": str(operator_snapshot.get("snapshot_id") or ""),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Outer watchdog that monitors and restarts bundle supervisor lanes"
    )
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--check-interval",
        type=float,
        default=float(os.environ.get("WATCHDOG_CHECK_INTERVAL_SECONDS", "120")),
    )
    parser.add_argument(
        "--lane-timeout",
        type=float,
        default=float(os.environ.get("WATCHDOG_LANE_TIMEOUT_SECONDS", "600")),
    )
    parser.add_argument(
        "--max-consecutive-restarts",
        type=int,
        default=int(os.environ.get("WATCHDOG_MAX_CONSECUTIVE_RESTARTS", "5")),
    )
    parser.add_argument("--log-aggregation-dir", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    watchdog = SupervisorWatchdog(
        manifest_path=args.manifest_path.resolve(),
        repo_root=args.repo_root.resolve(),
        check_interval=args.check_interval,
        lane_timeout=args.lane_timeout,
        max_consecutive_restarts=args.max_consecutive_restarts,
        log_aggregation_dir=args.log_aggregation_dir,
    )

    result = watchdog.run()
    logger.info("Watchdog exited: %s", json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
