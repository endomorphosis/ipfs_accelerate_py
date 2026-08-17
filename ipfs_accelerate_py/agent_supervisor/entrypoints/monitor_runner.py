"""Durable detached monitor runner (ASE3-008).

Only a :class:`ReviewedHostNamespaceReconciler` may start or adopt the monitor.
Client disconnect does not stop it. Injected monitor death has one restart
winner. Terminal shutdown stops only the exact owned generation.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_dag_json,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.run_monitor import (
    HEARTBEAT_INTERVAL_MS,
    JoinedRunningEvidence,
    MonitorAdoptionReceipt,
    ProcessEvidence,
    RecoveryAction,
    RecoveryPolicy,
    RecoveryReceipt,
    ReviewedHostNamespaceReconciler,
    RunHealthSnapshot,
    StallClass,
    StallClassifier,
    SupervisorDoctorService,
    join_running_evidence,
)


class DurableMonitorError(RuntimeError):
    """Monitor lifecycle invariant violation."""


@dataclass
class DurableMonitorState:
    run_id: str
    generation: int
    process_cid: str
    process_birth_identity: str
    lease_id: str
    fencing_generation: int
    event_cursor: str
    heartbeat_at_ms: int
    monitor_intent_cid: str
    guardian_cid: str
    terminal: bool = False
    recoveries: tuple[int, ...] = ()  # timestamps of recoveries

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "generation": self.generation,
            "process_cid": self.process_cid,
            "process_birth_identity": self.process_birth_identity,
            "lease_id": self.lease_id,
            "fencing_generation": self.fencing_generation,
            "event_cursor": self.event_cursor,
            "heartbeat_at_ms": self.heartbeat_at_ms,
            "monitor_intent_cid": self.monitor_intent_cid,
            "guardian_cid": self.guardian_cid,
            "terminal": self.terminal,
            "recoveries": list(self.recoveries),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DurableMonitorState":
        return cls(
            run_id=str(value.get("run_id") or ""),
            generation=int(value.get("generation") or 0),
            process_cid=str(value.get("process_cid") or ""),
            process_birth_identity=str(value.get("process_birth_identity") or ""),
            lease_id=str(value.get("lease_id") or ""),
            fencing_generation=int(value.get("fencing_generation") or 0),
            event_cursor=str(value.get("event_cursor") or ""),
            heartbeat_at_ms=int(value.get("heartbeat_at_ms") or 0),
            monitor_intent_cid=str(value.get("monitor_intent_cid") or ""),
            guardian_cid=str(value.get("guardian_cid") or ""),
            terminal=bool(value.get("terminal")),
            recoveries=tuple(int(x) for x in (value.get("recoveries") or ())),
        )


class DurableMonitorRunner:
    """File-backed durable monitor with guardian-only start/adopt."""

    def __init__(
        self,
        root: str | Path,
        *,
        guardian: ReviewedHostNamespaceReconciler,
        recovery_policy: RecoveryPolicy | None = None,
        doctor: SupervisorDoctorService | None = None,
        stall_classifier: StallClassifier | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise DurableMonitorError("monitor root must not be a symlink")
        self.guardian = guardian
        self.recovery_policy = recovery_policy or RecoveryPolicy()
        self.doctor = doctor or SupervisorDoctorService()
        self.stall_classifier = stall_classifier or StallClassifier()
        self._lock = threading.RLock()

    def _path(self, run_id: str) -> Path:
        digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.monitor.json"

    def _write(self, state: DurableMonitorState) -> None:
        path = self._path(state.run_id)
        tmp = path.with_suffix(".tmp")
        body = json.dumps(state.to_dict(), sort_keys=True).encode("utf-8")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(tmp, flags, 0o600)
        try:
            os.write(fd, body)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)

    def load(self, run_id: str) -> DurableMonitorState | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        if path.is_symlink():
            raise DurableMonitorError("monitor state is a symlink")
        return DurableMonitorState.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def persist_monitor_intent(
        self, *, run_id: str, intent_cid: str, now_ms: int | None = None
    ) -> str:
        """Persist monitor intent before any RUNNING claim is returned."""

        if not intent_cid:
            raise DurableMonitorError("monitor intent_cid is required")
        clock = int(now_ms if now_ms is not None else time.time() * 1000)
        receipt = {
            "run_id": run_id,
            "intent_cid": intent_cid,
            "persisted_at_ms": clock,
            "schema": "ipfs_accelerate_py/agent-supervisor/monitor-intent@1",
        }
        path = self.root / f"{hashlib.sha256(run_id.encode()).hexdigest()}.intent.json"
        path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return cid_for_dag_json(receipt)

    def start_or_adopt(
        self,
        *,
        run_id: str,
        requester: str,
        lifecycle: ProcessEvidence,
        now_ms: int | None = None,
    ) -> MonitorAdoptionReceipt:
        """Guardian-only start or adopt of the detached monitor process."""

        if not self.guardian.authorize_monitor_start(requester=requester):
            raise DurableMonitorError(
                "only ReviewedHostNamespaceReconciler may start/adopt monitor"
            )
        if lifecycle.role != "lifecycle":
            raise DurableMonitorError("lifecycle evidence required")
        clock = int(now_ms if now_ms is not None else time.time() * 1000)
        with self._lock:
            existing = self.load(run_id)
            if existing is not None and not existing.terminal:
                # Adopt exact generation winner.
                existing.heartbeat_at_ms = clock
                self._write(existing)
                return MonitorAdoptionReceipt(
                    run_id=run_id,
                    process_cid=existing.process_cid,
                    process_birth_identity=existing.process_birth_identity,
                    generation=existing.generation,
                    guardian_cid=self.guardian.content_id,
                    adopted=True,
                    created_at_ms=clock,
                )

            intent_cid = self.persist_monitor_intent(
                run_id=run_id,
                intent_cid=cid_for_dag_json(
                    {"run_id": run_id, "guardian": self.guardian.content_id}
                ),
                now_ms=clock,
            )
            generation = 1 if existing is None else existing.generation + 1
            process_birth = (
                f"monitor-birth:{run_id}:g{generation}:{clock}:{os.getpid()}"
            )
            process_cid = cid_for_dag_json(
                {
                    "run_id": run_id,
                    "generation": generation,
                    "birth": process_birth,
                    "role": "monitor",
                }
            )
            state = DurableMonitorState(
                run_id=run_id,
                generation=generation,
                process_cid=process_cid,
                process_birth_identity=process_birth,
                lease_id=f"monitor-lease:{run_id}:g{generation}",
                fencing_generation=lifecycle.fencing_generation,
                event_cursor=f"monitor-cursor:{generation}:0",
                heartbeat_at_ms=clock,
                monitor_intent_cid=intent_cid,
                guardian_cid=self.guardian.content_id,
            )
            self._write(state)
            return MonitorAdoptionReceipt(
                run_id=run_id,
                process_cid=process_cid,
                process_birth_identity=process_birth,
                generation=generation,
                guardian_cid=self.guardian.content_id,
                adopted=False,
                created_at_ms=clock,
            )

    def heartbeat(self, run_id: str, *, now_ms: int | None = None) -> DurableMonitorState:
        clock = int(now_ms if now_ms is not None else time.time() * 1000)
        with self._lock:
            state = self.load(run_id)
            if state is None or state.terminal:
                raise DurableMonitorError("monitor not running")
            state.heartbeat_at_ms = clock
            # Advance cursor monotonically on heartbeat (not log noise).
            seq = int(state.event_cursor.rsplit(":", 1)[-1] or "0") + 1
            state.event_cursor = f"monitor-cursor:{state.generation}:{seq}"
            self._write(state)
            return state

    def evaluate_running(
        self,
        *,
        run_id: str,
        run_revision: int,
        lifecycle: ProcessEvidence,
        semantic_progress_phase: str | None = None,
        semantic_progress_cursor: str | None = None,
        tree_reachable: bool = True,
        now_ms: int | None = None,
    ) -> JoinedRunningEvidence:
        clock = int(now_ms if now_ms is not None else time.time() * 1000)
        state = self.load(run_id)
        if state is None or state.terminal:
            # Fabricate empty monitor evidence so join fails closed.
            mon = ProcessEvidence(
                role="monitor",
                process_cid="missing",
                process_birth_identity="missing",
                lease_id="missing",
                fencing_generation=1,
                heartbeat_at_ms=0,
                event_cursor="missing",
                generation=1,
                healthy=False,
            )
        else:
            mon = ProcessEvidence(
                role="monitor",
                process_cid=state.process_cid,
                process_birth_identity=state.process_birth_identity,
                lease_id=state.lease_id,
                fencing_generation=state.fencing_generation,
                heartbeat_at_ms=state.heartbeat_at_ms,
                event_cursor=state.event_cursor,
                generation=state.generation,
                healthy=True,
            )
        from ipfs_accelerate_py.agent_supervisor.entrypoints.run_monitor import (
            SemanticProgressClock,
        )

        progress = None
        if semantic_progress_phase and semantic_progress_cursor:
            progress = SemanticProgressClock(
                phase=semantic_progress_phase,
                cursor_cid=semantic_progress_cursor,
                observed_at_ms=clock,
            )
        snapshot = RunHealthSnapshot(
            run_id=run_id,
            run_revision=run_revision,
            lifecycle=lifecycle,
            monitor=mon,
            semantic_progress=progress,
            tree_reachable=tree_reachable,
            observed_at_ms=clock,
        )
        return join_running_evidence(snapshot, now_ms=clock)

    def recover(
        self,
        *,
        run_id: str,
        stall: StallClass,
        authorized_callback: bool,
        lifecycle: ProcessEvidence,
        now_ms: int | None = None,
        requester: str = "",
    ) -> RecoveryReceipt:
        clock = int(now_ms if now_ms is not None else time.time() * 1000)
        with self._lock:
            state = self.load(run_id)
            generation = state.generation if state is not None else 0
            window_start = clock - self.recovery_policy.canary_window_ms
            recent = (
                tuple(t for t in (state.recoveries if state else ()) if t >= window_start)
            )
            action = self.recovery_policy.authorize(
                stall,
                recoveries_in_window=len(recent),
                authorized_callback=authorized_callback,
            )
            reasons = list(
                self.doctor.diagnose(
                    stall,
                    RunHealthSnapshot(
                        run_id=run_id,
                        run_revision=1,
                        lifecycle=lifecycle,
                        monitor=ProcessEvidence(
                            role="monitor",
                            process_cid=(state.process_cid if state else "none"),
                            process_birth_identity=(
                                state.process_birth_identity if state else "none"
                            ),
                            lease_id=(state.lease_id if state else "none"),
                            fencing_generation=1,
                            heartbeat_at_ms=state.heartbeat_at_ms if state else 0,
                            event_cursor=state.event_cursor if state else "",
                            generation=generation or 1,
                            healthy=False,
                        ),
                        semantic_progress=None,
                        tree_reachable=True,
                        observed_at_ms=clock,
                    ),
                )
            )
            if action in {RecoveryAction.RESTART, RecoveryAction.ADOPT, RecoveryAction.RESCUE}:
                if not self.guardian.authorize_monitor_start(
                    requester=requester or self.guardian.guardian_identity
                ):
                    action = RecoveryAction.OPERATOR
                    reasons.append("guardian_required")
                else:
                    # Dead/stale monitor: fence current generation so restart
                    # creates exactly one new generation winner.
                    if (
                        action is RecoveryAction.RESTART
                        and state is not None
                        and not state.terminal
                    ):
                        state.terminal = True
                        self._write(state)
                    receipt = self.start_or_adopt(
                        run_id=run_id,
                        requester=requester or self.guardian.guardian_identity,
                        lifecycle=lifecycle,
                        now_ms=clock,
                    )
                    state = self.load(run_id)
                    if state is not None:
                        state.recoveries = recent + (clock,)
                        self._write(state)
                    generation = receipt.generation
                    reasons.append(f"action:{action.value}")
            return RecoveryReceipt(
                run_id=run_id,
                stall=stall.value,
                action=action.value,
                authorized=authorized_callback and action is not RecoveryAction.OPERATOR,
                generation=generation,
                reason_codes=tuple(reasons),
                recovered_at_ms=clock,
            )

    def terminal_shutdown(
        self, run_id: str, *, generation: int, now_ms: int | None = None
    ) -> dict[str, Any]:
        """Stop only the exact owned monitor generation after run is terminal."""

        clock = int(now_ms if now_ms is not None else time.time() * 1000)
        with self._lock:
            state = self.load(run_id)
            if state is None:
                raise DurableMonitorError("monitor not found")
            if state.generation != generation:
                raise DurableMonitorError("terminal shutdown generation mismatch")
            state.terminal = True
            state.heartbeat_at_ms = clock
            self._write(state)
            return {
                "schema": "ipfs_accelerate_py/agent-supervisor/monitor-shutdown@1",
                "run_id": run_id,
                "generation": generation,
                "shutdown_at_ms": clock,
                "process_cid": state.process_cid,
            }

    def client_disconnect(self, run_id: str) -> DurableMonitorState:
        """Client disconnect must not stop the durable monitor."""

        state = self.load(run_id)
        if state is None or state.terminal:
            raise DurableMonitorError("monitor not running")
        return state


__all__ = [
    "DurableMonitorError",
    "DurableMonitorRunner",
    "DurableMonitorState",
]
