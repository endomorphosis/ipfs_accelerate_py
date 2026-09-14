"""Select other tasks while one exact expired callback remains unsettled.

This is an exclusion, never a retry/settlement capability. Every pass repeats
the ordinary terminal reconciliations before constructing this transient
observation. More than one retained obligation remains a bounded refusal.
"""

import json
from typing import Any

from .expired_attempt_custody import expired_execution_deferral


def _encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class RetainedAttemptFairness:
    def __init__(self, daemon: Any) -> None:
        self.daemon = daemon
        self.attempt = None
        self.error = None
        self._snapshot = None

    def _read(self, attempt: Any) -> str | None:
        daemon = self.daemon
        current = daemon.get_attempt(attempt.attempt_id)
        claim = daemon.coordinator.get_task_claim(attempt.claim_id)
        task = daemon.task_source.get(attempt.task_cid)
        if current is None or claim is None or task is None:
            return None
        if (
            current.status != "running"
            or current.owner_session_id != daemon.owner_session_id
            or _encoded(current.to_dict()) != _encoded(attempt.to_dict())
            or str(getattr(claim.state, "value", claim.state)) != "expired"
            or type(claim.expires_at_ms) is not int
            or claim.expires_at_ms > daemon._now_ms()
            or str(task.status) != "in_progress"
            or not daemon._task_is_in_lane(task, task_cid=attempt.task_cid)
            or not daemon._task_has_exact_database_claim_receipt(task, claim)
            or _encoded(task.body.get("completion_receipt"))
            != _encoded(daemon._database_claim_receipt(claim))
            or daemon.coordinator.get_prepared_task_completion(attempt.task_cid)
            is not None
        ):
            return None
        identity = claim.to_dict()
        names = (
            "claim_id", "task_cid", "attempt_id", "attempt_number",
            "owner_session_id", "lease_id", "fencing_token", "fence_epoch",
        )
        if any(
            _encoded(identity.get(name)) != _encoded(getattr(attempt, name))
            for name in names
        ):
            return None
        snapshot = _encoded({
            "attempt": current.to_dict(), "claim": identity, "task": task.to_dict(),
        })
        return snapshot if len(snapshot.encode()) <= 1024 * 1024 else None

    def retain(self, error: BaseException, attempt: Any) -> bool:
        deferred = expired_execution_deferral(error)
        if (
            self.attempt is not None
            or deferred is None
            or deferred["retained_attempt_evidence"]["reason"]
            != "claim_authority_expired"
            or any(
                _encoded(deferred["retained_attempt_evidence"][name])
                != _encoded(getattr(attempt, name))
                for name in deferred["retained_attempt_evidence"] if name != "reason"
            )
        ):
            return False
        snapshot = self._read(attempt)
        if snapshot is None:
            return False
        self.attempt, self.error, self._snapshot = attempt, error, snapshot
        return True

    def require_current(self) -> None:
        if self.attempt is not None and self._read(self.attempt) != self._snapshot:
            raise self.error

    @property
    def excluded_task_cids(self) -> tuple[str, ...]:
        return () if self.attempt is None else (self.attempt.task_cid,)

    def observation(self) -> list[dict[str, Any]]:
        return [] if self.error is None else [expired_execution_deferral(self.error)]

    def claim_boundary(self, control: Any) -> dict[str, Any]:
        self.require_current()
        if self.attempt is None:
            return control.before_claim()
        observe = getattr(control, "before_independent_claim", None)
        result = observe(self.daemon, self.attempt) if callable(observe) else None
        self.require_current()
        if result is not None and result.get("new_dispatch_permitted") is True:
            return result
        if (
            result is not None
            and result.get("reason") != "retained_attempt_dispatch_observation_unavailable"
        ):
            return result
        ordinary = control.before_claim()
        self.require_current()
        if ordinary.get("new_dispatch_permitted") is True:
            return ordinary
        if ordinary.get("reason") == "native_dispatch_observation_unavailable":
            # Missing owner/pause observation is not a pause and must not
            # starve a different ready task. The retained CID stays excluded.
            return {
                "new_dispatch_permitted": True,
                "reason": "retained_attempt_observation_unverified_independent_claim",
            }
        return ordinary

    def report_custody(self) -> None:
        if self.attempt is None:
            return
        control = getattr(self.daemon, "_native_dispatch_control", None)
        observe = getattr(control, "retained_attempt_custody", None)
        if callable(observe):
            try:
                observe(self.daemon, self.attempt)
            except Exception:
                pass  # Diagnostics cannot retire or cancel retained execution.
