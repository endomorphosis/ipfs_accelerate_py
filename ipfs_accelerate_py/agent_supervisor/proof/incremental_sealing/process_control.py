"""Cancellation, timeout, and process-tree termination fencing (IPS-034).

Uses existing OS process-group hooks.  Timeout and cancellation escalate
terminate then kill, fence descendants, and quarantine late output.  An
interrupted prove never fabricates a terminal proved result.

Interfaces: ``ProofProcessController``, ``CancellationToken``,
``ProcessTerminationResult``.
"""

from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Final

EVIDENCE_SUBSET: Final[str] = "ips/process-fencing@1"


class ProcessControlError(ValueError):
    """Fail-closed process-control contract violation."""


class ControlOutcome(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class TerminationStage(str, Enum):
    NONE = "none"
    TERMINATE = "terminate"
    KILL = "kill"
    REAPED = "reaped"


@dataclass
class CancellationToken:
    """Cooperative cancellation generation.  Late results must match it."""

    generation: int = 0
    cancelled: bool = False
    reason: str = ""

    def cancel(self, reason: str = "cancelled") -> None:
        self.cancelled = True
        self.reason = reason
        self.generation += 1

    def check(self) -> None:
        if self.cancelled:
            raise ProcessControlError(f"cancelled:{self.reason}")


@dataclass(frozen=True, slots=True)
class ProcessTerminationResult:
    """Outcome of fencing a proof process tree."""

    outcome: ControlOutcome
    stage: TerminationStage
    live_descendants: int
    generation: int
    admitted_proof: bool
    late_output_quarantined: bool
    pid: int | None = None

    @property
    def satisfies_required_unit(self) -> bool:
        return (
            self.outcome is ControlOutcome.COMPLETED
            and self.admitted_proof
            and not self.late_output_quarantined
        )

    def to_canonical(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "stage": self.stage.value,
            "live_descendants": self.live_descendants,
            "generation": self.generation,
            "admitted_proof": self.admitted_proof,
            "late_output_quarantined": self.late_output_quarantined,
            "satisfies_required_unit": self.satisfies_required_unit,
            "pid": self.pid,
        }


class ProofProcessController:
    """Fence one admitted proof process group.

    ``spawn`` is injected so unit tests never launch a real prover.  The
    controller still uses POSIX group signals when a real pid is provided.
    """

    def __init__(
        self,
        token: CancellationToken | None = None,
        *,
        terminate: Callable[[int], None] | None = None,
        kill: Callable[[int], None] | None = None,
        poll_alive: Callable[[int], bool] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.token = token or CancellationToken()
        self._terminate = terminate or _default_terminate
        self._kill = kill or _default_kill
        self._poll_alive = poll_alive or _default_alive
        self._clock = clock or time.monotonic
        self._pid: int | None = None
        self._generation_at_start = self.token.generation
        self._quarantine: list[bytes] = []

    @property
    def pid(self) -> int | None:
        return self._pid

    def attach(self, pid: int) -> None:
        if type(pid) is not int or pid <= 0:
            raise ProcessControlError("pid must be a positive int")
        self._pid = pid
        self._generation_at_start = self.token.generation

    def observe_output(self, payload: bytes, *, generation: int | None = None) -> bool:
        """Return True if output is accepted; quarantine late/cancelled bytes."""

        observed = self.token.generation if generation is None else generation
        if self.token.cancelled or observed != self._generation_at_start:
            self._quarantine.append(payload)
            return False
        return True

    def fence(
        self,
        *,
        timeout: bool = False,
        grace_seconds: float = 0.0,
    ) -> ProcessTerminationResult:
        """Escalate terminate → kill.  Never reports a proved result."""

        if timeout:
            self.token.cancel("timeout")
            outcome = ControlOutcome.TIMEOUT
        elif self.token.cancelled:
            outcome = ControlOutcome.CANCELLED
        else:
            self.token.cancel("fenced")
            outcome = ControlOutcome.CANCELLED

        stage = TerminationStage.NONE
        live = 0
        if self._pid is not None:
            try:
                self._terminate(self._pid)
                stage = TerminationStage.TERMINATE
            except ProcessControlError:
                outcome = ControlOutcome.UNKNOWN
            if grace_seconds > 0:
                deadline = self._clock() + grace_seconds
                while self._clock() < deadline and self._poll_alive(self._pid):
                    pass
            if self._poll_alive(self._pid):
                try:
                    self._kill(self._pid)
                    stage = TerminationStage.KILL
                except ProcessControlError:
                    outcome = ControlOutcome.UNKNOWN
            live = 1 if self._poll_alive(self._pid) else 0
            if live == 0:
                stage = TerminationStage.REAPED

        return ProcessTerminationResult(
            outcome=outcome,
            stage=stage,
            live_descendants=live,
            generation=self.token.generation,
            admitted_proof=False,
            late_output_quarantined=bool(self._quarantine),
            pid=self._pid,
        )


def _default_terminate(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError as nested:
            raise ProcessControlError(f"terminate failed: {nested}") from exc


def _default_kill(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError as exc:
            raise ProcessControlError(f"kill failed: {exc}") from exc


def _default_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


__all__ = (
    "EVIDENCE_SUBSET",
    "CancellationToken",
    "ControlOutcome",
    "ProcessControlError",
    "ProcessTerminationResult",
    "ProofProcessController",
    "TerminationStage",
)
