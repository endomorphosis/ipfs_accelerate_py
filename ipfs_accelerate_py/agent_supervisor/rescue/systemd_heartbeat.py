"""Main-process systemd liveness within the dispatcher's existing deadlines.

No thread or daemon is created. Callers notify between completed loop stages,
or poll a bounded child wait through ``wait_for_process``. A notification never
extends that child's absolute deadline or grants repair/task authority.
"""
from __future__ import annotations

import math
import os
import socket
import subprocess
import time
from typing import Any, Callable, Mapping


class SystemdHeartbeat:
    """A no-op unless this process owns a valid systemd watchdog environment."""

    def __init__(self, *, address: str | None, interval_seconds: float = 0,
                 owner_pid: int, disabled_reason: str = "", clock: Callable[[], float] = time.monotonic,
                 socket_factory: Callable[..., Any] = socket.socket):
        self.address = address
        self.interval_seconds = interval_seconds
        self.owner_pid = owner_pid
        self.disabled_reason = disabled_reason
        self.enabled = address is not None
        self._clock = clock
        self._socket_factory = socket_factory
        self._next_due = 0.0

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None, *,
                         pid: int | None = None, clock: Callable[[], float] = time.monotonic,
                         socket_factory: Callable[..., Any] = socket.socket) -> "SystemdHeartbeat":
        env = os.environ if environment is None else environment
        current_pid = os.getpid() if pid is None else pid

        def disabled(reason: str) -> "SystemdHeartbeat":
            return cls(address=None, owner_pid=current_pid, disabled_reason=reason,
                       clock=clock, socket_factory=socket_factory)

        raw_interval, raw_pid, address = (env.get("WATCHDOG_USEC"),
                                         env.get("WATCHDOG_PID"), env.get("NOTIFY_SOCKET"))
        if raw_interval is None or address is None:
            return disabled("watchdog_environment_absent")
        if (type(current_pid) is not int or current_pid < 1
                or not isinstance(raw_interval, str) or not raw_interval.isascii()
                or not raw_interval.isdecimal()):
            return disabled("watchdog_interval_invalid")
        # Bound conversion before int/float construction even for hostile env.
        if len(raw_interval) > 20:
            return disabled("watchdog_interval_invalid")
        usec = int(raw_interval)
        if usec < 1:
            return disabled("watchdog_interval_invalid")
        if raw_pid is not None:
            if (not isinstance(raw_pid, str) or not raw_pid.isascii()
                    or not raw_pid.isdecimal() or len(raw_pid) > 20
                    or int(raw_pid) != current_pid):
                return disabled("watchdog_pid_mismatch")
        if (not isinstance(address, str) or len(address) < 2
                or address[0] not in {"/", "@"}
                or any(value in address for value in ("\0", "\n", "\r"))):
            return disabled("notify_socket_invalid")
        target = "\0" + address[1:] if address.startswith("@") else address
        return cls(address=target, owner_pid=current_pid, interval_seconds=usec / 2_000_000,
                   clock=clock, socket_factory=socket_factory)

    def notify_due(self, *, force: bool = False) -> bool:
        """Send a bounded AF_UNIX datagram, only from the owning main PID."""
        if not self.enabled or os.getpid() != self.owner_pid:
            return False
        now = self._clock()
        if not force and now < self._next_due:
            return False
        client = None
        sent = False
        try:
            client = self._socket_factory(socket.AF_UNIX, socket.SOCK_DGRAM)
            client.settimeout(0.2)
            sent = client.sendto(b"WATCHDOG=1", self.address) == len(b"WATCHDOG=1")
        except OSError:
            # The service manager, rather than a fake successful ping, decides
            # recovery if its notification channel remains unavailable.
            sent = False
        finally:
            if client is not None:
                try:
                    client.close()
                except OSError:
                    pass
        self._next_due = now + (self.interval_seconds if sent
                                else min(1.0, self.interval_seconds))
        return sent

    def wait_for_process(self, process: Any, timeout: float, *, deadline: float | None = None,
                         poll_seconds: float | None = None) -> int:
        """Wait within one absolute budget, pinging only bounded wait progress.

        An externally supplied deadline can shorten the budget, never extend
        it. Disabled watchdogs retain the original one-call ``Popen.wait``
        behavior. No timeout, successful ping or child retry renews the budget.
        """
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("process timeout must be finite and nonnegative")
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("process timeout must be finite and nonnegative")
        started = self._clock()
        expires = started + timeout
        if not math.isfinite(started) or not math.isfinite(expires):
            raise ValueError("process clock and deadline must be finite")
        if deadline is not None:
            if (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
                    or not math.isfinite(deadline)):
                raise ValueError("process deadline must be finite")
            expires = min(expires, deadline)
        if not self.enabled:
            return process.wait(timeout=timeout if deadline is None
                                else max(0.0, expires - self._clock()))
        poll = min(30.0, max(0.001, self.interval_seconds / 3)) if poll_seconds is None else poll_seconds
        if (isinstance(poll, bool) or not isinstance(poll, (int, float))
                or not math.isfinite(poll) or poll <= 0):
            raise ValueError("process poll interval must be finite and positive")
        poll = min(float(poll), 30.0)
        while True:
            remaining = expires - self._clock()
            if remaining <= 0:
                return process.wait(timeout=0)
            self.notify_due()
            remaining = expires - self._clock()
            if remaining <= 0:
                return process.wait(timeout=0)
            try:
                return process.wait(timeout=min(poll, remaining))
            except subprocess.TimeoutExpired:
                if self._clock() >= expires:
                    raise
