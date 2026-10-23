"""Track command descendants that escape the command's original session.

Process groups alone miss Playwright web servers started with setsid(). Linux
pidfds let cleanup signal an observed descendant without risking a reused PID.
Other platforms retain the caller's process-group cleanup. Descendants must be
observed before they orphan; this is not a replacement for cgroup containment.
"""

from __future__ import annotations

import os
import signal
from pathlib import Path


def _identity(pid: int) -> tuple[str, str] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        # comm can contain spaces and closing parentheses.
        fields = raw[raw.rindex(")") + 2 :].split()
        return fields[0], fields[19]  # process state and starttime (field 22)
    except (OSError, ValueError, IndexError):
        return None


def _children(pid: int) -> set[int]:
    children: set[int] = set()
    try:
        tasks = list(Path(f"/proc/{pid}/task").iterdir())
    except OSError:
        return children
    for task in tasks:
        try:
            children.update(int(raw) for raw in (task / "children").read_text().split())
        except (OSError, ValueError):
            continue
    return children


class CommandProcessTree:
    """Keep identity-fenced handles for descendants of one owned command."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self._root_identity = _identity(pid)
        self._owned: dict[int, tuple[str, int]] = {}
        self._enabled = hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")

    def _remember(self, pid: int) -> bool:
        before = _identity(pid)
        if before is None or before[0] in {"Z", "X"}:
            return False
        prior = self._owned.get(pid)
        if prior is not None:
            return before[1] == prior[0]
        if len(self._owned) >= 4096:
            return False
        try:
            fd = os.pidfd_open(pid, 0)
        except OSError:
            return False
        after = _identity(pid)
        if after is None or after[1] != before[1]:
            os.close(fd)
            return False
        self._owned[pid] = (before[1], fd)
        return True

    def snapshot(self) -> None:
        """Observe descendants while their parentage is still available."""
        if not self._enabled:
            return
        # Previously observed detached children can themselves spawn children
        # after their original parent has exited.
        pending = [
            pid for pid, (start, _fd) in self._owned.items()
            if (current := _identity(pid)) is not None and current[1] == start
        ]
        current = _identity(self.pid)
        if (
            current is not None
            and self._root_identity is not None
            and current[1] == self._root_identity[1]
        ):
            pending.append(self.pid)
        visited: set[int] = set()
        while pending and len(visited) < 4096:
            parent = pending.pop()
            if parent in visited:
                continue
            visited.add(parent)
            for child in _children(parent):
                if child != self.pid and self._remember(child):
                    pending.append(child)

    def send_signal(self, sig: int) -> None:
        """Signal only processes opened while they were owned descendants."""
        for _pid, (_start, fd) in tuple(self._owned.items()):
            try:
                signal.pidfd_send_signal(fd, sig)
            except (ProcessLookupError, PermissionError):
                continue

    def close(self) -> None:
        for _start, fd in self._owned.values():
            os.close(fd)
        self._owned.clear()
