"""Graceful degradation for repeatedly failing submodules.

When a submodule causes repeated failures (merge conflicts, checkout errors,
dirty state), this module allows the supervisor to temporarily skip tasks that
depend on it rather than blocking all progress.

The degradation state is stored as a JSON file alongside the event log and
resets automatically after a configurable cooldown period.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Default: skip a submodule after 5 consecutive failures
_MAX_FAILURES_ENV = "IPFS_ACCELERATE_AGENT_SUBMODULE_MAX_FAILURES"
_DEFAULT_MAX_FAILURES = 5

# Default: re-enable a degraded submodule after 2 hours
_COOLDOWN_SECONDS_ENV = "IPFS_ACCELERATE_AGENT_SUBMODULE_COOLDOWN_SECONDS"
_DEFAULT_COOLDOWN_SECONDS = 7200


@dataclass
class SubmoduleHealth:
    """Health tracking for a single submodule."""

    path: str
    consecutive_failures: int = 0
    last_failure_time: float = 0.0
    last_failure_reason: str = ""
    degraded_since: float = 0.0
    total_failures: int = 0
    total_recoveries: int = 0

    def record_failure(self, reason: str = "") -> None:
        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_failure_time = time.time()
        self.last_failure_reason = reason

    def record_success(self) -> None:
        if self.consecutive_failures > 0:
            self.total_recoveries += 1
        self.consecutive_failures = 0
        self.degraded_since = 0.0

    def is_degraded(self, *, max_failures: int = 0, cooldown_seconds: float = 0) -> bool:
        if max_failures <= 0:
            max_failures = int(os.environ.get(_MAX_FAILURES_ENV, str(_DEFAULT_MAX_FAILURES)))
        if cooldown_seconds <= 0:
            cooldown_seconds = float(os.environ.get(_COOLDOWN_SECONDS_ENV, str(_DEFAULT_COOLDOWN_SECONDS)))

        if self.consecutive_failures < max_failures:
            return False

        # Check cooldown - if enough time has passed, give it another chance
        if self.degraded_since > 0:
            elapsed = time.time() - self.degraded_since
            if elapsed >= cooldown_seconds:
                return False

        # Mark degraded start time
        if self.degraded_since == 0.0:
            self.degraded_since = time.time()

        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "consecutive_failures": self.consecutive_failures,
            "last_failure_time": self.last_failure_time,
            "last_failure_reason": self.last_failure_reason,
            "degraded_since": self.degraded_since,
            "total_failures": self.total_failures,
            "total_recoveries": self.total_recoveries,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubmoduleHealth":
        return cls(
            path=str(data.get("path", "")),
            consecutive_failures=int(data.get("consecutive_failures", 0)),
            last_failure_time=float(data.get("last_failure_time", 0.0)),
            last_failure_reason=str(data.get("last_failure_reason", "")),
            degraded_since=float(data.get("degraded_since", 0.0)),
            total_failures=int(data.get("total_failures", 0)),
            total_recoveries=int(data.get("total_recoveries", 0)),
        )


@dataclass
class DegradationState:
    """Tracks health of all submodules for graceful degradation."""

    submodules: dict[str, SubmoduleHealth] = field(default_factory=dict)
    _path: Path | None = None

    def get_or_create(self, submodule_path: str) -> SubmoduleHealth:
        if submodule_path not in self.submodules:
            self.submodules[submodule_path] = SubmoduleHealth(path=submodule_path)
        return self.submodules[submodule_path]

    def record_failure(self, submodule_path: str, reason: str = "") -> None:
        health = self.get_or_create(submodule_path)
        health.record_failure(reason)
        self.save()

    def record_success(self, submodule_path: str) -> None:
        health = self.get_or_create(submodule_path)
        health.record_success()
        self.save()

    def is_degraded(self, submodule_path: str) -> bool:
        if submodule_path not in self.submodules:
            return False
        return self.submodules[submodule_path].is_degraded()

    def degraded_submodules(self) -> list[str]:
        """Return list of currently degraded submodule paths."""
        return [path for path, health in self.submodules.items() if health.is_degraded()]

    def should_skip_task(self, task_outputs: list[str], task_inputs: list[str] | None = None) -> str | None:
        """Check if a task should be skipped due to degraded submodules.

        Returns the degraded submodule path if the task should be skipped,
        or None if it can proceed.
        """
        all_paths = list(task_outputs)
        if task_inputs:
            all_paths.extend(task_inputs)

        for file_path in all_paths:
            for submodule_path in self.degraded_submodules():
                if file_path.startswith(submodule_path + "/") or file_path == submodule_path:
                    return submodule_path
        return None

    def summary(self) -> dict[str, Any]:
        """Return a summary of degradation state."""
        degraded = self.degraded_submodules()
        return {
            "total_tracked": len(self.submodules),
            "degraded_count": len(degraded),
            "degraded_paths": degraded,
            "submodules": {path: health.to_dict() for path, health in self.submodules.items()},
        }

    def save(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "schema": "submodule_degradation_state",
                "updated_at": time.time(),
                "submodules": {path: health.to_dict() for path, health in self.submodules.items()},
            }
            self._path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass

    @classmethod
    def load(cls, path: Path) -> "DegradationState":
        state = cls(_path=path)
        if not path.exists():
            return state
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for submodule_path, health_data in data.get("submodules", {}).items():
                state.submodules[submodule_path] = SubmoduleHealth.from_dict(health_data)
        except (json.JSONDecodeError, OSError, TypeError):
            pass
        return state
