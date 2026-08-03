"""Live local capacity producer for Grok/Codex reviewed supervisor lanes."""

from __future__ import annotations

import argparse
import errno
import json
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..todo_daemon.production_provider_cli import (
    DEFAULT_CONTEXT_BUDGET_TOKENS,
    production_cli_policy_readiness,
)
from .provider_capacity_snapshot import (
    DEFAULT_PROVIDER_CAPACITY_MAX_AGE_MS,
    DUAL_REVIEW_PROVIDER_ROLE_CAPABILITIES,
    PROVIDER_CAPACITY_BUDGET_SEMANTICS,
    provider_capacity_observation_floor,
    write_provider_capacity_snapshot,
)
from .resource_scheduler import ProviderCapacity

logger = logging.getLogger(__name__)

DEFAULT_MONITOR_INTERVAL_SECONDS: Final = 10.0
DEFAULT_PROVIDER_CONCURRENCY: Final = 2
DEFAULT_RESPONSE_TOKENS_PER_REQUEST: Final = 4_096
MAX_CLOCK_ADVANCE_WAIT_SECONDS: Final = 1.0
CLOCK_ADVANCE_POLL_SECONDS: Final = 0.001
_PROVIDER_NAMES: Final = ("grok_cli", "codex_cli")


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ProviderCapacityMonitorConfig:
    """Operator bounds used to produce local admission capacity.

    Request and token budgets are local scheduling ceilings, not account
    balances reported by either provider.
    """

    snapshot_path: Path
    max_age_ms: int = DEFAULT_PROVIDER_CAPACITY_MAX_AGE_MS
    interval_seconds: float = DEFAULT_MONITOR_INTERVAL_SECONDS
    grok_max_concurrency: int = DEFAULT_PROVIDER_CONCURRENCY
    codex_max_concurrency: int = DEFAULT_PROVIDER_CONCURRENCY
    grok_request_budget: int = DEFAULT_PROVIDER_CONCURRENCY
    codex_request_budget: int = DEFAULT_PROVIDER_CONCURRENCY
    grok_token_budget: int = (
        DEFAULT_PROVIDER_CONCURRENCY * DEFAULT_RESPONSE_TOKENS_PER_REQUEST
    )
    codex_token_budget: int = (
        DEFAULT_PROVIDER_CONCURRENCY * DEFAULT_RESPONSE_TOKENS_PER_REQUEST
    )
    context_budget_tokens: int = DEFAULT_CONTEXT_BUDGET_TOKENS

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_path", Path(self.snapshot_path))
        _positive_integer("max_age_ms", self.max_age_ms)
        if (
            isinstance(self.interval_seconds, bool)
            or not isinstance(self.interval_seconds, (int, float))
            or not 0 < float(self.interval_seconds) * 1_000 < self.max_age_ms
        ):
            raise ValueError(
                "interval_seconds must be positive and shorter than max_age_ms"
            )
        for name in ("grok_max_concurrency", "codex_max_concurrency"):
            _positive_integer(name, getattr(self, name))
        for name in (
            "grok_request_budget",
            "codex_request_budget",
            "grok_token_budget",
            "codex_token_budget",
        ):
            _non_negative_integer(name, getattr(self, name))
        _positive_integer("context_budget_tokens", self.context_budget_tokens)


def _command_provider(command: Sequence[str]) -> str:
    """Classify only non-interactive reviewed invocation argv patterns."""

    tokens = tuple(str(item) for item in command)
    if not tokens:
        return ""
    executable_index = 0
    executable = Path(tokens[0]).name.strip().lower()
    if executable == "node" and len(tokens) > 1:
        executable_index = 1
        executable = Path(tokens[1]).name.strip().lower()
    executable = executable.removesuffix(".exe").removesuffix(".js")
    arguments = tuple(item.lower() for item in tokens[executable_index + 1 :])
    if executable == "codex" and "exec" in arguments:
        return "codex_cli"
    if (
        executable == "grok"
        and "--prompt-file" in arguments
        and "--output-format" in arguments
    ):
        return "grok_cli"
    return ""


def _count_with_psutil() -> dict[str, int] | None:
    try:
        import psutil
    except ImportError:
        return None
    counts = {name: 0 for name in _PROVIDER_NAMES}
    effective_uid = os.geteuid() if hasattr(os, "geteuid") else None
    classified: list[tuple[int, int, str]] = []
    for process in psutil.process_iter(["cmdline", "uids", "ppid"]):
        try:
            info = process.info
            uids = info.get("uids")
            if (
                effective_uid is not None
                and uids is not None
                and int(uids.effective) != effective_uid
            ):
                continue
            command = info.get("cmdline") or ()
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise RuntimeError(
                "process inspection is inaccessible; capacity is unknown"
            ) from exc
        except OSError as exc:
            if exc.errno in {errno.ENOENT, errno.ESRCH}:
                continue
            raise RuntimeError(
                "process inspection failed; capacity is unknown"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "process identity is invalid; capacity is unknown"
            ) from exc
        provider = _command_provider(tuple(str(item) for item in command))
        if provider:
            classified.append(
                (int(process.pid), int(info.get("ppid") or 0), provider)
            )
    provider_by_pid = {pid: provider for pid, _ppid, provider in classified}
    for _pid, parent_pid, provider in classified:
        if provider_by_pid.get(parent_pid) != provider:
            counts[provider] += 1
    return counts


def _proc_identity(status_path: Path) -> tuple[int, int] | None:
    uid: int | None = None
    parent_pid = 0
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(
            "process identity is inaccessible; capacity is unknown"
        ) from exc
    try:
        for line in lines:
            if line.startswith("Uid:"):
                uid = int(line.split()[2])
            elif line.startswith("PPid:"):
                parent_pid = int(line.split()[1])
    except (UnicodeError, ValueError, IndexError) as exc:
        raise RuntimeError(
            "process identity is invalid; capacity is unknown"
        ) from exc
    if uid is None:
        raise RuntimeError("process owner is unknown; capacity is unknown")
    return uid, parent_pid


def _count_with_proc(
    *,
    maximum_processes: int = 8_192,
    proc_root: Path = Path("/proc"),
) -> dict[str, int]:
    counts = {name: 0 for name in _PROVIDER_NAMES}
    _positive_integer("maximum_processes", maximum_processes)
    effective_uid = os.geteuid() if hasattr(os, "geteuid") else None
    try:
        process_directories = sorted(
            (item for item in proc_root.iterdir() if item.name.isdigit()),
            key=lambda item: int(item.name),
        )
    except OSError as exc:
        raise RuntimeError(
            "process table is unavailable; capacity is unknown"
        ) from exc
    if len(process_directories) > maximum_processes:
        raise RuntimeError(
            "process table exceeds bounded fallback scan; capacity is unknown"
        )
    classified: list[tuple[int, int, str]] = []
    for process_dir in process_directories:
        identity = _proc_identity(process_dir / "status")
        if identity is None:
            continue
        uid, parent_pid = identity
        if effective_uid is not None and uid != effective_uid:
            continue
        try:
            raw = (process_dir / "cmdline").read_bytes()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RuntimeError(
                "process command is inaccessible; capacity is unknown"
            ) from exc
        command = tuple(
            token.decode("utf-8", errors="replace")
            for token in raw.split(b"\0")
            if token
        )
        provider = _command_provider(command)
        if provider:
            classified.append((int(process_dir.name), parent_pid, provider))
    provider_by_pid = {pid: provider for pid, _ppid, provider in classified}
    for _pid, parent_pid, provider in classified:
        if provider_by_pid.get(parent_pid) != provider:
            counts[provider] += 1
    return counts


def count_active_cli_processes() -> dict[str, int]:
    """Count current-user non-interactive invocation roots conservatively.

    Interactive ``codex``, ``codex resume``, and bare Grok control sessions do
    not consume this reviewed-route budget. Node launchers and their native
    children are de-duplicated into one invocation root.
    """

    measured = _count_with_psutil()
    return measured if measured is not None else _count_with_proc()


class ProviderCapacityMonitor:
    """Refresh an owner-private capacity snapshot before its hard TTL."""

    def __init__(
        self,
        config: ProviderCapacityMonitorConfig,
        *,
        readiness_source: Callable[[], Mapping[str, Any]] | None = None,
        process_counter: Callable[[], Mapping[str, Any]] | None = None,
        clock_ms: Callable[[], int] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config
        self._readiness_source = (
            readiness_source or production_cli_policy_readiness
        )
        self._process_counter = process_counter or count_active_cli_processes
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1_000))
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or time.sleep
        self._stop_event = threading.Event()
        self._observation_lock = threading.Lock()
        self._last_observed_at_ms = 0

    @staticmethod
    def _readiness_health(
        payload: Mapping[str, Any],
    ) -> dict[str, bool]:
        implementation = payload.get("implementation")
        review = payload.get("review")
        implementation = (
            implementation if isinstance(implementation, Mapping) else {}
        )
        review = review if isinstance(review, Mapping) else {}
        return {
            "grok_cli": bool(
                implementation.get("provider") == "grok_cli"
                and implementation.get("binary_available") is True
                and implementation.get("authenticated") is True
            ),
            "codex_cli": bool(
                review.get("provider") == "codex_cli"
                and review.get("binary_available") is True
                and review.get("authenticated") is True
                and review.get("independent") is True
            ),
        }

    @staticmethod
    def _readiness_diagnostics(payload: Mapping[str, Any]) -> dict[str, Any]:
        """Return a fixed, non-secret readiness projection for stdout."""

        result: dict[str, Any] = {
            "reported_ready": payload.get("ready") is True,
        }
        expected = {
            "implementation": "grok_cli",
            "review": "codex_cli",
        }
        for section_name, expected_provider in expected.items():
            raw = payload.get(section_name)
            section = raw if isinstance(raw, Mapping) else {}
            result[section_name] = {
                "provider": (
                    expected_provider
                    if section.get("provider") == expected_provider
                    else "unexpected-or-missing"
                ),
                "binary_available": section.get("binary_available") is True,
                "authenticated": section.get("authenticated") is True,
                "independent": section.get("independent") is True,
            }
        return result

    def _next_observation_time(self) -> tuple[int, bool]:
        """Wait boundedly for a real clock value newer than the CAS floor."""

        with self._observation_lock:
            floor = self._last_observed_at_ms
            if self.config.snapshot_path.exists():
                floor = max(
                    floor,
                    provider_capacity_observation_floor(
                        self.config.snapshot_path,
                        max_age_ms=self.config.max_age_ms,
                    ),
                )
            deadline = self._monotonic() + min(
                MAX_CLOCK_ADVANCE_WAIT_SECONDS,
                self.config.max_age_ms / 1_000,
            )
            waited = False
            while True:
                source_time = int(self._clock_ms())
                if source_time <= 0:
                    raise ValueError(
                        "monitor clock must return a positive millisecond value"
                    )
                if source_time > floor:
                    self._last_observed_at_ms = source_time
                    return source_time, waited
                if self._monotonic() >= deadline:
                    raise RuntimeError(
                        "monitor clock did not advance beyond the capacity CAS floor"
                    )
                waited = True
                self._sleep(CLOCK_ADVANCE_POLL_SECONDS)

    def sample(self) -> tuple[tuple[ProviderCapacity, ...], dict[str, Any]]:
        """Take one readiness/process sample and derive operator headroom."""

        readiness_error = ""
        try:
            readiness = self._readiness_source()
            if not isinstance(readiness, Mapping):
                raise ValueError("readiness source did not return a mapping")
            readiness = dict(readiness)
            health = self._readiness_health(readiness)
        except Exception:
            readiness_error = "readiness_probe_failed"
            readiness = {}
            health = {name: False for name in _PROVIDER_NAMES}

        process_error = ""
        try:
            raw_counts = self._process_counter()
            if not isinstance(raw_counts, Mapping):
                raise ValueError("process counter did not return a mapping")
            counts = {
                name: _non_negative_integer(
                    f"{name} active process count",
                    raw_counts.get(name, 0),
                )
                for name in _PROVIDER_NAMES
            }
        except Exception:
            process_error = "process_count_failed"
            counts = {
                "grok_cli": self.config.grok_max_concurrency,
                "codex_cli": self.config.codex_max_concurrency,
            }
            health = {name: False for name in _PROVIDER_NAMES}

        observed_at_ms, clock_waited = self._next_observation_time()
        ceilings = {
            "grok_cli": self.config.grok_max_concurrency,
            "codex_cli": self.config.codex_max_concurrency,
        }
        request_budgets = {
            "grok_cli": self.config.grok_request_budget,
            "codex_cli": self.config.codex_request_budget,
        }
        token_budgets = {
            "grok_cli": self.config.grok_token_budget,
            "codex_cli": self.config.codex_token_budget,
        }
        capacities = tuple(
            ProviderCapacity(
                provider_id=name,
                healthy=health[name] and counts[name] <= ceilings[name],
                quota_remaining=max(0, request_budgets[name] - counts[name]),
                latency_ms=0,
                context_window_tokens=self.config.context_budget_tokens,
                token_budget_remaining=max(
                    0,
                    token_budgets[name]
                    - counts[name] * DEFAULT_RESPONSE_TOKENS_PER_REQUEST,
                ),
                max_concurrency=ceilings[name],
                active_requests=counts[name],
                capabilities=tuple(
                    DUAL_REVIEW_PROVIDER_ROLE_CAPABILITIES[name]
                ),
                observed_at_ms=observed_at_ms,
            )
            for name in _PROVIDER_NAMES
        )
        auth_ready = all(health.values())
        admission_ready = all(
            item.healthy
            and item.available_concurrency > 0
            and item.quota_remaining > 0
            and item.token_budget_remaining
            >= DEFAULT_RESPONSE_TOKENS_PER_REQUEST
            for item in capacities
        )
        diagnostics = {
            "observed_at_ms": observed_at_ms,
            "clock_advance_waited": clock_waited,
            "ready": admission_ready,
            "auth_ready": auth_ready,
            "admission_ready": admission_ready,
            "health": dict(health),
            "active_processes": dict(counts),
            "process_count_scope": {
                "scope": "current-uid-noninteractive-cli-invocation-roots",
                "codex_pattern": "codex ... exec ...",
                "grok_pattern": (
                    "grok ... --prompt-file ... --output-format ..."
                ),
                "wrapper_child_deduplicated": True,
                "scheduler_reservation_interaction": (
                    "active CLI counts may conservatively overlap this "
                    "scheduler's long-lived lane reservations"
                ),
            },
            "operator_bounds": {
                "max_concurrency": ceilings,
                "request_admission_budget": request_budgets,
                "token_admission_budget": token_budgets,
                "context_budget_tokens": self.config.context_budget_tokens,
                "budget_semantics": PROVIDER_CAPACITY_BUDGET_SEMANTICS,
            },
            "readiness_error": readiness_error,
            "process_error": process_error,
            "readiness": self._readiness_diagnostics(readiness),
        }
        return capacities, diagnostics

    def publish_once(self) -> dict[str, Any]:
        capacities, diagnostics = self.sample()
        snapshot = write_provider_capacity_snapshot(
            self.config.snapshot_path,
            capacities,
            max_age_ms=self.config.max_age_ms,
            now_ms=diagnostics["observed_at_ms"],
        )
        return {
            **diagnostics,
            "snapshot_path": str(self.config.snapshot_path),
            "snapshot_id": snapshot["snapshot_id"],
            "expires_at_ms": snapshot["expires_at_ms"],
            "published": True,
        }

    def stop(self) -> None:
        self._stop_event.set()

    def run(
        self,
        *,
        max_cycles: int | None = None,
        stop_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Publish until stopped, or for a bounded number of test/one-shot cycles."""

        if max_cycles is not None and (
            isinstance(max_cycles, bool)
            or not isinstance(max_cycles, int)
            or max_cycles <= 0
        ):
            raise ValueError("max_cycles must be a positive integer")
        cycles = 0
        result: dict[str, Any] = {}
        while not self._stop_event.is_set() and not (
            stop_event is not None and stop_event.is_set()
        ):
            result = self.publish_once()
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            if stop_event is None:
                if self._stop_event.wait(self.config.interval_seconds):
                    break
            else:
                self._sleep(self.config.interval_seconds)
        return {**result, "cycles": cycles}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish fresh local Grok/Codex supervisor capacity"
    )
    parser.add_argument("--snapshot-path", type=Path, required=True)
    parser.add_argument(
        "--max-age-ms",
        type=int,
        default=DEFAULT_PROVIDER_CAPACITY_MAX_AGE_MS,
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=DEFAULT_MONITOR_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--grok-max-concurrency",
        type=int,
        default=DEFAULT_PROVIDER_CONCURRENCY,
    )
    parser.add_argument(
        "--codex-max-concurrency",
        type=int,
        default=DEFAULT_PROVIDER_CONCURRENCY,
    )
    parser.add_argument(
        "--grok-request-budget",
        type=int,
        default=DEFAULT_PROVIDER_CONCURRENCY,
        help="Local new-request admission budget, not provider account quota",
    )
    parser.add_argument(
        "--codex-request-budget",
        type=int,
        default=DEFAULT_PROVIDER_CONCURRENCY,
        help="Local new-request admission budget, not provider account quota",
    )
    parser.add_argument(
        "--grok-token-budget",
        type=int,
        default=(
            DEFAULT_PROVIDER_CONCURRENCY * DEFAULT_RESPONSE_TOKENS_PER_REQUEST
        ),
        help="Local new-response token admission budget",
    )
    parser.add_argument(
        "--codex-token-budget",
        type=int,
        default=(
            DEFAULT_PROVIDER_CONCURRENCY * DEFAULT_RESPONSE_TOKENS_PER_REQUEST
        ),
        help="Local new-response token admission budget",
    )
    parser.add_argument(
        "--context-budget-tokens",
        type=int,
        default=DEFAULT_CONTEXT_BUDGET_TOKENS,
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = ProviderCapacityMonitorConfig(
        snapshot_path=args.snapshot_path,
        max_age_ms=args.max_age_ms,
        interval_seconds=args.interval_seconds,
        grok_max_concurrency=args.grok_max_concurrency,
        codex_max_concurrency=args.codex_max_concurrency,
        grok_request_budget=args.grok_request_budget,
        codex_request_budget=args.codex_request_budget,
        grok_token_budget=args.grok_token_budget,
        codex_token_budget=args.codex_token_budget,
        context_budget_tokens=args.context_budget_tokens,
    )
    monitor = ProviderCapacityMonitor(config)

    previous_handlers: dict[int, Any] = {}

    def request_stop(_signum: int, _frame: object) -> None:
        monitor.stop()

    if not args.once and threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, request_stop)
    try:
        result = monitor.run(max_cycles=1 if args.once else None)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MONITOR_INTERVAL_SECONDS",
    "DEFAULT_PROVIDER_CONCURRENCY",
    "ProviderCapacityMonitor",
    "ProviderCapacityMonitorConfig",
    "build_arg_parser",
    "count_active_cli_processes",
    "main",
]
