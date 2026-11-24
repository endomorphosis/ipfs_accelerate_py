"""Keep a dependency-release supervisor from sitting idle on a blocked board.

One tick reads the catalogue. When no task is ready or in progress and some
task is still incomplete, the tick runs ``release_dependency_satisfied_blocks``.
That release creates resolution subgoals and completes a goal only from a
recorded result. When that release changes nothing, the tick starts one repair
subagent for that stall. The repair subagent runs the same release again. It
does not set a task ready, grant admission, remove an operator hold, or start
the scientific executor.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

TERMINAL_STATUSES = frozenset(
    {
        "completed",
        "complete",
        "done",
        "skipped",
        "cancelled",
        "canceled",
        "rejected",
        "quarantined",
    }
)
ACTIVE_STATUSES = frozenset({"ready", "in_progress", "retrying"})
QUIET_ACTIONS = frozenset(
    {"continue", "idle_complete", "stall_recorded", "database_busy"}
)


class BoardStallWatchdogError(RuntimeError):
    """The watchdog refused a tick that would break its own fence."""


def file_lock_clear(exit_code: int, stdout: str, stderr: str) -> bool:
    """True only when fuser reports no holder of the database file."""

    return exit_code == 1 and stdout == "" and stderr == ""


def _alias(record: Mapping[str, Any]) -> str:
    return str(record.get("task_alias") or record.get("task_cid") or "")


def _status(record: Mapping[str, Any]) -> str:
    return str(record.get("status") or "").strip().lower()


def _names(items: Sequence[Mapping[str, Any]] | None) -> list[str]:
    names: list[str] = []
    for item in items or ():
        if isinstance(item, Mapping):
            name = str(item.get("task_alias") or item.get("task_cid") or "")
            if name:
                names.append(name)
    return names


def observe_tasks(
    records: Sequence[Mapping[str, Any]],
    *,
    operator_hold_present: bool,
) -> dict[str, Any]:
    """Classify task rows. Active work is left alone. Incomplete work is not."""

    counts: dict[str, int] = {}
    blocked: list[str] = []
    active: list[str] = []
    incomplete: list[str] = []
    for record in records:
        status = _status(record)
        counts[status] = counts.get(status, 0) + 1
        alias = _alias(record)
        if status in ACTIVE_STATUSES:
            active.append(alias)
        if status not in TERMINAL_STATUSES:
            incomplete.append(alias)
            if status == "blocked":
                blocked.append(alias)
    blocked_aliases = tuple(sorted(alias for alias in blocked if alias))
    return {
        "counts": counts,
        "blocked_aliases": blocked_aliases,
        "active_aliases": tuple(sorted(alias for alias in active if alias)),
        "incomplete_count": len(incomplete),
        "active_count": len(active),
        "operator_hold_present": operator_hold_present,
        "signature": (
            ("hold" if operator_hold_present else "open")
            + "|"
            + ",".join(blocked_aliases)
        ),
    }


def _fence(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "preserve_operator_hold": bool(observation.get("operator_hold_present")),
        "executor_start_allowed": False,
        "set_ready": False,
        "grant_admission": False,
        "counts": dict(observation.get("counts") or {}),
        "blocked_aliases": list(observation.get("blocked_aliases") or ()),
    }


def _release_summary(release_result: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(release_result, Mapping):
        return {"completed": [], "spawned": []}
    return {
        "completed": _names(release_result.get("completed")),
        "spawned": _names(release_result.get("spawned")),
        "retired": _names(release_result.get("retired")),
    }


def decide_board_stall(
    observation: Mapping[str, Any],
    *,
    release_result: Mapping[str, Any] | None = None,
    repair_already_requested: bool = False,
) -> dict[str, Any]:
    """Choose the next stall action. Ready, admission, and hold removal are off."""

    decision = _fence(observation)
    if int(observation.get("active_count") or 0) > 0:
        decision["action"] = "continue"
        return decision
    if int(observation.get("incomplete_count") or 0) == 0:
        decision["action"] = "idle_complete"
        return decision
    if release_result is None:
        decision["action"] = "release"
        return decision
    summary = _release_summary(release_result)
    decision["release"] = summary
    if summary["completed"] or summary["spawned"] or summary["retired"]:
        decision["action"] = "progressed"
        return decision
    if repair_already_requested:
        decision["action"] = "stall_recorded"
        return decision
    decision["action"] = "start_repair_subagent"
    decision["repair_spec"] = {
        "kind": "board_stall_resolution_repair",
        "blocked_aliases": list(observation.get("blocked_aliases") or ()),
        "allowed": ["release_dependency_satisfied_blocks"],
        "forbidden": [
            "set_ready",
            "remove_operator_hold",
            "start_scientific_executor",
            "grant_admission",
            "invent_evidence",
        ],
        "operator_hold_present": bool(observation.get("operator_hold_present")),
        "executor_start_allowed": False,
        "set_ready": False,
        "grant_admission": False,
    }
    return decision


def resolution_repair_subagent(
    repo: Any,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the one release the repair spec allows.

    This is the watchdog's repair subagent. It creates subgoals only through
    the dependency release. A spec that asks for any other mutation is refused.
    """

    allowed = {str(item) for item in spec.get("allowed") or ()}
    if allowed != {"release_dependency_satisfied_blocks"}:
        return {"ran": False, "reason": "release_not_allowed", "set_ready": False}
    if spec.get("set_ready") is True or spec.get("grant_admission") is True:
        return {"ran": False, "reason": "forbidden_request", "set_ready": False}
    if spec.get("executor_start_allowed") is True:
        return {"ran": False, "reason": "executor_refused", "set_ready": False}
    result = repo.release_dependency_satisfied_blocks()
    summary = _release_summary(result if isinstance(result, Mapping) else {})
    return {
        "ran": True,
        "completed": summary["completed"],
        "spawned": summary["spawned"],
        "set_ready": False,
        "hold_removed": False,
        "admission_granted": False,
        "executor_started": False,
    }


def list_task_records(repo: Any) -> tuple[Mapping[str, Any], ...]:
    """Read every task page. The page size stays inside the repository limit."""

    records: list[Mapping[str, Any]] = []
    offset = 0
    page_size = 200
    while True:
        page = tuple(repo.list_tasks(limit=page_size, offset=offset))
        records.extend(page)
        if len(page) < page_size:
            return tuple(records)
        offset += page_size


def lock_conflict(exc: BaseException) -> bool:
    """True when DuckDB refused the open because another process holds the file."""

    text = f"{type(exc).__name__} {exc}".lower()
    return (
        "conflicting lock" in text
        or "could not set lock" in text
        or "database is locked" in text
    )


def database_busy_report(
    *,
    operator_hold_present: bool,
    error: str,
) -> dict[str, Any]:
    return {
        "action": "database_busy",
        "error": error,
        "executor_start_allowed": False,
        "set_ready": False,
        "grant_admission": False,
        "preserve_operator_hold": operator_hold_present,
    }


def observe_catalogue_readonly(
    database: Path,
    *,
    operator_hold_present: bool,
) -> dict[str, Any]:
    """Read task status without taking the catalogue write lock."""

    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        rows = connection.execute(
            "SELECT task_alias, status FROM tasks"
        ).fetchall()
    finally:
        connection.close()
    records = [
        {"task_alias": row[0], "status": row[1]}
        for row in rows
    ]
    return observe_tasks(records, operator_hold_present=operator_hold_present)


class BoardStallWatchdog:
    """One in-process supervisor loop for a blocked, stalled, or idle board."""

    def __init__(self) -> None:
        self._idle_signatures: set[str] = set()
        self._idle_checked_at: dict[str, float] = {}

    def idle_recorded(self, signature: str) -> bool:
        # Recheck dependency evidence after one normal 120-second interval.
        # Keep the separate repair signature so an unchanged stall does not
        # launch another repair callback merely because its check expired.
        return (signature in self._idle_signatures
                and time.monotonic() - self._idle_checked_at.get(signature, float("-inf")) < 120.0)

    def tick(
        self,
        repo: Any,
        *,
        operator_hold_present: bool,
        spawn_repair: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
    ) -> dict[str, Any]:
        observation = observe_tasks(
            list_task_records(repo),
            operator_hold_present=operator_hold_present,
        )
        active_work = int(observation["active_count"]) > 0
        if int(observation["incomplete_count"]) == 0 or (
            active_work and not observation["blocked_aliases"]
        ):
            return decide_board_stall(observation)
        signature = str(observation["signature"])
        repair_already_requested = signature in self._idle_signatures
        if not active_work and self.idle_recorded(signature):
            decision = decide_board_stall(
                observation,
                release_result={"completed": [], "spawned": []},
                repair_already_requested=True,
            )
            decision["signature"] = signature
            return decision
        try:
            release_result = repo.release_dependency_satisfied_blocks()
        except Exception as exc:
            if lock_conflict(exc):
                raise
            if active_work:
                decision = decide_board_stall(observation)
                decision["release_failure"] = {
                    "action": "release_failed", "error": type(exc).__name__
                }
                decision["signature"] = signature
                return decision
            decision = _fence(observation)
            decision["action"] = "release_failed"
            decision["error"] = type(exc).__name__
            decision["signature"] = signature
            return decision
        if not isinstance(release_result, Mapping):
            release_result = {}
        if active_work:
            # The repository resolves only eligible blocked metadata tasks. Its
            # claim, lease, evidence and revision fences retain authority.
            # Active work never starts a repair or consumes/resets its latch.
            decision = decide_board_stall(observation)
            decision["release"] = _release_summary(release_result)
            decision["signature"] = signature
            return decision
        decision = decide_board_stall(
            observation,
            release_result=release_result,
            repair_already_requested=repair_already_requested,
        )
        decision["signature"] = signature
        if decision["action"] != "start_repair_subagent":
            if decision["action"] == "stall_recorded":
                self._idle_checked_at[signature] = time.monotonic()
            elif decision["action"] == "progressed":
                self._idle_signatures.discard(signature)
                self._idle_checked_at.pop(signature, None)
            return decision
        self._idle_signatures.add(signature)
        self._idle_checked_at[signature] = time.monotonic()
        repair = spawn_repair or resolution_repair_subagent
        report = repair(repo, decision["repair_spec"])
        decision["repair_report"] = dict(report)
        if report.get("set_ready") is True or report.get("hold_removed") is True:
            raise BoardStallWatchdogError(
                "repair subagent reported a forbidden board mutation"
            )
        if report.get("spawned") or report.get("completed"):
            self._idle_signatures.discard(signature)
            self._idle_checked_at.pop(signature, None)
        return decision


def hold_bytes(path: Path | None) -> bytes | None:
    if path is None or not path.is_file():
        return None
    return path.read_bytes()


def assert_hold_unchanged(path: Path | None, before: bytes | None) -> None:
    if before is None:
        return
    if path is None or not path.is_file() or path.read_bytes() != before:
        raise BoardStallWatchdogError("operator hold was removed or changed")


def write_status(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["pid"] = os.getpid()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_once(
    repo: Any,
    *,
    watchdog: BoardStallWatchdog,
    hold_path: Path | None = None,
    status_path: Path | None = None,
    spawn_repair: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]
    | None = None,
) -> dict[str, Any]:
    before = hold_bytes(hold_path)
    report = watchdog.tick(
        repo,
        operator_hold_present=before is not None,
        spawn_repair=spawn_repair,
    )
    assert_hold_unchanged(hold_path, before)
    if status_path is not None:
        write_status(status_path, report)
    return report


def _stall_recorded(
    observation: Mapping[str, Any],
    *,
    hold_path: Path | None,
    status_path: Path | None,
) -> dict[str, Any]:
    before = hold_bytes(hold_path)
    report = decide_board_stall(
        observation,
        release_result={"completed": [], "spawned": []},
        repair_already_requested=True,
    )
    report["signature"] = str(observation.get("signature") or "")
    assert_hold_unchanged(hold_path, before)
    if status_path is not None:
        write_status(status_path, report)
    return report


def serve(
    repo: Any,
    *,
    watchdog: BoardStallWatchdog | None = None,
    hold_path: Path | None = None,
    status_path: Path | None = None,
    interval_seconds: float = 120.0,
    once: bool = False,
    spawn_repair: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]
    | None = None,
    sleep: Callable[[float], None] = time.sleep,
    stop_requested: Callable[[], bool] | None = None,
    readonly_observe: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run ticks until the process is asked to stop.

    A quiet stall stays in the status file and does not start another repair
    subagent. The operator hold is checked after every tick.
    """

    if interval_seconds <= 0:
        raise BoardStallWatchdogError("interval_seconds must be positive")
    owner = watchdog or BoardStallWatchdog()
    report: dict[str, Any] = {"action": "not_started"}
    hold_present = hold_path is not None and hold_path.is_file()
    while True:
        try:
            if readonly_observe is not None:
                observation = readonly_observe()
                signature = str(observation.get("signature") or "")
                if (int(observation.get("active_count") or 0) == 0
                        and owner.idle_recorded(signature)):
                    report = _stall_recorded(
                        observation,
                        hold_path=hold_path,
                        status_path=status_path,
                    )
                else:
                    report = run_once(
                        repo,
                        watchdog=owner,
                        hold_path=hold_path,
                        status_path=status_path,
                        spawn_repair=spawn_repair,
                    )
            else:
                report = run_once(
                    repo,
                    watchdog=owner,
                    hold_path=hold_path,
                    status_path=status_path,
                    spawn_repair=spawn_repair,
                )
        except BoardStallWatchdogError:
            raise
        except Exception as exc:
            if lock_conflict(exc):
                report = database_busy_report(
                    operator_hold_present=hold_present or (
                        hold_path is not None and hold_path.is_file()
                    ),
                    error=type(exc).__name__,
                )
            else:
                report = {
                    "action": "tick_failed",
                    "error": type(exc).__name__,
                    "executor_start_allowed": False,
                    "set_ready": False,
                    "grant_admission": False,
                    "preserve_operator_hold": hold_path is not None and hold_path.is_file(),
                }
            if status_path is not None:
                write_status(status_path, report)
        if once or (stop_requested is not None and stop_requested()):
            return report
        sleep(interval_seconds)
        if stop_requested is not None and stop_requested():
            return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the dependency-release supervisor watchdog. "
            "This process does not start the scientific executor and "
            "does not remove an operator hold."
        )
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--hold-file", type=Path, default=None)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=120.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--require-lock-clear",
        action="store_true",
        help="Refuse to open the database unless fuser reports it clear.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.require_lock_clear:
        import subprocess

        completed = subprocess.run(
            ["fuser", str(args.database)],
            check=False,
            capture_output=True,
            text=True,
        )
        if not file_lock_clear(
            completed.returncode,
            completed.stdout,
            completed.stderr,
        ):
            write_status(
                args.status_file,
                {
                    "action": "database_busy",
                    "executor_start_allowed": False,
                    "set_ready": False,
                    "grant_admission": False,
                },
            )
            return 2
    from .intent_repository import open_intent_repository

    repo = open_intent_repository(
        args.database,
        owner_id="dependency-block-release",
        install_schema=False,
        evidence_freshness_seconds=3600,
    )
    watchdog = BoardStallWatchdog()

    def readonly_observe() -> dict[str, Any]:
        return observe_catalogue_readonly(
            args.database,
            operator_hold_present=args.hold_file is not None and args.hold_file.is_file(),
        )

    stopping = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    report = serve(
        repo,
        watchdog=watchdog,
        hold_path=args.hold_file,
        status_path=args.status_file,
        interval_seconds=float(args.interval_seconds),
        once=bool(args.once),
        stop_requested=lambda: stopping,
        readonly_observe=readonly_observe,
    )
    print(json.dumps({"action": report.get("action"), "signature": report.get("signature")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
