#!/usr/bin/env python3
"""Fail-closed launcher for the incremental-verification planner board.

PR #176 removed the generic configured-board control plane while retaining the
ordinary implementation supervisor.  This adapter intentionally supports only
the reviewed IVP scheduler schema and maps it to that existing runtime.  It
does not implement a second scheduler, provider, or task authority.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
    CONFIGURATION_ROOT_ENV,
    PROFILE_ID_ENV,
    RUN_ID_ENV,
    TARGET_ID_ENV,
    LifecycleProfile,
    LinuxProcessAdapter,
    ProcessIdentity,
    ProcessIdentityMismatch,
    ProcessTreeNotFenced,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    ImplementationSupervisorTrackConfig,
    SupervisorTrack,
    build_configured_multi_supervisor_cli_runner,
    common_args_from_parsed_args,
    tracks_from_parsed_args,
    utc_run_stamp,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    build_arg_parser as build_runner_arg_parser,
)

SCHEDULER_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor."
    "incremental_verification_planner.scheduler_config@1"
)
LAUNCH_PLAN_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/incremental-verification-planner-launch-plan@1"
)
PREFLIGHT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/incremental-verification-planner-preflight@1"
)
LIFECYCLE_RECORD_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/incremental-verification-planner-lifecycle@1"
)
STATUS_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/incremental-verification-planner-status@1"
)
TERMINAL_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/incremental-verification-planner-terminal@1"
)
STOP_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/incremental-verification-planner-stop@1"
)
ENTRY_RELATIVE = Path("scripts/ops/agent_supervisor/implementation_supervisor_entry.py")
MASTER_TARGET_ID = "incremental-verification-planner:master"
MASTER_PID_FILENAME = "configured-board-master.pid"
MASTER_LIFECYCLE_FILENAME = "ivp-master-lifecycle.json"
MASTER_LIFECYCLE_LOCK_FILENAME = "ivp-master-lifecycle.lock"
MASTER_TERMINAL_FILENAME = "ivp-master-terminal.json"
MASTER_STOP_FILENAME = "ivp-master-stop.json"
MAX_JSON_BYTES = 1024 * 1024
PROVIDER_ENV = "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER"
FALLBACK_PROVIDER_ENV = "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER"
FALLBACK_TRIGGER_ENV = "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER"
GROK_MODEL_ENV = "IPFS_ACCELERATE_AGENT_GROK_MODEL"
CODEX_MODEL_ENV = "IPFS_ACCELERATE_AGENT_CODEX_MODEL"
CODEX_REASONING_EFFORT_ENV = "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT"
PROVIDER_FALLBACK_POLICY_ENV = "IPFS_ACCELERATE_AGENT_PROVIDER_FALLBACK_POLICY"
PROVIDER_ENV_NAMES = (
    PROVIDER_ENV,
    FALLBACK_PROVIDER_ENV,
    FALLBACK_TRIGGER_ENV,
    GROK_MODEL_ENV,
    CODEX_MODEL_ENV,
    CODEX_REASONING_EFFORT_ENV,
    PROVIDER_FALLBACK_POLICY_ENV,
)
FORBIDDEN_PROVIDER_OVERRIDE_ENV_NAMES = ("IMPLEMENTATION_DAEMON_COMMAND",)
EXPECTED_PROVIDER = {
    "primary_provider_id": "grok_cli",
    "primary_model_id": "grok-4.5",
    "fallback_provider_id": "codex",
    "fallback_model_id": "gpt-5.6-terra",
    "fallback_trigger": "primary_quota_exhausted",
    "fallback_reasoning_effort": "high",
}


class IVPSchedulerError(RuntimeError):
    """Raised when the sealed IVP launch boundary cannot be admitted."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise IVPSchedulerError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IVPSchedulerError(f"cannot load scheduler config: {path}") from exc
    if not isinstance(payload, dict):
        raise IVPSchedulerError("scheduler config must be a JSON object")
    return payload


def _safe_relative(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    candidate = PurePosixPath(text)
    if (
        not text
        or candidate.is_absolute()
        or ".." in candidate.parts
        or any(part in {"", "."} for part in candidate.parts)
    ):
        raise IVPSchedulerError(f"{field} must be a safe repository-relative path")
    return candidate.as_posix()


def _contained(repo_root: Path, relative: str) -> Path:
    candidate = (repo_root / relative).resolve(strict=False)
    try:
        candidate.relative_to(repo_root)
    except ValueError as exc:
        raise IVPSchedulerError(f"path escapes repository: {relative}") from exc
    return candidate


def _strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise IVPSchedulerError(f"{field} must be a JSON string array")
    items = tuple(_safe_relative(item, field=field) for item in value)
    if len(set(items)) != len(items):
        raise IVPSchedulerError(f"{field} contains duplicates")
    return items


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise IVPSchedulerError(f"{field} must be a positive integer")
    return int(value)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_root(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: object) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(dict(payload)) + b"\n"
    if len(encoded) > MAX_JSON_BYTES:
        raise IVPSchedulerError(f"runtime record is too large: {path.name}")
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _read_runtime_json(path: Path) -> dict[str, Any]:
    try:
        if path.is_symlink():
            raise IVPSchedulerError(f"runtime record must not be a symlink: {path}")
        with path.open("rb") as handle:
            raw = handle.read(MAX_JSON_BYTES + 1)
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise IVPSchedulerError(f"cannot read runtime record: {path}") from exc
    if len(raw) > MAX_JSON_BYTES:
        raise IVPSchedulerError(f"runtime record is too large: {path}")
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IVPSchedulerError(f"invalid runtime JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise IVPSchedulerError(f"runtime JSON must be an object: {path}")
    return payload


def _file_age_seconds(path: Path, *, now: float) -> float | None:
    try:
        return max(0.0, now - path.stat().st_mtime)
    except OSError:
        return None


@dataclass(frozen=True)
class IVPBoard:
    repo_root: Path
    config_path: Path
    payload: Mapping[str, Any]
    taskboard_path: str
    objectives_path: str
    validator_path: str
    merge_target_branch: str
    task_header_prefix: str
    max_lanes: int
    runtime_paths: Mapping[str, str]
    worktree_submodule_paths: tuple[str, ...]
    protected_paths: tuple[str, ...]

    def path(self, relative: str) -> Path:
        return _contained(self.repo_root, relative)

    @property
    def state_dir(self) -> Path:
        return self.path(self.runtime_paths["state"])

    @property
    def lifecycle_path(self) -> Path:
        return self.state_dir / MASTER_LIFECYCLE_FILENAME

    @property
    def lifecycle_lock_path(self) -> Path:
        return self.state_dir / MASTER_LIFECYCLE_LOCK_FILENAME

    @property
    def master_pid_path(self) -> Path:
        return self.state_dir / MASTER_PID_FILENAME

    @property
    def terminal_path(self) -> Path:
        return self.state_dir / MASTER_TERMINAL_FILENAME

    @property
    def stop_path(self) -> Path:
        return self.state_dir / MASTER_STOP_FILENAME


def load_board(config_path: Path | str, *, repo_root: Path | str) -> IVPBoard:
    root = Path(repo_root).resolve(strict=True)
    supplied = Path(config_path)
    config = (
        supplied.resolve(strict=True)
        if supplied.is_absolute()
        else (root / supplied).resolve(strict=True)
    )
    try:
        config.relative_to(root)
    except ValueError as exc:
        raise IVPSchedulerError("scheduler config is outside the repository") from exc
    payload = _load_json_object(config)
    if payload.get("schema") != SCHEDULER_SCHEMA:
        raise IVPSchedulerError("unsupported IVP scheduler schema")
    if payload.get("strict_task_sharding") is not True:
        raise IVPSchedulerError("IVP requires strict task sharding")
    if payload.get("objective_refill_enabled") is not False:
        raise IVPSchedulerError("IVP objective refill must remain disabled")
    if payload.get("codebase_refill_enabled") is not False:
        raise IVPSchedulerError("IVP codebase refill must remain disabled")
    if payload.get("exit_when_all_tracks_terminal") is not True:
        raise IVPSchedulerError("IVP requires terminal board drain")

    task_prefix = str(payload.get("task_prefix") or "").strip()
    if task_prefix != "IVP-":
        raise IVPSchedulerError("IVP task prefix differs from the sealed board")
    max_lanes = _positive_int(payload.get("max_lanes"), field="max_lanes")
    lanes = payload.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != max_lanes:
        raise IVPSchedulerError("lane count differs from max_lanes")
    for index, lane in enumerate(lanes):
        if (
            not isinstance(lane, Mapping)
            or lane.get("index") != index
            or lane.get("strict_shard_remainder") != index
        ):
            raise IVPSchedulerError(f"lane {index} is not its strict shard")

    runtime = payload.get("runtime_paths")
    if not isinstance(runtime, Mapping):
        raise IVPSchedulerError("runtime_paths must be an object")
    runtime_paths = {
        name: _safe_relative(runtime.get(name), field=f"runtime_paths.{name}")
        for name in ("root", "state", "worktrees", "merge_queue", "logs")
    }
    runtime_root = PurePosixPath(runtime_paths["root"])
    for name in ("state", "worktrees", "merge_queue", "logs"):
        try:
            PurePosixPath(runtime_paths[name]).relative_to(runtime_root)
        except ValueError as exc:
            raise IVPSchedulerError(
                f"runtime_paths.{name} escapes runtime root"
            ) from exc

    provider = payload.get("provider")
    if not isinstance(provider, Mapping):
        raise IVPSchedulerError("provider policy must be an object")
    observed_provider = {field: provider.get(field) for field in EXPECTED_PROVIDER}
    if observed_provider != EXPECTED_PROVIDER:
        raise IVPSchedulerError(
            "ordered provider route differs from the sealed IVP route"
        )
    if provider.get("max_concurrency") != max_lanes:
        raise IVPSchedulerError("provider concurrency differs from lane count")
    if (
        provider.get("secrets_from_environment_only") is not True
        or provider.get("secrets_in_argv_prompts_logs_or_receipts") is not False
    ):
        raise IVPSchedulerError("provider secret policy is not fail-closed")

    source = payload.get("source_binding")
    if not isinstance(source, Mapping):
        raise IVPSchedulerError("source_binding must be an object")
    if source.get("bootstrap_task_source") != "legacy-markdown":
        raise IVPSchedulerError(
            "only the explicit legacy-markdown task source is supported"
        )

    return IVPBoard(
        repo_root=root,
        config_path=config,
        payload=payload,
        taskboard_path=_safe_relative(
            payload.get("taskboard_path"), field="taskboard_path"
        ),
        objectives_path=_safe_relative(
            payload.get("objectives_path"), field="objectives_path"
        ),
        validator_path=_safe_relative(
            payload.get("validator_path"), field="validator_path"
        ),
        merge_target_branch=str(payload.get("merge_target_branch") or "").strip(),
        task_header_prefix="## IVP-",
        max_lanes=max_lanes,
        runtime_paths=runtime_paths,
        worktree_submodule_paths=_strings(
            payload.get("worktree_submodule_paths"),
            field="worktree_submodule_paths",
        ),
        protected_paths=_strings(
            payload.get("protected_paths"), field="protected_paths"
        ),
    )


def _git(
    board: IVPBoard, *args: str, timeout: float = 30.0
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(board.repo_root), *args],
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def _source_head(board: IVPBoard) -> str:
    completed = _git(board, "rev-parse", "--verify", "HEAD")
    value = completed.stdout.strip().lower()
    if (
        completed.returncode != 0
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise IVPSchedulerError("cannot bind an exact 40-hex source HEAD")
    return value


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _master_pid(board: IVPBoard) -> int:
    try:
        text = board.master_pid_path.read_text(encoding="ascii").strip()
    except OSError:
        return 0
    return int(text) if text.isdigit() else 0


def _validator_report(board: IVPBoard) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(board.repo_root / "ipfs_kit_py"),
            str(board.repo_root / "ipfs_datasets_py"),
            str(board.repo_root),
        )
    )
    completed = subprocess.run(
        [sys.executable, str(board.path(board.validator_path)), "--check-all"],
        cwd=board.repo_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    if completed.returncode != 0 or len(completed.stdout.encode("utf-8")) > 1024 * 1024:
        raise IVPSchedulerError("board validator did not complete successfully")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise IVPSchedulerError("board validator did not emit one JSON object") from exc
    if not isinstance(report, dict) or report.get("valid") is not True:
        raise IVPSchedulerError("board validator rejected the IVP controls")
    return report


def _exact_pythonpath(board: IVPBoard) -> str:
    return os.pathsep.join(
        (
            str(board.repo_root / "ipfs_kit_py"),
            str(board.repo_root / "ipfs_datasets_py"),
            str(board.repo_root),
        )
    )


def _ambient_provider_conflicts(
    expected: Mapping[str, str],
    *,
    ambient: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    environment = os.environ if ambient is None else ambient
    return tuple(
        sorted(
            name
            for name, expected_value in expected.items()
            if str(environment.get(name, "")).strip()
            and str(environment.get(name, "")).strip() != expected_value
        )
    )


@contextlib.contextmanager
def _temporary_provider_environment(expected: Mapping[str, str]):
    previous = {name: os.environ.get(name) for name in expected}
    try:
        os.environ.update(expected)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _probe_provider_readiness(environment: Mapping[str, str]) -> Any:
    from ipfs_accelerate_py.llm_router import (
        probe_grok_codex_agent_route_readiness,
    )

    return probe_grok_codex_agent_route_readiness(
        grok_model="grok-4.5",
        codex_model="gpt-5.6-terra",
        codex_reasoning_effort="high",
        timeout_seconds=10.0,
        environment=environment,
    )


def _build_provider_route_command(
    board: IVPBoard,
    expected: Mapping[str, str],
) -> list[str]:
    """Build, but never execute, the actual sealed implementation route."""

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        _codex_implementation_command,
        _configured_legacy_quota_only_fallback_policy,
        _grok_binary,
        _grok_cli_command,
        _ordered_provider_fallback_command,
    )

    with _temporary_provider_environment(expected):
        grok = _grok_binary()
        codex = shutil.which("codex")
        if not grok or not codex:
            raise IVPSchedulerError("sealed provider binaries are unavailable")
        policy = _configured_legacy_quota_only_fallback_policy("grok_cli")
        command = _ordered_provider_fallback_command(
            workspace_path=board.repo_root,
            primary_provider="grok",
            primary_command=_grok_cli_command(workspace_path=board.repo_root),
            fallback_provider="codex",
            fallback_command=_codex_implementation_command(
                codex=codex,
                workspace_path=board.repo_root,
            ),
            fallback_policy=policy,
        )
    return command


def _provider_preflight(board: IVPBoard) -> dict[str, Any]:
    expected = provider_environment(board)
    conflicts = _ambient_provider_conflicts(expected)
    if conflicts:
        raise IVPSchedulerError(
            "ambient provider policy conflicts with sealed IVP route: "
            + ", ".join(conflicts)
        )
    forbidden = tuple(
        name
        for name in FORBIDDEN_PROVIDER_OVERRIDE_ENV_NAMES
        if str(os.environ.get(name, "")).strip()
    )
    if forbidden:
        raise IVPSchedulerError(
            "ambient implementation command override is forbidden: "
            + ", ".join(forbidden)
        )
    environment = dict(os.environ)
    environment.update(expected)
    environment["PYTHONPATH"] = _exact_pythonpath(board)
    readiness = _probe_provider_readiness(environment)
    if not bool(getattr(readiness, "grok_ready", False)):
        raise IVPSchedulerError(
            "quota-only route requires a ready and authenticated Grok primary"
        )
    if not bool(getattr(readiness, "codex_ready", False)):
        raise IVPSchedulerError(
            "quota-only route requires a ready and authenticated Codex fallback"
        )
    if (
        str(getattr(readiness, "grok_model", "")) != "grok-4.5"
        or str(getattr(readiness, "codex_model", "")) != "gpt-5.6-terra"
        or str(getattr(readiness, "codex_reasoning_effort", "")) != "high"
    ):
        raise IVPSchedulerError("provider readiness returned a mismatched route")

    command = _build_provider_route_command(board, expected)
    expected_runner = (
        board.repo_root
        / "ipfs_accelerate_py/agent_supervisor/provider_fallback_runner.py"
    ).resolve()
    if (
        len(command) < 2
        or Path(command[0]).resolve(strict=False)
        != Path(sys.executable).resolve(strict=False)
        or Path(command[1]).resolve(strict=False) != expected_runner
        or "--fallback-policy" not in command
        or command[command.index("--fallback-policy") + 1] != "grok_quota_only"
        or "--primary-unavailable-kind" in command
    ):
        raise IVPSchedulerError("provider route dry proof did not bind quota-only")
    return {
        "grok_ready": True,
        "codex_ready": True,
        "effective_provider": str(getattr(readiness, "effective_provider", "")),
        "reason_code": str(getattr(readiness, "reason_code", "")),
        "fallback_policy": "grok_quota_only",
        "runner": str(expected_runner),
    }


def _validate_launch_plan(board: IVPBoard, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Dry-parse the exact runner, lane, supervisor, and reconciliation inputs."""

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        PortalImplementationSupervisor,
        supervisor_config_from_args,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
        parse_args as parse_supervisor_args,
    )

    parsed_runner = build_runner_arg_parser().parse_args(list(plan["argv"]))
    common = common_args_from_parsed_args(parsed_runner)
    tracks = tracks_from_parsed_args(parsed_runner)
    if len(tracks) != board.max_lanes or len(common) != 59:
        raise IVPSchedulerError("runner lane/common-argument expansion drifted")
    shards: list[dict[str, int]] = []
    for index, track in enumerate(tracks):
        parsed = parse_supervisor_args([*common, *track.extra_args])
        config = supervisor_config_from_args(parsed, repo_root=board.repo_root)
        if (
            not config.implement
            or config.objective_refill_enabled
            or config.codebase_refill_enabled
            or not config.strict_task_sharding
            or config.task_shard_count != board.max_lanes
            or config.task_shard_index != index
        ):
            raise IVPSchedulerError(f"lane {index} supervisor mapping drifted")
        reconciliation = PortalImplementationSupervisor(
            config
        )._build_worktree_reconciliation_daemon()
        if (
            reconciliation.task_shard_count != board.max_lanes
            or reconciliation.task_shard_index != index
            or not reconciliation.strict_task_sharding
        ):
            raise IVPSchedulerError(
                f"lane {index} reconciliation shard mapping drifted"
            )
        shards.append({"count": board.max_lanes, "index": index})
    return {
        "common_arg_count": len(common),
        "lane_count": len(tracks),
        "strict_shards": shards,
    }


def _plan_configuration_root(plan: Mapping[str, Any]) -> str:
    admitted = {
        key: plan.get(key)
        for key in (
            "argv",
            "environment",
            "expected_task_count",
            "lanes",
            "master_log",
            "master_pid_path",
            "runtime_root",
            "source_head",
            "stamp",
            "strict_task_sharding",
        )
    }
    return _sha256_root(admitted)


def _master_profile(board: IVPBoard, plan: Mapping[str, Any]) -> LifecycleProfile:
    configuration_root = _plan_configuration_root(plan)
    stamp = str(plan.get("stamp") or "").strip()
    if not stamp:
        raise IVPSchedulerError("launch plan has no run stamp")
    duration = float(plan.get("duration_seconds") or 0)
    expected_count = int(plan.get("expected_task_count") or 0)
    command = (
        sys.executable,
        str(Path(__file__).resolve()),
        "--repo-root",
        str(board.repo_root),
        "--config",
        str(board.config_path),
        "_run-master",
        "--duration-seconds",
        str(duration),
        "--stamp",
        stamp,
        "--expected-task-count",
        str(expected_count),
        "--expected-configuration-root",
        configuration_root,
    )
    environment = {
        **provider_environment(board),
        "PYTHONPATH": _exact_pythonpath(board),
    }
    run_component = hashlib.sha256(f"{stamp}:{configuration_root}".encode()).hexdigest()
    return LifecycleProfile(
        target_id=MASTER_TARGET_ID,
        run_id=f"ivp-master:{run_component}",
        configuration_root=configuration_root,
        repository_root=str(board.repo_root),
        state_root=str(board.state_dir),
        run_root=str(board.state_dir / "lifecycle-runs" / run_component),
        argv=command,
        cwd=str(board.repo_root),
        environment=tuple(environment.items()),
    )


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _lane_profiles(
    board: IVPBoard, plan: Mapping[str, Any]
) -> tuple[tuple[SupervisorTrack, LifecycleProfile], ...]:
    parsed_runner = build_runner_arg_parser().parse_args(list(plan["argv"]))
    common = common_args_from_parsed_args(parsed_runner)
    python_executable = str(parsed_runner.python_executable)
    result: list[tuple[SupervisorTrack, LifecycleProfile]] = []
    for track in tracks_from_parsed_args(parsed_runner):
        resolved = track.resolve(board.repo_root)
        command = (
            [python_executable, "-m", resolved.module_name, *resolved.extra_args]
            if resolved.module_name
            else [
                python_executable,
                str(resolved.script_path),
                *common,
                *resolved.extra_args,
            ]
        )
        configuration_root = _sha256_root(command)
        state_root = resolved.supervisor_pid_path.parent.resolve(strict=False)
        run_root = state_root / "lifecycle-runs" / resolved.name
        status_path = resolved.supervisor_status_path
        profile = LifecycleProfile(
            target_id=f"supervisor-track:{resolved.name}",
            run_id=(
                "multi-supervisor:"
                + hashlib.sha256(
                    f"{board.repo_root}:{resolved.name}".encode()
                ).hexdigest()
            ),
            configuration_root=configuration_root,
            repository_root=str(board.repo_root),
            state_root=str(state_root),
            run_root=str(run_root),
            argv=tuple(command),
            cwd=str(board.repo_root),
            health_path=(
                str(status_path.resolve(strict=False))
                if status_path is not None
                and _path_within(status_path.resolve(strict=False), state_root)
                else ""
            ),
        )
        result.append((resolved, profile))
    return tuple(result)


def _load_lifecycle_record(board: IVPBoard) -> dict[str, Any]:
    record = _read_runtime_json(board.lifecycle_path)
    if not record:
        return {}
    if record.get("schema") != LIFECYCLE_RECORD_SCHEMA:
        raise IVPSchedulerError("unsupported IVP lifecycle record")
    profile_payload = record.get("profile")
    plan = record.get("plan")
    if not isinstance(profile_payload, Mapping) or not isinstance(plan, Mapping):
        raise IVPSchedulerError("IVP lifecycle record is incomplete")
    profile = LifecycleProfile.from_dict(profile_payload)
    if (
        profile.repository_root != str(board.repo_root)
        or profile.state_root != str(board.state_dir)
        or profile.target_id != MASTER_TARGET_ID
        or profile.configuration_root != _plan_configuration_root(plan)
    ):
        raise IVPSchedulerError("IVP lifecycle record binding mismatch")
    identity_payload = record.get("identity")
    if identity_payload is not None:
        if not isinstance(identity_payload, Mapping):
            raise IVPSchedulerError("IVP lifecycle identity is malformed")
        identity = ProcessIdentity.from_dict(identity_payload)
        if (
            identity.profile_id != profile.profile_id
            or identity.run_id != profile.run_id
        ):
            raise IVPSchedulerError("IVP lifecycle identity binding mismatch")
    return record


def _collision_check(
    board: IVPBoard,
    plan: Mapping[str, Any],
    *,
    adapter: LinuxProcessAdapter | None = None,
) -> dict[str, Any]:
    process_adapter = adapter or LinuxProcessAdapter()
    record = _load_lifecycle_record(board)
    profiles: list[LifecycleProfile] = []
    if record:
        profiles.append(LifecycleProfile.from_dict(record["profile"]))
        profiles.extend(
            profile for _track, profile in _lane_profiles(board, record["plan"])
        )
    profiles.extend(profile for _track, profile in _lane_profiles(board, plan))
    live: list[str] = []
    seen: set[str] = set()
    for profile in profiles:
        if profile.profile_id in seen:
            continue
        seen.add(profile.profile_id)
        tree = process_adapter.snapshot(profile)
        if tree.members:
            live.append(profile.target_id)
    marker_pid = _master_pid(board)
    if _pid_alive(marker_pid) and MASTER_TARGET_ID not in live:
        raise IVPSchedulerError("live master PID has no exact lifecycle authority")
    return {
        "safe": not live,
        "live_targets": sorted(live),
        "stale_master_pid": marker_pid
        if marker_pid and not _pid_alive(marker_pid)
        else None,
    }


def preflight(board: IVPBoard) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, bool] = {}
    required_controls = tuple(
        dict.fromkeys(
            (
                board.config_path.relative_to(board.repo_root).as_posix(),
                board.taskboard_path,
                board.objectives_path,
                _safe_relative(board.payload.get("plan_path"), field="plan_path"),
                board.validator_path,
                *board.protected_paths,
            )
        )
    )
    checks["control_files_present"] = all(
        board.path(path).is_file() for path in required_controls
    )
    if not checks["control_files_present"]:
        errors.append("one or more sealed control files are missing")

    tracked = _git(board, "ls-files", "--error-unmatch", "--", *required_controls)
    checks["control_files_tracked"] = tracked.returncode == 0
    if not checks["control_files_tracked"]:
        errors.append("one or more sealed control files are untracked")

    status = _git(
        board,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    checks["checkout_clean"] = status.returncode == 0 and not status.stdout.strip()
    if not checks["checkout_clean"]:
        errors.append("checkout or configured submodules are dirty")

    branch = _git(board, "symbolic-ref", "--quiet", "--short", "HEAD")
    source = board.payload.get("source_binding")
    source = source if isinstance(source, Mapping) else {}
    required_branch = str(source.get("accelerator_required_branch") or "").strip()
    checks["branch_exact"] = (
        branch.returncode == 0
        and branch.stdout.strip() == required_branch == board.merge_target_branch
    )
    if not checks["branch_exact"]:
        errors.append("checkout branch differs from the sealed source/merge target")

    ancestor = str(source.get("accelerator_required_ancestor") or "").strip()
    ancestry = (
        _git(board, "merge-base", "--is-ancestor", ancestor, "HEAD")
        if ancestor
        else None
    )
    checks["required_ancestor"] = bool(
        ancestor and ancestry is not None and ancestry.returncode == 0
    )
    if not checks["required_ancestor"]:
        errors.append("required latest-main ancestor is absent")

    gitlink_specs = (
        (
            str(source.get("ipfs_kit_submodule_path") or ""),
            str(source.get("ipfs_kit_planning_revision") or ""),
        ),
        (
            str(source.get("ipfs_datasets_submodule_path") or ""),
            str(source.get("ipfs_datasets_planning_revision") or ""),
        ),
    )
    gitlinks_ok = True
    for relative, expected in gitlink_specs:
        try:
            safe_relative = _safe_relative(relative, field="source_binding submodule")
        except IVPSchedulerError:
            gitlinks_ok = False
            continue
        index = _git(board, "ls-files", "-s", "--", safe_relative)
        fields = index.stdout.strip().split()
        nested = board.path(safe_relative)
        nested_head = (
            subprocess.run(
                ["git", "-C", str(nested), "rev-parse", "HEAD"],
                text=True,
                capture_output=True,
                check=False,
            )
            if nested.is_dir()
            else None
        )
        nested_status = (
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(nested),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if nested_head is not None
            else None
        )
        gitlinks_ok = gitlinks_ok and bool(
            len(fields) >= 2
            and fields[0] == "160000"
            and fields[1] == expected
            and nested_head is not None
            and nested_head.returncode == 0
            and nested_head.stdout.strip() == expected
            and nested_status is not None
            and nested_status.returncode == 0
            and not nested_status.stdout.strip()
        )
    checks["gitlinks_exact_and_clean"] = gitlinks_ok
    if not gitlinks_ok:
        errors.append(
            "configured adapter gitlinks are not exact, initialized, and clean"
        )

    entry = board.repo_root / ENTRY_RELATIVE
    checks["implementation_entry_present"] = entry.is_file()
    if not checks["implementation_entry_present"]:
        errors.append("implementation supervisor entry script is missing")

    validator_report: dict[str, Any] = {}
    try:
        validator_report = _validator_report(board)
        checks["board_validator"] = True
    except (IVPSchedulerError, OSError, subprocess.SubprocessError) as exc:
        checks["board_validator"] = False
        errors.append(str(exc))

    runner_mapping: dict[str, Any] = {}
    collision: dict[str, Any] = {}
    dry_plan: dict[str, Any] | None = None
    if checks["board_validator"]:
        try:
            dry_plan = launch_plan(
                board,
                implement=True,
                foreground=False,
                duration_seconds=60.0,
                stamp="ivp-preflight",
                expected_task_count=int(validator_report.get("task_count") or 0),
            )
            runner_mapping = _validate_launch_plan(board, dry_plan)
            checks["runner_mapping"] = True
        except (IVPSchedulerError, OSError, SystemExit, ValueError) as exc:
            checks["runner_mapping"] = False
            errors.append(f"runner mapping rejected: {exc}")

        try:
            if dry_plan is None:
                raise IVPSchedulerError("runner mapping produced no launch plan")
            collision = _collision_check(board, dry_plan)
            checks["no_live_master_or_lane"] = bool(collision.get("safe"))
            if not checks["no_live_master_or_lane"]:
                errors.append(
                    "an IVP lifecycle is already live: "
                    + ", ".join(collision.get("live_targets") or ())
                )
        except (
            IVPSchedulerError,
            OSError,
            ProcessIdentityMismatch,
            ValueError,
        ) as exc:
            checks["no_live_master_or_lane"] = False
            errors.append(f"runtime collision check rejected: {exc}")
    else:
        checks["runner_mapping"] = False
        checks["no_live_master_or_lane"] = False

    provider_report: dict[str, Any] = {}
    try:
        provider_report = _provider_preflight(board)
        checks["provider_route_ready"] = True
    except (IVPSchedulerError, OSError, subprocess.SubprocessError, ValueError) as exc:
        checks["provider_route_ready"] = False
        errors.append(str(exc))

    return {
        "schema": PREFLIGHT_SCHEMA,
        "valid": not errors,
        "errors": errors,
        "checks": checks,
        "branch": branch.stdout.strip(),
        "required_ancestor": ancestor,
        "validator_report": validator_report,
        "runner_mapping": runner_mapping,
        "provider": provider_report,
        "collision": collision,
    }


def provider_environment(board: IVPBoard) -> dict[str, str]:
    provider = board.payload.get("provider")
    if not isinstance(provider, Mapping):
        raise IVPSchedulerError("provider policy must be an object")
    observed = {field: provider.get(field) for field in EXPECTED_PROVIDER}
    if observed != EXPECTED_PROVIDER:
        raise IVPSchedulerError(
            "ordered provider route differs from the sealed IVP route"
        )
    return {
        PROVIDER_ENV: "grok_cli",
        FALLBACK_PROVIDER_ENV: "codex",
        FALLBACK_TRIGGER_ENV: "primary_quota_exhausted",
        GROK_MODEL_ENV: "grok-4.5",
        CODEX_MODEL_ENV: "gpt-5.6-terra",
        CODEX_REASONING_EFFORT_ENV: "high",
        PROVIDER_FALLBACK_POLICY_ENV: "grok_quota_only",
    }


def common_supervisor_args(board: IVPBoard, *, implement: bool) -> tuple[str, ...]:
    payload = board.payload
    args: list[str] = [
        "--todo-path",
        str(board.path(board.taskboard_path)),
        "--task-prefix",
        board.task_header_prefix,
        "--worktree-root",
        str(board.path(board.runtime_paths["worktrees"])),
        "--merge-target-branch",
        board.merge_target_branch,
        "--merge-queue-dir",
        str(board.path(board.runtime_paths["merge_queue"])),
        "--stale-seconds",
        str(_positive_int(payload.get("stale_seconds"), field="stale_seconds")),
        "--check-interval",
        str(
            _positive_int(
                payload.get("check_interval_seconds"), field="check_interval_seconds"
            )
        ),
        "--watchdog-startup-grace-seconds",
        str(
            _positive_int(
                payload.get("watchdog_startup_grace_seconds"),
                field="watchdog_startup_grace_seconds",
            )
        ),
        "--max-restarts",
        str(_positive_int(payload.get("max_restarts"), field="max_restarts")),
        "--max-task-attempts",
        str(_positive_int(payload.get("max_task_attempts"), field="max_task_attempts")),
        "--daemon-interval",
        str(
            _positive_int(
                payload.get("daemon_interval_seconds"), field="daemon_interval_seconds"
            )
        ),
        "--implementation-timeout",
        str(
            _positive_int(
                payload.get("implementation_timeout_seconds"),
                field="implementation_timeout_seconds",
            )
        ),
        "--implementation-max-timeout",
        str(
            _positive_int(
                payload.get("implementation_max_timeout_seconds"),
                field="implementation_max_timeout_seconds",
            )
        ),
        "--implementation-log-stall-seconds",
        str(
            _positive_int(
                payload.get("implementation_log_stall_seconds"),
                field="implementation_log_stall_seconds",
            )
        ),
        "--implementation-retry-budget",
        str(
            _positive_int(
                payload.get("implementation_retry_budget"),
                field="implementation_retry_budget",
            )
        ),
        "--validation-retry-budget",
        str(
            _positive_int(
                payload.get("validation_retry_budget"), field="validation_retry_budget"
            )
        ),
        "--merge-retry-budget",
        str(
            _positive_int(payload.get("merge_retry_budget"), field="merge_retry_budget")
        ),
        "--no-objective-task-janitor",
        "--no-objective-goal-completion-reconcile",
        "--no-objective-goal-migration",
        "--strict-task-sharding",
        "--log-level",
        "INFO",
        "--implement" if implement else "--no-implement",
    ]
    for relative in board.worktree_submodule_paths:
        args.extend(("--worktree-submodule-path", relative))
    for relative in board.protected_paths:
        args.extend(("--implementation-protected-path", relative))
    return tuple(args)


def launch_plan(
    board: IVPBoard,
    *,
    implement: bool,
    foreground: bool,
    duration_seconds: float,
    stamp: str | None = None,
    expected_task_count: int = 0,
) -> dict[str, Any]:
    if not implement:
        raise IVPSchedulerError("launch requires explicit --implement authority")
    if duration_seconds <= 0:
        raise IVPSchedulerError("duration_seconds must be positive")
    run_stamp = stamp or utc_run_stamp()
    state_dir = board.path(board.runtime_paths["state"])
    log_dir = board.path(board.runtime_paths["logs"])
    if expected_task_count < 0:
        raise IVPSchedulerError("expected_task_count must not be negative")
    master_pid = board.master_pid_path
    master_log = log_dir / f"configured-board-{run_stamp}.log"
    runner = build_configured_multi_supervisor_cli_runner(
        repo_root=board.repo_root,
        duration_seconds=duration_seconds,
        heartbeat_interval_seconds=30,
        supervisor_status_stale_seconds=board.payload["stale_seconds"],
        stop_grace_seconds=30,
        stamp=run_stamp,
        master_dir=log_dir,
        master_log=master_log,
        master_pid_path=master_pid,
        label="incremental-verification-planner",
        python_executable=sys.executable,
        implementation_track_configs=(
            ImplementationSupervisorTrackConfig(
                name="incremental-verification-planner",
                script_path=board.repo_root / ENTRY_RELATIVE,
                state_dir=state_dir,
                state_prefix="ivp",
            ),
        ),
        common_args=common_supervisor_args(board, implement=True),
        # The IVP launcher owns detachment through an exact LifecycleProfile.
        # The generic runner's detached mode writes only a bare PID projection.
        detach=False,
    )
    argv = [
        *runner.args(),
        "--implementation-supervisor-lanes-per-track",
        str(board.max_lanes),
    ]
    plan = {
        "schema": LAUNCH_PLAN_SCHEMA,
        "argv": argv,
        "environment": provider_environment(board),
        "implement": True,
        "foreground": foreground,
        "duration_seconds": float(duration_seconds),
        "expected_task_count": int(expected_task_count),
        "lanes": board.max_lanes,
        "strict_task_sharding": True,
        "stamp": run_stamp,
        "master_pid_path": str(master_pid),
        "master_log": str(master_log),
        "runtime_root": str(board.path(board.runtime_paths["root"])),
        "source_head": _source_head(board),
    }
    plan["configuration_root"] = _plan_configuration_root(plan)
    return plan


def _validated_launch_environment(board: IVPBoard, supplied: object) -> dict[str, str]:
    expected = provider_environment(board)
    if not isinstance(supplied, Mapping) or dict(supplied) != expected:
        raise IVPSchedulerError("launch plan provider environment was modified")
    conflicts = _ambient_provider_conflicts(expected)
    if conflicts:
        raise IVPSchedulerError(
            "ambient provider policy conflicts with sealed IVP route: "
            + ", ".join(conflicts)
        )
    forbidden = tuple(
        name
        for name in FORBIDDEN_PROVIDER_OVERRIDE_ENV_NAMES
        if str(os.environ.get(name, "")).strip()
    )
    if forbidden:
        raise IVPSchedulerError(
            "ambient implementation command override is forbidden: "
            + ", ".join(forbidden)
        )
    return {**expected, "PYTHONPATH": _exact_pythonpath(board)}


def _lifecycle_record(
    profile: LifecycleProfile,
    plan: Mapping[str, Any],
    *,
    launched_at: str,
    identity: ProcessIdentity | None = None,
) -> dict[str, Any]:
    return {
        "schema": LIFECYCLE_RECORD_SCHEMA,
        "launched_at": launched_at,
        "profile": profile.to_dict(),
        "identity": identity.to_dict() if identity is not None else None,
        "plan": dict(plan),
    }


def _remove_pid_projection_if_dead(path: Path, *, expected_pid: int = 0) -> bool:
    try:
        raw = path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    pid = int(raw) if raw.isdigit() else 0
    if expected_pid and pid != expected_pid:
        return False
    if _pid_alive(pid):
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        return False
    return True


def run_launch(
    board: IVPBoard,
    plan: Mapping[str, Any],
    *,
    adapter: LinuxProcessAdapter | None = None,
) -> int:
    for relative in ("root", "state", "worktrees", "merge_queue", "logs"):
        board.path(board.runtime_paths[relative]).mkdir(parents=True, exist_ok=True)
    _validated_launch_environment(board, plan.get("environment"))
    if plan.get("configuration_root") != _plan_configuration_root(plan):
        raise IVPSchedulerError("launch plan configuration root mismatch")
    _validate_launch_plan(board, plan)
    if plan.get("source_head") != _source_head(board):
        raise IVPSchedulerError("source HEAD changed after launch-plan admission")
    profile = _master_profile(board, plan)
    process_adapter = adapter or LinuxProcessAdapter()
    lock_fd = os.open(
        board.lifecycle_lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise IVPSchedulerError(
                "another IVP launch/stop operation is active"
            ) from exc
        collision = _collision_check(board, plan, adapter=process_adapter)
        if not collision["safe"]:
            raise IVPSchedulerError(
                "an IVP lifecycle is already live: "
                + ", ".join(collision["live_targets"])
            )
        _remove_pid_projection_if_dead(board.master_pid_path)
        launched_at = _utc_now()
        _atomic_json(
            board.lifecycle_path,
            _lifecycle_record(profile, plan, launched_at=launched_at),
        )
        if plan.get("source_head") != _source_head(board):
            raise IVPSchedulerError("source HEAD changed immediately before launch")
        identity = process_adapter.launch(profile, fencing_epoch=0)
        _atomic_json(
            board.lifecycle_path,
            _lifecycle_record(
                profile,
                plan,
                launched_at=launched_at,
                identity=identity,
            ),
        )
    finally:
        os.close(lock_fd)

    if not bool(plan.get("foreground")):
        return 0
    try:
        while process_adapter.identity_alive(identity):
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop(board, adapter=process_adapter)
    report = status(board, adapter=process_adapter)
    return 0 if report["lifecycle"] in {"completed", "stopped"} else 2


def _read_lane_projection(
    board: IVPBoard,
    track: SupervisorTrack,
    profile: LifecycleProfile,
    *,
    adapter: LinuxProcessAdapter,
    launched_at: float,
    expected_task_count: int,
    now: float,
) -> dict[str, Any]:
    status_path = track.supervisor_status_path
    if status_path is None:
        raise IVPSchedulerError(f"lane {track.name} has no status path")
    prefix = status_path.name.removesuffix("_supervisor_status.json")
    task_path = status_path.parent / f"{prefix}_task_state.json"
    status_payload = _read_runtime_json(status_path)
    task_payload = _read_runtime_json(task_path)
    tree = adapter.snapshot(profile)
    status_age = _file_age_seconds(status_path, now=now)
    task_age = _file_age_seconds(task_path, now=now)
    status_fresh_for_run = bool(
        status_age is not None and status_path.stat().st_mtime + 1.0 >= launched_at
    )
    task_fresh_for_run = bool(
        task_age is not None and task_path.stat().st_mtime + 1.0 >= launched_at
    )

    def count(name: str) -> int | None:
        value = task_payload.get(name)
        return value if type(value) is int and value >= 0 else None

    counts = {
        name: count(name)
        for name in (
            "task_count",
            "completed_count",
            "ready_count",
            "selectable_ready_count",
            "eligible_ready_count",
            "external_reserved_count",
            "waiting_count",
            "blocked_count",
        )
    }
    active_task_id = str(task_payload.get("active_task_id") or "").strip()
    implementation_in_progress = task_payload.get("implementation_in_progress") is True
    heartbeat_at = str(task_payload.get("heartbeat_at") or "").strip()
    last_progress_at = str(task_payload.get("last_progress_at") or "").strip()
    progress_timestamp = _parse_time(last_progress_at)
    progress_age = (
        max(0.0, now - progress_timestamp) if progress_timestamp is not None else None
    )
    hard_progress_limit = float(board.payload["implementation_max_timeout_seconds"]) + (
        2.0 * float(board.payload["check_interval_seconds"])
    )
    active_progress_hard_timeout_exceeded = bool(
        active_task_id
        and implementation_in_progress
        and progress_age is not None
        and progress_age > hard_progress_limit
    )
    terminal_fields = bool(
        expected_task_count > 0
        and counts["task_count"] == expected_task_count
        and counts["completed_count"] == expected_task_count
        and all(
            counts[name] == 0
            for name in (
                "ready_count",
                "selectable_ready_count",
                "eligible_ready_count",
                "external_reserved_count",
                "waiting_count",
                "blocked_count",
            )
        )
        and not active_task_id
        and not implementation_in_progress
    )
    terminal = bool(terminal_fields and status_fresh_for_run and task_fresh_for_run)
    idle_reason = str(task_payload.get("selection_idle_reason") or "").strip()
    blockers: list[str] = []
    if (counts["blocked_count"] or 0) > 0:
        blockers.append(f"blocked_tasks_present:{counts['blocked_count']}")
    if idle_reason == "all_selectable_ready_tasks_reached_max_task_attempts":
        blockers.append("all_selectable_ready_tasks_reached_attempt_limit")
    return {
        "name": track.name,
        "profile_id": profile.profile_id,
        "status_path": str(status_path),
        "task_state_path": str(task_path),
        "status": status_payload.get("status"),
        "status_age_seconds": status_age,
        "task_state_age_seconds": task_age,
        "status_fresh_for_run": status_fresh_for_run,
        "task_state_fresh_for_run": task_fresh_for_run,
        "process_pids": [member.pid for member in tree.members],
        "process_root_count": len(tree.roots),
        "process_live": bool(tree.members),
        "active_task_id": active_task_id,
        "implementation_in_progress": implementation_in_progress,
        "heartbeat_at": heartbeat_at,
        "last_progress_at": last_progress_at,
        "progress_age_seconds": progress_age,
        "active_phase": str(task_payload.get("active_phase") or ""),
        "active_phase_started_at": str(
            task_payload.get("active_phase_started_at") or ""
        ),
        "active_phase_detail": str(task_payload.get("active_phase_detail") or ""),
        "active_progress_hard_timeout_seconds": hard_progress_limit,
        "active_progress_hard_timeout_exceeded": active_progress_hard_timeout_exceeded,
        "last_implementation_returncode": task_payload.get(
            "last_implementation_returncode"
        ),
        "last_implementation_error": str(
            task_payload.get("last_implementation_error") or ""
        ),
        "last_implementation_commit": str(
            task_payload.get("last_implementation_commit") or ""
        ),
        "last_merge_returncode": task_payload.get("last_merge_returncode"),
        "last_merge_commit": str(task_payload.get("last_merge_commit") or ""),
        "last_merge_error": str(task_payload.get("last_merge_error") or ""),
        "selection_idle_reason": idle_reason,
        "terminal_fields_satisfied": terminal_fields,
        "terminal": terminal,
        "blockers": blockers,
        **counts,
    }


def _terminal_evidence(
    board: IVPBoard,
    plan: Mapping[str, Any],
    *,
    adapter: LinuxProcessAdapter,
    launched_at: float,
    require_live_lanes: bool,
) -> tuple[bool, list[dict[str, Any]]]:
    now = time.time()
    expected = int(plan.get("expected_task_count") or 0)
    lanes = [
        _read_lane_projection(
            board,
            track,
            profile,
            adapter=adapter,
            launched_at=launched_at,
            expected_task_count=expected,
            now=now,
        )
        for track, profile in _lane_profiles(board, plan)
    ]
    terminal = bool(
        lanes
        and all(lane["terminal"] for lane in lanes)
        and (
            all(lane["process_live"] for lane in lanes) if require_live_lanes else True
        )
    )
    return terminal, lanes


def _same_process_identity_after_reparenting(
    recorded: ProcessIdentity,
    observed: ProcessIdentity,
) -> bool:
    """Match one process instance while allowing Linux parent adoption.

    ``ProcessIdentity.identity_id`` commits to ``parent_pid``.  A detached
    master can outlive the short-lived launcher and be adopted by init (or a
    subreaper), so a fresh snapshot has a different parent and therefore a
    different canonical identity.  Both identities have already validated
    their canonical IDs.  Comparing every canonical field except the mutable
    parent and its derived ID preserves the PID-birth, boot, executable,
    session, lifecycle-marker, fence, and configuration checks.  Any other
    drift continues to fail closed.
    """

    recorded_payload = recorded.to_dict()
    observed_payload = observed.to_dict()
    for mutable_field in ("parent_pid", "identity_id"):
        recorded_payload.pop(mutable_field)
        observed_payload.pop(mutable_field)
    return recorded_payload == observed_payload


def status(
    board: IVPBoard,
    *,
    adapter: LinuxProcessAdapter | None = None,
) -> dict[str, Any]:
    observed_at = _utc_now()
    now = time.time()
    process_adapter = adapter or LinuxProcessAdapter()
    issues: list[str] = []
    blockers: list[str] = []
    try:
        record = _load_lifecycle_record(board)
    except (IVPSchedulerError, OSError, ValueError) as exc:
        return {
            "schema": STATUS_SCHEMA,
            "observed_at": observed_at,
            "lifecycle": "unhealthy",
            "healthy": False,
            "issues": [str(exc)],
            "blockers": [],
            "lanes": [],
        }
    marker_pid = _master_pid(board)
    if not record:
        if _pid_alive(marker_pid):
            issues.append("live_master_pid_without_lifecycle_record")
            lifecycle = "unhealthy"
        else:
            lifecycle = "not_started"
        return {
            "schema": STATUS_SCHEMA,
            "observed_at": observed_at,
            "lifecycle": lifecycle,
            "healthy": lifecycle == "not_started",
            "master_pid": marker_pid or None,
            "issues": issues,
            "blockers": [],
            "lanes": [],
        }

    profile = LifecycleProfile.from_dict(record["profile"])
    plan = record["plan"]
    launched_at = _parse_time(record.get("launched_at")) or 0.0
    try:
        master_tree = process_adapter.snapshot(profile)
        terminal, lanes = _terminal_evidence(
            board,
            plan,
            adapter=process_adapter,
            launched_at=launched_at,
            require_live_lanes=False,
        )
    except (OSError, ValueError, ProcessIdentityMismatch) as exc:
        issues.append(f"exact lifecycle snapshot failed: {exc}")
        master_tree = None
        terminal = False
        lanes = []
    master_live = bool(master_tree and master_tree.members)
    master_roots = (
        [] if master_tree is None else [item.pid for item in master_tree.roots]
    )
    identity_payload = record.get("identity")
    if master_live:
        if len(master_roots) != 1:
            issues.append("master lifecycle does not have exactly one root")
        if marker_pid not in master_roots:
            issues.append("master PID projection disagrees with exact lifecycle root")
        if isinstance(identity_payload, Mapping):
            identity = ProcessIdentity.from_dict(identity_payload)
            if not any(
                _same_process_identity_after_reparenting(identity, item)
                for item in master_tree.members
            ):
                issues.append("recorded master identity is not live")

    startup_grace = float(board.payload["watchdog_startup_grace_seconds"])
    within_startup_grace = max(0.0, now - launched_at) <= startup_grace
    stale_seconds = float(board.payload["stale_seconds"])
    for lane in lanes:
        blockers.extend(f"{lane['name']}:{item}" for item in lane["blockers"])
        if master_live and not within_startup_grace:
            if not lane["process_live"]:
                issues.append(f"{lane['name']}:exact process tree is absent")
            elif lane["process_root_count"] != 1:
                issues.append(f"{lane['name']}:process tree root count is not one")
            if lane["status_age_seconds"] is None:
                issues.append(f"{lane['name']}:wrapper status is absent")
            elif lane["status_age_seconds"] > stale_seconds:
                issues.append(f"{lane['name']}:wrapper status is stale")
            if lane["active_progress_hard_timeout_exceeded"]:
                issues.append(
                    f"{lane['name']}:active progress exceeds configured hard timeout"
                )
        if not master_live and lane["process_live"]:
            issues.append(f"{lane['name']}:orphaned process tree")

    terminal_receipt = _read_runtime_json(board.terminal_path)
    terminal_receipt_valid = bool(
        terminal_receipt.get("schema") == TERMINAL_RECEIPT_SCHEMA
        and terminal_receipt.get("run_id") == profile.run_id
        and terminal_receipt.get("profile_id") == profile.profile_id
        and terminal_receipt.get("configuration_root") == profile.configuration_root
        and terminal_receipt.get("drained") is True
    )
    stop_receipt = _read_runtime_json(board.stop_path)
    stop_receipt_valid = bool(
        stop_receipt.get("schema") == STOP_RECEIPT_SCHEMA
        and stop_receipt.get("run_id") == profile.run_id
        and stop_receipt.get("profile_id") == profile.profile_id
        and stop_receipt.get("fenced") is True
    )
    any_lane_live = any(lane["process_live"] for lane in lanes)
    if terminal_receipt_valid and terminal and not master_live and not any_lane_live:
        lifecycle = "completed"
    elif stop_receipt_valid and not master_live and not any_lane_live:
        lifecycle = "stopped"
    elif issues:
        lifecycle = "unhealthy"
    elif blockers:
        lifecycle = "blocked"
    elif master_live:
        lifecycle = "running"
    else:
        lifecycle = "unhealthy"
        issues.append("master exited without terminal or stop evidence")
    return {
        "schema": STATUS_SCHEMA,
        "observed_at": observed_at,
        "lifecycle": lifecycle,
        "healthy": lifecycle in {"running", "completed", "stopped"},
        "run_id": profile.run_id,
        "profile_id": profile.profile_id,
        "configuration_root": profile.configuration_root,
        "launched_at": record.get("launched_at"),
        "master_pid": marker_pid or None,
        "master_root_pids": master_roots,
        "master_live": master_live,
        "within_startup_grace": within_startup_grace,
        "expected_task_count": int(plan.get("expected_task_count") or 0),
        "terminal_lane_count": sum(1 for lane in lanes if lane["terminal"]),
        "terminal_receipt_valid": terminal_receipt_valid,
        "stop_receipt_valid": stop_receipt_valid,
        "issues": issues,
        "blockers": blockers,
        "lanes": lanes,
    }


def _stop_exact(
    board: IVPBoard,
    *,
    grace_seconds: float = 180.0,
    adapter: LinuxProcessAdapter | None = None,
) -> dict[str, Any]:
    if grace_seconds <= 0:
        raise IVPSchedulerError("stop grace must be positive")
    process_adapter = adapter or LinuxProcessAdapter()
    record = _load_lifecycle_record(board)
    if not record:
        pid = _master_pid(board)
        if _pid_alive(pid):
            raise IVPSchedulerError("live master PID has no exact lifecycle record")
        return {"stopped": False, "fenced": True, "reason": "not_started"}
    profile = LifecycleProfile.from_dict(record["profile"])
    profiles = _lane_profiles(board, record["plan"])
    master_tree = process_adapter.snapshot(profile)
    observed_pids = [item.pid for item in master_tree.members]
    deadline = time.monotonic() + grace_seconds
    if master_tree.members:
        process_adapter.terminate(
            master_tree,
            grace_seconds=max(0.1, grace_seconds * 0.67),
            deadline_ms=max(1, int(grace_seconds * 1000)),
        )
    remaining_trees: list[str] = []
    for _track, lane_profile in profiles:
        tree = process_adapter.snapshot(lane_profile)
        observed_pids.extend(item.pid for item in tree.members)
        if tree.members:
            remaining = max(0.1, deadline - time.monotonic())
            process_adapter.terminate(
                tree,
                grace_seconds=min(30.0, remaining),
                deadline_ms=max(1, int(remaining * 1000)),
            )
        if process_adapter.snapshot(lane_profile).members:
            remaining_trees.append(lane_profile.target_id)
    if process_adapter.snapshot(profile).members:
        remaining_trees.append(profile.target_id)
    projections_clean = _remove_pid_projection_if_dead(
        board.master_pid_path,
        expected_pid=(master_tree.roots[0].pid if len(master_tree.roots) == 1 else 0),
    )
    for track, _profile in profiles:
        projections_clean = (
            _remove_pid_projection_if_dead(track.supervisor_pid_path)
            and _remove_pid_projection_if_dead(track.daemon_pid_path)
            and projections_clean
        )
    fenced = not remaining_trees and projections_clean
    receipt = {
        "schema": STOP_RECEIPT_SCHEMA,
        "stopped_at": _utc_now(),
        "run_id": profile.run_id,
        "profile_id": profile.profile_id,
        "configuration_root": profile.configuration_root,
        "fenced": fenced,
        "observed_pids": sorted(set(observed_pids)),
        "remaining_targets": remaining_trees,
    }
    _atomic_json(board.stop_path, receipt)
    if not fenced:
        raise ProcessTreeNotFenced(
            "IVP stop did not fence every exact process tree and PID projection"
        )
    return {
        "stopped": bool(observed_pids),
        "fenced": True,
        "reason": "exact_lifecycle_fenced",
        "pids": sorted(set(observed_pids)),
    }


def stop(
    board: IVPBoard,
    *,
    grace_seconds: float = 180.0,
    adapter: LinuxProcessAdapter | None = None,
) -> dict[str, Any]:
    if not board.state_dir.exists():
        return {"stopped": False, "fenced": True, "reason": "not_started"}
    lock_fd = os.open(
        board.lifecycle_lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise IVPSchedulerError(
                "another IVP launch/stop operation is active"
            ) from exc
        return _stop_exact(
            board,
            grace_seconds=grace_seconds,
            adapter=adapter,
        )
    finally:
        os.close(lock_fd)


def _run_master(
    board: IVPBoard,
    *,
    duration_seconds: float,
    stamp: str,
    expected_task_count: int,
    expected_configuration_root: str,
) -> int:
    """Run the ordinary multi-supervisor with an IVP terminal watcher."""

    if (
        os.environ.get(TARGET_ID_ENV) != MASTER_TARGET_ID
        or os.environ.get(CONFIGURATION_ROOT_ENV) != expected_configuration_root
        or not os.environ.get(RUN_ID_ENV)
        or not os.environ.get(PROFILE_ID_ENV)
    ):
        raise IVPSchedulerError("master entry lacks exact lifecycle authority")
    plan = launch_plan(
        board,
        implement=True,
        foreground=False,
        duration_seconds=duration_seconds,
        stamp=stamp,
        expected_task_count=expected_task_count,
    )
    if (
        plan["configuration_root"] != expected_configuration_root
        or _plan_configuration_root(plan) != expected_configuration_root
    ):
        raise IVPSchedulerError("master launch plan changed after admission")
    record = _load_lifecycle_record(board)
    profile = LifecycleProfile.from_dict(record["profile"])
    if (
        profile.profile_id != os.environ.get(PROFILE_ID_ENV)
        or profile.run_id != os.environ.get(RUN_ID_ENV)
        or profile.configuration_root != expected_configuration_root
    ):
        raise IVPSchedulerError("master lifecycle record changed after admission")
    launched_at = _parse_time(record.get("launched_at")) or time.time()
    process_adapter = LinuxProcessAdapter()
    watcher_stop = threading.Event()
    triggered: dict[str, Any] = {}

    def watch_terminal() -> None:
        poll = float(board.payload.get("poll_interval_seconds") or 5.0)
        while not watcher_stop.wait(max(0.25, poll)):
            try:
                terminal, lanes = _terminal_evidence(
                    board,
                    plan,
                    adapter=process_adapter,
                    launched_at=launched_at,
                    require_live_lanes=True,
                )
            except (IVPSchedulerError, OSError, ValueError, ProcessIdentityMismatch):
                continue
            if not terminal:
                continue
            triggered["lanes"] = lanes
            candidate = {
                "schema": TERMINAL_RECEIPT_SCHEMA,
                "observed_at": _utc_now(),
                "run_id": profile.run_id,
                "profile_id": profile.profile_id,
                "configuration_root": profile.configuration_root,
                "expected_task_count": expected_task_count,
                "drained": False,
                "lane_evidence": lanes,
            }
            _atomic_json(board.terminal_path, candidate)
            os.kill(os.getpid(), signal.SIGTERM)
            return

    watcher = threading.Thread(
        target=watch_terminal,
        name="ivp-terminal-watcher",
        daemon=True,
    )
    watcher.start()
    from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
        main as runner_main,
    )

    returncode = int(runner_main(list(plan["argv"])))
    watcher_stop.set()
    watcher.join(timeout=2.0)
    if _master_pid(board) == os.getpid():
        try:
            board.master_pid_path.unlink()
        except FileNotFoundError:
            pass
    if triggered:
        terminal, lanes = _terminal_evidence(
            board,
            plan,
            adapter=process_adapter,
            launched_at=launched_at,
            require_live_lanes=False,
        )
        lane_trees_empty = all(
            not process_adapter.snapshot(lane_profile).members
            for _track, lane_profile in _lane_profiles(board, plan)
        )
        projections_clean = all(
            _remove_pid_projection_if_dead(track.supervisor_pid_path)
            and _remove_pid_projection_if_dead(track.daemon_pid_path)
            for track, _profile in _lane_profiles(board, plan)
        )
        drained = bool(
            terminal
            and lane_trees_empty
            and projections_clean
            and not board.master_pid_path.exists()
        )
        _atomic_json(
            board.terminal_path,
            {
                "schema": TERMINAL_RECEIPT_SCHEMA,
                "observed_at": _utc_now(),
                "run_id": profile.run_id,
                "profile_id": profile.profile_id,
                "configuration_root": profile.configuration_root,
                "expected_task_count": expected_task_count,
                "drained": drained,
                "lane_evidence": lanes,
            },
        )
        if not drained:
            return 2
    return returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the PR-176-compatible IVP supervisor"
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/agent_supervisor_incremental_verification_planner_scheduler.json"
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    launch = commands.add_parser("launch")
    launch.add_argument("--implement", action="store_true")
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--foreground", action="store_true")
    launch.add_argument("--duration-seconds", type=float, default=28_800.0)
    commands.add_parser("status")
    stop_parser = commands.add_parser("stop")
    stop_parser.add_argument("--grace-seconds", type=float, default=180.0)
    hidden = commands.add_parser("_run-master", help=argparse.SUPPRESS)
    hidden.add_argument("--duration-seconds", type=float, required=True)
    hidden.add_argument("--stamp", required=True)
    hidden.add_argument("--expected-task-count", type=int, required=True)
    hidden.add_argument("--expected-configuration-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        board = load_board(args.config, repo_root=args.repo_root)
        if args.command == "_run-master":
            return _run_master(
                board,
                duration_seconds=float(args.duration_seconds),
                stamp=str(args.stamp),
                expected_task_count=int(args.expected_task_count),
                expected_configuration_root=str(args.expected_configuration_root),
            )
        if args.command == "preflight":
            report = preflight(board)
            print(json.dumps(report, sort_keys=True, separators=(",", ":")))
            return 0 if report["valid"] else 2
        if args.command == "status":
            report = status(board)
            print(json.dumps(report, sort_keys=True, separators=(",", ":")))
            return (
                0
                if report["lifecycle"]
                in {
                    "not_started",
                    "running",
                    "completed",
                    "stopped",
                }
                else 2
            )
        if args.command == "stop":
            print(
                json.dumps(
                    stop(board, grace_seconds=float(args.grace_seconds)),
                    sort_keys=True,
                )
            )
            return 0

        report = preflight(board)
        if not report["valid"]:
            print(json.dumps(report, sort_keys=True, separators=(",", ":")))
            return 2
        plan = launch_plan(
            board,
            implement=bool(args.implement),
            foreground=bool(args.foreground),
            duration_seconds=float(args.duration_seconds),
            expected_task_count=int(report["validator_report"].get("task_count") or 0),
        )
        print(json.dumps(plan, sort_keys=True, separators=(",", ":")))
        if args.dry_run:
            return 0
        return run_launch(board, plan)
    except (
        IVPSchedulerError,
        OSError,
        ProcessIdentityMismatch,
        ProcessTreeNotFenced,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        print(
            json.dumps({"error": str(exc), "type": type(exc).__name__}, sort_keys=True),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
