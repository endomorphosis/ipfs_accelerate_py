"""Start a resident supervisor and the workers for a prompt-admitted graph.

``supervisor run`` detaches one supervisor process under the state root,
outside the repository. That supervisor stays up and starts worker
processes for the ready tasks, at most ``width`` at a time. A product
worker is ``python -m ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner``
with the admitted prompt on stdin. That process is the worker the
supervisor counts. Pytest local-only, and an explicit executable, still
use a ``prompt_workers --worker`` process and do not launch the runner.
Validation commands stay recorded and unexecuted. Workers do not run git
and do not grant execution or completion authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

DEFAULT_PROMPT_WORKER_WIDTH = 4
_MODULE = "ipfs_accelerate_py.agent_supervisor.prompt.prompt_workers"
_RUNNER_MODULE = "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner"
_ENV_KEEP = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONNOUSERSITE",
        "VIRTUAL_ENV",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    }
)
_SAFE_NAME = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
_TRACKED: list["PromptWorkerLaunch"] = []
_STOP_REQUESTED = False
_AGENT_PID = 0
_SUPERVISOR_WAIT_SECONDS = 30.0
_AGENT_MAX_TURNS = 8
_DENY_PATH_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._-"
)


class PromptWorkerError(RuntimeError):
    """A prompt supervisor or worker could not be started or proven alive."""


@dataclass(frozen=True)
class PromptWorkerLaunch:
    """One resident supervisor and the workers it has started."""

    run_dir: str
    pids: tuple[int, ...]
    task_cids: tuple[str, ...]
    task_keys: tuple[str, ...]
    supervisor_pid: int = 0

    @property
    def active_worker_count(self) -> int:
        return len(self.pids)


def launch_prompt_workers(
    *,
    tasks: Sequence[Any],
    admitted_task_cids: Sequence[str],
    run_id: str,
    state_root: Path | str,
    repository_root: Path | str,
    width: int = DEFAULT_PROMPT_WORKER_WIDTH,
    popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    local_only: bool | None = None,
    agent_executable: str | None = None,
    prompt_text: str = "",
) -> PromptWorkerLaunch:
    """Start a resident supervisor. Return once it and its workers are alive.

    The call does not wait for the queue to finish. A second call for the
    same run reuses the live supervisor and the workers it already started.
    """

    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise PromptWorkerError("prompt worker width must be a positive integer")
    if not str(run_id) or any(char not in _SAFE_NAME for char in str(run_id)):
        raise PromptWorkerError("run id is not a safe worker directory name")
    state = Path(state_root).resolve()
    repository = Path(repository_root).resolve()
    run_dir = (state / "prompt-runs" / str(run_id)).resolve()
    if not _is_within(run_dir, state):
        raise PromptWorkerError("prompt worker directory escapes the state root")
    if _is_within(run_dir, repository) or _is_within(repository, run_dir):
        raise PromptWorkerError(
            "prompt workers refuse a run directory inside the repository"
        )
    ready = _ready_tasks(tasks, admitted_task_cids)
    if not ready:
        raise PromptWorkerError("prompt admitted no ready tasks to run")
    checked_agent = ""
    if agent_executable is not None:
        checked_agent = _checked_agent_executable(agent_executable, repository)
        if not checked_agent:
            raise PromptWorkerError(
                "prompt worker agent executable is not a file outside the repository"
            )
    if local_only is None:
        local_only = bool(os.environ.get("PYTEST_CURRENT_TEST")) and not checked_agent
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("workers", "logs", "queue", "claims", "workspaces", "agents", "prompts"):
        (run_dir / name).mkdir(exist_ok=True)
    supervisor_pid = _live_supervisor_pid(run_dir)
    if supervisor_pid is None:
        stop_path = run_dir / "stop"
        if stop_path.exists():
            stop_path.unlink()
        wave = run_dir / "wave"
        if wave.exists():
            wave.unlink()
        _requeue_unfinished(run_dir)
        for task in ready:
            _write_queue_packet(
                task,
                run_dir,
                repository,
                local_only=bool(local_only),
                agent_executable=checked_agent,
                prompt_text=prompt_text,
            )
        supervisor_pid = _spawn_supervisor(
            run_dir, repository, width, popen
        )
    try:
        launch = _wait_until_supervising(supervisor_pid, run_dir)
    except Exception:
        stop_prompt_workers(run_dir, (supervisor_pid,))
        raise
    _TRACKED.append(launch)
    return launch


def stop_tracked_prompt_workers() -> None:
    """Stop supervisors and workers started in this process."""

    pending = list(_TRACKED)
    _TRACKED.clear()
    for launch in pending:
        stop_prompt_workers(
            launch.run_dir,
            (launch.supervisor_pid, *launch.pids),
        )


def stop_prompt_workers(run_dir: Path | str, pids: Sequence[int]) -> None:
    """Ask one run to exit, then signal only processes bound to that run."""

    root = Path(run_dir)
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / "stop").write_text("stop\n", encoding="utf-8")
    except OSError:
        return
    _kill_recorded_agents(root)
    targets = []
    seen: set[int] = set()
    for pid in list(pids) + _pids_for_run(root):
        if isinstance(pid, bool) or not isinstance(pid, int) or pid in seen:
            continue
        if _belongs_to_run(pid, root):
            seen.add(pid)
            targets.append(pid)
    deadline = time.monotonic() + 2.0
    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
    for pid in targets:
        while time.monotonic() < deadline and _belongs_to_run(pid, root):
            time.sleep(0.05)
        if _belongs_to_run(pid, root):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                continue
    _kill_recorded_agents(root)
    for pid in targets:
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            continue


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the resident supervisor or one worker process."""

    args = list(sys.argv[1:] if argv is None else argv)
    if "--supervisor" in args:
        return _supervisor_main(args)
    return _worker_main(args)


def _supervisor_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog=_MODULE)
    parser.add_argument("--supervisor", action="store_true")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--width", type=int, required=True)
    args = parser.parse_args(list(argv))
    run_dir = Path(args.run_dir).resolve()
    repository = Path(args.repository_root).resolve()
    if not run_dir.is_dir() or (run_dir / ".git").exists():
        return 2
    if _is_within(run_dir, repository) or _is_within(repository, run_dir):
        return 2
    if isinstance(args.width, bool) or args.width < 1:
        return 2
    _install_stop_handlers()
    launched: dict[str, int] = {}
    attempts: dict[str, int] = {}
    _publish_supervisor(run_dir, (), (), ())
    first_wave = True
    while not _stopped(run_dir):
        _reconcile(run_dir, repository, int(args.width), launched, attempts)
        if first_wave:
            (run_dir / "wave").write_text("open\n", encoding="utf-8")
            first_wave = False
        time.sleep(0.1)
    for task_cid, pid in list(launched.items()):
        _kill_process_group(pid)
        _finish_direct_runner(run_dir, repository, task_cid, pid, _reap(pid))
    _kill_recorded_agents(run_dir)
    _publish_supervisor(run_dir, (), (), (), phase="stopped")
    return 0


def _worker_main(argv: Sequence[str]) -> int:
    """Prepare one admitted task, then keep taking admitted tasks."""

    parser = argparse.ArgumentParser(prog=_MODULE)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--task-cid", required=True)
    parser.add_argument("--task-key", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args(list(argv))
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir() or (run_dir / ".git").exists():
        return 2
    task_cid = str(args.task_cid)
    if not task_cid or any(char not in _SAFE_NAME for char in task_cid):
        return 2
    repository = Path(args.repository_root).resolve()
    if _is_within(run_dir, repository) or _is_within(repository, run_dir):
        return 2
    _install_stop_handlers()
    first = _load_packet(run_dir, task_cid)
    if first is None or str(first.get("task_cid") or "") != task_cid:
        return 2
    if first.get("authorizes_execution") is True or first.get("completion_authority") is True:
        return 2
    launched = first.get("launched_argv")
    if not isinstance(launched, list) or _MODULE not in launched:
        return 2
    if not _execute_and_publish(
        repository,
        run_dir,
        task_cid,
        str(args.task_key),
        first,
        local_only=bool(args.local_only),
    ):
        return 2
    while not _stopped(run_dir) and not (run_dir / "wave").exists():
        time.sleep(0.05)
    while not _stopped(run_dir):
        claimed = _claim_next(run_dir)
        if claimed is None:
            time.sleep(0.1)
            continue
        cid, packet = claimed
        if packet.get("authorizes_execution") is True or packet.get("completion_authority") is True:
            return 2
        command = packet.get("launched_argv")
        if not isinstance(command, list) or _MODULE not in command:
            return 2
        if not _execute_and_publish(
            repository,
            run_dir,
            cid,
            str(packet.get("task_key") or ""),
            packet,
            local_only=_packet_local_only(packet, bool(args.local_only)),
        ):
            return 2
    return 0


def _execute_and_publish(
    repository: Path,
    run_dir: Path,
    task_cid: str,
    task_key: str,
    packet: Mapping[str, Any],
    *,
    local_only: bool,
) -> bool:
    work = _prepare_workspace(repository, run_dir, task_cid, packet)
    launched = packet.get("launched_argv")
    payload = {
        "schema": "ipfs_accelerate_py.agent_supervisor.prompt-worker@1",
        "argv": [str(item) for item in launched] if isinstance(launched, list) else [],
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "task_cid": task_cid,
        "task_key": task_key,
        "cwd": str(run_dir),
        "phase": "running",
        "repository_edits": False,
        "authorizes_execution": False,
        "completion_authority": False,
        "work": work,
    }
    _atomic_json(run_dir / "workers" / f"{task_cid}.json", payload)
    print(
        f"running task {task_key} phase=running "
        f"files_prepared={work['files_prepared']} validation_executed=false",
        flush=True,
    )
    if local_only or _stopped(run_dir):
        return True
    agent = _run_workspace_agent(repository, run_dir, task_cid, task_key, packet, work)
    work["disposition"] = agent["disposition"]
    work["agent_returncode"] = agent["returncode"]
    work["repository_edits"] = agent["repository_edits"]
    payload["work"] = work
    payload["repository_edits"] = agent["repository_edits"]
    _atomic_json(run_dir / "workers" / f"{task_cid}.json", payload)
    return True


def _prepare_workspace(
    repository: Path,
    run_dir: Path,
    task_cid: str,
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy admitted scope files into this run's workspace. Do not touch the repo."""

    workspace = (run_dir / "workspaces" / task_cid).resolve()
    if not _is_within(workspace, run_dir):
        raise OSError("workspace escapes the run directory")
    workspace.mkdir(parents=True, exist_ok=True)
    scope: list[dict[str, Any]] = []
    prepared = 0
    if repository.is_dir():
        repository_status = "available"
        for relative in _packet_paths(packet):
            observed = _observe_scope(repository, relative)
            scope.append(observed)
            if observed.get("status") != "observed":
                continue
            target = workspace.joinpath(*PurePosixPath(relative).parts)
            if not _is_within(target, workspace):
                observed["status"] = "rejected"
                continue
            source = repository.joinpath(*PurePosixPath(relative).parts)
            try:
                data = source.read_bytes()
            except OSError:
                observed["status"] = "unreadable"
                continue
            if hashlib.sha256(data).hexdigest() != observed.get("sha256"):
                observed["status"] = "changed"
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            prepared += 1
    else:
        repository_status = "unavailable"
    execution = {
        "schema": "ipfs_accelerate_py.agent_supervisor.prompt-worker-execution@1",
        "task_cid": task_cid,
        "disposition": "prepared",
        "files_prepared": prepared,
        "repository_edits": False,
        "validation_executed": False,
        "authorizes_execution": False,
        "completion_authority": False,
    }
    _atomic_json(workspace / "execution.json", execution)
    return {
        "schema": "ipfs_accelerate_py.agent_supervisor.prompt-worker-run@1",
        "phase": "running",
        "repository_status": repository_status,
        "scope": scope,
        "workspace": str(workspace),
        "files_prepared": prepared,
        "disposition": "prepared",
        "validation_executed": False,
        "validation_skip_reason": "execution_not_authorized",
        "validation_argv": _packet_argv(packet),
        "repository_edits": False,
        "authorizes_execution": False,
        "completion_authority": False,
    }


def _reconcile(
    run_dir: Path,
    repository: Path,
    width: int,
    launched: dict[str, int],
    attempts: dict[str, int],
) -> None:
    if _stopped(run_dir):
        return
    for task_cid, pid in list(launched.items()):
        returncode = _reap(pid)
        if returncode is None and _process_alive(pid):
            continue
        _finish_direct_runner(
            run_dir, repository, task_cid, pid, returncode
        )
        launched.pop(task_cid, None)
        attempts[task_cid] = attempts.get(task_cid, 0) + 1
    pool = _live_pool(run_dir)
    live_pids = {pid for pid, _cid, _key in pool}
    starting = [
        pid
        for pid in launched.values()
        if pid not in live_pids and _process_alive(pid)
    ]
    occupied = len(live_pids) + len(starting)
    for task_cid, packet in _queued_packets(run_dir):
        if occupied >= width or _stopped(run_dir):
            break
        if attempts.get(task_cid, 0) >= 2:
            continue
        if not _claim_packet(run_dir, task_cid):
            continue
        try:
            pid = _spawn_worker(packet, repository, run_dir)
        except OSError as exc:
            _publish_supervisor(
                run_dir,
                tuple(row[0] for row in pool),
                tuple(row[1] for row in pool),
                tuple(row[2] for row in pool),
                last_error=f"prompt worker failed to start: {exc}",
            )
            return
        launched[task_cid] = pid
        occupied += 1
    pool = _live_pool(run_dir)
    last_error = ""
    if any(count >= 2 for count in attempts.values()) and not pool:
        last_error = "prompt worker failed twice for one admitted task"
    _publish_supervisor(
        run_dir,
        tuple(row[0] for row in pool),
        tuple(row[1] for row in pool),
        tuple(row[2] for row in pool),
        last_error=last_error,
    )


def _spawn_supervisor(
    run_dir: Path,
    repository: Path,
    width: int,
    popen: Callable[..., subprocess.Popen[Any]],
) -> int:
    command = [
        sys.executable,
        "-m",
        _MODULE,
        "--supervisor",
        "--run-dir",
        str(run_dir),
        "--repository-root",
        str(repository),
        "--width",
        str(width),
    ]
    log_handle = (run_dir / "logs" / "supervisor.log").open("ab")
    env = {key: value for key, value in os.environ.items() if key in _ENV_KEEP}
    try:
        process = popen(
            command,
            cwd=str(run_dir),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        log_handle.close()
        raise PromptWorkerError(f"prompt supervisor failed to start: {exc}") from exc
    log_handle.close()
    return int(process.pid)


def _spawn_worker(
    packet: Mapping[str, Any], repository: Path, run_dir: Path
) -> int:
    command = packet.get("launched_argv")
    if not isinstance(command, list) or not (
        _MODULE in command or _RUNNER_MODULE in command
    ):
        raise OSError("task packet has no worker command")
    argv = [str(item) for item in command]
    task_cid = str(packet.get("task_cid") or "")
    task_key = str(packet.get("task_key") or "")
    log_handle = (run_dir / "logs" / f"{task_cid}.log").open("ab")
    if _RUNNER_MODULE in argv:
        return _spawn_runner(
            argv,
            packet,
            repository,
            run_dir,
            task_cid,
            task_key,
            log_handle,
        )
    env = {key: value for key, value in os.environ.items() if key in _ENV_KEEP}
    env["PROMPT_WORKER"] = "1"
    env["PROMPT_WORKER_TASK_CID"] = task_cid
    try:
        process = subprocess.Popen(
            argv,
            cwd=str(run_dir),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    return int(process.pid)


def _spawn_runner(
    argv: list[str],
    packet: Mapping[str, Any],
    repository: Path,
    run_dir: Path,
    task_cid: str,
    task_key: str,
    log_handle: Any,
) -> int:
    """Start the package runner as the worker process for one admitted task."""

    workspace_flag = argv[argv.index("--workspace") + 1]
    if workspace_flag != str(repository):
        log_handle.close()
        raise OSError("runner workspace is not the admitted repository")
    prompt_dir = (run_dir / "prompts" / task_cid).resolve()
    if not _is_within(prompt_dir, run_dir) or _is_within(prompt_dir, repository):
        log_handle.close()
        raise OSError("runner prompt directory is not inside the run")
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / "prompt.txt"
    prompt_path.write_text(_agent_prompt(packet, repository), encoding="utf-8")
    prompt_bytes = prompt_path.read_bytes()
    if not prompt_bytes.strip():
        log_handle.close()
        raise OSError("runner prompt is empty")
    work = _prepare_workspace(repository, run_dir, task_cid, packet)
    process: subprocess.Popen[Any] | None = None
    try:
        process = subprocess.Popen(
            argv,
            cwd=str(repository),
            env=_agent_env(),
            stdin=subprocess.PIPE,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if process.stdin is not None:
            process.stdin.write(prompt_bytes)
            process.stdin.close()
    except OSError:
        log_handle.close()
        if process is not None and process.pid:
            _kill_process_group(int(process.pid))
        raise
    log_handle.close()
    pid = int(process.pid)
    _atomic_json(
        run_dir / "workers" / f"{task_cid}.json",
        {
            "schema": "ipfs_accelerate_py.agent_supervisor.prompt-worker@1",
            "argv": argv,
            "pid": pid,
            "ppid": os.getpid(),
            "task_cid": task_cid,
            "task_key": task_key,
            "cwd": str(repository),
            "phase": "running",
            "repository_edits": False,
            "authorizes_execution": False,
            "completion_authority": False,
            "work": work,
        },
    )
    _atomic_json(
        run_dir / "agents" / f"{task_cid}.json",
        {
            "pid": pid,
            "cwd": str(repository),
            "task_cid": task_cid,
            "task_key": task_key,
        },
    )
    return pid


def _finish_direct_runner(
    run_dir: Path,
    repository: Path,
    task_cid: str,
    pid: int,
    returncode: int | None = None,
) -> None:
    """Record a package-runner worker after its process has exited."""

    receipt = _read_receipt(run_dir / "workers" / f"{task_cid}.json") or {}
    argv = receipt.get("argv")
    if not isinstance(argv, list) or _RUNNER_MODULE not in argv:
        return
    if receipt.get("pid") != pid:
        return
    work = receipt.get("work")
    if not isinstance(work, dict):
        work = {}
    edited = _repository_scope_changed(repository, work)
    if returncode == 0:
        disposition = "agent_finished"
    elif returncode is None:
        disposition = "agent_stopped"
    else:
        disposition = "agent_failed"
    workspace = Path(str(work.get("workspace") or (run_dir / "workspaces" / task_cid)))
    _write_agent_execution(workspace, task_cid, disposition, returncode, edited)
    finished = dict(work)
    finished["disposition"] = disposition
    finished["agent_returncode"] = returncode
    finished["repository_edits"] = edited
    payload = dict(receipt)
    payload["work"] = finished
    payload["repository_edits"] = edited
    _atomic_json(run_dir / "workers" / f"{task_cid}.json", payload)
    agent_record = run_dir / "agents" / f"{task_cid}.json"
    if agent_record.exists():
        agent_record.unlink()


def _reap(pid: int) -> int | None:
    try:
        waited, status = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return None
    if waited != pid:
        return None
    if os.WIFEXITED(status):
        return int(os.WEXITSTATUS(status))
    if os.WIFSIGNALED(status):
        return 128 + int(os.WTERMSIG(status))
    return None


def _wait_until_supervising(supervisor_pid: int, run_dir: Path) -> PromptWorkerLaunch:
    deadline = time.monotonic() + _SUPERVISOR_WAIT_SECONDS
    while time.monotonic() < deadline:
        if not _belongs_to_run(supervisor_pid, run_dir):
            tail = _log_tail_path(run_dir / "logs" / "supervisor.log")
            detail = f": {tail}" if tail else ""
            raise PromptWorkerError(
                f"prompt supervisor exited before workers were running{detail}"
            )
        state = _read_receipt(run_dir / "supervisor.json") or {}
        if int(state.get("pid") or 0) != supervisor_pid:
            time.sleep(0.05)
            continue
        pids = _int_list(state.get("worker_pids"))
        cids = _str_list(state.get("worker_task_cids"))
        keys = _str_list(state.get("worker_task_keys"))
        if pids and len(pids) == len(cids) == len(keys) and _workers_running(
            run_dir, pids, cids
        ):
            return PromptWorkerLaunch(
                run_dir=str(run_dir),
                pids=tuple(pids),
                task_cids=tuple(cids),
                task_keys=tuple(keys),
                supervisor_pid=supervisor_pid,
            )
        if state.get("last_error") and not pids:
            raise PromptWorkerError(str(state["last_error"]))
        time.sleep(0.05)
    tail = _log_tail_path(run_dir / "logs" / "supervisor.log")
    detail = f": {tail}" if tail else ""
    raise PromptWorkerError(f"prompt supervisor did not start workers{detail}")


def _workers_running(run_dir: Path, pids: Sequence[int], cids: Sequence[str]) -> bool:
    for pid, task_cid in zip(pids, cids, strict=True):
        receipt = _read_receipt(run_dir / "workers" / f"{task_cid}.json")
        if not _receipt_running(receipt, pid, task_cid):
            return False
        argv = receipt.get("argv") if isinstance(receipt, Mapping) else None
        if isinstance(argv, list) and _RUNNER_MODULE in argv:
            if not _runner_matches(pid, receipt):
                return False
            continue
        if not _worker_matches(pid, task_cid, run_dir):
            return False
    return True


def _receipt_running(
    receipt: Mapping[str, Any] | None, pid: int, task_cid: str
) -> bool:
    if not receipt or str(receipt.get("task_cid") or "") != task_cid:
        return False
    if receipt.get("phase") != "running":
        return False
    if receipt.get("pid") != pid:
        return False
    if receipt.get("authorizes_execution") is True or receipt.get(
        "completion_authority"
    ) is True:
        return False
    work = receipt.get("work")
    if not isinstance(work, dict) or work.get("validation_executed") is not False:
        return False
    if work.get("repository_edits") not in (False, True):
        return False
    argv = receipt.get("argv")
    if not isinstance(argv, list) or "pytest" in argv or "git" in argv:
        return False
    if _RUNNER_MODULE in argv:
        return "--workspace" in argv and _arg(argv, "--workspace") != ""
    return _MODULE in argv and task_cid in argv


def _publish_supervisor(
    run_dir: Path,
    pids: Sequence[int],
    task_cids: Sequence[str],
    task_keys: Sequence[str],
    *,
    phase: str = "running",
    last_error: str = "",
) -> None:
    payload: dict[str, Any] = {
        "schema": "ipfs_accelerate_py.agent_supervisor.prompt-supervisor@1",
        "pid": os.getpid(),
        "phase": phase,
        "run_dir": str(run_dir),
        "worker_pids": list(pids),
        "worker_task_cids": list(task_cids),
        "worker_task_keys": list(task_keys),
        "authorizes_execution": False,
        "completion_authority": False,
    }
    if last_error:
        payload["last_error"] = last_error
    _atomic_json(run_dir / "supervisor.json", payload)


def _write_queue_packet(
    task: Any,
    run_dir: Path,
    repository: Path,
    *,
    local_only: bool,
    agent_executable: str,
    prompt_text: str = "",
) -> None:
    task_cid = _task_cid(task)
    task_key = str(getattr(task, "task_key", "") or "")
    if not task_cid or any(char not in _SAFE_NAME for char in task_cid):
        raise PromptWorkerError("admitted task cid is not a safe worker name")
    if "\0" in task_key or "\n" in task_key:
        raise PromptWorkerError("task key is not a safe worker argument")
    if local_only or agent_executable:
        command = [
            sys.executable,
            "-m",
            _MODULE,
            "--worker",
            "--run-dir",
            str(run_dir),
            "--task-cid",
            task_cid,
            "--task-key",
            task_key,
            "--repository-root",
            str(repository),
        ]
        if local_only:
            command.append("--local-only")
    else:
        command = _build_isolated_runner_command(repository)
    destination = run_dir / "queue" / f"{task_cid}.json"
    claim = run_dir / "claims" / f"{task_cid}.json"
    if destination.exists() or claim.exists():
        return
    _atomic_json(
        destination,
        {
            "schema": "ipfs_accelerate_py.agent_supervisor.prompt-worker-task@1",
            "task_cid": task_cid,
            "task_key": task_key,
            "objective": str(getattr(task, "objective", "") or "")[:512],
            "scope_paths": _scope_paths(task),
            "validation_argv": _validation_argv(task),
            "launched_argv": command,
            "agent_executable": agent_executable,
            "prompt_text": _bounded_prompt_text(prompt_text),
            "authorizes_execution": False,
            "completion_authority": False,
        },
    )


def _queued_packets(run_dir: Path) -> list[tuple[str, Mapping[str, Any]]]:
    found: list[tuple[str, Mapping[str, Any]]] = []
    queue = run_dir / "queue"
    if not queue.is_dir():
        return found
    for path in sorted(queue.glob("*.json")):
        packet = _read_receipt(path)
        if packet is None:
            continue
        task_cid = str(packet.get("task_cid") or "")
        if task_cid and task_cid == path.stem:
            found.append((task_cid, packet))
    return found


def _claim_packet(run_dir: Path, task_cid: str) -> bool:
    source = run_dir / "queue" / f"{task_cid}.json"
    target = run_dir / "claims" / f"{task_cid}.json"
    try:
        os.rename(source, target)
    except OSError:
        return False
    return True


def _claim_next(run_dir: Path) -> tuple[str, Mapping[str, Any]] | None:
    for task_cid, _packet in _queued_packets(run_dir):
        if not _claim_packet(run_dir, task_cid):
            continue
        packet = _load_packet(run_dir, task_cid)
        if packet is not None:
            return task_cid, packet
    return None


def _load_packet(run_dir: Path, task_cid: str) -> Mapping[str, Any] | None:
    for folder in ("claims", "queue"):
        packet = _read_receipt(run_dir / folder / f"{task_cid}.json")
        if packet is not None:
            return packet
    return None


def _requeue_unfinished(run_dir: Path) -> None:
    claims = run_dir / "claims"
    if not claims.is_dir():
        return
    for path in list(claims.glob("*.json")):
        task_cid = path.stem
        if _pid_for_task(task_cid, run_dir) is not None:
            continue
        target = run_dir / "queue" / path.name
        if target.exists():
            path.unlink()
        else:
            os.rename(path, target)
        stale = run_dir / "workers" / f"{task_cid}.json"
        if stale.exists():
            stale.unlink()


def _live_pool(run_dir: Path) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for pid in _pids_for_run(run_dir):
        text = _cmdline(pid)
        if text is None or "--worker" not in text:
            continue
        task_cid = _arg(text, "--task-cid")
        if not task_cid or not _worker_matches(pid, task_cid, run_dir):
            continue
        seen.add(task_cid)
        rows.append((pid, task_cid, _arg(text, "--task-key")))
    workers = run_dir / "workers"
    if workers.is_dir():
        for path in sorted(workers.glob("*.json")):
            receipt = _read_receipt(path)
            if not isinstance(receipt, Mapping):
                continue
            task_cid = str(receipt.get("task_cid") or "")
            if not task_cid or task_cid in seen or task_cid != path.stem:
                continue
            argv = receipt.get("argv")
            pid = receipt.get("pid")
            if (
                not isinstance(argv, list)
                or _RUNNER_MODULE not in argv
                or isinstance(pid, bool)
                or not isinstance(pid, int)
                or not _runner_matches(pid, receipt)
            ):
                continue
            seen.add(task_cid)
            rows.append((pid, task_cid, str(receipt.get("task_key") or "")))
    rows.sort(key=lambda row: row[1])
    return rows


def _process_alive(pid: int) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _pid_for_task(task_cid: str, run_dir: Path) -> int | None:
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if _worker_matches(pid, task_cid, run_dir):
            return pid
    return None


def live_prompt_supervisors(
    state_root: Path | str, repository: Path | str
) -> list[dict[str, Any]]:
    """Return prompt supervisors that are still alive for this repository."""

    root = Path(state_root) / "prompt-runs"
    if not root.is_dir():
        return []
    try:
        repository_root = Path(repository).resolve()
    except OSError:
        return []
    found: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        resolved = run_dir.resolve()
        state = _read_receipt(resolved / "supervisor.json")
        if not isinstance(state, Mapping) or state.get("phase") != "running":
            continue
        if (
            state.get("authorizes_execution") is True
            or state.get("completion_authority") is True
        ):
            continue
        pid = state.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or not _is_supervisor(
            pid, resolved
        ):
            continue
        recorded = _arg(_cmdline(pid) or [], "--repository-root")
        try:
            if not recorded or Path(recorded).resolve() != repository_root:
                continue
        except OSError:
            continue
        found.append(
            {
                "run_id": resolved.name,
                "run_dir": str(resolved),
                "supervisor_pid": pid,
                "worker_pids": _int_list(state.get("worker_pids")),
                "worker_task_cids": _str_list(state.get("worker_task_cids")),
                "worker_task_keys": _str_list(state.get("worker_task_keys")),
            }
        )
    return found


def _live_supervisor_pid(run_dir: Path) -> int | None:
    state = _read_receipt(run_dir / "supervisor.json") or {}
    pid = state.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int):
        return None
    if _is_supervisor(pid, run_dir):
        return pid
    return None


def _ready_tasks(
    tasks: Sequence[Any], admitted_task_cids: Sequence[str]
) -> list[Any]:
    admitted = {str(item) for item in admitted_task_cids if str(item)}
    if not admitted:
        return []
    ready: list[Any] = []
    seen: set[str] = set()
    for task in tasks:
        cid = _task_cid(task)
        if cid not in admitted or cid in seen:
            continue
        dependencies = {
            str(item)
            for item in (getattr(task, "dependency_task_cids", ()) or ())
            if str(item)
        }
        if dependencies & admitted:
            continue
        seen.add(cid)
        ready.append(task)
    return ready


def _task_cid(task: Any) -> str:
    return str(getattr(task, "task_cid", "") or "")


def _scope_paths(task: Any) -> list[str]:
    raw = getattr(task, "scope_paths", ()) or ()
    paths: list[str] = []
    for item in raw:
        text = str(item)
        if text and "\0" not in text and len(text) <= 1024:
            paths.append(text)
        if len(paths) >= 32:
            break
    return paths


def _validation_argv(task: Any) -> list[list[str]]:
    raw = getattr(task, "validations", ()) or ()
    rows: list[list[str]] = []
    for item in raw:
        argv = getattr(item, "argv", ()) or ()
        if isinstance(argv, (str, bytes)) or not argv:
            continue
        tokens = [str(part) for part in argv]
        if tokens and all("\0" not in part and len(part) <= 256 for part in tokens):
            rows.append(tokens[:32])
        if len(rows) >= 8:
            break
    return rows


def _packet_paths(packet: Mapping[str, Any]) -> list[str]:
    raw = packet.get("scope_paths") or []
    if not isinstance(raw, list):
        return []
    paths: list[str] = []
    for item in raw:
        if isinstance(item, str) and item and "\0" not in item:
            paths.append(item)
        if len(paths) >= 32:
            break
    return paths


def _packet_argv(packet: Mapping[str, Any]) -> list[list[str]]:
    raw = packet.get("validation_argv") or []
    if not isinstance(raw, list):
        return []
    rows: list[list[str]] = []
    for row in raw:
        if not isinstance(row, list) or not row:
            continue
        if all(isinstance(part, str) and "\0" not in part for part in row):
            rows.append([part[:256] for part in row[:32]])
        if len(rows) >= 8:
            break
    return rows


_CREDENTIAL_NAMES = frozenset({".env", ".env.local", "id_rsa", "id_ed25519"})
_CREDENTIAL_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx"})


def _packet_local_only(packet: Mapping[str, Any], fallback: bool) -> bool:
    command = packet.get("launched_argv")
    if isinstance(command, list) and "--local-only" in command:
        return True
    return fallback


def _checked_agent_executable(raw: str, repository: Path) -> str:
    if not raw or "\0" in raw or not raw.startswith("/"):
        return ""
    path = Path(raw)
    if path.name in {"git", "pytest", "sudo"}:
        return ""
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return ""
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        return ""
    if _is_within(resolved, repository) or ".git" in resolved.parts:
        return ""
    return str(resolved)


def _build_workspace_agent_command(
    executable: str,
    repository: Path,
    prompt_file: Path,
) -> list[str]:
    """Headless agent argv. Plan mode blocks edits even when approvals are on."""

    command = [
        executable,
        "--cwd",
        str(repository),
        "--no-plan",
        "--permission-mode",
        "bypassPermissions",
        "--always-approve",
        "--max-turns",
        str(_AGENT_MAX_TURNS),
        "--output-format",
        "plain",
        "--no-subagents",
        "--disable-web-search",
        "--prompt-file",
        str(prompt_file),
    ]
    git_dir = repository / ".git"
    if _deny_path(git_dir):
        text = str(git_dir)
        command.extend(["--deny", f"Edit({text}/**)", "--deny", f"Write({text}/**)"])
    command.extend(
        [
            "--deny",
            "Bash(git)",
            "--deny",
            "Bash(git *)",
            "--deny",
            "Bash(pytest)",
            "--deny",
            "Bash(pytest *)",
        ]
    )
    return command


def _deny_path(path: Path) -> bool:
    text = str(path)
    return bool(text) and all(char in _DENY_PATH_CHARS for char in text)


def _bounded_prompt_text(raw: str) -> str:
    return str(raw or "").replace("\0", "").strip()[:4000]


def _build_isolated_runner_command(repository: Path) -> list[str]:
    """Launch the package runner. It owns the sandbox and permission mode."""

    return [
        sys.executable,
        "-m",
        _RUNNER_MODULE,
        "--workspace",
        str(repository),
        "--mode",
        "agent",
        "--max-turns",
        str(_AGENT_MAX_TURNS),
    ]


def _agent_prompt(packet: Mapping[str, Any], repository: Path) -> str:
    instruction = _bounded_prompt_text(str(packet.get("prompt_text") or ""))
    fallback = str(packet.get("objective") or "").strip()
    objective = instruction or fallback or "Implement the admitted task in this repository."
    paths = ", ".join(_packet_paths(packet)[:32]) or "the files named by the task"
    parts = [objective[:4000]]
    parts.append(
        f"Repository: {repository}\n"
        f"Implement the task in this repository. Files in scope: {paths}. "
        "Do not run git. Do not commit. Do not run pytest.\n"
    )
    return "\n\n".join(parts)


def _agent_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key in _ENV_KEEP}
    env.pop("GROK_AGENT", None)
    env.pop("GROK_SESSION_ID", None)
    return env


def _resolve_agent_executable(
    packet: Mapping[str, Any], repository: Path
) -> str:
    """Return an explicitly configured executable, never an implicit agent."""

    configured = packet.get("agent_executable")
    if isinstance(configured, str) and configured:
        return _checked_agent_executable(configured, repository)
    return ""


def _run_workspace_agent(
    repository: Path,
    run_dir: Path,
    task_cid: str,
    task_key: str,
    packet: Mapping[str, Any],
    work: Mapping[str, Any],
) -> dict[str, Any]:
    global _AGENT_PID
    workspace = Path(str(work.get("workspace") or ""))
    result = {
        "disposition": "agent_unavailable",
        "returncode": None,
        "repository_edits": False,
    }
    if not repository.is_dir() or not workspace.is_dir():
        result["disposition"] = "agent_failed"
        return result
    if _is_within(run_dir, repository) or _is_within(repository, run_dir):
        result["disposition"] = "agent_failed"
        return result
    executable = _resolve_agent_executable(packet, repository)
    prompt_dir = (run_dir / "prompts" / task_cid).resolve()
    if not _is_within(prompt_dir, run_dir) or _is_within(prompt_dir, repository):
        result["disposition"] = "agent_failed"
        return result
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / "prompt.txt"
    prompt_path.write_text(_agent_prompt(packet, repository), encoding="utf-8")
    stdin_bytes: bytes | None = None
    if executable:
        command = _build_workspace_agent_command(executable, repository, prompt_path)
        if command[command.index("--cwd") + 1] != str(repository):
            result["disposition"] = "agent_failed"
            return result
    else:
        command = _build_isolated_runner_command(repository)
        if command[command.index("--workspace") + 1] != str(repository):
            result["disposition"] = "agent_failed"
            return result
        stdin_bytes = prompt_path.read_bytes()
    log_path = run_dir / "logs" / f"{task_cid}.agent.log"
    log_handle = log_path.open("ab")
    process: subprocess.Popen[Any] | None = None
    try:
        process = subprocess.Popen(
            command,
            cwd=str(repository),
            env=_agent_env(),
            stdin=subprocess.PIPE if stdin_bytes is not None else subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if stdin_bytes is not None and process.stdin is not None:
            process.stdin.write(stdin_bytes)
            process.stdin.close()
    except OSError:
        log_handle.close()
        if process is not None and process.pid:
            _kill_process_group(int(process.pid))
        _write_agent_execution(workspace, task_cid, "agent_unavailable", None, False)
        return result
    log_handle.close()
    if process is None:
        _write_agent_execution(workspace, task_cid, "agent_unavailable", None, False)
        return result
    _AGENT_PID = int(process.pid)
    _atomic_json(
        run_dir / "agents" / f"{task_cid}.json",
        {
            "pid": _AGENT_PID,
            "cwd": str(repository.resolve()),
            "task_cid": task_cid,
            "task_key": task_key,
        },
    )
    returncode: int | None = None
    while returncode is None:
        if _stopped(run_dir):
            _kill_process_group(_AGENT_PID)
            try:
                returncode = process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                returncode = None
            break
        try:
            returncode = process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            continue
    _AGENT_PID = 0
    agent_record = run_dir / "agents" / f"{task_cid}.json"
    if agent_record.exists():
        agent_record.unlink()
    if _stopped(run_dir):
        disposition = "agent_stopped"
    elif returncode == 0:
        disposition = "agent_finished"
    else:
        disposition = "agent_failed"
    edited = _repository_scope_changed(repository, work)
    _write_agent_execution(workspace, task_cid, disposition, returncode, edited)
    result["disposition"] = disposition
    result["returncode"] = returncode
    result["repository_edits"] = edited
    return result


def _repository_scope_changed(repository: Path, work: Mapping[str, Any]) -> bool:
    scope = work.get("scope")
    if not isinstance(scope, list):
        return False
    for item in scope:
        if not isinstance(item, dict) or item.get("status") != "observed":
            continue
        relative = str(item.get("path") or "")
        if not _safe_relative(relative):
            continue
        source = repository.joinpath(*PurePosixPath(relative).parts)
        try:
            data = source.read_bytes()
        except OSError:
            return True
        if hashlib.sha256(data).hexdigest() != item.get("sha256"):
            return True
    return False


def _write_agent_execution(
    workspace: Path,
    task_cid: str,
    disposition: str,
    returncode: int | None,
    repository_edits: bool,
) -> None:
    payload = {
        "schema": "ipfs_accelerate_py.agent_supervisor.prompt-worker-execution@1",
        "task_cid": task_cid,
        "disposition": disposition,
        "repository_edits": repository_edits,
        "validation_executed": False,
        "authorizes_execution": False,
        "completion_authority": False,
        "agent_returncode": returncode,
    }
    _atomic_json(workspace / "execution.json", payload)


def _kill_recorded_agents(run_dir: Path) -> None:
    folder = run_dir / "agents"
    if not folder.is_dir():
        return
    for path in list(folder.glob("*.json")):
        payload = _read_receipt(path) or {}
        pid = payload.get("pid")
        expected = payload.get("cwd")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
            continue
        if not isinstance(expected, str) or not expected:
            continue
        if not _cwd_is(pid, Path(expected)):
            continue
        _kill_process_group(pid)


def _cwd_is(pid: int, expected: Path) -> bool:
    try:
        cwd = Path(os.readlink(f"/proc/{pid}/cwd")).resolve()
        return cwd == expected.resolve()
    except OSError:
        return False


def _kill_process_group(pid: int) -> None:
    if pid <= 1:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline and _process_alive(pid):
        time.sleep(0.05)
    if not _process_alive(pid):
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            return


def _observe_scope(repository: Path, relative: str) -> dict[str, Any]:
    record: dict[str, Any] = {"path": relative, "status": "rejected"}
    if not _safe_relative(relative) or ".git" in PurePosixPath(relative).parts:
        return record
    name = PurePosixPath(relative).name.casefold()
    suffix = PurePosixPath(name).suffix
    if (
        name in _CREDENTIAL_NAMES
        or name.startswith(".env.")
        or suffix in _CREDENTIAL_SUFFIXES
    ):
        record["status"] = "credential"
        return record
    candidate = repository.joinpath(*PurePosixPath(relative).parts)
    try:
        info = candidate.lstat()
    except OSError:
        record["status"] = "missing"
        return record
    if stat.S_ISLNK(info.st_mode):
        record["status"] = "symlink"
        return record
    if not stat.S_ISREG(info.st_mode):
        record["status"] = "not_a_file"
        return record
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        record["status"] = "unreadable"
        return record
    if not _is_within(resolved, repository):
        record["status"] = "escapes_repository"
        return record
    if info.st_size > 1_048_576:
        record["status"] = "too_large"
        return record
    try:
        data = resolved.read_bytes()
    except OSError:
        record["status"] = "unreadable"
        return record
    if len(data) != info.st_size:
        record["status"] = "changed"
        return record
    record["status"] = "observed"
    record["sha256"] = hashlib.sha256(data).hexdigest()
    record["size_bytes"] = len(data)
    return record


def _safe_relative(relative: str) -> bool:
    if not relative or relative.startswith("/") or "\\" in relative:
        return False
    parts = PurePosixPath(relative).parts
    return bool(parts) and ".." not in parts


def _install_stop_handlers() -> None:
    def _request_stop(_signum: int, _frame: object) -> None:
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
        pid = _AGENT_PID
        if pid > 1:
            _kill_process_group(pid)

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)


def _stopped(run_dir: Path) -> bool:
    return _STOP_REQUESTED or (run_dir / "stop").exists()


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    rows: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return []
        rows.append(item)
    return rows


def _str_list(value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return []
    return [str(item) for item in value]


def _worker_matches(pid: int, task_cid: str, run_dir: Path) -> bool:
    if pid <= 1 or not _is_our_worker(pid):
        return False
    text = _cmdline(pid)
    if text is None:
        return False
    return (
        "--worker" in text
        and _MODULE in text
        and task_cid in text
        and _cmdline_has_run(text, run_dir)
    )


def _is_supervisor(pid: int, run_dir: Path) -> bool:
    if pid <= 1 or not _is_our_worker(pid):
        return False
    text = _cmdline(pid)
    if text is None:
        return False
    return "--supervisor" in text and _MODULE in text and _cmdline_has_run(text, run_dir)


def _belongs_to_run(pid: int, run_dir: Path) -> bool:
    if _is_supervisor(pid, run_dir):
        return True
    task_cid = _arg(_cmdline(pid) or [], "--task-cid")
    if task_cid and _worker_matches(pid, task_cid, run_dir):
        return True
    return _recorded_runner(pid, run_dir)


def _runner_matches(pid: int, receipt: Mapping[str, Any]) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return False
    if not _process_alive(pid):
        return False
    text = _cmdline(pid)
    if text is None or _RUNNER_MODULE not in text:
        return False
    argv = receipt.get("argv")
    if not isinstance(argv, list) or _RUNNER_MODULE not in argv:
        return False
    expected = _arg(argv, "--workspace")
    return bool(expected) and _arg(text, "--workspace") == expected


def _recorded_runner(pid: int, run_dir: Path) -> bool:
    workers = run_dir / "workers"
    if not workers.is_dir():
        return False
    for path in workers.glob("*.json"):
        receipt = _read_receipt(path)
        if not isinstance(receipt, Mapping) or receipt.get("pid") != pid:
            continue
        if _runner_matches(pid, receipt):
            return True
    return False


def _arg(text: Sequence[str], name: str) -> str:
    if name not in text:
        return ""
    index = list(text).index(name)
    if index + 1 >= len(text):
        return ""
    return text[index + 1]


def _pids_for_run(run_dir: Path) -> list[int]:
    found: list[int] = []
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError:
        return found
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if _belongs_to_run(pid, run_dir):
            found.append(pid)
    return found


def _is_our_worker(pid: object) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return False
    return _cmdline(pid) is not None and _MODULE in (_cmdline(pid) or [])


def _cmdline(pid: int) -> list[str] | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return None
    return [item.decode("utf-8", "surrogateescape") for item in raw if item]


def _cmdline_has_run(text: Sequence[str], run_dir: Path) -> bool:
    if "--run-dir" not in text:
        return False
    index = list(text).index("--run-dir")
    if index + 1 >= len(text):
        return False
    try:
        return Path(text[index + 1]).resolve() == run_dir.resolve()
    except OSError:
        return False


def _log_tail_path(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return " ".join(text[-500:].split())


def _read_receipt(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
