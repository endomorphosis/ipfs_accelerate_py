"""Explicit isolated-benchmark candidate execution; no ambient default opt-in.

The signed launch binds these root-controlled files before a provider runs.
The fixed launcher drops to the container worker identity; this module never
interprets candidate code in the owner process or accepts a worker command.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import threading
import time

CANDIDATE_RUNNER_ENV = "IPFS_SUPERVISOR_CANDIDATE_RUNNER_BINDING"
ROOT = Path("/opt/ipfs-supervisor")
SCHEMA = "supervisor-isolated-candidate-runner@1"
_EXECUTION_LOCK = threading.RLock()
GIT_OWNER_ENV = {
    "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_COUNT": "2", "GIT_CONFIG_KEY_0": "core.hooksPath",
    "GIT_CONFIG_VALUE_0": "/dev/null", "GIT_CONFIG_KEY_1": "core.fsmonitor",
    "GIT_CONFIG_VALUE_1": "false", "GIT_TERMINAL_PROMPT": "0",
}


def _root_file(path: Path) -> str:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("candidate runner requires exact non-symlink paths")
    for parent in (path, *path.parents):
        info = parent.lstat()
        expected = stat.S_ISREG if parent == path else stat.S_ISDIR
        if info.st_uid != 0 or info.st_mode & 0o022 or not expected(info.st_mode):
            raise ValueError("candidate runner files and ancestors must be root-controlled")
    if path.stat().st_size > 65_536:
        raise ValueError("candidate runner artifact exceeds bound")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind_candidate_runner(argv) -> dict:
    """Bind the one installed privilege-dropping entry, never arbitrary argv."""
    expected = (str(ROOT / "bin/validation-worker"),)
    if not isinstance(argv, (list, tuple)) or tuple(argv) != expected:
        raise ValueError("exact installed candidate validation launcher required")
    files = {str(path): _root_file(path) for path in (
        ROOT / "bin/validation-worker", ROOT / "bin/worker-entry", ROOT / "container-boundary.json",
    )}
    boundary = json.loads((ROOT / "container-boundary.json").read_text())
    if (boundary.get("schema") != "supervisor-container-worker-boundary@1"
            or type(boundary.get("owner_uid")) is not int
            or boundary["owner_uid"] != os.getuid() or os.geteuid() != os.getuid()
            or boundary.get("worker_uid") != 1001 or boundary["owner_uid"] == 1001
            or boundary.get("single_worker") is not True
            or boundary.get("namespaces") != {name: os.readlink("/proc/self/ns/" + name) for name in ("pid", "mnt", "net")}):
        raise ValueError("candidate runner differs from actual container owner or namespaces")
    return {"schema": SCHEMA, "argv": list(expected), "files": files,
            "owner_uid": boundary["owner_uid"], "worker_uid": boundary["worker_uid"],
            "namespaces": boundary["namespaces"], "completion_authority": False}


def verify_candidate_runner(binding) -> dict:
    if not isinstance(binding, dict) or binding != bind_candidate_runner(binding.get("argv")):
        raise ValueError("candidate runner no longer matches its signed launch binding")
    return binding


def candidate_runner_from_environment() -> dict | None:
    raw = os.environ.get(CANDIDATE_RUNNER_ENV)
    if raw is None:
        return None
    if not raw or len(raw) > 16_384:
        raise ValueError("bounded explicit candidate runner binding required")
    return verify_candidate_runner(json.loads(raw))


def run_candidate(binding, argv, *, cwd: Path, timeout: float,
                  stdout=None, stderr=None, text=False) -> subprocess.CompletedProcess:
    with _EXECUTION_LOCK:
        return _run_candidate(binding, argv, cwd=cwd, timeout=timeout,
                              stdout=stdout, stderr=stderr, text=text)


def _run_candidate(binding, argv, *, cwd: Path, timeout: float,
                   stdout=None, stderr=None, text=False) -> subprocess.CompletedProcess:
    """Run exact owner-selected arguments; on timeout let sudo reap its worker."""
    verify_candidate_runner(binding)
    if (not isinstance(argv, (tuple, list)) or not 1 <= len(argv) <= 128
            or any(not isinstance(arg, str) or not arg or len(arg) > 8192 or "\0" in arg for arg in argv)
            or not 0 < timeout <= 300):
        raise ValueError("bounded literal candidate argv and timeout required")
    cwd = Path(cwd).absolute()
    if cwd.resolve(strict=True) != cwd or not cwd.is_dir():
        raise ValueError("exact candidate validation cwd required")
    command = [*binding["argv"], *argv]
    process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
        stdout=stdout, stderr=stderr, text=text, start_new_session=True,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", **GIT_OWNER_ENV})
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Killing sudo immediately can orphan the downgraded process. TERM is
        # forwarded by the real monitor, which has authority to reap UID1001.
        process.send_signal(signal.SIGTERM)
        try:
            process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise RuntimeError("candidate monitor failed to reap worker after timeout")
        raise
    verify_candidate_runner(binding)
    return subprocess.CompletedProcess(command, process.returncode, out, err)


def install_portal_candidate_runner(portal, binding) -> None:
    """Use the native scheduler with an isolated subprocess execution seam."""
    verify_candidate_runner(binding)
    if getattr(portal, "manual_completion_authority_task_ids", ()):
        raise ValueError("isolated benchmark runner cannot replace authority validation policy")
    # The initial container profile uses one worker UID and reaps all of its
    # descendants between commands. Parallelism requires separate boundaries.
    portal.validation_scheduler.max_workers = 1
    portal.validation_scheduler.resource_budget = 1

    def runner(*, spec, workspace_path, timeout_seconds, environment):
        from ..validation.validation_runtime import validation_shell_command
        started = time.time()
        argv = validation_shell_command(str(spec.command))
        completed = run_candidate(binding, argv, cwd=workspace_path, timeout=min(float(timeout_seconds), 300),
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return {"command": str(spec.command), "raw_command": str(spec.raw_command or spec.command),
                "returncode": completed.returncode, "output": completed.stdout or "",
                "duration_seconds": time.time() - started, "candidate_runner": binding}

    portal._validation_command_runner = runner
    portal._isolated_candidate_runner = binding
    original = portal._run_validation_commands

    def uncached(*args, **kwargs):
        # Existing scheduler cache identities do not bind a privilege-dropping
        # runner, so earlier in-process receipts cannot satisfy this profile.
        kwargs["force_uncached"] = True
        return original(*args, **kwargs)

    portal._run_validation_commands = uncached

    def materialize(*, workspace_path, log_path, commands, task):
        from ..todo_daemon.implementation_daemon import _open_private_implementation_log
        from ..validation.validation_runtime import ValidationRuntimeError, validation_shell_command
        results = []
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with _open_private_implementation_log(log_path, "a") as log:
            for raw in commands:
                command = str(raw or "").strip()
                if not command:
                    continue
                try:
                    completed = run_candidate(binding, validation_shell_command(command),
                        cwd=workspace_path, timeout=min(float(portal.implementation_timeout or 300), 300),
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    entry = {"command": command, "ok": completed.returncode == 0,
                             "returncode": completed.returncode,
                             "output_tail": (completed.stdout or "")[-1200:], "candidate_runner": binding}
                except (OSError, subprocess.TimeoutExpired, ValidationRuntimeError) as exc:
                    entry = {"command": command, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
                results.append(entry)
                log.write("\n[isolated auto-rescue] " + json.dumps(entry, sort_keys=True) + "\n")
                if entry["ok"]:
                    break
        portal._record_event("implementation_auto_rescue_materialize_commands", {
            "task_id": task.task_id, "workspace_path": str(workspace_path), "results": results,
        })
        return results

    portal._run_auto_rescue_materialize_commands = materialize
    original_writer = portal._lgswf_writer_path

    def no_unqualified_writer(task_id):
        if original_writer(task_id) is not None:
            raise ValueError("registered deterministic writer is outside isolated benchmark execution policy")
        return None

    portal._lgswf_writer_path = no_unqualified_writer
