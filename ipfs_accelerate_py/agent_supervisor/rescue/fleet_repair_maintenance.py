"""Bounded, lock-owning maintenance observations without repair dispatch.

The normal dispatcher must be stopped separately after its current job finishes.
This mode never selects, reconciles, claims, starts, stops or completes a job.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import threading
import time

from .fleet_watchdog import command, load_config, repair_hold_paths, write_json

SCHEMA = "agent-supervisor/fleet-repair-maintenance-observation@1"
UNITS = {"dispatcher": "ipfs-taskboard-repair.service", "repair_job": "ipfs-taskboard-repair-job.service"}
FIELDS = {"Id", "LoadState", "ActiveState", "SubState", "MainPID"}


class ObservationBlocked(RuntimeError):
    pass


def _read(path: Path, *, absent=False):
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    except FileNotFoundError:
        if absent:
            return None
        raise
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid() or before.st_nlink != 1:
            raise ObservationBlocked("observation input is not an owned unique regular file")
        raw = os.read(descriptor, 4 * 1024 * 1024 + 1)
        after, current = os.fstat(descriptor), path.lstat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if (len(raw) > 4 * 1024 * 1024 or len(raw) != before.st_size
                or identity(before) != identity(after) or identity(before) != identity(current)):
            raise ObservationBlocked("observation input changed or exceeded bound")
        return raw
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def retained_dispatcher_lock(path: Path):
    """Use the existing native worker mutex; never replace or steal it."""
    descriptor = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
            raise ObservationBlocked("maintenance worker lock is not an owned unique regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def require_current():
            current, retained = path.lstat(), os.fstat(descriptor)
            if (not stat.S_ISREG(current.st_mode) or current.st_nlink != 1
                    or (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino)
                    or (retained.st_dev, retained.st_ino) != (info.st_dev, info.st_ino)):
                raise ObservationBlocked("maintenance worker lock changed")
        require_current()
        yield require_current
    finally:
        os.close(descriptor)


def _unit_snapshot(unit, runner=command):
    result = runner({"argv": ["systemctl", "--user", "show", unit, "--no-pager",
                              "--property=" + ",".join(sorted(FIELDS))]}, cwd="/", timeout=10)
    if result.get("returncode") != 0:
        raise ObservationBlocked("unit observation failed: " + unit)
    values = {}
    for line in result.get("stdout", "").splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in values:
            raise ObservationBlocked("unit observation ambiguous: " + unit)
        values[key] = value
    if set(values) != FIELDS or values["Id"] != unit:
        raise ObservationBlocked("unit observation incomplete: " + unit)
    return values


def _closed(unit, *, dispatcher):
    return (unit["MainPID"] == "0"
            and unit["LoadState"] in ({"loaded"} if dispatcher else {"loaded", "not-found"})
            and (unit["ActiveState"], unit["SubState"]) in {("inactive", "dead"), ("failed", "failed")})


def _jobs(config, now):
    root = Path(config["state_dir"])
    bindings, waiting, held = {}, [], []
    identifiers = [board["id"] for board in config["boards"]]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ObservationBlocked("configured boards are ambiguous")
    for board in config["boards"]:
        identifier = board["id"]
        if not isinstance(identifier, str) or not identifier or Path(identifier).name != identifier or identifier in {".", ".."}:
            raise ObservationBlocked("configured board identifier invalid")
        raw = _read(root / "repairs" / identifier / "job.json", absent=True)
        holds = repair_hold_paths(board)
        bindings[identifier] = {"sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
                                "holds": holds}
        if raw is None:
            continue
        job = json.loads(raw)
        if (not isinstance(job, dict) or job.get("board_id") != identifier
                or job.get("status") not in {"queued", "verified_healthy", "verified_published"}):
            raise ObservationBlocked("job claim is running or unknown: " + identifier)
        attempts, started, finished = job.get("attempts", 0), job.get("last_started_at"), job.get("finished_at")
        if type(attempts) is not int or attempts < 0:
            raise ObservationBlocked("job attempt count unknown: " + identifier)
        if started is None:
            if attempts != 0 or finished is not None:
                raise ObservationBlocked("job claim history inconsistent: " + identifier)
        elif (not all(type(v) in (int, float) and math.isfinite(v) for v in (started, finished))
              or not 0 < started <= finished <= now):
            raise ObservationBlocked("job attempt has not positively finished: " + identifier)
        if job["status"] == "queued":
            due = job.get("next_attempt_at", 0)
            if type(due) not in (int, float) or not math.isfinite(due) or due < 0:
                raise ObservationBlocked("job schedule unknown: " + identifier)
            (held if holds else waiting).append({"board_id": identifier, "next_attempt_at": due})
    return bindings, waiting, held


def observe(config, *, runner=command, unit_names=None):
    """Inspect a full stable queue and both real units; never alter claims."""
    names = UNITS if unit_names is None else unit_names
    units = {role: _unit_snapshot(name, runner) for role, name in names.items()}
    if not _closed(units["repair_job"], dispatcher=False):
        running = units["repair_job"]["ActiveState"] in {"active", "activating", "deactivating"}
        return {"status": "running" if running else "blocked", "reason": "repair_job_not_positively_inactive", "units": units}
    if not _closed(units["dispatcher"], dispatcher=True):
        return {"status": "blocked", "reason": "normal_dispatcher_not_positively_inactive", "units": units}
    jobs, waiting, held = _jobs(config, time.time())
    final_units = {role: _unit_snapshot(name, runner) for role, name in names.items()}
    final_jobs, _, _ = _jobs(config, time.time())
    if final_units != units or final_jobs != jobs:
        raise ObservationBlocked("unit or queue state changed during observation")
    return {"status": "waiting" if waiting or held else "idle", "waiting": waiting, "held": held,
            "units": units, "queue_bindings": jobs, "no_unfinished_claims": True,
            "next_attempt_at": min((row["next_attempt_at"] for row in waiting), default=None)}


def run(config, *, config_path: Path, duration_seconds=900.0, interval_seconds=5.0,
        stop=None, runner=command):
    if (not math.isfinite(duration_seconds) or not 1 <= duration_seconds <= 3600
            or not math.isfinite(interval_seconds) or not 0.2 <= interval_seconds <= 30):
        raise ValueError("maintenance duration must be 1..3600 seconds; interval 0.2..30 seconds")
    original_config = _read(config_path)
    current_config = load_config(config_path)
    current_config["_config_path"] = str(config_path.resolve())
    if current_config != config or _read(config_path) != original_config:
        raise ObservationBlocked("loaded maintenance configuration differs from retained bytes")
    root = Path(current_config["state_dir"])
    source = Path(__file__).resolve()
    base = {"schema": SCHEMA, "mode": "maintenance_observation", "dispatch_enabled": False,
            "observer_pid": os.getpid(), "observer_source": str(source),
            "observer_source_sha256": hashlib.sha256(_read(source)).hexdigest(),
            "configured_normal_runtime_release": config.get("runtime_release"),
            "config_sha256": hashlib.sha256(original_config).hexdigest(),
            "completion_authority": False, "claim_mutation": False}
    signal_handlers = {}
    if stop is None:
        stop = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda *_: stop.set())
    try:
        with retained_dispatcher_lock(root / "repair-worker.lock") as require_lock:
            deadline = time.monotonic() + duration_seconds
            while not stop.is_set() and time.monotonic() < deadline:
                try:
                    require_lock()
                    if _read(config_path) != original_config:
                        raise ObservationBlocked("maintenance configuration changed")
                    result = observe(config, runner=runner)
                    require_lock()
                    if _read(config_path) != original_config:
                        raise ObservationBlocked("maintenance configuration changed during observation")
                except Exception as exc:
                    result = {"status": "blocked", "reason": f"{type(exc).__name__}: {exc}"[:512]}
                require_lock()  # Lost mutex identity must never publish an idle record.
                record = dict(base, **result, observed_at=time.time())
                write_json(root / "repair-worker.json", record)
                print(json.dumps(record, sort_keys=True), flush=True)
                stop.wait(min(interval_seconds, max(0, deadline - time.monotonic())))
            require_lock()
            record = dict(base, status="maintenance_observer_stopped", observed_at=time.time(),
                          reason="signal" if stop.is_set() else "bounded_duration_complete")
            write_json(root / "repair-worker.json", record)
            print(json.dumps(record, sort_keys=True), flush=True)
    finally:
        for sig, handler in signal_handlers.items():
            signal.signal(sig, handler)
    return 0
