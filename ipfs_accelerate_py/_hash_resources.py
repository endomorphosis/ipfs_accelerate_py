"""Stdlib-only hashing resource admission, safe before supervisor verification.

Use ``with hashing_lock() as workers`` around a heavy batch, then pass
``workers`` to its *one* thread pool. Nested native pools should use one thread.
``hashing_worker_slot()`` admits one streaming file worker, sharing a bounded
slot budget across threads and processes. Heavy batches exclude all slots.
Every participant for a UID shares the same global lock, even on an idle
machine; ``kind`` and ``exclusive`` do not create separate budgets.
This coordinates upgraded callers sharing /tmp, not unrelated hashing programs
or containers with a separate /tmp mount. On platforms without flock, admission
is process-local and worker counts are restricted to one.
"""

from __future__ import annotations

import errno
import math
import os
import stat
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows: preserve a conservative, process-local fallback.
    fcntl = None  # type: ignore[assignment]


MAX_WORKERS_CAP = 4
DEFAULT_MAX_WORKERS = 2
DEFAULT_LOCK_TIMEOUT_SECONDS = 60.0
CPU_LOAD_LIMIT = 0.70
MIN_AVAILABLE_BYTES = 2 * 1024**3
BYTES_PER_WORKER = 512 * 1024**2
_process_lock = threading.RLock()
_local = threading.local()
_open_fds: set[int] = set()
_fd_registry_lock = threading.Lock()


def _read_proc(path: str | Path) -> str:
    return Path(path).read_text(encoding="ascii", errors="replace")


def _finite_nonnegative(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError("expected a finite nonnegative measurement")
    return parsed


def _cgroup_directories() -> tuple[Path, ...] | None:
    """Find the cgroup v2 leaf and visible ancestors, including delegated mounts."""
    try:
        membership = next(
            (line[3:] for line in _read_proc("/proc/self/cgroup").splitlines()
             if line.startswith("0::")), None
        )
        if membership is None:  # No v2 hierarchy (e.g. cgroup v1).
            return ()
        for line in _read_proc("/proc/self/mountinfo").splitlines():
            before, separator, after = line.partition(" - ")
            if not separator or after.split()[0] != "cgroup2":
                continue
            fields = before.split()
            # mountinfo escapes whitespace and backslashes with octal sequences.
            def unescape(value: str) -> str:
                for escape, replacement in (("\\040", " "), ("\\011", "\t"),
                                            ("\\012", "\n"), ("\\134", "\\")):
                    value = value.replace(escape, replacement)
                return value

            root, mount = Path(unescape(fields[3])), Path(unescape(fields[4]))
            member = Path(membership)
            if not member.is_absolute() or ".." in member.parts:
                return None
            if membership == "/":  # Membership relative to a cgroup namespace.
                relative = Path(".")
            elif member.is_relative_to(root):
                relative = member.relative_to(root)
            else:
                continue
            leaf = mount / relative
            return (leaf, *(p for p in leaf.parents if p == mount or mount in p.parents))
    except (OSError, ValueError, IndexError):
        pass
    return None


def _optional_read(path: Path) -> str | None:
    try:
        return _read_proc(path)
    except FileNotFoundError:  # A controller may not be enabled in this hierarchy.
        return None


def _psi_busy(text: str, resource: str) -> bool:
    records = {}
    for line in text.splitlines():
        fields = line.split()
        if fields:
            values = dict(field.split("=", 1) for field in fields[1:])
            records[fields[0]] = _finite_nonnegative(values["avg10"])
    if "some" not in records:
        raise ValueError("missing PSI measurement")
    # I/O and memory stalls deserve tighter limits than runnable CPU contention.
    some_limit = {"cpu": 20.0, "io": 5.0, "memory": 1.0}[resource]
    full_limit = {"cpu": 20.0, "io": 1.0, "memory": 0.1}[resource]
    return records["some"] >= some_limit or records.get("full", 0.0) >= full_limit


def host_hash_pressure() -> tuple[int, str]:
    """Return a live worker ceiling and diagnostic reason; unknown means serial.

    Limits account for CPU affinity, all visible cgroup v2 ancestor quotas and
    memory limits (including memory.high), host memory/load, and CPU/I/O/memory
    pressure. A one-worker floor allows progress on small or busy machines.
    """
    try:
        cpus = float(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        cpus = float(os.cpu_count() or 1)
    try:
        load = _finite_nonnegative(_read_proc("/proc/loadavg").split()[0])
        memory = {}
        for line in _read_proc("/proc/meminfo").splitlines():
            name, value = line.split(":", 1)
            memory[name] = int(value.split()[0]) * 1024
        available, total = memory["MemAvailable"], memory["MemTotal"]
        if available < 0 or total <= 0 or cpus <= 0:
            raise ValueError("invalid resource measurement")
        if memory.get("SwapTotal", 0) and memory.get("SwapFree", 0) < 256 * 1024**2:
            return 1, "host_swap_exhaustion"

        groups = _cgroup_directories()
        if groups is None:
            return 1, "unknown_cgroup_limits"
        for group in groups:
            quota = _optional_read(group / "cpu.max")
            if quota is not None:
                budget, period = quota.split()
                if budget != "max":
                    if int(budget) <= 0 or int(period) <= 0:
                        raise ValueError("invalid CPU quota")
                    cpus = min(cpus, int(budget) / int(period))
            for name in ("memory.max", "memory.high"):
                limit = _optional_read(group / name)
                if limit is None or limit.strip() == "max":
                    continue
                maximum = int(limit)
                current = int(_read_proc(group / "memory.current"))
                if maximum < 0 or current < 0:
                    raise ValueError("invalid memory quota")
                available = min(available, max(0, maximum - current))
                total = min(total, maximum)
            for resource in ("cpu", "io", "memory"):
                pressure = _optional_read(group / f"{resource}.pressure")
                if pressure is not None and _psi_busy(pressure, resource):
                    return 1, f"cgroup_{resource}_pressure"

        for resource in ("cpu", "io", "memory"):
            if _psi_busy(_read_proc(f"/proc/pressure/{resource}"), resource):
                return 1, f"host_{resource}_pressure"
        if available < MIN_AVAILABLE_BYTES or available <= total * 0.20:
            return 1, "memory_headroom"
        if load >= cpus * CPU_LOAD_LIMIT:
            return 1, "host_cpu_load"
        by_cpu = max(1, int(cpus - load))
        by_memory = max(1, available // BYTES_PER_WORKER)
        return max(1, min(MAX_WORKERS_CAP, by_cpu, by_memory)), "admitted"
    except (OSError, ValueError, KeyError, IndexError, ZeroDivisionError):
        return 1, "unknown_resource_pressure"


def hash_worker_limit(requested: int | None = None) -> int:
    """Cap an outer hash pool; IPFS_HASH_MAX_WORKERS defaults to 2, never above 4."""
    try:
        configured = int(os.environ.get("IPFS_HASH_MAX_WORKERS", DEFAULT_MAX_WORKERS))
    except ValueError:
        configured = 1
    ceiling = max(1, min(MAX_WORKERS_CAP, configured))
    if requested is not None:
        ceiling = min(ceiling, max(1, int(requested)))
    if fcntl is None:
        ceiling = 1
    if getattr(_local, "pid", None) == os.getpid():
        ceiling = min(ceiling, _local.workers)
    return min(ceiling, host_hash_pressure()[0])


def hash_lock_path(kind: str = "file-hash") -> Path:
    """Stable across kinds, checkouts, TMPDIR, and XDG_RUNTIME_DIR settings."""
    uid = os.geteuid() if hasattr(os, "geteuid") else "local"
    return Path("/tmp") / f"ipfs-accelerate-heavy-hash-{uid}.lock"


class HashingResourceTimeout(TimeoutError):
    """The global hashing admission budget remained occupied until its deadline."""


class HashingResourceUpgradeError(RuntimeError):
    """A per-file worker attempted an unsafe shared-to-exclusive lock upgrade."""


def _open_lock_file(path: Path) -> int:
    # Fork must not land between open/close and registry updates: the child
    # would otherwise retain an untracked parent lock or close a reused FD.
    with _fd_registry_lock:
        fd = os.open(path, os.O_RDWR | os.O_CREAT
                     | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0), 0o600)
        _open_fds.add(fd)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or info.st_nlink != 1 or info.st_mode & 0o022):
                raise PermissionError("unsafe hashing admission lock file")
        except BaseException:
            _open_fds.discard(fd)
            os.close(fd)
            raise
        return fd


def _close_lock_file(fd: int) -> None:
    with _fd_registry_lock:
        _open_fds.discard(fd)
        os.close(fd)


def _check_lock_path(fd: int, path: Path) -> None:
    held, named = os.fstat(fd), os.stat(path, follow_symlinks=False)
    if (held.st_nlink != 1 or held.st_dev != named.st_dev
            or held.st_ino != named.st_ino):
        raise PermissionError("hashing admission lock changed while waiting")


def _try_flock(fd: int, mode: int) -> bool:
    try:
        fcntl.flock(fd, mode | fcntl.LOCK_NB)
        return True
    except OSError as error:
        if error.errno not in (errno.EAGAIN, errno.EACCES, errno.EINTR):
            raise
        return False


def _lock_timeout(timeout: float | None) -> float:
    if timeout is None:
        timeout = float(os.environ.get(
            "IPFS_HASH_LOCK_TIMEOUT_SECONDS", DEFAULT_LOCK_TIMEOUT_SECONDS
        ))
    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("hash lock timeout must be finite and nonnegative")
    return timeout


@contextmanager
def hashing_lock(*, kind: str = "file-hash", exclusive: bool | None = None,
                 timeout: float | None = None) -> Iterator[int]:
    """Admit one heavy batch per UID, yielding its freshly sampled worker limit.

    ``exclusive=False`` still participates: leaving low-pressure entrants outside
    the lock defeats admission when many worktrees start at once. Same-thread
    nesting reuses admission across kinds; do not acquire in workers while their
    parent holds admission and waits for them. The timeout covers both thread
    and process contention. No nice/ionice or persistent environment changes.
    """
    timeout = _lock_timeout(timeout)
    if getattr(_local, "pid", None) == os.getpid():
        if getattr(_local, "mode", "exclusive") == "worker":
            raise HashingResourceUpgradeError(
                "cannot acquire exclusive hashing_lock while holding a worker slot; "
                "leave the worker slot before requesting heavy-batch admission"
            )
        yield hash_worker_limit()
        return
    owner_pid = os.getpid()
    process_lock = _process_lock
    deadline = time.monotonic() + timeout
    if not process_lock.acquire(timeout=timeout):
        raise HashingResourceTimeout(f"hash admission timed out for {kind!r}")
    fd = None
    locked = False
    try:
        if fcntl is not None:
            path = hash_lock_path(kind)
            fd = _open_lock_file(path)
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    locked = True
                    break
                except OSError as error:
                    if error.errno not in (errno.EAGAIN, errno.EACCES, errno.EINTR):
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise HashingResourceTimeout(
                            f"hash admission timed out for {kind!r} at {hash_lock_path()}"
                        ) from None
                    time.sleep(min(0.05, remaining))
            # A waiter must not acquire an unlinked/replaced lock inode while
            # newer entrants coordinate through a different file at this path.
            _check_lock_path(fd, path)
        # Sample after waiting: the pressure observed before contention is stale.
        workers = hash_worker_limit()
        _local.pid, _local.workers, _local.mode = os.getpid(), workers, "exclusive"
        try:
            yield workers
        finally:
            if os.getpid() == owner_pid:
                _local.pid = None
    finally:
        # atfork already closed inherited descriptors and replaced the lock.
        # A normally unwinding child must not close a reused descriptor number
        # or release either its fresh lock or the parent's admission.
        if os.getpid() == owner_pid:
            try:
                if fd is not None:
                    try:
                        if locked:
                            fcntl.flock(fd, fcntl.LOCK_UN)
                    finally:
                        _close_lock_file(fd)
            finally:
                process_lock.release()


def _worker_slot_paths() -> tuple[Path, Path, tuple[Path, ...]]:
    global_path = hash_lock_path()
    gate = global_path.with_name(global_path.name + ".slots.lock")
    slots = tuple(
        global_path.with_name(global_path.name + f".slot-{index}.lock")
        for index in range(MAX_WORKERS_CAP)
    )
    return global_path, gate, slots


@contextmanager
def hashing_worker_slot(*, timeout: float | None = None) -> Iterator[int]:
    """Admit ONE file worker across processes; yield its one-worker budget.

    Hold this only around streaming reads/hashing, not owner requests or waits.
    Numbered flock slots allow independent threads and processes to overlap;
    the global shared flock excludes ``hashing_lock`` heavy batches. A brief
    admission mutex counts active slots before admission, so reduced pressure
    limits drain existing work instead of admitting around occupied high slots.
    Already running workers are not interrupted. Nested calls in the same
    thread reuse its slot or exclusive admission. Shared-to-exclusive upgrades
    are rejected by ``hashing_lock`` rather than waiting on themselves.
    """
    timeout = _lock_timeout(timeout)
    owner_pid = os.getpid()
    if getattr(_local, "pid", None) == owner_pid:
        yield 1
        return
    if fcntl is None:
        with hashing_lock(timeout=timeout):
            _local.mode = "worker"
            try:
                yield 1
            finally:
                if os.getpid() == owner_pid:
                    _local.mode = "exclusive"
        return

    deadline = time.monotonic() + timeout
    global_path, gate_path, slot_paths = _worker_slot_paths()
    descriptors: list[int] = []
    try:
        for path in (global_path, gate_path, *slot_paths):
            descriptors.append(_open_lock_file(path))
        global_fd, gate_fd, *slot_fds = descriptors
        selected: int | None = None
        while selected is None:
            shared = gate_held = False
            free: list[int] = []
            try:
                shared = _try_flock(global_fd, fcntl.LOCK_SH)
                if shared:
                    _check_lock_path(global_fd, global_path)
                    gate_held = _try_flock(gate_fd, fcntl.LOCK_EX)
                    if gate_held:
                        _check_lock_path(gate_fd, gate_path)
                        # Protect all currently free slots while counting. Other
                        # entrants hold the same short-lived gate; workers may
                        # finish at any time, which only makes us conservative.
                        for index, fd in enumerate(slot_fds):
                            if _try_flock(fd, fcntl.LOCK_EX):
                                free.append(index)
                                _check_lock_path(fd, slot_paths[index])
                        ceiling = hash_worker_limit()
                        candidates = [index for index in free if index < ceiling]
                        if len(slot_fds) - len(free) < ceiling and candidates:
                            selected = candidates[0]
            finally:
                if os.getpid() == owner_pid:
                    for index in free:
                        if index != selected:
                            fcntl.flock(slot_fds[index], fcntl.LOCK_UN)
                    if gate_held:
                        fcntl.flock(gate_fd, fcntl.LOCK_UN)
                    if shared and selected is None:
                        # Waiting for a slot must not retain a shared lock and
                        # unnecessarily prevent an exclusive batch from running.
                        fcntl.flock(global_fd, fcntl.LOCK_UN)
            if selected is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise HashingResourceTimeout("hash worker slot admission timed out")
                time.sleep(min(0.05, remaining))

        # Keep only the global shared lock and this worker's exclusive slot
        # through the body. No Python mutex is held during file reads.
        keep = (global_fd, slot_fds[selected])
        for fd in tuple(descriptors):
            if fd not in keep:
                _close_lock_file(fd)
                descriptors.remove(fd)
        _local.pid, _local.workers, _local.mode = owner_pid, 1, "worker"
        try:
            yield 1
        finally:
            if os.getpid() == owner_pid:
                _local.pid = None
    finally:
        # The at-fork hook already closed inherited FDs. Never touch their old
        # numbers while an inherited child context unwinds.
        if os.getpid() == owner_pid:
            for fd in reversed(descriptors):
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    _close_lock_file(fd)


def _after_fork() -> None:
    global _process_lock, _local, _fd_registry_lock
    # Close, never LOCK_UN: a fork inherits the parent's open-file description.
    for fd in tuple(_open_fds):
        os.close(fd)
    _open_fds.clear()
    _process_lock, _local = threading.RLock(), threading.local()
    _fd_registry_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=lambda: _fd_registry_lock.acquire(),
        after_in_parent=lambda: _fd_registry_lock.release(),
        after_in_child=_after_fork,
    )
