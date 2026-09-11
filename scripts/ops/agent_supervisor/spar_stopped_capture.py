"""Fresh native stopped-owner capture after an interrupted workflow controller.

This protocol does not recreate a live-capture session from its audit JSON. A
new no-state keeper overlaps the old keeper before our exact failed controller
is retired. Fresh kernel fences, an immutable canonical task-store copy, and
accepted native plan/claim observations then produce a distinct capability.
"""
from __future__ import annotations

from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import stat
import struct
import subprocess
import time
from types import SimpleNamespace

from . import spar_legacy_capture as native
from . import spar_legacy_import_plan as producer
from . import spar_merge_owner as role
from . import spar_retained_capture_driver as workflow
from .spar_merge_owner_handoff import _scopes, configured_queue_root

SCHEMA = "spar/native-stopped-queue-capture@1"
TASK_SCHEMA = "spar/stopped-task-store-observation@1"
ADMISSION_SCHEMA = "spar/stopped-task-store-admission@1"
KEEPER_SCHEMA = "spar/workflow-keeper-succession@1"
CGROUP_ROOT = Path("/sys/fs/cgroup")
PROCESS_FIELDS = {"pid", "parent_pid", "start_time_ticks", "boot_id"}


def evidence_cid(value):
    """Bound full native evidence separately from small recovery RPC messages.

    Preserve the existing canonical identity encoding. This local stopped-state
    protocol admits at most 4 MiB; remote cursor/receipt RPC bounds do not change.
    """
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as exc:
        raise role.SparMergeOwnerError("stopped evidence is not bounded JSON") from exc
    if len(raw) > role.MAX_JSON_BYTES:
        raise role.SparMergeOwnerError("stopped evidence exceeds native origin bound")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def require(value, reason):
    if not value:
        raise role.SparMergeOwnerError(reason)


def _birth(pid):
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import read_process_birth
    value = read_process_birth(pid)
    require(value is not None, "workflow process birth unavailable")
    return value.to_dict()


def _exact_birth(value):
    require(type(value) is dict and set(value) == PROCESS_FIELDS
            and all(type(value[k]) is int and value[k] > 0 for k in PROCESS_FIELDS - {"boot_id"})
            and type(value["boot_id"]) is str, "workflow process birth invalid")
    require(_birth(value["pid"]) == value, "workflow process birth changed")


def _argv(pid):
    body = Path(f"/proc/{pid}/cmdline").read_bytes()
    require(len(body) <= 65536 and body.endswith(b"\0"), "workflow process argv unavailable")
    return body[:-1].decode().split("\0")


def _wait_pidfd(fd, seconds):
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    require(bool(poller.poll(int(seconds * 1000))), "workflow process exit remains unverified")


def _files(unit):
    fragment = workflow._property(unit, "Unit", "FragmentPath", "s")
    drops = workflow._property(unit, "Unit", "DropInPaths", "as")
    require(fragment and len(drops) < 64, "native unit file inventory unavailable")
    return {path: hashlib.sha256(native._bytes(Path(path))).hexdigest() for path in [fragment, *drops]}


def _verify_lock_line(line, info, *, ofd, pid):
    fields = line.split()
    require(len(fields) == 9 and fields[0] == "lock:" and fields[2:5] == [
                "OFDLCK" if ofd else "FLOCK", "ADVISORY", "WRITE"]
            and fields[5] == str(-1 if ofd else pid) and fields[7:] == ["0", "EOF"],
            "native retained kernel fence differs")
    try:
        major, minor, inode = fields[6].split(":")
        identity = (os.makedev(int(major, 16), int(minor, 16)), int(inode))
    except (ValueError, OverflowError) as exc:
        raise role.SparMergeOwnerError("native kernel fence identity unavailable") from exc
    require(identity == (info.st_dev, info.st_ino), "native retained kernel fence inode differs")


def _task_inventory(database):
    result = {}
    for path in (database, database.with_name(database.name + ".wal")):
        if path == database or path.exists() or path.is_symlink():
            info = path.lstat()
            require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and info.st_nlink == 1,
                    "canonical task database/WAL is not an owned unique regular inode")
            result[path.name] = role._file_identity(info)
    return result


class KeeperSuccession:
    """Exact workflow process succession; it grants no store/file authority."""

    def __init__(self, *, repository_root, config_path, controller_birth, helper_birth,
                 controller_argv, hold_sha256, source_head, source_tree, journal_root, fleet_config, task_runtime, owner_identity):
        self.root, self.config = Path(repository_root).absolute(), Path(config_path).absolute()
        self.operator = native._native_operator(self.root)
        self.board, _ = self.operator._load_config(self.config)
        self.paths = self.operator._runtime_paths(self.board)
        self.queue_root = configured_queue_root(self.board)
        self.hold = self.board.path(self.board.runtime_paths["root"]) / "HOLD"
        self.source = self.operator.source_binding(self.config)
        self.worker_path = Path(__file__).with_name("spar_stopped_task_observation.py")
        self.worker_bytes = native._bytes(self.worker_path, 1024 * 1024)
        self.worker_sha256 = hashlib.sha256(self.worker_bytes).hexdigest()
        self.task_runtime = dict(task_runtime)
        role._closed(self.task_runtime, {"manifest_path", "manifest_sha256", "helper_sha256"})
        self.runtime_probe = probe_task_runtime(self)
        require(self.runtime_probe.get("source") == self.source
                and self.runtime_probe.get("canonical_database_opened") is False,
                "native stopped dependency probe differs")
        require(self.source["head"] == source_head and self.source["tree"] == source_tree,
                "stopped capture source differs")
        self.hold_bytes = native._bytes(self.hold, 65536)
        require(hashlib.sha256(self.hold_bytes).hexdigest() == hold_sha256,
                "stopped capture hold differs")
        self.unit = workflow.unit_snapshot(native.UNIT)
        self.unit_files = _files(native.UNIT)
        self.controller_birth, self.helper_birth = dict(controller_birth), dict(helper_birth)
        _exact_birth(self.controller_birth)
        _exact_birth(self.helper_birth)
        require(helper_birth["parent_pid"] == controller_birth["pid"]
                and _argv(controller_birth["pid"]) == list(controller_argv),
                "failed controller/helper relationship differs")
        require(_argv(helper_birth["pid"]) == ["/usr/bin/python3", "-I", "-S", "-c", native.SENTINEL_CODE],
                "prior workflow helper is not the exact no-state keeper")
        require(os.readlink(f"/proc/{helper_birth['pid']}/cwd") == "/"
                and len(list(Path(f"/proc/{helper_birth['pid']}/fd").iterdir())) == 3,
                "prior workflow helper has state descriptors")
        self.controller_argv = list(controller_argv)
        self.fleet_config = Path(fleet_config).absolute()
        self.fleet_bytes = native._bytes(self.fleet_config)
        fleet = role._decode(self.fleet_bytes)
        require(fleet.get("schema") == "agent-supervisor/fleet-watchdog-config@1",
                "stopped recovery fleet configuration differs")
        fleet_root = Path(fleet["state_dir"])
        self.old_lock_paths = [(fleet_root / "repairs/spar/queue.lock", False),
                              (fleet_root / "spar/watchdog.lock", False)] + [
            (self.queue_root / name, ofd) for name, ofd in (
                (".merge_queue.duckdb.rebuild.lock", False), (".merge_queue.duckdb.lock", False),
                ("merge_queue.duckdb", True), ("train/consumer.lock", False))]
        self.old_locks = self.controller_locks()
        # Bind the actual queue while the old controller still holds every
        # native queue fence. A successful fresh lock acquisition cannot excuse
        # a writer which entered during the handover gap and already exited.
        self.queue_inventory = role.file_inventory(self.queue_root)
        self.queue_entries = producer._entries(self.queue_root, self.queue_inventory)
        require(self.controller_locks() == self.old_locks
                and role.file_inventory(self.queue_root) == self.queue_inventory,
                "failed controller queue changed during succession inspection")
        self.journal_root = Path(journal_root).absolute()
        require(not self.journal_root.is_relative_to(self.root), "succession journal inside native checkout")
        workflow._owned_directory(self.journal_root, create=True)
        self.sequence = 0
        self.resources = ExitStack()
        self.controller_fd = os.pidfd_open(controller_birth["pid"])
        self.resources.callback(os.close, self.controller_fd)
        self.old_helper_fd = os.pidfd_open(helper_birth["pid"])
        self.resources.callback(os.close, self.old_helper_fd)
        self.cgroup = CGROUP_ROOT / self.unit["ControlGroup"].lstrip("/")
        self.cgroup_fd = role._open_directory(self.cgroup)
        self.resources.callback(os.close, self.cgroup_fd)
        self.cgroup_identity = (os.fstat(self.cgroup_fd).st_dev, os.fstat(self.cgroup_fd).st_ino)
        self.owner_path = self.paths["owner"] / "quack-state-server.status.json"
        self.owner = native._json(self.owner_path)
        self.identity = {k: self.owner["identity"][k] for k in native.IDENTITY_FIELDS}
        role._closed(owner_identity, set(native.IDENTITY_FIELDS))
        require(self.identity == owner_identity, "nominated closed native owner identity differs")
        self.bootstrap_path = self.paths["bootstrap_receipt"]
        self.bootstrap = native._json(self.bootstrap_path)
        require(self.bootstrap.get("bootstrap_receipt_id") == self.operator._identity(
            {k: v for k, v in self.bootstrap.items() if k != "bootstrap_receipt_id"}),
            "immutable native bootstrap differs")
        self.broker_path = self.paths["owner"] / "spar-bootstrap-broker.json"
        self.broker = native._json(self.broker_path)
        records = self.broker.get("current_births")
        require(type(records) is dict and set(records) == {
            f"{self.board.board_namespace}-{i}" for i in range(self.board.max_lanes)},
            "native lane population differs")
        self.lanes = tuple(record[k] for record in records.values()
                           for k in ("supervisor_process_birth", "daemon_process_birth"))
        self.keeper = self.keeper_fd = self.keeper_birth = None
        self.stage = "inspected"
        self.retirement = None
        self.retirement_recorded = False
        self.finish_recorded = False
        self.require_current({helper_birth["pid"]})

    def controller_locks(self):
        """Read actual fdinfo only; never open/release another owner's lock fd."""
        require(native._bytes(self.fleet_config) == self.fleet_bytes,
                "stopped recovery fleet configuration changed")
        found = {}
        for item in Path(f"/proc/{self.controller_birth['pid']}/fdinfo").iterdir():
            lines = [line for line in item.read_text().splitlines() if line.startswith("lock:")]
            if not lines:
                continue
            fdpath = Path(f"/proc/{self.controller_birth['pid']}/fd/{item.name}")
            target = Path(os.readlink(fdpath))
            matches = [ofd for path, ofd in self.old_lock_paths if path == target]
            require(len(matches) == 1 and str(target) not in found and len(lines) == 1,
                    "failed controller native lock population differs")
            info, current = fdpath.stat(), target.lstat()
            require(role._file_identity(info) == role._file_identity(current),
                    "failed controller native lock inode differs")
            _verify_lock_line(lines[0], info, ofd=matches[0], pid=self.controller_birth["pid"])
            found[str(target)] = {"fd": int(item.name), "device": info.st_dev, "inode": info.st_ino,
                                  "ofd": matches[0]}
        require(set(found) == {str(path) for path, _ in self.old_lock_paths},
                "failed controller native locks unavailable")
        return found

    def record(self, phase):
        self.sequence += 1
        workflow._exclusive_write(self.journal_root / f"{self.sequence:03d}-{phase}.json",
            workflow._json_bytes({"schema": KEEPER_SCHEMA, "stage": self.stage,
                                  "phase": phase, "controller": self.controller_birth,
                                  "old_helper": self.helper_birth, "keeper": self.keeper_birth,
                                  "callback_settled": False, "at": time.time()}))

    def require_current(self, population):
        require(native._bytes(self.fleet_config) == self.fleet_bytes,
                "stopped recovery fleet configuration changed")
        unit = workflow.unit_snapshot(native.UNIT)
        require(_files(native.UNIT) == self.unit_files and unit == self.unit,
                "native inhibited unit changed")
        for key, value in {"Restart": "no", "SendSIGKILL": "no", "TimeoutStopUSec": "infinity",
                           "RefuseManualStart": "yes", "Delegate": "yes", "ExitType": "cgroup",
                           "MainPID": "0", "ActiveState": "active", "SubState": "running"}.items():
            require(unit[key] == value, "native stopped-owner inhibition incomplete: " + key)
        require(["ConditionPathExists", False, True, str(self.hold)] in unit["Conditions"],
                "native hold condition unavailable")
        current = self.cgroup.lstat()
        require((current.st_dev, current.st_ino) == self.cgroup_identity
                and set(native._cgroup_population(self.cgroup_fd)[0]) == set(population)
                and native._cgroup_population(self.cgroup_fd)[1] == "1",
                "native keeper population changed")
        require(native._bytes(self.hold, 65536) == self.hold_bytes
                and self.operator.source_binding(self.config) == self.source
                and native._json(self.bootstrap_path) == self.bootstrap
                and native._json(self.broker_path) == self.broker,
                "native hold/source/bootstrap/lane census changed")
        current_owner = native._json(self.owner_path)
        require(current_owner == self.owner and current_owner.get("lifecycle") == "stopped"
                and native._dead(self.identity["process_birth"])
                and all(native._dead(b) for b in self.lanes), "native owner/lane closure unavailable")
        return {"unit": unit, "hold_sha256": hashlib.sha256(self.hold_bytes).hexdigest(),
                "owner_status_cid": role._cid(current_owner), "native_actors_closed": True,
                "unit_files": self.unit_files, "cgroup_identity": list(self.cgroup_identity),
                "workflow_keeper": self.keeper_birth, "observed_cgroup_members": sorted(population),
                "callback_settled": False}

    def overlap(self):
        require(self.stage == "inspected", "keeper succession already used")
        _exact_birth(self.controller_birth)
        _exact_birth(self.helper_birth)
        self.require_current({self.helper_birth["pid"]})
        require(self.controller_locks() == self.old_locks, "failed controller locks changed")
        require(probe_task_runtime(self) == self.runtime_probe, "native stopped dependency probe changed")
        self.record("keeper-overlap-prepared")
        self.stage = "overlapping"
        self.keeper_resources = ExitStack()
        self.resources.callback(self.keeper_resources.close)
        self.keeper = subprocess.Popen(["/usr/bin/python3", "-I", "-S", "-c", native.SENTINEL_CODE],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd="/", env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}, close_fds=True)
        self.keeper_birth = _birth(self.keeper.pid)
        self.keeper_fd = os.pidfd_open(self.keeper.pid)
        self.keeper_resources.callback(os.close, self.keeper_fd)
        poller = select.poll(); poller.register(self.keeper.stdout.fileno(), select.POLLIN)
        require(poller.poll(5000) and os.read(self.keeper.stdout.fileno(), 128) == b"SPAR-CAPTURE-SENTINEL-READY\n",
                "new no-state keeper not ready")
        subprocess.run(["busctl", "--user", "call", "org.freedesktop.systemd1", "/org/freedesktop/systemd1",
            "org.freedesktop.systemd1.Manager", "AttachProcessesToUnit", "ssau", native.UNIT, "", "1", str(self.keeper.pid)],
            capture_output=True, check=True, timeout=10)
        self.require_current({self.helper_birth["pid"], self.keeper.pid})
        self.stage = "overlapped"
        self.record("keeper-overlap-observed")
        return self.keeper_birth

    def abort_overlap(self):
        """Release only our new helper while the prior custody is still proven."""
        require(self.stage in {"overlapping", "overlapped"}, "keeper overlap cannot be aborted after retirement")
        _exact_birth(self.controller_birth); _exact_birth(self.helper_birth)
        require(self.controller_locks() == self.old_locks, "prior kernel custody changed")
        if self.keeper is None:
            self.require_current({self.helper_birth["pid"]})
            self.keeper_resources.close()
            self.stage = "inspected"
            self.record("keeper-overlap-aborted")
            return
        population = set(native._cgroup_population(self.cgroup_fd)[0])
        require(population in ({self.helper_birth["pid"]}, {self.helper_birth["pid"], self.keeper.pid}),
                "unknown native population prevents helper rollback")
        self.require_current(population)
        self.keeper.stdin.close(); self.keeper.wait(timeout=10)
        require(self.keeper.returncode == 0 and native._exited(self.keeper_fd),
                "new no-state helper normal exit unavailable")
        self.require_current({self.helper_birth["pid"]})
        self.keeper_resources.close()
        for stream in (self.keeper.stdout, self.keeper.stderr): stream.close()
        self.keeper = self.keeper_fd = self.keeper_birth = None
        self.stage = "inspected"
        self.record("keeper-overlap-aborted")

    def retire(self, *, timeout=30):
        require(self.stage == "overlapped" and 0 < timeout <= 60, "failed workflow retirement not admitted")
        self.require_current({self.helper_birth["pid"], self.keeper.pid})
        _exact_birth(self.controller_birth); _exact_birth(self.helper_birth)
        require(_argv(self.controller_birth["pid"]) == self.controller_argv
                and not native._exited(self.keeper_fd), "failed workflow identity changed")
        for marker in ("native-legacy-profile-required.json", "native-stopped-profile-required.json",
                       ".native-legacy-database.prepared", ".native-stopped-database.prepared"):
            workflow._absent(self.queue_root / marker)
        self.record("failed-controller-graceful-exit-prepared")
        self.require_current({self.helper_birth["pid"], self.keeper.pid})
        _exact_birth(self.controller_birth)
        require(self.controller_locks() == self.old_locks, "failed controller locks changed")
        signal.pidfd_send_signal(self.controller_fd, signal.SIGTERM)
        self.stage = "retiring"
        return self.complete_retirement(timeout=timeout)

    def complete_retirement(self, *, timeout=1):
        require(self.stage in {"retiring", "retired"} and 0 < timeout <= 60,
                "workflow retirement was not previously admitted")
        _exact_birth(self.keeper_birth)
        require(not native._exited(self.keeper_fd), "new workflow keeper exited during retirement")
        _wait_pidfd(self.controller_fd, timeout)
        _wait_pidfd(self.old_helper_fd, timeout)
        self.require_current({self.keeper.pid})
        self.stage = "retired"
        self.retirement = {"schema": KEEPER_SCHEMA, "old_controller": self.controller_birth,
            "old_helper": self.helper_birth, "old_controller_pidfd_exit": True,
            "old_helper_pidfd_exit": True, "keeper": self.keeper_birth, "retired_kernel_locks": self.old_locks,
            "keeper_code_sha256": hashlib.sha256(native.SENTINEL_CODE.encode()).hexdigest(),
            "callback_settled": False, "task_or_store_authority": False}
        if not self.retirement_recorded:
            self.record("failed-workflow-exit-observed")
            self.retirement_recorded = True
        return dict(self.retirement)

    def closed_gate(self):
        require(self.stage == "retired" and self.retirement is not None and self.retirement_recorded
                and native._exited(self.controller_fd) and native._exited(self.old_helper_fd),
                "failed workflow closure not retained")
        _exact_birth(self.keeper_birth)
        require(not native._exited(self.keeper_fd), "new workflow keeper exited")
        return self.require_current({self.keeper.pid})

    def close_after_install(self):
        require(self.stage in {"retired", "closing", "finished"}, "keeper closure was not admitted")
        if self.stage == "retired":
            self.closed_gate()
            self.stage = "closing"
        if self.stage == "closing":
            if not self.keeper.stdin.closed:
                self.keeper.stdin.close()
            self.keeper.wait(timeout=10)
            require(self.keeper.returncode == 0 and native._exited(self.keeper_fd), "keeper normal exit unavailable")
            self.stage = "finished"
        if not self.finish_recorded:
            require(self.keeper.returncode == 0 and native._exited(self.keeper_fd), "keeper normal exit was not retained")
            self.record("keeper-normal-exit-observed")
            self.finish_recorded = True
        self.resources.close()


class StoppedCaptureSession:
    """Fresh real local fences plus native closed-copy facts, never old JSON."""

    def __init__(self, succession, *, fleet_config):
        require(type(succession) is KeeperSuccession, "retained keeper succession required")
        succession.closed_gate()
        require(Path(fleet_config).absolute() == succession.fleet_config,
                "fresh stopped fences require the inspected fleet configuration")
        self.succession = succession
        self.root, self.queue_root = succession.root, succession.queue_root
        self.resources = ExitStack()
        self.locks = []
        self._closed = False
        self._capture = None
        self.fleet = workflow.FleetCaptureExclusion(fleet_config, self.root, succession.hold)
        self.resources.callback(self.fleet.close)
        try:
            database = succession.paths["database"]
            for path in [database.with_name("." + database.name + suffix)
                         for suffix in (".lock", ".state-owner.lock", ".intent.lock", ".migration.lock")]:
                self._lock(path)
            for name in (".merge_queue.duckdb.rebuild.lock", ".merge_queue.duckdb.lock", "train/consumer.lock"):
                self._lock(self.queue_root / name)
            self._lock(database, ofd=True)
            self._lock(self.queue_root / "merge_queue.duckdb", ofd=True)
            self._closed_gate()
            require(role.file_inventory(self.queue_root) == succession.queue_inventory,
                    "canonical queue changed across workflow handover")
            for entry in succession.queue_entries:
                role.copy_entry(self.queue_root, entry, None, digest_only=True)
            self._closed_gate()
        except BaseException:
            self.resources.close()
            self._closed = True
            raise

    def _lock(self, path, *, ofd=False):
        parent = role._open_directory(path.parent)
        try:
            fd = os.open(path.name, (os.O_RDWR if ofd else os.O_RDONLY) | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                         dir_fd=parent)
        finally:
            os.close(parent)
        self.resources.callback(os.close, fd)
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid() and before.st_nlink == 1,
                "native store fence is not an owned unique regular inode")
        if ofd:
            fcntl.fcntl(fd, fcntl.F_OFD_SETLK, struct.pack("hhqqi", fcntl.F_WRLCK, os.SEEK_SET, 0, 0, 0))
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.locks.append((path, fd, (before.st_dev, before.st_ino), ofd))

    def _retain_replacement_queue(self, staging):
        require(getattr(self, "_replacement", None) is None, "replacement writer already retained")
        parent = role._open_directory(staging.parent)
        try:
            fd = os.open(staging.name, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        finally:
            os.close(parent)
        self.resources.callback(os.close, fd)
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and info.st_nlink == 1
                and role._file_identity(info) == role._file_identity(staging.lstat()),
                "replacement queue inode differs")
        fcntl.fcntl(fd, fcntl.F_OFD_SETLK, struct.pack("hhqqi", fcntl.F_WRLCK, os.SEEK_SET, 0, 0, 0))
        self._replacement = (staging, fd, (info.st_dev, info.st_ino))

    def _closed_gate(self):
        require(not self._closed and len(self.locks) == 9, "stopped-state native fences are not retained")
        self.fleet.require_current()
        closure = self.succession.closed_gate()
        for path, fd, identity, ofd in self.locks:
            old = os.fstat(fd); now = path.lstat()
            # Queue inode is replaced only by the distinct installer while old
            # writer fd and every native queue lock remain held in this session.
            replaced = getattr(self, "_installed_queue_inode", None)
            is_installed = path == self.queue_root / "merge_queue.duckdb" and replaced is not None
            require((old.st_dev, old.st_ino) == identity
                    and ((now.st_dev, now.st_ino) == (replaced if is_installed else identity)),
                    "native retained fence inode changed")
            lines = [line for line in Path(f"/proc/self/fdinfo/{fd}").read_text().splitlines()
                     if line.startswith("lock:")]
            require(len(lines) == 1, "native retained kernel fence missing")
            _verify_lock_line(lines[0], old, ofd=ofd, pid=os.getpid())
        if getattr(self, "_replacement", None) is not None:
            staging, fd, identity = self._replacement
            path = self.queue_root / "merge_queue.duckdb" if getattr(self, "_installed_queue_inode", None) else staging
            info = os.fstat(fd); current = path.lstat()
            require((info.st_dev, info.st_ino) == identity == (current.st_dev, current.st_ino),
                    "replacement queue writer inode changed")
            lines = [line for line in Path(f"/proc/self/fdinfo/{fd}").read_text().splitlines() if line.startswith("lock:")]
            require(len(lines) == 1, "replacement queue writer fence missing")
            _verify_lock_line(lines[0], info, ofd=True, pid=os.getpid())
        if hasattr(self, "_task_inventory"):
            require(_task_inventory(self.succession.paths["database"]) == self._task_inventory,
                    "canonical task database or WAL changed under fresh fences")
        return closure

    def capture(self, *, destination):
        require(self._capture is None, "stopped capture already produced")
        closure = self._closed_gate()
        destination = Path(destination).absolute()
        require(not destination.is_relative_to(self.root), "stopped capture destination inside canonical checkout")
        workflow._owned_directory(destination, create=True)
        queue_copy = destination / "queue"
        workflow._owned_directory(queue_copy, create=True)
        identities = role.file_inventory(self.queue_root)
        require(identities == self.succession.queue_inventory,
                "canonical queue changed since stopped succession admission")
        entries = producer._entries(self.queue_root, identities)
        require(entries == self.succession.queue_entries,
                "canonical queue content changed since stopped succession admission")
        for entry in entries:
            require(self._closed_gate() == closure, "stopped capture closure changed")
            role.copy_entry(self.queue_root, entry, queue_copy / entry["path"])
        database = self.succession.paths["database"]
        task_copy = destination / "task-original"
        workflow._owned_directory(task_copy, create=True)
        self._task_inventory = _task_inventory(database)
        task_paths = [database.parent / name for name in self._task_inventory]
        task_files = []
        for path in task_paths:
            self._closed_gate()
            entries_for_task = producer._entries(path.parent, {path.name: role._file_identity(path.lstat())})
            entry = entries_for_task[0]
            role.copy_entry(path.parent, entry, task_copy / path.name)
            task_files.append(entry)
        inspect = destination / "task-inspection"
        workflow._owned_directory(inspect, create=True)
        for entry in task_files:
            role.copy_entry(task_copy, entry, inspect / entry["path"])
        observed = observe_task_copy(self.succession, inspect / database.name, task_files=task_files)
        admission = validate_task_observation(observed, owner=self.succession.owner,
            bootstrap=self.succession.bootstrap, source=self.succession.source, task_files=task_files)
        context = {"repository_id": native._repository_id(self.root),
                   "target_branch": str(self.succession.board.payload.get("merge_target_branch") or ""),
                   "store_id": str(self.queue_root / "merge_queue.duckdb"),
                   "source_commit": self.succession.source["head"], "source_tree": self.succession.source["tree"],
                   "scope_bindings": _scopes(board=self.succession.board, paths=self.succession.paths,
                       amendment=SimpleNamespace(launch_config_cid="sha256:" + self.succession.source["config_sha256"],
                           bootstrap_plan_root_cid=self.succession.bootstrap["plan_root_cid"])),
                   "queue_policy": dict(role.DEFAULT_QUEUE_POLICY)}
        plan = producer.produce_offline_import_plan(offline_root=queue_copy,
                    destination=destination / "queue-inspection", context=context)
        require(plan["manifest"]["files"] == entries and role.file_inventory(self.queue_root) == identities,
                "stopped queue inventory changed")
        for entry in entries: role.copy_entry(self.queue_root, entry, None, digest_only=True)
        for entry in task_files: role.copy_entry(database.parent, entry, None, digest_only=True)
        require(self._closed_gate() == closure, "stopped capture closure changed after native observation")
        receipt = {"schema": SCHEMA, "queue_root": str(self.queue_root), "owner_identity": self.succession.identity,
                   "source": self.succession.source, "closure": closure, "keeper_succession": self.succession.retirement,
                   "task_admission": admission, "manifest": plan["manifest"],
                   "preserved_inventory_cid": "sha256:" + hashlib.sha256(role._json(plan["preserved_inventory"])).hexdigest(),
                   "capture_coherent": True, "consumer_processes_closed": True,
                   "callback_settled": False, "signing_authority": False, "source_admitted": False,
                   "completion_authority": False}
        raw_receipt = role._json(receipt)
        require(len(raw_receipt) <= role.MAX_JSON_BYTES, "fresh stopped receipt exceeds installed-origin bound")
        workflow._exclusive_write(destination / "stopped-capture-receipt.json", raw_receipt)
        self._capture = StoppedQueueCapture(self, queue_copy, receipt, identities, task_copy, task_files)
        return self._capture


_FROZEN_WORKER_BOOTSTRAP = "import hashlib,sys;path=sys.argv.pop(1);raw=sys.stdin.buffer.read(1048577);assert len(raw)<=1048576;exec(compile(raw,path,'exec'),{'__name__':'__main__','__file__':path,'_SPAR_STOPPED_WORKER_SHA256':hashlib.sha256(raw).hexdigest()})"


def _task_worker(succession, *, path=None, probe=False, task_files=None):
    require(native._bytes(succession.worker_path, 1024 * 1024) == succession.worker_bytes,
            "retained stopped task worker code changed")
    args = ["/usr/bin/python3", "-I", "-B", "-c", _FROZEN_WORKER_BOOTSTRAP, str(succession.worker_path),
        "--root", str(succession.root), "--config", str(succession.config),
        "--runtime-manifest", succession.task_runtime["manifest_path"],
        "--runtime-sha256", succession.task_runtime["manifest_sha256"],
        "--runtime-helper-sha256", succession.task_runtime["helper_sha256"]]
    with ExitStack() as inputs:
        bindings = []
        descriptors = []
        if probe:
            args += ["--probe"]
        else:
            require(type(task_files) is list and len(task_files) in (1, 2),
                    "retained task input inventory required")
            for entry in task_files:
                descriptor = role._open_regular(Path(path).parent, entry["path"])
                inputs.callback(os.close, descriptor)
                info = os.fstat(descriptor)
                require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and info.st_nlink == 1,
                        "retained task input is not an owned unique inode")
                bindings.append({"file": entry, "device": info.st_dev, "inode": info.st_ino,
                                 "descriptor": descriptor})
                descriptors.append((descriptor, entry["path"], role._file_identity(info)))
            args += ["--copy", str(path), "--input-manifest", role._json({"files": bindings}).decode()]
        result = subprocess.run(args, cwd="/", input=succession.worker_bytes, capture_output=True,
            timeout=120, check=False, pass_fds=tuple(fd for fd, _, _ in descriptors),
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
        require(result.returncode == 0 and len(result.stdout) <= 16 * 1024 * 1024,
                "native stopped task copy observation failed")
        for descriptor, name, before in descriptors:
            require(role._file_identity(os.fstat(descriptor)) == before
                    == role._file_identity((Path(path).parent / name).lstat()),
                    "retained task input pathname or bytes changed during observation")
        require(native._bytes(succession.worker_path, 1024 * 1024) == succession.worker_bytes,
                "retained stopped task worker code changed")
        value = json.loads(result.stdout)
        require(value.get("runtime", {}).get("worker_sha256") == succession.worker_sha256,
                "closed observation did not execute retained worker code")
        expected = [{k: v for k, v in item.items() if k != "descriptor"} for item in bindings]
        require(probe or value.get("input_binding") == expected,
                "closed observation does not bind the retained task input files")
        return value


def probe_task_runtime(succession):
    return _task_worker(succession, probe=True)


def observe_task_copy(succession, path, *, task_files):
    value = _task_worker(succession, path=path, task_files=task_files)
    require(value.get("runtime") == succession.runtime_probe.get("runtime"),
            "native closed observation changed its dependency runtime")
    return value


def validate_task_observation(value, *, owner, bootstrap, source, task_files):
    require(type(value) is dict and value.get("schema") == TASK_SCHEMA
            and value.get("source") == source and value.get("bootstrap") == bootstrap,
            "native stopped task source/bootstrap observation differs")
    require(all(value.get(k) is False for k in ("completion_authority", "callback_settled",
                "launch_amendment_admitted", "canonical_database_opened")), "closed observation claimed authority")
    require(type(owner) is dict and type(owner.get("identity")) is dict
            and type(bootstrap) is dict and bootstrap.get("bootstrap_receipt_id") == role._cid(
                {k: v for k, v in bootstrap.items() if k != "bootstrap_receipt_id"}),
            "closed owner or immutable bootstrap seal differs")
    require(type(task_files) is list and len(task_files) in (1, 2)
            and task_files[0].get("path") == "control.duckdb"
            and (len(task_files) == 1 or task_files[1].get("path") == "control.duckdb.wal"),
            "canonical task copy inventory differs")
    for entry in task_files:
        require(set(entry) == {"path", "size_bytes", "sha256"}
                and type(entry["size_bytes"]) is int and 0 < entry["size_bytes"] <= role.MAX_INPUT_BYTES
                and type(entry["sha256"]) is str and len(entry["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in entry["sha256"]),
                "canonical task copy digest differs")
    bindings = value.get("input_binding")
    require(type(bindings) is list and len(bindings) == len(task_files)
            and [item.get("file") for item in bindings] == task_files
            and all(set(item) == {"file", "device", "inode"}
                    and type(item["device"]) is int and type(item["inode"]) is int
                    and item["device"] > 0 and item["inode"] > 0 for item in bindings),
            "closed native facts do not bind the canonical task copy digests")
    identity = owner["identity"]
    require(all(type(identity.get(k)) is int and identity[k] > 0 for k in (
                "generation", "schema_revision", "fence_epoch", "startup_epoch")),
            "native closed identity integers differ")
    require(owner.get("lifecycle") == "stopped" and identity.get("status") == "stopped",
            "native owner did not record stopped state")
    generations, servers, epochs = (value.get(k) for k in ("store_generations", "state_servers", "server_epochs"))
    require(all(type(rows) is list and 0 < len(rows) <= 2 for rows in (generations, servers, epochs)),
            "canonical stopped identity rows unavailable")
    generation, server, epoch = generations[0], servers[0], epochs[0]
    require(all(type(generation.get(k)) is int for k in ("generation", "schema_revision", "fence_epoch"))
            and all(generation.get(k) == identity.get(k) for k in ("generation", "schema_revision", "fence_epoch", "database_uuid"))
            and generation.get("birth_id") == identity["process_birth_id"], "canonical stopped generation differs")
    require(all(type(server.get(k)) is int for k in ("generation", "schema_revision"))
            and all(server.get(k) == identity.get(k) for k in ("server_id", "store_id", "database_uuid", "process_birth_id", "schema_revision", "generation"))
            and server.get("status") == "stopped" and type(server.get("stopped_at")) is str and server["stopped_at"],
            "canonical server has not closed this exact generation")
    require(all(type(epoch.get(k)) is int for k in ("epoch", "fence_epoch"))
            and epoch.get("server_id") == identity["server_id"] and epoch.get("fence_epoch") == identity["fence_epoch"]
            and epoch.get("epoch") == identity["startup_epoch"]
            and type(epoch.get("ended_at")) is str and epoch["ended_at"], "canonical server epoch remains open")
    facts = value.get("facts", {})
    from ipfs_accelerate_py.agent_supervisor.task_sources.closeout_snapshot import RELATIONS
    require(type(facts) is dict and facts.get("all_relations_available") is True and facts.get("truncated") is False
            and set(facts.get("relations", {})) == set(RELATIONS), "native stopped facts incomplete")
    for relation in facts["relations"].values():
        require(type(relation) is dict and relation.get("available") is True and relation.get("truncated") is False
                and type(relation.get("rows")) is list and len(relation["rows"]) <= 512,
                "native stopped relation unavailable or truncated")
    require(facts["relations"]["leases"]["rows"] == [], "native stopped store retains active or unknown leases")
    task_cids = bootstrap["database_task_source_receipt"]["task_cids"]
    require(type(value.get("task_cids")) is list and len(value["task_cids"]) == len(set(value["task_cids"]))
            and set(value["task_cids"]) == set(task_cids)
            and {row["task_cid"] for row in facts["relations"]["tasks"]["rows"]} == set(task_cids),
            "stopped canonical task population differs")
    plan = value.get("plan", {})
    revisions = value.get("plan_revisions")
    require(plan.get("plan_cid") == bootstrap["plan_root_cid"] and plan.get("status") == "active"
            and type(plan.get("revision")) is int and type(revisions) is list and len(revisions) == plan["revision"]
            and all(row.get("plan_cid") == plan["plan_cid"] and row.get("revision") == i
                    for i, row in enumerate(revisions, 1))
            and revisions[-1].get("body") == plan.get("body")
            and type(plan.get("body")) is dict
            and all(plan["body"].get(key) == bootstrap.get(binding) for key, binding in (
                ("plan_cid", "plan_root_cid"), ("source_head", "source_head"),
                ("repository_tree_id", "repository_tree_id"))), "stopped native plan lineage differs")
    body = {"schema": ADMISSION_SCHEMA, "observation": value, "canonical_task_files": task_files,
            "owner_identity": {k: identity[k] for k in native.IDENTITY_FIELDS}, "closed_owner": owner,
            "native_lineage_verified": True, "completion_authority": False, "callback_settled": False}
    return {**body, "admission_cid": evidence_cid(body)}


class StoppedQueueCapture:
    def __init__(self, session, path, receipt, identities, task_copy, task_files):
        self._session, self.path = session, path
        self._receipt, self._identities = json.loads(role._json(receipt)), identities
        self.task_copy, self.task_files = task_copy, task_files

    @property
    def receipt(self):
        return json.loads(role._json(self._receipt))

    def require_current(self):
        require(type(self._session) is StoppedCaptureSession and self._session._capture is self,
                "stopped capture is not retained by its fresh producer")
        require(self._session._closed_gate() == self.receipt["closure"], "fresh stopped custody changed")
        require(native._bytes(self.path.parent / "stopped-capture-receipt.json", role.MAX_JSON_BYTES)
                == role._json(self.receipt), "fresh immutable stopped receipt changed")
        require(role.file_inventory(self._session.queue_root) == self._identities,
                "canonical queue changed after stopped capture")
        for entry in self.receipt["manifest"]["files"]: role.copy_entry(self.path, entry, None, digest_only=True)
        for entry in self.task_files:
            role.copy_entry(self.task_copy, entry, None, digest_only=True)
            role.copy_entry(self._session.succession.paths["database"].parent, entry, None, digest_only=True)
        return self.receipt
