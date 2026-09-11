"""Retain native process evidence across an independently authorized SPAR stop.

This module neither signals processes nor changes units, markers, source, or a
live database. A session must be armed against the actual native owner before
it exits. JSON reports cannot recreate a session. Its copied queue is still
not callback settlement, signing authority, source admission, or completion.
"""

from __future__ import annotations

from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import stat
import struct
import subprocess

from . import spar_merge_owner as role
from . import spar_legacy_import_plan as producer
from .spar_merge_owner_handoff import configured_queue_root

SCHEMA = "spar/native-legacy-closed-capture@1"
UNIT = "ipfs-taskboard-spar-supervisor.service"
IDENTITY_FIELDS = (
    "server_id", "process_birth_id", "process_birth", "store_id",
    "database_uuid", "generation", "fence_epoch",
)
UNIT_FIELDS = (
    "Id", "LoadState", "ActiveState", "SubState", "MainPID", "ControlGroup",
    "WorkingDirectory", "Restart", "SendSIGKILL",
    "TimeoutStopUSec", "RefuseManualStart", "NeedDaemonReload",
)


def _require(value, reason):
    if not value:
        raise role.SparMergeOwnerError(reason)


def _bytes(path, limit=4 * 1024 * 1024):
    path = Path(path).absolute()
    fd = role._open_regular(path.parent, path.name)
    try:
        before = os.fstat(fd)
        _require(before.st_size <= limit, "native public record exceeds bound")
        body = os.read(fd, limit + 1)
        _require(
            len(body) == before.st_size
            and role._file_identity(os.fstat(fd)) == role._file_identity(before),
            "native public record changed",
        )
        return body
    finally:
        os.close(fd)


def _json(path):
    return role._decode(_bytes(path))


def _unit():
    result = subprocess.run(
        ["systemctl", "--user", "show", UNIT, "--no-pager",
         "--property=" + ",".join(UNIT_FIELDS)],
        capture_output=True, text=True, check=True, timeout=10,
    )
    fields = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        _require(separator and key not in fields, "native unit response ambiguous")
        fields[key] = value
    _require(set(fields) == set(UNIT_FIELDS), "native unit response incomplete")
    _require(fields["Id"] == UNIT and fields["LoadState"] == "loaded",
             "native unit is unavailable")
    _require(fields["NeedDaemonReload"] == "no", "native unit reload pending")
    for interface, name, signature in (
        ("Unit", "Conditions", "a(sbbsi)"),
        ("Service", "ExecStart", "a(sasbttttuii)"),
    ):
        result = subprocess.run(
            ["busctl", "--user", "--json=short", "get-property", "org.freedesktop.systemd1",
             "/org/freedesktop/systemd1/unit/ipfs_2dtaskboard_2dspar_2dsupervisor_2eservice",
             "org.freedesktop.systemd1." + interface, name],
            capture_output=True, text=True, check=True, timeout=10,
        )
        decoded = json.loads(result.stdout)
        _require(decoded.get("type") == signature and type(decoded.get("data")) is list,
                 "native typed unit property unavailable")
        # Runtime start/stop timestamps are observations, not executable policy.
        fields[name] = ([record[:3] for record in decoded["data"]]
                        if name == "ExecStart" else [record[:4] for record in decoded["data"]])
    return fields


def _native_operator(root):
    from .spar_legacy_observation import NativeObservationClient, OPERATOR

    _bytes(root / OPERATOR, 4 * 1024 * 1024)
    return NativeObservationClient(root)


def _namespaces(pid):
    result = {}
    for name in ("pid", "mnt", "net"):
        try:
            result[name] = {"state": "observed", "value": os.readlink(f"/proc/{pid}/ns/{name}")}
        except OSError as exc:
            result[name] = {"state": "unknown", "error_class": type(exc).__name__}
    return result


def _dead(birth):
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness, ProcessBirthIdentity, owner_liveness,
    )
    return owner_liveness(ProcessBirthIdentity.from_dict(birth)) is OwnerLiveness.DEAD


def _alive(birth):
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        OwnerLiveness, ProcessBirthIdentity, owner_liveness,
    )
    return owner_liveness(ProcessBirthIdentity.from_dict(birth)) is OwnerLiveness.ALIVE


def _exited(fd):
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    events = poller.poll(0)
    return any(flags & select.POLLIN for _, flags in events)


def _cgroup_read(directory, name):
    fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
                 dir_fd=directory)
    try:
        body = os.read(fd, 1024 * 1024 + 1)
        _require(len(body) <= 1024 * 1024, "native cgroup response exceeds bound")
        return body.decode("ascii")
    finally:
        os.close(fd)


def _cgroup_population(directory):
    members = _cgroup_read(directory, "cgroup.procs").split()
    _require(all(re.fullmatch(r"[1-9][0-9]*", value) for value in members),
             "native cgroup member identity malformed")
    events = dict(line.split() for line in _cgroup_read(directory, "cgroup.events").splitlines())
    _require(events.get("populated") in {"0", "1"}, "native cgroup population unavailable")
    return tuple(int(value) for value in members), events["populated"]


class RetainedNativeLegacySession:
    """In-process retained pidfd/cgroup/lock custody; never reconstructed from JSON."""

    def __init__(self, *, repository_root, config_path, expected_source_commit,
                 expected_source_tree):
        self._resources = ExitStack()
        self._closed = False
        self._capture = None
        self.root = Path(repository_root).absolute()
        self.config_path = Path(config_path).absolute()
        self._empty_cgroup_observed = False
        try:
            self.operator = _native_operator(self.root)
            self.board, self.config = self.operator._load_config(self.config_path)
            _require(Path(self.board.repo_root).absolute() == self.root,
                     "native repository binding differs")
            self.paths = self.operator._runtime_paths(self.board)
            self.queue_root = configured_queue_root(self.board)
            self.hold_path = self.board.path(self.board.runtime_paths["root"]) / "HOLD"
            self._source = self._source_binding()
            _require(self._source["head"] == expected_source_commit
                     and self._source["tree"] == expected_source_tree,
                     "current native source differs from reviewed source")
            self._unit_before = _unit()
            _require(self._unit_before["WorkingDirectory"] == str(self.root)
                     and self._unit_before["ActiveState"] == "active"
                     and self._unit_before["SubState"] == "running"
                     and self._unit_before["ExecStart"] == [["/usr/bin/python3", [
                         "/usr/bin/python3", "scripts/materialize_semantic_preserving_remodularization_program.py",
                         "supervise", "--implement"], False]],
                     "native owner unit is not running at the current root")
            self.owner_path = self.paths["owner"] / "quack-state-server.status.json"
            self._owner_before = _json(self.owner_path)
            _require(self._owner_before.get("lifecycle") == "ready", "native owner not ready")
            self.identity = {key: self._owner_before["identity"][key] for key in IDENTITY_FIELDS}
            self.birth = self.identity["process_birth"]
            self.pid = self.birth["pid"]
            _require(type(self.pid) is int and self.pid > 1
                     and self._unit_before["MainPID"] == str(self.pid)
                     and _alive(self.birth), "native owner birth differs from unit")
            self.pidfd = os.pidfd_open(self.pid, 0)
            self._resources.callback(os.close, self.pidfd)
            _require(_alive(self.birth) and not _exited(self.pidfd), "native owner changed during retain")
            self.namespaces = _namespaces(self.pid)
            group = self._unit_before["ControlGroup"]
            _require(group.startswith("/") and ".." not in Path(group).parts and group != "/",
                     "native cgroup path invalid")
            self.cgroup_path = Path("/sys/fs/cgroup") / group.lstrip("/")
            self.cgroup_fd = role._open_directory(self.cgroup_path)
            self._resources.callback(os.close, self.cgroup_fd)
            _require(_cgroup_population(self.cgroup_fd) == ((self.pid,), "1"),
                     "native unit still has another process")
            self._broker_path = self.paths["owner"] / "spar-bootstrap-broker.json"
            self._broker = _json(self._broker_path)
            _require(self._broker.get("controller_pid") == self.pid
                     and self._broker.get("schema") == "spar/state-owner-bootstrap-broker@1"
                     and self._broker.get("failure") == "", "native broker unavailable or failed")
            sessions = {f"{self.board.board_namespace}-{index}" for index in range(self.board.max_lanes)}
            records = self._broker["current_births"]
            _require(type(records) is dict and set(records) == sessions,
                     "native lane census incomplete")
            self.lane_births = tuple(record[field] for record in records.values()
                                     for field in ("supervisor_process_birth", "daemon_process_birth"))
            _require(all(_dead(birth) for birth in self.lane_births), "native lane custody not closed")
            snapshot = self.operator.authoritative_status(self.config_path)
            _require(snapshot.get("authoritative_task_observation") is True
                     and snapshot.get("board_namespace") == self.board.board_namespace
                     and {key: snapshot["owner_identity"][key] for key in IDENTITY_FIELDS} == self.identity
                     and snapshot.get("leases") == [], "native snapshot differs or retains active leases")
            facts = snapshot["closeout_snapshot"]["closeout_facts"]
            _require(facts.get("truncated") is False and facts.get("all_relations_available") is True,
                     "native closeout observation incomplete")
            from .spar_legacy_observation import native_observation_cid

            self.snapshot_cid = native_observation_cid(snapshot)
            _require(self._source_binding() == self._source and _unit() == self._unit_before
                     and _json(self._broker_path) == self._broker and _alive(self.birth),
                     "native admission changed during retain")
        except BaseException:
            self.close()
            raise

    def _source_binding(self):
        binding = self.operator.source_binding(self.config_path)
        _require(binding["config_sha256"] == hashlib.sha256(_bytes(self.config_path)).hexdigest(),
                 "native configuration changed after source observation")
        return binding

    def _inhibition(self):
        unit = _unit()
        for key, expected in {"Restart": "no", "SendSIGKILL": "no",
                              "TimeoutStopUSec": "infinity", "RefuseManualStart": "yes"}.items():
            _require(unit[key] == expected, "native capture unit inhibition missing: " + key)
        _require(unit["WorkingDirectory"] == self._unit_before["WorkingDirectory"]
                 and unit["ExecStart"] == self._unit_before["ExecStart"],
                 "native executable changed before capture")
        condition = ["ConditionPathExists", False, True, str(self.hold_path)]
        _require(condition in unit["Conditions"],
                 "native unit lacks its negative retained hold condition")
        hold = _bytes(self.hold_path, 65536)
        _require(hold, "native capture hold is empty")
        return unit, hashlib.sha256(hold).hexdigest()

    def _closed_gate(self):
        _require(not self._closed, "native retained session already closed")
        unit, hold_digest = self._inhibition()
        _require(_exited(self.pidfd), "native retained pidfd has not exited")
        _require(unit["MainPID"] == "0" and unit["ActiveState"] == "inactive"
                 and unit["SubState"] == "dead", "native unit is not positively stopped")
        # A removed or unreadable cgroup is not a population observation.
        if not self._empty_cgroup_observed:
            _require(_cgroup_population(self.cgroup_fd) == ((), "0"),
                     "native cgroup closure unavailable")
            self._empty_cgroup_observed = True
        status = _json(self.owner_path)
        _require(status.get("lifecycle") == "stopped"
                 and {key: status["identity"][key] for key in IDENTITY_FIELDS} == self.identity,
                 "native owner did not publish its own unchanged closed identity")
        _require(all(_dead(birth) for birth in self.lane_births), "native lane birth closure changed")
        _require(self._source_binding() == self._source and _json(self._broker_path) == self._broker,
                 "native source or lane census changed during closure")
        return {"unit": unit, "hold_sha256": hold_digest,
                "owner_status_cid": role._cid(status), "pidfd_exit_observed": True,
                "cgroup_empty_observed": True}

    def observe_closure(self):
        """Nonblocking observation; callers may poll while graceful cleanup runs."""
        self._inhibition()
        if _exited(self.pidfd):
            try:
                if _cgroup_population(self.cgroup_fd) == ((), "0"):
                    self._empty_cgroup_observed = True
            except OSError:
                if not self._empty_cgroup_observed:
                    raise role.SparMergeOwnerError("native cgroup disappeared without empty observation") from None
        return self._closed_gate()

    def _queue_locks(self):
        descriptors = []
        for name in (".merge_queue.duckdb.rebuild.lock", ".merge_queue.duckdb.lock", "train/consumer.lock"):
            descriptor = role._open_regular(self.queue_root, name)
            self._resources.callback(os.close, descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            descriptors.append(descriptor)
        parent = role._open_directory(self.queue_root)
        try:
            fd = os.open("merge_queue.duckdb", os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                         dir_fd=parent)
        finally:
            os.close(parent)
        self._resources.callback(os.close, fd)
        info = os.fstat(fd)
        _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and info.st_nlink == 1,
                 "native queue writer inode unsafe")
        # OFD custody survives reading/closing the same inode in the copy loop;
        # traditional process locks would be silently dropped by those closes.
        _require(hasattr(fcntl, "F_OFD_SETLK"), "native capture requires OFD writer lock")
        fcntl.fcntl(fd, fcntl.F_OFD_SETLK, struct.pack("hhqqi", fcntl.F_WRLCK, os.SEEK_SET, 0, 0, 0))
        descriptors.append(fd)
        return descriptors

    def capture(self, *, destination, inspection_destination, context):
        """Copy the complete queue under retained closure and actual writer locks."""
        _require(self._capture is None, "native capture session already used")
        before = self._closed_gate()
        _require(context.get("source_commit") == self._source["head"]
                 and context.get("source_tree") == self._source["tree"]
                 and context.get("store_id") == str(self.queue_root / "merge_queue.duckdb"),
                 "native import context differs from captured source or store")
        self._queue_locks()
        identities = role.file_inventory(self.queue_root)
        entries = producer._entries(self.queue_root, identities)
        destination = Path(destination).absolute()
        fd = role._open_directory(destination.parent)
        os.close(fd)
        destination.mkdir(mode=0o700)
        for entry in entries:
            _require(self._closed_gate() == before, "native capture closure changed")
            role.copy_entry(self.queue_root, entry, destination / entry["path"])
        _require(role.file_inventory(self.queue_root) == identities,
                 "native queue population changed during capture")
        for entry in entries:
            role.copy_entry(self.queue_root, entry, None, digest_only=True)
        _require(self._closed_gate() == before, "native capture closure changed")
        result = producer.produce_offline_import_plan(
            offline_root=destination, destination=inspection_destination, context=context)
        _require(result["manifest"]["files"] == entries, "native captured inventory differs")
        receipt = {"schema": SCHEMA, "queue_root": str(self.queue_root),
                   "owner_identity": self.identity, "native_snapshot_cid": self.snapshot_cid,
                   "source": self._source, "pre_stop_namespaces": self.namespaces,
                   "closure": before, "manifest": result["manifest"],
                   "capture_coherent": True, "consumer_processes_closed": True,
                   "callback_settled": False, "signing_authority": False,
                   "source_admitted": False, "completion_authority": False}
        self._capture = CoherentLegacyCapture(self, destination, receipt, identities)
        return self._capture

    def close(self):
        self._closed = True
        self._resources.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class CoherentLegacyCapture:
    """A live session's result; its JSON receipt is an audit export only."""

    def __init__(self, session, path, receipt, identities):
        self._session = session
        self.path = path
        self._receipt = role._decode(role._json(receipt))
        self._identities = identities

    @property
    def receipt(self):
        return role._decode(role._json(self._receipt))

    def require_current(self):
        _require(type(self._session) is RetainedNativeLegacySession
                 and self._session._capture is self, "native capture is not retained by its producer")
        _require(self._session._closed_gate() == self.receipt["closure"],
                 "native capture closure changed")
        _require(role.file_inventory(self._session.queue_root) == self._identities,
                 "native queue changed after capture")
        for entry in self.receipt["manifest"]["files"]:
            role.copy_entry(self.path, entry, None, digest_only=True)
        return self.receipt
