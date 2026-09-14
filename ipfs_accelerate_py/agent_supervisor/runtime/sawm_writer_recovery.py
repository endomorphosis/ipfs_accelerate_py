"""Native SAWM writer-loss closure; startup and task authority stay native.

The expected manifest is a compare-and-swap input for an authorized operator,
not a credential or a statement that unresolved callbacks have completed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import hashlib
import itertools
import json
import os
import signal
import socket
import stat
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from . import native_graceful_recovery as process
from . import native_phased_graceful_recovery as phased_process
from . import owner_status_observation as observation

PROGRAM = "semantic-addressed-world-model-v1"
SCHEMA = "sawm/graceful-writer-loss-recovery@1"


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise process.GracefulRecoveryUnverified(reason)


def file_binding(path: Path, maximum: int = 32 * 1024 * 1024) -> dict[str, Any]:
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), "recovery_artifact_not_regular")
    value = observation._read_bounded(path, maximum)
    after = path.lstat()
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    identity = [getattr(before, name) for name in fields]
    require(
        identity == [getattr(after, name) for name in fields],
        "recovery_artifact_changed",
    )
    return {"identity": identity, "sha256": hashlib.sha256(value).hexdigest()}


def git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    require(len(result.stdout) <= 1024 * 1024, "source_observation_bound")
    return result.stdout.strip()


def source_binding(root: Path, config_path: Path) -> dict[str, Any]:
    source = git(root, "rev-parse", "HEAD", "HEAD^{tree}").splitlines()
    require(len(source) == 2, "source_identity_unavailable")
    index = Path(
        git(root, "rev-parse", "--path-format=absolute", "--git-path", "index")
    )
    result = {
        "head": source[0],
        "tree": source[1],
        "index": file_binding(index),
        "config": file_binding(config_path, 1024 * 1024),
    }
    require(
        not git(root, "diff", "--name-only", "HEAD"), "source_tracked_bytes_changed"
    )
    require(
        git(root, "rev-parse", "HEAD", "HEAD^{tree}").splitlines() == source,
        "source_ref_changed",
    )
    require(
        file_binding(index) == result["index"]
        and file_binding(config_path, 1024 * 1024) == result["config"],
        "source_index_or_config_changed",
    )
    return result


def canonical_writer_lost(
    database: Path, owner: process.ProcessBinding
) -> dict[str, Any]:
    """Observe exact loss without opening the canonical database descriptor."""
    process.require_exact_process(owner)
    for namespace in ("pid", "mnt"):
        require(
            os.readlink(f"/proc/{owner.pid}/ns/{namespace}")
            == os.readlink(f"/proc/self/ns/{namespace}"),
            "writer_namespace_unknown",
        )
    canonical = observation._regular_identity(database)
    native_locks = [
        observation._regular_identity(path) for path in observation._locks(database)
    ]
    identity = lambda value: (
        os.major(value["device"]),
        os.minor(value["device"]),
        value["inode"],
    )
    wanted = identity(canonical)
    held = set()
    for line in (
        observation._read_bounded(Path("/proc/locks"), 1024 * 1024)
        .decode("ascii")
        .splitlines()
    ):
        values = line.split()
        if len(values) == 9 and values[1] == "->":
            values = [values[0], *values[2:]]
        require(
            len(values) == 8 and values[1] in {"POSIX", "FLOCK", "OFDLCK"},
            "kernel_locks_unverified",
        )
        a, b, c = values[5].split(":")
        entry = (int(a, 16), int(b, 16), int(c))
        if values[1:4] == ["POSIX", "ADVISORY", "WRITE"] and entry == wanted:
            raise process.GracefulRecoveryUnverified("canonical_writer_present")
        if values[1:5] == ["FLOCK", "ADVISORY", "WRITE", str(owner.pid)] and values[
            6:
        ] == ["0", "EOF"]:
            held.add(entry)
    require(
        {identity(value) for value in native_locks} <= held,
        "owner_native_locks_unverified",
    )
    descriptors = list(itertools.islice(Path(f"/proc/{owner.pid}/fd").iterdir(), 4097))
    require(len(descriptors) <= 4096, "owner_descriptor_bound")
    opened = False
    for descriptor in descriptors:
        try:
            value = descriptor.stat()
        except FileNotFoundError:
            continue
        if (value.st_dev, value.st_ino) == (canonical["device"], canonical["inode"]):
            opened = True
    require(opened, "canonical_owner_descriptor_unverified")
    require(
        observation._regular_identity(database) == canonical
        and [
            observation._regular_identity(path) for path in observation._locks(database)
        ]
        == native_locks,
        "canonical_or_native_lock_identity_changed",
    )
    process.require_exact_process(owner)
    return {
        "canonical": canonical,
        "native_locks": native_locks,
        "writer_missing_verified": True,
        "callback_settlement_authority": False,
    }


def _path(root: Path, value: object) -> Path:
    require(type(value) is str and bool(value), "configured_path_invalid")
    path = root / value
    require(
        not Path(value).is_absolute() and ".." not in Path(value).parts,
        "configured_path_outside_root",
    )
    require(path.resolve().is_relative_to(root), "configured_path_symlink_escape")
    return path


def _pid(path: Path) -> int:
    value = observation._read_bounded(path, 64).decode("ascii").strip()
    require(value.isdecimal(), "native_pid_marker_invalid")
    return int(value)


def _argv(binding: process.ProcessBinding) -> list[str]:
    process.require_exact_process(binding)
    return (
        process._read_proc(Path(f"/proc/{binding.pid}/cmdline"))
        .decode()
        .rstrip("\0")
        .split("\0")
    )


def inspect(root: Path, config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    require(
        config.get("board_namespace") == PROGRAM and config.get("max_lanes") == 4,
        "native_program_scope_invalid",
    )
    require(
        str(config["database_program"]["store_generation"]) == "48",
        "native_generation_scope_invalid",
    )
    owner_config = config["quack_owner"]
    database = _path(root, owner_config["database_path"])
    state = _path(root, config["runtime_paths"]["state"])
    owner_status = observation._decode(
        observation._read_bounded(
            _path(root, owner_config["state_dir"]) / "quack-state-server.status.json",
            1024 * 1024,
        )
    )
    identity = owner_status["identity"]
    birth = identity["process_birth"]
    owner = process.observe_process(birth["pid"])
    require(
        owner_status.get("lifecycle") == "ready"
        and identity["generation"] == 48
        and identity["store_id"] == owner_config["store_id"]
        and identity["repository_id"] == owner_config["repository_id"]
        and owner.birth == birth["start_time_ticks"]
        and owner.boot_id == birth["boot_id"],
        "native_owner_identity_unverified",
    )
    owner_args = _argv(owner)
    require(
        "quack-start" in owner_args
        and any(a.endswith("/semantic_addressed_world_model.py") for a in owner_args),
        "native_owner_entry_unverified",
    )
    marker_paths = [
        state / "configured-board-wave.pid",
        state / "configured-board-master.pid",
    ]
    master_pid = _pid(marker_paths[0])
    require(_pid(marker_paths[1]) == master_pid, "native_master_markers_disagree")
    controller = process.observe_process(master_pid)
    require(
        "ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler"
        in _argv(controller),
        "native_controller_entry_unverified",
    )
    lanes = []
    for index in range(4):
        directory = state / f"lane-{index}"
        markers = [
            directory / f"sawm_lane_{index}_supervisor.pid",
            directory / f"sawm_lane_{index}_managed_daemon.pid",
        ]
        supervisor, daemon = [process.observe_process(_pid(path)) for path in markers]
        require(
            supervisor.parent == controller.pid and daemon.parent == supervisor.pid,
            "native_lane_parent_changed",
        )
        for actor, child in [(supervisor, False), (daemon, True)]:
            args = _argv(actor)
            require(
                "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor"
                in args
                and ("--run-sealed-daemon-child" in args) is child
                and "--accepted-control-plane-pin-json" in args
                and f"sawm_lane_{index}" in args,
                "native_lane_entry_unverified",
            )
        lanes.append(
            {
                "supervisor": dataclasses.asdict(supervisor),
                "daemon": dataclasses.asdict(daemon),
            }
        )
        marker_paths.extend(markers)
    actors = [
        owner,
        controller,
        *(process.ProcessBinding(**row) for lane in lanes for row in lane.values()),
    ]
    require(
        all(actor.pid == actor.process_group == actor.session for actor in actors),
        "native_owned_session_unverified",
    )
    require(
        len({actor.pid for actor in actors}) == len(actors),
        "native_actor_roster_duplicate",
    )
    return {
        "schema": SCHEMA,
        "root": str(root),
        "source": source_binding(root, config_path),
        "owner": dataclasses.asdict(owner),
        "owner_identity": identity,
        "controller": dataclasses.asdict(controller),
        "lanes": lanes,
        "custody": canonical_writer_lost(database, owner),
        "markers": {
            str(path.relative_to(root)): {
                "pid": _pid(path),
                "file": file_binding(path, 64),
            }
            for path in marker_paths
        },
    }


def _no_children(binding: process.ProcessBinding, allowed: set[int]) -> None:
    """All threads, plus visible direct children; uncertainty is a refusal."""
    process.require_exact_process(binding)
    tids = sorted(Path(f"/proc/{binding.pid}/task").iterdir())
    require(0 < len(tids) <= 512, "child_census_thread_bound")
    for tid in tids:
        text = process._read_proc(tid / "children").decode("ascii")
        require({int(pid) for pid in text.split()} <= allowed, "unclosed_native_child")
    require(
        tids == sorted(Path(f"/proc/{binding.pid}/task").iterdir()),
        "child_census_threads_changed",
    )


class NativeScopeProcessObserved(process.GracefulRecoveryUnverified):
    """The strict population refusal with bounded, non-authoritative evidence."""

    def __init__(self, diagnostic: Mapping[str, Any]):
        super().__init__("additional_native_scope_process")
        self.diagnostic = dict(diagnostic)


def _scope_refusal(pid, values, cwd, args, flags):
    # Do not expose argv or process environment; hashes identify the observation.
    diagnostic = {
        "schema": "sawm/rejected-scope-process@1",
        "pid": pid,
        "birth": int(values[19]),
        "parent": int(values[1]),
        "process_group": int(values[2]),
        "session": int(values[3]),
        "argv_sha256": hashlib.sha256(args).hexdigest(),
        "cwd": str(cwd),
        "scope_flags": dict(flags),
        "observed_at": time.time(),
        "stable_identity": False,
        "callback_settlement_authority": False,
    }
    try:
        after = process._stat(pid)
        diagnostic["stable_identity"] = (
            values[19] == after[19]
            and values[1:4] == after[1:4]
            and after[0] not in {"Z", "X"}
            and process._read_proc(Path("/proc") / str(pid) / "cmdline") == args
            and os.readlink(Path("/proc") / str(pid) / "cwd") == str(cwd)
        )
    except (OSError, ValueError, process.GracefulRecoveryUnverified):
        # An exited/replaced or unobservable actor still cannot authorize closure.
        pass
    return NativeScopeProcessObserved(diagnostic)


def scoped_census(root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Observe this checkout's remaining actors without claiming global visibility."""
    bindings = [
        expected["owner"],
        expected["controller"],
        *(actor for lane in expected["lanes"] for actor in lane.values()),
    ]
    known = {row["pid"]: process.ProcessBinding(**row) for row in bindings}
    groups = {row["process_group"] for row in bindings}
    sessions = {row["session"] for row in bindings}
    unavailable = []
    observed = []
    entries = list(itertools.islice(Path("/proc").iterdir(), 65537))
    require(len(entries) <= 65536, "process_census_bound")
    for entry in entries:
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        pid = int(entry.name)
        try:
            values = process._stat(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            require(pid not in known, "native_actor_observation_unavailable")
            unavailable.append(pid)
            continue
        if values[0] in {"Z", "X"}:
            continue
        # Kernel lineage is useful even if a descendant changes argv/cwd or is
        # reparented. A later denied read cannot turn a known scoped actor into
        # an unrelated visibility limitation.
        kernel_scoped = (
            pid in known
            or int(values[1]) in known
            or int(values[2]) in groups
            or int(values[3]) in sessions
        )
        try:
            cwd = Path(os.readlink(entry / "cwd"))
            args = process._read_proc(entry / "cmdline")
        except (FileNotFoundError, ProcessLookupError):
            try:
                current = process._stat(pid)
            except (FileNotFoundError, ProcessLookupError):
                continue
            if current[0] in {"Z", "X"}:
                continue
            require(not kernel_scoped, "native_actor_observation_unavailable")
            unavailable.append(pid)
            continue
        except PermissionError:
            require(not kernel_scoped, "native_actor_observation_unavailable")
            unavailable.append(pid)
            continue
        in_scope = (
            kernel_scoped
            or cwd == root
            or cwd.is_relative_to(root)
            or os.fsencode(root) in args
        )
        if not in_scope and pid not in known:
            continue
        if pid not in known:
            raise _scope_refusal(
                pid,
                values,
                cwd,
                args,
                {
                    "known_pid": False,
                    "known_parent": int(values[1]) in known,
                    "owned_group": int(values[2]) in groups,
                    "owned_session": int(values[3]) in sessions,
                    "cwd_under_root": cwd == root or cwd.is_relative_to(root),
                    "root_in_argv": os.fsencode(root) in args,
                },
            )
        process.require_exact_process(known[pid])
        observed.append(pid)
    for relative, original in expected["markers"].items():
        marker = _path(root, relative)
        if not os.path.lexists(marker):
            require(original["pid"] not in observed, "live_native_pid_marker_absent")
            continue
        current = _pid(marker)
        require(current in known, "foreign_native_pid_marker")
        if current in observed:
            require(
                file_binding(marker, 64) == original["file"],
                "live_native_pid_marker_changed",
            )
    return {
        "observed_native_pids": observed,
        "unavailable_unrelated_pids": unavailable,
        "global_process_visibility_claimed": False,
    }


def startup_exclusion(root: Path, config: Mapping[str, Any], watchdog_config: Path):
    """Bind the real deployed board HOLD and an inactive transient repair job."""
    watchdog = observation._decode(
        observation._read_bounded(watchdog_config, 1024 * 1024)
    )
    boards = watchdog.get("boards")
    require(type(boards) is list and len(boards) <= 64, "watchdog_scope_unverified")
    matches = [row for row in boards if type(row) is dict and row.get("id") == "sawm"]
    require(
        len(matches) == 1 and Path(matches[0]["cwd"]).resolve() == root,
        "watchdog_board_scope_changed",
    )
    hold = _path(root, config["runtime_paths"]["root"]) / "HOLD"
    require(
        str(hold) in matches[0].get("hold_files", []), "watchdog_hold_not_configured"
    )
    original = file_binding(hold, 65536)
    require(original["identity"][3] == os.getuid(), "watchdog_hold_owner_changed")
    watchdog_binding = file_binding(watchdog_config, 1024 * 1024)

    def gate():
        require(
            file_binding(hold, 65536) == original
            and file_binding(watchdog_config, 1024 * 1024) == watchdog_binding,
            "startup_exclusion_changed",
        )
        result = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                "ipfs-taskboard-repair-job.service",
                "--property=ActiveState",
                "--property=MainPID",
                "--no-pager",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        values = dict(line.split("=", 1) for line in result.stdout.splitlines())
        require(
            values.get("ActiveState") == "inactive" and values.get("MainPID") == "0",
            "repair_job_not_inactive",
        )

    gate()
    return gate


@contextlib.contextmanager
def recovery_journal(path: Path, manifest_sha256: str):
    """Durable exclusive output; a refusal preserves any already written prefix."""
    absolute = path.absolute()
    require(absolute.parent.resolve() == absolute.parent, "journal_parent_symlink")
    parent_fd = os.open(
        absolute.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    )
    try:
        descriptor = os.open(
            absolute.name,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=parent_fd,
        )
        try:
            original = os.fstat(descriptor)
            os.fsync(parent_fd)

            def record(phase: str) -> None:
                require(
                    type(phase) is str and len(phase) <= 128, "journal_phase_invalid"
                )
                current = os.stat(
                    absolute.name, dir_fd=parent_fd, follow_symlinks=False
                )
                require(
                    (current.st_dev, current.st_ino)
                    == (original.st_dev, original.st_ino),
                    "journal_identity_changed",
                )
                packet = (
                    json.dumps(
                        {
                            "phase": phase,
                            "timestamp": time.time(),
                            "manifest_sha256": manifest_sha256,
                            "callback_settlement_authority": False,
                        }
                    )
                    + "\n"
                ).encode()
                require(
                    os.write(descriptor, packet) == len(packet),
                    "journal_write_incomplete",
                )
                os.fsync(descriptor)

            yield record
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_fd)


def observe_owner_closed(
    root: Path, config: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    """Independently observe process/storage endpoint closure, never settlement."""
    identity = dict(expected["owner_identity"])
    owner_pid = expected["owner"]["pid"]
    require(not Path(f"/proc/{owner_pid}").exists(), "native_owner_not_reaped")
    status_path = (
        _path(root, config["quack_owner"]["state_dir"])
        / "quack-state-server.status.json"
    )
    status = observation._decode(observation._read_bounded(status_path, 1024 * 1024))
    identity["status"] = "stopped"
    require(
        status.get("lifecycle") == "stopped" and status.get("identity") == identity,
        "native_stopped_identity_unverified",
    )
    database = _path(root, config["quack_owner"]["database_path"])
    current = observation._regular_identity(database)
    require(current == expected["custody"]["canonical"], "canonical_inode_changed")
    lock_identities = [current, *expected["custody"]["native_locks"]]
    wanted = {
        (os.major(row["device"]), os.minor(row["device"]), row["inode"])
        for row in lock_identities
    }
    for line in (
        observation._read_bounded(Path("/proc/locks"), 1024 * 1024)
        .decode("ascii")
        .splitlines()
    ):
        fields = line.split()
        if len(fields) == 9 and fields[1] == "->":
            fields = [fields[0], *fields[2:]]
        require(len(fields) == 8, "closed_kernel_locks_unverified")
        a, b, c = fields[5].split(":")
        require(
            (int(a, 16), int(b, 16), int(c)) not in wanted, "closed_store_lock_present"
        )
    uri = identity["listen_uri"]
    prefix = "quack:127.0.0.1:"
    require(
        type(uri) is str and uri.startswith(prefix) and uri[len(prefix) :].isdecimal(),
        "closed_native_endpoint_scope_unverified",
    )
    with socket.socket() as probe:
        probe.settimeout(0.4)
        result = probe.connect_ex(("127.0.0.1", int(uri[len(prefix) :])))
    require(result == errno.ECONNREFUSED, "closed_native_endpoint_unverified")
    require(
        observation._regular_identity(database) == current,
        "closed_canonical_inode_changed",
    )
    return {
        "native_stopped_identity_observed": True,
        "canonical_inode_preserved": True,
        "store_locks_absent": True,
        "native_endpoint_refused": True,
        "callback_settlement_authority": False,
    }


def close_reviewed(
    *,
    root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    expected: Mapping[str, Any],
    startup_exclusion_gate,
    record_phase,
    timeout_seconds: float = 30,
) -> dict[str, Any]:
    """Operator-owned exclusion must remain held until native startup is admitted.

    The facade supplies the reviewed startup exclusion gate; the helper never
    changes a source ref, task, callback, grant, lease or generation.
    """
    require(
        inspect(root, config_path, config) == expected,
        "reviewed_recovery_manifest_changed",
    )
    owner = process.ProcessBinding(**expected["owner"])
    controller = process.ProcessBinding(**expected["controller"])
    lanes = [
        process.LaneBinding(
            process.ProcessBinding(**row["supervisor"]),
            process.ProcessBinding(**row["daemon"]),
        )
        for row in expected["lanes"]
    ]
    database = _path(root, config["quack_owner"]["database_path"])
    state = _path(root, config["runtime_paths"]["state"])
    from ..merge.checkout_lock import serialized_lock_update

    def bindings_gate():
        startup_exclusion_gate()
        require(
            source_binding(root, config_path) == expected["source"],
            "source_binding_changed",
        )
        status_path = (
            _path(root, config["quack_owner"]["state_dir"])
            / "quack-state-server.status.json"
        )
        status = observation._decode(
            observation._read_bounded(status_path, 1024 * 1024)
        )
        require(
            status.get("lifecycle") == "ready"
            and status.get("identity") == expected["owner_identity"],
            "current_owner_generation_identity_changed",
        )
        require(
            canonical_writer_lost(database, owner) == expected["custody"],
            "writer_custody_changed",
        )

    def population_gate():
        scoped_census(root, expected)

    def gate():
        bindings_gate()
        population_gate()

    def lane_gate(index):
        # Zombies are already exited; native cleanup may reap them.
        allowed = {lanes[index].daemon.pid}
        _no_children(lanes[index].supervisor, allowed)

    def final_gate():
        _no_children(controller, {lane.supervisor.pid for lane in lanes})
        # This is a separate final census, in addition to the following effect gate.
        scoped_census(root, expected)
        startup_exclusion_gate()

    result = phased_process.gracefully_close_native_lanes(
        controller=controller,
        lanes=lanes,
        lane_fence=lambda index: serialized_lock_update(
            state / f"lane-{index}" / f"sawm_lane_{index}_supervisor.lock",
            timeout_seconds=timeout_seconds,
        ),
        effect_gate=bindings_gate,
        population_gate=population_gate,
        population_refusal=NativeScopeProcessObserved,
        lane_children_gate=lane_gate,
        closed_children_gate=final_gate,
        record_phase=record_phase,
        timeout_seconds=timeout_seconds,
    )
    gate()
    with process._exact_pidfd(owner) as descriptor:
        record_phase("owner_graceful_exit_prepared")
        gate()
        signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        process._wait_exit(descriptor, time.monotonic() + timeout_seconds)
    record_phase("owner_exit_observed")
    deadline = time.monotonic() + timeout_seconds
    while Path(f"/proc/{owner.pid}").exists():
        require(time.monotonic() < deadline, "native_owner_reap_timeout")
        time.sleep(0.01)
    startup_exclusion_gate()
    require(
        source_binding(root, config_path) == expected["source"],
        "closed_source_binding_changed",
    )
    scoped_census(root, expected)
    closure = observe_owner_closed(root, config, expected)
    record_phase("native_owner_closure_observed")
    return {
        **result,
        "closure": closure,
        "owner_exited": True,
        "generation_changed": False,
        "native_restart_required": True,
        "source_transition_authority": False,
    }
