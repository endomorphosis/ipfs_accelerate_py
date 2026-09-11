"""Retain one reviewed SPAR capture transaction; never start a successor.

Inspection is the default CLI operation. ``--session`` keeps the exact native
pidfds and fleet locks in this process while an operator selects each stage.
Audit JSON cannot reconstruct a session. After arming, errors and input EOF
retain custody; they never undo HOLD, restore restart policy, or release claims.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import threading
import time

from . import spar_legacy_capture as native
from . import spar_legacy_origin as origin
from . import spar_merge_owner as role
from .spar_capture_runtime import CaptureRuntimeDenied, admit_runtime

SCHEMA = "spar/retained-capture-driver@1"
STOP_NAMES = ("HOLD", "OPERATOR_STOP", "watchdog.disabled", "watchdog.hold")
DROPIN = "95-native-legacy-capture.conf"
REPAIR_UNIT = "ipfs-taskboard-repair-job.service"


class CaptureDriverDenied(RuntimeError):
    pass


def require(value, reason):
    if not value:
        raise CaptureDriverDenied(reason)


def _cid(value):
    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


def _json_bytes(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    require(len(raw) <= 16 * 1024 * 1024, "driver_record_exceeds_bound")
    return raw


def _read(path, bound=1024 * 1024):
    return native._bytes(Path(path), bound)


def _object(path):
    value = role._decode(_read(path))
    require(type(value) is dict, "driver_record_not_object")
    return value


def _abs(path):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "driver_path_not_absolute_canonical")
    return path


def _absent(path):
    try:
        path.lstat()
    except FileNotFoundError:
        return
    raise CaptureDriverDenied("driver_path_already_exists")


def _exclusive_write(path, raw):
    """Create through a retained no-follow parent; never replace another actor."""
    descriptor = role._open_directory(path.parent)
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                     | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=descriptor)
        try:
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(fd)
        finally:
            os.close(fd)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require(_read(path, max(len(raw), 1)) == raw, "driver_created_record_changed")


def _owned_directory(path, *, create=False):
    if create:
        parent = role._open_directory(path.parent)
        try:
            os.mkdir(path.name, mode=0o700, dir_fd=parent)
            os.fsync(parent)
        finally:
            os.close(parent)
    descriptor = role._open_directory(path)
    try:
        info = os.fstat(descriptor)
        require(info.st_uid == os.geteuid() and not info.st_mode & 0o022,
                "driver_directory_not_owned_private")
    finally:
        os.close(descriptor)


def _command(argv):
    result = subprocess.run(argv, capture_output=True, timeout=10, check=False)
    require(result.returncode == 0 and len(result.stdout) <= 1024 * 1024,
            "driver_unit_observation_or_reload_failed")
    return result.stdout.decode("utf-8")


def _unit_path(unit):
    require(re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", unit), "driver_unit_name_invalid")
    return "/org/freedesktop/systemd1/unit/" + "".join(
        char if char.isascii() and char.isalnum() else f"_{ord(char):02x}" for char in unit)


def _property(unit, interface, name, signature):
    value = json.loads(_command([
        "busctl", "--user", "--json=short", "get-property", "org.freedesktop.systemd1",
        _unit_path(unit), "org.freedesktop.systemd1." + interface, name]))
    require(value.get("type") == signature
            and type(value.get("data")) is (str if signature == "s" else list),
            "driver_typed_unit_property_invalid")
    return value["data"]


def _scalars(unit, names):
    _unit_path(unit)
    raw = _command(["systemctl", "--user", "show", unit, "--no-pager",
                    "--property=" + ",".join(names)])
    value = {}
    for line in raw.splitlines():
        key, separator, body = line.partition("=")
        require(separator and key not in value, "driver_unit_property_ambiguous")
        value[key] = body
    require(set(value) == set(names) and value["Id"] == unit,
            "driver_unit_property_incomplete")
    return value


def unit_snapshot(unit):
    """The same typed executable/condition semantics as retained native capture."""
    value = _scalars(unit, native.UNIT_FIELDS)
    require(value["LoadState"] == "loaded" and value["NeedDaemonReload"] == "no",
            "driver_unit_not_loaded_or_reload_pending")
    value["ExecStart"] = [row[:3] for row in _property(unit, "Service", "ExecStart", "a(sasbttttuii)")]
    value["Conditions"] = [row[:4] for row in _property(unit, "Unit", "Conditions", "a(sbbsi)")]
    return value


def _job_unit():
    return _scalars(REPAIR_UNIT, ("Id", "LoadState", "ActiveState", "SubState", "MainPID"))


class FleetCaptureExclusion:
    """Existing board/queue locks plus positive no-claim/no-job observations."""

    def __init__(self, config_path, repo_root, hold_path):
        self.resources = ExitStack()
        self.locks = []
        self.closed = False
        self.config_path = _abs(config_path)
        self.config_bytes = _read(self.config_path)
        value = role._decode(self.config_bytes)
        require(type(value) is dict and value.get("schema") == "agent-supervisor/fleet-watchdog-config@1",
                "driver_fleet_config_invalid")
        boards = [b for b in value.get("boards", []) if type(b) is dict and b.get("id") == "spar"]
        require(len(boards) == 1, "driver_fleet_board_ambiguous")
        board = boards[0]
        self.holds = tuple(_abs(hold_path).parent / name for name in STOP_NAMES)
        require(board.get("cwd") == str(_abs(repo_root))
                and board.get("hold_files") == [str(path) for path in self.holds]
                and board.get("launch_hold_files") is None
                and board.get("ensure", {}).get("argv")
                == ["systemctl", "--user", "start", native.UNIT], "driver_fleet_board_binding_differs")
        self.state = _abs(value["state_dir"])
        self.job_path = self.state / "repairs/spar/job.json"
        try:
            for path in (self.state / "spar/watchdog.lock", self.state / "repairs/spar/queue.lock"):
                descriptor = role._open_regular(path.parent, path.name)
                self.resources.callback(os.close, descriptor)
                info = os.fstat(descriptor)
                require(info.st_nlink == 1, "driver_coordination_lock_has_alias")
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise CaptureDriverDenied("driver_fleet_coordination_busy") from exc
                self.locks.append((path, descriptor, (info.st_dev, info.st_ino)))
            self.job_bytes = _read(self.job_path)
            self.initial = self.require_idle()
        except BaseException:
            self.close()
            raise

    def require_current(self):
        require(not self.closed and len(self.locks) == 2, "driver_fleet_exclusion_not_retained")
        require(_read(self.config_path) == self.config_bytes, "driver_fleet_config_changed")
        for path, descriptor, expected in self.locks:
            current, retained = path.lstat(), os.fstat(descriptor)
            require(stat.S_ISREG(current.st_mode) and current.st_nlink == 1
                    and (current.st_dev, current.st_ino) == expected
                    and (retained.st_dev, retained.st_ino) == expected,
                    "driver_coordination_lock_replaced")
        require(_read(self.job_path) == self.job_bytes, "driver_repair_claim_changed_under_lock")

    def require_idle(self):
        self.require_current()
        job = role._decode(self.job_bytes)
        require(type(job) is dict and job.get("board_id") == "spar"
                and job.get("status") in {"queued", "verified_healthy", "verified_published"},
                "driver_repair_claim_not_positively_idle")
        attempts = job.get("attempts")
        started, finished = job.get("last_started_at"), job.get("finished_at")
        require(type(attempts) is int and attempts >= 0, "driver_repair_attempt_unknown")
        if started is not None:
            require(all(type(v) in (int, float) and math.isfinite(v) for v in (started, finished))
                    and 0 < started <= finished <= time.time(), "driver_repair_attempt_unfinished")
        else:
            require(attempts == 0 and finished is None, "driver_repair_attempt_inconsistent")
        worker = _object(self.state / "repair-worker.json")
        observed = worker.get("observed_at")
        require(worker.get("status") in {"waiting", "idle"}
                and type(observed) in (int, float) and math.isfinite(observed)
                and 0 <= time.time() - observed <= 120, "driver_repair_worker_not_positively_idle")
        unit = _job_unit()
        require(unit.get("LoadState") in {"loaded", "not-found"} and unit.get("MainPID") == "0"
                and (unit.get("ActiveState"), unit.get("SubState")) in {("inactive", "dead"), ("failed", "failed")},
                "driver_repair_job_not_positively_closed")
        return {"job_cid": _cid(job), "worker_observed_at": observed, "repair_unit": unit,
                "config_sha256": hashlib.sha256(self.config_bytes).hexdigest()}

    def require_no_stops(self):
        self.require_current()
        for path in self.holds:
            _absent(path)

    def close(self):
        self.closed = True
        self.resources.close()


class UnitCaptureInhibitor:
    """Install only one owned runtime drop-in; removal is a later reviewed stage."""

    def __init__(self, unit, hold_path):
        self.unit, self.hold = unit, _abs(hold_path)
        require(re.fullmatch(r"/[A-Za-z0-9_./-]+", str(self.hold)), "driver_hold_path_not_ini_safe")
        self.before = unit_snapshot(unit)
        require(self.before["ActiveState"] == "active" and self.before["SubState"] == "running"
                and int(self.before["MainPID"]) > 1, "driver_original_unit_not_running")
        self.files_before = self._files()
        self.directory = Path(f"/run/user/{os.geteuid()}/systemd/user") / (unit + ".d")
        self.path = self.directory / DROPIN
        _absent(self.path)
        self.raw = ("[Unit]\nRefuseManualStart=yes\nConditionPathExists=!" + str(self.hold)
                    + "\n[Service]\nRestart=no\nSendSIGKILL=no\nTimeoutStopSec=infinity\n"
                    "Delegate=yes\nExitType=cgroup\n").encode()
        self.applied = False

    def _files(self):
        fragment = _property(self.unit, "Unit", "FragmentPath", "s")
        drops = _property(self.unit, "Unit", "DropInPaths", "as")
        require(type(fragment) is str and fragment,
                "driver_native_unit_fragment_unknown")
        paths = [fragment, *drops]
        require(len(paths) <= 64 and all(type(p) is str for p in paths), "driver_unit_file_population_invalid")
        return {p: hashlib.sha256(_read(_abs(p))).hexdigest() for p in paths}

    def apply(self):
        require(not self.applied and unit_snapshot(self.unit) == self.before
                and self._files() == self.files_before, "driver_native_unit_changed_before_inhibition")
        _absent(self.hold)
        _absent(self.path)
        base = self.directory.parent
        if not base.exists():
            _owned_directory(base, create=True)
        _owned_directory(base)
        if not self.directory.exists():
            _owned_directory(self.directory, create=True)
        _owned_directory(self.directory)
        _exclusive_write(self.path, self.raw)
        self.applied = True
        _command(["systemctl", "--user", "daemon-reload"])
        self.require_current(owner_present=True)

    def require_current(self, *, owner_present=False):
        require(self.applied and _read(self.path) == self.raw, "driver_inhibitor_not_retained")
        current = unit_snapshot(self.unit)
        require(self._files() == self.files_before | {str(self.path): hashlib.sha256(self.raw).hexdigest()},
                "driver_native_unit_files_changed")
        require(current["WorkingDirectory"] == self.before["WorkingDirectory"]
                and current["ExecStart"] == self.before["ExecStart"], "driver_native_unit_executable_changed")
        for key, expected in {"Restart": "no", "SendSIGKILL": "no", "TimeoutStopUSec": "infinity",
                              "RefuseManualStart": "yes", "Delegate": "yes", "ExitType": "cgroup"}.items():
            require(current[key] == expected, "driver_native_inhibition_incomplete")
        require(["ConditionPathExists", False, True, str(self.hold)] in current["Conditions"],
                "driver_negative_hold_condition_missing")
        if owner_present:
            require(all(current[k] == self.before[k] for k in ("MainPID", "ActiveState", "SubState", "ControlGroup")),
                    "driver_native_owner_changed_during_inhibition")
        return current


class RetainedCaptureDriver:
    """One process owns inspection, arming, closure, capture, and installation."""

    def __init__(self, *, repository_root, config_path, fleet_config, operation_root,
                 expected_source_commit, expected_source_tree,
                 runtime_manifest=None, runtime_manifest_sha256=None):
        self.root, self.config = _abs(repository_root), _abs(config_path)
        self.fleet_config, self.output = _abs(fleet_config), _abs(operation_root)
        self.expected_head, self.expected_tree = expected_source_commit, expected_source_tree
        require(all(re.fullmatch(r"[0-9a-f]{40}", value) for value in (self.expected_head, self.expected_tree)),
                "driver_source_identity_invalid")
        self.stage = "new"
        self.exclusion = self.session = self.inhibitor = None
        self.inspection = self.captured = self.prepared = self.installed = None
        self.sequence = 0
        self.hold_bytes = None
        self.runtime_manifest, self.runtime_manifest_sha256 = runtime_manifest, runtime_manifest_sha256
        self.runtime = None
        self.failures = []
        self.attempts = {"capture": 0, "prepare": 0}
        self._retry = None
        self._closure = None

    def _result(self, **fields):
        return {"schema": SCHEMA, "stage": self.stage, "callback_settled": False,
                "signing_authority": False, "source_admitted": False,
                "successor_started": False, "completion_authority": False, **fields}

    def inspect(self):
        require(self.stage == "new", "driver_inspection_already_used")
        # Resolve every capture/prepare dependency in this exact process
        # before acquiring native handles or creating any operational artifact.
        self.runtime = admit_runtime(self.runtime_manifest, self.runtime_manifest_sha256)
        _absent(self.output)
        _owned_directory(self.output.parent)
        operator = native._native_operator(self.root)
        board, _ = operator._load_config(self.config)
        hold = board.path(board.runtime_paths["root"]) / "HOLD"
        require(not self.output.is_relative_to(self.root), "driver_output_inside_native_checkout")
        try:
            self.exclusion = FleetCaptureExclusion(self.fleet_config, self.root, hold)
            self.exclusion.require_no_stops()
            self.session = native.RetainedNativeLegacySession(repository_root=self.root, config_path=self.config,
                expected_source_commit=self.expected_head, expected_source_tree=self.expected_tree)
            require(self.session.hold_path == hold, "driver_native_hold_binding_changed")
            self.inhibitor = UnitCaptureInhibitor(native.UNIT, hold)
            require(self.inhibitor.before == self.session._unit_before, "driver_native_unit_inspection_changed")
            idle = self.exclusion.require_idle()
            self.stage = "inspected"
            self.inspection = self._result(owner_identity=self.session.identity, source=self.session._source,
                native_snapshot_cid=self.session.snapshot_cid, pre_stop_namespaces=self.session.namespaces,
                fleet=idle, original_unit=self.inhibitor.before, unit_files=self.inhibitor.files_before,
                operation_root=str(self.output), hold_path=str(hold),
                runtime={"manifest": self.runtime.manifest, "manifest_cid": _cid(self.runtime.manifest),
                         "isolated": bool(sys.flags.isolated), "executable": sys.executable})
            self.inspection["inspection_cid"] = _cid(self.inspection)
            self.inspection = json.loads(_json_bytes(self.inspection))
            return json.loads(_json_bytes(self.inspection))
        except BaseException:
            if self.session is not None:
                self.session.close()
            if self.exclusion is not None:
                self.exclusion.close()
            self.stage = "inspection-rejected"
            raise

    def _before_hold(self):
        self.runtime.require_current()
        self.exclusion.require_idle()
        self.exclusion.require_no_stops()
        require(self.session._source_binding() == self.session._source
                and native._alive(self.session.birth) and not native._exited(self.session.pidfd),
                "driver_retained_native_owner_or_source_changed")

    def _record(self, event):
        self.sequence += 1
        _exclusive_write(self.output / f"{self.sequence:02d}-{event}.json", _json_bytes(self._result(
            inspection_cid=self.inspection["inspection_cid"], event=event,
            resources_retained=not self.session._closed,
            installed=self.installed, failures=self.failures)))

    def arm(self, inspection_cid):
        require(self.stage == "inspected" and inspection_cid == self.inspection["inspection_cid"],
                "driver_arm_inspection_binding_differs")
        self._before_hold()
        _owned_directory(self.output, create=True)
        _exclusive_write(self.output / "inspection.json", _json_bytes(self.inspection))
        self.stage = "arming"
        self.inhibitor.apply()
        self._before_hold()
        self.sentinel = self.session.retain_workflow_sentinel()
        self.stage = "armed"
        self._record("armed")
        return self._result(sentinel=self.sentinel, hold_written=False)

    def request_closure(self):
        require(self.stage == "armed", "driver_closure_requires_retained_sentinel")
        self._before_hold()
        self.inhibitor.require_current(owner_present=True)
        require(self.session._sentinel is not None and not native._exited(self.session._sentinel_pidfd)
                and set(native._cgroup_population(self.session.cgroup_fd)[0])
                == {self.session.pid, self.session._sentinel.pid}, "driver_armed_population_changed")
        self.hold_bytes = _json_bytes({"schema": SCHEMA + "/hold", "operation_root": str(self.output),
                                      "inspection_cid": self.inspection["inspection_cid"]})
        _exclusive_write(self.session.hold_path, self.hold_bytes)
        self.stage = "close-requested"
        self._record("hold-written")
        return self._result(native_signal_sent=False)

    def _held(self):
        self.exclusion.require_current()
        self.inhibitor.require_current()
        require(self.hold_bytes is not None and _read(self.session.hold_path) == self.hold_bytes,
                "driver_hold_not_retained")

    def poll(self):
        require(self.stage in {"close-requested", "closed"}, "driver_closure_not_requested")
        self._held()
        closure = self.session.observe_closure()
        if closure is not None and self.stage != "closed":
            self.stage = "closed"
            self._closure = role._decode(role._json(closure))
            _exclusive_write(self.output / "native-closure.json", _json_bytes(closure))
            self._record("native-closed")
        return self._result(closure=closure, resources_retained=True)

    def _attempt_paths(self, action):
        self.attempts[action] += 1
        attempt = self.attempts[action]
        require(attempt <= 32, "driver_preinstall_attempt_bound_exhausted")
        suffix = "" if attempt == 1 else f"-{attempt:03d}"
        names = ("raw-capture", "inspection-copy") if action == "capture" else ("prepared-clone",)
        paths = tuple(self.output / (name + suffix) for name in names)
        for path in paths:
            _absent(path)
        return paths

    def _failed_preinstall(self, action, error, paths):
        self.stage = action + "-failed"
        # This tuple retains the actual original objects. A status document or
        # caller-supplied stage never substitutes for any element of it.
        self._retry = (action, self.session, self.captured)
        failure = {"action": action, "attempt": self.attempts[action],
                   "paths": [str(path) for path in paths], "diagnostic": _diagnostic(error)}
        self.failures.append(failure)
        try:
            self._record(action + "-failed")
        except Exception:
            # The original diagnostic and handles remain available over stdin
            # even when the failure is an inability to write an audit record.
            pass

    def _retry_admission(self, action):
        require(self.stage == action + "-failed" and self._retry is not None,
                "driver_retry_requires_same_process_failure")
        previous, session, captured = self._retry
        require(previous == action and session is self.session and captured is self.captured
                and self.prepared is None and self.installed is None,
                "driver_retry_retained_objects_differ")
        self._held()
        self.runtime.require_current()
        require(self._closure is not None and self.session._closed_gate() == self._closure,
                "driver_retry_native_closure_changed")
        if action == "capture":
            require(self.captured is None and self.session._capture is None,
                    "driver_retry_capture_already_admitted")
        else:
            self.captured.require_current()

    def capture(self):
        require(self.stage == "closed", "driver_capture_requires_positive_native_closure")
        return self._capture_attempt()

    def retry_capture(self):
        self._retry_admission("capture")
        return self._capture_attempt()

    def _capture_attempt(self):
        self._held()
        self.runtime.require_current()
        paths = self._attempt_paths("capture")
        self.stage = "capturing"
        try:
            self.captured = self.session.capture(destination=paths[0], inspection_destination=paths[1])
        except Exception as error:
            self._failed_preinstall("capture", error, paths)
            raise
        self._retry = None
        self.stage = "captured"
        _exclusive_write(self.output / "capture.json", _json_bytes(self.captured.receipt))
        self._record("captured")
        return self._result(capture_cid=role._cid(self.captured.receipt), capture_path=str(self.captured.path))

    def prepare(self):
        require(self.stage == "captured", "driver_prepare_requires_retained_capture")
        return self._prepare_attempt()

    def retry_prepare(self):
        self._retry_admission("prepare")
        return self._prepare_attempt()

    def _prepare_attempt(self):
        self._held()
        self.runtime.require_current()
        self.captured.require_current()
        paths = self._attempt_paths("prepare")
        self.stage = "preparing"
        try:
            self.prepared = role.prepare_offline_clone(offline_root=self.captured.path,
                destination=paths[0], manifest=self.captured.receipt["manifest"])
        except Exception as error:
            self._failed_preinstall("prepare", error, paths)
            raise
        self._retry = None
        self.stage = "prepared"
        self._record("prepared")
        return self._result(prepared_database_uuid=self.prepared.database_uuid)

    def status(self):
        return self._result(resources_retained=self.session is not None and not self.session._closed,
                            diagnostics=json.loads(_json_bytes(self.failures)), retry_available=(
                                self._retry[0] if self._retry is not None else None))

    def install(self):
        require(self.stage == "prepared", "driver_install_requires_retained_prepared_clone")
        self._held()
        self.stage = "installing"
        self.installed = origin.install_captured_queue(self.captured, self.prepared)
        self.stage = "installed"
        self._record("installed")
        return self._result(installed=self.installed)

    def finish(self):
        require(self.stage == "installed", "driver_finish_requires_complete_installation")
        self._held()
        require(self.session._closed_gate() == self.captured.receipt["closure"],
                "driver_closed_population_changed_after_installation")
        marker = _object(self.session.queue_root / origin.REQUIRED_MARKER)
        require(marker.get("origin_cid") == self.installed["origin_cid"]
                and marker.get("capture_cid") == role._cid(self.captured.receipt),
                "driver_installed_marker_differs")
        for entry in self.captured.receipt["manifest"]["files"]:
            role.copy_entry(self.captured.path, entry, None, digest_only=True)
        self.session.close()
        self.stage = "finished"
        self._record("finished")
        self.exclusion.close()
        return self._result(installed=self.installed, sentinel_close=self.session.sentinel_close_receipt,
                            hold_retained=True, unit_inhibition_retained=True)

    def close_inspection(self):
        require(self.stage == "inspected", "driver_cannot_abandon_armed_custody")
        self.session.close()
        self.exclusion.close()
        self.stage = "inspection-closed"
        return self._result()



def _diagnostic(error):
    """Bounded structural traceback; never export locals, source or arguments."""
    frames = []
    current = error.__traceback__
    while current is not None and len(frames) < 12:
        code = current.tb_frame.f_code
        frames.append({"file": Path(code.co_filename).name[:160],
                       "function": code.co_name[:160], "line": current.tb_lineno})
        current = current.tb_next
    missing = getattr(error, "name", None) if isinstance(error, ModuleNotFoundError) else None
    if type(missing) is not str or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9.]{0,159}", missing):
        missing = None
    reason = str(error) if type(error) in {CaptureDriverDenied, CaptureRuntimeDenied} else type(error).__name__
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,159}", reason):
        reason = type(error).__name__
    return {"code": reason, "exception": type(error).__name__[:160],
            "exception_module": type(error).__module__[:160], "missing_module": missing,
            "traceback": frames, "traceback_truncated": current is not None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repository-root", "config-path", "fleet-config", "operation-root",
                 "expected-source-commit", "expected-source-tree"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--session", action="store_true", help="retain handles for explicit JSON-line stages")
    parser.add_argument("--runtime-manifest", help="explicit reviewed DuckDB runtime code binding")
    parser.add_argument("--runtime-manifest-sha256", help="exact reviewed manifest digest")
    args = vars(parser.parse_args(argv))
    retain = args.pop("session")
    driver = RetainedCaptureDriver(**args)

    def emit(value):
        print(_json_bytes(value).decode(), flush=True)

    try:
        emit(driver.inspect())
        if not retain:
            emit(driver.close_inspection())
            return 0
        while driver.stage not in {"finished", "inspection-closed"}:
            line = sys.stdin.buffer.readline(65537)
            if not line:
                if driver.stage == "inspected":
                    emit(driver.close_inspection())
                    return 0
                emit(driver._result(error="input_closed_custody_retained", resources_retained=True))
                # There is no JSON rehydration or automatic last-actor cleanup.
                # The original controller must remain available for diagnosis.
                threading.Event().wait()
            try:
                require(len(line) <= 65536, "driver_command_exceeds_bound")
                command = role._decode(line)
                require(type(command) is dict and type(command.get("action")) is str,
                        "driver_command_invalid")
                action = command["action"]
                if action == "arm":
                    require(set(command) == {"action", "inspection_cid"}, "driver_command_fields_invalid")
                    emit(driver.arm(command["inspection_cid"]))
                else:
                    require(set(command) == {"action"} and action in {
                        "request_closure", "poll", "capture", "prepare", "install", "finish", "close_inspection",
                        "retry_capture", "retry_prepare", "status"},
                        "driver_command_not_admitted")
                    emit(getattr(driver, action)())
            except Exception as error:
                diagnostic = _diagnostic(error)
                emit(driver._result(error=diagnostic["code"], diagnostic=diagnostic,
                     resources_retained=driver.session is not None and not driver.session._closed))
        return 0
    except Exception as error:
        diagnostic = _diagnostic(error)
        emit(driver._result(error=diagnostic["code"], diagnostic=diagnostic))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
