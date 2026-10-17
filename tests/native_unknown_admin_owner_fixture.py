"""Bounded, disposable, genuine ADMIN creator-process fixture.

The caller must already own and admit the exact child argv/source. This helper
has no production admission or original-store authority. It opens only the
supplied copied canonical store explicitly, creates a real typed gateway and
issues its real narrow grant to the actual parent process. The sole product
seam is the visible, nonadmitting ``policy.verify_custody`` fixture boundary.

The private ready record contains a real token and is never printed. Graceful
exit stops the gateway and closes its connection in the actual child. The
separate crash method kills and reaps only that exact owned child, leaving any
genuine canonical WAL for the later native daemon to recover. No provider or
model is called, and no launch, terminal or admission receipt is manufactured.
Importing this helper performs no filesystem, process or database operation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import resource
import selectors
import signal
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

MARKER = "admin-owner-input.json"
CONTEXT = "admin-owner-context-input.json"
INSTALLATION = "duckdb-installation.json"
READY = "admin-owner-ready.json"
STOP = "admin-owner-stop.json"
EXIT = "admin-owner-exited.json"
FAILURE = "admin-owner-failure.json"
SOCKET = "admin.sock"
MAX_SOURCE = 8 * 1024 * 1024
MAX_DATABASE = 16 * 1024 * 1024
MAX_INPUT = 2 * 1024 * 1024
MAX_STREAM = 65536
MAX_PRODUCT_FILES = 4096
MAX_PRODUCT_BYTES = 256 * 1024 * 1024
# Parent-controlled hard termination starts by 25 seconds and has a separate
# five-second reap allowance. The child begins its own graceful teardown early.
CHILD_SECONDS = 25.0
REAP_SECONDS = 5.0
TEARDOWN_RESERVE = 8.0
POLICY_MODULE = "ipfs_accelerate_py.agent_supervisor.todo_daemon.planning_unknown_abandonment"
TYPED_MODULE = "ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner"
REQUIRED_RELATIVES = frozenset({
    "todo_daemon/planning_unknown_abandonment.py",
    "todo_daemon/planning_unknown_abandonment_settlement.py",
    "todo_daemon/implementation_daemon.py", "task_sources/typed_state_owner.py",
    "task_sources/database_task_source.py", "task_sources/intent_repository.py",
    "task_sources/control_plane_contracts.py", "task_sources/duckdb_state.py",
    "merge/database_coordination.py", "runtime/quack_state_server.py",
    "todo_daemon/native_execution_status_owner.py",
    "todo_daemon/native_execution_wal_maintenance.py",
})


def _json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate admin-owner fixture JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite fixture JSON")))


def _regular(path, maximum, *, nonempty=True, private=False):
    path = Path(path)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError("admin-owner input path alias")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid not in (0, os.geteuid())
                or before.st_nlink != 1 or before.st_mode & 0o002
                or not (int(nonempty) <= before.st_size <= maximum)
                or (private and (before.st_uid != os.geteuid()
                                 or stat.S_IMODE(before.st_mode) != 0o600))):
            raise ValueError("admin-owner regular-file custody/bound differs")
        chunks, count = [], 0
        while block := os.read(descriptor, min(65536, maximum - count + 1)):
            count += len(block)
            if count > maximum:
                raise ValueError("admin-owner input byte bound")
            chunks.append(block)
        fields = ("st_dev", "st_ino", "st_uid", "st_gid", "st_mode", "st_nlink",
                  "st_size", "st_mtime_ns", "st_ctime_ns")
        witness = lambda item: tuple(getattr(item, field) for field in fields)
        if (count != before.st_size or not witness(before)
                == witness(os.fstat(descriptor)) == witness(path.lstat())):
            raise ValueError("admin-owner input changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _pin(path, maximum=MAX_SOURCE, *, nonempty=True, private=False):
    raw = _regular(path, maximum, nonempty=nonempty, private=private)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _read_pin(value, maximum=MAX_SOURCE, *, private=False):
    if type(value) is not dict or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError("admin-owner fixture pin shape differs")
    raw = _regular(Path(value["path"]), maximum, nonempty=False, private=private)
    if type(value["bytes"]) is not int or value["bytes"] != len(raw) or hashlib.sha256(raw).hexdigest() != value["sha256"]:
        raise ValueError("admin-owner fixture pinned bytes differ")
    return raw


def _write(path, value):
    raw = _json(value)
    if len(raw) > MAX_INPUT:
        raise ValueError("admin-owner fixture record exceeds bound")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("admin-owner fixture short write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _ticks(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def _boot():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _identity(path):
    value = Path(path).stat()
    return dict(device=value.st_dev, inode=value.st_ino, size=value.st_size, mtime_ns=value.st_mtime_ns)


def _published(path, expected):
    pinned = _pin(path)
    if path.name != "native_execution_crash_wal_fixture.py" or pinned["sha256"] != expected:
        raise ValueError("published exact installed-DuckDB loader pin differs")
    spec = importlib.util.spec_from_file_location("_pinned_admin_owner_duckdb_loader", path,
                                                loader=_PinnedProductLoader(pinned))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if _pin(path)["sha256"] != expected:
        raise ValueError("published installed-DuckDB loader changed during load")
    return module


def _product_scope(context):
    rows = context.get("source_pins")
    if type(rows) is not list or len(rows) != len(REQUIRED_RELATIVES):
        raise ValueError("admin-owner fixed product source closure differs")
    by_path = {}
    for value in rows:
        _read_pin(value)
        if value["path"] in by_path:
            raise ValueError("duplicate admin-owner source pin")
        by_path[value["path"]] = value
    matches = [Path(name) for name in by_path if name.endswith(
        "/ipfs_accelerate_py/agent_supervisor/todo_daemon/planning_unknown_abandonment.py")]
    if len(matches) != 1:
        raise ValueError("admin-owner policy source root is ambiguous")
    product_root = matches[0].parents[3]
    prefix = product_root / "ipfs_accelerate_py/agent_supervisor"
    if set(by_path) != {str(prefix / relative) for relative in REQUIRED_RELATIVES}:
        raise ValueError("admin-owner source modules cross product roots")
    return product_root


def _package_pins(product_root):
    package = product_root / "ipfs_accelerate_py"
    paths = sorted(package.rglob("*.py"))
    if not 1 <= len(paths) <= MAX_PRODUCT_FILES:
        raise ValueError("admin-owner product Python population bound")
    rows, total = {}, 0
    for path in paths:
        value = _pin(path, nonempty=False)
        total += value["bytes"]
        if total > MAX_PRODUCT_BYTES:
            raise ValueError("admin-owner aggregate product source bound")
        rows[str(path)] = value
    if str(package / "__init__.py") not in rows:
        raise ValueError("admin-owner actual product initializer absent")
    return rows


class _PinnedProductLoader(importlib.abc.Loader):
    def __init__(self, pin):
        self.pin = pin

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Compile the source bytes just verified; existing bytecode is unused.
        raw = _read_pin(self.pin)
        exec(compile(raw, self.pin["path"], "exec", dont_inherit=True), module.__dict__)
        _read_pin(self.pin)


class _PinnedProductFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root, pins):
        self.root, self.pins = root, pins

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "ipfs_accelerate_py" and not fullname.startswith("ipfs_accelerate_py."):
            return None
        base = self.root.joinpath(*fullname.split("."))
        for source, package in ((base / "__init__.py", True), (base.with_suffix(".py"), False)):
            value = self.pins.get(str(source))
            if value is not None:
                return importlib.util.spec_from_file_location(fullname, source,
                    loader=_PinnedProductLoader(value),
                    submodule_search_locations=[str(base)] if package else None)
        # Genuine namespace directories can exist without an initializer.
        if base.is_dir() and any(Path(name).is_relative_to(base) for name in self.pins):
            spec = importlib.util.spec_from_loader(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = [str(base)]
            return spec
        raise ModuleNotFoundError("admin-owner product import absent from pinned source closure: " + fullname)


def _loaded_product_sources(pins):
    for module_name, module in tuple(sys.modules.items()):
        if module_name == "ipfs_accelerate_py" or module_name.startswith("ipfs_accelerate_py."):
            origin = getattr(module, "__file__", None)
            if origin is not None:
                value = pins.get(str(Path(origin)))
                if value is None:
                    raise ValueError("actual loaded product module outside source pins")
                _read_pin(value)


def fixture_argv(root, marker_sha):
    """Exact finite child argv for the outer source owner's explicit allowlist."""
    return (str(Path(sys.executable).resolve(strict=True)), "-I", "-S", "-B",
            str(Path(__file__).resolve(strict=True)), "--serve-admin",
            "--fixture-root", str(root), "--marker-sha256", marker_sha)


def _environment(root):
    # Closed allowlist: no model keys, proxy settings, loader overrides or
    # inherited provider configuration. HOME is neither set nor repurposed.
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
        "IPFS_ACCEL_SKIP_CORE": "1", "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
        "IPFS_DATASETS_AUTO_INSTALL": "false", "IPFS_AUTO_INSTALL": "false",
        "IPFS_ACCELERATE_AGENT_QUACK_PREFER": "0", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
        "TMPDIR": str(root), "TMP": str(root), "TEMP": str(root),
        "XDG_CONFIG_HOME": str(root / "private-config"), "XDG_CACHE_HOME": str(root / "private-cache"),
        "HF_HOME": str(root / "private-hf"), "HF_TOKEN_PATH": str(root / "absent-hf-token"),
        "IPFS_ACCELERATE_LLM_ALLOCATION_DB": str(root / "fixture-allocation.duckdb")}


class AdminOwner:
    """One exact owned child and its genuine public grant/context records."""
    def __init__(self, process, root, marker, marker_sha, started, typed):
        self._process, self.root, self._marker, self._marker_sha = process, root, marker, marker_sha
        self._started, self._typed = started, typed
        self._ticks, self._boot = _ticks(process.pid), _boot()
        self.child_pid, self.child_start_time_ticks, self.boot_id = process.pid, self._ticks, self._boot
        self.owner_process_group = os.getpgrp()
        self.argv = tuple(process.args)
        self.source_pins = {name: value["sha256"] for name, value in marker["product_sources"].items()}
        self.source_pins.update({marker["helper_pin"]["path"]: marker["helper_pin"]["sha256"],
                                marker["published_loader_pin"]["path"]: marker["published_loader_pin"]["sha256"]})
        self.installation_pin = marker["installation_pin"]
        self._streams, self._clients, self._witness = [bytearray(), bytearray()], [], None
        self._selector = selectors.DefaultSelector()
        for stream, destination in zip((process.stdout, process.stderr), self._streams):
            os.set_blocking(stream.fileno(), False)
            self._selector.register(stream, selectors.EVENT_READ, destination)

    def _remaining(self):
        return CHILD_SECONDS - (time.monotonic() - self._started)

    def _matching(self):
        return (_boot() == self._boot and _ticks(self.child_pid) == self._ticks
                and os.getpgid(self.child_pid) == self.owner_process_group)

    def _pump(self, maximum_wait=0.1):
        for key, _events in self._selector.select(max(0.0, min(maximum_wait, self._remaining()))):
            block = os.read(key.fd, 65536)
            if not block:
                self._selector.unregister(key.fileobj)
                key.fileobj.close()
                continue
            key.data.extend(block)
            if len(key.data) > MAX_STREAM:
                raise AssertionError("admin-owner private diagnostic stream bound")

    def _ready(self):
        while not (self.root / READY).exists():
            if self._process.poll() is not None:
                raise AssertionError("actual admin-owner child exited before ready; private diagnostics retained")
            if self._remaining() <= TEARDOWN_RESERVE:
                raise TimeoutError("actual admin-owner ready deadline")
            self._pump()
        value = _decode(_regular(self.root / READY, MAX_INPUT, private=True))
        self._ready_value = value
        self.context, self.socket_path, self.token = value["context"], Path(value["socket_path"]), value["token"]
        self.grant_json = value["grant"]
        self.grant = SimpleNamespace(**self.grant_json)
        from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import content_identity
        snapshot = _decode(_read_pin(self._marker["context_input"]["snapshot_pin"]))
        expected_scope = content_identity({"context_cid": self.context["context_cid"],
            "attempt_id": snapshot["attempt"]["attempt_id"], "claim_id": snapshot["attempt"]["claim_id"]})
        expected_creator = {"pid": self.child_pid,
                            "process_birth_id": self._typed.kernel_process_birth_id(self.child_pid)}
        installation = _decode(_read_pin(self.installation_pin, MAX_INPUT, private=True))
        previous = {key: item for key, item in self._marker["context_input"].items()
                    if key not in {"context_cid", "registered_creator", "registered_owner_identity"}}
        actual = {key: item for key, item in self.context.items()
                  if key not in {"context_cid", "registered_creator", "registered_owner_identity"}}
        if (value.get("schema") != "nonadmitting-real-native-admin-owner-ready@1"
                or value.get("fixture_only") is not True or value.get("production_custody_qualified") is not False
                or value.get("native_orchestration_receipts_created") is not False
                or value.get("child_pid") != self.child_pid or value.get("child_start_time_ticks") != self._ticks
                or value.get("boot_id") != self._boot or value.get("owner_process_group") != self.owner_process_group
                or value.get("marker_sha256") != self._marker_sha
                or value.get("duckdb_package_origin") != installation["package_origin"]
                or value.get("duckdb_native_origin") != installation["native_origin"]
                or value.get("duckdb_version") != installation["version"]
                or self.context.get("registered_creator") != expected_creator
                or _json(self.context.get("registered_owner_identity")) != _json(value.get("owner_identity"))
                or type(value.get("owner_identity")) is not dict
                or set(value["owner_identity"]) != {"server_id", "generation", "store_id"}
                or type(value["owner_identity"]["generation"]) is not int or value["owner_identity"]["generation"] != 1
                or value["owner_identity"]["store_id"] != self._marker["logical_store_id"]
                or _json(previous) != _json(actual) or self.socket_path != self.root / SOCKET
                or not self._matching() or type(self.token) is not str or not self.token
                or self.grant.client_id != self._typed.PLANNING_UNKNOWN_ABANDONMENT_ADMIN_CLIENT_ID
                or self.grant.allowed_operations != [self._typed.PLANNING_UNKNOWN_ABANDONMENT_OPERATION]
                or self.grant.allowed_command_operations or self.grant.allowed_database_task_commands
                or self.grant.peer_pid != os.getpid() or self.grant.peer_uid != os.geteuid()
                or self.grant.peer_start_time_ticks != _ticks(os.getpid())
                or self.grant.process_birth_id != self._typed.kernel_process_birth_id(os.getpid())
                or self.grant.entity_scopes != {"task_cid": self.context["task_cid"],
                    "abandonment_scope_cid": expected_scope}):
            raise AssertionError("actual child context/grant/creator crosswalk differs")
        if self.context["context_cid"] != content_identity({key: item for key, item in self.context.items() if key != "context_cid"}):
            raise AssertionError("actual child registered context identity differs")
        self._creator_birth = expected_creator["process_birth_id"]
        self._owner_identity_bytes = _json(value["owner_identity"])
        return self

    def client(self):
        if self._witness is not None or self._process.poll() is not None or self._remaining() <= TEARDOWN_RESERVE:
            raise RuntimeError("actual admin-owner child is unavailable/near teardown")
        value = self._typed.TypedStateOwnerConnection(socket_path=self.socket_path, token=self.token,
            client_id=self.grant.client_id, process_birth_id=self.grant.process_birth_id,
            store_id=self._marker["logical_store_id"], timeout_seconds=min(5.0, self._remaining() - TEARDOWN_RESERVE))
        if _json(dict(value.identity)) != self._owner_identity_bytes or _json(dict(value.grant)) != _json(self.grant_json):
            value.close()
            raise AssertionError("actual live gateway handshake identity/grant differs from ready record")
        self._clients.append(value)
        return value

    def _close_clients(self):
        for value in self._clients:
            value.close()
        self._clients.clear()

    def _force_reap(self):
        # This is a Popen child owned by this process. PID/start/boot/group must
        # still match before any signal; no group-wide or unrelated PID kill.
        if self._process.poll() is None:
            if not self._matching():
                raise AssertionError("refuse signaling an admin-owner identity mismatch")
            self._process.kill()
        self._process.wait(timeout=REAP_SECONDS)

    def _finish_streams(self):
        while self._selector.get_map():
            self._pump(0)
        self._selector.close()

    def _failure(self, error):
        if not (self.root / FAILURE).exists():
            _write(self.root / FAILURE, {"schema": "nonadmitting-real-native-admin-owner-failure@1",
                "fixture_only": True, "production_custody_qualified": False,
                "native_orchestration_receipts_created": False, "error_type": type(error).__name__,
                "child_pid": self.child_pid, "child_start_time_ticks": self._ticks, "boot_id": self._boot,
                "returncode": self._process.poll(), "stdout_text": bytes(self._streams[0]).decode("utf-8", "replace"),
                "stderr_text": bytes(self._streams[1]).decode("utf-8", "replace")})

    def _reap(self, *, crash):
        if self._witness is not None:
            if self._witness["termination_kind"] != ("crash_exit" if crash else "graceful_exit"):
                raise RuntimeError("admin-owner already reaped through a different termination path")
            return dict(self._witness)
        try:
            self._close_clients()
            if self._process.poll() is not None or not self._matching():
                raise AssertionError("actual admin-owner exited before requested termination")
            if crash:
                self._process.kill()
            else:
                _write(self.root / STOP, {"schema": "nonadmitting-real-native-admin-owner-stop@1",
                    "nonce": self._marker["nonce"], "marker_sha256": self._marker_sha,
                    "parent_pid": os.getpid(), "parent_start_time_ticks": _ticks(os.getpid()),
                    "child_pid": self.child_pid, "child_start_time_ticks": self._ticks, "boot_id": self._boot})
            while self._process.poll() is None:
                if self._remaining() <= 0:
                    raise TimeoutError("actual admin-owner exit deadline")
                self._pump()
            code = self._process.wait(timeout=max(0.01, self._remaining()))
            self._finish_streams()
            if code != (-signal.SIGKILL if crash else 0):
                raise AssertionError("actual admin-owner requested termination failed")
            if not crash:
                exited = _decode(_regular(self.root / EXIT, 32768, private=True))
                if (exited.get("schema") != "nonadmitting-real-native-admin-owner-exited@1"
                        or exited.get("child_pid") != self.child_pid or exited.get("child_start_time_ticks") != self._ticks
                        or exited.get("boot_id") != self._boot or exited.get("marker_sha256") != self._marker_sha
                        or exited.get("stop_requested") is not True or exited.get("gateway_stopped") is not True
                        or exited.get("connection_closed") is not True):
                    raise AssertionError("actual child graceful teardown witness differs")
            elif (self.root / EXIT).exists():
                raise AssertionError("crash path unexpectedly has a child graceful-close witness")
            try:
                same_process_present = _ticks(self.child_pid) == self._ticks
            except FileNotFoundError:
                same_process_present = False
            if same_process_present:
                raise AssertionError("reaped actual admin-owner still present")
            self._witness = {"schema": "nonadmitting-real-native-admin-owner-process-witness@1",
                "fixture_only": True, "production_custody_qualified": False,
                "native_orchestration_receipts_created": False, "child_pid": self.child_pid,
                "child_start_time_ticks": self._ticks, "boot_id": self._boot,
                "child_process_birth_id": self._creator_birth,
                "owner_process_group": self.owner_process_group, "returncode": code,
                "reaped": True, "child_reaped": True, "termination_kind": "crash_exit" if crash else "graceful_exit",
                "gateway_stopped_by_child": not crash, "connection_closed_by_child": not crash,
                "canonical_wal_present_after_exit": Path(self._marker["canonical_pin"]["path"] + ".wal").exists(),
                "marker_sha256": self._marker_sha, "stdout_bytes": len(self._streams[0]), "stderr_bytes": len(self._streams[1])}
            return dict(self._witness)
        except BaseException as error:
            self._failure(error)
            self._force_reap()
            raise
        finally:
            if self._process.poll() is not None:
                for stream in (self._process.stdout, self._process.stderr):
                    if stream is not None and not stream.closed:
                        stream.close()
                self._selector.close()

    def stop_and_reap(self):
        """Request real child gateway/connection close, then reap that child."""
        return self._reap(crash=False)

    def crash_and_reap(self):
        """SIGKILL only this exact owned child; preserve genuine crash WAL."""
        return self._reap(crash=True)


def start_admin_owner(root, canonical_path, context_input, logical_store_id):
    """Start one future-admitted, bounded child on an already closed copy.

    ``root`` is a NEW private directory beside copied ``canonical.duckdb``.
    The caller must first close its original fixture gateway/connection. Its
    snapshot, sidecars, source pins and nonadmitting fixture manifest are kept
    exact; only actual creator/owner/context identity fields are replaced.
    """
    root, canonical_path = Path(root), Path(canonical_path)
    if (not root.is_absolute() or root.parent.resolve(strict=True) != root.parent
            or root.exists() or root.is_symlink() or root.name in (".", "..")
            or canonical_path != root.parent / "canonical.duckdb"
            or len(os.fsencode(root / SOCKET)) > 107
            or Path(str(canonical_path) + ".wal").exists()
            or type(context_input) is not dict or type(logical_store_id) is not str or not logical_store_id
            or context_input.get("logical_store_id") != logical_store_id
            or context_input.get("stores", {}).get("canonical") != str(canonical_path)):
        raise ValueError("admin-owner new copied canonical scope differs")
    context = _decode(_json(context_input))
    custody = context.get("custody_artifacts")
    if type(custody) is not dict or set(custody) != {"nonadmitting_fixture_manifest"}:
        raise ValueError("admin-owner explicit nonadmitting fixture boundary absent")
    fixture = _decode(_read_pin(custody["nonadmitting_fixture_manifest"]))
    if (fixture.get("schema") != "nonadmitting-native-three-store-fixture@1"
            or fixture.get("fixture_only") is not True or fixture.get("production_custody_qualified") is not False
            or fixture.get("native_orchestration_receipts_created") is not False
            or Path(custody["nonadmitting_fixture_manifest"]["path"]).parent != root.parent
            or Path(context["snapshot_pin"]["path"]).parent != root.parent):
        raise ValueError("admin-owner fixture-only copy manifest scope differs")
    product_root = _product_scope(context)
    typed, policy = sys.modules.get(TYPED_MODULE), sys.modules.get(POLICY_MODULE)
    for name, module in ((TYPED_MODULE, typed), (POLICY_MODULE, policy)):
        wanted = product_root.joinpath(*name.split(".")).with_suffix(".py")
        if module is None or Path(module.__file__).resolve(strict=True) != wanted:
            raise ValueError("admin-owner parent must already load the exact genuine product")
    producer = product_root / "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py"
    published_path = product_root / "test/api/native_execution_crash_wal_fixture.py"
    published_pin = _pin(published_path)
    published = _published(published_path, published_pin["sha256"])
    installation = published.installed_duckdb_binding()
    package_pins = _package_pins(product_root)
    for value in context["source_pins"]:
        if package_pins.get(value["path"]) != value:
            raise ValueError("admin-owner required product source pin drift")
    root.mkdir(mode=0o700)
    _write(root / CONTEXT, context)
    _write(root / INSTALLATION, installation)
    started_ns = time.monotonic_ns()
    marker = {"schema": "nonadmitting-real-native-admin-owner-input@1", "fixture_only": True,
        "production_custody_qualified": False, "native_orchestration_receipts_created": False,
        "nonce": uuid.uuid4().hex, "uid": os.geteuid(), "parent_pid": os.getpid(),
        "parent_start_time_ticks": _ticks(os.getpid()), "parent_process_birth_id": typed.kernel_process_birth_id(os.getpid()),
        "owner_process_group": os.getpgrp(), "boot_id": _boot(), "logical_store_id": logical_store_id,
        "start_monotonic_ns": started_ns,
        "context_pin": _pin(root / CONTEXT, MAX_INPUT, private=True),
        "installation_pin": _pin(root / INSTALLATION, MAX_INPUT, private=True),
        "canonical_pin": _pin(canonical_path, MAX_DATABASE, private=True),
        "product_root": str(product_root), "product_sources": package_pins,
        "helper_pin": _pin(Path(__file__).resolve(strict=True)), "published_loader_pin": published_pin,
        "python_pin": _pin(Path(sys.executable).resolve(strict=True), 128 * 1024 * 1024),
        "producer_source": str(producer)}
    _write(root / MARKER, marker)
    marker_sha = _pin(root / MARKER, MAX_INPUT, private=True)["sha256"]
    # In-memory parent comparison value is not a second serialized authority.
    marker["context_input"] = context
    started = started_ns / 1_000_000_000
    process = subprocess.Popen(fixture_argv(root, marker_sha), cwd=root, env=_environment(root),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        close_fds=True, start_new_session=False)
    owner = None
    try:
        owner = AdminOwner(process, root, marker, marker_sha, started, typed)
        if os.getpgid(process.pid) != os.getpgrp():
            raise AssertionError("actual admin-owner child escaped owner process group")
        return owner._ready()
    except BaseException as error:
        if owner is not None:
            owner._failure(error)
            owner._force_reap()
            for stream in (process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
            owner._selector.close()
        else:
            # Still the exact unreaped Popen child; do not signal a group.
            if process.poll() is None and os.getpgid(process.pid) == os.getpgrp():
                process.kill()
            process.wait(timeout=REAP_SECONDS)
        raise


def _no_external_actions(event, args):
    # Host fixture restrictions; no product scope/audit/native-handle validator
    # is replaced. The genuine AF_UNIX listener remains available to the parent.
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.exec", "socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("admin-owner fixture forbids external process/network actions")
    if event == "socket.__new__" and len(args) > 1 and args[1] != 1:  # AF_UNIX == 1 on Linux.
        raise RuntimeError("admin-owner fixture permits AF_UNIX sockets only")


def serve(args):
    # Independent child wall-clock fence also covers input checks and imports.
    # No handler converts expiry into a fabricated successful close witness.
    signal.setitimer(signal.ITIMER_REAL, CHILD_SECONDS + REAP_SECONDS)
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_DATABASE, MAX_DATABASE))
    resource.setrlimit(resource.RLIMIT_NOFILE, (192, 192))
    root = args.fixture_root
    sys.pycache_prefix = str(root / "unwritten-isolated-pycache")
    if (not root.is_absolute() or root.resolve(strict=True) != root or root.stat().st_uid != os.geteuid()
            or stat.S_IMODE(root.stat().st_mode) != 0o700
            or set(path.name for path in root.iterdir()) != {MARKER, CONTEXT, INSTALLATION}
            or _pin(root / MARKER, MAX_INPUT, private=True)["sha256"] != args.marker_sha256):
        raise ValueError("actual admin-owner private input directory differs")
    marker = _decode(_regular(root / MARKER, MAX_INPUT, private=True))
    if (marker.get("schema") != "nonadmitting-real-native-admin-owner-input@1"
            or marker.get("fixture_only") is not True or marker.get("production_custody_qualified") is not False
            or marker.get("native_orchestration_receipts_created") is not False
            or marker.get("uid") != os.geteuid() or marker.get("parent_pid") != os.getppid()
            or marker.get("parent_start_time_ticks") != _ticks(os.getppid())
            or marker.get("owner_process_group") != os.getpgrp() or marker.get("boot_id") != _boot()
            or type(marker.get("start_monotonic_ns")) is not int or marker["start_monotonic_ns"] <= 0
            or marker.get("helper_pin", {}).get("path") != str(Path(__file__).resolve(strict=True))
            or marker.get("python_pin", {}).get("path") != str(Path(sys.executable).resolve(strict=True))):
        raise ValueError("actual admin-owner input/parent/source identity differs")
    elapsed = (time.monotonic_ns() - marker["start_monotonic_ns"]) / 1_000_000_000
    if not 0 <= elapsed < CHILD_SECONDS - TEARDOWN_RESERVE:
        raise TimeoutError("actual admin-owner input/start deadline")
    signal.setitimer(signal.ITIMER_REAL, CHILD_SECONDS + REAP_SECONDS - elapsed)
    _read_pin(marker["helper_pin"])
    _read_pin(marker["python_pin"], 128 * 1024 * 1024)
    context = _decode(_read_pin(marker["context_pin"], MAX_INPUT, private=True))
    _read_pin(marker["installation_pin"], MAX_INPUT, private=True)
    canonical = Path(marker["canonical_pin"]["path"])
    if (canonical != root.parent / "canonical.duckdb" or Path(str(canonical) + ".wal").exists()
            or context["stores"]["canonical"] != str(canonical)
            or context["logical_store_id"] != marker["logical_store_id"]):
        raise ValueError("actual admin-owner copied canonical path differs")
    _read_pin(marker["canonical_pin"], MAX_DATABASE, private=True)
    product_root = _product_scope(context)
    if str(product_root) != marker["product_root"]:
        raise ValueError("actual admin-owner product root changed")
    pins = marker["product_sources"]
    if type(pins) is not dict or not 1 <= len(pins) <= MAX_PRODUCT_FILES:
        raise ValueError("actual admin-owner product closure bound differs")
    total = 0
    actual_paths = {str(path) for path in (product_root / "ipfs_accelerate_py").rglob("*.py")}
    if set(pins) != actual_paths:
        raise ValueError("actual admin-owner product source population changed")
    for name, value in pins.items():
        if name != value["path"] or not Path(name).is_relative_to(product_root / "ipfs_accelerate_py"):
            raise ValueError("actual admin-owner product pin crosses root")
        total += len(_read_pin(value))
        if total > MAX_PRODUCT_BYTES:
            raise ValueError("actual admin-owner aggregate source bound")
    for value in context["source_pins"]:
        if pins.get(value["path"]) != value:
            raise ValueError("actual admin-owner context/closure source mismatch")
    published_path = product_root / "test/api/native_execution_crash_wal_fixture.py"
    if marker["published_loader_pin"]["path"] != str(published_path):
        raise ValueError("actual admin-owner published loader crosses product root")
    sys.addaudithook(_no_external_actions)
    published = _published(published_path, marker["published_loader_pin"]["sha256"])
    duckdb, installed = published._load_exact_installed_duckdb(root, marker["installation_pin"]["sha256"])
    sys.path.insert(0, str(product_root))
    sys.meta_path.insert(0, _PinnedProductFinder(product_root, pins))
    typed = __import__(TYPED_MODULE, fromlist=["TypedStateOwnerGateway"])
    policy = __import__(POLICY_MODULE, fromlist=["verify_custody"])
    contracts = __import__("ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts", fromlist=["content_identity"])
    _loaded_product_sources(pins)
    if typed.kernel_process_birth_id(os.getppid()) != marker["parent_process_birth_id"]:
        raise ValueError("actual parent kernel birth differs")
    fixture_pin = context["custody_artifacts"]["nonadmitting_fixture_manifest"]
    fixture = policy.decode(policy.read_pin(fixture_pin))
    snapshot = policy.decode(policy.read_pin(context["snapshot_pin"]))
    if (set(context["custody_artifacts"]) != {"nonadmitting_fixture_manifest"}
            or fixture.get("schema") != "nonadmitting-native-three-store-fixture@1"
            or fixture.get("fixture_only") is not True or fixture.get("production_custody_qualified") is not False
            or fixture.get("native_orchestration_receipts_created") is not False):
        raise ValueError("actual child nonadmitting custody manifest differs")

    def explicit_nonadmitting_custody(value, registered, *, check_current_file):
        # The only product seam. No production orchestration witness exists.
        if (value["custody_artifacts"] != {"nonadmitting_fixture_manifest": fixture_pin}
                or policy.decode(policy.read_pin(fixture_pin)) != fixture
                or policy.decode(policy.read_pin(value["snapshot_pin"])) != registered
                or registered != snapshot
                or (check_current_file and _identity(value["stores"]["execution"])
                    != fixture["maintenance_receipt"]["execution_file_identity"])):
            raise ValueError("explicit nonadmitting custody boundary differs")

    policy.verify_custody = explicit_nonadmitting_custody
    owner_identity = {"server_id": "fixture-admin-owner-" + uuid.uuid4().hex,
                      "generation": 1, "store_id": marker["logical_store_id"]}
    context.pop("context_cid", None)
    context["registered_owner_identity"] = owner_identity
    context["registered_creator"] = {"pid": os.getpid(), "process_birth_id": typed.kernel_process_birth_id(os.getpid())}
    context["context_cid"] = contracts.content_identity(context)
    connection, gateway = None, None
    stop_requested = False
    # Child lifetime is also bounded locally, independent of parent polling.
    deadline = marker["start_monotonic_ns"] / 1_000_000_000 + CHILD_SECONDS - TEARDOWN_RESERVE
    try:
        connection = duckdb.connect(str(canonical), config={"threads": "1", "memory_limit": "256MB",
            "autoinstall_known_extensions": "false", "autoload_known_extensions": "false",
            "enable_external_access": "false", "allow_unsigned_extensions": "false", "lock_configuration": "true"})
        gateway = typed.TypedStateOwnerGateway(connection=connection, socket_path=root / SOCKET,
            store_id=marker["logical_store_id"], identity=owner_identity)
        gateway.start()
        gateway.bind_planning_unknown_abandonment_service(expected_identity=owner_identity, authority_context=context)
        token, grant = gateway.issue_planning_unknown_abandonment_grant(peer_pid=marker["parent_pid"], ttl_seconds=20.0)
        _loaded_product_sources(pins)
        ready = {"schema": "nonadmitting-real-native-admin-owner-ready@1", "fixture_only": True,
            "production_custody_qualified": False, "native_orchestration_receipts_created": False,
            "child_pid": os.getpid(), "child_start_time_ticks": _ticks(os.getpid()), "boot_id": _boot(),
            "owner_process_group": os.getpgrp(), "marker_sha256": args.marker_sha256,
            "owner_identity": dict(gateway.identity), "context": context, "socket_path": str(gateway.socket_path),
            "token": token, "grant": grant.public_dict(),
            "duckdb_package_origin": installed["package_origin"], "duckdb_native_origin": installed["native_origin"],
            "duckdb_version": installed["version"]}
        _write(root / READY, ready)
        while time.monotonic() < deadline:
            if os.getppid() != marker["parent_pid"] or _ticks(os.getppid()) != marker["parent_start_time_ticks"]:
                raise RuntimeError("actual admin-owner parent disappeared or changed")
            if (root / STOP).exists():
                expected = {"schema": "nonadmitting-real-native-admin-owner-stop@1",
                    "nonce": marker["nonce"], "marker_sha256": args.marker_sha256,
                    "parent_pid": marker["parent_pid"], "parent_start_time_ticks": marker["parent_start_time_ticks"],
                    "child_pid": os.getpid(), "child_start_time_ticks": _ticks(os.getpid()), "boot_id": _boot()}
                if _decode(_regular(root / STOP, 8192, private=True)) != expected:
                    raise ValueError("actual admin-owner exact stop marker differs")
                stop_requested = True
                break
            time.sleep(0.025)
        if not stop_requested:
            raise TimeoutError("actual admin-owner finite stop deadline")
        gateway.stop()
        gateway = None
        connection.close()
        connection = None
        _read_pin(marker["helper_pin"])
        _read_pin(marker["published_loader_pin"])
        _loaded_product_sources(pins)
        _write(root / EXIT, {"schema": "nonadmitting-real-native-admin-owner-exited@1",
            "fixture_only": True, "production_custody_qualified": False,
            "native_orchestration_receipts_created": False, "child_pid": os.getpid(),
            "child_start_time_ticks": _ticks(os.getpid()), "boot_id": _boot(),
            "marker_sha256": args.marker_sha256, "stop_requested": True,
            "gateway_stopped": True, "connection_closed": True})
    finally:
        if gateway is not None:
            gateway.stop()
        if connection is not None:
            connection.close()
    os._exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve-admin", action="store_true", required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--marker-sha256", required=True)
    args = parser.parse_args()
    try:
        serve(args)
    except Exception as error:
        # No tokens/context/raw native rows are emitted into outer diagnostics.
        sys.stderr.write(json.dumps({"fixture_failed": True, "error_type": type(error).__name__,
            "error_message": str(error).encode("utf-8", "replace")[:4096].decode("utf-8", "ignore")}) + "\n")
        sys.stderr.flush()
        os._exit(2)


if __name__ == "__main__":
    main()
