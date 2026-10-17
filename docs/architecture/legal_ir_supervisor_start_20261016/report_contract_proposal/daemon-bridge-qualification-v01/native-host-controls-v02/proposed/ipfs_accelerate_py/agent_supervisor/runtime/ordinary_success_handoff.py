"""Opt-in trusted ordinary runner handoff; no accepting CLI or native grant.

Only run_ordinary_with_success_handoff is an effectful public entry. It retains
actual same-interpreter native/process objects; serialized summaries are inert.
Default CLI callers never establish this optional context and keep behavior.
"""
from __future__ import annotations

import argparse
from contextvars import ContextVar
from contextlib import contextmanager
import threading
from dataclasses import dataclass
import hashlib
import importlib
import os
from pathlib import Path
import re
import select
import stat
import subprocess
import sys
import tempfile
from types import MappingProxyType, FunctionType
import uuid

_ACTIVE = ContextVar("ordinary_success_handoff", default=None)
_NATIVE = "ipfs_datasets_py.optimizers.logic_theorem_optimizer."
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_DISPATCH_SOURCES = (
    "ipfs_accelerate_py.llm_router", "ipfs_accelerate_py.agent_implementation_route",
    "ipfs_accelerate_py.agent_supervisor.runtime.hash_pressure",
    "ipfs_accelerate_py.agent_supervisor.runtime.provider_command_binding",
    "ipfs_accelerate_py.agent_supervisor.runtime.provider_command_environment",
    "ipfs_accelerate_py.agent_supervisor.runtime.provider_failure_policy",
    "ipfs_accelerate_py.agent_supervisor.validation.validation_runtime",
    "ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator",
    "ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle",
    "ipfs_accelerate_py.agent_supervisor.control.provider_attempt_store",
)

_START_MARKER = b"LEGAL_IR_ORDINARY_PROVIDER_START_V1\n"
_START_GATE = """import os,select,sys,time
expected=b'LEGAL_IR_ORDINARY_PROVIDER_START_V1\\n';received=bytearray();deadline=time.monotonic()+180
while len(received)<len(expected):
 remaining=deadline-time.monotonic()
 if remaining<=0 or not select.select([0],[],[],remaining)[0]:raise SystemExit(125)
 byte=os.read(0,1)
 if not byte:raise SystemExit(125)
 received.extend(byte)
if bytes(received)!=expected:raise SystemExit(125)
os.execv(sys.argv[1],sys.argv[1:])
"""


class OrdinaryHandoffRefusal(RuntimeError):
    """Required handoff/closeout unavailable; successful dispatch not inferred."""


class OrdinaryCustodyPending(OrdinaryHandoffRefusal):
    """Actual trusted caller must hold native contexts; never serialize refs.

    These are actual in-memory cleanup references, not a native grant/receipt.
    The caller must not unwind native parents until independent cleanup proves
    quiescence. Legacy watchdog group is distinct from attached transport.
    """
    def __init__(self, owner, watchdog_owner, leases, processes, fence, daemon_id, runtime, *, custody_hold=None, private_allocations=()):
        super().__init__("actual_ordinary_allocations_require_retained_native_custody")
        self.actual_owner = owner
        self.actual_watchdog_owner = watchdog_owner
        self.actual_leases = tuple(leases)
        self.actual_transport_processes = tuple(processes)
        self.actual_container_fence = _readonly(fence) if fence is not None else None
        self.actual_daemon_id = daemon_id
        self.actual_runtime = _readonly(runtime) if runtime is not None else None
        self.actual_custody_hold = custody_hold
        self.actual_private_allocations = tuple(private_allocations)

    def __reduce__(self):
        raise OrdinaryHandoffRefusal("actual_custody_references_not_serializable")


def need(condition, reason):
    if not condition:
        raise OrdinaryHandoffRefusal(reason)


def _readonly(value):
    if type(value) is dict:
        return MappingProxyType({key: _readonly(item) for key, item in value.items()})
    if type(value) in (list, tuple):
        return tuple(_readonly(item) for item in value)
    return value


@dataclass(frozen=True)
class OrdinaryPrimaryRoute:
    """Source-pinned runner's actual selected primary; no fallback authority."""
    primary_provider_id: str = "grok_cli"
    primary_model_id: str = "grok-4.7"
    invocation_binding: object = None
    route_kind: str = "primary_without_preflight"
    execution_id: str = ""


def check_primary_only_options(args, receipt_fd=-1, environment=None):
    """Pure scope refusal before invocation; no route/execution authority."""
    need(type(args) is argparse.Namespace, "actual_parsed_runner_namespace_required")
    environment = os.environ if environment is None else environment
    need(type(receipt_fd) is int and receipt_fd == -1
         and getattr(args, "receipt_fd_declared", False) is False
         and not any(name in environment for name in (
             "IPFS_ACCELERATE_GROK_TERMINAL_RECEIPT_FD",
             "IPFS_ACCELERATE_AGENT_TRUSTED_FAILURE_RECEIPT_FD")),
         "primary_only_handoff_forbids_supervised_receipt_dispatch")
    need(not str(getattr(args, "agent_implementation_recovery_json", "") or "").strip(),
         "ordinary_handoff_forbids_protected_recovery")
    need(str(getattr(args, "codex_fallback_command_json", "") or "").strip() in {"", "[]"}
         and not str(getattr(args, "grok_failure_receipt_nonce", "") or "").strip()
         and not str(getattr(args, "agent_implementation_route_json", "") or "").strip()
         and getattr(args, "canonical_legacy_preflight_route", False) is False,
         "primary_only_handoff_forbids_ungated_preflight_or_fallback")


@dataclass(frozen=True)
class OrdinaryProviderSuccess:
    """Actual objects delivered by the trusted wrapper, never acceptance authority.

    Merely constructing this data class does not authenticate an outcome. The
    supported path is the actual source-pinned wrapper callback invocation.
    Native checker/coordinator must independently verify their own full inputs.
    """
    execution_id: str
    owner: object
    source_sha256: object
    route: object
    provider_id: str
    model_id: str
    reasoning_effort: str
    provider_trigger: str
    route_invocation_id: str
    transport_process: subprocess.Popen
    model_host_pid: int
    model_start_time_ticks: int
    container_fence: object
    docker_daemon_id: str
    container_id: str
    container_name: str
    actual_returncode: int
    watchdog_owner: object
    callable_invocation: object = None
    observation_only: bool = True
    native_validation_qualified: bool = False
    acceptance_authority: bool = False

    def __reduce__(self):
        raise OrdinaryHandoffRefusal("same_interpreter_outcome_not_serializable")

    def __copy__(self):
        raise OrdinaryHandoffRefusal("actual_outcome_not_copyable")

    def __deepcopy__(self, memo):
        raise OrdinaryHandoffRefusal("actual_outcome_not_copyable")


def check_lifecycle_projection(value):
    """Pure necessary conditions; a caller mapping never authenticates an effect."""
    need(type(value) is dict, "lifecycle_projection_required")
    need(type(value.get("execution_id")) is str and re.fullmatch(r"[0-9a-f]{32}", value["execution_id"]),
         "fresh_execution_identity_required")
    need(value.get("provider_calls") == 1 and type(value.get("provider_calls")) is int,
         "exactly_one_task_provider_required")
    need(type(value.get("returncode")) is int and value["returncode"] == 0,
         "successful_provider_return_required")
    for key in ("process_reaped", "lease_closed", "watchdog_reaped", "watchdog_owner_group_quiescent", "kernel_scope_quiescent",
                "exact_container_absent", "lease_root_absent", "provider_home_absent", "prompt_path_absent", "daemon_unchanged", "runtime_unchanged"):
        need(value.get(key) is True, "actual_closeout_missing:" + key)
    route = (value.get("provider_id"), value.get("model_id"), value.get("reasoning_effort"), value.get("provider_trigger"))
    need(route == ("grok_cli", "grok-4.7", "", "primary"),
         "actual_supported_ordinary_route_required")
    worker, owner, transport = (value.get(key) for key in ("model_host_pid", "owner_pid", "transport_pid"))
    need(all(type(item) is int and item > 1 for item in (worker, owner, transport))
         and worker not in {owner, transport}, "actual_model_process_role_required")
    need(type(value.get("model_start_time_ticks")) is int and value["model_start_time_ticks"] > 0,
         "actual_model_birth_required")
    return None


def check_success_order(events):
    """Pure lifecycle ordering constraint; event strings supply no authority."""
    required = ("invocation_started", "route_selected", "task_started", "watchdog_native_attached", "lease_created",
                "transport_started", "native_child_attached", "container_scope_captured", "provider_start_released", "provider_reaped", "lease_closed",
                "kernel_scope_quiescent", "native_owner_checked", "callback_ready")
    need(type(events) in (tuple, list) and tuple(events) == required,
         "successful_handoff_must_follow_actual_closeout")


def check_external_storage_declarations(public, declarations):
    """Join explicit host declarations to actual SDK hashed charge labels.

    This is a necessary correspondence check, not path measurement or a disk
    grant. The trusted caller separately pins its declaration/source request.
    Actual account_external_bytes stores key->bytes, not filesystem paths.
    """
    need(type(public) is dict and type(declarations) in (tuple, list)
         and 1 <= len(declarations) <= 16, "declared_external_storage_required")
    record = public.get("record")
    charges = record.get("external_charges") if type(record) is dict else None
    roots = public.get("roots")
    need(type(charges) is dict and type(roots) is list, "actual_external_charge_inventory_required")
    paths, result = set(), []
    for declaration in declarations:
        need(type(declaration) is dict and set(declaration) == {"path", "bytes"},
             "exact_external_storage_declaration_required")
        spelling, size = declaration["path"], declaration["bytes"]
        need(type(spelling) is str and Path(spelling).is_absolute() and os.path.abspath(spelling) == spelling
             and ".." not in Path(spelling).parts and type(size) is int and size > 0 and spelling not in paths,
             "canonical_unique_positive_external_storage_required")
        # This is the existing public-owner recipe's explicit label convention.
        # No label is ever interpreted as a path or asserted to measure bytes.
        key = hashlib.sha256(spelling.encode()).hexdigest()
        need(type(charges.get(key)) is int and charges[key] == size,
             "actual_hashed_external_charge_differs")
        path = Path(spelling)
        need(any(type(root) is dict and type(root.get("path")) is str
                 and Path(root["path"]) in path.parents for root in roots),
             "declared_external_storage_outside_named_roots")
        paths.add(spelling)
        result.append((spelling, size))
    return tuple(result)


def _read_pinned(path, expected):
    path = Path(path)
    need(path.is_absolute() and str(path) == os.path.abspath(path)
         and not any(item.is_symlink() for item in (path, *path.parents)), "unaliased_pinned_source_required")
    need(type(expected) is str and _SHA.fullmatch(expected), "exact_source_digest_required")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_uid in {0, os.getuid()}
             and before.st_nlink == 1 and not before.st_mode & 0o002
             and 0 < before.st_size <= 256 * 1024 * 1024, "bounded_owned_trusted_source_required")
        count, digest = 0, hashlib.sha256()
        while True:
            raw = stream.read(1024 * 1024)
            if not raw:
                break
            count += len(raw)
            need(count <= before.st_size, "trusted_source_changed_during_read")
            digest.update(raw)
        after = os.fstat(stream.fileno())
    witness = lambda m: (m.st_dev, m.st_ino, m.st_mode, m.st_uid, m.st_gid, m.st_nlink,
                         m.st_size, m.st_mtime_ns, m.st_ctime_ns)
    need(witness(before) == witness(after) == witness(path.lstat()) and count == before.st_size,
         "trusted_source_changed_during_read")
    need(digest.hexdigest() == expected, "actual_handoff_source_changed")


def _guard_owner(owner, *, watchdog=False):
    need(type(owner).__module__ == _NATIVE + "parented_phase_resources"
         and type(owner).__name__ == "ParentedPhaseResourceReservation", "genuine_parented_owner_required")
    native = importlib.import_module(_NATIVE + "parented_phase_resources")
    need(type(owner) is native.ParentedPhaseResourceReservation, "genuine_parented_owner_required")
    need(owner.heartbeat(), "actual_owner_heartbeat_failed")
    owner.check_usage(owner.directory, full=True)
    lease, public = owner.native_lease.to_dict(), owner.to_dict()
    need(lease.get("owner_pid") == os.getpid() and lease.get("lane") == "orchestration"
         and lease.get("released") is False and lease.get("cancelled") is False
         and lease.get("cpu_slots", 0) >= 1 and lease.get("memory_mb", 0) >= (256 if watchdog else 512)
         and lease.get("child_process_slots", 0) >= (4 if watchdog else 8) and public.get("status") == "active",
         "actual_live_ordinary_owner_required")


def check_owner_pair_projection(primary, watchdog):
    """Pure necessary sibling/freshness check; mappings are never admission."""
    need(type(primary) is dict and type(watchdog) is dict,
         "actual_two_owner_inventory_required")
    first, second = primary.get("resource_lease"), watchdog.get("resource_lease")
    need(type(first) is dict and type(second) is dict,
         "actual_two_native_lease_inventories_required")
    need(type(primary.get("reservation_id")) is str and primary["reservation_id"]
         and type(watchdog.get("reservation_id")) is str and watchdog["reservation_id"]
         and primary["reservation_id"] != watchdog["reservation_id"]
         and type(first.get("lease_id")) is str and first["lease_id"]
         and type(second.get("lease_id")) is str and second["lease_id"]
         and first["lease_id"] != second["lease_id"]
         and type(first.get("parent_lease_id")) is str and first["parent_lease_id"]
         and first["parent_lease_id"] == second.get("parent_lease_id"),
         "distinct_sibling_native_owners_required")
    for public in (primary, watchdog):
        record = public.get("record")
        need(type(record) is dict and record.get("child") is None
             and record.get("prior_children", []) == [], "fresh_native_owner_child_inventory_required")


def _verify_sources(owner, pins, runner_path):
    need(type(pins) is dict and all(type(path) is str and type(value) is str and _SHA.fullmatch(value) for path, value in pins.items()),
         "exact_handoff_source_pins_required")
    paths = {str(Path(__file__).absolute()), str(runner_path),
             str(Path(__file__).with_name("ordinary_custody_hold.py").absolute()),
             str(runner_path.with_name("process_security.py")), "/usr/bin/docker", str(Path(sys.executable).resolve())}
    for name in ("parented_phase_resources", "autoencoder_daemon_resources", "existing_resource_scheduler", "resource_scheduler"):
        module = sys.modules.get(_NATIVE + name)
        need(module is not None and getattr(module, "__file__", None), "actual_native_source_module_required")
        paths.add(str(Path(module.__file__).absolute()))
    need(paths <= set(pins) and len(pins) <= 128, "exact_handoff_source_pins_required")
    for name in _DISPATCH_SOURCES:
        suffix = "/" + name.replace(".", "/") + ".py"
        need(sum(path.endswith(suffix) for path in pins) == 1, "exact_dispatch_helper_source_pins_required")
    for path, expected in pins.items():
        _read_pinned(path, expected)
    # Before import, every required helper file is already exact-pinned. After
    # import, join actually executed module paths to the same supplied closure.
    runner = sys.modules.get("ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner")
    if runner is not None and Path(getattr(runner, "__file__", "")).absolute() == runner_path:
        for name in _DISPATCH_SOURCES:
            # Some ordinary helpers are imported lazily by _run. Their source
            # has already been pinned above; load definitions without dispatch
            # and join the actual import path before allowing provider work.
            module = importlib.import_module(name)
            need(module is not None and getattr(module, "__file__", None)
                 and str(Path(module.__file__).absolute()) in pins,
                 "actual_dispatch_helper_source_differs")


class _Lifecycle:
    def __init__(self, owner, watchdog_owner, source_sha256, runner, execution_id, external_storage):
        self.owner, self.source_sha256, self.runner = owner, source_sha256, runner
        self.watchdog_owner = watchdog_owner
        self.execution_id, self.route, self.model = execution_id, None, None
        self.provider_id, self.effort, self.trigger = "grok_cli", "", "primary"
        self.task_active, self.leases, self.processes = False, {}, []
        self.selected_lease, self.fence, self.returncode = None, None, None
        self.closed, self.failed, self.daemon_id, self.runtime = False, False, None, None
        self.events = ["invocation_started"]
        self.gated_create = None
        self.external_storage = tuple(dict(row) for row in external_storage)
        self.allocated_leases = []
        self.invocation = None
        self.trusted_binary = None
        self.custody_hold = None
        self.custody_module = None
        self.callback_outcome = None
        self.runner_entered = False
        self.creation_intents = set()
        self.known_created = {}
        self.private_allocations = []
        self.original_groups = {}
        self.owner_thread = threading.get_ident()
        self.primary_native, self.watchdog_native = owner.native_lease, watchdog_owner.native_lease
        self.callback_capture, self.freezer_scope, self.stopped_pending = None, None, None

    def guard_live(self):
        # Main-thread bounded I/O calls this between <=0.5s select/wait slices.
        # Heartbeat cannot reset either admitted native deadline.
        need(self.selected_lease is not None and self.selected_lease._watchdog.poll() is None,
             "actual_watchdog_must_remain_alive_during_provider_work")
        for owner in (self.owner, self.watchdog_owner):
            need(owner.heartbeat(), "actual_owner_heartbeat_failed")
            owner.check_usage(owner.directory)

    def query(self, arguments):
        _ordinary_admin_capacity(self)
        code, output, _error = self.runner._bounded_docker_query(
            ["/usr/bin/docker", "--host=unix:///var/run/docker.sock", "--config", "/nonexistent", *arguments], timeout=5)
        need(code == 0 and len(output) <= 4096, "actual_docker_observation_unavailable")
        return output.strip()

    def daemon(self):
        raw = self.query(["info", "--format", "{{.ID}}"])
        value = raw.decode("ascii")
        need(re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", value), "actual_docker_daemon_identity_required")
        return value

    def select(self, route, model):
        need(not self.task_active and self.route is None and model == "grok-4.7" and route is None,
             "actual_primary_without_preflight_required")
        self.route, self.model = OrdinaryPrimaryRoute(primary_model_id=model, execution_id=self.execution_id), model
        self.events.append("route_selected")

    def fallback(self, route, decision):
        # V01 refuses preflight/fallback configuration before _run. Retain a
        # fail-closed internal guard if a future callsite drifts into this API.
        raise OrdinaryHandoffRefusal("primary_only_handoff_has_no_fallback_authority")

    def lease_created(self, lease, provider):
        if not self.task_active:
            return
        # Retain actual factory allocation before any fallible observation.
        self.allocated_leases.append(lease)
        self.selected_lease = lease
        need(type(lease) is self.runner._LegacyDockerContainerLease and provider == "grok"
             and len(self.allocated_leases) == 1 and not self.leases, "actual_single_ordinary_docker_lease_required")
        need(lease.docker_bin == "/usr/bin/docker" and type(lease._watchdog) is subprocess.Popen
             and lease._watchdog.poll() is None, "actual_ordinary_watchdog_required")
        _capture_native_group(self, self.watchdog_owner, lease._watchdog)
        self.watchdog_owner.check_usage(self.watchdog_owner.directory,
                                        child_pid=lease._watchdog.pid, full=True)
        need(self.watchdog_owner.heartbeat(), "actual_watchdog_owner_heartbeat_failed")
        self.events.append("watchdog_native_attached")
        declarations = check_external_storage_declarations(self.owner.to_dict(), self.external_storage)
        need(any(Path(path) in lease.lease_root.parents for path, _bytes in declarations),
             "actual_provider_temporary_storage_precharge_required")
        self.daemon_id = self.daemon()
        self.runtime = self.runner._docker_runtime_receipt(lease.docker_bin)
        self.leases[id(lease)] = (lease, provider, self.runner._runner_process_start_ticks(lease._watchdog.pid))
        self.selected_lease = lease
        self.events.append("lease_created")

    def started(self, process, command):
        if not self.task_active:
            return
        # Retain the exact actual child before any further fallible observation.
        self.processes.append(process)
        need(type(process) is subprocess.Popen and len(self.processes) == 1
             and process.poll() is None and self.selected_lease is not None,
             "actual_single_running_provider_transport_required")
        lease, provider, _watchdog_birth = self.leases[id(self.selected_lease)]
        cid = lease.cidfile.read_text(encoding="ascii").strip()
        need(re.fullmatch(r"[0-9a-f]{64}", cid), "actual_created_ordinary_cid_required")
        expected = ["/usr/bin/docker", "--host=unix:///var/run/docker.sock", "--config", str(lease.docker_config),
                    "start", "--attach", "--interactive", cid]
        need(list(command) == expected and process.args == expected,
             "actual_provider_transport_argv_differs")
        need(self.gated_create is not None and process.stdin is not None, "actual_gated_ordinary_start_required")
        self.events.append("transport_started")
        _capture_native_group(self, self.owner, process)
        self.owner.check_usage(self.owner.directory, child_pid=process.pid, full=True)
        need(self.owner.heartbeat(), "actual_owner_heartbeat_failed")
        self.events.append("native_child_attached")
        image = self.runner._docker_ordinary_task_toolchain_image_id(lease.docker_bin, docker_config=lease.docker_config)
        need(image and self.daemon() == self.daemon_id, "actual_ordinary_image_or_daemon_changed")
        self.fence = self.runner._attest_exact_docker_execution(
            docker_bin=lease.docker_bin, docker_config=lease.docker_config, provider=provider,
            container_name=lease.container_name, container_id=cid, image_id=image, timeout=2)
        need(self.fence.get("init_pid", 0) > 1 and self.fence["init_pid"] not in {os.getpid(), process.pid}
             and self.fence.get("kernel_scope"), "actual_untrusted_model_container_birth_required")
        self.events.append("container_scope_captured")
        _verify_sources(self.owner, self.source_sha256, Path(self.runner.__file__).absolute())
        self.guard_live()
        _release_primary_start_and_close_stdin(process.stdin)
        self.events.append("provider_start_released")

    def returned(self, process, returncode):
        if not self.task_active:
            return
        need(len(self.processes) == 1 and process is self.processes[0]
             and type(returncode) is int and process.poll() == returncode and process.returncode == returncode,
             "actual_provider_wait_correspondence_required")
        self.returncode = returncode
        self.failed |= returncode != 0
        self.events.append("provider_reaped")

    def lease_closed(self, lease):
        if not self.task_active or id(lease) not in self.leases:
            return
        need(lease is self.selected_lease and lease._closed is True and not lease.preserve_for_recovery,
             "actual_provider_lease_closeout_incomplete")
        need(type(lease._watchdog.returncode) is int and lease._watchdog.poll() == lease._watchdog.returncode,
             "actual_watchdog_not_reaped")
        need(not os.path.lexists(lease.lease_root) and not os.path.lexists(lease.cidfile),
             "actual_provider_lease_paths_remain")
        self.closed = True
        self.events.append("lease_closed")

    def finish(self, actual_returncode):
        need(not self.failed and len(self.processes) == 1 and self.fence is not None,
             "actual_task_provider_lifecycle_missing")
        need(type(self.route) is OrdinaryPrimaryRoute and self.route.execution_id == self.execution_id,
             "actual_primary_route_execution_identity_changed")
        process, lease = self.processes[0], self.selected_lease
        need(self.returncode == actual_returncode == process.returncode and process.poll() == 0,
             "actual_successful_reaped_provider_required")
        cid, name = self.fence["container_id"], self.fence["container_name"]
        absent_id = not self.query(["container", "ls", "--all", "--no-trunc", "--filter", "id=" + cid, "--format", "{{.ID}}"])
        absent_name = not self.query(["container", "ls", "--all", "--no-trunc", "--filter", "name=^/" + name + "$", "--format", "{{.Names}}"])
        scope = self.fence["kernel_scope"]
        projection = dict(execution_id=self.execution_id, provider_calls=len(self.processes), returncode=actual_returncode,
            process_reaped=process.poll() == 0, lease_closed=self.closed,
            watchdog_reaped=type(lease._watchdog.returncode) is int and lease._watchdog.poll() == lease._watchdog.returncode,
            watchdog_owner_group_quiescent=self.watchdog_owner.check_usage(self.watchdog_owner.directory, full=True)["group_rss"]["live_processes"] == 0,
            kernel_scope_quiescent=self.runner._docker_termination_scope_quiescent(self.fence),
            exact_container_absent=bool(absent_id and absent_name), lease_root_absent=not os.path.lexists(lease.lease_root),
            provider_home_absent=not os.path.lexists(lease.provider_home), prompt_path_absent=not os.path.lexists(lease.prompt_path),
            daemon_unchanged=self.daemon() == self.daemon_id, runtime_unchanged=self.runner._docker_runtime_receipt("/usr/bin/docker") == self.runtime,
            provider_id=self.provider_id, model_id=self.model, reasoning_effort=self.effort, provider_trigger=self.trigger,
            model_host_pid=self.fence["init_pid"], model_start_time_ticks=scope["start_time_ticks"], owner_pid=os.getpid(), transport_pid=process.pid)
        check_lifecycle_projection(projection)
        self.events.append("kernel_scope_quiescent")
        actual_usage = self.owner.check_usage(self.owner.directory, full=True)
        need(actual_usage["group_rss"]["live_processes"] == 0, "actual_provider_transport_group_not_quiescent")
        _guard_owner(self.owner)
        _guard_owner(self.watchdog_owner, watchdog=True)
        _verify_sources(self.owner, self.source_sha256, Path(self.runner.__file__).absolute())
        self.events.extend(("native_owner_checked", "callback_ready"))
        check_success_order(self.events)
        invocation = getattr(self.route, "invocation_binding", None)
        return OrdinaryProviderSuccess(self.execution_id, self.owner, _readonly(dict(self.source_sha256)), self.route,
            self.provider_id, self.model, self.effort, self.trigger,
            getattr(invocation, "invocation_id", "") if invocation is not None else "", process,
            self.fence["init_pid"], scope["start_time_ticks"], _readonly(self.fence), self.daemon_id, cid, name, actual_returncode,
            self.watchdog_owner, callable_invocation=self.invocation)

    def reap_observed_transport_after_failure(self, *, stopped=False):
        # Existing _run owns Docker cleanup. Never signal a reused PID or
        # blindly call close again. Capture actual references on unknown state.
        pending = bool(self.creation_intents - set(self.known_created))
        unresolved_scope = self.processes and not (type(self.fence) is dict
                and type(self.fence.get("init_pid")) is int and self.fence["init_pid"] > 1
                and bool(self.fence.get("kernel_scope")))
        if unresolved_scope:
            pending = True
        for process in self.processes:
            if type(process) is subprocess.Popen:
                if unresolved_scope:
                    # Preserve the original gate input while its real init is
                    # still available for trusted stopped-cleanup observation.
                    # EOF here would erase that observation opportunity.
                    continue
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pending = True
                if process.stdin is not None and not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except OSError:
                        pending = True
        for lease in self.allocated_leases:
            if (type(lease) is not self.runner._LegacyDockerContainerLease
                    or lease.preserve_for_recovery
                    or type(lease._watchdog.returncode) is not int
                    or lease._watchdog.poll() != lease._watchdog.returncode
                    or os.path.lexists(lease.lease_root)
                    or os.path.lexists(lease.provider_home)
                    or os.path.lexists(lease.prompt_path)):
                pending = True
        # Paths/watchdog exit alone cannot prove the actual container/kernel
        # scope ended. Reobserve exact same daemon and actual captured fence,
        # even when finish/callback raised. Never issue cleanup from this hook.
        if self.daemon_id is not None:
            try:
                need(self.daemon() == self.daemon_id
                     and self.runtime is not None
                     and self.runner._docker_runtime_receipt("/usr/bin/docker") == self.runtime,
                     "actual_cleanup_daemon_or_runtime_changed")
                for lease in self.allocated_leases:
                    need(not self.query(["container", "ls", "--all", "--no-trunc", "--filter",
                                         "name=^/" + lease.container_name + "$", "--format", "{{.Names}}"]),
                         "actual_cleanup_named_container_remains")
                if self.fence is not None:
                    need(not self.query(["container", "ls", "--all", "--no-trunc", "--filter",
                                         "id=" + self.fence["container_id"], "--format", "{{.ID}}"])
                         and self.runner._docker_termination_scope_quiescent(self.fence),
                         "actual_cleanup_container_or_kernel_scope_remains")
                elif "provider_start_released" in self.events:
                    pending = True
            except Exception:
                pending = True
        elif self.processes or self.gated_create is not None:
            pending = True
        if stopped:
            try:
                _original_native_groups_stopped(self)
            except Exception:
                pending = True
        elif self.processes or self.allocated_leases:
            for owner in (self.owner, self.watchdog_owner):
                try:
                    need(owner.check_usage(owner.directory, full=True)["group_rss"]["live_processes"] == 0,
                         "actual_cleanup_native_child_group_remains")
                except Exception:
                    # Revocation/deadline refusal is not permission to infer
                    # dead children. Independent trusted cleanup stays needed.
                    pending = True
        if pending:
            raise _retain_ordinary_pending(self)


def _release_primary_start_and_close_stdin(stream):
    """Release exact primary marker, then EOF; primary prompt is a bound file.

    Called only after actual Popen/owner/container checks. V01 forbids live
    fallback/preflight. A future Codex prompt path needs its separate lifecycle
    and cannot reuse this primary-only EOF behavior.
    """
    sent = 0
    while sent < len(_START_MARKER):
        count = os.write(stream.fileno(), _START_MARKER[sent:])
        need(type(count) is int and 0 < count <= len(_START_MARKER) - sent,
             "actual_provider_start_gate_write_failed")
        sent += count
    stream.close()


def gated_create_command(command, *, image, original_entrypoint, python_entrypoint):
    """Pure bounded argv transform; preserves every original isolation option.

    Actual ordinary builders pass their generated full create argv here. It
    replaces only the entrypoint and prefixes a fixed non-secret stdin barrier
    before the exact original entrypoint/tail. This pure transform grants no
    native/Docker/start authority and performs no writes.
    """
    need(type(command) is list and all(type(item) is str and item for item in command),
         "actual_generated_ordinary_create_argv_required")
    need(type(image) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", image) and command.count(image) == 1,
         "exact_ordinary_create_image_required")
    image_index = command.index(image)
    prefix = command[:image_index]
    need(len(prefix) >= 5 and prefix[:5] == ["/usr/bin/docker", "--host=unix:///var/run/docker.sock", "--config", prefix[3], "create"]
         and "--interactive" in prefix and "--pull=never" in prefix,
         "actual_interactive_inert_ordinary_create_required")
    entries = [index for index, value in enumerate(prefix) if value.startswith("--entrypoint=")]
    need(len(entries) == 1 and prefix[entries[0]] == "--entrypoint=" + original_entrypoint
         and original_entrypoint.startswith("/") and python_entrypoint.startswith("/"),
         "exact_ordinary_entrypoint_required")
    prefix[entries[0]] = "--entrypoint=" + python_entrypoint
    return [*prefix, image, "-I", "-c", _START_GATE, original_entrypoint, *command[image_index + 1:]]


def prepare_ordinary_gated_create(command, *, image, original_entrypoint, python_entrypoint):
    current = _current()
    if current is None or not current.task_active:
        return command
    need(current.selected_lease is not None and current.gated_create is None,
         "single_actual_ordinary_create_required")
    prepared = gated_create_command(command, image=image, original_entrypoint=original_entrypoint,
                                    python_entrypoint=python_entrypoint)
    current.gated_create = tuple(prepared)
    return prepared


def ordinary_requires_docker():
    return _current() is not None


def ordinary_transport_kwargs(*, stdin_already_present=False):
    current = _current()
    if current is None or not current.task_active:
        return {}
    need(type(stdin_already_present) is bool and current.gated_create is not None,
         "actual_gated_create_before_transport_required")
    return {"start_new_session": True, **({} if stdin_already_present else {"stdin": subprocess.PIPE})}


def _read_periodically_guarded(stream, guard, maximum):
    """Private finite I/O primitive; supplied guard is not admission authority."""
    while True:
        guard()
        if select.select([stream.fileno()], [], [], 0.5)[0]:
            return os.read(stream.fileno(), maximum)


def ordinary_stderr_chunk(process, maximum):
    current = _current()
    if current is None:
        return process.stderr.read(maximum)
    need(current.task_active and len(current.processes) == 1 and process is current.processes[0],
         "actual_retained_provider_before_monitored_read_required")
    return _read_periodically_guarded(process.stderr, current.guard_live, maximum)


def _wait_periodically_guarded(process, guard):
    while True:
        guard()
        try:
            return process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass


def ordinary_provider_wait(process):
    current = _current()
    if current is None:
        return process.wait()
    need(current.task_active and len(current.processes) == 1 and process is current.processes[0],
         "actual_retained_provider_before_monitored_wait_required")
    return _wait_periodically_guarded(process, current.guard_live)


def _current():
    return _ACTIVE.get()


def select_ordinary_route(route, model):
    current = _current()
    if current is not None:
        current.select(route, model)


def begin_ordinary_task():
    current = _current()
    if current is not None:
        need(current.route is not None and not current.task_active, "actual_selected_ordinary_task_required")
        current.task_active = True
        current.events.append("task_started")


def select_ordinary_fallback(route, decision):
    current = _current()
    if current is not None:
        current.fallback(route, decision)


def ordinary_lease_created(lease, provider):
    current = _current()
    if current is not None:
        current.lease_created(lease, provider)


def ordinary_provider_started(process, command):
    current = _current()
    if current is not None:
        current.started(process, command)


def ordinary_provider_returned(process, returncode):
    current = _current()
    if current is not None:
        current.returned(process, returncode)


def ordinary_lease_closed(lease):
    current = _current()
    if current is not None:
        current.lease_closed(lease)


def run_ordinary_with_success_handoff(args, receipt_fd, *, owner=None, watchdog_owner=None,
                                      source_sha256=None, declared_external_storage=None, on_success=None,
                                      invocation=None, custody_hold=None, custody_source_sha256=None,
                                      before_run=None):
    """Run one primary-only actual ordinary invocation and hand off after fence.

    The caller already holds genuine finite owner admission including storage,
    native process inventory and provider effects. This API creates no grant,
    changes no task/counter and accepts no authority from a worker CLI flag.
    on_success(outcome) runs in this trusted interpreter after _run's finally.
    Required callback exceptions propagate; they cannot return successful exit.
    """
    if on_success is None:
        need(invocation is None and custody_hold is None and custody_source_sha256 is None
             and before_run is None, "explicit_callable_handoff_requires_success_callback")
        runner = importlib.import_module("ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner")
        return runner._run(args, receipt_fd)
    need(callable(on_success), "explicit_required_success_callback")
    need(_current() is None, "nested_ordinary_handoff_forbidden")
    need(sys.dont_write_bytecode is True, "native_host_no_bytecode_required")
    need(type(source_sha256) is dict and 1 <= len(source_sha256) <= 128,
         "finite_exact_callable_source_closure_required")
    for spelling, digest in source_sha256.items():
        _read_pinned(spelling, digest)
    _verify_callback_source(on_success, source_sha256)
    if before_run is not None:
        _verify_callback_source(before_run, source_sha256)
    _guard_owner(owner)
    _guard_owner(watchdog_owner, watchdog=True)
    need(owner is not watchdog_owner, "distinct_sibling_native_owners_required")
    check_owner_pair_projection(owner.to_dict(), watchdog_owner.to_dict())
    need(invocation is not None, "explicit_ordinary_callable_invocation_required")
    check_primary_only_options(args, receipt_fd, environment=invocation.environment)
    declarations = check_external_storage_declarations(owner.to_dict(), declared_external_storage)
    for spelling, _size in declarations:
        declared_path = Path(spelling)
        need(declared_path.is_dir() and declared_path.resolve(strict=True) == declared_path
             and not any(path.is_symlink() for path in (declared_path, *declared_path.parents)),
             "actual_unaliased_external_storage_required")
    for path in (Path(args.workspace).resolve(strict=True), Path(invocation.temporary_root).resolve(strict=True)):
        need(path != owner.directory and owner.directory not in path.parents
             and any(Path(root) == path or Path(root) in path.parents for root, _bytes in declarations),
             "actual_workspace_and_provider_temp_precharge_required")
    runner_path = Path(__file__).with_name("grok_cli_runner.py").absolute()
    _verify_sources(owner, source_sha256, runner_path)
    runner = importlib.import_module("ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner")
    need(Path(runner.__file__).absolute() == runner_path, "actual_source_pinned_runner_required")
    _verify_sources(owner, source_sha256, runner_path)
    need(type(invocation) is runner.OrdinaryCallableInvocation,
         "actual_explicit_ordinary_invocation_type_required")
    need(Path(args.workspace).absolute() == invocation.cwd,
         "actual_explicit_workspace_identity_required")
    _check_output_storage(owner, invocation, declarations)
    custody_path = Path(__file__).with_name("ordinary_custody_hold.py").absolute()
    need(type(custody_source_sha256) is dict and all(source_sha256.get(path) == digest
         for path, digest in custody_source_sha256.items()), "exact_custody_source_closure_join_required")
    custody = importlib.import_module("ipfs_accelerate_py.agent_supervisor.runtime.ordinary_custody_hold")
    need(Path(custody.__file__).absolute() == custody_path, "actual_source_pinned_custody_module_required")
    custody.verify_source_pins(custody_source_sha256)
    custody.require_bound_ordinary_custody_hold(custody_hold, primary_owner=owner, watchdog_owner=watchdog_owner)
    workspace = Path(args.workspace).resolve(strict=True)
    selected_binary = str(getattr(args, "grok_bin", "") or "").strip()
    need(selected_binary, "explicit_trusted_grok_binary_required")
    trusted_binary = runner._resolve_trusted_grok_bin(configured=selected_binary, workspace=workspace)
    need(trusted_binary and trusted_binary in source_sha256, "actual_trusted_grok_binary_pin_required")
    _read_pinned(Path(trusted_binary), source_sha256[trusted_binary])
    need(str(getattr(args, "model", "") or "").strip() in {"", "grok-4.7"},
         "actual_primary_grok_4_7_model_required")
    bound_args = argparse.Namespace(**vars(args))
    bound_args.workspace, bound_args.grok_bin, bound_args.model = workspace, selected_binary, "grok-4.7"
    lifecycle = _Lifecycle(owner, watchdog_owner, dict(source_sha256), runner, uuid.uuid4().hex, declared_external_storage)
    lifecycle.invocation, lifecycle.custody_hold, lifecycle.custody_module = invocation, custody_hold, custody
    lifecycle.trusted_binary = trusted_binary
    custody.retain_ordinary_custody_references(custody_hold, (lifecycle, invocation))
    token = _ACTIVE.set(lifecycle)
    try:
        runner.validate_ordinary_callable_invocation(bound_args, receipt_fd, invocation)
        if before_run is not None:
            need(before_run() is None, "before_run_must_return_none_or_raise")
        lifecycle.runner_entered = True
        result = runner._run(bound_args, receipt_fd, callable_invocation=invocation)
        if type(result) is not int or result != 0:
            return result
        outcome = lifecycle.finish(result)
        lifecycle.callback_outcome = outcome
        try:
            callback_result = on_success(outcome)
        finally:
            lifecycle.callback_outcome = None
        need(callback_result is None, "required_success_callback_must_return_none")
        _verify_sources(owner, source_sha256, runner_path)
        return result
    finally:
        lifecycle.callback_outcome = None
        try:
            _acknowledge_independently_stopped_custody(lifecycle)
        finally:
            _ACTIVE.reset(token)


def _verify_callback_source(callback, pins):
    need(type(callback) is FunctionType, "source_pinned_plain_host_callback_required")
    module = sys.modules.get(callback.__module__)
    path = Path(callback.__code__.co_filename).absolute()
    need(module is not None and getattr(module, "__file__", None)
         and Path(module.__file__).absolute() == path and str(path) in pins,
         "actual_host_callback_source_pin_required")
    _read_pinned(path, pins[str(path)])


def _check_output_storage(owner, invocation, declarations):
    for path in (invocation.stdout_path, invocation.stderr_path):
        need(path.is_absolute() and path.resolve(strict=True) == path
             and not any(part.is_symlink() for part in (path, *path.parents)),
             "actual_unaliased_callable_output_required")
        need(owner.directory in path.parents or any(Path(root) in path.parents
             for root, _bytes in declarations), "actual_callable_output_precharge_required")


def ordinary_callable_context_required(invocation):
    """Guard actual same-process callable admission before allocation/start."""
    current = _current()
    need(current is not None and current.invocation is invocation and invocation is not None
         and current.custody_module is not None and current.custody_hold is not None,
         "actual_admitted_callable_context_required")
    current.custody_module.require_bound_ordinary_custody_hold(current.custody_hold,
        primary_owner=current.owner, watchdog_owner=current.watchdog_owner)
    for owner in (current.owner, current.watchdog_owner):
        need(owner.heartbeat(), "actual_callable_owner_heartbeat_failed")
        owner.check_usage(owner.directory)


def ordinary_trusted_binary_resolved(binary):
    """Join the actual callable resolver result to the original pinned binary.

    No configured alias, basename, UUID or source map alone supplies authority.
    The trusted wrapper retains the actual original resolver result and both
    genuine owners before G can allocate private paths or dispatch a provider.
    """
    current = _current()
    need(current is not None and current.invocation is not None
         and type(binary) is str and current.trusted_binary == binary
         and binary in current.source_sha256, "actual_original_callable_binary_required")
    ordinary_callable_context_required(current.invocation)
    _read_pinned(Path(binary), current.source_sha256[binary])


def ordinary_callable_binary_mount_allowed(source):
    """Only the active callable's identical pinned resolved binary may mount.

    Returns False outside the optional context, retaining all legacy/protected
    resolver policy. G limits this necessary join to its existing readonly Grok
    mount at the fixed provider binary destination; no other mount is allowed.
    """
    current = _current()
    if current is None or current.invocation is None:
        return False
    need(type(source) is Path or isinstance(source, Path), "actual_resolved_callable_binary_path_required")
    need(source.is_absolute() and source.resolve(strict=True) == source
         and not any(item.is_symlink() for item in (source, *source.parents)),
         "actual_unaliased_callable_binary_mount_required")
    ordinary_trusted_binary_resolved(str(source))
    return True


def ordinary_provider_guard():
    current = _current()
    need(current is not None and current.invocation is not None,
         "actual_callable_provider_guard_required")
    current.guard_live()


def ordinary_private_allocations_created(isolated_home, prompt_path):
    current = _current()
    if current is None or not current.task_active:
        return
    # Retain the ORIGINAL TemporaryDirectory before any fallible check. Its
    # finalizer must not erase custody paths when caller frames unwind.
    current.private_allocations.append((isolated_home, Path(prompt_path)))
    need(type(isolated_home) is tempfile.TemporaryDirectory,
         "actual_original_private_home_owner_required")
    root = current.invocation.temporary_root
    for path in (Path(isolated_home.name), Path(prompt_path)):
        need(root == path.parent and path.resolve(strict=True) == path
             and not any(item.is_symlink() for item in (path, *path.parents)),
             "actual_precharged_private_allocation_required")


def ordinary_container_creation_started(lease, command):
    current = _current()
    if current is None:
        return
    need(current.task_active and lease is current.selected_lease
         and id(lease) in current.leases and current.gated_create == tuple(command)
         and id(lease) not in current.creation_intents,
         "actual_single_bound_creation_intent_required")
    current.creation_intents.add(id(lease))


def ordinary_container_created(lease, container_id):
    current = _current()
    if current is None:
        return
    need(lease is current.selected_lease and id(lease) in current.creation_intents
         and id(lease) not in current.known_created and type(container_id) is str
         and re.fullmatch("[0-9a-f]{64}", container_id)
         and lease.cidfile.read_text(encoding="ascii").strip() == container_id,
         "actual_created_container_correspondence_required")
    current.known_created[id(lease)] = container_id


def ordinary_allocation_close_permitted(lease):
    current = _current()
    if current is None:
        return True
    if lease is not current.selected_lease or id(lease) not in current.leases:
        return False
    if id(lease) in current.creation_intents and id(lease) not in current.known_created:
        return False
    if current.processes and not (type(current.fence) is dict
            and type(current.fence.get("init_pid")) is int and current.fence["init_pid"] > 1
            and current.fence.get("kernel_scope")):
        return False
    return True


def require_active_ordinary_success(outcome):
    """Authenticate this actual callback scope; never task/acceptance authority.

    Constructing or copying a dataclass cannot satisfy object identity with the
    original source-pinned lifecycle's currently executing callback argument.
    The daemon separately joins original task/attempt/board/commit identities.
    """
    current = _current()
    need(current is not None and current.callback_outcome is outcome
         and type(outcome) is OrdinaryProviderSuccess and current.invocation is not None
         and outcome.callable_invocation is current.invocation
         and outcome.owner is current.owner and outcome.watchdog_owner is current.watchdog_owner
         and outcome.execution_id == current.execution_id and outcome.route is current.route,
         "actual_active_original_success_callback_required")
    check_success_order(current.events)
    need(current.closed and current.returncode == 0 and len(current.processes) == 1
         and outcome.transport_process is current.processes[0]
         and outcome.transport_process.poll() == 0,
         "actual_original_success_transport_required")
    current.custody_module.require_bound_ordinary_custody_hold(current.custody_hold,
        primary_owner=current.owner, watchdog_owner=current.watchdog_owner)
    _verify_sources(current.owner, current.source_sha256, Path(current.runner.__file__).absolute())
    if current.freezer_scope is None:
        current.reap_observed_transport_after_failure()
    else:
        scope = current.freezer_scope
        capture = scope["capture"]
        entry = _AUTHENTICATED_CALLBACKS.get(capture)
        need(entry is not None and entry["current"] is current and entry["outcome"] is outcome
             and current.callback_capture is capture and entry["pid"] == os.getpid()
             and entry["thread"] == threading.get_ident() and entry["used"]
             and not scope["lease"].released, "actual_active_authenticated_freezer_scope_required")
        _original_native_groups_stopped(current)


def _pending_custody(current):
    return _retain_ordinary_pending(current)


def _acknowledge_independently_stopped_custody(current):
    try:
        current.reap_observed_transport_after_failure()
        if not current.runner_entered:
            need(not current.processes and not current.allocated_leases and current.gated_create is None
                 and not current.private_allocations and not current.creation_intents,
                 "actual_known_unstarted_scope_required")
            for owner in (current.owner, current.watchdog_owner):
                public = owner.to_dict()
                need(public["record"].get("child") is None
                     and public["record"].get("prior_children", []) == [],
                     "actual_known_unstarted_native_history_required")
                need(owner.check_usage(owner.directory, full=True)["group_rss"]["live_processes"] == 0,
                     "actual_known_unstarted_group_not_quiescent")
        else:
            # A runner-entry flag or missing lease is not zero-effect proof.
            # Every productive path needs retained allocation correspondence.
            need(bool(current.allocated_leases), "entered_runner_without_stopped_allocation_proof")
            for home, prompt in current.private_allocations:
                need(not os.path.lexists(home.name) and not os.path.lexists(prompt),
                     "actual_original_private_allocation_remains")
        _verify_sources(current.owner, current.source_sha256, Path(current.runner.__file__).absolute())
        current.custody_module.require_bound_ordinary_custody_hold(current.custody_hold,
            primary_owner=current.owner, watchdog_owner=current.watchdog_owner)
        consumers = current.custody_module.held_native_consumers(current.custody_hold)
        for consumer in consumers:
            consumer.release()
        current.custody_module.forget_acknowledged_custody_hold(current.custody_hold)
        _forget_unused_callback(current)
    except OrdinaryCustodyPending:
        raise
    except BaseException as error:
        # Original objects remain strongly held by the binder's registry.
        # No second close, portable receipt, release permission or retry reset.
        raise _pending_custody(current) from error


# Source appended by build_sources.py to the exact frozen H3 baseline.
class AuthenticatedOrdinaryCallback:
    """Opaque actual same-callback identity; never admission or portable proof."""
    __slots__ = ()
    def __new__(cls, *args, **kwargs):
        raise OrdinaryHandoffRefusal('internal_authenticated_callback_constructor_only')
    def __reduce__(self):
        raise OrdinaryHandoffRefusal('authenticated_callback_not_serializable')
    def __copy__(self):
        raise OrdinaryHandoffRefusal('authenticated_callback_not_copyable')
    def __deepcopy__(self, memo):
        raise OrdinaryHandoffRefusal('authenticated_callback_not_copyable')


_AUTHENTICATED_CALLBACKS = {}
_ORDINARY_PENDING = {}
_UNSTARTED_ACKNOWLEDGMENTS = {}


def _actual_process_row(pid):
    try:
        raw = (Path('/proc') / str(pid) / 'stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = raw.rsplit(')', 1)[1].split()
    need(len(fields) >= 22, 'actual_original_process_row_required')
    return dict(pid=pid, state=fields[0], parent=int(fields[1]), group=int(fields[2]), birth=fields[19])


def _capture_native_group(current, owner, process):
    need(type(process) is subprocess.Popen and process.poll() is None, 'actual_live_original_Popen_required')
    row = _actual_process_row(process.pid)
    need(row is not None and row['parent'] == os.getpid() and row['group'] == process.pid,
         'actual_owned_isolated_original_group_required')
    current.original_groups[id(owner)] = (owner, process, row['birth'])


def _original_native_groups_stopped(current):
    for owner in (current.owner, current.watchdog_owner):
        retained = current.original_groups.get(id(owner))
        record = owner.to_dict()['record']
        if retained is None:
            expected = current.processes if owner is current.owner else current.allocated_leases
            need(not expected and record.get('child') is None and record.get('prior_children', []) == [],
                 'original_unstarted_native_group_required')
            continue
        actual_owner, process, birth = retained
        need(actual_owner is owner and type(process) is subprocess.Popen and type(process.poll()) is int,
             'original_actual_process_not_reaped')
        leader = _actual_process_row(process.pid)
        need(leader is None or leader['birth'] == birth, 'original_process_PID_reused')
        entries = list(Path('/proc').iterdir())
        need(len(entries) <= 100000, 'bounded_original_group_census_required')
        for entry in entries:
            if entry.name.isdecimal():
                row = _actual_process_row(int(entry.name))
                need(row is None or row['group'] != process.pid or row['state'] == 'Z',
                     'original_native_group_still_alive')


def _ordinary_admin_capacity(current):
    """Retained phase capacity only; no admin attachment/hard-quota claim."""
    need(current.primary_native.owner_pid == os.getpid() and current.owner_thread == threading.get_ident(),
         'original_same_PID_thread_cleanup_required')
    if current.freezer_scope is not None:
        need(current.freezer_scope['lease'].released, 'Docker_queries_forbidden_while_freezer_owner_held')
    record = current.owner.to_dict()['record']
    existing = importlib.import_module(_NATIVE + 'existing_resource_scheduler')
    scheduler = existing.open_existing_scheduler(Path(record['scheduler_state_path']), record['scheduler_policy_sha256'])
    rows = scheduler.active_leases()
    phase_id = current.primary_native.parent_lease_id
    phase = next((row for row in rows if row['lease_id'] == phase_id), None)
    children = [row for row in rows if row.get('parent_lease_id') == phase_id]
    need(phase is not None and phase.get('owner_pid') == os.getpid()
         and sum(row.get('child_process_slots', 0) for row in children) + 1 <= phase['child_process_slots']
         and sum(row.get('cpu_slots', 0) for row in children) + 1 <= phase['cpu_slots'],
         'one_cooperative_phase_admin_slot_required_before_Docker_query')


def _retain_ordinary_pending(current):
    if current.stopped_pending is None:
        pending = OrdinaryCustodyPending(current.owner, current.watchdog_owner, current.allocated_leases,
            current.processes, current.fence, current.daemon_id, current.runtime,
            custody_hold=current.custody_hold, private_allocations=current.private_allocations)
        pending.actual_lifecycle = current
        current.stopped_pending = pending
        _ORDINARY_PENDING[pending] = current
    return current.stopped_pending


def authenticate_ordinary_callback_for_freezer(outcome):
    """Do every original Docker/source/kernel check BEFORE admitting freezer4."""
    current = _current()
    need(current is not None and current.freezer_scope is None and current.callback_capture is None,
         'fresh_original_callback_authentication_required')
    _ordinary_admin_capacity(current)
    require_active_ordinary_success(outcome)
    need(len(_AUTHENTICATED_CALLBACKS) < 128, 'bounded_actual_callback_registry_required')
    capture = object.__new__(AuthenticatedOrdinaryCallback)
    _AUTHENTICATED_CALLBACKS[capture] = dict(current=current, outcome=outcome, pid=os.getpid(),
                                            thread=threading.get_ident(), used=False)
    current.callback_capture = capture
    return capture


def discard_unused_authenticated_callback(capture):
    entry = _AUTHENTICATED_CALLBACKS.get(capture)
    current = _current()
    need(entry is not None and entry['current'] is current and current.callback_capture is capture
         and not entry['used'] and current.freezer_scope is None
         and current.callback_outcome is entry['outcome'], 'actual_unused_same_callback_capture_required')
    current.callback_capture = None
    del _AUTHENTICATED_CALLBACKS[capture]


@contextmanager
def authenticated_ordinary_freezer_scope(capture, *, freezer_owner):
    """Consume actual callback capture; no Docker queries while freezer held."""
    need(type(capture) is AuthenticatedOrdinaryCallback and capture in _AUTHENTICATED_CALLBACKS,
         'actual_authenticated_callback_capture_required')
    entry = _AUTHENTICATED_CALLBACKS[capture]
    current = _current()
    need(current is entry['current'] and entry['pid'] == os.getpid() and entry['thread'] == threading.get_ident()
         and current.callback_capture is capture and current.callback_outcome is entry['outcome']
         and not entry['used'] and current.freezer_scope is None, 'actual_same_callback_capture_scope_required')
    native = importlib.import_module(_NATIVE + 'parented_phase_resources')
    need(type(freezer_owner) is native.ParentedPhaseResourceReservation
         and freezer_owner not in (current.owner, current.watchdog_owner), 'genuine_distinct_freezer_owner_required')
    lease = freezer_owner.native_lease
    public = freezer_owner.to_dict()
    need(lease.owner_pid == os.getpid() and lease.parent_lease_id == current.primary_native.parent_lease_id
         and lease.cpu_slots == 1 and lease.memory_mb >= 512 and lease.child_process_slots == 4
         and public['record'].get('child') is None and public['record'].get('prior_children', []) == [],
         'fresh_same_phase_one_CPU_four_slot_freezer_required')
    entry['used'] = True
    current.freezer_scope = dict(capture=capture, owner=freezer_owner, lease=lease)
    try:
        require_active_ordinary_success(entry['outcome'])
        yield entry['outcome']
    finally:
        need(lease.released and freezer_owner.to_dict()['status'] == 'released',
             'freezer_public_stopped_release_required_before_callback_exit')
        current.freezer_scope = None
        current.callback_capture = None
        del _AUTHENTICATED_CALLBACKS[capture]


def acknowledge_stopped_ordinary_custody(pending):
    """Original same-process observations only; never close/kill/retry provider."""
    need(type(pending) is OrdinaryCustodyPending and pending in _ORDINARY_PENDING,
         'actual_retained_ordinary_pending_required')
    current = _ORDINARY_PENDING[pending]
    need(current.primary_native.owner_pid == os.getpid() and current.owner_thread == threading.get_ident(),
         'actual_original_pending_PID_thread_required')
    need(getattr(pending, 'actual_lifecycle', None) is current and pending.actual_owner is current.owner
         and pending.actual_watchdog_owner is current.watchdog_owner, 'actual_original_pending_identity_required')
    try:
        if current.freezer_scope is not None:
            scope = current.freezer_scope
            if not (scope['lease'].released and scope['owner'].to_dict()['status'] == 'released'):
                return False  # No Docker query with live freezer4.
            current.freezer_scope = None
            capture = current.callback_capture
            current.callback_capture = None
            _AUTHENTICATED_CALLBACKS.pop(capture, None)
        current.reap_observed_transport_after_failure(stopped=True)
        _original_native_groups_stopped(current)
        if not current.runner_entered:
            need(not current.processes and not current.allocated_leases and current.gated_create is None
                 and not current.private_allocations and not current.creation_intents, 'actual_known_unstarted_scope_required')
        else:
            need(bool(current.allocated_leases), 'entered_runner_without_original_allocation_proof')
        for home, prompt in current.private_allocations:
            need(not os.path.lexists(home.name) and not os.path.lexists(prompt), 'original_private_allocation_remains')
        _verify_sources(current.owner, current.source_sha256, Path(current.runner.__file__).absolute())
        # held_native_consumers authenticates the original registered handle.
        # A prior release may have succeeded for only one original consumer;
        # require_bound would incorrectly demand both still unreleased here.
        for consumer in current.custody_module.held_native_consumers(current.custody_hold):
            if not consumer.released:
                consumer.release()
        current.custody_module.forget_acknowledged_custody_hold(current.custody_hold)
    except Exception:
        return False
    del _ORDINARY_PENDING[pending]
    _forget_unused_callback(current)
    return True


def _forget_unused_callback(current):
    capture = current.callback_capture
    entry = _AUTHENTICATED_CALLBACKS.get(capture)
    if entry is not None and entry['current'] is current and not entry['used'] and current.freezer_scope is None:
        del _AUTHENTICATED_CALLBACKS[capture]
        current.callback_capture = None


def acknowledge_unstarted_custody_hold(hold, *, primary_owner, watchdog_owner):
    """Only genuine prelude failure: no registered actual allocation/history."""
    need(threading.current_thread() is threading.main_thread() and _current() is None
         and not any(row.custody_hold is hold for row in _ORDINARY_PENDING.values()),
         'actual_allocated_or_active_custody_cannot_be_classified_unstarted')
    custody = importlib.import_module('ipfs_accelerate_py.agent_supervisor.runtime.ordinary_custody_hold')
    entry = _UNSTARTED_ACKNOWLEDGMENTS.get(hold)
    if entry is None:
        need(len(_UNSTARTED_ACKNOWLEDGMENTS) < 128, 'bounded_original_unstarted_acknowledgments_required')
        custody.require_bound_ordinary_custody_hold(hold, primary_owner=primary_owner, watchdog_owner=watchdog_owner)
        entry = (primary_owner,watchdog_owner,custody.held_native_consumers(hold))
        _UNSTARTED_ACKNOWLEDGMENTS[hold] = entry
    need(entry[0] is primary_owner and entry[1] is watchdog_owner
         and all(left is right for left,right in zip(entry[2],custody.held_native_consumers(hold))),
         'original_unstarted_hold_owners_and_consumers_required')
    for owner in (primary_owner, watchdog_owner):
        row = owner.to_dict()['record']
        need(row.get('child') is None and row.get('prior_children', []) == [], 'actual_zero_native_child_history_required')
    for consumer in entry[2]:
        if not consumer.released:
            consumer.release()
    custody.forget_acknowledged_custody_hold(hold)
    del _UNSTARTED_ACKNOWLEDGMENTS[hold]
    return True
