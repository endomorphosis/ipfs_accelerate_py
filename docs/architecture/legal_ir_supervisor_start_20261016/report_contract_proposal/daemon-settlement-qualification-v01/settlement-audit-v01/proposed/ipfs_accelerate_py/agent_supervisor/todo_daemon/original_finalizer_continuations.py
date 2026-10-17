"""Same-process original finally continuations; never cleanup authority by ID.

The host must prove original physical custody stopped before calling drain.
Running a finally body does not settle lifecycle/candidate/task disposition.
"""
import hashlib
import inspect
import os
from pathlib import Path
import stat
import sys
import threading
from types import CodeType, FrameType, FunctionType, ModuleType


MODULE = "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
STAGES = (
    ("implementation_claim_release", "_run_implementation", "_settle_report_only_implementation_claims"),
    ("pass_dispatch_intent_reconciliation", "_run_once", "_settle_report_only_dispatch_intents"),
)
MAX_SOURCE_BYTES = 8 * 1024 * 1024


class OriginalFinalizerRefusal(RuntimeError):
    pass


class OriginalFinalizerPending(BaseException):
    def __init__(self, actual_registry, actual_cause):
        super().__init__("actual_original_finalizer_settlement_pending")
        self.actual_registry = actual_registry
        self.actual_cause = actual_cause

    def __reduce__(self):
        raise OriginalFinalizerRefusal("actual_original_finalizer_pending_not_serializable")


def require(condition, reason):
    if not condition:
        raise OriginalFinalizerRefusal(reason)


def _owned_source(path):
    require(path.is_absolute() and str(path) == os.path.abspath(path)
            and not any(item.is_symlink() for item in (path, *path.parents)),
            "unaliased_actual_daemon_source_required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                and before.st_uid in {0, os.geteuid()} and not before.st_mode & 0o002
                and 0 < before.st_size <= MAX_SOURCE_BYTES,
                "bounded_owned_actual_daemon_source_required")
        raw = stream.read(MAX_SOURCE_BYTES + 1)
        after = os.fstat(stream.fileno())
    witness = lambda row: (row.st_dev, row.st_ino, row.st_mode, row.st_uid,
                           row.st_gid, row.st_nlink, row.st_size,
                           row.st_mtime_ns, row.st_ctime_ns)
    require(len(raw) == before.st_size and witness(before) == witness(after)
            == witness(path.lstat()), "actual_daemon_source_changed_during_observation")
    return hashlib.sha256(raw).hexdigest()


class OriginalFinalizerContinuations:
    """Retain two actual source-defined callbacks, with an at-most-once drain."""

    def __init__(self, daemon_module, daemon):
        require(type(daemon_module) is ModuleType and daemon_module.__name__ == MODULE
                and sys.modules.get(MODULE) is daemon_module,
                "actual_loaded_original_daemon_module_required")
        cls = daemon_module.__dict__.get("PortalImplementationDaemon")
        require(type(cls) is type and type(daemon) is cls
                and cls.__module__ == MODULE,
                "actual_exact_original_daemon_class_required")
        self.module, self.daemon = daemon_module, daemon
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.path = Path(daemon_module.__file__).absolute()
        self.source_sha256 = _owned_source(self.path)
        self.specifications = {}
        for stage, method_name, nested_name in STAGES:
            method = cls.__dict__.get(method_name)
            seen = set()
            for _ in range(9):
                require(type(method) is FunctionType and id(method) not in seen,
                        "bounded_actual_original_method_unwrap_required")
                seen.add(id(method))
                wrapped = method.__dict__.get("__wrapped__")
                if wrapped is None:
                    break
                method = wrapped
            else:
                raise OriginalFinalizerRefusal("actual_original_method_unwrap_bound")
            require(method.__globals__ is daemon_module.__dict__
                    and method.__code__.co_name == method_name
                    and Path(method.__code__.co_filename).absolute() == self.path,
                    "actual_source_original_method_body_required")
            codes = [value for value in method.__code__.co_consts
                     if type(value) is CodeType and value.co_name == nested_name]
            require(len(codes) == 1
                    and Path(codes[0].co_filename).absolute() == self.path
                    and codes[0].co_argcount == 0 and codes[0].co_posonlyargcount == 0
                    and codes[0].co_kwonlyargcount == 0
                    and not codes[0].co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS),
                    "exact_original_finally_nested_code_required")
            self.specifications[stage] = (method.__code__, codes[0])
        self.entries = {}
        self.rejected = None
        self.pending = None
        self.highest_registered = -1
        self.draining = False

    def __reduce__(self):
        raise OriginalFinalizerRefusal("actual_finalizer_registry_not_serializable")

    def __copy__(self):
        raise OriginalFinalizerRefusal("actual_finalizer_registry_not_copyable")

    def __deepcopy__(self, memo):
        raise OriginalFinalizerRefusal("actual_finalizer_registry_not_copyable")

    def _same_process(self):
        require(self.pid == os.getpid() and self.thread == threading.get_ident(),
                "original_finalizer_PID_thread_required")

    def verify_source(self):
        self._same_process()
        require(sys.modules.get(MODULE) is self.module
                and self.module.__dict__.get("PortalImplementationDaemon") is type(self.daemon)
                and Path(self.module.__file__).absolute() == self.path
                and _owned_source(self.path) == self.source_sha256,
                "unchanged_original_daemon_module_source_required")

    def _failed(self, cause):
        if self.pending is None:
            self.pending = OriginalFinalizerPending(self, cause)
        raise self.pending

    def register(self, stage, continuation, caller_frame, original_frame):
        # Capture even malformed input before shape/source/connection observations.
        # At most two accepted carriers plus the first rejected carrier are retained.
        carrier = (stage, continuation, caller_frame, original_frame)
        try:
            self._same_process()
            require(self.rejected is None and self.pending is None and not self.draining,
                    "original_finalizer_registry_already_failed_or_draining")
            require(type(stage) is str and stage in self.specifications
                    and type(original_frame) is dict,
                    "original_finalizer_stage_and_reference_carrier_required")
            require(stage not in self.entries, "original_finalizer_stage_already_registered")
            index = tuple(row[0] for row in STAGES).index(stage)
            require(index > self.highest_registered, "original_finalizer_registration_order_changed")
            enclosing_code, nested_code = self.specifications[stage]
            current = inspect.currentframe()
            try:
                # The sole intervening frame is the actual host's pinned wrapper.
                host_frame = current.f_back
                require(type(caller_frame) is FrameType and host_frame is not None
                        and host_frame.f_back is caller_frame
                        and caller_frame.f_code is enclosing_code
                        and caller_frame.f_globals is self.module.__dict__
                        and caller_frame.f_locals.get("self") is self.daemon,
                        "actual_immediate_original_method_caller_required")
            finally:
                del current
            require(type(continuation) is FunctionType
                    and continuation.__code__ is nested_code
                    and continuation.__globals__ is self.module.__dict__
                    and continuation.__defaults__ is None
                    and continuation.__kwdefaults__ is None,
                    "exact_actual_original_finally_function_required")
            cells = continuation.__closure__ or ()
            names = continuation.__code__.co_freevars
            require(len(cells) == len(names), "exact_original_finally_cells_required")
            for name, cell in zip(names, cells):
                require(name in caller_frame.f_locals
                        and cell.cell_contents is caller_frame.f_locals[name],
                        "original_finally_cell_identity_changed")
            require("self" in names and cells[names.index("self")].cell_contents is self.daemon,
                    "actual_daemon_finally_cell_required")
            # From this point a fallible source/DB/productive-phase check may not
            # lose this original closure. Source authentication follows retention.
            self.entries[stage] = {"carrier": carrier, "continuation": continuation,
                                   "status": "registered", "cause": None}
            self.highest_registered = index
            self.verify_source()
            return self.diagnostics()
        except BaseException as cause:
            if self.rejected is None:
                self.rejected = (carrier, cause)
            raise

    def drain(self):
        """Host calls only after independent stopped proof; no observation grant."""
        try:
            self._same_process()
            require(self.rejected is None and not self.draining,
                    "original_finalizer_registry_rejected_or_reentrant")
            if self.pending is not None:
                raise self.pending
            self.verify_source()
        except BaseException as cause:
            self._failed(cause)
        self.draining = True
        try:
            for stage, _method, _nested in STAGES:
                entry = self.entries.get(stage)
                if entry is None or entry["status"] == "done":
                    continue
                require(entry["status"] == "registered", "original_finalizer_may_not_be_replayed")
                entry["status"] = "running"
                try:
                    entry["continuation"]()
                except BaseException as cause:
                    entry["status"], entry["cause"] = "failed", cause
                    self._failed(cause)
                entry["status"] = "done"
            return self.diagnostics()
        finally:
            self.draining = False

    def diagnostics(self):
        return {
            "registered": len(self.entries),
            "done": sum(row["status"] == "done" for row in self.entries.values()),
            "failed": sum(row["status"] == "failed" for row in self.entries.values()),
            "rejected": self.rejected is not None,
            "stages": {stage: self.entries[stage]["status"] for stage, _, _ in STAGES
                       if stage in self.entries},
            "physical_custody_authority": False,
            "disposition_settled": False,
            "acceptance_authority": False,
        }

    def complete(self):
        """Diagnostic only; never physical custody or disposition authority."""
        return (self.rejected is None and self.pending is None and not self.draining
                and all(row["status"] == "done" for row in self.entries.values()))
