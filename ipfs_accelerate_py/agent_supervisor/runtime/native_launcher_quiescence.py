"""Original-pidfd launcher custody before a single daemon nomination.

This mechanism has no task, source or child-adoption authority. Its caller owns
those gates. It never signals a process observed as somebody else's child.

Use this Linux primitive from a separate admitted custodian, never from a
master or wrapper that it will suspend. Retain the instance before calling
``__enter__``: an entry or cleanup exception can leave original suspensions
that only this same instance's open pidfds may restore through ``close()``.
Each lane-fence context must yield a callable that positively verifies its
still-held original lock. Hold the caller's independent native guards until
cleanup succeeds; ``normal_exit_required`` reports uncertain descriptor
cleanup, not proof that custody was released. No runtime handler or automatic
recovery request is installed by importing this module.
"""
from __future__ import annotations

import contextlib
import os
import signal
import time

from . import native_process_custody as process


class LauncherQuiescence:
    def __init__(self, *, controller, supervisors, lane_fence, effect_gate,
                 record_phase, timeout_seconds=30):
        self.controller = controller
        self.supervisors = tuple(supervisors)
        self.bindings = (controller, *self.supervisors)
        process._require(0 < len(self.supervisors) <= 32, "launcher_slot_bound")
        process._require(len({b.pid for b in self.bindings}) == len(self.bindings),
                         "launcher_pid_collision")
        process._require(all(b.parent == controller.pid for b in self.supervisors),
                         "launcher_parent_changed")
        process._require(0 < timeout_seconds <= 300, "launcher_timeout_bound")
        self.owner = process.observe_process(os.getpid())
        process._require(self.owner.pid not in {b.pid for b in self.bindings},
                         "launcher_custodian_cannot_freeze_itself")
        self.lane_fence = lane_fence
        self.effect_gate = effect_gate
        self.record_phase = record_phase
        self.timeout_seconds = timeout_seconds
        self.handles = contextlib.ExitStack()
        self.fences = contextlib.ExitStack()
        self.fence_checks = []
        self.fences_released = False
        self.descriptors = {}
        self.descriptor_identities = {}
        self.suspended = []
        self.state = "new"
        self.closed = False
        self.normal_exit_required = False
        self.daemon_bindings = None

    def _retain_pidfd(self, binding):
        process._require(binding.pid not in self.descriptors, "launcher_pidfd_population_collision")
        self.descriptors[binding.pid] = self.handles.enter_context(process._exact_pidfd(binding))
        current = os.fstat(self.descriptors[binding.pid])
        self.descriptor_identities[binding.pid] = (current.st_dev, current.st_ino, current.st_mode)

    def require_owner(self):
        process._require(os.getpid() == self.owner.pid, "launcher_custody_foreign_process")
        process.require_exact_process(self.owner)
        process._require(not self.closed, "launcher_custody_closed")
        for pid, expected in self.descriptor_identities.items():
            current = os.fstat(self.descriptors[pid])
            process._require((current.st_dev, current.st_ino, current.st_mode) == expected,
                             "launcher_original_pidfd_replaced")
            from pathlib import Path
            fields = [row.split()[1:] for row in process._read_proc(
                Path('/proc/self/fdinfo') / str(self.descriptors[pid])).decode('ascii').splitlines()
                if row.startswith('Pid:')]
            process._require(fields == [[str(pid)]] or
                             (fields == [['-1']] and process._exited(self.descriptors[pid])),
                             "launcher_original_pidfd_target_changed")

    def require_current(self):
        self.require_owner()
        self.require_fences()
        for binding in self.bindings:
            process.require_exact_process(binding)
            process._require(not process._exited(self.descriptors[binding.pid]),
                             "launcher_original_pidfd_exited")
        for binding in self.suspended:
            process._require(process.all_threads_stopped(binding), "launcher_suspension_lost")

    def _pause(self, binding, role):
        self.effect_gate()
        self.require_current()
        process._require(process._stat(binding.pid)[0] not in {"T", "t"},
                         "launcher_already_stopped")
        self.record_phase(role + "_suspend_prepared")
        self.effect_gate()
        self.require_current()
        self.suspended.append(binding)
        signal.pidfd_send_signal(self.descriptors[binding.pid], signal.SIGSTOP)
        deadline = time.monotonic() + self.timeout_seconds
        while not process.all_threads_stopped(binding):
            process._require(time.monotonic() < deadline, "launcher_suspend_timeout")
            time.sleep(.01)
        self.record_phase(role + "_all_threads_stopped")

    def __enter__(self):
        self.require_owner()
        process._require(self.state == "new", "launcher_custody_reentered")
        self.state = "acquiring"
        try:
            for binding in self.bindings:
                self._retain_pidfd(binding)
            self.require_current()
            self._pause(self.controller, "controller")
            # A held launch fence can require another wrapper to run. Acquire
            # every fence before stopping any wrapper, exactly as in closure.
            for index in range(len(self.supervisors)):
                self.effect_gate()
                self.require_current()
                check = self.fences.enter_context(self.lane_fence(index))
                process._require(callable(check), "launcher_fence_check_unavailable")
                self.fence_checks.append(check)
                self.effect_gate()
                self.require_current()
            self.record_phase("all_native_launch_fences_acquired_before_nomination")
            for binding in self.supervisors:
                self._pause(binding, "supervisor")
            self.state = "frozen"
            self.require_frozen()
            self.record_phase("original_launchers_frozen_before_daemon_nomination")
            return self
        except BaseException:
            self.close()
            raise

    def require_frozen(self):
        process._require(self.state == "frozen", "launcher_nomination_not_admitted")
        process._require(not self.fences_released and len(self.fence_checks) == len(self.supervisors),
                         "launcher_fences_not_retained_for_nomination")
        self.require_current()
        process._require(self.suspended == list(self.bindings), "launcher_frozen_population_changed")

    def nominate_daemons(self, daemons):
        self.require_frozen()
        process._require(self.daemon_bindings is None, "frozen_daemon_nomination_replayed")
        daemons = tuple(daemons)
        process._require(self.owner.pid not in {b.pid for b in daemons},
                         "launcher_custodian_cannot_nominate_itself")
        process._require(len(daemons) <= len(self.supervisors)
                         and len({b.parent for b in daemons}) == len(daemons)
                         and all(b.parent in {s.pid for s in self.supervisors} for b in daemons),
                         "frozen_daemon_slot_parent_changed")
        self.daemon_bindings = daemons
        for binding in daemons:
            self._retain_pidfd(binding)
        self.require_frozen()

    def consume_for_closure(self, controller, supervisors, daemons):
        self.require_frozen()
        process._require(controller == self.controller
                         and len(supervisors) == len(self.supervisors)
                         and set(supervisors) == set(self.supervisors),
                         "launcher_closure_binding_changed")
        process._require(self.daemon_bindings is not None
                         and tuple(daemons) == self.daemon_bindings,
                         "launcher_final_daemon_nomination_changed")
        for binding in self.daemon_bindings:
            process.require_exact_process(binding)
            process._require(not process._exited(self.descriptors[binding.pid]),
                             "launcher_nominated_daemon_exited")
        self.state = "closure"
        return self

    def require_fences(self):
        if not self.fences_released:
            for check in self.fence_checks:
                check()

    def release_fences(self):
        self.require_owner()
        if self.fences_released:
            return
        self.fences_released = True
        try:
            self.fences.close()
        except BaseException:
            self.normal_exit_required = True
            raise
        finally:
            # ExitStack consumes all callbacks even when cleanup raises. Its
            # closed descriptors cannot gate a same-controller restoration.
            self.fence_checks.clear()

    def restore_suspensions(self):
        """Continue only the original open pidfds; never re-observe numeric PIDs."""
        self.require_owner()
        errors = []
        for binding in reversed(self.suspended[:]):
            try:
                signal.pidfd_send_signal(self.descriptors[binding.pid], signal.SIGCONT)
            except ProcessLookupError:
                self.suspended.remove(binding)
            except OSError as error:
                errors.append(error)
            else:
                self.suspended.remove(binding)
        if errors:
            raise process.GracefulRecoveryUnverified("launcher_original_resume_failed") from errors[0]

    def close(self):
        if self.closed:
            return
        self.require_owner()
        try:
            try:
                self.release_fences()
            finally:
                self.restore_suspensions()
        finally:
            # Never close original pidfds while any original suspension still
            # needs restoration. The original controller retains that custody.
            if not self.suspended:
                self.closed = True
                self.state = "closed"
                self.handles.close()

    def __exit__(self, *exception):
        self.close()
