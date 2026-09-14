"""Pause an exact roster before graceful closure; never signal unknown helpers.

This separate mechanism preserves the earlier sequential implementation. All
known actors are suspended before waiting for strict population quiescence.
A helper requiring parent progress therefore refuses and resumes its parents.
Population evidence is not callback, owner replacement or source authority.
"""

from __future__ import annotations

import contextlib
import signal
import time
from pathlib import Path

from . import native_graceful_recovery as process


def gracefully_close_native_lanes(
    *,
    controller,
    lanes,
    lane_fence,
    effect_gate,
    population_gate,
    population_refusal,
    lane_children_gate,
    closed_children_gate,
    record_phase,
    idle_supervisors=(),
    lane_indices=None,
    idle_lane_indices=None,
    timeout_seconds=30,
):
    """Close bound active lanes followed by explicitly observed idle wrappers.

    Explicit indices preserve configured lane identity when active and idle
    wrappers are separated. Both index sequences must be supplied together
    and form an exact permutation of the full configured lane population.
    Without indices, callbacks retain their historical active-first ordering.
    Idle wrappers must have no live child; omitting a daemon is not closure
    evidence. Population and source gates still apply to the complete roster.
    """
    process._require(
        type(timeout_seconds) in {int, float} and 0 < timeout_seconds <= 300,
        "recovery_timeout_bound",
    )
    lanes = tuple(lanes)
    idle_supervisors = tuple(idle_supervisors)
    supervisors = [lane.supervisor for lane in lanes] + list(idle_supervisors)
    process._require(1 <= len(supervisors) <= 32, "recovery_lane_bound")
    if lane_indices is None and idle_lane_indices is None:
        configured_indices = tuple(range(len(supervisors)))
    else:
        process._require(
            type(lane_indices) in {tuple, list}
            and type(idle_lane_indices) in {tuple, list}
            and len(lane_indices) == len(lanes)
            and len(idle_lane_indices) == len(idle_supervisors),
            "recovery_lane_index_population_changed",
        )
        configured_indices = (*lane_indices, *idle_lane_indices)
        process._require(
            all(type(index) is int for index in configured_indices)
            and sorted(configured_indices) == list(range(len(supervisors))),
            "recovery_lane_indices_not_exact_permutation",
        )
    bindings = [
        controller,
        *supervisors,
        *[lane.daemon for lane in lanes],
    ]
    process._require(
        len({b.pid for b in bindings}) == len(bindings),
        "recovery_actor_population_changed",
    )
    process._require(
        all(
            lane.supervisor.parent == controller.pid
            and lane.daemon.parent == lane.supervisor.pid
            for lane in lanes
        ) and all(wrapper.parent == controller.pid for wrapper in idle_supervisors),
        "lane_parent_binding_changed",
    )
    suspended = []
    controller_terminated = False

    def idle_children_closed():
        for wrapper in idle_supervisors:
            if process._exited(descriptors[wrapper.pid]):
                continue
            process.require_exact_process(wrapper)
            task = Path("/proc") / str(wrapper.pid) / "task"
            tids = sorted(task.iterdir())
            process._require(0 < len(tids) <= 512, "idle_wrapper_thread_bound")
            for tid in tids:
                children = process._read_proc(tid / "children").decode("ascii").split()
                for child in children:
                    try:
                        state = process._stat(int(child))[0]
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    process._require(state in {"Z", "X"}, "idle_supervisor_has_live_child")
            process._require(tids == sorted(task.iterdir()), "idle_wrapper_threads_changed")
            process.require_exact_process(wrapper)

    with contextlib.ExitStack() as actors:
        descriptors = {
            b.pid: actors.enter_context(process._exact_pidfd(b)) for b in bindings
        }
        fences = contextlib.ExitStack()

        def still_suspended():
            for binding in suspended:
                process.require_exact_process(binding)
                process._require(
                    process.all_threads_stopped(binding), "actor_suspension_lost"
                )

        def pause(binding, role):
            effect_gate()
            process.require_exact_process(binding)
            process._require(
                process._stat(binding.pid)[0] not in {"T", "t"}, "actor_already_stopped"
            )
            record_phase(role + "_suspend_prepared")
            effect_gate()
            process.require_exact_process(binding)
            suspended.append(binding)
            signal.pidfd_send_signal(descriptors[binding.pid], signal.SIGSTOP)
            deadline = time.monotonic() + timeout_seconds
            while not process.all_threads_stopped(binding):
                process._require(
                    time.monotonic() < deadline, "actor_suspend_unverified"
                )
                time.sleep(0.01)
            record_phase(role + "_all_threads_stopped")

        def resume(binding):
            signal.pidfd_send_signal(descriptors[binding.pid], signal.SIGCONT)
            suspended.remove(binding)

        def wait_quiet():
            deadline = time.monotonic() + timeout_seconds
            while True:
                effect_gate()
                still_suspended()
                idle_children_closed()
                try:
                    population_gate()
                    return
                except population_refusal:
                    if time.monotonic() >= deadline:
                        raise
                time.sleep(0.02)

        try:
            pause(controller, "controller")
            # No wrapper may launch another daemon or maintenance helper while
            # another lane closes. Keep the existing native launch fences.
            for index, supervisor in zip(configured_indices, supervisors):
                fences.enter_context(lane_fence(index))
                pause(supervisor, "supervisor")
            for lane in lanes:
                pause(lane.daemon, "daemon")
            wait_quiet()
            record_phase("all_actors_paused_and_population_quiet")

            # Daemon handlers raise SystemExit and close native handles. Keep
            # every wrapper/master paused and every launch fence until all
            # daemons and independently observed helper remnants are gone.
            for lane in lanes:
                effect_gate()
                still_suspended()
                record_phase("daemon_graceful_exit_prepared")
                effect_gate()
                still_suspended()
                signal.pidfd_send_signal(descriptors[lane.daemon.pid], signal.SIGTERM)
                resume(lane.daemon)
            for lane in lanes:
                process._wait_exit(
                    descriptors[lane.daemon.pid], time.monotonic() + timeout_seconds
                )
                record_phase("daemon_exit_observed")
            wait_quiet()
            for index in configured_indices:
                lane_children_gate(index)
            record_phase("all_daemons_and_helpers_closed")

            # Queue every native wrapper stop flag while all threads remain
            # stopped, then release fences BEFORE any wrapper cleanup runs.
            for supervisor in supervisors:
                record_phase("supervisor_graceful_exit_prepared")
                effect_gate()
                still_suspended()
                population_gate()
                signal.pidfd_send_signal(
                    descriptors[supervisor.pid], signal.SIGTERM
                )
            fences.close()
            for supervisor in supervisors:
                resume(supervisor)
            for supervisor in supervisors:
                process._wait_exit(
                    descriptors[supervisor.pid], time.monotonic() + timeout_seconds
                )
                record_phase("lane_exit_observed")
            wait_quiet()
            closed_children_gate()
            effect_gate()
            still_suspended()
            population_gate()
            record_phase("controller_graceful_exit_prepared")
            effect_gate()
            still_suspended()
            population_gate()
            signal.pidfd_send_signal(descriptors[controller.pid], signal.SIGTERM)
            controller_terminated = True
        finally:
            # Release locks first, including on a partial failure, so a resumed
            # native wrapper can perform its own existing reconciliation.
            try:
                fences.close()
            finally:
                errors = []
                for binding in reversed(suspended):
                    try:
                        signal.pidfd_send_signal(
                            descriptors[binding.pid], signal.SIGCONT
                        )
                    except ProcessLookupError:
                        pass
                    except OSError as exc:
                        errors.append(exc)
                if errors:
                    raise process.GracefulRecoveryUnverified(
                        "actor_resume_unverified"
                    ) from errors[0]
        process._require(controller_terminated, "controller_exit_not_requested")
        process._wait_exit(
            descriptors[controller.pid], time.monotonic() + timeout_seconds
        )
        record_phase("controller_exit_observed")
    return {
        "controller_exited": True,
        "closed_lanes": sorted(configured_indices),
        "callback_settlement_authority": False,
        "completion_authority": False,
    }
