"""Exercise actual phase routing with explicit, simulated process observations.

No OS signal, child process or native database is used. The existing phased
recovery suite separately qualifies real pidfd and signal behavior.
"""

from contextlib import contextmanager
from pathlib import Path
import signal

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import (
    native_phased_graceful_recovery as phased,
)


def binding(pid, parent):
    return phased.process.ProcessBinding(
        pid, pid * 100, parent, pid, pid, "private-boot", 1000, str(pid), "/private"
    )


@pytest.fixture
def simulated(monkeypatch):
    controller = binding(100, 1)
    wrappers = [binding(200 + i, controller.pid) for i in range(4)]
    daemon = binding(300, wrappers[3].pid)
    actors = [controller, *wrappers, daemon]
    state = {b.pid: "S" for b in actors}
    pending_term = set()
    events = []
    held = set()

    @contextmanager
    def exact(binding):
        assert binding in actors
        yield binding.pid

    def send(pid, sig):
        events.append(("signal", pid, sig))
        assert state[pid] != "X"
        if sig == signal.SIGSTOP:
            state[pid] = "T"
        elif sig == signal.SIGTERM:
            assert state[pid] == "T"
            pending_term.add(pid)
        elif sig == signal.SIGCONT:
            state[pid] = "X" if pid in pending_term else "S"
        else:
            pytest.fail("unexpected signal")

    class TaskPath:
        def __truediv__(self, _):
            return self

        def iterdir(self):
            return [self]

    monkeypatch.setattr(phased, "Path", lambda *_: TaskPath())
    monkeypatch.setattr(phased.process, "_exact_pidfd", exact)
    monkeypatch.setattr(phased.process, "require_exact_process", lambda b: (
        None if b in actors and state[b.pid] != "X" else pytest.fail("wrong birth")
    ))
    monkeypatch.setattr(phased.process, "_stat", lambda pid: [state[pid]])
    monkeypatch.setattr(phased.process, "_read_proc", lambda _: b"")
    monkeypatch.setattr(phased.process, "all_threads_stopped", lambda b: state[b.pid] == "T")
    monkeypatch.setattr(phased.process, "_exited", lambda pid: state[pid] == "X")
    monkeypatch.setattr(phased.process, "_wait_exit", lambda pid, _: (
        None if state[pid] == "X" else pytest.fail("exit not observed")
    ))
    monkeypatch.setattr(phased.signal, "pidfd_send_signal", send)

    @contextmanager
    def fence(index):
        assert index not in held
        held.add(index)
        events.append(("fence", index, wrappers[index].pid))
        yield
        held.remove(index)

    def children(index):
        assert state[wrappers[index].pid] == "T"
        assert state[daemon.pid] == "X"
        events.append(("children", index))

    return dict(
        controller=controller,
        lanes=[phased.process.LaneBinding(wrappers[3], daemon)],
        idle_supervisors=wrappers[:3],
        lane_indices=[3], idle_lane_indices=[0, 1, 2],
        lane_fence=fence,
        effect_gate=lambda: None,
        population_gate=lambda: None,
        population_refusal=RuntimeError,
        lane_children_gate=children,
        closed_children_gate=lambda: None,
        record_phase=lambda _: None,
    ), events, state, held


def test_lane_three_active_retains_original_fences_and_child_gates(simulated):
    options, events, state, held = simulated
    result = phased.gracefully_close_native_lanes(**options)
    assert [e[1] for e in events if e[0] == "fence"] == [3, 0, 1, 2]
    assert [e[1] for e in events if e[0] == "children"] == [3, 0, 1, 2]
    # Each wrapper STOP is immediately preceded by its actual configured fence.
    filtered = [e for e in events if e[0] == "fence" or
                (e[0] == "signal" and e[2] == signal.SIGSTOP and 200 <= e[1] < 204)]
    assert filtered == [event for i in (3, 0, 1, 2) for event in (
        ("fence", i, 200 + i), ("signal", 200 + i, signal.SIGSTOP)
    )]
    assert result["closed_lanes"] == [0, 1, 2, 3]
    assert result["controller_exited"] is True
    assert result["completion_authority"] is False
    assert result["callback_settlement_authority"] is False
    assert all(value == "X" for value in state.values())
    assert not held


@pytest.mark.parametrize(("active", "idle"), [
    ([3], None), (None, [0, 1, 2]), ([0], [0, 1, 2]),
    ([4], [0, 1, 2]), ([-1], [0, 1, 2]), ([True], [0, 2, 3]),
    ([3.0], [0, 1, 2]), (["3"], [0, 1, 2]), ([], [0, 1, 2]),
    ([3, 2], [0, 1]), ([3], [0, 1]), ("3", [0, 1, 2]),
])
def test_invalid_mapping_refuses_before_any_fence_or_signal(simulated, active, idle):
    options, events, state, held = simulated
    options.update(lane_indices=active, idle_lane_indices=idle)
    with pytest.raises(phased.process.GracefulRecoveryUnverified, match="lane_ind"):
        phased.gracefully_close_native_lanes(**options)
    assert events == []
    assert set(state.values()) == {"S"}
    assert not held


def test_source_under_test_is_this_candidate():
    expected = Path(__file__).resolve().parents[2]
    assert Path(phased.__file__).resolve().is_relative_to(expected)
