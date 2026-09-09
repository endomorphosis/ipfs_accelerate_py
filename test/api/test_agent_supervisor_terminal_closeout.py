from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.terminal_closeout import (
    retain_owner_for_closeout,
)


@pytest.mark.parametrize("observation", [
    {"task_count": 51, "completed_count": 51},
    {"complete": True, "completion_authority": False},
    {"complete": True, "completion_authority": 1},
    {"complete": True},
    {"completion_authority": True, "complete": False},
    {"completion_authority": True, "complete": 1},
    {"completion_authority": False, "authenticated_task_observation": True,
     "task_count": 51, "goal_count": 32},
])
def test_task_drain_and_progress_do_not_retire_native_owner(observation):
    waits = []
    result = retain_owner_for_closeout(
        observe=lambda: observation,
        wait=lambda seconds: waits.append(seconds) or True,
        stopped=lambda: False, check_owner=lambda: None, output=lambda _: None,
    )
    assert result == "stopped"
    assert waits == [10.0]


def test_native_acceptance_is_rechecked_after_pending_observation():
    observations = iter([{"completion_authority": False},
                         {"completion_authority": True, "complete": True}])
    checks = []
    result = retain_owner_for_closeout(
        observe=lambda: next(observations), wait=lambda _: False,
        stopped=lambda: False, check_owner=lambda: checks.append("live"),
        output=lambda _: None,
    )
    assert result == "accepted"
    assert checks == ["live", "live", "live"]


def test_hold_prevents_further_authority_reads():
    result = retain_owner_for_closeout(
        observe=lambda: pytest.fail("read after hold"),
        wait=lambda _: pytest.fail("wait after hold"),
        stopped=lambda: True, check_owner=lambda: pytest.fail("check after hold"),
        output=lambda _: None,
    )
    assert result == "stopped"


def test_owner_failure_reaches_existing_restart_path():
    def failed():
        raise RuntimeError("owner fence lost")
    with pytest.raises(RuntimeError, match="owner fence lost"):
        retain_owner_for_closeout(
            observe=lambda: pytest.fail("read after owner failure"),
            wait=lambda _: False, stopped=lambda: False,
            check_owner=failed, output=lambda _: None,
        )


def test_stop_during_acceptance_read_does_not_report_completion():
    stopping = []
    def observe():
        stopping.append(True)
        return {"completion_authority": True, "complete": True}
    assert retain_owner_for_closeout(
        observe=observe, wait=lambda _: False, stopped=lambda: bool(stopping),
        check_owner=lambda: None, output=lambda _: None,
    ) == "stopped"
