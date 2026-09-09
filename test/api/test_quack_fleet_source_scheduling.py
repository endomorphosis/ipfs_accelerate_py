"""One slow native reader must not impose its polling period on other sources."""
from concurrent.futures import Future
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation.fleet_observation import SCHEMA
from ipfs_accelerate_py.agent_supervisor.runtime.quack_fleet_observer import (
    FleetObserver,
    _SourcePolls,
)


class Executor:
    def __init__(self):
        self.calls = []
        self.closed = False

    def submit(self, fn, board):
        future = Future()
        future.set_running_or_notify_cancel()
        self.calls.append((dict(board), future))
        return future

    def shutdown(self, **kwargs):
        assert kwargs == {"wait": False, "cancel_futures": True}
        self.closed = True


def sample(identifier):
    return {"schema": SCHEMA, "source_id": identifier, "observed_at": "2026-09-09T19:00:00+00:00",
            "availability": "unavailable", "source_identity": {}, "native_receipt": {},
            "reason": "native_owner_not_ready_or_alive", "completion_authority": False}


def test_fast_source_repolls_while_other_read_stays_in_flight():
    executor = Executor()
    polls = _SourcePolls(10, max_workers=2, executor=executor)
    boards = [{"id": "slow"}, {"id": "fast"}]
    assert polls.step(boards, 0) == []
    slow, fast = [future for _, future in executor.calls]
    fast.set_result(sample("fast"))
    assert polls.step(boards, 1) == [sample("fast")]
    assert not slow.done()
    assert polls.step(boards, 10) == [] and len(executor.calls) == 2
    assert polls.step(boards, 11) == []
    assert [board["id"] for board, _ in executor.calls] == ["slow", "fast", "fast"]
    executor.calls[-1][1].set_result(sample("fast"))
    assert polls.step(boards, 12) == [sample("fast")]
    assert not slow.done()  # already two fast publications, no batch barrier
    polls.close()
    assert executor.closed


def test_changed_or_removed_binding_discards_inflight_result():
    executor = Executor()
    polls = _SourcePolls(10, max_workers=2, executor=executor)
    polls.step([{"id": "first", "endpoint": "old"}, {"id": "removed"}], 0)
    changed = [{"id": "first", "endpoint": "new"}]
    assert polls.step(changed, 1) == []
    assert len(executor.calls) == 2  # no overlapping read of changed source
    for board, future in executor.calls:
        future.set_result(sample(board["id"]))
    assert polls.step(changed, 2) == []
    assert executor.calls[-1][0]["endpoint"] == "new"
    executor.calls[-1][1].set_result(sample("first"))
    assert polls.step(changed, 3) == [sample("first")]


def test_pool_is_bounded_and_oldest_unsampled_source_gets_next_slot():
    executor = Executor()
    polls = _SourcePolls(10, max_workers=1, executor=executor)
    boards = [{"id": identifier} for identifier in ("first", "second", "third")]
    polls.step(boards, 0)
    assert len(executor.calls) == 1
    executor.calls[0][1].set_result(sample("first"))
    polls.step(boards, 1)
    assert executor.calls[-1][0]["id"] == "second"
    assert len(polls.pending) == 1


@pytest.mark.parametrize("failure", ["exception", "malformed", "foreign"])
def test_one_failed_reader_becomes_unavailable_without_losing_other_results(failure):
    executor = Executor()
    polls = _SourcePolls(10, max_workers=2, executor=executor)
    boards = [{"id": "broken"}, {"id": "healthy"}]
    polls.step(boards, 0)
    broken, healthy = [future for _, future in executor.calls]
    if failure == "exception":
        broken.set_exception(RuntimeError("secret failure text must not escape"))
    else:
        broken.set_result({} if failure == "malformed" else sample("foreign"))
    healthy.set_result(sample("healthy"))
    results = polls.step(boards, 1)
    assert {row["source_id"] for row in results} == {"broken", "healthy"}
    error = next(row for row in results if row["source_id"] == "broken")
    assert error["availability"] == "unavailable" and not error["native_receipt"]
    assert error["reason"].startswith("native_read_failed:") and "secret" not in str(error)


def test_no_finished_reads_do_not_refresh_successful_write_deadline(tmp_path, monkeypatch):
    observer = FleetObserver(None, tmp_path / "inventory.json", tmp_path / "view.json")
    observer.last_progress = 100
    def cycle():
        observer.stop_event.set()
        return None
    monkeypatch.setattr(observer, "cycle", cycle)
    monkeypatch.setattr(observer.stop_event, "wait", lambda _: None)
    observer._run()
    assert observer.last_progress == 100
    assert not observer.output_path.exists()


@pytest.mark.parametrize("workers", [0, 257, True, 1.5])
def test_source_concurrency_requires_bounded_integer(workers):
    with pytest.raises(ValueError, match="source workers"):
        FleetObserver(None, Path("inventory"), Path("view"), source_workers=workers)
