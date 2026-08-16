"""Restart, CAS, lease, and fence tests for SqliteAuthorityState@1 (MCPP-037).

Acceptance (todo MCPP-037 / plan gate 10 / ADR-0004 §2):

* Restart recovers committed state.
* Stale fence tokens are rejected.
* CAS version mismatches fail closed.
* No acknowledged committed write is lost across the declared crash matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.state.sqlite_authority import (
    CAS_INTERRUPTION_POINTS,
    CONSISTENCY_MODE,
    POST_COMMIT_INTERRUPTION_POINTS,
    PRE_COMMIT_INTERRUPTION_POINTS,
    PROVIDER_ID,
    SCHEMA_MARKER,
    CasMismatchError,
    LeaseError,
    SqliteAuthorityState,
    StaleFenceError,
)


class InjectedCrash(RuntimeError):
    """Stand-in for process death at a named durable boundary."""


HOLDER_A = "did:key:z6MkpTHR8VNsBxYAAWHut2Geadd9jSwuBV8xRoAnwWsdvktH"
HOLDER_B = "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"
STATE_ID = "state:test/sqlite-authority"


def _clock(start_ms: int = 1_700_000_000_000) -> Callable[[], int]:
    state = {"now": start_ms}

    def now() -> int:
        return int(state["now"])

    now.advance = lambda ms: state.__setitem__("now", state["now"] + ms)  # type: ignore[attr-defined]
    now.set = lambda ms: state.__setitem__("now", int(ms))  # type: ignore[attr-defined]
    return now


def _open(
    path: Path,
    *,
    clock_ms: Optional[Callable[[], int]] = None,
    crash_injector: Optional[Callable[[str], None]] = None,
) -> SqliteAuthorityState:
    return SqliteAuthorityState.open(path, clock_ms=clock_ms, crash_injector=crash_injector)


def _db(tmp_path: Path, name: str = "authority.sqlite3") -> Path:
    return tmp_path / name


# ---------------------------------------------------------------------------
# Basics: WAL, create, StateRef shape
# ---------------------------------------------------------------------------


def test_opens_with_wal_journal_mode(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as store:
        assert store.journal_mode() == "wal"
        assert store.db_version() == SqliteAuthorityState.DB_VERSION


def test_create_and_get_state_ref_shape(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as store:
        record = store.create(
            STATE_ID,
            {"counter": 0},
            authority={"kind": "principal", "principal": HOLDER_A},
        )
        assert record["version"] == 0
        assert record["value"] == {"counter": 0}
        ref = record["state_ref"]
        assert ref["schema"] == SCHEMA_MARKER
        assert ref["mode"] == CONSISTENCY_MODE
        assert ref["provider"] == PROVIDER_ID
        assert ref["id"] == STATE_ID
        assert ref["version"] == 0
        assert ref["authority"]["kind"] == "principal"


# ---------------------------------------------------------------------------
# Restart recovers committed state
# ---------------------------------------------------------------------------


def test_restart_recovers_committed_state(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as store:
        store.create(STATE_ID, {"counter": 0}, authority={"kind": "principal", "principal": HOLDER_A})
        result = store.cas_write(
            STATE_ID,
            expected_version=0,
            value={"counter": 1},
            operation_id="op-commit-1",
        )
        assert result.status == "updated"
        assert result.version == 1
        assert result.value == {"counter": 1}

    # Process restart: new connection to the same durable file.
    with _open(path) as recovered:
        record = recovered.get(STATE_ID)
        assert record["version"] == 1
        assert record["value"] == {"counter": 1}
        assert record["state_ref"]["mode"] == CONSISTENCY_MODE
        assert recovered.get_ref(STATE_ID)["version"] == 1


def test_restart_recovers_multiple_committed_versions(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as store:
        store.create(STATE_ID, {"n": 0})
        for expected in range(0, 5):
            store.cas_write(
                STATE_ID,
                expected_version=expected,
                value={"n": expected + 1},
                operation_id=f"op-{expected}",
            )

    with _open(path) as recovered:
        record = recovered.get(STATE_ID)
        assert record["version"] == 5
        assert record["value"] == {"n": 5}


# ---------------------------------------------------------------------------
# CAS mismatch
# ---------------------------------------------------------------------------


def test_cas_mismatch_fails_closed(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as store:
        store.create(STATE_ID, {"v": "a"})
        store.cas_write(STATE_ID, expected_version=0, value={"v": "b"})
        with pytest.raises(CasMismatchError) as excinfo:
            store.cas_write(STATE_ID, expected_version=0, value={"v": "stale"})
        err = excinfo.value
        assert err.code == "cas_mismatch"
        assert err.expected_version == 0
        assert err.actual_version == 1
        # Live value unchanged by the failed CAS.
        assert store.get(STATE_ID)["value"] == {"v": "b"}
        assert store.get(STATE_ID)["version"] == 1


def test_cas_mismatch_survives_restart(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as store:
        store.create(STATE_ID, {"v": 1})
        store.cas_write(STATE_ID, expected_version=0, value={"v": 2})
        with pytest.raises(CasMismatchError):
            store.cas_write(STATE_ID, expected_version=0, value={"v": 99})

    with _open(path) as recovered:
        assert recovered.get(STATE_ID)["value"] == {"v": 2}
        with pytest.raises(CasMismatchError) as excinfo:
            recovered.cas_write(STATE_ID, expected_version=0, value={"v": 99})
        assert excinfo.value.actual_version == 1


# ---------------------------------------------------------------------------
# Stale fence rejection
# ---------------------------------------------------------------------------


def test_stale_fence_rejected(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as store:
        store.create(STATE_ID, {"phase": "init"}, fence_token=1)
        # Advance fence (reclaim / takeover).
        advanced = store.issue_fence(STATE_ID, issued_to=HOLDER_B)
        assert advanced["fence_token"] == 2

        with pytest.raises(StaleFenceError) as excinfo:
            store.cas_write(
                STATE_ID,
                expected_version=0,
                value={"phase": "stale-writer"},
                fence_token=1,  # stale relative to accepted token 2
                writer=HOLDER_A,
            )
        err = excinfo.value
        assert err.code == "stale_fence"
        assert err.presented_token == 1
        assert err.accepted_token == 2
        assert store.get(STATE_ID)["value"] == {"phase": "init"}

        # Current fence succeeds.
        ok = store.cas_write(
            STATE_ID,
            expected_version=0,
            value={"phase": "current-writer"},
            fence_token=2,
            writer=HOLDER_B,
        )
        assert ok.version == 1
        assert ok.value == {"phase": "current-writer"}


def test_stale_fence_rejected_after_restart(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as store:
        store.create(STATE_ID, {"x": 0}, fence_token=3)
        store.cas_write(
            STATE_ID,
            expected_version=0,
            value={"x": 1},
            fence_token=3,
        )

    with _open(path) as recovered:
        with pytest.raises(StaleFenceError) as excinfo:
            recovered.cas_write(
                STATE_ID,
                expected_version=1,
                value={"x": 2},
                fence_token=2,
            )
        assert excinfo.value.accepted_token == 3
        assert recovered.get(STATE_ID)["value"] == {"x": 1}


def test_missing_fence_rejected_when_fence_active(tmp_path: Path) -> None:
    path = _db(tmp_path)
    with _open(path) as store:
        store.create(STATE_ID, {"x": 0}, fence_token=1)
        with pytest.raises(StaleFenceError):
            store.cas_write(STATE_ID, expected_version=0, value={"x": 1})


# ---------------------------------------------------------------------------
# Leases
# ---------------------------------------------------------------------------


def test_lease_enforces_exclusive_writer(tmp_path: Path) -> None:
    clock = _clock()
    path = _db(tmp_path)
    with _open(path, clock_ms=clock) as store:
        store.create(STATE_ID, {"work": 0})
        leased = store.acquire_lease(STATE_ID, holder=HOLDER_A, ttl_ms=60_000)
        assert leased["lease"]["holder"] == HOLDER_A
        token = leased["fence_token"]
        assert token >= 1

        with pytest.raises(LeaseError):
            store.cas_write(
                STATE_ID,
                expected_version=0,
                value={"work": 1},
                fence_token=token,
                writer=HOLDER_B,
            )

        ok = store.cas_write(
            STATE_ID,
            expected_version=0,
            value={"work": 1},
            fence_token=token,
            writer=HOLDER_A,
        )
        assert ok.version == 1


def test_lease_reclaim_after_expiry_advances_fence(tmp_path: Path) -> None:
    clock = _clock()
    path = _db(tmp_path)
    with _open(path, clock_ms=clock) as store:
        store.create(STATE_ID, {"work": 0})
        first = store.acquire_lease(STATE_ID, holder=HOLDER_A, ttl_ms=1_000)
        old_token = first["fence_token"]
        clock.advance(2_000)  # type: ignore[attr-defined]

        reclaimed = store.acquire_lease(STATE_ID, holder=HOLDER_B, ttl_ms=60_000)
        assert reclaimed["lease"]["holder"] == HOLDER_B
        assert reclaimed["fence_token"] > old_token
        assert reclaimed["epoch"] > first["epoch"]

        with pytest.raises(StaleFenceError):
            store.cas_write(
                STATE_ID,
                expected_version=0,
                value={"work": "stale"},
                fence_token=old_token,
                writer=HOLDER_A,
            )


# ---------------------------------------------------------------------------
# Crash matrix: no acknowledged committed write is lost
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("boundary", CAS_INTERRUPTION_POINTS)
def test_crash_matrix_preserves_acknowledged_commits(
    tmp_path: Path, boundary: str
) -> None:
    """Declared crash matrix for CAS durability (MCPP-037 acceptance).

    * Pre-commit interruptions: the write is not acknowledged; restart keeps
      the prior committed value (no partial publish).
    * Post-commit interruption (``after_sqlite_commit``): durability has been
      achieved even though the caller did not receive an ack; restart must
      show the new value. A subsequent ack'd write must also survive restart.
    """

    path = _db(tmp_path, f"crash-{boundary}.sqlite3")

    with _open(path) as store:
        store.create(STATE_ID, {"seq": 0})
        # Establish a durable baseline that must never be lost.
        baseline = store.cas_write(
            STATE_ID,
            expected_version=0,
            value={"seq": 1},
            operation_id="baseline",
        )
        assert baseline.version == 1

    def interrupt(point: str) -> None:
        if point == boundary:
            raise InjectedCrash(point)

    with _open(path, crash_injector=interrupt) as store:
        with pytest.raises(InjectedCrash, match=boundary):
            store.cas_write(
                STATE_ID,
                expected_version=1,
                value={"seq": 2},
                operation_id=f"crash-op-{boundary}",
            )

    with _open(path) as recovered:
        record = recovered.get(STATE_ID)
        if boundary in PRE_COMMIT_INTERRUPTION_POINTS:
            # Write was never acknowledged and must not partially apply.
            assert record["version"] == 1
            assert record["value"] == {"seq": 1}
            # Retry after recovery succeeds and is then durable.
            retry = recovered.cas_write(
                STATE_ID,
                expected_version=1,
                value={"seq": 2},
                operation_id=f"crash-op-{boundary}",
            )
            assert retry.status == "updated"
            assert retry.version == 2
        else:
            assert boundary in POST_COMMIT_INTERRUPTION_POINTS
            # Committed before the post-commit crash seam: must be present.
            assert record["version"] == 2
            assert record["value"] == {"seq": 2}
            # Idempotent replay of the same operation id.
            replay = recovered.cas_write(
                STATE_ID,
                expected_version=1,
                value={"seq": 2},
                operation_id=f"crash-op-{boundary}",
            )
            assert replay.status == "unchanged"
            assert recovered.get(STATE_ID)["version"] == 2

    # Final restart: whatever is committed remains committed.
    with _open(path) as final:
        final_record = final.get(STATE_ID)
        assert final_record["value"] == {"seq": 2}
        assert final_record["version"] == 2


def test_acknowledged_write_never_lost_across_clean_restart_sequence(
    tmp_path: Path,
) -> None:
    """Every successful (acknowledged) cas_write is visible after reopen."""

    path = _db(tmp_path)
    acknowledged = []
    with _open(path) as store:
        store.create(STATE_ID, {"seq": 0})
        for i in range(1, 6):
            result = store.cas_write(
                STATE_ID,
                expected_version=i - 1,
                value={"seq": i},
                operation_id=f"ack-{i}",
            )
            acknowledged.append((result.version, result.value))

    with _open(path) as recovered:
        record = recovered.get(STATE_ID)
        last_version, last_value = acknowledged[-1]
        assert record["version"] == last_version
        assert record["value"] == last_value
        # Intermediate versions were superseding commits; the last ack is law.
        for version, value in acknowledged[:-1]:
            assert version < last_version
            assert value != last_value or version == last_version


def test_interruption_points_cover_pre_and_post_commit() -> None:
    assert set(CAS_INTERRUPTION_POINTS) == (
        PRE_COMMIT_INTERRUPTION_POINTS | POST_COMMIT_INTERRUPTION_POINTS
    )
    assert "before_sqlite_commit" in PRE_COMMIT_INTERRUPTION_POINTS
    assert "after_sqlite_commit" in POST_COMMIT_INTERRUPTION_POINTS


# ---------------------------------------------------------------------------
# Combined acceptance scenario
# ---------------------------------------------------------------------------


def test_acceptance_restart_cas_and_stale_fence(tmp_path: Path) -> None:
    """Single narrative covering the MCPP-037 acceptance sentence."""

    path = _db(tmp_path)
    clock = _clock()

    with _open(path, clock_ms=clock) as store:
        store.create(STATE_ID, {"balance": 100}, authority={"kind": "principal", "principal": HOLDER_A})
        leased = store.acquire_lease(STATE_ID, holder=HOLDER_A, ttl_ms=10_000)
        token = leased["fence_token"]
        written = store.cas_write(
            STATE_ID,
            expected_version=0,
            value={"balance": 90},
            fence_token=token,
            writer=HOLDER_A,
            operation_id="debit-10",
        )
        assert written.version == 1

    # Restart recovers the committed debit.
    with _open(path, clock_ms=clock) as store:
        assert store.get(STATE_ID)["value"] == {"balance": 90}

        # CAS mismatch fails closed.
        with pytest.raises(CasMismatchError):
            store.cas_write(
                STATE_ID,
                expected_version=0,
                value={"balance": 0},
                fence_token=token,
                writer=HOLDER_A,
            )

        # Stale fence fails closed after reclaim.
        clock.advance(20_000)  # type: ignore[attr-defined]
        reclaimed = store.acquire_lease(STATE_ID, holder=HOLDER_B, ttl_ms=10_000)
        new_token = reclaimed["fence_token"]
        assert new_token > token
        with pytest.raises(StaleFenceError):
            store.cas_write(
                STATE_ID,
                expected_version=1,
                value={"balance": 80},
                fence_token=token,
                writer=HOLDER_A,
            )
        ok = store.cas_write(
            STATE_ID,
            expected_version=1,
            value={"balance": 80},
            fence_token=new_token,
            writer=HOLDER_B,
            operation_id="debit-10-b",
        )
        assert ok.value == {"balance": 80}

    with _open(path) as final:
        assert final.get(STATE_ID)["value"] == {"balance": 80}
        assert final.get(STATE_ID)["version"] == 2
