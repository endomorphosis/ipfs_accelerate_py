from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.autonomy.recovery_leases import (
    DEFAULT_DAEMON_RECOVERY_LEASE_TTL_SECONDS,
    RecoveryLeaseTable,
    bundle_recovery_lease_ttl_seconds,
    recovery_lease_ttl_from_env,
    recovery_leases_from_env,
)


def test_second_owner_is_blocked_until_ttl_expires(tmp_path) -> None:
    table = RecoveryLeaseTable(tmp_path / "leases.json")
    now = 1_000.0
    ok, first = table.try_acquire("task", "task-a", "lane-0", now=now, ttl_seconds=10)
    assert ok
    assert first["reason"] == "recovery_lease_acquired"
    denied, view = table.try_acquire("task", "task-a", "lane-1", now=now + 1, ttl_seconds=10)
    assert denied is False
    assert view["reason"] == "recovery_lease_held"
    stolen, steal = table.try_acquire("task", "task-a", "lane-1", now=now + 11, ttl_seconds=10)
    assert stolen
    assert steal["reason"] == "recovery_lease_stolen"
    assert steal["lease"]["owner_id"] == "lane-1"


def test_same_owner_renews_and_release_frees(tmp_path) -> None:
    table = RecoveryLeaseTable(tmp_path / "leases.json")
    ok, _ = table.try_acquire("lane", "lane-0", "owner-a", now=5.0, ttl_seconds=10)
    assert ok
    ok, renewed = table.try_acquire("lane", "lane-0", "owner-a", now=8.0, ttl_seconds=10)
    assert ok
    assert renewed["reason"] == "recovery_lease_renewed"
    assert renewed["lease"]["expires_at"] == 18.0
    released = table.release("lane", "lane-0", "owner-a")
    assert released["released"] is True
    ok, _ = table.try_acquire("lane", "lane-0", "owner-b", now=9.0, ttl_seconds=10)
    assert ok


def test_unreadable_lease_file_is_not_stolen(tmp_path) -> None:
    path = tmp_path / "leases.json"
    path.write_text("{not json", encoding="utf-8")
    table = RecoveryLeaseTable(path)
    ok, view = table.try_acquire("checkout", "main", "owner-a", now=1.0)
    assert ok is False
    assert view["reason"] == "recovery_lease_unreadable"


def test_expired_other_keys_are_swept_on_acquire(tmp_path) -> None:
    table = RecoveryLeaseTable(tmp_path / "leases.json")
    table.try_acquire("task", "old", "lane-0", now=1.0, ttl_seconds=1)
    table.try_acquire("task", "live", "lane-0", now=10.0, ttl_seconds=10)
    assert table.inspect("task", "old") is None
    live = table.inspect("task", "live")
    assert live is not None
    assert live["owner_id"] == "lane-0"


def test_env_path_shares_file_backed_table(tmp_path) -> None:
    path = tmp_path / "shared.json"
    first = recovery_leases_from_env({"AUTONOMY_RECOVERY_LEASE_PATH": str(path)})
    second = recovery_leases_from_env({"AUTONOMY_RECOVERY_LEASE_PATH": str(path)})
    ok, _ = first.try_acquire("task", "shared", "lane-0", ttl_seconds=60)
    assert ok
    denied, view = second.try_acquire("task", "shared", "lane-1", ttl_seconds=60)
    assert denied is False
    assert view["reason"] == "recovery_lease_held"


def test_state_root_env_selects_shared_lease_file(tmp_path) -> None:
    table = recovery_leases_from_env({"AGENT_SUPERVISOR_STATE_ROOT": str(tmp_path)})
    table.try_acquire("task", "from-root", "lane-0", ttl_seconds=60)
    assert (tmp_path / "recovery-leases.json").is_file()


def test_daemon_ttl_env_defaults_and_rejects_nonpositive() -> None:
    assert recovery_lease_ttl_from_env({}, default=300.0) == 300.0
    assert recovery_lease_ttl_from_env(
        {"AUTONOMY_RECOVERY_LEASE_TTL_SECONDS": "120"},
        default=300.0,
    ) == 120.0
    assert recovery_lease_ttl_from_env(
        {"AUTONOMY_RECOVERY_LEASE_TTL_SECONDS": "-1"},
        default=DEFAULT_DAEMON_RECOVERY_LEASE_TTL_SECONDS,
    ) == DEFAULT_DAEMON_RECOVERY_LEASE_TTL_SECONDS


def test_bundle_ttl_is_at_least_daemon_default_and_twice_duckdb_lease() -> None:
    assert bundle_recovery_lease_ttl_seconds(60_000) == 300.0
    assert bundle_recovery_lease_ttl_seconds(600_000) == 1_200.0
