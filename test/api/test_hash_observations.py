"""Owner-side hash reuse and producer fencing, using a hermetic DuckDB."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources import hash_observations as hashes

duckdb = pytest.importorskip("duckdb")
DIGEST = hashlib.sha256(b"test bytes").hexdigest()


class Clock:
    wall = 1_000_000
    mono = 1_000

    def advance(self, milliseconds):
        self.wall += milliseconds
        self.mono += milliseconds


@pytest.fixture
def database():
    connection = duckdb.connect(":memory:", config={"threads": 1})
    sql = Path(hashes.__file__).parent / "sql" / "0004_hash_observations.sql"
    connection.execute(sql.read_text(encoding="utf-8"))
    yield connection
    connection.close()


@pytest.fixture
def clock():
    return Clock()


def store(database, clock, generation="owner:1", **kwargs):
    return hashes.HashObservationStore(
        database, generation=generation, clock_ms=lambda: clock.wall,
        monotonic_ms=lambda: clock.mono, **kwargs,
    )


def identity(**overrides):
    return {
        "schema": hashes.IDENTITY_SCHEMA, "algorithm": "sha256",
        "profile": hashes.IDENTITY_PROFILE, "host_boot_id": "boot-test",
        "mount_id": "mount-test", "dev": 1, "ino": 2, "mode": 0o100644,
        "uid": 1000, "gid": 1000, "nlink": 1, "size": 10,
        "mtime_ns": 123, "ctime_ns": 123, **overrides,
    }


def request(action="claim", *, witness=None, **kwargs):
    witness = identity() if witness is None else witness
    return {"action": action, "identity": witness, "key": hashes.identity_key(witness), **kwargs}


def finish(service, claim, *, principal="producer", witness=None, **kwargs):
    return service.handle(
        request(
            "complete", witness=witness, generation=claim["generation"],
            lease_token=claim["lease_token"], fence=claim["fence"], sha256=DIGEST, **kwargs,
        ), principal=principal,
    )


def test_default_day_ttl_is_owner_timed_and_hits_do_not_extend_it(database, clock):
    service = store(database, clock)
    claim = service.handle(request(), principal="producer")
    assert claim["expires_at_ms"] - clock.wall == 86_400_000
    clock.advance(9000)
    observation = finish(service, claim)
    assert observation["observed_at_ms"] == 1_000_000
    assert observation["expires_at_ms"] == claim["expires_at_ms"]
    clock.advance(86_400_000 - 9001)
    hit = service.handle(request(), principal="follower")
    assert hit == observation
    clock.advance(1)
    refresh = service.handle(request(), principal="follower")
    assert refresh["status"] == "claimed"
    assert refresh["fence"] == claim["fence"] + 1


def test_shared_backend_clients_deduplicate_same_key_allow_distinct_keys(database, clock):
    first, second = store(database, clock), store(database, clock)
    claim = first.handle(request(), principal="producer")
    busy = second.handle(request(), principal="follower")
    assert busy["status"] == "busy" and 0 < busy["retry_after_ms"] <= 100
    assert "lease_token" not in busy and "principal" not in busy
    other = second.handle(request(witness=identity(ino=3)), principal="follower")
    assert other["status"] == "claimed"
    finish(first, claim)
    assert second.handle(request(), principal="follower")["sha256"] == DIGEST


@pytest.mark.parametrize("field,value", [
    ("dev", 4), ("ino", 3), ("mode", 0o100600), ("uid", 12), ("gid", 13),
    ("nlink", 2), ("size", 11), ("mtime_ns", 124), ("ctime_ns", 124),
    ("host_boot_id", "new-boot"), ("mount_id", "new-mount"),
])
def test_any_file_witness_change_immediately_invalidates_reuse(database, clock, field, value):
    service = store(database, clock)
    claim = service.handle(request(), principal="producer")
    finish(service, claim)
    assert service.handle(request(witness=identity(**{field: value})), principal="follower")["status"] == "claimed"


def test_expired_producer_cannot_publish_over_replacement(database, clock):
    service = store(database, clock)
    old = service.handle(request(lease_seconds=1), principal="producer")
    clock.advance(1000)
    new = service.handle(request(), principal="replacement")
    assert new["fence"] == old["fence"] + 1
    assert new["lease_token"] != old["lease_token"]
    with pytest.raises(hashes.HashObservationError, match="stale or foreign"):
        finish(service, old)
    assert finish(service, new, principal="replacement")["sha256"] == DIGEST


def test_restart_reuses_complete_observation_but_fences_abandoned_claim(database, clock):
    old = store(database, clock)
    completed = old.handle(request(), principal="producer")
    finish(old, completed)
    other_witness = identity(ino=3)
    abandoned = old.handle(request(witness=other_witness), principal="producer")
    new = store(database, clock, generation="owner:2")
    assert new.handle(request(), principal="reader")["status"] == "hit"
    replacement = new.handle(request(witness=other_witness), principal="replacement")
    assert replacement["status"] == "claimed"
    with pytest.raises(hashes.HashObservationError, match="stale or foreign"):
        finish(new, abandoned, witness=other_witness)


@pytest.mark.parametrize("patch", [
    {"sha256": "bad"}, {"sha256": "A" * 64}, {"lease_token": "0" * 64},
    {"generation": "other-owner"}, {"fence": 222}, {"fence": True},
    {"observed_at_ms": 1}, {"expires_at_ms": 999999999999},
])
def test_malformed_and_poisoned_publications_fail_closed(database, clock, patch):
    service = store(database, clock)
    claim = service.handle(request(), principal="producer")
    publication = request(
        "complete", generation=claim["generation"], lease_token=claim["lease_token"],
        fence=claim["fence"], sha256=DIGEST,
    )
    with pytest.raises(hashes.HashObservationError):
        service.handle({**publication, **patch}, principal="producer")
    assert service.handle(request(), principal="follower")["status"] == "busy"


def test_wrong_principal_or_identity_cannot_complete_or_abort(database, clock):
    service = store(database, clock)
    claim = service.handle(request(), principal="producer")
    with pytest.raises(hashes.HashObservationError, match="stale or foreign"):
        finish(service, claim, principal="impostor")
    with pytest.raises(hashes.HashObservationError, match="stale or foreign"):
        finish(service, claim, witness=identity(ctime_ns=125))
    abort = request("abort", generation=claim["generation"], lease_token=claim["lease_token"], fence=claim["fence"])
    with pytest.raises(hashes.HashObservationError, match="stale or foreign"):
        service.handle(abort, principal="impostor")
    assert service.handle(abort, principal="producer")["status"] == "aborted"
    assert service.handle(request(), principal="next")["fence"] == claim["fence"] + 1


def test_shorter_requested_ttl_is_respected(database, clock):
    service = store(database, clock)
    claim = service.handle(request(), principal="producer")
    finish(service, claim)
    clock.advance(1000)
    assert service.handle(request("lookup", ttl_seconds=1), principal="reader")["status"] == "miss"
    assert service.handle(request("lookup"), principal="reader")["status"] == "hit"


def test_completion_after_observation_ttl_cutoff_is_rejected(database, clock):
    service = store(database, clock)
    claim = service.handle(request(ttl_seconds=1), principal="producer")
    clock.advance(1000)
    with pytest.raises(hashes.HashObservationError, match="TTL expired"):
        finish(service, claim)


@pytest.mark.parametrize("ttl_ms,lease_ms", [(1, 1), (1500, 500), (604800000, 300000)])
def test_integer_wire_durations_preserve_subsecond_precision(database, clock, ttl_ms, lease_ms):
    service = store(database, clock)
    claim = service.handle(request(ttl_ms=ttl_ms, lease_ms=lease_ms), principal="producer")
    assert claim["expires_at_ms"] == clock.wall + ttl_ms
    assert claim["lease_expires_ms"] == clock.wall + lease_ms
    finish(service, claim)
    assert service.handle(request("lookup", ttl_ms=ttl_ms), principal="reader")["status"] == "hit"
    clock.advance(ttl_ms)
    assert service.handle(request("lookup", ttl_ms=ttl_ms), principal="reader")["status"] == "miss"


@pytest.mark.parametrize("durations", [
    {"ttl_ms": 0}, {"ttl_ms": -1}, {"ttl_ms": 604800001},
    {"ttl_ms": True}, {"ttl_ms": 1000.0}, {"ttl_ms": 1.5},
    {"lease_ms": 0}, {"lease_ms": 300001}, {"lease_ms": False},
    {"lease_ms": 1000.0}, {"lease_ms": 1.5},
    {"ttl_ms": 1000, "ttl_seconds": 1},
    {"lease_ms": 1000, "lease_seconds": 1},
    {"ttl_seconds": 10**1000}, {"lease_seconds": 10**1000},
])
def test_invalid_or_conflicting_duration_units_raise_typed_error(database, clock, durations):
    service = store(database, clock)
    with pytest.raises(hashes.HashObservationError):
        service.handle(request(**durations), principal="producer")
    assert database.execute("SELECT COUNT(*) FROM hash_observations").fetchone()[0] == 0


def test_live_claims_are_never_evicted_to_make_room(database, clock):
    service = store(database, clock, max_rows=2)
    first = service.handle(request(), principal="producer")
    service.handle(request(witness=identity(ino=3)), principal="other")
    with pytest.raises(hashes.HashObservationError, match="capacity"):
        service.handle(request(witness=identity(ino=4)), principal="new")
    finish(service, first)
    assert service.handle(request(witness=identity(ino=4)), principal="new")["status"] == "claimed"
    assert database.execute("SELECT COUNT(*) FROM hash_observations").fetchone()[0] == 2
    assert service.handle(request("lookup", witness=identity(ino=3)), principal="reader")["status"] == "busy"


def test_expired_rows_are_cleaned_up_before_capacity_admission(database, clock):
    service = store(database, clock, max_rows=1)
    service.handle(request(lease_seconds=1), principal="producer")
    clock.advance(1000)
    assert service.handle(request(witness=identity(ino=3)), principal="new")["status"] == "claimed"
    assert database.execute("SELECT COUNT(*) FROM hash_observations").fetchone()[0] == 1


@pytest.mark.parametrize("rollback", ["wall", "monotonic", "lost_wall_time"])
def test_clock_rollback_invalidates_observations_and_old_fences(database, clock, rollback):
    service = store(database, clock)
    first = service.handle(request(), principal="producer")
    finish(service, first)
    other = service.handle(request(witness=identity(ino=3)), principal="producer")
    if rollback == "wall":
        clock.wall -= 1
    elif rollback == "monotonic":
        clock.mono -= 1
    else:
        clock.mono += 2001
    with pytest.raises(hashes.HashObservationError, match="stale or foreign"):
        finish(service, other, witness=identity(ino=3))
    fresh = service.handle(request(), principal="new")
    assert fresh["status"] == "claimed"
    assert fresh["generation"] != first["generation"]


def test_future_observation_is_not_reused_after_restart(database, clock):
    service = store(database, clock)
    claim = service.handle(request(), principal="producer")
    finish(service, claim)
    clock.wall -= 1
    restarted = store(database, clock, generation="owner:2")
    assert restarted.handle(request(), principal="new")["status"] == "claimed"


@pytest.mark.parametrize("patch", [
    {"key": "0" * 64}, {"ttl_seconds": 604801}, {"ttl_seconds": 0},
    {"lease_seconds": 301}, {"lease_seconds": float("nan")},
    {"ttl_seconds": True}, {"action": []}, {"extra": "forbidden"},
])
def test_invalid_requests_do_not_mutate_database(database, clock, patch):
    service = store(database, clock)
    with pytest.raises(hashes.HashObservationError):
        service.handle({**request(), **patch}, principal="producer")
    assert database.execute("SELECT COUNT(*) FROM hash_observations").fetchone()[0] == 0


@pytest.mark.parametrize("patch", [
    {"mode": 0o120777}, {"mode": True}, {"nlink": 0}, {"size": -1},
    {"host_boot_id": ""}, {"profile": "other"}, {"algorithm": "md5"},
])
def test_identity_contract_rejects_unsafe_or_unsupported_witnesses(patch):
    with pytest.raises(hashes.HashObservationError):
        hashes.identity_key(identity(**patch))


def test_unmigrated_connection_fails_without_ddl():
    connection = duckdb.connect(":memory:", config={"threads": 1})
    try:
        with pytest.raises(hashes.HashObservationUnavailableError):
            hashes.HashObservationStore(connection, generation="owner:1")
        assert connection.execute("SHOW TABLES").fetchall() == []
    finally:
        connection.close()
