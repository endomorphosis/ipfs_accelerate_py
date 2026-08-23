"""EAAEF-093: one fenced in-memory DuckDB/Quack owner and failover."""

from __future__ import annotations

from threading import Thread

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.external_quack_owner import (
    INITIAL_EPOCH,
    LIVE_QUACK_PORT,
    REMOTE_CAPABILITIES,
    BoundedQuackTransport,
    DuplicateOwnerError,
    ExternalQuackOwner,
    RemoteSqlRefusedError,
    StaleOwnerError,
    TransportAuthError,
    UnsignedEnvelopeError,
    issue_envelope,
)


OWNER_A = "owner-a"
OWNER_B = "owner-b"


def _owner(owner_id: str = OWNER_A) -> ExternalQuackOwner:
    return ExternalQuackOwner(owner_id, shard_id="disposable-test-shard")


def _put(
    key: str = "task-1",
    *,
    status: str = "claimed",
    principal_id: str = "principal:worker",
    idempotency_key: str = "idem-1",
) -> dict[str, object]:
    return issue_envelope(
        operation="put",
        key=key,
        value={"status": status},
        principal_id=principal_id,
        idempotency_key=idempotency_key,
    )


def test_single_owner_epoch_starts_at_one() -> None:
    owner = _owner()
    lease = owner.lease()
    assert owner.epoch == INITIAL_EPOCH == 1
    assert owner.owner_id == OWNER_A
    assert lease.owner_id == OWNER_A
    assert lease.epoch == 1
    assert lease.fence == 1
    assert owner.claim(OWNER_A, epoch=1).epoch == 1
    assert owner.operational_table_exposed is False
    assert owner.remote_capabilities == frozenset({"append", "read"})
    assert owner.bound_port is None
    assert owner.transport.bound_port is None
    assert owner.transport.listen_uri == ""
    assert LIVE_QUACK_PORT == 19495


def test_failover_advances_epoch() -> None:
    owner = _owner()
    first = owner.lease()
    owner.apply(_put(), owner_id=OWNER_A, epoch=first.epoch)
    takeover = owner.failover(OWNER_B)
    assert takeover.owner_id == OWNER_B
    assert takeover.epoch == first.epoch + 1
    assert takeover.fence == first.fence + 1
    assert owner.epoch == 2
    assert owner.owner_id == OWNER_B
    restart = owner.failover()
    assert restart.owner_id == OWNER_B
    assert restart.epoch == 3
    assert owner.get("task-1") is not None


def test_stale_owner_with_old_epoch_is_rejected() -> None:
    owner = _owner()
    stale = owner.lease()
    owner.apply(_put(), owner_id=stale.owner_id, epoch=stale.epoch)
    owner.failover(OWNER_B)
    with pytest.raises(StaleOwnerError, match="stale owner") as err:
        owner.apply(
            _put(status="running", idempotency_key="idem-stale"),
            owner_id=stale.owner_id,
            epoch=stale.epoch,
        )
    assert err.value.reason_code == "stale_owner"
    assert owner.get("task-1")["status"] == "claimed"
    receipt = owner.apply(
        _put(status="running", idempotency_key="idem-2"),
        owner_id=OWNER_B,
        epoch=2,
    )
    assert receipt["status"] == "applied"
    assert receipt["epoch"] == 2
    assert owner.get("task-1")["status"] == "running"


def test_second_owner_without_failover_fails_closed() -> None:
    owner = _owner()
    with pytest.raises(DuplicateOwnerError, match="second owner") as err:
        owner.claim(OWNER_B, epoch=1)
    assert err.value.reason_code == "duplicate_owner"
    with pytest.raises(StaleOwnerError, match="stale owner"):
        owner.apply(_put(), owner_id=OWNER_B, epoch=1)
    assert owner.get("task-1") is None


def test_remote_update_sql_and_arbitrary_sql_are_refused() -> None:
    owner = _owner()
    sql = "UPDATE tasks SET status = 'hijacked'"
    with pytest.raises(RemoteSqlRefusedError, match="remote UPDATE") as remote:
        owner.remote_update_sql(sql)
    assert remote.value.reason_code == "remote_sql_refused"
    with pytest.raises(RemoteSqlRefusedError, match="arbitrary SQL") as arbitrary:
        owner.execute_sql("SELECT * FROM tasks")
    assert arbitrary.value.reason_code == "remote_sql_refused"
    with pytest.raises(RemoteSqlRefusedError, match="remote UPDATE"):
        owner.transport.remote_update_sql(sql)
    with pytest.raises(RemoteSqlRefusedError, match="arbitrary SQL"):
        owner.transport.execute_sql("DROP TABLE tasks")
    assert owner.get("task-1") is None
    assert owner.operational_table_exposed is False


def test_unsigned_and_forged_envelopes_fail_closed() -> None:
    owner = _owner()
    with pytest.raises(UnsignedEnvelopeError, match="missing") as missing:
        owner.apply({}, owner_id=OWNER_A, epoch=1)
    assert missing.value.reason_code == "unsigned_envelope"
    forged = _put()
    forged["value"] = {"status": "forged"}
    with pytest.raises(UnsignedEnvelopeError, match="forged") as err:
        owner.apply(forged, owner_id=OWNER_A, epoch=1)
    assert err.value.reason_code == "forged_envelope"
    assert owner.get("task-1") is None


def test_signed_envelope_serializes_private_duckdb_transaction() -> None:
    owner = _owner()
    first = owner.apply(_put(), owner_id=OWNER_A, epoch=1)
    replay = owner.apply(_put(), owner_id=OWNER_A, epoch=1)
    assert first["status"] == "applied"
    assert replay["envelope_content_id"] == first["envelope_content_id"]
    assert dict(replay) == dict(first)
    assert owner.get("task-1")["status"] == "claimed"
    owner.apply(
        _put(status="running", idempotency_key="idem-2"),
        owner_id=OWNER_A,
        epoch=1,
    )
    assert owner.get("task-1")["status"] == "running"


def test_transport_is_authenticated_multi_reader_multi_writer() -> None:
    owner = _owner()
    transport = owner.transport
    writer_a = transport.attach("writer-a", role="writer", token="token-a")
    writer_b = transport.attach("writer-b", role="writer", token="token-b")
    reader_a = transport.attach("reader-a", role="reader", token="token-c")
    reader_b = transport.attach("reader-b", role="reader", token="token-d")
    first = _put(key="task-a", idempotency_key="idem-a")
    second = _put(key="task-b", status="queued", idempotency_key="idem-b")
    transport.append(writer_a, first)
    transport.append(writer_b, second)
    seen_a = transport.read(reader_a)
    seen_b = transport.read(reader_b)
    assert len(seen_a) == 2
    assert seen_a == seen_b
    assert REMOTE_CAPABILITIES == frozenset({"append", "read"})
    with pytest.raises(TransportAuthError, match="reader"):
        transport.append(reader_a, _put(idempotency_key="idem-reader"))
    with pytest.raises(TransportAuthError, match="writer"):
        transport.read(writer_a)
    with pytest.raises(TransportAuthError, match="token is required") as blank:
        transport.attach("blank", role="writer", token="")
    assert blank.value.reason_code == "transport_auth"
    receipts = owner.apply_from_transport(owner_id=OWNER_A, epoch=1)
    assert {receipt["key"] for receipt in receipts} == {"task-a", "task-b"}
    assert owner.get("task-a")["status"] == "claimed"
    assert owner.get("task-b")["status"] == "queued"


def test_never_binds_live_quack_port() -> None:
    owner = _owner()
    assert owner.bound_port is not LIVE_QUACK_PORT
    assert owner.bound_port is None
    assert owner.transport.bound_port is None
    assert "19495" not in owner.transport.listen_uri
    assert isinstance(owner.transport, BoundedQuackTransport)


def test_private_duckdb_transactions_are_serialized() -> None:
    owner = _owner()
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        envelope = issue_envelope(
            operation="increment",
            key="counter",
            value={"n": 1},
            principal_id=f"principal:{index}",
            idempotency_key=f"inc-{index}",
        )
        try:
            owner.apply(envelope, owner_id=OWNER_A, epoch=1)
        except BaseException as exc:  # noqa: BLE001 — collect then assert
            errors.append(exc)

    threads = tuple(Thread(target=worker, args=(index,)) for index in range(16))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert owner.get("counter")["n"] == 16
