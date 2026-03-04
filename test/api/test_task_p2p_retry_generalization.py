import anyio
import pytest


def test_retryable_transport_error_includes_discovery_timeout():
    from ipfs_accelerate_py.p2p_tasks.client import _is_retryable_transport_error

    assert _is_retryable_transport_error(RuntimeError("discovery_timeout"))


@pytest.mark.parametrize(
    "message",
    [
        "discovery timeout",
        "no response from peer",
        "connection reset by peer",
        "connection refused",
        "temporarily unavailable",
        "broken pipe",
    ],
)
def test_retryable_transport_error_additional_transient_markers(message: str):
    from ipfs_accelerate_py.p2p_tasks.client import _is_retryable_transport_error

    assert _is_retryable_transport_error(RuntimeError(message))


def test_submit_task_retries_on_retryable_response(monkeypatch):
    import ipfs_accelerate_py.p2p_tasks.client as client
    from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, get_p2p_retry_metrics, reset_p2p_retry_metrics

    responses = [
        {"ok": False, "error": "discovery_timeout"},
        {"ok": True, "task_id": "task-123"},
    ]

    async def fake_dial_and_request(*, remote, message, dial_timeout_s, allow_broad_discovery_override=None):
        assert isinstance(message, dict)
        return responses.pop(0)

    async def fake_acquire_dial_slot(*, op_label):
        assert op_label == "submit"

        def _release() -> None:
            return None

        return _release

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_SUBMIT_RETRIES", "1")
    monkeypatch.setattr(client, "_dial_and_request", fake_dial_and_request)
    monkeypatch.setattr(client, "_acquire_dial_slot", fake_acquire_dial_slot)

    reset_p2p_retry_metrics()

    async def _do() -> None:
        remote = RemoteQueue(peer_id="peerA", multiaddr="")
        task_id = await client.submit_task(
            remote=remote,
            task_type="text_generation",
            model_name="gpt2",
            payload={"prompt": "hello"},
        )
        assert task_id == "task-123"

    anyio.run(_do, backend="trio")

    metrics = get_p2p_retry_metrics()
    assert metrics.get("submit.retry_response", 0) >= 1
    assert metrics.get("submit.retry", 0) >= 1
    assert metrics.get("submit.recovered", 0) >= 1


def test_submit_task_retry_uses_lightweight_discovery_override(monkeypatch):
    import ipfs_accelerate_py.p2p_tasks.client as client
    from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, get_p2p_retry_metrics, reset_p2p_retry_metrics

    seen_overrides: list[bool | None] = []
    attempts = {"n": 0}

    async def fake_dial_and_request(*, remote, message, dial_timeout_s, allow_broad_discovery_override=None):
        del remote, message, dial_timeout_s
        seen_overrides.append(allow_broad_discovery_override)
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("p2p request failed: no response")
        return {"ok": True, "task_id": "task-456"}

    async def fake_acquire_dial_slot(*, op_label):
        assert op_label == "submit"

        def _release() -> None:
            return None

        return _release

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_SUBMIT_RETRIES", "1")
    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_RETRY_LIGHTWEIGHT_DISCOVERY", "1")
    monkeypatch.setattr(client, "_dial_and_request", fake_dial_and_request)
    monkeypatch.setattr(client, "_acquire_dial_slot", fake_acquire_dial_slot)

    reset_p2p_retry_metrics()

    async def _do() -> None:
        remote = RemoteQueue(peer_id="peerB", multiaddr="")
        task_id = await client.submit_task(
            remote=remote,
            task_type="text_generation",
            model_name="gpt2",
            payload={"prompt": "hello"},
        )
        assert task_id == "task-456"

    anyio.run(_do, backend="trio")

    assert seen_overrides == [None, False]
    metrics = get_p2p_retry_metrics()
    assert metrics.get("submit.retry_lightweight_discovery", 0) >= 1


def test_dial_and_request_with_retries_uses_lightweight_override(monkeypatch):
    import ipfs_accelerate_py.p2p_tasks.client as client
    from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, get_p2p_retry_metrics, reset_p2p_retry_metrics

    seen_overrides: list[bool | None] = []
    attempts = {"n": 0}

    async def fake_dial_and_request(*, remote, message, dial_timeout_s, allow_broad_discovery_override=None):
        del remote, message, dial_timeout_s
        seen_overrides.append(allow_broad_discovery_override)
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("p2p request failed: no response")
        return {"ok": True}

    async def fake_acquire_dial_slot(*, op_label):
        assert op_label == "status"

        def _release() -> None:
            return None

        return _release

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_RETRY_LIGHTWEIGHT_DISCOVERY", "1")
    monkeypatch.setattr(client, "_dial_and_request", fake_dial_and_request)
    monkeypatch.setattr(client, "_acquire_dial_slot", fake_acquire_dial_slot)

    reset_p2p_retry_metrics()

    async def _do() -> None:
        remote = RemoteQueue(peer_id="peerC", multiaddr="")
        resp = await client._dial_and_request_with_retries(
            remote=remote,
            message={"op": "status"},
            retries=1,
            retry_base_ms=10,
            dial_timeout_s=1.0,
            op_label="status",
        )
        assert resp.get("ok") is True

    anyio.run(_do, backend="trio")

    assert seen_overrides == [None, False]
    metrics = get_p2p_retry_metrics()
    assert metrics.get("status.retry_lightweight_discovery", 0) >= 1


def test_explicit_addr_cooldown_mark_failure_and_success():
    import ipfs_accelerate_py.p2p_tasks.client as client

    ma = "/ip4/127.0.0.1/tcp/9100/p2p/12D3KooWExample"
    # Ensure clean state before assertions.
    client._explicit_addr_cooldown_mark_success(ma)

    client._explicit_addr_cooldown_mark_failure(ma)
    assert client._explicit_addr_cooldown_wait_s(ma) > 0.0

    client._explicit_addr_cooldown_mark_success(ma)
    assert client._explicit_addr_cooldown_wait_s(ma) == 0.0


def test_retry_delay_is_bounded_for_large_attempts(monkeypatch):
    import ipfs_accelerate_py.p2p_tasks.client as client

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_RETRY_DELAY_MAX_MS", "200")
    # The helper includes jitter in [0, min(base_ms, stage_delay_ms)], so
    # assert bounds instead of an exact value.
    delay_s = client._retry_delay_s(attempt=999, base_ms=50)
    assert delay_s >= 0.200
    assert delay_s <= 0.250


def test_request_status_retries_on_retryable_response(monkeypatch):
    import ipfs_accelerate_py.p2p_tasks.client as client
    from ipfs_accelerate_py.p2p_tasks.client import RemoteQueue, get_p2p_retry_metrics, reset_p2p_retry_metrics

    responses = [
        {"ok": False, "error": "discovery_timeout"},
        {"ok": True, "status": "ready"},
    ]

    async def fake_dial_and_request(*, remote, message, dial_timeout_s, allow_broad_discovery_override=None):
        del remote, dial_timeout_s, allow_broad_discovery_override
        assert message.get("op") == "status"
        return responses.pop(0)

    async def fake_acquire_dial_slot(*, op_label):
        assert op_label == "status"

        def _release() -> None:
            return None

        return _release

    monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_STATUS_RETRIES", "1")
    monkeypatch.setattr(client, "_dial_and_request", fake_dial_and_request)
    monkeypatch.setattr(client, "_acquire_dial_slot", fake_acquire_dial_slot)

    reset_p2p_retry_metrics()

    async def _do() -> None:
        remote = RemoteQueue(peer_id="peer-status", multiaddr="")
        resp = await client.request_status(remote=remote, timeout_s=1.0, detail=False)
        assert resp.get("ok") is True
        assert resp.get("status") == "ready"

    anyio.run(_do, backend="trio")

    metrics = get_p2p_retry_metrics()
    assert metrics.get("status.retry_response", 0) >= 1
    assert metrics.get("status.retry", 0) >= 1
    assert metrics.get("status.recovered", 0) >= 1
