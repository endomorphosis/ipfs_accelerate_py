"""Migration invokes one owner and preserves the gateway's admission state.

All invokers and the coordinator below are authored, bounded test doubles.
No provider, child process, ledger file, or repository target is executed.
"""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import json
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.provider_execution import (
    ProviderExecutionGateway,
    ProviderExecutionMode,
    ProviderExecutionPhase,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_usage_migration import (
    ConsumerId,
    SupervisorUsageMode,
    build_consumer_call_context,
    dispatch_migrated_provider_call,
)
from ipfs_accelerate_py.endpoint_usage import UsageVector


class _Coordinator:
    """Count exact physical transitions without supplying external capacity."""

    def __init__(self):
        self.reserved = []
        self.dispatched = []
        self.committed = []
        self.lock = threading.Lock()

    def reserve(self, scope_id, requested, **kwargs):
        with self.lock:
            reservation_id = "reservation:" + kwargs["request_id"]
            self.reserved.append((reservation_id, scope_id, requested))
        return SimpleNamespace(granted=True, reservation_id=reservation_id,
            usage_revision="usage:reserved", reason_codes=("reserved",))

    def mark_dispatched(self, reservation_id):
        with self.lock:
            self.dispatched.append(reservation_id)
        return SimpleNamespace(reservation_id=reservation_id, state="held")

    def commit(self, reservation_id, actual, **kwargs):
        with self.lock:
            self.committed.append(reservation_id)
        return SimpleNamespace(reservation_id=reservation_id, charged=actual, event_id="event:" + reservation_id,
            usage_revision="usage:settled", state="committed")


def _context(name="one"):
    return build_consumer_call_context(
        consumer_id=ConsumerId.TASK_PROPOSAL_ROUTER,
        provider_id="provider:authored-test-double",
        request_id="request:" + name,
        idempotency_key="idempotency:" + name,
        fence_id="1",
    )


def _observation():
    return {"status": "ok", "units": UsageVector.of(requests=1).to_dict()}


@pytest.mark.parametrize("mode", [SupervisorUsageMode.OBSERVE,
    SupervisorUsageMode.SHADOW, SupervisorUsageMode.ASSIST, SupervisorUsageMode.ENFORCE])
def test_migration_forwards_mode_to_configured_execution_owner(mode):
    seen = []
    coordinator = _Coordinator()

    def provider(request):
        seen.append(request.mode)
        return _observation()

    def forbidden_consumer():
        pytest.fail("configured gateway already owns execution")

    gateway = ProviderExecutionGateway(coordinator=coordinator, invoker=provider)
    result = dispatch_migrated_provider_call(context=_context(),
        invoke=forbidden_consumer, mode=mode, gateway=gateway)

    assert seen == [ProviderExecutionMode(mode.value)]
    assert result.execution_result.mode is ProviderExecutionMode(mode.value)
    assert result.execution_result.phase is ProviderExecutionPhase.SETTLED
    assert gateway.invoke_count("request:one#1") == 1
    assert len(coordinator.reserved) == len(coordinator.dispatched) == len(coordinator.committed) == 1
    assert result.text == ""
    assert "generated_text_unavailable" in result.reason_codes


def test_shared_gateway_preserves_replay_without_reinvoking_consumer():
    coordinator = _Coordinator()
    gateway = ProviderExecutionGateway(coordinator=coordinator)
    calls = []
    context = _context()

    def consumer():
        calls.append("once")
        return "authored output kept outside operational receipts"

    first = dispatch_migrated_provider_call(context=context, invoke=consumer,
        mode="enforce", gateway=gateway)
    replay = dispatch_migrated_provider_call(context=context,
        invoke=lambda: pytest.fail("replay must not execute a replacement callback"),
        mode="enforce", gateway=gateway)

    assert calls == ["once"]
    assert first.text == "authored output kept outside operational receipts"
    assert replay.execution_result.phase is ProviderExecutionPhase.REPLAYED
    assert replay.usage_receipt == first.usage_receipt
    assert replay.metered and replay.text == ""
    assert "generated_text_unavailable" in replay.reason_codes
    assert gateway.invoke_count("request:one#1") == 1
    assert len(coordinator.reserved) == len(coordinator.dispatched) == len(coordinator.committed) == 1
    assert first.text not in json.dumps(first.to_record())
    assert first.text not in json.dumps(replay.to_record())
    assert gateway.has_invoker is False


def test_configured_owner_replay_is_not_replaced_by_consumer_callback():
    coordinator = _Coordinator()
    owner_calls = []

    def owner(request):
        owner_calls.append(request.attempt_key)
        return _observation()

    gateway = ProviderExecutionGateway(coordinator=coordinator, invoker=owner)
    results = [dispatch_migrated_provider_call(context=_context(),
        invoke=lambda: pytest.fail("gateway has its own execution owner"),
        mode="enforce", gateway=gateway) for _ in range(2)]
    assert owner_calls == ["request:one#1"]
    assert results[1].execution_result.replayed is True
    assert results[1].usage_receipt == results[0].usage_receipt
    assert len(coordinator.reserved) == len(coordinator.committed) == 1
    assert gateway.has_invoker is True


def test_concurrent_calls_keep_per_call_dispatch_owners_separate():
    coordinator = _Coordinator()
    gateway = ProviderExecutionGateway(coordinator=coordinator)
    barrier = threading.Barrier(2)
    calls = []
    lock = threading.Lock()

    def submit(name):
        def invoke():
            with lock:
                calls.append(name)
            barrier.wait(timeout=5)
            return "authored:" + name
        return dispatch_migrated_provider_call(context=_context(name), invoke=invoke,
            mode="enforce", gateway=gateway)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(submit, ("one", "two")))
    assert sorted(calls) == ["one", "two"]
    assert [result.text for result in results] == ["authored:one", "authored:two"]
    assert len(coordinator.reserved) == len(coordinator.committed) == 2
    assert gateway.invoke_count("request:one#1") == gateway.invoke_count("request:two#1") == 1
    assert gateway.has_invoker is False


def test_enforce_without_coordinator_refuses_before_consumer_execution():
    result = dispatch_migrated_provider_call(context=_context(), mode="enforce",
        invoke=lambda: pytest.fail("missing authoritative coordination must refuse"))
    assert result.execution_result.phase is ProviderExecutionPhase.DENIED
    assert result.execution_result.granted is False
    assert result.text == "" and result.metered is False
    assert result.usage_receipt is None
    assert "coordination_fail_closed" in result.reason_codes


def test_cancelled_request_does_not_invoke_either_execution_owner():
    coordinator = _Coordinator()
    gateway = ProviderExecutionGateway(coordinator=coordinator,
        invoker=lambda _: pytest.fail("cancelled request must not invoke gateway"))
    result = dispatch_migrated_provider_call(context=_context(), mode="enforce",
        invoke=lambda: pytest.fail("cancelled request must not invoke consumer"),
        cancelled=True, gateway=gateway)
    assert result.execution_result.phase is ProviderExecutionPhase.CANCELLED
    assert coordinator.reserved == []
    assert gateway.invoke_count("request:one#1") == 0


def test_explicit_off_mode_preserves_existing_consumer_path():
    gateway = ProviderExecutionGateway(
        invoker=lambda _: pytest.fail("off mode uses legacy consumer dispatch"))
    result = dispatch_migrated_provider_call(context=_context(),
        invoke=lambda: "legacy authored text", mode="off", gateway=gateway)
    assert result.text == "legacy authored text" and result.phase == "off"
    assert result.execution_result is None
