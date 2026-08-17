"""SCH-005 existing-supervisor scheduling adapter tests."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import threading
import types
from pathlib import Path
from typing import Any, Mapping

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
    HostResourceSnapshot,
    ResourcePolicy,
    ResourceScheduler,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessError,
    HarnessMode,
    WorkKind,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.scheduling_contracts import (
    ProviderBinding,
    SemanticWorkRequest,
    SemanticWorkStatus,
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _provider(
    *,
    mode: str = HarnessMode.DEVELOPMENT.value,
    simulated: bool = True,
    reservation_id: str | None = None,
) -> ProviderBinding:
    if reservation_id is None:
        reservation_id = "sim:dev-1" if simulated else "resv:prod-1"
    return ProviderBinding.from_dict(
        {
            "provider_id": "provider-a",
            "reservation_id": reservation_id,
            "mode": mode,
            "simulated": simulated,
        }
    )


def _request(
    work_kind: str = WorkKind.SCAN.value,
    *,
    mode: str = HarnessMode.DEVELOPMENT.value,
    attempt_id: str = "attempt-1",
    provider: ProviderBinding | None = None,
    **overrides: object,
) -> SemanticWorkRequest:
    if work_kind == WorkKind.MODEL_INVOCATION.value and provider is None:
        provider = _provider(
            mode=mode,
            simulated=mode == HarnessMode.DEVELOPMENT.value,
        )
    request = SemanticWorkRequest.build(
        work_kind=work_kind,
        attempt_id=attempt_id,
        repository_id="example/repo",
        mode=mode,
        input_artifact_cids=[_cid("input-a")],
        base_root_cid=_cid("base-root"),
        provider=provider,
    )
    if overrides:
        payload = request.to_dict()
        payload.update(overrides)
        return SemanticWorkRequest.from_dict(payload)
    return request


def _ample_host(**overrides: object) -> HostResourceSnapshot:
    payload = {
        "observed_at_ms": 1_700_000_000_000,
        "cpu_percent": 5,
        "memory_percent": 10,
        "disk_percent": 10,
        "memory_total_bytes": 16 * 1024 * 1024 * 1024,
        "memory_available_bytes": 8 * 1024 * 1024 * 1024,
        "disk_total_bytes": 200 * 1024 * 1024 * 1024,
        "disk_available_bytes": 100 * 1024 * 1024 * 1024,
        "active_workers": 0,
        "worker_limit": 8,
        "available_worker_capacity": 8,
        "capabilities": ("cpu",),
    }
    payload.update(overrides)
    return HostResourceSnapshot(**payload)  # type: ignore[arg-type]


def _exhausted_host() -> HostResourceSnapshot:
    return _ample_host(
        active_workers=1,
        worker_limit=1,
        available_worker_capacity=0,
        cpu_percent=95,
        memory_percent=95,
    )


class _FakeProviderResult:
    def __init__(
        self,
        *,
        phase: str = "settled",
        final_status: str = "committed",
        success: bool = True,
        reason_codes: tuple[str, ...] = (),
        replayed: bool = False,
        coordination_state: str = "available",
        observation: Mapping[str, Any] | None = None,
    ) -> None:
        self.phase = phase
        self.final_status = final_status
        self._success = success
        self.reason_codes = reason_codes
        self.replayed = replayed
        self.coordination_state = coordination_state
        self.observation = dict(observation or {})

    @property
    def success(self) -> bool:
        return self._success


class FakeProviderGateway:
    """Hermetic provider gateway double with exact-attempt replay."""

    def __init__(
        self,
        *,
        result: _FakeProviderResult | None = None,
        raise_on_execute: Exception | None = None,
    ) -> None:
        self.result = result or _FakeProviderResult()
        self.raise_on_execute = raise_on_execute
        self.execute_calls = 0
        self.invoke_calls = 0
        self.seen_requests: list[Any] = []
        self._terminals: dict[str, _FakeProviderResult] = {}

    def invoke_count(self, attempt_key: str) -> int:
        return self.invoke_calls if attempt_key else 0

    def execute(self, request: Any) -> _FakeProviderResult:
        self.execute_calls += 1
        self.seen_requests.append(request)
        attempt_key = getattr(request, "attempt_key", "default")
        prior = self._terminals.get(attempt_key)
        if prior is not None:
            return _FakeProviderResult(
                phase=prior.phase,
                final_status=prior.final_status,
                success=prior.success,
                reason_codes=tuple(prior.reason_codes) + ("exact_replay",),
                replayed=True,
                coordination_state=prior.coordination_state,
                observation=prior.observation,
            )
        if bool(getattr(request, "cancelled", False)):
            outcome = _FakeProviderResult(
                phase="cancelled",
                final_status="cancelled",
                success=False,
                reason_codes=("pre_dispatch_cancelled",),
            )
            self._terminals[attempt_key] = outcome
            return outcome
        if self.raise_on_execute is not None:
            raise self.raise_on_execute
        self.invoke_calls += 1
        outcome = self.result
        self._terminals[attempt_key] = outcome
        return outcome


class _FakeProviderRequest:
    def __init__(
        self,
        *,
        attempt_key: str = "request:1#1",
        cancelled: bool = False,
    ) -> None:
        self.attempt_key = attempt_key
        self.cancelled = cancelled


class _FakeProcess:
    def __init__(self) -> None:
        self.terminate_calls = 0
        self.kill_calls = 0
        self.returncode: int | None = None

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def poll(self) -> int | None:
        return self.returncode


@pytest.fixture
def scheduling_mod():
    return importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.scheduling"
    )


def test_cold_import_starts_no_resources_threads_processes_or_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing scheduling.py must not start threads, processes, DBs, or network."""

    module_name = "ipfs_accelerate_py.agent_supervisor.semantic_state.scheduling"
    sys.modules.pop(module_name, None)

    before_threads = {t.ident for t in threading.enumerate()}

    real_thread_start = threading.Thread.start
    started_threads: list[str] = []

    def guarded_start(self: threading.Thread, *args: Any, **kwargs: Any) -> None:
        started_threads.append(self.name)
        return real_thread_start(self, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "start", guarded_start)

    real_popen = subprocess.Popen
    popen_calls: list[Any] = []

    def guarded_popen(*args: Any, **kwargs: Any):
        popen_calls.append((args, kwargs))
        raise AssertionError("cold import must not spawn subprocesses")

    monkeypatch.setattr(subprocess, "Popen", guarded_popen)

    socket_mod = importlib.import_module("socket")
    real_socket = socket_mod.socket
    socket_calls: list[Any] = []

    class GuardedSocket(real_socket):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            socket_calls.append((args, kwargs))
            raise AssertionError("cold import must not open sockets")

    monkeypatch.setattr(socket_mod, "socket", GuardedSocket)

    # DuckDB / sqlite open guards via builtins open on .duckdb is too broad;
    # instead ensure no connection factories are used at import time.
    created_engines: list[str] = []

    class _NoDB:
        def connect(self, *args: Any, **kwargs: Any):
            created_engines.append("connect")
            raise AssertionError("cold import must not open databases")

    fake_duckdb = types.ModuleType("duckdb")
    fake_duckdb.connect = _NoDB().connect  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "duckdb", fake_duckdb)

    mod = importlib.import_module(module_name)
    assert mod.SEMANTIC_SCHEDULING_ADAPTER_INTERFACE == "SemanticSchedulingAdapter@1"
    assert started_threads == []
    assert popen_calls == []
    assert socket_calls == []
    assert created_engines == []

    after_threads = {t.ident for t in threading.enumerate()}
    # No new non-main threads from import.
    assert after_threads == before_threads or after_threads.issuperset(before_threads)
    new_ids = after_threads - before_threads
    assert new_ids == set()


def test_descriptor_declares_composition_and_invariants(scheduling_mod) -> None:
    desc = scheduling_mod.semantic_scheduling_adapter_descriptor()
    assert desc["interface"] == "SemanticSchedulingAdapter@1"
    assert "ResourceScheduler" in desc["composes"]
    assert "ProviderExecutionGateway" in desc["composes"]
    assert "runtime.event_log" in desc["composes"]
    assert "expired_fences_cannot_publish" in desc["invariants"]
    assert "cold_import_starts_no_resources" in desc["invariants"]


def test_capacity_absence_returns_typed_unavailable(scheduling_mod) -> None:
    adapter = scheduling_mod.SemanticSchedulingAdapter(
        resource_scheduler=ResourceScheduler(ResourcePolicy(max_lanes=1)),
        host=_exhausted_host(),
    )
    attempt = adapter.schedule(_request(WorkKind.SCAN.value))
    assert attempt.result.status == SemanticWorkStatus.UNAVAILABLE.value
    assert attempt.result.unavailable is not None
    assert attempt.result.unavailable.adapter_id == "resource-scheduler"
    assert attempt.result.unavailable.retryable is True
    assert attempt.result.scheduling_success is False
    assert attempt.result.verification_success is False
    assert "capacity" in attempt.result.unavailable.reason_code or any(
        "capacity" in code or "host" in code or "worker" in code
        for code in attempt.result.reason_codes
    )


def test_provider_gateway_absence_returns_typed_unavailable(scheduling_mod) -> None:
    adapter = scheduling_mod.SemanticSchedulingAdapter(
        resource_scheduler=ResourceScheduler(ResourcePolicy(max_lanes=4)),
        host=_ample_host(),
        provider_gateway=None,
    )
    attempt = adapter.schedule(
        _request(WorkKind.MODEL_INVOCATION.value),
        provider_request=_FakeProviderRequest(),
    )
    assert attempt.result.status == SemanticWorkStatus.UNAVAILABLE.value
    assert attempt.result.unavailable is not None
    assert attempt.result.unavailable.adapter_id == "provider-execution-gateway"
    assert attempt.result.unavailable.reason_code == "provider_gateway_absent"
    assert attempt.result.verification_success is False


def test_provider_request_absence_returns_typed_unavailable(scheduling_mod) -> None:
    gateway = FakeProviderGateway()
    adapter = scheduling_mod.SemanticSchedulingAdapter(
        resource_scheduler=ResourceScheduler(ResourcePolicy(max_lanes=4)),
        host=_ample_host(),
        provider_gateway=gateway,
    )
    attempt = adapter.schedule(_request(WorkKind.MODEL_INVOCATION.value))
    assert attempt.result.status == SemanticWorkStatus.UNAVAILABLE.value
    assert attempt.result.unavailable is not None
    assert attempt.result.unavailable.reason_code == "provider_request_absent"
    assert gateway.execute_calls == 0


def test_local_work_succeeds_and_journals(scheduling_mod, tmp_path: Path) -> None:
    event_log = tmp_path / "events.jsonl"
    adapter = scheduling_mod.SemanticSchedulingAdapter(
        resource_scheduler=ResourceScheduler(ResourcePolicy(max_lanes=4)),
        host=_ample_host(),
        event_log_path=event_log,
    )

    def executor(request, *, lease, cancellation, cancel_boundary):
        assert lease.fencing_token >= 1
        assert cancellation.cancelled is False
        return {
            "output_artifact_cids": [_cid("out-1")],
            "diagnostic": "scan complete",
        }

    attempt = adapter.schedule(_request(WorkKind.SCAN.value), work_executor=executor)
    assert attempt.result.status == SemanticWorkStatus.SUCCEEDED.value
    assert attempt.lease is not None
    assert attempt.fence is not None
    assert attempt.result.output_artifact_cids == (_cid("out-1"),)
    assert attempt.result.verification_success is False
    assert event_log.is_file()
    lines = event_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 2
    events = [json.loads(line) for line in lines]
    types = {event["type"] for event in events}
    assert "semantic_work_admitted" in types
    assert "semantic_work_terminal" in types


def test_cancellation_reaches_subprocess_and_provider_boundary(
    scheduling_mod,
) -> None:
    gateway = FakeProviderGateway()
    adapter = scheduling_mod.SemanticSchedulingAdapter(
        resource_scheduler=ResourceScheduler(ResourcePolicy(max_lanes=4)),
        host=_ample_host(),
        provider_gateway=gateway,
    )
    request = _request(WorkKind.MODEL_INVOCATION.value, attempt_id="attempt-cancel")
    provider_request = _FakeProviderRequest(attempt_key="request:cancel#1")

    # Pre-cancel before schedule.
    assert adapter.cancel(
        cancellation_id=request.cancellation_id, reason="user_abort"
    )
    attempt = adapter.schedule(request, provider_request=provider_request)
    assert attempt.result.status == SemanticWorkStatus.CANCELLED.value
    assert "cancelled" in attempt.result.reason_codes
    assert attempt.cancellation.cancelled is True
    assert attempt.cancellation.reason == "user_abort"

    # In-flight style: schedule local work that binds a process then cancel.
    proc = _FakeProcess()
    seen: dict[str, Any] = {}

    def local_with_process(request, *, lease, cancellation, cancel_boundary):
        cancel_boundary.bind_process(proc)
        seen["boundary"] = cancel_boundary
        # Simulate cooperative check after external cancel.
        if cancellation.is_cancelled() or cancel_boundary.cancelled:
            return {"status": "cancelled", "reason": cancellation.reason}
        return {"diagnostic": "should-not-finish"}

    request2 = _request(WorkKind.PYTEST.value, attempt_id="attempt-proc")
    # Start schedule path: cancel after boundary exists via executor.
    def executor_then_cancel(request, *, lease, cancellation, cancel_boundary):
        cancel_boundary.bind_process(proc)
        adapter.cancel(cancellation_id=request.cancellation_id, reason="timeout")
        cancel_boundary.cancel(reason="timeout")
        assert cancel_boundary.cancelled is True
        assert proc.terminate_calls >= 1
        return {"status": "cancelled", "reason": "timeout"}

    attempt2 = adapter.schedule(request2, work_executor=executor_then_cancel)
    assert attempt2.result.status == SemanticWorkStatus.CANCELLED.value
    assert proc.terminate_calls >= 1
    assert attempt2.cancel_boundary.cancelled is True


def test_provider_cancelled_bit_is_propagated(scheduling_mod) -> None:
    gateway = FakeProviderGateway()
    adapter = scheduling_mod.SemanticSchedulingAdapter(
        resource_scheduler=ResourceScheduler(ResourcePolicy(max_lanes=4)),
        host=_ample_host(),
        provider_gateway=gateway,
    )
    request = _request(WorkKind.MODEL_INVOCATION.value, attempt_id="attempt-p-cancel")
    provider_request = _FakeProviderRequest(
        attempt_key="request:p-cancel#1", cancelled=True
    )
    attempt = adapter.schedule(request, provider_request=provider_request)
    assert attempt.result.status == SemanticWorkStatus.CANCELLED.value
    assert gateway.execute_calls == 1
    assert gateway.invoke_calls == 0


def test_replay_does_not_reinvoke_terminal_provider_call(scheduling_mod) -> None:
    gateway = FakeProviderGateway(
        result=_FakeProviderResult(
            phase="settled",
            final_status="committed",
            reason_codes=("ok",),
            observation={"output_artifact_cids": [_cid("model-out")]},
        )
    )
    adapter = scheduling_mod.SemanticSchedulingAdapter(
        resource_scheduler=ResourceScheduler(ResourcePolicy(max_lanes=4)),
        host=_ample_host(),
        provider_gateway=gateway,
    )
    request = _request(
        WorkKind.MODEL_INVOCATION.value,
        attempt_id="attempt-replay",
        provider=_provider(simulated=True),
    )
    provider_request = _FakeProviderRequest(attempt_key="request:replay#1")

    first = adapter.schedule(request, provider_request=provider_request)
    assert first.result.status == SemanticWorkStatus.SIMULATED.value
    assert first.replayed is False
    assert gateway.execute_calls == 1
    assert gateway.invoke_calls == 1
    assert adapter.provider_invoke_count(request) == 1

    second = adapter.schedule(request, provider_request=provider_request)
    assert second.replayed is True
    assert second.result.status == first.result.status
    assert gateway.execute_calls == 1  # adapter did not call gateway again
    assert gateway.invoke_calls == 1
    assert adapter.provider_invoke_count(request) == 1

    third = scheduling_mod.replay_semantic_work(request, adapter=adapter)
    assert third.replayed is True
    assert gateway.execute_calls == 1
    assert gateway.invoke_calls == 1
    assert third.result.verification_success is False


def test_expired_fence_cannot_publish(scheduling_mod) -> None:
    clock = {"now": 1_000_000}

    adapter = scheduling_mod.SemanticSchedulingAdapter(
        resource_scheduler=ResourceScheduler(ResourcePolicy(max_lanes=4)),
        host=_ample_host(),
        clock_ms=lambda: clock["now"],
        fence_ttl_ms=1_000,
    )
    attempt = adapter.schedule(_request(WorkKind.TASK_PARSING.value))
    assert attempt.lease is not None
    token = attempt.lease.fencing_token

    observation = adapter.publish(attempt, fencing_token=token, now_ms=clock["now"])
    assert observation.work_id == attempt.request.work_id
    assert observation.verification_success is False

    # Advance past expiry.
    clock["now"] = attempt.fence.expires_at_ms + 1  # type: ignore[union-attr]
    with pytest.raises(HarnessError, match="expired fence cannot publish"):
        adapter.publish(attempt, fencing_token=token, now_ms=clock["now"])

    # Stale token also denied while fence is live on a fresh attempt.
    clock["now"] = 2_000_000
    attempt2 = adapter.schedule(
        _request(WorkKind.TASK_PARSING.value, attempt_id="attempt-2")
    )
    with pytest.raises(HarnessError, match="stale fencing token cannot publish"):
        adapter.publish(
            attempt2,
            fencing_token=attempt2.lease.fencing_token + 99,  # type: ignore[union-attr]
            now_ms=clock["now"],
        )


def test_force_expire_fence_blocks_publish(scheduling_mod) -> None:
    adapter = scheduling_mod.SemanticSchedulingAdapter(
        resource_scheduler=ResourceScheduler(ResourcePolicy(max_lanes=4)),
        host=_ample_host(),
        fence_ttl_ms=60_000,
    )
    attempt = adapter.schedule(_request(WorkKind.PERSISTENCE.value))
    assert attempt.fence is not None
    expired = adapter.expire_fence(attempt, now_ms=adapter._clock_ms())
    assert expired.is_expired(adapter._clock_ms())
    with pytest.raises(HarnessError, match="expired fence cannot publish"):
        adapter.publish(
            attempt,
            fencing_token=attempt.lease.fencing_token,  # type: ignore[union-attr]
        )


def test_schedule_semantic_work_module_entrypoint(scheduling_mod) -> None:
    attempt = scheduling_mod.schedule_semantic_work(
        _request(WorkKind.CONTEXT_PACKING.value),
        resource_scheduler=ResourceScheduler(ResourcePolicy(max_lanes=2)),
        host=_ample_host(),
    )
    assert isinstance(attempt, scheduling_mod.ScheduledAttempt)
    assert attempt.result.status == SemanticWorkStatus.SUCCEEDED.value
    assert attempt.fencing_token is not None


def test_all_work_kinds_can_be_scheduled(scheduling_mod) -> None:
    adapter = scheduling_mod.SemanticSchedulingAdapter(
        resource_scheduler=ResourceScheduler(ResourcePolicy(max_lanes=8)),
        host=_ample_host(),
        provider_gateway=FakeProviderGateway(
            result=_FakeProviderResult(observation={"output_artifact_cids": []})
        ),
    )
    for kind in WorkKind:
        req = _request(kind.value, attempt_id=f"attempt-{kind.value}")
        kwargs: dict[str, Any] = {}
        if kind is WorkKind.MODEL_INVOCATION:
            kwargs["provider_request"] = _FakeProviderRequest(
                attempt_key=f"request:{kind.value}#1"
            )
        attempt = adapter.schedule(req, **kwargs)
        assert attempt.result.status in {
            SemanticWorkStatus.SUCCEEDED.value,
            SemanticWorkStatus.SIMULATED.value,
        }, kind
        assert attempt.result.verification_success is False


def test_work_executor_unavailable_is_typed(scheduling_mod) -> None:
    adapter = scheduling_mod.SemanticSchedulingAdapter(
        resource_scheduler=ResourceScheduler(ResourcePolicy(max_lanes=4)),
        host=_ample_host(),
    )

    def unavailable_executor(request, *, lease, cancellation, cancel_boundary):
        return {
            "status": "unavailable",
            "reason_code": "tooling_missing",
            "diagnostic": "pytest binary not found",
            "retryable": True,
        }

    attempt = adapter.schedule(
        _request(WorkKind.PYTEST.value), work_executor=unavailable_executor
    )
    assert attempt.result.status == SemanticWorkStatus.UNAVAILABLE.value
    assert attempt.result.unavailable is not None
    assert attempt.result.unavailable.reason_code == "tooling_missing"


def test_provider_capacity_denied_maps_to_unavailable(scheduling_mod) -> None:
    gateway = FakeProviderGateway(
        result=_FakeProviderResult(
            phase="denied",
            final_status="capacity_unavailable",
            success=False,
            reason_codes=("capacity_denied",),
        )
    )
    adapter = scheduling_mod.SemanticSchedulingAdapter(
        resource_scheduler=ResourceScheduler(ResourcePolicy(max_lanes=4)),
        host=_ample_host(),
        provider_gateway=gateway,
    )
    attempt = adapter.schedule(
        _request(
            WorkKind.MODEL_INVOCATION.value,
            mode=HarnessMode.PRODUCTION.value,
            provider=_provider(
                mode=HarnessMode.PRODUCTION.value,
                simulated=False,
                reservation_id="resv:real-9",
            ),
        ),
        provider_request=_FakeProviderRequest(attempt_key="request:cap#1"),
    )
    assert attempt.result.status == SemanticWorkStatus.UNAVAILABLE.value
    assert attempt.result.unavailable is not None
    assert "capacity" in attempt.result.unavailable.reason_code


def test_cancelled_fence_cannot_publish(scheduling_mod) -> None:
    adapter = scheduling_mod.SemanticSchedulingAdapter(
        resource_scheduler=ResourceScheduler(ResourcePolicy(max_lanes=4)),
        host=_ample_host(),
    )
    request = _request(WorkKind.STATIC_CHECK.value, attempt_id="attempt-fence-cancel")
    attempt = adapter.schedule(request)
    assert attempt.lease is not None
    adapter.cancel(cancellation_id=request.cancellation_id, reason="operator_cancel")
    with pytest.raises(HarnessError, match="cancelled fence cannot publish"):
        adapter.publish(attempt, fencing_token=attempt.lease.fencing_token)


def test_scheduled_attempt_to_dict_is_secret_free(scheduling_mod) -> None:
    adapter = scheduling_mod.SemanticSchedulingAdapter(
        resource_scheduler=ResourceScheduler(ResourcePolicy(max_lanes=4)),
        host=_ample_host(),
    )
    attempt = adapter.schedule(_request(WorkKind.CAPSULE_COMPILATION.value))
    payload = attempt.to_dict()
    encoded = json.dumps(payload)
    assert "api_key" not in encoded
    assert "source_body" not in encoded
    assert "prompt" not in encoded
    assert payload["result"]["verification_success"] is False
    assert payload["terminal"] is True
