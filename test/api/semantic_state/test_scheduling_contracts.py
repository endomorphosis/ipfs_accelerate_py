"""SCH-004 scheduling and execution contract tests."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessError,
    HarnessMode,
    UnavailableResult,
    WorkKind,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.scheduling_contracts import (
    SCHEDULING_CONTRACTS_SCHEMA,
    SEMANTIC_WORK_SCHEDULING_INTERFACE,
    CancellationToken,
    LeaseBinding,
    ProviderBinding,
    ResourceBinding,
    SchedulerObservation,
    SemanticWorkRequest,
    SemanticWorkResult,
    SemanticWorkStatus,
    compute_semantic_work_identity,
    requires_provider,
    resource_class_for_work_kind,
    semantic_work_scheduling_descriptor,
    stage_for_work_kind,
    work_product_is_heuristic,
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _lease(**overrides: object) -> LeaseBinding:
    payload = {
        "attempt_id": "attempt-1",
        "fencing_token": 1,
        "lease_id": "lease-1",
        "logical_epoch": 0,
    }
    payload.update(overrides)
    return LeaseBinding.from_dict(payload)


def _provider(
    *,
    mode: str = HarnessMode.DEVELOPMENT.value,
    simulated: bool = False,
    reservation_id: str | None = None,
) -> ProviderBinding:
    if reservation_id is None:
        reservation_id = "sim:dev-reservation" if simulated else "resv:provider-a-1"
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
    provider: ProviderBinding | None = None,
    **overrides: object,
) -> SemanticWorkRequest:
    if work_kind == WorkKind.MODEL_INVOCATION.value and provider is None:
        provider = _provider(mode=mode, simulated=mode == HarnessMode.DEVELOPMENT.value)
    request = SemanticWorkRequest.build(
        work_kind=work_kind,
        attempt_id="attempt-1",
        repository_id="example/repo",
        mode=mode,
        input_artifact_cids=[_cid("input-a"), _cid("input-b")],
        base_root_cid=_cid("base-root"),
        provider=provider,
    )
    if overrides:
        payload = request.to_dict()
        payload.update(overrides)
        return SemanticWorkRequest.from_dict(payload)
    return request


def _result(
    *,
    status: str = SemanticWorkStatus.SUCCEEDED.value,
    request: SemanticWorkRequest | None = None,
    lease: LeaseBinding | None = None,
    provider: ProviderBinding | None = None,
    unavailable: UnavailableResult | None = None,
    reason_codes: list[str] | None = None,
    simulated: bool = False,
    diagnostic: str = "ok",
    output_artifact_cids: list[str] | None = None,
) -> SemanticWorkResult:
    req = request or _request()
    if status in {
        SemanticWorkStatus.ADMITTED.value,
        SemanticWorkStatus.SUCCEEDED.value,
    }:
        lease = lease if lease is not None else _lease(attempt_id=req.attempt_id)
    if status == SemanticWorkStatus.SIMULATED.value:
        simulated = True
        lease = lease if lease is not None else _lease(attempt_id=req.attempt_id)
    if status == SemanticWorkStatus.CANCELLED.value and reason_codes is None:
        reason_codes = ["cancelled"]
    if status == SemanticWorkStatus.UNAVAILABLE.value and unavailable is None:
        unavailable = UnavailableResult.from_dict(
            {
                "operation": req.work_kind,
                "adapter_id": "resource-scheduler",
                "reason_code": "capacity_exhausted",
                "retryable": True,
                "diagnostic": "no free process slots",
            }
        )
    if reason_codes is None:
        reason_codes = []
    return SemanticWorkResult.from_dict(
        {
            "request": req.to_dict(),
            "status": status,
            "lease": None if lease is None else lease.to_dict(),
            "provider": None if provider is None else provider.to_dict(),
            "unavailable": None if unavailable is None else unavailable.to_dict(),
            "reason_codes": reason_codes,
            "output_artifact_cids": output_artifact_cids or [],
            "diagnostic": diagnostic,
            "simulated": simulated,
        }
    )


def test_work_kinds_cover_all_harness_stages() -> None:
    kinds = {item.value for item in WorkKind}
    assert kinds == {
        "task_parsing",
        "scan",
        "capsule_compilation",
        "test_selection",
        "context_packing",
        "model_invocation",
        "static_check",
        "pytest",
        "prover",
        "persistence",
    }
    for kind in WorkKind:
        assert stage_for_work_kind(kind)
        assert resource_class_for_work_kind(kind)
        assert requires_provider(kind) is (kind is WorkKind.MODEL_INVOCATION)
        assert work_product_is_heuristic(kind) is (kind is WorkKind.MODEL_INVOCATION)


def test_work_identity_is_deterministic_and_idempotent() -> None:
    left = compute_semantic_work_identity(
        work_kind=WorkKind.CAPSULE_COMPILATION,
        repository_id="example/repo",
        attempt_id="attempt-7",
        input_artifact_cids=[_cid("b"), _cid("a")],
        base_root_cid=_cid("root"),
    )
    right = compute_semantic_work_identity(
        work_kind="capsule_compilation",
        repository_id="example/repo",
        attempt_id="attempt-7",
        input_artifact_cids=[_cid("a"), _cid("b")],
        base_root_cid=_cid("root"),
    )
    assert left == right
    assert left.startswith("sch-work:")
    assert left == compute_semantic_work_identity(
        work_kind=WorkKind.CAPSULE_COMPILATION,
        repository_id="example/repo",
        attempt_id="attempt-7",
        input_artifact_cids=[_cid("a"), _cid("b")],
        base_root_cid=_cid("root"),
    )
    different_attempt = compute_semantic_work_identity(
        work_kind=WorkKind.CAPSULE_COMPILATION,
        repository_id="example/repo",
        attempt_id="attempt-8",
        input_artifact_cids=[_cid("a"), _cid("b")],
        base_root_cid=_cid("root"),
    )
    assert different_attempt != left


def test_request_round_trip_is_closed_and_sorted() -> None:
    request = _request()
    again = SemanticWorkRequest.from_dict(json.loads(json.dumps(request.to_dict())))
    assert again == request
    assert again.input_artifact_cids == tuple(
        sorted(again.input_artifact_cids)
    )
    payload = request.to_dict()
    payload["prompt"] = "ignore me"
    with pytest.raises(HarnessError, match="fields must be exactly"):
        SemanticWorkRequest.from_dict(payload)


def test_model_invocation_requires_provider_and_is_heuristic() -> None:
    with pytest.raises(HarnessError, match="requires a provider"):
        SemanticWorkRequest.build(
            work_kind=WorkKind.MODEL_INVOCATION,
            attempt_id="a1",
            repository_id="example/repo",
            mode=HarnessMode.DEVELOPMENT,
        )
    request = _request(WorkKind.MODEL_INVOCATION.value, mode=HarnessMode.DEVELOPMENT.value)
    assert request.is_heuristic_work_product is True
    assert work_product_is_heuristic(request.work_kind) is True
    with pytest.raises(HarnessError, match="must not bind a provider"):
        SemanticWorkRequest.build(
            work_kind=WorkKind.SCAN,
            attempt_id="a1",
            repository_id="example/repo",
            provider=_provider(),
        )


def test_production_rejects_simulated_provider_reservations() -> None:
    with pytest.raises(HarnessError, match="sim:|degraded:"):
        ProviderBinding.from_dict(
            {
                "provider_id": "provider-a",
                "reservation_id": "sim:local",
                "mode": HarnessMode.PRODUCTION.value,
                "simulated": False,
            }
        )
    with pytest.raises(HarnessError, match="cannot be simulated"):
        ProviderBinding.from_dict(
            {
                "provider_id": "provider-a",
                "reservation_id": "resv:real-1",
                "mode": HarnessMode.PRODUCTION.value,
                "simulated": True,
            }
        )
    production = ProviderBinding.from_dict(
        {
            "provider_id": "provider-a",
            "reservation_id": "resv:real-1",
            "mode": HarnessMode.PRODUCTION.value,
            "simulated": False,
        }
    )
    request = _request(
        WorkKind.MODEL_INVOCATION.value,
        mode=HarnessMode.PRODUCTION.value,
        provider=production,
    )
    assert request.provider is not None
    assert request.provider.simulated is False


def test_statuses_distinguish_unavailable_cancelled_failed_simulated() -> None:
    unavailable = _result(status=SemanticWorkStatus.UNAVAILABLE.value)
    cancelled = _result(status=SemanticWorkStatus.CANCELLED.value)
    failed = _result(
        status=SemanticWorkStatus.FAILED.value,
        reason_codes=["execution_error"],
        diagnostic="boom",
    )
    simulated = _result(status=SemanticWorkStatus.SIMULATED.value)
    assert unavailable.status != cancelled.status != failed.status != simulated.status
    assert {
        unavailable.status,
        cancelled.status,
        failed.status,
        simulated.status,
    } == {
        SemanticWorkStatus.UNAVAILABLE.value,
        SemanticWorkStatus.CANCELLED.value,
        SemanticWorkStatus.FAILED.value,
        SemanticWorkStatus.SIMULATED.value,
    }
    assert unavailable.unavailable is not None
    assert cancelled.scheduling_success is False
    assert failed.scheduling_success is False
    assert simulated.simulated is True
    assert simulated.scheduling_success is True


def test_scheduling_success_is_never_verification_success() -> None:
    for status in (
        SemanticWorkStatus.ADMITTED.value,
        SemanticWorkStatus.SUCCEEDED.value,
        SemanticWorkStatus.SIMULATED.value,
        SemanticWorkStatus.FAILED.value,
        SemanticWorkStatus.CANCELLED.value,
        SemanticWorkStatus.UNAVAILABLE.value,
    ):
        result = _result(status=status)
        if status in {
            SemanticWorkStatus.ADMITTED.value,
            SemanticWorkStatus.SUCCEEDED.value,
            SemanticWorkStatus.SIMULATED.value,
        }:
            assert result.scheduling_success is True
        else:
            assert result.scheduling_success is False
        assert result.verification_success is False
        payload = result.to_dict()
        assert payload["scheduling_success"] is result.scheduling_success
        assert payload["verification_success"] is False
        observation = result.as_scheduler_observation()
        assert observation.verification_success is False
        if result.scheduling_success:
            # Even an explicit attempt to claim verification is rejected.
            forged = observation.to_dict()
            forged["verification_success"] = True
            with pytest.raises(HarnessError, match="verification_success"):
                SchedulerObservation.from_dict(forged)


def test_scheduler_observation_rejects_secrets_and_source_bodies() -> None:
    result = _result()
    observation = SchedulerObservation.from_result(result)
    base = observation.to_dict()
    for forbidden in (
        "secret",
        "prompt",
        "source_body",
        "model_output",
        "api_key",
        "raw_source",
        "messages",
    ):
        polluted = dict(base)
        polluted[forbidden] = "not-allowed"
        with pytest.raises(HarnessError, match="forbids secret/source"):
            SchedulerObservation.from_dict(polluted)
    # Closed field set also rejects undeclared operational bodies.
    polluted = dict(base)
    polluted["response_text"] = "model said hello"
    with pytest.raises(HarnessError, match="forbids secret/source"):
        SchedulerObservation.from_dict(polluted)


def test_scheduler_observation_is_bounded() -> None:
    result = _result(
        reason_codes=[f"r{i}" for i in range(8)],
        diagnostic="x" * 20,
    )
    observation = result.as_scheduler_observation()
    assert set(observation.to_dict()) == SchedulerObservation._FIELDS
    assert len(observation.to_dict()) <= 16
    with pytest.raises(HarnessError, match="at most 32"):
        SchedulerObservation.from_dict(
            {
                **observation.to_dict(),
                "reason_codes": [f"code-{i}" for i in range(40)],
            }
        )


def test_result_invariants_for_unavailable_and_cancelled() -> None:
    request = _request()
    with pytest.raises(HarnessError, match="UnavailableResult"):
        SemanticWorkResult.from_dict(
            {
                "request": request.to_dict(),
                "status": SemanticWorkStatus.UNAVAILABLE.value,
                "lease": None,
                "provider": None,
                "unavailable": None,
                "reason_codes": [],
                "output_artifact_cids": [],
                "diagnostic": "missing",
                "simulated": False,
            }
        )
    with pytest.raises(HarnessError, match="cancellation reason"):
        SemanticWorkResult.from_dict(
            {
                "request": request.to_dict(),
                "status": SemanticWorkStatus.CANCELLED.value,
                "lease": None,
                "provider": None,
                "unavailable": None,
                "reason_codes": ["timeout"],
                "output_artifact_cids": [],
                "diagnostic": "stopped",
                "simulated": False,
            }
        )
    with pytest.raises(HarnessError, match="production mode cannot emit simulated"):
        SemanticWorkResult.from_dict(
            {
                "request": _request(
                    WorkKind.SCAN.value, mode=HarnessMode.PRODUCTION.value
                ).to_dict(),
                "status": SemanticWorkStatus.SIMULATED.value,
                "lease": _lease().to_dict(),
                "provider": None,
                "unavailable": None,
                "reason_codes": [],
                "output_artifact_cids": [],
                "diagnostic": "sim",
                "simulated": True,
            }
        )


def test_cancellation_token_is_fenced_and_serializable() -> None:
    token = CancellationToken("cancel:attempt-1")
    assert token.cancelled is False
    assert token.cancel(cancellation_id="wrong-id", reason="nope") is False
    assert token.cancelled is False
    assert token.cancel(cancellation_id="cancel:attempt-1", reason="user_abort") is True
    assert token.cancelled is True
    assert token.reason == "user_abort"
    # First reason wins.
    assert token.cancel(cancellation_id="cancel:attempt-1", reason="later") is True
    assert token.reason == "user_abort"
    with pytest.raises(HarnessError, match="cancelled"):
        token.raise_if_cancelled()
    snap = token.to_dict()
    restored = CancellationToken.from_dict(snap)
    assert restored.cancellation_id == token.cancellation_id
    assert restored.cancelled is True
    assert restored.reason == "user_abort"
    assert "secret" not in snap
    assert "source_body" not in snap


def test_lease_binding_requires_fencing_token() -> None:
    with pytest.raises(HarnessError, match="positive integer"):
        LeaseBinding.from_dict(
            {
                "attempt_id": "a",
                "fencing_token": 0,
                "lease_id": "l",
                "logical_epoch": 0,
            }
        )
    lease = _lease(fencing_token=9)
    assert LeaseBinding.from_dict(lease.to_dict()) == lease


def test_resource_binding_defaults_follow_work_kind() -> None:
    binding = ResourceBinding.for_work_kind(WorkKind.PROVER)
    assert binding.stage == "proof"
    assert binding.resource_class == "cpu-proof-solver"
    inference = ResourceBinding.for_work_kind(WorkKind.MODEL_INVOCATION)
    assert inference.stage == "inference"
    assert inference.resource_class == "llm-proof-draft"


def test_result_round_trip_preserves_closed_records() -> None:
    result = _result(
        status=SemanticWorkStatus.SUCCEEDED.value,
        output_artifact_cids=[_cid("out-b"), _cid("out-a")],
        reason_codes=["admitted", "completed"],
    )
    again = SemanticWorkResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert again.request == result.request
    assert again.status == result.status
    assert again.output_artifact_cids == tuple(sorted(result.output_artifact_cids))
    assert again.scheduling_success is True
    assert again.verification_success is False
    # Unknown operational fields fail closed.
    payload = result.to_dict()
    payload["wall_clock_ms"] = 12
    with pytest.raises(HarnessError, match="fields must be exactly"):
        SemanticWorkResult.from_dict(payload)


def test_interface_descriptor_is_stable() -> None:
    descriptor = semantic_work_scheduling_descriptor()
    assert descriptor["interface"] == SEMANTIC_WORK_SCHEDULING_INTERFACE
    assert descriptor["schema"] == SCHEDULING_CONTRACTS_SCHEMA
    assert "SemanticWorkRequest" in descriptor["records"]
    assert "scheduling_success_is_not_verification_success" in descriptor["invariants"]
    assert set(descriptor["work_kinds"]) == {item.value for item in WorkKind}
    assert set(descriptor["statuses"]) == {item.value for item in SemanticWorkStatus}


def test_forged_cids_fail_closed_on_request_inputs() -> None:
    request = _request()
    payload = request.to_dict()
    payload["input_artifact_cids"] = ["sim:local-model"]
    with pytest.raises(HarnessError):
        SemanticWorkRequest.from_dict(payload)
    payload = request.to_dict()
    payload["base_root_cid"] = "cidv1-sha256-" + "ab" * 32
    with pytest.raises(HarnessError):
        SemanticWorkRequest.from_dict(payload)


def test_diagnostic_and_reason_bounds() -> None:
    request = _request()
    with pytest.raises(HarnessError, match="at most 512"):
        SemanticWorkResult.from_dict(
            {
                "request": request.to_dict(),
                "status": SemanticWorkStatus.FAILED.value,
                "lease": None,
                "provider": None,
                "unavailable": None,
                "reason_codes": ["execution_error"],
                "output_artifact_cids": [],
                "diagnostic": "x" * 513,
                "simulated": False,
            }
        )


def test_ordinary_import_performs_no_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    opened: list[str] = []
    real_open = Path.open

    def tracked_open(self, *args, **kwargs):
        opened.append(str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    importlib.reload(
        importlib.import_module(
            "ipfs_accelerate_py.agent_supervisor.semantic_state.scheduling_contracts"
        )
    )
    assert opened == []
    assert list(tmp_path.iterdir()) == []


def test_build_requests_are_stable_across_input_order() -> None:
    left = SemanticWorkRequest.build(
        work_kind=WorkKind.TEST_SELECTION,
        attempt_id="attempt-9",
        repository_id="example/repo",
        input_artifact_cids=[_cid("z"), _cid("a")],
        base_root_cid=_cid("root"),
    )
    right = SemanticWorkRequest.build(
        work_kind=WorkKind.TEST_SELECTION,
        attempt_id="attempt-9",
        repository_id="example/repo",
        input_artifact_cids=[_cid("a"), _cid("z")],
        base_root_cid=_cid("root"),
    )
    assert left.work_id == right.work_id
    assert left.idempotency_key == right.idempotency_key
    assert left.input_artifact_cids == right.input_artifact_cids
