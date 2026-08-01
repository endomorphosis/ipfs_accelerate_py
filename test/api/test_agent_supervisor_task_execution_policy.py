"""SCA-167 / SCA-188 SCAEV167ROUTE symbolic-only and bounded provider tests.

Proves objective evidence SCAEV167ROUTE for SCA-G167: deterministic-only tasks
never call providers; model-assisted work is exact-bound Grok then independent
Codex review; context and prompt budgets, quota, admission, and labels are
hard-enforced and receipted.
"""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.contract_packet_provider_router import (
    ProviderQuotaError,
    ProviderQuotaLatch,
    ProviderRole,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.task_execution_policy import (
    SCAEV167ROUTE,
    SCAEV167ROUTE_COVERAGE,
    SCAEV167ROUTE_EVIDENCE,
    ExecutionMode,
    ExecutionReason,
    ExecutionStatus,
    LocalOperationType,
    ProviderExecutable,
    TaskContextMetadata,
    TaskExecutionPolicy,
    TaskExecutionRequest,
    TypedLocalOperation,
    builtin_local_operation_handlers,
)


def _bounds(*, max_bytes: int = 1024, max_tokens: int = 1024):
    return TaskContextMetadata(max_bytes=max_bytes, max_tokens=max_tokens)


def _model_request(**overrides: Any) -> TaskExecutionRequest:
    values = {
        "task_id": "SCA-167-fixture",
        "mode": ExecutionMode.GROK_CODEX,
        "context": {"obligations": ["symbolic-only", "independent-review"]},
        "context_metadata": _bounds(),
        "grok_executable_id": "provider:grok:build-v1",
        "codex_executable_id": "provider:codex:review-v1",
    }
    values.update(overrides)
    return TaskExecutionRequest(**values)


def _providers(events: list[str]):
    def grok(request):
        events.append("grok")
        assert request["role"] == ProviderRole.GROK_IMPLEMENT.value
        assert request["authority"]["proposal_only"] is True
        assert request["authority"]["completion_authoritative"] is False
        return {"patch": "bounded implementation"}

    def codex(request):
        events.append("codex")
        assert events == ["grok", "codex"]
        assert request["role"] == ProviderRole.CODEX_REVIEW.value
        assert request["grok_implementation"] == {
            "patch": "bounded implementation"
        }
        return {"decision": "approve", "findings": []}

    return (
        ProviderExecutable(
            role=ProviderRole.GROK_IMPLEMENT,
            executable_id="provider:grok:build-v1",
            display_label="implementation",
            invoke=grok,
        ),
        ProviderExecutable(
            role=ProviderRole.CODEX_REVIEW,
            executable_id="provider:codex:review-v1",
            display_label="review",
            invoke=codex,
        ),
    )


def test_scaev167route_evidence_term_is_declared_and_receipted() -> None:
    """Exact-text SCAEV167ROUTE markers for objective evidence admission."""

    assert SCAEV167ROUTE == "SCAEV167ROUTE"
    assert SCAEV167ROUTE_EVIDENCE == SCAEV167ROUTE
    assert "deterministic-only-typed-allowlist-zero-provider-calls" in SCAEV167ROUTE_COVERAGE
    assert "proposal-only-admission-no-completion-or-proof-authority" in SCAEV167ROUTE_COVERAGE

    events: list[str] = []
    grok, codex = _providers(events)
    receipt = TaskExecutionPolicy(grok=grok, codex=codex).execute(_model_request())
    payload = receipt.to_dict()

    assert SCAEV167ROUTE in payload["evidence"]["requirement_ids"]
    assert payload["evidence"]["coverage"] == list(SCAEV167ROUTE_COVERAGE)
    assert payload["admission"] == {
        "proposal_only": True,
        "repository_write_allowed": False,
        "completion_authoritative": False,
        "proof_authoritative": False,
        "provider_result_admitted": True,
        "labels_may_select_provider": False,
    }
    assert payload["completion_authoritative"] is False
    assert payload["proof_authoritative"] is False
    assert payload["prompt_usage"]["bytes"] > 0
    assert payload["prompt_usage"]["tokens"] > 0
    assert payload["context_usage"]["byte_limit"] == 1024
    assert all(attempt["admitted"] for attempt in payload["attempts"])


def test_deterministic_only_runs_typed_allowlist_and_records_zero_model_calls() -> None:
    provider_calls: list[str] = []
    grok = ProviderExecutable(
        ProviderRole.GROK_IMPLEMENT,
        "provider:grok:build-v1",
        "same label",
        lambda _request: provider_calls.append("grok") or {},
    )
    codex = ProviderExecutable(
        ProviderRole.CODEX_REVIEW,
        "provider:codex:review-v1",
        "same label",
        lambda _request: provider_calls.append("codex") or {},
    )
    request = TaskExecutionRequest(
        task_id="symbolic:1",
        mode=ExecutionMode.DETERMINISTIC_ONLY,
        context={"facts": [True, True]},
        context_metadata=_bounds(),
        local_operations=(
            TypedLocalOperation(
                LocalOperationType.EXACT_EQUAL,
                {"left": {"a": 1}, "right": {"a": 1}},
            ),
            TypedLocalOperation(
                LocalOperationType.SHA256,
                {"value": {"proof": "symbolic"}},
            ),
        ),
    )

    result = TaskExecutionPolicy(
        local_operation_handlers=builtin_local_operation_handlers(),
        grok=grok,
        codex=codex,
    ).execute(request)

    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.result[0] is True
    assert result.result[1].startswith("sha256:")
    assert provider_calls == []
    assert result.model_call_count == result.provider_call_count == 0
    assert result.provider_result_admitted is False
    assert result.to_dict()["isolation_audit"] == {
        "llm_call_count": 0,
        "model_call_count": 0,
        "provider_call_count": 0,
    }
    assert result.to_dict()["admission"]["provider_result_admitted"] is False
    assert all(attempt.role == "deterministic-local" for attempt in result.attempts)
    assert SCAEV167ROUTE in result.to_dict()["evidence"]["requirement_ids"]


def test_free_form_or_unallowlisted_local_operations_cannot_execute() -> None:
    with pytest.raises(TypeError, match="LocalOperationType"):
        TypedLocalOperation("shell", {"command": "python arbitrary.py"})  # type: ignore[arg-type]

    request = TaskExecutionRequest(
        task_id="symbolic:closed",
        mode=ExecutionMode.DETERMINISTIC_ONLY,
        context={},
        context_metadata=_bounds(),
        local_operations=(
            TypedLocalOperation(LocalOperationType.ALL_TRUE, {"values": [True]}),
        ),
    )
    result = TaskExecutionPolicy(local_operation_handlers={}).execute(request)

    assert result.status is ExecutionStatus.REJECTED
    assert result.reason_code == ExecutionReason.LOCAL_OPERATION_NOT_ALLOWED.value
    assert result.model_call_count == 0
    assert result.attempts[0].invoked is False
    assert result.attempts[0].admitted is False


@pytest.mark.parametrize(
    ("mode", "context", "bounds", "token_counter"),
    [
        (
            ExecutionMode.DETERMINISTIC_ONLY,
            {"payload": "too many bytes"},
            _bounds(max_bytes=5),
            None,
        ),
        (
            ExecutionMode.DETERMINISTIC_ONLY,
            {"small": True},
            _bounds(max_tokens=2),
            lambda _data: 3,
        ),
        (
            ExecutionMode.GROK_CODEX,
            {"payload": "too many bytes for model path"},
            _bounds(max_bytes=8),
            None,
        ),
    ],
)
def test_task_context_metadata_is_a_pre_execution_hard_limit(
    mode, context, bounds, token_counter
) -> None:
    local_calls = 0
    model_calls = 0

    def local(_arguments, _context):
        nonlocal local_calls
        local_calls += 1
        return True

    def model(_request):
        nonlocal model_calls
        model_calls += 1
        return {}

    request = TaskExecutionRequest(
        task_id="bounded",
        mode=mode,
        context=context,
        context_metadata=bounds,
        local_operations=(
            (TypedLocalOperation(LocalOperationType.EXACT_EQUAL, {}),)
            if mode is ExecutionMode.DETERMINISTIC_ONLY
            else ()
        ),
        grok_executable_id="grok:exact",
        codex_executable_id="codex:exact",
    )
    policy = TaskExecutionPolicy(
        local_operation_handlers={LocalOperationType.EXACT_EQUAL: local},
        grok=ProviderExecutable(
            ProviderRole.GROK_IMPLEMENT, "grok:exact", "grok", model
        ),
        codex=ProviderExecutable(
            ProviderRole.CODEX_REVIEW, "codex:exact", "codex", model
        ),
        token_counter=token_counter,
    )

    result = policy.execute(request)

    assert result.status is ExecutionStatus.REJECTED
    assert result.reason_code == ExecutionReason.CONTEXT_LIMIT_EXCEEDED.value
    assert local_calls == model_calls == 0
    assert result.model_call_count == 0
    assert result.attempts == ()
    assert result.to_dict()["admission"]["provider_result_admitted"] is False


def test_grok_implements_before_an_independent_codex_review() -> None:
    events: list[str] = []
    grok, codex = _providers(events)

    result = TaskExecutionPolicy(grok=grok, codex=codex).execute(_model_request())

    assert result.status is ExecutionStatus.SUCCEEDED
    assert events == ["grok", "codex"]
    assert [attempt.role for attempt in result.attempts] == [
        ProviderRole.GROK_IMPLEMENT.value,
        ProviderRole.CODEX_REVIEW.value,
    ]
    assert result.model_call_count == result.provider_call_count == 2
    assert result.grok_implementation == {"patch": "bounded implementation"}
    assert result.codex_review == {"decision": "approve", "findings": ()}
    assert result.provider_result_admitted is True
    assert all(attempt.prompt_bytes > 0 for attempt in result.attempts)
    assert all(attempt.admitted for attempt in result.attempts)


def test_display_labels_never_select_or_swap_executables() -> None:
    calls: list[str] = []
    grok = ProviderExecutable(
        ProviderRole.GROK_IMPLEMENT,
        "provider:grok:build-v1",
        "shared human label",
        lambda _request: calls.append("grok") or {"proposal": 1},
    )
    codex = ProviderExecutable(
        ProviderRole.CODEX_REVIEW,
        "provider:codex:review-v1",
        "shared human label",
        lambda _request: calls.append("codex") or {"review": 1},
    )
    policy = TaskExecutionPolicy(grok=grok, codex=codex)

    mismatch = policy.execute(
        _model_request(grok_executable_id="provider:codex:review-v1")
    )
    assert mismatch.status is ExecutionStatus.REJECTED
    assert (
        mismatch.reason_code == ExecutionReason.EXECUTABLE_BINDING_MISMATCH.value
    )
    assert mismatch.model_call_count == 0
    assert calls == []
    assert mismatch.to_dict()["admission"]["labels_may_select_provider"] is False

    success = policy.execute(_model_request())
    assert success.status is ExecutionStatus.SUCCEEDED
    assert calls == ["grok", "codex"]
    assert [attempt.display_label for attempt in success.attempts] == [
        "shared human label",
        "shared human label",
    ]
    assert [attempt.executable_id for attempt in success.attempts] == [
        "provider:grok:build-v1",
        "provider:codex:review-v1",
    ]


def test_same_executable_or_callback_cannot_claim_independent_review() -> None:
    def shared(_request):
        raise AssertionError("non-independent providers must not execute")

    policy = TaskExecutionPolicy(
        grok=ProviderExecutable(
            ProviderRole.GROK_IMPLEMENT, "model:shared", "grok", shared
        ),
        codex=ProviderExecutable(
            ProviderRole.CODEX_REVIEW, "model:shared", "codex", shared
        ),
    )
    result = policy.execute(
        _model_request(
            grok_executable_id="model:shared",
            codex_executable_id="model:shared",
        )
    )

    assert result.status is ExecutionStatus.REJECTED
    assert result.reason_code == ExecutionReason.PROVIDERS_NOT_INDEPENDENT.value
    assert result.model_call_count == 0


def test_grok_quota_exhaustion_defers_without_codex_or_local_fallback() -> None:
    calls: list[str] = []
    grok, codex = _providers(calls)
    result = TaskExecutionPolicy(
        grok=grok,
        codex=codex,
        grok_quota=ProviderQuotaLatch(remaining_calls=0),
        codex_quota=ProviderQuotaLatch(remaining_calls=4),
        local_operation_handlers=builtin_local_operation_handlers(),
    ).execute(_model_request())

    assert result.status is ExecutionStatus.DEFERRED
    assert result.reason_code == ExecutionReason.GROK_QUOTA_EXHAUSTED.value
    assert calls == []
    assert result.model_call_count == 0
    assert result.result is None
    assert result.provider_result_admitted is False


def test_codex_quota_exhaustion_defers_and_never_promotes_grok_proposal() -> None:
    events: list[str] = []
    grok, codex = _providers(events)
    result = TaskExecutionPolicy(
        grok=grok,
        codex=codex,
        codex_quota=ProviderQuotaLatch(remaining_calls=0),
    ).execute(_model_request())

    assert result.status is ExecutionStatus.DEFERRED
    assert result.reason_code == ExecutionReason.CODEX_QUOTA_EXHAUSTED.value
    assert events == ["grok"]
    assert result.model_call_count == 1
    assert result.grok_implementation == {"patch": "bounded implementation"}
    assert result.codex_review is None
    assert result.result is None
    assert result.provider_result_admitted is False
    assert result.to_dict()["admission"]["provider_result_admitted"] is False


@pytest.mark.parametrize(
    ("failing_role", "exception", "expected_reason", "expected_calls"),
    [
        (
            "grok",
            RuntimeError("offline"),
            ExecutionReason.GROK_FAILED,
            ["grok"],
        ),
        (
            "grok",
            ProviderQuotaError(),
            ExecutionReason.GROK_QUOTA_EXHAUSTED,
            ["grok"],
        ),
        (
            "codex",
            RuntimeError("offline"),
            ExecutionReason.CODEX_FAILED,
            ["grok", "codex"],
        ),
    ],
)
def test_provider_failures_defer_safely(
    failing_role, exception, expected_reason, expected_calls
) -> None:
    events: list[str] = []

    def grok(_request):
        events.append("grok")
        if failing_role == "grok":
            raise exception
        return {"patch": "not independently reviewed yet"}

    def codex(_request):
        events.append("codex")
        raise exception

    result = TaskExecutionPolicy(
        grok=ProviderExecutable(
            ProviderRole.GROK_IMPLEMENT,
            "provider:grok:build-v1",
            "implementation",
            grok,
        ),
        codex=ProviderExecutable(
            ProviderRole.CODEX_REVIEW,
            "provider:codex:review-v1",
            "review",
            codex,
        ),
    ).execute(_model_request())

    assert result.status is ExecutionStatus.DEFERRED
    assert result.reason_code == expected_reason.value
    assert events == expected_calls
    assert result.model_call_count == len(expected_calls)
    assert result.result is None
    assert result.provider_result_admitted is False


def test_provider_reported_quota_latches_and_defers() -> None:
    codex_calls = 0
    grok_quota = ProviderQuotaLatch(remaining_calls=3)

    def grok(_request):
        return {"status": "quota-exhausted", "reason_code": "quota_exhausted"}

    def codex(_request):
        nonlocal codex_calls
        codex_calls += 1
        return {}

    result = TaskExecutionPolicy(
        grok=ProviderExecutable(
            ProviderRole.GROK_IMPLEMENT,
            "provider:grok:build-v1",
            "grok",
            grok,
        ),
        codex=ProviderExecutable(
            ProviderRole.CODEX_REVIEW,
            "provider:codex:review-v1",
            "codex",
            codex,
        ),
        grok_quota=grok_quota,
    ).execute(_model_request())

    assert result.status is ExecutionStatus.DEFERRED
    assert result.reason_code == ExecutionReason.GROK_QUOTA_EXHAUSTED.value
    assert result.model_call_count == 1
    assert codex_calls == 0
    assert grok_quota.exhausted is True


def test_provider_prompt_hard_limits_are_enforced_and_receipted() -> None:
    calls: list[str] = []

    def grok(_request):
        calls.append("grok")
        return {"patch": "should not run"}

    def codex(_request):
        calls.append("codex")
        return {}

    result = TaskExecutionPolicy(
        grok=ProviderExecutable(
            ProviderRole.GROK_IMPLEMENT,
            "provider:grok:build-v1",
            "grok",
            grok,
        ),
        codex=ProviderExecutable(
            ProviderRole.CODEX_REVIEW,
            "provider:codex:review-v1",
            "codex",
            codex,
        ),
        max_context_bytes=64,
        max_context_tokens=4_096,
        token_counter=lambda data: len(data),
    ).execute(
        _model_request(
            context={"payload": "x" * 8},
            context_metadata=_bounds(max_bytes=256, max_tokens=4_096),
        )
    )

    assert result.status is ExecutionStatus.REJECTED
    assert result.reason_code == ExecutionReason.PROVIDER_PROMPT_TOO_LARGE.value
    assert calls == []
    assert result.model_call_count == 0
    assert result.attempts[0].prompt_bytes > 64
    assert result.attempts[0].invoked is False
    assert result.to_dict()["prompt_usage"]["bytes"] == result.attempts[0].prompt_bytes


def test_provider_authority_claims_are_not_admitted() -> None:
    events: list[str] = []

    def grok(_request):
        events.append("grok")
        return {
            "patch": "forged completion",
            "completion_authoritative": True,
        }

    def codex(_request):
        events.append("codex")
        return {"decision": "approve"}

    result = TaskExecutionPolicy(
        grok=ProviderExecutable(
            ProviderRole.GROK_IMPLEMENT,
            "provider:grok:build-v1",
            "grok",
            grok,
        ),
        codex=ProviderExecutable(
            ProviderRole.CODEX_REVIEW,
            "provider:codex:review-v1",
            "codex",
            codex,
        ),
    ).execute(_model_request())

    assert result.status is ExecutionStatus.DEFERRED
    assert result.reason_code == ExecutionReason.PROVIDER_RESULT_NOT_ADMITTED.value
    assert events == ["grok"]
    assert result.result is None
    assert result.provider_result_admitted is False
    assert result.attempts[0].admitted is False


def test_oversized_provider_response_defers_without_admission() -> None:
    def grok(_request):
        return {"blob": "y" * 200}

    def codex(_request):
        raise AssertionError("codex must not run after oversized grok response")

    result = TaskExecutionPolicy(
        grok=ProviderExecutable(
            ProviderRole.GROK_IMPLEMENT,
            "provider:grok:build-v1",
            "grok",
            grok,
        ),
        codex=ProviderExecutable(
            ProviderRole.CODEX_REVIEW,
            "provider:codex:review-v1",
            "codex",
            codex,
        ),
        max_provider_response_bytes=32,
    ).execute(_model_request())

    assert result.status is ExecutionStatus.DEFERRED
    assert result.reason_code == ExecutionReason.PROVIDER_RESPONSE_TOO_LARGE.value
    assert result.model_call_count == 1
    assert result.attempts[0].response_bytes > 32
    assert result.attempts[0].admitted is False
    assert result.result is None
