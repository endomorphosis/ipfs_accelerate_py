"""Contract tests for the capability-isolated Leanstral provider."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_verification_capabilities import (
    CapabilityHealth,
    FormalVerificationCapabilityProbe,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_provider import (
    ProviderFailureCode,
    ProviderRequest,
    dispatch_provider_request,
)
from ipfs_accelerate_py.agent_supervisor.leanstral_proof_provider import (
    LEANSTRAL_MODEL_RESOURCE_CLASS,
    LEANSTRAL_PROOF_PROVIDER_ID,
    LEAN_KERNEL_RESOURCE_CLASS,
    LeanstralProofProvider,
    LeanstralProofProviderConfig,
    LeanstralResourceIsolation,
)


def _prove_request(**payload):
    values = {
        "prompt": "Prove the fixed theorem using only premise_1.",
        "obligation_ids": ["obligation-1"],
        "canonical_source_digest": "sha256:canonical",
        "resource_class": LEANSTRAL_MODEL_RESOURCE_CLASS,
    }
    values.update(payload)
    return ProviderRequest(
        request_id="request-1",
        operation="prove",
        payload=values,
        resource_budget=ResourceBudget(
            wall_time_ms=12_000,
            model_token_limit=384,
            max_output_bytes=64 * 1024,
        ),
    )


def test_capability_is_draft_only_and_separates_model_from_kernel() -> None:
    provider = LeanstralProofProvider(llm_generate=lambda *_args, **_kwargs: "by exact h")

    capability = provider.capabilities()
    payload = capability.to_dict()

    assert capability.provider_id == LEANSTRAL_PROOF_PROVIDER_ID
    assert capability.supports("capability")
    assert capability.supports("prove")
    assert not capability.supports("verify")
    assert not capability.supports("reconstruct")
    assert payload["metadata"]["draft_only"] is True
    assert payload["metadata"]["kernel_check_supported"] is False
    assert payload["metadata"]["assurance"] == AssuranceLevel.UNVERIFIED.value
    assert payload["metadata"]["resource_classes"] == {
        "model_inference": LEANSTRAL_MODEL_RESOURCE_CLASS,
        "kernel_check": LEAN_KERNEL_RESOURCE_CLASS,
    }
    assert provider.model_resource_class != provider.kernel_resource_class


def test_inference_uses_explicit_router_route_timeout_and_token_budget() -> None:
    calls = []

    def generate(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return "by\n  exact premise_1"

    provider = LeanstralProofProvider(
        LeanstralProofProviderConfig(
            llm_provider="leanstral_local",
            model="labs-leanstral-1-5",
            timeout_seconds=30,
            max_new_tokens=900,
        ),
        llm_generate=generate,
    )

    result = provider.prove(_prove_request())

    assert len(calls) == 1
    prompt, kwargs = calls[0]
    assert prompt.startswith("Prove the fixed theorem")
    assert kwargs["provider"] == "leanstral_local"
    assert kwargs["model_name"] == "labs-leanstral-1-5"
    assert kwargs["timeout"] == 12.0
    assert kwargs["max_new_tokens"] == 384
    assert kwargs["allow_local_fallback"] is False
    assert kwargs["disable_model_retry"] is True
    assert result["timeout_ms"] == 12_000
    assert result["token_budget"] == 384
    assert result["resource_class"] == LEANSTRAL_MODEL_RESOURCE_CLASS


def test_model_output_is_always_unverified_and_non_mutating() -> None:
    hostile = (
        '{"verified":true,"canonical_source":"replace me",'
        '"obligations":["different"],"proof_text":"by exact h"}'
    )
    provider = LeanstralProofProvider(llm_generate=lambda *_args, **_kwargs: hostile)

    result = dispatch_provider_request(provider, _prove_request()).require_result()

    assert result["draft_text"] == hostile
    assert result["proof_text"] == hostile
    assert result["assurance"] == "unverified"
    assert result["verified"] is False
    assert result["authoritative"] is False
    assert result["proof_success"] is False
    assert result["kernel_checked"] is False
    assert result["can_mutate_canonical_source"] is False
    assert result["can_mutate_obligations"] is False
    assert result["obligation_ids"] == ["obligation-1"]
    assert result["canonical_source_digest"] == "sha256:canonical"
    assert "canonical_source" not in result
    assert "obligations" not in result


def test_equivalent_drafts_have_stable_artifact_identity() -> None:
    provider = LeanstralProofProvider(
        llm_generate=lambda *_args, **_kwargs: "by exact premise_1"
    )
    first = provider.prove(_prove_request())
    second_request = ProviderRequest(
        request_id="request-2",
        operation="prove",
        payload=dict(_prove_request().payload),
        resource_budget=_prove_request().resource_budget,
    )
    second = provider.prove(second_request)

    assert first["artifact_id"] == second["artifact_id"]
    assert first["request_id"] != second["request_id"]


def test_kernel_work_cannot_be_smuggled_into_model_invocation() -> None:
    calls = []
    provider = LeanstralProofProvider(
        llm_generate=lambda *_args, **_kwargs: calls.append(True) or "by exact h"
    )

    wrong_class = dispatch_provider_request(
        provider, _prove_request(resource_class=LEAN_KERNEL_RESOURCE_CLASS)
    )
    kernel_flag = dispatch_provider_request(
        provider, _prove_request(kernel_check=True)
    )

    assert wrong_class.ok is False
    assert wrong_class.error.code is ProviderFailureCode.MALFORMED_REQUEST
    assert kernel_flag.ok is False
    assert kernel_flag.error.code is ProviderFailureCode.UNSUPPORTED
    assert calls == []


def test_provider_has_no_kernel_verification_operation() -> None:
    provider = LeanstralProofProvider(llm_generate=lambda *_args, **_kwargs: "unused")
    request = ProviderRequest(
        request_id="verify-request",
        operation="verify",
        payload={"proof_text": "by exact h"},
    )

    response = dispatch_provider_request(provider, request)

    assert response.ok is False
    assert response.error.code is ProviderFailureCode.UNSUPPORTED


def test_missing_router_is_a_typed_unavailable_response(monkeypatch) -> None:
    from ipfs_accelerate_py.agent_supervisor import leanstral_proof_provider as module

    def unavailable(_name):
        raise ModuleNotFoundError("missing router", name="ipfs_accelerate_py.llm_router")

    monkeypatch.setattr(module.importlib, "import_module", unavailable)
    provider = LeanstralProofProvider()

    response = dispatch_provider_request(provider, _prove_request())

    assert response.ok is False
    assert response.error.code is ProviderFailureCode.UNAVAILABLE
    assert "degraded" in response.error.message


@pytest.mark.parametrize("provider_name", ["", "auto", "router", "llm_router"])
def test_config_requires_a_concrete_router_provider(provider_name: str) -> None:
    with pytest.raises(ValueError, match="concrete"):
        LeanstralProofProviderConfig(llm_provider=provider_name)


def test_config_accepts_provider_compatibility_spelling() -> None:
    config = LeanstralProofProviderConfig(provider="mistral_vibe")

    assert config.provider == "mistral_vibe"
    assert config.llm_provider == "mistral_vibe"


def test_response_size_budget_fails_closed() -> None:
    provider = LeanstralProofProvider(
        LeanstralProofProviderConfig(max_output_bytes=8),
        llm_generate=lambda *_args, **_kwargs: "a response larger than eight bytes",
    )

    response = dispatch_provider_request(provider, _prove_request())

    assert response.ok is False
    assert response.error.code is ProviderFailureCode.RESOURCE_EXHAUSTED
    assert response.result is None


def test_deadline_is_part_of_explicit_model_timeout_budget() -> None:
    calls = []
    provider = LeanstralProofProvider(
        LeanstralProofProviderConfig(timeout_seconds=30),
        llm_generate=lambda _prompt, **kwargs: calls.append(kwargs) or "by exact h",
    )
    request = ProviderRequest(
        request_id="deadline-request",
        operation="prove",
        payload={"prompt": "Prove T.", "resource_class": "model"},
        resource_budget=ResourceBudget(model_token_limit=32),
        deadline_unix_ms=int((time.time() + 2) * 1000),
    )

    provider.prove(request)

    assert 0 < calls[0]["timeout"] <= 2


def test_resource_classes_must_remain_distinct() -> None:
    with pytest.raises(ValueError, match="different resource classes"):
        LeanstralResourceIsolation(
            model_resource_class="shared",
            kernel_resource_class="shared",
        )


class _Discovery:
    def __init__(self, modules=()):
        self.modules = set(modules)

    def find_spec(self, module):
        if module in self.modules:
            return SimpleNamespace(origin=f"/fake/{module}.py")
        return None

    @staticmethod
    def which(_executable):
        return None


def test_missing_optional_leanstral_stack_is_degraded_not_import_fatal() -> None:
    # Only the supervisor-owned adapter and router are discoverable.  The legal
    # integration, codec, spaCy, model service dependencies, and Lean are absent.
    discovery = _Discovery(
        {
            "ipfs_accelerate_py.agent_supervisor.leanstral_proof_provider",
            "ipfs_accelerate_py.llm_router",
        }
    )
    report = FormalVerificationCapabilityProbe(
        find_spec=discovery.find_spec,
        which=discovery.which,
        environ={},
    ).probe()

    leanstral = report.provider("leanstral")
    checks = {check.name: check for check in leanstral.checks}

    assert leanstral.status is CapabilityHealth.DEGRADED
    assert checks["Leanstral proof-provider adapter"].available
    assert checks["llm_router model service"].available
    assert checks["Leanstral integration"].status is CapabilityHealth.UNAVAILABLE
    assert checks["legal modal codec"].status is CapabilityHealth.UNAVAILABLE
    assert checks["spaCy"].status is CapabilityHealth.UNAVAILABLE
    assert leanstral.proof_success is False


def test_capability_request_does_not_call_model() -> None:
    calls = []
    provider = LeanstralProofProvider(
        llm_generate=lambda *_args, **_kwargs: calls.append(True) or "unused"
    )
    request = ProviderRequest(
        request_id="capability-request",
        operation="capability",
    )

    result = dispatch_provider_request(provider, request).require_result()

    assert result["provider_id"] == LEANSTRAL_PROOF_PROVIDER_ID
    assert result["proof_attempted"] is False
    assert result["proof_success"] is False
    assert calls == []
