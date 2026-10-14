"""Real direct consumer paths with isolated quota owners and authored transports.

The production wrappers, gateway, and durable/in-memory CAS implementations run
normally. Only the final router transport is replaced; no model, subprocess,
solver, or inspected program is invoked.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
    PromptGoalPlannerConfig,
    _default_router,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    DirectoryScanPolicy,
    OutputMode,
    PromptOutputPolicy,
    PromptPlanningPolicy,
    PromptSource,
    PromptWorkflowBudget,
    PromptWorkflowRequest,
    prompt_workflow_cid,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import ResourceBudget
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_provider import (
    ProofProviderError,
    ProviderRequest,
)
from ipfs_accelerate_py.agent_supervisor.proof.leanstral_proof_provider import (
    LEANSTRAL_MODEL_RESOURCE_CLASS,
    LeanstralProofProvider,
    LeanstralProofProviderConfig,
    _default_llm_generate,
    create_leanstral_proof_provider,
)
from ipfs_accelerate_py.agent_supervisor.runtime.direct_provider_usage import DirectProviderUsageBinding
from ipfs_accelerate_py.agent_supervisor.runtime.provider_execution import ProviderExecutionPhase
from ipfs_accelerate_py.agent_supervisor.runtime.provider_usage_migration import (
    ConsumerId,
    ProviderUsageMigrationError,
    clear_last_call_results,
    last_call_result,
)
from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import (
    credential_configuration_pseudonym,
    stable_id,
)
from ipfs_accelerate_py.endpoint_usage.schema import (
    EndpointUsageScope,
    LimitWindow,
    ProtocolKind,
    Quantity,
    UsageDimension,
    UsageLimit,
    WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import (
    DurableUsageLedgerStore,
    InMemoryUsageLedgerStore,
    SystemClock,
)


PROVIDER = "authored-provider"
MODEL = "authored-model"
PROMPT = "Prove the fixed theorem ∀ n, n = n using the reviewed premise."
OUTPUT_TOKENS = 32


def _limits(scope_id, *, requests=10, input_tokens=10_000, output_tokens=10_000):
    return [UsageLimit(
        scope_id=scope_id, dimension=UsageDimension(name), ceiling=Quantity.finite(amount),
        window=LimitWindow(kind=WindowKind.LIFETIME),
    ) for name, amount in {
        "requests": requests, "input_tokens": input_tokens, "output_tokens": output_tokens,
    }.items()]


@pytest.fixture(params=["memory", "durable"])
def owner(request, tmp_path, monkeypatch):
    # A configured trusted binding must stay enforced even under the historical
    # environment default. Tests do not globally change the default mode.
    monkeypatch.setenv("IPFS_ACCELERATE_SUPERVISOR_USAGE_MODE", "off")
    clear_last_call_results()
    clock = SystemClock()
    arguments = {"clock": clock, "writer_id": "direct-test-writer", "fence": 1}
    if request.param == "durable":
        store = DurableUsageLedgerStore(tmp_path / "direct-usage.json", **arguments)
    else:
        store = InMemoryUsageLedgerStore(**arguments)
    coord = UsageCoordinator(store, writer_id="direct-test-writer", fence=1)
    scope = EndpointUsageScope(
        provider_id=stable_id("provider", "direct-authored-provider"),
        protocol=ProtocolKind.HTTPS,
        operation="text.generate",
        deployment_id=stable_id("deployment", "direct-authored-endpoint"),
        model_id=stable_id("model", MODEL),
        credential_pseudonym=credential_configuration_pseudonym(
            "env:DIRECT_AUTHORED_TEST_KEY", key_id="direct-fixture"
        ),
    )
    coord.configure_limits(scope.scope_id, _limits(scope.scope_id))
    binding = DirectProviderUsageBinding(
        coordinator=coord, scope=scope, provider_id=PROVIDER, model=MODEL,
        reservation_ttl_ms=60_000, fence=1,
    )
    yield SimpleNamespace(coord=coord, store=store, scope=scope, binding=binding, root=tmp_path)
    clear_last_call_results()
    store.close()


def _prompt_request(root):
    def cid(name):
        return prompt_workflow_cid({"direct-consumer-fixture": name})

    return PromptWorkflowRequest(
        prompt_source=PromptSource.inline("Plan the reviewed proof task"),
        repository_root=str(root), directory=str(root),
        repository_root_cid=cid("repository"), allowlist_cid=cid("allowlist"),
        scan_policy=DirectoryScanPolicy(policy_id="scan:authored", scanner_version="1"),
        planning_policy=PromptPlanningPolicy(
            policy_id="planning:authored", provider_preferences=(PROVIDER,),
            model_preferences=(MODEL,),
        ),
        output_policy=PromptOutputPolicy(
            policy_id="output:authored", mode=OutputMode.MARKDOWN,
            output_root=str(root), allowed_output_roots=(str(root),),
            markdown_path="plan.md",
        ),
        budget=PromptWorkflowBudget(
            max_prompt_tokens=4096, max_provider_tokens=OUTPUT_TOKENS, max_latency_ms=2000,
        ),
        caller="principal:authored", program_root=cid("program"),
        intent_ir_root=cid("intent"), legal_ir_root=cid("legal"),
        security_ir_root=cid("security"), policy_root=cid("policy"),
    )


def _proof_request():
    return ProviderRequest(
        request_id="direct-proof-request", operation="prove",
        payload={
            "prompt": PROMPT, "obligation_ids": ["reviewed-obligation"],
            "canonical_source_digest": "sha256:reviewed-source",
            "resource_class": LEANSTRAL_MODEL_RESOURCE_CLASS,
        },
        resource_budget=ResourceBudget(
            wall_time_ms=2000, model_token_limit=OUTPUT_TOKENS, max_output_bytes=8192,
        ),
    )


@pytest.fixture(params=["prompt", "proof"])
def consumer(request, owner, monkeypatch):
    calls = []
    holds_at_dispatch = []
    configs = []
    is_prompt = request.param == "prompt"
    consumer_id = ConsumerId.PROMPT_GOAL_PLANNER if is_prompt else ConsumerId.LEANSTRAL_PROOF_PROVIDER

    def observe(prompt, arguments):
        calls.append((prompt, arguments))
        records = owner.store.read()["reservations"].values()
        holds_at_dispatch.append(tuple(record for record in records if record["state"] == "held"))
        return "authored plan" if is_prompt else "by rfl"

    if is_prompt:
        from ipfs_accelerate_py.agent_supervisor.todo_daemon import llm

        def transport(prompt, invocation):
            return observe(prompt, vars(invocation))

        monkeypatch.setattr(llm, "call_llm_router", transport)
    else:
        from ipfs_accelerate_py import llm_router

        def transport(prompt, **kwargs):
            return observe(prompt, kwargs)

        monkeypatch.setattr(llm_router, "generate_text", transport)

    def run(*, provider=PROVIDER, model=MODEL, binding=owner.binding, max_new_tokens=OUTPUT_TOKENS):
        if is_prompt:
            config = PromptGoalPlannerConfig(
                repo_root=owner.root, provider=provider, model=model,
                timeout_seconds=2, max_new_tokens=max_new_tokens, usage_binding=binding,
            )
            configs.append(config)
            return _default_router(PROMPT, _prompt_request(owner.root), config)
        config = LeanstralProofProviderConfig(
            llm_provider=provider, model=model, timeout_seconds=2,
            max_new_tokens=max_new_tokens,
        )
        configs.append(config)
        return LeanstralProofProvider(config, usage_binding=binding).prove(_proof_request())

    return SimpleNamespace(
        run=run, calls=calls, holds=holds_at_dispatch, configs=configs,
        consumer_id=consumer_id, is_prompt=is_prompt,
    )


def test_configured_direct_consumer_uses_real_owner_with_paired_pins_even_when_env_off(owner, consumer):
    initial = owner.coord.admission_snapshot(owner.scope.scope_id)
    output = consumer.run()
    assert len(consumer.calls) == 1
    assert consumer.calls[0][0] == PROMPT
    assert len(consumer.holds[0]) == 1  # Reservation precedes the real dispatch seam.
    hold = consumer.holds[0][0]
    assert hold["scope_id"] == owner.scope.scope_id
    assert hold["admission_revision_at_decision"] == initial.admission_revision
    assert hold["reserved_amounts"] == {
        "requests": 1, "input_tokens": len(PROMPT.encode("utf-8")),
        "output_tokens": OUTPUT_TOKENS,
    }
    result = last_call_result(consumer.consumer_id)
    assert result is not None and result.metered and result.mode.value == "enforce"
    assert result.execution_result.phase is ProviderExecutionPhase.SETTLED
    assert result.usage_receipt.scope.admission_revision == initial.admission_revision
    assert result.usage_receipt.usage_revision.startswith("usnap_")
    assert result.usage_receipt.usage_revision != initial.snapshot.usage_revision
    assert result.usage_receipt.scope.endpoint_scope_id == owner.scope.scope_id
    document = owner.store.read()
    assert len(document["reservations"]) == 1
    settled = next(iter(document["reservations"].values()))
    assert settled["state"] == "committed"
    assert settled["charged_amounts"] == hold["reserved_amounts"]
    assert not any(row["state"] in ("held", "pending") for row in document["reservations"].values())
    if consumer.is_prompt:
        assert output == "authored plan"
    else:
        assert output["draft_text"] == "by rfl"
        assert output["verified"] is False and output["authoritative"] is False


def test_direct_consumer_quota_denial_never_reaches_transport(owner, consumer):
    owner.coord.configure_limits(owner.scope.scope_id, _limits(owner.scope.scope_id, input_tokens=0))
    if consumer.is_prompt:
        with pytest.raises(ProviderUsageMigrationError):
            consumer.run()
    else:
        with pytest.raises(ProofProviderError):
            consumer.run()
    assert consumer.calls == []
    assert owner.store.read()["reservations"] == {}
    result = last_call_result(consumer.consumer_id)
    assert result is not None and result.execution_result.phase is ProviderExecutionPhase.DENIED
    assert "limit_exhausted" in result.reason_codes
    assert result.usage_receipt is None


def test_direct_consumer_observes_new_limits_on_each_call(owner, consumer):
    consumer.run()
    first = last_call_result(consumer.consumer_id)
    owner.coord.configure_limits(owner.scope.scope_id, _limits(owner.scope.scope_id, requests=1))
    if consumer.is_prompt:
        with pytest.raises(ProviderUsageMigrationError):
            consumer.run()
    else:
        with pytest.raises(ProofProviderError):
            consumer.run()
    assert len(consumer.calls) == 1
    assert last_call_result(consumer.consumer_id).execution_result.phase is ProviderExecutionPhase.DENIED
    assert first.execution_result.phase is ProviderExecutionPhase.SETTLED
    assert len(owner.store.read()["reservations"]) == 1


@pytest.mark.parametrize("mismatch", ["provider", "model"])
def test_direct_consumer_route_mismatch_cannot_spend_another_scope(owner, consumer, mismatch):
    before = owner.store.read()
    kwargs = {mismatch: "other-" + mismatch}
    with pytest.raises((ProviderUsageMigrationError, ProofProviderError, ValueError)):
        consumer.run(**kwargs)
    assert consumer.calls == [] and owner.store.read() == before


def test_direct_consumer_requires_every_configured_token_limit(owner, consumer):
    owner.coord.configure_limits(owner.scope.scope_id, _limits(owner.scope.scope_id)[:2])
    before = owner.store.read()
    with pytest.raises((ProviderUsageMigrationError, ProofProviderError)):
        consumer.run()
    assert consumer.calls == [] and owner.store.read() == before


def test_runtime_binding_is_not_sent_to_provider_or_operational_records(owner, consumer):
    output = consumer.run()
    arguments = consumer.calls[0][1]
    assert "usage_binding" not in arguments
    assert "coordinator" not in arguments
    assert "gateway" not in arguments
    assert all(value is not owner.binding and value is not owner.coord for value in arguments.values())
    record = last_call_result(consumer.consumer_id).to_record()
    serialized = json.dumps(record, sort_keys=True)
    assert "DirectProviderUsageBinding" not in serialized
    assert "UsageCoordinator" not in serialized
    assert "usage_binding" not in serialized
    assert PROMPT not in serialized
    if consumer.is_prompt:
        assert "usage_binding=" not in repr(consumer.configs[0])
    else:
        assert "usage_binding" not in consumer.configs[0].to_dict()
        assert "usage_binding" not in json.dumps(output, sort_keys=True)
        assert arguments["disable_model_retry"] is True
        assert arguments["allow_local_fallback"] is False


def test_without_binding_environment_off_keeps_legacy_consumer_behavior(owner, consumer):
    before = owner.store.read()
    consumer.run(binding=None)
    assert len(consumer.calls) == 1
    assert owner.store.read() == before
    assert last_call_result(consumer.consumer_id) is None


def test_prompt_omitted_output_override_uses_bounded_workflow_budget(owner, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import llm

    calls = []
    monkeypatch.setattr(llm, "call_llm_router", lambda prompt, invocation: calls.append(invocation) or "plan")
    config = PromptGoalPlannerConfig(
        repo_root=owner.root, provider=PROVIDER, model=MODEL, timeout_seconds=2,
        max_new_tokens=None, usage_binding=owner.binding,
    )
    assert _default_router(PROMPT, _prompt_request(owner.root), config) == "plan"
    assert len(calls) == 1 and calls[0].max_new_tokens == OUTPUT_TOKENS
    record = next(iter(owner.store.read()["reservations"].values()))
    assert record["charged_amounts"]["output_tokens"] == OUTPUT_TOKENS


def test_prompt_configured_binding_rejects_local_fallback(owner):
    with pytest.raises(ValueError):
        PromptGoalPlannerConfig(
            provider=PROVIDER, model=MODEL, allow_local_fallback=True,
            usage_binding=owner.binding,
        )
    assert owner.store.read()["reservations"] == {}


@pytest.mark.parametrize("override", [
    {"allow_local_fallback": True},
    {"allow_cross_provider_fallback": True},
])
def test_default_proof_transport_rejects_route_expansion_before_callback(owner, monkeypatch, override):
    from ipfs_accelerate_py import llm_router

    calls = []
    monkeypatch.setattr(llm_router, "generate_text", lambda *_a, **_k: calls.append(True) or "by rfl")
    arguments = {
        "provider": PROVIDER, "model_name": MODEL, "timeout": 2,
        "max_new_tokens": OUTPUT_TOKENS, "allow_local_fallback": False,
        "disable_model_retry": True, "usage_binding": owner.binding,
    }
    arguments.update(override)
    with pytest.raises((ProviderUsageMigrationError, ProofProviderError, ValueError)):
        _default_llm_generate(PROMPT, **arguments)
    assert calls == [] and owner.store.read()["reservations"] == {}


def test_default_proof_transport_disables_model_retry_with_a_binding(owner, monkeypatch):
    from ipfs_accelerate_py import llm_router

    calls = []
    monkeypatch.setattr(llm_router, "generate_text", lambda _prompt, **kwargs: calls.append(kwargs) or "by rfl")
    output = _default_llm_generate(
        PROMPT, provider=PROVIDER, model_name=MODEL, timeout=2,
        max_new_tokens=OUTPUT_TOKENS, disable_model_retry=False,
        usage_binding=owner.binding,
    )
    assert output == "by rfl" and len(calls) == 1
    assert calls[0]["disable_model_retry"] is True
    assert calls[0]["allow_cross_provider_fallback"] is False
    assert len(owner.store.read()["reservations"]) == 1


def test_proof_factory_preserves_binding_through_public_prove_path(owner, monkeypatch):
    from ipfs_accelerate_py import llm_router

    calls = []
    monkeypatch.setattr(llm_router, "generate_text", lambda *_a, **_k: calls.append(True) or "by rfl")
    provider = create_leanstral_proof_provider(
        LeanstralProofProviderConfig(
            llm_provider=PROVIDER, model=MODEL, timeout_seconds=2, max_new_tokens=OUTPUT_TOKENS,
        ),
        usage_binding=owner.binding,
    )
    result = provider.prove(_proof_request())
    assert result["draft_text"] == "by rfl"
    assert calls == [True]
    assert last_call_result(ConsumerId.LEANSTRAL_PROOF_PROVIDER).metered
    assert len(owner.store.read()["reservations"]) == 1


def test_proof_custom_generator_cannot_bypass_configured_binding(owner):
    calls = []
    with pytest.raises(ValueError):
        LeanstralProofProvider(
            LeanstralProofProviderConfig(llm_provider=PROVIDER, model=MODEL),
            llm_generate=lambda *_a, **_k: calls.append(True) or "by rfl",
            usage_binding=owner.binding,
        )
    assert calls == [] and owner.store.read()["reservations"] == {}
