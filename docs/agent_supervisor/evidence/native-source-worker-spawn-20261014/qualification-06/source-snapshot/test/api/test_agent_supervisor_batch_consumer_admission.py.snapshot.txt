"""Consumer attribution reuses the physical batch owner's real quota charge.

Only authored native batch callbacks execute. The scheduler, receipt producer,
quota adapter, and isolated memory/durable CAS stores are the production owners.
"""

from __future__ import annotations

from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.objectives.goal_development_contracts import (
    GoalDevelopmentMode, GoalDevelopmentPolicy, GoalDevelopmentRequest,
)
from ipfs_accelerate_py.agent_supervisor.planning.task_proposal_router import _call_text_provider
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    ContractValidationError, ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.proof.leanstral_goal_development import (
    GoalDevelopmentContext, GoalDevelopmentTemplate, ImmutableGoalRecord,
    LEANSTRAL_GOAL_DEVELOPMENT_OPERATION, LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_ID,
    LeanstralGoalDevelopmentInvocation, LeanstralGoalDevelopmentProvider,
    LeanstralGoalDevelopmentProviderConfig, build_leanstral_goal_development_batch_dispatch,
    create_leanstral_goal_development_batch_scheduler,
)
from ipfs_accelerate_py.agent_supervisor.rescue.rescue_planner import (
    DEFAULT_RESCUE_ROUTE, RescuePlanner, RescuePlannerPolicy,
)
from ipfs_accelerate_py.agent_supervisor.runtime.batch_provider_usage import (
    BatchProviderUsageBinding, BatchProviderUsageError, BatchUsageAttribution, summarize_batch_usage,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_scheduler import (
    ProviderBatchCapacity, ProviderBatchRequest, ProviderBatchScheduler,
    ProviderBatchSchedulerConfig, ProviderBatchStatus,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_batch_usage import PhysicalBatchUsageAdmission
from ipfs_accelerate_py.agent_supervisor.runtime.provider_execution import ProviderExecutionGateway
from ipfs_accelerate_py.agent_supervisor.runtime.provider_usage_migration import ConsumerId
from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm import LlmRouterInvocation
from ipfs_accelerate_py.endpoint_usage.coordinator import UsageCoordinator
from ipfs_accelerate_py.endpoint_usage.identity import stable_id
from ipfs_accelerate_py.endpoint_usage.schema import (
    EndpointUsageScope, LimitWindow, ProtocolKind, Quantity, UsageDimension,
    UsageLimit, UsageVector, WindowKind,
)
from ipfs_accelerate_py.endpoint_usage.store import (
    DurableUsageLedgerStore, InMemoryUsageLedgerStore, SystemClock,
)


PROVIDER = "authored-native-batch"
MODEL = "authored-model"
PROMPT = "Plan the reviewed goal ∀ n, n = n."
RESPONSE = "authored native batch response"
INPUT_BOUND = 256
OUTPUT_BOUND = 32


@pytest.fixture(autouse=True)
def no_second_dispatch_owner(monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_SUPERVISOR_USAGE_MODE", "off")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("batch consumer must not enter a scalar provider or outer usage gateway")

    monkeypatch.setattr(ProviderExecutionGateway, "execute", forbidden)
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.llm.call_llm_router", forbidden,
    )
    monkeypatch.setattr("ipfs_accelerate_py.llm_router.generate_text", forbidden)


def _limits(scope):
    values = {
        UsageDimension.CONCURRENT_REQUESTS: 1,
        UsageDimension.REQUESTS: 100,
        UsageDimension.BATCH_ITEMS: 100,
        UsageDimension.INPUT_TOKENS: 100_000,
        UsageDimension.OUTPUT_TOKENS: 100_000,
        UsageDimension.TOTAL_TOKENS: 200_000,
    }
    return [UsageLimit(
        scope_id=scope.scope_id, dimension=dimension, ceiling=Quantity.finite(amount),
        window=LimitWindow(kind=(
            WindowKind.CONCURRENT if dimension is UsageDimension.CONCURRENT_REQUESTS
            else WindowKind.LIFETIME
        )),
    ) for dimension, amount in values.items()]


def _goal_invocation():
    policy = GoalDevelopmentPolicy(mode=GoalDevelopmentMode.SHADOW)
    request = GoalDevelopmentRequest(
        root_goal_id="goal:root", root_goal_content_id="cid:reviewed-root",
        satisfaction_formula_id="formula:root", assumption_ids=("assumption:reviewed",),
        evidence_requirement_ids=("evidence:tests",), vocabulary_profile_id="vocabulary:reviewed",
        vocabulary_version=1, repository_tree_id="tree:fixed", scope_ids=("scope:package",),
        policy_digest=policy.policy_digest, mode=policy.mode,
    )
    template = GoalDevelopmentTemplate(
        template_id="template:reviewed", satisfaction_formula_id="formula:subgoal",
        evidence_requirement_ids=("evidence:tests",), assurance_ids=("assurance:typed",),
        resource_ids=("resource:authored",), scope_ids=("scope:package",),
        validation_check_ids=("check:review",),
    )
    return LeanstralGoalDevelopmentInvocation(
        request=request, policy=policy,
        context=GoalDevelopmentContext(
            goal=ImmutableGoalRecord(request.root_goal_id, request.root_goal_content_id,
                                     request.satisfaction_formula_id),
            templates=(template,),
        ),
        resource_budget=ResourceBudget(
            wall_time_ms=2000, model_token_limit=OUTPUT_BOUND, max_output_bytes=4096,
        ),
    )


@pytest.fixture(params=["memory", "durable"])
def owner_factory(request, tmp_path):
    owners = []

    def create(consumer="task", *, blocked=False, capacity_blocked=False):
        clock = SystemClock()
        arguments = {"clock": clock, "writer_id": "batch-consumer-writer", "fence": 1}
        if request.param == "durable":
            store = DurableUsageLedgerStore(tmp_path / ("usage-%s.json" % len(owners)), **arguments)
        else:
            store = InMemoryUsageLedgerStore(**arguments)
        coordinator = UsageCoordinator(store, writer_id="batch-consumer-writer", fence=1)
        scope = EndpointUsageScope(
            provider_id=stable_id("provider", "batch-consumer-fixture"),
            protocol=ProtocolKind.HTTPS, operation="text.generate",
            deployment_id=stable_id("deployment", "authored-native-batch"),
            credential_pseudonym=stable_id("credential", "authored-batch-fixture"),
        )
        coordinator.configure_limits(scope.scope_id, _limits(scope))
        route, operation, consumer_id = {
            "task": ("authored-task-route", "task.proposal.v1", ConsumerId.TASK_PROPOSAL_ROUTER),
            "rescue": (DEFAULT_RESCUE_ROUTE, "rescue_plan.v1", ConsumerId.RESCUE_PLANNER),
            "goal": (LEANSTRAL_GOAL_DEVELOPMENT_PROVIDER_ID,
                     LEANSTRAL_GOAL_DEVELOPMENT_OPERATION, ConsumerId.LEANSTRAL_GOAL_DEVELOPMENT),
        }[consumer]
        started, finish, second_submitted, denied = (threading.Event() for _ in range(4))
        first_submitted = threading.Event()
        capacity_ready = threading.Event()
        if not capacity_blocked:
            capacity_ready.set()
        if not blocked:
            finish.set()
        calls, holds, refusals = [], [], []

        class ObservedAdmission(PhysicalBatchUsageAdmission):
            def __call__(self, *args, **kwargs):
                grant = super().__call__(*args, **kwargs)
                if not grant.admitted:
                    refusals.append(grant.reason)
                    denied.set()
                return grant

        admission = ObservedAdmission(
            coordinator, scope_supplier=lambda _provider: scope.scope_id,
            max_batch_tokens=100_000, reservation_ttl_ms=60_000,
            shared_overhead_tokens=3, retry_initial_ms=20, retry_max_ms=40,
            require_hard_limits=True,
        )

        def dispatch(requests):
            calls.append(tuple(requests))
            holds.append(tuple(record for record in store.read()["reservations"].values()
                               if record["state"] in ("held", "pending") and record["dispatched"]))
            started.set()
            assert finish.wait(5), "authored physical callback was not released"
            return [RESPONSE for _ in requests]

        class ObservedScheduler(ProviderBatchScheduler):
            def __init__(self, *args, **kwargs):
                self.submit_count = 0
                self.submit_lock = threading.Lock()
                super().__init__(*args, **kwargs)

            def submit(self, request):
                future = super().submit(request)
                with self.submit_lock:
                    self.submit_count += 1
                    first_submitted.set()
                    if self.submit_count >= 2:
                        second_submitted.set()
                return future

        scheduler = ObservedScheduler(
            dispatch, admission=admission,
            capacity_supplier=lambda provider: ProviderBatchCapacity(
                provider_id=provider, max_batch_size=8,
                available_concurrent_batches=int(capacity_ready.is_set()),
            ),
            config=ProviderBatchSchedulerConfig(
                max_batch_size=8, batch_window_ms=0, max_parallel_batches=1,
                admission_retry_ms=5,
            ),
        )
        binding = BatchProviderUsageBinding(
            scheduler=scheduler, admission=admission, scope=scope, provider_id=PROVIDER,
            model=MODEL, route=route, operation=operation, fence=1,
            single_physical_request=True,
        )
        owner = SimpleNamespace(
            coordinator=coordinator, store=store, scope=scope, scheduler=scheduler,
            admission=admission, binding=binding, calls=calls, holds=holds, refusals=refusals,
            started=started, finish=finish, second_submitted=second_submitted, denied=denied,
            capacity_ready=capacity_ready,
            first_submitted=first_submitted,
            route=route, operation=operation, consumer_id=consumer_id, consumer=consumer,
        )
        owners.append(owner)
        if consumer == "rescue":
            planner = RescuePlanner(
                RescuePlannerPolicy.permit(provider=PROVIDER, model=MODEL),
                batch_usage_binding=binding,
            )

            def run():
                text = planner._default_provider(
                    PROMPT, repository_root=str(tmp_path), output_tokens=OUTPUT_BOUND,
                    timeout_ms=2000, prompt_token_limit=INPUT_BOUND,
                )
                return text, None
        elif consumer == "goal":
            provider = LeanstralGoalDevelopmentProvider(
                LeanstralGoalDevelopmentProviderConfig(
                    llm_provider=PROVIDER, model=MODEL, timeout_seconds=2,
                    max_new_tokens=OUTPUT_BOUND, max_context_tokens=INPUT_BOUND,
                ),
                batch_usage_binding=binding,
            )
            invocation = _goal_invocation()

            def run():
                result = provider._invoke_shared(
                    PROMPT, invocation, timeout=2, token_budget=OUTPUT_BOUND, cancellation=None,
                )
                return result.output, result
        else:
            invocation = LlmRouterInvocation(
                repo_root=tmp_path, provider=PROVIDER, model_name=MODEL,
                timeout_seconds=2, max_new_tokens=OUTPUT_BOUND,
            )

            def run(*, prompt=PROMPT, output_tokens=OUTPUT_BOUND):
                return _call_text_provider(
                    prompt, replace(invocation, max_new_tokens=output_tokens),
                    scheduler=None, route=route, operation=operation,
                    context_limit=INPUT_BOUND, response_contract="AuthoredProposal/v1",
                    provenance={"task_id": "task:authored"}, batch_usage_binding=binding,
                )
        owner.run = run
        return owner

    yield create
    for owner in reversed(owners):
        owner.finish.set()
        owner.capacity_ready.set()
        owner.scheduler.shutdown(wait=True, cancel_pending=True)
        owner.store.close()


@pytest.mark.parametrize("consumer", ["task", "rescue", "goal"])
@pytest.mark.parametrize("mode", ["off", "enforce"])
def test_actual_consumer_uses_one_physical_owner_and_retains_logical_attribution(
    owner_factory, monkeypatch, consumer, mode,
):
    monkeypatch.setenv("IPFS_ACCELERATE_SUPERVISOR_USAGE_MODE", mode)
    owner = owner_factory(consumer)
    text, result = owner.run()
    assert text == RESPONSE
    assert len(owner.calls) == len(owner.calls[0]) == len(owner.holds[0]) == 1
    assert owner.calls[0][0].usage_scope_id == owner.scope.scope_id
    assert owner.calls[0][0].context_limit >= len(PROMPT.encode("utf-8"))
    assert owner.scheduler.flush(3)
    records = tuple(owner.store.read()["reservations"].values())
    assert len(records) == 1 and records[0]["state"] == "committed"
    assert records[0]["charged_amounts"] == {
        "requests": 1, "batch_items": 1, "input_tokens": INPUT_BOUND + 3,
        "output_tokens": OUTPUT_BOUND, "total_tokens": INPUT_BOUND + 3 + OUTPUT_BOUND,
    }
    attribution = owner.binding.last_attribution(owner.consumer_id)
    assert attribution is not None and attribution.consumer_id == owner.consumer_id
    assert attribution.additional_physical_charge == 0
    assert attribution.usage_state == "confirmed"
    assert attribution.physical_charge_id
    assert attribution.physical_usage["reservation_id"] == records[0]["reservation_id"]
    assert attribution.physical_usage["state"] == "committed"
    assert owner.binding.attribution_for(attribution.logical_request_id) == attribution
    if result is not None:
        assert result.request_id == attribution.logical_request_id
        assert result.status is ProviderBatchStatus.SUCCEEDED
    serialized = json.dumps(attribution.to_dict(), sort_keys=True)
    assert PROMPT not in serialized and RESPONSE not in serialized
    assert "BatchProviderUsageBinding" not in serialized and "UsageCoordinator" not in serialized
    summary = summarize_batch_usage([attribution])
    assert summary["logical_requests"] == summary["physical_calls"] == 1
    assert summary["settled"] == records[0]["charged_amounts"]


def test_late_singleflight_consumers_share_charge_but_keep_distinct_logical_ids(owner_factory):
    owner = owner_factory("task", blocked=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(owner.run)
        try:
            assert owner.started.wait(3)
            second = pool.submit(owner.run)
            assert owner.second_submitted.wait(3)
        finally:
            owner.finish.set()
        first_text, first_result = first.result(timeout=5)
        second_text, second_result = second.result(timeout=5)
    assert first_text == second_text == RESPONSE
    assert first_result.request_id != second_result.request_id
    assert first_result.execution_id == second_result.execution_id
    assert first_result.singleflight_shared and second_result.singleflight_shared
    assert len(owner.calls) == len(owner.calls[0]) == 1
    assert owner.scheduler.flush(3)
    records = tuple(owner.store.read()["reservations"].values())
    assert len(records) == 1 and records[0]["charged_amounts"]["requests"] == 1
    attributions = [owner.binding.attribution_for(result.request_id)
                    for result in (first_result, second_result)]
    assert all(item is not None and item.additional_physical_charge == 0 for item in attributions)
    assert attributions[0].physical_charge_id == attributions[1].physical_charge_id
    late_digest = hashlib.sha256(second_result.request_id.encode("utf-8")).hexdigest()
    assert late_digest not in second_result.physical_usage["reserved_request_digests"]
    summary = summarize_batch_usage(attributions)
    assert summary["logical_requests"] == 2 and summary["physical_calls"] == 1
    assert summary["settled"] == records[0]["charged_amounts"]


def test_distinct_task_prompts_share_one_physical_batch_and_one_aggregate_charge(owner_factory):
    owner = owner_factory("task", capacity_blocked=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(owner.run, prompt="Plan authored task one.")
        second = pool.submit(owner.run, prompt="Plan authored task two.")
        try:
            assert owner.second_submitted.wait(3)
            assert owner.calls == []
        finally:
            owner.capacity_ready.set()
        first_text, first_result = first.result(timeout=5)
        second_text, second_result = second.result(timeout=5)
    assert first_text == second_text == RESPONSE
    assert first_result.execution_id != second_result.execution_id
    assert first_result.batch_id == second_result.batch_id
    assert not first_result.singleflight_shared and not second_result.singleflight_shared
    assert len(owner.calls) == 1 and len(owner.calls[0]) == 2
    assert owner.calls[0][0].batch_key == owner.calls[0][1].batch_key
    assert owner.scheduler.flush(3)
    records = tuple(owner.store.read()["reservations"].values())
    assert len(records) == 1 and records[0]["state"] == "committed"
    expected = {"requests": 1, "batch_items": 2,
                "input_tokens": 2 * INPUT_BOUND + 3, "output_tokens": 2 * OUTPUT_BOUND,
                "total_tokens": 2 * INPUT_BOUND + 3 + 2 * OUTPUT_BOUND}
    assert records[0]["charged_amounts"] == expected
    attributions = [owner.binding.attribution_for(result.request_id)
                    for result in (first_result, second_result)]
    assert all(item.usage_state == "confirmed" for item in attributions)
    assert attributions[0].physical_charge_id == attributions[1].physical_charge_id
    assert attributions[0].physical_usage["dispatched_member_count"] == 2
    summary = summarize_batch_usage(attributions)
    assert summary["logical_requests"] == 2 and summary["physical_calls"] == 1
    assert summary["settled"] == expected


def test_task_output_ceilings_partition_batches_before_physical_dispatch(owner_factory):
    owner = owner_factory("task", capacity_blocked=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(owner.run, output_tokens=16)
        second = pool.submit(owner.run, output_tokens=32)
        try:
            assert owner.second_submitted.wait(3)
            assert owner.calls == []
        finally:
            owner.capacity_ready.set()
        _, first_result = first.result(timeout=5)
        _, second_result = second.result(timeout=5)
    assert owner.scheduler.flush(3)
    assert len(owner.calls) == 2 and all(len(call) == 1 for call in owner.calls)
    assert owner.calls[0][0].batch_key != owner.calls[1][0].batch_key
    assert first_result.batch_id != second_result.batch_id
    records = tuple(owner.store.read()["reservations"].values())
    assert sorted(record["charged_amounts"]["output_tokens"] for record in records) == [16, 32]
    attributions = [owner.binding.attribution_for(result.request_id)
                    for result in (first_result, second_result)]
    summary = summarize_batch_usage(attributions)
    assert summary["logical_requests"] == summary["physical_calls"] == 2
    assert summary["settled"]["requests"] == 2
    assert summary["settled"]["output_tokens"] == 48


def test_cancelled_original_does_not_revoke_late_consumer_or_refund_dispatched_work(owner_factory):
    owner = owner_factory("task", blocked=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(owner.run)
        try:
            assert owner.started.wait(3)
            original_request_id = owner.calls[0][0].request_id
            second = pool.submit(owner.run)
            assert owner.second_submitted.wait(3)
            assert owner.scheduler.cancel(original_request_id)
            assert next(iter(owner.store.read()["reservations"].values()))["state"] == "held"
        finally:
            owner.finish.set()
        with pytest.raises(CancelledError):
            first.result(timeout=5)
        text, result = second.result(timeout=5)
    assert text == RESPONSE and result.successful and result.singleflight_shared
    assert owner.scheduler.flush(3)
    records = tuple(owner.store.read()["reservations"].values())
    assert len(owner.calls) == len(records) == 1
    assert records[0]["state"] == "committed"
    assert records[0]["charged_amounts"]["requests"] == 1
    survivor = owner.binding.attribution_for(result.request_id)
    assert survivor is not None and survivor.usage_state == "confirmed"
    cancelled = owner.binding.attribution_for(original_request_id)
    assert cancelled is None or cancelled.usage_state == "not_observed"


def test_completed_singleflight_group_is_not_a_free_result_cache(owner_factory):
    owner = owner_factory("task")
    _, first = owner.run()
    assert owner.scheduler.flush(3)
    _, second = owner.run()
    assert owner.scheduler.flush(3)
    assert first.execution_id != second.execution_id
    assert len(owner.calls) == 2
    records = tuple(owner.store.read()["reservations"].values())
    assert len(records) == 2
    attributions = [owner.binding.attribution_for(result.request_id) for result in (first, second)]
    assert summarize_batch_usage(attributions)["physical_calls"] == 2
    assert sum(record["charged_amounts"]["requests"] for record in records) == 2


def test_external_quota_pressure_defers_then_recovers_without_outer_reservation(owner_factory):
    owner = owner_factory("task")
    outside = owner.coordinator.reserve(
        owner.scope.scope_id, UsageVector.of(concurrent_requests=1), request_id="external-work",
        idempotency_key="external-work", owner_id="external-owner", ttl_ms=60_000,
    )
    assert outside.granted
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(owner.run)
        try:
            assert owner.denied.wait(3)
            assert owner.calls == []
            assert any(reason.startswith("usage_reservation_denied:") for reason in owner.refusals)
        finally:
            owner.coordinator.cancel(outside.reservation_id)
        text, result = future.result(timeout=5)
    assert text == RESPONSE and result.successful
    assert owner.scheduler.flush(3)
    assert len(owner.calls) == 1
    records = tuple(owner.store.read()["reservations"].values())
    committed = [record for record in records if record["state"] == "committed"]
    assert len(committed) == 1 and committed[0]["charged_amounts"]["requests"] == 1
    assert not any(record["state"] in ("held", "pending") for record in records)
    attribution = owner.binding.attribution_for(result.request_id)
    assert attribution.additional_physical_charge == 0 and attribution.usage_state == "confirmed"


def test_binding_requires_explicit_single_physical_request_declaration(owner_factory):
    owner = owner_factory("task")
    before = owner.store.read()
    with pytest.raises(BatchProviderUsageError):
        BatchProviderUsageBinding(
            scheduler=owner.scheduler, admission=owner.admission, scope=owner.scope,
            provider_id=PROVIDER, model=MODEL, route=owner.route,
            operation=owner.operation, fence=1,
        )
    assert owner.calls == [] and owner.store.read() == before


def test_known_generic_goal_batch_dispatch_cannot_claim_single_physical_request(owner_factory):
    owner = owner_factory("goal")
    calls = []
    generic_dispatch = build_leanstral_goal_development_batch_dispatch(
        LeanstralGoalDevelopmentProviderConfig(llm_provider=PROVIDER, model=MODEL),
        llm_generate_batch=lambda *_a, **_k: calls.append(True) or [],
    )
    before = owner.store.read()
    with ProviderBatchScheduler(generic_dispatch, admission=owner.admission) as scheduler:
        with pytest.raises(BatchProviderUsageError):
            BatchProviderUsageBinding(
                scheduler=scheduler, admission=owner.admission, scope=owner.scope,
                provider_id=PROVIDER, model=MODEL, route=owner.route,
                operation=owner.operation, fence=1, single_physical_request=True,
            )
    assert calls == [] and owner.calls == [] and owner.store.read() == before


def test_generic_goal_factory_rejects_physical_owner_before_constructing_host_owner(
    owner_factory, monkeypatch,
):
    owner = owner_factory("goal")
    calls = []

    def forbidden_host(*_args, **_kwargs):
        raise AssertionError("invalid physical owner must be rejected before host construction")

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.runtime.provider_host_admission.ProviderHostAdmission",
        forbidden_host,
    )
    before = owner.store.read()
    with pytest.raises(ContractValidationError, match="may fan out"):
        create_leanstral_goal_development_batch_scheduler(
            LeanstralGoalDevelopmentProviderConfig(llm_provider=PROVIDER, model=MODEL),
            admission=owner.admission,
            llm_generate_batch=lambda *_args, **_kwargs: calls.append(True) or [],
        )
    assert calls == [] and owner.calls == [] and owner.store.read() == before


def test_provider_registration_cannot_replace_owner_while_bound_consumer_is_queued(owner_factory):
    owner = owner_factory("task", capacity_blocked=True)
    replacement_calls = []
    original = owner.scheduler.dispatch_callback_for(PROVIDER)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(owner.run)
        try:
            assert owner.first_submitted.wait(3)
            assert owner.calls == [] and owner.store.read()["reservations"] == {}
            with pytest.raises(ValueError, match="pinned"):
                owner.scheduler.register_provider(
                    PROVIDER,
                    lambda members: replacement_calls.append(tuple(members)) or [RESPONSE] * len(members),
                )
            owner.scheduler.register_provider(PROVIDER, original)
            owner.scheduler.register_provider("unrelated-provider", lambda members: [])
            assert owner.scheduler.dispatch_callback_for(PROVIDER) is original
        finally:
            owner.capacity_ready.set()
        text, result = future.result(timeout=5)
    assert text == RESPONSE and result.successful
    assert owner.scheduler.flush(3)
    assert replacement_calls == [] and len(owner.calls) == 1
    records = tuple(owner.store.read()["reservations"].values())
    assert len(records) == 1 and records[0]["charged_amounts"]["requests"] == 1
    assert owner.binding.attribution_for(result.request_id).usage_state == "confirmed"


def test_first_dispatch_pin_cannot_retroactively_bind_previously_used_provider():
    calls = []

    def authored_dispatch(members):
        calls.append(tuple(members))
        return [RESPONSE] * len(members)

    with ProviderBatchScheduler(
        authored_dispatch, config=ProviderBatchSchedulerConfig(batch_window_ms=0),
    ) as scheduler:
        result = scheduler.execute(ProviderBatchRequest(
            request_id="prior-unbound-authored-call", payload=PROMPT, provider_id=PROVIDER,
            model=MODEL, route="authored-task-route", operation="task.proposal.v1",
            context_limit=INPUT_BOUND, token_budget=OUTPUT_BOUND, timeout_ms=2000,
        ))
        assert result.successful and scheduler.flush(3)
        with pytest.raises(ValueError, match="before queued or physical work"):
            scheduler.pin_dispatch_callback(PROVIDER, authored_dispatch)
        assert scheduler.dispatch_callback_for(PROVIDER) is authored_dispatch
    assert len(calls) == 1


def test_repeated_valid_attribution_counts_one_logical_request_and_one_charge(owner_factory):
    owner = owner_factory("task")
    _, result = owner.run()
    attribution = owner.binding.attribution_for(result.request_id)
    assert attribution is not None
    summary = summarize_batch_usage([attribution, attribution, replace(attribution)])
    assert summary["logical_requests"] == summary["physical_calls"] == 1
    assert summary["settled"]["requests"] == 1
    assert summary["additional_physical_charge"] == 0
    assert len(owner.calls) == len(owner.store.read()["reservations"]) == 1


def test_replacing_charge_json_cannot_reuse_a_valid_attribution_issuer(owner_factory):
    owner = owner_factory("task")
    _, result = owner.run()
    attribution = owner.binding.attribution_for(result.request_id)
    changed = attribution.physical_usage
    changed["reservation_id"] = "invented-reservation"
    changed["settled"] = UsageVector.of(requests=999, input_tokens=999).to_dict()
    forged = replace(
        attribution, _physical_json=json.dumps(changed, sort_keys=True),
        physical_charge_id="invented",
    )
    before = owner.store.read()
    with pytest.raises(BatchProviderUsageError) as error:
        summarize_batch_usage([forged])
    assert "batch_usage_attribution_invalid" in error.value.reason_codes
    assert summarize_batch_usage([attribution])["settled"]["requests"] == 1
    assert owner.store.read() == before and len(owner.calls) == 1


def test_caller_constructed_or_serialized_attributions_cannot_authorize_aggregation(owner_factory):
    owner = owner_factory("task")
    _, result = owner.run()
    valid = owner.binding.attribution_for(result.request_id)
    unissued = BatchUsageAttribution(
        consumer_id=valid.consumer_id, logical_request_id=valid.logical_request_id,
        execution_fingerprint=valid.execution_fingerprint, status=valid.status,
        provider_id=valid.provider_id, batch_id=valid.batch_id, execution_id=valid.execution_id,
        batch_receipt_id=valid.batch_receipt_id, singleflight_shared=valid.singleflight_shared,
        usage_state=valid.usage_state, physical_charge_id=valid.physical_charge_id,
        _physical_json=json.dumps(valid.physical_usage, sort_keys=True),
    )
    for projection in (unissued, valid.to_dict()):
        with pytest.raises(BatchProviderUsageError) as error:
            summarize_batch_usage([projection])
        assert "batch_usage_attribution_invalid" in error.value.reason_codes
    assert len(owner.calls) == len(owner.store.read()["reservations"]) == 1


def test_physical_usage_property_returns_an_independent_deep_copy(owner_factory):
    owner = owner_factory("task")
    _, result = owner.run()
    attribution = owner.binding.attribution_for(result.request_id)
    original = attribution.to_dict()
    exposed = attribution.physical_usage
    exposed["reservation_id"] = "caller-mutated"
    exposed["settled"]["entries"].clear()
    exposed["reserved_request_digests"].append("caller-mutation")
    assert attribution.to_dict() == original
    assert summarize_batch_usage([attribution])["settled"]["requests"] == 1


def test_native_receipt_is_retained_when_authoritative_validation_requires_reconciliation(
    owner_factory, monkeypatch,
):
    owner = owner_factory("task")

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("authored authoritative receipt validation failure")

    monkeypatch.setattr(owner.admission, "validate_receipt", unavailable)
    with pytest.raises(BatchProviderUsageError) as error:
        owner.run()
    assert "reconciliation_required" in error.value.reason_codes
    assert len(owner.calls) == 1 and owner.scheduler.flush(3)
    records = tuple(owner.store.read()["reservations"].values())
    assert len(records) == 1 and records[0]["state"] == "committed"
    attribution = owner.binding.last_attribution(ConsumerId.TASK_PROPOSAL_ROUTER)
    assert attribution is not None
    assert attribution.usage_state == "reconciliation_required"
    assert attribution.physical_charge_id == ""
    assert attribution.batch_receipt_id == owner.scheduler.evidence_receipts()[-1].evidence_id
    assert attribution.physical_usage["reservation_id"] == records[0]["reservation_id"]
    assert attribution.physical_usage["state"] == "committed"
    summary = summarize_batch_usage([attribution])
    assert summary["logical_requests"] == summary["unobserved_logical_requests"] == 1
    assert summary["physical_calls"] == 0 and summary["settled"] == {}


@pytest.mark.parametrize("changed", ["input_ceiling", "model", "scope"])
def test_request_binding_mismatch_cannot_dispatch_or_reserve(owner_factory, changed):
    owner = owner_factory("task")
    request = ProviderBatchRequest(
        request_id="mismatch", payload=PROMPT, provider_id=PROVIDER, model=MODEL,
        route=owner.route, operation=owner.operation, context_limit=INPUT_BOUND,
        token_budget=OUTPUT_BOUND, timeout_ms=2000,
    )
    if changed == "input_ceiling":
        request = replace(request, context_limit=1)
    elif changed == "model":
        request = replace(request, model="other-model")
    else:
        request = replace(request, usage_scope_id=stable_id("scope", "foreign"))
    before = owner.store.read()
    with pytest.raises(BatchProviderUsageError):
        owner.binding.execute(request, consumer_id=ConsumerId.TASK_PROPOSAL_ROUTER)
    assert owner.calls == [] and owner.store.read() == before


def test_binding_cannot_claim_a_different_configured_fence(owner_factory):
    owner = owner_factory("task")
    before = owner.store.read()
    with pytest.raises(BatchProviderUsageError) as error:
        replace(owner.binding, fence=2)
    assert "batch_usage_fence_mismatch" in error.value.reason_codes
    assert owner.calls == [] and owner.store.read() == before


def test_owner_drift_after_native_settlement_keeps_reconciliation_evidence(owner_factory, monkeypatch):
    owner = owner_factory("task")
    validate = owner.admission.validate_receipt

    def validate_then_change_owner(*args, **kwargs):
        physical = validate(*args, **kwargs)
        monkeypatch.setattr(owner.coordinator, "_fence", 2)
        return physical

    monkeypatch.setattr(owner.admission, "validate_receipt", validate_then_change_owner)
    with pytest.raises(BatchProviderUsageError) as error:
        owner.run()
    assert "reconciliation_required" in error.value.reason_codes
    assert len(owner.calls) == 1 and owner.scheduler.flush(3)
    records = tuple(owner.store.read()["reservations"].values())
    assert len(records) == 1 and records[0]["state"] == "committed"
    attribution = owner.binding.last_attribution(ConsumerId.TASK_PROPOSAL_ROUTER)
    assert attribution is not None and attribution.usage_state == "reconciliation_required"
    assert attribution.physical_charge_id == ""
    assert attribution.physical_usage["reservation_id"] == records[0]["reservation_id"]
