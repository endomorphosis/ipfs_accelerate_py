"""WPD-031: Formal replan + failure memory residual path.

Acceptance (from the sealed WPD board):

* Repeated identical failure triggers backoff.
* Replan edits only bound records (dependency-minimal suffix).
* Residual packet is required for LLM retry; free re-prompt is forbidden.
* Evidence subset: budgets, unchanged-failure backoff, packet seal.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.formal_replanner import (
    DeltaPlan,
    DeltaPlanStep,
    DeltaReplanStopReason,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_failure_memory import (
    BranchFailureKind,
    BranchFailureObservation,
    FailureBackoffPolicy,
    FailureMemoryDisposition,
    FailureMemoryScope,
    PlanFailureMemory,
    TypedBranchFailure,
)
from ipfs_accelerate_py.agent_supervisor.planning.residual_llm_packet import (
    ResidualLlmPacket,
    packet_satisfies_residual_llm_contract,
    seal_residual_llm_packet,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.failure_replan_policy import (
    FAILURE_REPLAN_POLICY_EVIDENCE,
    FAILURE_REPLAN_POLICY_INTERFACE,
    FAILURE_REPLAN_POLICY_VERSION,
    REASON_FREE_REPROMPT_FORBIDDEN,
    REASON_IDENTICAL_FAILURE_EXHAUSTED,
    REASON_REPLAN_ABSTAIN_NO_PACKET,
    REASON_REPLAN_RESIDUAL_AUTHORIZED,
    REASON_RESIDUAL_PACKET_REQUIRED,
    REASON_UNCHANGED_FAILURE_BACKOFF,
    FailureReplanOutcome,
    FailureReplanPolicy,
    FailureReplanPolicyInputError,
    FailureReplanRequest,
    FreeRepromptForbiddenError,
    ResidualPacketMaterials,
    ResidualPacketRequiredError,
    authorize_llm_retry_after_failure,
    build_failure_replan_policy,
    evaluate_failure_replan,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_disposition import (
    ImplementationDisposition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _scope(
    *,
    tree: str = "tree:failure-replan",
    policy: str = "policy:failure-replan-v1",
    environment: str = "environment:linux-py312",
    planner: str = "and-or-planner-v1",
) -> FailureMemoryScope:
    return FailureMemoryScope(
        repository_tree_id=tree,
        policy_revision=policy,
        environment_id=environment,
        planner_version=planner,
    )


def _plan(scope: FailureMemoryScope | None = None) -> DeltaPlan:
    scope = scope or _scope()
    return DeltaPlan(
        scope=scope,
        steps=(
            DeltaPlanStep(
                step_id="step:base",
                branch_id="branch:base",
                accepted=True,
                evidence_ids=("evidence:base",),
            ),
            DeltaPlanStep(
                step_id="step:target",
                branch_id="branch:target",
                dependency_ids=("step:base",),
                accepted=True,
                evidence_ids=("evidence:target",),
                obligation_ids=("obligation:target",),
                alternative_ids=("alternative:target",),
                constraint_ids=("constraint:scope",),
                validation_signature_ids=("validation:pytest-failed",),
                capability_ids=("capability:gpu",),
                conflict_scope_ids=("scope:target",),
                resource_ids=("resource:gpu-memory",),
            ),
            DeltaPlanStep(
                step_id="step:suffix",
                branch_id="branch:suffix",
                dependency_ids=("step:target",),
                accepted=True,
                evidence_ids=("evidence:suffix",),
            ),
            DeltaPlanStep(
                step_id="step:independent",
                branch_id="branch:independent",
                dependency_ids=("step:base",),
                accepted=True,
                evidence_ids=("evidence:independent",),
            ),
        ),
    )


def _observation(
    kind: BranchFailureKind = BranchFailureKind.COUNTEREXAMPLE,
    *,
    scope: FailureMemoryScope | None = None,
    evidence_id: str = "evidence:failure-v1",
    delivery_id: str = "delivery:one",
) -> BranchFailureObservation:
    return BranchFailureObservation(
        features=TypedBranchFailure(
            scope=scope or _scope(),
            kind=kind,
            failure_code=f"failure:{kind.value}",
            branch_id="branch:target",
            step_ids=("step:target",),
            obligation_ids=("obligation:target",),
            alternative_ids=("alternative:target",),
            constraint_ids=("constraint:scope",),
            validation_signature_ids=("validation:pytest-failed",),
            capability_ids=("capability:gpu",),
            conflict_scope_ids=("scope:target",),
            resource_ids=("resource:gpu-memory",),
        ),
        evidence_id=evidence_id,
        delivery_id=delivery_id,
    )


def _capsule(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "schema": "ipfs_accelerate_py/agent-supervisor/counterexample-context-capsule@1",
        "target_ids": ["symbol:target"],
        "counterexamples": [
            {
                "counterexample_id": "cex:wpd-031",
                "kind": "generic_failure",
                "summary": "focused residual repair required",
                "violated_property": "acceptance must hold",
            }
        ],
        "nodes": [],
        "edges": [],
        "usage": {
            "counterexamples": 1,
            "graph_nodes": 0,
            "graph_edges": 0,
            "encoded_bytes": 128,
            "omitted_counterexamples": 0,
        },
        "limits": {"max_bytes": 4096},
        "minimized": True,
        "redacted": True,
        "contains_private_material": False,
        "contains_raw_prover_output": False,
        "contains_source": False,
    }
    base.update(overrides)
    return base


def _materials(**overrides: object) -> ResidualPacketMaterials:
    base: dict[str, object] = {
        "task_id": "WPD-031",
        "repository_id": "repository:sha256:wpd-031",
        "tree_id": "tree:failure-replan",
        "write_paths": (
            "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/"
            "todo_daemon/failure_replan_policy.py",
        ),
        "obligation_ids": ("obligation:target",),
        "counterexample_capsule": _capsule(),
        "validation_commands": (
            "python3 -m pytest external/ipfs_accelerate/test/api/"
            "test_agent_supervisor_failure_replan_residual.py -q",
        ),
        "policy_id": "policy:implementation-daemon",
        "policy_revision": "sha256:policy-wpd-031",
        "forest_id": "forest:wpd-031",
        "acceptance_ids": (FAILURE_REPLAN_POLICY_EVIDENCE,),
        "authority_roots": {
            "repository_id": "repository:sha256:wpd-031",
            "tree_id": "tree:failure-replan",
        },
    }
    base.update(overrides)
    return ResidualPacketMaterials(**base)  # type: ignore[arg-type]


def _sealed_packet(**overrides: object) -> ResidualLlmPacket:
    materials = _materials(**overrides)
    return materials.seal()


def _request(
    *,
    with_materials: bool = False,
    with_packet: bool = False,
    evidence_id: str = "evidence:failure-v1",
    delivery_id: str = "delivery:one",
    observed_at_milliseconds: int = 100,
    **changes: Any,
) -> FailureReplanRequest:
    values: dict[str, Any] = {
        "plan": _plan(),
        "observation": _observation(
            evidence_id=evidence_id, delivery_id=delivery_id
        ),
        "observed_at_milliseconds": observed_at_milliseconds,
    }
    if with_materials:
        values["residual_materials"] = _materials()
    if with_packet:
        values["residual_packet"] = _sealed_packet()
    values.update(changes)
    return FailureReplanRequest(**values)


# ---------------------------------------------------------------------------
# Interface / cold import / discovery
# ---------------------------------------------------------------------------


def test_interface_and_evidence_identity_are_stable() -> None:
    assert FAILURE_REPLAN_POLICY_INTERFACE == "FailureReplanPolicy@1"
    assert FAILURE_REPLAN_POLICY_VERSION == 1
    assert FAILURE_REPLAN_POLICY_EVIDENCE == "wpd/formal-replan-on-failure@1"
    discovery = FailureReplanPolicy.discovery()
    assert discovery["interface"] == FAILURE_REPLAN_POLICY_INTERFACE
    assert discovery["evidence_key"] == FAILURE_REPLAN_POLICY_EVIDENCE
    assert discovery["uses_formal_replanner"] is True
    assert discovery["uses_plan_failure_memory"] is True
    assert discovery["edits_only_bound_records"] is True
    assert discovery["residual_packet_required_for_llm_retry"] is True
    assert discovery["free_reprompt_allowed"] is False
    assert discovery["llm_router_enabled"] is False
    assert discovery["network_access"] is False
    assert discovery["provider_hooks"] == 0
    assert discovery["backoff_on_identical_failure"] is True


def test_cold_import_does_not_load_llm_or_network_clients() -> None:
    llm_import_roots = (
        "openai",
        "anthropic",
        "litellm",
        "groq",
        "together",
        "requests",
        "httpx",
        "aiohttp",
    )
    import importlib

    importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.failure_replan_policy"
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
        failure_replan_policy as mod,
    )

    source = inspect.getsource(mod)
    for root in llm_import_roots:
        pattern = re.compile(
            rf"(?m)^\s*(?:import|from)\s+{re.escape(root)}(?:\.|\s|$)"
        )
        assert pattern.search(source) is None, f"forbidden import of {root!r}"
        assert f'"{root}"' not in source
        assert f"'{root}'" not in source


# ---------------------------------------------------------------------------
# Bound-record replan + residual packet seal
# ---------------------------------------------------------------------------


def test_replan_edits_only_bound_dependent_suffix_and_seals_residual() -> None:
    policy = build_failure_replan_policy()
    result = policy.evaluate(_request(with_materials=True))

    assert result.outcome is FailureReplanOutcome.RESIDUAL_LLM_AUTHORIZED
    assert result.disposition is ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED
    assert result.reason_code == REASON_REPLAN_RESIDUAL_AUTHORIZED
    assert result.should_replan is True
    assert result.edits_only_bound_records is True
    assert result.authorizes_provider is True
    assert result.provider_hook_count == 0
    assert result.free_reprompt_allowed is False
    assert result.residual_packet_required is True
    assert result.residual_packet_sealed is True
    assert result.residual_packet is not None
    assert packet_satisfies_residual_llm_contract(result.residual_packet)

    assert result.invalidated_step_ids == ("step:suffix", "step:target")
    assert set(result.bound_step_ids) == {"step:suffix", "step:target"}
    assert result.preserved_step_ids == ("step:base", "step:independent")
    assert result.delta_decision.stop_reason is DeltaReplanStopReason.REPLAN_REQUIRED
    assert result.delta_decision.direct_failure_step_ids == ("step:target",)

    resulting = {
        item.step_id: item for item in result.delta_decision.resulting_plan.steps
    }
    assert resulting["step:base"].accepted
    assert resulting["step:independent"].accepted
    assert resulting["step:independent"].evidence_ids == ("evidence:independent",)
    assert not resulting["step:target"].accepted
    assert not resulting["step:suffix"].accepted
    assert resulting["step:target"].evidence_ids == ()
    assert resulting["step:suffix"].evidence_ids == ()


def test_replan_without_residual_materials_abstains() -> None:
    result = evaluate_failure_replan(_request(with_materials=False))

    assert result.outcome is FailureReplanOutcome.ABSTAIN_REVIEW
    assert result.disposition is ImplementationDisposition.ABSTAIN_REVIEW
    assert result.reason_code == REASON_REPLAN_ABSTAIN_NO_PACKET
    assert result.should_replan is True
    assert result.edits_only_bound_records is True
    assert result.invalidated_step_ids == ("step:suffix", "step:target")
    assert result.residual_packet is None
    assert result.residual_packet_required is True
    assert result.residual_packet_sealed is False
    assert result.authorizes_provider is False
    assert result.provider_hook_count == 0
    assert "residual_packet_required" in result.notes


def test_presealed_residual_packet_authorizes_llm() -> None:
    packet = _sealed_packet()
    result = evaluate_failure_replan(_request(with_packet=True))

    assert result.outcome is FailureReplanOutcome.RESIDUAL_LLM_AUTHORIZED
    assert result.residual_packet is not None
    assert result.residual_packet.packet_id == packet.packet_id
    assert result.authorizes_provider is True


# ---------------------------------------------------------------------------
# Identical failure → backoff
# ---------------------------------------------------------------------------


def test_repeated_identical_failure_triggers_backoff() -> None:
    memory = PlanFailureMemory(
        policy=FailureBackoffPolicy(
            base_backoff_milliseconds=10,
            max_backoff_milliseconds=40,
            max_identical_failures=4,
            max_records=10,
            max_records_per_branch=5,
        )
    )
    policy = build_failure_replan_policy(failure_memory=memory)

    first = policy.evaluate(
        _request(with_materials=True, observed_at_milliseconds=100)
    )
    assert first.outcome is FailureReplanOutcome.RESIDUAL_LLM_AUTHORIZED
    assert first.should_replan is True
    assert first.backoff_milliseconds == 0

    # Same evidence + different delivery id → identical failure noise.
    second = policy.evaluate(
        _request(
            with_materials=True,
            delivery_id="delivery:transport-redelivery",
            observed_at_milliseconds=101,
        )
    )
    assert second.outcome is FailureReplanOutcome.BACKOFF
    assert second.reason_code == REASON_UNCHANGED_FAILURE_BACKOFF
    assert second.should_backoff is True
    assert second.should_replan is False
    assert second.backoff_milliseconds == 10
    assert second.backoff_attempt == 1
    assert second.disposition is ImplementationDisposition.ABSTAIN_REVIEW
    assert second.authorizes_provider is False
    assert second.residual_packet is None
    assert second.invalidated_step_ids == ()
    assert second.memory_disposition == FailureMemoryDisposition.UNCHANGED_BACKOFF.value
    assert second.delta_decision.stop_reason is (
        DeltaReplanStopReason.UNCHANGED_FAILURE_BACKOFF
    )

    third = policy.evaluate(
        _request(
            with_materials=True,
            delivery_id="delivery:after-restart",
            observed_at_milliseconds=102,
        )
    )
    assert third.outcome is FailureReplanOutcome.BACKOFF
    assert third.backoff_attempt == 2
    assert third.backoff_milliseconds == 20
    assert third.authorizes_provider is False


def test_identical_failure_exhausts_and_blocks_provider() -> None:
    memory = PlanFailureMemory(
        policy=FailureBackoffPolicy(
            base_backoff_milliseconds=2,
            max_backoff_milliseconds=4,
            max_identical_failures=2,
            max_records=4,
            max_records_per_branch=4,
        )
    )
    policy = build_failure_replan_policy(failure_memory=memory)

    assert policy.evaluate(
        _request(with_materials=True, observed_at_milliseconds=1)
    ).should_replan
    backed_off = policy.evaluate(
        _request(with_materials=True, observed_at_milliseconds=2)
    )
    exhausted = policy.evaluate(
        _request(with_materials=True, observed_at_milliseconds=3)
    )

    assert backed_off.outcome is FailureReplanOutcome.BACKOFF
    assert backed_off.backoff_milliseconds == 2
    assert exhausted.outcome is FailureReplanOutcome.EXHAUSTED
    assert exhausted.reason_code == REASON_IDENTICAL_FAILURE_EXHAUSTED
    assert exhausted.authorizes_provider is False
    assert exhausted.disposition is ImplementationDisposition.ABSTAIN_REVIEW


def test_changed_evidence_reopens_bound_suffix() -> None:
    memory = PlanFailureMemory(
        policy=FailureBackoffPolicy(
            base_backoff_milliseconds=10,
            max_backoff_milliseconds=40,
            max_identical_failures=4,
            max_records=10,
            max_records_per_branch=5,
        )
    )
    policy = build_failure_replan_policy(failure_memory=memory)
    first = policy.evaluate(
        _request(with_materials=True, observed_at_milliseconds=100)
    )
    noise = policy.evaluate(
        _request(
            with_materials=True,
            delivery_id="delivery:noise",
            observed_at_milliseconds=101,
        )
    )
    changed = policy.evaluate(
        _request(
            with_materials=True,
            evidence_id="evidence:failure-v2",
            delivery_id="delivery:new-evidence",
            observed_at_milliseconds=103,
        )
    )

    assert first.should_replan
    assert noise.should_backoff
    assert changed.should_replan
    assert changed.invalidated_step_ids == ("step:suffix", "step:target")
    assert changed.edits_only_bound_records is True
    assert changed.outcome is FailureReplanOutcome.RESIDUAL_LLM_AUTHORIZED
    assert changed.backoff_milliseconds == 0


# ---------------------------------------------------------------------------
# Residual packet required for LLM retry / free re-prompt ban
# ---------------------------------------------------------------------------


def test_llm_retry_requires_sealed_residual_packet() -> None:
    result = evaluate_failure_replan(_request(with_materials=False))
    assert result.authorizes_provider is False

    denied = authorize_llm_retry_after_failure(result)
    assert denied.authorized is False
    assert denied.reason_code == REASON_RESIDUAL_PACKET_REQUIRED
    assert denied.disposition is ImplementationDisposition.ABSTAIN_REVIEW
    assert denied.free_reprompt_allowed is False
    assert denied.residual_packet is None

    # Supplying a sealed packet after an active replan authorizes residual LLM.
    packet = _sealed_packet()
    allowed = authorize_llm_retry_after_failure(result, residual_packet=packet)
    assert allowed.authorized is True
    assert allowed.reason_code == REASON_REPLAN_RESIDUAL_AUTHORIZED
    assert allowed.disposition is ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED
    assert allowed.residual_packet is not None
    assert allowed.residual_packet_id
    assert allowed.free_reprompt_allowed is False


def test_llm_retry_denied_during_backoff_even_with_packet() -> None:
    memory = PlanFailureMemory(
        policy=FailureBackoffPolicy(
            base_backoff_milliseconds=5,
            max_backoff_milliseconds=20,
            max_identical_failures=4,
            max_records=8,
            max_records_per_branch=4,
        )
    )
    policy = build_failure_replan_policy(failure_memory=memory)
    policy.evaluate(_request(with_materials=True, observed_at_milliseconds=1))
    backed_off = policy.evaluate(
        _request(with_materials=True, observed_at_milliseconds=2)
    )
    assert backed_off.should_backoff

    auth = policy.authorize_llm_retry(
        backed_off, residual_packet=_sealed_packet()
    )
    assert auth.authorized is False
    assert auth.reason_code == REASON_UNCHANGED_FAILURE_BACKOFF
    assert auth.free_reprompt_allowed is False


def test_free_reprompt_context_rejected_on_request() -> None:
    with pytest.raises(FreeRepromptForbiddenError) as excinfo:
        FailureReplanRequest(
            plan=_plan(),
            observation=_observation(),
            free_reprompt_context={"task_body": "rewrite the entire module"},
        )
    assert excinfo.value.reason_code == REASON_FREE_REPROMPT_FORBIDDEN


def test_free_reprompt_context_rejected_on_llm_retry() -> None:
    result = evaluate_failure_replan(_request(with_materials=True))
    with pytest.raises(FreeRepromptForbiddenError):
        authorize_llm_retry_after_failure(
            result,
            residual_packet=result.residual_packet,
            free_reprompt_context="full task prose re-injection",
        )


def test_source_bodies_and_task_prose_rejected_in_metadata() -> None:
    with pytest.raises(FailureReplanPolicyInputError, match="secrets|source|prose"):
        FailureReplanRequest(
            plan=_plan(),
            observation=_observation(),
            metadata={"source_body": "def evil(): pass"},
        )
    with pytest.raises(FailureReplanPolicyInputError, match="secrets|source|prose"):
        FailureReplanRequest(
            plan=_plan(),
            observation=_observation(),
            metadata={"task_prose": "please rewrite everything"},
        )
    with pytest.raises(FailureReplanPolicyInputError, match="secrets|source|prose"):
        ResidualPacketMaterials.from_dict(
            {
                "task_id": "WPD-031",
                "repository_id": "repository:x",
                "tree_id": "tree:x",
                "write_paths": ["pkg/a.py"],
                "obligation_ids": ["obligation:1"],
                "counterexample_capsule": {
                    "target_ids": ["t"],
                    "source_body": "leak",
                },
                "validation_commands": ["python3 -m pytest -q"],
            }
        )


def test_path_escape_on_write_paths_rejected() -> None:
    with pytest.raises(FailureReplanPolicyInputError, match="relative repository path"):
        ResidualPacketMaterials(
            task_id="WPD-031",
            repository_id="repository:x",
            tree_id="tree:x",
            write_paths=("../outside.py",),
            obligation_ids=("obligation:1",),
            counterexample_capsule=_capsule(),
            validation_commands=("python3 -m pytest -q",),
        )


# ---------------------------------------------------------------------------
# Budgets / unbound / result surface
# ---------------------------------------------------------------------------


def test_unbound_failure_abstains_without_provider() -> None:
    unbound = BranchFailureObservation(
        features=TypedBranchFailure(
            scope=_scope(),
            kind=BranchFailureKind.COUNTEREXAMPLE,
            failure_code="failure:counterexample",
            branch_id="branch:missing",
            step_ids=("step:does-not-exist",),
            obligation_ids=("obligation:missing",),
        ),
        evidence_id="evidence:unbound",
    )
    result = evaluate_failure_replan(
        FailureReplanRequest(
            plan=_plan(),
            observation=unbound,
            residual_materials=_materials(),
            observed_at_milliseconds=1,
        )
    )
    assert result.outcome is FailureReplanOutcome.UNBOUND
    assert result.authorizes_provider is False
    assert result.should_replan is False
    assert result.invalidated_step_ids == ()
    assert result.disposition is ImplementationDisposition.ABSTAIN_REVIEW


def test_result_payload_is_body_free_and_provider_hooks_zero() -> None:
    result = evaluate_failure_replan(_request(with_materials=True))
    payload = result.to_dict()
    assert payload["provider_hook_count"] == 0
    assert payload["free_reprompt_allowed"] is False
    assert payload["edits_only_bound_records"] is True
    assert payload["evidence"] == FAILURE_REPLAN_POLICY_EVIDENCE
    assert "source" not in payload
    assert "prompt" not in payload
    assert "task_body" not in payload
    assert result.content_id
    assert " " not in result.content_id


def test_authorize_from_authorized_result_reuses_sealed_packet() -> None:
    result = evaluate_failure_replan(_request(with_materials=True))
    auth = authorize_llm_retry_after_failure(result)
    assert auth.authorized is True
    assert auth.residual_packet is not None
    assert auth.residual_packet.packet_id == result.residual_packet.packet_id  # type: ignore[union-attr]


def test_mapping_request_round_trip() -> None:
    request = _request(with_materials=True)
    result = evaluate_failure_replan(request.to_dict())
    assert result.outcome is FailureReplanOutcome.RESIDUAL_LLM_AUTHORIZED
    assert result.edits_only_bound_records is True
    assert result.residual_packet_sealed is True


def test_residual_llm_authorized_requires_packet_on_result() -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.failure_replan_policy import (
        FailureReplanResult,
    )

    # Active replan without materials abstains rather than forging authorization.
    first = evaluate_failure_replan(_request(with_materials=False))
    assert first.disposition is ImplementationDisposition.ABSTAIN_REVIEW
    assert first.residual_packet is None

    # Forged residual_llm_authorized without a sealed packet is rejected.
    with pytest.raises(ResidualPacketRequiredError):
        FailureReplanResult(
            outcome=FailureReplanOutcome.RESIDUAL_LLM_AUTHORIZED,
            reason_code=REASON_REPLAN_RESIDUAL_AUTHORIZED,
            disposition=ImplementationDisposition.RESIDUAL_LLM_AUTHORIZED,
            delta_decision=first.delta_decision,
            memory_disposition=first.memory_disposition,
            backoff_milliseconds=0,
            backoff_attempt=0,
            bound_step_ids=first.bound_step_ids,
            invalidated_step_ids=first.invalidated_step_ids,
            preserved_step_ids=first.preserved_step_ids,
            residual_packet=None,
            residual_packet_required=True,
            residual_packet_sealed=False,
        )


def test_seal_materials_produce_valid_residual_packet() -> None:
    packet = _materials().seal()
    assert packet_satisfies_residual_llm_contract(packet)
    # Direct seal helper stays aligned.
    again = seal_residual_llm_packet(
        task_id=packet.task_id,
        repository_id=packet.repository_id,
        tree_id=packet.tree_id,
        write_paths=packet.write_paths,
        obligation_ids=packet.obligation_ids,
        counterexample_capsule=packet.counterexample_capsule,
        validation_commands=packet.validation_commands,
        policy_id=packet.policy_id,
        policy_revision=packet.policy_revision,
        forest_id=packet.forest_id,
        acceptance_ids=packet.acceptance_ids,
        authority_roots=dict(packet.authority_roots or {}),
    )
    assert again.packet_id == packet.packet_id
