"""Sealed single-coordinator bootstrap for the CASF first tranche.

This module does not create a second control plane.  It compiles the
operator-reviewed bootstrap profile into the existing closed contracts and
submits them through :class:`FederationControlGateway` and the canonical
``FederationStateRepository``.  The bootstrap profile admits one coordinator
and one *logical* subagent; it does not claim multi-supervisor execution.
It is a local operator bootstrap and is not evidence that the external-agent
trigger or a remote transport has passed live qualification.

The HMAC key accepted here is an ephemeral authentication credential.  It is
never persisted or returned.  Policy permission comes only from the sealed
server-side profile supplied by the operator.
"""

# Python 3.8 compatibility requires ``datetime.timezone.utc``.
# ruff: noqa: UP017

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from ..task_sources.control_plane_contracts import content_identity
from .budgets import AuthoritativeBudgetAuthority
from .contracts import (
    PROGRAM_ID,
    ROOT_OBJECTIVE,
    BudgetDimension,
    BudgetDimensionName,
    FederationBinding,
    FederationContractError,
    FederationIdentity,
    FederationLifecycleState,
    FederationPolicy,
    FederationReceipt,
    FederationRequest,
    ResourceBudget,
    SubagentAssignment,
    SubagentCapability,
    SubagentDefinition,
    SubagentInstance,
    SupervisorAssignment,
    SupervisorCapability,
    SupervisorDefinition,
    SupervisorInstance,
    SupervisorRole,
    TokenBudget,
)
from .events import (
    EventClass,
    EventSelector,
    EventSubscription,
    SelectorKind,
    SubscriptionState,
)
from .trigger import (
    AuthenticationAlgorithm,
    AuthenticationEvidence,
    FederationControlGateway,
    HmacAuthenticationAuthority,
    ResolvedRepository,
)

BOOTSTRAP_PROFILE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/causal-federation-bootstrap-profile@1"
)


def _required_text(profile: Mapping[str, Any], name: str) -> str:
    value = str(profile.get(name) or "").strip()
    if not value or len(value) > 512:
        raise FederationContractError(f"bootstrap profile {name} is invalid")
    return value


def _positive_integer(
    profile: Mapping[str, Any],
    name: str,
    *,
    minimum: int = 1,
    maximum: int = 1_000_000_000,
) -> int:
    value = profile.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FederationContractError(f"bootstrap profile {name} must be integral")
    if not minimum <= value <= maximum:
        raise FederationContractError(f"bootstrap profile {name} is outside its bound")
    return value


def _profile_operations(profile: Mapping[str, Any]) -> tuple[str, ...]:
    raw = profile.get("allowed_operations")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise FederationContractError(
            "bootstrap profile allowed_operations must be an array"
        )
    values = tuple(str(item or "").strip() for item in raw)
    if not values or any(not item for item in values) or len(set(values)) != len(values):
        raise FederationContractError(
            "bootstrap profile allowed_operations is empty or duplicated"
        )
    return values


def validate_bootstrap_profile(profile: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and freeze the exact local bootstrap policy surface."""

    if not isinstance(profile, Mapping):
        raise FederationContractError("bootstrap profile must be an object")
    allowed = {
        "schema",
        "tenant_id",
        "caller_did",
        "issuer_did",
        "audience",
        "policy_ref",
        "policy_revision",
        "requested_supervisor_profile",
        "allowed_operations",
        "allowed_effect",
        "risk_ceiling",
        "expires_at",
        "cpu_millis",
        "processes",
        "input_tokens",
        "output_tokens",
        "model_calls",
        "maximum_supervisors",
        "maximum_subagents",
        "maximum_concurrent_subagents",
    }
    unknown = set(profile) - allowed
    if unknown:
        raise FederationContractError(
            f"bootstrap profile contains unknown fields: {sorted(unknown)}"
        )
    if profile.get("schema") != BOOTSTRAP_PROFILE_SCHEMA:
        raise FederationContractError("bootstrap profile schema differs")
    for name in (
        "tenant_id",
        "caller_did",
        "issuer_did",
        "audience",
        "policy_ref",
        "requested_supervisor_profile",
        "allowed_effect",
        "risk_ceiling",
        "expires_at",
    ):
        _required_text(profile, name)
    _profile_operations(profile)
    for name in (
        "policy_revision",
        "cpu_millis",
        "processes",
        "input_tokens",
        "output_tokens",
        "model_calls",
        "maximum_supervisors",
        "maximum_subagents",
        "maximum_concurrent_subagents",
    ):
        _positive_integer(profile, name)
    if _positive_integer(profile, "maximum_supervisors", maximum=12) != 1:
        raise FederationContractError("first-tranche bootstrap admits one supervisor")
    if _positive_integer(profile, "maximum_subagents", maximum=256) != 1:
        raise FederationContractError("first-tranche bootstrap registers one logical agent")
    if _positive_integer(profile, "maximum_concurrent_subagents", maximum=64) != 1:
        raise FederationContractError("first-tranche bootstrap concurrency must remain one")
    try:
        expiry = datetime.fromisoformat(
            _required_text(profile, "expires_at").replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise FederationContractError("bootstrap profile expiry is invalid") from exc
    if expiry.tzinfo is None:
        raise FederationContractError("bootstrap profile expiry must include a timezone")
    return MappingProxyType(dict(profile))


class _NoDelegationAuthority:
    def verify_chain(
        self,
        request: FederationRequest,
        grant_refs: Sequence[str],
    ) -> tuple[()]:
        del request
        if grant_refs:
            raise FederationContractError(
                "local bootstrap profile does not admit delegated authority"
            )
        return ()


class _StaticPolicyAuthority:
    def __init__(self, policy: FederationPolicy) -> None:
        self._policy = policy

    def get_policy(self, policy_ref: str) -> FederationPolicy:
        if policy_ref != self._policy.record_id:
            raise FederationContractError("bootstrap policy identity differs")
        return self._policy


class _StaticRepositoryAuthority:
    def __init__(self, repository: ResolvedRepository) -> None:
        self._repository = repository

    def resolve(self, repository_refs: Sequence[str]) -> tuple[ResolvedRepository, ...]:
        if tuple(repository_refs) != (self._repository.requested_ref,):
            raise FederationContractError("bootstrap repository reference differs")
        return (self._repository,)


@dataclass(frozen=True)
class BootstrapAdmission:
    """Public, credential-free result of the bounded bootstrap admission."""

    federation_identity: FederationIdentity
    federation_receipt: FederationReceipt
    supervisor: SupervisorInstance
    subagent: SubagentInstance
    subscription: EventSubscription
    fencing_epoch: int

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "causal-federation-bootstrap-admission@1"
            ),
            "federation_id": self.federation_identity.record_id,
            "federation_receipt_id": self.federation_receipt.record_id,
            "supervisor_id": self.supervisor.record_id,
            "supervisor_state": self.supervisor.state,
            "registered_logical_subagents": 1,
            "subagent_id": self.subagent.record_id,
            "active_subagent_processes": 0,
            "subscription_id": self.subscription.subscription_id,
            "consumer_id": self.subscription.consumer_id,
            "fencing_epoch": self.fencing_epoch,
            "multi_supervisor_qualified": False,
            "parallel_execution_qualified": False,
        }


def _dimension(name: BudgetDimensionName, ceiling: int) -> BudgetDimension:
    return BudgetDimension(name=name, ceiling=ceiling, reserved=0, consumed=0)


def _definition(
    contract_type: type[SupervisorDefinition] | type[SubagentDefinition],
    capability_type: type[SupervisorCapability] | type[SubagentCapability],
    *,
    binding: FederationBinding,
    suffix: str,
    operations: tuple[str, ...],
    effect_ceiling: str,
    risk_ceiling: str,
    resource_budget_ref: str,
    token_budget_ref: str,
) -> tuple[
    SupervisorDefinition | SubagentDefinition,
    tuple[SupervisorCapability | SubagentCapability, ...],
]:
    capability = capability_type(
        record_id=f"capability:{suffix}",
        revision=1,
        binding=binding,
        name=f"CASF {suffix} bounded capability",
        capabilities=(),
        allowed_operations=operations,
        effect_ceiling=effect_ceiling,
        risk_ceiling=risk_ceiling,
        resource_budget_ref=resource_budget_ref,
        token_budget_ref=token_budget_ref,
    )
    definition = contract_type(
        record_id=f"definition:{suffix}",
        revision=1,
        binding=binding,
        name=f"CASF {suffix}",
        capabilities=(capability.record_id,),
        allowed_operations=operations,
        effect_ceiling=effect_ceiling,
        risk_ceiling=risk_ceiling,
        resource_budget_ref=resource_budget_ref,
        token_budget_ref=token_budget_ref,
    )
    return definition, (capability,)


def admit_bootstrap_federation(
    repository: Any,
    *,
    profile: Mapping[str, Any],
    repository_id: str,
    repository_tree_id: str,
    plan_root_ref: str,
    operation_catalog_ref: str,
    control_plane_generation: int,
    fencing_epoch: int,
    ready_task_refs: Sequence[str],
    authentication_key: bytes,
    now: datetime | None = None,
) -> BootstrapAdmission:
    """Authenticate and transactionally admit one bounded CASF coordinator."""

    sealed = validate_bootstrap_profile(profile)
    if not isinstance(authentication_key, bytes) or len(authentication_key) < 16:
        raise FederationContractError(
            "bootstrap authentication credential is unavailable"
        )
    observed = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    issued_at = observed.isoformat().replace("+00:00", "Z")
    expiry = _required_text(sealed, "expires_at")
    if observed.timestamp() >= datetime.fromisoformat(
        expiry.replace("Z", "+00:00")
    ).timestamp():
        raise FederationContractError("bootstrap policy has expired")
    tasks = tuple(str(item or "").strip() for item in ready_task_refs)
    if not tasks or any(not item for item in tasks) or len(tasks) > 44:
        raise FederationContractError("bootstrap ready task references are invalid")
    plan_root = str(plan_root_ref or "").strip()
    if not plan_root or len(plan_root) > 512:
        raise FederationContractError("bootstrap plan root is invalid")
    effect = _required_text(sealed, "allowed_effect")
    risk = _required_text(sealed, "risk_ceiling")
    operations = _profile_operations(sealed)
    supervisor_operations = tuple(
        item for item in operations if item != "federation.create"
    )
    if not supervisor_operations:
        raise FederationContractError(
            "bootstrap policy admits no bounded supervisor operations"
        )
    subagent_operations = tuple(
        item for item in supervisor_operations if item == "event.wait"
    )
    if not subagent_operations:
        raise FederationContractError(
            "bootstrap policy admits no bounded logical-agent operation"
        )
    semantic_root = "semantic-state:unavailable:" + repository_tree_id
    authentication_evidence_ref = "authentication:casf-local-bootstrap:" + content_identity(
        {
            "caller_did": sealed["caller_did"],
            "repository_id": repository_id,
            "repository_tree_id": repository_tree_id,
            "plan_root_ref": plan_root,
            "policy_ref": sealed["policy_ref"],
        }
    )
    binding = FederationBinding(
        tenant_id=_required_text(sealed, "tenant_id"),
        repository_ids=(repository_id,),
        repository_tree_ids=(repository_tree_id,),
        program_id=PROGRAM_ID,
        objective_ref=ROOT_OBJECTIVE,
        objective_revision=1,
        policy_ref=_required_text(sealed, "policy_ref"),
        policy_revision=_positive_integer(sealed, "policy_revision"),
        operation_catalog_ref=operation_catalog_ref,
        control_plane_generation=control_plane_generation,
        causal_graph_revision=0,
        semantic_state_roots=(semantic_root,),
        supervisor_population=0,
        budget_ref="budget:federation:casf-bootstrap-v1",
        expires_at=expiry,
        issuer=_required_text(sealed, "issuer_did"),
        authorization_evidence_ref=authentication_evidence_ref,
    )
    resource_budget = ResourceBudget(
        record_id="budget:resource:casf-bootstrap-v1",
        revision=1,
        binding=binding,
        parent_budget_id=binding.budget_ref,
        owner_id="federation:casf-bootstrap-pending",
        dimensions=(
            _dimension(
                BudgetDimensionName.CPU_MILLIS,
                _positive_integer(sealed, "cpu_millis"),
            ),
            _dimension(
                BudgetDimensionName.PROCESSES,
                _positive_integer(sealed, "processes"),
            ),
        ),
        status="requested",
    )
    token_budget = TokenBudget(
        record_id="budget:token:casf-bootstrap-v1",
        revision=1,
        binding=binding,
        parent_budget_id=binding.budget_ref,
        owner_id="federation:casf-bootstrap-pending",
        dimensions=(
            _dimension(
                BudgetDimensionName.INPUT_TOKENS,
                _positive_integer(sealed, "input_tokens"),
            ),
            _dimension(
                BudgetDimensionName.OUTPUT_TOKENS,
                _positive_integer(sealed, "output_tokens"),
            ),
            _dimension(
                BudgetDimensionName.MODEL_CALLS,
                _positive_integer(sealed, "model_calls"),
            ),
        ),
        status="requested",
    )
    request = FederationRequest(
        caller_did=_required_text(sealed, "caller_did"),
        delegation_chain=(),
        audience=_required_text(sealed, "audience"),
        program_id=PROGRAM_ID,
        repository_roots=binding.repository_ids,
        objective_ref=ROOT_OBJECTIVE,
        requested_supervisor_profile=_required_text(
            sealed, "requested_supervisor_profile"
        ),
        maximum_supervisors=_positive_integer(sealed, "maximum_supervisors"),
        maximum_subagents=_positive_integer(sealed, "maximum_subagents"),
        resource_budget=resource_budget,
        token_budget=token_budget,
        effect_scope=(effect,),
        policy_ref=binding.policy_ref,
        expiry=expiry,
        nonce="nonce:casf-bootstrap:" + repository_tree_id[:24],
        idempotency_key="idempotency:casf-bootstrap:" + repository_tree_id[:24],
        binding=binding,
    )
    policy = FederationPolicy(
        record_id=binding.policy_ref,
        revision=binding.policy_revision,
        binding=binding,
        allowed_callers=(request.caller_did,),
        allowed_audiences=(request.audience,),
        allowed_operations=operations,
        allowed_effects=(effect,),
        maximum_supervisors=request.maximum_supervisors,
        maximum_subagents=request.maximum_subagents,
        maximum_concurrent_subagents=_positive_integer(
            sealed, "maximum_concurrent_subagents"
        ),
        conservative_abstraction_scheduling=False,
    )
    key_handle = "handle:casf-bootstrap-ephemeral"
    authenticator = HmacAuthenticationAuthority(
        lambda caller, handle: (
            authentication_key
            if caller == request.caller_did and handle == key_handle
            else b""
        ),
        now=lambda: observed.timestamp(),
    )
    evidence = AuthenticationEvidence(
        evidence_id=authentication_evidence_ref,
        caller_did=request.caller_did,
        algorithm=AuthenticationAlgorithm.HMAC_SHA256,
        key_handle=key_handle,
        request_cid=request.cid,
        audience=request.audience,
        nonce=request.nonce,
        issued_at=issued_at,
        expires_at=request.expiry,
        signature=HmacAuthenticationAuthority.sign_request(
            request, authentication_key
        ),
    )
    resolved = ResolvedRepository(
        requested_ref=repository_id,
        repository_id=repository_id,
        tree_id=repository_tree_id,
        semantic_state_root=semantic_root,
    )
    capacity = {
        item.name: item.ceiling
        for budget in (resource_budget, token_budget)
        for item in budget.dimensions
    }
    budget_authority = AuthoritativeBudgetAuthority(
        repository,
        capacity=capacity,
        authority_id="authority:casf-bootstrap-budget-v1",
        now=lambda: observed,
    )
    gateway = FederationControlGateway(
        audience=request.audience,
        authenticator=authenticator,
        delegations=_NoDelegationAuthority(),
        policies=_StaticPolicyAuthority(policy),
        repositories=_StaticRepositoryAuthority(resolved),
        budgets=budget_authority,
        store=repository,
        now=lambda: observed.timestamp(),
    )
    federation_identity, federation_receipt = gateway.create(request, evidence)

    supervisor_suffix = "casf-coordinator-" + repository_tree_id[:16]
    supervisor_id = "supervisor:" + supervisor_suffix
    supervisor = SupervisorInstance(
        record_id=supervisor_id,
        revision=1,
        binding=binding,
        state=FederationLifecycleState.ADMITTED.value,
        federation_id=federation_identity.record_id,
        parent_supervisor_id="",
        role=SupervisorRole.COORDINATOR,
        lease_id="lease:" + supervisor_suffix,
        fencing_epoch=fencing_epoch,
    )
    supervisor_assignment = SupervisorAssignment(
        record_id="assignment:" + supervisor_suffix,
        revision=1,
        binding=binding,
        subject_id=supervisor_id,
        repository_ids=binding.repository_ids,
        goal_refs=(ROOT_OBJECTIVE,),
        task_refs=tasks,
        allowed_task_families=("causal-federation-coordination",),
        fencing_epoch=fencing_epoch,
    )
    supervisor_definition, supervisor_capabilities = _definition(
        SupervisorDefinition,
        SupervisorCapability,
        binding=binding,
        suffix=supervisor_suffix,
        operations=supervisor_operations,
        effect_ceiling=effect,
        risk_ceiling=risk,
        resource_budget_ref=resource_budget.record_id,
        token_budget_ref=token_budget.record_id,
    )
    supervisor = repository.register_supervisor(
        supervisor,
        supervisor_assignment,
        definition=supervisor_definition,
        capabilities=supervisor_capabilities,
        idempotency_key="register:" + supervisor_suffix,
    )

    subagent_binding = replace(binding, supervisor_population=1)
    subagent_suffix = "casf-bounded-agent-" + repository_tree_id[:16]
    subagent_id = "subagent:" + subagent_suffix
    subagent = SubagentInstance(
        record_id=subagent_id,
        revision=1,
        binding=subagent_binding,
        state=FederationLifecycleState.ADMITTED.value,
        federation_id=federation_identity.record_id,
        supervisor_id=supervisor_id,
        task_id=tasks[0],
        lease_id="lease:" + subagent_suffix,
        fencing_epoch=fencing_epoch,
    )
    subagent_assignment = SubagentAssignment(
        record_id="assignment:" + subagent_suffix,
        revision=1,
        binding=subagent_binding,
        subject_id=subagent_id,
        repository_ids=subagent_binding.repository_ids,
        goal_refs=(ROOT_OBJECTIVE,),
        task_refs=(tasks[0],),
        allowed_task_families=("causal-federation-coordination",),
        fencing_epoch=fencing_epoch,
    )
    subagent_definition, subagent_capabilities = _definition(
        SubagentDefinition,
        SubagentCapability,
        binding=subagent_binding,
        suffix=subagent_suffix,
        operations=subagent_operations,
        effect_ceiling=effect,
        risk_ceiling=risk,
        resource_budget_ref=resource_budget.record_id,
        token_budget_ref=token_budget.record_id,
    )
    subagent = repository.register_subagent(
        subagent,
        definition=subagent_definition,
        assignment=subagent_assignment,
        capabilities=subagent_capabilities,
        idempotency_key="register:" + subagent_suffix,
    )

    subscription = EventSubscription(
        subscription_id="subscription:" + supervisor_suffix,
        tenant_id=binding.tenant_id,
        federation_id=federation_identity.record_id,
        consumer_id="consumer:" + supervisor_suffix,
        revision=1,
        event_classes=tuple(EventClass),
        selectors=(
            EventSelector(kind=SelectorKind.SUPERVISOR, value=supervisor_id),
        ),
        maximum_batch=64,
        maximum_pending=1_024,
        retry_budget=8,
        expires_at=binding.expires_at,
        state=SubscriptionState.ACTIVE,
    )
    repository.register_subscription(
        subscription,
        supervisor_id=supervisor_id,
        maximum_fanout=64,
        idempotency_key="register:" + subscription.subscription_id,
    )
    return BootstrapAdmission(
        federation_identity=federation_identity,
        federation_receipt=federation_receipt,
        supervisor=supervisor,
        subagent=subagent,
        subscription=subscription,
        fencing_epoch=fencing_epoch,
    )


__all__ = [
    "BOOTSTRAP_PROFILE_SCHEMA",
    "BootstrapAdmission",
    "admit_bootstrap_federation",
    "validate_bootstrap_profile",
]
