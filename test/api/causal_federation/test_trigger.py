# ``datetime.UTC`` is unavailable on the supported Python 3.8 baseline.
# ruff: noqa: UP017

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from ipfs_accelerate_py.agent_supervisor.federation import contracts
from ipfs_accelerate_py.agent_supervisor.federation.trigger import (
    AuthenticationAlgorithm,
    AuthenticationEvidence,
    AuthenticationRejected,
    BudgetReconciliationRequired,
    BudgetRejected,
    DelegationGrant,
    DelegationRejected,
    FederationControlGateway,
    HmacAuthenticationAuthority,
    HmacDelegationAuthority,
    PolicyRejected,
    RepositoryResolutionRejected,
    ResolvedRepository,
    resolved_authorization_scope_identity,
    validated_delegation_chain_identity,
)
from test.api.causal_federation.test_contracts import (
    EXPIRY,
    NOW,
    sample_binding,
    sample_budget,
)

AUTH_KEY = b"authentication-key-material"
DELEGATION_KEY = b"delegation-key-material"
AUDIENCE = "agent-supervisor:test"
NOW_EPOCH = datetime(2030, 1, 2, tzinfo=timezone.utc).timestamp()  # noqa: UP017


def sample_request(
    *,
    binding: contracts.FederationBinding | None = None,
    caller_did: str = "did:test:caller",
    audience: str = AUDIENCE,
    delegation_chain: tuple[str, ...] = (),
    maximum_supervisors: int = 4,
    maximum_subagents: int = 16,
    effect_scope: tuple[str, ...] = ("effect.read",),
) -> contracts.FederationRequest:
    binding = binding or sample_binding()
    return contracts.FederationRequest(
        caller_did=caller_did,
        delegation_chain=delegation_chain,
        audience=audience,
        program_id=binding.program_id,
        repository_roots=binding.repository_ids,
        objective_ref=binding.objective_ref,
        requested_supervisor_profile="profile:test",
        maximum_supervisors=maximum_supervisors,
        maximum_subagents=maximum_subagents,
        resource_budget=sample_budget(
            contracts.ResourceBudget,
            binding=binding,
            record_id="budget:resource",
        ),
        token_budget=sample_budget(
            contracts.TokenBudget,
            binding=binding,
            record_id="budget:token",
        ),
        effect_scope=effect_scope,
        policy_ref=binding.policy_ref,
        expiry=binding.expires_at,
        nonce="nonce:test",
        idempotency_key="idempotency:test",
        binding=binding,
    )


def sample_policy(
    binding: contracts.FederationBinding,
    **overrides: object,
) -> contracts.FederationPolicy:
    values: dict[str, object] = {
        "record_id": binding.policy_ref,
        "revision": binding.policy_revision,
        "binding": binding,
        "allowed_callers": ("did:test:caller",),
        "allowed_audiences": (AUDIENCE,),
        "allowed_operations": ("federation.create",),
        "allowed_effects": ("effect.read",),
        "maximum_supervisors": 12,
        "maximum_subagents": 256,
        "maximum_concurrent_subagents": 64,
        "conservative_abstraction_scheduling": False,
    }
    values.update(overrides)
    return contracts.FederationPolicy(**values)  # type: ignore[arg-type]


def sample_authentication(
    request: contracts.FederationRequest,
    *,
    signature: str | None = None,
    caller_did: str | None = None,
    audience: str | None = None,
    expires_at: str = EXPIRY,
) -> AuthenticationEvidence:
    return AuthenticationEvidence(
        evidence_id="authentication:test",
        caller_did=caller_did or request.caller_did,
        algorithm=AuthenticationAlgorithm.HMAC_SHA256,
        key_handle="handle:authentication",
        request_cid=request.cid,
        audience=audience or request.audience,
        nonce=request.nonce,
        issued_at=NOW,
        expires_at=expires_at,
        signature=signature
        if signature is not None
        else HmacAuthenticationAuthority.sign_request(request, AUTH_KEY),
    )


def unsigned_grant(
    request: contracts.FederationRequest,
    *,
    grant_id: str = "grant:test",
    issuer_did: str = "did:test:root",
    subject_did: str | None = None,
    parent_grant_ref: str = "",
    audience: str | None = None,
    operations: tuple[str, ...] = ("federation.create",),
    repository_ids: tuple[str, ...] | None = None,
    policy_ref: str | None = None,
    expires_at: str = EXPIRY,
) -> DelegationGrant:
    return DelegationGrant(
        grant_id=grant_id,
        issuer_did=issuer_did,
        subject_did=subject_did or request.caller_did,
        parent_grant_ref=parent_grant_ref,
        audience=audience or request.audience,
        operations=operations,
        repository_ids=repository_ids or request.repository_roots,
        policy_ref=policy_ref or request.policy_ref,
        issued_at=NOW,
        expires_at=expires_at,
        key_handle="handle:delegation",
        signature="unsigned",
    )


def signed_grant(
    request: contracts.FederationRequest,
    **overrides: object,
) -> DelegationGrant:
    grant = unsigned_grant(request, **overrides)  # type: ignore[arg-type]
    return replace(
        grant,
        signature=HmacDelegationAuthority.sign_grant(grant, DELEGATION_KEY),
    )


class StaticPolicyAuthority:
    def __init__(self, policy: contracts.FederationPolicy) -> None:
        self.policy = policy

    def get_policy(self, policy_ref: str) -> contracts.FederationPolicy:
        assert policy_ref == self.policy.record_id
        return self.policy


class StaticRepositoryAuthority:
    def __init__(
        self,
        repositories: tuple[ResolvedRepository, ...],
    ) -> None:
        self.repositories = repositories
        self.requested_refs: tuple[str, ...] = ()

    def resolve(
        self,
        repository_refs: Sequence[str],
    ) -> tuple[ResolvedRepository, ...]:
        self.requested_refs = tuple(repository_refs)
        return self.repositories


class IdempotentBudgetAuthority:
    def __init__(self, *, reject: bool = False, release_fails: bool = False) -> None:
        self.reject = reject
        self.release_fails = release_fails
        self.reservations: dict[str, contracts.BudgetReservation] = {}
        self.releases: list[tuple[str, str, str]] = []
        self.calls = 0

    def reserve(
        self,
        request: contracts.FederationRequest,
        policy: contracts.FederationPolicy,
    ) -> contracts.BudgetReservation | str:
        self.calls += 1
        if self.reject:
            return ""
        return self.reservations.setdefault(
            request.idempotency_key,
            contracts.BudgetReservation(
                record_id="budget-reservation:test",
                revision=1,
                binding=request.binding,
                parent_budget_id=request.binding.budget_ref,
                owner_id=f"federation:{request.cid}",
                dimensions=tuple(
                    list(request.resource_budget.dimensions)
                    + list(request.token_budget.dimensions)
                ),
                status="reserved",
                request_cid=request.cid,
                idempotency_key=request.idempotency_key,
                policy_ref=policy.record_id,
                policy_revision=policy.revision,
                resource_budget_ref=request.resource_budget.record_id,
                token_budget_ref=request.token_budget.record_id,
                issued_at=NOW,
                expires_at=request.expiry,
                authorization_evidence_ref="budget-admission:test",
            ),
        )

    def release(
        self,
        reservation: contracts.BudgetReservation,
        *,
        idempotency_key: str,
        reason: str,
    ) -> None:
        if self.release_fails:
            raise RuntimeError("injected release failure")
        disposition = (reservation.record_id, idempotency_key, reason)
        if disposition not in self.releases:
            self.releases.append(disposition)


class IdempotentAdmissionStore:
    def __init__(
        self,
        *,
        fail: bool = False,
        commit_then_fail: bool = False,
        lookup_fails: bool = False,
    ) -> None:
        self.fail = fail
        self.commit_then_fail = commit_then_fail
        self.lookup_fails = lookup_fails
        self.records: dict[
            str,
            tuple[contracts.FederationIdentity, contracts.FederationReceipt],
        ] = {}
        self.authorization_decisions: dict[
            str, contracts.FederationAuthorizationDecision
        ] = {}
        self.mutations = 0

    def create_federation(
        self,
        *,
        request: contracts.FederationRequest,
        policy: contracts.FederationPolicy,
        repositories: Sequence[ResolvedRepository],
        budget_reservation: contracts.BudgetReservation,
        authentication_evidence_ref: str,
        authorization_decision: contracts.FederationAuthorizationDecision,
    ) -> tuple[contracts.FederationIdentity, contracts.FederationReceipt]:
        del policy, repositories
        if self.fail:
            raise RuntimeError("injected authoritative transaction failure")
        existing = self.records.get(request.idempotency_key)
        if existing is not None:
            return existing
        self.mutations += 1
        self.authorization_decisions[authorization_decision.cid] = authorization_decision
        identity = contracts.FederationIdentity(
            record_id=f"federation:{request.cid}",
            revision=1,
            binding=request.binding,
        )
        receipt = contracts.FederationReceipt(
            record_id="federation-receipt:test",
            revision=1,
            binding=request.binding,
            outcome="created",
            evidence_refs=(
                budget_reservation.record_id,
                authentication_evidence_ref,
                authorization_decision.authentication_evidence_cid,
                authorization_decision.cid,
            ),
            recorded_at=NOW,
        )
        self.records[request.idempotency_key] = (identity, receipt)
        if self.commit_then_fail:
            raise RuntimeError("injected lost response after commit")
        return identity, receipt

    def lookup_federation_creation(
        self,
        *,
        idempotency_key: str,
        tenant_id: str,
        federation_id: str,
    ) -> tuple[contracts.FederationIdentity, contracts.FederationReceipt] | None:
        if self.lookup_fails:
            raise RuntimeError("injected authoritative lookup failure")
        record = self.records.get(idempotency_key)
        if record is None:
            return None
        identity, _receipt = record
        if identity.binding.tenant_id != tenant_id or identity.record_id != federation_id:
            return None
        return record


def resolved_for(
    binding: contracts.FederationBinding,
) -> tuple[ResolvedRepository, ...]:
    return tuple(
        ResolvedRepository(
            requested_ref=repository_id,
            repository_id=repository_id,
            tree_id=tree_id,
            semantic_state_root=semantic_root,
        )
        for repository_id, tree_id, semantic_root in zip(  # noqa: B905
            binding.repository_ids,
            binding.repository_tree_ids,
            binding.semantic_state_roots,
        )
    )


def gateway_for(
    request: contracts.FederationRequest,
    *,
    policy: contracts.FederationPolicy | None = None,
    repositories: tuple[ResolvedRepository, ...] | None = None,
    budgets: IdempotentBudgetAuthority | None = None,
    store: IdempotentAdmissionStore | None = None,
    grants: dict[str, DelegationGrant] | None = None,
) -> tuple[
    FederationControlGateway,
    IdempotentBudgetAuthority,
    IdempotentAdmissionStore,
]:
    policy = policy or sample_policy(request.binding)
    budget_authority = budgets or IdempotentBudgetAuthority()
    admission_store = store or IdempotentAdmissionStore()
    grant_map = grants or {}
    authenticator = HmacAuthenticationAuthority(
        lambda caller, handle: AUTH_KEY,
        now=lambda: NOW_EPOCH,
    )
    delegations = HmacDelegationAuthority(
        lambda grant_ref: grant_map[grant_ref],
        lambda issuer, handle: DELEGATION_KEY,
        now=lambda: NOW_EPOCH,
    )
    gateway = FederationControlGateway(
        audience=AUDIENCE,
        authenticator=authenticator,
        delegations=delegations,
        policies=StaticPolicyAuthority(policy),
        repositories=StaticRepositoryAuthority(
            repositories if repositories is not None else resolved_for(request.binding)
        ),
        budgets=budget_authority,
        store=admission_store,
        now=lambda: NOW_EPOCH,
    )
    return gateway, budget_authority, admission_store


def test_authentication_binds_request_caller_audience_nonce_and_signature() -> None:
    request = sample_request()
    authority = HmacAuthenticationAuthority(
        lambda caller, handle: AUTH_KEY,
        now=lambda: NOW_EPOCH,
    )

    authority.verify(request, sample_authentication(request))

    with pytest.raises(AuthenticationRejected):
        authority.verify(
            request,
            sample_authentication(request, signature="invalid-signature"),
        )
    with pytest.raises(AuthenticationRejected):
        authority.verify(
            request,
            sample_authentication(request, caller_did="did:test:other"),
        )
    with pytest.raises(AuthenticationRejected):
        authority.verify(
            request,
            sample_authentication(request, audience="agent-supervisor:other"),
        )


@pytest.mark.parametrize("record_factory", [sample_authentication, signed_grant])
def test_trigger_contracts_round_trip_and_reject_unknown_fields(
    record_factory,
) -> None:
    request = sample_request()
    record = record_factory(request)

    decoded = type(record).from_dict(record.to_dict())
    assert decoded == record
    assert decoded.cid == record.cid

    payload = record.to_dict()
    payload["database_path"] = "/tmp/control.duckdb"
    with pytest.raises(contracts.UnknownNormativeFieldError):
        type(record).from_dict(payload)


def test_authentication_and_request_expiry_fail_closed() -> None:
    request = sample_request()
    authority = HmacAuthenticationAuthority(
        lambda caller, handle: AUTH_KEY,
        now=lambda: NOW_EPOCH,
    )

    with pytest.raises(AuthenticationRejected):
        authority.verify(
            request,
            sample_authentication(
                request,
                expires_at="2020-01-01T00:00:00Z",
            ),
        )

    expired_binding = sample_binding(expires_at="2020-01-01T00:00:00Z")
    expired_request = sample_request(binding=expired_binding)
    with pytest.raises(AuthenticationRejected):
        authority.verify(
            expired_request,
            sample_authentication(expired_request),
        )


def test_authentication_lifetime_cannot_be_amplified_by_longer_request() -> None:
    request = sample_request()
    authority = HmacAuthenticationAuthority(
        lambda caller, handle: AUTH_KEY,
        now=lambda: NOW_EPOCH,
    )

    with pytest.raises(AuthenticationRejected, match="exceeds authentication"):
        authority.verify(
            request,
            sample_authentication(
                request,
                expires_at="2030-01-03T00:00:00Z",
            ),
        )
    with pytest.raises(AuthenticationRejected, match="not yet valid"):
        authority.verify(
            request,
            replace(
                sample_authentication(request),
                issued_at="2030-01-03T00:00:00Z",
            ),
        )


def test_authentication_uses_opaque_key_handles_only() -> None:
    request = sample_request()

    with pytest.raises(contracts.FederationContractError):
        replace(
            sample_authentication(request),
            key_handle="raw-secret-key",
        )


def test_valid_delegation_chain_is_authenticated_and_scope_bounded() -> None:
    request = sample_request(delegation_chain=("grant:test",))
    grant = signed_grant(request)
    authority = HmacDelegationAuthority(
        lambda grant_ref: {"grant:test": grant}[grant_ref],
        lambda issuer, handle: DELEGATION_KEY,
        now=lambda: NOW_EPOCH,
    )

    assert authority.verify_chain(request, request.delegation_chain) == (grant,)


def test_delegation_lifetime_cannot_be_amplified_by_longer_federation() -> None:
    request = sample_request(delegation_chain=("grant:test",))
    grant = signed_grant(
        request,
        expires_at="2030-01-03T00:00:00Z",
    )
    authority = HmacDelegationAuthority(
        lambda grant_ref: grant,
        lambda issuer, handle: DELEGATION_KEY,
        now=lambda: NOW_EPOCH,
    )

    with pytest.raises(DelegationRejected, match="exceeds delegation"):
        authority.verify_chain(request, request.delegation_chain)


@pytest.mark.parametrize(
    "grant",
    [
        lambda request: signed_grant(
            request,
            expires_at="2020-01-01T00:00:00Z",
        ),
        lambda request: signed_grant(
            request,
            audience="agent-supervisor:other",
        ),
        lambda request: signed_grant(
            request,
            operations=("federation.status",),
        ),
        lambda request: signed_grant(
            request,
            repository_ids=("repo:other",),
        ),
        lambda request: signed_grant(
            request,
            policy_ref="policy:other",
        ),
        lambda request: signed_grant(
            request,
            subject_did="did:test:other",
        ),
    ],
)
def test_delegation_expiry_audience_operation_and_scope_fail_closed(grant) -> None:
    request = sample_request(delegation_chain=("grant:test",))
    observed = grant(request)
    authority = HmacDelegationAuthority(
        lambda grant_ref: observed,
        lambda issuer, handle: DELEGATION_KEY,
        now=lambda: NOW_EPOCH,
    )

    with pytest.raises(DelegationRejected):
        authority.verify_chain(request, request.delegation_chain)


def test_delegation_contract_rejects_raw_sql_operation() -> None:
    request = sample_request()

    with pytest.raises(contracts.FederationContractError):
        unsigned_grant(
            request,
            operations=("DROP TABLE control",),
        )


def test_gateway_creates_one_idempotent_federation_through_typed_authorities() -> None:
    request = sample_request()
    gateway, budgets, store = gateway_for(request)
    evidence = sample_authentication(request)

    first = gateway.create(request, evidence)
    second = gateway.create(request, evidence)

    assert first == second
    assert first[0].record_id == f"federation:{request.cid}"
    assert first[1].outcome == "created"
    assert budgets.calls == 2
    assert len(budgets.reservations) == 1
    assert store.mutations == 1
    assert len(store.authorization_decisions) == 1
    decision = next(iter(store.authorization_decisions.values()))
    assert decision.request_cid == request.cid
    assert decision.caller_did == request.caller_did
    assert decision.audience == request.audience
    assert decision.operation is contracts.FederationOperation.CREATE
    assert decision.policy_id == request.policy_ref
    assert decision.policy_revision == request.binding.policy_revision
    assert decision.authentication_evidence_cid == evidence.cid
    assert decision.authentication_evidence_cid != evidence.evidence_id
    assert decision.resolved_scope_cid == resolved_authorization_scope_identity(
        resolved_for(request.binding),
        request.effect_scope,
    )
    assert decision.delegation_chain_cid == validated_delegation_chain_identity((), ())
    assert decision.cid in first[1].evidence_refs
    assert evidence.cid in first[1].evidence_refs
    serialized = decision.to_dict()
    assert "signature" not in serialized
    assert "key_handle" not in serialized


def test_gateway_decision_binds_validated_delegation_content_not_request_labels() -> None:
    request = sample_request(delegation_chain=("grant:test",))
    grant = signed_grant(request)
    gateway, _, store = gateway_for(
        request,
        grants={"grant:test": grant},
    )

    gateway.create(request, sample_authentication(request))

    decision = next(iter(store.authorization_decisions.values()))
    assert decision.delegation_chain_cid == validated_delegation_chain_identity(
        request.delegation_chain,
        (grant,),
    )
    assert decision.delegation_chain_cid != grant.grant_id


def test_gateway_rejects_wrong_audience_before_mutation() -> None:
    request = sample_request(audience="agent-supervisor:other")
    gateway, _, store = gateway_for(request)

    with pytest.raises(AuthenticationRejected):
        gateway.create(request, sample_authentication(request))

    assert store.mutations == 0
    assert store.authorization_decisions == {}


def test_gateway_rejects_expired_request_before_mutation() -> None:
    request = sample_request(binding=sample_binding(expires_at="2020-01-01T00:00:00Z"))
    gateway, _, store = gateway_for(request)

    with pytest.raises(AuthenticationRejected):
        gateway.create(request, sample_authentication(request))

    assert store.mutations == 0


def test_gateway_rejects_shorter_authentication_and_delegation_authority() -> None:
    request = sample_request(delegation_chain=("grant:test",))
    short_grant = signed_grant(
        request,
        expires_at="2030-01-03T00:00:00Z",
    )
    gateway, _, store = gateway_for(
        request,
        grants={"grant:test": short_grant},
    )

    with pytest.raises(AuthenticationRejected, match="exceeds authentication"):
        gateway.create(
            request,
            sample_authentication(
                request,
                expires_at="2030-01-03T00:00:00Z",
            ),
        )
    with pytest.raises(DelegationRejected, match="exceeds delegation"):
        gateway.create(request, sample_authentication(request))

    assert store.mutations == 0


@pytest.mark.parametrize(
    "policy_overrides",
    [
        {"allowed_callers": ("did:test:other",)},
        {"allowed_audiences": ("agent-supervisor:other",)},
        {"allowed_operations": ("federation.status",)},
        {"allowed_effects": ("effect.write",)},
        {"maximum_supervisors": 3},
        {
            "maximum_subagents": 15,
            "maximum_concurrent_subagents": 15,
        },
    ],
)
def test_gateway_enforces_server_side_policy(
    policy_overrides: dict[str, object],
) -> None:
    request = sample_request()
    policy = sample_policy(request.binding, **policy_overrides)
    gateway, _, store = gateway_for(request, policy=policy)

    with pytest.raises(PolicyRejected):
        gateway.create(request, sample_authentication(request))

    assert store.mutations == 0


@pytest.mark.parametrize(
    "repositories",
    [
        (),
        (
            ResolvedRepository(
                requested_ref="repo:test",
                repository_id="repo:other",
                tree_id="tree:test",
                semantic_state_root="semantic:test",
            ),
        ),
        (
            ResolvedRepository(
                requested_ref="repo:test",
                repository_id="repo:test",
                tree_id="tree:other",
                semantic_state_root="semantic:test",
            ),
        ),
        (
            ResolvedRepository(
                requested_ref="repo:test",
                repository_id="repo:test",
                tree_id="tree:test",
                semantic_state_root="semantic:other",
            ),
        ),
    ],
)
def test_gateway_accepts_only_server_resolved_roots(
    repositories: tuple[ResolvedRepository, ...],
) -> None:
    request = sample_request()
    gateway, _, store = gateway_for(request, repositories=repositories)

    with pytest.raises(RepositoryResolutionRejected):
        gateway.create(request, sample_authentication(request))

    assert store.mutations == 0


def test_gateway_rejects_missing_budget_reservation() -> None:
    request = sample_request()
    budgets = IdempotentBudgetAuthority(reject=True)
    gateway, _, store = gateway_for(request, budgets=budgets)

    with pytest.raises(BudgetRejected):
        gateway.create(request, sample_authentication(request))

    assert store.mutations == 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: replace(item, owner_id="federation:foreign"),
        lambda item: replace(item, request_cid="request:foreign"),
        lambda item: replace(item, idempotency_key="idempotency:foreign"),
        lambda item: replace(item, policy_ref="policy:foreign"),
        lambda item: replace(item, resource_budget_ref="budget:foreign"),
        lambda item: replace(item, status="released"),
    ],
)
def test_gateway_rejects_foreign_or_fabricated_budget_reservation(
    mutation: object,
) -> None:
    request = sample_request()

    class ForgingBudgetAuthority(IdempotentBudgetAuthority):
        def reserve(
            self,
            request: contracts.FederationRequest,
            policy: contracts.FederationPolicy,
        ) -> contracts.BudgetReservation:
            admitted = super().reserve(request, policy)
            assert isinstance(admitted, contracts.BudgetReservation)
            return mutation(admitted)  # type: ignore[operator,no-any-return]

    gateway, _, store = gateway_for(request, budgets=ForgingBudgetAuthority())

    with pytest.raises(BudgetRejected, match="scope differs"):
        gateway.create(request, sample_authentication(request))

    assert store.mutations == 0


def test_gateway_releases_budget_when_authoritative_admission_fails() -> None:
    request = sample_request()
    budgets = IdempotentBudgetAuthority()
    store = IdempotentAdmissionStore(fail=True)
    gateway, _, _ = gateway_for(request, budgets=budgets, store=store)

    with pytest.raises(RuntimeError, match="authoritative transaction failure"):
        gateway.create(request, sample_authentication(request))

    assert budgets.releases == [
        (
            "budget-reservation:test",
            request.idempotency_key,
            "federation_admission_failed",
        )
    ]
    assert store.mutations == 0


def test_gateway_surfaces_typed_blocker_when_budget_release_fails() -> None:
    request = sample_request()
    budgets = IdempotentBudgetAuthority(release_fails=True)
    store = IdempotentAdmissionStore(fail=True)
    gateway, _, _ = gateway_for(request, budgets=budgets, store=store)

    with pytest.raises(BudgetReconciliationRequired):
        gateway.create(request, sample_authentication(request))

    assert store.mutations == 0


def test_gateway_recovers_committed_create_after_response_is_lost() -> None:
    request = sample_request()
    budgets = IdempotentBudgetAuthority()
    store = IdempotentAdmissionStore(commit_then_fail=True)
    gateway, _, _ = gateway_for(request, budgets=budgets, store=store)

    identity, receipt = gateway.create(request, sample_authentication(request))

    assert identity.record_id == f"federation:{request.cid}"
    assert receipt.outcome == "created"
    assert store.mutations == 1
    assert budgets.releases == []


def test_gateway_retains_budget_when_admission_outcome_cannot_be_resolved() -> None:
    request = sample_request()
    budgets = IdempotentBudgetAuthority()
    store = IdempotentAdmissionStore(fail=True, lookup_fails=True)
    gateway, _, _ = gateway_for(request, budgets=budgets, store=store)

    with pytest.raises(BudgetReconciliationRequired, match="outcome is unknown"):
        gateway.create(request, sample_authentication(request))

    assert budgets.releases == []
    assert store.mutations == 0


def test_gateway_rejects_unadmitted_program_identity() -> None:
    binding = sample_binding(program_id="another-program")
    request = sample_request(binding=binding)
    gateway, _, store = gateway_for(request)

    with pytest.raises(PolicyRejected):
        gateway.create(request, sample_authentication(request))

    assert store.mutations == 0
