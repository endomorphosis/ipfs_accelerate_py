"""SCA-175 tests for runtime state-machine obligation compilation."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.code_claim_contracts import (
    ClaimStatus,
    CodeClaimRecord,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel,
    CodeProofObligation,
)
from ipfs_accelerate_py.agent_supervisor.proof.runtime_contract_obligations import (
    RUNTIME_CONTRACT_OBLIGATIONS_INTERFACE,
    LogicFragment,
    RuntimeClaimFamily,
    RuntimeClaimState,
    RuntimeContractClaim,
    RuntimeContractObligation,
    RuntimeContractObligationError,
    RuntimeCounterexample,
    RuntimeCounterexampleKind,
    RuntimeLogicView,
    compile_runtime_claim,
    compile_runtime_claims,
)


def _counterexample(
    *,
    kind: RuntimeCounterexampleKind = RuntimeCounterexampleKind.TRANSITION,
    subject_id: str = "orchestrator.task",
    reason_code: str = "illegal_transition",
    failed_edge: str = "",
    failed_transition: str = "running->absent",
    failed_invariant: str = "",
) -> RuntimeCounterexample:
    return RuntimeCounterexample(
        kind=kind,
        subject_id=subject_id,
        reason_code=reason_code,
        failed_edge=failed_edge,
        failed_transition=failed_transition,
        failed_invariant=failed_invariant,
        expected="running->completed",
        actual=failed_transition or failed_edge or failed_invariant,
        premise_ids=("premise:lifecycle-table",),
    )


def _claim(
    family: RuntimeClaimFamily = RuntimeClaimFamily.LIFECYCLE,
    *,
    state: RuntimeClaimState = RuntimeClaimState.PROVED,
    premises: tuple[str, ...] = ("premise:edge", "premise:machine"),
    bounds: tuple[str, ...] = (),
    reason_codes: tuple[str, ...] | None = None,
    counterexamples: tuple[RuntimeCounterexample, ...] = (),
    property_id: str = "property:lifecycle-edge-legal",
    subject_id: str = "orchestrator.task",
) -> RuntimeContractClaim:
    if reason_codes is None:
        if state is RuntimeClaimState.PROVED:
            reason_codes = ("lifecycle_edges_closed",)
        elif state is RuntimeClaimState.REFUTED:
            reason_codes = ("illegal_transition",)
        elif state is RuntimeClaimState.UNSUPPORTED:
            reason_codes = ("schema_keyword_unsupported",)
        elif state is RuntimeClaimState.TIMED_OUT:
            reason_codes = ("solver_wall_time_exceeded",)
        else:
            reason_codes = ("unsupported_program_semantics",)
    if state is RuntimeClaimState.REFUTED and not counterexamples:
        counterexamples = (_counterexample(),)
    return RuntimeContractClaim(
        family=family,
        state=state,
        subject_id=subject_id,
        property_id=property_id,
        premise_ids=premises,
        reason_codes=reason_codes,
        component_ids=("orchestrator", "scheduler"),
        bound_ids=bounds,
        counterexamples=counterexamples,
    )


def _compile(
    family: RuntimeClaimFamily = RuntimeClaimFamily.LIFECYCLE,
    *,
    state: RuntimeClaimState = RuntimeClaimState.PROVED,
    premises: tuple[str, ...] = ("premise:edge", "premise:machine"),
    bounds: tuple[str, ...] = (),
    claim: RuntimeContractClaim | None = None,
    **overrides,
) -> RuntimeContractObligation:
    payload = claim or _claim(
        family, state=state, premises=premises, bounds=bounds
    )
    kwargs = {
        "catalog_id": "catalog:runtime-fixture",
        "catalog_version": "1",
        "repository_id": "repository:fixture",
        "snapshot_id": "tree:fixture",
        "scope_ids": ("scope:orchestrator", "scope:scheduler"),
        "assumption_ids": ("assumption:closed-machine",),
        "bound_ids": bounds or ("bound:workers<=4",),
        "toolchain_id": "toolchain:python-3.12",
        "policy_id": "policy:runtime-v1",
        "required_assurance": AssuranceLevel.KERNEL_VERIFIED,
    }
    kwargs.update(overrides)
    return compile_runtime_claim(payload, **kwargs)


def test_compiler_binds_snapshot_catalog_policy_toolchain_bounds_and_premises() -> None:
    result = _compile()

    assert RUNTIME_CONTRACT_OBLIGATIONS_INTERFACE == "RuntimeContractObligation@1"
    assert isinstance(result.code_obligation, CodeProofObligation)
    assert isinstance(result.code_claim, CodeClaimRecord)
    assert result.logic_fragment is LogicFragment.GRAPH
    assert result.supported is True
    assert result.snapshot_id == "tree:fixture"
    assert result.catalog_id == "catalog:runtime-fixture"
    assert result.catalog_version == "1"
    assert result.property_id == "property:lifecycle-edge-legal"
    assert result.toolchain_id == "toolchain:python-3.12"
    assert result.policy_id == "policy:runtime-v1"
    assert result.required_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert result.bound_ids == ("bound:workers<=4",)
    assert result.premise_ids == ("premise:edge", "premise:machine")
    assert result.assumption_ids == ("assumption:closed-machine",)
    assert result.invalidators
    kinds = {
        item["kind"]
        for item in result.invalidators
        if item["source"] == "compiler"
    }
    assert {
        "assumption_set",
        "bound_set",
        "catalog",
        "policy",
        "premise_set",
        "required_assurance",
        "scope_set",
        "snapshot",
        "toolchain",
    }.issubset(kinds)
    assert result.observation_state is RuntimeClaimState.PROVED
    assert result.code_claim.status is ClaimStatus.OPEN
    assert result.code_claim.derived_assurance is AssuranceLevel.UNVERIFIED

    metadata = result.code_obligation.metadata
    assert metadata["catalog_id"] == result.catalog_id
    assert metadata["snapshot_id"] == result.snapshot_id
    assert metadata["toolchain_id"] == result.toolchain_id
    assert metadata["policy_id"] == result.policy_id
    assert metadata["bound_ids"] == list(result.bound_ids)
    assert metadata["observation_state"] == "proved"
    assert metadata["supported"] is True
    assert metadata["logic_fragment"] == "graph"


def test_logic_view_uses_shared_identity_profile_and_canonical_round_trip() -> None:
    result = _compile()
    view = result.logic_view

    assert view.identity_profile == "ir-canonical-identity-v1"
    assert view.logic_id.startswith("b")
    assert view.identity.multicodec == "raw"
    assert RuntimeLogicView.from_json(view.to_json()).to_json() == view.to_json()

    encoded = result.to_json()
    decoded = RuntimeContractObligation.from_json(encoded)
    assert decoded.to_json() == encoded
    assert decoded.compiled_obligation_id == result.compiled_obligation_id
    assert decoded.code_obligation.obligation_id == result.obligation_id
    assert decoded.shared_ir_claim.obligations[0].obligation_id == view.logic_id
    assert decoded.observation_state is RuntimeClaimState.PROVED


def test_premise_and_bound_order_do_not_change_identity() -> None:
    first = _compile(
        premises=("premise:z", "premise:a", "premise:z"),
        bounds=("bound:b", "bound:a", "bound:b"),
    )
    second = _compile(
        premises=("premise:a", "premise:z"),
        bounds=("bound:a", "bound:b"),
    )

    assert first.premise_ids == second.premise_ids
    assert first.bound_ids == second.bound_ids
    assert first.logic_view.logic_id == second.logic_view.logic_id
    assert first.obligation_id == second.obligation_id
    assert first.compiled_obligation_id == second.compiled_obligation_id
    assert first.canonical_bytes() == second.canonical_bytes()


@pytest.mark.parametrize(
    "state,claim_status",
    [
        (RuntimeClaimState.PROVED, ClaimStatus.OPEN),
        (RuntimeClaimState.REFUTED, ClaimStatus.REFUTED),
        (RuntimeClaimState.UNKNOWN, ClaimStatus.UNKNOWN),
        (RuntimeClaimState.UNSUPPORTED, ClaimStatus.UNSUPPORTED),
        (RuntimeClaimState.TIMED_OUT, ClaimStatus.NOT_MEASURED),
    ],
)
def test_observation_states_remain_distinct(
    state: RuntimeClaimState,
    claim_status: ClaimStatus,
) -> None:
    result = _compile(state=state)
    assert result.observation_state is state
    assert result.code_claim.status is claim_status
    assert result.code_obligation.metadata["observation_state"] == state.value


def test_unsupported_analysis_is_explicit_unsupported_fragment() -> None:
    result = _compile(state=RuntimeClaimState.UNSUPPORTED)

    assert result.supported is False
    assert result.logic_fragment is LogicFragment.UNSUPPORTED
    assert result.logic_view.unsupported_reason == "schema_keyword_unsupported"
    assert result.code_claim.status is ClaimStatus.UNSUPPORTED
    assert result.code_obligation.fallback_checks == (
        "runtime-contract:unsupported-fragment",
    )
    assert result.observation_state is RuntimeClaimState.UNSUPPORTED


def test_unsupported_program_semantics_stay_unknown_not_refuted() -> None:
    result = _compile(
        state=RuntimeClaimState.UNKNOWN,
        claim=_claim(
            RuntimeClaimFamily.TEMPORAL,
            state=RuntimeClaimState.UNKNOWN,
            reason_codes=("unsupported_program_semantics",),
        ),
    )

    assert result.observation_state is RuntimeClaimState.UNKNOWN
    assert result.code_claim.status is ClaimStatus.UNKNOWN
    assert result.supported is True
    assert result.counterexamples == ()

    with pytest.raises(
        RuntimeContractObligationError,
        match="must remain unknown, not refuted",
    ):
        _claim(
            state=RuntimeClaimState.REFUTED,
            reason_codes=("unsupported_program_semantics",),
            counterexamples=(_counterexample(),),
        )


def test_compact_counterexamples_identify_failed_edge_transition_or_invariant() -> None:
    edge_cx = _counterexample(
        kind=RuntimeCounterexampleKind.EDGE,
        failed_edge="owned->running",
        failed_transition="",
        reason_code="missing_edge",
    )
    transition_cx = _counterexample(
        kind=RuntimeCounterexampleKind.TRANSITION,
        failed_transition="running->absent",
        reason_code="illegal_transition",
    )
    invariant_cx = _counterexample(
        kind=RuntimeCounterexampleKind.INVARIANT,
        failed_transition="",
        failed_invariant="queue_accounting_conserved",
        reason_code="conservation_broken",
    )

    edge_result = _compile(
        claim=_claim(
            state=RuntimeClaimState.REFUTED,
            counterexamples=(edge_cx,),
        )
    )
    transition_result = _compile(
        claim=_claim(
            state=RuntimeClaimState.REFUTED,
            counterexamples=(transition_cx,),
        )
    )
    invariant_result = _compile(
        claim=_claim(
            RuntimeClaimFamily.CONSERVATION,
            state=RuntimeClaimState.REFUTED,
            property_id="property:queue-conserved",
            counterexamples=(invariant_cx,),
        )
    )

    assert edge_result.counterexamples[0].failed_edge == "owned->running"
    assert edge_result.counterexamples[0].kind is RuntimeCounterexampleKind.EDGE
    assert (
        transition_result.counterexamples[0].failed_transition
        == "running->absent"
    )
    assert (
        transition_result.counterexamples[0].kind
        is RuntimeCounterexampleKind.TRANSITION
    )
    assert (
        invariant_result.counterexamples[0].failed_invariant
        == "queue_accounting_conserved"
    )
    assert (
        invariant_result.counterexamples[0].kind
        is RuntimeCounterexampleKind.INVARIANT
    )
    assert edge_result.observation_state is RuntimeClaimState.REFUTED
    assert edge_result.code_claim.status is ClaimStatus.REFUTED


@pytest.mark.parametrize(
    "family,fragment",
    [
        (RuntimeClaimFamily.LIFECYCLE, LogicFragment.GRAPH),
        (RuntimeClaimFamily.SCHEMA, LogicFragment.SCHEMA),
        (RuntimeClaimFamily.REACHABILITY, LogicFragment.GRAPH),
        (RuntimeClaimFamily.DOMINANCE, LogicFragment.DEONTIC),
        (RuntimeClaimFamily.TEMPORAL, LogicFragment.TEMPORAL),
        (RuntimeClaimFamily.CONSERVATION, LogicFragment.RELATION),
        (RuntimeClaimFamily.IDEMPOTENCE, LogicFragment.RELATION),
        (
            RuntimeClaimFamily.BOUNDED_CONCURRENCY,
            LogicFragment.BOUNDED_CONCURRENCY,
        ),
    ],
)
def test_closed_families_select_reviewed_compact_fragments(
    family: RuntimeClaimFamily,
    fragment: LogicFragment,
) -> None:
    bounds = (
        ("bound:interleavings<=8",)
        if family is RuntimeClaimFamily.BOUNDED_CONCURRENCY
        else ("bound:workers<=4",)
    )
    result = _compile(
        family,
        claim=_claim(
            family,
            property_id=f"property:{family.value}",
            bounds=bounds,
        ),
        bounds=bounds,
    )
    assert result.logic_fragment is fragment
    expression = result.logic_view.expression_dict()
    assert set(expression) == {"schema", "operator", "terms"}
    assert set(expression["terms"]) == {
        "claim_id",
        "subject_id",
        "property_id",
        "bound_ids",
    }


def test_bounded_concurrency_requires_bounds() -> None:
    claim = _claim(
        RuntimeClaimFamily.BOUNDED_CONCURRENCY,
        property_id="property:bounded-interleaving",
        bounds=(),
    )
    with pytest.raises(
        RuntimeContractObligationError,
        match="require non-empty bound_ids",
    ):
        compile_runtime_claim(
            claim,
            catalog_id="catalog:x",
            repository_id="repository:x",
            snapshot_id="tree:x",
            scope_ids=("scope:x",),
            toolchain_id="toolchain:x",
            policy_id="policy:x",
            bound_ids=(),
        )


@pytest.mark.parametrize(
    "bad_premise",
    [
        {"node": "premise:x", "source": "def mutate(): pass"},
        "def mutate():\n    pass",
        '{"nodes": ["entire", "graph"]}',
    ],
)
def test_source_and_graph_dumps_are_rejected_as_premises(bad_premise) -> None:
    claim = _claim()
    object.__setattr__(claim, "premise_ids", ("premise:ok", bad_premise))

    with pytest.raises(
        RuntimeContractObligationError,
        match="compact identifier|source or graph|must be a string",
    ):
        compile_runtime_claim(
            claim,
            catalog_id="catalog:x",
            repository_id="repository:x",
            snapshot_id="tree:x",
            scope_ids=("scope:x",),
            toolchain_id="toolchain:x",
            policy_id="policy:x",
        )


def test_no_freeform_theorem_or_missing_refutation_witness_is_admitted() -> None:
    payload = _claim().to_dict()
    payload["theorem"] = "Everything is safe."

    with pytest.raises(RuntimeContractObligationError, match="unsupported fields"):
        compile_runtime_claim(
            payload,
            catalog_id="catalog:x",
            repository_id="repository:x",
            snapshot_id="tree:x",
            scope_ids=("scope:x",),
            toolchain_id="toolchain:x",
            policy_id="policy:x",
        )

    with pytest.raises(
        RuntimeContractObligationError,
        match="requires a compact counterexample",
    ):
        RuntimeContractClaim(
            family=RuntimeClaimFamily.LIFECYCLE,
            state=RuntimeClaimState.REFUTED,
            subject_id="orchestrator.task",
            property_id="property:x",
            premise_ids=("premise:edge",),
            reason_codes=("illegal_transition",),
            counterexamples=(),
        )


def test_tampering_with_canonical_binding_fails_round_trip_validation() -> None:
    result = _compile()
    payload = result.to_dict()
    payload["code_obligation"]["metadata"]["policy_id"] = "policy:other"

    with pytest.raises(
        RuntimeContractObligationError,
        match="mandatory binding|bindings disagree",
    ):
        RuntimeContractObligation.from_dict(payload)


def test_compile_claims_batch_is_deterministic_and_deduped_by_identity() -> None:
    claims = (
        _claim(RuntimeClaimFamily.LIFECYCLE, property_id="property:a"),
        _claim(
            RuntimeClaimFamily.DOMINANCE,
            property_id="property:b",
            subject_id="supervisor.mutation",
        ),
    )
    results = compile_runtime_claims(
        claims,
        catalog_id="catalog:runtime-fixture",
        repository_id="repository:fixture",
        snapshot_id="tree:fixture",
        scope_ids=("scope:orchestrator", "scope:scheduler"),
        bound_ids=("bound:workers<=4",),
        toolchain_id="toolchain:python-3.12",
        policy_id="policy:runtime-v1",
    )
    assert len(results) == 2
    ids = [item.compiled_obligation_id for item in results]
    assert ids == sorted(ids)
    again = compile_runtime_claims(
        claims,
        catalog_id="catalog:runtime-fixture",
        repository_id="repository:fixture",
        snapshot_id="tree:fixture",
        scope_ids=("scope:orchestrator", "scope:scheduler"),
        bound_ids=("bound:workers<=4",),
        toolchain_id="toolchain:python-3.12",
        policy_id="policy:runtime-v1",
    )
    assert [item.compiled_obligation_id for item in again] == ids


def test_counterexample_missing_structural_locator_fails_closed() -> None:
    with pytest.raises(
        RuntimeContractObligationError,
        match="requires failed_transition",
    ):
        RuntimeCounterexample(
            kind=RuntimeCounterexampleKind.TRANSITION,
            subject_id="x",
            reason_code="illegal_transition",
            failed_transition="",
        )
