"""Tests for per-call-site compatibility inventory (RPR-029)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_consumer_inventory import (
    ActualArgument,
    ArgumentForm,
    CallSiteObservation,
    CallerKind,
    ChangeConsumerInventory,
    ChangeConsumerInventoryError,
    ConsumerCompatibilityEntry,
    ConsumerCompatibilityLedger,
    ConsumerDisposition,
    ConsumerMigrationObligation,
    RouteStatus,
    build_change_consumer_inventory,
    required_caller_kinds,
)
from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    ContractClauseDelta,
    DeltaDisposition,
    DeltaKind,
    GraphNodeRef,
    GraphProvenance,
    ProgramContractDelta,
    PropagationAuthorityRoots,
)


@pytest.fixture
def roots() -> PropagationAuthorityRoots:
    return PropagationAuthorityRoots(
        repository_id="repository:one",
        base_forest_id="forest:base",
        base_tree_id="tree:base",
        base_overlay_id="overlay:base",
        candidate_forest_id="forest:candidate",
        candidate_tree_id="tree:candidate",
        candidate_overlay_id="overlay:candidate",
        graph_id="graph:one",
        index_id="index:one",
        model_id="model:one",
        config_id="config:one",
        translator_id="translator:one",
        toolchain_id="toolchain:one",
        policy_id="policy:one",
    )


def _clause(
    *,
    clause_id: str = "clause:param-add",
    kind: DeltaKind = DeltaKind.PARAMETER_ADD,
    disposition: DeltaDisposition = DeltaDisposition.BREAKING,
    subject: str = "symbol:process",
    reason: str = "third argument required: parameter=context",
    after: str = "contract:process(left: A, right: B, context: C) -> R",
) -> ContractClauseDelta:
    return ContractClauseDelta(
        clause_id=clause_id,
        kind=kind,
        disposition=disposition,
        subject_symbol_id=subject,
        consumer_domain="domain:python-callers",
        before_contract_ref="contract:process(left: A, right: B) -> R",
        after_contract_ref=after,
        reason=reason,
    )


def _delta(
    roots: PropagationAuthorityRoots,
    *clauses: ContractClauseDelta,
    subject: str = "symbol:process",
) -> ProgramContractDelta:
    if not clauses:
        clauses = (_clause(subject=subject),)
    return ProgramContractDelta(
        roots=roots,
        change_set_id="changeset:process-arity",
        subject_symbol_id=subject,
        before_contract_ref="contract:before",
        after_contract_ref="contract:after",
        clauses=clauses,
        evidence_refs=("evidence:extract",),
    )


def _node(
    consumer_id: str,
    path: str,
    symbol_id: str,
    *,
    kind: str = "function",
) -> GraphNodeRef:
    return GraphNodeRef(
        node_id=f"node:{consumer_id}",
        kind=kind,
        path=path,
        symbol_id=symbol_id,
        artifact_id=f"blob:{consumer_id}",
        provenance=GraphProvenance.TRUSTED,
        extractor_id="extractor:test",
    )


def _two_arg_observation(
    *,
    consumer_id: str,
    kind: CallerKind,
    path: str,
    symbol_id: str,
    route_hops: tuple[str, ...] = (),
    route_status: RouteStatus = RouteStatus.RESOLVED,
    awaited: bool = False,
    result_uses: tuple[str, ...] = (),
    defaults_applied: tuple[str, ...] = (),
    callee_default_refs: tuple[str, ...] = (),
    supplies: tuple[str, ...] = (),
    receiver_state_refs: tuple[str, ...] = (),
    path_condition_ref: str = "",
    handled_errors: tuple[str, ...] = (),
    effects: tuple[str, ...] = (),
    capabilities: tuple[str, ...] = (),
    provided: int | None = 2,
    required: int | None = 3,
    kwargs: tuple[ActualArgument, ...] | None = None,
) -> CallSiteObservation:
    args = kwargs
    if args is None:
        args = (
            ActualArgument(0, ArgumentForm.POSITIONAL, type_ref="type:A"),
            ActualArgument(1, ArgumentForm.POSITIONAL, type_ref="type:B"),
        )
    return CallSiteObservation(
        consumer_id=consumer_id,
        caller_kind=kind,
        path=path,
        symbol_id=symbol_id,
        callee_symbol_id="symbol:process",
        actual_arguments=args,
        defaults_applied=defaults_applied,
        receiver_state_refs=receiver_state_refs,
        path_condition_ref=path_condition_ref,
        awaited=awaited,
        result_uses=result_uses,
        handled_error_refs=handled_errors,
        effect_refs=effects,
        capability_refs=capabilities,
        route_hops=route_hops or (f"route:{consumer_id}",),
        route_status=route_status,
        callee_default_refs=callee_default_refs,
        required_argument_count=required,
        provided_argument_count=provided,
        supplies_parameter_names=supplies,
        node=_node(consumer_id, path, symbol_id),
        span_ref=f"span:{consumer_id}",
        call_requirement_ref=f"callreq:{consumer_id}",
        evidence_refs=(f"evidence:{consumer_id}",),
        attributes={"added_parameter": "context"},
    )


def _all_kind_callers() -> list[CallSiteObservation]:
    specs = [
        ("direct", CallerKind.DIRECT, "src/client.py", "symbol:client.run"),
        ("aliased", CallerKind.ALIASED, "src/alias_api.py", "symbol:alias.handle"),
        ("re_exported", CallerKind.RE_EXPORTED, "src/public_api.py", "symbol:public.process"),
        ("wrapped", CallerKind.WRAPPED, "src/wrapper.py", "symbol:wrapper.proxy"),
        ("decorated", CallerKind.DECORATED, "src/decorators.py", "symbol:decorated.call"),
        ("callback", CallerKind.CALLBACK, "src/hooks.py", "symbol:hooks.on_event"),
        ("overload", CallerKind.OVERLOAD, "src/overloads.py", "symbol:overloads.dispatch"),
        (
            "method",
            CallerKind.METHOD_OVERRIDE,
            "src/service.py",
            "symbol:Service.run",
        ),
        ("factory", CallerKind.FACTORY, "src/factory.py", "symbol:factory.build_worker"),
        ("test_mock", CallerKind.TEST_MOCK, "tests/test_process.py", "symbol:test_process"),
        (
            "generated",
            CallerKind.GENERATED_CLIENT,
            "generated/client/process.py",
            "symbol:generated.process",
        ),
    ]
    return [
        _two_arg_observation(
            consumer_id=f"consumer:{name}",
            kind=kind,
            path=path,
            symbol_id=symbol,
            route_hops=(f"hop:{name}",),
            awaited=(name == "callback"),
            result_uses=("assigned:result",) if name == "direct" else (),
            receiver_state_refs=("receiver:self",) if name == "method" else (),
            path_condition_ref="pathcond:always" if name == "direct" else "",
            handled_errors=("error:ValueError",) if name == "wrapped" else (),
            effects=("effect:io",) if name == "factory" else (),
            capabilities=("cap:process",) if name == "direct" else (),
        )
        for name, kind, path, symbol in specs
    ]


# ---------------------------------------------------------------------------
# Catalogue / schema
# ---------------------------------------------------------------------------


def test_required_caller_kinds_cover_acceptance_catalogue() -> None:
    kinds = required_caller_kinds()
    assert kinds == {
        "direct",
        "aliased",
        "re_exported",
        "wrapped",
        "decorated",
        "callback",
        "overload",
        "method_override",
        "factory",
        "test_mock",
        "generated_client",
    }
    assert {item.value for item in CallerKind} == kinds


def test_observation_records_args_defaults_route_and_effects(roots: PropagationAuthorityRoots) -> None:
    observation = _two_arg_observation(
        consumer_id="consumer:direct",
        kind=CallerKind.DIRECT,
        path="src/client.py",
        symbol_id="symbol:client.run",
        awaited=True,
        result_uses=("assigned:receipt", "returned"),
        defaults_applied=(),
        receiver_state_refs=("receiver:client",),
        path_condition_ref="pathcond:authenticated",
        handled_errors=("error:TimeoutError",),
        effects=("effect:network",),
        capabilities=("cap:send",),
        kwargs=(
            ActualArgument(0, ArgumentForm.POSITIONAL, type_ref="type:A", value_ref="expr:left"),
            ActualArgument(1, ArgumentForm.KEYWORD, name="right", type_ref="type:B"),
            ActualArgument(2, ArgumentForm.SPLAT_KWARGS, name="options"),
        ),
        provided=3,
        required=3,
    )
    assert observation.awaited is True
    assert observation.result_uses == ("assigned:receipt", "returned")
    assert observation.handled_error_refs == ("error:TimeoutError",)
    assert observation.effect_refs == ("effect:network",)
    assert observation.capability_refs == ("cap:send",)
    assert observation.path_condition_ref == "pathcond:authenticated"
    assert observation.receiver_state_refs == ("receiver:client",)
    assert observation.has_splat is True
    assert observation.actual_arguments[1].form is ArgumentForm.KEYWORD
    assert CallSiteObservation.from_dict(observation.to_dict()) == observation


# ---------------------------------------------------------------------------
# Core: two-to-three argument change
# ---------------------------------------------------------------------------


def test_two_to_three_flags_every_two_arg_caller_independently(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(roots)
    callers = _all_kind_callers()
    ledger = ChangeConsumerInventory(roots=roots).inventory(delta, callers)

    assert len(ledger.entries) == len(callers)
    assert len(ledger.migrate_entries) == len(callers)
    assert not ledger.compatible_entries

    # Every still-two-argument caller gets its own obligation.
    consumer_ids = {entry.observation.consumer_id for entry in ledger.entries}
    assert len(consumer_ids) == len(callers)
    obligations = ledger.obligations
    assert len(obligations) == len(callers)
    assert all(isinstance(item, ConsumerMigrationObligation) for item in obligations)
    assert all(item.disposition is ConsumerDisposition.MIGRATE for item in obligations)
    assert all(item.missing_input_ids for item in obligations)

    # Each obligation is independently keyed.
    obligation_ids = {item.obligation_id for item in obligations}
    assert len(obligation_ids) == len(callers)

    for entry in ledger.entries:
        assert "context" in entry.missing_parameter_names
        assert "each_two_arg_caller_gets_obligation" in entry.reason_codes
        assert entry.observation.provided_argument_count == 2
        assert entry.observation.required_argument_count == 3


def test_one_compatible_caller_cannot_discharge_others(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(roots)
    migrate_callers = _all_kind_callers()[:4]
    compatible = _two_arg_observation(
        consumer_id="consumer:already-three",
        kind=CallerKind.DIRECT,
        path="src/modern.py",
        symbol_id="symbol:modern.run",
        supplies=("context",),
        provided=3,
        required=3,
        kwargs=(
            ActualArgument(0, ArgumentForm.POSITIONAL, type_ref="type:A"),
            ActualArgument(1, ArgumentForm.POSITIONAL, type_ref="type:B"),
            ActualArgument(2, ArgumentForm.KEYWORD, name="context", type_ref="type:C"),
        ),
    )
    # Force compatible by also marking the clause as satisfied via supplies.
    ledger = ChangeConsumerInventory(roots=roots).inventory(
        delta, [*migrate_callers, compatible]
    )

    assert len(ledger.migrate_entries) == 4
    assert len(ledger.compatible_entries) == 1
    assert ledger.compatible_entries[0].observation.consumer_id == "consumer:already-three"
    assert not ledger.compatible_entries[0].missing_parameter_names
    assert ledger.compatible_entries[0].disposition is ConsumerDisposition.COMPATIBLE
    # Compatible row does not clear migrate obligations.
    assert ledger.one_compatible_cannot_discharge_others()
    assert {e.observation.consumer_id for e in ledger.migrate_entries} == {
        item.consumer_id for item in migrate_callers
    }


def test_callee_default_does_not_discharge_callers_without_it(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(
        roots,
        _clause(
            reason="third argument required: parameter=context (callee may default)",
            after="contract:process(left: A, right: B, context: C = default) -> R",
        ),
    )
    # One caller applies the default at the call site → compatible for that route.
    with_default = _two_arg_observation(
        consumer_id="consumer:uses-default",
        kind=CallerKind.DIRECT,
        path="src/with_default.py",
        symbol_id="symbol:with_default.run",
        defaults_applied=("context",),
        callee_default_refs=("default:context",),
        supplies=("context",),
        provided=2,
        required=3,
    )
    # Other callers see the same callee default but do not apply it.
    without = [
        _two_arg_observation(
            consumer_id=f"consumer:no-default-{index}",
            kind=kind,
            path=f"src/caller_{index}.py",
            symbol_id=f"symbol:caller_{index}",
            callee_default_refs=("default:context",),
            provided=2,
            required=3,
        )
        for index, kind in enumerate(
            (CallerKind.DIRECT, CallerKind.ALIASED, CallerKind.WRAPPED)
        )
    ]
    ledger = ChangeConsumerInventory(roots=roots).inventory(
        delta, [with_default, *without]
    )

    assert len(ledger.compatible_entries) == 1
    assert ledger.compatible_entries[0].observation.consumer_id == "consumer:uses-default"
    assert len(ledger.migrate_entries) == 3
    for entry in ledger.migrate_entries:
        assert "context" in entry.missing_parameter_names
        assert "compatible_default_does_not_discharge_others" in entry.reason_codes
        assert entry.obligation is not None
        assert entry.obligation.disposition is ConsumerDisposition.MIGRATE


def test_duplicate_paths_do_not_duplicate_obligations(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(roots)
    first = _two_arg_observation(
        consumer_id="consumer:dup",
        kind=CallerKind.DIRECT,
        path="src/client.py",
        symbol_id="symbol:client.run",
        route_hops=("route:same",),
        path_condition_ref="pathcond:x",
    )
    duplicate = _two_arg_observation(
        consumer_id="consumer:dup",
        kind=CallerKind.DIRECT,
        path="src/client.py",
        symbol_id="symbol:client.run",
        route_hops=("route:same",),
        path_condition_ref="pathcond:x",
    )
    # Different path condition → distinct route, two obligations.
    other_route = _two_arg_observation(
        consumer_id="consumer:dup-branch",
        kind=CallerKind.DIRECT,
        path="src/client.py",
        symbol_id="symbol:client.run",
        route_hops=("route:same",),
        path_condition_ref="pathcond:y",
    )
    ledger = ChangeConsumerInventory(roots=roots).inventory(
        delta, [first, duplicate, other_route]
    )
    assert len(ledger.entries) == 2
    assert len(ledger.obligations) == 2
    assert ledger.obligation_set_id()


def test_ambiguous_and_dynamic_calls_remain_frontier(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(roots)
    sites = [
        _two_arg_observation(
            consumer_id="consumer:ambiguous",
            kind=CallerKind.DIRECT,
            path="src/dispatch.py",
            symbol_id="symbol:dispatch",
            route_status=RouteStatus.AMBIGUOUS,
        ),
        _two_arg_observation(
            consumer_id="consumer:dynamic",
            kind=CallerKind.DIRECT,
            path="src/dynamic.py",
            symbol_id="symbol:dynamic.run",
            route_status=RouteStatus.DYNAMIC,
        ),
        _two_arg_observation(
            consumer_id="consumer:external",
            kind=CallerKind.DIRECT,
            path="src/ffi.py",
            symbol_id="symbol:ffi.call",
            route_status=RouteStatus.EXTERNAL,
        ),
        _two_arg_observation(
            consumer_id="consumer:resolved",
            kind=CallerKind.DIRECT,
            path="src/ok.py",
            symbol_id="symbol:ok.run",
            route_status=RouteStatus.RESOLVED,
        ),
    ]
    ledger = ChangeConsumerInventory(roots=roots).inventory(delta, sites)
    assert len(ledger.frontier_entries) == 3
    assert set(ledger.frontier_consumer_ids) == {
        "consumer:ambiguous",
        "consumer:dynamic",
        "consumer:external",
    }
    for entry in ledger.frontier_entries:
        assert entry.disposition is ConsumerDisposition.FRONTIER
        assert entry.obligation is not None
        assert entry.obligation.proof_refs == ()
        assert not entry.missing_parameter_names
    assert len(ledger.migrate_entries) == 1
    assert ledger.migrate_entries[0].observation.consumer_id == "consumer:resolved"


def test_all_caller_kinds_enumerated_in_ledger(roots: PropagationAuthorityRoots) -> None:
    delta = _delta(roots)
    ledger = ChangeConsumerInventory(roots=roots).inventory(delta, _all_kind_callers())
    present = {entry.observation.caller_kind for entry in ledger.entries}
    assert present == set(CallerKind)


def test_ledger_round_trip_and_canonical_obligations(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(roots)
    ledger = build_change_consumer_inventory(delta, _all_kind_callers()[:3], roots=roots)
    rebuilt = ConsumerCompatibilityLedger.from_dict(ledger.to_dict())
    assert rebuilt.ledger_id == ledger.ledger_id
    assert rebuilt.entries == ledger.entries
    for obligation in rebuilt.obligations:
        assert ConsumerMigrationObligation.from_dict(obligation.to_record()) == obligation


def test_compatible_entry_cannot_carry_missing_inputs(
    roots: PropagationAuthorityRoots,
) -> None:
    observation = _two_arg_observation(
        consumer_id="consumer:x",
        kind=CallerKind.DIRECT,
        path="src/x.py",
        symbol_id="symbol:x",
        supplies=("context",),
        provided=3,
        required=3,
        kwargs=(
            ActualArgument(0, ArgumentForm.POSITIONAL),
            ActualArgument(1, ArgumentForm.POSITIONAL),
            ActualArgument(2, ArgumentForm.KEYWORD, name="context"),
        ),
    )
    node = observation.node
    assert node is not None
    with pytest.raises(ChangeConsumerInventoryError, match="compatible/excluded"):
        ConsumerCompatibilityEntry(
            observation=observation,
            disposition=ConsumerDisposition.COMPATIBLE,
            clause_ids=("clause:param-add",),
            missing_parameter_names=("context",),
        )


def test_inventory_rejects_root_mismatch(roots: PropagationAuthorityRoots) -> None:
    delta = _delta(roots)
    other = PropagationAuthorityRoots(
        repository_id="repository:other",
        base_forest_id="forest:b2",
        base_tree_id="tree:b2",
        base_overlay_id="overlay:b2",
        candidate_forest_id="forest:c2",
        candidate_tree_id="tree:c2",
        candidate_overlay_id="overlay:c2",
        graph_id="graph:two",
        index_id="index:two",
        model_id="model:two",
        config_id="config:two",
        translator_id="translator:two",
        toolchain_id="toolchain:two",
        policy_id="policy:two",
    )
    with pytest.raises(ChangeConsumerInventoryError, match="roots must match"):
        ChangeConsumerInventory(roots=other).inventory(delta, _all_kind_callers()[:1])


def test_inventory_requires_call_sites(roots: PropagationAuthorityRoots) -> None:
    delta = _delta(roots)
    with pytest.raises(ChangeConsumerInventoryError, match="at least one call site"):
        ChangeConsumerInventory(roots=roots).inventory(delta, [])


def test_inventory_parameter_add_convenience(roots: PropagationAuthorityRoots) -> None:
    delta = _delta(roots)
    sites = _all_kind_callers()[:2]
    ledger = ChangeConsumerInventory(roots=roots).inventory_parameter_add(
        delta, sites, new_parameter="context", required_arity=3
    )
    assert len(ledger.migrate_entries) == 2
    assert all("context" in entry.missing_parameter_names for entry in ledger.migrate_entries)


def test_keyword_and_splat_arguments_recorded(roots: PropagationAuthorityRoots) -> None:
    delta = _delta(roots)
    site = _two_arg_observation(
        consumer_id="consumer:kwargs",
        kind=CallerKind.DIRECT,
        path="src/kwargs.py",
        symbol_id="symbol:kwargs.run",
        kwargs=(
            ActualArgument(0, ArgumentForm.POSITIONAL, type_ref="type:A"),
            ActualArgument(1, ArgumentForm.SPLAT_ARGS, name="rest"),
            ActualArgument(2, ArgumentForm.SPLAT_KWARGS, name="options"),
        ),
        provided=3,
        required=3,
    )
    ledger = ChangeConsumerInventory(roots=roots).inventory(delta, [site])
    entry = ledger.entries[0]
    forms = {item.form for item in entry.observation.actual_arguments}
    assert ArgumentForm.SPLAT_ARGS in forms
    assert ArgumentForm.SPLAT_KWARGS in forms
    # Splat arity uncertainty still produces a migration obligation when the
    # named new parameter is not supplied.
    assert entry.disposition is ConsumerDisposition.MIGRATE
    assert "splat_argument_arity_uncertain" in entry.reason_codes


def test_excluded_consumers_do_not_migrate(roots: PropagationAuthorityRoots) -> None:
    delta = _delta(roots)
    sites = _all_kind_callers()[:2]
    ledger = ChangeConsumerInventory(roots=roots).inventory(
        delta,
        sites,
        excluded_consumer_ids=(sites[0].consumer_id,),
    )
    excluded = [
        entry
        for entry in ledger.entries
        if entry.disposition is ConsumerDisposition.EXCLUDED
    ]
    assert len(excluded) == 1
    assert excluded[0].observation.consumer_id == sites[0].consumer_id
    assert not excluded[0].missing_parameter_names
    assert len(ledger.migrate_entries) == 1


def test_entries_for_kind_filter(roots: PropagationAuthorityRoots) -> None:
    delta = _delta(roots)
    ledger = ChangeConsumerInventory(roots=roots).inventory(delta, _all_kind_callers())
    methods = ledger.entries_for_kind(CallerKind.METHOD_OVERRIDE)
    assert len(methods) == 1
    assert methods[0].observation.caller_kind is CallerKind.METHOD_OVERRIDE


def test_path_escape_rejected() -> None:
    with pytest.raises(ChangeConsumerInventoryError, match="repository-relative"):
        CallSiteObservation(
            consumer_id="consumer:bad",
            caller_kind=CallerKind.DIRECT,
            path="../escape.py",
            symbol_id="symbol:bad",
            callee_symbol_id="symbol:process",
        )


def test_compatible_clause_domain_marks_compatible_when_satisfied(
    roots: PropagationAuthorityRoots,
) -> None:
    # Fully compatible delta → all callers compatible even with two args.
    delta = _delta(
        roots,
        _clause(disposition=DeltaDisposition.COMPATIBLE, reason="defaulted third arg"),
    )
    ledger = ChangeConsumerInventory(roots=roots).inventory(
        delta, _all_kind_callers()[:3]
    )
    assert len(ledger.compatible_entries) == 3
    assert not ledger.migrate_entries
    for entry in ledger.entries:
        assert entry.disposition is ConsumerDisposition.COMPATIBLE
        assert entry.obligation is not None
        assert not entry.obligation.missing_input_ids


def test_each_obligation_is_canonical_consumer_migration_obligation(
    roots: PropagationAuthorityRoots,
) -> None:
    delta = _delta(roots)
    ledger = ChangeConsumerInventory(roots=roots).inventory(
        delta, _all_kind_callers()[:1]
    )
    obligation = ledger.obligations[0]
    assert obligation.SCHEMA.endswith("consumer-migration-obligation@1")
    assert obligation.roots == roots
    assert obligation.consumer_id == "consumer:direct"
    assert "clause:param-add" in obligation.clause_ids
    assert obligation.missing_input_ids == (
        "missing:consumer:direct:context",
    )
