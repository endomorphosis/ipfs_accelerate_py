"""Hermetic tests for CASF abstraction maps and intervention consistency."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.federation import contracts
from ipfs_accelerate_py.agent_supervisor.federation.causal_abstraction import (
    CausalAbstractionAuthorityError,
    CausalAbstractionError,
    CausalAbstractionStore,
    evaluate_intervention,
    map_may_control_scheduling,
    refuse_work_suppression,
    resulting_faithfulness,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations import (
    duckdb_available,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
    install_control_plane_schema,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions import (
    TransactionError,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client import (
    open_embedded_client,
)
from test.api.causal_federation.test_contracts import sample_binding, sample_contract
from test.api.causal_federation.test_registry import _create
from test.api.causal_federation.test_trigger import sample_policy, sample_request


def _map(**overrides: object) -> contracts.CausalAbstractionMap:
    abstraction = sample_contract(contracts.CausalAbstractionMap)
    assert isinstance(abstraction, contracts.CausalAbstractionMap)
    values = {
        "record_id": "abstraction:map-one",
        "admitted_domain_refs": ("intervention:low",),
        "excluded_domain_refs": ("intervention:out",),
        "faithfulness_status": contracts.AbstractionFaithfulness.EXACT,
        "policy_admitted": True,
    }
    values.update(overrides)
    return replace(abstraction, **values)


def _test(abstraction: contracts.CausalAbstractionMap, **overrides: object) -> contracts.InterventionTest:
    test = sample_contract(contracts.InterventionTest)
    assert isinstance(test, contracts.InterventionTest)
    values = {
        "record_id": "intervention:one",
        "abstraction_map_id": abstraction.record_id,
        "low_level_intervention_ref": "intervention:low",
        "abstracted_outcome_ref": "outcome:high",
        "high_level_outcome_ref": "outcome:high",
        "outcome": "matched",
        "mismatch_ref": "",
    }
    values.update(overrides)
    return replace(test, **values)


def test_matched_intervention_keeps_exact_authority() -> None:
    abstraction = _map()
    test = evaluate_intervention(abstraction, _test(abstraction))
    assert test.outcome == "matched"
    assert resulting_faithfulness(abstraction, (test,)) is contracts.AbstractionFaithfulness.EXACT
    assert map_may_control_scheduling(
        abstraction.faithfulness_status,
        policy_admitted=True,
        resulting_status=resulting_faithfulness(abstraction, (test,)),
    )
    refuse_work_suppression(abstraction, resulting_status=contracts.AbstractionFaithfulness.EXACT)


def test_mismatch_is_durable_and_refutes_scheduling_authority() -> None:
    abstraction = _map()
    test = evaluate_intervention(
        abstraction,
        _test(
            abstraction,
            abstracted_outcome_ref="outcome:abstracted",
            high_level_outcome_ref="outcome:high",
            outcome="mismatched",
            mismatch_ref="counterexample:low-vs-high",
        ),
    )
    assert test.outcome == "mismatched"
    assert test.mismatch_ref == "counterexample:low-vs-high"
    assert resulting_faithfulness(abstraction, (test,)) is contracts.AbstractionFaithfulness.REFUTED
    with pytest.raises(CausalAbstractionAuthorityError, match="refuted"):
        refuse_work_suppression(
            abstraction,
            resulting_status=contracts.AbstractionFaithfulness.REFUTED,
        )


def test_mismatched_intervention_requires_mismatch_ref() -> None:
    abstraction = _map()
    with pytest.raises(contracts.FederationContractError, match="mismatch_ref"):
        evaluate_intervention(
            abstraction,
            _test(
                abstraction,
                abstracted_outcome_ref="outcome:abstracted",
                outcome="mismatched",
                mismatch_ref="",
            ),
        )


def test_claimed_match_cannot_hide_a_mismatch() -> None:
    abstraction = _map()
    with pytest.raises(CausalAbstractionAuthorityError, match="does not match"):
        evaluate_intervention(
            abstraction,
            _test(
                abstraction,
                abstracted_outcome_ref="outcome:abstracted",
                high_level_outcome_ref="outcome:high",
                outcome="matched",
            ),
        )


def test_excluded_domain_is_durable_and_does_not_refute() -> None:
    abstraction = _map()
    test = evaluate_intervention(
        abstraction,
        _test(
            abstraction,
            low_level_intervention_ref="intervention:out",
            outcome="excluded",
        ),
    )
    assert test.outcome == "excluded"
    assert resulting_faithfulness(abstraction, (test,)) is contracts.AbstractionFaithfulness.EXACT


def test_unknown_domain_cannot_suppress_via_the_map() -> None:
    abstraction = _map()
    test = evaluate_intervention(
        abstraction,
        _test(
            abstraction,
            low_level_intervention_ref="intervention:unknown",
            outcome="excluded",
        ),
    )
    assert test.outcome == "excluded"


@pytest.mark.parametrize(
    "status",
    [
        contracts.AbstractionFaithfulness.EMPIRICALLY_SUPPORTED,
        contracts.AbstractionFaithfulness.HEURISTIC,
        contracts.AbstractionFaithfulness.UNKNOWN,
        contracts.AbstractionFaithfulness.REFUTED,
    ],
)
def test_nomination_only_maps_cannot_control_scheduling(
    status: contracts.AbstractionFaithfulness,
) -> None:
    abstraction = _map(faithfulness_status=status, policy_admitted=False)
    assert not map_may_control_scheduling(status, policy_admitted=False)
    with pytest.raises(CausalAbstractionAuthorityError, match="cannot suppress"):
        refuse_work_suppression(abstraction)


def test_conservative_map_requires_separate_policy_admission() -> None:
    pending = _map(
        faithfulness_status=contracts.AbstractionFaithfulness.CONSERVATIVE,
        policy_admitted=False,
    )
    admitted = replace(pending, policy_admitted=True)
    assert not map_may_control_scheduling(
        pending.faithfulness_status, policy_admitted=False
    )
    assert map_may_control_scheduling(
        admitted.faithfulness_status, policy_admitted=True
    )
    with pytest.raises(CausalAbstractionAuthorityError, match="not admitted"):
        refuse_work_suppression(pending)
    refuse_work_suppression(admitted)


def test_stale_map_revision_cannot_suppress_work() -> None:
    abstraction = _map()
    with pytest.raises(CausalAbstractionAuthorityError, match="stale"):
        refuse_work_suppression(abstraction, expected_revision=1, live_revision=2)


def _open_abstraction_store(
    tmp_path: Path,
) -> tuple[CausalAbstractionStore, contracts.FederationBinding, str]:
    database = tmp_path / "control.duckdb"
    report = install_control_plane_schema(
        database, owner_id="owner:causal-abstraction-migration"
    )
    assert report.to_version == 3
    client = open_embedded_client(
        database,
        owner_id="owner:causal-abstraction",
        seed_generation=True,
    )
    generation = client.load_generation()
    store = CausalAbstractionStore(client)
    binding = sample_binding(
        control_plane_generation=generation.generation,
        supervisor_population=0,
        causal_graph_revision=1,
    )
    request = sample_request(
        binding=binding,
        maximum_supervisors=2,
        maximum_subagents=2,
    )
    policy = sample_policy(
        binding,
        maximum_supervisors=2,
        maximum_subagents=2,
        maximum_concurrent_subagents=2,
    )
    identity, _receipt = _create(store, request=request, policy=policy)
    return store, binding, identity.record_id


def test_store_rejects_database_path(tmp_path: Path) -> None:
    with pytest.raises(CausalAbstractionError, match="database path"):
        CausalAbstractionStore(tmp_path / "control.duckdb")  # type: ignore[arg-type]


@pytest.mark.skipif(not duckdb_available(), reason="DuckDB required for abstraction store")
def test_store_records_maps_and_durable_intervention_results(tmp_path: Path) -> None:
    store, binding, federation_id = _open_abstraction_store(tmp_path)
    abstraction = _map(binding=binding)
    first = store.record_map(
        abstraction,
        federation_id=federation_id,
        expected_graph_revision=store.graph_revision(
            tenant_id=binding.tenant_id, federation_id=federation_id
        ),
        idempotency_key="idempotency:map-one",
    )
    loaded = store.load_map(
        map_id=abstraction.record_id,
        tenant_id=binding.tenant_id,
        federation_id=federation_id,
    )
    assert loaded.content_ref == abstraction.cid
    assert loaded.may_control_scheduling is True
    store.scheduling_authority(
        map_id=abstraction.record_id,
        tenant_id=binding.tenant_id,
        federation_id=federation_id,
        expected_revision=abstraction.revision,
    )
    matched = _test(abstraction, binding=binding)
    store.record_intervention(
        abstraction,
        matched,
        federation_id=federation_id,
        expected_graph_revision=first.graph_revision,
        expected_map_revision=abstraction.revision,
        idempotency_key="idempotency:matched",
    )
    mismatch = _test(
        abstraction,
        binding=binding,
        record_id="intervention:mismatch",
        abstracted_outcome_ref="outcome:abstracted",
        outcome="mismatched",
        mismatch_ref="counterexample:low-vs-high",
    )
    store.record_intervention(
        abstraction,
        mismatch,
        federation_id=federation_id,
        expected_graph_revision=store.graph_revision(
            tenant_id=binding.tenant_id, federation_id=federation_id
        ),
        expected_map_revision=abstraction.revision,
        idempotency_key="idempotency:mismatch",
    )
    refuted = store.load_map(
        map_id=abstraction.record_id,
        tenant_id=binding.tenant_id,
        federation_id=federation_id,
    )
    assert refuted.resulting_status is contracts.AbstractionFaithfulness.REFUTED
    assert refuted.may_control_scheduling is False
    with pytest.raises(CausalAbstractionAuthorityError, match="refuted"):
        store.scheduling_authority(
            map_id=abstraction.record_id,
            tenant_id=binding.tenant_id,
            federation_id=federation_id,
            expected_revision=abstraction.revision,
        )
    with pytest.raises(CausalAbstractionAuthorityError, match="stale"):
        store.scheduling_authority(
            map_id=abstraction.record_id,
            tenant_id=binding.tenant_id,
            federation_id=federation_id,
            expected_revision=abstraction.revision + 1,
        )
    with pytest.raises(TransactionError, match="already bound"):
        store.record_map(
            abstraction,
            federation_id=federation_id,
            expected_graph_revision=store.graph_revision(
                tenant_id=binding.tenant_id, federation_id=federation_id
            ),
            idempotency_key="idempotency:map-dup",
        )
