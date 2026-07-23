from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
)
from ipfs_accelerate_py.agent_supervisor.multi_prover_router import (
    AttemptOutcome,
    PortfolioAttempt,
    PortfolioPlan,
    PortfolioResult,
    PortfolioVerdict,
    PropertyKind,
    PropertyObligation,
    ProverLane,
    ProverRole,
)
from ipfs_accelerate_py.agent_supervisor.prover_evidence_store import (
    ConformanceBinding,
    EvidenceLookupStatus,
    EvidenceRejectionReason,
    ProverEvidenceStore,
    build_prover_evidence_key,
    query_prover_evidence,
)


def _key(**changes: object):
    values: dict[str, object] = {
        "property_class": PropertyKind.STATE_MACHINE,
        "normalized_model": {
            "initial": "idle",
            "transitions": [["idle", "running"], ["running", "done"]],
        },
        "translator_profile": {
            "id": "tla-translation",
            "version": "2.1",
            "semantics": "exact",
        },
        "assumptions": ("finite-workers", "fair-scheduling"),
        "finite_bounds": {"workers": 3, "steps": 12},
        "prover_versions": {"apalache": "0.45.2", "tlc": "2.19"},
        "kernel_versions": {"apalache-typechecker": "0.45.2"},
        "policy": {"id": "state-machine-portfolio@3", "fail_closed": True},
        "repository_tree_id": "git-tree:abc123",
        "conformance_fixture_set_id": "fixture-set:state-machine@8",
    }
    values.update(changes)
    return build_prover_evidence_key(**values)


def _binding(
    *,
    fixture_set_id: str = "fixture-set:state-machine@8",
    passed: bool = True,
    assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED,
) -> ConformanceBinding:
    return ConformanceBinding(
        fixture_set_id=fixture_set_id,
        report_ids=("conformance-report:1",),
        passed=passed,
        permitted_assurance=assurance if passed else AssuranceLevel.UNVERIFIED,
    )


def _result(
    *,
    assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED,
    disagreement: bool = False,
) -> PortfolioResult:
    obligation = PropertyObligation(
        obligation_id="state-machine-safety",
        property_kind=PropertyKind.STATE_MACHINE,
        statement="No two workers own the same task.",
        required_assurance=(
            assurance
            if assurance.rank >= AssuranceLevel.SOLVER_CHECKED.rank
            else AssuranceLevel.SOLVER_CHECKED
        ),
    )
    lanes = [
        ProverLane(
            "apalache",
            ProverRole.MODEL_CHECKER,
            authority_capability="bounded_state_machine",
        )
    ]
    attempts = [
        PortfolioAttempt(
            prover_id="apalache",
            role=ProverRole.MODEL_CHECKER,
            stage=0,
            reported_outcome=AttemptOutcome.VERIFIED,
            effective_outcome=AttemptOutcome.VERIFIED,
            authoritative=True,
            conclusive=False,
            detail="bounded model accepted",
            capability_receipt_id="capability:apalache:0.45.2",
            conformance_gate_id="gate:fixture-set:state-machine@8",
            duration_ms=15,
        )
    ]
    counterexample_attempt_id = ""
    if disagreement:
        lanes.append(
            ProverLane(
                "tlc",
                ProverRole.MODEL_CHECKER,
                authority_capability="bounded_state_machine",
            )
        )
        counterexample = PortfolioAttempt(
            prover_id="tlc",
            role=ProverRole.MODEL_CHECKER,
            stage=0,
            reported_outcome=AttemptOutcome.COUNTEREXAMPLE,
            effective_outcome=AttemptOutcome.COUNTEREXAMPLE,
            authoritative=True,
            conclusive=True,
            detail="counterexample retained only in the durable result",
            evidence={"counterexample_id": "counterexample:tlc:7"},
            capability_receipt_id="capability:tlc:2.19",
            conformance_gate_id="gate:fixture-set:state-machine@8",
            duration_ms=9,
        )
        attempts.append(counterexample)
        counterexample_attempt_id = counterexample.attempt_id
    plan = PortfolioPlan(
        obligation=obligation,
        policy_id="state-machine-portfolio@3",
        lanes=tuple(lanes),
    )
    if disagreement:
        return PortfolioResult(
            plan=plan,
            verdict=PortfolioVerdict.INCONCLUSIVE,
            assurance=AssuranceLevel.UNVERIFIED,
            attempts=tuple(attempts),
            reason="authoritative provers disagree",
            authority_attempt_ids=(attempts[0].attempt_id,),
            counterexample_attempt_id=counterexample_attempt_id,
            disagreement=True,
        )
    return PortfolioResult(
        plan=plan,
        verdict=PortfolioVerdict.PROVED,
        assurance=assurance,
        attempts=tuple(attempts),
        reason="model-checking authority accepted the bounded model",
        authority_attempt_ids=(attempts[0].attempt_id,),
    )


def test_identity_binds_every_semantic_toolchain_and_trust_dimension() -> None:
    baseline = _key()
    assert baseline.key_id == _key(
        assumptions=("fair-scheduling", "finite-workers")
    ).key_id

    mutations = {
        "property_class": PropertyKind.FINITE_CONSTRAINT,
        "normalized_model": {"initial": "different"},
        "translator_profile": {"id": "tla-translation", "version": "2.2"},
        "assumptions": ("finite-workers",),
        "finite_bounds": {"workers": 3, "steps": 13},
        "prover_versions": {"apalache": "0.46.0", "tlc": "2.19"},
        "kernel_versions": {"apalache-typechecker": "0.46.0"},
        "policy": {"id": "state-machine-portfolio@4", "fail_closed": True},
        "repository_tree_id": "git-tree:def456",
        "conformance_fixture_set_id": "fixture-set:state-machine@9",
    }
    for field, value in mutations.items():
        assert _key(**{field: value}).key_id != baseline.key_id, field

    payload = baseline.to_dict()
    assert payload["property_class"] == "state_machine"
    for field in mutations:
        assert field in payload


def test_lookup_fails_closed_for_stale_weak_model_only_and_nonconformant(
    tmp_path: Path,
) -> None:
    now = [1_000.0]
    store = ProverEvidenceStore(tmp_path, clock=lambda: now[0])
    strong = store.put(
        _key(), _result(), conformance=_binding(), ttl_seconds=30
    )
    assert strong.stored
    assert store.lookup(_key()).status is EvidenceLookupStatus.HIT

    too_strong = store.lookup(
        _key(), required_assurance=AssuranceLevel.ATTESTED
    )
    assert EvidenceRejectionReason.INSUFFICIENT_ASSURANCE.value in (
        too_strong.reason_codes
    )

    now[0] += 31
    stale = store.lookup(_key())
    assert EvidenceRejectionReason.STALE.value in stale.reason_codes

    model_store = ProverEvidenceStore(tmp_path / "model")
    assert model_store.put(
        _key(), _result(), conformance=_binding(), model_only=True
    )
    model_only = model_store.lookup(_key())
    assert EvidenceRejectionReason.MODEL_ONLY.value in model_only.reason_codes

    conformance_store = ProverEvidenceStore(tmp_path / "nonconformant")
    assert conformance_store.put(
        _key(), _result(), conformance=_binding(passed=False)
    )
    nonconformant = conformance_store.lookup(_key())
    assert (
        EvidenceRejectionReason.NON_CONFORMANT.value
        in nonconformant.reason_codes
    )

    weak_store = ProverEvidenceStore(tmp_path / "weak")
    assert weak_store.put(
        _key(),
        _result(assurance=AssuranceLevel.SOLVER_CHECKED),
        conformance=_binding(assurance=AssuranceLevel.SOLVER_CHECKED),
    )
    weak = weak_store.lookup(
        _key(), required_assurance=AssuranceLevel.KERNEL_VERIFIED
    )
    assert EvidenceRejectionReason.INSUFFICIENT_ASSURANCE.value in weak.reason_codes


def test_disagreements_are_retained_but_never_cache_hits(tmp_path: Path) -> None:
    store = ProverEvidenceStore(tmp_path)
    saved = store.put(_key(), _result(disagreement=True), conformance=_binding())
    assert saved.stored
    lookup = store.lookup(_key(), required_assurance=AssuranceLevel.UNVERIFIED)
    assert lookup.status is EvidenceLookupStatus.REJECTED
    assert EvidenceRejectionReason.DISAGREEMENT.value in lookup.reason_codes
    assert EvidenceRejectionReason.INCONCLUSIVE.value in lookup.reason_codes


def test_json_and_duckdb_projection_expose_public_matrix_and_lineage(
    tmp_path: Path,
) -> None:
    store = ProverEvidenceStore(tmp_path / "store")
    first = store.put(_key(), _result(), conformance=_binding())
    assert first.receipt is not None
    assert store.invalidate_receipt(
        first.receipt.receipt_id,
        reason="translator conformance fixture changed",
        invalidated_by_receipt_id="conformance-report:2",
    )
    disagreement = store.put(
        _key(finite_bounds={"workers": 2, "steps": 4}),
        _result(disagreement=True),
        conformance=_binding(),
    )
    assert disagreement.stored

    projection = store.project(tmp_path / "projection.json")
    assert (tmp_path / "projection.json").is_file()
    assert (tmp_path / "projection.duckdb").is_file()
    assert projection["receipts"]
    assert projection["capabilities"]
    assert projection["attempts"]
    assert projection["disagreements"]
    assert projection["counterexamples"]
    assert projection["assurance"]["kernel_verified"] == 1
    assert projection["freshness"]["stale"] == 1
    assert projection["invalidations"][0]["reason"].startswith("translator")

    # Public summaries never contain raw attempt evidence, detail, or transcripts.
    public_text = json.dumps(projection)
    assert "counterexample retained only in the durable result" not in public_text
    assert '"evidence"' not in public_text

    attempts = query_prover_evidence(
        tmp_path / "projection.duckdb",
        table="prover_evidence_attempts",
        columns=("prover_id", "effective_outcome", "counterexample_id"),
        where="effective_outcome = ?",
        parameters=("counterexample",),
    )
    assert attempts["rows"] == [
        {
            "prover_id": "tlc",
            "effective_outcome": "counterexample",
            "counterexample_id": disagreement.receipt.result.counterexample_attempt_id,
        }
    ]
    lineage = query_prover_evidence(
        tmp_path / "projection.json",
        table="prover_evidence_invalidations",
    )
    assert lineage["rows"][0]["invalidated_by_receipt_id"] == "conformance-report:2"


def test_single_flight_deduplicates_serial_and_parallel_supervisors(
    tmp_path: Path,
) -> None:
    database = tmp_path / "shared.sqlite3"
    stores = [ProverEvidenceStore(database), ProverEvidenceStore(database)]
    calls = 0
    calls_lock = threading.Lock()
    started = threading.Event()

    def producer() -> dict[str, object]:
        nonlocal calls
        with calls_lock:
            calls += 1
        started.set()
        time.sleep(0.08)
        return {"result_id": "heavy-result:1", "proved": True}

    def run(store: ProverEvidenceStore):
        return store.single_flight(
            _key(),
            producer,
            wait_timeout_seconds=3,
            poll_interval_seconds=0.005,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(run, stores[0])
        assert started.wait(1)
        second_future = executor.submit(run, stores[1])
        first, second = first_future.result(), second_future.result()

    assert calls == 1
    assert first.value == second.value
    assert {first.owner, second.owner} == {True, False}
    assert first.fencing_token == second.fencing_token

    serial = stores[1].single_flight(_key(), producer)
    assert serial.value == first.value
    assert serial.shared
    assert calls == 1

    changed_bounds = stores[1].single_flight(
        _key(finite_bounds={"workers": 4, "steps": 12}), producer
    )
    assert changed_bounds.owner
    assert calls == 2


def test_single_flight_heartbeats_a_long_running_prover_owner(
    tmp_path: Path,
) -> None:
    database = tmp_path / "heartbeat.sqlite3"
    stores = [ProverEvidenceStore(database), ProverEvidenceStore(database)]
    calls = 0
    lock = threading.Lock()
    started = threading.Event()

    def producer() -> dict[str, str]:
        nonlocal calls
        with lock:
            calls += 1
        started.set()
        time.sleep(1.15)  # Longer than the deliberately short lease.
        return {"receipt_id": "long-running-result"}

    def run(store: ProverEvidenceStore):
        return store.single_flight(
            _key(),
            producer,
            lease_seconds=1,
            wait_timeout_seconds=3,
            poll_interval_seconds=0.01,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        leader = executor.submit(run, stores[0])
        assert started.wait(1)
        follower = executor.submit(run, stores[1])
        results = (leader.result(), follower.result())

    assert calls == 1
    assert {item.owner for item in results} == {True, False}
    assert results[0].value == results[1].value
