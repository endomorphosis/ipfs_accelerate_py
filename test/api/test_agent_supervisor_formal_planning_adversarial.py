from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.formal_planning_adversarial import (
    AdversarialAdmission,
    AdversarialPolicy,
    AdversarialValidationCoordinator,
    AdversarialValidationError,
    BoundaryKind,
    EvidenceClass,
    EvidenceConclusion,
    EvidenceExecutionStatus,
    EvidenceSource,
    FindingCode,
    FormalPlanningAdversarialGate,
    PlanTrustBinding,
    ProverBoundaryEvidence,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    AssuranceLevel,
)
from ipfs_accelerate_py.agent_supervisor.multi_prover_router import PropertyKind


NOW_MS = 1_800_000_000_000
SECRET = "private-witness-a76d47"


def _binding(**changes: object) -> PlanTrustBinding:
    values: dict[str, object] = {
        "plan_id": "plan:formal-adversarial",
        "task_id": "task:claim-lease",
        "repository_tree_id": "git-tree:current",
        "policy_id": "policy:formal-enforcement@4",
        "lane_id": "lane:claim-lease",
        "actor_id": "actor:worker-a",
        "authority_ids": ("authority:claim", "authority:implement"),
        "temporal_bounds": {"max_steps": 32, "max_time_ms": 10_000},
        "dependency_ids": ("task:prepare",),
        "formula_id": "formula:unique-live-lease",
        "normalized_model_id": "model:lease-state@7",
        "premise_ids": ("premise:fencing", "premise:task-state"),
        "tool_versions": {"z3": "4.16.0", "lean": "4.31.0"},
        "executable_digests": {
            "z3": "sha256:" + "1" * 64,
            "lean": "sha256:" + "2" * 64,
        },
        "conformance_fixture_set_id": "fixtures:prover-matrix@9",
        "cache_key_id": "cache-key:exact-request",
        "receipt_id": "receipt:exact-result",
        "trace_id": "trace:lane-a-epoch-9",
    }
    values.update(changes)
    return PlanTrustBinding(**values)  # type: ignore[arg-type]


def _evidence(
    binding: PlanTrustBinding | None = None,
    **changes: object,
) -> ProverBoundaryEvidence:
    current = binding or _binding()
    values: dict[str, object] = {
        "property_class": PropertyKind.FINITE_CONSTRAINT,
        "source": EvidenceSource.SOLVER,
        "status": EvidenceExecutionStatus.SUCCEEDED,
        "conclusion": EvidenceConclusion.HOLDS,
        **{
            name: getattr(current, name)
            for name in (
                "plan_id",
                "task_id",
                "repository_tree_id",
                "policy_id",
                "lane_id",
                "actor_id",
                "authority_ids",
                "temporal_bounds",
                "dependency_ids",
                "formula_id",
                "normalized_model_id",
                "premise_ids",
                "tool_versions",
                "executable_digests",
                "conformance_fixture_set_id",
                "cache_key_id",
                "receipt_id",
                "trace_id",
            )
        },
        "claimed_assurance": AssuranceLevel.SOLVER_CHECKED,
        "solver_verdicts": {"z3": "holds", "cvc5": "holds"},
    }
    values.update(changes)
    return ProverBoundaryEvidence(**values)  # type: ignore[arg-type]


def _policy(
    property_class: PropertyKind = PropertyKind.FINITE_CONSTRAINT,
    **changes: object,
) -> AdversarialPolicy:
    values: dict[str, object] = {
        "property_class": property_class,
        "required_assurance": AssuranceLevel.SOLVER_CHECKED,
        "now_ms": NOW_MS,
    }
    values.update(changes)
    return AdversarialPolicy(**values)  # type: ignore[arg-type]


def _codes(result: AdversarialAdmission) -> set[str]:
    return set(result.reason_codes)


def _assert_fail_closed(
    result: AdversarialAdmission,
    code: FindingCode,
) -> None:
    assert not result.admitted
    assert result.fail_closed
    assert not result.promotable
    assert result.authoritative_assurance is AssuranceLevel.UNVERIFIED
    assert code.value in result.reason_codes
    rendered = result.to_dict()
    assert SECRET not in str(rendered)
    assert all(len(item.message.encode("utf-8")) <= 384 for item in result.findings)


def test_exact_solver_evidence_is_admitted_but_never_promoted_above_solver() -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _binding(), _evidence(), _policy()
    )

    assert result.admitted
    assert result.promotable
    assert result.evidence_class is EvidenceClass.SOLVER_CANDIDATE
    assert result.authoritative_assurance is AssuranceLevel.SOLVER_CHECKED
    assert result.findings == ()
    assert AdversarialAdmission.from_dict(result.to_dict()) == result


@pytest.mark.parametrize(
    ("field_name", "mutated_value", "boundary"),
    (
        ("plan_id", "plan:forged", BoundaryKind.PLAN),
        ("actor_id", "actor:intruder", BoundaryKind.ACTOR_AUTHORITY),
        (
            "authority_ids",
            ("authority:claim", "authority:merge"),
            BoundaryKind.ACTOR_AUTHORITY,
        ),
        (
            "temporal_bounds",
            {"max_steps": 1, "max_time_ms": 10_000},
            BoundaryKind.TEMPORAL_BOUNDS,
        ),
        (
            "dependency_ids",
            (),
            BoundaryKind.TASK_DEPENDENCIES,
        ),
        ("formula_id", "formula:weaker", BoundaryKind.FORMULA),
        ("normalized_model_id", "model:attacker", BoundaryKind.MODEL),
        ("premise_ids", ("premise:substituted",), BoundaryKind.PREMISES),
        (
            "tool_versions",
            {"z3": "999.0-fake", "lean": "4.31.0"},
            BoundaryKind.TOOLCHAIN,
        ),
        (
            "executable_digests",
            {
                "z3": "sha256:" + "9" * 64,
                "lean": "sha256:" + "2" * 64,
            },
            BoundaryKind.TOOLCHAIN,
        ),
        ("cache_key_id", "cache-key:poisoned", BoundaryKind.CACHE),
        ("receipt_id", "receipt:substituted", BoundaryKind.RECEIPT),
        ("trace_id", "trace:other-lane", BoundaryKind.TRACE),
    ),
)
def test_every_semantic_and_execution_binding_mutation_is_rejected(
    field_name: str,
    mutated_value: object,
    boundary: BoundaryKind,
) -> None:
    evidence = replace(_evidence(), **{field_name: mutated_value})
    result = FormalPlanningAdversarialGate().evaluate(_binding(), evidence, _policy())

    _assert_fail_closed(result, FindingCode.BINDING_MISMATCH)
    assert boundary in {item.boundary for item in result.findings}


def test_canonical_identities_detect_mutation_of_binding_evidence_and_result() -> None:
    binding_payload = _binding().to_dict()
    binding_payload["formula_id"] = "formula:changed"
    with pytest.raises(AdversarialValidationError, match="identity mismatch"):
        PlanTrustBinding.from_dict(binding_payload)

    evidence_payload = _evidence().to_dict()
    evidence_payload["claimed_assurance"] = "attested"
    with pytest.raises(AdversarialValidationError, match="identity mismatch"):
        ProverBoundaryEvidence.from_dict(evidence_payload)

    result = FormalPlanningAdversarialGate().evaluate(
        _binding(), _evidence(), _policy()
    )
    result_payload = result.to_dict()
    result_payload["disposition"] = "rejected"
    with pytest.raises(AdversarialValidationError, match="identity mismatch"):
        AdversarialAdmission.from_dict(result_payload)


def test_actor_authority_boolean_is_independent_of_matching_actor_text() -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _binding(), _evidence(actor_authorized=False), _policy()
    )
    _assert_fail_closed(result, FindingCode.ACTOR_NOT_AUTHORIZED)


def test_provider_output_cannot_self_declare_an_authoritative_lane() -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _binding(), _evidence(authoritative=False), _policy()
    )
    _assert_fail_closed(result, FindingCode.SOURCE_NOT_AUTHORITATIVE)


@pytest.mark.parametrize(
    ("changes", "code"),
    (
        (
            {
                "status": EvidenceExecutionStatus.UNAVAILABLE,
                "tool_available": False,
            },
            FindingCode.TOOL_UNAVAILABLE,
        ),
        (
            {
                "tool_versions": {"z3": "4.16.0", "lean": "4.31.0"},
                "executable_versions_verified": False,
            },
            FindingCode.TOOL_VERSION_UNVERIFIED,
        ),
        (
            {"conformance_passed": False},
            FindingCode.CONFORMANCE_NOT_PASSED,
        ),
        (
            {"receipt_digest_verified": False},
            FindingCode.RECEIPT_NOT_VERIFIED,
        ),
        (
            {"receipt_bindings_verified": False},
            FindingCode.RECEIPT_NOT_VERIFIED,
        ),
    ),
)
def test_unavailable_fake_nonconformant_and_malformed_tools_fail_closed(
    changes: dict[str, object],
    code: FindingCode,
) -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _binding(), _evidence(**changes), _policy()
    )
    _assert_fail_closed(result, code)


def test_solver_disagreement_is_not_hidden_by_a_success_label() -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _binding(),
        _evidence(
            solver_verdicts={"z3": "holds", "cvc5": "violated"},
            claimed_assurance=AssuranceLevel.ATTESTED,
        ),
        _policy(),
    )

    _assert_fail_closed(result, FindingCode.SOLVER_DISAGREEMENT)
    assert FindingCode.FORGED_ASSURANCE.value in result.reason_codes


@pytest.mark.parametrize(
    ("bounds", "bounded", "complete", "code"),
    (
        ({}, True, True, FindingCode.BOUNDS_MISSING),
        ({"max_steps": 0}, True, True, FindingCode.BOUNDS_MISSING),
        ({"max_steps": 32}, False, True, FindingCode.BOUNDS_MISSING),
        (
            {"max_steps": 32},
            True,
            False,
            FindingCode.EXPLORATION_INCOMPLETE,
        ),
    ),
)
def test_bounded_model_results_require_visible_bounds_and_complete_exploration(
    bounds: dict[str, int],
    bounded: bool,
    complete: bool,
    code: FindingCode,
) -> None:
    binding = _binding(temporal_bounds=bounds)
    evidence = _evidence(
        binding,
        property_class=PropertyKind.STATE_MACHINE,
        source=EvidenceSource.MODEL_CHECKER,
        temporal_bounds=bounds,
        bounded=bounded,
        exploration_complete=complete,
        solver_verdicts={},
    )
    result = FormalPlanningAdversarialGate().evaluate(
        binding, evidence, _policy(PropertyKind.STATE_MACHINE)
    )
    _assert_fail_closed(result, code)


def test_complete_bounded_model_check_retains_its_property_specific_label() -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _binding(),
        _evidence(
            property_class=PropertyKind.STATE_MACHINE,
            source=EvidenceSource.MODEL_CHECKER,
            bounded=True,
            exploration_complete=True,
            solver_verdicts={},
        ),
        _policy(PropertyKind.STATE_MACHINE),
    )
    assert result.admitted
    assert result.evidence_class is EvidenceClass.BOUNDED_MODEL_CHECKED
    assert result.authoritative_assurance is AssuranceLevel.SOLVER_CHECKED


def test_protocol_positive_result_must_survive_reviewed_attack_fixtures() -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _binding(),
        _evidence(
            property_class=PropertyKind.PROTOCOL,
            source=EvidenceSource.PROTOCOL_ENGINE,
            bounded=True,
            negative_fixtures_passed=False,
            solver_verdicts={},
        ),
        _policy(PropertyKind.PROTOCOL),
    )
    _assert_fail_closed(result, FindingCode.PROTOCOL_FALSE_POSITIVE)


@pytest.mark.parametrize(
    "changes",
    (
        {
            "hypertrace": {
                "trace_a": {"private_witness": SECRET},
                "trace_b": {"public": "same"},
            }
        },
        {"hypertrace_redacted": False},
        {"hypertrace_isolated": False},
        {"hypertrace": {"observation": SECRET}},
    ),
)
def test_hypertraces_are_redacted_and_cannot_leak_across_lanes(
    changes: dict[str, object],
) -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _binding(),
        _evidence(
            property_class=PropertyKind.HYPERPROPERTY,
            source=EvidenceSource.HYPERPROPERTY_ENGINE,
            bounded=True,
            solver_verdicts={},
            **changes,
        ),
        _policy(
            PropertyKind.HYPERPROPERTY,
            forbidden_public_values=(SECRET,),
        ),
    )
    _assert_fail_closed(result, FindingCode.HYPERTRACE_LEAKAGE)
    assert SECRET not in str(result.to_dict())


@pytest.mark.parametrize(
    "changes",
    (
        {"monitor_coverage_complete": False},
        {"monitor_gaps": ("rotated-log:17",)},
    ),
)
def test_monitor_gaps_cannot_be_described_as_runtime_checked(
    changes: dict[str, object],
) -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _binding(),
        _evidence(
            property_class=PropertyKind.RUNTIME_TRACE,
            source=EvidenceSource.RUNTIME_MONITOR,
            claimed_assurance=AssuranceLevel.CANDIDATE,
            solver_verdicts={},
            **changes,
        ),
        _policy(
            PropertyKind.RUNTIME_TRACE,
            required_assurance=AssuranceLevel.CANDIDATE,
        ),
    )
    _assert_fail_closed(result, FindingCode.MONITOR_GAP)


def test_complete_runtime_trace_is_not_promoted_to_solver_or_kernel_proof() -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _binding(),
        _evidence(
            property_class=PropertyKind.RUNTIME_TRACE,
            source=EvidenceSource.RUNTIME_MONITOR,
            claimed_assurance=AssuranceLevel.CANDIDATE,
            solver_verdicts={},
        ),
        _policy(
            PropertyKind.RUNTIME_TRACE,
            required_assurance=AssuranceLevel.CANDIDATE,
        ),
    )
    assert result.admitted
    assert result.evidence_class is EvidenceClass.RUNTIME_CHECKED
    assert result.authoritative_assurance is AssuranceLevel.CANDIDATE

    stronger = FormalPlanningAdversarialGate().evaluate(
        _binding(),
        _evidence(
            property_class=PropertyKind.RUNTIME_TRACE,
            source=EvidenceSource.RUNTIME_MONITOR,
            claimed_assurance=AssuranceLevel.KERNEL_VERIFIED,
            solver_verdicts={},
        ),
        _policy(PropertyKind.RUNTIME_TRACE),
    )
    _assert_fail_closed(stronger, FindingCode.FORGED_ASSURANCE)


@pytest.mark.parametrize(
    ("source", "code"),
    (
        (EvidenceSource.MODEL_TEXT, FindingCode.MODEL_TEXT_UNTRUSTED),
        (
            EvidenceSource.NATIVE_HEURISTIC,
            FindingCode.HEURISTIC_PROOF_UNTRUSTED,
        ),
    ),
)
def test_model_text_and_native_heuristic_success_never_become_proofs(
    source: EvidenceSource,
    code: FindingCode,
) -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _binding(),
        _evidence(
            source=source,
            claimed_assurance=AssuranceLevel.ATTESTED,
            solver_verdicts={},
        ),
        _policy(),
    )
    _assert_fail_closed(result, code)
    assert FindingCode.FORGED_ASSURANCE.value in result.reason_codes


@pytest.mark.parametrize(
    ("simulated", "receipt_assurance", "expected"),
    (
        (True, AssuranceLevel.KERNEL_VERIFIED, FindingCode.SIMULATED_ZKP),
        (
            False,
            AssuranceLevel.SOLVER_CHECKED,
            FindingCode.ATTESTATION_SUBJECT_UNTRUSTED,
        ),
    ),
)
def test_zkp_requires_a_real_backend_and_an_already_trusted_receipt(
    simulated: bool,
    receipt_assurance: AssuranceLevel,
    expected: FindingCode,
) -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _binding(),
        _evidence(
            property_class=PropertyKind.KERNEL_CHECK,
            source=EvidenceSource.ZKP,
            claimed_assurance=AssuranceLevel.ATTESTED,
            simulated=simulated,
            attested_receipt_assurance=receipt_assurance,
            solver_verdicts={},
        ),
        _policy(
            PropertyKind.KERNEL_CHECK,
            required_assurance=AssuranceLevel.ATTESTED,
            accepted_evidence_classes=(EvidenceClass.ATTESTED,),
        ),
    )
    _assert_fail_closed(result, expected)


def test_real_zkp_can_only_attest_the_exact_kernel_receipt() -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _binding(),
        _evidence(
            property_class=PropertyKind.KERNEL_CHECK,
            source=EvidenceSource.ZKP,
            claimed_assurance=AssuranceLevel.ATTESTED,
            attested_receipt_assurance=AssuranceLevel.KERNEL_VERIFIED,
            solver_verdicts={},
        ),
        _policy(
            PropertyKind.KERNEL_CHECK,
            required_assurance=AssuranceLevel.ATTESTED,
            accepted_evidence_classes=(EvidenceClass.ATTESTED,),
        ),
    )
    assert result.admitted
    assert result.authoritative_assurance is AssuranceLevel.ATTESTED


def _cache_evidence(
    *,
    origin: ProverBoundaryEvidence | None,
    created_at_ms: int = NOW_MS - 1_000,
    expires_at_ms: int = NOW_MS + 10_000,
    **changes: object,
) -> ProverBoundaryEvidence:
    values: dict[str, object] = {
        "source": EvidenceSource.CACHE,
        "origin": origin,
        "cache_created_at_ms": created_at_ms,
        "cache_expires_at_ms": expires_at_ms,
        "solver_verdicts": {},
    }
    values.update(changes)
    return replace(_evidence(), **values)


@pytest.mark.parametrize(
    ("changes", "expected"),
    (
        ({"origin": None}, FindingCode.CACHE_ORIGIN_MISSING),
        (
            {"created_at_ms": NOW_MS - 100_000, "expires_at_ms": NOW_MS - 1},
            FindingCode.CACHE_STALE,
        ),
        ({"cache_invalidated": True}, FindingCode.CACHE_INVALIDATED),
        ({"cache_digest_verified": False}, FindingCode.CACHE_POISONED),
    ),
)
def test_cache_entries_are_not_a_trust_root_and_stale_entries_fail_policy(
    changes: dict[str, object],
    expected: FindingCode,
) -> None:
    origin = _evidence()
    cache_args = dict(changes)
    supplied_origin = cache_args.pop("origin", origin)
    result = FormalPlanningAdversarialGate().evaluate(
        _binding(),
        _cache_evidence(origin=supplied_origin, **cache_args),
        _policy(max_cache_age_ms=5_000),
    )
    _assert_fail_closed(result, expected)


def test_fresh_cache_revalidates_origin_and_never_upgrades_it() -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _binding(),
        _cache_evidence(origin=_evidence()),
        _policy(max_cache_age_ms=5_000),
    )
    assert result.admitted
    assert result.evidence_class is EvidenceClass.SOLVER_CANDIDATE
    assert result.authoritative_assurance is AssuranceLevel.SOLVER_CHECKED

    forged_origin = replace(
        _evidence(source=EvidenceSource.MODEL_TEXT, solver_verdicts={}),
        claimed_assurance=AssuranceLevel.ATTESTED,
    )
    rejected = FormalPlanningAdversarialGate().evaluate(
        _binding(),
        _cache_evidence(origin=forged_origin),
        _policy(max_cache_age_ms=5_000),
    )
    _assert_fail_closed(rejected, FindingCode.CACHE_ORIGIN_MISSING)


def test_property_specific_evidence_cannot_cross_satisfy_another_lane() -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _binding(),
        _evidence(
            property_class=PropertyKind.PROTOCOL,
            source=EvidenceSource.PROTOCOL_ENGINE,
            bounded=True,
            solver_verdicts={},
        ),
        _policy(PropertyKind.KERNEL_CHECK),
    )
    _assert_fail_closed(result, FindingCode.PROPERTY_CLASS_MISMATCH)
    assert FindingCode.EVIDENCE_CLASS_NOT_ALLOWED.value in result.reason_codes


def test_authoritative_counterexample_is_admitted_but_not_promotable() -> None:
    result = FormalPlanningAdversarialGate().evaluate(
        _binding(),
        _evidence(conclusion=EvidenceConclusion.VIOLATED),
        _policy(),
    )
    assert result.admitted
    assert not result.promotable
    assert result.authoritative_assurance is AssuranceLevel.SOLVER_CHECKED


def test_parallel_duplicate_claims_execute_exactly_once(tmp_path: Path) -> None:
    coordinator = AdversarialValidationCoordinator(tmp_path / "flights.duckdb")
    calls = 0
    lock = threading.Lock()

    def evaluator(
        binding: PlanTrustBinding,
        evidence: ProverBoundaryEvidence,
        policy: AdversarialPolicy,
    ) -> AdversarialAdmission:
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.05)
        return FormalPlanningAdversarialGate().evaluate(binding, evidence, policy)

    def run(_: int):
        return coordinator.evaluate(
            _binding(),
            _evidence(),
            _policy(),
            evaluator=evaluator,
            wait_timeout_seconds=5,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, range(16)))

    assert calls == 1
    assert sum(item.owner for item in results) == 1
    assert all(item.admission.admitted for item in results)
    assert len({item.admission.content_id for item in results}) == 1
    assert len({item.fencing_token for item in results}) == 1


def test_restart_reuses_the_same_fenced_admission(tmp_path: Path) -> None:
    path = tmp_path / "restart.duckdb"
    first = AdversarialValidationCoordinator(path).evaluate(
        _binding(), _evidence(), _policy(), owner_id="supervisor:first"
    )
    restarted = AdversarialValidationCoordinator(path).evaluate(
        _binding(), _evidence(), _policy(), owner_id="supervisor:restarted"
    )

    assert first.owner
    assert restarted.shared
    assert restarted.fencing_token == first.fencing_token
    assert restarted.admission == first.admission
    assert restarted.admission.admitted


def test_cancellation_is_durable_and_fail_closed_for_duplicates(
    tmp_path: Path,
) -> None:
    event = threading.Event()
    event.set()
    coordinator = AdversarialValidationCoordinator(tmp_path / "cancel.duckdb")
    calls = 0

    def should_not_run(*_: object) -> AdversarialAdmission:
        nonlocal calls
        calls += 1
        raise AssertionError("cancelled evaluator ran")

    first = coordinator.evaluate(
        _binding(),
        _evidence(),
        _policy(),
        cancel_event=event,
        evaluator=should_not_run,  # type: ignore[arg-type]
    )
    duplicate = AdversarialValidationCoordinator(tmp_path / "cancel.duckdb").evaluate(
        _binding(), _evidence(), _policy()
    )

    assert calls == 0
    assert first.owner
    assert duplicate.shared
    _assert_fail_closed(first.admission, FindingCode.VALIDATION_CANCELLED)
    assert duplicate.admission == first.admission


def test_parallel_cancelled_claims_share_one_fail_closed_outcome(
    tmp_path: Path,
) -> None:
    path = tmp_path / "parallel-cancel.duckdb"
    event = threading.Event()
    event.set()

    def run(index: int):
        # Separate instances exercise the same restart/process-facing durable
        # boundary rather than sharing in-memory coordinator state.
        return AdversarialValidationCoordinator(path).evaluate(
            _binding(),
            _evidence(),
            _policy(),
            cancel_event=event,
            owner_id=f"cancelled-supervisor:{index}",
            wait_timeout_seconds=5,
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(run, range(12)))

    assert sum(item.owner for item in results) == 1
    assert len({item.fencing_token for item in results}) == 1
    assert len({item.admission.content_id for item in results}) == 1
    for item in results:
        _assert_fail_closed(item.admission, FindingCode.VALIDATION_CANCELLED)


def test_crash_and_restart_never_manufacture_an_admission(tmp_path: Path) -> None:
    path = tmp_path / "crash.duckdb"

    def crash(*_: object) -> AdversarialAdmission:
        raise RuntimeError(f"provider crashed with {SECRET}")

    first = AdversarialValidationCoordinator(path).evaluate(
        _binding(),
        _evidence(),
        _policy(),
        evaluator=crash,  # type: ignore[arg-type]
    )
    restarted = AdversarialValidationCoordinator(path).evaluate(
        _binding(), _evidence(), _policy()
    )

    _assert_fail_closed(first.admission, FindingCode.SINGLE_FLIGHT_FAILED)
    _assert_fail_closed(restarted.admission, FindingCode.SINGLE_FLIGHT_FAILED)
    assert SECRET not in str(first.admission.to_dict())
    assert SECRET not in str(restarted.admission.to_dict())
    connection = duckdb.connect(str(path), read_only=True)
    try:
        outcomes = connection.execute(
            "SELECT outcome_json FROM prover_evidence_flight_outcomes"
        ).fetchall()
    finally:
        connection.close()
    assert outcomes
    assert SECRET not in str(outcomes)


def test_parallel_crash_claims_all_fail_closed(tmp_path: Path) -> None:
    coordinator = AdversarialValidationCoordinator(tmp_path / "parallel-crash.duckdb")
    calls = 0
    lock = threading.Lock()

    def crash(*_: object) -> AdversarialAdmission:
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.03)
        raise RuntimeError("crash")

    def run(_: int):
        return coordinator.evaluate(
            _binding(),
            _evidence(),
            _policy(),
            evaluator=crash,  # type: ignore[arg-type]
            wait_timeout_seconds=5,
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(run, range(12)))

    assert calls == 1
    assert all(item.admission.fail_closed for item in results)
    assert all(
        FindingCode.SINGLE_FLIGHT_FAILED.value in item.admission.reason_codes
        for item in results
    )
