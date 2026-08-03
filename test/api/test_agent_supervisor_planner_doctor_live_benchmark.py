"""PDR-070: live paired Planner/Doctor benchmarks on hermetic repositories.

Asserts:

* runner materializes hermetic mini-repos from compact seeded recipes;
* real PlanCreateService / PlanSteerService / DeterministicDoctorService entry
  points are invoked;
* admitted work runs in isolated arm worktrees;
* independent quality oracle is consulted only after output seal;
* fixture ``expected`` fields never choose diagnosis/disposition/repair/
  completion;
* deterministic Doctor and V2 synthetic producers are labeled conformance-only;
* paired inputs match exactly across primary arms;
* cold / exact-warm / delta / restart and concurrency 1/2/4/6 cells are
  replayable;
* skips cannot qualify promotion.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.validation.planner_doctor_live_benchmark import (
    CACHE_STRATA,
    CONFIGURED_MAXIMUM_WORKERS,
    CONFORMANCE_ONLY_EVIDENCE_SOURCES,
    LIVE_BENCHMARK_MANIFEST_SCHEMA,
    LIVE_BENCHMARK_PAIR_RECEIPT_INTERFACE,
    LIVE_BENCHMARK_PAIR_RECEIPT_SCHEMA,
    LIVE_EVIDENCE_SOURCES,
    PAIR_FAMILIES,
    PLANNER_DOCTOR_LIVE_BENCHMARK_INTERFACE,
    PRIMARY_ARM_IDS,
    PRODUCER_TASK_ID,
    REQUESTED_CONCURRENCY,
    SCORED_REPETITIONS,
    ArmExecutionReceipt,
    ArmExecutionStatus,
    ArmId,
    CacheStratum,
    EvidenceAuthorityClass,
    ExecutionKind,
    HermeticFileRecipe,
    LiveBenchmarkCase,
    LiveBenchmarkError,
    LiveBenchmarkManifest,
    LiveBenchmarkPairReceipt,
    LiveBenchmarkPairSeal,
    PairReceiptDisposition,
    PlannerDoctorLiveBenchmark,
    assert_no_fixture_decision_fields,
    build_default_live_cases,
    build_default_live_manifest,
    create_isolated_worktree,
    create_planner_doctor_live_benchmark,
    effective_workers,
    evidence_authority_for_source,
    materialize_hermetic_repository,
    scored_cell_count,
    skip_qualifies_for_promotion,
)

ROOT = Path(__file__).resolve().parents[2]
LIVE_MANIFEST_PATH = (
    ROOT / "test/fixtures/agent_supervisor/planner_doctor_live/manifest.json"
)
HOLDOUT_MANIFEST_PATH = (
    ROOT / "test/fixtures/agent_supervisor/planner_doctor_holdout/manifest.json"
)
BENCHMARK_POLICY_PATH = (
    ROOT / "config/agent_supervisor_planner_doctor_benchmark.json"
)
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py/agent_supervisor/validation/planner_doctor_live_benchmark.py"
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        assert key not in result, f"duplicate JSON key: {key}"
        result[key] = value
    return result


def _load(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


@pytest.fixture(scope="module")
def live_document() -> dict[str, Any]:
    return _load(LIVE_MANIFEST_PATH)


@pytest.fixture(scope="module")
def live_manifest(live_document: dict[str, Any]) -> LiveBenchmarkManifest:
    return LiveBenchmarkManifest.from_dict(live_document)


@pytest.fixture(scope="module")
def holdout_manifest() -> dict[str, Any]:
    return _load(HOLDOUT_MANIFEST_PATH)


@pytest.fixture(scope="module")
def benchmark_policy() -> dict[str, Any]:
    return _load(BENCHMARK_POLICY_PATH)


@pytest.fixture
def runner(tmp_path: Path) -> PlannerDoctorLiveBenchmark:
    engine = create_planner_doctor_live_benchmark(
        repo_root=ROOT,
        work_root=tmp_path / "work",
    )
    yield engine
    engine.close()


# ---------------------------------------------------------------------------
# Module / fixture contracts
# ---------------------------------------------------------------------------


def test_module_and_fixture_exist() -> None:
    assert MODULE_PATH.is_file()
    assert LIVE_MANIFEST_PATH.is_file()
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert "PlannerDoctorLiveBenchmark@1" in text
    assert "LiveBenchmarkPairReceipt@1" in text
    assert "PDR-070" in text
    assert "never reads fixture" in text.lower() or "fixture expected" in text.lower()


def test_live_manifest_schema_and_identity(
    live_document: dict[str, Any],
    live_manifest: LiveBenchmarkManifest,
) -> None:
    assert live_document["schema"] == LIVE_BENCHMARK_MANIFEST_SCHEMA
    assert live_document["interface"] == PLANNER_DOCTOR_LIVE_BENCHMARK_INTERFACE
    assert live_document["task_id"] == PRODUCER_TASK_ID
    assert live_document["manifest_cid"] == live_manifest.manifest_cid
    assert live_manifest.automatic_promotion_enabled is False
    assert live_manifest.fixture_expected_fields_are_not_execution_authority is True
    assert live_manifest.skips_cannot_qualify_promotion is True
    assert live_manifest.synthetic_results_are_execution_authority is False


def test_manifest_binds_holdout_and_policy(
    live_manifest: LiveBenchmarkManifest,
    holdout_manifest: dict[str, Any],
    benchmark_policy: dict[str, Any],
) -> None:
    assert live_manifest.holdout_manifest_cid == holdout_manifest["manifest_cid"]
    assert live_manifest.benchmark_policy_cid == benchmark_policy["policy_cid"]
    assert live_manifest.cache_strata == CACHE_STRATA
    assert live_manifest.requested_concurrency == REQUESTED_CONCURRENCY
    assert live_manifest.configured_maximum_workers == CONFIGURED_MAXIMUM_WORKERS
    assert live_manifest.scored_repetitions == SCORED_REPETITIONS
    assert live_manifest.primary_arm_ids == PRIMARY_ARM_IDS


def test_cases_cover_all_pair_families_without_expected_fields(
    live_document: dict[str, Any],
    live_manifest: LiveBenchmarkManifest,
) -> None:
    families = {case.pair_family for case in live_manifest.cases}
    assert families == set(PAIR_FAMILIES)
    assert len(live_manifest.cases) == 6
    for case in live_manifest.cases:
        assert case.promotion_eligible is False
        assert case.files
        assert case.holdout_case_id
        assert case.oracle_slot_id
        assert case.case_cid

    forbidden = {
        "expected",
        "expected_outcome",
        "expected_disposition",
        "expected_diagnosis",
        "expected_repair",
        "expected_completion",
        "gold",
        "golden",
        "oracle_body",
        "gold_outcome",
        "correct_disposition",
        "correct_diagnosis",
    }
    for mapping in _walk(live_document):
        assert forbidden.isdisjoint(mapping.keys())


def test_fixture_expected_fields_are_rejected() -> None:
    with pytest.raises(LiveBenchmarkError, match="decision field"):
        assert_no_fixture_decision_fields(
            {"case_id": "x", "expected_disposition": "succeed"}
        )
    with pytest.raises(LiveBenchmarkError, match="decision field"):
        LiveBenchmarkCase.from_dict(
            {
                "case_id": "bad",
                "pair_family": "plan-create",
                "execution_kind": "planner-create",
                "partition": "hermetic-live",
                "deterministic_seed": 1,
                "prompt_template_id": "p@1",
                "mutation_recipe_id": "none",
                "task_source_seed_id": "t@1",
                "oracle_slot_id": "oracle:x@1",
                "holdout_case_id": "pdr-dev-plan-create-control-plane",
                "files": [{"path": "a.py", "content": "x=1\n"}],
                "expected": {"disposition": "succeed"},
            }
        )


def test_forged_manifest_cid_rejected(live_document: dict[str, Any]) -> None:
    tampered = copy.deepcopy(live_document)
    tampered["manifest_cid"] = "baguqeera" + "b" * 52
    with pytest.raises(LiveBenchmarkError, match="manifest_cid"):
        LiveBenchmarkManifest.from_dict(tampered)


def test_denominator_mutations_rejected(live_manifest: LiveBenchmarkManifest) -> None:
    payload = live_manifest.to_dict()
    payload.pop("manifest_cid", None)
    bad = copy.deepcopy(payload)
    bad["requested_concurrency"] = [1, 2, 3]
    with pytest.raises(LiveBenchmarkError, match="requested_concurrency"):
        LiveBenchmarkManifest.from_dict(bad)

    bad2 = copy.deepcopy(payload)
    bad2["cache_strata"] = ["cold", "warm"]
    with pytest.raises(LiveBenchmarkError, match="cache_strata"):
        LiveBenchmarkManifest.from_dict(bad2)

    bad3 = copy.deepcopy(payload)
    bad3["automatic_promotion_enabled"] = True
    with pytest.raises(LiveBenchmarkError, match="automatic_promotion"):
        LiveBenchmarkManifest.from_dict(bad3)


# ---------------------------------------------------------------------------
# Evidence authority and skip rules
# ---------------------------------------------------------------------------


def test_conformance_only_labels_for_synthetic_benchmarks(
    live_manifest: LiveBenchmarkManifest,
) -> None:
    labels = set(live_manifest.conformance_only_sources)
    assert "deterministic-doctor-fixture-benchmark" in labels
    assert "supervisor-v2-synthetic-benchmark" in labels
    for source in CONFORMANCE_ONLY_EVIDENCE_SOURCES:
        assert (
            evidence_authority_for_source(source)
            is EvidenceAuthorityClass.CONFORMANCE_ONLY
        )
    for source in LIVE_EVIDENCE_SOURCES:
        assert evidence_authority_for_source(source) is EvidenceAuthorityClass.LIVE_SERVICE
    assert evidence_authority_for_source("skipped") is EvidenceAuthorityClass.SKIPPED
    assert skip_qualifies_for_promotion(ArmExecutionStatus.SKIPPED) is False
    assert skip_qualifies_for_promotion(ArmExecutionStatus.MEASURED) is True
    assert skip_qualifies_for_promotion("xfail") is False
    assert skip_qualifies_for_promotion("dry-run") is False


def test_scored_cell_denominator_matches_policy(
    live_manifest: LiveBenchmarkManifest,
    benchmark_policy: dict[str, Any],
) -> None:
    # 6 hermetic cases × 3 arms × 4 strata × 4 concurrency × 3 scored reps
    required = scored_cell_count(case_count=len(live_manifest.cases))
    assert required == 6 * 3 * 4 * 4 * 3
    assert required == 864
    # Policy preregisters 12 holdout cases → 1728 scored executions.
    policy_required = scored_cell_count(
        case_count=int(benchmark_policy["population"]["qualifying_case_count"])
    )
    assert policy_required == benchmark_policy["budgets"]["qualifying_run"][
        "scored_cell_executions"
    ]
    assert effective_workers(4) == 4
    assert effective_workers(8) == CONFIGURED_MAXIMUM_WORKERS
    assert effective_workers(4, admitted_dag_width=2) == 2


# ---------------------------------------------------------------------------
# Hermetic repository materialization
# ---------------------------------------------------------------------------


def test_materialize_hermetic_repository_and_worktree(
    live_manifest: LiveBenchmarkManifest,
    tmp_path: Path,
) -> None:
    case = live_manifest.cases[0]
    repo = materialize_hermetic_repository(
        case,
        parent=tmp_path / "repos",
        arm_id=ArmId.DETERMINISTIC_SYMBOLIC.value,
        stratum_id="cold",
    )
    try:
        assert (repo.root / "pkg" / "math_ops.py").is_file()
        assert repo.head_commit
        assert repo.tree_id
        assert repo.forest_cid
        worktree = create_isolated_worktree(
            repo, worktree_parent=tmp_path / "worktrees"
        )
        assert worktree.is_dir()
        assert (worktree / "pkg" / "math_ops.py").is_file()
        # Replayable forest across arms (same stratum family).
        repo2 = materialize_hermetic_repository(
            case,
            parent=tmp_path / "repos2",
            arm_id=ArmId.HYBRID_RESIDUAL.value,
            stratum_id="cold",
        )
        try:
            assert repo.tree_id == repo2.tree_id
            assert repo.forest_cid == repo2.forest_cid
            assert repo.head_commit == repo2.head_commit
        finally:
            repo2.cleanup()
    finally:
        repo.cleanup()


def test_delta_stratum_applies_mutation_overlay(
    live_manifest: LiveBenchmarkManifest,
    tmp_path: Path,
) -> None:
    case = next(c for c in live_manifest.cases if c.mutation_recipe_id != "none")
    base = materialize_hermetic_repository(
        case, parent=tmp_path / "base", stratum_id="cold"
    )
    delta = materialize_hermetic_repository(
        case, parent=tmp_path / "delta", stratum_id="delta", apply_delta=True
    )
    try:
        assert not (base.root / "pkg" / "delta_overlay.py").exists()
        assert (delta.root / "pkg" / "delta_overlay.py").is_file()
        assert base.forest_cid != delta.forest_cid
    finally:
        base.cleanup()
        delta.cleanup()


# ---------------------------------------------------------------------------
# Live service execution
# ---------------------------------------------------------------------------


def test_run_pair_invokes_real_services_and_matches_inputs(
    runner: PlannerDoctorLiveBenchmark,
) -> None:
    receipt = runner.run_pair(
        "live-hermetic-plan-create",
        stratum_id="cold",
        concurrency=1,
        repetition=0,
    )
    assert receipt.INTERFACE == LIVE_BENCHMARK_PAIR_RECEIPT_INTERFACE
    assert receipt.disposition is PairReceiptDisposition.PAIRED
    assert receipt.inputs_match_across_primary_arms is True
    assert receipt.promotion_eligible is False
    assert len(receipt.arm_receipts) == 3
    arm_ids = {r.arm_id for r in receipt.arm_receipts}
    assert arm_ids == set(PRIMARY_ARM_IDS)
    for arm in receipt.arm_receipts:
        assert arm.evidence_authority is EvidenceAuthorityClass.LIVE_SERVICE
        assert arm.status in {
            ArmExecutionStatus.MEASURED,
            ArmExecutionStatus.CAPABILITY_ABSTAINED,
            ArmExecutionStatus.FAILED,
        }
        assert arm.status is not ArmExecutionStatus.SKIPPED
        assert "PlanCreateService@1" in arm.service_interfaces_invoked
        assert arm.process_tree_terminated
        assert arm.capabilities_revoked
        assert arm.output_root_sealed
        assert arm.wall_seconds_measured
        assert arm.cache_namespace
    assert "public_hermetic_corpus_conformance_and_live_runner_only" in (
        receipt.reason_codes
    )


def test_doctor_and_security_and_steer_and_degradation_cases(
    runner: PlannerDoctorLiveBenchmark,
) -> None:
    doctor = runner.run_pair("live-hermetic-doctor-contract", concurrency=2)
    assert doctor.inputs_match_across_primary_arms
    for arm in doctor.arm_receipts:
        assert "DeterministicDoctorService@1" in arm.service_interfaces_invoked

    security = runner.run_pair("live-hermetic-security-ir", stratum_id="exact-warm")
    assert security.inputs_match_across_primary_arms
    for arm in security.arm_receipts:
        assert "DeterministicDoctorService@1" in arm.service_interfaces_invoked
        assert "PlanCreateService@1" in arm.service_interfaces_invoked

    steer = runner.run_pair("live-hermetic-plan-steer", stratum_id="restart")
    assert steer.inputs_match_across_primary_arms
    for arm in steer.arm_receipts:
        assert any("Steer" in iface or "steer" in iface.lower() for iface in arm.service_interfaces_invoked) or any(
            "PlanSteer" in iface for iface in arm.service_interfaces_invoked
        )

    degrade = runner.run_pair(
        "live-hermetic-capability-degradation", stratum_id="delta", concurrency=4
    )
    assert degrade.inputs_match_across_primary_arms
    for arm in degrade.arm_receipts:
        assert "DeterministicDoctorService@1" in arm.service_interfaces_invoked
        # Degradation without backends is typed abstention, not invented success.
        assert arm.typed_abstention or arm.status is ArmExecutionStatus.FAILED


def test_skips_cannot_qualify_promotion(
    runner: PlannerDoctorLiveBenchmark,
) -> None:
    receipt = runner.run_pair(
        "live-hermetic-plan-create",
        force_skip_arms=[ArmId.DETERMINISTIC_SYMBOLIC.value],
    )
    assert receipt.promotion_eligible is False
    assert "skip_not_promotion_eligible" in receipt.reason_codes
    skipped = [
        r for r in receipt.arm_receipts if r.status is ArmExecutionStatus.SKIPPED
    ]
    assert len(skipped) == 1
    assert skipped[0].evidence_authority is EvidenceAuthorityClass.SKIPPED
    assert skipped[0].promotion_eligible() is False


def test_arm_execution_skip_receipt_not_promotion_eligible() -> None:
    receipt = ArmExecutionReceipt(
        seal_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        arm_id="deterministic-symbolic",
        status=ArmExecutionStatus.SKIPPED,
        evidence_authority=EvidenceAuthorityClass.LIVE_SERVICE,  # coerced to SKIPPED
        service_interfaces_invoked=(),
        worktree_root_cid="baguqeerabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        output_root_cid="baguqeeracccccccccccccccccccccccccccccccccccccccccccccccc",
        service_disposition="skipped",
        service_reason_codes=("explicit_skip",),
        process_tree_terminated=True,
        capabilities_revoked=True,
        output_root_sealed=True,
    )
    assert receipt.evidence_authority is EvidenceAuthorityClass.SKIPPED
    assert receipt.promotion_eligible() is False


# ---------------------------------------------------------------------------
# Cache strata + concurrency replay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stratum", list(CACHE_STRATA))
@pytest.mark.parametrize("concurrency", list(REQUESTED_CONCURRENCY))
def test_stratum_and_concurrency_cells_are_paired(
    runner: PlannerDoctorLiveBenchmark,
    stratum: str,
    concurrency: int,
) -> None:
    receipt = runner.run_pair(
        "live-hermetic-transaction-rollback",
        stratum_id=stratum,
        concurrency=concurrency,
        repetition=0,
    )
    assert receipt.cache_stratum_id == stratum
    assert receipt.concurrency_requested == concurrency
    assert receipt.inputs_match_across_primary_arms is True
    assert receipt.disposition is PairReceiptDisposition.PAIRED
    for arm in receipt.arm_receipts:
        assert arm.effective_workers == effective_workers(concurrency)
        assert stratum in arm.cache_namespace or arm.cache_namespace


def test_pair_replay_is_identity_equivalent(
    runner: PlannerDoctorLiveBenchmark,
) -> None:
    first, second, match = runner.replay_pair(
        "live-hermetic-plan-create",
        stratum_id="cold",
        concurrency=2,
        repetition=1,
    )
    assert match is True
    assert first.pair_input_cid == second.pair_input_cid
    assert first.case_id == second.case_id
    assert [r.seal_cid for r in first.arm_receipts] == [
        r.seal_cid for r in second.arm_receipts
    ]
    # Pair input fields (excluding arm treatments) match across arms.
    assert first.inputs_match_across_primary_arms
    assert second.inputs_match_across_primary_arms


def test_all_cache_strata_and_concurrency_constants() -> None:
    assert CACHE_STRATA == ("cold", "exact-warm", "delta", "restart")
    assert REQUESTED_CONCURRENCY == (1, 2, 4, 6)
    assert CONFIGURED_MAXIMUM_WORKERS == 6
    assert set(CacheStratum)  # enum populated


# ---------------------------------------------------------------------------
# Pair seal and matrix
# ---------------------------------------------------------------------------


def test_pair_seal_excludes_arm_treatments_from_shared_fields(
    runner: PlannerDoctorLiveBenchmark,
    live_manifest: LiveBenchmarkManifest,
) -> None:
    case = live_manifest.case_by_id("live-hermetic-plan-create")
    seals = []
    for arm_id in PRIMARY_ARM_IDS:
        seal = runner.build_pair_seal(
            case,
            arm_id=arm_id,
            stratum_id="cold",
            concurrency=1,
            repetition=0,
            scored=True,
            repository_forest_cid="baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        seals.append(seal)
        assert seal.arm_id == arm_id
        assert seal.planner_doctor_mode
        assert seal.provider_call_permission
    shared = [s.pair_identity_fields() for s in seals]
    assert all(item == shared[0] for item in shared[1:])
    # Treatments differ.
    assert len({s.arm_id for s in seals}) == 3
    assert len({s.planner_doctor_mode for s in seals}) == 3


def test_run_matrix_subset_reports_incomplete_denominator(
    runner: PlannerDoctorLiveBenchmark,
) -> None:
    report = runner.run_matrix(
        case_ids=["live-hermetic-plan-create"],
        strata=["cold"],
        concurrency_values=[1],
        scored_repetitions=1,
        max_pairs=1,
    )
    assert report.incomplete is True
    assert report.promotion_eligible is False
    assert report.scored_cells_required == scored_cell_count(
        case_count=len(runner.manifest.cases)
    )
    assert report.scored_cells_observed < report.scored_cells_required
    assert "incomplete_required_cell_population" in report.reason_codes
    assert "deterministic-doctor-fixture-benchmark" in report.conformance_only_labels
    assert "supervisor-v2-synthetic-benchmark" in report.conformance_only_labels
    assert len(report.pair_receipts) == 1
    assert report.report_cid


def test_default_factory_loads_repo_fixture() -> None:
    engine = create_planner_doctor_live_benchmark(repo_root=ROOT)
    try:
        assert engine.manifest.task_id == PRODUCER_TASK_ID
        assert engine.INTERFACE == PLANNER_DOCTOR_LIVE_BENCHMARK_INTERFACE
        assert (
            engine.PAIR_RECEIPT_INTERFACE == LIVE_BENCHMARK_PAIR_RECEIPT_INTERFACE
        )
        assert len(engine.manifest.cases) == 6
        assert engine.label_conformance_only_sources()
    finally:
        engine.close()


def test_build_default_manifest_round_trip() -> None:
    manifest = build_default_live_manifest()
    cases = build_default_live_cases()
    assert len(cases) == 6
    again = LiveBenchmarkManifest.from_dict(manifest.to_dict())
    assert again.manifest_cid == manifest.manifest_cid
    assert [c.case_id for c in again.cases] == [c.case_id for c in cases]


def test_pair_receipt_schema_and_interfaces(
    runner: PlannerDoctorLiveBenchmark,
) -> None:
    receipt = runner.run_pair("live-hermetic-capability-degradation")
    payload = receipt.to_dict()
    assert payload["schema"] == LIVE_BENCHMARK_PAIR_RECEIPT_SCHEMA
    assert payload["interface"] == LIVE_BENCHMARK_PAIR_RECEIPT_INTERFACE
    assert payload["receipt_cid"] == receipt.receipt_cid
    assert payload["promotion_eligible"] is False


def test_execute_arm_never_reads_expected_from_case_dict(
    runner: PlannerDoctorLiveBenchmark,
    live_manifest: LiveBenchmarkManifest,
) -> None:
    """Guard: even if a caller smuggles expected into a case mapping, load fails."""

    case = live_manifest.cases[0]
    as_dict = case.to_dict()
    assert "expected" not in as_dict
    # Service path uses LiveBenchmarkCase objects; fixture decision keys blocked.
    with pytest.raises(LiveBenchmarkError):
        assert_no_fixture_decision_fields(
            {**as_dict, "expected_repair": {"patch": "nope"}}
        )


def test_holdout_case_ids_align_with_public_slots(
    live_manifest: LiveBenchmarkManifest,
    holdout_manifest: dict[str, Any],
) -> None:
    holdout_ids = {c["case_id"] for c in holdout_manifest["cases"]}
    for case in live_manifest.cases:
        assert case.holdout_case_id in holdout_ids
        slot = next(
            c for c in holdout_manifest["cases"] if c["case_id"] == case.holdout_case_id
        )
        assert case.oracle_slot_id == slot["oracle_slot_id"]
        assert case.pair_family == slot["pair_family"]
