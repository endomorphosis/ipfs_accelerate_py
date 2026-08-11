"""IVP-015: selected-versus-full-suite semantic fixture evaluation.

Acceptance coverage:

* Selected/full runs use fresh identical tree/environment/lock/fixture/policy
  snapshots
* Fixture ground-truth affected tests define false positives/negatives
* Full-suite failure comparison is a separate oracle
* Passing selected tests are not automatically false positives
* Flaky/order/outcome discrepancies are inconclusive
* Full-suite timeout/unavailable or absent corpus is not_measured, never zero
* Uncertain/uncovered selector or missing validation mapping requires broader
  suite before acceptance
* Equivalent controlled fixtures are labeled separately
* Evidence binds corpus/evaluated count/repository/policy/environment/selector
  identities and measured timing without asserting target success
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.verification.evaluation import (
    CANONICAL_CORPUS_ID,
    CORPUS_MANIFEST_NAME,
    REASON_BROADER_REQUIRED,
    REASON_CORPUS_ABSENT,
    REASON_FLAKY_OUTCOME,
    REASON_FULL_SUITE_TIMEOUT,
    REASON_FULL_SUITE_UNAVAILABLE,
    REASON_GROUND_TRUTH_OMISSION,
    REASON_ORACLE_FAILURE_NOT_OBSERVED,
    REASON_ORDER_DEPENDENT,
    REASON_OUTCOME_DISCREPANCY,
    REASON_PASSING_NOT_FALSE_POSITIVE,
    REASON_SELECTED_OUTSIDE_GROUND_TRUTH,
    REASON_SNAPSHOT_MISMATCH,
    REASON_TARGET_SUCCESS_NOT_ASSERTED,
    REASON_UNCERTAIN_SELECTOR,
    REASON_VALIDATION_MAPPING_MISSING,
    REASON_ZERO_EVALUATED,
    SELECTION_EVALUATION_EVIDENCE,
    TEST_SELECTION_EVALUATION_INTERFACE,
    TEST_SELECTION_EVALUATION_SCHEMA,
    ControlledSemanticFixture,
    EvaluationSnapshotIdentity,
    MeasurementStatus,
    ObservedTestOutcome,
    SuiteMode,
    SuiteObservation,
    SuiteRunStatus,
    TestSelectionEvaluation,
    compare_selected_with_full_suite,
    default_fixture_root,
    evaluate_controlled_fixture_corpus,
    evaluate_default_corpus,
    fresh_identical_observations,
    load_controlled_fixtures,
    make_suite_observation,
)
from ipfs_accelerate_py.agent_supervisor.verification.selection import (
    FallbackMode,
    select_affected_verification,
)


# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "test" / "fixtures" / "incremental_verification"
MINI_REPO = FIXTURE_ROOT / "mini_repo"

TEST_A = "tests/test_mod.py::test_fn"
TEST_B = "tests/test_mod.py::test_helper"
TEST_C = "tests/test_other.py::test_unrelated"
TEST_FAIL = "tests/test_mod.py::test_deliberately_fails"


def _by_id(
    fixtures: tuple[ControlledSemanticFixture, ...], fixture_id: str
) -> ControlledSemanticFixture:
    for item in fixtures:
        if item.fixture_id == fixture_id:
            return item
    raise AssertionError(f"fixture {fixture_id!r} not found")


@pytest.fixture(scope="module")
def corpus_fixtures() -> tuple[ControlledSemanticFixture, ...]:
    fixtures = load_controlled_fixtures(FIXTURE_ROOT, require_present=True)
    assert fixtures, "expected controlled corpus cases"
    return fixtures


# ---------------------------------------------------------------------------
# Interface / schema surface
# ---------------------------------------------------------------------------


def test_module_exports_required_interfaces() -> None:
    assert TEST_SELECTION_EVALUATION_INTERFACE == "TestSelectionEvaluation@1"
    assert TEST_SELECTION_EVALUATION_SCHEMA.endswith(
        "test-selection-evaluation@1"
    )
    assert SELECTION_EVALUATION_EVIDENCE == "ivp/test-selection-evaluation@1"
    assert CANONICAL_CORPUS_ID == "ivp-semantic-capsule-controlled-v1"
    assert default_fixture_root(REPO_ROOT) == FIXTURE_ROOT.resolve()


def test_corpus_manifest_present_and_loads(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    manifest_path = FIXTURE_ROOT / CORPUS_MANIFEST_NAME
    assert manifest_path.is_file()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["corpus_id"] == CANONICAL_CORPUS_ID
    assert payload["authoritative"] is False
    assert payload["target_success_asserted"] is False
    assert payload["case_count"] == len(corpus_fixtures)
    assert len(corpus_fixtures) >= 15
    kinds = {item.change_kind for item in corpus_fixtures}
    for required in (
        "direct_symbol",
        "transitive",
        "unrelated",
        "fixture_edge",
        "config_edge",
        "opaque",
        "dynamic",
        "deliberately_failing",
        "false_negative_seed",
        "false_positive_seed",
        "flaky",
        "order_dependent",
        "full_suite_timeout",
        "full_suite_unavailable",
        "validation_mapping",
        "equivalent_controlled",
    ):
        assert required in kinds


# ---------------------------------------------------------------------------
# Fresh identical snapshots
# ---------------------------------------------------------------------------


def test_selected_and_full_use_fresh_identical_snapshots(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    fixture = _by_id(corpus_fixtures, "direct-symbol-fn")
    assert fixture.selected_observation is not None
    assert fixture.full_suite_observation is not None
    selected = fixture.selected_observation
    full = fixture.full_suite_observation
    # Distinct observation objects, identical identity.
    assert selected is not full
    assert selected.snapshot is not full.snapshot
    assert selected.snapshot.matches(full.snapshot)
    assert selected.snapshot.matches(fixture.snapshot)
    for attr in (
        "tree_id",
        "environment_id",
        "lock_id",
        "fixture_id",
        "policy_id",
    ):
        assert getattr(selected.snapshot, attr) == getattr(
            full.snapshot, attr
        )
        assert getattr(selected.snapshot, attr) == getattr(
            fixture.snapshot, attr
        )

    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.snapshots_identical is True
    assert result.measurement_status is MeasurementStatus.MEASURED
    assert result.authoritative is False
    assert result.target_success_asserted is False
    assert REASON_TARGET_SUCCESS_NOT_ASSERTED in result.reason_codes


def test_fresh_identical_observations_helper() -> None:
    snap = EvaluationSnapshotIdentity(
        tree_id="tree:t",
        environment_id="env:e",
        lock_id="lock:l",
        fixture_id="fixture:f",
        policy_id="policy:p",
        repository_id="repo:r",
    )
    selected, full = fresh_identical_observations(
        snapshot=snap,
        selected_outcomes={TEST_A: "passed"},
        full_outcomes={TEST_A: "passed", TEST_C: "passed"},
        selected_duration_ms=3,
        full_duration_ms=9,
    )
    assert selected.mode is SuiteMode.SELECTED
    assert full.mode is SuiteMode.FULL_SUITE
    assert selected.snapshot.matches(full.snapshot)
    assert selected.duration_ms == 3
    assert full.duration_ms == 9


def test_snapshot_mismatch_is_inconclusive() -> None:
    snap_a = EvaluationSnapshotIdentity(
        tree_id="tree:a",
        environment_id="env:e",
        lock_id="lock:l",
        fixture_id="fx:mismatch",
        policy_id="policy:p",
    )
    snap_b = EvaluationSnapshotIdentity(
        tree_id="tree:b",
        environment_id="env:e",
        lock_id="lock:l",
        fixture_id="fx:mismatch",
        policy_id="policy:p",
    )
    fixture = ControlledSemanticFixture(
        fixture_id="fx:mismatch",
        change_kind="direct_symbol",
        snapshot=snap_a,
        ground_truth_affected_tests=(TEST_A,),
        all_tests=(TEST_A, TEST_C),
        forced_selected_tests=(TEST_A,),
        selected_observation=make_suite_observation(
            mode=SuiteMode.SELECTED,
            snapshot=snap_a,
            test_outcomes={TEST_A: "passed"},
        ),
        full_suite_observation=make_suite_observation(
            mode=SuiteMode.FULL_SUITE,
            snapshot=snap_b,
            test_outcomes={TEST_A: "passed", TEST_C: "passed"},
        ),
    )
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.snapshots_identical is False
    assert result.measurement_status is MeasurementStatus.INCONCLUSIVE
    assert REASON_SNAPSHOT_MISMATCH in result.reason_codes


# ---------------------------------------------------------------------------
# Ground-truth false negatives / positives
# ---------------------------------------------------------------------------


def test_ground_truth_false_negative_from_omission(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    fixture = _by_id(corpus_fixtures, "seeded-false-negative")
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.measurement_status is MeasurementStatus.MEASURED
    assert TEST_FAIL in result.ground_truth_false_negatives
    assert TEST_FAIL in result.false_negative_tests
    # Separate oracle: full-suite failure not selected.
    assert TEST_FAIL in result.full_suite_oracle_false_negatives
    assert TEST_FAIL in result.full_suite_failures
    assert result.false_negative_count == len(result.false_negative_tests)
    assert result.false_negative_count is not None
    assert result.false_negative_count >= 1
    assert REASON_GROUND_TRUTH_OMISSION in result.reason_codes
    assert REASON_ORACLE_FAILURE_NOT_OBSERVED in result.reason_codes


def test_ground_truth_false_positive_not_from_passing(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    fixture = _by_id(corpus_fixtures, "seeded-false-positive")
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.measurement_status is MeasurementStatus.MEASURED
    assert TEST_C in result.ground_truth_false_positives
    assert TEST_C in result.false_positive_tests
    # TEST_A is ground-truth and passes — not a false positive.
    assert TEST_A not in result.false_positive_tests
    assert REASON_SELECTED_OUTSIDE_GROUND_TRUTH in result.reason_codes
    assert REASON_PASSING_NOT_FALSE_POSITIVE in result.reason_codes


def test_direct_symbol_zero_false_negatives_when_measured(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    fixture = _by_id(corpus_fixtures, "direct-symbol-fn")
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.measurement_status is MeasurementStatus.MEASURED
    assert result.false_negative_count == 0
    assert result.false_positive_count == 0
    assert result.false_negative_tests == ()
    assert result.false_positive_tests == ()
    assert TEST_A in result.selected_tests
    assert TEST_A in result.ground_truth_affected_tests


def test_deliberately_failing_observed_is_not_false_negative(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    fixture = _by_id(corpus_fixtures, "deliberately-failing-observed")
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.measurement_status is MeasurementStatus.MEASURED
    assert TEST_FAIL in result.full_suite_failures
    assert TEST_FAIL in result.selected_failures
    assert TEST_FAIL not in result.false_negative_tests
    assert result.false_negative_count == 0


def test_full_suite_oracle_separate_from_ground_truth() -> None:
    """Oracle FN without GT membership is recorded separately and aggregated."""

    snap = EvaluationSnapshotIdentity(
        tree_id="tree:oracle",
        environment_id="env:e",
        lock_id="lock:l",
        fixture_id="fx:oracle-only",
        policy_id="policy:p",
    )
    # GT says only TEST_A; selection has TEST_A; full suite also fails TEST_C
    # which was never selected → oracle FN for TEST_C, not GT FN.
    fixture = ControlledSemanticFixture(
        fixture_id="fx:oracle-only",
        change_kind="deliberately_failing",
        snapshot=snap,
        ground_truth_affected_tests=(TEST_A,),
        all_tests=(TEST_A, TEST_C),
        forced_selected_tests=(TEST_A,),
        selected_observation=make_suite_observation(
            mode=SuiteMode.SELECTED,
            snapshot=snap,
            test_outcomes={TEST_A: "passed"},
        ),
        full_suite_observation=make_suite_observation(
            mode=SuiteMode.FULL_SUITE,
            snapshot=snap,
            test_outcomes={TEST_A: "passed", TEST_C: "failed"},
        ),
    )
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.ground_truth_false_negatives == ()
    assert TEST_C in result.full_suite_oracle_false_negatives
    assert TEST_C in result.false_negative_tests
    assert TEST_C in result.full_suite_failures


# ---------------------------------------------------------------------------
# Inconclusive classifications
# ---------------------------------------------------------------------------


def test_flaky_outcomes_are_inconclusive(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    fixture = _by_id(corpus_fixtures, "flaky-outcome")
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.measurement_status is MeasurementStatus.INCONCLUSIVE
    assert TEST_A in result.inconclusive_tests
    assert REASON_FLAKY_OUTCOME in result.reason_codes
    assert REASON_FLAKY_OUTCOME in result.inconclusive_reasons


def test_order_dependent_outcomes_are_inconclusive(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    fixture = _by_id(corpus_fixtures, "order-dependent")
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.measurement_status is MeasurementStatus.INCONCLUSIVE
    assert TEST_B in result.inconclusive_tests
    assert REASON_ORDER_DEPENDENT in result.reason_codes


def test_outcome_discrepancy_is_inconclusive(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    fixture = _by_id(corpus_fixtures, "outcome-discrepancy")
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.measurement_status is MeasurementStatus.INCONCLUSIVE
    assert TEST_A in result.inconclusive_tests
    assert REASON_OUTCOME_DISCREPANCY in result.reason_codes
    # Not automatically counted as a hard oracle FN when selected observed pass.
    assert TEST_A not in result.full_suite_oracle_false_negatives


# ---------------------------------------------------------------------------
# not_measured (never zero)
# ---------------------------------------------------------------------------


def test_full_suite_timeout_is_not_measured_never_zero(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    fixture = _by_id(corpus_fixtures, "full-suite-timeout")
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.measurement_status is MeasurementStatus.NOT_MEASURED
    assert result.false_negative_count is None
    assert result.false_positive_count is None
    assert result.false_negative_count != 0  # None, not zero
    assert REASON_FULL_SUITE_TIMEOUT in result.not_measured_reasons
    assert REASON_FULL_SUITE_TIMEOUT in result.reason_codes
    assert result.evaluated is False


def test_full_suite_unavailable_is_not_measured_never_zero(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    fixture = _by_id(corpus_fixtures, "full-suite-unavailable")
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.measurement_status is MeasurementStatus.NOT_MEASURED
    assert result.false_negative_count is None
    assert result.false_positive_count is None
    assert REASON_FULL_SUITE_UNAVAILABLE in result.not_measured_reasons


def test_absent_canonical_corpus_is_not_measured_never_zero(
    tmp_path: Path,
) -> None:
    empty_root = tmp_path / "missing_corpus"
    empty_root.mkdir()
    fixtures = load_controlled_fixtures(empty_root, require_present=False)
    assert fixtures == ()
    summary = evaluate_controlled_fixture_corpus(
        fixtures,
        corpus_id=CANONICAL_CORPUS_ID,
        corpus_present=False,
    )
    assert summary.measurement_status is MeasurementStatus.NOT_MEASURED
    assert summary.evaluated_count == 0
    assert summary.total_false_negatives is None
    assert summary.total_false_positives is None
    assert summary.total_false_negatives != 0
    assert REASON_CORPUS_ABSENT in summary.reason_codes
    assert summary.authoritative is False
    assert summary.target_success_asserted is False


def test_zero_evaluated_fixtures_is_not_measured_never_zero() -> None:
    summary = evaluate_controlled_fixture_corpus(
        (),
        corpus_id=CANONICAL_CORPUS_ID,
        corpus_present=True,
    )
    assert summary.measurement_status is MeasurementStatus.NOT_MEASURED
    assert summary.evaluated_count == 0
    assert summary.total_false_negatives is None
    assert summary.total_false_positives is None
    assert REASON_ZERO_EVALUATED in summary.reason_codes


def test_fixture_corpus_present_false_on_case() -> None:
    snap = EvaluationSnapshotIdentity(
        tree_id="tree:x",
        environment_id="env:e",
        lock_id="lock:l",
        fixture_id="fx:absent",
        policy_id="policy:p",
    )
    fixture = ControlledSemanticFixture(
        fixture_id="fx:absent",
        change_kind="direct_symbol",
        snapshot=snap,
        ground_truth_affected_tests=(TEST_A,),
        all_tests=(TEST_A,),
        corpus_present=False,
        forced_selected_tests=(TEST_A,),
        selected_observation=make_suite_observation(
            mode=SuiteMode.SELECTED,
            snapshot=snap,
            test_outcomes={TEST_A: "passed"},
        ),
        full_suite_observation=make_suite_observation(
            mode=SuiteMode.FULL_SUITE,
            snapshot=snap,
            test_outcomes={TEST_A: "passed"},
        ),
    )
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.measurement_status is MeasurementStatus.NOT_MEASURED
    assert result.false_negative_count is None
    assert REASON_CORPUS_ABSENT in result.not_measured_reasons


# ---------------------------------------------------------------------------
# Broader / full suite before acceptance
# ---------------------------------------------------------------------------


def test_opaque_requires_broader_before_acceptance(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    fixture = _by_id(corpus_fixtures, "opaque-critical")
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.broader_suite_required_before_acceptance is True
    assert (
        REASON_BROADER_REQUIRED in result.reason_codes
        or REASON_UNCERTAIN_SELECTOR in result.reason_codes
        or result.selection_broader_required
    )
    assert REASON_BROADER_REQUIRED in result.acceptance_blocked_reasons or (
        result.selection_broader_required
        or result.selection_full_suite_required
    )


def test_dynamic_requires_broader_before_acceptance(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    fixture = _by_id(corpus_fixtures, "dynamic-critical")
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.broader_suite_required_before_acceptance is True


def test_missing_validation_mapping_requires_broader(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    fixture = _by_id(corpus_fixtures, "validation-mapping-incomplete")
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.broader_suite_required_before_acceptance is True
    assert REASON_VALIDATION_MAPPING_MISSING in result.reason_codes
    assert REASON_VALIDATION_MAPPING_MISSING in result.acceptance_blocked_reasons
    # Selection itself should also report broader requirement.
    selection = select_affected_verification(
        changed_symbols=fixture.changed_symbols,
        edges=list(fixture.edges),
        validation_selection=fixture.validation_selection,
        catalog=fixture.catalog,
        policy=fixture.policy,
    )
    assert selection.broader_selection_required is True


def test_uncovered_selector_requires_broader() -> None:
    snap = EvaluationSnapshotIdentity(
        tree_id="tree:u",
        environment_id="env:e",
        lock_id="lock:l",
        fixture_id="fx:uncovered",
        policy_id="policy:p",
    )
    fixture = ControlledSemanticFixture(
        fixture_id="fx:uncovered",
        change_kind="opaque",
        snapshot=snap,
        ground_truth_affected_tests=(TEST_A,),
        all_tests=(TEST_A, TEST_C),
        changed_symbols=("pkg.mod.fn",),
        edges=(
            {
                "source": "pkg.mod.fn",
                "target": TEST_A,
                "kind": "tested_by",
                "disposition": "exact",
            },
        ),
        uncovered_symbols=("pkg.mod.fn",),
        catalog={"tests": [TEST_A, TEST_C]},
        policy={"critical_uncertainty_requires_full_suite": False},
        forced_selected_tests=(TEST_A,),
        selected_observation=make_suite_observation(
            mode=SuiteMode.SELECTED,
            snapshot=snap,
            test_outcomes={TEST_A: "passed"},
        ),
        full_suite_observation=make_suite_observation(
            mode=SuiteMode.FULL_SUITE,
            snapshot=snap,
            test_outcomes={TEST_A: "passed", TEST_C: "passed"},
        ),
    )
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.broader_suite_required_before_acceptance is True
    assert REASON_UNCERTAIN_SELECTOR in result.reason_codes or (
        result.selection_broader_required
    )


# ---------------------------------------------------------------------------
# Equivalent fixtures labeled separately
# ---------------------------------------------------------------------------


def test_equivalent_controlled_fixtures_labeled_separately(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    a = _by_id(corpus_fixtures, "equivalent-controlled-a")
    b = _by_id(corpus_fixtures, "equivalent-controlled-b")
    assert a.fixture_id != b.fixture_id
    assert a.equivalence_label == b.equivalence_label
    assert a.equivalence_label == "equivalent-pair:mod-fn@1"
    ra = compare_selected_with_full_suite(fixture=a)
    rb = compare_selected_with_full_suite(fixture=b)
    assert ra.fixture_id != rb.fixture_id
    assert ra.equivalence_label == rb.equivalence_label
    assert ra.measurement_status is MeasurementStatus.MEASURED
    assert rb.measurement_status is MeasurementStatus.MEASURED


# ---------------------------------------------------------------------------
# Evidence bindings / corpus aggregate
# ---------------------------------------------------------------------------


def test_evidence_binds_identities_and_timing_without_target_success(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    fixture = _by_id(corpus_fixtures, "direct-symbol-fn")
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.evidence == SELECTION_EVALUATION_EVIDENCE
    assert result.interface == TEST_SELECTION_EVALUATION_INTERFACE
    assert result.corpus_id == CANONICAL_CORPUS_ID
    assert result.repository_id
    assert result.policy_id
    assert result.environment_id
    assert result.selector_identity
    assert result.selected_duration_ms >= 0
    assert result.full_suite_duration_ms >= 0
    assert result.evaluation_duration_ms >= 0
    assert result.authoritative is False
    assert result.target_success_asserted is False
    payload = result.to_dict()
    assert payload["authoritative"] is False
    assert payload["target_success_asserted"] is False
    assert payload["false_negative_count"] == 0
    assert "content_id" in payload
    # Round-trip.
    restored = TestSelectionEvaluation.from_dict(payload)
    assert restored.fixture_id == result.fixture_id
    assert restored.measurement_status is result.measurement_status
    assert restored.false_negative_count == result.false_negative_count


def test_evaluate_default_corpus_summary(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    summary = evaluate_default_corpus(FIXTURE_ROOT)
    assert summary.corpus_id == CANONICAL_CORPUS_ID
    assert summary.corpus_present is True
    assert summary.evaluated_count == len(corpus_fixtures)
    assert summary.evaluated_count > 0
    assert summary.measurement_status is MeasurementStatus.MEASURED
    assert summary.measured_count >= 1
    assert summary.not_measured_count >= 1  # timeout + unavailable cases
    assert summary.total_false_negatives is not None
    assert summary.total_false_positives is not None
    assert "equivalent-pair:mod-fn@1" in summary.equivalence_labels
    assert summary.authoritative is False
    assert summary.target_success_asserted is False
    assert summary.repository_id
    assert summary.policy_id
    assert summary.environment_id
    assert summary.evaluation_duration_ms >= 0
    payload = summary.to_dict()
    assert payload["evaluated_count"] == summary.evaluated_count
    assert len(payload["evaluations"]) == summary.evaluated_count


def test_selection_integrates_with_affected_selector(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    fixture = _by_id(corpus_fixtures, "transitive-helper")
    selection = select_affected_verification(
        changed_symbols=fixture.changed_symbols,
        edges=list(fixture.edges),
        catalog=fixture.catalog,
        policy=fixture.policy,
    )
    # Without forced selected, evaluation uses selector output.
    bare = ControlledSemanticFixture.from_value(
        {
            **fixture.to_dict(),
            "forced_selected_tests": None,
        }
    )
    result = compare_selected_with_full_suite(
        fixture=bare, selection=selection
    )
    assert result.measurement_status is MeasurementStatus.MEASURED
    assert set(selection.selected_tests).issubset(set(result.selected_tests) | set(selection.affected_tests) | set(selection.fallback_tests))
    for node in fixture.ground_truth_affected_tests:
        assert node in selection.selected_tests or selection.broader_selection_required


# ---------------------------------------------------------------------------
# Real mini-repo execution (controlled pytest repository)
# ---------------------------------------------------------------------------


def _collect_pytest_node_ids(repo: Path) -> list[str]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            str(repo / "tests"),
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    nodes: list[str] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if "::" not in line:
            continue
        node = line.split()[0] if line.split() else line
        if "::" not in node:
            continue
        # Normalize to tests/... form when present.
        if "tests/" in node:
            node = node[node.index("tests/") :]
        if any(
            node.endswith(suffix)
            for suffix in (
                "::test_fn",
                "::test_helper",
                "::test_deliberately_fails",
                "::test_unrelated",
            )
        ) or node.startswith("tests/"):
            nodes.append(node)
    return sorted(set(nodes))


def _run_pytest_nodes(
    repo: Path, node_ids: list[str] | None
) -> dict[str, str]:
    """Run selected nodes (or full suite) and map node id → pass/fail."""

    argv = [sys.executable, "-m", "pytest", "-q", "--tb=no"]
    if node_ids:
        argv.extend(node_ids)
    else:
        argv.append(str(repo / "tests"))
    proc = subprocess.run(
        argv,
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    # Parse short pytest summary lines of form "FAILED tests/...::name"
    outcomes: dict[str, str] = {}
    # Default: if we selected specific nodes, start as passed then mark fails.
    targets = list(node_ids or [])
    if not targets:
        # Full suite: collect first.
        targets = _collect_pytest_node_ids(repo)
    for node in targets:
        outcomes[node] = "passed"
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    for line in combined.splitlines():
        line = line.strip()
        for prefix, status in (
            ("FAILED ", "failed"),
            ("ERROR ", "error"),
            ("PASSED ", "passed"),
        ):
            if line.startswith(prefix):
                rest = line[len(prefix) :].strip()
                node = rest.split()[0] if rest.split() else rest
                if "tests/" in node:
                    node = node[node.index("tests/") :]
                if "::" in node:
                    outcomes[node] = status
    # If process failed hard with no parse, mark all selected failed.
    if proc.returncode not in (0, 1) and not any(
        v == "failed" for v in outcomes.values()
    ):
        for node in targets:
            outcomes[node] = "error"
    return outcomes


def test_real_mini_repo_selected_versus_full_suite(tmp_path: Path) -> None:
    """Run a real controlled pytest repository for selected + full observations.

    Uses the shared mini-repo recipe (not production authority). Selected and
    full observations bind identical snapshot identities.
    """

    if not MINI_REPO.is_dir():
        pytest.skip("mini_repo fixture absent")

    work = tmp_path / "mini_repo"
    shutil.copytree(MINI_REPO, work)

    # Fresh snapshot identity shared by both runs.
    snap = EvaluationSnapshotIdentity(
        tree_id="tree:mini-repo-real",
        environment_id="env:mini-hermetic",
        lock_id="lock:mini-none",
        fixture_id="mini-repo-direct-fn",
        policy_id="policy:mini-eval@1",
        repository_id="repository:mini-repo@1",
    )

    all_nodes = _collect_pytest_node_ids(work)
    assert any("test_fn" in n for n in all_nodes)
    # Map collected nodes to short forms used in recipes when possible.
    node_fn = next(n for n in all_nodes if n.endswith("::test_fn"))
    node_fail = next(
        n for n in all_nodes if n.endswith("::test_deliberately_fails")
    )
    node_unrelated = next(
        n for n in all_nodes if n.endswith("::test_unrelated")
    )

    selected_nodes = [node_fn]
    selected_outcomes = _run_pytest_nodes(work, selected_nodes)
    full_outcomes = _run_pytest_nodes(work, None)

    assert selected_outcomes.get(node_fn) == "passed"
    assert full_outcomes.get(node_fail) == "failed"

    selected_obs, full_obs = fresh_identical_observations(
        snapshot=snap,
        selected_outcomes=selected_outcomes,
        full_outcomes=full_outcomes,
        selected_duration_ms=1,
        full_duration_ms=2,
        selected_selector_identity="selector:mini:selected",
        full_selector_identity="selector:mini:full",
    )
    assert selected_obs.snapshot.matches(full_obs.snapshot)

    # GT: only test_fn is affected by a direct symbol change of fn.
    # Full suite still fails deliberately_fails (pre-existing), which is NOT
    # a mutation-caused failure for this change — but as an oracle observation
    # the evaluation will flag it if not selected. For this real-run case we
    # include the failing test in GT so the deliberate failure is expected.
    fixture = ControlledSemanticFixture(
        fixture_id="mini-repo-direct-fn",
        change_kind="direct_symbol",
        snapshot=snap,
        ground_truth_affected_tests=(node_fn,),
        all_tests=tuple(sorted(full_outcomes)),
        changed_symbols=("pkg.mod.fn",),
        edges=(
            {
                "source": "pkg.mod.fn",
                "target": node_fn,
                "kind": "tested_by",
                "disposition": "exact",
            },
        ),
        catalog={"tests": list(sorted(full_outcomes))},
        policy={"critical_uncertainty_requires_full_suite": False},
        forced_selected_tests=(node_fn,),
        selected_observation=selected_obs,
        full_suite_observation=full_obs,
        description="Real mini-repo differential observation",
    )
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.snapshots_identical is True
    assert result.measurement_status is MeasurementStatus.MEASURED
    assert result.authoritative is False
    assert result.target_success_asserted is False
    # Direct GT selection is correct for node_fn.
    assert node_fn in result.selected_tests
    assert node_fn not in result.ground_truth_false_negatives
    # Pre-existing full-suite failure outside GT is a separate oracle FN.
    assert node_fail in result.full_suite_failures
    assert node_fail in result.full_suite_oracle_false_negatives
    assert node_unrelated not in result.false_positive_tests
    assert result.selector_identity
    assert result.full_suite_duration_ms >= 0


def test_not_measured_serialization_omits_zero_counts() -> None:
    result = TestSelectionEvaluation(
        fixture_id="fx:nm",
        measurement_status=MeasurementStatus.NOT_MEASURED,
        false_negative_tests=(),
        false_positive_tests=(),
        false_negative_count=0,  # should be forced to None
        false_positive_count=0,
        not_measured_reasons=(REASON_FULL_SUITE_TIMEOUT,),
    )
    assert result.false_negative_count is None
    assert result.false_positive_count is None
    payload = result.to_dict()
    assert payload["false_negative_count"] is None
    assert payload["false_positive_count"] is None
    assert payload["measurement_status"] == "not_measured"


def test_compare_accepts_mapping_inputs() -> None:
    snap = {
        "tree_id": "tree:m",
        "environment_id": "env:e",
        "lock_id": "lock:l",
        "fixture_id": "fx:map",
        "policy_id": "policy:p",
        "repository_id": "repo:m",
    }
    fixture = {
        "fixture_id": "fx:map",
        "change_kind": "direct_symbol",
        "snapshot": snap,
        "ground_truth_affected_tests": [TEST_A],
        "all_tests": [TEST_A, TEST_C],
        "forced_selected_tests": [TEST_A],
        "selected_observation": {
            "mode": "selected",
            "snapshot": snap,
            "suite_status": "completed",
            "test_outcomes": {TEST_A: "passed"},
            "duration_ms": 1,
        },
        "full_suite_observation": {
            "mode": "full_suite",
            "snapshot": snap,
            "suite_status": "completed",
            "test_outcomes": {TEST_A: "passed", TEST_C: "passed"},
            "duration_ms": 2,
        },
    }
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.measurement_status is MeasurementStatus.MEASURED
    assert result.false_negative_count == 0
    assert result.false_positive_count == 0
    assert result.repository_id == "repo:m"


def test_unrelated_edit_has_empty_exact_selection(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    fixture = _by_id(corpus_fixtures, "unrelated-edit")
    result = compare_selected_with_full_suite(fixture=fixture)
    assert result.measurement_status is MeasurementStatus.MEASURED
    assert result.selected_tests == ()
    assert result.false_negative_count == 0
    assert result.false_positive_count == 0


def test_fixture_and_config_edge_kinds_present(
    corpus_fixtures: tuple[ControlledSemanticFixture, ...],
) -> None:
    fx = _by_id(corpus_fixtures, "fixture-edge-change")
    cfg = _by_id(corpus_fixtures, "config-edge-change")
    r_fx = compare_selected_with_full_suite(fixture=fx)
    r_cfg = compare_selected_with_full_suite(fixture=cfg)
    assert r_fx.measurement_status is MeasurementStatus.MEASURED
    assert r_cfg.measurement_status is MeasurementStatus.MEASURED
    assert r_fx.false_negative_count == 0
    assert r_cfg.false_negative_count == 0
