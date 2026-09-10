"""ASEH-015 live shadow/canary cohort admission.

The sealed manifest binds an immutable enrollment deadline. Qualification
requires ten distinct newly encountered live tasks with real provider and
verifier evidence. Shadow stays non-mutating and precedes canary. Fixtures
and simulations cannot satisfy live. Honest shortfalls emit a typed
insufficiency or non-admission receipt. This module does not import sibling
tests.
"""

from __future__ import annotations

import ast
import copy
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.control.live_cohort_admission import (
    CID_RE,
    CLOSED_DISPOSITIONS,
    ENROLLMENT_DEADLINE,
    LIVE_COHORT_GATES,
    LIVE_COHORT_MANIFEST_SCHEMA,
    LIVE_ENROLLMENT_DEADLINE_MAXIMUM_DAYS,
    LIVE_MINIMUM,
    LiveCohortAdmissionError,
    LiveCohortAdmissionPolicy,
    LiveCohortAdmissionService,
    LiveCohortReason,
    MANIFEST_PATH,
    REQUIRED_PROVENANCE_FIELDS,
    SEALED_AT,
    TaskDisposition,
    admit_cohort,
    admit_task,
    live_task_evidence_payload,
    parse_utc,
    seal_enrollment_deadline,
    sealed_policy,
    verify_sealed_artifacts,
    write_sealed_artifacts,
)


SIBLING_TEST_PREFIXES: tuple[str, ...] = (
    "test.api.agent_supervisor.efficiency_state_hardening.test_",
    "test.api.test_agent_supervisor_",
)

write_sealed_artifacts()


def _imported_module_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def _shadow_tasks(count: int = LIVE_MINIMUM) -> list[dict[str, Any]]:
    return [live_task_evidence_payload(index) for index in range(1, count + 1)]


def test_module_installs_without_sibling_test_imports() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    imported = _imported_module_names(ast.parse(source))
    sibling_stems = {
        path.stem
        for path in Path(__file__).parent.glob("test_*.py")
        if path.name != Path(__file__).name
    }
    for name in imported:
        assert not name.startswith("test."), name
        assert name.rsplit(".", 1)[-1] not in sibling_stems
        for prefix in SIBLING_TEST_PREFIXES:
            assert not name.startswith(prefix), name
    assert "ipfs_accelerate_py.agent_supervisor.control.live_cohort_admission" in imported


def test_sealed_manifest_binds_immutable_deadline_and_honest_insufficiency() -> None:
    verified = verify_sealed_artifacts()
    manifest = write_sealed_artifacts()
    payload = verify_sealed_artifacts(manifest)
    assert payload == verified
    loaded = verify_sealed_artifacts()
    assert loaded["count"] == 0
    assert loaded["qualification"] is False
    assert loaded["disposition"] == "insufficient_evidence"
    assert loaded["enrollment_deadline"] == ENROLLMENT_DEADLINE
    service = LiveCohortAdmissionService()
    snapshot = service.seal_manifest()
    assert snapshot["schema"] == LIVE_COHORT_MANIFEST_SCHEMA
    assert snapshot["status"] == "sealed"
    assert snapshot["live"] is False
    assert snapshot["authority"] is False
    assert snapshot["qualification"] is False
    assert snapshot["count"] == 0
    assert snapshot["minimum"] == LIVE_MINIMUM
    assert snapshot["enrollment_deadline"] == ENROLLMENT_DEADLINE
    assert snapshot["enrollment_deadline_immutable"] is True
    assert snapshot["enrollment_deadline_maximum_days"] == LIVE_ENROLLMENT_DEADLINE_MAXIMUM_DAYS
    assert snapshot["sealed_at"] == SEALED_AT
    assert snapshot["shadow_before_canary"] is True
    assert snapshot["canary_requires_safety_admission"] is True
    assert snapshot["fixtures_satisfy_live"] is False
    assert snapshot["simulation_satisfies_live"] is False
    assert snapshot["canary_mutation_permitted"] is False
    assert snapshot["shadow_admitted"] is False
    assert snapshot["canary_admitted"] is False
    assert snapshot["hermetic_sufficient_for_production_promotion"] is False
    assert snapshot["historical_sufficient_for_production_promotion"] is False
    assert snapshot["population_kind"] == "new_live_shadow_canary"
    assert snapshot["population_status"] == "enrolling"
    assert snapshot["disposition"] in CLOSED_DISPOSITIONS
    assert snapshot["disposition"] == "insufficient_evidence"
    assert LiveCohortReason.NOT_YET_MEASURED.value in snapshot["reasons"]
    assert tuple(snapshot["required_gates"]) == LIVE_COHORT_GATES
    assert tuple(snapshot["provenance_fields"]) == REQUIRED_PROVENANCE_FIELDS
    assert snapshot["tasks"] == []
    assert MANIFEST_PATH.is_file()
    assert snapshot["identity"] == loaded["identity"]
    assert CID_RE.fullmatch(str(snapshot["identity"]))


def test_enrollment_deadline_is_sealed_and_bounded_to_thirty_calendar_days() -> None:
    deadline = seal_enrollment_deadline(
        sealed_at=SEALED_AT,
        enrollment_deadline=ENROLLMENT_DEADLINE,
    )
    assert deadline == ENROLLMENT_DEADLINE
    start = parse_utc(SEALED_AT, name="sealed_at")
    bound = start + timedelta(days=LIVE_ENROLLMENT_DEADLINE_MAXIMUM_DAYS)
    assert parse_utc(deadline, name="enrollment_deadline") == bound
    with pytest.raises(LiveCohortAdmissionError, match="30 calendar day"):
        seal_enrollment_deadline(
            sealed_at=SEALED_AT,
            enrollment_deadline="2026-10-03T00:00:00Z",
        )
    with pytest.raises(LiveCohortAdmissionError, match="must be after sealed_at"):
        seal_enrollment_deadline(sealed_at=SEALED_AT, enrollment_deadline=SEALED_AT)
    with pytest.raises(LiveCohortAdmissionError, match="sealed UTC timestamp"):
        seal_enrollment_deadline(sealed_at=SEALED_AT, enrollment_deadline="2026-10-02")


def test_missing_or_mutable_deadline_is_rejected() -> None:
    with pytest.raises(LiveCohortAdmissionError, match="enrollment_deadline"):
        LiveCohortAdmissionPolicy(sealed_at=SEALED_AT, enrollment_deadline="")
    with pytest.raises(LiveCohortAdmissionError, match="immutable"):
        LiveCohortAdmissionPolicy(
            sealed_at=SEALED_AT,
            enrollment_deadline=ENROLLMENT_DEADLINE,
            enrollment_deadline_immutable=False,
        )
    with pytest.raises(LiveCohortAdmissionError, match="30-day bound"):
        LiveCohortAdmissionPolicy(
            sealed_at=SEALED_AT,
            enrollment_deadline=ENROLLMENT_DEADLINE,
            enrollment_deadline_maximum_days=29,
        )


def test_policy_cannot_weaken_live_or_shadow_gates() -> None:
    with pytest.raises(LiveCohortAdmissionError, match="shadow_before_canary"):
        LiveCohortAdmissionPolicy(
            sealed_at=SEALED_AT,
            enrollment_deadline=ENROLLMENT_DEADLINE,
            shadow_before_canary=False,
        )
    with pytest.raises(LiveCohortAdmissionError, match="fixtures cannot satisfy live"):
        LiveCohortAdmissionPolicy(
            sealed_at=SEALED_AT,
            enrollment_deadline=ENROLLMENT_DEADLINE,
            fixtures_satisfy_live=True,
        )
    with pytest.raises(LiveCohortAdmissionError, match="simulation cannot satisfy live"):
        LiveCohortAdmissionPolicy(
            sealed_at=SEALED_AT,
            enrollment_deadline=ENROLLMENT_DEADLINE,
            simulation_satisfies_live=True,
        )
    with pytest.raises(LiveCohortAdmissionError, match="minimum"):
        LiveCohortAdmissionPolicy(
            sealed_at=SEALED_AT,
            enrollment_deadline=ENROLLMENT_DEADLINE,
            minimum=9,
        )


def test_evidence_qualified_cohort_requires_ten_distinct_new_live_tasks() -> None:
    short = admit_cohort(_shadow_tasks(LIVE_MINIMUM - 1), now=SEALED_AT)
    assert short.disposition == "insufficient_evidence"
    assert short.qualification is False
    assert short.live is False
    assert short.count == LIVE_MINIMUM - 1
    assert short.canary_mutation_permitted is False
    assert LiveCohortReason.INSUFFICIENT_DISTINCT_LIVE_TASKS.value in short.reasons

    qualified = admit_cohort(_shadow_tasks(LIVE_MINIMUM), now=SEALED_AT)
    assert qualified.disposition == "evidence_qualified"
    assert qualified.qualification is True
    assert qualified.admitted is True
    assert qualified.live is True
    assert qualified.shadow_admitted is True
    assert qualified.count == LIVE_MINIMUM
    assert qualified.canary_admitted is False
    assert qualified.canary_mutation_permitted is False
    assert len(qualified.distinct_task_ids) == LIVE_MINIMUM


def test_duplicate_tasks_do_not_satisfy_the_ten_task_minimum() -> None:
    tasks = _shadow_tasks(LIVE_MINIMUM - 1)
    tasks.append(copy.deepcopy(tasks[0]))
    receipt = admit_cohort(tasks, now=SEALED_AT)
    assert receipt.disposition == "insufficient_evidence"
    assert receipt.count == LIVE_MINIMUM - 1
    assert LiveCohortReason.DUPLICATE_TASK.value in receipt.reasons
    assert receipt.task_receipts[-1]["disposition"] == TaskDisposition.NOT_ADMITTED.value


def test_live_provenance_fields_are_required_and_bound() -> None:
    complete = live_task_evidence_payload(1)
    for field in REQUIRED_PROVENANCE_FIELDS:
        assert complete[field]
    receipt = admit_task(complete)
    assert receipt.admitted is True
    assert receipt.disposition == TaskDisposition.ADMITTED_SHADOW.value
    assert receipt.live is True
    missing = dict(complete)
    missing.pop("task_cid")
    missing.pop("identity", None)
    missing.pop("receipt_cid", None)
    with pytest.raises(LiveCohortAdmissionError, match="task_cid"):
        admit_task(missing)
    not_live = live_task_evidence_payload(2, live=False)
    denied = admit_task(not_live)
    assert denied.admitted is False
    assert denied.live is False
    assert LiveCohortReason.NOT_LIVE.value in denied.reasons


def test_shadow_execution_is_non_mutating() -> None:
    mutating = admit_task(live_task_evidence_payload(1, mutating=True))
    assert mutating.admitted is False
    assert mutating.mutation_permitted is False
    assert LiveCohortReason.SHADOW_MUTATING.value in mutating.reasons
    shadow = admit_task(live_task_evidence_payload(2))
    assert shadow.admitted is True
    assert shadow.mutating is False
    assert shadow.mutation_permitted is False


def test_canary_requires_prior_shadow_and_safety_admission() -> None:
    shadow = admit_task(live_task_evidence_payload(1))
    bypass = admit_task(
        live_task_evidence_payload(1, execution_mode="canary", safety_admitted=True)
    )
    assert bypass.admitted is False
    assert bypass.mutation_permitted is False
    assert LiveCohortReason.SHADOW_BYPASS.value in bypass.reasons
    assert LiveCohortReason.CANARY_WITHOUT_SHADOW.value in bypass.reasons

    unsafe = admit_task(
        live_task_evidence_payload(
            1,
            execution_mode="canary",
            safety_admitted=False,
            shadow_receipt_cid=shadow.receipt_id,
        ),
        shadow_receipt=shadow,
    )
    assert unsafe.admitted is False
    assert LiveCohortReason.CANARY_WITHOUT_SAFETY.value in unsafe.reasons

    canary = admit_task(
        live_task_evidence_payload(
            1,
            execution_mode="canary",
            safety_admitted=True,
            shadow_receipt_cid=shadow.receipt_id,
        ),
        shadow_receipt=shadow,
    )
    assert canary.admitted is True
    assert canary.disposition == TaskDisposition.ADMITTED_CANARY.value
    assert canary.mutation_permitted is True
    assert canary.shadow_receipt_cid == shadow.receipt_id


def test_canary_cohort_is_separately_gated_behind_qualified_shadow() -> None:
    shadows = _shadow_tasks()
    shadow_receipts = [admit_task(item) for item in shadows]
    canaries = [
        live_task_evidence_payload(
            index,
            execution_mode="canary",
            safety_admitted=True,
            shadow_receipt_cid=shadow_receipts[index - 1].receipt_id,
        )
        for index in range(1, LIVE_MINIMUM + 1)
    ]
    denied = admit_cohort(canaries, now=SEALED_AT, request_canary=True)
    assert denied.disposition == "not_admitted"
    assert denied.canary_mutation_permitted is False
    assert LiveCohortReason.SHADOW_BYPASS.value in denied.reasons

    admitted = admit_cohort(
        canaries,
        shadow_receipts=shadow_receipts,
        now=SEALED_AT,
        request_canary=True,
    )
    assert admitted.disposition == "evidence_qualified"
    assert admitted.shadow_admitted is True
    assert admitted.canary_admitted is True
    assert admitted.canary_mutation_permitted is True
    assert admitted.count == LIVE_MINIMUM


def test_fixture_and_hermetic_substitution_are_rejected() -> None:
    fixture = live_task_evidence_payload(
        1,
        fixture_id="aseh-h01",
        hermetic=True,
        source_kind="hermetic_fixture",
        task_id="aseh-h01",
        task_cid=live_task_evidence_payload(1)["task_cid"],
    )
    receipt = admit_task(fixture)
    assert receipt.admitted is False
    assert LiveCohortReason.FIXTURE_SUBSTITUTION.value in receipt.reasons
    cohort = admit_cohort(
        [fixture, *(_shadow_tasks(LIVE_MINIMUM))],
        now=SEALED_AT,
    )
    assert cohort.disposition == "not_admitted"
    assert cohort.live is False
    assert cohort.qualification is False
    assert LiveCohortReason.FIXTURE_SUBSTITUTION.value in cohort.reasons


def test_simulation_substitution_is_rejected() -> None:
    simulated = live_task_evidence_payload(
        1,
        simulated=True,
        truth_state="simulated",
        provider_usage_truth_state="simulated",
    )
    receipt = admit_task(simulated)
    assert receipt.admitted is False
    assert LiveCohortReason.SIMULATION_SUBSTITUTION.value in receipt.reasons
    cohort = admit_cohort([simulated], now=SEALED_AT)
    assert cohort.disposition == "not_admitted"
    assert LiveCohortReason.SIMULATION_SUBSTITUTION.value in cohort.reasons


def test_historical_replay_cannot_satisfy_live() -> None:
    historical = live_task_evidence_payload(
        1,
        task_id="ASEH-000",
        historical_replay=True,
        source_kind="historical_replay",
        live=True,
        newly_encountered=False,
    )
    receipt = admit_task(historical)
    assert receipt.admitted is False
    assert LiveCohortReason.HISTORICAL_SUBSTITUTION.value in receipt.reasons
    assert LiveCohortReason.NOT_NEWLY_ENCOUNTERED.value in receipt.reasons
    cohort = admit_cohort([historical], now=SEALED_AT)
    assert cohort.disposition == "not_admitted"
    assert cohort.live is False


def test_deadline_elapsed_emits_typed_insufficiency() -> None:
    elapsed = admit_cohort((), now="2026-10-03T00:00:00Z")
    assert elapsed.disposition == "deadline_elapsed"
    assert elapsed.qualification is False
    assert elapsed.live is False
    assert elapsed.population_status == "insufficient"
    assert LiveCohortReason.DEADLINE_ELAPSED.value in elapsed.reasons
    assert LiveCohortReason.INSUFFICIENT_DISTINCT_LIVE_TASKS.value in elapsed.reasons
    nine = admit_cohort(_shadow_tasks(9), now="2026-10-02T00:00:01Z")
    assert nine.disposition == "deadline_elapsed"
    assert nine.count == 9


def test_in_window_shortfall_emits_insufficient_evidence() -> None:
    empty = admit_cohort((), now=SEALED_AT)
    assert empty.disposition == "insufficient_evidence"
    assert empty.population_status == "enrolling"
    assert empty.enrollment_deadline_immutable is True
    assert LiveCohortReason.NOT_YET_MEASURED.value in empty.reasons
    assert empty.disposition in CLOSED_DISPOSITIONS


def test_untyped_or_nontruthful_disposition_is_rejected() -> None:
    qualified = admit_cohort(_shadow_tasks(), now=SEALED_AT)
    payload = qualified.to_dict()
    payload["disposition"] = "almost_qualified"
    with pytest.raises(LiveCohortAdmissionError, match="closed"):
        type(qualified).from_dict(payload)
    payload = qualified.to_dict()
    payload["live"] = True
    payload["qualification"] = False
    payload["admitted"] = False
    payload["disposition"] = "insufficient_evidence"
    payload["count"] = 0
    payload["distinct_task_ids"] = []
    payload["shadow_admitted"] = False
    payload["canary_admitted"] = False
    payload["canary_mutation_permitted"] = False
    payload["population_status"] = "enrolling"
    payload["reasons"] = [LiveCohortReason.NOT_YET_MEASURED.value]
    payload["task_receipts"] = []
    with pytest.raises(LiveCohortAdmissionError, match="live cannot be claimed"):
        type(qualified).from_dict(payload)


def test_mixed_shadow_then_canary_in_one_request_respects_shadow_first() -> None:
    shadows = _shadow_tasks()
    shadow_receipts = [admit_task(item) for item in shadows]
    mixed: list[dict[str, Any]] = []
    for index, shadow in enumerate(shadows, start=1):
        mixed.append(shadow)
        mixed.append(
            live_task_evidence_payload(
                index,
                execution_mode="canary",
                safety_admitted=True,
                shadow_receipt_cid=shadow_receipts[index - 1].receipt_id,
            )
        )
    receipt = admit_cohort(mixed, now=SEALED_AT)
    assert receipt.disposition == "evidence_qualified"
    assert receipt.shadow_admitted is True
    assert receipt.canary_admitted is True
    assert receipt.canary_mutation_permitted is True
    assert receipt.count == LIVE_MINIMUM


def test_sealed_policy_identity_is_stable() -> None:
    first = sealed_policy()
    second = LiveCohortAdmissionPolicy.from_dict(first.to_dict())
    assert first.policy_identity == second.policy_identity
    assert first.enrollment_deadline == ENROLLMENT_DEADLINE
    mutated = first.to_dict()
    mutated["enrollment_deadline"] = "2026-09-15T00:00:00Z"
    changed = LiveCohortAdmissionPolicy.from_dict(mutated)
    assert changed.policy_identity != first.policy_identity
