"""Independent current-tree checks for DOEP-055 oscillation/runaway/nonconvergence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement.campaign_refill_policy import (
    AUTOMATIC_BOUNDED_TASK_REFILL_BINDING,
    CAMPAIGN_REFILL_POLICY_BINDING,
    CAMPAIGN_REFILL_POLICY_INTERFACE,
    CAMPAIGN_REFILL_POLICY_SCHEMA,
    MAX_GENERATED_TASKS,
    MAX_OSCILLATION_WINDOW,
    MAX_REPEATED_PLAN_HASHES,
    MAX_REPLAN_EPOCHS,
    MAX_STABLE_FRONTIER_ROUNDS,
    MAX_TASKS_PER_OBJECTIVE,
    MIN_EVENT_DEBOUNCE_MS,
    OSCILLATION_RUNAWAY_NONCONVERGENCE_BINDING,
    OSCILLATION_RUNAWAY_NONCONVERGENCE_CONSUMES,
    OSCILLATION_RUNAWAY_NONCONVERGENCE_INTERFACE,
    OSCILLATION_RUNAWAY_NONCONVERGENCE_SCHEMA,
    TASK_SEMANTIC_DEDUPLICATION_BINDING,
    CampaignRefillCandidate,
    CampaignRefillController,
    CampaignRefillDecision,
    CampaignRefillError,
    CampaignRefillHistory,
    CampaignRefillPolicy,
    RefillDisposition,
    RefillTrigger,
    all_refill_triggers_are_bounded,
    apply_oscillation_runaway_nonconvergence_controls,
    convergence_controls_are_armed,
    detect_oscillation,
    detect_repeated_plan_hash,
    detect_runaway,
    detect_stable_frontier,
    evaluate_convergence_controls,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py/agent_supervisor/self_improvement/campaign_refill_policy.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-055.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-055.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/self_improvement/campaign_refill_policy.py",
    "test/api/doep/test_doep_055_add_oscillation_runaway_and_nonconvergence_controls.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-055.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-055.json",
)
TASK_CID = "sha256:09c8bfe34519ce11e93ef8ed69fa7cccc297b8deb0c15e0684490d97e886fab8"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {
        "commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f",
        "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7",
    },
    "ipfs_datasets_py": {
        "commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7",
        "tree": "456e09b51d6a07a3a5873436df24054768195320",
    },
    "ipfs_kit_py": {
        "commit": "b6c65ba732733d7e33852713ba18aa3b12235668",
        "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2",
    },
    "lift_coding": {
        "commit": "bb8869ed72eb7002434345d9969efee729c4f7f6",
        "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42",
    },
}


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(
    candidate_id: str = "residual-1",
    *,
    trigger: RefillTrigger = RefillTrigger.PROOF_RESIDUAL,
    residual_count: int = 1,
    curriculum_key: str = "",
    **metadata: object,
) -> CampaignRefillCandidate:
    return CampaignRefillCandidate(
        candidate_id=candidate_id,
        trigger=trigger,
        residual_count=residual_count,
        curriculum_key=curriculum_key,
        metadata=metadata,
    )


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_binding_extends_campaign_refill_without_competing_subsystem() -> None:
    assert CAMPAIGN_REFILL_POLICY_INTERFACE == "CampaignRefillPolicy@1"
    assert CAMPAIGN_REFILL_POLICY_BINDING == CAMPAIGN_REFILL_POLICY_INTERFACE
    assert CAMPAIGN_REFILL_POLICY_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/campaign-refill-policy@1"
    )
    assert OSCILLATION_RUNAWAY_NONCONVERGENCE_INTERFACE == (
        "OscillationRunawayNonconvergence@1"
    )
    assert OSCILLATION_RUNAWAY_NONCONVERGENCE_BINDING == (
        OSCILLATION_RUNAWAY_NONCONVERGENCE_INTERFACE
    )
    assert OSCILLATION_RUNAWAY_NONCONVERGENCE_SCHEMA == (
        "ipfs_accelerate_py/agent-supervisor/oscillation-runaway-nonconvergence@1"
    )
    assert OSCILLATION_RUNAWAY_NONCONVERGENCE_CONSUMES == (
        CAMPAIGN_REFILL_POLICY_SCHEMA,
        AUTOMATIC_BOUNDED_TASK_REFILL_BINDING,
        TASK_SEMANTIC_DEDUPLICATION_BINDING,
    )
    assert CampaignRefillController.CONVERGENCE_INTERFACE == (
        OSCILLATION_RUNAWAY_NONCONVERGENCE_INTERFACE
    )
    assert CampaignRefillController.CONVERGENCE_BINDING == (
        OSCILLATION_RUNAWAY_NONCONVERGENCE_BINDING
    )
    assert CampaignRefillController.BINDING == CAMPAIGN_REFILL_POLICY_BINDING
    assert apply_oscillation_runaway_nonconvergence_controls.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.self_improvement.campaign_refill_policy"
    )
    assert evaluate_convergence_controls.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.self_improvement.campaign_refill_policy"
    )
    assert detect_oscillation.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.self_improvement.campaign_refill_policy"
    )
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "class CampaignRefillController" in source
    assert "def decide(" in source
    assert "def apply_oscillation_runaway_nonconvergence_controls(" in source
    assert "def detect_oscillation(" in source
    assert "def detect_runaway(" in source
    assert "quarantined_nonconvergent" in source
    assert "not a second refill owner" in source
    assert "never writes DuckDB" in source
    assert "cannot skip these controls" in source
    assert "class CompetingOscillationDetector" not in source
    assert "class OscillationEngine" not in source
    assert "class RunawayController" not in source
    assert "class CompetingRefill" not in source
    assert "class NonconvergenceBus" not in source
    assert "CREATE TABLE" not in source
    assert "import duckdb" not in source.casefold()
    assert "connect(" not in source


def test_existing_no_progress_round_and_repetition_bounds_remain() -> None:
    controller = CampaignRefillController()
    assert all_refill_triggers_are_bounded()
    assert convergence_controls_are_armed()
    candidate = _candidate("stuck-1", trigger=RefillTrigger.NO_PROGRESS, curriculum_key="curr:stuck")
    no_progress = controller.decide(
        (candidate,),
        history=CampaignRefillHistory(
            refill_rounds=1,
            last_progress_identity="curr:stuck",
            no_progress_streak=2,
        ),
        cursor_advanced=False,
        progress_identity="curr:stuck",
    )
    assert no_progress.disposition is RefillDisposition.NO_PROGRESS_BOUNDED
    assert no_progress.admitted == ()
    assert no_progress.quarantined is False
    repeated = _candidate(
        "repeat-1",
        trigger=RefillTrigger.CURRICULUM_GAP,
        curriculum_key="curr:same",
    )
    repetition = controller.decide(
        (repeated,),
        history=CampaignRefillHistory(curriculum_repetitions={"curr:same": 2}),
    )
    assert repetition.disposition is RefillDisposition.REPETITION_BOUNDED
    rounds = controller.decide(
        (repeated,),
        history=CampaignRefillHistory(refill_rounds=8),
    )
    assert rounds.disposition is RefillDisposition.ROUND_BOUNDED


def test_progressing_work_is_still_admitted() -> None:
    decision = apply_oscillation_runaway_nonconvergence_controls(
        (_candidate("proof-a"),),
        history=CampaignRefillHistory(
            progress_identities=("frontier:a",),
            plan_hashes=("plan:1",),
            frontier_identities=("ready:1",),
        ),
        progress_identity="frontier:b",
        plan_hash="plan:2",
        frontier_identity="ready:2",
        worker_assertion=True,
    )
    assert decision.disposition is RefillDisposition.ADMITTED
    assert [item.candidate_id for item in decision.admitted] == ["proof-a"]
    assert decision.quarantined is False
    payload = decision.to_dict()
    assert payload["completion_authoritative"] is False
    assert payload["empty_queue_is_completion"] is False
    assert payload["database_write"] is False
    assert payload["carrier"] == "CampaignRefillController"
    assert payload["binding"] == OSCILLATION_RUNAWAY_NONCONVERGENCE_BINDING


def test_oscillating_progress_identities_quarantine_even_with_worker_assertion() -> None:
    assert detect_oscillation(("A", "B", "A")) is True
    assert detect_oscillation(("A", "B", "A", "B")) is True
    assert detect_oscillation(("A", "A", "A")) is False
    assert detect_oscillation(("A", "B")) is False
    assert detect_oscillation(("A", "B", "C")) is False
    decision = apply_oscillation_runaway_nonconvergence_controls(
        (_candidate("cycle-1"),),
        history=CampaignRefillHistory(progress_identities=("state:a", "state:b")),
        progress_identity="state:a",
        cursor_advanced=True,
        worker_assertion=True,
    )
    assert decision.disposition is RefillDisposition.QUARANTINED_NONCONVERGENT
    assert decision.reason_code == "oscillation_detected"
    assert decision.control == "oscillation"
    assert decision.quarantined is True
    assert decision.admitted == ()
    assert decision.changed is False
    assert decision.bounded is True
    payload = decision.to_dict()
    assert payload["quarantined"] is True
    assert payload["empty_queue_is_completion"] is False
    assert payload["worker_assertion_is_authority"] is False
    assert payload["completion_authoritative"] is False
    assert '"completion_authoritative": true' not in json.dumps(payload)


def test_repeated_plan_hash_is_quarantined() -> None:
    assert detect_repeated_plan_hash((), "plan:a") is False
    assert detect_repeated_plan_hash(("plan:a",), "plan:a") is True
    assert MAX_REPEATED_PLAN_HASHES == 2
    decision = CampaignRefillController().decide(
        (_candidate("hash-1"),),
        history=CampaignRefillHistory(plan_hashes=("plan:same",)),
        plan_hash="plan:same",
        worker_assertion=True,
    )
    assert decision.disposition is RefillDisposition.QUARANTINED_NONCONVERGENT
    assert decision.reason_code == "repeated_plan_hash"
    assert decision.control == "oscillation"


def test_stable_frontier_is_quarantined() -> None:
    assert detect_stable_frontier((), "frontier:a") is False
    assert detect_stable_frontier(("frontier:a",), "frontier:a") is True
    assert MAX_STABLE_FRONTIER_ROUNDS == 2
    decision = apply_oscillation_runaway_nonconvergence_controls(
        (_candidate("stable-1"),),
        history=CampaignRefillHistory(frontier_identities=("frontier:open",)),
        frontier_identity="frontier:open",
    )
    assert decision.disposition is RefillDisposition.QUARANTINED_NONCONVERGENT
    assert decision.reason_code == "stable_frontier"
    assert decision.control == "nonconvergence"
    via_last = CampaignRefillController().decide(
        (_candidate("stable-2"),),
        history=CampaignRefillHistory(
            last_frontier_identity="frontier:open",
            stable_frontier_streak=1,
        ),
        frontier_identity="frontier:open",
    )
    assert via_last.disposition is RefillDisposition.QUARANTINED_NONCONVERGENT
    assert via_last.reason_code == "stable_frontier"


def test_runaway_generated_tasks_replan_epochs_and_objective_caps() -> None:
    policy = CampaignRefillPolicy()
    assert detect_runaway(
        generated_task_count=MAX_GENERATED_TASKS,
        replan_epochs=0,
        tasks_for_objective=0,
        policy=policy,
    ) == "runaway_generated_tasks"
    generated = apply_oscillation_runaway_nonconvergence_controls(
        (_candidate("runaway-gen"),),
        history=CampaignRefillHistory(generated_task_count=MAX_GENERATED_TASKS),
        worker_assertion=True,
    )
    assert generated.disposition is RefillDisposition.QUARANTINED_NONCONVERGENT
    assert generated.reason_code == "runaway_generated_tasks"
    assert generated.control == "runaway"
    epochs = CampaignRefillController().decide(
        (_candidate("runaway-epochs"),),
        history=CampaignRefillHistory(replan_epochs=MAX_REPLAN_EPOCHS),
    )
    assert epochs.disposition is RefillDisposition.QUARANTINED_NONCONVERGENT
    assert epochs.reason_code == "runaway_replan_epochs"
    objective = apply_oscillation_runaway_nonconvergence_controls(
        (_candidate("runaway-obj", objective_id="DOEP-G060.S3"),),
        history=CampaignRefillHistory(
            tasks_per_objective={"DOEP-G060.S3": MAX_TASKS_PER_OBJECTIVE}
        ),
        objective_id="DOEP-G060.S3",
    )
    assert objective.disposition is RefillDisposition.QUARANTINED_NONCONVERGENT
    assert objective.reason_code == "runaway_tasks_per_objective"


def test_event_debounce_rejects_without_completing_or_quarantining() -> None:
    decision = apply_oscillation_runaway_nonconvergence_controls(
        (_candidate("debounce-1"),),
        last_event_ms=10_000,
        now_ms=10_000 + MIN_EVENT_DEBOUNCE_MS - 1,
        worker_assertion=True,
    )
    assert decision.disposition is RefillDisposition.EVENT_DEBOUNCED
    assert decision.reason_code == "event_debounced"
    assert decision.quarantined is False
    assert decision.admitted == ()
    payload = decision.to_dict()
    assert payload["empty_queue_is_completion"] is False
    assert payload["completion_authoritative"] is False
    released = apply_oscillation_runaway_nonconvergence_controls(
        (_candidate("debounce-2"),),
        last_event_ms=10_000,
        now_ms=10_000 + MIN_EVENT_DEBOUNCE_MS,
    )
    assert released.disposition is RefillDisposition.ADMITTED
    with pytest.raises(CampaignRefillError, match="both last_event_ms and now_ms"):
        apply_oscillation_runaway_nonconvergence_controls(
            (_candidate("debounce-3"),),
            last_event_ms=10_000,
        )


def test_worker_assertion_cannot_skip_controls_or_complete_empty_queue() -> None:
    empty = apply_oscillation_runaway_nonconvergence_controls(
        (),
        worker_assertion=True,
    )
    assert empty.disposition is RefillDisposition.REJECTED
    assert empty.reason_code == "no_admissible_candidates"
    assert empty.admitted == ()
    payload = empty.to_dict()
    assert payload["empty_queue_is_completion"] is False
    assert payload["completion_authoritative"] is False
    assert payload["worker_assertion_is_authority"] is False
    blocked = evaluate_convergence_controls(
        (_candidate("cycle-2"),),
        history=CampaignRefillHistory(progress_identities=("x", "y")),
        progress_identity="x",
        worker_assertion=True,
    )
    assert blocked is not None
    assert blocked.disposition is RefillDisposition.QUARANTINED_NONCONVERGENT
    with pytest.raises(CampaignRefillError, match="worker_assertion"):
        apply_oscillation_runaway_nonconvergence_controls(
            (_candidate("bad-assertion"),),
            worker_assertion="yes",  # type: ignore[arg-type]
        )


def test_safety_ceilings_cannot_be_raised_or_disabled() -> None:
    with pytest.raises(CampaignRefillError, match="max_generated_tasks"):
        CampaignRefillPolicy(max_generated_tasks=MAX_GENERATED_TASKS + 1)
    with pytest.raises(CampaignRefillError, match="max_replan_epochs"):
        CampaignRefillPolicy(max_replan_epochs=MAX_REPLAN_EPOCHS + 1)
    with pytest.raises(CampaignRefillError, match="max_tasks_per_objective"):
        CampaignRefillPolicy(max_tasks_per_objective=MAX_TASKS_PER_OBJECTIVE + 1)
    with pytest.raises(CampaignRefillError, match="oscillation_window"):
        CampaignRefillPolicy(oscillation_window=MAX_OSCILLATION_WINDOW + 1)
    with pytest.raises(CampaignRefillError, match="oscillation_detection cannot be disabled"):
        CampaignRefillPolicy(oscillation_detection=False)
    with pytest.raises(CampaignRefillError, match="runaway_detection cannot be disabled"):
        CampaignRefillPolicy(runaway_detection=False)
    with pytest.raises(CampaignRefillError, match="nonconvergence_quarantine cannot be disabled"):
        CampaignRefillPolicy(nonconvergence_quarantine=False)
    with pytest.raises(CampaignRefillError, match="repeated_plan_hash_detection"):
        CampaignRefillPolicy(repeated_plan_hash_detection=False)
    with pytest.raises(CampaignRefillError, match="stable_frontier_detection"):
        CampaignRefillPolicy(stable_frontier_detection=False)
    tightened = CampaignRefillPolicy(max_generated_tasks=4, oscillation_window=3)
    assert tightened.max_generated_tasks == 4
    assert tightened.oscillation_window == 3
    assert convergence_controls_are_armed(tightened)


def test_same_identity_stall_is_no_progress_not_oscillation() -> None:
    assert detect_oscillation(("stuck", "stuck", "stuck")) is False
    decision = CampaignRefillController().decide(
        (_candidate("stall-1", trigger=RefillTrigger.NO_PROGRESS, curriculum_key="curr:stuck"),),
        history=CampaignRefillHistory(
            progress_identities=("curr:stuck", "curr:stuck"),
            last_progress_identity="curr:stuck",
            no_progress_streak=2,
        ),
        cursor_advanced=False,
        progress_identity="curr:stuck",
    )
    assert decision.disposition is RefillDisposition.NO_PROGRESS_BOUNDED
    assert decision.quarantined is False


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-055"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["entrypoint"] == (
        "apply_oscillation_runaway_nonconvergence_controls"
    )
    assert manifest["canonical_extension"]["carrier"] == "CampaignRefillController"
    assert manifest["canonical_extension"]["binding"] == (
        OSCILLATION_RUNAWAY_NONCONVERGENCE_BINDING
    )
    assert manifest["canonical_extension"]["authority"] == (
        "model_free_oscillation_runaway_nonconvergence_quarantine"
    )
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(SOURCE_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
    assert CampaignRefillDecision(
        disposition=RefillDisposition.QUARANTINED_NONCONVERGENT,
        policy_id="policy:test",
    ).to_dict()["model_free"] is True
