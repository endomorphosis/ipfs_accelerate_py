"""WWL-106: fixture proofs for native completion verification.

Done flags, a zero exit, stale receipts, skipped work, unresolved source
blockers and absent release evidence cannot establish completion. The valid
case carries fixture release evidence, so this task does not need the later
public release.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_completion import (
    BLOCKER_KINDS,
    NON_AUTHORITATIVE,
    OBSERVATION_SCHEMA,
    RELEASE_FIELDS,
    RELEASE_SCHEMA,
    RECEIPT_FIELDS,
    SOURCE_SCHEMA,
    TASK_RECEIPT_SCHEMA,
    UNRESOLVED_BLOCKERS,
    VERIFIED_REASON,
    CompletionContractError,
    as_watchdog_verifier,
    digest,
    implementation_obligations,
    sealed,
    verify_completion,
)

NOW = 1_700_000_000
RUN = "wwl-test"
GOAL = "WWL-G000"
CORE = "WWL-000"
CHILD = "WWL-SRC-0123456789abcdef-verify-0001"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
ADAPTER = "ipfs_accelerate_py/agent_supervisor/project_adapters/worldwide_legal_search_completion.py"
ARGV = ["python3", "-m", "pytest", "test/api/test_worldwide_legal_search_completion.py", "-q"]


def sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


TREE = sha("accepted-tree")
PLAN = sha("admitted-plan")
INVENTORY = sha("inventory")
SOURCE_REV = sha("source-revision")
OTHER_TREE = sha("other-tree")


def task_receipt(task_id: str, **overrides) -> dict:
    body = {
        "schema": TASK_RECEIPT_SCHEMA,
        "task_id": task_id,
        "run_id": RUN,
        "goal_id": GOAL,
        "accepted_tree": TREE,
        "admitted_plan_revision": PLAN,
        "source_revision": SOURCE_REV,
        "validation_argv": list(ARGV),
        "validation_exit_code": 0,
        "validation_result": "4 passed",
        "validation_skipped": 0,
        "output_hashes": {ADAPTER: sha("adapter-" + task_id)},
        "merge_receipt": sha("merge-" + task_id),
        "observed_at": NOW,
        "independent": True,
        "completion_authority": False,
    }
    body.update(overrides)
    receipt_id = body.pop("receipt_id", None)
    sealed_body = sealed(RECEIPT_FIELDS, body)
    if receipt_id is not None:
        sealed_body["receipt_id"] = receipt_id
    return sealed_body


def source_row(source_id: str, **overrides) -> dict:
    row = {
        "schema": SOURCE_SCHEMA,
        "source_id": source_id,
        "inventory_revision": INVENTORY,
        "disposition": "ingested",
        "blocker_kind": "",
        "reason": "",
        "ingest_receipt": sha("ingest-" + source_id),
        "resolved_revision": sha("resolved-" + source_id),
    }
    row.update(overrides)
    return row


def release_evidence(**overrides) -> dict:
    body = {
        "schema": RELEASE_SCHEMA,
        "run_id": RUN,
        "goal_id": GOAL,
        "accepted_tree": TREE,
        "admitted_plan_revision": PLAN,
        "commit": COMMIT,
        "artifact_digest": sha("release-artifact"),
        "public_url": "https://example.invalid/worldwide/fixture",
        "rollback_preserved": True,
        "observed_at": NOW,
        "fixture": True,
        "live_fetch": False,
    }
    body.update(overrides)
    receipt_id = body.pop("receipt_id", None)
    sealed_body = sealed(RELEASE_FIELDS, body)
    if receipt_id is not None:
        sealed_body["receipt_id"] = receipt_id
    return sealed_body


def task_row(task_id: str, **receipt_overrides) -> dict:
    return {
        "task_id": task_id,
        "status": "",
        "exit_code": None,
        "skipped": False,
        "receipt": task_receipt(task_id, **receipt_overrides),
    }


def valid_observation() -> dict:
    return {
        "schema": OBSERVATION_SCHEMA,
        "run_id": RUN,
        "goal_id": GOAL,
        "accepted_tree": TREE,
        "admitted_plan_revision": PLAN,
        "inventory_revision": INVENTORY,
        "source_revision": SOURCE_REV,
        "observed_at": NOW,
        "required_task_ids": [CORE, CHILD],
        "tasks": [task_row(CORE), task_row(CHILD)],
        "sources": [
            source_row("nl-national"),
            source_row(
                "code-only-repo",
                disposition="excluded",
                reason="code repository is not a legal corpus",
                ingest_receipt="",
                resolved_revision="",
            ),
        ],
        "release": release_evidence(),
        "hold": False,
        "effects_unknown": False,
        "runnable_count": 0,
        "active_task_id": "",
        "expansion_child_ids": [CHILD],
    }


def drop_receipts(observation: dict, *, status: str = "", exit_code: int | None = None, skipped: bool = False) -> dict:
    for task in observation["tasks"]:
        task["status"] = status
        task["exit_code"] = exit_code
        task["skipped"] = skipped
        task["receipt"] = None
    return observation


def test_valid_fixture_verifies_without_the_future_release():
    verdict = verify_completion(valid_observation(), now=NOW)
    assert verdict["completion_verified"] is True
    assert verdict["completion_authority"] is False
    assert verdict["task_retry_authority"] is False
    assert verdict["live_release_consulted"] is False
    assert verdict["reason"] == VERIFIED_REASON
    assert verdict["rejections"] == []
    assert verdict["ignored_claims"] == []
    assert verdict["completion_receipt"] == digest(verdict["proof"])
    assert verdict["proof"]["required_task_ids"] == [CORE, CHILD]
    assert verdict["proof"]["expansion_child_ids"] == [CHILD]
    assert verdict["proof"]["live_release_consulted"] is False
    assert verdict["proof"]["release_receipt"] == valid_observation()["release"]["receipt_id"]
    again = verify_completion(valid_observation(), now=NOW)
    assert again == verdict
    obligations = implementation_obligations()
    assert obligations["future_release_required"] is False
    assert obligations["required_release_tasks"] == []
    assert obligations["establishes_campaign_completion"] is False
    assert obligations["live_release_consulted"] is False


def test_done_flag_and_zero_exit_are_not_part_of_the_proof():
    plain = verify_completion(valid_observation(), now=NOW)
    flagged = valid_observation()
    flagged["done"] = True
    flagged["completion_verified"] = True
    flagged["completion_receipt"] = "sha256:" + ("ab" * 32)
    flagged["process_exit_code"] = 0
    flagged["prose"] = "the worker says the campaign is done"
    for task in flagged["tasks"]:
        task["status"] = "done"
        task["exit_code"] = 0
    verdict = verify_completion(flagged, now=NOW)
    assert verdict["completion_verified"] is True
    assert verdict["completion_receipt"] == plain["completion_receipt"]
    assert verdict["proof"] == plain["proof"]
    assert verdict["ignored_claims"] == ["done_flag", "zero_exit", "prose"]
    assert verdict["completion_authority"] is False


def test_done_flags_cannot_establish_completion():
    observation = drop_receipts(valid_observation(), status="done")
    observation["completion_verified"] = True
    observation["done"] = True
    verdict = verify_completion(observation, now=NOW)
    assert verdict["completion_verified"] is False
    assert verdict["completion_receipt"] == ""
    assert verdict["rejections"] == ["done_flag"]
    assert verdict["completion_authority"] is False


def test_observation_done_flag_without_task_status_cannot_establish_completion():
    observation = drop_receipts(valid_observation())
    observation["status"] = "completed"
    verdict = verify_completion(observation, now=NOW)
    assert verdict["completion_verified"] is False
    assert verdict["rejections"] == ["done_flag"]


@pytest.mark.parametrize("claim", ["exit_code", "process_exit_code", "returncode"])
def test_zero_exit_cannot_establish_completion(claim: str):
    observation = drop_receipts(valid_observation())
    if claim == "exit_code":
        for task in observation["tasks"]:
            task["exit_code"] = 0
    else:
        observation[claim] = 0
    verdict = verify_completion(observation, now=NOW)
    assert verdict["completion_verified"] is False
    assert verdict["rejections"] == ["zero_exit"]
    assert verdict["reason"] == "zero_exit"


def test_zero_exit_without_merge_or_outputs_cannot_establish_completion():
    observation = valid_observation()
    for task in observation["tasks"]:
        task["exit_code"] = 0
        task["receipt"] = task_receipt(
            task["task_id"],
            validation_argv=[],
            output_hashes={},
            merge_receipt="",
        )
    verdict = verify_completion(observation, now=NOW)
    assert verdict["completion_verified"] is False
    assert verdict["rejections"] == ["zero_exit"]


@pytest.mark.parametrize(
    "mutate",
    ["tree", "plan", "source", "run", "goal", "task", "digest", "old", "future", "observation"],
)
def test_stale_receipts_cannot_establish_completion(mutate: str):
    observation = valid_observation()
    if mutate == "observation":
        observation["observed_at"] = NOW - 3601
    elif mutate == "task":
        for task in observation["tasks"]:
            # The row keeps its required id. The sealed receipt names another task.
            task["receipt"] = task_receipt("WWL-001")
    else:
        overrides = {
            "tree": {"accepted_tree": OTHER_TREE},
            "plan": {"admitted_plan_revision": sha("other-plan")},
            "source": {"source_revision": sha("other-source")},
            "run": {"run_id": "other-run"},
            "goal": {"goal_id": "WWL-G090"},
            "digest": {},
            "old": {"observed_at": NOW - 3601},
            "future": {"observed_at": NOW + 31},
        }[mutate]
        for task in observation["tasks"]:
            task["receipt"] = task_receipt(task["task_id"], **overrides)
            if mutate == "digest":
                task["receipt"]["validation_result"] = "5 passed"
    verdict = verify_completion(observation, now=NOW)
    assert verdict["completion_verified"] is False
    assert "stale_receipt" in verdict["rejections"]
    assert verdict["reason"] == "stale_receipt"


def test_freshness_boundaries_keep_current_receipts():
    for observed in (NOW - 3600, NOW + 30):
        observation = valid_observation()
        observation["observed_at"] = observed
        observation["release"] = release_evidence(observed_at=observed)
        for task in observation["tasks"]:
            task["receipt"] = task_receipt(task["task_id"], observed_at=observed)
        verdict = verify_completion(observation, now=NOW)
        assert verdict["completion_verified"] is True


def test_release_tree_mismatch_is_a_stale_receipt():
    observation = valid_observation()
    observation["release"] = release_evidence(accepted_tree=OTHER_TREE)
    verdict = verify_completion(observation, now=NOW)
    assert verdict["completion_verified"] is False
    assert verdict["rejections"] == ["stale_receipt"]


@pytest.mark.parametrize(
    "mutate",
    ["flag", "status", "validation_count", "validation_text", "missing_task", "source", "pruned_child"],
)
def test_skipped_work_cannot_establish_completion(mutate: str):
    observation = valid_observation()
    if mutate == "flag":
        observation["tasks"][0]["skipped"] = True
    elif mutate == "status":
        observation["tasks"][0]["status"] = "skipped"
    elif mutate == "validation_count":
        observation["tasks"][0]["receipt"] = task_receipt(CORE, validation_skipped=1)
    elif mutate == "validation_text":
        observation["tasks"][0]["receipt"] = task_receipt(CORE, validation_result="4 passed, 1 skipped")
    elif mutate == "missing_task":
        observation["tasks"] = [task for task in observation["tasks"] if task["task_id"] != CHILD]
    elif mutate == "source":
        observation["sources"][1]["disposition"] = "skipped"
        observation["sources"][1]["reason"] = ""
    else:
        observation["required_task_ids"] = [CORE]
        observation["tasks"] = [task for task in observation["tasks"] if task["task_id"] == CORE]
    verdict = verify_completion(observation, now=NOW)
    assert verdict["completion_verified"] is False
    assert "skipped_work" in verdict["rejections"]
    assert verdict["reason"] == "skipped_work"


@pytest.mark.parametrize("kind", sorted(UNRESOLVED_BLOCKERS))
def test_unresolved_source_blockers_cannot_establish_completion(kind: str):
    observation = valid_observation()
    observation["sources"][0]["disposition"] = "unresolved"
    observation["sources"][0]["blocker_kind"] = kind
    verdict = verify_completion(observation, now=NOW)
    assert verdict["completion_verified"] is False
    assert verdict["rejections"] == ["unresolved_source_blocker"]


def test_unknown_rights_stay_unresolved_when_labeled_excluded():
    observation = valid_observation()
    observation["sources"][0]["disposition"] = "excluded"
    observation["sources"][0]["blocker_kind"] = "unknown-rights"
    observation["sources"][0]["reason"] = "rights were not reviewed"
    verdict = verify_completion(observation, now=NOW)
    assert verdict["completion_verified"] is False
    assert verdict["rejections"] == ["unresolved_source_blocker"]


def test_excluded_without_a_reason_and_empty_inventory_stay_unresolved():
    missing_reason = valid_observation()
    missing_reason["sources"][1]["reason"] = ""
    assert verify_completion(missing_reason, now=NOW)["rejections"] == ["unresolved_source_blocker"]
    empty = valid_observation()
    empty["sources"] = []
    assert verify_completion(empty, now=NOW)["rejections"] == ["unresolved_source_blocker"]
    no_ingest = valid_observation()
    no_ingest["sources"][0]["ingest_receipt"] = ""
    assert verify_completion(no_ingest, now=NOW)["rejections"] == ["unresolved_source_blocker"]


@pytest.mark.parametrize(
    "release",
    [
        None,
        {"note": "released"},
        "the site is live",
    ],
)
def test_absent_release_evidence_cannot_establish_completion(release):
    observation = valid_observation()
    if release == "the site is live":
        observation["prose"] = release
        observation["release"] = None
    else:
        observation["release"] = release
    observation["release_required"] = False
    observation["future_release_required"] = False
    verdict = verify_completion(observation, now=NOW)
    assert verdict["completion_verified"] is False
    assert "absent_release_evidence" in verdict["rejections"]
    assert verdict["reason"] == "absent_release_evidence"
    assert "release_opt_out" in verdict["ignored_claims"]


def test_release_opt_out_does_not_change_a_bound_fixture():
    observation = valid_observation()
    observation["future_release_required"] = False
    observation["release_required"] = False
    plain = verify_completion(valid_observation(), now=NOW)
    verdict = verify_completion(observation, now=NOW)
    assert verdict["completion_verified"] is True
    assert verdict["completion_receipt"] == plain["completion_receipt"]
    assert verdict["ignored_claims"] == ["release_opt_out"]


@pytest.mark.parametrize("fixture", [True, False])
def test_recorded_release_evidence_is_not_a_live_fetch(fixture: bool):
    observation = valid_observation()
    observation["release"] = release_evidence(fixture=fixture)
    verdict = verify_completion(observation, now=NOW)
    assert verdict["completion_verified"] is True
    assert verdict["live_release_consulted"] is False


def test_live_fetch_flag_is_not_release_evidence():
    observation = valid_observation()
    observation["release"] = release_evidence(live_fetch=True)
    verdict = verify_completion(observation, now=NOW)
    assert verdict["completion_verified"] is False
    assert verdict["rejections"] == ["absent_release_evidence"]
    assert verdict["live_release_consulted"] is False


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("hold", True, "hold"),
        ("effects_unknown", True, "effects_unknown"),
        ("runnable_count", 1, "work_remains"),
        ("active_task_id", CORE, "work_remains"),
    ],
)
def test_remaining_work_hold_and_unknown_effects_fail_closed(field: str, value, reason: str):
    observation = valid_observation()
    observation[field] = value
    verdict = verify_completion(observation, now=NOW)
    assert verdict["completion_verified"] is False
    assert verdict["rejections"] == [reason]


def test_self_certified_and_authority_claims_cannot_establish_completion():
    certified = valid_observation()
    for task in certified["tasks"]:
        task["receipt"] = task_receipt(task["task_id"], independent=False)
    verdict = verify_completion(certified, now=NOW)
    assert verdict["completion_verified"] is False
    assert verdict["rejections"] == ["self_certified"]
    claimed = valid_observation()
    claimed["completion_authority"] = True
    for task in claimed["tasks"]:
        task["receipt"] = task_receipt(task["task_id"], completion_authority=True)
    verdict = verify_completion(claimed, now=NOW)
    assert verdict["completion_verified"] is False
    assert verdict["completion_authority"] is False
    assert verdict["task_retry_authority"] is False
    assert verdict["rejections"] == ["completion_authority_refused"]
    assert "completion_authority" in verdict["ignored_claims"]


def test_model_prose_cannot_establish_completion():
    observation = drop_receipts(valid_observation())
    observation["prose"] = "All worldwide sources are complete."
    observation["model_output"] = "exit 0"
    verdict = verify_completion(observation, now=NOW)
    assert verdict["completion_verified"] is False
    assert verdict["rejections"] == ["incomplete_evidence"]
    assert verdict["ignored_claims"] == ["prose"]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        ("schema", "schema"),
        ("unknown", "unknown field"),
        ("duplicate_required", "required task"),
        ("outside", "outside the required set"),
        ("clock", "clock"),
        ("freshness", "freshness"),
        ("release_type", "release evidence"),
        ("duplicate_source", "repeats a source"),
        ("empty_required", "required task"),
    ],
)
def test_malformed_observations_are_refused(mutate: str, match: str):
    observation = valid_observation()
    kwargs = {"now": NOW}
    if mutate == "schema":
        observation["schema"] = "wwl.done@1"
    elif mutate == "unknown":
        observation["verified"] = True
    elif mutate == "duplicate_required":
        observation["required_task_ids"] = [CORE, CORE]
    elif mutate == "outside":
        observation["tasks"][0]["task_id"] = "WWL-001"
        observation["tasks"][0]["receipt"] = task_receipt("WWL-001")
    elif mutate == "clock":
        kwargs["now"] = math.nan
    elif mutate == "freshness":
        kwargs["freshness_seconds"] = 0
    elif mutate == "release_type":
        observation["release"] = "released"
    elif mutate == "duplicate_source":
        observation["sources"][1]["source_id"] = observation["sources"][0]["source_id"]
    else:
        observation["required_task_ids"] = []
        observation["tasks"] = []
        observation["expansion_child_ids"] = []
    with pytest.raises(CompletionContractError, match=match):
        verify_completion(observation, **kwargs)


def test_watchdog_snapshot_claims_cannot_establish_completion():
    snapshot = {
        "schema": "wwl.watchdog.snapshot@1",
        "run_id": RUN,
        "goal_id": GOAL,
        "accepted_tree": "tree-1",
        "required_task_ids": [CORE, "WWL-001"],
        "verified_task_ids": [CORE, "WWL-001"],
        "task_statuses": {CORE: "done", "WWL-001": "completed"},
        "process_exit_code": 0,
        "completion_verified": True,
        "completion_receipt": "sha256:" + ("cd" * 32),
        "native_completion_receipts_by_alias": {CORE: "sha256:" + ("ab" * 32)},
        "runnable_count": 0,
        "task_id": "",
        "hold": False,
        "effects_unknown": False,
    }
    verdict = as_watchdog_verifier(snapshot, now=NOW)
    assert verdict["completion_verified"] is False
    assert verdict["completion_authority"] is False
    assert verdict["completion_receipt"] == ""
    assert verdict["live_release_consulted"] is False
    for code in ("done_flag", "zero_exit", "stale_receipt", "unresolved_source_blocker", "absent_release_evidence"):
        assert code in verdict["rejections"]
    hooked = as_watchdog_verifier(valid_observation(), now=NOW)
    assert hooked["completion_verified"] is True
    assert hooked["completion_authority"] is False


def test_verifier_does_not_contain_a_live_release_client():
    source = Path(ADAPTER).read_text(encoding="utf-8")
    for banned in ("urllib", "requests", "socket", "urlopen", "subprocess", "http.client"):
        assert banned not in source
    obligations = implementation_obligations()
    assert obligations["non_authoritative"] == list(NON_AUTHORITATIVE)
    assert set(UNRESOLVED_BLOCKERS) <= set(BLOCKER_KINDS)
    for task_id in ("WWL-111", "WWL-112", "WWL-113", "WWL-114"):
        assert task_id not in obligations["required_release_tasks"]


def test_receipt_matches_the_completion_contract():
    receipt = json.loads(Path("artifacts/worldwide-legal-search/receipts/WWL-106.json").read_text(encoding="utf-8"))
    assert receipt["schema"] == "worldwide-legal-search.task-receipt@1"
    assert receipt["task_id"] == "WWL-106"
    assert receipt["title"] == "Implement and test native completion verification"
    assert receipt["goal_id"] == "WWL-G090"
    assert receipt["subgoal_id"] == "WWL-G090.S2"
    assert receipt["root_goal_id"] == "WWL-G000"
    assert receipt["depends_on"] == ["WWL-105"]
    assert receipt["non_authoritative"] == list(NON_AUTHORITATIVE)
    assert receipt["future_release_required"] is False
    assert receipt["live_release_consulted"] is False
    assert receipt["required_release_tasks"] == []
    assert receipt["completion_authority"] is False
    assert receipt["self_certified_completion"] is False
    assert receipt["execution_completed"] is False
    assert receipt["outputs"] == [
        "test/api/test_worldwide_legal_search_completion.py",
        ADAPTER,
        "artifacts/worldwide-legal-search/receipts/WWL-106.json",
    ]
    assert receipt["validation_argv"] == ARGV
    assert "does not grant campaign completion" in receipt["limitations"][-1]
    text = "\n".join(receipt["limitations"])
    for phrase in ("done flag", "zero", "stale", "skipped", "source blocker", "release", "WWL-111"):
        assert phrase in text
