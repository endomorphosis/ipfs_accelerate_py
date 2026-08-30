"""Independent current-tree checks for the DOEP-040 task-state contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CANONICAL_TASK_STATE_MACHINE_INTERFACE,
    ControlPlaneContractError,
    TaskState,
    TaskStateSnapshot,
    allowed_task_transitions,
    assert_task_transition,
    canonical_task_state,
    is_terminal_task_state,
    task_transition_allowed,
)

ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_contracts.py"
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = ACCELERATE_ROOT / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-040.json"
RECEIPT_PATH = ACCELERATE_ROOT / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-040.json"
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_contracts.py",
    "test/api/doep/test_doep_040_formalize_canonical_task_state_machine.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-040.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-040.json",
)
TASK_CID = "sha256:3c431b8c065f167f973c1358b1806c6ddd27148743f5fdd454065475327a9acc"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {"commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f", "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7"},
    "ipfs_datasets_py": {"commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7", "tree": "456e09b51d6a07a3a5873436df24054768195320"},
    "ipfs_kit_py": {"commit": "b6c65ba732733d7e33852713ba18aa3b12235668", "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2"},
    "lift_coding": {"commit": "bb8869ed72eb7002434345d9969efee729c4f7f6", "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42"},
}


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(**updates: object) -> TaskStateSnapshot:
    values: dict[str, object] = {
        "task_cid": TASK_CID, "state": TaskState.IN_PROGRESS, "revision": 7,
        "lease_id": "lease:current", "fence_epoch": 3,
        "policy_cid": "policy:current", "repository_tree_id": "tree:current",
        "plan_cid": PLAN_CID, "plan_epoch": 1,
    }
    values.update(updates)
    return TaskStateSnapshot(**values)


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_canonical_lifecycle_is_closed_and_unknown_effects_reconcile() -> None:
    assert CANONICAL_TASK_STATE_MACHINE_INTERFACE == "CanonicalTaskStateMachine@1"
    assert canonical_task_state("todo") is TaskState.READY
    assert canonical_task_state("done") is TaskState.COMPLETED
    assert task_transition_allowed(TaskState.READY, TaskState.CLAIMED)
    assert task_transition_allowed(TaskState.CLAIMED, TaskState.IN_PROGRESS)
    assert TaskState.RECONCILING in allowed_task_transitions(TaskState.PROVIDER_OUTCOME_UNKNOWN)
    assert not task_transition_allowed(TaskState.PROVIDER_OUTCOME_UNKNOWN, TaskState.RETRYING)
    assert is_terminal_task_state(TaskState.COMPLETED)
    assert not task_transition_allowed(TaskState.COMPLETED, TaskState.READY)
    with pytest.raises(ControlPlaneContractError):
        assert_task_transition(TaskState.READY, TaskState.COMPLETED)
    with pytest.raises(ControlPlaneContractError):
        canonical_task_state("worker_says_complete")


def test_completion_is_fenced_by_current_lease_fence_policy_tree_and_plan() -> None:
    candidate = _snapshot()
    assert candidate.may_complete_against(_snapshot())
    for field, stale in (
        ("revision", 8), ("lease_id", "lease:replaced"), ("fence_epoch", 4),
        ("policy_cid", "policy:replaced"), ("repository_tree_id", "tree:replaced"),
        ("plan_cid", "sha256:" + "a" * 64), ("plan_epoch", 2),
    ):
        assert not candidate.may_complete_against(_snapshot(**{field: stale}))
    assert not _snapshot(state=TaskState.CLAIMED).may_complete_against(_snapshot(state=TaskState.CLAIMED))
    assert not _snapshot(state=TaskState.PROVIDER_OUTCOME_UNKNOWN).may_complete_against(_snapshot(state=TaskState.PROVIDER_OUTCOME_UNKNOWN))


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in ((manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"), (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1")):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-040"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["expected_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["write_scope"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["base_repositories"] == BASE_REPOSITORIES
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(CONTRACT_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == "pending_independent_fenced_supervisor"
