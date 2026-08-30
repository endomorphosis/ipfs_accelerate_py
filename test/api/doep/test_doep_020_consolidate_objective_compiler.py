"""Independent current-tree checks for the DOEP-020 canonical compiler seam."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_compiler import (
    CANONICAL_OBJECTIVE_COMPILER,
    CompilationStatus,
    FormalPlanCompiler,
    compile_formal_plan,
    compile_objective_snapshot,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
COMPILER_PATH = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "planning"
    / "formal_plan_compiler.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "outputs"
    / "DOEP-020.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "receipts"
    / "DOEP-020.json"
)

OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/planning/formal_plan_compiler.py",
    "test/api/doep/test_doep_020_consolidate_objective_compiler.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-020.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-020.json",
)
TASK_CID = "sha256:aa601db60ea44648faa566d805529e8e980d464498d232db49f78ffbe838eb40"
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


def _source() -> dict[str, Any]:
    return {
        "repository_tree_id": "tree:doep-020",
        "objectives": [{"goal_cid": "goal:compiler", "owner_actor_id": "supervisor"}],
        "tasks": [
            {
                "task_cid": "task:compiler",
                "goal_cid": "goal:compiler",
                "actor_id": "agent:compiler",
                "acceptance_criteria": ["focused tests pass"],
            }
        ],
        "policies": [{"policy_cid": "policy:compiler"}],
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_objective_snapshot_is_a_thin_adapter_to_the_existing_compiler() -> None:
    source = _source()
    canonical = compile_objective_snapshot(source)
    compatibility = compile_formal_plan(source)
    instance = FormalPlanCompiler().compile(source)

    assert CANONICAL_OBJECTIVE_COMPILER == "FormalPlanCompiler@1"
    assert canonical.status is CompilationStatus.COMPILED
    assert canonical.to_dict() == compatibility.to_dict() == instance.to_dict()
    assert canonical.plan is not None
    assert canonical.plan.repository_tree_id == "tree:doep-020"
    assert {task.task_id for task in canonical.plan.tasks} == {"task:compiler"}


def test_manifest_and_candidate_receipt_bind_the_exact_current_tree_outputs() -> None:
    manifest = _load_json(OUTPUT_PATH)
    receipt = _load_json(RECEIPT_PATH)

    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-020"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["plan_revision"] == "DOEP-PLAN-V5"
        assert payload["board_namespace"] == "agent-supervisor-direct-objective-and-event-driven-planning-v1"
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
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(COMPILER_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
    assert receipt["required_evidence"]["source_commit_tree_gitlinks"] == BASE_REPOSITORIES
