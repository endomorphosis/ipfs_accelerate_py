"""Independent current-tree checks for the DOEP-012 objective-submission service."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.entrypoints.intent_service import (
    CANONICAL_OBJECTIVE_SUBMISSION_ENTRYPOINT,
    CANONICAL_OBJECTIVE_SUBMISSION_SERVICE,
    ObjectiveSubmissionContractError,
    ObjectiveSubmissionPolicyError,
    PromptToRunUnavailableError,
    SupervisorIntentService,
    submit_objective,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.objective_resolver import (
    content_addressed_prompt_objective,
)
from ipfs_datasets_py.logic.intent_ir.schema import (
    OBJECTIVE_MATERIALIZATION_RECEIPT_FORBIDDEN_FIELDS,
    OBJECTIVE_MATERIALIZATION_RECEIPT_SCHEMA,
    SUPERVISOR_OBJECTIVE_INTENT_FORBIDDEN_FIELDS,
    ObjectiveMaterializationReceipt,
    SupervisorObjectiveIntent,
    SupervisorObjectiveSubmitterKind,
    idea_text_sha256,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
SERVICE_PATH = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "entrypoints"
    / "intent_service.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "outputs"
    / "DOEP-012.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "receipts"
    / "DOEP-012.json"
)

OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/entrypoints/intent_service.py",
    "test/api/doep/test_doep_012_implement_canonical_objective_submission_service.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-012.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-012.json",
)
TASK_CID = "sha256:0c1390dfe50331773e0280c83b9648d7b8d4c1a4ccdfc4a0f6387c7c5ca21dc6"
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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _minimal_intent(**overrides: Any) -> SupervisorObjectiveIntent:
    idea = "Submit one bounded high-level idea to the existing supervisor."
    values: dict[str, Any] = {
        "intent_id": "doep.objective.intent.example",
        "idea_text": idea,
        "idea_sha256": idea_text_sha256(idea),
        "submitter_kind": SupervisorObjectiveSubmitterKind.HUMAN,
        "caller": "caller:example-principal",
        "repository_id": "repository:sha256:example",
        "board_namespace": "agent-supervisor-direct-objective-and-event-driven-planning-v1",
        "title_hint": "Example direct objective",
        "tags": ("direct-objective", "doep"),
    }
    values.update(overrides)
    return SupervisorObjectiveIntent(**values)


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_canonical_service_is_the_existing_intent_service() -> None:
    assert CANONICAL_OBJECTIVE_SUBMISSION_SERVICE == "SupervisorIntentService@1"
    assert CANONICAL_OBJECTIVE_SUBMISSION_ENTRYPOINT == "submit_objective"
    assert (
        submit_objective.__module__
        == "ipfs_accelerate_py.agent_supervisor.entrypoints.intent_service"
    )
    assert hasattr(SupervisorIntentService, "submit_objective")
    assert hasattr(SupervisorIntentService, "run")
    assert hasattr(SupervisorIntentService, "start_or_resume")
    assert hasattr(SupervisorIntentService, "adopt_or_resume")
    source = SERVICE_PATH.read_text(encoding="utf-8")
    assert "class SupervisorIntentService" in source
    assert "def submit_objective" in source
    assert "class CanonicalObjectiveSubmissionService" not in source
    assert "FormalPlanCompiler" not in source
    assert "PromptProgramMaterializer" not in source
    assert "content_addressed_prompt_objective" in source


def test_submit_objective_materializes_datasets_receipt() -> None:
    intent = _minimal_intent()
    receipt = submit_objective(intent)
    assert isinstance(receipt, ObjectiveMaterializationReceipt)
    assert receipt.schema == OBJECTIVE_MATERIALIZATION_RECEIPT_SCHEMA
    assert receipt.is_completion_authority is False
    assert receipt.intent_id == intent.intent_id
    assert receipt.objective_id == intent.intent_id
    prompt_cid = cid_for_dag_json(
        {"schema": intent.schema, "intent": intent.to_dict()}
    )
    expected_objective, expected_revision, _pending = content_addressed_prompt_objective(
        prompt_cid,
        repository_id=intent.repository_id,
    )
    assert receipt.objective_cid == expected_objective
    assert receipt.objective_revision_cid == expected_revision
    assert "plan" not in receipt.to_dict()
    assert "plan_root_cid" not in receipt.to_dict()
    assert submit_objective(intent.to_dict()).to_dict() == receipt.to_dict()
    assert SupervisorIntentService().submit_objective(intent).to_dict() == receipt.to_dict()
    assert submit_objective(intent).to_dict() == receipt.to_dict()
    payload_keys = set(receipt.to_dict())
    assert payload_keys.isdisjoint(OBJECTIVE_MATERIALIZATION_RECEIPT_FORBIDDEN_FIELDS)
    assert SUPERVISOR_OBJECTIVE_INTENT_FORBIDDEN_FIELDS & {
        "policy",
        "policy_id",
        "lease_id",
        "authorization",
        "terminalize",
    }
    assert payload_keys.isdisjoint(
        {
            "policy",
            "policy_id",
            "lease_id",
            "authorization",
            "terminalize",
            "duckdb",
            "completion_authoritative",
        }
    )


def test_caller_supplied_policy_and_authority_fail_closed() -> None:
    intent = _minimal_intent()
    with pytest.raises(ObjectiveSubmissionPolicyError):
        submit_objective(intent, policy_id="policy:forbidden")
    with pytest.raises(ObjectiveSubmissionPolicyError):
        submit_objective(intent, lease_id="lease:x")
    with pytest.raises(ObjectiveSubmissionPolicyError):
        submit_objective({**intent.to_dict(), "policy_id": "policy:forbidden"})
    with pytest.raises(ObjectiveSubmissionPolicyError):
        submit_objective({**intent.to_dict(), "objective_cid": "cid:caller-supplied"})
    with pytest.raises(ObjectiveSubmissionContractError):
        submit_objective("raw idea text is not an intent")
    with pytest.raises(ObjectiveSubmissionContractError):
        submit_objective(_minimal_intent(idea_sha256="0" * 64))


def test_submission_does_not_require_runtime_factory_or_start_a_run() -> None:
    service = SupervisorIntentService()
    receipt = service.submit_objective(_minimal_intent())
    assert receipt.is_completion_authority is False
    with pytest.raises(PromptToRunUnavailableError):
        service.run(object())  # type: ignore[arg-type]


def test_manifest_and_candidate_receipt_bind_the_exact_current_tree_outputs() -> None:
    manifest = _load_json(OUTPUT_PATH)
    receipt = _load_json(RECEIPT_PATH)

    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-012"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["plan_revision"] == "DOEP-PLAN-V5"
        assert payload["board_namespace"] == (
            "agent-supervisor-direct-objective-and-event-driven-planning-v1"
        )
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True

    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    service = manifest["canonical_objective_submission_service"]
    assert service["identifier"] == CANONICAL_OBJECTIVE_SUBMISSION_SERVICE
    assert service["entrypoint"] == CANONICAL_OBJECTIVE_SUBMISSION_ENTRYPOINT
    assert service["carrier"] == "SupervisorIntentService"
    assert service["callers_supply_authoritative_policy"] is False
    assert service["completion_authority"] is False
    assert service["competing_subsystem_created"] is False
    assert service["prompt_to_run_saga_preserved"] is True
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["expected_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["write_scope"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["base_repositories"] == BASE_REPOSITORIES
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(SERVICE_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
    assert receipt["required_evidence"]["source_commit_tree_gitlinks"] == BASE_REPOSITORIES
