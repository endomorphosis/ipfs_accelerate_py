"""Independent current-tree checks for the DOEP-013 Python client facade."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.entrypoints.authority_resolver import (
    AuthorityAdmissionRequest,
    AuthorityAuthenticationError,
    AuthorityIdempotencyError,
    AuthorityIdempotencyStore,
    AuthorityResolutionRequest,
    ExpectedEffect,
    InvocationMode,
    admit_authority as service_admit_authority,
    install_local_worktree_authority,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import (
    CANONICAL_PYTHON_CLIENT,
    CANONICAL_PYTHON_CLIENT_AUTHORITY_ENTRYPOINT,
    CANONICAL_PYTHON_CLIENT_ENTRYPOINT,
    CANONICAL_PYTHON_CLIENT_MODULE,
    Supervisor,
    admit_authority,
    submit_objective,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.intent_service import (
    ObjectiveSubmissionContractError,
    ObjectiveSubmissionPolicyError,
    submit_objective as service_submit_objective,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.objective_resolver import (
    content_addressed_prompt_objective,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.service_factory import (
    ACTIVATION_TASK_ID,
    COMPOSITION_MANIFEST_SCHEMA,
    ProductionServiceComposition,
    ProductionServiceCompositionManifest,
    _PRODUCTION_BACKENDS,
)
from ipfs_datasets_py.logic.intent_ir.schema import (
    OBJECTIVE_MATERIALIZATION_RECEIPT_FORBIDDEN_FIELDS,
    OBJECTIVE_MATERIALIZATION_RECEIPT_SCHEMA,
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
    / "facade.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "outputs"
    / "DOEP-013.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "receipts"
    / "DOEP-013.json"
)

OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/entrypoints/facade.py",
    "test/api/doep/test_doep_013_add_python_client.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-013.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-013.json",
)
TASK_CID = "sha256:9512c44dd447a08c3b2991756943a55ee76cc927eb5025988bbbfd334d0793c5"
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
    idea = "Submit one bounded high-level idea through the Python client."
    values: dict[str, Any] = {
        "intent_id": "doep.objective.intent.python-client",
        "idea_text": idea,
        "idea_sha256": idea_text_sha256(idea),
        "submitter_kind": SupervisorObjectiveSubmitterKind.HUMAN,
        "caller": "caller:python-client-example",
        "repository_id": "repository:sha256:example",
        "board_namespace": "agent-supervisor-direct-objective-and-event-driven-planning-v1",
        "title_hint": "Example Python-client objective",
        "tags": ("direct-objective", "doep", "python-client"),
    }
    values.update(overrides)
    return SupervisorObjectiveIntent(**values)


def _client_supervisor() -> Supervisor:
    manifest = ProductionServiceCompositionManifest(
        schema=COMPOSITION_MANIFEST_SCHEMA,
        composition_cid=cid_for_dag_json({"fixture": "doep-013-python-client"}),
        activation_task_id=ACTIVATION_TASK_ID,
        generation=1,
        backends=dict(_PRODUCTION_BACKENDS),
        objective_refill_enabled=False,
        monitor_enabled=False,
    )
    return Supervisor(ProductionServiceComposition(manifest=manifest))


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_canonical_python_client_extends_existing_supervisor_facade() -> None:
    assert CANONICAL_PYTHON_CLIENT == "Supervisor@1"
    assert CANONICAL_PYTHON_CLIENT_ENTRYPOINT == "submit_objective"
    assert CANONICAL_PYTHON_CLIENT_AUTHORITY_ENTRYPOINT == "admit_authority"
    assert CANONICAL_PYTHON_CLIENT_MODULE == (
        "ipfs_accelerate_py.agent_supervisor.entrypoints.facade"
    )
    assert submit_objective.__module__ == CANONICAL_PYTHON_CLIENT_MODULE
    assert admit_authority.__module__ == CANONICAL_PYTHON_CLIENT_MODULE
    assert hasattr(Supervisor, "open")
    assert hasattr(Supervisor, "run")
    assert hasattr(Supervisor, "submit_objective")
    assert hasattr(Supervisor, "admit_authority")
    source = SERVICE_PATH.read_text(encoding="utf-8")
    assert "class Supervisor" in source
    assert "def submit_objective" in source
    assert "def admit_authority" in source
    assert "from .intent_service import submit_objective" in source
    assert "from .authority_resolver import admit_authority" in source
    assert "class PythonClient" not in source
    assert "class DirectObjectiveClient" not in source
    assert "class CanonicalPythonClient" not in source
    assert "FormalPlanCompiler" not in source
    assert "class AuthService" not in source


def test_python_client_submit_objective_matches_canonical_service() -> None:
    intent = _minimal_intent()
    via_client = submit_objective(intent)
    via_service = service_submit_objective(intent)
    assert isinstance(via_client, ObjectiveMaterializationReceipt)
    assert via_client.schema == OBJECTIVE_MATERIALIZATION_RECEIPT_SCHEMA
    assert via_client.is_completion_authority is False
    assert via_client.to_dict() == via_service.to_dict()
    assert _client_supervisor().submit_objective(intent).to_dict() == via_service.to_dict()
    assert submit_objective(intent.to_dict()).to_dict() == via_service.to_dict()
    prompt_cid = cid_for_dag_json(
        {"schema": intent.schema, "intent": intent.to_dict()}
    )
    expected_objective, expected_revision, _pending = content_addressed_prompt_objective(
        prompt_cid,
        repository_id=intent.repository_id,
    )
    assert via_client.objective_cid == expected_objective
    assert via_client.objective_revision_cid == expected_revision
    assert set(via_client.to_dict()).isdisjoint(OBJECTIVE_MATERIALIZATION_RECEIPT_FORBIDDEN_FIELDS)
    assert "plan" not in via_client.to_dict()
    assert "duckdb" not in via_client.to_dict()


def test_python_client_rejects_caller_supplied_policy() -> None:
    intent = _minimal_intent()
    with pytest.raises(ObjectiveSubmissionPolicyError):
        submit_objective(intent, policy_id="policy:forbidden")
    with pytest.raises(ObjectiveSubmissionPolicyError):
        submit_objective(intent, lease_id="lease:x")
    with pytest.raises(ObjectiveSubmissionPolicyError):
        submit_objective({**intent.to_dict(), "policy_id": "policy:forbidden"})
    with pytest.raises(ObjectiveSubmissionContractError):
        submit_objective("raw idea text is not an intent")
    with pytest.raises(ObjectiveSubmissionPolicyError):
        _client_supervisor().submit_objective(intent, authorization="auth:forged")


def test_python_client_admit_authority_matches_canonical_service() -> None:
    local = install_local_worktree_authority(
        "did:key:local-owner",
        signing_key_handle="key:local-worktree-1",
        installed_at_ms=1_700_000_000_000,
    )
    request = AuthorityAdmissionRequest(
        resolution=AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            local_worktree_authority=local,
        )
    )
    via_client = admit_authority(request)
    via_service = service_admit_authority(request)
    assert via_client.authorized is True
    assert (
        via_client.resolution.decision_reference_cid
        == via_service.resolution.decision_reference_cid
    )
    assert _client_supervisor().admit_authority(request).authorized is True

    store = AuthorityIdempotencyStore()
    keyed = AuthorityAdmissionRequest(
        resolution=AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            local_worktree_authority=local,
        ),
        idempotency_key="doep-013-admission-1",
    )
    first = admit_authority(keyed, store=store)
    second = admit_authority(keyed, store=store)
    assert first.idempotency is not None and first.idempotency.replayed is False
    assert second.idempotency is not None and second.idempotency.replayed is True

    with pytest.raises(AuthorityAuthenticationError):
        admit_authority(
            AuthorityAdmissionRequest(
                resolution=AuthorityResolutionRequest(
                    credentials_present=True,
                    prompt_claimed_principal="did:key:forged",
                ),
                require_authenticated_principal=True,
            )
        )
    with pytest.raises(AuthorityIdempotencyError):
        admit_authority(
            AuthorityAdmissionRequest(
                resolution=AuthorityResolutionRequest(
                    mode=InvocationMode.WORKTREE,
                    local_worktree_authority=local,
                ),
                require_idempotency_key=True,
                idempotency_key="",
            )
        )
    # Local worktree effects remain the ASE-008 ceiling; client does not widen.
    assert (
        ExpectedEffect.EDIT_ISOLATED_WORKTREE
        in via_client.resolution.effect_ceiling.allowed_effects
    )


def test_manifest_and_candidate_receipt_bind_the_exact_current_tree_outputs() -> None:
    manifest = _load_json(OUTPUT_PATH)
    receipt = _load_json(RECEIPT_PATH)

    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-013"
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
    client = manifest["canonical_python_client"]
    assert client["identifier"] == CANONICAL_PYTHON_CLIENT
    assert client["entrypoint"] == CANONICAL_PYTHON_CLIENT_ENTRYPOINT
    assert client["authority_entrypoint"] == CANONICAL_PYTHON_CLIENT_AUTHORITY_ENTRYPOINT
    assert client["carrier"] == "Supervisor"
    assert client["module"] == CANONICAL_PYTHON_CLIENT_MODULE
    assert client["objective_submission_delegate"] == (
        "ipfs_accelerate_py.agent_supervisor.entrypoints.intent_service.submit_objective"
    )
    assert client["authority_admission_delegate"] == (
        "ipfs_accelerate_py.agent_supervisor.entrypoints.authority_resolver.admit_authority"
    )
    assert client["ase3009_preserved"] is True
    assert client["callers_supply_authoritative_policy"] is False
    assert client["completion_authority"] is False
    assert client["competing_subsystem_created"] is False

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
