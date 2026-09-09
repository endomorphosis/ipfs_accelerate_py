"""Independent current-tree checks for the DOEP-015 MCP/MCP++ adapters."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.authority_resolver import (
    AuthorityAdmission,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    prompt_entrypoints as adapters,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
ADAPTER_PATH = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/prompt_entrypoints.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-015.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-015.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/mcp_server/tools/agent_supervisor_tools/prompt_entrypoints.py",
    "test/api/doep/test_doep_015_add_mcp_mcp_adapters.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-015.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-015.json",
)
TASK_CID = "sha256:9022e48c40b4c930a6a78c67f9ae6d71100ea114147a5800d662d52fd413ff23"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _admission(*, delegated: bool = False) -> AuthorityAdmission:
    # The adapter only accepts the typed, host-produced DOEP-016 record.  Its
    # resolution is intentionally opaque to the MCP boundary.
    return AuthorityAdmission(
        resolution=SimpleNamespace(authorized=True),
        delegation=object() if delegated else None,
    )


def _submit(intent: Any) -> dict[str, Any]:
    return {"receipt_id": "cid:receipt", "intent_id": intent["intent_id"]}


def test_declared_outputs_exist_and_artifacts_describe_this_tree() -> None:
    assert ADAPTER_PATH.is_file()
    assert TEST_PATH.is_file()
    assert OUTPUT_PATH.is_file()
    assert RECEIPT_PATH.is_file()
    output = _load_json(OUTPUT_PATH)
    receipt = _load_json(RECEIPT_PATH)
    assert output["task_id"] == receipt["task_id"] == "DOEP-015"
    assert output["task_cid"] == receipt["task_cid"] == TASK_CID
    assert tuple(output["declared_outputs"]) == OWNER_RELATIVE_OUTPUTS
    assert tuple(receipt["expected_outputs"]) == OWNER_RELATIVE_OUTPUTS
    assert receipt["outputs_present"] == {item: True for item in OWNER_RELATIVE_OUTPUTS}
    assert receipt["path_digests"][OWNER_RELATIVE_OUTPUTS[0]] == _sha256_file(ADAPTER_PATH)
    assert receipt["path_digests"][OWNER_RELATIVE_OUTPUTS[1]] == _sha256_file(TEST_PATH)
    assert receipt["path_digests"][OWNER_RELATIVE_OUTPUTS[2]] == _sha256_file(OUTPUT_PATH)


def test_both_adapters_delegate_to_the_one_canonical_submission_callable() -> None:
    intent = {"intent_id": "intent:one"}
    expected = _submit(intent)
    assert adapters.submit_objective_mcp(
        intent, admission=_admission(), submit_objective=_submit
    ) == expected
    assert adapters.submit_objective_mcpplusplus(
        intent, admission=_admission(delegated=True), submit_objective=_submit
    ) == expected
    source = ADAPTER_PATH.read_text(encoding="utf-8")
    assert "from ipfs_accelerate_py.agent_supervisor.entrypoints.intent_service import" in source
    assert "submit_objective as _submit_objective" in source
    assert "class ObjectiveSubmissionService" not in source
    assert "class MCPObjectiveStore" not in source


def test_adapters_fail_closed_for_untrusted_or_undelegated_authority() -> None:
    intent = {"intent_id": "intent:one"}
    with pytest.raises(adapters.ObjectiveSubmissionMCPError):
        adapters.submit_objective_mcp(intent, admission=None, submit_objective=_submit)
    with pytest.raises(adapters.ObjectiveSubmissionMCPError):
        adapters.submit_objective_mcp(intent, admission={"authorized": True}, submit_objective=_submit)
    with pytest.raises(adapters.ObjectiveSubmissionMCPError):
        adapters.submit_objective_mcpplusplus(
            intent, admission=_admission(), submit_objective=_submit
        )


def test_public_tools_never_accept_authority_in_client_schema_and_return_typed_errors() -> None:
    class Manager:
        def __init__(self) -> None:
            self.entries: list[dict[str, Any]] = []

        def register_tool(self, **kwargs: Any) -> None:
            self.entries.append(kwargs)

    manager = Manager()
    adapters.register_prompt_lifecycle_tools(manager)
    entries = {entry["name"]: entry for entry in manager.entries}
    assert set(adapters.MCP_OBJECTIVE_SUBMISSION_TOOLS) <= set(entries)
    for name in adapters.MCP_OBJECTIVE_SUBMISSION_TOOLS:
        assert "authority_admission" not in entries[name]["input_schema"]["properties"]
    response = asyncio.run(
        adapters.agent_supervisor_submit_objective({"intent_id": "intent:one"})
    )
    assert response["ok"] is False
    assert response["error_code"] == "ObjectiveSubmissionMCPError"


def test_discovery_is_cold_and_states_the_authority_contract() -> None:
    manifest = adapters.objective_submission_mcp_discovery_manifest()
    assert manifest["adapter"] == "ObjectiveSubmissionMCPAdapter@1"
    assert manifest["tools"] == list(adapters.MCP_OBJECTIVE_SUBMISSION_TOOLS)
    assert manifest["submission_delegate"].endswith("intent_service.submit_objective")
    assert manifest["authority_delegate"].endswith("authority_resolver.admit_authority")
    assert manifest["mcp_requires_authenticated_admission"] is True
    assert manifest["mcpplusplus_requires_verified_delegation"] is True
    assert manifest["callers_supply_authoritative_policy"] is False
    assert manifest["completion_authority"] is False
    assert manifest["cold_registration"] is True
