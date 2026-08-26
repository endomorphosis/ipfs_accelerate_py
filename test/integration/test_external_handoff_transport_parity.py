"""EAAEF-115: closed-vocabulary parity across Python, CLI, MCP, and MCP++."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.api.external_handoff import (
    HANDOFF_API_OPERATIONS,
    ExternalHandoffAPI,
    ExternalHandoffAPIError,
    ExternalHandoffReceipt,
    ExternalHandoffRequest,
    HandoffApiVerdict,
)
from ipfs_accelerate_py.cli.supervisor_handoff import (
    CLI_COMMANDS,
    CLI_TO_OPERATION,
    SupervisorHandoffCLIError,
    request_from_args,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools.external_handoff import (
    HANDOFF_MCP_OPERATIONS,
    execute_external_handoff_operation,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
KIT_ROOT = REPO_ROOT / "ipfs_kit_py"
if str(KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(KIT_ROOT))

from ipfs_kit_py.mcp_server.mcplusplus.external_agent_handoff import (  # noqa: E402
    HANDOFF_OPERATIONS as MCPP_OPERATIONS,
    ExternalAgentHandoffBindError,
    bind_handoff_operation,
)

RECEIPT_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "external_agent_autonomous_execution_fabric"
    / "receipts"
    / "transport_parity.json"
)

OPERATOR = "principal:operator"
WORKER = "principal:worker"
SESSION = "session:parity"
REPO = "repo:example"
UNKNOWN = "mutate_production"


def _start_request(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "principal_id": OPERATOR,
        "worker_principal_id": WORKER,
        "session_id": SESSION,
        "repository_id": REPO,
        "objective_id": "objective:handoff",
        "idempotency_key": "idem:parity-1",
    }
    values.update(changes)
    return values


def _load_receipt() -> dict[str, object]:
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


def test_receipt_contract_is_fail_closed_and_not_live() -> None:
    receipt = _load_receipt()
    assert receipt["schema"].endswith("external-handoff-transport-parity@1")
    assert receipt["task_id"] == "EAAEF-115"
    assert receipt["evidence_mode"] == "contract_fail_closed"
    assert receipt["qualification_scope"] == "offline_transport_contract_only"
    assert receipt["task_completion_claimed"] is False
    assert receipt["production_qualification_claimed"] is False
    assert receipt["live_runtime_invoked"] is False
    assert receipt["live_quack"] is False
    assert receipt["live_docker"] is False
    assert receipt["preview_is_handoff"] is False
    assert receipt["self_approval"] is False
    assert receipt["unknown_operation_reason_code"] == "unknown_operation"


def test_closed_operation_vocabulary_matches_receipt() -> None:
    receipt = _load_receipt()
    python_ops = list(HANDOFF_API_OPERATIONS)
    cli_ops = [CLI_TO_OPERATION[name] for name in CLI_COMMANDS]
    mcp_ops = list(HANDOFF_MCP_OPERATIONS)
    mcpp_ops = list(MCPP_OPERATIONS)
    assert python_ops == receipt["transports"]["python"]["operations"]
    assert list(CLI_COMMANDS) == receipt["transports"]["cli"]["commands"]
    assert cli_ops == receipt["transports"]["cli"]["operations"]
    assert mcp_ops == receipt["transports"]["mcp"]["operations"]
    assert mcpp_ops == receipt["transports"]["mcplusplus"]["operations"]
    assert python_ops == receipt["canonical_operations"]
    shared = set(python_ops).intersection(cli_ops, mcp_ops, mcpp_ops)
    assert sorted(shared) == receipt["shared_operations"]
    for operations in (python_ops, cli_ops, mcp_ops, mcpp_ops):
        extra = set(operations).difference(HANDOFF_API_OPERATIONS)
        assert extra == set()


def test_unknown_operations_fail_closed_on_every_transport() -> None:
    with pytest.raises(ExternalHandoffAPIError) as python_err:
        ExternalHandoffRequest(operation=UNKNOWN, principal_id=OPERATOR)
    assert python_err.value.reason_code == "unknown_operation"

    with pytest.raises(SupervisorHandoffCLIError) as cli_err:
        request_from_args(argparse.Namespace(command=UNKNOWN, request_file=None))
    assert cli_err.value.reason_code == "unknown_operation"

    mcp_result = execute_external_handoff_operation(
        UNKNOWN, {"principal_id": OPERATOR}
    )
    assert mcp_result["ok"] is False
    assert mcp_result["error_code"] == "unknown_operation"

    with pytest.raises(ExternalAgentHandoffBindError) as mcpp_err:
        bind_handoff_operation(UNKNOWN)
    assert mcpp_err.value.reason_code == "unknown_operation"


def test_equivalent_handoff_inputs_share_canonical_identities() -> None:
    request = _start_request()
    python_receipt = ExternalHandoffAPI().handoff(request)

    cli_request = request_from_args(
        argparse.Namespace(
            command="handoff",
            request_file=None,
            principal_id=OPERATOR,
            worker_principal_id=WORKER,
            session_id=SESSION,
            repository_id=REPO,
            objective_id="objective:handoff",
            idempotency_key="idem:parity-1",
            instruction="",
            instruction_file=None,
            instruction_stdin=False,
            reason="",
            reason_file=None,
        )
    )
    cli_receipt = ExternalHandoffAPI().handoff(cli_request)

    mcp_result = execute_external_handoff_operation(
        "handoff", request, api=ExternalHandoffAPI()
    )
    assert mcp_result["ok"] is True
    mcp_receipt = mcp_result["receipt"]

    mcpp_bound = bind_handoff_operation(
        "handoff",
        profiles={
            "interface": "profile-a",
            "artifact": "profile-b",
            "delegation": "profile-c",
            "event": "profile-f",
            "fencing": "durable-executor",
        },
        durable_executor_configured=True,
        runtime_method="handoff",
    )
    assert mcpp_bound["new_profile"] is False
    assert mcpp_bound["storage_authority"] is False
    assert mcpp_bound["live_runtime_invoked"] is False

    identities = {
        "content_id": python_receipt.content_id,
        "request_id": python_receipt.request_id,
        "run_id": python_receipt.run_id,
        "authority_id": python_receipt.authority_id,
    }
    assert cli_receipt.content_id == identities["content_id"]
    assert cli_receipt.request_id == identities["request_id"]
    assert cli_receipt.run_id == identities["run_id"]
    assert cli_receipt.authority_id == identities["authority_id"]
    mcp_obj = ExternalHandoffReceipt.from_dict(mcp_receipt)
    assert mcp_obj.content_id == identities["content_id"]
    assert mcp_obj.request_id == identities["request_id"]
    assert mcp_obj.run_id == identities["run_id"]
    assert mcp_obj.authority_id == identities["authority_id"]
    assert mcp_obj.verdict == HandoffApiVerdict.ADMITTED.value


def test_preview_is_not_handoff_on_python_mcp_and_mcplusplus() -> None:
    request = _start_request(idempotency_key="idem:parity-preview")
    python_preview = ExternalHandoffAPI().preview(request)
    mcp_preview = execute_external_handoff_operation(
        "preview", request, api=ExternalHandoffAPI()
    )
    mcpp_preview = bind_handoff_operation("preview")
    assert python_preview.verdict == HandoffApiVerdict.PREVIEW_ONLY.value
    assert mcp_preview["ok"] is True
    assert mcp_preview["receipt"]["verdict"] == HandoffApiVerdict.PREVIEW_ONLY.value
    assert (
        ExternalHandoffReceipt.from_dict(mcp_preview["receipt"]).content_id
        == python_preview.content_id
    )
    assert mcpp_preview["operation"] == "preview"
    assert mcpp_preview["preview_is_handoff"] is False
    assert mcpp_preview["live_runtime_invoked"] is False
    assert "preview" not in {CLI_TO_OPERATION[name] for name in CLI_COMMANDS}
