"""ASI-153: Python/CLI/MCP parity for prompt workflow control operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py import cli
from ipfs_accelerate_py.agent_supervisor.control_cli import COMMAND_OPERATIONS
from ipfs_accelerate_py.agent_supervisor.control_contracts import (
    Operation,
    OperationRequest,
    OperationStatus,
)
from ipfs_accelerate_py.agent_supervisor.control_plane import (
    BackendResponse,
    InMemoryControlStateStore,
    SupervisorControlService,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    AGENT_SUPERVISOR_OPERATION_TOOLS,
    configure_agent_supervisor_control,
)


@pytest.fixture(autouse=True)
def _reset_mcp() -> Any:
    configure_agent_supervisor_control()
    yield
    configure_agent_supervisor_control()


def _binding(repository_root: Path, state_root: Path) -> dict[str, Any]:
    return {
        "repository_root": str(repository_root),
        "state_root": str(state_root),
        "repository_id": "repository:prompt",
        "tree_id": "tree:current",
        "objective_id": "ASI-153",
        "objective_revision": "objective:1",
        "policy_id": "policy:prompt-control",
        "policy_revision": "policy:1",
        "caller": "operator:alice",
    }


def _cli_command(operation: Operation) -> str:
    return next(
        command
        for command, candidate in COMMAND_OPERATIONS.items()
        if candidate is operation
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        Operation.WORKFLOW_PREVIEW,
        Operation.RESCUE_PREVIEW,
        Operation.RESTART,
    ],
)
async def test_python_cli_mcp_records_are_identical(
    operation: Operation,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    if operation is Operation.WORKFLOW_PREVIEW:
        parameters = {
            "directory": str(repository_root),
            "prompt_source": {"kind": "inline", "content_cid": "prompt:one"},
            "output_mode": "both",
        }
    elif operation is Operation.RESCUE_PREVIEW:
        parameters = {
            "incident_cid": "incident:one",
            "incident_root": "incident-root:one",
            "incident_repository_id": "repository:prompt",
            "incident_tree_id": "tree:current",
            "incident_objective_id": "ASI-153",
            "incident_objective_revision": "objective:1",
            "incident_policy_id": "policy:prompt-control",
            "incident_policy_revision": "policy:1",
        }
    else:
        parameters = {
            "target_id": "supervisor:prompt",
            "run_id": "run:old",
            "configuration_root": "configuration:1",
            "expected_revision": 1,
            "deadline_ms": 30_000,
            "health_window_ms": 5_000,
            "reason": "parity restart",
        }
    request = OperationRequest(
        operation=operation,
        **_binding(repository_root, state_root),
        parameters=parameters,
        dry_run=True,
    )

    def handler(_request: OperationRequest) -> BackendResponse:
        return BackendResponse(
            data={"operation": operation.value, "ok": True},
            changed=False,
            checks=("schema",),
        )

    service = SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        handlers={operation: handler},
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 4_000,
    )
    python_result = service.execute(request)
    assert python_result.status is OperationStatus.SUCCEEDED

    exit_status = cli.main(
        [
            "agent",
            _cli_command(operation),
            "--request-json",
            request.to_json(),
            "--output-json",
        ],
        agent_control_service=service,
    )
    captured = capsys.readouterr()
    assert exit_status == 0
    cli_record = json.loads(captured.out)

    configure_agent_supervisor_control(service=service)
    mcp_record = await AGENT_SUPERVISOR_OPERATION_TOOLS[operation](
        request=request.to_record()
    )

    assert cli_record == python_result.to_record()
    assert mcp_record == python_result.to_record()
