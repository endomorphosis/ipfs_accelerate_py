"""ASI-153: lazy MCP tools for prompt workflow and rescue operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    Operation,
    OperationRequest,
    OperationStatus,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    BackendResponse,
    InMemoryControlStateStore,
    SupervisorControlService,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    AGENT_SUPERVISOR_OPERATION_TOOLS,
    agent_supervisor_discovery_manifest,
    configure_agent_supervisor_control,
    register_native_agent_supervisor_tools,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    native_agent_supervisor_tools as native_tools,
)


PROMPT_OPS = (
    Operation.WORKFLOW_PREVIEW,
    Operation.WORKFLOW_MATERIALIZE,
    Operation.RESTART,
    Operation.RESCUE_PREVIEW,
    Operation.RESCUE,
)


class _RecordingToolManager:
    def __init__(self) -> None:
        self.definitions: list[dict[str, Any]] = []

    def register_tool(self, **definition: Any) -> None:
        self.definitions.append(definition)


@pytest.fixture(autouse=True)
def _reset_mcp_configuration() -> Any:
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


def test_prompt_mcp_tools_are_lazily_named_and_catalog_complete() -> None:
    for operation in PROMPT_OPS:
        tool = AGENT_SUPERVISOR_OPERATION_TOOLS[operation]
        assert tool.__name__ == f"agent_supervisor_{operation.value}"
        assert tool.__agent_supervisor_operation__ is operation

    manager = _RecordingToolManager()
    register_native_agent_supervisor_tools(manager)
    names = {definition["name"] for definition in manager.definitions}
    for operation in PROMPT_OPS:
        assert operation.value in names
        # Public callable name keeps the agent_supervisor_ prefix.
        assert any(
            definition["func"].__name__
            == f"agent_supervisor_{operation.value}"
            for definition in manager.definitions
            if definition["name"] == operation.value
        )

    manifest = agent_supervisor_discovery_manifest()
    assert set(manifest.operations) == set(Operation)


@pytest.mark.asyncio
async def test_mcp_workflow_preview_matches_python_service(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    request = OperationRequest(
        operation=Operation.WORKFLOW_PREVIEW,
        **_binding(repository_root, state_root),
        parameters={
            "directory": str(repository_root),
            "prompt_source": {"kind": "inline", "content_cid": "prompt:one"},
            "output_mode": "markdown",
        },
        dry_run=True,
    )

    def handler(_request: OperationRequest) -> BackendResponse:
        return BackendResponse(
            data={"proposal_root": "plan:mcp"},
            changed=False,
            checks=("schema",),
        )

    service = SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        handlers={Operation.WORKFLOW_PREVIEW: handler},
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 3_000,
    )
    configure_agent_supervisor_control(service=service)
    python_result = service.workflow_preview(request)
    tool = AGENT_SUPERVISOR_OPERATION_TOOLS[Operation.WORKFLOW_PREVIEW]
    mcp_result = await tool(request=request.to_record())
    assert mcp_result["status"] == OperationStatus.SUCCEEDED.value
    assert mcp_result == python_result.to_record()


@pytest.mark.asyncio
async def test_mcp_requires_server_allowlists_and_rejects_path_widening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    monkeypatch.delenv(
        native_tools.AGENT_SUPERVISOR_REPOSITORY_ALLOWLIST_ENV, raising=False
    )
    monkeypatch.delenv(
        native_tools.AGENT_SUPERVISOR_STATE_ALLOWLIST_ENV, raising=False
    )
    configure_agent_supervisor_control()
    request = OperationRequest(
        operation=Operation.WORKFLOW_PREVIEW,
        **_binding(repository_root, state_root),
        parameters={
            "directory": str(repository_root),
            "prompt_source": {"kind": "inline", "content_cid": "prompt:one"},
        },
        dry_run=True,
    )
    tool = AGENT_SUPERVISOR_OPERATION_TOOLS[Operation.WORKFLOW_PREVIEW]
    with pytest.raises(native_tools.AgentSupervisorMCPConfigurationError):
        await tool(request=request.to_record())
