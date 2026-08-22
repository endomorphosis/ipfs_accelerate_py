"""EAAEF-112: canonical MCP handoff tools over ExternalHandoffAPI."""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.api.external_handoff import ExternalHandoffAPI
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools.external_handoff import (
    HANDOFF_TOOL_NAMES,
    configure_external_handoff_api,
    execute_external_handoff_operation,
    external_handoff_discovery_manifest,
    register_external_handoff_tools,
)


OPERATOR = "principal:operator"
WORKER = "principal:worker"
REVIEWER = "principal:reviewer"


class _FakeManager:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def register_tool(self, **definition: Any) -> None:
        self.tools[str(definition["name"])] = definition


@pytest.fixture(autouse=True)
def _reset_api() -> None:
    configure_external_handoff_api(None)
    yield
    configure_external_handoff_api(None)


def test_cold_registration_lists_handoff_tools() -> None:
    manager = _FakeManager()
    register_external_handoff_tools(manager)
    for name in HANDOFF_TOOL_NAMES:
        assert name in manager.tools
        assert manager.tools[name]["category"] == "agent_supervisor"
    manifest = external_handoff_discovery_manifest()
    assert manifest["cold_registration"] is True
    assert manifest["preview_is_handoff"] is False
    assert manifest["self_approval"] is False


def test_preview_does_not_admit_mutation() -> None:
    api = ExternalHandoffAPI()
    configure_external_handoff_api(api)
    preview = execute_external_handoff_operation(
        "preview",
        {
            "principal_id": OPERATOR,
            "worker_principal_id": WORKER,
            "session_id": "session:mcp",
            "repository_id": "repo:mcp",
            "objective_id": "objective:mcp",
            "idempotency_key": "idem:preview",
        },
        api=api,
    )
    assert preview["ok"] is True
    assert preview["result"]["reason_code"] == "preview_only"
    started = execute_external_handoff_operation(
        "handoff",
        {
            "principal_id": OPERATOR,
            "worker_principal_id": WORKER,
            "session_id": "session:mcp",
            "repository_id": "repo:mcp",
            "objective_id": "objective:mcp",
            "idempotency_key": "idem:handoff",
        },
        api=api,
    )
    assert started["ok"] is True
    assert started["result"]["reason_code"] == "admitted"


def test_worker_cannot_self_approve() -> None:
    api = ExternalHandoffAPI()
    configure_external_handoff_api(api)
    started = execute_external_handoff_operation(
        "handoff",
        {
            "principal_id": OPERATOR,
            "worker_principal_id": WORKER,
            "session_id": "session:mcp",
            "repository_id": "repo:mcp",
            "objective_id": "objective:mcp",
            "idempotency_key": "idem:approve",
        },
        api=api,
    )
    denied = execute_external_handoff_operation(
        "approve",
        {
            "principal_id": WORKER,
            "worker_principal_id": WORKER,
            "reviewer_principal_id": WORKER,
            "run_id": started["result"]["run_id"],
            "authority_id": started["result"]["authority_id"],
        },
        api=api,
    )
    assert denied["ok"] is False
    approved = execute_external_handoff_operation(
        "approve",
        {
            "principal_id": REVIEWER,
            "worker_principal_id": WORKER,
            "reviewer_principal_id": REVIEWER,
            "run_id": started["result"]["run_id"],
            "authority_id": started["result"]["authority_id"],
        },
        api=api,
    )
    assert approved["ok"] is True
    assert approved["result"]["reason_code"] == "approved"


def test_unknown_operation_fails_closed() -> None:
    result = execute_external_handoff_operation("merge_everything", {})
    assert result["ok"] is False
    assert result["error_code"] == "unknown_operation"
