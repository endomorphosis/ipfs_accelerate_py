"""LGCVF-102: one typed semantic service across Python, CLI, and MCP.

Mutation defaults to preview. Wrappers perform no independent semantics.
"""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider import (
    SEMANTIC_SERVICE_OPERATIONS,
    LgcvfSemanticService,
    SemanticServiceRequest,
    invoke_semantic_service_cli,
    invoke_semantic_service_mcp,
    invoke_semantic_service_python,
)


def test_all_declared_operations_share_one_service() -> None:
    service = LgcvfSemanticService()
    assert len(SEMANTIC_SERVICE_OPERATIONS) == 16
    for operation in SEMANTIC_SERVICE_OPERATIONS:
        receipt = service.invoke(
            SemanticServiceRequest(operation=operation, payload={"id": operation})
        )
        assert receipt.operation == operation
        assert receipt.wrote is False
        assert receipt.preview is True


def test_python_cli_mcp_parity() -> None:
    payload = {"root": "tree:fixture"}
    python = invoke_semantic_service_python("snapshot", payload)
    cli = invoke_semantic_service_cli("snapshot", payload)
    mcp = invoke_semantic_service_mcp("snapshot", payload)
    for item in (python, cli, mcp):
        item.pop("transport")
    assert python == cli == mcp


def test_mutation_defaults_to_preview_and_does_not_write() -> None:
    applied_preview = invoke_semantic_service_python(
        "repair", {"path": "pkg/mod.py"}, preview=True
    )
    defaulted = invoke_semantic_service_python("repair", {"path": "pkg/mod.py"})
    forced = invoke_semantic_service_cli(
        "synthesize", {"operator": "add_argument"}, preview=False
    )
    assert applied_preview["wrote"] is False
    assert defaulted["wrote"] is False
    assert defaulted["preview"] is True
    # Even an explicit apply request remains preview: mutation defaults to preview.
    assert forced["wrote"] is False
    assert forced["preview"] is True
