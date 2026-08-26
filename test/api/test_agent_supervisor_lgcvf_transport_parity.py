"""LGCVF-102: one typed semantic service across Python, CLI, and MCP.

Mutation defaults to preview. Wrappers perform no independent semantics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.integrations.ipfs_datasets_logic_provider import (
    SEMANTIC_SERVICE_INTERFACE,
    SEMANTIC_SERVICE_MUTATING_OPERATIONS,
    SEMANTIC_SERVICE_MUTATION_POLICY,
    SEMANTIC_SERVICE_OPERATIONS,
    SEMANTIC_SERVICE_TRANSPORTS,
    LgcvfSemanticService,
    SemanticServiceError,
    SemanticServiceRequest,
    get_lgcvf_semantic_service,
    invoke_semantic_service_cli,
    invoke_semantic_service_mcp,
    invoke_semantic_service_python,
)


_TRANSPORT_INVOKERS = {
    "python": invoke_semantic_service_python,
    "cli": invoke_semantic_service_cli,
    "mcp": invoke_semantic_service_mcp,
}


def _without_transport(envelope: dict[str, Any]) -> dict[str, Any]:
    payload = dict(envelope)
    payload.pop("transport", None)
    return payload


def test_all_declared_operations_share_one_service() -> None:
    service = LgcvfSemanticService()
    assert len(SEMANTIC_SERVICE_OPERATIONS) == 16
    assert SEMANTIC_SERVICE_OPERATIONS == (
        "capability",
        "snapshot",
        "impact",
        "contracts",
        "abstract",
        "discharge",
        "verify",
        "prove",
        "counterexample",
        "interpolate",
        "synthesize",
        "repair",
        "context",
        "benchmark",
        "explain",
        "replay",
    )
    for operation in SEMANTIC_SERVICE_OPERATIONS:
        receipt = service.invoke(
            SemanticServiceRequest(operation=operation, payload={"id": operation})
        )
        assert receipt.operation == operation
        assert receipt.wrote is False
        assert receipt.preview is True
        assert receipt.interface == SEMANTIC_SERVICE_INTERFACE
        assert receipt.result["shared_service"] == SEMANTIC_SERVICE_INTERFACE
        assert receipt.result["mutation_policy"] == SEMANTIC_SERVICE_MUTATION_POLICY
        assert receipt.result["mcp_plus_plus_profile"] is False
        assert tuple(receipt.result["transports"]) == SEMANTIC_SERVICE_TRANSPORTS


def test_python_cli_mcp_parity() -> None:
    payload = {"root": "tree:fixture"}
    python = invoke_semantic_service_python("snapshot", payload)
    cli = invoke_semantic_service_cli("snapshot", payload)
    mcp = invoke_semantic_service_mcp("snapshot", payload)
    assert python["transport"] == "python"
    assert cli["transport"] == "cli"
    assert mcp["transport"] == "mcp"
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
    assert forced["result"]["requested_preview"] is False
    assert forced["result"]["mutating"] is True
    assert forced["status"] == "preview"


def test_all_operations_have_python_cli_mcp_parity() -> None:
    service = get_lgcvf_semantic_service()
    for operation in SEMANTIC_SERVICE_OPERATIONS:
        payload = {"id": operation, "root": "tree:fixture"}
        envelopes = {
            transport: invoker(operation, payload, service=service)
            for transport, invoker in _TRANSPORT_INVOKERS.items()
        }
        semantic = [_without_transport(item) for item in envelopes.values()]
        assert semantic[0] == semantic[1] == semantic[2]
        for transport, envelope in envelopes.items():
            assert envelope["transport"] == transport
            assert envelope["interface"] == SEMANTIC_SERVICE_INTERFACE
            assert envelope["wrote"] is False
            assert set(envelope) == {
                "interface",
                "operation",
                "preview",
                "result",
                "schema",
                "status",
                "transport",
                "wrote",
            }


def test_wrappers_share_one_typed_service() -> None:
    first = get_lgcvf_semantic_service()
    second = get_lgcvf_semantic_service()
    assert first is second
    payload = {"root": "tree:shared"}
    python = invoke_semantic_service_python("impact", payload, service=first)
    cli = invoke_semantic_service_cli("impact", payload, service=first)
    mcp = invoke_semantic_service_mcp("impact", payload, service=first)
    assert _without_transport(python) == _without_transport(cli) == _without_transport(mcp)
    assert python["result"]["payload_digest"] == cli["result"]["payload_digest"]
    assert python["result"]["shared_service"] == SEMANTIC_SERVICE_INTERFACE


def test_capability_lists_closed_catalog_without_mcp_plus_plus_profile() -> None:
    receipt = invoke_semantic_service_mcp("capability", {})
    result = receipt["result"]
    assert tuple(result["operations"]) == SEMANTIC_SERVICE_OPERATIONS
    assert set(result["mutating_operations"]) == set(SEMANTIC_SERVICE_MUTATING_OPERATIONS)
    assert result["mcp_plus_plus_profile"] is False
    assert result["mutation_policy"] == "preview"
    serialized = json.dumps(receipt, sort_keys=True)
    assert "mcp++" not in serialized.lower()
    assert "mcplusplus" not in serialized.lower()


def test_unknown_operation_is_rejected() -> None:
    with pytest.raises(SemanticServiceError, match="unsupported semantic operation"):
        SemanticServiceRequest(operation="invent", payload={})
    with pytest.raises(SemanticServiceError, match="unsupported semantic operation"):
        invoke_semantic_service_python("invent", {})


def test_preview_mutation_does_not_write_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    for operation in sorted(SEMANTIC_SERVICE_MUTATING_OPERATIONS):
        for transport, invoker in _TRANSPORT_INVOKERS.items():
            envelope = invoker(
                operation,
                {"path": "pkg/mod.py", "operator": "add_argument"},
                preview=False,
            )
            assert envelope["transport"] == transport
            assert envelope["wrote"] is False
            assert envelope["preview"] is True
            assert envelope["result"]["wrote"] is False
    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert after == before
