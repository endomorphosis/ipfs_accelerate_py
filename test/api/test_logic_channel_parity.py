"""LPC-130 / LogicOperationCatalog@1 — accelerate-side live channel parity.

Validates that the datasets logic verification catalog projects consistently
through Python, CLI, and MCP when imported from the accelerate test surface.
Complements ``ipfs_datasets_py/tests/unit/logic/test_channel_parity.py`` and
the existing FormalVerificationMCPParity@1 / GoalTacticianCLIMCP@1 suites.
"""

from __future__ import annotations

import importlib
from typing import Any

import anyio
import pytest

from ipfs_datasets_py.logic.verification_api import (
    FORMAL_VERIFICATION_MCP_PARITY_INTERFACE,
    GOAL_TACTICIAN_CLI_MCP_INTERFACE,
    GOAL_TACTICIAN_OPERATIONS,
    GOAL_TACTICIAN_TOOL_TO_OPERATION,
    GOAL_TACTICIAN_CLI_TO_OPERATION,
    LOGIC_VERIFICATION_API_INTERFACE,
    STABLE_OPERATIONS,
    VerificationAuthority,
    VerificationStatus,
    get_verification_api,
    list_goal_tactician_cli_mcp_surface,
)


ENVELOPE_KEYS = frozenset(
    {
        "status",
        "authority",
        "operation",
        "result",
        "assumptions",
        "bounds",
        "translations",
        "witnesses",
        "unsupported_features",
        "diagnostics",
        "cache",
        "interface",
    }
)


def _run(coro: Any) -> Any:
    return anyio.run(lambda: coro)


def _load_datasets_mcp():
    return importlib.import_module(
        "ipfs_datasets_py.mcp_server.tools.logic_verification"
    )


def _assert_envelope(payload: dict[str, Any], *, operation: str | None = None) -> None:
    missing = ENVELOPE_KEYS - set(payload)
    assert not missing, f"missing envelope keys: {sorted(missing)}"
    assert payload["interface"] == LOGIC_VERIFICATION_API_INTERFACE
    assert isinstance(payload["status"], str) and payload["status"]
    assert isinstance(payload["authority"], str) and payload["authority"]
    assert isinstance(payload["result"], dict)
    if operation is not None:
        assert payload["operation"] == operation


def test_logic_operation_catalog_interface_markers() -> None:
    """Catalog identity markers remain stable for LPC-130 consumers."""

    assert FORMAL_VERIFICATION_MCP_PARITY_INTERFACE == "FormalVerificationMCPParity@1"
    assert LOGIC_VERIFICATION_API_INTERFACE == "LogicVerificationAPI@1"
    surface = list_goal_tactician_cli_mcp_surface()
    assert surface["interface"] == GOAL_TACTICIAN_CLI_MCP_INTERFACE
    assert surface["python_interface"] == "GoalTacticianAPI@1"


def test_stable_catalog_python_mcp_cli_name_agreement() -> None:
    lv = _load_datasets_mcp()
    from ipfs_datasets_py.logic.cli import create_parser
    import argparse

    mapped = set(lv.TOOL_TO_OPERATION.values())
    for operation in STABLE_OPERATIONS:
        assert operation in mapped, f"MCP lost stable op {operation}"

    # Every STABLE_OPERATIONS name has at least one MCP tool inverse.
    inverse: dict[str, list[str]] = {}
    for tool, operation in lv.TOOL_TO_OPERATION.items():
        inverse.setdefault(operation, []).append(tool)
    for operation in STABLE_OPERATIONS:
        assert inverse[operation], f"no MCP tool for {operation}"

    parser = create_parser()
    actions = [
        action
        for action in parser._actions  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction)  # type: ignore[attr-defined]
    ]
    choices = set(actions[0].choices or {})
    expected_cli = {
        "list-families",
        "list-providers",
        "provider-capabilities",
        "compile",
        "check",
        "monitor",
        "portfolio",
        "counterexample",
        "verify-receipt",
        "attest-receipt",
        "advise",
        "probe-provider",
        "install-provider",
        "list-features",
        "verification-capabilities",
    }
    assert expected_cli <= choices


def test_goal_tactician_catalog_is_additive_and_closed() -> None:
    surface = list_goal_tactician_cli_mcp_surface()
    assert set(surface["operations"]) == set(GOAL_TACTICIAN_OPERATIONS)
    assert set(GOAL_TACTICIAN_TOOL_TO_OPERATION.values()) == set(GOAL_TACTICIAN_OPERATIONS)
    assert set(GOAL_TACTICIAN_CLI_TO_OPERATION.values()) == set(GOAL_TACTICIAN_OPERATIONS)
    assert set(surface["legacy_operations_preserved"]) == set(STABLE_OPERATIONS)
    assert surface["transport_success_implies_proof_success"] is False

    lv = _load_datasets_mcp()
    mapped = set(lv.TOOL_TO_OPERATION.values())
    for operation in STABLE_OPERATIONS:
        assert operation in mapped
    # Goal-tactician ops stay off the legacy LogicVerificationMCP@1 identity.
    for operation in GOAL_TACTICIAN_OPERATIONS:
        if operation == "list_goal_tactician_operations":
            continue
        assert operation not in mapped


def test_python_mcp_list_features_and_capabilities_parity() -> None:
    api = get_verification_api(reset=True)
    py = api.list_features().to_dict()
    _assert_envelope(py, operation="list_features")
    assert set(py["result"]["operations"]) >= set(STABLE_OPERATIONS)

    lv = _load_datasets_mcp()
    mcp_features = _run(lv.verification_list_features())
    mcp_caps = _run(lv.verification_capabilities())
    _assert_envelope(mcp_features, operation="list_features")
    assert set(mcp_features["result"]["operations"]) >= set(STABLE_OPERATIONS)
    assert set(mcp_caps["operations"]) == set(STABLE_OPERATIONS)
    assert mcp_caps["status"] == "declarative"
    assert mcp_caps["authority"] == "declarative"
    assert mcp_caps["bounds"]["max_json_bytes"] == 256_000


def test_install_is_opt_in_mutation_not_ordinary_verify() -> None:
    api = get_verification_api(reset=True)
    denied = api.install_provider("z3").to_dict()
    _assert_envelope(denied, operation="install_provider")
    assert denied["status"] == VerificationStatus.UNSUPPORTED.value
    assert "install_without_opt_in" in denied["unsupported_features"]
    assert denied["result"].get("install_attempted") is False

    lv = _load_datasets_mcp()
    mcp = _run(lv.verification_install_provider(provider_id="z3"))
    _assert_envelope(mcp, operation="install_provider")
    assert mcp["status"] == "unsupported"
    assert mcp["result"].get("install_attempted") is False

    dry = api.install_provider("z3", dry_run=True).to_dict()
    assert dry["status"] == VerificationStatus.DECLARATIVE.value
    assert dry["authority"] in {
        VerificationAuthority.NONE.value,
        VerificationAuthority.DECLARATIVE.value,
    }
    assert dry["result"]["mutation_authorized"] is False


def test_probe_is_explicit_opt_in_surface() -> None:
    """probe_provider is a named opt-in op; discovery never substitutes for it."""

    api = get_verification_api(reset=True)
    discovery = api.list_providers().to_dict()
    assert discovery["status"] == "declarative"
    # list_providers must not report a live probe result field as success proof.
    for provider in discovery["result"]["providers"]:
        assert "probe_status" not in provider or provider.get("probed") is not True

    lv = _load_datasets_mcp()
    probe = api.probe_provider("runtime_mtl").to_dict()
    mcp_probe = _run(lv.verification_probe_provider(provider_id="runtime_mtl"))
    _assert_envelope(probe, operation="probe_provider")
    _assert_envelope(mcp_probe, operation="probe_provider")
    assert probe["status"] == mcp_probe["status"]
    assert "available" in probe["result"]
    assert "available" in mcp_probe["result"]


def test_supervisor_only_controls_absent_from_public_catalogs() -> None:
    forbidden = set(list_goal_tactician_cli_mcp_surface()["forbidden_controls"])
    assert forbidden
    lv = _load_datasets_mcp()
    public_names = (
        set(STABLE_OPERATIONS)
        | set(GOAL_TACTICIAN_OPERATIONS)
        | set(lv.TOOL_TO_OPERATION)
        | set(lv.TOOL_TO_OPERATION.values())
        | set(GOAL_TACTICIAN_TOOL_TO_OPERATION)
        | set(GOAL_TACTICIAN_CLI_TO_OPERATION)
    )
    assert forbidden.isdisjoint(public_names)


def test_status_authority_vocabularies_shared() -> None:
    statuses = {member.value for member in VerificationStatus}
    authorities = {member.value for member in VerificationAuthority}
    assert "succeeded" in statuses and "declarative" in statuses
    assert "unsupported" in statuses and "unavailable" in statuses
    assert "theorem" in authorities and "declarative" in authorities
    assert "none" in authorities


def test_transport_success_does_not_imply_proof_for_goal_list() -> None:
    from ipfs_datasets_py.logic.verification_api import (
        invoke_goal_tactician,
        invoke_goal_tactician_cli,
        invoke_goal_tactician_mcp_tool,
    )

    py = invoke_goal_tactician("list_goal_tactician_operations").to_dict()
    mcp = invoke_goal_tactician_mcp_tool("goal_tactician_list_operations")
    cli = invoke_goal_tactician_cli("goal-list-operations")
    for payload in (py, mcp, cli):
        _assert_envelope(payload, operation="list_goal_tactician_operations")
        assert payload["status"] == "declarative"
        assert payload["result"].get("transport_success_implies_proof_success", False) is False
        assert payload["result"].get("proof_success", False) is False
