"""LPC-130 API gate: LogicOperationCatalog@1 across Python, CLI, and MCP.

Acceptance:

* Channels agree on names, schemas, status, authority, failure codes, and opt-in.
* Installation is not an ordinary verify operation.
* Supervisor-only mutation controls remain outside datasets logic surfaces.
"""

from __future__ import annotations

from typing import Any

import anyio
import pytest

from ipfs_datasets_py.logic import cli as logic_cli
from ipfs_datasets_py.logic.cli import LOGIC_VERIFICATION_CLI_INTERFACE, create_parser
from ipfs_datasets_py.logic.verification_api import (
    FORMAL_VERIFICATION_MCP_PARITY_INTERFACE,
    GOAL_TACTICIAN_CLI_MCP_INTERFACE,
    GOAL_TACTICIAN_CLI_TO_OPERATION,
    GOAL_TACTICIAN_OPERATIONS,
    GOAL_TACTICIAN_TOOL_TO_OPERATION,
    LOGIC_VERIFICATION_API_INTERFACE,
    LOGIC_VERIFICATION_RESPONSE_SCHEMA,
    STABLE_OPERATIONS,
    get_verification_api,
    invoke_goal_tactician,
    invoke_goal_tactician_cli,
    invoke_goal_tactician_mcp_tool,
    list_goal_tactician_cli_mcp_surface,
    list_stable_features,
)
from ipfs_datasets_py.mcp_server.tools import logic_verification as lv


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

# Closed CLI projection documented by LogicOperationCatalog@1.
CLI_TO_OPERATION: dict[str, str] = {
    "list-features": "list_features",
    "list-families": "list_logic_families",
    "list-providers": "list_providers",
    "provider-capabilities": "provider_capabilities",
    "compile": "compile_verification_artifact",
    "check": "check",
    "monitor": "monitor",
    "portfolio": "run_portfolio",
    "counterexample": "explain_counterexample",
    "verify-receipt": "verify_receipt",
    "advise": "advise",
    "attest-receipt": "attest_receipt",
    "probe-provider": "probe_provider",
    "install-provider": "install_provider",
    "verification-capabilities": "list_features",
}

FORBIDDEN_SUPERVISOR_CONTROLS = frozenset(
    {
        "admit_goal",
        "close_plan",
        "mutate_supervisor",
        "force_complete",
        "lease_steal",
        "rewrite_event_log",
        "bypass_resource_policy",
        "promote_proof_authority",
        "supervisor_mutate",
        "supervisor_only",
    }
)


def _run(coro):
    async def _awaiter():
        return await coro

    return anyio.run(_awaiter)


def _cli(*argv: str) -> dict[str, Any]:
    ns = create_parser().parse_args(list(argv))
    payload = _run(logic_cli._run_async(ns))
    assert isinstance(payload, dict)
    return payload


def _assert_envelope(payload: dict[str, Any], *, operation: str | None = None) -> None:
    missing = ENVELOPE_KEYS - set(payload)
    assert not missing, f"missing envelope keys: {sorted(missing)}"
    assert payload["interface"] == LOGIC_VERIFICATION_API_INTERFACE
    assert payload.get("schema_version") in {
        None,
        LOGIC_VERIFICATION_RESPONSE_SCHEMA,
        "logic-verification-response/v1",
    } or isinstance(payload.get("schema_version"), str)
    assert isinstance(payload["status"], str) and payload["status"]
    assert isinstance(payload["authority"], str)
    assert isinstance(payload["result"], dict)
    if operation is not None:
        assert payload["operation"] == operation


def _tool_for(operation: str) -> str:
    for tool, mapped in lv.TOOL_TO_OPERATION.items():
        if mapped == operation and tool != "verification_capabilities":
            return tool
    raise AssertionError(f"no MCP tool for {operation}")


def _command_for(operation: str) -> str:
    for command, mapped in CLI_TO_OPERATION.items():
        if mapped == operation and command != "verification-capabilities":
            return command
    raise AssertionError(f"no CLI command for {operation}")


# ---------------------------------------------------------------------------
# Catalog agreement
# ---------------------------------------------------------------------------


def test_logic_operation_catalog_channel_name_agreement() -> None:
    assert FORMAL_VERIFICATION_MCP_PARITY_INTERFACE == "FormalVerificationMCPParity@1"
    assert lv.LOGIC_VERIFICATION_MCP_INTERFACE == "LogicVerificationMCP@1"
    assert LOGIC_VERIFICATION_CLI_INTERFACE == "LogicVerificationCLI@1"

    mcp_ops = set(lv.TOOL_TO_OPERATION.values())
    cli_ops = set(CLI_TO_OPERATION.values())
    for operation in STABLE_OPERATIONS:
        assert operation in mcp_ops
        assert operation in cli_ops

    # Schemas agree that every MCP tool projects to a Python operation and
    # returns the shared response envelope.
    for tool, operation in lv.TOOL_TO_OPERATION.items():
        schema = lv.TOOL_SCHEMAS[tool]
        assert schema["python_operation"] == operation
        assert schema["returns"]["envelope"] == LOGIC_VERIFICATION_RESPONSE_SCHEMA
        assert schema["interface"] == lv.LOGIC_VERIFICATION_MCP_INTERFACE

    # Capabilities projection is the channel-neutral catalog document.
    caps = _run(lv.verification_capabilities())
    assert set(caps["operations"]) == set(STABLE_OPERATIONS)
    assert caps["tool_to_operation"] == dict(lv.TOOL_TO_OPERATION)
    assert caps["python_interface"] == LOGIC_VERIFICATION_API_INTERFACE
    assert caps["cli_interface"] == LOGIC_VERIFICATION_CLI_INTERFACE

    cli_caps = _cli("verification-capabilities")
    assert set(cli_caps["operations"]) == set(STABLE_OPERATIONS)
    assert cli_caps["cli_interface"] == LOGIC_VERIFICATION_CLI_INTERFACE
    assert cli_caps["tool_to_operation"] == dict(lv.TOOL_TO_OPERATION)


def test_goal_tactician_catalog_stays_additive_and_closed() -> None:
    surface = list_goal_tactician_cli_mcp_surface()
    assert surface["interface"] == GOAL_TACTICIAN_CLI_MCP_INTERFACE
    assert set(surface["operations"]) == set(GOAL_TACTICIAN_OPERATIONS)
    assert set(surface["tool_to_operation"].values()) == set(GOAL_TACTICIAN_OPERATIONS)
    assert set(surface["cli_to_operation"].values()) == set(GOAL_TACTICIAN_OPERATIONS)
    assert set(surface["forbidden_controls"]) == FORBIDDEN_SUPERVISOR_CONTROLS
    assert set(surface["legacy_operations_preserved"]) == set(STABLE_OPERATIONS)
    assert surface["transport_success_implies_proof_success"] is False

    # Forbidden controls must not collide with any stable or goal-tactician name.
    public = (
        set(STABLE_OPERATIONS)
        | set(GOAL_TACTICIAN_OPERATIONS)
        | set(lv.TOOL_NAMES)
        | set(CLI_TO_OPERATION)
        | set(GOAL_TACTICIAN_TOOL_TO_OPERATION)
        | set(GOAL_TACTICIAN_CLI_TO_OPERATION)
    )
    assert not (FORBIDDEN_SUPERVISOR_CONTROLS & public)


# ---------------------------------------------------------------------------
# Live channel parity on discovery + failure codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operation,python_call,mcp_call,cli_argv",
    [
        (
            "list_logic_families",
            lambda api: api.list_logic_families().to_dict(),
            lambda: lv.verification_list_logic_families(),
            ("list-families",),
        ),
        (
            "list_providers",
            lambda api: api.list_providers().to_dict(),
            lambda: lv.verification_list_providers(),
            ("list-providers",),
        ),
        (
            "provider_capabilities",
            lambda api: api.provider_capabilities().to_dict(),
            lambda: lv.verification_provider_capabilities(),
            ("provider-capabilities",),
        ),
        (
            "list_features",
            lambda api: api.list_features().to_dict(),
            lambda: lv.verification_list_features(),
            ("list-features",),
        ),
    ],
)
def test_discovery_channels_agree_on_status_authority_and_payload(
    operation: str,
    python_call,
    mcp_call,
    cli_argv: tuple[str, ...],
) -> None:
    api = get_verification_api(reset=True)
    py = python_call(api)
    mcp = _run(mcp_call())
    cli = _cli(*cli_argv)

    for payload, channel in ((py, "python"), (mcp, "mcp"), (cli, "cli")):
        _assert_envelope(payload, operation=operation)
        assert payload["status"] == py["status"], channel
        assert payload["authority"] == py["authority"], channel

    if operation == "list_providers":
        py_ids = {item["provider_id"] for item in py["result"]["providers"]}
        assert py_ids == {
            item["provider_id"] for item in mcp["result"]["providers"]
        }
        assert py_ids == {
            item["provider_id"] for item in cli["result"]["providers"]
        }
        assert py["result"]["count"] == mcp["result"]["count"] == cli["result"]["count"]
    elif operation == "list_logic_families":
        assert py["status"] == "declarative"
        assert py["result"]["count"] == mcp["result"]["count"] == cli["result"]["count"]
        py_ids = {item["family_id"] for item in py["result"]["families"]}
        assert py_ids == {item["family_id"] for item in mcp["result"]["families"]}
        assert py_ids == {item["family_id"] for item in cli["result"]["families"]}
    elif operation == "list_features":
        py_ids = {item["feature_id"] for item in py["result"]["features"]}
        mcp_ids = {item["feature_id"] for item in mcp["result"]["features"]}
        cli_ids = {item["feature_id"] for item in cli["result"]["features"]}
        assert set(STABLE_OPERATIONS) <= py_ids
        assert py_ids == mcp_ids == cli_ids
        assert set(py["result"]["operations"]) == set(STABLE_OPERATIONS)
        assert set(mcp["result"]["operations"]) == set(STABLE_OPERATIONS)
        assert set(cli["result"]["operations"]) == set(STABLE_OPERATIONS)

    # MCP channel markers do not alter semantic identity.
    assert mcp["python_operation"] == operation
    assert mcp["mcp_interface"] == lv.LOGIC_VERIFICATION_MCP_INTERFACE


def test_unsupported_provider_failure_code_parity() -> None:
    api = get_verification_api(reset=True)
    py = api.provider_capabilities("not-a-backend").to_dict()
    mcp = _run(lv.verification_provider_capabilities(provider_id="not-a-backend"))
    cli = _cli("provider-capabilities", "--provider-id", "not-a-backend")
    for payload in (py, mcp, cli):
        _assert_envelope(payload, operation="provider_capabilities")
        assert payload["status"] == "unsupported"
        assert payload["authority"] in {"none", "declarative", "advisory", "bounded"}


def test_probe_provider_opt_in_parity() -> None:
    features = {item.feature_id: item for item in list_stable_features()}
    assert features["probe_provider"].requires_opt_in is True

    api = get_verification_api(reset=True)
    py = api.probe_provider("runtime_mtl").to_dict()
    mcp = _run(lv.verification_probe_provider(provider_id="runtime_mtl"))
    cli = _cli("probe-provider", "runtime_mtl")

    for payload in (py, mcp, cli):
        _assert_envelope(payload, operation="probe_provider")
        assert "available" in payload["result"]
        assert payload["status"] == py["status"]
        assert payload["authority"] == py["authority"]


# ---------------------------------------------------------------------------
# Installation is not ordinary verify
# ---------------------------------------------------------------------------


def test_install_provider_opt_in_and_non_verify_parity() -> None:
    features = {item.feature_id: item for item in list_stable_features()}
    assert features["install_provider"].requires_opt_in is True
    assert features["install_provider"].availability.value == "opt_in"

    # Catalog maps agree on the install name across channels.
    assert _tool_for("install_provider") == "verification_install_provider"
    assert _command_for("install_provider") == "install-provider"
    assert CLI_TO_OPERATION["install-provider"] == "install_provider"
    assert lv.TOOL_TO_OPERATION["verification_install_provider"] == "install_provider"

    api = get_verification_api(reset=True)
    py_denied = api.install_provider("z3").to_dict()
    mcp_denied = _run(
        lv.verification_install_provider(provider_id="z3", allow_install=False)
    )
    cli_denied = _cli("install-provider", "z3")

    for payload, channel in (
        (py_denied, "python"),
        (mcp_denied, "mcp"),
        (cli_denied, "cli"),
    ):
        _assert_envelope(payload, operation="install_provider")
        assert payload["status"] == "unsupported", channel
        assert payload["authority"] == "none", channel
        assert payload["result"]["status"] == "authorization_required", channel
        assert payload["result"]["install_attempted"] is False, channel
        assert payload["result"]["mutation_authorized"] is False, channel
        assert "install_without_opt_in" in payload["unsupported_features"], channel
        # Install denial is never ordinary verify success.
        assert payload["status"] != "succeeded", channel
        assert payload["result"].get("installed") is not True, channel

    py_plan = api.install_provider("z3", dry_run=True).to_dict()
    mcp_plan = _run(
        lv.verification_install_provider(
            provider_id="z3", allow_install=False, dry_run=True
        )
    )
    cli_plan = _cli("install-provider", "z3", "--dry-run")

    for payload, channel in ((py_plan, "python"), (mcp_plan, "mcp"), (cli_plan, "cli")):
        _assert_envelope(payload, operation="install_provider")
        assert payload["status"] == "declarative", channel
        assert payload["authority"] == "none", channel
        assert payload["result"]["status"] == "planned", channel
        assert payload["result"]["install_attempted"] is False, channel
        assert payload["result"]["mutation_authorized"] is False, channel

    # Ordinary verify discovery remains a separate operation identity.
    verify_like = api.list_providers().to_dict()
    assert verify_like["operation"] == "list_providers"
    assert verify_like["status"] == "declarative"
    assert verify_like["operation"] != "install_provider"


def test_mcp_install_schema_documents_opt_in_default() -> None:
    schema = lv.TOOL_SCHEMAS["verification_install_provider"]
    assert "Opt-in" in schema["description"] or "opt-in" in schema["description"].lower()
    assert schema["parameters"]["properties"]["allow_install"]["default"] is False
    assert "allow_install" in schema["parameters"]["properties"]


# ---------------------------------------------------------------------------
# Goal-tactician channel identity + supervisor control refusal
# ---------------------------------------------------------------------------


def test_goal_tactician_list_operations_channel_parity() -> None:
    py = invoke_goal_tactician("list_goal_tactician_operations").to_dict()
    mcp = invoke_goal_tactician_mcp_tool("goal_tactician_list_operations")
    cli = invoke_goal_tactician_cli("goal-list-operations")
    for payload in (py, mcp, cli):
        _assert_envelope(payload, operation="list_goal_tactician_operations")
        assert payload["status"] == "declarative"
        assert set(payload["result"]["operations"]) == set(GOAL_TACTICIAN_OPERATIONS)
        assert payload["result"]["transport_success_implies_proof_success"] is False


def test_goal_tactician_supervisor_control_refusal_parity() -> None:
    request = {
        "plan_id": "plan:forbidden-parity",
        "steps": [{"step_id": "s", "obligation_id": "o", "statement": "x"}],
        "controls": {"mutate_supervisor": True},
    }
    py = invoke_goal_tactician("execute_proof_plan", request).to_dict()
    mcp = invoke_goal_tactician_mcp_tool("goal_tactician_execute_proof_plan", request)
    cli = invoke_goal_tactician_cli("goal-execute-plan", request)
    for payload in (py, mcp, cli):
        assert payload["status"] == "invalid"
        assert "supervisor_only_control" in payload["unsupported_features"]
        assert payload["status"] != "succeeded"
