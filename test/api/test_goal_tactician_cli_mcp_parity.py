"""CLI / MCP / Python parity for GoalTacticianCLIMCP@1 (FVT-G050 / FVT-029).

Acceptance:

* Python ``GoalTacticianAPI@1``, datasets MCP tools, and CLI commands share
  closed operation names, schemas, envelopes, identities, status, authority,
  diagnostics, redaction, bounds, cancellation, and availability;
* legacy ``LogicVerificationMCP@1`` / ``STABLE_OPERATIONS`` remain compatible;
* transport success never implies proof success.
"""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_datasets_py.logic.verification_api import (
    GOAL_TACTICIAN_API_INTERFACE,
    GOAL_TACTICIAN_CLI_COMMANDS,
    GOAL_TACTICIAN_CLI_MCP_INTERFACE,
    GOAL_TACTICIAN_CLI_TO_OPERATION,
    GOAL_TACTICIAN_OPERATIONS,
    GOAL_TACTICIAN_RESPONSE_SCHEMA,
    GOAL_TACTICIAN_TOOL_NAMES,
    GOAL_TACTICIAN_TOOL_TO_OPERATION,
    LOGIC_VERIFICATION_API_INTERFACE,
    STABLE_OPERATIONS,
    get_verification_api,
    goal_tactician_tool_schemas,
    invoke_goal_tactician,
    invoke_goal_tactician_cli,
    invoke_goal_tactician_mcp_tool,
    list_goal_tactician_cli_mcp_surface,
)


ENVELOPE_KEYS = {
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


def _assert_envelope(payload: dict[str, Any], *, operation: str | None = None) -> None:
    missing = ENVELOPE_KEYS - set(payload)
    assert not missing, f"missing envelope keys: {sorted(missing)}"
    assert payload["interface"] == LOGIC_VERIFICATION_API_INTERFACE
    assert payload.get("schema_version") in {
        None,
        GOAL_TACTICIAN_RESPONSE_SCHEMA,
        "logic-verification-response/v1",
    } or isinstance(payload.get("schema_version"), str)
    assert isinstance(payload["status"], str) and payload["status"]
    assert isinstance(payload["authority"], str)
    assert isinstance(payload["result"], dict)
    assert isinstance(payload["assumptions"], list)
    assert isinstance(payload["bounds"], dict)
    assert isinstance(payload["translations"], list)
    assert isinstance(payload["witnesses"], list)
    assert isinstance(payload["unsupported_features"], list)
    assert isinstance(payload["diagnostics"], list)
    assert isinstance(payload["cache"], dict)
    if operation is not None:
        assert payload["operation"] == operation


def _formalize_request() -> dict[str, Any]:
    return {
        "caller_text": "\n".join(
            [
                "PROPERTY existential_reachability",
                "QUANTIFIER exists",
                "QUANTIFIER eventually",
                "ACTOR scheduler",
                "STATE phase",
                "CURRENT phase=init",
                "TARGET phase=ready",
                "TRANSITION claim",
                "ASSUME must_prove: tokens are totally ordered",
                "BOUND wall_time_ms=5000",
                "BOUND max_steps=32",
                "ASSURANCE bounded",
                "LOGIC temporal.ltl",
                "PROVIDER provider:z3",
                "ACCEPT receipt:kernel",
                "RECEIPT proof-receipt",
            ]
        ),
        "source": {
            "tree_id": "tree:repo@abc",
            "source_ref_ids": ["source:prompt", "source:lease.py"],
            "span_ids": ["span:caller"],
            "ast_scope_ids": ["symbol:claim_lease"],
            "snapshot_id": "snap:1",
        },
        "goal_id": "goal:lease-ready",
        "root_goal_id": "goal:lease-ready",
        "known_identifiers": ["scheduler", "phase", "claim", "init", "ready"],
        "repository_source_ref_ids": ["source:lease.py"],
        "prefer_controlled_language": True,
        "max_candidates": 8,
        "logic_family": "temporal.ltl",
        "provider_ids": ["provider:z3"],
        "bounds": {"wall_time_ms": 1000, "max_steps": 8, "network_allowed": False},
    }


def _witness() -> dict[str, Any]:
    return {
        "kind": "model",
        "assignments": {"x": 1, "y": 0},
        "tool_id": "z3",
        "property_id": "property:lease-safety",
        "tree_id": "tree:repo@abc",
        "assumption_ids": ["assume:tokens-ordered"],
        "bounds": {"max_steps": 4},
    }


def _channel_request(operation: str) -> dict[str, Any]:
    if operation == "list_goal_tactician_operations":
        return {}
    if operation == "formalize_goal":
        return _formalize_request()
    if operation == "compare_interpretations":
        return {"source": "the system reaches ready", "goal_id": "goal:compare"}
    if operation == "discover_missing_proofs":
        from ipfs_datasets_py.logic.software_verification.tactician.contracts import (
            SourceSpanBinding,
        )
        from ipfs_datasets_py.logic.software_verification.tactician.proof_holes import (
            CompilationSurface,
            loop_site,
        )

        surface = CompilationSurface(
            surface_id="surface:parity",
            formal_goal_id="formal:parity",
            tree_id="tree:repo@abc",
            sites=(
                loop_site(
                    "site:loop",
                    source=SourceSpanBinding(
                        tree_id="tree:repo@abc",
                        source_ref_ids=("source:lease.py",),
                        span_ids=("span:loop",),
                        ast_scope_ids=("symbol:loop",),
                        snapshot_id="snap:1",
                    ),
                    has_invariant=False,
                    has_variant=False,
                    require_variant=False,
                ),
            ),
        )
        return surface.to_dict()
    if operation == "plan_proof":
        from ipfs_datasets_py.logic.software_verification.tactician.contracts import (
            AuthorityCeiling,
        )
        from ipfs_datasets_py.logic.software_verification.tactician.proof_plan import (
            build_missing_proof_plan,
            complete_step,
        )

        plan = build_missing_proof_plan(
            "plan:parity",
            formal_goal_id="formal:parity",
            graph_id="graph:parity",
            tree_id="tree:repo@abc",
            steps=(
                complete_step(
                    "step:parity:0",
                    "obligation:lease-safety",
                    authority=AuthorityCeiling.BOUNDED,
                    provider_ids=("provider:z3",),
                ),
            ),
            required_obligation_ids=("obligation:lease-safety",),
        )
        return {
            "alternatives": [plan.to_dict()],
            "policy": {
                "minimum_authority": "bounded",
                "available_resource_classes": ["solver", "kernel", "artifact_store"],
                "satisfied_dependencies": ["root:goal"],
                "required_obligation_ids": ["obligation:lease-safety"],
            },
        }
    if operation == "validate_proof_candidate":
        return {"candidate": {"step_id": "s"}}
    if operation == "execute_proof_plan":
        return {
            "plan_id": "plan:parity",
            "steps": [
                {
                    "step_id": "step:1",
                    "obligation_id": "obligation:lease-safety",
                    "statement": "safe",
                }
            ],
        }
    if operation == "proof_status":
        return {
            "plan_id": "plan:parity",
            "status": "complete",
            "steps": [{"step_id": "s1"}],
            "receipts": [],
        }
    if operation in {
        "minimize_counterexample",
        "explain_counterexample_causal",
        "replay_counterexample",
    }:
        return {"witness": _witness(), "tool_available": True, "family": "smt_model"}
    raise AssertionError(f"no fixture for {operation}")


# ---------------------------------------------------------------------------
# Schema / registration parity
# ---------------------------------------------------------------------------


def test_goal_tactician_cli_mcp_surface_is_closed_and_complete() -> None:
    surface = list_goal_tactician_cli_mcp_surface()
    assert surface["interface"] == GOAL_TACTICIAN_CLI_MCP_INTERFACE
    assert surface["python_interface"] == GOAL_TACTICIAN_API_INTERFACE
    assert set(surface["operations"]) == set(GOAL_TACTICIAN_OPERATIONS)
    assert set(surface["tools"]) == set(GOAL_TACTICIAN_TOOL_NAMES)
    assert set(surface["cli_commands"]) == set(GOAL_TACTICIAN_CLI_COMMANDS)
    assert set(GOAL_TACTICIAN_TOOL_TO_OPERATION.values()) == set(GOAL_TACTICIAN_OPERATIONS)
    assert set(GOAL_TACTICIAN_CLI_TO_OPERATION.values()) == set(GOAL_TACTICIAN_OPERATIONS)
    assert surface["transport_success_implies_proof_success"] is False
    assert set(surface["legacy_operations_preserved"]) == set(STABLE_OPERATIONS)

    schemas = goal_tactician_tool_schemas()
    assert set(schemas) == set(GOAL_TACTICIAN_TOOL_NAMES)
    for tool_name, schema in schemas.items():
        assert schema["interface"] == GOAL_TACTICIAN_CLI_MCP_INTERFACE
        assert schema["python_interface"] == GOAL_TACTICIAN_API_INTERFACE
        assert schema["python_operation"] == GOAL_TACTICIAN_TOOL_TO_OPERATION[tool_name]
        assert schema["returns"]["envelope"] == GOAL_TACTICIAN_RESPONSE_SCHEMA
        assert schema["returns"]["interface"] == LOGIC_VERIFICATION_API_INTERFACE
        assert schema["bounds"]["supervisor_mutation"] is False
        assert schema["bounds"]["cancellation"] is True
        assert schema["bounds"]["redaction"] == "public"


def test_legacy_logic_verification_mcp_still_covers_stable_operations() -> None:
    """Goal tactician wiring is additive and must not break LogicVerificationMCP@1."""

    from ipfs_datasets_py.mcp_server.tools import logic_verification as lv

    assert lv.LOGIC_VERIFICATION_MCP_INTERFACE == "LogicVerificationMCP@1"
    mapped = set(lv.TOOL_TO_OPERATION.values())
    for operation in STABLE_OPERATIONS:
        assert operation in mapped, f"legacy MCP lost mapping for {operation}"
    # Goal tactician ops intentionally live on GoalTacticianCLIMCP@1, not the
    # closed LFV-G071 tool set.
    for operation in GOAL_TACTICIAN_OPERATIONS:
        if operation != "list_goal_tactician_operations":
            assert operation not in mapped or operation in STABLE_OPERATIONS


# ---------------------------------------------------------------------------
# Channel parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operation", list(GOAL_TACTICIAN_OPERATIONS))
def test_python_mcp_cli_share_envelope_for_each_operation(operation: str) -> None:
    get_verification_api(reset=True)
    request = _channel_request(operation)

    python = invoke_goal_tactician(operation, request, request_id=f"req:py:{operation}")
    py = python.to_dict()
    _assert_envelope(py, operation=operation)

    tool_name = next(
        name for name, op in GOAL_TACTICIAN_TOOL_TO_OPERATION.items() if op == operation
    )
    command = next(
        name for name, op in GOAL_TACTICIAN_CLI_TO_OPERATION.items() if op == operation
    )

    mcp = invoke_goal_tactician_mcp_tool(
        tool_name, request, request_id=f"req:mcp:{operation}"
    )
    cli = invoke_goal_tactician_cli(
        command, request, request_id=f"req:cli:{operation}"
    )

    for channel_payload, channel in ((mcp, "mcp"), (cli, "cli")):
        _assert_envelope(channel_payload, operation=operation)
        assert channel_payload["channel"] == channel
        assert channel_payload["mcp_interface"] == GOAL_TACTICIAN_CLI_MCP_INTERFACE
        assert channel_payload["cli_interface"] == GOAL_TACTICIAN_CLI_MCP_INTERFACE
        assert channel_payload["python_operation"] == operation
        assert channel_payload["status"] == py["status"]
        assert channel_payload["authority"] == py["authority"]
        # Core result semantics must match (ignore channel metadata / request ids).
        for key in (
            "proof_success",
            "admitted",
            "cancelled",
            "ready",
            "executed",
            "supervisor_mutated",
            "mode",
            "count",
            "missing_proof_count",
            "requires_selection",
            "transport_ok",
        ):
            if key in py["result"] or key in channel_payload["result"]:
                assert channel_payload["result"].get(key) == py["result"].get(key), key

    # Transport / API success never silently becomes proof success.
    if py["status"] == "succeeded":
        if operation == "proof_status" and py["result"].get("receipt_count", 0) > 0:
            pass
        else:
            assert py["result"].get("proof_success", False) is False or operation == "proof_status"
            if operation != "proof_status":
                assert py["result"].get("proof_success", False) is False


def test_channels_agree_on_cancellation() -> None:
    request = _formalize_request()
    cancellation = {"cancelled": True}
    py = invoke_goal_tactician(
        "formalize_goal", request, cancellation=cancellation
    ).to_dict()
    mcp = invoke_goal_tactician_mcp_tool(
        "goal_tactician_formalize_goal",
        request,
        cancellation=cancellation,
    )
    cli = invoke_goal_tactician_cli(
        "goal-formalize",
        request,
        cancellation=cancellation,
    )
    for payload in (py, mcp, cli):
        _assert_envelope(payload, operation="formalize_goal")
        assert payload["status"] == "partial"
        assert payload["result"]["cancelled"] is True
        assert payload["result"].get("proof_success", False) is False


def test_channels_agree_on_supervisor_control_refusal() -> None:
    request = {
        "plan_id": "plan:forbidden",
        "steps": [{"step_id": "s", "obligation_id": "o", "statement": "x"}],
        "controls": {"mutate_supervisor": True},
    }
    py = invoke_goal_tactician("execute_proof_plan", request).to_dict()
    mcp = invoke_goal_tactician_mcp_tool(
        "goal_tactician_execute_proof_plan", request
    )
    cli = invoke_goal_tactician_cli("goal-execute-plan", request)
    for payload in (py, mcp, cli):
        assert payload["status"] == "invalid"
        assert "supervisor-only" in payload["diagnostics"][0]
        assert "supervisor_only_control" in payload["unsupported_features"]


def test_unknown_tool_and_command_are_unsupported_and_stable() -> None:
    mcp = invoke_goal_tactician_mcp_tool("goal_tactician_not_a_tool", {})
    cli = invoke_goal_tactician_cli("goal-not-a-command", {})
    assert mcp["status"] == "unsupported"
    assert cli["status"] == "unsupported"
    assert mcp["success"] is False
    assert cli["success"] is False
    assert "mcp_tool:goal_tactician_not_a_tool" in mcp["unsupported_features"]
    assert "cli_command:goal-not-a-command" in cli["unsupported_features"]


def test_list_operations_parity_across_channels() -> None:
    py = invoke_goal_tactician("list_goal_tactician_operations").to_dict()
    mcp = invoke_goal_tactician_mcp_tool("goal_tactician_list_operations")
    cli = invoke_goal_tactician_cli("goal-list-operations")
    for payload in (py, mcp, cli):
        _assert_envelope(payload, operation="list_goal_tactician_operations")
        assert payload["status"] == "declarative"
        assert set(payload["result"]["operations"]) == set(GOAL_TACTICIAN_OPERATIONS)
        assert payload["result"]["interface"] == GOAL_TACTICIAN_CLI_MCP_INTERFACE
        assert payload["result"]["transport_success_implies_proof_success"] is False


def test_proof_status_parity_marks_transport_without_proof() -> None:
    request = {
        "plan_id": "plan:claim",
        "status": "complete",
        "steps": [{"step_id": "s1"}],
        "receipts": [],
    }
    py = invoke_goal_tactician("proof_status", request).to_dict()
    mcp = invoke_goal_tactician_mcp_tool("goal_tactician_proof_status", request)
    cli = invoke_goal_tactician_cli("goal-proof-status", request)
    for payload in (py, mcp, cli):
        assert payload["result"]["transport_ok"] is True
        assert payload["result"]["proof_success"] is False
        assert payload["status"] == "partial"
        assert payload["result"]["identity"]
