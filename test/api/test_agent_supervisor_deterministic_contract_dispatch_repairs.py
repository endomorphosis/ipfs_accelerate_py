"""DCR-043: dispatcher, handler, and datasets logic-route repair operators.

Acceptance
----------
* Same typed obligation reaches ``cec_prove`` locally and through live datasets
  MCP ``tools_dispatch`` with equivalent canonical output/receipt identity.
* Structural operators bind dispatcher/handler/logic routes without inventing
  handler bodies or routing to models.
* Unknown owners, model routes, and missing semantics fail closed or abstain.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.dispatch_repairs import (
    CEC_PROVE_TOOL,
    DEFAULT_CEC_PROVE_GOAL,
    DISPATCH_REPAIR_EVIDENCE,
    DISPATCH_REPAIR_OPERATORS_INTERFACE,
    LOGIC_CEC_PROVE_ROUTE,
    LOGIC_TOOLS_CATEGORY,
    REVIEWED_LOGIC_TOOL_OWNERS,
    BindDispatcherOperator,
    BindHandlerOperator,
    BindLogicToolOperator,
    DispatchBinding,
    DispatchOperatorKind,
    DispatchRepairAbstention,
    DispatchRepairError,
    DispatchTable,
    build_logic_tools_dispatch_table,
    canonicalize_logic_result,
    compare_local_and_mcp_cec_prove,
    dispatch_operator_vectors,
    ensure_cec_prove_bound,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.operators.registry import (
    OperatorFamily,
    OperatorKind,
    build_default_operator_registry,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_dispatch_repair_interface_and_registry_family() -> None:
    assert DISPATCH_REPAIR_OPERATORS_INTERFACE == "DispatchRepairOperators@1"
    assert DISPATCH_REPAIR_EVIDENCE == "dcr/dispatch-repair@1"
    reg = build_default_operator_registry()
    descriptor = reg.require_known(OperatorKind.REPAIR_DISPATCH_BINDING)
    assert descriptor.family is OperatorFamily.DISPATCH
    assert "handler_binding" in descriptor.aliases or descriptor.kind is OperatorKind.REPAIR_DISPATCH_BINDING
    for operator in (
        BindDispatcherOperator(),
        BindHandlerOperator(),
        BindLogicToolOperator(),
    ):
        assert operator.descriptor.kind is OperatorKind.REPAIR_DISPATCH_BINDING
        assert operator.operator_id.startswith("dcr-operator:")


def test_bind_dispatcher_and_handler_preview_inverse_idempotent() -> None:
    table = DispatchTable.empty()
    dispatcher = BindDispatcherOperator()
    preview, after = dispatcher.preview(
        table,
        category=LOGIC_TOOLS_CATEGORY,
        tool=CEC_PROVE_TOOL,
    )
    assert preview.operator_kind == DispatchOperatorKind.BIND_DISPATCHER.value
    assert preview.proposal_only is True
    assert preview.grants_write_authority is False
    assert after.contains(LOGIC_CEC_PROVE_ROUTE)
    assert "datasets.tools_dispatch" in after.dispatcher_ids

    # Inverse of a pure bind removes the route.
    restored = dispatcher.inverse(after, preview)
    assert restored.contains(LOGIC_CEC_PROVE_ROUTE) is False

    # Idempotent re-apply leaves table identity stable.
    after2, preview2 = dispatcher.apply(
        after,
        category=LOGIC_TOOLS_CATEGORY,
        tool=CEC_PROVE_TOOL,
    )
    assert after2.table_id == after.table_id
    assert preview2.before_table_id == preview2.after_table_id
    assert dispatcher.inverse(after2, preview2).table_id == after2.table_id

    # Handler bind requires the target dispatcher to already be registered.
    handler = BindHandlerOperator()
    foreign = DispatchTable(bindings=(), dispatcher_ids=("foreign.dispatcher",))
    with pytest.raises(DispatchRepairAbstention, match="dispatcher_not_bound"):
        handler.preview(
            foreign,
            category=LOGIC_TOOLS_CATEGORY,
            tool=CEC_PROVE_TOOL,
            dispatcher_id="datasets.tools_dispatch",
        )
    bound, handler_preview = handler.apply(
        after,
        category=LOGIC_TOOLS_CATEGORY,
        tool=CEC_PROVE_TOOL,
    )
    assert bound.contains(LOGIC_CEC_PROVE_ROUTE)
    assert handler_preview.binding.owner_ref == REVIEWED_LOGIC_TOOL_OWNERS[CEC_PROVE_TOOL]
    assert handler_preview.binding.model_routed is False


def test_bind_logic_tool_operator_rejects_non_logic_and_model_routes() -> None:
    logic = BindLogicToolOperator()
    table = ensure_cec_prove_bound()
    assert table.contains(LOGIC_CEC_PROVE_ROUTE)

    with pytest.raises(DispatchRepairError, match="logic_tools"):
        logic.preview(table, category="dataset_tools", tool="load_dataset")

    with pytest.raises(DispatchRepairError, match="model|provider"):
        DispatchBinding(
            route="logic_tools/cec_prove",
            category=LOGIC_TOOLS_CATEGORY,
            tool=CEC_PROVE_TOOL,
            owner_ref="ipfs_datasets_py.mcp_server.tools.logic_tools.cec_prove_tool:llm_prompt",
            input_schema_ref="schema:logic_tools/cec_prove/input@1",
        )

    with pytest.raises(DispatchRepairError, match="forbidden body"):
        DispatchBinding.from_dict(
            {
                "route": LOGIC_CEC_PROVE_ROUTE,
                "category": LOGIC_TOOLS_CATEGORY,
                "tool": CEC_PROVE_TOOL,
                "owner_ref": REVIEWED_LOGIC_TOOL_OWNERS[CEC_PROVE_TOOL],
                "input_schema_ref": "schema:logic_tools/cec_prove/input@1",
                "source_body": "def handler(): pass",
            }
        )

    with pytest.raises(DispatchRepairAbstention):
        logic.preview(table, category=LOGIC_TOOLS_CATEGORY, tool="unknown_formula_solver")


def test_build_logic_tools_table_and_vectors_are_content_addressed() -> None:
    table = build_logic_tools_dispatch_table(tools=(CEC_PROVE_TOOL, "logic_health"))
    assert set(table.routes()) == {
        LOGIC_CEC_PROVE_ROUTE,
        f"{LOGIC_TOOLS_CATEGORY}/logic_health",
    }
    rebuilt = DispatchTable.from_dict(table.to_dict())
    assert rebuilt.table_id == table.table_id
    vectors = dispatch_operator_vectors()
    assert vectors["interface"] == DISPATCH_REPAIR_OPERATORS_INTERFACE
    assert vectors["evidence_id"] == DISPATCH_REPAIR_EVIDENCE
    assert LOGIC_CEC_PROVE_ROUTE in vectors["routes"]
    assert vectors["vector_digest"].startswith("sha256:")


def test_datasets_tools_dispatch_exposes_logic_tools_cec_prove() -> None:
    from ipfs_datasets_py.mcp_server import tools_dispatch as datasets_dispatch

    assert datasets_dispatch.LOGIC_CEC_PROVE_ROUTE == LOGIC_CEC_PROVE_ROUTE
    route_table = datasets_dispatch.logic_tools_route_table()
    assert route_table["required_route"] == LOGIC_CEC_PROVE_ROUTE
    assert "cec_prove" in route_table["routes"]
    assert route_table["model_routed"] is False

    listed = _run(datasets_dispatch.tools_list_tools(LOGIC_TOOLS_CATEGORY))
    names = {item["name"] for item in listed["tools"]}
    assert CEC_PROVE_TOOL in names

    schema = _run(datasets_dispatch.tools_get_schema(LOGIC_TOOLS_CATEGORY, CEC_PROVE_TOOL))
    assert schema["status"] == "success"
    assert schema["schema"]["name"] == CEC_PROVE_TOOL
    assert "goal" in schema["schema"]["parameters"]

    handler = datasets_dispatch.resolve_logic_handler(CEC_PROVE_TOOL)
    assert callable(handler)


def test_local_and_mcp_cec_prove_share_canonical_identity() -> None:
    from ipfs_datasets_py.mcp_server import tools_dispatch as datasets_dispatch

    goal = DEFAULT_CEC_PROVE_GOAL
    pair = datasets_dispatch.prove_local_and_mcp(goal=goal, timeout=5)
    assert pair["canonically_equivalent"] is True
    assert pair["receipt"]["tool"] == LOGIC_CEC_PROVE_ROUTE
    assert pair["receipt"]["process_local_cid"] == pair["receipt"]["mcp_result_cid"]
    assert pair["receipt"]["receipt_cid"]

    local_proj = pair["process_local_projection"]
    mcp_proj = pair["mcp_projection"]
    assert local_proj == mcp_proj
    assert local_proj["goal"] == goal
    assert local_proj["surface"] == datasets_dispatch.PROCESS_LOCAL_SURFACE

    # Operator-level comparison uses the same typed obligation and receipt.
    receipt = compare_local_and_mcp_cec_prove(
        goal=goal,
        process_local=pair["process_local"],
        mcp_result=pair["mcp"],
    )
    assert receipt.canonically_equivalent is True
    assert receipt.tool == LOGIC_CEC_PROVE_ROUTE
    assert receipt.process_local_cid == receipt.mcp_result_cid
    assert receipt.receipt_cid == content_identity(
        {
            "tool": LOGIC_CEC_PROVE_ROUTE,
            "goal": goal,
            "process_local_cid": receipt.process_local_cid,
            "mcp_result_cid": receipt.mcp_result_cid,
            "canonically_equivalent": True,
            "dispatcher": "tools_dispatch",
        }
    )

    # Live operator path (imports datasets tools_dispatch) also holds.
    live = compare_local_and_mcp_cec_prove(goal=goal)
    assert live.canonically_equivalent is True
    assert live.process_local_cid == live.mcp_result_cid


def test_tools_dispatch_async_path_matches_process_local_identity() -> None:
    from ipfs_datasets_py.mcp_server import tools_dispatch as datasets_dispatch

    goal = "True"

    async def _both() -> tuple[dict[str, Any], dict[str, Any]]:
        local = await datasets_dispatch.cec_prove_process_local(goal=goal, timeout=5)
        mcp = await datasets_dispatch.cec_prove_via_dispatch(goal=goal, timeout=5)
        return local, mcp

    local, mcp = _run(_both())
    receipt = compare_local_and_mcp_cec_prove(
        goal=goal,
        process_local=local,
        mcp_result=mcp,
    )
    assert receipt.canonically_equivalent is True
    assert receipt.process_local_cid == receipt.mcp_result_cid

    # Direct hierarchical-style call through tools_dispatch(category, tool, params).
    dispatched = _run(
        datasets_dispatch.tools_dispatch(
            LOGIC_TOOLS_CATEGORY,
            CEC_PROVE_TOOL,
            {"goal": goal, "timeout": 5},
        )
    )
    assert dispatched.get("model_routed") is False
    assert dispatched.get("route") == LOGIC_CEC_PROVE_ROUTE
    direct_receipt = compare_local_and_mcp_cec_prove(
        goal=goal,
        process_local=local,
        mcp_result=dispatched,
    )
    assert direct_receipt.canonically_equivalent is True


def test_canonicalize_strips_wall_clock_and_envelope_noise() -> None:
    noisy = {
        "success": True,
        "proved": True,
        "elapsed_ms": 12.5,
        "request_id": "abc",
        "category": LOGIC_TOOLS_CATEGORY,
        "tool": CEC_PROVE_TOOL,
        "dispatcher": "datasets.tools_dispatch",
        "interface": "tools_dispatch@1",
        "model_routed": False,
        "proof_steps": [{"step": 1, "rule": "axiom"}],
    }
    clean = canonicalize_logic_result(noisy, goal="True", surface="process_local")
    assert "elapsed_ms" not in clean
    assert "request_id" not in clean
    assert "category" not in clean
    assert "dispatcher" not in clean
    assert clean["goal"] == "True"
    assert clean["surface"] == "process_local"
    assert clean["proved"] is True
    assert clean["success"] is True


def test_binding_rejects_authority_and_body_smuggling() -> None:
    with pytest.raises(DispatchRepairError):
        DispatchBinding(
            route=LOGIC_CEC_PROVE_ROUTE,
            category=LOGIC_TOOLS_CATEGORY,
            tool=CEC_PROVE_TOOL,
            owner_ref=REVIEWED_LOGIC_TOOL_OWNERS[CEC_PROVE_TOOL],
            input_schema_ref="schema:logic_tools/cec_prove/input@1",
            semantic_authority=True,
        )
    with pytest.raises(DispatchRepairError):
        DispatchBinding(
            route=LOGIC_CEC_PROVE_ROUTE,
            category=LOGIC_TOOLS_CATEGORY,
            tool=CEC_PROVE_TOOL,
            owner_ref=REVIEWED_LOGIC_TOOL_OWNERS[CEC_PROVE_TOOL],
            input_schema_ref="schema:logic_tools/cec_prove/input@1",
            allows_source_generation=True,
        )
    with pytest.raises(DispatchRepairError, match="module:callable"):
        DispatchBinding(
            route=LOGIC_CEC_PROVE_ROUTE,
            category=LOGIC_TOOLS_CATEGORY,
            tool=CEC_PROVE_TOOL,
            owner_ref="not-a-module-path",
            input_schema_ref="schema:logic_tools/cec_prove/input@1",
        )
