"""Contract tests for native agent-supervisor control and goal/task extraction.

SCA-174 / SCA-G174 acceptance:

* Every SwissKnife capability maps to one exact native ``agent_supervisor_*``
  request/result/dispatcher/function identity or is explicitly refuted.
* Generic proxy selection is rejected.
* Goal completion requires current child/evidence/health/exhaustion closure.
* Mutation paths are policy dominated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.supervisor_contract_extractor import (
    GENERIC_CONTROL_ADAPTER_TOOL,
    GENERIC_USAGE_TOOL,
    GOAL_COMPLETION_EVALUATOR_IDENTITY,
    MUTATION_POLICY_PATH,
    MappingDisposition,
    NATIVE_EXECUTOR_IDENTITY,
    REQUIRED_COMPLETION_CLOSURE,
    SUPERVISOR_CONTRACT_CATALOG_INTERFACE,
    SupervisorContractCatalog,
    GenericProxySelectionError,
    SupervisorContractExtractorError,
    assert_no_generic_proxy_selection,
    build_native_operation_identity,
    build_supervisor_contract_catalog,
    extract_goal_completion_contract,
    extract_mutation_policy_contracts,
    extract_native_operations,
    extract_swissknife_capabilities,
    is_generic_proxy_reason,
    map_swissknife_capability,
    native_tool_name,
    parse_native_tool_name,
    resolve_task_control_action,
    select_native_identity,
    summarize_catalog,
    write_supervisor_contract_catalog,
)
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    MUTATION_OPERATIONS,
    OPERATION_CATALOG_V2,
    Operation,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    DIRECT_CONTROL_SERVICE_DISPATCHER_ID,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools.native_agent_supervisor_tools import (
    AGENT_SUPERVISOR_OPERATION_TOOLS,
    execute_agent_supervisor_operation,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SWISSKNIFE_ROOT = REPOSITORY_ROOT / "swissknife"


@pytest.fixture(scope="module")
def catalog() -> SupervisorContractCatalog:
    root = SWISSKNIFE_ROOT if SWISSKNIFE_ROOT.is_dir() else None
    return build_supervisor_contract_catalog(swissknife_root=root)


def test_native_operations_cover_catalog_exactly(catalog: SupervisorContractCatalog) -> None:
    expected = {operation.value for operation in OPERATION_CATALOG_V2.operations}
    actual = {item.operation for item in catalog.native_operations}
    assert actual == expected
    assert len(catalog.native_operations) == len(expected)

    for identity in catalog.native_operations:
        assert identity.tool_name == native_tool_name(identity.operation)
        assert identity.tool_name.startswith("agent_supervisor_")
        assert identity.dispatcher_id == DIRECT_CONTROL_SERVICE_DISPATCHER_ID
        assert identity.executor_identity == NATIVE_EXECUTOR_IDENTITY
        assert identity.request_schema_id.startswith("b")
        assert identity.result_schema_id.startswith("b")
        assert identity.behavior_id.startswith("sha256:")
        assert identity.function_identity.endswith(f":{identity.tool_name}")
        assert identity.identity_cid.startswith("b")

        # Live MCP publication binds the same tool callable.
        operation = Operation(identity.operation)
        tool = AGENT_SUPERVISOR_OPERATION_TOOLS[operation]
        assert tool.__name__ == identity.tool_name
        assert getattr(tool, "__agent_supervisor_executor__", None) is (
            execute_agent_supervisor_operation
        )


def test_every_swissknife_capability_is_mapped_or_refuted(
    catalog: SupervisorContractCatalog,
) -> None:
    capability_ids = {item.capability_id for item in catalog.capabilities}
    mapping_ids = {item.capability_id for item in catalog.mappings}
    assert capability_ids == mapping_ids
    assert capability_ids  # non-empty inventory

    # Required console capabilities from the schema must appear.
    for required in (
        "supervisor.health.read",
        "supervisor.queue.read",
        "supervisor.goals.read",
        "supervisor.subgoals.read",
        "supervisor.taskboard.links.read",
        "supervisor.logs.read",
        "supervisor.receipts.read",
        "supervisor.run-history.search",
        "supervisor.prompt-steering.request",
        "supervisor.task-control.request",
    ):
        assert required in capability_ids

    for mapping in catalog.mappings:
        if mapping.disposition is MappingDisposition.MAPPED:
            assert mapping.identity is not None
            assert mapping.operation
            assert mapping.identity.tool_name == f"agent_supervisor_{mapping.operation}"
            assert mapping.identity.dispatcher_id == DIRECT_CONTROL_SERVICE_DISPATCHER_ID
            assert mapping.reason_code == ""
        elif mapping.disposition is MappingDisposition.REFUTED:
            assert mapping.identity is None
            assert mapping.reason_code
            assert is_generic_proxy_reason(mapping.reason_code) or mapping.reason_code
        elif mapping.disposition is MappingDisposition.ACTION_MAPPED:
            assert mapping.action_bindings
            assert mapping.details.get("generic_control_adapter_rejected") is True
            for binding in mapping.action_bindings:
                if binding.disposition is MappingDisposition.MAPPED:
                    assert binding.identity is not None
                    assert binding.identity.tool_name == (
                        f"agent_supervisor_{binding.operation}"
                    )
                else:
                    assert binding.reason_code
        else:  # pragma: no cover
            raise AssertionError(f"unexpected disposition {mapping.disposition}")


def test_exact_capability_mappings_resolve_to_one_native_identity(
    catalog: SupervisorContractCatalog,
) -> None:
    expected = {
        "supervisor.health.read": "health",
        "supervisor.queue.read": "tasks",
        "supervisor.goals.read": "goals",
        "supervisor.subgoals.read": "goals",
        "supervisor.logs.read": "events",
        "supervisor.receipts.read": "receipts",
        "supervisor.prompt-steering.request": "objective_refine",
    }
    for capability_id, operation in expected.items():
        mapping = catalog.mapping_for(capability_id)
        assert mapping.disposition is MappingDisposition.MAPPED
        assert mapping.operation == operation
        assert mapping.identity is not None
        assert mapping.identity.tool_name == f"agent_supervisor_{operation}"
        assert mapping.identity.function_identity.endswith(
            f":agent_supervisor_{operation}"
        )


def test_generic_proxy_capabilities_are_refuted(
    catalog: SupervisorContractCatalog,
) -> None:
    refuted = {
        "supervisor.taskboard.links.read": "datasets_search_proxy",
        "supervisor.run-history.search": "datasets_search_proxy",
        "supervisor.policy.assist": "datasets_search_proxy",
        "supervisor.receipts.persist": "kit_storage_proxy",
        "supervisor.content.retrieve": "kit_storage_proxy",
        "supervisor.profile-g.read": "generic_mcp_plus_plus_proxy",
        "supervisor.schedule.claim": "generic_mcp_plus_plus_proxy",
        "supervisor.schedule.reconcile": "generic_mcp_plus_plus_proxy",
    }
    for capability_id, reason in refuted.items():
        mapping = catalog.mapping_for(capability_id)
        assert mapping.disposition is MappingDisposition.REFUTED
        assert mapping.reason_code == reason
        assert mapping.identity is None


def test_generic_proxy_selection_is_rejected() -> None:
    for selection, reason in (
        (GENERIC_CONTROL_ADAPTER_TOOL, "generic_control_adapter"),
        (GENERIC_USAGE_TOOL, "generic_usage_tool"),
        ("mcp++/schedule/claim", "workflow_data_storage_proxy"),
        ("workflow/materialize", "workflow_data_storage_proxy"),
        ("data/search", "workflow_data_storage_proxy"),
        ("storage/put", "workflow_data_storage_proxy"),
        ("some_other_tool", "unknown_tool_name"),
    ):
        with pytest.raises(GenericProxySelectionError) as excinfo:
            select_native_identity(selection)
        assert excinfo.value.reason_code == reason
        with pytest.raises(GenericProxySelectionError):
            assert_no_generic_proxy_selection(selection)

    # Exact native selection succeeds.
    identity = select_native_identity("agent_supervisor_health")
    assert identity.operation == "health"
    assert identity.tool_name == "agent_supervisor_health"
    assert parse_native_tool_name("agent_supervisor_pause") is Operation.PAUSE


def test_task_control_action_mapping_is_exact_and_policy_aware(
    catalog: SupervisorContractCatalog,
) -> None:
    mapping = catalog.mapping_for("supervisor.task-control.request")
    assert mapping.disposition is MappingDisposition.ACTION_MAPPED
    by_action = {item.action: item for item in mapping.action_bindings}

    for action in ("pause", "resume", "retry", "cancel"):
        binding = by_action[action]
        assert binding.disposition is MappingDisposition.MAPPED
        assert binding.operation == action
        assert binding.identity is not None
        assert binding.identity.authority == "mutation"
        assert binding.identity.policy_dominated is True
        assert binding.identity.requires_authorization is True
        assert binding.identity.supports_dry_run is True
        resolved = resolve_task_control_action(action)
        assert resolved.tool_name == f"agent_supervisor_{action}"

    for action in ("claim", "release"):
        binding = by_action[action]
        assert binding.disposition is MappingDisposition.REFUTED
        with pytest.raises(GenericProxySelectionError):
            resolve_task_control_action(action)


def test_goal_completion_requires_child_evidence_health_exhaustion(
    catalog: SupervisorContractCatalog,
) -> None:
    contract = catalog.goal_completion
    assert contract.evaluator_identity == GOAL_COMPLETION_EVALUATOR_IDENTITY
    assert contract.fail_closed is True
    assert contract.claim_family == "GoalTaskClosure"
    for member in REQUIRED_COMPLETION_CLOSURE:
        assert member in contract.required_closure
    for member in (
        "child_goals",
        "required_validations",
        "analyzer_health",
        "exhaustion_quorum",
    ):
        assert member in contract.required_closure
        assert member in contract.gate_check_names or member == "required_validations"
    assert "child_goals" in contract.gate_check_names
    assert "analyzer_health" in contract.gate_check_names
    assert "exhaustion_quorum" in contract.gate_check_names
    assert "required_validations" in contract.gate_check_names

    # Standalone extractor agrees with the catalog projection.
    standalone = extract_goal_completion_contract()
    assert standalone.required_closure == contract.required_closure
    assert standalone.evaluator_identity == contract.evaluator_identity


def test_mutation_paths_are_policy_dominated(
    catalog: SupervisorContractCatalog,
) -> None:
    mutation_ops = {item.value for item in MUTATION_OPERATIONS}
    policy_ops = {item.operation for item in catalog.mutation_policies}
    assert policy_ops == mutation_ops

    for policy in catalog.mutation_policies:
        assert policy.authority == "mutation"
        assert policy.requires_authorization is True
        assert policy.requires_idempotency is True
        assert policy.supports_dry_run is True
        assert policy.policy_dominated is True
        assert tuple(policy.policy_path) == MUTATION_POLICY_PATH
        assert policy.identity.policy_dominated is True
        assert policy.tool_name == f"agent_supervisor_{policy.operation}"
        # Preview/permit/receipt path members are present and ordered.
        assert policy.policy_path[0] == "dry_run_or_preview"
        assert "authorization" in policy.policy_path
        assert "execution_permit" in policy.policy_path
        assert "audit_receipt" in policy.policy_path


def test_catalog_is_cid_stable_and_round_trips(
    catalog: SupervisorContractCatalog, tmp_path: Path
) -> None:
    first = catalog.to_dict()
    second = SupervisorContractCatalog.from_dict(first).to_dict()
    assert first["catalog_cid"] == second["catalog_cid"]
    assert first["catalog_cid"].startswith("b")

    path = tmp_path / "supervisor-contract-catalog.json"
    written_cid = write_supervisor_contract_catalog(catalog, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["catalog_cid"] == written_cid
    assert loaded["interface"] == SUPERVISOR_CONTRACT_CATALOG_INTERFACE

    summary = summarize_catalog(catalog)
    assert summary["native_operation_count"] == len(catalog.native_operations)
    assert summary["capability_count"] == len(catalog.capabilities)
    assert summary["mapped_count"] + summary["refuted_count"] + summary[
        "action_mapped_count"
    ] == len(catalog.mappings)


def test_swissknife_source_extraction_when_tree_present() -> None:
    if not SWISSKNIFE_ROOT.is_dir():
        pytest.skip("SwissKnife tree is not present in this checkout")
    capabilities = extract_swissknife_capabilities(SWISSKNIFE_ROOT)
    ids = {item.capability_id for item in capabilities}
    assert "supervisor.health.read" in ids
    assert "supervisor.task-control.request" in ids
    # Schema-closed methods that are mcp++ proxies remain visible so they can
    # be refuted rather than silently dropped.
    assert "supervisor.schedule.claim" in ids or any(
        item.method.startswith("mcp++/") for item in capabilities
    )


def test_ui_owner_is_not_native_reachability() -> None:
    mapping = map_swissknife_capability(
        {
            "capability_id": "supervisor.custom.unknown",
            "method": "agent_supervisor.custom.unknown",
            "owner": "ipfs_datasets_py",
            "access": "read",
            "policy_class": "read",
        }
    )
    assert mapping.disposition is MappingDisposition.REFUTED
    assert mapping.reason_code == "workflow_data_storage_proxy"


def test_build_identity_matches_live_descriptor() -> None:
    identity = build_native_operation_identity(Operation.HEALTH)
    descriptor = OPERATION_CATALOG_V2.operation(Operation.HEALTH)
    assert identity.request_schema_id == descriptor.request_schema_id
    assert identity.result_schema_id == descriptor.result_schema_id
    assert identity.backend_capability == descriptor.backend_capability


def test_extract_native_operations_sorted_and_complete() -> None:
    operations = extract_native_operations()
    values = [item.operation for item in operations]
    assert values == sorted(values)
    assert set(values) == {item.value for item in OPERATION_CATALOG_V2.operations}


def test_mutation_policy_extractor_matches_catalog(
    catalog: SupervisorContractCatalog,
) -> None:
    standalone = extract_mutation_policy_contracts()
    assert {item.operation for item in standalone} == {
        item.operation for item in catalog.mutation_policies
    }


def test_invalid_swissknife_root_fails_closed() -> None:
    with pytest.raises(SupervisorContractExtractorError) as excinfo:
        extract_swissknife_capabilities("/no/such/swissknife/root")
    assert excinfo.value.reason_code == "invalid_swissknife_root"
