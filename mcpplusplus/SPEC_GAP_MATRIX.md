# MCP++ Spec Gap Matrix

This matrix tracks source-to-target parity for the unified MCP runtime migration.

## Snapshot

- Source runtime: `ipfs_datasets_py/ipfs_datasets_py/mcp_server`
- Target runtime: `ipfs_accelerate_py/mcp_server`
- Source MCP++ primitives: `ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus`
- Target MCP++ primitives: `ipfs_accelerate_py/mcp_server/mcplusplus` (task queue + workflow + peer + cache primitives created)
- Source tool categories: 51
- Target migrated tool categories: 3 (`ipfs`, `workflow`, `p2p`)
- Unified bootstrap tests: 45 in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py`

## Capability Matrix

| Capability | Source Reference | Target Reference | Status | Test Evidence | Priority | Next Action |
| --- | --- | --- | --- | --- | --- | --- |
| Runtime router and timeout semantics | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/runtime_router.py` | `ipfs_accelerate_py/mcp_server/runtime_router.py` | PASS | `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` | P0 | Keep expanding router coverage as new runtime metadata is added. |
| Hierarchical registration and metadata path | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/hierarchical_tool_manager.py` | `ipfs_accelerate_py/mcp_server/hierarchical_tool_manager.py`, `ipfs_accelerate_py/mcp_server/registration_adapter.py` | PASS | `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` | P0 | Maintain single registration path per tool. |
| Unified meta-tools control plane | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/tool_registry.py` | `ipfs_accelerate_py/mcp_server/server.py` | PASS | `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` | P0 | Keep schema and dispatch behavior deterministic. |
| Native IPFS tools | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/ipfs_tools/` | `ipfs_accelerate_py/mcp_server/tools/ipfs/native_ipfs_tools.py` | PARTIAL | `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` | P1 | Continue parity expansion for remaining source IPFS functionality. |
| Native workflow tools | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/workflow_tools/` | `ipfs_accelerate_py/mcp_server/tools/workflow/native_workflow_tools.py` | PARTIAL | `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` | P1 | Port remaining workflow operations and preserve schemas. |
| Native p2p tools and task queue calls | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/p2p_tools/`, `ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/mcplusplus_taskqueue_tools.py` | `ipfs_accelerate_py/mcp_server/tools/p2p/native_p2p_tools.py` | PARTIAL | `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` | P1 | Continue p2p parity and extend deterministic dispatch coverage. |
| MCP++ task queue primitive | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/task_queue.py` | `ipfs_accelerate_py/mcp_server/mcplusplus/task_queue.py` | PASS | `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_task_queue.py` | P0 | Extend parity from wrapper-level support to full runtime service integration. |
| MCP++ workflow engine, scheduler, and DAG primitives | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/workflow_engine.py`, `ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/workflow_scheduler.py`, `ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/workflow_dag.py` | `ipfs_accelerate_py/mcp_server/mcplusplus/workflow_scheduler.py`, `ipfs_accelerate_py/mcp_server/mcplusplus/workflow_engine.py`, `ipfs_accelerate_py/mcp_server/mcplusplus/workflow_dag.py` | PASS | `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_workflow_scheduler.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_workflow_dag.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_workflow_engine.py` | P0 | Integrate primitives into shared runtime services and transport-level execution paths. |
| MCP++ peer discovery and peer registry primitives | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/peer_discovery.py`, `ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/peer_registry.py` | `ipfs_accelerate_py/mcp_server/mcplusplus/peer_registry.py`, `ipfs_accelerate_py/mcp_server/mcplusplus/peer_discovery.py` | PASS | `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_peer_primitives.py` | P0 | Integrate peer lifecycle hooks into server bootstrap and runtime service container. |
| MCP++ result cache primitive | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/result_cache.py` | `ipfs_accelerate_py/mcp_server/mcplusplus/result_cache.py` | PASS | `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_result_cache.py` | P0 | Connect result cache to dispatch and workflow execution reuse paths. |
| Transport parity (stdio/http/trio-p2p) | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/standalone_server.py`, `ipfs_datasets_py/ipfs_datasets_py/mcp_server/fastapi_service.py`, `ipfs_datasets_py/ipfs_datasets_py/mcp_server/trio_adapter.py` | `ipfs_accelerate_py/mcp_server/server.py`, `ipfs_accelerate_py/mcp_server/runtime_router.py`, `ipfs_accelerate_py/mcp/integration.py`, `ipfs_accelerate_py/mcp/standalone.py` | PARTIAL | `ipfs_accelerate_py/mcp/tests/test_mcp_server_transport_parity.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_transport_e2e_matrix.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_transport_process_level.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_transport_subprocess_contracts.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_transport_trio_p2p_networked.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` | P0 | Promote optional networked trio-p2p test to required CI lane where libp2p is available and assert legacy-vs-unified parity contracts there. |
| Security and policy subsystems | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/policy_audit_log.py`, `ipfs_datasets_py/ipfs_datasets_py/mcp_server/secrets_vault.py`, `ipfs_datasets_py/ipfs_datasets_py/mcp_server/risk_scorer.py` | `ipfs_accelerate_py/mcp_server/` | GAP | None | P1 | Port policy/audit/vault/risk primitives with conformance tests. |
| Observability and exporters | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/monitoring.py`, `ipfs_datasets_py/ipfs_datasets_py/mcp_server/otel_tracing.py`, `ipfs_datasets_py/ipfs_datasets_py/mcp_server/prometheus_exporter.py` | `ipfs_accelerate_py/mcp_server/` | GAP | None | P1 | Port monitoring and tracing parity, then add SLO checks. |
| Compatibility facade cutover readiness | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/server.py` | `ipfs_accelerate_py/mcp_server/server.py`, `ipfs_accelerate_py/mcp/server.py` | PARTIAL | `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` | P0 | Hold facade until all P0 gaps are closed and rollback is validated. |

## Tracking Rules

1. Update this matrix whenever a capability moves between `GAP`, `PARTIAL`, and `PASS`.
2. No capability should move to `PASS` without a deterministic test path in `ipfs_accelerate_py/mcp/tests/`.
3. Any deferred item must include rationale and a target phase in `MCP_SERVER_UNIFICATION_PLAN.md`.
