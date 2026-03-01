# MCP Server Unification Master Plan

## 1. Objective

Create a single canonical MCP runtime in `ipfs_accelerate_py/mcp_server` by porting from
`ipfs_datasets_py/ipfs_datasets_py/mcp_server`, merging MCP++ capabilities from
`ipfs_accelerate_py/mcplusplus_module`, and closing conformance against spec artifacts in
`ipfs_accelerate_py/mcpplusplus`.

Target outcome:

- `ipfs_accelerate_py/mcp_server` is the default runtime path.
- `ipfs_accelerate_py/mcp` remains a temporary compatibility facade.
- Spec tracking in `mcpplusplus/CONFORMANCE_CHECKLIST.md` and `mcpplusplus/SPEC_GAP_MATRIX.md`
  is the merge gate for cutover.

## 2. Source and Target Reality

Verified source inputs:

- Full source server: `ipfs_datasets_py/ipfs_datasets_py/mcp_server`
- Source MCP++ primitives: `ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/*`
- Existing MCP++ runtime package: `ipfs_accelerate_py/mcplusplus_module/*`

Verified target outputs:

- Canonical runtime package: `ipfs_accelerate_py/mcp_server`
- Target MCP++ primitive package: `ipfs_accelerate_py/mcp_server/mcplusplus`
- Spec directory: `ipfs_accelerate_py/mcpplusplus`

## 3. Current Progress (As Of Now)

Completed:

- Core unification control-plane is active in target:
  - `runtime_router.py`, `hierarchical_tool_manager.py`, `registration_adapter.py`, `tool_registry.py`, `tool_metadata.py`
- Unified bridge/bootstrap and meta-tools are active:
  - `tools_list_categories`, `tools_list_tools`, `tools_get_schema`, `tools_dispatch`, `tools_runtime_metrics`
- Wave A native tool categories in target:
  - `ipfs`, `workflow`, `p2p`
- MCP++ primitive package added in target:
  - `task_queue.py`, `workflow_scheduler.py`, `workflow_dag.py`, `workflow_engine.py`, `peer_registry.py`, `peer_discovery.py`, `result_cache.py`
- Unified service container wiring added at bootstrap:
  - `server._unified_services` factories for queue/workflow/peer/cache
- Transport validation expanded:
  - router parity tests
  - in-process E2E transport matrix tests
  - process helper wiring tests
  - subprocess startup contract tests
  - optional real-network trio-p2p test

Open high-priority gaps:

- Tool-surface parity (source has 51 categories; target has 3 migrated categories).
- Security/policy/audit parity.
- Monitoring/tracing/exporter parity.
- `MCPP-013` transport parity CI enforcement in libp2p-enabled lane.

## 4. End-State Architecture

### 4.1 Runtime Layout

- Canonical runtime: `ipfs_accelerate_py/mcp_server`
- Compatibility facade: `ipfs_accelerate_py/mcp`
- Spec source-of-truth: `ipfs_accelerate_py/mcpplusplus`

### 4.2 Control Plane

- Single registration/dispatch path via hierarchical manager.
- Runtime routing resolved through `RuntimeRouter` metadata and timeout semantics.
- Deterministic category loaders and native-over-legacy precedence.

### 4.3 Execution Plane

- Unified support for stdio/http/trio-p2p dispatch contracts.
- Shared service container for MCP++ queue/workflow/peer/cache primitives.

## 5. Conformance Status Summary

From `mcpplusplus/CONFORMANCE_CHECKLIST.md`:

- PASS: `MCPP-001` through `MCPP-011`
- PARTIAL: `MCPP-012` (tool category parity), `MCPP-013` (transport parity enforcement)
- GAP: `MCPP-014` (security/policy/audit), `MCPP-015` (observability/exporters)

## 6. Workstreams

### W1. Spec and Governance (P0)

- Keep checklist and matrix as authoritative merge gates.
- Require evidence links for each status transition.

### W2. Runtime Hardening (P0)

- Stabilize bridge/bootstrap defaults and recursion protections.
- Ensure service container factories remain side-effect-light at startup.

### W3. MCP++ Primitive Integration (P0)

- Move from primitive existence to runtime integration in server lifecycle:
  - queue usage in p2p dispatch paths
  - workflow orchestration hook points
  - peer lifecycle hook points
  - cache reuse hook points in dispatch/workflow execution

### W4. Tool Parity Waves (P0/P1)

- Port source tool categories in waves with deterministic tests.
- Add schema parity checks for migrated tools.

### W5. Transport Parity and CI Lanes (P0)

- Keep existing transport parity suites as required.
- Add libp2p-enabled CI lane that runs `test_mcp_transport_trio_p2p_networked.py` as required.

### W6. Security and Policy (P1)

- Port `policy_audit_log.py`, `secrets_vault.py`, `risk_scorer.py` equivalents.
- Add enforcement/integrity tests.

### W7. Observability and Reliability (P1)

- Port monitoring/tracing/exporters.
- Add SLO and failure-rate assertions for critical paths.

### W8. Cutover and Deprecation (P0/P1)

- Transition default runtime to `mcp_server` once P0 gates are green.
- Keep rollback path and compatibility facade for one release window.

## 7. Detailed Porting Plan

### Phase A: Lock Core and Transport Gates (1 week)

Deliverables:

- Ensure transport test suites are green in default lane:
  - `test_mcp_server_transport_parity.py`
  - `test_mcp_server_transport_e2e_matrix.py`
  - `test_mcp_transport_process_level.py`
  - `test_mcp_transport_subprocess_contracts.py`
- Add CI lane for libp2p-enabled optional test as required.

Exit criteria:

- `MCPP-013` evidence is executable in both default and libp2p CI lanes.

### Phase B: Expand Tool Surface (2-4 weeks)

Priority order:

1. `ipfs_tools` parity expansion
2. `workflow_tools` parity expansion
3. `p2p_tools` and `mcplusplus_*_tools` completion
4. Next critical categories: `security_tools`, `monitoring_tools`, `storage_tools`, `vector_tools`

Per-category checklist:

- Register native category loader in `mcp_server`.
- Add deterministic `tools_dispatch` tests.
- Add schema parity assertions for representative tools.
- Add native precedence skip rules in wave loaders when needed.

Exit criteria:

- `MCPP-012` moved from `PARTIAL` to agreed threshold.

### Phase C: Security and Observability Parity (2 weeks)

Deliverables:

- Security/policy/audit modules ported and covered.
- Monitoring/tracing/exporter modules ported and covered.

Exit criteria:

- `MCPP-014` and `MCPP-015` moved to `PASS` with evidence.

### Phase D: Default Runtime Cutover (1 week)

Deliverables:

- Default startup path uses unified runtime without migration flags.
- Compatibility facade remains as rollback path.

Exit criteria:

- All P0 checklist items green.
- Rollback procedure documented and tested.

## 8. Source-to-Target Module Mapping (High-Level)

Core runtime:

- Source `runtime_router.py` -> Target `mcp_server/runtime_router.py`
- Source `hierarchical_tool_manager.py` -> Target `mcp_server/hierarchical_tool_manager.py`
- Source `dispatch_pipeline.py` -> integrate into target routing/dispatch layers as needed

MCP++ primitives:

- Source `mcplusplus/task_queue.py` -> Target `mcp_server/mcplusplus/task_queue.py`
- Source `mcplusplus/workflow_*` -> Target `mcp_server/mcplusplus/workflow_*`
- Source `mcplusplus/peer_*` -> Target `mcp_server/mcplusplus/peer_*`
- Source `mcplusplus/result_cache.py` -> Target `mcp_server/mcplusplus/result_cache.py`

Transports and lifecycle:

- Source `standalone_server.py`/`fastapi_service.py`/`trio_adapter.py`
  -> Target `mcp_server/server.py` + `mcp/integration.py` + `mcp/standalone.py` parity path

## 9. Execution Gates

Gate G1: Conformance gate

- No merge of major migration slice without checklist/matrix update and test evidence.

Gate G2: Regression gate

- New category migrations require deterministic dispatch tests.

Gate G3: Transport gate

- Transport suites must pass in default lane.
- Networked trio-p2p suite must pass in libp2p-enabled lane.

Gate G4: Cutover gate

- `MCPP-012` threshold met.
- `MCPP-013`, `MCPP-014`, `MCPP-015` at `PASS`.

## 10. Risks and Mitigations

Risk: Tool registry drift between legacy and native paths.

- Mitigation: single adapter path + native precedence skip rules + schema tests.

Risk: Transport parity appears green locally but fails in real p2p runtime.

- Mitigation: required libp2p CI lane and subprocess/networked tests.

Risk: Primitive wrappers exist but are not actually used in runtime.

- Mitigation: service container wiring and integration tests through `tools_dispatch`.

Risk: Scope expansion across 51 source categories.

- Mitigation: wave sequencing with category-level exit criteria.

## 11. Immediate Next Actions (Concrete)

1. Promote `test_mcp_transport_trio_p2p_networked.py` to required in libp2p CI lane.
2. Start Wave B category migration with one high-impact category from source `tools/`.
3. Add runtime integration tests proving queue/cache/workflow/peer services are invoked from dispatch paths.
4. Start security parity port (`policy_audit_log`, `secrets_vault`, `risk_scorer`) with tests.

## 12. Success Definition

This plan is complete when:

- `ipfs_accelerate_py/mcp_server` is default runtime.
- `mcpplusplus/CONFORMANCE_CHECKLIST.md` critical requirements are all `PASS`.
- Transport parity is enforced in both default and libp2p-enabled CI lanes.
- Legacy `mcp` facade remains only as a temporary rollback-compatible layer.
