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
- Full chapter-by-chapter MCP++ spec implementation beyond current primitive parity.

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
- PARTIAL: `MCPP-012` (tool category parity)
- PASS: `MCPP-013` (transport parity enforcement)
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

1. Start Wave B category migration with one high-impact category from source `tools/`.
2. Add runtime integration tests proving queue/cache/workflow/peer services are invoked from dispatch paths.
3. Start security parity port (`policy_audit_log`, `secrets_vault`, `risk_scorer`) with tests.
4. Begin MCP-IDL profile implementation (descriptor canonicalization + repository APIs).

## 12. Success Definition

This plan is complete when:

- `ipfs_accelerate_py/mcp_server` is default runtime.
- `mcpplusplus/CONFORMANCE_CHECKLIST.md` critical requirements are all `PASS`.
- Transport parity is enforced in both default and libp2p-enabled CI lanes.
- Legacy `mcp` facade remains only as a temporary rollback-compatible layer.

## 13. Sprint Execution Board

Legend:

- Status: `todo` | `in-progress` | `done`
- Owner lanes: `runtime`, `tools`, `transport`, `security`, `observability`, `release`

### Sprint 1 (Week 1): Transport Gate Closure

1. Status: `done`  Owner: `transport`
  Task: Promote libp2p trio networked test to required CI lane.
  Files: `.github/workflows/*`, `ipfs_accelerate_py/mcp/tests/test_mcp_transport_trio_p2p_networked.py`
  Verify:
  - `python3 -m unittest ipfs_accelerate_py.mcp.tests.test_mcp_transport_trio_p2p_networked`

2. Status: `todo`  Owner: `transport`
  Task: Keep non-libp2p transport suite green and required in default lane.
  Files: `ipfs_accelerate_py/mcp/tests/test_mcp_server_transport_parity.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_transport_e2e_matrix.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_transport_process_level.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_transport_subprocess_contracts.py`
  Verify:
  - `python3 -m unittest ipfs_accelerate_py.mcp.tests.test_mcp_server_transport_parity ipfs_accelerate_py.mcp.tests.test_mcp_server_transport_e2e_matrix ipfs_accelerate_py.mcp.tests.test_mcp_transport_process_level ipfs_accelerate_py.mcp.tests.test_mcp_transport_subprocess_contracts`

3. Status: `todo`  Owner: `runtime`
  Task: Add integration tests proving `_unified_services` factories are consumed by runtime paths (not only present).
  Files: `ipfs_accelerate_py/mcp_server/server.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py`
  Verify:
  - `python3 -m unittest ipfs_accelerate_py.mcp.tests.test_mcp_server_unified_bootstrap`

Sprint 1 Exit:

- `MCPP-013` has required CI coverage in both default and libp2p-enabled lanes.

### Sprint 2 (Weeks 2-3): Tool Surface Wave B

1. Status: `todo`  Owner: `tools`
  Task: Port next high-impact source categories (start with `security_tools`, `monitoring_tools`).
  Files: `ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/*`, `ipfs_accelerate_py/mcp_server/tools/*`, `ipfs_accelerate_py/mcp_server/wave_a_loaders.py`
  Verify:
  - deterministic `tools_dispatch` tests per migrated tool in `ipfs_accelerate_py/mcp/tests/`

2. Status: `todo`  Owner: `tools`
  Task: Add schema parity checks for representative tools per migrated category.
  Files: `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (or category-specific test files)
  Verify:
  - `python3 -m unittest <new_category_test_modules>`

3. Status: `todo`  Owner: `runtime`
  Task: Update native precedence skip lists to prevent duplicate registrations.
  Files: `ipfs_accelerate_py/mcp_server/wave_a_loaders.py`
  Verify:
  - `python3 -m unittest ipfs_accelerate_py.mcp.tests.test_mcp_server_unified_bootstrap`

Sprint 2 Exit:

- `MCPP-012` parity percentage increased with evidence updates in both conformance docs.

### Sprint 3 (Week 4): Security and Policy Parity

1. Status: `todo`  Owner: `security`
  Task: Port policy/audit/vault/risk primitives from source server.
  Files: `ipfs_datasets_py/ipfs_datasets_py/mcp_server/policy_audit_log.py`, `ipfs_datasets_py/ipfs_datasets_py/mcp_server/secrets_vault.py`, `ipfs_datasets_py/ipfs_datasets_py/mcp_server/risk_scorer.py`, `ipfs_accelerate_py/mcp_server/*`
  Verify:
  - New unit/integration tests under `ipfs_accelerate_py/mcp/tests/`

2. Status: `todo`  Owner: `security`
  Task: Add policy enforcement and audit integrity tests for migrated paths.
  Files: `ipfs_accelerate_py/mcp/tests/*security*`
  Verify:
  - `python3 -m unittest <security_test_modules>`

Sprint 3 Exit:

- `MCPP-014` moved to `PASS`.

### Sprint 4 (Week 5): Observability Parity

1. Status: `todo`  Owner: `observability`
  Task: Port monitoring/tracing/exporter modules from source server.
  Files: `ipfs_datasets_py/ipfs_datasets_py/mcp_server/monitoring.py`, `ipfs_datasets_py/ipfs_datasets_py/mcp_server/otel_tracing.py`, `ipfs_datasets_py/ipfs_datasets_py/mcp_server/prometheus_exporter.py`, `ipfs_accelerate_py/mcp_server/*`
  Verify:
  - observability tests in `ipfs_accelerate_py/mcp/tests/`

2. Status: `todo`  Owner: `observability`
  Task: Add SLO/latency/error-rate assertions for critical dispatch paths.
  Files: `ipfs_accelerate_py/mcp/tests/*transport*`, `ipfs_accelerate_py/mcp/tests/*metrics*`
  Verify:
  - `python3 -m unittest <observability_test_modules>`

Sprint 4 Exit:

- `MCPP-015` moved to `PASS`.

### Sprint 5 (Week 6): Cutover and Release

1. Status: `todo`  Owner: `release`
  Task: Flip default runtime to unified `mcp_server` path.
  Files: `ipfs_accelerate_py/mcp/server.py`, `ipfs_accelerate_py/mcp_server/server.py`, startup scripts/config
  Verify:
  - full MCP regression suite and transport suite pass

2. Status: `todo`  Owner: `release`
  Task: Keep compatibility facade and validate rollback path.
  Files: `ipfs_accelerate_py/mcp/*`, release docs/changelog
  Verify:
  - rollback smoke test in CI

Sprint 5 Exit:

- Unified runtime is default.
- Compatibility rollback path is validated.

## 14. Operational Rules

1. Every status move in `mcpplusplus/CONFORMANCE_CHECKLIST.md` must include runnable test evidence.
2. No category migration is complete without deterministic `tools_dispatch` tests.
3. No cutover without `MCPP-012`/`MCPP-013`/`MCPP-014`/`MCPP-015` meeting gate thresholds.

## 15. MCP++ Spec-Complete Plan (Chapter-Driven)

This section aligns implementation directly to:

- `docs/spec/mcp++-profiles-draft.md`
- `docs/spec/mcp-idl.md`
- `docs/spec/cid-native-artifacts.md`
- `docs/spec/ucan-delegation.md`
- `docs/spec/temporal-deontic-policy.md`
- `docs/spec/event-dag-ordering.md`
- `docs/spec/risk-scheduling.md`
- `docs/spec/transport-mcp-p2p.md`

### 15.1 Scope and Principles

1. Preserve baseline MCP JSON-RPC semantics; MCP++ is additive by capability/profile negotiation.
2. Implement deterministic canonicalization for all CID-materialized artifacts.
3. Treat transport identity (`PeerID`) and execution authority (`proof_cid`/policy) as separate checks.
4. Require executable conformance evidence for each profile before marking `PASS`.

### 15.2 Profile Delivery Map

| Profile / Chapter | Implementation Scope | Source Inputs | Target Delivery | Primary Tests | Gate |
| --- | --- | --- | --- | --- | --- |
| Profiles registry and negotiation (`mcp++-profiles-draft`) | Advertise/negotiate supported profiles in init metadata | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/server.py` | `ipfs_accelerate_py/mcp_server/server.py`, `ipfs_accelerate_py/mcp_server/tool_metadata.py` | `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (extend) | `MCPP-016` |
| Profile A MCP-IDL (`mcp-idl`) | Interface descriptor canonicalization + repository APIs (`interfaces/list`, `interfaces/get`, `interfaces/compat`, optional `interfaces/select`) | Tool metadata in `ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/*` | New `ipfs_accelerate_py/mcp_server/mcplusplus/idl_registry.py`; new tools under `ipfs_accelerate_py/mcp_server/tools/idl/` | New `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_idl.py` | `MCPP-017` |
| Profile B CID-native artifacts (`cid-native-artifacts`) | Canonical artifact models for intent/decision/receipt/event and CID computation pipelines | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/event_dag.py` | New `ipfs_accelerate_py/mcp_server/mcplusplus/artifacts.py`, `.../canonicalization.py` | New `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_artifacts.py` | `MCPP-018` |
| Profile C UCAN delegation (`ucan-delegation`) | Proof bundle ingestion, validation hooks, execution-time authorization checks | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/ucan_delegation.py`, `.../nl_ucan_policy.py` | New `ipfs_accelerate_py/mcp_server/mcplusplus/delegation.py`; dispatch middleware in `runtime_router.py` | New `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_ucan.py` | `MCPP-019` |
| Profile D temporal deontic policy (`temporal-deontic-policy`) | `policy_cid` model, evaluator runtime, `decision_cid` output, obligations/deadlines | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/temporal_policy.py`, `.../policy_audit_log.py` | New `ipfs_accelerate_py/mcp_server/mcplusplus/policy_engine.py`, `.../policy_audit.py` | New `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_policy.py` | `MCPP-020` |
| Event DAG and ordering (`event-dag-ordering`) | Event node schema with immutable parent links, conflict hooks, replay traversal | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/event_dag.py` | New `ipfs_accelerate_py/mcp_server/mcplusplus/event_dag.py` integration with workflow engine | New `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_event_dag.py` | `MCPP-021` |
| Risk, consensus, scheduling (`risk-scheduling`) | Risk scoring inputs from immutable artifacts, frontier scheduling, optional neighborhood consensus signals | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/risk_scorer.py`, `ipfs_accelerate_py/mcplusplus_module/p2p/workflow.py` | New `ipfs_accelerate_py/mcp_server/mcplusplus/risk_scheduler.py` | New `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_risk_scheduler.py` | `MCPP-022` |
| Profile E `mcp+p2p` (`transport-mcp-p2p`) | Protocol IDs, framing limits, session init semantics, abuse resistance, optional dissemination separation | `ipfs_datasets_py/ipfs_datasets_py/mcp_server/trio_adapter.py`, `ipfs_accelerate_py/mcplusplus_module/p2p/connectivity.py` | `ipfs_accelerate_py/mcp/standalone.py`, `ipfs_accelerate_py/mcp/integration.py`, plus framing helpers in `mcp_server` | Existing and expanded transport tests, including `ipfs_accelerate_py/mcp/tests/test_mcp_transport_p2p_framing_limits.py`, plus `.github/workflows/mcp-transport-libp2p.yml` | `MCPP-013`, `MCPP-023` |

### 15.3 Implementation Phases

Phase 1 (Week 1): Profile negotiation + MCP-IDL baseline

1. Add profile capability advertisement and negotiation contract tests.
2. Implement interface descriptor canonicalization and `interface_cid` generation.
3. Implement `interfaces/list|get|compat` endpoints with deterministic outputs.

Phase 2 (Week 2): CID-native artifact layer

1. Add canonical artifact schemas (`intent`, `decision`, `receipt`, `event`).
2. Add canonicalization and CID utility functions with test vectors.
3. Wire artifact emission to dispatch paths with correlation IDs.

Phase 3 (Week 3): Delegation and policy enforcement

1. Port UCAN delegation verification and proof bundle handling.
2. Port temporal policy evaluator and `decision_cid` generation.
3. Enforce execution-time authz before tool dispatch; emit policy audit logs.

Phase 4 (Week 4): Event DAG and risk scheduler integration

1. Persist event nodes with immutable `parents[]` links.
2. Add replay/rollback traversal APIs and conflict detection hooks.
3. Integrate risk scoring and frontier scheduling with workflow engine.

Phase 5 (Week 5): Tool surface and observability completion

1. Continue category parity migration from `ipfs_datasets_py` source tools.
2. Port observability modules (`monitoring.py`, `otel_tracing.py`, `prometheus_exporter.py`).
3. Add profile-aware metrics (`decision_cid` counts, denial rates, obligation misses).

Phase 6 (Week 6): Cutover and compliance lock

1. Make `ipfs_accelerate_py/mcp_server` the default runtime path.
2. Keep compatibility facade and validate rollback.
3. Require all P0 profile gates to be `PASS` for release.

### 15.4 Conformance Test Plan

1. Add one deterministic unit test module per profile (`MCPP-017`..`MCPP-023`).
2. Add cross-profile integration tests for invocation path:
`intent_cid -> proof_cid -> policy_cid -> decision_cid -> receipt_cid -> event_cid`.
3. Add negative tests:
- invalid canonicalization bytes,
- expired/attenuation-breaking delegation chains,
- policy denial and obligation violation,
- oversized p2p frames and rate-limit enforcement.
4. Keep libp2p lane required for networked transport parity and abuse-resistance regressions.

### 15.5 Release Gates for Full Spec Progress

1. Profile coverage gate:
`MCPP-016` through `MCPP-023` are at least `PARTIAL` before cutover dry-run.
2. Baseline gate:
`MCPP-013` and all non-optional baseline transport checks are `PASS`.
3. Security gate:
UCAN + temporal policy checks are enforced at execution time in integration tests.
4. Provenance gate:
every accepted tool call can emit immutable artifact chain references.
