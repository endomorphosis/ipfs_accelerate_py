# MCP Server Unification Master Plan

## 1. Purpose

Create one canonical runtime in `ipfs_accelerate_py/mcp_server` by:

1. Porting and validating parity from `ipfs_datasets_py/ipfs_datasets_py/mcp_server`.
2. Merging the remaining operational value in `ipfs_accelerate_py/mcplusplus_module` into canonical runtime paths.
3. Completing chapter-level implementation and hardening for MCP++ spec documents referenced by:
   - `https://raw.githubusercontent.com/endomorphosis/Mcp-Plus-Plus/refs/heads/main/docs/index.md`
   - `ipfs_accelerate_py/mcpplusplus/spec/*`

## 2. Canonical Outcomes

1. `ipfs_accelerate_py/mcp_server` is the authoritative runtime implementation.
2. `ipfs_accelerate_py/mcp` becomes a temporary compatibility facade only.
3. `mcplusplus_module` remains as a compatibility shim, not a second runtime.
4. Conformance evidence is tracked only through:
   - `mcpplusplus/CONFORMANCE_CHECKLIST.md`
   - `mcpplusplus/SPEC_GAP_MATRIX.md`

## 3. Baseline (As Of This Plan)

From current in-repo conformance artifacts:

1. Tool category presence parity is reached (`51/51` categories represented).
2. MCP++ checklist items `MCPP-001` through `MCPP-023` are currently marked `PASS`.
3. Main remaining work is not "first implementation" of core spec chapters, but:
   - depth parity inside each category,
   - production hardening,
   - adapter/shim reduction,
   - stricter end-to-end evidence and release gates,
   - clean cutover and deprecation execution.

## 4. Strategic Execution Model

### 4.1 Three Parallel Programs

1. Program A: Source-to-target parity deepening
   - Move from category-level parity to function/schema/behavior parity.
2. Program B: `mcplusplus_module` convergence
   - Convert compatibility paths to canonical-first delegation.
   - Reduce duplicated logic and fallback drift.
3. Program C: Spec chapter hardening
   - Treat each MCP++ chapter as a profile-quality gate with explicit regression evidence.

### 4.2 Non-Negotiable Rules

1. No status changes to `PASS` without deterministic tests.
2. No runtime feature landing without transport and dispatch evidence.
3. No cutover without rollback validation.
4. No direct feature growth in shim layers unless required for compatibility.

## 5. Porting and Merge Workstreams

## W1. Runtime Core and Dispatch Integrity (P0)

Scope:

1. `runtime_router.py`, `hierarchical_tool_manager.py`, `registration_adapter.py`, `tool_registry.py`, `tool_metadata.py`.
2. Uniform native-first registration and conflict handling.
3. Deterministic timeout and metadata behavior for all dispatch paths.

Deliverables:

1. Schema and dispatch consistency checks for all migrated categories.
2. Centralized registration conflict policy (single behavior path).
3. Runtime metrics and structured error envelope consistency.

Exit Criteria:

1. No duplicate registration side effects in bootstrap tests.
2. Deterministic `tools_dispatch` behavior across representative categories.

## W2. Tool Surface Deep Parity (P0/P1)

Scope:

1. Migrate behavior-level parity for each category, not only category presence.
2. Fill schema and argument compatibility deltas category-by-category.

Execution Pattern per Category:

1. Inventory source operations and schemas.
2. Map to existing native implementations.
3. Implement missing operations or compatibility wrappers in `mcp_server/tools/*`.
4. Add deterministic tests for:
   - discovery (`tools_list_tools`, `tools_get_schema`),
   - dispatch behavior,
   - error contracts,
   - edge-case argument handling.

Priority Waves:

1. Wave B1 (P0): `ipfs_tools`, `workflow_tools`, `p2p_tools`, `mcplusplus`, `security_tools`, `monitoring_tools`.
2. Wave B2 (P1): `dataset_tools`, `embedding_tools`, `vector_tools`, `search_tools`, `storage_tools`, `pdf_tools`, `graph_tools`.
3. Wave B3 (P1): remaining long-tail categories (`bespoke_tools`, `legacy_mcp_tools`, `lizard*`, etc.).

Exit Criteria:

1. `SPEC_GAP_MATRIX` rows for each category are either `PASS` or have explicit deferred rationale.

## W3. MCP++ Primitive Runtime Integration (P0)

Scope:

1. Ensure `task_queue`, `workflow_engine/scheduler/dag`, `peer_registry/discovery`, `result_cache` are operationally consumed in canonical dispatch paths.
2. Remove "primitive exists but not used" risk.

Deliverables:

1. Dispatch-to-queue integration checks.
2. Workflow scheduler binding checks.
3. Peer lifecycle hook checks.
4. Cache reuse behavior checks.

Exit Criteria:

1. Integration tests prove runtime consumption of unified services, not just object creation.

## W4. `mcplusplus_module` Convergence and Shim Stabilization (P0)

Scope:

1. Keep `mcplusplus_module` as compatibility shim only.
2. Eliminate duplicated resolver and fallback logic.
3. Standardize optional dependency behavior to explicit stubs across module boundaries.

Deliverables:

1. Canonical-first delegation from shim tools, p2p, and trio surfaces.
2. Shared fallback helper usage (`_missing_dependency_stub`) across shim packages.
3. Contract tests for unavailable dependency behavior.

Exit Criteria:

1. Shim code paths do not implement unique runtime behavior that diverges from canonical runtime.

## W5. Security, Policy, and Governance (P0/P1)

Scope:

1. Policy audit log, secrets vault, risk scorer/risk scheduler behavior hardening.
2. UCAN + temporal policy enforcement reliability and observability.

Deliverables:

1. Strict execution-time enforcement tests (allow/deny/obligation flows).
2. Decision artifact lineage consistency (`decision_cid` correlation).
3. Structured audit evidence for policy and delegation denials.

Exit Criteria:

1. Deterministic security tests cover both positive and negative authorization paths.

## W6. Transport, Reliability, and Abuse Resistance (P0)

Scope:

1. `stdio`, `http`, `trio-p2p`, and `mcp+p2p` parity and hard limits.
2. Handler/client framing and token bucket controls.

Deliverables:

1. Required default lane transport tests.
2. Required libp2p-enabled lane with networked trio-p2p tests.
3. Abuse path assertions (oversized frames, init ordering, unauthorized counters).

Exit Criteria:

1. Transport matrix is green in both lanes with deterministic evidence links.

## W7. Observability and Operational Readiness (P1)

Scope:

1. Monitoring, tracing, exporter parity and consistency.
2. Profile-aware telemetry for MCP++ chapter features.

Deliverables:

1. Stable runtime metrics contract.
2. Trace/span and exporter behavior under optional dependency constraints.
3. Dashboards/alerts tied to dispatch failures and policy denials.

Exit Criteria:

1. Observability tests pass and telemetry payloads are deterministic.

## 6. Chapter-by-Chapter MCP++ Spec Plan

Spec chapters from docs index:

1. `mcp++-profiles-draft.md`
2. `transport-mcp-p2p.md`
3. `mcp-idl.md`
4. `cid-native-artifacts.md`
5. `ucan-delegation.md`
6. `temporal-deontic-policy.md`
7. `event-dag-ordering.md`
8. `risk-scheduling.md`

### 6.1 Profiles (`mcp++-profiles-draft`)

Current posture: Implemented.
Next plan:

1. Validate additive negotiation behavior for all active transports.
2. Add downgrade/unknown-profile compatibility regression cases.
3. Add profile capability snapshot tests to prevent accidental capability drift.

### 6.2 Transport (`transport-mcp-p2p`)

Current posture: Implemented.
Next plan:

1. Expand networked interop tests for mixed-version peers.
2. Add stricter fuzz-style framing boundary tests (decode + rate limits).
3. Harden status counter semantics under sustained abuse scenarios.

### 6.3 MCP-IDL (`mcp-idl`)

Current posture: Implemented.
Next plan:

1. Ensure descriptor generation covers all loaded migrated categories consistently.
2. Add canonicalization hash stability tests across Python/runtime environments.
3. Add compatibility algorithm regression corpus (`interfaces/compat`).

### 6.4 CID Artifacts (`cid-native-artifacts`)

Current posture: Implemented.
Next plan:

1. Add durable backend option validation beyond in-memory storage.
2. Add replayability tests for artifact chain reconstruction.
3. Verify envelope emission policy across all dispatch modes.

### 6.5 UCAN Delegation (`ucan-delegation`)

Current posture: Implemented.
Next plan:

1. Expand cryptographic verification test vectors (`did:key`, signature encodings).
2. Add policy + delegation combined denial matrix (cross-feature interactions).
3. Add explicit proof lineage telemetry for deny/allow outcomes.

### 6.6 Temporal Policy (`temporal-deontic-policy`)

Current posture: Implemented.
Next plan:

1. Expand obligation lifecycle validation (deadline and fulfillment semantics).
2. Add policy version migration tests (`policy_cid` evolution).
3. Validate decision persistence and retrieval parity across transports.

### 6.7 Event DAG (`event-dag-ordering`)

Current posture: Implemented.
Next plan:

1. Expand DAG conflict and fork handling test scenarios.
2. Add deterministic replay and rollback tests for larger graphs.
3. Add durability and snapshot compatibility tests.

### 6.8 Risk Scheduling (`risk-scheduling`)

Current posture: Implemented.
Next plan:

1. Strengthen frontier execution binding tests under load and retries.
2. Add neighborhood consensus signal integration as optional enhancement.
3. Validate risk-state lineage integrity with event/artifact linkage.

## 7. Milestones and Timeline

## Milestone M1 (Weeks 1-2): Convergence Hardening

1. Finish shim fallback/resolver dedup (`mcplusplus_module` convergence).
2. Lock deterministic optional-dependency contract behavior.
3. Add integration assertions proving canonical runtime ownership of behavior.

Exit:

1. No remaining high-risk duplicated runtime logic between shim and canonical paths.

## Milestone M2 (Weeks 3-5): Deep Tool Parity Wave

1. Complete B1 categories (high-impact runtime categories).
2. Add schema parity checks and behavioral regression suites.
3. Update `SPEC_GAP_MATRIX` capability rows with evidence.

Exit:

1. High-impact categories at behavior parity threshold with deterministic tests.

## Milestone M3 (Weeks 6-8): Spec Hardening and Ops

1. Profile chapter hardening pass across 8 chapters.
2. Transport abuse and interop expansion.
3. Observability and security interaction tests.

Exit:

1. Chapter-level hardening test suites green in CI.

## Milestone M4 (Weeks 9-10): Cutover Readiness

1. Validate compatibility facade rollback path.
2. Run release candidate matrix across transport and profile features.
3. Freeze migration deltas and publish cutover checklist.

Exit:

1. Cutover gate approved.

## 8. Cutover and Deprecation Plan

1. Default startup path points to `mcp_server`.
2. Keep `mcp` facade for one release window.
3. Instrument facade usage telemetry.
4. Deprecate shim runtime behavior in phases:
   - Phase D1: warn-only,
   - Phase D2: opt-in only,
   - Phase D3: remove runtime duplication.

## 9. Testing and CI Gate Matrix

Required test classes:

1. Unit tests for each migrated primitive/category module.
2. Integration tests through canonical `tools_dispatch`.
3. Transport matrix tests (`stdio/http/trio-p2p/mcp+p2p`).
4. Process/subprocess startup contract tests.
5. Optional dependency fallback contract tests.
6. Networked libp2p tests in required dedicated CI lane.

Required gates:

1. Gate G1: Conformance docs updated with evidence before merge.
2. Gate G2: Deterministic test evidence for every status change.
3. Gate G3: Transport tests pass in default + libp2p lanes.
4. Gate G4: Cutover only after rollback validation.

## 10. Risk Register and Mitigation

1. Risk: behavior drift between canonical runtime and shim wrappers.
   - Mitigation: canonical-first delegation + shim contract tests.
2. Risk: category presence parity mistaken for full behavior parity.
   - Mitigation: per-category operation/schema parity matrix and tests.
3. Risk: optional dependency breakage causes import-time failures.
   - Mitigation: explicit stubs and smoke import checks.
4. Risk: transport parity passes locally but fails in real p2p lanes.
   - Mitigation: required libp2p CI lane and networked assertions.

## 11. Immediate Execution Backlog (Next 2 Sprints)

Sprint S1:

1. Finish B1 deep parity for `ipfs_tools`, `workflow_tools`, `p2p_tools`, `mcplusplus`.
2. Add integration tests proving `_unified_services` are consumed in dispatch paths.
3. Complete shim convergence remaining fallback surfaces in `bootstrap`/`peer_registry` boundaries where needed.

Sprint S2:

1. Harden chapter interaction tests:
   - UCAN + temporal policy + artifacts + risk scheduler.
2. Expand transport abuse and mixed-peer compatibility tests.
3. Run cutover dry-run with compatibility facade rollback scenario.

## 12. Definition of Done

This initiative is complete when:

1. Canonical runtime ownership is uncontested (`mcp_server` default).
2. Shim layers are compatibility-only and minimal.
3. MCP++ chapter hardening evidence is green and stable in CI.
4. `CONFORMANCE_CHECKLIST` and `SPEC_GAP_MATRIX` remain consistent with executable tests.
5. Rollback path is validated and deprecation milestones are scheduled.

## 13. Execution-Ready Issue Backlog

Use this section as the canonical issue queue for implementation. Issue IDs are ordered by critical path.

### P0 Issues (Critical Path)

1. `UNI-001` Canonical Dispatch Service Consumption
   - Scope: Prove `_unified_services` are consumed on live dispatch paths.
   - Target files: `ipfs_accelerate_py/mcp_server/server.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py`
   - Acceptance:
     - Dispatch path tests assert service factory invocation and usage.
     - No behavior regression in unified bootstrap suite.
   - Depends on: none.

2. `UNI-002` IPFS Tools Deep Parity Wave
   - Scope: Close operation/schema deltas in `ipfs_tools` category.
   - Target files: `ipfs_accelerate_py/mcp_server/tools/ipfs_tools/*`, `ipfs_accelerate_py/mcp/tests/*ipfs*`
   - Acceptance:
     - Representative source operations and schemas mapped and passing.
     - `SPEC_GAP_MATRIX` IPFS rows updated with evidence.
   - Depends on: `UNI-001`.

3. `UNI-003` Workflow Tools Deep Parity Wave
   - Scope: Complete workflow operation parity and schema behavior.
   - Target files: `ipfs_accelerate_py/mcp_server/tools/workflow_tools/*`, `ipfs_accelerate_py/mcp/tests/*workflow*`
   - Acceptance:
     - End-to-end workflow dispatch parity for source-equivalent operations.
     - Deterministic tests for status/submit/next/edge cases.
   - Depends on: `UNI-001`.

4. `UNI-004` P2P and MCP++ Tool Surface Deep Parity
   - Scope: Fill behavior gaps in `p2p_tools` and `mcplusplus` tool category.
   - Target files: `ipfs_accelerate_py/mcp_server/tools/p2p_tools/*`, `ipfs_accelerate_py/mcp_server/tools/mcplusplus/*`, tests under `ipfs_accelerate_py/mcp/tests/`
   - Acceptance:
     - Remote-call/workflow-helper parity validated.
     - Taskqueue/workflow operation coverage expanded.
   - Depends on: `UNI-001`.

5. `UNI-005` Shim Convergence Finalization
   - Scope: Remove remaining duplicated fallback/resolver behavior in `mcplusplus_module` boundaries.
   - Target files: `ipfs_accelerate_py/mcplusplus_module/p2p/bootstrap.py`, `ipfs_accelerate_py/mcplusplus_module/p2p/peer_registry.py`, tests under `ipfs_accelerate_py/mcplusplus_module/tests/`
   - Acceptance:
     - Canonical-first delegation where applicable.
     - Optional dependency boundaries use explicit compatibility contracts.
   - Depends on: none.

6. `UNI-006` Transport Interop and Abuse Regression Expansion
   - Scope: Strengthen `mcp+p2p` framing, init-order, and abuse-path matrix.
   - Target files: `ipfs_accelerate_py/mcp/tests/test_mcp_transport_*`, `ipfs_accelerate_py/p2p_tasks/*`
   - Acceptance:
     - Added regressions for mixed-version peer behavior.
     - Required transport lanes remain green.
   - Depends on: none.

7. `UNI-007` Cutover Dry-Run and Rollback Verification
   - Scope: Exercise runtime default switch and rollback procedure in CI-compatible flow.
   - Target files: `ipfs_accelerate_py/mcp/server.py`, startup scripts/config, tests/workflow docs
   - Acceptance:
     - Dry-run validates canonical path as default.
     - Rollback path tested and documented.
   - Depends on: `UNI-001` to `UNI-006`.

### P1 Issues (Hardening and Breadth)

1. `UNI-101` Security Tools Behavior Parity Expansion
   - Scope: Expand auth/security tools beyond currently implemented baseline.
   - Target files: `ipfs_accelerate_py/mcp_server/tools/security_tools/*`, `.../auth_tools/*`, related tests
   - Acceptance: representative operation and schema parity coverage added.

2. `UNI-102` Monitoring and Observability Behavior Parity Expansion
   - Scope: Expand advanced monitoring/alert/diagnostic operations.
   - Target files: `ipfs_accelerate_py/mcp_server/tools/monitoring_tools/*`, `.../alert_tools/*`, tests
   - Acceptance: advanced telemetry and diagnostics calls validated.

3. `UNI-103` Dataset and Embedding Pipeline Parity Expansion
   - Scope: Expand `dataset_tools` + `embedding_tools` behavior-level parity.
   - Target files: `ipfs_accelerate_py/mcp_server/tools/dataset_tools/*`, `.../embedding_tools/*`, tests
   - Acceptance: source-equivalent conversion and endpoint-management flows validated.

4. `UNI-104` Vector/Search/Storage Integration Parity Expansion
   - Scope: Expand cross-category backend orchestration parity.
   - Target files: `ipfs_accelerate_py/mcp_server/tools/vector_tools/*`, `.../search_tools/*`, `.../storage_tools/*`, tests
   - Acceptance: integration behavior and schema parity for representative flows.

5. `UNI-105` PDF/Graph/Logic Advanced Surface Parity Expansion
   - Scope: Close high-value advanced operations in document/graph/logic categories.
   - Target files: `ipfs_accelerate_py/mcp_server/tools/pdf_tools/*`, `.../graph_tools/*`, `.../logic_tools/*`, tests
   - Acceptance: advanced ingestion/query/reasoning flows covered.

### Spec Chapter Hardening Issues

1. `SPEC-201` Profiles Negotiation Hardening
   - Chapter: `mcp++-profiles-draft.md`
   - Acceptance: downgrade/unknown-profile regression cases and capability snapshot checks.

2. `SPEC-202` MCP-IDL Stability Corpus
   - Chapter: `mcp-idl.md`
   - Acceptance: canonicalization stability corpus and compat matching regression suite.

3. `SPEC-203` Artifact Durability and Replay
   - Chapter: `cid-native-artifacts.md`
   - Acceptance: durable store path and replay integrity tests beyond in-memory behavior.

4. `SPEC-204` UCAN Verification Vector Expansion
   - Chapter: `ucan-delegation.md`
   - Acceptance: expanded signature/caveat/proof-link vectors and deny/allow telemetry checks.

5. `SPEC-205` Temporal Policy Lifecycle Hardening
   - Chapter: `temporal-deontic-policy.md`
   - Acceptance: obligation/deadline/version migration tests and transport parity assertions.

6. `SPEC-206` Event DAG Scale and Conflict Scenarios
   - Chapter: `event-dag-ordering.md`
   - Acceptance: large-DAG replay/rollback and fork/conflict handling tests.

7. `SPEC-207` Risk Frontier + Consensus Enhancements
   - Chapter: `risk-scheduling.md`
   - Acceptance: load/retry frontier binding tests and optional consensus signal integration tests.

8. `SPEC-208` mcp+p2p Mixed-Version Interop Matrix
   - Chapter: `transport-mcp-p2p.md`
   - Acceptance: mixed-version interop and abuse-resistance matrix incorporated into CI evidence.

### Issue Lifecycle Rules

1. Every issue closure must update evidence links in `mcpplusplus/SPEC_GAP_MATRIX.md`.
2. Any requirement status change must also update `mcpplusplus/CONFORMANCE_CHECKLIST.md`.
3. No issue closes without deterministic tests in `ipfs_accelerate_py/mcp/tests/` or `ipfs_accelerate_py/mcplusplus_module/tests/`.

## 14. GitHub Issue Templates (P0 Ready)

Copy/paste the following blocks as GitHub issues. Keep labels aligned with your repository conventions.

### `UNI-001` Canonical Dispatch Service Consumption

Title: `UNI-001: Prove canonical dispatch consumes _unified_services`

Body:

```markdown
## Summary
Prove that canonical dispatch paths in `ipfs_accelerate_py/mcp_server/server.py` actively consume `_unified_services` factories rather than only constructing them at bootstrap.

## Scope
- Add/extend integration assertions that dispatch invokes queue/workflow/peer/cache service paths.
- Keep behavior deterministic and dependency-light.

## Target Files
- `ipfs_accelerate_py/mcp_server/server.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py`

## Acceptance Criteria
- [ ] Tests assert service factory invocation and runtime usage.
- [ ] No regressions in unified bootstrap tests.
- [ ] Evidence references added to `mcpplusplus/SPEC_GAP_MATRIX.md`.

## Verification
- `python3 -m unittest ipfs_accelerate_py.mcp.tests.test_mcp_server_unified_bootstrap`
```

### `UNI-002` IPFS Tools Deep Parity Wave

Title: `UNI-002: Deep parity for native ipfs_tools`

Body:

```markdown
## Summary
Close operation-level and schema-level parity deltas for `ipfs_tools` between source and canonical runtime.

## Scope
- Map source operations to canonical implementations.
- Add missing operations/schemas or compatibility behavior.
- Add deterministic dispatch and schema tests.

## Target Files
- `ipfs_accelerate_py/mcp_server/tools/ipfs_tools/*`
- `ipfs_accelerate_py/mcp/tests/*ipfs*`

## Acceptance Criteria
- [ ] Representative source operations are implemented or explicitly deferred with rationale.
- [ ] `tools_get_schema` parity validated for representative tools.
- [ ] `SPEC_GAP_MATRIX` IPFS rows updated with evidence.

## Verification
- `python3 -m unittest <ipfs parity test modules>`
```

### `UNI-003` Workflow Tools Deep Parity Wave

Title: `UNI-003: Deep parity for native workflow_tools`

Body:

```markdown
## Summary
Complete behavior and schema parity for workflow operations in canonical runtime.

## Scope
- Expand workflow status/submit/next and edge-case behavior parity.
- Validate deterministic schema and dispatch behavior.

## Target Files
- `ipfs_accelerate_py/mcp_server/tools/workflow_tools/*`
- `ipfs_accelerate_py/mcp/tests/*workflow*`

## Acceptance Criteria
- [ ] Workflow dispatch parity for representative source operations.
- [ ] Edge-case tests for arguments and error envelopes.
- [ ] Evidence links updated in conformance docs.

## Verification
- `python3 -m unittest <workflow parity test modules>`
```

### `UNI-004` P2P and MCP++ Tool Surface Deep Parity

Title: `UNI-004: Deep parity for p2p_tools and mcplusplus tools`

Body:

```markdown
## Summary
Close high-impact behavior gaps in `p2p_tools` and `mcplusplus` category surfaces.

## Scope
- Complete remote-call/workflow-helper parity.
- Expand taskqueue/workflow operation coverage in canonical runtime.

## Target Files
- `ipfs_accelerate_py/mcp_server/tools/p2p_tools/*`
- `ipfs_accelerate_py/mcp_server/tools/mcplusplus/*`
- `ipfs_accelerate_py/mcp/tests/*p2p*`

## Acceptance Criteria
- [ ] Representative source-equivalent p2p operations dispatch correctly.
- [ ] MCP++ tool wrappers expose expected schemas and behavior.
- [ ] Matrix/checklist evidence updated.

## Verification
- `python3 -m unittest <p2p and mcplusplus parity test modules>`
```

### `UNI-005` Shim Convergence Finalization

Title: `UNI-005: Finalize mcplusplus_module shim convergence`

Body:

```markdown
## Summary
Finish convergence of `mcplusplus_module` so it remains compatibility-only and delegates canonical behavior.

## Scope
- Remove remaining duplicated fallback/resolver behavior in shim boundaries.
- Standardize optional dependency contracts to explicit stubs.

## Target Files
- `ipfs_accelerate_py/mcplusplus_module/p2p/bootstrap.py`
- `ipfs_accelerate_py/mcplusplus_module/p2p/peer_registry.py`
- `ipfs_accelerate_py/mcplusplus_module/tests/*`

## Acceptance Criteria
- [ ] Canonical-first delegation or explicit compatibility rationale in all touched boundaries.
- [ ] No raw `None` export surfaces at module boundaries where explicit stubs are required.
- [ ] Contract tests updated and passing.

## Verification
- `python3 -m unittest ipfs_accelerate_py.mcplusplus_module.tests.test_tool_adapters ipfs_accelerate_py.mcplusplus_module.tests.test_trio_server`
```

### `UNI-006` Transport Interop and Abuse Regression Expansion

Title: `UNI-006: Expand mcp+p2p interop and abuse regression matrix`

Body:

```markdown
## Summary
Strengthen transport interop and abuse-resistance evidence for `mcp+p2p`.

## Scope
- Add mixed-version interop scenarios.
- Expand framing/limit/init-order abuse tests.
- Keep default and libp2p CI lanes green.

## Target Files
- `ipfs_accelerate_py/mcp/tests/test_mcp_transport_*`
- `ipfs_accelerate_py/p2p_tasks/*`
- `.github/workflows/mcp-transport-libp2p.yml`

## Acceptance Criteria
- [ ] Additional regression tests merged for interop/abuse scenarios.
- [ ] Default transport lane remains green.
- [ ] libp2p lane remains green and required.

## Verification
- `python3 -m unittest ipfs_accelerate_py.mcp.tests.test_mcp_transport_p2p_framing_limits`
- `python3 -m unittest ipfs_accelerate_py.mcp.tests.test_mcp_transport_mcp_p2p_handler_limits`
- `python3 -m unittest ipfs_accelerate_py.mcp.tests.test_mcp_transport_mcp_p2p_client_limits`
```

### `UNI-007` Cutover Dry-Run and Rollback Verification

Title: `UNI-007: Execute canonical runtime cutover dry-run and rollback`

Body:

```markdown
## Summary
Run a release-grade dry-run where canonical runtime is default, then verify rollback path.

## Scope
- Validate startup and dispatch through `mcp_server` default path.
- Verify compatibility facade rollback behavior.

## Target Files
- `ipfs_accelerate_py/mcp/server.py`
- `ipfs_accelerate_py/mcp_server/server.py`
- startup scripts/config and CI docs

## Acceptance Criteria
- [ ] Canonical default path validated in dry-run.
- [ ] Rollback scenario tested and documented.
- [ ] Conformance docs updated with cutover evidence.

## Verification
- `python3 -m unittest <cutover/rollback regression modules>`
```
