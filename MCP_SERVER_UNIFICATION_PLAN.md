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

## 15. GitHub Issue Templates (Spec Chapters)

### `SPEC-201` Profiles Negotiation Hardening

Title: `SPEC-201: Harden MCP++ profile negotiation and downgrade behavior`

Body:

```markdown
## Summary
Harden profile negotiation behavior for additive compatibility, downgrade behavior, and unknown-profile handling.

## Chapter
- `mcp++-profiles-draft.md`

## Scope
- Add regression tests for requested supported profile selection.
- Add tests for unknown/unsupported profile fallbacks.
- Add capability snapshot tests to prevent accidental profile drift.

## Target Files
- `ipfs_accelerate_py/mcp_server/server.py`
- `ipfs_accelerate_py/p2p_tasks/mcp_p2p.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py`

## Acceptance Criteria
- [ ] Negotiation behavior remains additive and backward compatible.
- [ ] Unknown profiles produce deterministic fallback.
- [ ] Conformance evidence updated.

## Verification
- `python3 -m unittest ipfs_accelerate_py.mcp.tests.test_mcp_server_unified_bootstrap`
```

### `SPEC-202` MCP-IDL Stability Corpus

Title: `SPEC-202: Add MCP-IDL canonicalization and compat stability corpus`

Body:

```markdown
## Summary
Expand deterministic MCP-IDL descriptor and compatibility tests to lock cross-runtime stability.

## Chapter
- `mcp-idl.md`

## Scope
- Add canonicalization stability corpus across representative descriptors.
- Expand `interfaces/compat` regression cases.
- Validate descriptor generation coverage for migrated categories.

## Target Files
- `ipfs_accelerate_py/mcp_server/mcplusplus/idl_registry.py`
- `ipfs_accelerate_py/mcp_server/tools/idl/native_idl_tools.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_idl.py`

## Acceptance Criteria
- [ ] Descriptor CIDs are deterministic for corpus inputs.
- [ ] Compatibility matching remains stable under case/whitespace/version variants.
- [ ] Evidence updated in matrix/checklist.

## Verification
- `python3 -m unittest ipfs_accelerate_py.mcp.tests.test_mcp_server_mcplusplus_idl`
```

### `SPEC-203` Artifact Durability and Replay

Title: `SPEC-203: Harden CID artifact durability and replay integrity`

Body:

```markdown
## Summary
Extend CID-native artifact pipeline validation beyond in-memory behavior into durable/replayable paths.

## Chapter
- `cid-native-artifacts.md`

## Scope
- Add durable storage backend validation.
- Add replay/reconstruction tests for artifact chains.
- Validate artifact emission policy consistency across dispatch modes.

## Target Files
- `ipfs_accelerate_py/mcp_server/mcplusplus/artifacts.py`
- `ipfs_accelerate_py/mcp_server/server.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_artifacts.py`

## Acceptance Criteria
- [ ] Durable artifact retrieval passes deterministic integrity checks.
- [ ] Replay reconstructs chain correctly.
- [ ] Emission policy is deterministic across modes.

## Verification
- `python3 -m unittest ipfs_accelerate_py.mcp.tests.test_mcp_server_mcplusplus_artifacts`
```

### `SPEC-204` UCAN Verification Vector Expansion

Title: `SPEC-204: Expand UCAN verification vectors and deny matrix`

Body:

```markdown
## Summary
Broaden UCAN verification coverage for signature formats, caveats, proof linkage, and negative execution-time authorization paths.

## Chapter
- `ucan-delegation.md`

## Scope
- Add signature vector cases (`ed25519`, hex variants, `did:key`).
- Expand caveat and context-CID matrix.
- Add combined policy+delegation denial interaction tests.

## Target Files
- `ipfs_accelerate_py/mcp_server/mcplusplus/delegation.py`
- `ipfs_accelerate_py/mcp_server/server.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_ucan.py`

## Acceptance Criteria
- [ ] Expanded positive and negative UCAN vectors pass deterministically.
- [ ] Denial behavior is explicit and auditable.
- [ ] Conformance evidence updated.

## Verification
- `python3 -m unittest ipfs_accelerate_py.mcp.tests.test_mcp_server_mcplusplus_ucan`
```

### `SPEC-205` Temporal Policy Lifecycle Hardening

Title: `SPEC-205: Harden temporal policy obligations and decision lifecycle`

Body:

```markdown
## Summary
Strengthen temporal policy lifecycle semantics for obligations, deadlines, and policy evolution behavior.

## Chapter
- `temporal-deontic-policy.md`

## Scope
- Add obligation/deadline progression tests.
- Add policy version migration tests.
- Validate decision persistence parity across transport paths.

## Target Files
- `ipfs_accelerate_py/mcp_server/mcplusplus/policy_engine.py`
- `ipfs_accelerate_py/mcp_server/server.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_policy.py`

## Acceptance Criteria
- [ ] Obligation lifecycle behavior is deterministic.
- [ ] Decision CID persistence remains stable across dispatch modes/transports.
- [ ] Evidence links updated.

## Verification
- `python3 -m unittest ipfs_accelerate_py.mcp.tests.test_mcp_server_mcplusplus_policy`
```

### `SPEC-206` Event DAG Scale and Conflict Scenarios

Title: `SPEC-206: Expand Event DAG replay, scale, and conflict coverage`

Body:

```markdown
## Summary
Expand Event DAG tests for larger graphs, replay/rollback correctness, and fork/conflict handling behavior.

## Chapter
- `event-dag-ordering.md`

## Scope
- Add large DAG fixture traversal tests.
- Add fork/conflict scenario validation.
- Validate snapshot export/import compatibility.

## Target Files
- `ipfs_accelerate_py/mcp_server/mcplusplus/event_dag.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_event_dag.py`

## Acceptance Criteria
- [ ] Replay and rollback paths are deterministic for larger graphs.
- [ ] Conflict behavior is explicit and tested.
- [ ] Snapshot compatibility preserved.

## Verification
- `python3 -m unittest ipfs_accelerate_py.mcp.tests.test_mcp_server_mcplusplus_event_dag`
```

### `SPEC-207` Risk Frontier and Consensus Enhancements

Title: `SPEC-207: Harden risk frontier execution and optional consensus signals`

Body:

```markdown
## Summary
Strengthen risk frontier execution under load/retry patterns and add optional consensus signal integration tests.

## Chapter
- `risk-scheduling.md`

## Scope
- Add load/retry frontier binding tests.
- Add optional neighborhood consensus signal integration checks.
- Validate risk lineage consistency with event/artifact links.

## Target Files
- `ipfs_accelerate_py/mcp_server/mcplusplus/risk_scheduler.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_risk_scheduler.py`

## Acceptance Criteria
- [ ] Frontier execution binding deterministic under load/retries.
- [ ] Consensus signal path remains optional and non-breaking.
- [ ] Evidence updated in conformance docs.

## Verification
- `python3 -m unittest ipfs_accelerate_py.mcp.tests.test_mcp_server_mcplusplus_risk_scheduler`
```

### `SPEC-208` mcp+p2p Mixed-Version Interop Matrix

Title: `SPEC-208: Expand mcp+p2p mixed-version and abuse-resistance interop matrix`

Body:

```markdown
## Summary
Expand `mcp+p2p` interop and abuse-resistance regression matrix to include mixed-version and advanced edge conditions.

## Chapter
- `transport-mcp-p2p.md`

## Scope
- Add mixed-version handshake and call-flow coverage.
- Expand abuse counters and malformed frame handling assertions.
- Keep default + libp2p transport lanes stable.

## Target Files
- `ipfs_accelerate_py/p2p_tasks/mcp_p2p.py`
- `ipfs_accelerate_py/p2p_tasks/mcp_p2p_client.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_transport_*`
- `.github/workflows/mcp-transport-libp2p.yml`

## Acceptance Criteria
- [ ] Mixed-version matrix behavior is deterministic.
- [ ] Abuse-resistance assertions expanded and passing.
- [ ] CI evidence remains green in required lanes.

## Verification
- `python3 -m unittest ipfs_accelerate_py.mcp.tests.test_mcp_transport_p2p_framing_limits`
- `python3 -m unittest ipfs_accelerate_py.mcp.tests.test_mcp_transport_mcp_p2p_handler_limits`
- `python3 -m unittest ipfs_accelerate_py.mcp.tests.test_mcp_transport_mcp_p2p_client_limits`
```

## 16. GitHub Issue Templates (P1 Ready)

### `UNI-101` Security Tools Behavior Parity Expansion

Title: `UNI-101: Expand auth/security behavior parity in canonical runtime`

Body:

```markdown
## Summary
Expand behavior-level parity in `security_tools` and `auth_tools` beyond baseline parity.

## Scope
- Add representative source-equivalent operations not yet behavior-complete.
- Validate schema parity for newly covered operations.
- Expand deterministic deny/allow behavior tests.

## Target Files
- `ipfs_accelerate_py/mcp_server/tools/security_tools/*`
- `ipfs_accelerate_py/mcp_server/tools/auth_tools/*`
- `ipfs_accelerate_py/mcp/tests/*security*`

## Acceptance Criteria
- [ ] Representative missing operations implemented or deferred with rationale.
- [ ] `tools_get_schema` and dispatch parity tests added.
- [ ] Evidence links updated in conformance docs.

## Verification
- `python3 -m unittest <security and auth parity test modules>`
```

### `UNI-102` Monitoring and Observability Behavior Parity Expansion

Title: `UNI-102: Expand monitoring/alert observability behavior parity`

Body:

```markdown
## Summary
Deepen parity for advanced monitoring, diagnostics, and alert operations.

## Scope
- Expand diagnostic and alerting behavior parity against source.
- Validate observability payload stability and compatibility.

## Target Files
- `ipfs_accelerate_py/mcp_server/tools/monitoring_tools/*`
- `ipfs_accelerate_py/mcp_server/tools/alert_tools/*`
- `ipfs_accelerate_py/mcp/tests/*monitor*`

## Acceptance Criteria
- [ ] Advanced monitoring operations behave deterministically.
- [ ] Alert/diagnostic schemas and outputs are parity-validated.
- [ ] Matrix/checklist evidence updated.

## Verification
- `python3 -m unittest <monitoring and alert parity test modules>`
```

### `UNI-103` Dataset and Embedding Pipeline Parity Expansion

Title: `UNI-103: Expand dataset and embedding pipeline behavior parity`

Body:

```markdown
## Summary
Expand behavior-level parity for dataset and embedding operations, including advanced conversion and endpoint flows.

## Scope
- Add missing source-equivalent dataset operations.
- Add missing embedding endpoint/management operations.
- Validate schema and dispatch behavior parity.

## Target Files
- `ipfs_accelerate_py/mcp_server/tools/dataset_tools/*`
- `ipfs_accelerate_py/mcp_server/tools/embedding_tools/*`
- `ipfs_accelerate_py/mcp/tests/*dataset*`
- `ipfs_accelerate_py/mcp/tests/*embedding*`

## Acceptance Criteria
- [ ] Representative dataset and embedding deltas closed.
- [ ] Edge-case arguments and error envelopes parity-tested.
- [ ] Evidence links updated in conformance artifacts.

## Verification
- `python3 -m unittest <dataset and embedding parity test modules>`
```

### `UNI-104` Vector/Search/Storage Integration Parity Expansion

Title: `UNI-104: Expand vector-search-storage integration parity`

Body:

```markdown
## Summary
Close cross-category behavior gaps for vector, search, and storage orchestration.

## Scope
- Expand integration behavior parity for index/search/storage workflows.
- Validate schema compatibility and dispatch invariants across categories.

## Target Files
- `ipfs_accelerate_py/mcp_server/tools/vector_tools/*`
- `ipfs_accelerate_py/mcp_server/tools/search_tools/*`
- `ipfs_accelerate_py/mcp_server/tools/storage_tools/*`
- `ipfs_accelerate_py/mcp/tests/*vector*`
- `ipfs_accelerate_py/mcp/tests/*search*`
- `ipfs_accelerate_py/mcp/tests/*storage*`

## Acceptance Criteria
- [ ] Representative integration flows are parity-validated.
- [ ] Deterministic schema and result contracts are asserted.
- [ ] Matrix rows/evidence are updated.

## Verification
- `python3 -m unittest <vector search storage parity test modules>`
```

### `UNI-105` PDF/Graph/Logic Advanced Surface Parity Expansion

Title: `UNI-105: Expand PDF/Graph/Logic advanced behavior parity`

Body:

```markdown
## Summary
Deepen parity for advanced document, graph, and logic operations.

## Scope
- Add advanced ingestion/query/reasoning behavior parity coverage.
- Validate schema compatibility and deterministic output behavior.

## Target Files
- `ipfs_accelerate_py/mcp_server/tools/pdf_tools/*`
- `ipfs_accelerate_py/mcp_server/tools/graph_tools/*`
- `ipfs_accelerate_py/mcp_server/tools/logic_tools/*`
- `ipfs_accelerate_py/mcp/tests/*pdf*`
- `ipfs_accelerate_py/mcp/tests/*graph*`
- `ipfs_accelerate_py/mcp/tests/*logic*`

## Acceptance Criteria
- [ ] Advanced operations implemented or deferred with rationale.
- [ ] Deterministic behavior and schema parity tests added.
- [ ] Evidence updated in conformance docs.

## Verification
- `python3 -m unittest <pdf graph logic parity test modules>`
```

## 17. Execution Sequence and Gate Mapping

This section defines the practical run order and ties each issue to conformance gates.

### 17.1 Critical Path Order (Recommended)

1. `UNI-001` canonical dispatch service consumption
2. `UNI-005` shim convergence finalization
3. `UNI-002` IPFS deep parity
4. `UNI-003` workflow deep parity
5. `UNI-004` p2p + mcplusplus deep parity
6. `UNI-006` transport interop/abuse expansion
7. `SPEC-201` through `SPEC-208` hardening passes
8. `UNI-007` cutover dry-run and rollback verification
9. `UNI-101` through `UNI-105` breadth hardening waves

### 17.2 Issue to Conformance IDs

| Issue | Primary Conformance Targets | Secondary Targets |
| --- | --- | --- |
| `UNI-001` | `MCPP-005`, `MCPP-007` | `MCPP-022` |
| `UNI-002` | `MCPP-012` | `MCPP-007` |
| `UNI-003` | `MCPP-012` | `MCPP-022` |
| `UNI-004` | `MCPP-012`, `MCPP-023` | `MCPP-008`, `MCPP-009` |
| `UNI-005` | `MCPP-001` | `MCPP-013` |
| `UNI-006` | `MCPP-013`, `MCPP-023` | `MCPP-016` |
| `UNI-007` | cutover gate (`MCPP-012`, `MCPP-013`, `MCPP-014`, `MCPP-015`) | `MCPP-016`..`MCPP-023` |
| `UNI-101` | `MCPP-014` | `MCPP-019`, `MCPP-020` |
| `UNI-102` | `MCPP-015` | `MCPP-013` |
| `UNI-103` | `MCPP-012` | `MCPP-017` |
| `UNI-104` | `MCPP-012` | `MCPP-015` |
| `UNI-105` | `MCPP-012` | `MCPP-017`, `MCPP-020` |
| `SPEC-201` | `MCPP-016` | `MCPP-013` |
| `SPEC-202` | `MCPP-017` | `MCPP-012` |
| `SPEC-203` | `MCPP-018` | `MCPP-021` |
| `SPEC-204` | `MCPP-019` | `MCPP-014` |
| `SPEC-205` | `MCPP-020` | `MCPP-014` |
| `SPEC-206` | `MCPP-021` | `MCPP-018` |
| `SPEC-207` | `MCPP-022` | `MCPP-023` |
| `SPEC-208` | `MCPP-023` | `MCPP-013`, `MCPP-016` |

### 17.3 Merge Gating Checklist (Per PR)

1. Implementation links added in PR description.
2. Deterministic test evidence attached (module-level command outputs).
3. `mcpplusplus/SPEC_GAP_MATRIX.md` updated if capability status changed.
4. `mcpplusplus/CONFORMANCE_CHECKLIST.md` updated if requirement status changed.
5. Transport lane impact assessed (`default` and `libp2p` lanes).

## 18. Operating Cadence (Weekly)

1. Monday: select 1 P0 + 1 spec hardening issue; lock test scope.
2. Midweek: merge implementation + tests with evidence updates.
3. Friday: run transport matrix smoke and conformance doc consistency review.
4. End of week: publish delta summary listing closed issues, moved conformance IDs, and residual risks.

## 19. First 30 Days Kickoff Playbook

This section defines a practical startup sequence for executing the plan immediately.

### 19.1 Day 1-3 (Setup and Baseline Lock)

1. Open issues from templates in this order:
   - `UNI-001`, `UNI-005`, `UNI-002`, `UNI-003`, `UNI-004`, `UNI-006`, `UNI-007`
   - `SPEC-201` through `SPEC-208`
2. Assign initial milestones:
   - `M1-Convergence`, `M2-DeepParity`, `M3-SpecHardening`, `M4-Cutover`
3. Freeze baseline references in issue bodies:
   - target conformance IDs from section 17.2
   - current matrix/checklist rows used as baseline evidence

### 19.2 Day 4-10 (Critical Path Start)

1. Execute `UNI-001` and `UNI-005` in parallel if staffing permits.
2. Require every merged PR to include:
   - test command outputs,
   - changed conformance references,
   - risk note if transport or policy paths were touched.
3. Keep `SPEC-201` and `SPEC-202` in progress as the first chapter-hardening pair.

### 19.3 Day 11-20 (Deep Parity Wave)

1. Execute `UNI-002`, `UNI-003`, and `UNI-004` in sequence.
2. For each category wave, publish a parity mini-report in PR description:
   - operations covered,
   - schemas covered,
   - deferred items with rationale.
3. Expand chapter hardening with `SPEC-203` to `SPEC-206`.

### 19.4 Day 21-30 (Transport + Cutover Prep)

1. Execute `UNI-006` to harden transport interop and abuse matrix.
2. Execute `SPEC-207` and `SPEC-208`.
3. Run `UNI-007` cutover dry-run checklist:
   - canonical default startup check,
   - compatibility facade rollback check,
   - conformance evidence synchronization check.

## 20. Milestones, Labels, and PR Taxonomy

Use consistent labels and milestone names so issue state is queryable.

### 20.1 Recommended Milestones

1. `M1-Convergence-Hardening`
2. `M2-Deep-Tool-Parity`
3. `M3-Spec-Chapter-Hardening`
4. `M4-Cutover-Rollback-Validation`

### 20.2 Recommended Labels

1. Priority labels:
   - `priority:p0`
   - `priority:p1`
2. Type labels:
   - `type:parity`
   - `type:spec`
   - `type:transport`
   - `type:security`
   - `type:observability`
   - `type:cutover`
3. Status labels:
   - `status:blocked`
   - `status:needs-evidence`
   - `status:ready-for-review`

### 20.3 PR Title and Evidence Format

PR title format:

1. `[UNI-001] Prove canonical dispatch consumes unified services`
2. `[SPEC-204] Expand UCAN verification vectors and deny matrix`

PR checklist format:

1. Linked issue ID.
2. Conformance IDs touched (`MCPP-*`).
3. Deterministic test commands and outcomes.
4. Matrix/checklist doc updates completed.
5. Transport impact assessed.

## 21. Weekly Review Template

Use this template each Friday for project-level status reviews.

1. Closed this week:
   - issue IDs
   - conformance IDs moved
2. In progress:
   - issue IDs
   - current blockers
3. Evidence quality:
   - missing tests
   - missing doc updates
4. Transport/security risk summary:
   - new risks
   - mitigations planned
5. Next week commitments:
   - exactly 1-2 P0 items
   - exactly 1 spec hardening item

## 22. Execution Appendix (GitHub CLI)

Use these commands to operationalize this plan in GitHub.

### 22.1 Repository Variables

```bash
export REPO="endomorphosis/ipfs_accelerate_py"
```

### 22.2 Create Labels (One-Time)

```bash
gh label create "priority:p0" --repo "$REPO" --color B60205 --description "Critical path"
gh label create "priority:p1" --repo "$REPO" --color D93F0B --description "Important but non-critical"
gh label create "type:parity" --repo "$REPO" --color 0E8A16 --description "Source-to-target parity"
gh label create "type:spec" --repo "$REPO" --color 5319E7 --description "MCP++ spec hardening"
gh label create "type:transport" --repo "$REPO" --color 1D76DB --description "Transport and protocol behavior"
gh label create "type:security" --repo "$REPO" --color E99695 --description "Security/policy/auth"
gh label create "type:observability" --repo "$REPO" --color C2E0C6 --description "Monitoring/tracing/exporters"
gh label create "type:cutover" --repo "$REPO" --color F9D0C4 --description "Runtime cutover and rollback"
gh label create "status:blocked" --repo "$REPO" --color 000000 --description "Blocked work"
gh label create "status:needs-evidence" --repo "$REPO" --color FBCA04 --description "Missing conformance evidence"
gh label create "status:ready-for-review" --repo "$REPO" --color 006B75 --description "Ready for reviewer"
```

### 22.3 Create Milestones (One-Time)

```bash
gh api -X POST "repos/$REPO/milestones" -f title="M1-Convergence-Hardening"
gh api -X POST "repos/$REPO/milestones" -f title="M2-Deep-Tool-Parity"
gh api -X POST "repos/$REPO/milestones" -f title="M3-Spec-Chapter-Hardening"
gh api -X POST "repos/$REPO/milestones" -f title="M4-Cutover-Rollback-Validation"
```

### 22.4 Create Issues from Templates

For each template section in this file:

1. Copy issue `Title` and markdown `Body`.
2. Create issue:

```bash
gh issue create --repo "$REPO" --title "<TITLE>" --body-file <BODY_FILE.md> --label "priority:p0,type:parity"
```

Suggested label mappings:

1. `UNI-001` to `UNI-007`: `priority:p0` + one of `type:parity|type:transport|type:cutover`.
2. `SPEC-201` to `SPEC-208`: `priority:p0` + `type:spec` (+ optional `type:transport`/`type:security`).
3. `UNI-101` to `UNI-105`: `priority:p1` + `type:parity` (+ optional domain label).

### 22.5 Query Active Critical Path

```bash
gh issue list --repo "$REPO" --search "is:open label:priority:p0" --limit 200
gh issue list --repo "$REPO" --search "is:open label:type:spec label:priority:p0" --limit 200
```

### 22.6 Weekly Review Export

```bash
gh issue list --repo "$REPO" --search "closed:>=2026-01-01" --limit 200 --json number,title,labels,closedAt
gh issue list --repo "$REPO" --search "is:open label:status:blocked" --limit 200 --json number,title,labels
```

### 22.7 PR Evidence Checklist (Copy Into PR Description)

```markdown
- Linked issue ID(s):
- Conformance IDs touched (`MCPP-*`):
- Deterministic test command output attached:
- `SPEC_GAP_MATRIX.md` updated (if capability status changed):
- `CONFORMANCE_CHECKLIST.md` updated (if requirement status changed):
- Transport impact assessed (default/libp2p lanes):
```
