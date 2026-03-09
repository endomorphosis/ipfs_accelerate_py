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

### 5.0 Comprehensive Port-and-Merge Blueprint (Source -> Canonical)

This section is the concrete migration blueprint for porting
`ipfs_datasets_py/ipfs_datasets_py/mcp_server` (sometimes referenced as `mcp_serve`)
into `ipfs_accelerate_py/mcp_server`, while folding in the remaining runtime value from
`ipfs_accelerate_py/mcplusplus_module`.

#### 5.0.1 Scope Boundaries

1. Canonical runtime destination: `ipfs_accelerate_py/mcp_server/*`.
2. Source parity baseline: `ipfs_datasets_py/ipfs_datasets_py/mcp_server/*`.
3. Compatibility-only shim source: `ipfs_accelerate_py/mcplusplus_module/*`.
4. Spec acceptance source: `ipfs_accelerate_py/mcpplusplus/spec/*`.

#### 5.0.2 Porting Architecture Contract

1. All behavior lands in canonical modules under `ipfs_accelerate_py/mcp_server`.
2. `mcplusplus_module` may delegate, adapt, or stub, but must not become feature-authoritative.
3. Any source feature with no canonical analog must first land in canonical runtime, then be wired via shim delegation.
4. Existing canonical-first resolver patterns remain mandatory for new shim integration points.

#### 5.0.3 Directory-Level Migration Map

| Source (`ipfs_datasets_py/.../mcp_server`) | Canonical Target (`ipfs_accelerate_py/mcp_server`) | Merge Rule | Priority |
| --- | --- | --- | --- |
| `server.py`, `runtime_router.py`, `tool_registry.py`, `registration_adapter.py`, `hierarchical_tool_manager.py` | same top-level runtime modules | Reconcile behavior into one dispatch path; remove duplicate decision branches | P0 |
| `tools/*` (51 categories) | `tools/*` native category modules | Keep category parity and close operation/schema deltas with deterministic tests | P0/P1 |
| `mcplusplus/*` primitives | `mcplusplus/*` primitives | Ensure dispatch consumption and transport integration, not only primitive existence | P0 |
| `policy_audit_log.py`, `secrets_vault.py`, `risk_scorer.py` | same top-level modules | Preserve deterministic policy/risk behavior and decision lineage contracts | P0 |
| `monitoring.py`, `otel_tracing.py`, `prometheus_exporter.py` | same top-level modules | Preserve optional dependency safety and payload stability | P1 |
| transport entrypoints (`standalone_server.py`, trio/http adapters) | `mcp_server` + `mcp/standalone.py` + `p2p_tasks/*` integration boundary | Keep parity across stdio/http/trio-p2p/mcp+p2p with required CI lanes | P0 |

#### 5.0.4 `mcplusplus_module` Feature Merge Map

| Shim Area (`ipfs_accelerate_py/mcplusplus_module`) | Canonical Ownership Target | Required End State |
| --- | --- | --- |
| `tools/*` adapters | `ipfs_accelerate_py/mcp_server/tools/*` | shim tools become thin registrars/delegators only |
| `p2p/*` bootstrap/peer/workflow/taskqueue helpers | `ipfs_accelerate_py/mcp_server/mcplusplus/*` and `ipfs_accelerate_py/mcp_server/tools/p2p_tools/*` | no unique scheduler/queue/peer behavior remains in shim |
| `trio/*` server and resolver surfaces | canonical runtime registration and transport helpers | shared resolver path and explicit compatibility stubs |
| top-level fallback utilities | centralized helper functions | no duplicated optional-dependency/fallback logic across shim subpackages |

#### 5.0.5 Four-Phase Delivery Sequence

1. Phase A: Inventory and mapping lock
   - Generate operation/schema diff per category (source -> canonical).
   - Classify each delta: implement now, delegate, or explicitly defer.
2. Phase B: Canonical feature landing
   - Implement missing behavior in canonical runtime only.
   - Add discovery/schema/dispatch/error-envelope tests.
3. Phase C: Shim convergence
   - Replace shim logic with canonical-first delegation.
   - Convert residual fallback branches to shared explicit stubs/helpers.
4. Phase D: Spec and cutover hardening
   - Prove chapter-level behavior in `mcpplusplus/spec/*` using deterministic evidence.
   - Validate rollout and rollback for canonical default runtime.

#### 5.0.6 Per-Feature Acceptance Gate

1. Source operation mapped to canonical implementation or explicit deferred rationale.
2. `tools_list_tools` and `tools_get_schema` evidence for the migrated operation.
3. `tools_dispatch` behavioral parity evidence, including negative/error paths.
4. Conformance artifacts updated:
   - `mcpplusplus/SPEC_GAP_MATRIX.md`
   - `mcpplusplus/CONFORMANCE_CHECKLIST.md` (when requirement status changes)
5. Shim paths (if any) verified as delegation-only for the migrated operation.

#### 5.0.7 Definition of Merge Completion

1. `ipfs_accelerate_py/mcp_server` is feature-authoritative for runtime + tools + MCP++ primitives.
2. `mcplusplus_module` has no unique runtime behavior and only compatibility delegation/stubs.
3. Every spec chapter in `mcpplusplus/spec/*` is represented by deterministic execution evidence.
4. Canonical runtime can be the default path with tested rollback.

#### 5.0.8 Top-Level Runtime Module Delta Triage

Current top-level module delta (source minus canonical) includes:

1. Dispatch/transport/control modules:
   - `dispatch_pipeline.py`, `fastapi_service.py`, `fastapi_config.py`, `standalone_server.py`, `trio_adapter.py`, `mcp_p2p_transport.py`, `register_p2p_tools.py`, `server_context.py`, `mcp_interfaces.py`.
2. Spec-adjacent modules:
   - `ucan_delegation.py`, `temporal_policy.py`, `event_dag.py`, `cid_artifacts.py`, `interface_descriptor.py`.
3. Operational/auxiliary modules:
   - `client.py`, `simple_server.py`, `enterprise_api.py`, `grpc_transport.py`, `compliance_checker.py`, `did_key_manager.py`, `investigation_mcp_client.py`, `p2p_service_manager.py`, `p2p_mcp_registry_adapter.py`, `logger.py`, `validators.py`, `nl_ucan_policy.py`, `temporal_deontic_mcp_server.py`.

Decision workflow for each module:

1. `MERGE`: functionality is missing in canonical runtime and must be ported into `ipfs_accelerate_py/mcp_server/*`.
2. `ADAPT`: functionality already exists in canonical runtime; add compatibility wrapper only.
3. `FACADE`: keep only as compatibility entrypoint in `ipfs_accelerate_py/mcp/*`.
4. `DEFER`: document explicit rationale and target phase in `SPEC_GAP_MATRIX`.

Required output artifact for this triage:

1. `module_port_matrix.csv`-style issue table in tracker (module, decision, target file, tests, owner, phase).

#### 5.0.9 Tool Category Consolidation Rules

Observed category state:

1. Source categories: `51`.
2. Canonical categories: `56`.
3. Canonical-only alias/organizational categories: `idl`, `ipfs`, `p2p`, `rate_limiting`, `workflow`.

Consolidation rules:

1. Treat source category names as compatibility contracts (`ipfs_tools`, `workflow_tools`, `p2p_tools`, `rate_limiting_tools`).
2. Canonical alias categories (`ipfs`, `workflow`, `p2p`, `rate_limiting`) may remain for internal organization, but must not break source-compatible discovery/dispatch behavior.
3. New features land in canonical native categories first, then backfill source-compatible aliases/schemas.

Wave execution order (operation-level parity):

1. Wave P0-A: `ipfs_tools`, `workflow_tools`, `p2p_tools`, `mcplusplus`.
2. Wave P0-B: `security_tools`, `monitoring_tools`, `auth_tools`, `session_tools`.
3. Wave P1-C: `dataset_tools`, `embedding_tools`, `vector_tools`, `search_tools`, `storage_tools`.
4. Wave P1-D: `pdf_tools`, `graph_tools`, `logic_tools`, `web_archive_tools`, long-tail categories.

#### 5.0.10 `mcplusplus_module` Function-Level Merge Plan

This maps known shim functions/surfaces to canonical ownership targets.

| Shim Function/Surface | Current File | Canonical Destination | Merge Action |
| --- | --- | --- | --- |
| `_resolve_storage_wrapper_factory` | `ipfs_accelerate_py/mcplusplus_module/__init__.py` | `ipfs_accelerate_py/mcp_server` shared compatibility utility module | Keep single resolver implementation; import from one canonical location |
| `_missing_dependency_stub` contract | `ipfs_accelerate_py/mcplusplus_module/__init__.py` + subpackages | canonical optional-dependency helper (shared) | Eliminate duplicate stub classes across shim subpackages |
| `_detect_runner_name` | `ipfs_accelerate_py/mcplusplus_module/__init__.py` | `ipfs_accelerate_py/mcp_server` shared compatibility utility module | Keep one runner-identity detection helper and alias from shim |
| `_detect_public_ip` | `ipfs_accelerate_py/mcplusplus_module/__init__.py` | `ipfs_accelerate_py/mcp_server` shared compatibility utility module | Keep one public-IP detection helper and alias from shim |
| Trio registrar resolution (`_resolve_p2p_registrars`) | `ipfs_accelerate_py/mcplusplus_module/trio/server.py` and `.../tools/__init__.py` | `ipfs_accelerate_py/mcp_server` shared compatibility utility module | Preserve a single shared resolver owned canonically; shim should alias/delegate |
| Peer registration payload builder (`_build_peer_registration_record`) | `ipfs_accelerate_py/mcplusplus_module/p2p/bootstrap.py` + `.../peer_registry.py` | `ipfs_accelerate_py/mcp_server` shared compatibility utility module | Keep one peer-info/timestamp builder and reuse across shim backends |
| Peer bootstrap service wrapper (`create_peer_bootstrap`, `PeerBootstrapWrapper`) | `ipfs_accelerate_py/mcplusplus_module/p2p/bootstrap.py` | `ipfs_accelerate_py/mcp_server/mcplusplus/peer_bootstrap.py` | Expose canonical async-friendly wrapper while retaining shim helper import surface |
| `register_p2p_taskqueue_tools` | `ipfs_accelerate_py/mcplusplus_module/tools/taskqueue_tools.py` | `ipfs_accelerate_py/mcp_server/tools/p2p_tools/native_p2p_tools.py` | maintain delegation-only shim; no unique business logic |
| `register_p2p_workflow_tools` | `ipfs_accelerate_py/mcplusplus_module/tools/workflow_tools.py` | `ipfs_accelerate_py/mcp_server/tools/p2p_workflow_tools/native_p2p_workflow_tools.py` | move scheduler semantics to canonical runtime; keep compatibility names in shim |
| Peer/bootstrap adapters (`SimplePeerBootstrap`, `P2PPeerRegistry`) | `ipfs_accelerate_py/mcplusplus_module/p2p/bootstrap.py`, `.../peer_registry.py` | canonical p2p service integration boundary (`mcp_server` + `p2p_tasks`) | converge on canonical-first service calls and transport-neutral persistence |

Integration acceptance for each row:

1. No shim-exclusive runtime branch remains.
2. Shim function calls canonical implementation or explicit compatibility stub.
3. Deterministic coverage exists in `ipfs_accelerate_py/mcplusplus_module/tests/` plus canonical dispatch tests when applicable.

#### 5.0.11 Spec-First Porting Gates (MCP++)

For each spec chapter in `ipfs_accelerate_py/mcpplusplus/spec/*`, porting closes only when both are true:

1. Runtime behavior exists in canonical modules (`ipfs_accelerate_py/mcp_server/*`).
2. At least one transport-level execution path validates behavior (`stdio`, `http`, or `mcp+p2p` where applicable).

Mandatory chapter evidence bundle:

1. Unit/module test evidence.
2. Unified bootstrap dispatch evidence.
3. Matrix updates in `mcpplusplus/SPEC_GAP_MATRIX.md`.
4. Checklist status consistency in `mcpplusplus/CONFORMANCE_CHECKLIST.md`.

#### 5.0.12 Initial `module_port_matrix` (Execution Baseline)

Use this as the first implementation backlog for source top-level modules that are not yet mirrored in canonical runtime.

| Source Module | Decision | Canonical Target | Phase | Deterministic Evidence |
| --- | --- | --- | --- | --- |
| `__main__.py` | `FACADE` | `ipfs_accelerate_py/mcp/__main__.py` and canonical `mcp_server/server.py` startup path | A | startup contract tests |
| `audit_metrics_bridge.py` | `MERGE` | `ipfs_accelerate_py/mcp_server/monitoring.py` + `policy_audit_log.py` integration hooks | B | observability + policy audit integration tests |
| `cid_artifacts.py` | `ADAPT` | `ipfs_accelerate_py/mcp_server/mcplusplus/artifacts.py` | B | artifact chain tests + unified bootstrap assertions |
| `client.py` | `FACADE` | `ipfs_accelerate_py/mcp/integration.py` or `p2p_tasks/client.py` | C | process/subprocess client contract tests |
| `compliance_checker.py` | `ADAPT` | `ipfs_accelerate_py/mcp_server/compliance_checker.py` compatibility facade delegating to source compliance checker | C | `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni205_compliance_checker_adapter.py` |
| `did_key_manager.py` | `MERGE` | `ipfs_accelerate_py/mcp_server/did_key_manager.py` canonical key-management module | B | `test_mcp_server_did_key_manager.py`, `test_mcp_server_secrets_vault.py` |
| `dispatch_pipeline.py` | `MERGE` | `ipfs_accelerate_py/mcp_server/runtime_router.py` + `server.py` dispatch flow | B | unified bootstrap dispatch regression suite |
| `enterprise_api.py` | `ADAPT` | `ipfs_accelerate_py/mcp_server/enterprise_api.py` compatibility facade delegating to source enterprise API surface | C | `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni207_deferred_module_adapters.py` |
| `event_dag.py` | `ADAPT` | `ipfs_accelerate_py/mcp_server/mcplusplus/event_dag.py` | B | event DAG lineage/replay tests |
| `fastapi_config.py` | `MERGE` | new canonical config helper under `ipfs_accelerate_py/mcp_server/` | B | FastAPI startup/config parsing tests |
| `fastapi_service.py` | `MERGE` | canonical HTTP entrypoint under `ipfs_accelerate_py/mcp_server/` with facade in `mcp/` | B | transport parity HTTP tests |
| `grpc_transport.py` | `ADAPT` | `ipfs_accelerate_py/mcp_server/grpc_transport.py` compatibility facade delegating to source gRPC transport surface | C | `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni207_deferred_module_adapters.py` |
| `interface_descriptor.py` | `ADAPT` | `ipfs_accelerate_py/mcp_server/mcplusplus/idl_registry.py` + `tools/idl/*` | B | IDL descriptor/canonicalization tests |
| `investigation_mcp_client.py` | `ADAPT` | `ipfs_accelerate_py/mcp_server/investigation_mcp_client.py` compatibility facade delegating to source investigation client | C | `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni206_investigation_client_adapter.py` |
| `logger.py` | `ADAPT` | shared logger usage in `mcp_server/*` (no standalone runtime module) | C | import/runtime smoke tests |
| `mcp_interfaces.py` | `MERGE` | canonical metadata/interfaces helpers (`tool_metadata.py` + IDL tooling) | B | schema/interface exposure tests |
| `mcp_p2p_transport.py` | `ADAPT` | `ipfs_accelerate_py/p2p_tasks/mcp_p2p.py` + `mcp_server/mcplusplus/p2p_framing.py` | B | MCP+p2p framing/handler tests |
| `nl_ucan_policy.py` | `ADAPT` | `ipfs_accelerate_py/mcp_server/nl_ucan_policy.py` compatibility facade delegating to source NL-UCAN policy surface | C | `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni207_deferred_module_adapters.py` |
| `p2p_mcp_registry_adapter.py` | `MERGE` | `ipfs_accelerate_py/p2p_tasks/service.py` integration boundary | B | p2p registry/handler integration tests |
| `p2p_service_manager.py` | `MERGE` | `ipfs_accelerate_py/p2p_tasks/service.py` | B | process-level transport service tests |
| `register_p2p_tools.py` | `ADAPT` | canonical tool-loader path (`mcp_server/server.py` + `wave_a_loaders.py`) | C | bootstrap registration idempotency tests |
| `server_context.py` | `MERGE` | unified services context in `ipfs_accelerate_py/mcp_server/server.py` | B | `_unified_services` consumption tests |
| `simple_server.py` | `FACADE` | `ipfs_accelerate_py/mcp/standalone.py` | C | standalone startup contract tests |
| `standalone_server.py` | `FACADE` | `ipfs_accelerate_py/mcp/standalone.py` + canonical runtime wiring | C | subprocess startup parity tests |
| `temporal_deontic_mcp_server.py` | `ADAPT` | `ipfs_accelerate_py/mcp_server/mcplusplus/policy_engine.py` + dispatch hooks | B | temporal policy decision/obligation tests |
| `temporal_policy.py` | `ADAPT` | `ipfs_accelerate_py/mcp_server/mcplusplus/policy_engine.py` | B | policy engine unit + bootstrap tests |
| `trio_adapter.py` | `ADAPT` | `ipfs_accelerate_py/p2p_tasks/*` trio transport path | C | trio networked transport tests |
| `trio_bridge.py` | `DEFER` | compatibility-only bridge track | D | deferred rationale in matrix |
| `ucan_delegation.py` | `ADAPT` | `ipfs_accelerate_py/mcp_server/mcplusplus/delegation.py` | B | UCAN execution-time enforcement tests |
| `validators.py` | `MERGE` | validation helpers in canonical runtime modules (router/tools/policy) | C | invalid-input/error-envelope tests |

Phase ordering for this matrix:

1. Phase A: validate and lock decisions (`MERGE/ADAPT/FACADE/DEFER`) per row.
2. Phase B: implement all `MERGE` and `ADAPT` rows that affect P0 spec/runtime paths.
3. Phase C: finalize `FACADE` rows and compatibility entrypoints.
4. Phase D: document and track `DEFER` rows with explicit rationale and target milestones.

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

Recent execution (2026-03-06):

1. Added explicit profile capability snapshot regression coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` to lock the canonical supported-profile list and normalized negotiation payload across bootstrap attachments, runtime context, and context snapshots.
2. Added HTTP transport process-level coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_transport_process_level.py` to verify mounted standalone apps preserve the same additive unified profile negotiation metadata exposed by the canonical bootstrap.

### 6.2 Transport (`transport-mcp-p2p`)

Current posture: Implemented.
Next plan:

1. Expand networked interop tests for mixed-version peers.
2. Add stricter fuzz-style framing boundary tests (decode + rate limits).
3. Harden status counter semantics under sustained abuse scenarios.

Recent execution (2026-03-06):

1. Completed stricter framing boundary regressions in `ipfs_accelerate_py/mcp/tests/test_mcp_transport_p2p_framing_limits.py` covering malformed UTF-8 payloads, malformed JSON payloads, and over-capacity token bucket costs.
2. Added handler regression in `ipfs_accelerate_py/mcp/tests/test_mcp_transport_mcp_p2p_handler_limits.py` validating malformed UTF-8 payload classification as deterministic `invalid_json` transport error.
3. Hardened canonical framing decode normalization in `ipfs_accelerate_py/mcp_server/mcplusplus/p2p_framing.py` with explicit `FramingError` codes for malformed UTF-8/JSON, and updated `ipfs_accelerate_py/p2p_tasks/mcp_p2p.py` to preserve `invalid_json` classification.
4. Added sustained mixed-abuse counter regression in `ipfs_accelerate_py/mcp/tests/test_mcp_transport_mcp_p2p_handler_limits.py` validating exact cumulative status counters across oversized-frame, malformed-UTF8, unauthorized, rate-limited, and write-failure sessions.
5. Expanded live mixed-version libp2p interop coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_transport_trio_mcp_p2p_networked.py` to validate unknown-profile fallback and canonical `tools/list` vs alias `tools.list` parity over a real MCP+p2p network session.

### 6.3 MCP-IDL (`mcp-idl`)

Current posture: Implemented.
Next plan:

1. Ensure descriptor generation covers all loaded migrated categories consistently.
2. Add canonicalization hash stability tests across Python/runtime environments.
3. Add compatibility algorithm regression corpus (`interfaces/compat`).

Recent execution (2026-03-06):

1. Added descriptor-generation coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_idl.py` to verify all loaded migrated categories registered in the manager produce consistent MCP-IDL descriptors with deterministic names, method surfaces, and capability requirements.
2. Added cross-runtime canonicalization stability coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_idl.py` to verify canonical descriptor bytes and computed interface CIDs remain stable across isolated Python processes with different runtime hash seeds.
3. Added an `interfaces/compat` regression corpus in `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_idl.py` to lock deterministic compatibility verdicts for missing-capability ordering, unknown-interface handling, and sorted suggested-alternative selection.

### 6.4 CID Artifacts (`cid-native-artifacts`)

Current posture: Implemented.
Next plan:

1. Add durable backend option validation beyond in-memory storage.
2. Add replayability tests for artifact chain reconstruction.
3. Verify envelope emission policy across all dispatch modes.

Recent execution (2026-03-06):

1. Added config-driven artifact-store backend normalization in `ipfs_accelerate_py/mcp_server/configs.py` and JSON-backed artifact persistence/reload support in `ipfs_accelerate_py/mcp_server/server.py`, with focused coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_artifacts.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` to validate durable backend selection, on-disk persistence, and startup reload semantics.
2. Added dispatch-backed replay coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_artifacts.py` to reconstruct persisted parent/child artifact chains from the JSON backend after server reload, verifying event -> receipt -> decision -> intent linkage without relying on live runtime state.
3. Verified dispatch-mode envelope emission policy in `ipfs_accelerate_py/mcp_server/server.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_artifacts.py`, ensuring explicit `__emit_artifacts=False` suppresses artifact envelope fields even when config-default emission is enabled, while cache-hit dispatches still emit full artifact envelopes when `__emit_artifacts=True`.

### 6.5 UCAN Delegation (`ucan-delegation`)

Current posture: Implemented.
Next plan:

1. Expand cryptographic verification test vectors (`did:key`, signature encodings). ✅
2. Add policy + delegation combined denial matrix (cross-feature interactions). ✅
3. Add explicit proof lineage telemetry for deny/allow outcomes. ✅

Recent execution (2026-03-07):

1. Expanded the policy + delegation interaction matrix in `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_ucan.py` to cover `UCAN deny + policy allow` and `UCAN allow + policy deny`, locking deterministic precedence and response-shape behavior across combined execution-time controls.
2. Extended `ipfs_accelerate_py/mcp_server/mcplusplus/delegation.py` and `ipfs_accelerate_py/mcp_server/server.py` so UCAN validation results now emit deterministic `proof_lineage` and `failure_hop` telemetry for both allow and deny outcomes, and unified dispatch success responses now surface the same authorization telemetry shape as denial responses.
3. Revalidated the focused UCAN lane with `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_ucan.py` (`24 passed`), including new dispatch assertions for proof-lineage telemetry on both allow and deny paths.

### 6.6 Temporal Policy (`temporal-deontic-policy`)

Current posture: Implemented.
Next plan:

1. Expand obligation lifecycle validation (deadline and fulfillment semantics). ✅
2. Add policy version migration tests (`policy_cid` evolution). ✅
3. Validate decision persistence and retrieval parity across transports. ✅

Recent execution (2026-03-07):

1. Extended `ipfs_accelerate_py/mcp_server/mcplusplus/policy_engine.py` so outstanding obligations now expose deterministic deadline status (`pending` / `overdue`) while fulfilled obligations are treated as completed lifecycle entries that no longer force `allow_with_obligations` decisions.
2. Added focused policy-engine coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_policy.py` for pending-vs-overdue deadline progression and fulfilled-obligation semantics.
3. Expanded unified-dispatch coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` to verify persisted policy decisions reflect fulfilled obligations as plain `allow` outcomes and that `decision_cid` changes deterministically when `policy_cid` evolves even if `policy_version` is unchanged.
4. Added transport E2E coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_transport_e2e_matrix.py` to prove HTTP-style meta-tool dispatch and MCP+p2p `tools/call` return the same persisted temporal-policy `policy_decision` / `decision_cid` contract and artifact-store retrieval shape.
5. Revalidated with `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_policy.py` (`7 passed`), a focused unified-bootstrap temporal policy slice (`5 passed`), and `ipfs_accelerate_py/mcp/tests/test_mcp_server_transport_e2e_matrix.py` (`6 passed`).

### 6.7 Event DAG (`event-dag-ordering`)

Current posture: Implemented.
Next plan:

1. Expand DAG conflict and fork handling test scenarios. ✅
2. Add deterministic replay and rollback tests for larger graphs. ✅
3. Add durability and snapshot compatibility tests. ✅

Recent execution (2026-03-07):

1. Expanded `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_event_dag.py` with deterministic fork/merge regressions covering lexical-parent lineage selection for multi-parent nodes, replay deduplication of shared merge descendants, and merge-fork snapshot roundtrip stability.
2. Added unified-dispatch fork/merge coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` to verify artifact-emitted event DAG snapshots preserve deterministic merge lineage, replay ordering, and rollback behavior when a merge node references two parents in non-lexical input order.
3. Added a larger layered unified-dispatch Event DAG regression in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` to confirm repeated replay/rollback determinism, event-count accounting, and snapshot rebuild stability for a multi-level artifact-emitted graph.
4. Added snapshot compatibility regressions in `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_event_dag.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` to verify deterministic rebuild from reordered snapshot entries while ignoring malformed/noise payloads.
5. Revalidated with `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_event_dag.py` plus focused unified-bootstrap Event DAG coverage (`18 passed`).

### 6.8 Risk Scheduling (`risk-scheduling`)

Current posture: Implemented.
Next plan:

1. Strengthen frontier execution binding tests under load and retries. ✅
2. Add neighborhood consensus signal integration as optional enhancement. ✅
3. Validate risk-state lineage integrity with event/artifact linkage. ✅

Recent execution (2026-03-08):

1. Added unified-dispatch risk lineage coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` to verify the emitted artifact `event_cid` is the same value tracked in risk state, frontier metadata, Event DAG lineage, and workflow-scheduler execution binding.
2. Added a frontier load/retry regression in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` to verify `__execute_frontier` pops older ready work first while keeping a penalized retry item queued behind both the executed item and the newly emitted event.
3. Wired optional frontier consensus inputs through `ipfs_accelerate_py/mcp_server/server.py` and added unified-dispatch coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` proving a high-confidence consensus signal can prioritize the dispatched frontier item and propagates through execution metadata.
4. Revalidated with focused risk scheduling coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_risk_scheduler.py` plus unified-bootstrap risk/frontier tests (`15 passed`).
5. Added a focused chapter-interaction regression in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` covering the combined UCAN allow path, temporal-policy obligations, CID artifact emission, and risk-frontier execution binding, then revalidated that interaction slice together with adjacent deny/policy/risk lineage assertions (`5 passed, 144 deselected`).

## 7. Milestones and Timeline

## Milestone M1 (Weeks 1-2): Convergence Hardening

1. Finish shim fallback/resolver dedup (`mcplusplus_module` convergence).
2. Lock deterministic optional-dependency contract behavior.
3. Add integration assertions proving canonical runtime ownership of behavior.

Recent execution (2026-03-08):

1. Moved shared shim compatibility helpers (`_missing_dependency_stub`, `_resolve_storage_wrapper_factory`, `_create_storage_wrapper`) into canonical `ipfs_accelerate_py/mcp_server/compatibility.py` and reduced `mcplusplus_module.__init__` to thin aliasing for those behaviors.
2. Moved shared runner/public-IP detection helpers (`_detect_runner_name`, `_detect_public_ip`) into canonical `ipfs_accelerate_py/mcp_server/compatibility.py`, keeping `mcplusplus_module.__init__` as a thin alias layer for P2P bootstrap/peer-registry consumers.
3. Moved shared Trio/P2P registrar resolution (`_resolve_p2p_registrars`) into canonical `ipfs_accelerate_py/mcp_server/compatibility.py`, reducing `mcplusplus_module.tools` to aliasing while preserving Trio server delegation through the same shared resolver.
4. Moved shared peer-registration payload construction (`_build_peer_registration_record`) into canonical `ipfs_accelerate_py/mcp_server/compatibility.py`, eliminating duplicate peer-info/timestamp shaping across `mcplusplus_module.p2p.bootstrap` and `mcplusplus_module.p2p.peer_registry` while preserving backend-specific persistence.
5. Revalidated shim convergence with focused helper/peer-registration coverage in `ipfs_accelerate_py/mcplusplus_module/tests/test_tool_adapters.py -k "missing_dependency_stub or storage_wrapper or detect_runner_name or detect_public_ip or peer_registration_record or register_peer"` (`12 passed, 16 deselected`) plus broader trio/tool adapter regression coverage in `ipfs_accelerate_py/mcplusplus_module/tests/test_tool_adapters.py ipfs_accelerate_py/mcplusplus_module/tests/test_trio_server.py -k "resolve_p2p_registrars or missing_dependency_stub or storage_wrapper or register_p2p_tools or register_peer"` (`14 passed, 31 deselected`).
6. Added canonical `ipfs_accelerate_py.mcp_server.mcplusplus.peer_bootstrap` wrapper ownership (`PeerBootstrapWrapper`, `create_peer_bootstrap`) so bootstrap discovery/cleanup/address retrieval can be consumed through canonical async-friendly runtime primitives instead of only through the shim helper class.
7. Wired unified bootstrap service ownership through `peer_bootstrap_factory` and `tools_dispatch` bootstrap-address probing so canonical runtime consumers can resolve bootstrap addresses without direct shim imports, validated by targeted unified-bootstrap coverage (`2 passed`).
8. Added canonical `create_peer_service_bundle` ownership in `ipfs_accelerate_py/mcp_server/mcplusplus/peer_services.py` and wired `P2PServiceManager` to initialize peer-registry/bootstrap wrappers through that shared construction path, validated by `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni013_p2p_adapters.py` (`6 passed`) and focused unified-bootstrap revalidation (`2 passed, 148 deselected`).
9. Added canonical `create_peer_discovery` ownership in `ipfs_accelerate_py/mcp_server/mcplusplus/peer_discovery.py` and wired unified bootstrap `peer_discovery_factory` through that shared peer-service construction path, validated by focused peer primitive and unified-bootstrap coverage (`3 passed, 152 deselected`).

Exit:

1. No remaining high-risk duplicated runtime logic between shim and canonical paths.

## Milestone M2 (Weeks 3-5): Deep Tool Parity Wave

1. Complete B1 categories (high-impact runtime categories).
2. Add schema parity checks and behavioral regression suites.
3. Update `SPEC_GAP_MATRIX` capability rows with evidence.

Recent execution (2026-03-08):

1. Hardened enhanced `monitoring_tools` response normalization in `ipfs_accelerate_py/mcp_server/tools/monitoring_tools/native_monitoring_tools.py` so `check_health`, `collect_metrics`, and `manage_alerts` preserve deterministic source-like defaults for timestamps, diagnostics, trend/anomaly sections, alert filters/metrics, and threshold-update payloads even when delegates return sparse success payloads.
2. Extended focused monitoring parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni125_monitoring_tools.py` and revalidated unified bootstrap dispatch behavior in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`14 passed, 147 deselected`).
3. Hardened enhanced `auth_tools` response normalization in `ipfs_accelerate_py/mcp_server/tools/auth_tools/native_auth_tools.py` so sparse success payloads still preserve deterministic source-like defaults for authentication metadata, refresh/decode/validation subpayloads, and `get_user_info` permission/profile shaping.
4. Extended focused auth parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni111_auth_tools.py` and revalidated unified bootstrap dispatch behavior in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`10 passed, 148 deselected`).
5. Hardened enhanced `session_tools` response normalization in `ipfs_accelerate_py/mcp_server/tools/session_tools/native_session_tools.py` so sparse success payloads preserve deterministic source-like session metadata for `create_session`, `manage_session`, and `get_session_state`, including a normalized nested `session` payload plus stable timestamps, config/resources, metadata, tags, and request counters.
6. Extended focused session parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni109_session_tools.py`, revalidated cleanup-option contracts in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni164_session_tools.py`, and confirmed unified bootstrap dispatch behavior in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`13 passed, 149 deselected`; `1 passed, 148 deselected`).
7. Hardened `p2p_tools` remote cache, remote submit, and service-status wrappers in `ipfs_accelerate_py/mcp_server/tools/p2p_tools/native_p2p_tools.py` so `p2p_remote_cache_get` / `set` / `has` / `delete`, `p2p_remote_submit_task`, and `p2p_service_status` now enforce canonical validation contracts and preserve deterministic sparse-success defaults for key/peer addressing, timeout handling, cache hit/delete fields, task submission metadata, and service/peer status envelopes.
8. Extended focused P2P parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni135_p2p_tools.py` and revalidated unified bootstrap discovery/dispatch behavior in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`16 passed, 148 deselected`).
9. Hardened `mcplusplus` wrapper normalization in `ipfs_accelerate_py/mcp_server/tools/mcplusplus/native_mcplusplus_tools.py` so list/stats/status/result/discovery and mutation surfaces preserve deterministic sparse-success defaults when delegates return minimal success envelopes, including `mcplusplus_taskqueue_submit`, `mcplusplus_taskqueue_priority`, `mcplusplus_taskqueue_cancel`, `mcplusplus_taskqueue_list`, `mcplusplus_taskqueue_stats`, `mcplusplus_taskqueue_retry`, `mcplusplus_taskqueue_pause`, `mcplusplus_taskqueue_resume`, `mcplusplus_taskqueue_clear`, `mcplusplus_worker_register`, `mcplusplus_worker_unregister`, `mcplusplus_worker_status`, `mcplusplus_taskqueue_result`, `mcplusplus_workflow_submit`, `mcplusplus_workflow_cancel`, `mcplusplus_workflow_get_status`, `mcplusplus_workflow_list`, `mcplusplus_workflow_dependencies`, `mcplusplus_workflow_result`, `mcplusplus_peer_list`, `mcplusplus_peer_discover`, `mcplusplus_peer_connect`, `mcplusplus_peer_disconnect`, `mcplusplus_peer_metrics`, and `mcplusplus_peer_bootstrap_network`.
10. Extended focused `mcplusplus` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni147_mcplusplus_tools.py` and revalidated unified bootstrap `mcplusplus` dispatch behavior in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`9 passed, 149 deselected`).
11. Hardened `security_tools` envelope normalization in `ipfs_accelerate_py/mcp_server/tools/security_tools/native_security_tools.py` so `check_access_permission` now infers deterministic error status from sparse delegate error payloads, while `check_access_permissions_batch` rejects non-boolean `fail_fast` values and preserves stable aggregate error-envelope fields (`processed`, `requested`, `all_allowed`, and counts) on invalid batch inputs.
12. Extended focused `security_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni127_security_tools.py` and revalidated security dispatch/bootstrap behavior in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni182_security_dispatch_compat.py` plus `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`13 passed, 149 deselected`).
13. Hardened `dataset_tools` delegate-payload normalization in `ipfs_accelerate_py/mcp_server/tools/dataset_tools/native_dataset_tools.py` so `load_dataset`, `save_dataset`, `process_dataset`, `convert_dataset_format`, `text_to_fol`, `legal_text_to_deontic`, and `dataset_tools_claudes` now infer deterministic `status="error"` when delegates return contradictory failed payloads such as `{"status": "success", "success": false, ...}`.
14. Extended focused `dataset_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni157_dataset_tools.py` and revalidated dataset dispatch compatibility in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni183_dataset_dispatch_compat.py` (`12 passed`).
15. Hardened `storage_tools` delegate-payload normalization in `ipfs_accelerate_py/mcp_server/tools/storage_tools/native_storage_tools.py` so `store_data`, `retrieve_data`, `manage_collections`, and `query_storage` now infer deterministic `status="error"` when delegates return contradictory failed payloads such as `{"status": "success", "success": false, ...}`.
16. Extended focused `storage_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni156_storage_tools.py` and revalidated storage dispatch/bootstrap behavior in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 154 deselected`).
17. Hardened `audit_tools` delegate-payload normalization in `ipfs_accelerate_py/mcp_server/tools/audit_tools/native_audit_tools.py` so `record_audit_event`, `generate_audit_report`, and `audit_tools` now infer deterministic `status="error"` when delegates return contradictory failed payloads such as `{"status": "success", "success": false, ...}`.
18. Extended focused `audit_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni118_audit_tools.py` and revalidated audit dispatch/bootstrap behavior in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 160 deselected`).
19. Hardened `index_management_tools` delegate-payload normalization in `ipfs_accelerate_py/mcp_server/tools/index_management_tools/native_index_management_tools.py` so `load_index`, `manage_shards`, `monitor_index_status`, and `manage_index_configuration` now infer deterministic `status="error"` when delegates return contradictory failed payloads such as `{"status": "success", "success": false, ...}`.
20. Extended focused `index_management_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni114_index_management_tools.py` and revalidated index-management dispatch/bootstrap behavior in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 163 deselected`).
21. Hardened `ipfs_cluster_tools` delegate-payload normalization in `ipfs_accelerate_py/mcp_server/tools/ipfs_cluster_tools/native_ipfs_cluster_tools.py` so `manage_ipfs_cluster` and `manage_ipfs_content` now infer deterministic `status="error"` when delegates return contradictory failed payloads such as `{"status": "success", "success": false, ...}`.
22. Extended focused `ipfs_cluster_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni126_ipfs_cluster_tools.py` and revalidated IPFS-cluster dispatch/bootstrap behavior in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 160 deselected`).
23. Hardened `web_scraping_tools` delegate-payload normalization in `ipfs_accelerate_py/mcp_server/tools/web_scraping_tools/native_web_scraping_tools.py` so `scrape_url_tool`, `scrape_multiple_urls_tool`, and `check_scraper_methods_tool` now infer deterministic `status="error"` when delegates return contradictory failed payloads such as `{"status": "success", "success": false, ...}`.
24. Extended focused `web_scraping_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni123_web_scraping_tools.py` and revalidated web-scraping dispatch/bootstrap behavior in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 162 deselected`).
25. Revalidated `dashboard_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/dashboard_tools/native_dashboard_tools.py` and added the missing unified bootstrap dispatch regression for `export_tdfol_statistics` so contradictory failed payloads now stay covered through dispatch.
26. Extended focused `dashboard_tools` parity evidence in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni122_dashboard_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 163 deselected`).
27. Extended `alert_tools` parity with the missing unified bootstrap dispatch regression for contradictory failed delegate payloads so `remove_alert_rule` now stays covered through unified dispatch in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py`.
28. Revalidated focused `alert_tools` direct and bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni119_alert_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 166 deselected`).
29. Hardened `p2p_tools` delegate-payload normalization in `ipfs_accelerate_py/mcp_server/tools/p2p_tools/native_p2p_tools.py` so contradictory failed payloads like `{"status": "success", "success": false, ...}` now infer deterministic `status="error"` across local and remote helper wrappers.
30. Extended focused `p2p_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni135_p2p_tools.py` and revalidated unified `p2p_tools` dispatch/bootstrap behavior in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 170 deselected`).
31. Hardened `admin_tools` delegate-payload normalization in `ipfs_accelerate_py/mcp_server/tools/admin_tools/native_admin_tools.py` so `manage_endpoints`, `system_maintenance`, `configure_system`, `system_health`, `get_system_status`, `manage_service`, `update_configuration`, and `cleanup_resources` now infer deterministic `status="error"` when delegates return contradictory failed payloads such as `{"status": "success", "success": false, ...}`.
32. Extended focused `admin_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni116_admin_tools.py` and revalidated unified admin dispatch/bootstrap behavior in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 172 deselected`).
33. Hardened `graph_tools` delegate-payload normalization in `ipfs_accelerate_py/mcp_server/tools/graph_tools/native_graph_tools.py` so contradictory failed payloads such as `{"status": "success", "success": false, ...}` now infer deterministic `status="error"` across graph creation, mutation, query, transaction, search, visualization, explanation, and provenance wrappers.
34. Extended focused `graph_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni159_graph_tools.py` and added isolated dispatch compatibility coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni260_graph_dispatch_compat.py` (`14 passed`).
35. Hardened `alert_tools` delegate-payload normalization in `ipfs_accelerate_py/mcp_server/tools/alert_tools/native_alert_tools.py` so contradictory failed payloads such as `{"status": "success", "success": false, ...}` now infer deterministic `status="error"` across Discord send, rule evaluation, rule listing, and rule removal wrappers.
36. Extended focused `alert_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni119_alert_tools.py` and added isolated dispatch compatibility coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni261_alert_dispatch_compat.py` (`13 passed`).
37. Hardened `email_tools` delegate-payload normalization in `ipfs_accelerate_py/mcp_server/tools/email_tools/native_email_tools.py` so contradictory failed payloads such as `{"status": "success", "success": false, ...}` now infer deterministic `status="error"` across connection, folder listing, export analysis/search, and EML parsing wrappers.
38. Extended focused `email_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni120_email_tools.py` and added isolated dispatch compatibility coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni262_email_dispatch_compat.py` (`15 passed`).
39. Tightened `sparse_embedding_tools` delegate-payload normalization in `ipfs_accelerate_py/mcp_server/tools/sparse_embedding_tools/native_sparse_embedding_tools.py` so contradictory failed payloads such as `{"status": "success", "success": false, ...}` now infer deterministic `status="error"` across sparse embedding generation, indexing, search, and model-management wrappers.
40. Extended focused `sparse_embedding_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni124_sparse_embedding_tools.py` and added isolated dispatch compatibility coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni263_sparse_embedding_dispatch_compat.py` (`11 passed`).
41. Hardened `file_detection_tools` delegate-payload normalization in `ipfs_accelerate_py/mcp_server/tools/file_detection_tools/native_file_detection_tools.py` so contradictory failed payloads such as `{"status": "success", "success": false, ...}` now infer deterministic `status="error"` across single-file, batch, and detection-accuracy wrappers.
42. Extended focused `file_detection_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni117_file_detection_tools.py` and added isolated dispatch compatibility coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni264_file_detection_dispatch_compat.py` (`12 passed`).
43. Tightened base `monitoring_tools` delegate-payload normalization in `ipfs_accelerate_py/mcp_server/tools/monitoring_tools/native_monitoring_tools.py` so contradictory failed payloads such as `{"status": "success", "success": false, ...}` now infer deterministic `status="error"` across base health, performance-metrics, service-status, and reporting wrappers while preserving healthy/ok normalization.
44. Extended focused `monitoring_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni125_monitoring_tools.py` and added isolated dispatch compatibility coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni266_monitoring_dispatch_compat.py` (`15 passed`).
45. Tightened `search_tools` delegate-payload normalization in `ipfs_accelerate_py/mcp_server/tools/search_tools/native_search_tools.py` so contradictory failed payloads such as `{"status": "success", "success": false, ...}` now infer deterministic `status="error"` across semantic, similarity, and faceted search wrappers.
46. Extended focused `search_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni165_search_tools.py` and revalidated isolated dispatch compatibility coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni256_search_dispatch_compat.py` (`13 passed`).
47. Revalidated `file_detection_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/file_detection_tools/native_file_detection_tools.py` after recent edits and added the missing unified bootstrap dispatch regression for `detect_file_type` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
48. Revalidated focused `file_detection_tools` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni117_file_detection_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 169 deselected`).
49. Revalidated `file_converter_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/file_converter_tools/native_file_converter_tools.py` and added the missing unified bootstrap dispatch regression for `convert_file_tool` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
50. Revalidated focused `file_converter_tools` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni131_file_converter_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 177 deselected`).
51. Revalidated `function_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/functions/native_function_tools.py` and added the missing unified bootstrap dispatch regression for `execute_python_snippet` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
52. Revalidated focused `function_tools` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni129_function_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`4 passed, 166 deselected`).
53. Revalidated `data_processing_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/data_processing_tools/native_data_processing_tools.py` and added the missing unified bootstrap dispatch regression for `chunk_text` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
54. Revalidated focused `data_processing_tools` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni111_data_processing_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 172 deselected`).
55. Revalidated `graph_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/graph_tools/native_graph_tools.py` and added the missing unified bootstrap dispatch regression for `graph_add_entity` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
56. Revalidated focused `graph_tools` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni159_graph_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 170 deselected`).
57. Revalidated `pdf_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/pdf_tools/native_pdf_tools.py` and added the missing unified bootstrap dispatch regression for `pdf_query_corpus` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
58. Revalidated focused `pdf_tools` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni160_pdf_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 173 deselected`).
59. Revalidated `email_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/email_tools/native_email_tools.py` and added the missing unified bootstrap dispatch regression for `email_test_connection` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
60. Revalidated focused `email_tools` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni120_email_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 178 deselected`).
61. Revalidated `discord_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/discord_tools/native_discord_tools.py` and added the missing unified bootstrap dispatch regression for `discord_list_guilds` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
62. Revalidated focused `discord_tools` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni130_discord_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 176 deselected`).
63. Revalidated `vector_store_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/vector_store_tools/native_vector_store_tools.py` and added the missing unified bootstrap dispatch regression for `vector_index` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
64. Revalidated focused `vector_store_tools` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni128_vector_store_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 178 deselected`).
65. Revalidated `p2p_workflow_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/p2p_workflow_tools/native_p2p_workflow_tools.py` and added the missing unified bootstrap dispatch regression for `schedule_p2p_workflow` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
66. Revalidated focused `p2p_workflow_tools` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni136_p2p_workflow_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 178 deselected`).
67. Hardened contradictory failed-delegate normalization across `rate_limiting_tools` and canonical `rate_limiting` wrappers in `ipfs_accelerate_py/mcp_server/tools/rate_limiting_tools/native_rate_limiting_tools_category.py` and `ipfs_accelerate_py/mcp_server/tools/rate_limiting/native_rate_limiting_tools.py` so delegate payloads such as `{"status": "success", "success": false, "error": ...}` now infer deterministic `status="error"` for rate-limit checks and management flows.
68. Extended focused `rate_limiting_tools` and `rate_limiting` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni151_rate_limiting_tools_category.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni152_native_rate_limiting_tools.py`, and added isolated dispatch compatibility coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni292_rate_limiting_tools_dispatch_compat.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni293_rate_limiting_dispatch_compat.py` (`19 passed`).
69. Added the missing isolated dispatch compatibility regression for `background_task_tools` in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni294_background_task_dispatch_compat.py`, covering contradictory failed delegate payload normalization across `check_task_status`, `manage_background_tasks`, `manage_task_queue`, and `get_task_status` through canonical `tools_dispatch`.
70. Revalidated focused `background_task_tools` direct, import-compat, isolated dispatch, and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni121_background_task_tools.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni214_background_task_import_compat.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni294_background_task_dispatch_compat.py`, and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`15 passed`, `1 passed`).
71. Hardened contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/logic_tools/native_logic_tools.py` so delegate payloads such as `{"status": "success", "success": false, "error": ...}` now infer deterministic `status="error"` while preserving `success=False` across representative TDFOL and CEC wrappers.
72. Extended focused `logic_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni161_logic_tools.py` and added isolated dispatch compatibility coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni294_logic_dispatch_compat.py` (`15 passed`).
73. Revalidated `workflow_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/workflow_tools/native_workflow_tools_category.py` and added the missing unified bootstrap dispatch regression for `schedule_p2p_workflow` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
74. Revalidated focused `workflow_tools` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni134_workflow_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 183 deselected`).
75. Revalidated `finance_data_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/finance_data_tools/native_finance_data_tools.py` and added the missing unified bootstrap dispatch regression for `scrape_stock_data` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
76. Revalidated focused `finance_data_tools` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni137_finance_data_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 181 deselected`).
77. Hardened contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/cache_tools/native_cache_tools.py` so cache-manager payloads such as `{"status": "success", "success": false, "error": ...}` now infer deterministic `status="error"` across representative cache read, stats, optimization, and embedding lookup wrappers.
78. Extended focused `cache_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni162_cache_tools.py` and added isolated dispatch compatibility coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni295_cache_dispatch_compat.py` alongside the existing direct and import-compat suites (`16 passed`).
79. Added the missing isolated dispatch compatibility regression for `session_tools` in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni296_session_dispatch_compat.py`, covering enhanced `create_session`, `manage_session`, and `get_session_state` validation and success contracts through canonical `tools_dispatch` without relying on the shared bootstrap suite.
80. Revalidated focused `session_tools` direct, cleanup-option, import-compat, and isolated dispatch coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni109_session_tools.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni164_session_tools.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni210_session_import_compat.py`, and `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni296_session_dispatch_compat.py` (`15 passed`).
81. Added the missing isolated dispatch compatibility regression for `ipfs_tools` in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni297_ipfs_tools_dispatch_compat.py`, covering canonical `tools_dispatch` passthrough and validation contracts for `pin_to_ipfs` and `get_from_ipfs` without relying on the shared bootstrap suite.
82. Revalidated focused `ipfs_tools` direct, import-compat, and isolated dispatch coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni002_ipfs_tools.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni238_ipfs_tools_import_compat.py`, and `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni297_ipfs_tools_dispatch_compat.py` (`12 passed`).
83. Revalidated `investigation_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/investigation_tools/native_investigation_tools.py` and added the missing unified bootstrap dispatch regression for `analyze_entities` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
84. Revalidated focused `investigation_tools` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni139_investigation_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 180 deselected`).
85. Hardened contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/development_tools/native_development_tools.py` so development-tool delegate payloads such as `{"status": "success", "success": false, "error": ...}` now infer deterministic `status="error"` while preserving `success=False` across representative code-search, documentation, test-runner, and VSCode CLI wrappers.
86. Extended focused `development_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni132_development_tools.py`, added isolated dispatch compatibility coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni298_development_tools_dispatch_compat.py`, and revalidated import-compat coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni222_development_import_compat.py` (`17 passed`).
87. Revalidated `cli` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/cli/native_cli_tools.py` and added the missing unified bootstrap dispatch regression for `execute_command` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
88. Revalidated focused `cli` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni142_cli_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 179 deselected`).
89. Revalidated `embedding_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/embedding_tools/native_embedding_tools.py` and added the missing unified bootstrap dispatch regression for `generate_embedding` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
90. Revalidated focused `embedding_tools` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni158_embedding_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 185 deselected`).
91. Revalidated `legal_dataset_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/legal_dataset_tools/native_legal_dataset_tools.py` and added the missing unified bootstrap dispatch regression for `scrape_state_laws` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
92. Revalidated focused `legal_dataset_tools` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni138_legal_dataset_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed, 187 deselected`).
93. Hardened contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/auth_tools/native_auth_tools.py` so auth delegate payloads such as `{"status": "success", "success": false, "error": ...}` now infer deterministic `status="error"` while preserving `success=False` across `authenticate_user`, `validate_token`, and `get_user_info`.
94. Extended focused `auth_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni111_auth_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni181_auth_dispatch_compat.py`, and revalidated the unified bootstrap dispatch regression in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`13 passed`, `2 passed`).
95. Hardened contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/web_archive_tools/native_web_archive_tools.py` so representative web-archive delegate payloads such as `{"status": "success", "success": false, "error": ...}` now infer deterministic `status="error"` while preserving `success=False` across representative archive, Common Crawl, GitHub-provider, and unified-search/fetch wrappers.
96. Extended focused `web_archive_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni106_web_archive_tools.py`, added isolated dispatch compatibility coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni299_web_archive_tools_dispatch_compat.py`, and revalidated import-compat coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni208_web_archive_import_compat.py` (`22 passed`).
97. Hardened contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/security_tools/native_security_tools.py` so security delegate payloads such as `{"status": "success", "success": false, "error": ...}` now infer deterministic `status="error"` while preserving `success=False` across `check_access_permission` results that feed direct, batch, and dispatch-driven authorization flows.
98. Extended focused `security_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni127_security_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni182_security_dispatch_compat.py`, and added the missing unified bootstrap dispatch regression in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`15 passed`, `2 passed`).
99. Revalidated `medical_research_scrapers` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/medical_research_scrapers/native_medical_research_scrapers.py` and added the missing unified bootstrap dispatch regression for `scrape_pubmed_medical_research` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
100. Revalidated focused `medical_research_scrapers` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni143_medical_research_scrapers.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed`).
101. Revalidated `search_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/search_tools/native_search_tools.py` and added the missing unified bootstrap dispatch regression for `semantic_search` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
102. Revalidated focused `search_tools` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni165_search_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed`).
103. Revalidated `vector_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/vector_tools/native_vector_tools.py` and added the missing unified bootstrap dispatch regression for `create_vector_index` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
104. Revalidated focused `vector_tools` direct and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni155_vector_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`3 passed`).
105. Revalidated `file_detection_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/file_detection_tools/native_file_detection_tools.py` and added isolated `tools_dispatch` compatibility coverage for representative single-file, batch, and detection-accuracy flows so contradictory failed payloads stay normalized through canonical dispatch.
106. Revalidated focused `file_detection_tools` direct, import-compat, and isolated dispatch coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni117_file_detection_tools.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni217_file_detection_import_compat.py`, and `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni300_file_detection_dispatch_compat.py` (`13 passed`).
107. Hardened contradictory failed-manager normalization in `ipfs_accelerate_py/mcp_server/tools/workflow/native_workflow_tools.py` so native `workflow` delegate payloads such as `{"status": "success", "success": false, "error": ...}` now infer deterministic `status="error"` while preserving `success=False` across list/detail, mutation, and lifecycle wrappers.
108. Extended focused native workflow parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni149_native_workflow_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni301_workflow_dispatch_compat.py`, and added the missing unified bootstrap dispatch regression in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`9 passed`; bootstrap slice `3 passed, 180 deselected`).
109. Hardened contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/ipfs/native_ipfs_tools.py` so native `ipfs` delegate payloads such as `{"status": "success", "success": false, "error": ...}` now infer deterministic `status="error"` while preserving `success=False` across native CID validation and the shared kit-response normalization path used by the IPFS file operations.
110. Extended focused native IPFS parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni148_native_ipfs_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni302_native_ipfs_dispatch_compat.py`, and added the missing unified bootstrap dispatch regression in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`9 passed`; bootstrap slice `1 passed, 184 deselected`).
111. Hardened contradictory failed-manager normalization in `ipfs_accelerate_py/mcp_server/tools/session_tools/native_session_tools.py` so enhanced native `session_tools` delegate payloads such as `{"status": "success", "success": false, "error": ...}` now infer deterministic `status="error"` while preserving `success=False` across create, create/get/update/delete/list/cleanup session management flows, cleanup helpers, and `get_session_state`.
112. Extended focused native session parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni109_session_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni296_session_dispatch_compat.py`, and added the missing unified bootstrap dispatch regression in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`17 passed`; bootstrap slice `2 passed, 182 deselected`).
113. Revalidated native `p2p` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/p2p/native_p2p_tools.py` and added the missing unified bootstrap dispatch regression for `p2p_taskqueue_status` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
114. Revalidated focused native `p2p` direct, isolated dispatch, and unified bootstrap coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni150_native_p2p_tools.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni285_native_p2p_dispatch_compat.py`, and `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`7 passed`; bootstrap slice `1 passed, 185 deselected`).
115. Hardened contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/ipfs_tools/native_ipfs_tools_category.py` so source-compatible `ipfs_tools` delegate payloads such as `{"status": "success", "success": false, "error": ...}` now infer deterministic `status="error"` while preserving `success=False` across `pin_to_ipfs` and `get_from_ipfs` without changing the existing validation or JSON-entrypoint behavior.
116. Extended focused `ipfs_tools` parity coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni002_ipfs_tools.py` and `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni297_ipfs_tools_dispatch_compat.py`, and added the missing unified bootstrap dispatch regression in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`13 passed`; bootstrap slice `1 passed, 186 deselected`).
117. Revalidated `monitoring_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/monitoring_tools/native_monitoring_tools.py` and added the missing unified bootstrap dispatch regression for `get_performance_metrics` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
118. Revalidated focused `monitoring_tools` isolated dispatch coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni266_monitoring_dispatch_compat.py` and the new unified bootstrap slice in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`1 passed`; bootstrap slice `1 passed, 188 deselected`).
119. Revalidated `rate_limiting_tools` contradictory failed-delegate normalization in `ipfs_accelerate_py/mcp_server/tools/rate_limiting_tools/native_rate_limiting_tools_category.py` and added the missing unified bootstrap dispatch regression for alias-category `configure_rate_limits`, `check_rate_limit`, and `manage_rate_limits` so contradictory failed payloads now stay covered through canonical `tools_dispatch`.
120. Revalidated focused `rate_limiting_tools` isolated dispatch coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni292_rate_limiting_tools_dispatch_compat.py` and the new unified bootstrap slice in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`1 passed`; bootstrap slice `1 passed, 188 deselected`).

Exit:

1. High-impact categories at behavior parity threshold with deterministic tests.

## Milestone M3 (Weeks 6-8): Spec Hardening and Ops

1. Profile chapter hardening pass across 8 chapters.
2. Transport abuse and interop expansion.
3. Observability and security interaction tests.

Recent execution (2026-03-08):

1. Expanded `ipfs_accelerate_py/mcp/tests/test_mcp_transport_mcp_p2p_handler_limits.py` with a focused mixed-version transport regression proving malformed registry negotiation metadata is sanitized during `initialize` while preserving follow-on alias `tools.list` and `tools.call` compatibility in the same session.
2. Revalidated the focused transport compatibility slice in `ipfs_accelerate_py/mcp/tests/test_mcp_transport_mcp_p2p_handler_limits.py` (`2 passed, 17 deselected`).
3. Added a focused unified-dispatch observability/security interaction regression in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` proving policy-audit allow/deny outcomes remain aligned with audit-metrics bridge forwarding and monitoring counters in `tools_runtime_metrics` (`3 passed, 147 deselected`).

Exit:

1. Chapter-level hardening test suites green in CI.

## Milestone M4 (Weeks 9-10): Cutover Readiness

1. Validate compatibility facade rollback path.
2. Run release candidate matrix across transport and profile features.
3. Freeze migration deltas and publish cutover checklist. ✅

Recent execution (2026-03-08):

1. Extended `ipfs_accelerate_py/mcp/server.py` with compatibility-facade usage telemetry so legacy-wrapper fallback, unified-bridge handoff, dry-run validation, rollback forcing, and bridge-failure fallback all emit deterministic `_mcp_facade_telemetry` metadata.
2. Expanded `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni007_cutover_rollback.py` to validate facade telemetry snapshots and aggregate counters for dry-run success, dry-run failure, force-rollback, and unified-bridge handoff paths (`4 passed`).
3. Added a focused cutover precedence regression in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni007_cutover_rollback.py` proving force-rollback takes precedence even when cutover dry-run is enabled, while still preserving deterministic dry-run intent telemetry and legacy-fallback metadata (`4 passed, 3 deselected`).
4. Revalidated process-level and direct-entry startup behavior after the canonical-default startup flip with focused subprocess, p2p bridge, standalone-app, direct component, and script-style initialization coverage (`3 passed, 6 deselected`; `5 passed, 5 deselected`; script smoke paths completed successfully with populated tool/resource/prompt inventories).
5. Added focused FastAPI integration/helper coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_transport_process_level.py` for `initialize_mcp_server()` and `integrate_mcp_with_fastapi()`, then revalidated those helpers plus canonical FastAPI facade delegation in `ipfs_accelerate_py/mcp/tests/test_mcp_server_fastapi_service.py` (`9 passed, 2 deselected`).
6. Added focused CLI startup coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_cli.py`, confirming the CLI still constructs the compatibility facade and delegates to `mcp_server.run()` with the parsed host/port contract in normal and `--dev` modes after the canonical-default startup change (`2 passed`).
7. Ran an aggregate post-cutover validation bundle across rollback telemetry, legacy bootstrap compatibility, subprocess/FastAPI entrypoints, process helpers, p2p bridge dispatch, and CLI startup (`28 passed, 148 deselected`), confirming the canonical-default startup change remains stable across the main compatibility-facade entry surfaces.
8. Ran a focused release-candidate matrix across cutover, transport entrypoints, MCP+p2p handler interop, and representative MCP++ profile chapters (`test_mcp_server_uni007_cutover_rollback.py`, `test_mcp_transport_process_level.py`, `test_mcp_transport_mcp_p2p_handler_limits.py`, `test_mcp_server_transport_e2e_matrix.py`, `test_mcp_server_mcplusplus_idl.py`, `test_mcp_server_mcplusplus_artifacts.py`, `test_mcp_server_mcplusplus_ucan.py`, `test_mcp_server_mcplusplus_policy.py`, `test_mcp_server_mcplusplus_event_dag.py`, `test_mcp_server_mcplusplus_risk_scheduler.py`), with the matrix completing successfully (`120 passed`).
9. Published [MCP_CUTOVER_CHECKLIST.md](MCP_CUTOVER_CHECKLIST.md) to freeze the cutover delta set, record the approved operational gates, and anchor the release-candidate evidence bundle used for canonical-default startup approval.

Exit:

1. Cutover gate approved.

## 8. Cutover and Deprecation Plan

1. Default startup path points to `mcp_server`.
2. Keep `mcp` facade for one release window.
3. Instrument facade usage telemetry. ✅
4. Published cutover checklist: [MCP_CUTOVER_CHECKLIST.md](MCP_CUTOVER_CHECKLIST.md). ✅
5. Deprecate shim runtime behavior in phases:
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
   - UCAN + temporal policy + artifacts + risk scheduler. ✅
2. Expand transport abuse and mixed-peer compatibility tests. ✅
3. Run cutover dry-run with compatibility facade rollback scenario. ✅

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
   - Status: COMPLETE (2026-03-06)
   - Evidence:
     - `test_tools_dispatch_frontier_execution_binds_to_workflow_scheduler`
     - `test_tools_dispatch_frontier_execution_binds_to_task_queue_fallback`
     - `test_tools_dispatch_result_cache_factory_consumed_on_cache_hit`
     - `test_tools_dispatch_peer_registry_factory_consumed_for_probe`
   - Acceptance:
     - Dispatch path tests assert service factory invocation and usage.
     - No behavior regression in unified bootstrap suite.
   - Depends on: none.

2. `UNI-002` IPFS Tools Deep Parity Wave
   - Scope: Close operation/schema deltas in `ipfs_tools` category.
   - Target files: `ipfs_accelerate_py/mcp_server/tools/ipfs_tools/*`, `ipfs_accelerate_py/mcp/tests/*ipfs*`
   - Status: COMPLETE (2026-03-06)
   - Evidence:
     - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni002_ipfs_tools.py` (`10 passed`)
     - `test_ipfs_tools_discovery_schema_and_dispatch_parity` in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (`1 passed`)
   - Acceptance:
     - Representative source operations and schemas mapped and passing.
     - `SPEC_GAP_MATRIX` IPFS rows updated with evidence.
   - Depends on: `UNI-001`.

3. `UNI-003` Workflow Tools Deep Parity Wave
   - Scope: Complete workflow operation parity and schema behavior.
   - Target files: `ipfs_accelerate_py/mcp_server/tools/workflow_tools/*`, `ipfs_accelerate_py/mcp/tests/*workflow*`
   - Status: COMPLETE (2026-03-06)
   - Evidence:
     - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni134_workflow_tools.py`
     - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni149_native_workflow_tools.py`
     - `test_workflow_tools_discovery_schema_and_dispatch_parity` in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py`
     - `test_workflow_tools_expanded_p2p_parity_operations` in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py`
   - Acceptance:
     - End-to-end workflow dispatch parity for source-equivalent operations.
     - Deterministic tests for status/submit/next/edge cases.
   - Depends on: `UNI-001`.

4. `UNI-004` P2P and MCP++ Tool Surface Deep Parity
   - Scope: Fill behavior gaps in `p2p_tools` and `mcplusplus` tool category.
   - Target files: `ipfs_accelerate_py/mcp_server/tools/p2p_tools/*`, `ipfs_accelerate_py/mcp_server/tools/mcplusplus/*`, tests under `ipfs_accelerate_py/mcp/tests/`
   - Status: COMPLETE (2026-03-06)
   - Evidence:
     - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni135_p2p_tools.py`
     - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni147_mcplusplus_tools.py`
     - `test_p2p_tools_discovery_schema_and_dispatch_parity` in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py`
     - `test_p2p_tools_expanded_parity_operations` in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py`
     - `test_mcplusplus_tools_engine_status_operations` in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py`
   - Acceptance:
     - Remote-call/workflow-helper parity validated.
     - Taskqueue/workflow operation coverage expanded.
   - Depends on: `UNI-001`.

5. `UNI-005` Shim Convergence Finalization
   - Scope: Remove remaining duplicated fallback/resolver behavior in `mcplusplus_module` boundaries.
   - Target files: `ipfs_accelerate_py/mcplusplus_module/p2p/bootstrap.py`, `ipfs_accelerate_py/mcplusplus_module/p2p/peer_registry.py`, tests under `ipfs_accelerate_py/mcplusplus_module/tests/`
   - Status: COMPLETE (2026-03-06)
   - Evidence:
     - `ipfs_accelerate_py/mcplusplus_module/tests/test_tool_adapters.py` (`24 passed`)
     - Includes canonical-first resolver fallback checks and explicit optional-dependency compatibility-stub contracts.
     - Includes shared-helper delegation checks for `bootstrap`/`peer_registry` detection paths.
   - Acceptance:
     - Canonical-first delegation where applicable.
     - Optional dependency boundaries use explicit compatibility contracts.
   - Depends on: none.

6. `UNI-006` Transport Interop and Abuse Regression Expansion
   - Scope: Strengthen `mcp+p2p` framing, init-order, and abuse-path matrix.
   - Target files: `ipfs_accelerate_py/mcp/tests/test_mcp_transport_*`, `ipfs_accelerate_py/p2p_tasks/*`
   - Status: COMPLETE (2026-03-06)
   - Evidence:
     - `ipfs_accelerate_py/mcp/tests/test_mcp_transport_p2p_framing_limits.py`
     - `ipfs_accelerate_py/mcp/tests/test_mcp_transport_mcp_p2p_handler_limits.py`
     - `ipfs_accelerate_py/mcp/tests/test_mcp_transport_mcp_p2p_client_limits.py`
     - `ipfs_accelerate_py/mcp/tests/test_mcp_server_transport_e2e_matrix.py`
     - `ipfs_accelerate_py/mcp/tests/test_mcp_server_transport_parity.py`
     - Targeted run result: `37 passed`.
   - Acceptance:
     - Added regressions for mixed-version peer behavior.
     - Required transport lanes remain green.
   - Depends on: none.

7. `UNI-007` Cutover Dry-Run and Rollback Verification
   - Scope: Exercise runtime default switch and rollback procedure in CI-compatible flow.
   - Target files: `ipfs_accelerate_py/mcp/server.py`, startup scripts/config, tests/workflow docs
   - Status: COMPLETE (2026-03-06)
   - Evidence:
     - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni007_cutover_rollback.py` (`3 passed`)
     - Covers dry-run success path, dry-run failure fallback path, and force-legacy rollback override behavior.
   - Acceptance:
     - Dry-run validates canonical path as default.
     - Rollback path tested and documented.
   - Depends on: `UNI-001` to `UNI-006`.

### P0/P1 Module-Port Issues (Derived from 5.0.12)

1. `UNI-008` Runtime Pipeline and Context Module Convergence
    - Scope: Merge source runtime pipeline/context modules into canonical dispatch path.
    - Target files: `ipfs_accelerate_py/mcp_server/runtime_router.py`, `ipfs_accelerate_py/mcp_server/server.py`, new/merged helpers for `dispatch_pipeline` and `server_context` semantics.
    - Status: COMPLETE (2026-03-06)
    - Evidence:
       - `ipfs_accelerate_py/mcp/tests/test_mcp_server_dispatch_pipeline.py`
       - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni008_runtime_context.py`
       - Targeted run result: `7 passed`.
    - Acceptance:
       - Source `dispatch_pipeline.py`, `server_context.py`, and `mcp_interfaces.py` behaviors are represented in canonical runtime.
       - Unified bootstrap tests confirm deterministic routing and context metadata behavior.
    - Depends on: `UNI-001`.

2. `UNI-009` Canonical HTTP Service Entry Convergence
    - Scope: Port `fastapi_service.py` and `fastapi_config.py` behavior into canonical runtime, with compatibility facade where required.
    - Target files: canonical HTTP entry module(s) under `ipfs_accelerate_py/mcp_server/`, facade updates in `ipfs_accelerate_py/mcp/*`, transport tests.
    - Status: COMPLETE (2026-03-06)
    - Evidence:
       - `ipfs_accelerate_py/mcp/tests/test_mcp_server_fastapi_service.py`
       - Targeted run result: `4 passed`.
    - Acceptance:
       - HTTP startup path is canonical and deterministic.
       - Transport parity tests validate HTTP lane behavior.
    - Depends on: `UNI-008`.

3. `UNI-010` MCP+p2p Transport Adapter Consolidation
    - Scope: Reconcile `mcp_p2p_transport.py`, `trio_adapter.py`, and `register_p2p_tools.py` semantics into canonical transport wiring.
    - Target files: `ipfs_accelerate_py/p2p_tasks/mcp_p2p.py`, `ipfs_accelerate_py/mcp_server/mcplusplus/p2p_framing.py`, `ipfs_accelerate_py/mcp_server/server.py`.
    - Status: COMPLETE (2026-03-06)
    - Evidence:
       - `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcp_p2p_transport.py`
       - `ipfs_accelerate_py/mcp/tests/test_mcp_server_trio_adapter.py`
       - `ipfs_accelerate_py/mcp/tests/test_mcp_server_register_p2p_tools.py`
       - Targeted run result: `12 passed`.
    - Acceptance:
       - No duplicate registration or transport adapter branches.
       - Existing `mcp+p2p` limit and handshake tests remain green.
    - Depends on: `UNI-004`, `UNI-006`.

4. `UNI-011` Policy/Delegation Legacy Surface Adaptation
    - Scope: Adapt source `ucan_delegation.py`, `temporal_policy.py`, and `temporal_deontic_mcp_server.py` surfaces to canonical MCP++ policy/delegation modules.
    - Target files: `ipfs_accelerate_py/mcp_server/mcplusplus/delegation.py`, `ipfs_accelerate_py/mcp_server/mcplusplus/policy_engine.py`, `ipfs_accelerate_py/mcp_server/server.py`.
    - Status: COMPLETE (2026-03-06)
    - Evidence:
       - `ipfs_accelerate_py/mcp/tests/test_mcp_server_policy_delegation_legacy_adapters.py`
       - `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_ucan.py`
       - `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_policy.py`
       - Targeted run result: `32 passed`.
    - Acceptance:
       - Legacy surface behavior is represented without introducing duplicate runtimes.
       - UCAN + temporal policy tests pass with stable decision artifacts.
    - Depends on: `SPEC-204`, `SPEC-205`.

5. `UNI-012` Artifact/IDL/Event Module Adaptation
    - Scope: Fold source `cid_artifacts.py`, `interface_descriptor.py`, and `event_dag.py` surfaces into canonical MCP++ implementations.
    - Target files: `ipfs_accelerate_py/mcp_server/mcplusplus/artifacts.py`, `ipfs_accelerate_py/mcp_server/mcplusplus/idl_registry.py`, `ipfs_accelerate_py/mcp_server/mcplusplus/event_dag.py`.
    - Status: COMPLETE (2026-03-06)
    - Evidence:
       - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni012_legacy_adapters.py`
       - `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_artifacts.py`
       - `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_idl.py`
       - `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_event_dag.py`
       - Targeted run result: `32 passed`.
    - Acceptance:
       - Source-equivalent interfaces are reachable through canonical APIs.
       - Artifact, IDL, and Event DAG tests remain deterministic.
    - Depends on: `SPEC-202`, `SPEC-203`, `SPEC-206`.

6. `UNI-013` P2P Service Manager and Registry Adapter Merge
    - Scope: Port `p2p_service_manager.py` and `p2p_mcp_registry_adapter.py` semantics into canonical p2p service boundary.
    - Target files: `ipfs_accelerate_py/p2p_tasks/service.py`, related canonical bootstrap wiring.
    - Status: COMPLETE (2026-03-06)
    - Evidence:
       - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni013_p2p_adapters.py`
       - `ipfs_accelerate_py/mcp/tests/test_mcp_transport_process_level.py`
       - `ipfs_accelerate_py/mcp/tests/test_mcp_transport_trio_p2p_networked.py`
       - Targeted run result: `10 passed`.
    - Acceptance:
       - P2P service lifecycle paths are canonical-owned.
       - Process-level and networked p2p tests validate integration.
    - Depends on: `UNI-010`.

7. `UNI-014` Legacy Entry Facade Normalization
    - Scope: Normalize source entry surfaces (`__main__.py`, `simple_server.py`, `standalone_server.py`, `client.py`) as compatibility facades only.
    - Target files: `ipfs_accelerate_py/mcp/__main__.py`, `ipfs_accelerate_py/mcp/standalone.py`, `ipfs_accelerate_py/mcp/integration.py`.
    - Status: COMPLETE (2026-03-06)
    - Evidence:
       - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni014_entry_facades.py`
       - `ipfs_accelerate_py/mcp/tests/test_mcp_transport_subprocess_contracts.py`
       - Targeted run result: `11 passed`.
    - Acceptance:
       - Legacy entrypoints route to canonical runtime paths.
       - Startup and subprocess contract tests remain stable.
    - Depends on: `UNI-007`.

8. `UNI-015` Validation and Logging Surface Consolidation
    - Scope: Consolidate `validators.py` and `logger.py` semantics into canonical runtime modules without standalone runtime duplication.
    - Target files: `ipfs_accelerate_py/mcp_server/runtime_router.py`, `ipfs_accelerate_py/mcp_server/server.py`, category modules that enforce schemas.
    - Status: COMPLETE (2026-03-06)
    - Evidence:
       - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni015_validation_logging.py`
       - `ipfs_accelerate_py/mcp/tests/test_mcp_server_dispatch_flag_coercion.py`
       - Targeted run result: `9 passed`.
    - Acceptance:
       - Validation and logging behavior is deterministic and centralized.
       - Error-envelope and invalid-input tests pass across representative categories.
    - Depends on: `UNI-008`.

9. `UNI-016` Deferred Source Module Governance
    - Scope: Track deferred modules (`enterprise_api.py`, `grpc_transport.py`, `nl_ucan_policy.py`, `trio_bridge.py`, `compliance_checker.py`, `investigation_mcp_client.py`) with explicit phase/rationale.
    - Target files: `mcpplusplus/SPEC_GAP_MATRIX.md`, this plan, optional follow-on backlog docs.
    - Status: COMPLETE (2026-03-06)
    - Evidence:
       - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni016_deferred_governance_docs.py`
       - Targeted run result: `4 passed`.
    - Acceptance:
       - Each deferred module has rationale, risk, and target milestone.
       - No deferred module is treated as implicitly complete.
    - Depends on: none.

   Deferred governance ledger (`UNI-016` baseline):

   | Module | Defer Rationale | Primary Risk if Untracked | Target Milestone |
   | --- | --- | --- | --- |
   | `enterprise_api.py` | Compatibility imports for enterprise API utilities are now canonically represented via adapter delegation. | Enterprise integrations could break if canonical adapter exports drift from source exports. | `UNI-201` COMPLETE: canonical adapter shipped at `ipfs_accelerate_py/mcp_server/enterprise_api.py` with focused coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni207_deferred_module_adapters.py`. |
   | `grpc_transport.py` | Compatibility imports for gRPC transport utilities are now canonically represented via adapter delegation. | Transport integrations could break if canonical adapter exports drift from source exports. | `UNI-202` COMPLETE: canonical adapter shipped at `ipfs_accelerate_py/mcp_server/grpc_transport.py` with focused coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni207_deferred_module_adapters.py`. |
   | `nl_ucan_policy.py` | Compatibility imports for NL-UCAN policy utilities are now canonically represented via adapter delegation. | Policy-language integrations could break if canonical adapter exports drift from source exports. | `UNI-203` COMPLETE: canonical adapter shipped at `ipfs_accelerate_py/mcp_server/nl_ucan_policy.py` with focused coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni207_deferred_module_adapters.py`. |
   | `trio_bridge.py` | Compatibility bridge helpers are legacy adjuncts and not canonical runtime ownership targets. | Duplicate bridge logic may silently reintroduce non-canonical runtime routing paths. | `UNI-204` COMPLETE: post-`SPEC-208` bridge audit confirms canonical runtime avoids source `trio_bridge.py` imports and keeps bridge behavior on canonical compatibility surfaces. |
   | `compliance_checker.py` | Compatibility imports for compliance-checker utilities are now canonically represented via adapter delegation while preserving source behavior. | Legacy import consumers could break if adapter symbols drift from source exports. | `UNI-205` COMPLETE: canonical adapter shipped at `ipfs_accelerate_py/mcp_server/compliance_checker.py` with focused coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni205_compliance_checker_adapter.py`. |
   | `investigation_mcp_client.py` | Compatibility imports for investigation MCP client utilities are now canonically represented via adapter delegation while preserving source behavior. | Dashboard/client integrations could break if adapter symbols drift from source exports. | `UNI-206` COMPLETE: canonical adapter shipped at `ipfs_accelerate_py/mcp_server/investigation_mcp_client.py` with focused coverage in `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni206_investigation_client_adapter.py`. |

### P1 Issues (Hardening and Breadth)

1. `UNI-101` Security Tools Behavior Parity Expansion
   - Scope: Expand auth/security tools beyond currently implemented baseline.
   - Target files: `ipfs_accelerate_py/mcp_server/tools/security_tools/*`, `.../auth_tools/*`, related tests
   - Status: COMPLETE (2026-03-06)
   - Evidence:
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni101_security_tools.py`
      - Targeted run result: `4 passed`.
   - Acceptance: representative operation and schema parity coverage added.

2. `UNI-102` Monitoring and Observability Behavior Parity Expansion
   - Scope: Expand advanced monitoring/alert/diagnostic operations.
   - Target files: `ipfs_accelerate_py/mcp_server/tools/monitoring_tools/*`, `.../alert_tools/*`, tests
   - Status: COMPLETE (2026-03-06)
   - Evidence:
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni102_monitoring_tools.py`
      - Targeted run result: `5 passed`.
   - Acceptance: advanced telemetry and diagnostics calls validated.

3. `UNI-103` Dataset and Embedding Pipeline Parity Expansion
   - Scope: Expand `dataset_tools` + `embedding_tools` behavior-level parity.
   - Target files: `ipfs_accelerate_py/mcp_server/tools/dataset_tools/*`, `.../embedding_tools/*`, tests
   - Status: COMPLETE (2026-03-06)
   - Evidence:
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni103_dataset_logic_tools.py`
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni103_embedding_tools.py`
      - Targeted run result: `9 passed`.
   - Acceptance: source-equivalent conversion and endpoint-management flows validated.

4. `UNI-104` Vector/Search/Storage Integration Parity Expansion
   - Scope: Expand cross-category backend orchestration parity.
   - Target files: `ipfs_accelerate_py/mcp_server/tools/vector_tools/*`, `.../search_tools/*`, `.../storage_tools/*`, tests
   - Status: COMPLETE (2026-03-06)
   - Evidence:
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni104_vector_search_storage_tools.py`
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni104_storage_tools.py`
      - Targeted run result: `10 passed`.
   - Acceptance: integration behavior and schema parity for representative flows.

5. `UNI-105` PDF/Graph/Logic Advanced Surface Parity Expansion
   - Scope: Close high-value advanced operations in document/graph/logic categories.
   - Target files: `ipfs_accelerate_py/mcp_server/tools/pdf_tools/*`, `.../graph_tools/*`, `.../logic_tools/*`, tests
   - Status: COMPLETE (2026-03-06)
   - Evidence:
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni105_pdf_tools.py`
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni105_graph_tools.py`
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni105_logic_tools.py`
      - Targeted run result: `15 passed`.
   - Acceptance: advanced ingestion/query/reasoning flows covered.

### Spec Chapter Hardening Issues

1. `SPEC-201` Profiles Negotiation Hardening
   - Chapter: `mcp++-profiles-draft.md`
   - Status: COMPLETE (2026-03-06)
   - Evidence:
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py`
      - `ipfs_accelerate_py/mcp/tests/test_mcp_transport_mcp_p2p_handler_limits.py`
      - Targeted run result: `6 passed`.
   - Acceptance: downgrade/unknown-profile regression cases and capability snapshot checks.

2. `SPEC-202` MCP-IDL Stability Corpus
   - Chapter: `mcp-idl.md`
   - Status: COMPLETE (2026-03-06)
   - Evidence:
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_idl.py`
      - Targeted run result: `11 passed`.
   - Acceptance: canonicalization stability corpus and compat matching regression suite.

3. `SPEC-203` Artifact Durability and Replay
   - Chapter: `cid-native-artifacts.md`
   - Status: COMPLETE (2026-03-06)
   - Evidence:
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_artifacts.py`
      - Targeted run result: `12 passed`.
   - Acceptance: durable store path and replay integrity tests beyond in-memory behavior.

4. `SPEC-204` UCAN Verification Vector Expansion
   - Chapter: `ucan-delegation.md`
   - Status: COMPLETE (2026-03-07)
   - Evidence:
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_ucan.py`
      - Targeted run result: `24 passed`.
   - Acceptance: expanded signature/caveat/proof-link vectors and deny/allow telemetry checks.

5. `SPEC-205` Temporal Policy Lifecycle Hardening
   - Chapter: `temporal-deontic-policy.md`
   - Status: COMPLETE (2026-03-07)
   - Evidence:
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_policy.py`
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py`
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_transport_e2e_matrix.py`
      - Focused policy-engine run result: `7 passed`.
      - Focused unified-bootstrap temporal policy slice: `5 passed`.
      - Focused transport matrix result: `6 passed`.
   - Acceptance: obligation/deadline/version migration tests and transport parity assertions.

6. `SPEC-206` Event DAG Scale and Conflict Scenarios
   - Chapter: `event-dag-ordering.md`
   - Status: COMPLETE (2026-03-07)
   - Evidence:
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_event_dag.py`
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py`
      - Focused Event DAG run result: `15 passed`.
   - Acceptance: large-DAG replay/rollback and fork/conflict handling tests.

7. `SPEC-207` Risk Frontier + Consensus Enhancements
   - Chapter: `risk-scheduling.md`
   - Status: COMPLETE (2026-03-06)
   - Evidence:
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_risk_scheduler.py`
      - Combined targeted SPEC-205/SPEC-206/SPEC-207 run result: `20 passed`.
   - Acceptance: load/retry frontier binding tests and optional consensus signal integration tests.

8. `SPEC-208` mcp+p2p Mixed-Version Interop Matrix
   - Chapter: `transport-mcp-p2p.md`
   - Status: COMPLETE (2026-03-06)
   - Evidence:
      - `ipfs_accelerate_py/mcp/tests/test_mcp_transport_p2p_framing_limits.py`
      - `ipfs_accelerate_py/mcp/tests/test_mcp_transport_mcp_p2p_handler_limits.py`
      - `ipfs_accelerate_py/mcp/tests/test_mcp_transport_mcp_p2p_client_limits.py`
      - `ipfs_accelerate_py/mcp/tests/test_mcp_server_transport_e2e_matrix.py`
      - Targeted run result: `37 passed`.
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
- [x] Tests assert service factory invocation and runtime usage.
- [x] No regressions in unified bootstrap tests.
- [x] Evidence references added to `mcpplusplus/SPEC_GAP_MATRIX.md`.

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
- [x] Representative source operations are implemented or explicitly deferred with rationale.
- [x] `tools_get_schema` parity validated for representative tools.
- [x] `SPEC_GAP_MATRIX` IPFS rows updated with evidence.

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
- [x] Workflow dispatch parity for representative source operations.
- [x] Edge-case tests for arguments and error envelopes.
- [x] Evidence links updated in conformance docs.

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
- [x] Representative source-equivalent p2p operations dispatch correctly.
- [x] MCP++ tool wrappers expose expected schemas and behavior.
- [x] Matrix/checklist evidence updated.

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
- [x] Canonical-first delegation or explicit compatibility rationale in all touched boundaries.
- [x] No raw `None` export surfaces at module boundaries where explicit stubs are required.
- [x] Contract tests updated and passing.

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
- [x] Additional regression tests merged for interop/abuse scenarios.
- [x] Default transport lane remains green.
- [x] libp2p lane remains green and required.

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
- [x] Canonical default path validated in dry-run.
- [x] Rollback scenario tested and documented.
- [x] Conformance docs updated with cutover evidence.

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
- [x] Negotiation behavior remains additive and backward compatible.
- [x] Unknown profiles produce deterministic fallback.
- [x] Conformance evidence updated.

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
- [x] Descriptor CIDs are deterministic for corpus inputs.
- [x] Compatibility matching remains stable under case/whitespace/version variants.
- [x] Evidence updated in matrix/checklist.

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
- [x] Durable artifact retrieval passes deterministic integrity checks.
- [x] Replay reconstructs chain correctly.
- [x] Emission policy is deterministic across modes.

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
- [x] Expanded positive and negative UCAN vectors pass deterministically.
- [x] Denial behavior is explicit and auditable.
- [x] Conformance evidence updated.

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
- [x] Obligation lifecycle behavior is deterministic.
- [x] Decision CID persistence remains stable across dispatch modes/transports.
- [x] Evidence links updated.

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
- [x] Replay and rollback paths are deterministic for larger graphs.
- [x] Conflict behavior is explicit and tested.
- [x] Snapshot compatibility preserved.

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
- [x] Frontier execution binding deterministic under load/retries.
- [x] Consensus signal path remains optional and non-breaking.
- [x] Evidence updated in conformance docs.

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
- [x] Mixed-version matrix behavior is deterministic.
- [x] Abuse-resistance assertions expanded and passing.
- [x] CI evidence remains green in required lanes.

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
- [x] Representative missing operations implemented or deferred with rationale.
- [x] `tools_get_schema` and dispatch parity tests added.
- [x] Evidence links updated in conformance docs.

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
- [x] Advanced monitoring operations behave deterministically.
- [x] Alert/diagnostic schemas and outputs are parity-validated.
- [x] Matrix/checklist evidence updated.

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
- [x] Representative dataset and embedding deltas closed.
- [x] Edge-case arguments and error envelopes parity-tested.
- [x] Evidence links updated in conformance artifacts.

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
- [x] Representative integration flows are parity-validated.
- [x] Deterministic schema and result contracts are asserted.
- [x] Matrix rows/evidence are updated.

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
- [x] Advanced operations implemented or deferred with rationale.
- [x] Deterministic behavior and schema parity tests added.
- [x] Evidence updated in conformance docs.

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

### 22.8 Safe Bulk Issue Creation (Idempotent)

Use this pattern to avoid duplicate issue creation when running backlog setup repeatedly.

```bash
issue_exists() {
  local title="$1"
  gh issue list --repo "$REPO" --search "in:title \"$title\" state:open" --limit 1 --json title \
    | grep -q '"title"'
}

create_issue_if_missing() {
  local title="$1"
  local body_file="$2"
  local labels="$3"
  local milestone="$4"

  if issue_exists "$title"; then
    echo "[skip] $title"
    return 0
  fi

  gh issue create \
    --repo "$REPO" \
    --title "$title" \
    --body-file "$body_file" \
    --label "$labels" \
    --milestone "$milestone"
}
```

Example usage:

```bash
create_issue_if_missing \
  "UNI-001: Prove canonical dispatch consumes _unified_services" \
  ./tmp/UNI-001.md \
  "priority:p0,type:parity" \
  "M1-Convergence-Hardening"

create_issue_if_missing \
  "SPEC-204: Expand UCAN verification vectors and deny matrix" \
  ./tmp/SPEC-204.md \
  "priority:p0,type:spec,type:security" \
  "M3-Spec-Chapter-Hardening"
```

### 22.9 Milestone and Label Defaults by Issue Family

1. `UNI-001` to `UNI-005`:
   - milestone: `M1-Convergence-Hardening`
   - labels: `priority:p0,type:parity`
2. `UNI-002` to `UNI-004` (if run as deep parity wave):
   - milestone: `M2-Deep-Tool-Parity`
   - labels: `priority:p0,type:parity`
3. `UNI-006`:
   - milestone: `M3-Spec-Chapter-Hardening`
   - labels: `priority:p0,type:transport`
4. `UNI-007`:
   - milestone: `M4-Cutover-Rollback-Validation`
   - labels: `priority:p0,type:cutover`
5. `SPEC-201` to `SPEC-208`:
   - milestone: `M3-Spec-Chapter-Hardening`
   - labels: `priority:p0,type:spec` (+ domain label where needed)
6. `UNI-101` to `UNI-105`:
   - milestone: `M2-Deep-Tool-Parity`
   - labels: `priority:p1,type:parity`
