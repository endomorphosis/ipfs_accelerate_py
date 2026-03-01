# MCP Server Unification Master Plan

## 1. Goal

Create a single canonical MCP implementation at `ipfs_accelerate_py/mcp_server` by:

- Porting feature-complete capabilities from `ipfs_datasets_py/ipfs_datasets_py/mcp_server`.
- Merging MCP++ runtime features (task queue, workflow engine, peer discovery, caching, scheduling).
- Aligning implementation and tests to the MCP++ spec target at `ipfs_accelerate_py/mcpplusplus`.
- Preserving backward compatibility through feature-flagged bridge/cutover stages.

## 2. Path Reality Check (Current Workspace)

Verified in this workspace:

- Canonical target runtime package exists: `ipfs_accelerate_py/mcp_server`.
- Full source implementation exists: `ipfs_datasets_py/ipfs_datasets_py/mcp_server`.
- MCP++ implementation set is currently present under source server at:
  - `ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus/*`
- Existing target package currently contains core framework plus native Wave A tools:
  - `ipfs_accelerate_py/mcp_server/tools/ipfs/native_ipfs_tools.py`
  - `ipfs_accelerate_py/mcp_server/tools/workflow/native_workflow_tools.py`
  - `ipfs_accelerate_py/mcp_server/tools/p2p/native_p2p_tools.py`

Action item from this path check:

- Establish `ipfs_accelerate_py/mcpplusplus` as the canonical spec directory if not present, and keep spec docs versioned there.
- Stage MCP++ runtime internals under `ipfs_accelerate_py/mcp_server/mcplusplus/` (or equivalent subpackage) as target runtime primitives.

## 3. Current Progress Snapshot

Already completed in target `ipfs_accelerate_py/mcp_server`:

- Core framework modules ported and active:
  - `configs.py`, `exceptions.py`, `runtime_router.py`, `hierarchical_tool_manager.py`, `registration_adapter.py`, `tool_registry.py`, `tool_metadata.py`.
- Bridge/bootstrap path implemented and feature-flagged.
- Unified meta-tools implemented:
  - `tools_list_categories`, `tools_list_tools`, `tools_get_schema`, `tools_dispatch`, `tools_runtime_metrics`.
- Native Wave A migration advanced for IPFS, workflow, and p2p.
- Unified bootstrap regression suite currently green at 45 tests.

## 4. Target Architecture (End State)

### 4.1 Runtime and Package Structure

- Canonical runtime: `ipfs_accelerate_py/mcp_server`
- Compatibility facade: `ipfs_accelerate_py/mcp` (temporary)
- Spec and conformance docs: `ipfs_accelerate_py/mcpplusplus`
- MCP++ runtime internals:
  - `ipfs_accelerate_py/mcp_server/mcplusplus/bootstrap.py`
  - `ipfs_accelerate_py/mcp_server/mcplusplus/executor.py`
  - `ipfs_accelerate_py/mcp_server/mcplusplus/task_queue.py`
  - `ipfs_accelerate_py/mcp_server/mcplusplus/workflow_*`
  - `ipfs_accelerate_py/mcp_server/mcplusplus/peer_*`
  - `ipfs_accelerate_py/mcp_server/mcplusplus/result_cache.py`

### 4.2 Control Plane

- Single hierarchical tool registry and dispatch path.
- Runtime router with explicit runtime metadata and timeout semantics.
- One registration path per tool name (no duplicates across legacy/native).

### 4.3 Execution Plane

- FastAPI/stdio/trio-p2p runtime modes normalized behind unified router.
- MCP++ queue, scheduler, DAG, cache, and peer registry integrated as shared services.

## 5. Scope Inventory

Port and merge scope includes:

- Core server lifecycle and transports (`server.py`, `standalone_server.py`, `fastapi_service.py`, transport adapters).
- Dispatch and registry internals (`dispatch_pipeline.py`, `hierarchical_tool_manager.py`, metadata layers).
- Security and policy subsystems (auth/policy/audit/vault/risk).
- Observability and reliability subsystems (monitoring, tracing, metrics exporters).
- MCP++ runtime primitives (`mcplusplus/*`).
- Tool categories under `ipfs_datasets_py/ipfs_datasets_py/mcp_server/tools/*`.

Out of scope for initial cutover:

- Deleting legacy facade immediately.
- Big-bang replacement of all categories in one release.

## 6. Workstreams

### W1. Spec and Conformance Baseline

- Create/refresh `ipfs_accelerate_py/mcpplusplus`.
- Add/maintain:
  - `README.md`
  - `CONFORMANCE_CHECKLIST.md`
  - `SPEC_GAP_MATRIX.md` (required/implemented/tested per capability).
- Treat checklist entries as testable requirements with explicit evidence links.

### W2. Core Runtime Convergence

- Finish parity for server startup, runtime routing, transport mode selection, and error taxonomy.
- Normalize config model and environment flags.
- Ensure bridge mode cannot recurse and has deterministic precedence.

### W3. MCP++ Runtime Merge

- Port `mcplusplus` runtime modules from source into target `mcp_server/mcplusplus`.
- Introduce adapters for any existing in-target alternatives.
- Unify task queue and workflow engine interfaces behind one abstraction.

### W4. Tool Surface Port (Wave Strategy)

- Continue native tool migration by category waves.
- Each migrated tool must include:
  - schema parity
  - deterministic `tools_dispatch` test
  - legacy override skip rule where needed
  - runtime metadata tags

### W5. Transport Parity

- Ensure parity across stdio/http/trio-p2p execution paths.
- Define transport behavior contract and compatibility tests.

### W6. Security and Policy Parity

- Port and validate auth, policy, audit, and secrets integrations.
- Add conformance tests for policy enforcement and audit trail integrity.

### W7. Observability and Reliability

- Port metrics/logging/tracing parity.
- Add SLO dashboards and timeout/error burn-rate alerts.

### W8. Compatibility and Cutover

- Keep `mcp` facade as compatibility layer until gates pass.
- Gradually move defaults to unified `mcp_server`.
- Publish deprecation schedule and rollback instructions.

## 7. Phased Delivery Plan

### Phase 0: Baseline and Planning (Now)

- Path inventory and scope map complete.
- Existing unification baseline acknowledged.
- Master plan and gap matrix authored.

Exit criteria:

- Approved migration plan with owner/date per phase.
- Spec checklist and gap matrix created/updated.

### Phase 1: Core and Runtime Hardening

- Complete core parity for runtime router/transport selection/server lifecycle.
- Stabilize bridge/bootstrap behavior.

Exit criteria:

- Core parity tests pass.
- Runtime mode compatibility tests pass for all configured modes.

### Phase 2: MCP++ Runtime Merge

- Port and wire MCP++ primitives (queue/scheduler/workflow/peer/cache).
- Integrate these primitives into unified services layer.

Exit criteria:

- MCP++ primitive test suite passes in target package.
- Conformance matrix shows all core MCP++ requirements implemented or deferred with rationale.

### Phase 3: Tool Category Completion

- Finish remaining tool categories from source `tools/*`.
- Resolve naming collisions and runtime metadata consistency.

Exit criteria:

- Tool parity matrix reaches agreed threshold (recommended 95%+ required tools).
- All required categories have deterministic dispatch tests.

### Phase 4: Security/Observability/Performance

- Port security/policy/audit and telemetry parity.
- Execute performance and reliability validation.

Exit criteria:

- Security regression suite green.
- SLO and latency benchmarks meet target thresholds.

### Phase 5: Cutover and Deprecation

- Flip unified runtime as default.
- Maintain compatibility facade for one release window.
- Execute controlled deprecation of legacy paths.

Exit criteria:

- Default startup uses unified runtime without flags.
- Rollback path documented and validated.

## 8. Test Strategy and Gates

### Unit Gates

- Core modules: router, manager, adapter, metadata, configs.
- MCP++ primitives: queue, scheduler, cache, peer registry, workflow engine.
- Tool wrappers: schema and return contract checks.

### Integration Gates

- Unified bootstrap and dispatch tests.
- Transport-mode tests (stdio/http/trio-p2p).
- Cross-runtime interop tests.

### Conformance Gates

- `CONFORMANCE_CHECKLIST.md` requirements mapped to executable tests.
- No cutover unless all critical checklist items are green.

### Release Gates

- Performance smoke benchmarks within tolerance.
- Security/audit checks pass.
- Feature-flag rollback verified.

## 9. Risk Register

- Runtime divergence across fastapi/trio/bridge paths.
  - Mitigation: single runtime router contract + transport-specific tests.
- Duplicate tool registrations and schema drift.
  - Mitigation: centralized registration adapter + parity matrix.
- Scope explosion from source tool breadth.
  - Mitigation: strict wave planning, per-wave exit criteria, no speculative ports.
- Path/spec ambiguity.
  - Mitigation: path inventory gate and explicit spec source-of-truth docs.

## 10. Execution Backlog (Prioritized)

### Immediate (Next 1-2 weeks)

1. Create/refresh `ipfs_accelerate_py/mcpplusplus` spec files and gap matrix.
2. Build source-to-target capability matrix:
   - Source: `ipfs_datasets_py/ipfs_datasets_py/mcp_server`
   - Target: `ipfs_accelerate_py/mcp_server`
   - MCP++ runtime: `ipfs_datasets_py/ipfs_datasets_py/mcp_server/mcplusplus`
3. Port MCP++ runtime primitives into target subpackage with adapter seams.
4. Finish transport-mode parity tests.

### Near Term (Next 3-6 weeks)

1. Complete remaining tool-category waves.
2. Finish security/policy/audit parity.
3. Harden observability and performance baselines.
4. Run staged cutover in CI, then default runtime switch.

## 11. Success Metrics

- 100% critical spec requirements implemented and tested.
- 0 critical parity gaps in core MCP++ runtime primitives.
- 0 unresolved blocking tool-category gaps for required categories.
- Stable unified bootstrap suite with trendline growth and no regression flapping.
- Unified `mcp_server` becomes default runtime with rollback path retained.
