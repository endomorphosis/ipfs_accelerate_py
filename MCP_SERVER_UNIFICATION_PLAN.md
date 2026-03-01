# MCP Server Unification Plan

## Objective

Port `ipfs_datasets_py/ipfs_datasets_py/mcp_server` into `ipfs_accelerate_py/ipfs_accelerate_py/mcp_server`, then merge `ipfs_accelerate_py/ipfs_accelerate_py/mcplusplus_module` features to produce a single, spec-driven MCP++ implementation aligned to `ipfs_accelerate_py/ipfs_accelerate_py/mcplusplus`.

## Current Baseline (Verified)

- Source implementation exists at `ipfs_datasets_py/ipfs_datasets_py/mcp_server`.
- Existing IPFS Accelerate MCP implementation exists at `ipfs_accelerate_py/ipfs_accelerate_py/mcp`.
- MCP++ implementation module exists at `ipfs_accelerate_py/ipfs_accelerate_py/mcplusplus_module`.
- Spec baseline now exists in `ipfs_accelerate_py/ipfs_accelerate_py/mcplusplus` (`README.md`, `CONFORMANCE_CHECKLIST.md`).

## Current Implementation Status (Live)

- [x] `ipfs_accelerate_py/ipfs_accelerate_py/mcp_server` package scaffolded and importable.
- [x] Core modules implemented in unified package:
  - `tool_registry.py`
  - `runtime_router.py`
  - `hierarchical_tool_manager.py`
  - `registration_adapter.py`
  - `tool_metadata.py`
  - `wave_a_loaders.py`
- [x] Feature-flagged unified bootstrap wired into `mcp_server/server.py`.
- [x] Feature-flagged legacy-to-unified bridge wired into `mcp/server.py`.
- [x] Canonical unified meta-tools implemented:
  - `tools_list_categories`
  - `tools_list_tools`
  - `tools_get_schema`
  - `tools_dispatch`
  - `tools_runtime_metrics`
- [x] Timeout observability added in runtime metrics (`timeout_count`).
- [x] First native Wave A IPFS tools ported in unified package:
  - `ipfs_files_validate_cid`
  - `ipfs_files_list_files`
  - `ipfs_files_add_file`
  - `ipfs_files_pin_file`
  - `ipfs_files_unpin_file`
  - `ipfs_files_get_file`
- [x] First native Wave A workflow tool ported in unified package:
  - `get_workflow_templates`
- [x] Additional native Wave A workflow tool ported in unified package:
  - `list_workflows`
- [x] Additional native Wave A workflow tool ported in unified package:
  - `get_workflow`
- [x] Additional native Wave A workflow tool ported in unified package:
  - `create_workflow`
- [x] Additional native Wave A workflow tool ported in unified package:
  - `update_workflow`
- [x] Additional native Wave A workflow tool ported in unified package:
  - `delete_workflow`
- [x] Additional native Wave A workflow tool ported in unified package:
  - `start_workflow`
- [x] Additional native Wave A workflow tool ported in unified package:
  - `pause_workflow`
- [x] Additional native Wave A workflow tool ported in unified package:
  - `stop_workflow`
- [x] First native Wave A p2p tool ported in unified package:
  - `p2p_taskqueue_status`
- [x] Additional native Wave A p2p tool ported in unified package:
  - `p2p_taskqueue_list_tasks`
- [x] Additional native Wave A p2p tool ported in unified package:
  - `p2p_taskqueue_get_task`
- [x] Additional native Wave A p2p tool ported in unified package:
  - `p2p_taskqueue_wait_task`
- [x] Additional native Wave A p2p tool ported in unified package:
  - `p2p_taskqueue_complete_task`
- [x] Additional native Wave A p2p tool ported in unified package:
  - `p2p_taskqueue_heartbeat`
- [x] Unified bootstrap E2E tests passing in `ipfs_accelerate_py/mcp/tests/test_mcp_server_unified_bootstrap.py` (37 tests).

## Rollout Flags

Use these environment variables to control migration behavior safely:

| Flag | Default | Effect |
|---|---|---|
| `IPFS_MCP_ENABLE_UNIFIED_BRIDGE` | `false` | When true, legacy `create_mcp_server()` in `ipfs_accelerate_py.mcp.server` delegates creation through `ipfs_accelerate_py.mcp_server.server.create_server()`. |
| `IPFS_MCP_SERVER_ENABLE_UNIFIED_BOOTSTRAP` | `false` | When true, unified server attaches runtime router, hierarchical manager, Wave A loaders, and unified meta-tools to delegated server instance. |
| `IPFS_MCP_UNIFIED_PRELOAD_CATEGORIES` | empty | Optional eager category preload for unified manager. Accepts comma-separated values (`ipfs,workflow,p2p`) or `all`. Defaults to lazy-load only. |

## Target Architecture

- Canonical runtime package: `ipfs_accelerate_py/ipfs_accelerate_py/mcp_server`.
- Compatibility facade: `ipfs_accelerate_py/ipfs_accelerate_py/mcp` (temporary).
- Trio/P2P provider module: `ipfs_accelerate_py/ipfs_accelerate_py/mcplusplus_module`.
- Normative spec and conformance checklist: `ipfs_accelerate_py/ipfs_accelerate_py/mcplusplus`.

## Non-Goals (Initial Migration)

- No immediate removal of `ipfs_accelerate_py/ipfs_accelerate_py/mcp`.
- No big-bang migration of all tool categories in one change.
- No immediate deletion of compatibility shims.

## Workstreams

### 1. Spec Bootstrapping

- Add baseline MCP++ spec in `mcplusplus/README.md`.
- Add `mcplusplus/CONFORMANCE_CHECKLIST.md` with testable requirements.
- Keep requirements implementation-oriented, not aspirational.

### 2. Runtime Consolidation

- Create `mcp_server` package in `ipfs_accelerate_py`.
- Start with compatibility entry points that delegate to existing `mcp` server.
- Incrementally replace internals with ported code from `ipfs_datasets_py.mcp_server`.

### 3. Feature Merge (MCP++)

- Integrate Trio server path from `mcplusplus_module/trio`.
- Integrate P2P workflow/taskqueue tools from `mcplusplus_module/tools`.
- Integrate connectivity and peer registry from `mcplusplus_module/p2p`.
- Ensure one registration path per tool to avoid duplicates.

### 4. Tool Surface Migration (Wave Strategy)

- Wave A: dataset/ipfs/vector/graph/workflow/p2p/audit.
- Wave B: auth/monitoring/cache/storage/security/admin.
- Wave C: legal/pdf/media/web archive/finance/investigation.

Each wave must satisfy:
- import checks
- registration parity checks
- targeted integration tests

### 5. Compatibility and Cutover

- Keep `mcp` importing from `mcp_server` until migration is complete.
- Deprecate direct old entry points with warnings.
- Flip default startup path to `mcp_server` after parity gates pass.

## Phased Delivery

### Phase 0 (This Start)

- [x] Create unification plan.
- [x] Seed spec folder with baseline docs/checklist.
- [x] Create `mcp_server` package with compatibility-safe server entry point.

### Phase 1

- Port core framework files from source server:
  - [x] `tool_registry.py`
  - [x] `runtime_router.py`
  - [x] `hierarchical_tool_manager.py`
  - [x] `configs.py`
  - [x] `exceptions.py`
- [x] Wire startup lifecycle and dispatch pipeline (feature-flagged bootstrap path).
- [x] Add bridge guardrails and anti-recursion wiring.

### Phase 2

- [~] Merge MCP++ Trio and P2P execution paths (runtime router + trio bridge execution path in place; full parity pending).
- [x] Add unified registration layer (`registration_adapter.py`).
- [ ] Add transport-mode selection parity (stdio/http/trio-p2p) in unified package.

### Phase 3

- [~] Complete wave-based tool migration (Wave A bootstrap loaders for `ipfs/workflow/p2p` in place).
- [x] Enforce schema/metadata consistency baseline (`tool_metadata.py`, runtime precedence tests).
- [~] Stabilize runtime routing (timeouts + timeout metrics implemented; broader production hardening pending).

### Phase 4

- Execute cutover to `mcp_server` default.
- Maintain compatibility bridge for one release window.

## Validation Gates

Primary test gates during migration:

- `ipfs_accelerate_py/mcp/tests/test_mcp_server.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_integration.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_mcplusplus_tool_parity.py`
- `ipfs_accelerate_py/mcp/tests/test_p2p_taskqueue_mcp_tools.py`
- `ipfs_accelerate_py/mcp/tests/test_p2p_call_tool_bridge.py`
- `ipfs_accelerate_py/mcplusplus_module/tests/test_trio_server.py`
- `ipfs_accelerate_py/mcplusplus_module/tests/test_trio_bridge.py`
- `ipfs_datasets_py/tests/integration/test_mcp_tools_integration.py`
- `ipfs_datasets_py/tests/integration/test_mcp_p2p_libp2p_smoke.py`
- `test/integration/test_mcp_mcplusplus_interop_smoke.py`

## Risks and Mitigations

- Three runtime models drifting apart.
  - Mitigation: single canonical runtime package (`mcp_server`) with adapters only.
- Duplicate tool registrations.
  - Mitigation: central registry guard + parity tests.
- Import shadowing issues around `mcp` package names.
  - Mitigation: startup import diagnostics and explicit import checks in CI.
- Scope explosion due to broad source categories.
  - Mitigation: wave-based migration with strict per-wave exit criteria.

## Next 3 Implementation Tasks

1. Continue Wave A native `ipfs` migration by porting one additional operation and validating it via `tools_dispatch` tests.
2. Continue native `p2p` extraction into `ipfs_accelerate_py/mcp_server/tools/` with parity checks against legacy behavior.
3. Add transport parity increment in unified package (`stdio` + explicit HTTP mode selection), then gate with a dedicated integration test path.
