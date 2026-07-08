# MCP Cutover Checklist

<!-- markdownlint-disable MD013 -->

Status: Approved for canonical-default cutover on 2026-03-08.

## Scope Freeze

Frozen migration delta set:

- Canonical runtime default remains `ipfs_accelerate_py.mcp_server`.
- Legacy `ipfs_accelerate_py.mcp.server` remains compatibility-only in deprecation phase `D2_opt_in_only`.
- Rollback and dry-run controls remain available via `IPFS_MCP_FORCE_LEGACY_ROLLBACK` and `IPFS_MCP_UNIFIED_CUTOVER_DRY_RUN`.
- Remaining follow-up work is limited to maintenance hardening, deprecation-phase progression (`D2`/`D3`), and explicitly deferred governance items already tracked in `SERVER_UNIFICATION_PLAN.md`.

## Cutover Gates

- [x] Canonical startup is the default path.
  - Evidence: `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni007_cutover_rollback.py`
- [x] Compatibility facade emits deterministic telemetry for unified handoff, rollback, and dry-run fallback.
  - Evidence: `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni007_cutover_rollback.py`
- [x] Forced rollback takes precedence over dry-run/bridge enablement.
  - Evidence: `test_force_rollback_takes_precedence_over_cutover_dry_run`
- [x] Process-level and FastAPI helper entrypoints remain stable.
  - Evidence: `ipfs_accelerate_py/mcp/tests/test_mcp_transport_process_level.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_fastapi_service.py`
- [x] CLI and direct-entry startup remain stable.
  - Evidence: `ipfs_accelerate_py/mcp/tests/test_mcp_cli.py`, subprocess/direct-entry coverage recorded in `SERVER_UNIFICATION_PLAN.md`
- [x] MCP+p2p handler abuse and mixed-version compatibility are validated.
  - Evidence: `ipfs_accelerate_py/mcp/tests/test_mcp_transport_mcp_p2p_handler_limits.py`
- [x] Cross-entrypoint transport parity is validated.
  - Evidence: `ipfs_accelerate_py/mcp/tests/test_mcp_server_transport_e2e_matrix.py`
- [x] Representative MCP++ profile chapters are green in the release-candidate matrix.
  - Evidence: `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_idl.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_artifacts.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_ucan.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_policy.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_event_dag.py`, `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_risk_scheduler.py`
- [x] Conformance docs are synchronized with executable evidence.
  - Evidence: `mcpplusplus/CONFORMANCE_CHECKLIST.md`, `mcpplusplus/SPEC_GAP_MATRIX.md`, `SERVER_UNIFICATION_PLAN.md`

## Focused Release-Candidate Matrix

Executed on 2026-03-08:

- `ipfs_accelerate_py/mcp/tests/test_mcp_server_uni007_cutover_rollback.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_transport_process_level.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_transport_mcp_p2p_handler_limits.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_server_transport_e2e_matrix.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_idl.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_artifacts.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_ucan.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_policy.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_event_dag.py`
- `ipfs_accelerate_py/mcp/tests/test_mcp_server_mcplusplus_risk_scheduler.py`

Result:

- `120 passed`

## Operational Notes

- Expected CI/local warnings from optional dependencies do not block cutover when the deterministic target suites remain green.
- The required libp2p-enabled lane remains part of transport governance and should continue to enforce networked parity separately from this focused local matrix.
- Future work after cutover should update this checklist if any gate re-opens or if deprecation phase progression changes runtime behavior.
