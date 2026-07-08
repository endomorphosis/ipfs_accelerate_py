# MCP++ Spec Baseline

<!-- markdownlint-disable MD013 -->

This directory is the source of truth for MCP++ conformance in this workspace.

## Scope

- Canonical target runtime: `ipfs_accelerate_py/mcp_server`
- Compatibility facade (temporary): `ipfs_accelerate_py/mcp`
- Source implementation for parity reference: `ipfs_datasets_py/ipfs_datasets_py/mcp_server`
- Existing alternate runtime package to integrate with: `ipfs_accelerate_py/mcplusplus_module`

## Files

- `mcpplusplus/CONFORMANCE_CHECKLIST.md`: requirement-level conformance status and evidence.
- `mcpplusplus/SPEC_GAP_MATRIX.md`: capability matrix from source to target with implementation and test status.
- `mcpplusplus/SERVER_UNIFICATION_PLAN.md`: migration backlog, milestones, and deferred-governance tracking.
- `mcpplusplus/CUTOVER_CHECKLIST.md`: approved cutover gates and release-candidate evidence.
- `mcpplusplus/history/`: earlier MCP++ phase completion records.

## Status Model

- `PASS`: implemented and covered by automated evidence.
- `PARTIAL`: implemented in part, or implemented without complete parity tests.
- `GAP`: not implemented in canonical target runtime.
- `DEFERRED`: intentionally postponed with rationale and tracking.

## Evidence Rules

- Every checklist item must include at least one repo-local evidence reference.
- Evidence references should point to implementation and test paths when both exist.
- Any item marked `PASS` must have executable test coverage.

## Update Workflow

1. Update `SPEC_GAP_MATRIX.md` first when code changes parity status.
2. Update requirement status in `CONFORMANCE_CHECKLIST.md`.
3. Add or update deterministic tests in `ipfs_accelerate_py/mcp/tests/`.
4. Keep `SERVER_UNIFICATION_PLAN.md` aligned with milestone and phase changes.
