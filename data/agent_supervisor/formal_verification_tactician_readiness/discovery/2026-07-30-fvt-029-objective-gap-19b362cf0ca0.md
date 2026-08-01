# FVT-029 Objective Goal Gap

Date: 2026-07-30
Fingerprint: 19b362cf0ca0bf35c8d69df5016f63c83d79d6ca
Goal id: FVT-G050
Goal title: Expose stable goal-directed verification operations everywhere
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: public-api
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 28
Bundle: formal-verification-tactician/provider-surface
Parallel lane: formal-verification-tactician/provider-surface
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Add schema-equivalent Python, CLI, datasets MCP, and parent MCP operations for goal formalization, interpretation comparison, missing-proof discovery, proof planning/validation/execution/status, counterexample minimization/explanation/replay.
AST query: ipfs_datasets_py/tests/integration/logic/test_goal_tactician_public_api.py, test/api/test_goal_tactician_cli_mcp_parity.py
Conflict policy: Own stable public wiring and conformance tests; version additive operations and do not expose supervisor-only mutation controls through datasets APIs.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/verification_api.py, ipfs_datasets_py/tests/integration/logic/test_goal_tactician_public_api.py, test/api/test_goal_tactician_cli_mcp_parity.py
AST symbols: ipfs_datasets_py/tests/integration/logic/test_goal_tactician_public_api.py, test/api/test_goal_tactician_cli_mcp_parity.py
Interfaces: GoalTacticianAPI@1, GoalTacticianCLIMCP@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/19e237396a73396bffbfdbca0691afbbfc9b14fca41d1526e130e4ceb7cba9bf
Acceptance subset: All channels share closed requests/responses, identities, status, authority, diagnostics, redaction, bounds, cancellation, and availability, imports are side-effect free, legacy operations remain compatible, transport success never implies proof success.
Preconditions: objective goal FVT-G050 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/tests/integration/logic/test_goal_tactician_public_api.py, satisfy evidence requirement: test/api/test_goal_tactician_cli_mcp_parity.py
Evidence subset: ipfs_datasets_py/tests/integration/logic/test_goal_tactician_public_api.py, test/api/test_goal_tactician_cli_mcp_parity.py
Dependencies: FVT-G036, FVT-G042, FVT-G044
Resource class: cpu-api
Token class: medium
Estimated tokens: 0
Resources: cpu-api
Merge fate: objective/FVT-G050
Rejection reasons: none (accepted)

## Goal

Add schema-equivalent Python, CLI, datasets MCP, and parent MCP operations for goal formalization, interpretation comparison, missing-proof discovery, proof planning/validation/execution/status, counterexample minimization/explanation/replay.

## Missing Evidence

- ipfs_datasets_py/tests/integration/logic/test_goal_tactician_public_api.py
- test/api/test_goal_tactician_cli_mcp_parity.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
